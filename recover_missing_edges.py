"""
Recover edges lost by incremental inference due to:
  1. Incomplete checkpoint (old run had pairs not in checkpoint)
  2. Asymmetric kNN (incremental only queried FROM changed frames)

Strategy:
  - Untouched edges (neither frame changed): copy directly from old semantic_edges.json
  - Touched pairs (>= 1 frame changed): re-classify with LLM
  - Merge with current results

Usage:
    python recover_missing_edges.py --dry-run     # show what would happen
    python recover_missing_edges.py               # run recovery
"""

import json
import os
import sys
import subprocess
import threading
import concurrent.futures
import time
import pickle
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter

from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
import networkx as nx

BASE_DIR      = Path(__file__).parent
GRAPH_PICKLE  = BASE_DIR / "provenance_graph.gpickle"
GRAPH_GRAPHML = BASE_DIR / "provenance_graph.graphml"
EDGES_OUT     = BASE_DIR / "semantic_edges.json"
CHECKPOINT    = BASE_DIR / "infer_edges_checkpoint.json"

MODEL    = os.getenv("OPENAI_MODEL", "gpt-5")
PARALLEL = int(os.getenv("PARALLEL_REQUESTS", "10"))

_XML_ILLEGAL = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def _clean_xml(v):
    return _XML_ILLEGAL.sub("", v) if isinstance(v, str) else v


# ── Reuse schema and prompt from infer_edges.py ──

class EdgeClassification(BaseModel):
    label: Literal["CONTRADICTS", "MITIGATES", "SPECIFIES", "CAUSAL_LINK", "UNRELATED"]
    direction: Literal["A_to_B", "B_to_A", "bidirectional", "none"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


SYSTEM_PROMPT = """You classify semantic relationships between pairs of machining and engineering knowledge frames.

Given Frame A and Frame B, output exactly one of these labels:

CONTRADICTS   A and B give directly opposing guidance for the same situation or subject.
              Example: A says "always use flood coolant on aluminum",
              B says "aluminum machines better without coolant to avoid built-up edge".
              This is symmetric -- set direction to "bidirectional".

MITIGATES     One frame provides a concrete solution, workaround, or prevention for a risk
              or problem described in the other. The mitigating frame is the source.
              Example: A is a heuristic "use sharp HSS tools to prevent built-up edge on Al";
              B is a risk "built-up edge on aluminum causes poor surface finish".
              Direction: A_to_B (A mitigates B).

SPECIFIES     One frame adds a condition, context, or constraint that narrows when the
              other applies. The specifying frame is the source.
              Example: A says "this only applies when machining without coolant";
              B is a general principle about aluminum machinability.
              Direction: A_to_B (A specifies B).

CAUSAL_LINK   One frame explains the physical or mechanical ROOT CAUSE behind an
              observation or effect described in the other. The causal frame is the source.
              Example: A explains "aluminum has low hardness so it tends to smear onto tool faces";
              B observes "built-up edge is common when machining aluminum".
              Direction: A_to_B (A explains B).

UNRELATED     The frames are topically similar but have no meaningful engineering relationship.
              Choose this when in doubt. Superficial overlap (same material, same scope) is not
              enough -- the relationship must be directional and specific.

Only label CONTRADICTS / MITIGATES / SPECIFIES / CAUSAL_LINK when the relationship
is clear, specific, and unambiguous. Default to UNRELATED for weak connections."""


def make_pair_prompt(a: Dict, b: Dict) -> str:
    def fmt(f: Dict, label: str) -> str:
        mp = (f.get("main_point") or "")[:300]
        sq = (f.get("source_quote") or "")[:200]
        return (
            f"Frame {label}:\n"
            f"  frame_type : {f.get('frame_type', '')}\n"
            f"  scope      : {f.get('scope', '')}\n"
            f"  subject    : {f.get('subject', '')}\n"
            f"  main_point : {mp}\n"
            f"  quote      : {sq}"
        )
    return fmt(a, "A") + "\n\n" + fmt(b, "B")


def classify_pair(client: OpenAI, fid_a: str, fid_b: str, frame_data: Dict[str, Dict]):
    a = frame_data.get(fid_a, {})
    b = frame_data.get(fid_b, {})
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": make_pair_prompt(a, b)},
            ],
            response_format=EdgeClassification,
        )
        result: EdgeClassification = completion.choices[0].message.parsed
        return {
            "frame_a":    fid_a,
            "frame_b":    fid_b,
            "label":      result.label,
            "direction":  result.direction,
            "confidence": result.confidence,
            "reasoning":  result.reasoning,
        }
    except Exception as e:
        return {
            "frame_a":    fid_a,
            "frame_b":    fid_b,
            "label":      "ERROR",
            "direction":  "none",
            "confidence": "low",
            "reasoning":  f"{type(e).__name__}: {e}",
        }


def save_graph(G):
    with open(GRAPH_PICKLE, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pickle saved : {GRAPH_PICKLE}")
    for node_id, data in G.nodes(data=True):
        for k, v in list(data.items()):
            if v is None:
                G.nodes[node_id][k] = ""
            elif isinstance(v, bool):
                G.nodes[node_id][k] = str(v)
            elif isinstance(v, str):
                G.nodes[node_id][k] = _clean_xml(v)
    for u, v, data in G.edges(data=True):
        for k, val in list(data.items()):
            if val is None:
                data[k] = ""
            elif isinstance(val, bool):
                data[k] = str(val)
            elif isinstance(val, str):
                data[k] = _clean_xml(val)
    nx.write_graphml(G, str(GRAPH_GRAPHML), encoding="utf-8")
    print(f"GraphML saved: {GRAPH_GRAPHML}")


def add_semantic_edges(G, results: List[Dict]):
    counts: Dict[str, int] = {}
    for r in results:
        label     = r.get("label", "")
        direction = r.get("direction", "none")
        fid_a     = r.get("frame_a", "")
        fid_b     = r.get("frame_b", "")
        if label in ("UNRELATED", "ERROR") or not label:
            counts[f"skipped_{label.lower() if label else 'empty'}"] = \
                counts.get(f"skipped_{label.lower() if label else 'empty'}", 0) + 1
            continue
        if not G.has_node(fid_a) or not G.has_node(fid_b):
            counts["skipped_node_missing"] = counts.get("skipped_node_missing", 0) + 1
            continue
        attrs = {
            "edge_type":  label,
            "confidence": r.get("confidence", ""),
            "reasoning":  r.get("reasoning", ""),
        }
        if label == "CONTRADICTS" or direction == "bidirectional":
            G.add_edge(fid_a, fid_b, **attrs)
            G.add_edge(fid_b, fid_a, **attrs)
            counts[label] = counts.get(label, 0) + 2
        elif direction == "B_to_A":
            G.add_edge(fid_b, fid_a, **attrs)
            counts[label] = counts.get(label, 0) + 1
        else:
            G.add_edge(fid_a, fid_b, **attrs)
            counts[label] = counts.get(label, 0) + 1
    return counts


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recover missing edges from incremental run.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without running LLM")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("OPENAI_API_KEY not set.")

    # ── Load current state ──
    print("Loading current semantic_edges.json...")
    current_data = json.load(open(EDGES_OUT, "r", encoding="utf-8"))
    current_edges = current_data["edges"]
    current_keys = {(e["frame_a"], e["frame_b"]) for e in current_edges}
    print(f"  Current edges: {len(current_edges)}")

    # ── Load current checkpoint (from incremental run) ──
    current_ckpt = json.load(open(CHECKPOINT, "r", encoding="utf-8"))["results"]
    current_ckpt_keys = {(r["frame_a"], r["frame_b"]) for r in current_ckpt}
    print(f"  Current checkpoint: {len(current_ckpt)} pairs")

    # ── Load old state from git ──
    print("Loading old semantic_edges.json from git HEAD...")
    old_edges_data = json.loads(
        subprocess.check_output(["git", "show", "HEAD:semantic_edges.json"]).decode("utf-8"))
    old_edges = old_edges_data["edges"]
    print(f"  Old edges: {len(old_edges)}")

    old_ckpt = json.loads(
        subprocess.check_output(["git", "show", "HEAD:infer_edges_checkpoint.json"]).decode("utf-8"))["results"]
    old_ckpt_keys = {(r["frame_a"], r["frame_b"]) for r in old_ckpt}
    print(f"  Old checkpoint: {len(old_ckpt)} pairs")

    # ── Load frame diff ──
    print("Computing frame diff...")
    old_ext = json.load(open("extractions_se_all.json", "r", encoding="utf-8"))
    new_ext = json.load(open("extractions_revised_reviews_all.json", "r", encoding="utf-8"))

    def build_frame_map(data):
        fmap = {}
        for r in data["results"]:
            pid = r["post_id"]
            for i, f in enumerate(r.get("frames", [])):
                fmap[f"{pid}-{i}"] = f
        return fmap

    old_fm = build_frame_map(old_ext)
    new_fm = build_frame_map(new_ext)
    changed = {fid for fid in set(old_fm) & set(new_fm) if old_fm[fid] != new_fm[fid]}
    print(f"  Changed frames: {len(changed)}")

    # ══════════════════════════════════════════════
    # IDENTIFY MISSING EDGES
    # ══════════════════════════════════════════════

    # 1. Checkpoint gap: edges in old semantic_edges but NOT in old checkpoint
    old_edges_not_in_ckpt = [e for e in old_edges
                             if (e["frame_a"], e["frame_b"]) not in old_ckpt_keys]

    # Split into untouched (can copy) vs touched (need re-classify)
    gap_untouched = [e for e in old_edges_not_in_ckpt
                     if e["frame_a"] not in changed and e["frame_b"] not in changed]
    gap_touched = [e for e in old_edges_not_in_ckpt
                   if e["frame_a"] in changed or e["frame_b"] in changed]

    # Filter out any that are already in current edges
    gap_untouched_new = [e for e in gap_untouched
                         if (e["frame_a"], e["frame_b"]) not in current_keys]
    gap_touched_new = [e for e in gap_touched
                       if (e["frame_a"], e["frame_b"]) not in current_keys]

    print(f"\n--- Checkpoint Gap ---")
    print(f"  Old edges NOT in old checkpoint: {len(old_edges_not_in_ckpt)}")
    print(f"  Untouched (copy directly): {len(gap_untouched)} ({len(gap_untouched_new)} not already in current)")
    print(f"  Touched (re-classify):     {len(gap_touched)} ({len(gap_touched_new)} not already in current)")

    # 2. Asymmetric kNN: pairs in old checkpoint that touched changed frames
    #    but were NOT re-gathered by incremental
    old_ckpt_touched = [r for r in old_ckpt
                        if r["frame_a"] in changed or r["frame_b"] in changed]
    asymmetric_lost = [r for r in old_ckpt_touched
                       if (r["frame_a"], r["frame_b"]) not in current_ckpt_keys
                       and (r["frame_b"], r["frame_a"]) not in current_ckpt_keys]

    print(f"\n--- Asymmetric kNN ---")
    print(f"  Old checkpoint pairs touching changed frames: {len(old_ckpt_touched)}")
    print(f"  Not re-gathered by incremental: {len(asymmetric_lost)}")
    asym_real = [r for r in asymmetric_lost if r["label"] not in ("UNRELATED", "ERROR")]
    print(f"  Of which were real edges: {len(asym_real)}")

    # Build pairs to re-classify (touched gap + asymmetric)
    pairs_to_classify: List[Tuple[str, str]] = []
    seen_pairs: Set[frozenset] = set()

    for e in gap_touched_new:
        pk = frozenset({e["frame_a"], e["frame_b"]})
        if pk not in seen_pairs:
            seen_pairs.add(pk)
            pairs_to_classify.append((e["frame_a"], e["frame_b"]))

    for r in asymmetric_lost:
        pk = frozenset({r["frame_a"], r["frame_b"]})
        if pk not in seen_pairs:
            # Also skip if already in current results
            if (r["frame_a"], r["frame_b"]) not in current_keys and \
               (r["frame_b"], r["frame_a"]) not in current_keys:
                seen_pairs.add(pk)
                pairs_to_classify.append((r["frame_a"], r["frame_b"]))

    print(f"\n{'='*60}")
    print(f"RECOVERY PLAN")
    print(f"{'='*60}")
    print(f"  Copy directly (untouched gap edges): {len(gap_untouched_new)}")
    print(f"  Re-classify (touched + asymmetric):  {len(pairs_to_classify)}")
    print(f"  Estimated API cost: ~${len(pairs_to_classify) * 0.007:.2f}")
    print(f"  Expected new total: ~{len(current_edges) + len(gap_untouched_new)} + re-classified edges")

    if args.dry_run:
        print("\n[DRY RUN] Stopping here.")

        # Show label distribution of what we'd copy
        copy_labels = Counter(e["label"] for e in gap_untouched_new)
        print(f"\n  Untouched edges to copy by label:")
        for lbl, cnt in copy_labels.most_common():
            print(f"    {lbl}: {cnt}")

        reclassify_old_labels = Counter()
        for a, b in pairs_to_classify:
            # Check old edges for this pair's label
            for e in old_edges:
                if e["frame_a"] == a and e["frame_b"] == b:
                    reclassify_old_labels[e["label"]] += 1
                    break
            else:
                for r in old_ckpt:
                    if r["frame_a"] == a and r["frame_b"] == b:
                        reclassify_old_labels[r["label"]] += 1
                        break

        print(f"\n  Pairs to re-classify (old labels):")
        for lbl, cnt in reclassify_old_labels.most_common():
            print(f"    {lbl}: {cnt}")
        return

    # ══════════════════════════════════════════════
    # EXECUTE RECOVERY
    # ══════════════════════════════════════════════

    # Load graph for frame data
    print("\nLoading graph for frame data...")
    with open(GRAPH_PICKLE, "rb") as f:
        G = pickle.load(f)
    frame_data: Dict[str, Dict] = {
        n: dict(d) for n, d in G.nodes(data=True) if d.get("node_type") == "Frame"
    }

    # Classify missing pairs
    print(f"\nClassifying {len(pairs_to_classify)} pairs with {MODEL}...")
    client = OpenAI(api_key=api_key)
    results: List[Dict] = []
    lock = threading.Lock()
    completed = [0]
    start = time.time()

    def _worker(pair: Tuple[str, str]) -> Dict:
        r = classify_pair(client, pair[0], pair[1], frame_data)
        with lock:
            completed[0] += 1
            results.append(r)
            if completed[0] % 100 == 0:
                elapsed = time.time() - start
                label_dist = Counter(x["label"] for x in results)
                label_str = " ".join(f"{k}:{v}" for k, v in sorted(label_dist.items()))
                print(f"  [{completed[0]}/{len(pairs_to_classify)}] {elapsed:.0f}s | {label_str}",
                      flush=True)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        list(executor.map(_worker, pairs_to_classify))

    elapsed = time.time() - start
    print(f"Classification done: {len(results)} pairs in {elapsed:.1f}s")

    # ── Merge everything ──
    print("\nMerging results...")

    # Start with current edges
    merged_edges = list(current_edges)
    merged_keys = set(current_keys)

    # Add untouched gap edges (direct copy)
    copied = 0
    for e in gap_untouched_new:
        k = (e["frame_a"], e["frame_b"])
        if k not in merged_keys:
            merged_edges.append(e)
            merged_keys.add(k)
            copied += 1

    # Add freshly classified non-UNRELATED
    classified_added = 0
    for r in results:
        if r["label"] in ("UNRELATED", "ERROR"):
            continue
        k = (r["frame_a"], r["frame_b"])
        if k not in merged_keys:
            merged_edges.append(r)
            merged_keys.add(k)
            classified_added += 1

    print(f"  Copied untouched edges: {copied}")
    print(f"  Added from classification: {classified_added}")
    print(f"  Total edges: {len(merged_edges)}")

    # ── Also merge into checkpoint ──
    # Load current checkpoint and add all new results
    ckpt_results = list(current_ckpt)
    ckpt_keys = set(current_ckpt_keys)
    # Add untouched gap edges as "results" with their existing data
    for e in gap_untouched_new:
        k = (e["frame_a"], e["frame_b"])
        if k not in ckpt_keys:
            ckpt_results.append(e)
            ckpt_keys.add(k)
    # Add all freshly classified (including UNRELATED)
    for r in results:
        k = (r["frame_a"], r["frame_b"])
        if k not in ckpt_keys:
            ckpt_results.append(r)
            ckpt_keys.add(k)

    # ── Save ──
    # Save merged semantic edges
    with open(EDGES_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "model":            MODEL,
            "top_k":            current_data.get("top_k", 5),
            "total_classified": len(ckpt_results),
            "total_edges":      len(merged_edges),
            "edges":            merged_edges,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nEdges saved: {EDGES_OUT} ({len(merged_edges)} edges)")

    # Save merged checkpoint
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"results": ckpt_results}, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved: {CHECKPOINT} ({len(ckpt_results)} results)")

    # ── Rebuild graph with all edges ──
    print("\nRebuilding graph with recovered edges...")
    # Reload clean graph (structural only)
    with open(GRAPH_PICKLE, "rb") as f:
        G = pickle.load(f)

    # Remove any existing semantic edges first
    semantic_edge_types = {"CONTRADICTS", "MITIGATES", "SPECIFIES", "CAUSAL_LINK"}
    to_remove = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get("edge_type") in semantic_edge_types]
    G.remove_edges_from(to_remove)
    print(f"  Cleared {len(to_remove)} old semantic edges")

    # Add all merged edges
    edge_counts = add_semantic_edges(G, merged_edges)
    save_graph(G)

    # Stats
    print(f"\n{'='*60}")
    print("RECOVERY COMPLETE")
    print(f"{'='*60}")
    print(f"Total edges: {len(merged_edges)}")
    label_dist = Counter(e["label"] for e in merged_edges)
    for lbl, cnt in label_dist.most_common():
        print(f"  {lbl}: {cnt}")

    fresh_label_dist = Counter(r["label"] for r in results)
    print(f"\nFreshly classified {len(results)} pairs:")
    for lbl, cnt in fresh_label_dist.most_common():
        print(f"  {lbl}: {cnt}")


if __name__ == "__main__":
    main()
