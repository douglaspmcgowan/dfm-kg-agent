"""
Infer semantic edges between knowledge frames using GPT-5.

For each Frame node in the provenance graph, finds the top-K most similar
frames via Qdrant vector neighbor lookup (uses stored vectors -- no re-embedding),
then uses GPT-5 structured output to classify each unique pair relationship.

Edge labels:
  CONTRADICTS   A and B give directly opposing guidance for the same situation.
  MITIGATES     One frame is a solution/prevention for a risk in the other.
  SPECIFIES     One frame adds a condition/context that scopes the other.
  CAUSAL_LINK   One frame explains the physical cause behind the other.
  UNRELATED     Superficial similarity; no real semantic relationship.

Adds new directed edges to the provenance graph and saves output to:
  semantic_edges.json          -- all non-UNRELATED classified edges
  provenance_graph.gpickle     -- updated graph with semantic edges
  provenance_graph.graphml     -- updated GraphML export
  infer_edges_checkpoint.json  -- incremental checkpoint (used with --resume)

Usage:
    python infer_edges.py                  # full run, all indexed frames
    python infer_edges.py --limit 100      # prototype on first 100 frames
    python infer_edges.py --resume         # continue from checkpoint
    python infer_edges.py --dry-run        # Qdrant lookup only, skip LLM
    OPENAI_MODEL=gpt-5 python infer_edges.py
    TOP_K=3 PARALLEL_REQUESTS=5 python infer_edges.py

Environment variables:
    OPENAI_API_KEY       required
    OPENAI_MODEL         default: gpt-5
    TOP_K                default: 5 (neighbors per frame)
    PARALLEL_REQUESTS    default: 10 (concurrent LLM calls)
    CHECKPOINT_INTERVAL  default: 100 (save checkpoint every N completions)
"""

import json
import os
import sys
import re
import argparse
import pickle
import threading
import concurrent.futures
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from pydantic import BaseModel
from typing import Literal

import networkx as nx

BASE_DIR      = Path(__file__).parent
GRAPH_PICKLE  = BASE_DIR / "provenance_graph.gpickle"
GRAPH_GRAPHML = BASE_DIR / "provenance_graph.graphml"
QDRANT_PATH   = str(BASE_DIR / "qdrant_data")
CHECKPOINT    = BASE_DIR / "infer_edges_checkpoint.json"
EDGES_OUT     = BASE_DIR / "semantic_edges.json"

MODEL        = os.getenv("OPENAI_MODEL", "gpt-5")
TOP_K        = int(os.getenv("TOP_K", "5"))
PARALLEL     = int(os.getenv("PARALLEL_REQUESTS", "10"))
CHKPT_EVERY  = int(os.getenv("CHECKPOINT_INTERVAL", "100"))

KNOWLEDGE_COLLECTION = "frames_knowledge"

_XML_ILLEGAL = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def _clean_xml(v):
    return _XML_ILLEGAL.sub("", v) if isinstance(v, str) else v


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

class EdgeClassification(BaseModel):
    label: Literal["CONTRADICTS", "MITIGATES", "SPECIFIES", "CAUSAL_LINK", "UNRELATED"]
    direction: Literal["A_to_B", "B_to_A", "bidirectional", "none"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

def load_graph():
    if not GRAPH_PICKLE.exists():
        print(f"ERROR: {GRAPH_PICKLE} not found. Run build_graph.py first.")
        sys.exit(1)
    with open(GRAPH_PICKLE, "rb") as f:
        G = pickle.load(f)
    frame_count = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "Frame")
    print(f"Graph loaded : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {frame_count} Frame nodes")
    return G


def save_graph(G):
    """Save updated graph to pickle and GraphML (sanitizing for XML)."""
    with open(GRAPH_PICKLE, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pickle saved : {GRAPH_PICKLE}")

    # Sanitize for GraphML: no None, no control chars
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


# ─────────────────────────────────────────────
# QDRANT NEIGHBOR LOOKUP
# ─────────────────────────────────────────────

def build_qdrant_index(qdrant):
    """
    Scroll all points in frames_knowledge to build frame_id <-> qdrant_int_id maps.
    Returns: (fid_to_int dict, int_to_fid dict)
    """
    fid_to_int: Dict[str, int] = {}
    int_to_fid: Dict[int, str] = {}

    offset = None
    batch_size = 500

    while True:
        result = qdrant.scroll(
            collection_name=KNOWLEDGE_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=["frame_id"],
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            fid = point.payload.get("frame_id", "")
            if fid:
                fid_to_int[fid] = point.id
                int_to_fid[point.id] = fid
        if next_offset is None:
            break
        offset = next_offset

    print(f"Qdrant index : {len(fid_to_int)} frames mapped")
    return fid_to_int, int_to_fid


def get_frame_neighbors(qdrant, int_id: int, top_k: int, int_to_fid: Dict[int, str]):
    """
    Find the top_k nearest neighbors of frame int_id using its stored vector.
    Returns list of frame_id strings (excludes the query frame itself).
    """
    results = qdrant.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=int_id,          # integer ID -> uses stored vector, no re-embedding
        limit=top_k + 1,       # +1 in case self is returned
        with_payload=["frame_id"],
    ).points

    neighbors = []
    for point in results:
        if point.id == int_id:
            continue
        fid = point.payload.get("frame_id") or int_to_fid.get(point.id, "")
        if fid:
            neighbors.append(fid)

    return neighbors[:top_k]


# ─────────────────────────────────────────────
# PAIR GATHERING
# ─────────────────────────────────────────────

def gather_pairs(G, qdrant, fid_to_int: Dict[str, int], int_to_fid: Dict[int, str],
                 top_k: int, limit: Optional[int] = None):
    """
    For each Frame node in the graph that is also in the Qdrant index,
    find its top-K neighbors. Returns a deduplicated list of (fid_a, fid_b) pairs.

    Skips:
      - Pairs from the same post (post_id derived by stripping the trailing "-{i}" index)
      - Duplicate pairs (treats {A,B} and {B,A} as the same pair)
    """
    frame_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "Frame" and n in fid_to_int
    ]

    not_indexed = sum(
        1 for n, d in G.nodes(data=True)
        if d.get("node_type") == "Frame" and n not in fid_to_int
    )
    if not_indexed:
        print(f"  Note: {not_indexed} Frame nodes not in Qdrant index (will be skipped)")

    if limit:
        frame_nodes = frame_nodes[:limit]

    seen: Set[frozenset] = set()
    pairs: List[Tuple[str, str]] = []
    skipped_same_post = 0
    skipped_dup = 0

    print(f"Gathering pairs for {len(frame_nodes)} frames (top_k={top_k})...")

    for i, fid_a in enumerate(frame_nodes):
        int_id_a = fid_to_int[fid_a]
        post_a = fid_a.rsplit("-", 1)[0]  # strip trailing "-{i}" to get post_id

        try:
            neighbors = get_frame_neighbors(qdrant, int_id_a, top_k, int_to_fid)
        except Exception as e:
            print(f"  [WARN] Neighbor lookup failed for {fid_a}: {e}")
            continue

        for fid_b in neighbors:
            if fid_b not in fid_to_int:
                continue

            # Skip same-post pairs (structural edges already capture intra-post relationships)
            post_b = fid_b.rsplit("-", 1)[0]
            if post_a == post_b:
                skipped_same_post += 1
                continue

            pair_key = frozenset({fid_a, fid_b})
            if pair_key in seen:
                skipped_dup += 1
                continue

            seen.add(pair_key)
            pairs.append((fid_a, fid_b))

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(frame_nodes)} frames processed, {len(pairs)} pairs so far")

    print(f"Pairs gathered   : {len(pairs)} unique")
    print(f"  Skipped same-post  : {skipped_same_post}")
    print(f"  Skipped duplicates : {skipped_dup}")
    return pairs


# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────

def classify_pair(client: OpenAI, fid_a: str, fid_b: str, frame_data: Dict[str, Dict]):
    """Call GPT to classify relationship between two frames. Returns result dict."""
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
            "frame_a":   fid_a,
            "frame_b":   fid_b,
            "label":     result.label,
            "direction": result.direction,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        return {
            "frame_a":   fid_a,
            "frame_b":   fid_b,
            "label":     "ERROR",
            "direction": "none",
            "confidence": "low",
            "reasoning": f"{type(e).__name__}: {e}",
        }


def _save_checkpoint(results: List[Dict]) -> None:
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)


def load_checkpoint():
    """Returns (previous_results, set of already-done pair keys)."""
    if not CHECKPOINT.exists():
        return [], set()
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    done = {f"{r['frame_a']}|{r['frame_b']}" for r in results}
    print(f"Checkpoint   : {len(results)} previously classified pairs")
    return results, done


def run_classification(client: OpenAI, pairs: List[Tuple[str, str]],
                       frame_data: Dict[str, Dict], already_done: Set[str],
                       dry_run: bool = False):
    """Classify all pairs in parallel. Returns list of result dicts."""
    # Filter pairs already classified in a previous run
    remaining = [
        (a, b) for a, b in pairs
        if f"{a}|{b}" not in already_done and f"{b}|{a}" not in already_done
    ]
    skipped = len(pairs) - len(remaining)
    print(f"Pairs to classify: {len(remaining)}"
          + (f" ({skipped} already done from checkpoint)" if skipped else ""))

    if not remaining or dry_run:
        if dry_run:
            print("[DRY RUN] Skipping LLM classification.")
        return []

    results: List[Dict] = []
    lock = threading.Lock()
    completed = [0]
    start = time.time()

    def _worker(pair: Tuple[str, str]) -> Dict:
        r = classify_pair(client, pair[0], pair[1], frame_data)
        with lock:
            completed[0] += 1
            results.append(r)
            if completed[0] % CHKPT_EVERY == 0:
                _save_checkpoint(results)
                elapsed = time.time() - start
                done_labels = {}
                for x in results:
                    done_labels[x["label"]] = done_labels.get(x["label"], 0) + 1
                label_str = " ".join(f"{k}:{v}" for k, v in sorted(done_labels.items()))
                print(f"  [{completed[0]}/{len(remaining)}] checkpoint | {elapsed:.0f}s | {label_str}",
                      flush=True)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        list(executor.map(_worker, remaining))

    # Final checkpoint
    _save_checkpoint(results)
    elapsed = time.time() - start
    print(f"Classification done: {len(results)} pairs in {elapsed:.1f}s "
          f"({elapsed/max(len(results),1):.2f}s/pair)")
    return results


# ─────────────────────────────────────────────
# EDGE WRITING
# ─────────────────────────────────────────────

def add_semantic_edges(G, results: List[Dict]):
    """
    Add non-UNRELATED classified edges to the graph.
    Returns dict of edge_type -> count.

    CONTRADICTS: bidirectional (A->B and B->A)
    MITIGATES, SPECIFIES, CAUSAL_LINK: directed per the 'direction' field
    """
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
        else:  # A_to_B or "none"
            G.add_edge(fid_a, fid_b, **attrs)
            counts[label] = counts.get(label, 0) + 1

    return counts


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────

def print_stats(results: List[Dict], edge_counts: Dict[str, int]) -> None:
    label_dist: Dict[str, int] = {}
    conf_dist:  Dict[str, int] = {}
    for r in results:
        lbl  = r.get("label", "?")
        conf = r.get("confidence", "?")
        label_dist[lbl]  = label_dist.get(lbl, 0) + 1
        conf_dist[conf]  = conf_dist.get(conf, 0) + 1

    n = max(len(results), 1)
    print(f"\n{'='*60}")
    print(f"SEMANTIC EDGE INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Total pairs classified : {len(results)}")
    print(f"\nLabel distribution:")
    for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"  {lbl:<15}: {cnt:>5}  ({100*cnt/n:.1f}%)")
    print(f"\nConfidence distribution:")
    for conf, cnt in sorted(conf_dist.items()):
        print(f"  {conf:<10}: {cnt}")
    print(f"\nEdges added to graph:")
    for et, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"  {et:<25}: {cnt}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Infer semantic edges between knowledge frames."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N frames (for prototyping -- skip full run cost)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from infer_edges_checkpoint.json"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run Qdrant neighbor lookup and pair gathering only; skip LLM calls"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("OPENAI_API_KEY not set.")

    print(f"Model        : {MODEL}")
    print(f"Top-K        : {TOP_K}")
    print(f"Parallel     : {PARALLEL}")
    print(f"Limit        : {args.limit if args.limit else 'all frames'}")
    print(f"Resume       : {args.resume}")
    print(f"Dry run      : {args.dry_run}")
    print()

    # Load graph
    G = load_graph()

    # Open Qdrant
    if not os.path.exists(QDRANT_PATH):
        print(f"ERROR: {QDRANT_PATH} not found. Run build_embeddings.py first.")
        sys.exit(1)
    qdrant = QdrantClient(path=QDRANT_PATH)

    # Build frame_id <-> qdrant int_id maps
    fid_to_int, int_to_fid = build_qdrant_index(qdrant)

    # Gather candidate pairs via Qdrant neighbor lookup (fast, no API cost)
    pairs = gather_pairs(G, qdrant, fid_to_int, int_to_fid, TOP_K, limit=args.limit)
    qdrant.close()

    if not pairs:
        print("No pairs to classify.")
        return

    # Load checkpoint if resuming
    prev_results, already_done = load_checkpoint() if args.resume else ([], set())

    # Extract frame content from graph nodes (for LLM prompt)
    frame_data: Dict[str, Dict] = {
        n: dict(d) for n, d in G.nodes(data=True) if d.get("node_type") == "Frame"
    }

    # Classify pairs
    client = OpenAI(api_key=api_key) if not args.dry_run else None
    new_results = run_classification(
        client, pairs, frame_data, already_done, dry_run=args.dry_run
    )

    all_results = prev_results + new_results

    # Save semantic edges JSON
    if not args.dry_run and all_results:
        non_unrelated = [r for r in all_results if r.get("label") not in ("UNRELATED", "ERROR")]
        with open(EDGES_OUT, "w", encoding="utf-8") as f:
            json.dump({
                "model":            MODEL,
                "top_k":            TOP_K,
                "total_classified": len(all_results),
                "total_edges":      len(non_unrelated),
                "edges":            non_unrelated,
            }, f, indent=2, ensure_ascii=False)
        print(f"Edges saved  : {EDGES_OUT} ({len(non_unrelated)} non-UNRELATED edges)")

    # Add edges to graph and save
    edge_counts = add_semantic_edges(G, all_results)
    if not args.dry_run:
        save_graph(G)

    print_stats(all_results, edge_counts)

    if not args.dry_run and not args.limit:
        print(f"\nCheckpoint kept at: {CHECKPOINT}")
        print("Delete it manually once you're satisfied with the run.")


if __name__ == "__main__":
    main()
