"""
benchmark_ablation.py — Three-way ablation for IDETC §4.4 (R1f, R3.5).

Configurations:
  A. LLM-only           Question -> gpt-5, no retrieval, no graph.
  B. LLM + RAG          Vector retrieval (frames_knowledge) only, no graph.
                        Retrieved frames passed in context to gpt-5.
  C. Full pipeline      Vector retrieval + graph traversal + multi-vector
                        ranking + structured synthesis (delegates to
                        dfm_agent.py in Quick-Search-equivalent mode).

Question set (8): mix of well-covered, sparse, and out-of-scope topics.
Out-of-scope queries test refusal behavior — Configs B & C should
produce explicit "no corpus evidence" responses, while Config A should
produce confident hallucinated answers.

Metrics per (config, query):
  - elapsed_s, tokens, cost_usd
  - n_cited_frames (RAG configs only)
  - n_design_rules (heuristic line count)
  - n_disciplines (heuristic header count)
  - frame_id_faithfulness: cited frame_ids exist in corpus?
  - refusal: does the answer flag absent corpus coverage?

Outputs:
  benchmark_ablation_results.json  -- machine-readable
  benchmark_ablation_summary.md    -- LaTeX-ready summary table

Usage:
    python benchmark_ablation.py
    python benchmark_ablation.py --queries 1,2,3       # subset
    python benchmark_ablation.py --configs A,B         # subset
    python benchmark_ablation.py --quick               # smaller top_k
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
import io
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

BASE_DIR = Path(__file__).parent
GRAPH_PICKLE = BASE_DIR / "provenance_graph.gpickle"
QDRANT_PATH = str(BASE_DIR / "qdrant_data")
EXTRACTIONS = BASE_DIR / "extractions_se_all.json"

LLM_MODEL = os.getenv("ABLATION_MODEL", "gpt-5")
EMBEDDING_MODEL = "text-embedding-3-large"

PRICING_PER_M = {
    "gpt-5":              {"in": 1.25, "out": 10.00},
    "gpt-5-mini":         {"in": 0.25, "out":  2.00},
    "gpt-5-nano":         {"in": 0.05, "out":  0.40},
    "gpt-4.1":            {"in": 2.00, "out":  8.00},
    "gpt-4.1-mini":       {"in": 0.40, "out":  1.60},
    "text-embedding-3-large": {"in": 0.13, "out": 0.0},
}


QUERIES: list[dict[str, str]] = [
    {
        "id": "Q1",
        "coverage": "well-covered",
        "scope": "design_geometry",
        "question": "What should I consider when designing thin-walled aluminum parts for CNC machining?",
    },
    {
        "id": "Q2",
        "coverage": "well-covered",
        "scope": "machining_process",
        "question": "How do I choose between climb and conventional milling?",
    },
    {
        "id": "Q3",
        "coverage": "well-covered",
        "scope": "design_geometry",
        "question": "What DFM rules apply to deep pockets and tall narrow ribs in CNC parts?",
    },
    {
        "id": "Q4",
        "coverage": "sparse",
        "scope": "cost",
        "question": "How does adding chamfers and edge breaks affect machining cost on small batch parts?",
    },
    {
        "id": "Q5",
        "coverage": "sparse",
        "scope": "shop_operations",
        "question": "How should I label and track parts on the shop floor for traceability between design and machining?",
    },
    {
        "id": "Q6",
        "coverage": "sparse",
        "scope": "case",
        "question": "Give me a real-world example of a tolerance stack-up failure in a turned shaft assembly.",
    },
    {
        "id": "Q7",
        "coverage": "out-of-scope",
        "scope": "sheet_metal",
        "question": "What is the minimum bend radius for 16-gauge stainless steel in air bending?",
    },
    {
        "id": "Q8",
        "coverage": "out-of-scope",
        "scope": "additive",
        "question": "What support-structure rules apply to FDM overhangs greater than 45 degrees in PLA?",
    },
]


# ─────────────────────────────────────────────
# Config A — LLM-only baseline
# ─────────────────────────────────────────────

LLM_ONLY_PROMPT = """You are a design-for-manufacturing advisor. Answer the
user's question with the best practical guidance you can give. Be concrete:
include specific rules of thumb, numeric ranges where possible, and
common failure modes. Keep the response under 350 words. Do not cite
sources you cannot verify.

Question: {q}
"""


def run_llm_only(client: OpenAI, query: dict) -> dict:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": LLM_ONLY_PROMPT.format(q=query["question"])}],
    )
    elapsed = time.time() - t0
    u = resp.usage
    return {
        "config": "A_llm_only",
        "query_id": query["id"],
        "coverage": query["coverage"],
        "model": LLM_MODEL,
        "elapsed_s": round(elapsed, 1),
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "n_cited_frames": 0,
        "answer": resp.choices[0].message.content or "",
    }


# ─────────────────────────────────────────────
# Config B — LLM + RAG (no graph)
# ─────────────────────────────────────────────

RAG_PROMPT = """You are a design-for-manufacturing advisor. Below are
knowledge frames retrieved from a corpus of machining-forum discourse.
Each frame has a frame_id and a main_point. Answer the user's question
using ONLY information present in the retrieved frames. Cite frame_ids
in square brackets, e.g. [se-engineering-12345-a67890-2]. If the
retrieved frames do not contain enough information to answer, say so
explicitly and identify the specific gap.

Retrieved frames:
{frames}

Question: {q}
"""


def embed_query(client: OpenAI, text: str) -> tuple[list[float], int]:
    r = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return r.data[0].embedding, r.usage.total_tokens


def retrieve_rag(qdrant: QdrantClient, vector: list[float], top_k: int) -> list[dict]:
    hits = qdrant.query_points(
        collection_name="frames_knowledge",
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "frame_id": p.get("frame_id", ""),
            "main_point": p.get("main_point", ""),
            "subject": p.get("subject", ""),
            "frame_type": p.get("frame_type", ""),
            "scope": p.get("scope", ""),
            "score": float(h.score),
        })
    return out


def run_llm_rag(client: OpenAI, qdrant: QdrantClient | None, query: dict, top_k: int = 12) -> dict:
    """qdrant=None opens a fresh local client and closes it before returning,
    so the lock does not collide with Config C's dfm_agent QdrantClient."""
    t0 = time.time()
    own_qdrant = qdrant is None
    if own_qdrant:
        qdrant = QdrantClient(path=QDRANT_PATH)
    try:
        vector, embed_toks = embed_query(client, query["question"])
        frames = retrieve_rag(qdrant, vector, top_k)
    finally:
        if own_qdrant:
            qdrant.close()

    blob = "\n\n".join(
        f"[{f['frame_id']}] ({f['frame_type']}, {f['scope']}) {f['main_point']}"
        for f in frames
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": RAG_PROMPT.format(frames=blob, q=query["question"])},
        ],
    )
    elapsed = time.time() - t0
    u = resp.usage

    return {
        "config": "B_llm_rag",
        "query_id": query["id"],
        "coverage": query["coverage"],
        "model": LLM_MODEL,
        "elapsed_s": round(elapsed, 1),
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "embedding_tokens": embed_toks,
        "retrieved_frame_ids": [f["frame_id"] for f in frames],
        "n_retrieved": len(frames),
        "answer": resp.choices[0].message.content or "",
    }


# ─────────────────────────────────────────────
# Config C — Full pipeline (delegates to dfm_agent)
# ─────────────────────────────────────────────

def _serialize_report_v2(report) -> str:
    """Render a DFMReportV2 (Pydantic) as a string for downstream metrics."""
    try:
        if hasattr(report, "model_dump_json"):
            return report.model_dump_json()
        if hasattr(report, "json"):
            return report.json()
    except Exception:
        pass
    return str(report)


def run_full_pipeline(query: dict, G=None) -> dict:
    """Delegate to dfm_agent in a Quick-Search-equivalent configuration:
    decompose + multi-vector retrieve + 2-hop graph traversal + structured
    synthesis. Returns metrics dict."""
    import dfm_agent as agent
    from dfm_agent import query_dfm

    t0 = time.time()
    agent.MODEL = LLM_MODEL
    report, enrichment, ranked, sub_aspects, diagnostic, backfill_result, tracker = query_dfm(
        question=query["question"],
        top_k=10,
        max_frames=20,
        max_hops=2,
        decompose=True,
        backfill=True,
        report_format="v2",
        verbose=False,
        G=G,
    )
    elapsed = time.time() - t0

    # Pull metrics out of the structured report
    n_cited = 0
    n_rules = 0
    n_disciplines = 0
    n_contradictions = 0
    n_gaps = 0
    summary = ""
    try:
        n_disciplines = len(getattr(report, "discipline_sections", []) or [])
        n_rules = len(getattr(report, "design_rules", []) or [])
        n_contradictions = len(getattr(report, "contradictions", []) or [])
        n_gaps = len(getattr(report, "knowledge_gaps", []) or [])
        summary = getattr(report, "summary", "") or ""
        # Cited frame IDs across all evidence + design_rules
        cited_ids = set()
        for sec in getattr(report, "discipline_sections", []) or []:
            for ev in getattr(sec, "evidence", []) or []:
                fid = getattr(ev, "frame_id", "") or (ev.get("frame_id", "") if isinstance(ev, dict) else "")
                if fid:
                    cited_ids.add(fid)
        for rule in getattr(report, "design_rules", []) or []:
            fid = getattr(rule, "frame_id", "") or (rule.get("frame_id", "") if isinstance(rule, dict) else "")
            if fid:
                cited_ids.add(fid)
        n_cited = len(cited_ids)
    except Exception as e:
        print(f"  warning: report parse error: {e}")

    return {
        "config": "C_full_pipeline",
        "query_id": query["id"],
        "coverage": query["coverage"],
        "model": LLM_MODEL,
        "elapsed_s": round(elapsed, 1),
        "prompt_tokens": tracker.prompt_tokens,
        "completion_tokens": tracker.completion_tokens,
        "embedding_tokens": tracker.embedding_tokens,
        "n_cited_frames": n_cited,
        "n_design_rules": n_rules,
        "n_disciplines": n_disciplines,
        "n_contradictions": n_contradictions,
        "n_knowledge_gaps": n_gaps,
        "answer": summary or _serialize_report_v2(report),
    }


# ─────────────────────────────────────────────
# Post-hoc analysis
# ─────────────────────────────────────────────

FRAME_ID_RE = re.compile(r"se-[a-z]+-\d+-[a-z]?\d+-\d+", re.IGNORECASE)


def analyze_answer(record: dict, real_frame_ids: set[str]) -> dict:
    """Add derived metrics to one config-query record."""
    text = record.get("answer", "") or ""

    cited = set(FRAME_ID_RE.findall(text))
    fabricated = cited - real_frame_ids
    record.setdefault("n_cited_frames", len(cited))

    # Refusal heuristic: explicit "no evidence" / "no information" phrases
    refusal_phrases = [
        "no corpus evidence",
        "no information",
        "not enough information",
        "the retrieved frames do not",
        "frames do not contain",
        "outside the scope",
        "knowledge gap",
        "no extracted frames",
    ]
    record["refusal_signal"] = any(p in text.lower() for p in refusal_phrases)

    # Crude design-rule count: lines starting with "- ", "* ", or numbered
    rule_lines = re.findall(r"(?m)^\s*(?:[-\*]|\d+\.)\s+\S", text)
    record.setdefault("n_design_rules", len(rule_lines))

    # Crude discipline count: bolded headers or section titles
    disciplines = set()
    for m in re.findall(r"(?m)^\s*#{1,3}\s+(.+)$", text):
        disciplines.add(m.lower().strip())
    for m in re.findall(r"\*\*([A-Z][^*]+)\*\*", text):
        disciplines.add(m.lower().strip())
    record.setdefault("n_disciplines", len(disciplines))

    record["frame_id_fabrication_rate"] = (
        round(len(fabricated) / max(len(cited), 1), 3)
    )
    record["fabricated_frame_ids"] = sorted(fabricated)

    # Cost
    in_t = record.get("prompt_tokens", 0) + record.get("embedding_tokens", 0)
    out_t = record.get("completion_tokens", 0)
    p = PRICING_PER_M.get(record.get("model", ""), {"in": 0, "out": 0})
    record["cost_usd"] = round((in_t / 1e6) * p["in"] + (out_t / 1e6) * p["out"], 4)
    record["total_tokens"] = in_t + out_t

    return record


def load_real_frame_ids() -> set[str]:
    """Best-effort set of canonical frame IDs. The extractions JSON does not
    store a top-level frame_id, so we also try the graph nodes."""
    ids: set[str] = set()
    if EXTRACTIONS.exists():
        data = json.loads(EXTRACTIONS.read_text(encoding="utf-8"))
        if isinstance(data, list):
            ids |= {f.get("frame_id", "") for f in data if isinstance(f, dict) and f.get("frame_id")}
        elif isinstance(data, dict) and "results" in data:
            for post in data["results"]:
                for fr in post.get("frames", []) or []:
                    if fr.get("frame_id"):
                        ids.add(fr["frame_id"])
    if GRAPH_PICKLE.exists():
        try:
            with open(GRAPH_PICKLE, "rb") as f:
                G = pickle.load(f)
            for n, d in G.nodes(data=True):
                if d.get("node_type") == "Frame":
                    ids.add(n)
        except Exception:
            pass
    return {x for x in ids if x}


# ─────────────────────────────────────────────
# Summary / output
# ─────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = r["config"]
        bucket = out.setdefault(key, {
            "n": 0, "elapsed_s": 0.0, "cost_usd": 0.0, "tokens": 0,
            "cited": 0, "rules": 0, "disciplines": 0,
            "fabrications": 0, "refusals": 0,
            "by_coverage": {},
        })
        bucket["n"] += 1
        bucket["elapsed_s"] += r.get("elapsed_s", 0)
        bucket["cost_usd"] += r.get("cost_usd", 0)
        bucket["tokens"] += r.get("total_tokens", 0)
        bucket["cited"] += r.get("n_cited_frames", 0)
        bucket["rules"] += r.get("n_design_rules", 0)
        bucket["disciplines"] += r.get("n_disciplines", 0)
        bucket["fabrications"] += len(r.get("fabricated_frame_ids", []))
        bucket["refusals"] += int(bool(r.get("refusal_signal", False)))
        cov = r.get("coverage", "?")
        cb = bucket["by_coverage"].setdefault(cov, {"refusals": 0, "n": 0})
        cb["n"] += 1
        cb["refusals"] += int(bool(r.get("refusal_signal", False)))
    return out


def render_markdown_summary(rows: list[dict], agg: dict) -> str:
    lines = ["# Ablation Results (IDETC §4.4)", "", f"Model: {LLM_MODEL}    Run: {time.strftime('%Y-%m-%d %H:%M')}", ""]

    lines.append("## Per-config averages")
    lines.append("")
    lines.append("| Config | Latency (s) | Cost (USD) | Tokens | Cited frames | Rules | Disciplines | Fabricated frame IDs | Refusals (out-of-scope correct) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cfg, b in agg.items():
        n = max(b["n"], 1)
        oos = b["by_coverage"].get("out-of-scope", {"n": 0, "refusals": 0})
        oos_correct = f"{oos['refusals']}/{oos['n']}" if oos["n"] else "n/a"
        lines.append(
            f"| {cfg} | {b['elapsed_s']/n:.1f} | ${b['cost_usd']/n:.3f} | "
            f"{b['tokens']/n:.0f} | {b['cited']/n:.1f} | {b['rules']/n:.1f} | "
            f"{b['disciplines']/n:.1f} | {b['fabrications']} total | {oos_correct} |"
        )
    lines.append("")

    lines.append("## Per-query")
    lines.append("")
    lines.append("| Query | Coverage | Config | Latency | Cost | Cited | Rules | Refusal? | Fabricated |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['query_id']} | {r['coverage']} | {r['config']} | "
            f"{r.get('elapsed_s', 0):.1f}s | ${r.get('cost_usd', 0):.3f} | "
            f"{r.get('n_cited_frames', 0)} | {r.get('n_design_rules', 0)} | "
            f"{'yes' if r.get('refusal_signal') else 'no'} | "
            f"{len(r.get('fabricated_frame_ids', []))} |"
        )
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="", help="Comma-separated query IDs (Q1..Q8) to run")
    ap.add_argument("--configs", default="A,B,C", help="Subset of A,B,C")
    ap.add_argument("--quick", action="store_true", help="Smaller top_k for debugging")
    ap.add_argument("--out-json", default="benchmark_ablation_results.json")
    ap.add_argument("--out-md", default="benchmark_ablation_summary.md")
    args = ap.parse_args()

    queries = QUERIES
    if args.queries.strip():
        wanted = {q.strip() for q in args.queries.split(",")}
        queries = [q for q in QUERIES if q["id"] in wanted]

    configs = {c.strip().upper() for c in args.configs.split(",")}
    top_k = 5 if args.quick else 12

    client = OpenAI()
    # Qdrant is opened per-call inside run_llm_rag to avoid lock collision
    # with Config C's dfm_agent.QdrantClient.
    qdrant = None

    # Pre-load graph once for Config C
    G = None
    if "C" in configs:
        print("Loading graph...")
        with open(GRAPH_PICKLE, "rb") as f:
            G = pickle.load(f)
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    real_frame_ids = load_real_frame_ids()
    print(f"Loaded {len(real_frame_ids)} real frame IDs for fabrication check.")

    rows: list[dict] = []
    for q in queries:
        print(f"\n=== {q['id']} ({q['coverage']}) === {q['question']}")
        if "A" in configs:
            print("  [A] LLM only ...")
            try:
                rows.append(run_llm_only(client, q))
            except Exception as e:
                print(f"    A failed: {e}")
        if "B" in configs:
            print("  [B] LLM + RAG ...")
            try:
                rows.append(run_llm_rag(client, qdrant, q, top_k=top_k))
            except Exception as e:
                print(f"    B failed: {e}")
        if "C" in configs:
            print("  [C] Full pipeline ...")
            try:
                rows.append(run_full_pipeline(q, G=G))
            except Exception as e:
                print(f"    C failed: {e}")
                import traceback; traceback.print_exc()

    for r in rows:
        analyze_answer(r, real_frame_ids)

    agg = aggregate(rows)

    Path(args.out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(render_markdown_summary(rows, agg), encoding="utf-8")
    print(f"\nWrote {args.out_json} and {args.out_md} ({len(rows)} runs).")


if __name__ == "__main__":
    main()
