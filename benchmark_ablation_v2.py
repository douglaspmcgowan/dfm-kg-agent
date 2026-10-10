"""
benchmark_ablation_v2.py — Verbosity-matched three-way ablation
================================================================

Re-runs Configs A (LLM-only) / B (LLM+RAG) / C (Full pipeline) under a
COMMON output schema and a COMMON output budget, so the per-query
metrics (rule count, gap count, summary length) are comparable across
configurations.

Key differences from benchmark_ablation.py:

1. **Common output schema (AblationOutputV2)** — every config returns
   the same JSON shape: design_rules (≤MAX_RULES), knowledge_gaps
   (≤MAX_GAPS), cited_frame_ids (≤MAX_CITES), summary (≤MAX_SUMMARY_WORDS).
   This makes rule/gap counts directly comparable.

2. **Hard token cap on every config** — `max_output_tokens=MAX_OUTPUT_TOKENS`
   on every API call (configurable, default 1500).

3. **Reasoning-effort balancing for C** — Config C uses
   `reasoning_effort="low"` to drop its verbosity from ~24.5k tokens
   (v1, default reasoning + Pydantic schema with many fields) into the
   same ~1.5k band as A/B.

4. **Iterative tuning pass** — after the first 2 queries, the script
   measures per-config mean completion tokens. If any config's mean
   differs from the panel mean by more than VERBOSITY_TOLERANCE_PCT,
   the script auto-adjusts `max_output_tokens` per-config and re-runs
   those configs. Up to MAX_ITERATIONS rounds.

Usage:
    $env:OPENAI_API_KEY="..."
    python benchmark_ablation_v2.py                       # full 8-query run
    python benchmark_ablation_v2.py --queries Q1,Q2       # subset
    python benchmark_ablation_v2.py --no-tune             # skip auto-tune
    python benchmark_ablation_v2.py --pilot               # 2 queries only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

# Optional: load OPENAI_API_KEY from .env in the project root.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Strip any trailing whitespace/newlines from the API key
# (Windows env vars often retain a trailing \n that breaks the HTTP header).
if os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()
# Also clear any proxy env vars that block requests.
for _p in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
          "https_proxy", "all_proxy"):
    os.environ.pop(_p, None)

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).parent
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

# ─────────────────────────────────────────────────────────────────────
# COMMON OUTPUT SCHEMA & BUDGETS
# ─────────────────────────────────────────────────────────────────────

MAX_RULES = 10
MAX_GAPS = 10
MAX_CITES = 12
MAX_SUMMARY_WORDS = 200

# Initial token cap per config — will be auto-tuned if --tune is on.
# Note: max_completion_tokens for gpt-5 covers BOTH reasoning tokens AND
# visible output. With reasoning_effort=low the model can still spend 500–1500
# tokens reasoning. Set the cap generously so the visible JSON isn't truncated;
# verbosity is matched via the schema (10 rules, 10 gaps, 200-word summary).
INIT_MAX_OUTPUT_TOKENS = {
    "A_llm_only":      4096,
    "B_llm_rag":       4096,
    "C_full_pipeline": 4096,
}

# Reasoning effort per config. "low" everywhere keeps wall-clock predictable.
REASONING_EFFORT = {
    "A_llm_only":      "low",
    "B_llm_rag":       "low",
    "C_full_pipeline": "low",
}

VERBOSITY_TOLERANCE_PCT = 30   # acceptable spread vs panel-mean completion tokens
MAX_ITERATIONS = 3             # auto-tune passes (incl. the initial run)


class AblationOutputV2(BaseModel):
    """Common schema every config emits."""
    design_rules: list[str] = Field(default_factory=list, max_length=MAX_RULES)
    knowledge_gaps: list[str] = Field(default_factory=list, max_length=MAX_GAPS)
    cited_frame_ids: list[str] = Field(default_factory=list, max_length=MAX_CITES)
    summary: str = Field(default="", max_length=MAX_SUMMARY_WORDS * 7)  # ~7 chars/word ceiling


# ─────────────────────────────────────────────────────────────────────
# QUERIES (same as v1)
# ─────────────────────────────────────────────────────────────────────

QUERIES: list[dict[str, str]] = [
    {"id": "Q1", "coverage": "well-covered",  "scope": "design_geometry",
     "question": "What should I consider when designing thin-walled aluminum parts for CNC machining?"},
    {"id": "Q2", "coverage": "well-covered",  "scope": "machining_process",
     "question": "How do I choose between climb and conventional milling?"},
    {"id": "Q3", "coverage": "well-covered",  "scope": "design_geometry",
     "question": "What DFM rules apply to deep pockets and tall narrow ribs in CNC parts?"},
    {"id": "Q4", "coverage": "sparse",        "scope": "cost",
     "question": "How does adding chamfers and edge breaks affect machining cost on small batch parts?"},
    {"id": "Q5", "coverage": "sparse",        "scope": "shop_operations",
     "question": "How should I label and track parts on the shop floor for traceability between design and machining?"},
    {"id": "Q6", "coverage": "sparse",        "scope": "case",
     "question": "Give me a real-world example of a tolerance stack-up failure in a turned shaft assembly."},
    {"id": "Q7", "coverage": "out-of-scope",  "scope": "sheet_metal",
     "question": "What is the minimum bend radius for 16-gauge stainless steel in air bending?"},
    {"id": "Q8", "coverage": "out-of-scope",  "scope": "additive",
     "question": "What support-structure rules apply to FDM overhangs greater than 45 degrees in PLA?"},
]

# ─────────────────────────────────────────────────────────────────────
# CONFIG A — LLM-only
# ─────────────────────────────────────────────────────────────────────

LLM_ONLY_SYSTEM = f"""You are a design-for-manufacturing advisor. Respond using ONLY the JSON schema described in the user message. Cap your response at {MAX_RULES} design_rules, {MAX_GAPS} knowledge_gaps, {MAX_CITES} cited_frame_ids (always empty for this configuration — you have no retrieval), and a summary of {MAX_SUMMARY_WORDS} words or fewer. Do not cite sources you cannot verify."""

LLM_ONLY_USER = """Question: {q}

Respond with a JSON object matching this schema (no markdown, no code fences):
{{
  "design_rules":     [list of strings, up to {max_rules}],
  "knowledge_gaps":   [list of strings, up to {max_gaps}],
  "cited_frame_ids":  [],
  "summary":          "string, up to {max_summary} words"
}}"""

def run_llm_only(client: OpenAI, query: dict, max_output_tokens: int) -> dict:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": LLM_ONLY_SYSTEM},
            {"role": "user", "content": LLM_ONLY_USER.format(
                q=query["question"],
                max_rules=MAX_RULES, max_gaps=MAX_GAPS,
                max_summary=MAX_SUMMARY_WORDS,
            )},
        ],
        max_completion_tokens=max_output_tokens,
        response_format={"type": "json_object"},
        reasoning_effort=REASONING_EFFORT["A_llm_only"],
    )
    elapsed = time.time() - t0
    u = resp.usage
    raw = resp.choices[0].message.content or "{}"
    parsed = _safe_json(raw)
    return _record("A_llm_only", query, elapsed, u, parsed, raw)


# ─────────────────────────────────────────────────────────────────────
# CONFIG B — LLM + RAG
# ─────────────────────────────────────────────────────────────────────

RAG_SYSTEM = f"""You are a design-for-manufacturing advisor. You will receive a set of knowledge frames retrieved from a forum corpus. Respond using ONLY information present in the retrieved frames; cite every claim with a frame_id. Use the JSON schema in the user message. Cap your response at {MAX_RULES} design_rules, {MAX_GAPS} knowledge_gaps, {MAX_CITES} cited_frame_ids, and a summary of {MAX_SUMMARY_WORDS} words or fewer. If the retrieved frames do not contain enough information to answer, populate knowledge_gaps explicitly with what is missing."""

RAG_USER = """Retrieved frames:
{frames}

Question: {q}

Respond with a JSON object (no markdown, no code fences):
{{
  "design_rules":     [list of strings, up to {max_rules}],
  "knowledge_gaps":   [list of strings, up to {max_gaps}],
  "cited_frame_ids":  [list of frame_id strings, up to {max_cites}],
  "summary":          "string, up to {max_summary} words"
}}"""

def embed_query(client: OpenAI, text: str) -> tuple[list[float], int]:
    r = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return r.data[0].embedding, r.usage.total_tokens

def retrieve_rag(qdrant: QdrantClient, vector: list[float], top_k: int) -> list[dict]:
    hits = qdrant.query_points(
        collection_name="frames_knowledge",
        query=vector, limit=top_k, with_payload=True,
    ).points
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "frame_id":  p.get("frame_id", ""),
            "main_point": p.get("main_point", ""),
            "subject":   p.get("subject", ""),
            "frame_type":p.get("frame_type", ""),
            "scope":     p.get("scope", ""),
            "score":     float(h.score),
        })
    return out

def run_llm_rag(client: OpenAI, qdrant: QdrantClient | None,
                query: dict, max_output_tokens: int, top_k: int = 12) -> dict:
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
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": RAG_USER.format(
                frames=blob, q=query["question"],
                max_rules=MAX_RULES, max_gaps=MAX_GAPS,
                max_cites=MAX_CITES, max_summary=MAX_SUMMARY_WORDS,
            )},
        ],
        max_completion_tokens=max_output_tokens,
        response_format={"type": "json_object"},
        reasoning_effort=REASONING_EFFORT["B_llm_rag"],
    )
    elapsed = time.time() - t0
    u = resp.usage
    raw = resp.choices[0].message.content or "{}"
    parsed = _safe_json(raw)
    record = _record("B_llm_rag", query, elapsed, u, parsed, raw)
    record["embedding_tokens"] = embed_toks
    record["retrieved_frame_ids"] = [f["frame_id"] for f in frames]
    record["n_retrieved"] = len(frames)
    return record


# ─────────────────────────────────────────────────────────────────────
# CONFIG C — Full pipeline (delegates to dfm_agent)
# ─────────────────────────────────────────────────────────────────────

C_RESYNTH_SYSTEM = f"""You are a design-for-manufacturing advisor. You have been given a structured DFM report produced by an agentic pipeline. Compress it into the common ablation output schema (defined in the user message). Take only the design rules and knowledge gaps that are most directly supported by retrieved evidence in the report. Cap your output at {MAX_RULES} design_rules, {MAX_GAPS} knowledge_gaps, {MAX_CITES} cited_frame_ids, and a summary of {MAX_SUMMARY_WORDS} words or fewer."""

C_RESYNTH_USER = """Full pipeline report (verbose):
{report_json}

Question: {q}

Respond with a JSON object (no markdown, no code fences):
{{
  "design_rules":     [list of strings, up to {max_rules}],
  "knowledge_gaps":   [list of strings, up to {max_gaps}],
  "cited_frame_ids":  [list of frame_id strings, up to {max_cites}],
  "summary":          "string, up to {max_summary} words"
}}"""

def run_full_pipeline(client: OpenAI, query: dict, max_output_tokens: int,
                      G=None) -> dict:
    """Run the full pipeline at low reasoning effort, then compress the
    Pydantic report to the common AblationOutputV2 schema via a final
    resynth call. This pulls C's per-query token count into the same
    band as A and B without changing what C actually does."""
    # dfm_agent lives under archive/v1_streamlit_app/ — add to path on demand.
    _v1 = BASE_DIR / "archive" / "v1_streamlit_app"
    if str(_v1) not in sys.path:
        sys.path.insert(0, str(_v1))
    import dfm_agent as agent
    from dfm_agent import query_dfm
    # dfm_agent's QDRANT_PATH defaults to its own folder; redirect to the
    # project-root qdrant_data where the real data lives.
    agent.QDRANT_PATH = str(BASE_DIR / "qdrant_data")

    t0 = time.time()
    agent.MODEL = LLM_MODEL
    # The agent has its own reasoning effort handling — try to force low.
    if hasattr(agent, "REASONING_EFFORT"):
        agent.REASONING_EFFORT = "low"
    report, enrichment, ranked, sub_aspects, diagnostic, backfill_result, tracker = query_dfm(
        question=query["question"],
        top_k=10, max_frames=20, max_hops=2,
        decompose=True, backfill=True,
        report_format="v2", verbose=False, G=G,
    )
    pipeline_elapsed = time.time() - t0

    # Serialize the Pydantic report for the resynth call.
    try:
        report_json = report.model_dump_json()
    except Exception:
        report_json = str(report)

    # Resynth into the common schema.
    t1 = time.time()
    resynth = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": C_RESYNTH_SYSTEM},
            {"role": "user", "content": C_RESYNTH_USER.format(
                report_json=report_json, q=query["question"],
                max_rules=MAX_RULES, max_gaps=MAX_GAPS,
                max_cites=MAX_CITES, max_summary=MAX_SUMMARY_WORDS,
            )},
        ],
        max_completion_tokens=max_output_tokens,
        response_format={"type": "json_object"},
        reasoning_effort=REASONING_EFFORT["C_full_pipeline"],
    )
    resynth_elapsed = time.time() - t1
    elapsed = pipeline_elapsed + resynth_elapsed

    u = resynth.usage
    # Approximate tracker totals + resynth.
    p_in = (getattr(tracker, "prompt_tokens", 0) if tracker else 0) + (u.prompt_tokens or 0)
    p_out = (getattr(tracker, "completion_tokens", 0) if tracker else 0) + (u.completion_tokens or 0)

    raw = resynth.choices[0].message.content or "{}"
    parsed = _safe_json(raw)
    record = _record_raw("C_full_pipeline", query, elapsed,
                         p_in, p_out, parsed, raw)
    record["pipeline_elapsed_s"] = round(pipeline_elapsed, 1)
    record["resynth_elapsed_s"]  = round(resynth_elapsed, 1)
    return record


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        # Try to recover a JSON object from extra text
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return {}

def _record(config: str, query: dict, elapsed: float, usage,
            parsed: dict, raw: str) -> dict:
    return _record_raw(config, query, elapsed,
                       getattr(usage, "prompt_tokens", 0),
                       getattr(usage, "completion_tokens", 0),
                       parsed, raw)

def _record_raw(config: str, query: dict, elapsed: float,
                p_in: int, p_out: int, parsed: dict, raw: str) -> dict:
    cited = parsed.get("cited_frame_ids", []) or []
    rules = parsed.get("design_rules", []) or []
    gaps  = parsed.get("knowledge_gaps", []) or []
    summary = parsed.get("summary", "") or ""
    return {
        "config": config,
        "query_id": query["id"],
        "coverage": query["coverage"],
        "model": LLM_MODEL,
        "elapsed_s": round(elapsed, 1),
        "prompt_tokens": p_in,
        "completion_tokens": p_out,
        "n_cited_frames": len(cited),
        "n_design_rules": len(rules),
        "n_knowledge_gaps": len(gaps),
        "summary_word_count": len(summary.split()),
        "refusal_flag": _detect_refusal(parsed, raw),
        "answer_json": parsed,
    }

def _detect_refusal(parsed: dict, raw: str) -> bool:
    """A refusal is any explicit knowledge_gap that flags absent corpus
    evidence on the question itself (not adjacent gaps)."""
    gaps = parsed.get("knowledge_gaps", []) or []
    if not gaps:
        return False
    keywords = ["no direct corpus evidence", "no frames", "corpus does not",
                "outside the scope", "out of scope", "not in the corpus",
                "no evidence in the corpus"]
    return any(any(k in g.lower() for k in keywords) for g in gaps)

def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING_PER_M.get(model, {"in": 0, "out": 0})
    return (prompt_tokens / 1e6) * p["in"] + (completion_tokens / 1e6) * p["out"]


# ─────────────────────────────────────────────────────────────────────
# ITERATIVE VERBOSITY TUNING
# ─────────────────────────────────────────────────────────────────────

def per_config_means(records: list[dict]) -> dict:
    by_config = {}
    for r in records:
        by_config.setdefault(r["config"], []).append(r["completion_tokens"])
    return {c: sum(v) / len(v) for c, v in by_config.items()}

def tune_budgets(means: dict, current: dict, tolerance_pct: int) -> tuple[dict, bool]:
    """Adjust per-config max_output_tokens to push spread within tolerance."""
    if not means:
        return current, True
    panel_mean = sum(means.values()) / len(means)
    adjusted = dict(current)
    converged = True
    for config, m in means.items():
        diff_pct = abs(m - panel_mean) / panel_mean * 100
        if diff_pct > tolerance_pct:
            converged = False
            if m > panel_mean:
                # Too verbose — tighten the cap.
                adjusted[config] = max(500, int(current[config] * panel_mean / m * 0.95))
            else:
                # Too terse — loosen the cap (within reason).
                adjusted[config] = min(4096, int(current[config] * panel_mean / m * 1.05))
    return adjusted, converged


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verbosity-matched ablation v2")
    parser.add_argument("--queries", default=None,
                        help="Comma-separated query IDs (default: all 8)")
    parser.add_argument("--configs", default="A,B,C",
                        help="Comma-separated configs (default: A,B,C)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip the iterative verbosity tuning pass")
    parser.add_argument("--pilot", action="store_true",
                        help="Run only Q1+Q2 (cheap dry-run for budget tuning)")
    parser.add_argument("--out", default="benchmark_ablation_v2_results.json")
    args = parser.parse_args()

    # Query subset
    queries = list(QUERIES)
    if args.pilot:
        queries = [q for q in QUERIES if q["id"] in ("Q1", "Q2")]
    if args.queries:
        wanted = set(args.queries.split(","))
        queries = [q for q in QUERIES if q["id"] in wanted]

    # Config subset
    configs = set(args.configs.split(","))
    print(f"[v2] queries={[q['id'] for q in queries]}  configs={configs}")
    print(f"[v2] model={LLM_MODEL}  cap={INIT_MAX_OUTPUT_TOKENS}")

    client = OpenAI()
    # Note: Qdrant local-file mode is single-instance. We pass `None` to
    # run_llm_rag so it opens/closes its own client per-call — otherwise
    # Config C's dfm_agent collides with the persistent client.
    qdrant = None

    # Maybe load graph for Config C
    G = None
    if "C" in configs:
        import pickle
        try:
            with open(BASE_DIR / "provenance_graph.gpickle", "rb") as f:
                G = pickle.load(f)
            print(f"[v2] loaded provenance graph: {len(G.nodes)} nodes")
        except Exception as e:
            print(f"[v2] WARN: could not load graph ({e}); C will load its own")

    current_caps = dict(INIT_MAX_OUTPUT_TOKENS)
    pilot_records = []

    if not args.no_tune and not args.pilot:
        # Phase 1: pilot pass on Q1+Q2 to measure verbosity, then tune.
        print("\n[v2] ── Phase 1: 2-query pilot for verbosity tuning ──")
        pilot_queries = [q for q in queries if q["id"] in ("Q1", "Q2")] or queries[:2]
        for it in range(MAX_ITERATIONS):
            print(f"[v2] tuning iter {it+1}/{MAX_ITERATIONS}  caps={current_caps}")
            pilot_records = []
            for q in pilot_queries:
                if "A" in configs:
                    pilot_records.append(run_llm_only(client, q, current_caps["A_llm_only"]))
                if "B" in configs:
                    pilot_records.append(run_llm_rag(client, qdrant, q, current_caps["B_llm_rag"]))
                if "C" in configs:
                    pilot_records.append(run_full_pipeline(client, q, current_caps["C_full_pipeline"], G=G))
            means = per_config_means(pilot_records)
            print(f"[v2] iter {it+1} means: {means}")
            current_caps, converged = tune_budgets(means, current_caps, VERBOSITY_TOLERANCE_PCT)
            if converged:
                print(f"[v2] converged at iter {it+1}")
                break
        else:
            print("[v2] tuning did NOT converge — using last caps anyway")

    # Phase 2: full pass with tuned caps
    print(f"\n[v2] ── Phase 2: full pass  caps={current_caps} ──")
    records = []
    for q in queries:
        for cfg in ("A", "B", "C"):
            if cfg not in configs:
                continue
            try:
                if cfg == "A":
                    rec = run_llm_only(client, q, current_caps["A_llm_only"])
                elif cfg == "B":
                    rec = run_llm_rag(client, qdrant, q, current_caps["B_llm_rag"])
                else:
                    rec = run_full_pipeline(client, q, current_caps["C_full_pipeline"], G=G)
                records.append(rec)
                print(f"[v2] {cfg} {q['id']:<4}  toks={rec['completion_tokens']:>5}  "
                      f"rules={rec['n_design_rules']:>2}  gaps={rec['n_knowledge_gaps']:>2}  "
                      f"refusal={'yes' if rec['refusal_flag'] else 'no':<3}")
            except Exception as e:
                print(f"[v2] {cfg} {q['id']} FAILED: {e}")

    # Summary
    summary = _build_summary(records, current_caps)
    out_path = BASE_DIR / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary,
                   "tuned_caps": current_caps,
                   "model": LLM_MODEL}, f, indent=2, ensure_ascii=False)
    print(f"\n[v2] wrote {out_path}")
    print(json.dumps(summary, indent=2))

    if qdrant: qdrant.close()


def _build_summary(records: list[dict], caps: dict) -> dict:
    by_config = {}
    for r in records:
        c = r["config"]
        by_config.setdefault(c, []).append(r)

    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0

    out = {"per_config": {}, "tuned_caps": caps}
    for c, rs in by_config.items():
        out["per_config"][c] = {
            "n": len(rs),
            "mean_completion_tokens": int(avg([r["completion_tokens"] for r in rs])),
            "mean_prompt_tokens": int(avg([r["prompt_tokens"] for r in rs])),
            "mean_latency_s": avg([r["elapsed_s"] for r in rs]),
            "mean_cost_usd": round(avg([
                cost_usd(r["model"], r["prompt_tokens"], r["completion_tokens"])
                for r in rs
            ]), 4),
            "mean_rules": avg([r["n_design_rules"] for r in rs]),
            "mean_gaps":  avg([r["n_knowledge_gaps"] for r in rs]),
            "mean_summary_words": avg([r["summary_word_count"] for r in rs]),
            "refusal_count": sum(1 for r in rs if r["refusal_flag"]),
            "refusal_oos":   sum(1 for r in rs
                                  if r["refusal_flag"] and r["coverage"] == "out-of-scope"),
            "n_oos": sum(1 for r in rs if r["coverage"] == "out-of-scope"),
        }
    return out


if __name__ == "__main__":
    main()
