"""
benchmark_ablation_v2_C_hardened.py — Config C only, hardened.

Why this exists: the prior v2 C run hung for 94 min wall / 47 CPU-s, which is
the classic "OpenAI HTTP call wedged past timeout" pattern. This runner adds:

  1. Hard per-query timeout (4 min) via concurrent.futures, not the OpenAI
     SDK's own timeout (which has been observed to wedge).
  2. Progressive save after EACH query (so a single bad query can't lose all
     work).
  3. Verbose, flushed stdout (so we see exactly where any future hang is).
  4. No tuning phase, no pilot. Single sequential pass through Q1..Q8.
  5. --skip-done to resume from partial output.

Usage:
    python benchmark_ablation_v2_C_hardened.py
    python benchmark_ablation_v2_C_hardened.py --queries Q1,Q2
    python benchmark_ablation_v2_C_hardened.py --timeout 240 --skip-done
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

# Reuse the v2 building blocks.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from benchmark_ablation_v2 import (
    QUERIES, run_full_pipeline, cost_usd, LLM_MODEL, INIT_MAX_OUTPUT_TOKENS,
)
from openai import OpenAI

# Strip the trailing newline that Windows env vars sometimes carry.
if os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()
for _p in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_p, None)

OUT = HERE / "benchmark_ablation_v2_C_only.json"


def log(msg: str) -> None:
    """Flushed log — so we see progress in real time."""
    sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stdout.flush()


def load_graph():
    """Load provenance graph once and reuse across queries."""
    import pickle
    pkl = HERE / "provenance_graph.gpickle"
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl} not found.")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run_one_with_timeout(client: OpenAI, query: dict, cap: int, G,
                         timeout_s: int) -> dict:
    """Run a single Config C call in a worker thread with a hard timeout."""
    log(f"  → Q={query['id']} starting (cap={cap}, timeout={timeout_s}s)")
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(run_full_pipeline, client, query, cap, G)
        try:
            rec = fut.result(timeout=timeout_s)
            log(f"  ← Q={query['id']} OK  toks={rec.get('completion_tokens', '?')}  "
                f"rules={rec.get('n_design_rules', '?')}  gaps={rec.get('n_knowledge_gaps', '?')}")
            return rec
        except FuturesTimeout:
            log(f"  ✗ Q={query['id']} TIMED OUT after {timeout_s}s — abandoning thread")
            return {
                "config": "C_full_pipeline", "query_id": query["id"],
                "coverage": query["coverage"], "model": LLM_MODEL,
                "error": f"hard_timeout_{timeout_s}s",
                "elapsed_s": timeout_s, "prompt_tokens": 0,
                "completion_tokens": 0, "n_cited_frames": 0,
                "n_design_rules": 0, "n_knowledge_gaps": 0,
                "summary_word_count": 0, "refusal_flag": False,
                "answer_json": {},
            }
        except Exception as e:
            log(f"  ✗ Q={query['id']} FAILED: {type(e).__name__}: {e}")
            return {
                "config": "C_full_pipeline", "query_id": query["id"],
                "coverage": query["coverage"], "model": LLM_MODEL,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "n_cited_frames": 0,
                "n_design_rules": 0, "n_knowledge_gaps": 0,
                "summary_word_count": 0, "refusal_flag": False,
                "answer_json": {},
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=None,
                    help="Comma-separated query IDs (default: all 8).")
    ap.add_argument("--timeout", type=int, default=240,
                    help="Hard per-query timeout in seconds (default 240 = 4 min).")
    ap.add_argument("--skip-done", action="store_true",
                    help="Skip queries already in the output file.")
    ap.add_argument("--cap", type=int,
                    default=INIT_MAX_OUTPUT_TOKENS["C_full_pipeline"],
                    help="max_completion_tokens for the resynth call (default 4096).")
    args = ap.parse_args()

    queries = list(QUERIES)
    if args.queries:
        wanted = set(args.queries.split(","))
        queries = [q for q in QUERIES if q["id"] in wanted]

    log(f"START  model={LLM_MODEL}  cap={args.cap}  timeout={args.timeout}s  "
        f"queries={[q['id'] for q in queries]}")

    # Load prior results if resuming.
    prior_records: list[dict] = []
    done_ids: set[str] = set()
    if args.skip_done and OUT.exists():
        try:
            blob = json.loads(OUT.read_text(encoding="utf-8"))
            prior_records = blob.get("records", [])
            done_ids = {r["query_id"] for r in prior_records if "error" not in r}
            log(f"RESUME  skipping {len(done_ids)} already-done queries: {sorted(done_ids)}")
        except Exception as e:
            log(f"RESUME warning: could not parse existing output ({e}); starting fresh")

    log("Loading provenance graph ...")
    G = load_graph()
    log(f"Loaded graph: {len(G.nodes)} nodes")

    client = OpenAI()  # uses default timeout

    records = list(prior_records)
    t_panel = time.time()
    for q in queries:
        if q["id"] in done_ids:
            log(f"SKIP  {q['id']} (already done)")
            continue
        rec = run_one_with_timeout(client, q, args.cap, G, args.timeout)
        records.append(rec)

        # Progressive save after each query.
        OUT.write_text(json.dumps(
            {"records": records, "model": LLM_MODEL, "cap": args.cap},
            indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"SAVED  {len(records)} records to {OUT.name}")

    elapsed = (time.time() - t_panel) / 60
    log(f"DONE   {elapsed:.1f} min total wall time")

    # Summary
    ok = [r for r in records if "error" not in r]
    err = [r for r in records if "error" in r]
    log(f"SUMMARY  ok={len(ok)}  err={len(err)}")
    if ok:
        mean_toks = sum(r["completion_tokens"] for r in ok) / len(ok)
        mean_lat = sum(r["elapsed_s"] for r in ok) / len(ok)
        mean_rules = sum(r["n_design_rules"] for r in ok) / len(ok)
        mean_gaps = sum(r["n_knowledge_gaps"] for r in ok) / len(ok)
        oos_records = [r for r in ok if r["coverage"] == "out-of-scope"]
        oos_refusals = sum(1 for r in oos_records if r["refusal_flag"])
        log(f"  mean_tokens={mean_toks:.0f}  mean_latency={mean_lat:.1f}s  "
            f"mean_rules={mean_rules:.1f}  mean_gaps={mean_gaps:.1f}  "
            f"oos_refusals={oos_refusals}/{len(oos_records)}")


if __name__ == "__main__":
    main()
