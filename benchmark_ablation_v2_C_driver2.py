"""Simpler driver: subprocess.run with DEVNULL stdout (no pipe-buffer
issues), longer timeout, no auxiliary log file (only stdout that the
shell captures).

Why this exists: the previous driver wedged silently. Suspected causes:
  - OS pipe-buffer filling from chatty worker stdout
  - File-write hang on G:\ for the auxiliary log
  - subprocess.communicate() interaction with Windows pipes

This version sidesteps both:
  - stdout=DEVNULL on the subprocess (the worker can chatter all it wants
    without filling any buffer); errors come through the JSON output file
    (or absence thereof).
  - sys.stdout.flush() after every print() — shell-captured stdout only.

Usage:
    python benchmark_ablation_v2_C_driver2.py --skip-done --timeout 480
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, tempfile
from pathlib import Path

HERE = Path(__file__).parent
WORKER = HERE / "benchmark_ablation_v2_C_worker.py"
OUT = HERE / "benchmark_ablation_v2_C_only.json"

QUERIES_ALL = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
COVERAGE = {
    "Q1": "well-covered", "Q2": "well-covered", "Q3": "well-covered",
    "Q4": "sparse", "Q5": "sparse", "Q6": "sparse",
    "Q7": "out-of-scope", "Q8": "out-of-scope",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(query_id: str, cap: int, timeout_s: int) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(HERE)
    ) as tmp:
        tmp_path = Path(tmp.name)
    t0 = time.time()
    try:
        log(f"  spawn  {query_id}  cap={cap}  timeout={timeout_s}s  tmp={tmp_path.name}")
        cmd = [sys.executable, str(WORKER), query_id, str(tmp_path), str(cap)]
        # stdout=DEVNULL prevents pipe-buffer deadlock; stderr=DEVNULL too.
        # Errors propagate via non-zero exit + missing tmp file.
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout_s,
            )
            elapsed = time.time() - t0
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            log(f"  ✗ {query_id} TIMED OUT after {elapsed:.0f}s")
            return {
                "config": "C_full_pipeline", "query_id": query_id,
                "coverage": COVERAGE.get(query_id, "?"), "model": "gpt-5",
                "error": f"hard_timeout_{timeout_s}s",
                "elapsed_s": round(elapsed, 1), "prompt_tokens": 0,
                "completion_tokens": 0, "n_cited_frames": 0,
                "n_design_rules": 0, "n_knowledge_gaps": 0,
                "summary_word_count": 0, "refusal_flag": False,
                "answer_json": {},
            }
        if proc.returncode != 0:
            log(f"  ✗ {query_id} exit {proc.returncode} after {elapsed:.0f}s")
            return {
                "config": "C_full_pipeline", "query_id": query_id,
                "coverage": COVERAGE.get(query_id, "?"), "model": "gpt-5",
                "error": f"exit_code_{proc.returncode}",
                "elapsed_s": round(elapsed, 1), "prompt_tokens": 0,
                "completion_tokens": 0, "n_cited_frames": 0,
                "n_design_rules": 0, "n_knowledge_gaps": 0,
                "summary_word_count": 0, "refusal_flag": False,
                "answer_json": {},
            }
        try:
            rec = json.loads(tmp_path.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"  ✗ {query_id} parse error: {e}")
            return {
                "config": "C_full_pipeline", "query_id": query_id,
                "coverage": COVERAGE.get(query_id, "?"), "model": "gpt-5",
                "error": f"parse_error: {e}",
                "elapsed_s": round(elapsed, 1), "prompt_tokens": 0,
                "completion_tokens": 0, "n_cited_frames": 0,
                "n_design_rules": 0, "n_knowledge_gaps": 0,
                "summary_word_count": 0, "refusal_flag": False,
                "answer_json": {},
            }
        log(f"  ✓ {query_id}  {elapsed:.0f}s  toks={rec['completion_tokens']}  "
            f"rules={rec['n_design_rules']}  gaps={rec['n_knowledge_gaps']}  "
            f"refusal={'yes' if rec.get('refusal_flag') else 'no'}")
        return rec
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=None)
    ap.add_argument("--timeout", type=int, default=480,
                    help="Hard per-query timeout (default 480s = 8 min).")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()

    queries = list(QUERIES_ALL)
    if args.queries:
        wanted = set(args.queries.split(","))
        queries = [q for q in QUERIES_ALL if q in wanted]

    records: list[dict] = []
    done_ids: set[str] = set()
    if args.skip_done and OUT.exists():
        try:
            blob = json.loads(OUT.read_text(encoding="utf-8"))
            records = blob.get("records", [])
            done_ids = {r["query_id"] for r in records if "error" not in r}
            log(f"RESUME  {len(done_ids)} done: {sorted(done_ids)}")
        except Exception as e:
            log(f"RESUME failed ({e}); starting fresh")

    log(f"START  queries={queries}  cap={args.cap}  timeout={args.timeout}s")
    t_panel = time.time()
    for q in queries:
        if q in done_ids:
            log(f"SKIP  {q}")
            continue
        rec = run_one(q, args.cap, args.timeout)
        records.append(rec)
        OUT.write_text(json.dumps(
            {"records": records, "model": "gpt-5", "cap": args.cap},
            indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"SAVED  n={len(records)}")
    log(f"DONE  {(time.time()-t_panel)/60:.1f} min total")

    ok = [r for r in records if "error" not in r]
    log(f"SUMMARY  ok={len(ok)}  err={len(records)-len(ok)}")
    if ok:
        oos = [r for r in ok if r["coverage"] == "out-of-scope"]
        oos_refusals = sum(1 for r in oos if r.get("refusal_flag"))
        log(f"  OOS refusals: {oos_refusals}/{len(oos)}")


if __name__ == "__main__":
    main()
