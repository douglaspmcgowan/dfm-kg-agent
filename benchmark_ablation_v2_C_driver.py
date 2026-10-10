"""
Config C hardened driver — v2. Lessons from the prior run:

  - concurrent.futures timeout does NOT kill the underlying thread,
    and ThreadPoolExecutor.__exit__ blocks on shutdown(wait=True),
    which made the previous "hardened" runner hang anyway.

  - Subprocess timeout via Popen.communicate(timeout=N) + kill() DOES
    forcibly reap the wedged process. This driver spawns one Python
    subprocess per query (via benchmark_ablation_v2_C_worker.py), kills
    it on timeout, and moves on. Progressive save after each query.

Usage:
    python benchmark_ablation_v2_C_driver.py
    python benchmark_ablation_v2_C_driver.py --queries Q3,Q4
    python benchmark_ablation_v2_C_driver.py --timeout 360 --skip-done
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, tempfile
from pathlib import Path

HERE = Path(__file__).parent
WORKER = HERE / "benchmark_ablation_v2_C_worker.py"
OUT = HERE / "benchmark_ablation_v2_C_only.json"
LOG = HERE / "benchmark_ablation_v2_C_driver.log"

QUERIES_ALL = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
COVERAGE = {
    "Q1": "well-covered", "Q2": "well-covered", "Q3": "well-covered",
    "Q4": "sparse", "Q5": "sparse", "Q6": "sparse",
    "Q7": "out-of-scope", "Q8": "out-of-scope",
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    sys.stdout.write(line); sys.stdout.flush()
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def run_one_subprocess(query_id: str, cap: int, timeout_s: int) -> dict:
    """Spawn the worker subprocess. Kill it if it exceeds timeout."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
        dir=str(HERE)
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [sys.executable, str(WORKER), query_id, str(tmp_path), str(cap)]
        log(f"  spawn  {query_id}  cap={cap}  timeout={timeout_s}s")
        t0 = time.time()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log(f"  ✗ {query_id} TIMED OUT after {timeout_s}s — killing PID {proc.pid}")
            proc.kill()
            try:
                stdout, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                stdout = "(no output captured after kill)"
            return {
                "config": "C_full_pipeline", "query_id": query_id,
                "coverage": COVERAGE.get(query_id, "?"), "model": "gpt-5",
                "error": f"hard_timeout_{timeout_s}s",
                "elapsed_s": timeout_s, "prompt_tokens": 0,
                "completion_tokens": 0, "n_cited_frames": 0,
                "n_design_rules": 0, "n_knowledge_gaps": 0,
                "summary_word_count": 0, "refusal_flag": False,
                "answer_json": {}, "worker_stdout_tail": stdout[-1000:],
            }
        elapsed = time.time() - t0
        if proc.returncode != 0:
            log(f"  ✗ {query_id} exit code {proc.returncode}")
            return {
                "config": "C_full_pipeline", "query_id": query_id,
                "coverage": COVERAGE.get(query_id, "?"), "model": "gpt-5",
                "error": f"exit_code_{proc.returncode}",
                "elapsed_s": round(elapsed, 1), "prompt_tokens": 0,
                "completion_tokens": 0, "n_cited_frames": 0,
                "n_design_rules": 0, "n_knowledge_gaps": 0,
                "summary_word_count": 0, "refusal_flag": False,
                "answer_json": {}, "worker_stdout_tail": stdout[-1000:],
            }
        # Load the result the worker dropped.
        try:
            rec = json.loads(tmp_path.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"  ✗ {query_id} could not parse worker output: {e}")
            return {
                "config": "C_full_pipeline", "query_id": query_id,
                "coverage": COVERAGE.get(query_id, "?"), "model": "gpt-5",
                "error": f"parse_error: {e}",
                "elapsed_s": round(elapsed, 1), "prompt_tokens": 0,
                "completion_tokens": 0, "n_cited_frames": 0,
                "n_design_rules": 0, "n_knowledge_gaps": 0,
                "summary_word_count": 0, "refusal_flag": False,
                "answer_json": {}, "worker_stdout_tail": stdout[-1000:],
            }
        log(f"  ✓ {query_id}  {elapsed:.1f}s  toks={rec['completion_tokens']}  "
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
    ap.add_argument("--timeout", type=int, default=360,
                    help="Hard per-query timeout (default 360s = 6 min).")
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()

    queries = list(QUERIES_ALL)
    if args.queries:
        wanted = set(args.queries.split(","))
        queries = [q for q in QUERIES_ALL if q in wanted]

    # Resume
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
        rec = run_one_subprocess(q, args.cap, args.timeout)
        records.append(rec)
        OUT.write_text(json.dumps(
            {"records": records, "model": "gpt-5", "cap": args.cap},
            indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"SAVED  n={len(records)}")
    log(f"DONE  {(time.time()-t_panel)/60:.1f} min")

    # Summary
    ok = [r for r in records if "error" not in r]
    log(f"SUMMARY  ok={len(ok)}  err={len(records)-len(ok)}")
    if ok:
        oos = [r for r in ok if r["coverage"] == "out-of-scope"]
        oos_refusals = sum(1 for r in oos if r.get("refusal_flag"))
        log(f"  OOS refusals: {oos_refusals}/{len(oos)}")


if __name__ == "__main__":
    main()
