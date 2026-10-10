"""Single-query worker for Config C. Designed to be spawned by the
hardened runner as a subprocess, so a hard OS-level kill is possible
when an OpenAI call wedges past the SDK's own timeout.

Reads QUERY_ID from argv[1], writes the single result record to argv[2]
(JSON). Exits 0 on success, non-zero on error.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Strip trailing whitespace from key — Windows env vars often have a stray \n.
if os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()
for _p in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_p, None)

from benchmark_ablation_v2 import (
    QUERIES, run_full_pipeline, INIT_MAX_OUTPUT_TOKENS,
)
from openai import OpenAI


def main():
    if len(sys.argv) < 3:
        print("usage: benchmark_ablation_v2_C_worker.py QUERY_ID OUT_PATH [CAP]")
        sys.exit(2)
    query_id = sys.argv[1]
    out_path = Path(sys.argv[2])
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else INIT_MAX_OUTPUT_TOKENS["C_full_pipeline"]

    query = next((q for q in QUERIES if q["id"] == query_id), None)
    if query is None:
        print(f"unknown query id: {query_id}")
        sys.exit(2)

    print(f"[worker] {query_id}: loading graph ...", flush=True)
    import pickle
    with open(HERE / "provenance_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    print(f"[worker] {query_id}: graph loaded ({len(G.nodes)} nodes)", flush=True)

    client = OpenAI()
    t0 = time.time()
    print(f"[worker] {query_id}: calling run_full_pipeline ...", flush=True)
    rec = run_full_pipeline(client, query, cap, G=G)
    elapsed = time.time() - t0
    print(f"[worker] {query_id}: done in {elapsed:.1f}s  toks={rec['completion_tokens']}  "
          f"rules={rec['n_design_rules']}  gaps={rec['n_knowledge_gaps']}", flush=True)

    out_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    print(f"[worker] {query_id}: wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
