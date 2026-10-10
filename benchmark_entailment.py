"""benchmark_entailment.py — Quote → main_point entailment check.

Addresses Path B (condition c) of the Machine-Readability Defense:
"the source_quote provably supports the main_point."

For each frame in extractions_se_all.json, ask GPT-5 (low reasoning,
structured output) whether the source_quote entails the main_point in
the NLI sense (entailment / contradiction / neutral). Report the
overall entailment rate and the list of frames that fail.

This is a different signal from the LLM-judge's `extraction_accuracy`
rating (which is a 1-3 rubric assessment, ceiling-effect-prone). The
entailment check is a binary forced choice — a stricter test.

Usage:
    py benchmark_entailment.py                     # all 3,879 frames
    py benchmark_entailment.py --sample 100        # pilot subset
    py benchmark_entailment.py --resume            # resume from checkpoint

Cost: ~$20 for full corpus at gpt-5 low-reasoning (~$0.005/frame).
      ~$0.50 for n=100 pilot.
Time: ~30 min full corpus at 20 parallel workers, ~3 min pilot.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

if os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()
for _p in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"): os.environ.pop(_p, None)

BASE = Path(__file__).parent
EXTR = BASE / "archive" / "v1_streamlit_app" / "extractions_se_all.json"
OUT  = BASE / "benchmark_entailment_results.json"
CKPT = BASE / "benchmark_entailment_checkpoint.json"
MODEL = "gpt-5"

SYSTEM_PROMPT = """You are a strict natural-language inference judge.
Given a Premise (the verbatim source quote from a forum post) and a
Hypothesis (a 2-3 sentence summary of the transferable knowledge claim
in the post), decide whether the Premise entails the Hypothesis.

Return ONLY a JSON object:
{
  "entailment": "entailment" | "contradiction" | "neutral",
  "rationale": "<= 25 words"
}

Definitions:
- "entailment": the Premise straightforwardly supports the Hypothesis.
  Every claim in the Hypothesis can be inferred from the Premise alone,
  without external context.
- "contradiction": the Premise contradicts the Hypothesis (rare).
- "neutral": the Hypothesis adds claims the Premise does not support,
  or omits context that makes the Premise's claim conditional.

Be strict. If the Hypothesis adds detail not in the Premise (e.g., a
specific number, a generalization, a cause), return "neutral"."""

USER_TMPL = """Premise (source quote):
{q}

Hypothesis (main point):
{m}

Does Premise entail Hypothesis?"""

def check_one(client, frame):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":USER_TMPL.format(
                    q=frame.get("source_quote",""),
                    m=frame.get("main_point",""),
                )},
            ],
            response_format={"type":"json_object"},
            max_completion_tokens=512,
            reasoning_effort="low",
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            d = json.loads(raw)
        except Exception:
            d = {"entailment":"parse_error","rationale":raw[:100]}
        return {
            "frame_id": frame.get("frame_id",""),
            "post_id":  frame.get("post_id",""),
            "verdict":  d.get("entailment","parse_error"),
            "rationale": d.get("rationale","")[:200],
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    except Exception as e:
        return {
            "frame_id": frame.get("frame_id",""),
            "post_id":  frame.get("post_id",""),
            "verdict":  "error",
            "rationale": f"{type(e).__name__}: {e}",
            "prompt_tokens": 0, "completion_tokens": 0,
        }

def load_frames():
    with open(EXTR, encoding="utf-8") as f: d = json.load(f)
    posts = d.get("results", d) if isinstance(d, dict) else d
    out = []
    for p in posts:
        if not isinstance(p, dict): continue
        for fr in p.get("frames",[]) or []:
            if isinstance(fr,dict) and fr.get("source_quote") and fr.get("main_point"):
                out.append(fr)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0=all, N=first N frames")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    frames = load_frames()
    if args.sample: frames = frames[:args.sample]
    print(f"[entail] {len(frames)} frames to check")

    results, done_ids = [], set()
    if args.resume and CKPT.exists():
        with open(CKPT, encoding="utf-8") as f: results = json.load(f)
        done_ids = {r["frame_id"] for r in results}
        print(f"[entail] resumed: {len(done_ids)} already done")

    todo = [fr for fr in frames if fr.get("frame_id") not in done_ids]
    print(f"[entail] {len(todo)} remaining")

    client = OpenAI()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(check_one, client, fr): fr for fr in todo}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            results.append(r)
            if (i+1) % 50 == 0:
                with open(CKPT, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
                print(f"  [{i+1}/{len(todo)}] {(time.time()-t0)/60:.1f} min")

    # Save
    with open(OUT, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
    if CKPT.exists(): CKPT.unlink()

    # Summary
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    total = len(results)
    print(f"\n=== Entailment results (n={total}) ===")
    for v, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {v:<15} {c:>5}  ({c/total*100:.1f}%)")
    ent_pct = counts.get("entailment", 0) / total * 100 if total else 0
    print(f"\nEntailment rate: {ent_pct:.1f}% — paper-quotable number")

    # Cost
    total_in = sum(r["prompt_tokens"] for r in results)
    total_out = sum(r["completion_tokens"] for r in results)
    cost = total_in/1e6 * 1.25 + total_out/1e6 * 10
    print(f"Cost: ${cost:.2f}  (in={total_in}, out={total_out})")

if __name__ == "__main__":
    main()
