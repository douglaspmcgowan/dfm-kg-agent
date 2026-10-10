"""benchmark_repeatability.py — GPT-5 judge repeatability test.

Re-runs the LLM-judge (gpt-5, default reasoning) on a fixed 10-frame
sample N=3 times under identical prompt, schema, and SEED=42, and
reports per-frame and per-dimension agreement across runs.

This isolates STOCHASTICITY of the judge from the variance we see
across model families (the gpt-5 vs gpt-5.4 ablation already covered).

Cost: ~$0.30 (30 judge calls × $0.01 each)
Time: ~5 min wall clock

Usage:
    $env:OPENAI_API_KEY = ([Environment]::GetEnvironmentVariable("OPENAI_API_KEY","User")).Trim()
    py benchmark_repeatability.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field

# strip newlines from key
if os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()
for _p in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"): os.environ.pop(_p, None)

BASE_DIR = Path(__file__).parent
N_RUNS = 3
N_FRAMES = 10
SEED = 42

class JudgeRating(BaseModel):
    extraction_accuracy: int = Field(ge=1, le=3)
    tribal_knowledge: int = Field(ge=1, le=3)
    clarity: int = Field(ge=1, le=3)
    frame_type_rating: int = Field(ge=1, le=5)
    post_role_rating: int = Field(ge=1, le=5)
    scope_rating: int = Field(ge=1, le=5)
    applicability_rating: int = Field(ge=1, le=5)
    main_point_quality: int = Field(ge=1, le=5)
    epistemic_stance_rating: int = Field(ge=1, le=5)
    overall: str  # Keep / Revise / Remove
    comments: str = ""

SYSTEM_PROMPT = """You are an expert engineering reviewer rating extracted DFM knowledge frames. Apply the rubric below strictly and consistently. Return ONLY a JSON object conforming to the JudgeRating schema; do not add commentary.

Rubric (1-3 scale): extraction_accuracy (1=misrepresents; 3=accurately captures); tribal_knowledge (1=textbook; 3=experiential); clarity (1=ambiguous; 3=clear).
Rubric (1-5 scale): for frame_type/post_role/scope/applicability/main_point_quality/epistemic_stance, rate label appropriateness (1=wrong; 3=acceptable; 5=excellent).
Overall: Keep / Revise / Remove."""

USER_TEMPLATE = """Frame to rate:

Source quote: "{source_quote}"
Main point: {main_point}
Frame type: {frame_type}
Post role: {post_role}
Scope: {scope}
Applicability: {applicability}
Epistemic stance: {epistemic_stance}
Subject: {subject}

Rate the frame on each dimension and return the JSON."""

def load_sample():
    extr_path = BASE_DIR / "archive" / "v1_streamlit_app" / "extractions_se_all.json"
    if not extr_path.exists():
        extr_path = BASE_DIR / "extractions_se_all.json"
    with open(extr_path, encoding="utf-8") as f:
        all_extr = json.load(f)
    posts = all_extr.get("results", all_extr) if isinstance(all_extr, dict) else all_extr
    all_frames = []
    for post in posts:
        if not isinstance(post, dict): continue
        for fr in post.get("frames", []) or []:
            if isinstance(fr, dict) and fr.get("source_quote") and fr.get("main_point"):
                all_frames.append(fr)
    print(f"Total frames available: {len(all_frames)}")
    import random
    random.seed(SEED)
    return random.sample(all_frames, N_FRAMES)

def rate(client, frame, seed):
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":USER_TEMPLATE.format(
                source_quote=frame.get("source_quote",""),
                main_point=frame.get("main_point",""),
                frame_type=frame.get("frame_type",""),
                post_role=frame.get("post_role",""),
                scope=frame.get("scope",""),
                applicability=frame.get("applicability",""),
                epistemic_stance=frame.get("epistemic_stance",""),
                subject=frame.get("subject",""),
            )},
        ],
        response_format={"type":"json_object"},
        max_completion_tokens=4096,
        reasoning_effort="low",
        seed=seed,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw), resp.usage.prompt_tokens, resp.usage.completion_tokens
    except Exception:
        return {}, resp.usage.prompt_tokens, resp.usage.completion_tokens

def main():
    sample = load_sample()
    print(f"Sampled {len(sample)} frames, running {N_RUNS} times each.")
    client = OpenAI()
    results = []  # [(frame_idx, run, rating_dict)]
    for frame_idx, frame in enumerate(sample):
        for run in range(N_RUNS):
            t0 = time.time()
            rating, pin, pout = rate(client, frame, seed=SEED+run)
            results.append({
                "frame_idx": frame_idx,
                "frame_id": frame.get("frame_id",""),
                "run": run,
                "rating": rating,
                "prompt_tokens": pin,
                "completion_tokens": pout,
                "elapsed_s": round(time.time()-t0,1),
            })
            print(f"  frame {frame_idx} run {run}: {rating.get('overall','?')} "
                  f"toks={pout} elapsed={time.time()-t0:.1f}s")
    out = BASE_DIR / "benchmark_repeatability_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"sample": [s.get("frame_id","") for s in sample], "results": results}, f, indent=2)
    print(f"\nWrote {out}")

    # Quick per-frame agreement summary
    fields_1to3 = ["extraction_accuracy","tribal_knowledge","clarity"]
    fields_1to5 = ["frame_type_rating","post_role_rating","scope_rating",
                  "applicability_rating","main_point_quality","epistemic_stance_rating"]
    fields = fields_1to3 + fields_1to5

    from collections import defaultdict
    by_frame = defaultdict(list)
    for r in results:
        by_frame[r["frame_idx"]].append(r["rating"])

    print("\n=== Per-frame intra-judge agreement ===")
    print(f"{'Field':<28} {'pct_all_3_same':>15} {'pct_within_1':>15}")
    for fld in fields:
        all_same = 0
        within_1 = 0
        n = 0
        for fi, runs in by_frame.items():
            vals = [int(r.get(fld,0) or 0) for r in runs]
            if len(set(vals)) <= 1: all_same += 1
            if max(vals) - min(vals) <= 1: within_1 += 1
            n += 1
        print(f"{fld:<28} {all_same/n*100:>13.0f}% {within_1/n*100:>14.0f}%")

    # Overall verdict stability
    print("\nOverall verdict stability (all 3 runs same Keep/Revise/Remove):")
    same = 0
    for fi, runs in by_frame.items():
        verdicts = [r.get("overall","?") for r in runs]
        if len(set(verdicts)) <= 1: same += 1
    print(f"  {same}/{len(by_frame)} frames ({same/len(by_frame)*100:.0f}%)")

if __name__ == "__main__":
    main()
