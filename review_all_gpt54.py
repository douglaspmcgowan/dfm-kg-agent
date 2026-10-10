"""
Run full-corpus LLM frame review with gpt-5.4 + medium reasoning effort.

Wraps review_all.py logic but adds `reasoning_effort` parameter to the API call,
which the base script doesn't expose. Output schema is identical to reviews_all.json
so disagreement analysis is apples-to-apples.

Usage:
    python review_all_gpt54.py --input <archive>/extractions_se_all.json --count 50
    python review_all_gpt54.py --input <archive>/extractions_se_all.json --all --resume

Environment:
    OPENAI_API_KEY        (required)
    OPENAI_MODEL          (default: gpt-5.4)
    REASONING_EFFORT      (default: medium)
    PARALLEL_REQUESTS     (default: 10)
    CHECKPOINT_INTERVAL   (default: 50)
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openai import OpenAI

# schema_se lives in the archive; add it to path
_HERE = Path(__file__).parent
_ARCHIVE = _HERE / "archive" / "v1_streamlit_app"
if _ARCHIVE.exists():
    sys.path.insert(0, str(_ARCHIVE))

from schema_review import FrameReview, REVIEW_SYSTEM_PROMPT, build_user_message
from review_all import load_extractions, build_frame_list, print_summary


def review_one_frame_with_reasoning(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    post: dict,
    frame: dict,
    global_idx: int,
    post_id: str,
    frame_index: int,
) -> dict:
    """Send a single frame for review with reasoning_effort parameter."""
    post_text = post.get("non_quoted_text")
    user_msg = build_user_message(frame, post, global_idx, post_text=post_text)

    kwargs = dict(
        model=model,
        seed=42,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=FrameReview,
    )
    if reasoning_effort and reasoning_effort.lower() != "none":
        kwargs["reasoning_effort"] = reasoning_effort

    completion = client.beta.chat.completions.parse(**kwargs)
    parsed: FrameReview = completion.choices[0].message.parsed
    result = parsed.model_dump()
    result["post_id"] = post_id
    result["frame_index"] = frame_index
    result["global_idx"] = global_idx

    # Token usage for cost analysis
    usage = completion.usage
    if usage is not None:
        result["_usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        # Reasoning tokens are nested in CompletionTokensDetails
        ctd = getattr(usage, "completion_tokens_details", None)
        if ctd is not None:
            result["_usage"]["reasoning_tokens"] = getattr(ctd, "reasoning_tokens", None)

    result["_meta"] = {
        "thread_title": post.get("thread_title", ""),
        "post_score": post.get("post_score"),
        "frame_type": frame.get("frame_type", ""),
        "scope": frame.get("scope", ""),
        "applicability": frame.get("applicability", ""),
        "source_quote_preview": frame.get("source_quote", "")[:100],
        "main_point_preview": frame.get("main_point", "")[:100],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--count", "-n", type=int, default=100)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if os.getenv("RESUME", "false").lower() == "true":
        args.resume = True

    if args.output is None:
        args.output = "reviews_all_gpt54.json" if args.all else f"reviews_pilot_{args.count}_gpt54.json"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    reasoning = os.getenv("REASONING_EFFORT", "medium")
    parallel = int(os.getenv("PARALLEL_REQUESTS", "10"))
    checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL", "50"))

    client = OpenAI(api_key=api_key)

    print(f"Loading extractions from: {args.input}")
    posts = load_extractions(args.input)
    total_frames = sum(len(p["frames"]) for p in posts)
    print(f"Loaded {len(posts)} posts with {total_frames} total frames.")

    if args.all:
        frame_list = build_frame_list(posts, count=None)
        print(f"Reviewing ALL {len(frame_list)} frames.")
    else:
        frame_list = build_frame_list(posts, count=args.count, seed=args.seed)
        print(f"Sampled {len(frame_list)} frames (seed={args.seed}).")

    if not frame_list:
        print("ERROR: No frames to review.", file=sys.stderr)
        sys.exit(1)

    checkpoint_file = args.output.replace(".json", "_checkpoint.json")
    done_keys = set()
    resumed_reviews = []

    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            cp_data = json.load(f)
        resumed_reviews = cp_data.get("reviews", [])
        done_keys = {(r["post_id"], r["frame_index"]) for r in resumed_reviews if "error" not in r}
        print(f"Resumed: {len(resumed_reviews)} reviews ({len(done_keys)} successful)")
    elif args.resume:
        print("Resume requested but no checkpoint found.")

    remaining = [f for f in frame_list if (f[3], f[4]) not in done_keys]
    print(f"Remaining: {len(remaining)} frames to review.")
    print(f"Model: {model}, reasoning_effort: {reasoning}, parallel: {parallel}")

    if not remaining:
        reviews_list = resumed_reviews
        elapsed = 0.0
    else:
        start_time = time.time()
        reviews_dict = {}
        progress_lock = threading.Lock()
        completed = [0]
        total_remaining = len(remaining)
        overall_tally = Counter()
        token_totals = Counter()
        last_print_time = [time.time()]

        def worker(task):
            post, frame, gidx, pid, fidx = task
            try:
                result = review_one_frame_with_reasoning(
                    client, model, reasoning, post, frame, gidx, pid, fidx
                )
            except Exception as e:
                result = {
                    "post_id": pid,
                    "frame_index": fidx,
                    "global_idx": gidx,
                    "error": f"{type(e).__name__}: {e}",
                }
            with progress_lock:
                completed[0] += 1
                n = completed[0]
                if "error" not in result:
                    overall_tally[result["overall"]] += 1
                    u = result.get("_usage", {})
                    for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                        if u.get(k):
                            token_totals[k] += u[k]
                if n % 10 == 0 or n == total_remaining or time.time() - last_print_time[0] > 30:
                    last_print_time[0] = time.time()
                    elapsed = time.time() - start_time
                    rate = n / elapsed if elapsed > 0 else 0
                    eta_s = (total_remaining - n) / rate if rate > 0 else 0
                    k = overall_tally.get("Keep", 0)
                    rv = overall_tally.get("Revise", 0)
                    rm = overall_tally.get("Remove", 0)
                    print(f"  [{n}/{total_remaining}] rate={rate:.2f}/s eta={eta_s/60:.1f}min "
                          f"Keep={k} Rev={rv} Rem={rm} "
                          f"prompt_tok={token_totals['prompt_tokens']} "
                          f"comp_tok={token_totals['completion_tokens']} "
                          f"reas_tok={token_totals['reasoning_tokens']}", flush=True)
            return (pid, fidx), result

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(worker, t): t for t in remaining}
            for future in concurrent.futures.as_completed(futures):
                key, result = future.result()
                reviews_dict[key] = result

                if checkpoint_interval > 0 and len(reviews_dict) % checkpoint_interval == 0:
                    cp_reviews = resumed_reviews + sorted(
                        reviews_dict.values(),
                        key=lambda r: r.get("global_idx", 0),
                    )
                    cp_data = {
                        "metadata": {
                            "model": model,
                            "reasoning_effort": reasoning,
                            "checkpoint": True,
                            "reviewed": len(cp_reviews),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        "reviews": cp_reviews,
                    }
                    tmp_file = checkpoint_file + ".tmp"
                    with open(tmp_file, "w", encoding="utf-8") as f:
                        json.dump(cp_data, f, ensure_ascii=False)
                    os.replace(tmp_file, checkpoint_file)

        elapsed = time.time() - start_time
        new_reviews = sorted(reviews_dict.values(), key=lambda r: r.get("global_idx", 0))
        reviews_list = resumed_reviews + new_reviews

    reviews_list.sort(key=lambda r: r.get("global_idx", 0))

    # Aggregate token usage
    total_prompt = sum(r.get("_usage", {}).get("prompt_tokens") or 0 for r in reviews_list)
    total_comp = sum(r.get("_usage", {}).get("completion_tokens") or 0 for r in reviews_list)
    total_reas = sum(r.get("_usage", {}).get("reasoning_tokens") or 0 for r in reviews_list)

    output = {
        "metadata": {
            "model": model,
            "reasoning_effort": reasoning,
            "total_frames_in_corpus": total_frames,
            "reviewed": len(reviews_list),
            "errors": len([r for r in reviews_list if "error" in r]),
            "seed": args.seed if not args.all else None,
            "mode": "all" if args.all else f"sample_{args.count}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_comp,
            "total_reasoning_tokens": total_reas,
        },
        "reviews": reviews_list,
    }

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(reviews_list)} reviews to {out_path}")
    print(f"Tokens: prompt={total_prompt:,}  completion={total_comp:,}  reasoning={total_reas:,}")

    if os.path.exists(checkpoint_file) and not args.resume:
        os.remove(checkpoint_file)

    print_summary(reviews_list, model, elapsed)


if __name__ == "__main__":
    main()
