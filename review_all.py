"""
Full-corpus LLM frame reviewer -- reviews all (or N) frames from the extraction corpus.

Phase 1 of the review-revise pipeline (Phase 2 = revise_frames.py).

  - 9-dimension rubric-guided evaluation via Pydantic structured output
  - Calibration guidance in system prompt from 26-frame human expert gold standard
  - Uses (post_id, frame_index) composite key for stable frame addressing
  - Includes non_quoted_text in review prompt for source_quote verification
  - Checkpoint/resume support for large runs
  - Paper-ready summary statistics

Usage:
    python review_all.py --input extractions_se_all.json --count 100 [--seed 42]
    python review_all.py --input extractions_se_all.json --all [--resume]

Environment variables:
    OPENAI_API_KEY      (required)
    OPENAI_MODEL        (default: gpt-5)
    PARALLEL_REQUESTS   (default: 10)
    CHECKPOINT_INTERVAL (default: 50)
    RESUME              (default: false)
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

from schema_review import FrameReview, REVIEW_SYSTEM_PROMPT, build_user_message


# ---------------------------------------------------------------------------
# Load extraction data
# ---------------------------------------------------------------------------
def load_extractions(filepath: str) -> list[dict]:
    """Load the extraction JSON file and return the results list."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats: {"results": [...]} and flat list
    if isinstance(data, dict):
        results = data.get("results", [])
    elif isinstance(data, list):
        results = data
    else:
        results = []

    # Filter to posts with frames
    return [r for r in results if r.get("frames")]


# ---------------------------------------------------------------------------
# Build flat frame list
# ---------------------------------------------------------------------------
def build_frame_list(
    posts: list[dict],
    count: int | None = None,
    seed: int = 42,
) -> list[tuple[dict, dict, int, str, int]]:
    """
    Build a flat list of (post, frame, global_idx, post_id, frame_index_within_post).

    If count is None, returns all frames. Otherwise, randomly samples count frames.
    """
    all_frames = []
    global_idx = 0
    for post in posts:
        for fi, frame in enumerate(post["frames"]):
            global_idx += 1
            all_frames.append((post, frame, global_idx, post["post_id"], fi))

    if count is not None and count < len(all_frames):
        rng = random.Random(seed)
        rng.shuffle(all_frames)
        all_frames = all_frames[:count]
        # Re-sort by global_idx for consistent ordering
        all_frames.sort(key=lambda x: x[2])

    return all_frames


# ---------------------------------------------------------------------------
# Review one frame
# ---------------------------------------------------------------------------
def review_one_frame(
    client: OpenAI,
    model: str,
    post: dict,
    frame: dict,
    global_idx: int,
    post_id: str,
    frame_index: int,
) -> dict:
    """Send a single frame to the LLM for review."""
    # Include non_quoted_text for source_quote verification
    post_text = post.get("non_quoted_text")

    user_msg = build_user_message(frame, post, global_idx, post_text=post_text)

    completion = client.beta.chat.completions.parse(
        model=model,
        seed=42,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=FrameReview,
    )

    parsed: FrameReview = completion.choices[0].message.parsed
    result = parsed.model_dump()
    result["post_id"] = post_id
    result["frame_index"] = frame_index
    result["global_idx"] = global_idx

    # Frame metadata snapshot for analysis
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


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_summary(reviews: list[dict], model: str, elapsed: float):
    """Print paper-ready summary statistics."""
    valid = [r for r in reviews if "error" not in r]
    errors = [r for r in reviews if "error" in r]

    print(f"\n{'='*60}")
    print("REVIEW SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Frames reviewed: {len(valid)}")
    print(f"Errors: {len(errors)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(len(valid),1):.2f}s/frame)")

    if not valid:
        return

    # Overall judgment
    overall_dist = Counter(r["overall"] for r in valid)
    n = len(valid)
    print(f"\nOverall Judgment:")
    for label in ["Keep", "Revise", "Remove"]:
        c = overall_dist.get(label, 0)
        print(f"  {label:<8} {c:>5} ({100*c/n:.1f}%)")

    # Content ratings (1-3)
    print("\nKnowledge Content (1-3 scale):")
    print(f"  {'Field':<25} {'Mean':>6} {'Median':>7} {'SD':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*6}")
    for field in ["extraction_accuracy", "tribal_knowledge", "clarity"]:
        vals = [r[field] for r in valid if field in r]
        if vals:
            print(f"  {field:<25} {np.mean(vals):>6.2f} {np.median(vals):>7.1f} {np.std(vals, ddof=1):>6.2f}")

    # Metadata ratings (1-5)
    print("\nMetadata Accuracy (1-5 scale):")
    print(f"  {'Field':<25} {'Mean':>6} {'Median':>7} {'SD':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*6}")
    for field in ["frame_type_rating", "post_role_rating", "scope_rating",
                   "applicability_rating", "main_point_quality", "epistemic_stance_rating"]:
        vals = [r[field] for r in valid if field in r]
        if vals:
            print(f"  {field:<25} {np.mean(vals):>6.2f} {np.median(vals):>7.1f} {np.std(vals, ddof=1):>6.2f}")

    # Breakdown by frame type
    print("\nOverall judgment by frame type:")
    ft_groups = {}
    for r in valid:
        ft = r.get("_meta", {}).get("frame_type", "unknown")
        ft_groups.setdefault(ft, []).append(r)

    for ft in sorted(ft_groups.keys()):
        group = ft_groups[ft]
        dist = Counter(r["overall"] for r in group)
        print(f"  {ft:<20} n={len(group):>3}  "
              f"Keep={dist.get('Keep',0):>3}  "
              f"Revise={dist.get('Revise',0):>3}  "
              f"Remove={dist.get('Remove',0):>3}")

    # Scope rating by assigned scope
    print("\nScope rating by assigned scope:")
    scope_groups = {}
    for r in valid:
        sc = r.get("_meta", {}).get("scope", "unknown")
        scope_groups.setdefault(sc, []).append(r)

    for sc in sorted(scope_groups.keys(), key=lambda s: -len(scope_groups[s])):
        group = scope_groups[sc]
        vals = [r["scope_rating"] for r in group if "scope_rating" in r]
        if vals:
            print(f"  {sc:<25} n={len(group):>3}  scope_rating mean={np.mean(vals):.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM frame reviewer -- review all or N frames from the corpus."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to extraction JSON file.",
    )
    parser.add_argument(
        "--count", "-n", type=int, default=100,
        help="Number of frames to review (default: 100). Ignored if --all is set.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Review ALL frames in the corpus.",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for sampling (default: 42). Ignored if --all is set.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON file (default: reviews_test_{count}.json or reviews_all.json).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint file.",
    )

    args = parser.parse_args()

    # Also check env var for resume
    if os.getenv("RESUME", "false").lower() == "true":
        args.resume = True

    # Determine output filename
    if args.output is None:
        if args.all:
            args.output = "reviews_all.json"
        else:
            args.output = f"reviews_test_{args.count}.json"

    # Environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    parallel = int(os.getenv("PARALLEL_REQUESTS", "10"))
    checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL", "50"))

    client = OpenAI(api_key=api_key)

    # Load extractions
    print(f"Loading extractions from: {args.input}")
    posts = load_extractions(args.input)
    total_frames = sum(len(p["frames"]) for p in posts)
    print(f"Loaded {len(posts)} posts with {total_frames} total frames.")

    # Build frame list
    if args.all:
        frame_list = build_frame_list(posts, count=None)
        print(f"Reviewing ALL {len(frame_list)} frames.")
    else:
        frame_list = build_frame_list(posts, count=args.count, seed=args.seed)
        print(f"Sampled {len(frame_list)} frames (seed={args.seed}).")

    if not frame_list:
        print("ERROR: No frames to review.", file=sys.stderr)
        sys.exit(1)

    # Resume from checkpoint
    checkpoint_file = args.output.replace(".json", "_checkpoint.json")
    done_keys = set()
    resumed_reviews = []

    if args.resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            cp_data = json.load(f)
        resumed_reviews = cp_data.get("reviews", [])
        done_keys = {(r["post_id"], r["frame_index"]) for r in resumed_reviews if "error" not in r}
        print(f"Resumed: {len(resumed_reviews)} reviews from checkpoint "
              f"({len(done_keys)} successful)")
    elif args.resume:
        print("Resume requested but no checkpoint found. Starting fresh.")

    # Filter to remaining frames
    remaining = [f for f in frame_list if (f[3], f[4]) not in done_keys]
    print(f"Remaining: {len(remaining)} frames to review.")

    if not remaining:
        print("All frames already reviewed. Outputting final results.")
        reviews_list = resumed_reviews
    else:
        # Review frames in parallel
        print(f"Reviewing with {model} (parallel={parallel})...")
        start_time = time.time()

        reviews_dict = {}
        progress_lock = threading.Lock()
        completed = [0]
        total_remaining = len(remaining)
        overall_tally = Counter()
        rating_sums = Counter()

        def worker(task):
            post, frame, gidx, pid, fidx = task
            result = review_one_frame(client, model, post, frame, gidx, pid, fidx)
            with progress_lock:
                completed[0] += 1
                n = completed[0]
                if "error" not in result:
                    overall_tally[result["overall"]] += 1
                    for f in ["extraction_accuracy", "tribal_knowledge", "clarity"]:
                        if f in result:
                            rating_sums[f] += result[f]

                # Running tally every 25 frames or at end
                if n % 25 == 0 or n == total_remaining:
                    k = overall_tally.get("Keep", 0)
                    rv = overall_tally.get("Revise", 0)
                    rm = overall_tally.get("Remove", 0)
                    pct_keep = 100 * k / n if n else 0
                    valid_n = k + rv + rm
                    stats = (f"  [{n}/{total_remaining}] "
                             f"Keep={k} ({pct_keep:.0f}%) "
                             f"Revise={rv} Remove={rm}")
                    if valid_n > 0:
                        ea_mean = rating_sums["extraction_accuracy"] / valid_n
                        tk_mean = rating_sums["tribal_knowledge"] / valid_n
                        stats += f"  EA={ea_mean:.2f} TK={tk_mean:.2f}"
                    print(stats)
            return (pid, fidx), result

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(worker, t): t for t in remaining}
            for future in concurrent.futures.as_completed(futures):
                try:
                    key, result = future.result()
                    reviews_dict[key] = result
                except Exception as e:
                    task = futures[future]
                    pid, fidx, gidx = task[3], task[4], task[2]
                    print(f"  ERROR on frame {gidx} ({pid}:{fidx}): "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
                    reviews_dict[(pid, fidx)] = {
                        "post_id": pid,
                        "frame_index": fidx,
                        "global_idx": gidx,
                        "error": str(e),
                    }

                # Checkpoint (atomic write: temp file -> rename)
                if checkpoint_interval > 0 and len(reviews_dict) % checkpoint_interval == 0:
                    cp_reviews = resumed_reviews + sorted(
                        reviews_dict.values(),
                        key=lambda r: r.get("global_idx", 0),
                    )
                    cp_data = {
                        "metadata": {
                            "model": model,
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

        # Merge with resumed
        new_reviews = sorted(reviews_dict.values(), key=lambda r: r.get("global_idx", 0))
        reviews_list = resumed_reviews + new_reviews

    # Sort by global_idx
    reviews_list.sort(key=lambda r: r.get("global_idx", 0))

    # Build output
    output = {
        "metadata": {
            "model": model,
            "total_frames_in_corpus": total_frames,
            "reviewed": len(reviews_list),
            "errors": len([r for r in reviews_list if "error" in r]),
            "seed": args.seed if not args.all else None,
            "mode": "all" if args.all else f"sample_{args.count}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "reviews": reviews_list,
    }

    # Write output
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(reviews_list)} reviews to {out_path}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_file) and not args.resume:
        os.remove(checkpoint_file)

    # Print summary
    elapsed_total = time.time() - (start_time if 'start_time' in dir() else time.time())
    print_summary(reviews_list, model, elapsed_total if remaining else 0)


if __name__ == "__main__":
    main()
