"""
Batch LLM frame reviewer -- reviews N frames from the full extraction corpus.

This extends llm_reviewer.py to scale beyond the 26-frame sample:
  - Samples frames from the full corpus (not just the reviewer sample)
  - Supports reviewing a configurable number of frames (default: 100)
  - Outputs detailed JSON results + summary statistics
  - Can exclude already-reviewed frames to avoid overlap

Usage:
    python llm_batch_review.py --input extractions_se_all.json --count 100 [--seed 99] [--output batch_review_100.json]

Environment variables:
    OPENAI_API_KEY  (required)
    OPENAI_MODEL    (default: gpt-5)
    PARALLEL_REQUESTS (default: 10)
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
from pathlib import Path

import numpy as np
from openai import OpenAI

from schema_review import FrameReview, REVIEW_SYSTEM_PROMPT as SYSTEM_PROMPT, build_user_message
from generate_reviewer_docs import load_posts


# ---------------------------------------------------------------------------
# Sampling: draw N frames from the full corpus
# ---------------------------------------------------------------------------
def sample_frames(
    posts: list[dict],
    count: int,
    seed: int,
    exclude_frame_ids: set[str] | None = None,
) -> list[tuple[dict, dict, int]]:
    """
    Sample `count` frames from all posts.

    Returns list of (post, frame, global_frame_index) tuples.

    If exclude_frame_ids is provided, skip frames whose source_quote hash
    matches (to avoid re-reviewing frames from the 26-frame sample).
    """
    rng = random.Random(seed)

    # Build flat list of (post, frame, global_idx)
    all_frames = []
    idx = 0
    for post in posts:
        for frame in post["frames"]:
            idx += 1
            # Create a simple ID from source_quote for exclusion matching
            fid = hash(frame.get("source_quote", ""))
            if exclude_frame_ids and fid in exclude_frame_ids:
                continue
            all_frames.append((post, frame, idx))

    # Shuffle and take first `count`
    rng.shuffle(all_frames)
    selected = all_frames[:count]

    return selected


# ---------------------------------------------------------------------------
# Review one frame
# ---------------------------------------------------------------------------
def review_frame(
    client: OpenAI,
    model: str,
    frame: dict,
    post: dict,
    frame_idx: int,
) -> dict:
    """Send a single frame to the LLM for review."""
    user_msg = build_user_message(frame, post, frame_idx)

    completion = client.beta.chat.completions.parse(
        model=model,
        seed=42,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=FrameReview,
    )

    parsed: FrameReview = completion.choices[0].message.parsed
    result = parsed.model_dump()
    result["frame_idx"] = frame_idx

    # Include frame metadata for analysis
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
    """Print summary statistics for the batch review."""
    valid = [r for r in reviews if "error" not in r]
    errors = [r for r in reviews if "error" in r]

    print(f"\n{'='*60}")
    print("BATCH REVIEW SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Frames reviewed: {len(valid)}")
    print(f"Errors: {len(errors)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(len(valid),1):.2f}s/frame)")

    if not valid:
        return

    # Overall judgment
    overall_dist = Counter(r["overall"] for r in valid)
    print(f"\nOverall: Keep={overall_dist.get('Keep',0)} ({100*overall_dist.get('Keep',0)/len(valid):.0f}%), "
          f"Revise={overall_dist.get('Revise',0)} ({100*overall_dist.get('Revise',0)/len(valid):.0f}%), "
          f"Remove={overall_dist.get('Remove',0)} ({100*overall_dist.get('Remove',0)/len(valid):.0f}%)")

    # Content ratings
    print("\nKnowledge Content (1-3 scale):")
    for field in ["extraction_accuracy", "tribal_knowledge", "clarity"]:
        vals = [r[field] for r in valid]
        print(f"  {field:<25} mean={np.mean(vals):.2f}, std={np.std(vals, ddof=1):.2f}")

    # Metadata ratings
    print("\nMetadata Accuracy (1-5 scale):")
    for field in ["frame_type_rating", "post_role_rating", "scope_rating",
                   "applicability_rating", "main_point_quality", "epistemic_stance_rating"]:
        vals = [r[field] for r in valid]
        print(f"  {field:<25} mean={np.mean(vals):.2f}, std={np.std(vals, ddof=1):.2f}")

    # By frame type
    print("\nOverall judgment by frame type:")
    ft_groups = {}
    for r in valid:
        ft = r.get("_meta", {}).get("frame_type", "unknown")
        ft_groups.setdefault(ft, []).append(r)

    for ft in sorted(ft_groups.keys()):
        group = ft_groups[ft]
        dist = Counter(r["overall"] for r in group)
        print(f"  {ft:<20} n={len(group):>3}  Keep={dist.get('Keep',0):>3}  "
              f"Revise={dist.get('Revise',0):>3}  Remove={dist.get('Remove',0):>3}")

    # By scope
    print("\nScope rating by assigned scope:")
    scope_groups = {}
    for r in valid:
        sc = r.get("_meta", {}).get("scope", "unknown")
        scope_groups.setdefault(sc, []).append(r)

    for sc in sorted(scope_groups.keys(), key=lambda s: -len(scope_groups[s])):
        group = scope_groups[sc]
        vals = [r["scope_rating"] for r in group]
        print(f"  {sc:<25} n={len(group):>3}  scope_rating mean={np.mean(vals):.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch LLM frame reviewer -- review N frames from the corpus."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to extraction JSON file.",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="Number of frames to review (default: 100).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=99,
        help="Random seed for sampling (default: 99).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (default: batch_review_{count}.json).",
    )
    parser.add_argument(
        "--exclude-reviews",
        default=None,
        help="Path to existing review JSON to exclude those frames (e.g., reviewer_docs/llm_review.json).",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = f"batch_review_{args.count}.json"

    # Environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    parallel = int(os.getenv("PARALLEL_REQUESTS", "10"))

    client = OpenAI(api_key=api_key)

    # Load posts
    print(f"Loading posts from: {args.input}")
    all_posts = load_posts(args.input)
    total_frames = sum(len(p["frames"]) for p in all_posts)
    print(f"Loaded {len(all_posts)} posts with {total_frames} total frames.")

    # Build exclusion set
    exclude_ids = None
    if args.exclude_reviews and os.path.exists(args.exclude_reviews):
        with open(args.exclude_reviews, "r", encoding="utf-8") as f:
            existing = json.load(f)
        # We can't perfectly match by frame_num since numbering depends on sampling,
        # so we load the original sample to get the actual quotes
        # For now, just note the exclusion
        print(f"Note: --exclude-reviews provided but exact matching requires "
              f"same seed/target as original. Proceeding without exclusion.")

    # Sample frames
    print(f"Sampling {args.count} frames (seed={args.seed})...")
    sampled = sample_frames(all_posts, args.count, args.seed, exclude_ids)
    print(f"Selected {len(sampled)} frames for review.")

    if not sampled:
        print("ERROR: No frames selected.", file=sys.stderr)
        sys.exit(1)

    # Review frames in parallel
    print(f"Reviewing {len(sampled)} frames with {model} (parallel={parallel})...")
    start_time = time.time()

    reviews = {}
    progress_lock = threading.Lock()
    completed = [0]

    def worker(task):
        post, frame, fidx = task
        result = review_frame(client, model, frame, post, fidx)
        with progress_lock:
            completed[0] += 1
            if completed[0] % 10 == 0 or completed[0] == len(sampled):
                print(f"  [{completed[0]}/{len(sampled)}] "
                      f"Frame {fidx}: {result['overall']} "
                      f"(EA={result['extraction_accuracy']}, "
                      f"TK={result['tribal_knowledge']})")
        return fidx, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(worker, t): t for t in sampled}
        for future in concurrent.futures.as_completed(futures):
            try:
                fidx, result = future.result()
                reviews[fidx] = result
            except Exception as e:
                task = futures[future]
                fidx = task[2]
                print(f"  ERROR on frame {fidx}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                reviews[fidx] = {"frame_idx": fidx, "error": str(e)}

    elapsed = time.time() - start_time

    # Sort and save
    sorted_reviews = [reviews[k] for k in sorted(reviews.keys())]

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted_reviews, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(sorted_reviews)} reviews to {out_path}")

    # Print summary
    print_summary(sorted_reviews, model, elapsed)


if __name__ == "__main__":
    main()
