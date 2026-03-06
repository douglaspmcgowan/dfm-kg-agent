"""
Build training data from reviewer ratings for fine-tuning an LLM frame reviewer.

Converts the existing LLM review (llm_review.json) + any human reviews into
structured training examples in OpenAI fine-tuning JSONL format.

Each training example is a (system, user, assistant) message tuple where:
  - system: the reviewer rubric/instructions
  - user: the frame to review (same format as llm_reviewer.py)
  - assistant: the structured review output (JSON)

Usage:
    python build_review_training_data.py --input extractions_se_all.json --reviews reviewer_docs/llm_review.json --seed 42 [--output review_training.jsonl]

Output:
    - review_training.jsonl -- OpenAI fine-tuning format
"""

import argparse
import json
import os
import sys
from pathlib import Path

from generate_reviewer_docs import load_posts, stratified_sample_v2
from schema_review import REVIEW_SYSTEM_PROMPT as SYSTEM_PROMPT, build_user_message


def build_training_examples(
    posts: list[dict],
    reviews: list[dict],
) -> list[dict]:
    """
    Build training examples by matching reviews to their source frames.

    Returns a list of dicts in OpenAI fine-tuning format:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    # Build frame lookup: frame_num -> (post, frame)
    frame_lookup = {}
    frame_num = 0
    for post in posts:
        for frame in post["frames"]:
            frame_num += 1
            frame_lookup[frame_num] = (post, frame)

    examples = []
    for review in reviews:
        fn = review.get("frame_num")
        if fn is None or fn not in frame_lookup:
            continue

        post, frame = frame_lookup[fn]

        # Build the user message (same format as llm_reviewer.py)
        user_msg = build_user_message(frame, post, fn)

        # Build the assistant response (the review as structured JSON)
        # Remove frame_num from the training output (it's positional, not semantic)
        review_output = {k: v for k, v in review.items() if k != "frame_num" and k != "error"}

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps(review_output, ensure_ascii=False)},
            ]
        }
        examples.append(example)

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Build training data from reviewer ratings."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to extraction JSON file (same as used for reviewer doc generation).",
    )
    parser.add_argument(
        "--reviews", "-r",
        required=True,
        help="Path to review JSON file (e.g., reviewer_docs/llm_review.json).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        required=True,
        help="Random seed (must match the seed used for reviewer doc generation).",
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=25,
        help="Target frames (must match reviewer doc generation, default: 25).",
    )
    parser.add_argument(
        "--output", "-o",
        default="review_training.jsonl",
        help="Output JSONL file (default: review_training.jsonl).",
    )
    parser.add_argument(
        "--add-human",
        nargs="*",
        help="Additional human review JSON files to include as training examples.",
    )

    args = parser.parse_args()

    # Load posts and resample to get the same frames
    print(f"Loading posts from: {args.input}")
    all_posts = load_posts(args.input)
    print(f"Loaded {len(all_posts)} posts.")

    print(f"Sampling with seed={args.seed}, target={args.target}...")
    selected_posts = stratified_sample_v2(all_posts, args.seed, args.target)
    total_frames = sum(len(p["frames"]) for p in selected_posts)
    print(f"Selected {len(selected_posts)} posts with {total_frames} frames.")

    # Load reviews
    with open(args.reviews, "r", encoding="utf-8") as f:
        reviews = json.load(f)
    print(f"Loaded {len(reviews)} reviews from {args.reviews}")

    # Build training examples from primary reviews
    examples = build_training_examples(selected_posts, reviews)
    print(f"Built {len(examples)} training examples from primary reviews.")

    # Add human reviews if provided
    if args.add_human:
        for human_file in args.add_human:
            if not os.path.exists(human_file):
                print(f"WARNING: {human_file} not found, skipping.")
                continue
            with open(human_file, "r", encoding="utf-8") as f:
                human_reviews = json.load(f)
            human_examples = build_training_examples(selected_posts, human_reviews)
            examples.extend(human_examples)
            print(f"Added {len(human_examples)} examples from {human_file}")

    # Write JSONL
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(examples)} training examples to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    # Token estimate (rough)
    total_chars = sum(
        sum(len(m["content"]) for m in ex["messages"])
        for ex in examples
    )
    est_tokens = total_chars // 4
    print(f"Estimated tokens: ~{est_tokens:,}")


if __name__ == "__main__":
    main()
