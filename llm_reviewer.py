"""
LLM-based reviewer: sends the same sampled frames to GPT-5 for automated ratings.

Uses the same sampling logic as generate_reviewer_docs.py so the LLM reviews
exactly the same frames the human reviewers see.

Usage:
    python llm_reviewer.py --input extractions_se_all.json --seed <int> [--target 25] [--output-dir reviewer_docs]

Environment variables:
    OPENAI_API_KEY  (required)
    OPENAI_MODEL    (default: gpt-5)
    PARALLEL_REQUESTS (default: 10)

Output:
    - reviewer_docs/llm_review.json   -- structured JSON with all ratings
    - reviewer_docs/reviewer_llm.md   -- filled-in markdown matching human reviewer format
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading
from pathlib import Path
from openai import OpenAI

from generate_reviewer_docs import load_posts, stratified_sample_v2
from schema_review import FrameReview, REVIEW_SYSTEM_PROMPT, build_user_message

# Backward compat: expose SYSTEM_PROMPT name used by other scripts
SYSTEM_PROMPT = REVIEW_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Review one frame via API
# ---------------------------------------------------------------------------
def review_frame(
    client: OpenAI,
    model: str,
    frame: dict,
    post: dict,
    frame_num: int,
) -> dict:
    """Send a single frame to the LLM for review. Returns the review as a dict."""
    user_msg = build_user_message(frame, post, frame_num)

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
    result["frame_num"] = frame_num
    return result


# ---------------------------------------------------------------------------
# Generate filled-in reviewer markdown
# ---------------------------------------------------------------------------
def generate_llm_reviewer_markdown(
    posts: list[dict],
    reviews: dict[int, dict],
) -> str:
    """
    Generate a reviewer markdown file with LLM ratings filled in,
    matching the human reviewer file format.
    """
    # Import the header generation from generate_reviewer_docs
    from generate_reviewer_docs import generate_reviewer_markdown

    # Generate the blank template (reviewer_num=-1 so title says "LLM")
    # Instead, build it manually to control the title
    frame_num = 0
    post_frame_groups = []
    for post in posts:
        group = []
        for frame in post["frames"]:
            frame_num += 1
            group.append((frame, frame_num))
        post_frame_groups.append((post, group))

    total_frames = frame_num
    lines = []

    lines.append("# Reviewer LLM (GPT-5) -- Knowledge Frame Evaluation")
    lines.append("")
    lines.append(
        "This file contains automated ratings produced by GPT-5 using the same "
        "rubric and frame sample as the human reviewers."
    )
    lines.append("")
    lines.append(f"**Total frames reviewed:** {total_frames}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for display_idx, (post, frame_group) in enumerate(post_frame_groups):
        lines.append(f"## Post Group {display_idx + 1}")
        lines.append("")

        if post.get("thread_title"):
            lines.append(f"**Thread:** {post['thread_title']}")

        post_role = (
            frame_group[0][0].get("post_role", "unknown") if frame_group else "unknown"
        )
        lines.append(f"**Post Role:** {post_role}")

        if post.get("post_score") is not None:
            lines.append(f"**Post Score:** {post['post_score']}")

        if post.get("se_tags"):
            lines.append(f"**Tags:** {', '.join(post['se_tags'])}")

        lines.append("")

        for frame_idx, (frame, global_num) in enumerate(frame_group):
            lines.append(f"### Frame {global_num}")
            lines.append("")

            # Source quote
            lines.append("**Source Quote:**")
            lines.append("")
            quote_text = frame.get("source_quote", "(no quote)")
            for quote_line in quote_text.split("\n"):
                lines.append(f"> {quote_line}")
            lines.append("")

            # Main point
            lines.append(
                f"**Main Point (system-generated summary):** "
                f"{frame.get('main_point', '(none)')}"
            )
            lines.append("")

            # System-assigned labels
            lines.append("**System-assigned labels:**")
            lines.append("")
            lines.append(f"- **Frame Type:** {frame.get('frame_type', '?')}")
            lines.append(f"- **Post Role:** {frame.get('post_role', '?')}")
            lines.append(f"- **Scope:** {frame.get('scope', '?')}")
            lines.append(f"- **Applicability:** {frame.get('applicability', '?')}")
            lines.append(
                f"- **Epistemic Stance:** {frame.get('epistemic_stance', '?')}"
            )
            if frame.get("subject"):
                lines.append(f"- **Subject:** {frame['subject']}")
            if frame.get("example_value"):
                lines.append(f"- **Example Value:** {frame['example_value']}")
            if frame.get("example_context"):
                lines.append(f"- **Example Context:** {frame['example_context']}")
            if frame.get("extraction_rationale"):
                lines.append(
                    f"- **Extraction Rationale:** {frame['extraction_rationale']}"
                )
            lines.append("")

            # Filled-in response block
            r = reviews.get(global_num, {})
            lines.append("```")
            lines.append(f"---REVIEW FRAME {global_num}---")
            lines.append(f"Extraction Accuracy (1-3): {r.get('extraction_accuracy', '')}")
            lines.append(f"Tribal Knowledge (1-3): {r.get('tribal_knowledge', '')}")
            lines.append(f"Clarity (1-3): {r.get('clarity', '')}")
            lines.append(f"Frame Type (1-5): {r.get('frame_type_rating', '')}")
            lines.append(f"Post Role (1-5): {r.get('post_role_rating', '')}")
            lines.append(f"Scope (1-5): {r.get('scope_rating', '')}")
            lines.append(f"Applicability (1-5): {r.get('applicability_rating', '')}")
            lines.append(f"Main Point Quality (1-5): {r.get('main_point_quality', '')}")
            lines.append(f"Epistemic Stance (1-5): {r.get('epistemic_stance_rating', '')}")
            lines.append(f"Overall (Keep / Revise / Remove): {r.get('overall', '')}")
            lines.append(f"Comments: {r.get('comments', '')}")
            lines.append("```")
            lines.append("")

            if frame_idx < len(frame_group) - 1:
                lines.append("- - -")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-5 as an automated reviewer on sampled frames."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to extraction JSON file or directory.",
    )
    parser.add_argument(
        "--seed", "-s", type=int, required=True,
        help="Random seed (must match generate_reviewer_docs.py for same sample).",
    )
    parser.add_argument(
        "--target", "-t", type=int, default=25,
        help="Target frames (default: 25).",
    )
    parser.add_argument(
        "--output-dir", "-o", default="reviewer_docs",
        help="Output directory (default: reviewer_docs).",
    )

    args = parser.parse_args()

    # Environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    parallel = int(os.getenv("PARALLEL_REQUESTS", "10"))

    client = OpenAI(api_key=api_key)

    # Load and sample -- same logic as generate_reviewer_docs.py
    print(f"Loading posts from: {args.input}")
    all_posts = load_posts(args.input)
    total_frames = sum(len(p["frames"]) for p in all_posts)
    print(f"Loaded {len(all_posts)} posts with {total_frames} total frames.")

    print(f"Sampling with seed={args.seed}, target={args.target} frames...")
    selected_posts = stratified_sample_v2(all_posts, args.seed, args.target)

    if not selected_posts:
        print("ERROR: No posts selected.", file=sys.stderr)
        sys.exit(1)

    # Build flat list of (post, frame, frame_num) for API calls
    frame_tasks = []
    frame_num = 0
    for post in selected_posts:
        for frame in post["frames"]:
            frame_num += 1
            frame_tasks.append((post, frame, frame_num))

    print(f"Reviewing {len(frame_tasks)} frames with {model} (parallel={parallel})...")

    # Parallel API calls
    reviews = {}  # frame_num -> review dict
    progress_lock = threading.Lock()
    completed = [0]

    def worker(task):
        post, frame, fnum = task
        result = review_frame(client, model, frame, post, fnum)
        with progress_lock:
            completed[0] += 1
            print(f"  [{completed[0]}/{len(frame_tasks)}] Frame {fnum}: "
                  f"{result['overall']} (EA={result['extraction_accuracy']}, "
                  f"TK={result['tribal_knowledge']}, C={result['clarity']})")
        return fnum, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(worker, t): t for t in frame_tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                fnum, result = future.result()
                reviews[fnum] = result
            except Exception as e:
                task = futures[future]
                fnum = task[2]
                print(f"  ERROR on frame {fnum}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                reviews[fnum] = {"frame_num": fnum, "error": str(e)}

    # Write JSON output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "llm_review.json"
    # Sort by frame number for readability
    sorted_reviews = [reviews[k] for k in sorted(reviews.keys())]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_reviews, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {json_path} ({len(sorted_reviews)} reviews)")

    # Write filled-in markdown
    md_path = out_dir / "reviewer_llm.md"
    md = generate_llm_reviewer_markdown(selected_posts, reviews)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {md_path}")

    # Summary stats
    print(f"\n{'='*50}")
    print("LLM REVIEW SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {model}")
    print(f"Frames reviewed: {len([r for r in reviews.values() if 'error' not in r])}")
    errors = len([r for r in reviews.values() if 'error' in r])
    if errors:
        print(f"Errors: {errors}")

    # Rating distributions
    valid = [r for r in reviews.values() if "error" not in r]
    if valid:
        from collections import Counter
        overall_dist = Counter(r["overall"] for r in valid)
        print(f"\nOverall: Keep={overall_dist.get('Keep',0)}, "
              f"Revise={overall_dist.get('Revise',0)}, "
              f"Remove={overall_dist.get('Remove',0)}")

        for field, scale in [
            ("extraction_accuracy", 3),
            ("tribal_knowledge", 3),
            ("clarity", 3),
        ]:
            vals = [r[field] for r in valid]
            avg = sum(vals) / len(vals)
            print(f"{field}: mean={avg:.2f}")

        for field in [
            "frame_type_rating", "post_role_rating", "scope_rating",
            "applicability_rating", "main_point_quality", "epistemic_stance_rating",
        ]:
            vals = [r[field] for r in valid]
            avg = sum(vals) / len(vals)
            print(f"{field}: mean={avg:.2f}")


if __name__ == "__main__":
    main()
