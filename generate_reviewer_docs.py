"""
Generate reviewer markdown documents for human evaluation of extracted knowledge frames.

Usage:
    python generate_reviewer_docs.py --input <path> --seed <int> [--reviewers 4] [--target 25] [--output-dir reviewer_docs]

Input can be:
  - A single JSON file with {"results": [...]} structure (extractions_se_all.json style)
  - A directory of JSON files, each either a single SEPostExtractionResult dict
    or a {"results": [...]} wrapper

Output:
  - reviewer_1.md through reviewer_N.md in the output directory
  - Console summary of sampling decisions

Sampling strategy:
  - POST-BASED stratified sampling by frame_type (most frequent type per post)
  - Proportional draw from each stratum in round-robin order
  - All reviewers get the same posts/frames; only frame ORDER differs per reviewer
  - Per-reviewer shuffle seed = base_seed + reviewer_index
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# PII fields to strip from frames before writing to markdown
# ---------------------------------------------------------------------------
PII_FRAME_FIELDS = {
    "author_username",
    "source_url",
    "post_id",
    "post_date",
    "is_accepted_answer",
    "question_id_ref",
    # Review fields (not PII but not relevant for reviewers)
    "contradicted_by",
    "related_frames",
    "thread_context",
    "validated",
    "validation_notes",
}

PII_POST_FIELDS = {
    "post_id",
    "post_type",
    "thread_url",
    "post_date",
    "post_position",
    "author_username",
    "author_reputation",
    "author_credibility_tier",
    "se_site",
    "is_accepted_answer",
    "question_id_ref",
    "non_quoted_text",
    "has_relevant_content",
    "skip_reason",
}


# ---------------------------------------------------------------------------
# Load posts from input path
# ---------------------------------------------------------------------------
def load_posts(input_path: str) -> list[dict]:
    """Load all posts (with frames) from input file or directory."""
    path = Path(input_path)
    raw_posts = []

    if path.is_file():
        raw_posts = _load_file(path)
    elif path.is_dir():
        for f in sorted(path.glob("*.json")):
            raw_posts.extend(_load_file(f))
    else:
        print(f"ERROR: {input_path} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    # Keep only posts that have at least one frame
    posts = [p for p in raw_posts if p.get("frames")]
    return posts


def _load_file(filepath: Path) -> list[dict]:
    """Load a single JSON file. Handles both wrapper and bare post formats."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "results" in data:
            return data["results"]
        # Bare SEPostExtractionResult
        return [data]
    elif isinstance(data, list):
        return data
    else:
        print(f"WARNING: unexpected JSON type in {filepath}, skipping", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Assign each post to a frame_type stratum (most frequent type in that post)
# ---------------------------------------------------------------------------
def assign_post_stratum(post: dict) -> str:
    """Return the most frequent frame_type among this post's frames.
    Ties broken by: whichever type appears first in the frames list."""
    type_counts = Counter(f["frame_type"] for f in post["frames"])
    max_count = max(type_counts.values())
    # Among types with max count, pick the one whose first occurrence is earliest
    for frame in post["frames"]:
        if type_counts[frame["frame_type"]] == max_count:
            return frame["frame_type"]
    # Fallback (shouldn't reach here)
    return post["frames"][0]["frame_type"]


# ---------------------------------------------------------------------------
# Stratified post sampling
# ---------------------------------------------------------------------------
def stratified_sample(posts: list[dict], seed: int, target_frames: int) -> list[dict]:
    """
    Draw posts from frame_type strata proportionally in round-robin order
    until cumulative frame count >= target_frames.
    """
    rng = random.Random(seed)

    # Group posts by stratum
    strata: dict[str, list[dict]] = defaultdict(list)
    for post in posts:
        stratum = assign_post_stratum(post)
        strata[stratum].append(post)

    # Shuffle within each stratum
    for ft in strata:
        rng.shuffle(strata[ft])

    # Compute proportional weights (based on total posts per stratum)
    total_posts = sum(len(v) for v in strata.values())
    stratum_weights = {ft: len(v) / total_posts for ft, v in strata.items()}

    # Build round-robin draw order: sort strata by weight descending for determinism
    sorted_strata = sorted(strata.keys(), key=lambda ft: -stratum_weights[ft])

    # Track how many posts drawn from each stratum and current index
    draw_index = {ft: 0 for ft in sorted_strata}
    selected = []
    frame_total = 0

    # Round-robin: in each round, try to draw one post from each stratum
    # proportionally. We use fractional accumulation to maintain proportions.
    accumulators = {ft: 0.0 for ft in sorted_strata}

    while frame_total < target_frames:
        drew_any = False
        for ft in sorted_strata:
            accumulators[ft] += stratum_weights[ft]
            if accumulators[ft] >= 1.0 / len(sorted_strata):
                # Try to draw from this stratum
                idx = draw_index[ft]
                if idx < len(strata[ft]):
                    post = strata[ft][idx]
                    selected.append(post)
                    frame_total += len(post["frames"])
                    draw_index[ft] += 1
                    accumulators[ft] -= 1.0 / len(sorted_strata)
                    drew_any = True
                    if frame_total >= target_frames:
                        break

        if not drew_any:
            # All strata exhausted
            break

    return selected


# ---------------------------------------------------------------------------
# Alternative simpler proportional sampling (cleaner logic)
# ---------------------------------------------------------------------------
def stratified_sample_v2(posts: list[dict], seed: int, target_frames: int) -> list[dict]:
    """
    1. Group posts by stratum, shuffle each with seed.
    2. Compute how many posts to draw from each stratum proportionally.
    3. Draw in round-robin across strata until frame count >= target.
    """
    rng = random.Random(seed)

    # Group and shuffle
    strata: dict[str, list[dict]] = defaultdict(list)
    for post in posts:
        stratum = assign_post_stratum(post)
        strata[stratum].append(post)
    for ft in strata:
        rng.shuffle(strata[ft])

    total_posts = sum(len(v) for v in strata.values())

    # Sort strata by size descending for deterministic round-robin order
    sorted_strata = sorted(strata.keys(), key=lambda ft: (-len(strata[ft]), ft))

    # Build interleaved draw queue: repeat strata proportionally
    # E.g. if heuristic=35%, observation=34%, risk=12%... we draw in proportion
    draw_queue = []
    # Create a priority queue using fractional counters
    counters = {ft: 0.0 for ft in sorted_strata}
    weights = {ft: len(strata[ft]) / total_posts for ft, v in strata.items()}
    draw_pointers = {ft: 0 for ft in sorted_strata}

    # Generate draw order for up to sum-of-all-posts entries
    max_draws = sum(len(v) for v in strata.values())
    for _ in range(max_draws):
        # Add weight to each counter, pick the one with highest accumulated value
        for ft in sorted_strata:
            counters[ft] += weights[ft]
        # Pick stratum with highest counter that still has posts
        best_ft = None
        best_val = -1
        for ft in sorted_strata:
            if draw_pointers[ft] < len(strata[ft]) and counters[ft] > best_val:
                best_val = counters[ft]
                best_ft = ft
        if best_ft is None:
            break
        draw_queue.append((best_ft, draw_pointers[best_ft]))
        draw_pointers[best_ft] += 1
        counters[best_ft] -= 1.0

    # Draw from queue until frame count >= target
    selected = []
    frame_total = 0
    for ft, idx in draw_queue:
        post = strata[ft][idx]
        selected.append(post)
        frame_total += len(post["frames"])
        if frame_total >= target_frames:
            break

    return selected


# ---------------------------------------------------------------------------
# Generate markdown for one reviewer
# ---------------------------------------------------------------------------
def generate_reviewer_markdown(
    posts: list[dict],
    reviewer_num: int,
) -> str:
    """
    Generate a reviewer markdown document.

    All reviewers see the same posts and frames in the same order.
    Frame numbering is sequential (1..N) by post appearance then frame position.
    """
    # Build global frame numbering (stable across all reviewers)
    frame_num = 0
    post_frame_groups = []  # list of (post, [(frame, global_num)])
    for post in posts:
        group = []
        for frame in post["frames"]:
            frame_num += 1
            group.append((frame, frame_num))
        post_frame_groups.append((post, group))

    total_frames = frame_num

    lines = []

    # ---------------------------------------------------------------
    # HEADER: study context, definitions, and rating instructions
    # ---------------------------------------------------------------
    lines.append(f"# Reviewer {reviewer_num + 1} -- Knowledge Frame Evaluation")
    lines.append("")
    lines.append("## Background")
    lines.append("")
    lines.append(
        "This study evaluates an AI-powered knowledge extraction pipeline. "
        "The pipeline reads Q&A posts from engineering Stack Exchange sites "
        "(topics include CNC machining, materials, tooling, and shop practices) "
        "and uses a large language model to extract structured **knowledge frames** "
        "from each post."
    )
    lines.append("")
    lines.append(
        'A **knowledge frame** is one discrete, self-contained piece of knowledge '
        "extracted from a forum post. Each frame captures a single claim, rule, "
        "observation, or practice described by the post's author. The extraction "
        "system reads the original post text, identifies passages that contain "
        "transferable knowledge, and produces a structured record for each one."
    )
    lines.append("")
    lines.append("## What you are reviewing")
    lines.append("")
    lines.append(
        "Below you will find a set of extracted frames, grouped by the source post "
        "they came from. For each frame you will see:"
    )
    lines.append("")
    lines.append(
        "- **Source Quote** (in a blockquote) -- the verbatim text from the "
        "original forum post that the system identified as supporting this frame."
    )
    lines.append(
        "- **Main Point** -- a 2-3 sentence summary the system wrote to capture "
        "the transferable knowledge claim from the quote."
    )
    lines.append("- **Classified metadata** -- labels the system assigned:")
    lines.append(
        "  - **Frame Type** -- what kind of knowledge this is "
        "(see reference table below)"
    )
    lines.append(
        "  - **Post Role** -- how the source post functions in the discussion "
        "(e.g., a direct answer, supplementary info, a narrative, etc.)"
    )
    lines.append(
        "  - **Scope** -- the engineering domain the frame belongs to "
        "(e.g., machining_process, material, tooling, cost)"
    )
    lines.append(
        "  - **Applicability** -- how broadly the claim holds: "
        "*universal* (true in virtually all cases), "
        "*context_dependent* (true under specific conditions), "
        "or *situational* (specific to a particular shop/machine)"
    )
    lines.append(
        "  - **Epistemic Stance** -- the tone/certainty of the original quote: "
        "ranging from *absolute_authoritative* (strong, no hedges) "
        "to *narrative_implied* (inferred from a story rather than stated directly)"
    )
    lines.append("")
    lines.append(
        "Each post group also shows contextual metadata from the source forum:"
    )
    lines.append("")
    lines.append(
        "- **Thread** -- the title of the Q&A thread the post appeared in."
    )
    lines.append(
        "- **Post Role** -- what role the post played in the thread."
    )
    lines.append(
        "- **Post Score** -- the community vote score on the original post "
        "(upvotes minus downvotes; higher = more community agreement)."
    )
    lines.append(
        "- **Tags** -- topic tags assigned to the thread by the original poster."
    )
    lines.append("")
    lines.append("## Frame type reference")
    lines.append("")
    lines.append("| Frame Type | Meaning |")
    lines.append("|---|---|")
    lines.append(
        "| risk | A condition that leads to a failure mode or operational limitation |"
    )
    lines.append(
        "| heuristic | A personal practice, preference, or directed advice with rationale |"
    )
    lines.append(
        "| principle | A causal explanation of WHY something happens |"
    )
    lines.append(
        "| observation | An empirical, factual, or definitional claim (no action, no causal mechanism) |"
    )
    lines.append(
        "| comparison | Explicitly weighs two or more options against each other |"
    )
    lines.append(
        "| workaround | A non-standard fix done because of a stated limitation |"
    )
    lines.append(
        "| case | A specific real-world job or incident narrative |"
    )
    lines.append(
        "| other | Does not fit the above (should include an extraction rationale) |"
    )
    lines.append("")
    lines.append("## Rating instructions")
    lines.append("")
    lines.append(
        "For each frame, fill in every field in the response block below it. "
        "Type your rating directly after the colon on each line."
    )
    lines.append("")
    lines.append("**Quality ratings (1-3 scale):**")
    lines.append("")
    lines.append("| Rating | Extraction Accuracy | Tribal Knowledge | Clarity |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| 1 | Quote does not support the main point, or main point misrepresents the quote "
        "| Generic textbook knowledge -- easily found in any reference "
        "| Poorly written, ambiguous, or hard to understand |"
    )
    lines.append(
        "| 2 | Main point mostly captures the quote, but has minor omissions or overstatements "
        "| Moderately specialized -- a practitioner might know this, but it is not trivial "
        "| Understandable but could be clearer or more precise |"
    )
    lines.append(
        "| 3 | Main point accurately and completely captures the knowledge in the quote "
        "| Genuine tribal/experiential knowledge -- unlikely to be found in textbooks "
        "| Clear, precise, and immediately actionable |"
    )
    lines.append("")
    lines.append("**Classification ratings (1-5 scale):**")
    lines.append("")
    lines.append(
        "For Frame Type, Post Role, Scope, Applicability, Epistemic Stance, "
        "and Main Point Quality, rate how appropriate the system's label is:"
    )
    lines.append("")
    lines.append("| Rating | Meaning |")
    lines.append("|---|---|")
    lines.append("| 1 | Completely wrong |")
    lines.append("| 2 | Poor -- a different label would clearly be better |")
    lines.append("| 3 | Acceptable -- reasonable but not ideal |")
    lines.append("| 4 | Good -- correct and appropriate |")
    lines.append("| 5 | Excellent -- could not be improved |")
    lines.append("")
    lines.append("**Overall judgment:**")
    lines.append("")
    lines.append("- **Keep** -- the frame is accurate, useful, and correctly classified")
    lines.append(
        "- **Revise** -- the frame captures real knowledge but needs corrections "
        "to labels or main point"
    )
    lines.append(
        "- **Remove** -- the frame is wrong, trivial, or does not represent "
        "meaningful extracted knowledge"
    )
    lines.append("")
    lines.append(
        "Use the **Comments** field for anything you want to note: "
        "suggested label corrections, concerns about the quote, "
        "or general observations."
    )
    lines.append("")
    lines.append(f"**Total frames to review:** {total_frames}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---------------------------------------------------------------
    # FRAME CONTENT
    # ---------------------------------------------------------------
    for display_idx, (post, frame_group) in enumerate(post_frame_groups):
        # Post header (PII-stripped)
        lines.append(f"## Post Group {display_idx + 1}")
        lines.append("")

        if post.get("thread_title"):
            lines.append(f"**Thread:** {post['thread_title']}")

        # post_role comes from the frames (all frames in a post share post_role)
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

            # Source quote as blockquote
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

            # Classified metadata
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

            # Response block
            lines.append("```")
            lines.append(f"---REVIEW FRAME {global_num}---")
            lines.append("Extraction Accuracy (1-3): ")
            lines.append("Tribal Knowledge (1-3): ")
            lines.append("Clarity (1-3): ")
            lines.append("Frame Type (1-5): ")
            lines.append("Post Role (1-5): ")
            lines.append("Scope (1-5): ")
            lines.append("Applicability (1-5): ")
            lines.append("Main Point Quality (1-5): ")
            lines.append("Epistemic Stance (1-5): ")
            lines.append("Overall (Keep / Revise / Remove): ")
            lines.append("Comments: ")
            lines.append("```")
            lines.append("")

            # Frame divider (lighter than post divider)
            if frame_idx < len(frame_group) - 1:
                lines.append("- - -")
                lines.append("")

        # Post group divider
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(
    seed: int,
    all_posts: list[dict],
    selected_posts: list[dict],
    num_reviewers: int,
    target_frames: int,
):
    total_frames = sum(len(p["frames"]) for p in selected_posts)

    print("\n" + "=" * 60)
    print("REVIEWER DOCUMENT GENERATION SUMMARY")
    print("=" * 60)
    print(f"Random seed:          {seed}")
    print(f"Reviewers:            {num_reviewers}")
    print(f"Target frames/reviewer: {target_frames}")
    print(f"Total posts in corpus:  {len(all_posts)}")
    print(f"Total posts sampled:    {len(selected_posts)}")
    print(f"Total frames in docs:   {total_frames}")
    print()

    # Corpus-level frame_type distribution
    corpus_strata = Counter()
    for p in all_posts:
        corpus_strata[assign_post_stratum(p)] += 1

    # Sampled distribution
    sample_strata = Counter()
    sample_frame_counts = Counter()
    for p in selected_posts:
        st = assign_post_stratum(p)
        sample_strata[st] += 1
        sample_frame_counts[st] += len(p["frames"])

    all_types = sorted(
        set(corpus_strata.keys()) | set(sample_strata.keys()),
        key=lambda ft: -corpus_strata.get(ft, 0),
    )

    print("Frame-type stratum breakdown:")
    print(f"  {'Stratum':<24} {'Corpus Posts':>12} {'Sampled Posts':>13} {'Sampled Frames':>14}")
    print(f"  {'-'*24} {'-'*12} {'-'*13} {'-'*14}")
    for ft in all_types:
        cp = corpus_strata.get(ft, 0)
        sp = sample_strata.get(ft, 0)
        sf = sample_frame_counts.get(ft, 0)
        pct = f"({100*cp/len(all_posts):.1f}%)"
        print(f"  {ft:<24} {cp:>6} {pct:>5} {sp:>13} {sf:>14}")
    print()

    # List selected posts (anonymized index)
    print("Selected posts (anonymized index, frame count):")
    for i, post in enumerate(selected_posts):
        st = assign_post_stratum(post)
        print(f"  Post #{i+1:>3}: {len(post['frames'])} frames, stratum={st}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate reviewer markdown documents for human evaluation."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to extraction JSON file or directory of JSON files.",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        required=True,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--reviewers", "-r",
        type=int,
        default=4,
        help="Number of reviewers (default: 4).",
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=25,
        help="Target frames per reviewer (default: 25).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="reviewer_docs",
        help="Output directory for markdown files (default: reviewer_docs).",
    )

    args = parser.parse_args()

    # Load all posts with frames
    print(f"Loading posts from: {args.input}")
    all_posts = load_posts(args.input)
    total_frames = sum(len(p["frames"]) for p in all_posts)
    print(f"Loaded {len(all_posts)} posts with {total_frames} total frames.")

    if not all_posts:
        print("ERROR: No posts with frames found.", file=sys.stderr)
        sys.exit(1)

    # Stratified sampling
    print(f"\nSampling with seed={args.seed}, target={args.target} frames...")
    selected_posts = stratified_sample_v2(all_posts, args.seed, args.target)

    if not selected_posts:
        print("ERROR: No posts selected.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate one markdown file per reviewer
    for reviewer_idx in range(args.reviewers):
        reviewer_num = reviewer_idx + 1
        md = generate_reviewer_markdown(selected_posts, reviewer_idx)
        out_path = out_dir / f"reviewer_{reviewer_num}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        frame_count = sum(len(p["frames"]) for p in selected_posts)
        print(f"  Wrote {out_path} ({frame_count} frames)")

    # Print summary
    print_summary(args.seed, all_posts, selected_posts, args.reviewers, args.target)


if __name__ == "__main__":
    main()
