"""
Revise flagged frames based on LLM review feedback.

Phase 2 of the review-revise pipeline (Phase 1 = review_all.py).

Pipeline:
  1. review_all.py produces reviews (ratings + comments for each frame)
  2. This script identifies frames needing revision based on calibrated thresholds
  3. An LLM generates corrected fields using structured output (FrameRevision)
  4. Produces a revised extraction file + detailed changelog

Usage:
    python revise_frames.py --extractions extractions_se_all.json --reviews reviews_test_100.json
    python revise_frames.py --extractions extractions_se_all.json --reviews reviews_all.json \\
        --output extractions_se_all_revised.json --changelog revision_changelog.json

Environment variables:
    OPENAI_API_KEY      (required)
    OPENAI_MODEL        (default: gpt-5)
    PARALLEL_REQUESTS   (default: 10)
    CHECKPOINT_INTERVAL (default: 25)
"""

import argparse
import concurrent.futures
import copy
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openai import OpenAI

from schema_review import (
    FrameRevision,
    REVISION_SYSTEM_PROMPT,
    CALIBRATION,
    build_revision_message,
)


# ---------------------------------------------------------------------------
# Identify frames needing revision
# ---------------------------------------------------------------------------
def identify_revisions(
    reviews: list[dict],
    thresholds: dict[str, int] | None = None,
) -> list[dict]:
    """
    Filter reviews to find frames that need revision based on calibrated thresholds.

    Returns list of reviews for frames that should be sent to the revision LLM,
    with a 'flagged_fields' key indicating which fields triggered revision.
    """
    if thresholds is None:
        thresholds = {k: v["revise_threshold"] for k, v in CALIBRATION.items()}

    to_revise = []
    for review in reviews:
        if "error" in review:
            continue

        overall = review.get("overall", "Keep")

        if overall == "Keep":
            continue
        elif overall == "Remove":
            # Flag for manual review, don't auto-revise
            review["flagged_fields"] = ["REMOVE_FLAGGED"]
            to_revise.append(review)
            continue

        # overall == "Revise": check which specific fields are below threshold
        flagged = []
        for field, threshold in thresholds.items():
            rating = review.get(field)
            if rating is not None and rating <= threshold:
                flagged.append(field)

        # Also flag if extraction_accuracy or main_point_quality is at threshold
        # (these affect main_point text, not just labels)
        if review.get("extraction_accuracy", 3) <= thresholds.get("extraction_accuracy", 1):
            if "extraction_accuracy" not in flagged:
                flagged.append("extraction_accuracy")

        if flagged:
            review["flagged_fields"] = flagged
            to_revise.append(review)

    return to_revise


# ---------------------------------------------------------------------------
# Revise one frame
# ---------------------------------------------------------------------------
def revise_one_frame(
    client: OpenAI,
    model: str,
    frame: dict,
    post: dict,
    review: dict,
    flagged_fields: list[str],
) -> dict:
    """Send a frame + review feedback to the LLM for revision."""
    post_text = post.get("non_quoted_text")

    user_msg = build_revision_message(
        frame=frame,
        post=post,
        review=review,
        flagged_fields=flagged_fields,
        post_text=post_text,
    )

    completion = client.beta.chat.completions.parse(
        model=model,
        seed=42,
        messages=[
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=FrameRevision,
    )

    parsed: FrameRevision = completion.choices[0].message.parsed
    return parsed.model_dump()


# ---------------------------------------------------------------------------
# Apply revision to a frame
# ---------------------------------------------------------------------------
def apply_revision(frame: dict, revision: dict) -> tuple[dict, list[dict]]:
    """
    Apply a FrameRevision to a frame dict. Returns (revised_frame, changes_list).

    Only changes fields that actually differ from the original.
    """
    revised = copy.deepcopy(frame)
    actual_changes = []

    # Categorical fields
    for field in ["frame_type", "scope", "applicability", "epistemic_stance"]:
        if field in revision and revision[field] != frame.get(field):
            actual_changes.append({
                "field_name": field,
                "original_value": str(frame.get(field, "")),
                "revised_value": str(revision[field]),
                "justification": next(
                    (c["justification"] for c in revision.get("changes", [])
                     if c["field_name"] == field),
                    revision.get("revision_rationale", ""),
                ),
            })
            revised[field] = revision[field]

    # Main point
    if revision.get("main_point") and revision["main_point"] != frame.get("main_point", ""):
        actual_changes.append({
            "field_name": "main_point",
            "original_value": frame.get("main_point", "")[:200],
            "revised_value": revision["main_point"][:200],
            "justification": next(
                (c["justification"] for c in revision.get("changes", [])
                 if c["field_name"] == "main_point"),
                revision.get("revision_rationale", ""),
            ),
        })
        revised["main_point"] = revision["main_point"]

    # Example context
    if revision.get("example_context") and revision["example_context"] != frame.get("example_context"):
        actual_changes.append({
            "field_name": "example_context",
            "original_value": str(frame.get("example_context", "")),
            "revised_value": str(revision["example_context"]),
            "justification": "Added or corrected example_context for applicability",
        })
        revised["example_context"] = revision["example_context"]

    return revised, actual_changes


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_revision_summary(changelog: dict):
    """Print paper-ready revision statistics."""
    meta = changelog["metadata"]
    revisions = changelog["revisions"]

    print(f"\n{'='*60}")
    print("REVISION SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames in review: {meta['total_frames_reviewed']}")
    print(f"Frames flagged as Revise: {meta['frames_flagged_revise']}")
    print(f"Frames passing threshold: {meta['frames_passing_threshold']}")
    print(f"Frames actually revised: {meta['frames_actually_revised']}")
    print(f"Frames kept (reviser disagreed): {meta['frames_kept_by_reviser']}")
    print(f"Frames removed: {meta['frames_removed']}")
    print(f"Errors: {meta['errors']}")

    if not revisions:
        return

    # Most commonly revised fields
    field_counts = Counter()
    for rev in revisions:
        if rev["action"] == "revise":
            for ch in rev.get("actual_changes", []):
                field_counts[ch["field_name"]] += 1

    print("\nMost commonly revised fields:")
    for field, count in field_counts.most_common():
        print(f"  {field:<25} {count:>4} revisions")

    # Scope migrations
    scope_changes = [(ch["original_value"], ch["revised_value"])
                     for rev in revisions
                     for ch in rev.get("actual_changes", [])
                     if ch["field_name"] == "scope"]

    if scope_changes:
        print("\nScope label migrations:")
        migration_counts = Counter(scope_changes)
        for (orig, new), count in migration_counts.most_common(10):
            print(f"  {orig:<25} -> {new:<25} ({count}x)")

    # Frame type migrations
    ft_changes = [(ch["original_value"], ch["revised_value"])
                  for rev in revisions
                  for ch in rev.get("actual_changes", [])
                  if ch["field_name"] == "frame_type"]

    if ft_changes:
        print("\nFrame type migrations:")
        migration_counts = Counter(ft_changes)
        for (orig, new), count in migration_counts.most_common(10):
            print(f"  {orig:<25} -> {new:<25} ({count}x)")


# ---------------------------------------------------------------------------
# Before / after comparison on revised frames
# ---------------------------------------------------------------------------
def print_before_after_stats(
    original_results: list[dict],
    revised_results: list[dict],
    changelog_entries: list[dict],
):
    """Compare original vs revised frames on categorical field distributions."""
    # Collect only the frames that were actually revised
    revised_keys = {
        (e["post_id"], e["frame_index"])
        for e in changelog_entries
        if e.get("action") == "revise"
    }

    if not revised_keys:
        return

    orig_lookup = {r["post_id"]: r for r in original_results}
    rev_lookup = {r["post_id"]: r for r in revised_results}

    orig_frames = []
    rev_frames = []
    for pid, fidx in revised_keys:
        op = orig_lookup.get(pid)
        rp = rev_lookup.get(pid)
        if op and rp and fidx < len(op.get("frames", [])) and fidx < len(rp.get("frames", [])):
            orig_frames.append(op["frames"][fidx])
            rev_frames.append(rp["frames"][fidx])

    if not orig_frames:
        return

    print(f"\n{'='*60}")
    print(f"BEFORE / AFTER COMPARISON  (n={len(orig_frames)} revised frames)")
    print(f"{'='*60}")

    for field in ["frame_type", "scope", "applicability", "epistemic_stance"]:
        orig_dist = Counter(f.get(field, "?") for f in orig_frames)
        rev_dist = Counter(f.get(field, "?") for f in rev_frames)
        all_vals = sorted(set(list(orig_dist.keys()) + list(rev_dist.keys())))

        changed_vals = [v for v in all_vals if orig_dist.get(v, 0) != rev_dist.get(v, 0)]
        if not changed_vals:
            continue

        print(f"\n  {field}:")
        print(f"    {'Value':<30} {'Before':>7} {'After':>7} {'Delta':>7}")
        print(f"    {'-'*30} {'-'*7} {'-'*7} {'-'*7}")
        for val in changed_vals:
            b = orig_dist.get(val, 0)
            a = rev_dist.get(val, 0)
            d = a - b
            sign = "+" if d > 0 else ""
            print(f"    {val:<30} {b:>7} {a:>7} {sign}{d:>6}")

    # Main point changes: show how many changed and avg length delta
    mp_changes = [
        (o.get("main_point", ""), r.get("main_point", ""))
        for o, r in zip(orig_frames, rev_frames)
        if o.get("main_point") != r.get("main_point")
    ]
    if mp_changes:
        avg_orig_len = np.mean([len(o) for o, _ in mp_changes])
        avg_rev_len = np.mean([len(r) for _, r in mp_changes])
        print(f"\n  main_point:")
        print(f"    {len(mp_changes)} frames had main_point revised")
        print(f"    Avg length: {avg_orig_len:.0f} chars -> {avg_rev_len:.0f} chars")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Revise flagged frames based on LLM review feedback."
    )
    parser.add_argument(
        "--extractions", "-e", required=True,
        help="Path to extraction JSON file.",
    )
    parser.add_argument(
        "--reviews", "-r", required=True,
        help="Path to review JSON file (from review_all.py).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output revised extraction file (default: extractions_revised_{basename}.json).",
    )
    parser.add_argument(
        "--changelog", "-c", default=None,
        help="Output changelog file (default: revision_changelog_{basename}.json).",
    )

    # Configurable thresholds
    parser.add_argument("--scope-threshold", type=int, default=None)
    parser.add_argument("--main-point-threshold", type=int, default=None)
    parser.add_argument("--extraction-threshold", type=int, default=None)
    parser.add_argument("--frame-type-threshold", type=int, default=None)
    parser.add_argument("--applicability-threshold", type=int, default=None)
    parser.add_argument("--epistemic-threshold", type=int, default=None)

    args = parser.parse_args()

    # Build thresholds
    thresholds = {k: v["revise_threshold"] for k, v in CALIBRATION.items()}
    if args.scope_threshold is not None:
        thresholds["scope_rating"] = args.scope_threshold
    if args.main_point_threshold is not None:
        thresholds["main_point_quality"] = args.main_point_threshold
    if args.extraction_threshold is not None:
        thresholds["extraction_accuracy"] = args.extraction_threshold
    if args.frame_type_threshold is not None:
        thresholds["frame_type_rating"] = args.frame_type_threshold
    if args.applicability_threshold is not None:
        thresholds["applicability_rating"] = args.applicability_threshold
    if args.epistemic_threshold is not None:
        thresholds["epistemic_stance_rating"] = args.epistemic_threshold

    # Defaults for output names
    review_basename = Path(args.reviews).stem
    if args.output is None:
        args.output = f"extractions_revised_{review_basename}.json"
    if args.changelog is None:
        args.changelog = f"revision_changelog_{review_basename}.json"

    # Environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    parallel = int(os.getenv("PARALLEL_REQUESTS", "10"))

    client = OpenAI(api_key=api_key)

    # Load extractions
    print(f"Loading extractions from: {args.extractions}")
    with open(args.extractions, "r", encoding="utf-8") as f:
        extraction_data = json.load(f)

    if isinstance(extraction_data, dict):
        all_results = extraction_data.get("results", [])
    else:
        all_results = extraction_data

    total_frames = sum(len(r.get("frames", [])) for r in all_results)
    print(f"Loaded {len(all_results)} posts with {total_frames} total frames.")

    # Build post lookup by post_id
    post_lookup = {r["post_id"]: r for r in all_results}

    # Load reviews
    print(f"Loading reviews from: {args.reviews}")
    with open(args.reviews, "r", encoding="utf-8") as f:
        review_data = json.load(f)

    if isinstance(review_data, dict):
        reviews = review_data.get("reviews", [])
    else:
        reviews = review_data

    print(f"Loaded {len(reviews)} reviews.")

    # Identify frames needing revision
    to_revise = identify_revisions(reviews, thresholds)
    n_revise_overall = len([r for r in reviews if r.get("overall") == "Revise" and "error" not in r])
    n_remove = len([r for r in to_revise if "REMOVE_FLAGGED" in r.get("flagged_fields", [])])
    to_revise = [r for r in to_revise if "REMOVE_FLAGGED" not in r.get("flagged_fields", [])]

    print(f"\nThresholds: {thresholds}")
    print(f"Frames with overall=Revise: {n_revise_overall}")
    print(f"Frames passing threshold (sent to revision LLM): {len(to_revise)}")
    if n_remove:
        print(f"Frames flagged for removal (manual review needed): {n_remove}")

    if not to_revise:
        print("No frames need revision. Copying extractions unchanged.")
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}")
        return

    # Revise frames in parallel
    print(f"\nRevising {len(to_revise)} frames with {model} (parallel={parallel})...")
    start_time = time.time()

    revision_results = {}
    progress_lock = threading.Lock()
    completed = [0]

    action_tally = Counter()
    field_tally = Counter()

    def worker(review):
        pid = review["post_id"]
        fidx = review["frame_index"]
        flagged = review["flagged_fields"]

        post = post_lookup.get(pid)
        if not post or fidx >= len(post.get("frames", [])):
            return (pid, fidx), {"error": f"Post/frame not found: {pid}:{fidx}"}

        frame = post["frames"][fidx]

        try:
            result = revise_one_frame(client, model, frame, post, review, flagged)
            result["post_id"] = pid
            result["frame_index"] = fidx
        except Exception as e:
            result = {"post_id": pid, "frame_index": fidx, "error": str(e)}

        with progress_lock:
            completed[0] += 1
            action = result.get("action", "error")
            action_tally[action] += 1

            # Track which fields the revision LLM is changing
            if action == "revise":
                for ch in result.get("changes", []):
                    field_tally[ch.get("field_name", "?")] += 1

            # Progress every 20 frames or at end
            n = completed[0]
            total = len(to_revise)
            if n % 20 == 0 or n == total:
                tally_str = ", ".join(f"{k}={v}" for k, v in sorted(action_tally.items()))
                fields_str = ""
                if field_tally:
                    fields_str = "  fields: " + ", ".join(
                        f"{f}={c}" for f, c in field_tally.most_common(4))
                print(f"  [{n}/{total}] {tally_str}{fields_str}")

        return (pid, fidx), result

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(worker, r): r for r in to_revise}
        for future in concurrent.futures.as_completed(futures):
            key, result = future.result()
            revision_results[key] = result

    elapsed = time.time() - start_time
    print(f"\nRevision complete in {elapsed:.1f}s "
          f"({elapsed/max(len(to_revise),1):.2f}s/frame)")
    if field_tally:
        print(f"Fields changed by reviser: "
              + ", ".join(f"{f}={c}" for f, c in field_tally.most_common()))

    # Apply revisions to extraction data
    revised_data = copy.deepcopy(extraction_data)
    if isinstance(revised_data, dict):
        revised_results = revised_data.get("results", [])
    else:
        revised_results = revised_data

    changelog_entries = []
    stats = {
        "frames_actually_revised": 0,
        "frames_kept_by_reviser": 0,
        "frames_removed": 0,
        "errors": 0,
    }

    for (pid, fidx), revision in revision_results.items():
        if "error" in revision:
            stats["errors"] += 1
            changelog_entries.append({
                "post_id": pid,
                "frame_index": fidx,
                "action": "error",
                "error": revision["error"],
            })
            continue

        action = revision.get("action", "keep")

        if action == "keep":
            stats["frames_kept_by_reviser"] += 1
            changelog_entries.append({
                "post_id": pid,
                "frame_index": fidx,
                "action": "keep",
                "revision_rationale": revision.get("revision_rationale", ""),
            })
            continue

        if action == "remove":
            stats["frames_removed"] += 1
            # Don't actually delete -- mark in changelog for manual review
            changelog_entries.append({
                "post_id": pid,
                "frame_index": fidx,
                "action": "remove",
                "revision_rationale": revision.get("revision_rationale", ""),
            })
            continue

        # action == "revise"
        # Find the post in revised_results and apply changes
        for post in revised_results:
            if post["post_id"] == pid and fidx < len(post.get("frames", [])):
                frame = post["frames"][fidx]
                revised_frame, actual_changes = apply_revision(frame, revision)

                if actual_changes:
                    post["frames"][fidx] = revised_frame
                    stats["frames_actually_revised"] += 1
                    changelog_entries.append({
                        "post_id": pid,
                        "frame_index": fidx,
                        "action": "revise",
                        "actual_changes": actual_changes,
                        "revision_rationale": revision.get("revision_rationale", ""),
                    })
                else:
                    stats["frames_kept_by_reviser"] += 1
                    changelog_entries.append({
                        "post_id": pid,
                        "frame_index": fidx,
                        "action": "keep",
                        "revision_rationale": "Reviser output matched original (no actual changes).",
                    })
                break

    # Write revised extractions
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(revised_data, f, indent=2, ensure_ascii=False)
    print(f"\nWrote revised extractions to {out_path}")

    # Write changelog
    changelog = {
        "metadata": {
            "source_file": args.extractions,
            "review_file": args.reviews,
            "model": model,
            "thresholds": thresholds,
            "total_frames_in_corpus": total_frames,
            "total_frames_reviewed": len(reviews),
            "frames_flagged_revise": n_revise_overall,
            "frames_passing_threshold": len(to_revise),
            "frames_actually_revised": stats["frames_actually_revised"],
            "frames_kept_by_reviser": stats["frames_kept_by_reviser"],
            "frames_removed": stats["frames_removed"],
            "errors": stats["errors"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "revisions": changelog_entries,
    }

    cl_path = Path(args.changelog)
    with open(cl_path, "w", encoding="utf-8") as f:
        json.dump(changelog, f, indent=2, ensure_ascii=False)
    print(f"Wrote changelog to {cl_path}")

    # Print summary
    print_revision_summary(changelog)

    # Before/after comparison on revised frames
    print_before_after_stats(all_results, revised_results, changelog_entries)


if __name__ == "__main__":
    main()
