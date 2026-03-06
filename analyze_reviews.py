"""
Statistical analysis of reviewer ratings for the DFM knowledge frame evaluation.

Supports:
  - Single-rater descriptive statistics (LLM review)
  - Multi-rater inter-rater agreement when human reviews are added
  - Human vs LLM comparison
  - Comment theme analysis

Usage:
    python analyze_reviews.py [--human-dir reviewer_docs] [--llm-review reviewer_docs/llm_review.json]

Outputs:
    - Console report with all statistics
    - reviewer_docs/analysis_results.json -- machine-readable results
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Rating field definitions
CONTENT_FIELDS = ["extraction_accuracy", "tribal_knowledge", "clarity"]
CONTENT_SCALE = (1, 3)

METADATA_FIELDS = [
    "frame_type_rating",
    "post_role_rating",
    "scope_rating",
    "applicability_rating",
    "main_point_quality",
    "epistemic_stance_rating",
]
METADATA_SCALE = (1, 5)

ALL_RATING_FIELDS = CONTENT_FIELDS + METADATA_FIELDS

FIELD_LABELS = {
    "extraction_accuracy": "Extraction Accuracy",
    "tribal_knowledge": "Tribal Knowledge",
    "clarity": "Clarity",
    "frame_type_rating": "Frame Type",
    "post_role_rating": "Post Role",
    "scope_rating": "Scope",
    "applicability_rating": "Applicability",
    "main_point_quality": "Main Point Quality",
    "epistemic_stance_rating": "Epistemic Stance",
}


# ---------------------------------------------------------------------------
# Parse human reviewer markdown files
# ---------------------------------------------------------------------------
def _strip_bold(value: str) -> str:
    """Strip markdown bold markers from a value string: **3** -> 3."""
    value = value.strip()
    if value.startswith("**") and value.endswith("**"):
        value = value[2:-2]
    return value.strip()


def _parse_int(value: str) -> int | None:
    """Parse an integer from a possibly bold-wrapped string."""
    v = _strip_bold(value)
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_review_block(block: str, frame_num: int) -> dict | None:
    """Parse a single review block (the lines after ---REVIEW FRAME N---)."""
    review = {"frame_num": frame_num}

    for line in block.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, _, value = line.partition(":")
        value = value.strip()

        if not value:
            continue  # Skip blank fields

        key_lower = key.strip().lower()

        if "extraction accuracy" in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["extraction_accuracy"] = v
        elif "tribal knowledge" in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["tribal_knowledge"] = v
        elif "clarity" in key_lower and "main point" not in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["clarity"] = v
        elif "frame type" in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["frame_type_rating"] = v
        elif "post role" in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["post_role_rating"] = v
        elif "scope" in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["scope_rating"] = v
        elif "applicability" in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["applicability_rating"] = v
        elif "main point" in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["main_point_quality"] = v
        elif "epistemic" in key_lower:
            v = _parse_int(value)
            if v is not None:
                review["epistemic_stance_rating"] = v
        elif "overall" in key_lower:
            val = _strip_bold(value).lower()
            if "keep" in val:
                review["overall"] = "Keep"
            elif "revise" in val:
                review["overall"] = "Revise"
            elif "remove" in val:
                review["overall"] = "Remove"
            else:
                review["overall"] = _strip_bold(value)
        elif "comment" in key_lower:
            review["comments"] = _strip_bold(value)

    if any(f in review for f in ALL_RATING_FIELDS):
        return review
    return None


def parse_human_review_md(filepath: str) -> list[dict]:
    """Parse a filled-in human reviewer markdown file into a list of review dicts.

    Handles both code-block (```...```) and bare ---REVIEW FRAME N--- formats,
    and both **bold** and plain number ratings.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    reviews = []

    # Try code-block format first: ```\n---REVIEW FRAME N---\n...\n```
    pattern_code = r"```\s*\n\s*---REVIEW FRAME (\d+)---\s*\n(.*?)```"
    matches = list(re.finditer(pattern_code, text, re.DOTALL))

    if not matches:
        # Try bare format: ---REVIEW FRAME N---\n...\n(next section or EOF)
        pattern_bare = r"---REVIEW FRAME (\d+)---\s*\n(.*?)(?=---REVIEW FRAME|\n-{5,}|\n#{1,2} |\Z)"
        matches = list(re.finditer(pattern_bare, text, re.DOTALL))

    for m in matches:
        frame_num = int(m.group(1))
        block = m.group(2)
        review = _parse_review_block(block, frame_num)
        if review is not None:
            reviews.append(review)

    return reviews


def parse_multi_reviewer_md(filepath: str) -> dict[str, list[dict]]:
    """Parse a markdown file that contains multiple reviewers separated by
    '# Reviewer N' headers. Returns {reviewer_name: [reviews]}.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by reviewer headers
    # Match: # Reviewer 1, # Reviewer 2, etc. (with optional backslash-escaped dashes)
    sections = re.split(r"(?=^# Reviewer \d+)", text, flags=re.MULTILINE)

    result = {}
    for section in sections:
        # Extract reviewer number
        header_match = re.match(r"# Reviewer (\d+)", section)
        if not header_match:
            continue

        reviewer_num = header_match.group(1)
        reviewer_name = f"reviewer_{reviewer_num}"

        # Parse reviews from this section
        reviews = []

        # Try bare format (most common in this file)
        pattern = r"---REVIEW FRAME (\d+)---\s*\n(.*?)(?=---REVIEW FRAME|\n-{5,}|\n#{1,3} |\Z)"
        matches = re.finditer(pattern, section, re.DOTALL)

        for m in matches:
            frame_num = int(m.group(1))
            block = m.group(2)
            review = _parse_review_block(block, frame_num)
            if review is not None:
                reviews.append(review)

        if reviews:
            result[reviewer_name] = reviews

    return result


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_stats(values: list[int | float]) -> dict:
    """Compute descriptive statistics for a list of numeric values."""
    arr = np.array(values, dtype=float)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "distribution": dict(Counter(int(v) for v in values)),
    }


# ---------------------------------------------------------------------------
# Krippendorff's alpha (ordinal)
# ---------------------------------------------------------------------------
def krippendorff_alpha_ordinal(ratings_matrix: np.ndarray) -> float:
    """
    Compute Krippendorff's alpha for ordinal data.

    ratings_matrix: shape (n_items, n_raters), NaN for missing values.
    """
    n_items, n_raters = ratings_matrix.shape

    # Get all unique values (excluding NaN)
    all_vals = sorted(set(v for v in ratings_matrix.flatten() if not np.isnan(v)))
    if len(all_vals) <= 1:
        return 1.0  # Perfect agreement if all same value

    n_vals = len(all_vals)
    val_to_idx = {v: i for i, v in enumerate(all_vals)}

    # Ordinal difference function: squared difference of cumulative sums
    # For ordinal data, d(c,k) = sum_{g=c}^{k-1} (n_g)  where n_g is count of value g
    # Simplified: use squared numeric difference as approximation
    def ordinal_diff_sq(c, k):
        """Squared ordinal distance."""
        return (c - k) ** 2

    # Observed disagreement
    Do = 0.0
    total_pairs = 0

    for i in range(n_items):
        vals_i = [ratings_matrix[i, r] for r in range(n_raters) if not np.isnan(ratings_matrix[i, r])]
        m_i = len(vals_i)
        if m_i < 2:
            continue
        for a in range(m_i):
            for b in range(a + 1, m_i):
                Do += ordinal_diff_sq(vals_i[a], vals_i[b])
                total_pairs += 1

    if total_pairs == 0:
        return 1.0

    Do /= total_pairs

    # Expected disagreement
    all_ratings = [v for v in ratings_matrix.flatten() if not np.isnan(v)]
    n_total = len(all_ratings)

    De = 0.0
    count_pairs = 0
    for a in range(n_total):
        for b in range(a + 1, n_total):
            De += ordinal_diff_sq(all_ratings[a], all_ratings[b])
            count_pairs += 1

    if count_pairs == 0:
        return 1.0

    De /= count_pairs

    if De == 0:
        return 1.0

    alpha = 1.0 - Do / De
    return alpha


# ---------------------------------------------------------------------------
# Cohen's kappa (weighted, linear)
# ---------------------------------------------------------------------------
def cohens_weighted_kappa(rater1: list[int], rater2: list[int], weights="linear") -> float:
    """
    Compute Cohen's weighted kappa for ordinal ratings.
    Both lists must be the same length, no missing values.
    """
    assert len(rater1) == len(rater2)
    n = len(rater1)
    if n == 0:
        return 0.0

    all_vals = sorted(set(rater1 + rater2))
    k = len(all_vals)
    if k <= 1:
        return 1.0

    val_to_idx = {v: i for i, v in enumerate(all_vals)}

    # Build confusion matrix
    matrix = np.zeros((k, k))
    for a, b in zip(rater1, rater2):
        matrix[val_to_idx[a], val_to_idx[b]] += 1

    # Weight matrix
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == "linear":
                w[i, j] = abs(i - j) / (k - 1)
            else:  # quadratic
                w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    # Normalize confusion matrix
    matrix_norm = matrix / n

    # Marginals
    row_sums = matrix_norm.sum(axis=1)
    col_sums = matrix_norm.sum(axis=0)
    expected = np.outer(row_sums, col_sums)

    # Weighted observed and expected
    po = np.sum(w * matrix_norm)
    pe = np.sum(w * expected)

    if pe == 1.0:
        return 1.0 if po == 0.0 else 0.0

    kappa = 1.0 - po / pe
    return kappa


# ---------------------------------------------------------------------------
# Percent agreement
# ---------------------------------------------------------------------------
def percent_agreement(rater1: list, rater2: list) -> float:
    """Exact percent agreement between two raters."""
    assert len(rater1) == len(rater2)
    if len(rater1) == 0:
        return 0.0
    matches = sum(1 for a, b in zip(rater1, rater2) if a == b)
    return matches / len(rater1)


# ---------------------------------------------------------------------------
# Analyze comments for themes
# ---------------------------------------------------------------------------
def analyze_comments(reviews: list[dict]) -> dict:
    """Analyze reviewer comments for common themes."""
    theme_patterns = {
        "scope_issue": r"scope|domain|categoriz",
        "main_point_adds_info": r"adds|overreach|not in the quote|not stated|unstated|beyond",
        "frame_type_issue": r"frame.?type|better labeled|should be (heuristic|observation|risk|principle)",
        "applicability_issue": r"applicability|universal|context.?dependent|situational",
        "textbook_knowledge": r"textbook|standard|basic|not tribal",
        "quote_fidelity": r"verbatim|paraphras|quote|fidelity|faithful",
    }

    theme_counts = Counter()
    theme_examples = defaultdict(list)

    for r in reviews:
        comment = r.get("comments", "")
        if not comment:
            continue
        for theme, pattern in theme_patterns.items():
            if re.search(pattern, comment, re.IGNORECASE):
                theme_counts[theme] += 1
                if len(theme_examples[theme]) < 2:
                    theme_examples[theme].append(
                        f"Frame {r.get('frame_num', '?')}: {comment[:120]}..."
                        if len(comment) > 120
                        else f"Frame {r.get('frame_num', '?')}: {comment}"
                    )

    return {
        "theme_counts": dict(theme_counts),
        "theme_examples": dict(theme_examples),
        "frames_with_comments": sum(1 for r in reviews if r.get("comments")),
        "total_frames": len(reviews),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(llm_reviews: list[dict], human_reviews: dict[str, list[dict]] | None = None):
    """Run full analysis and return results dict."""
    results = {}

    # ------------------------------------------------------------------
    # 1. LLM reviewer descriptive statistics
    # ------------------------------------------------------------------
    llm_stats = {}
    for field in ALL_RATING_FIELDS:
        values = [r[field] for r in llm_reviews if field in r]
        if values:
            llm_stats[field] = descriptive_stats(values)

    # Overall judgment distribution
    overall_dist = Counter(r.get("overall", "?") for r in llm_reviews)
    llm_stats["overall_judgment"] = dict(overall_dist)

    results["llm_descriptive"] = llm_stats

    # Comment analysis
    results["llm_comments"] = analyze_comments(llm_reviews)

    # ------------------------------------------------------------------
    # 2. Quality composite scores
    # ------------------------------------------------------------------
    content_scores = []
    metadata_scores = []
    for r in llm_reviews:
        # Content: average of 3 content fields (normalized to 0-1)
        cv = [(r.get(f, 2) - 1) / 2.0 for f in CONTENT_FIELDS]
        content_scores.append(np.mean(cv))
        # Metadata: average of 6 metadata fields (normalized to 0-1)
        mv = [(r.get(f, 3) - 1) / 4.0 for f in METADATA_FIELDS]
        metadata_scores.append(np.mean(mv))

    results["composite_scores"] = {
        "content_quality": {
            "mean": float(np.mean(content_scores)),
            "std": float(np.std(content_scores, ddof=1)),
            "per_frame": [float(v) for v in content_scores],
        },
        "metadata_quality": {
            "mean": float(np.mean(metadata_scores)),
            "std": float(np.std(metadata_scores, ddof=1)),
            "per_frame": [float(v) for v in metadata_scores],
        },
    }

    # ------------------------------------------------------------------
    # 3. Per-reviewer descriptive statistics (human reviewers)
    # ------------------------------------------------------------------
    if human_reviews:
        per_reviewer = {}
        for name, revs in human_reviews.items():
            reviewer_stats = {}
            for field in ALL_RATING_FIELDS:
                values = [r[field] for r in revs if field in r]
                if values:
                    reviewer_stats[field] = descriptive_stats(values)
            overall_dist = Counter(r.get("overall", "?") for r in revs)
            reviewer_stats["overall_judgment"] = dict(overall_dist)
            reviewer_stats["comments_analysis"] = analyze_comments(revs)
            per_reviewer[name] = reviewer_stats
        results["per_reviewer"] = per_reviewer

    # ------------------------------------------------------------------
    # 4. Inter-rater agreement (when human reviews available)
    # ------------------------------------------------------------------
    if human_reviews:
        # Build rater lookups
        human_rater_map = {}
        for name, revs in human_reviews.items():
            human_rater_map[name] = {r["frame_num"]: r for r in revs}

        all_rater_map = dict(human_rater_map)
        all_rater_map["llm"] = {r["frame_num"]: r for r in llm_reviews}

        human_names = sorted(human_rater_map.keys())
        all_names = sorted(all_rater_map.keys())
        frame_nums = sorted(set().union(*(set(all_rater_map[r].keys()) for r in all_names)))

        def _compute_irr(rater_map, rater_names, frame_nums):
            """Compute IRR for a given set of raters."""
            irr = {}
            for field in ALL_RATING_FIELDS:
                matrix = np.full((len(frame_nums), len(rater_names)), np.nan)
                for j, rater in enumerate(rater_names):
                    for i, fn in enumerate(frame_nums):
                        if fn in rater_map[rater] and field in rater_map[rater][fn]:
                            matrix[i, j] = rater_map[rater][fn][field]

                alpha = krippendorff_alpha_ordinal(matrix)

                pairwise = {}
                for a_idx, a_name in enumerate(rater_names):
                    for b_idx, b_name in enumerate(rater_names):
                        if a_idx >= b_idx:
                            continue
                        a_vals, b_vals = [], []
                        for i, fn in enumerate(frame_nums):
                            va = matrix[i, a_idx]
                            vb = matrix[i, b_idx]
                            if not np.isnan(va) and not np.isnan(vb):
                                a_vals.append(int(va))
                                b_vals.append(int(vb))
                        if a_vals:
                            kappa = cohens_weighted_kappa(a_vals, b_vals)
                            pct = percent_agreement(a_vals, b_vals)
                            pairwise[f"{a_name}_vs_{b_name}"] = {
                                "kappa_linear": round(kappa, 3),
                                "percent_agreement": round(pct, 3),
                                "n": len(a_vals),
                            }

                irr[field] = {
                    "krippendorff_alpha": round(alpha, 3),
                    "pairwise": pairwise,
                }

            # Overall judgment
            overall_matrix = np.full((len(frame_nums), len(rater_names)), np.nan)
            judgment_map = {"Keep": 1, "Revise": 2, "Remove": 3}
            for j, rater in enumerate(rater_names):
                for i, fn in enumerate(frame_nums):
                    if fn in rater_map[rater] and "overall" in rater_map[rater][fn]:
                        overall_matrix[i, j] = judgment_map.get(
                            rater_map[rater][fn]["overall"], np.nan
                        )
            irr["overall_judgment"] = {
                "krippendorff_alpha": round(krippendorff_alpha_ordinal(overall_matrix), 3),
            }
            return irr

        # Human-only IRR
        results["human_irr"] = _compute_irr(human_rater_map, human_names, frame_nums)

        # All raters IRR (humans + LLM)
        results["all_raters_irr"] = _compute_irr(all_rater_map, all_names, frame_nums)

        # ------------------------------------------------------------------
        # 5. Human vs LLM comparison
        # ------------------------------------------------------------------
        if "llm" in all_rater_map:
            comparison = {}
            for field in ALL_RATING_FIELDS:
                llm_vals = []
                human_means = []
                for fn in frame_nums:
                    if fn not in all_rater_map["llm"] or field not in all_rater_map["llm"][fn]:
                        continue
                    llm_v = all_rater_map["llm"][fn][field]
                    h_vals = [
                        human_rater_map[h][fn][field]
                        for h in human_names
                        if fn in human_rater_map[h] and field in human_rater_map[h][fn]
                    ]
                    if h_vals:
                        llm_vals.append(llm_v)
                        human_means.append(np.mean(h_vals))

                if llm_vals:
                    diff = [l - h for l, h in zip(llm_vals, human_means)]
                    comparison[field] = {
                        "llm_mean": round(float(np.mean(llm_vals)), 3),
                        "human_mean": round(float(np.mean(human_means)), 3),
                        "mean_diff": round(float(np.mean(diff)), 3),
                        "std_diff": round(float(np.std(diff, ddof=1)), 3) if len(diff) > 1 else 0,
                        "n": len(diff),
                    }

            results["human_vs_llm"] = comparison

    return results


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def _interpret_alpha(alpha: float) -> str:
    """Interpret Krippendorff's alpha value."""
    if alpha >= 0.800:
        return "Good"
    elif alpha >= 0.667:
        return "Tentative"
    elif alpha >= 0.400:
        return "Fair"
    elif alpha >= 0.200:
        return "Slight"
    elif alpha >= 0.0:
        return "None"
    else:
        return "None (negative)"


def _print_best_pairwise(irr: dict):
    """Print best pairwise kappa/agreement for each dimension."""
    for f in ALL_RATING_FIELDS:
        if f not in irr:
            continue
        pw = irr[f].get("pairwise", {})
        if not pw:
            continue
        best_pair = max(pw.items(), key=lambda x: x[1].get("kappa_linear", 0))
        pair_name = best_pair[0].replace("_vs_", " vs ")
        k = best_pair[1]["kappa_linear"]
        pct = best_pair[1]["percent_agreement"]
        print(f"    {FIELD_LABELS[f]:<23} {pair_name:<25} kappa={k:.3f}  agree={pct:.0%}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
def print_report(results: dict):
    """Print a formatted console report."""
    print("\n" + "=" * 70)
    print("REVIEWER ANALYSIS REPORT")
    print("=" * 70)

    # LLM descriptive stats
    llm = results.get("llm_descriptive", {})
    if llm:
        print("\n--- LLM Reviewer: Descriptive Statistics ---\n")

        print("Knowledge Content (1-3 scale):")
        print(f"  {'Field':<25} {'Mean':>6} {'Med':>5} {'SD':>6} {'Distribution'}")
        print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*20}")
        for f in CONTENT_FIELDS:
            s = llm.get(f, {})
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(s.get("distribution", {}).items()))
            print(f"  {FIELD_LABELS[f]:<25} {s.get('mean', 0):>6.2f} {s.get('median', 0):>5.1f} {s.get('std', 0):>6.2f} {dist_str}")

        print("\nMetadata Accuracy (1-5 scale):")
        print(f"  {'Field':<25} {'Mean':>6} {'Med':>5} {'SD':>6} {'Distribution'}")
        print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*30}")
        for f in METADATA_FIELDS:
            s = llm.get(f, {})
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(s.get("distribution", {}).items()))
            print(f"  {FIELD_LABELS[f]:<25} {s.get('mean', 0):>6.2f} {s.get('median', 0):>5.1f} {s.get('std', 0):>6.2f} {dist_str}")

        oj = llm.get("overall_judgment", {})
        print(f"\nOverall Judgment: Keep={oj.get('Keep', 0)}, Revise={oj.get('Revise', 0)}, Remove={oj.get('Remove', 0)}")

    # Composite scores
    comp = results.get("composite_scores", {})
    if comp:
        print("\n--- Composite Quality Scores (0-1 normalized) ---\n")
        cq = comp.get("content_quality", {})
        mq = comp.get("metadata_quality", {})
        print(f"  Content Quality:  mean={cq.get('mean', 0):.3f}, std={cq.get('std', 0):.3f}")
        print(f"  Metadata Quality: mean={mq.get('mean', 0):.3f}, std={mq.get('std', 0):.3f}")

    # Comment themes
    ct = results.get("llm_comments", {})
    if ct:
        print(f"\n--- Comment Analysis ({ct.get('frames_with_comments', 0)}/{ct.get('total_frames', 0)} frames have comments) ---\n")
        for theme, count in sorted(ct.get("theme_counts", {}).items(), key=lambda x: -x[1]):
            print(f"  {theme:<30} {count:>3} occurrences")

    # Per-reviewer descriptive stats
    per_rev = results.get("per_reviewer", {})
    if per_rev:
        print("\n--- Per-Reviewer Mean Ratings ---\n")
        # Header row
        rev_names = sorted(per_rev.keys())
        header = f"  {'Field':<25}"
        for rn in rev_names:
            label = rn.replace("reviewer_", "R")
            header += f" {label:>8}"
        print(header)
        print(f"  {'-'*25}" + f" {'-'*8}" * len(rev_names))

        for f in ALL_RATING_FIELDS:
            row = f"  {FIELD_LABELS[f]:<25}"
            for rn in rev_names:
                s = per_rev[rn].get(f, {})
                row += f" {s.get('mean', 0):>8.2f}"
            print(row)

        # Overall judgment per reviewer
        print()
        for rn in rev_names:
            oj = per_rev[rn].get("overall_judgment", {})
            label = rn.replace("reviewer_", "R")
            print(f"  {label} Overall: Keep={oj.get('Keep', 0)}, Revise={oj.get('Revise', 0)}, Remove={oj.get('Remove', 0)}")

    # Human-only inter-rater agreement
    hirr = results.get("human_irr", {})
    if hirr:
        print("\n--- Inter-Rater Agreement: Human Raters Only (Krippendorff's alpha) ---\n")
        print(f"  {'Field':<25} {'Alpha':>8}  {'Interpretation'}")
        print(f"  {'-'*25} {'-'*8}  {'-'*20}")
        for f in ALL_RATING_FIELDS:
            if f in hirr:
                alpha = hirr[f].get("krippendorff_alpha", "N/A")
                interp = _interpret_alpha(alpha) if isinstance(alpha, (int, float)) else ""
                print(f"  {FIELD_LABELS[f]:<25} {alpha:>8}  {interp}")
        if "overall_judgment" in hirr:
            alpha = hirr["overall_judgment"].get("krippendorff_alpha", "N/A")
            interp = _interpret_alpha(alpha) if isinstance(alpha, (int, float)) else ""
            print(f"  {'Overall Judgment':<25} {alpha:>8}  {interp}")

        # Best pairwise agreements (human-human)
        print("\n  Best Pairwise Human-Human Agreement:")
        _print_best_pairwise(hirr)

    # All-raters inter-rater agreement
    airr = results.get("all_raters_irr", {})
    if airr:
        print("\n--- Inter-Rater Agreement: All Raters incl. LLM (Krippendorff's alpha) ---\n")
        print(f"  {'Field':<25} {'Alpha':>8}  {'Interpretation'}")
        print(f"  {'-'*25} {'-'*8}  {'-'*20}")
        for f in ALL_RATING_FIELDS:
            if f in airr:
                alpha = airr[f].get("krippendorff_alpha", "N/A")
                interp = _interpret_alpha(alpha) if isinstance(alpha, (int, float)) else ""
                print(f"  {FIELD_LABELS[f]:<25} {alpha:>8}  {interp}")
        if "overall_judgment" in airr:
            alpha = airr["overall_judgment"].get("krippendorff_alpha", "N/A")
            interp = _interpret_alpha(alpha) if isinstance(alpha, (int, float)) else ""
            print(f"  {'Overall Judgment':<25} {alpha:>8}  {interp}")

    # Human vs LLM
    hvl = results.get("human_vs_llm", {})
    if hvl:
        print("\n--- Human vs. LLM Comparison ---\n")
        print(f"  {'Field':<25} {'LLM Mean':>9} {'Human Mean':>11} {'Diff':>6}")
        print(f"  {'-'*25} {'-'*9} {'-'*11} {'-'*6}")
        for f in ALL_RATING_FIELDS:
            if f in hvl:
                d = hvl[f]
                print(f"  {FIELD_LABELS[f]:<25} {d['llm_mean']:>9.2f} {d['human_mean']:>11.2f} {d['mean_diff']:>+6.2f}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze reviewer ratings.")
    parser.add_argument(
        "--llm-review",
        default="reviewer_docs/llm_review.json",
        help="Path to LLM review JSON (default: reviewer_docs/llm_review.json).",
    )
    parser.add_argument(
        "--human-dir",
        default="reviewer_docs",
        help="Directory containing filled-in human reviewer markdown files.",
    )
    parser.add_argument(
        "--human-file",
        default=None,
        help="Path to a single multi-reviewer markdown file (e.g., 'Reviewer Data.md').",
    )
    parser.add_argument(
        "--output",
        default="reviewer_docs/analysis_results.json",
        help="Path for JSON output (default: reviewer_docs/analysis_results.json).",
    )

    args = parser.parse_args()

    # Load LLM reviews
    llm_reviews = []
    if os.path.exists(args.llm_review):
        with open(args.llm_review, "r", encoding="utf-8") as f:
            llm_reviews = json.load(f)
        print(f"Loaded {len(llm_reviews)} LLM reviews from {args.llm_review}")
    else:
        print(f"WARNING: LLM review file not found: {args.llm_review}")

    # Try to load human reviews
    human_reviews = {}

    # First try the multi-reviewer file (e.g., "Reviewer Data.md")
    human_file = args.human_file
    if human_file is None:
        # Auto-detect common filenames
        for candidate in ["Reviewer Data.md", "reviewer_data.md", "Reviewer_Data.md"]:
            if os.path.exists(candidate):
                human_file = candidate
                break

    if human_file and os.path.exists(human_file):
        multi = parse_multi_reviewer_md(human_file)
        for name, revs in multi.items():
            human_reviews[name] = revs
            print(f"Loaded {len(revs)} reviews for {name} from {human_file}")

    # Also try individual reviewer files in directory
    if os.path.isdir(args.human_dir):
        for md_file in sorted(Path(args.human_dir).glob("reviewer_[0-9]*.md")):
            reviews = parse_human_review_md(str(md_file))
            if reviews and any(f in reviews[0] for f in ALL_RATING_FIELDS):
                name = md_file.stem
                if name not in human_reviews:  # Don't overwrite multi-reviewer data
                    human_reviews[name] = reviews
                    print(f"Loaded {len(reviews)} reviews from {md_file.name}")
            else:
                print(f"Skipped {md_file.name} (no filled ratings found)")

    if not llm_reviews and not human_reviews:
        print("ERROR: No review data found.", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    results = run_analysis(llm_reviews, human_reviews if human_reviews else None)

    # Print report
    print_report(results)

    # Save JSON output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved analysis results to {out_path}")

    # Export human reviews as JSON for training data pipeline
    if human_reviews:
        for name, revs in human_reviews.items():
            json_path = Path(args.human_dir) / f"{name}_review.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(revs, f, indent=2, ensure_ascii=False)
            print(f"Exported {len(revs)} reviews to {json_path}")


if __name__ == "__main__":
    main()
