"""
Comprehensive IRR benchmark: computes all candidate metrics on the 26-frame
expert review panel to find which tell a defensible story.

Metrics computed:
  1. Krippendorff's alpha (ordinal) -- current baseline
  2. Gwet's AC1 -- prevalence-paradox-resistant
  3. Cohen's weighted kappa (linear) -- current baseline
  4. Cohen's weighted kappa (quadratic) -- penalizes big misses more
  5. Percent exact agreement
  6. Percent agreement within +/- 1 ("practical tolerance")
  7. Collapsed binary agreement (correct 4-5 vs incorrect 1-3 for metadata)
  8. Gwet's AC1 on collapsed binary
  9. TOST equivalence tests (human vs LLM, +/- 0.5 margin)
  10. Bland-Altman bias analysis (human vs LLM)

Usage:
    python benchmark_irr.py
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ── Load review data ─────────────────────────────────────────────────

DATA_DIR = Path("reviewer_docs")

def load_reviews():
    """Load all reviewer JSON files, return dict of {name: {frame_num: review}}."""
    raters = {}
    for name, fname in [
        ("R1", "reviewer_1_review.json"),
        ("R2", "reviewer_2_review.json"),
        ("R4", "reviewer_4_review.json"),
        ("LLM", "llm_review.json"),
    ]:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} not found", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            reviews = json.load(f)
        raters[name] = {r.get("frame_num", i + 1): r for i, r in enumerate(reviews)}
    return raters


CONTENT_FIELDS = ["extraction_accuracy", "tribal_knowledge", "clarity"]
METADATA_FIELDS = [
    "frame_type_rating", "post_role_rating", "scope_rating",
    "applicability_rating", "main_point_quality", "epistemic_stance_rating",
]
ALL_FIELDS = CONTENT_FIELDS + METADATA_FIELDS

LABELS = {
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


# ── Helper: build paired arrays ──────────────────────────────────────

def get_paired(raters, rater_a, rater_b, field, frames):
    """Get paired ratings for two raters on a field, skipping missing."""
    a_vals, b_vals = [], []
    for fn in frames:
        va = raters.get(rater_a, {}).get(fn, {}).get(field)
        vb = raters.get(rater_b, {}).get(fn, {}).get(field)
        if va is not None and vb is not None:
            a_vals.append(va)
            b_vals.append(vb)
    return a_vals, b_vals


def get_all_rater_matrix(raters, rater_names, field, frames):
    """Build (n_items, n_raters) matrix, NaN for missing."""
    matrix = np.full((len(frames), len(rater_names)), np.nan)
    for j, rn in enumerate(rater_names):
        for i, fn in enumerate(frames):
            v = raters.get(rn, {}).get(fn, {}).get(field)
            if v is not None:
                matrix[i, j] = v
    return matrix


# ── Metric implementations ──────────────────────────────────────────

def krippendorff_alpha_ordinal(matrix):
    """Krippendorff's alpha for ordinal data (squared-distance metric)."""
    n_items, n_raters = matrix.shape
    all_vals = sorted(set(v for v in matrix.flatten() if not np.isnan(v)))
    if len(all_vals) <= 1:
        return 1.0

    # Observed disagreement
    Do = 0.0
    total_pairs = 0
    for i in range(n_items):
        vals_i = [matrix[i, r] for r in range(n_raters) if not np.isnan(matrix[i, r])]
        m_i = len(vals_i)
        if m_i < 2:
            continue
        for a in range(m_i):
            for b in range(a + 1, m_i):
                Do += (vals_i[a] - vals_i[b]) ** 2
                total_pairs += 1
    if total_pairs == 0:
        return 1.0
    Do /= total_pairs

    # Expected disagreement
    all_ratings = [v for v in matrix.flatten() if not np.isnan(v)]
    n_total = len(all_ratings)
    De = 0.0
    count_pairs = 0
    for a in range(n_total):
        for b in range(a + 1, n_total):
            De += (all_ratings[a] - all_ratings[b]) ** 2
            count_pairs += 1
    if count_pairs == 0 or De == 0:
        return 1.0
    De /= count_pairs

    return 1.0 - Do / De


def gwets_ac1(rater1, rater2):
    """
    Gwet's AC1 for two raters with ordinal/nominal data.
    Handles prevalence bias better than Cohen's kappa.
    """
    n = len(rater1)
    if n == 0:
        return 0.0
    assert len(rater1) == len(rater2)

    all_cats = sorted(set(rater1 + rater2))
    q = len(all_cats)
    if q <= 1:
        return 1.0

    # Observed agreement
    pa = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # Expected agreement under AC1
    # pi_k = marginal proportion for category k (average of both raters)
    pi = {}
    for k in all_cats:
        n1k = sum(1 for x in rater1 if x == k)
        n2k = sum(1 for x in rater2 if x == k)
        pi[k] = (n1k + n2k) / (2 * n)

    pe = sum(pi[k] * (1 - pi[k]) for k in all_cats) / (q - 1)

    if pe == 1.0:
        return 1.0 if pa == 1.0 else 0.0
    return (pa - pe) / (1 - pe)


def gwets_ac2_ordinal(rater1, rater2):
    """
    Gwet's AC2 for ordinal data (weighted version of AC1).
    Uses quadratic weights like weighted kappa but with AC1's chance correction.
    """
    n = len(rater1)
    if n == 0:
        return 0.0
    assert len(rater1) == len(rater2)

    all_cats = sorted(set(rater1 + rater2))
    q = len(all_cats)
    if q <= 1:
        return 1.0

    cat_to_idx = {c: i for i, c in enumerate(all_cats)}

    # Quadratic weight matrix
    def weight(a, b):
        ia, ib = cat_to_idx[a], cat_to_idx[b]
        return 1.0 - ((ia - ib) ** 2) / ((q - 1) ** 2)

    # Observed weighted agreement
    pa = sum(weight(a, b) for a, b in zip(rater1, rater2)) / n

    # Marginal proportions
    pi = {}
    for k in all_cats:
        n1k = sum(1 for x in rater1 if x == k)
        n2k = sum(1 for x in rater2 if x == k)
        pi[k] = (n1k + n2k) / (2 * n)

    # Expected weighted agreement
    pe = 0.0
    for k in all_cats:
        for l in all_cats:
            pe += weight(k, l) * pi[k] * pi[l]

    if pe == 1.0:
        return 1.0 if pa == 1.0 else 0.0
    return (pa - pe) / (1 - pe)


def cohens_weighted_kappa(rater1, rater2, weights="linear"):
    """Cohen's weighted kappa (linear or quadratic weights)."""
    n = len(rater1)
    if n == 0:
        return 0.0

    all_vals = sorted(set(rater1 + rater2))
    k = len(all_vals)
    if k <= 1:
        return 1.0

    val_to_idx = {v: i for i, v in enumerate(all_vals)}

    # Confusion matrix
    matrix = np.zeros((k, k))
    for a, b in zip(rater1, rater2):
        matrix[val_to_idx[a], val_to_idx[b]] += 1

    # Weight matrix
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == "linear":
                w[i, j] = abs(i - j) / (k - 1)
            else:
                w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    matrix_norm = matrix / n
    row_sums = matrix_norm.sum(axis=1)
    col_sums = matrix_norm.sum(axis=0)
    expected = np.outer(row_sums, col_sums)

    po = np.sum(w * matrix_norm)
    pe = np.sum(w * expected)

    if pe == 1.0:
        return 1.0 if po == 0.0 else 0.0
    return 1.0 - po / pe


def pct_exact(rater1, rater2):
    """Percent exact agreement."""
    if not rater1:
        return 0.0
    return sum(1 for a, b in zip(rater1, rater2) if a == b) / len(rater1)


def pct_within_1(rater1, rater2):
    """Percent agreement within +/- 1."""
    if not rater1:
        return 0.0
    return sum(1 for a, b in zip(rater1, rater2) if abs(a - b) <= 1) / len(rater1)


def collapsed_binary(rater1, rater2, threshold=4):
    """Collapse to binary (>= threshold = correct, < threshold = incorrect), return agreement."""
    b1 = [1 if v >= threshold else 0 for v in rater1]
    b2 = [1 if v >= threshold else 0 for v in rater2]
    if not b1:
        return 0.0, 0.0
    pct = sum(1 for a, b in zip(b1, b2) if a == b) / len(b1)
    ac1 = gwets_ac1(b1, b2)
    return pct, ac1


def tost_test(values_a, values_b, margin=0.5):
    """
    Two One-Sided Tests for equivalence.
    Tests whether the mean difference between A and B is within +/- margin.
    Returns (mean_diff, t_lower, p_lower, t_upper, p_upper, equivalent).
    """
    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    if n < 2:
        return (0, 0, 1, 0, 1, False)

    mean_d = np.mean(diffs)
    se = np.std(diffs, ddof=1) / math.sqrt(n)

    if se == 0:
        # Perfect agreement
        return (mean_d, float('inf'), 0, float('inf'), 0, abs(mean_d) <= margin)

    # Test H0: mean_diff <= -margin (lower test, one-sided)
    t_lower = (mean_d - (-margin)) / se
    p_lower = 1 - sp_stats.t.cdf(t_lower, df=n - 1)

    # Test H0: mean_diff >= +margin (upper test, one-sided)
    t_upper = (mean_d - margin) / se
    p_upper = sp_stats.t.cdf(t_upper, df=n - 1)

    equivalent = (p_lower < 0.05) and (p_upper < 0.05)
    return (mean_d, t_lower, p_lower, t_upper, p_upper, equivalent)


def bland_altman(values_a, values_b):
    """
    Bland-Altman analysis.
    Returns (mean_diff, sd_diff, lower_loa, upper_loa, bias_p_value).
    """
    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    if n < 2:
        return (0, 0, 0, 0, 1.0)

    mean_d = np.mean(diffs)
    sd_d = np.std(diffs, ddof=1)
    lower_loa = mean_d - 1.96 * sd_d
    upper_loa = mean_d + 1.96 * sd_d

    # Test if bias is significantly different from zero
    t_stat, p_val = sp_stats.ttest_1samp(diffs, 0)

    return (mean_d, sd_d, lower_loa, upper_loa, p_val)


# ── Main report ──────────────────────────────────────────────────────

def main():
    raters = load_reviews()
    if not raters:
        print("No review data found.")
        return

    frames = sorted(set().union(*(set(raters[r].keys()) for r in raters)))
    human_names = [n for n in ["R1", "R2", "R4"] if n in raters]
    all_names = human_names + (["LLM"] if "LLM" in raters else [])

    print("=" * 80)
    print("COMPREHENSIVE IRR BENCHMARK REPORT")
    print(f"Raters: {', '.join(all_names)}  |  Frames: {len(frames)}")
    print("=" * 80)

    # ── Section 1: Multi-rater metrics (humans only) ──────────────

    print("\n" + "=" * 80)
    print("SECTION 1: MULTI-RATER AGREEMENT (HUMANS ONLY)")
    print("=" * 80)

    print(f"\n{'Field':<23} {'K-Alpha':>8} {'Interpretation':<18}")
    print("-" * 55)
    for f in ALL_FIELDS:
        matrix = get_all_rater_matrix(raters, human_names, f, frames)
        alpha = krippendorff_alpha_ordinal(matrix)
        interp = (
            "Good" if alpha >= 0.8 else
            "Tentative" if alpha >= 0.667 else
            "Fair" if alpha >= 0.4 else
            "Slight" if alpha >= 0.2 else
            "None"
        )
        print(f"  {LABELS[f]:<21} {alpha:>8.3f} {interp}")

    # ── Section 2: Pairwise metrics comparison ────────────────────

    print("\n" + "=" * 80)
    print("SECTION 2: PAIRWISE METRIC COMPARISON (ALL PAIRS)")
    print("Linear Kappa vs Quadratic Kappa vs Gwet's AC1 vs AC2 vs %Exact vs %Within1")
    print("=" * 80)

    pairs = []
    for i, a in enumerate(all_names):
        for b in all_names[i + 1:]:
            pairs.append((a, b))

    for field in ALL_FIELDS:
        scale = "(1-3)" if field in CONTENT_FIELDS else "(1-5)"
        print(f"\n  --- {LABELS[field]} {scale} ---")
        print(f"  {'Pair':<10} {'K-Lin':>7} {'K-Quad':>7} {'AC1':>7} {'AC2':>7} {'%Exact':>7} {'%+/-1':>7} {'n':>4}")
        print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*4}")

        for ra, rb in pairs:
            a_vals, b_vals = get_paired(raters, ra, rb, field, frames)
            if not a_vals:
                continue
            kl = cohens_weighted_kappa(a_vals, b_vals, "linear")
            kq = cohens_weighted_kappa(a_vals, b_vals, "quadratic")
            ac1 = gwets_ac1(a_vals, b_vals)
            ac2 = gwets_ac2_ordinal(a_vals, b_vals)
            pe = pct_exact(a_vals, b_vals)
            pw1 = pct_within_1(a_vals, b_vals)
            label = f"{ra}/{rb}"
            print(f"  {label:<10} {kl:>7.3f} {kq:>7.3f} {ac1:>7.3f} {ac2:>7.3f} {pe:>6.0%} {pw1:>6.0%} {len(a_vals):>4}")

    # ── Section 3: Collapsed binary (metadata only) ──────────────

    print("\n" + "=" * 80)
    print("SECTION 3: COLLAPSED BINARY AGREEMENT (Metadata: 4-5=Correct, 1-3=Incorrect)")
    print("=" * 80)

    for field in METADATA_FIELDS:
        print(f"\n  --- {LABELS[field]} ---")
        print(f"  {'Pair':<10} {'%Agree':>7} {'AC1':>7}")
        print(f"  {'-'*10} {'-'*7} {'-'*7}")
        for ra, rb in pairs:
            a_vals, b_vals = get_paired(raters, ra, rb, field, frames)
            if not a_vals:
                continue
            pct, ac1 = collapsed_binary(a_vals, b_vals, threshold=4)
            label = f"{ra}/{rb}"
            print(f"  {label:<10} {pct:>6.0%} {ac1:>7.3f}")

    # ── Section 4: Overall Judgment agreement ────────────────────

    print("\n" + "=" * 80)
    print("SECTION 4: OVERALL JUDGMENT (Keep vs Revise)")
    print("=" * 80)

    judgment_map = {"Keep": 1, "Revise": 2, "Remove": 3}

    for ra, rb in pairs:
        a_vals, b_vals = [], []
        for fn in frames:
            va = raters.get(ra, {}).get(fn, {}).get("overall")
            vb = raters.get(rb, {}).get(fn, {}).get("overall")
            if va and vb:
                a_vals.append(judgment_map.get(va, 0))
                b_vals.append(judgment_map.get(vb, 0))
        if a_vals:
            ac1 = gwets_ac1(a_vals, b_vals)
            pe = pct_exact(a_vals, b_vals)
            label = f"{ra}/{rb}"
            print(f"  {label:<10} %Agree={pe:.0%}  AC1={ac1:.3f}  n={len(a_vals)}")

    # ── Section 5: TOST Equivalence (Human mean vs LLM) ──────────

    if "LLM" in raters:
        print("\n" + "=" * 80)
        print("SECTION 5: TOST EQUIVALENCE (Human Mean vs LLM, margin=+/-0.5)")
        print("=" * 80)
        print(f"\n  {'Field':<23} {'Diff':>6} {'p_lo':>7} {'p_hi':>7} {'Equiv?':>7} {'Verdict'}")
        print(f"  {'-'*23} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*25}")

        for field in ALL_FIELDS:
            llm_vals = []
            human_means = []
            for fn in frames:
                lv = raters["LLM"].get(fn, {}).get(field)
                if lv is None:
                    continue
                h_vals = []
                for hn in human_names:
                    hv = raters[hn].get(fn, {}).get(field)
                    if hv is not None:
                        h_vals.append(hv)
                if h_vals:
                    llm_vals.append(lv)
                    human_means.append(np.mean(h_vals))

            if len(llm_vals) < 3:
                continue

            mean_d, t_lo, p_lo, t_hi, p_hi, equiv = tost_test(
                llm_vals, human_means, margin=0.5
            )

            verdict = "EQUIVALENT" if equiv else "NOT equivalent"
            if not equiv and abs(mean_d) <= 0.5:
                verdict += " (underpowered?)"

            print(f"  {LABELS[field]:<23} {mean_d:>+6.2f} {p_lo:>7.4f} {p_hi:>7.4f} {'YES' if equiv else 'NO':>7} {verdict}")

        # Also try with +/- 1.0 margin
        print(f"\n  Relaxed margin = +/- 1.0:")
        print(f"  {'Field':<23} {'Diff':>6} {'p_lo':>7} {'p_hi':>7} {'Equiv?':>7}")
        print(f"  {'-'*23} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

        for field in ALL_FIELDS:
            llm_vals = []
            human_means = []
            for fn in frames:
                lv = raters["LLM"].get(fn, {}).get(field)
                if lv is None:
                    continue
                h_vals = []
                for hn in human_names:
                    hv = raters[hn].get(fn, {}).get(field)
                    if hv is not None:
                        h_vals.append(hv)
                if h_vals:
                    llm_vals.append(lv)
                    human_means.append(np.mean(h_vals))

            if len(llm_vals) < 3:
                continue

            mean_d, t_lo, p_lo, t_hi, p_hi, equiv = tost_test(
                llm_vals, human_means, margin=1.0
            )
            print(f"  {LABELS[field]:<23} {mean_d:>+6.2f} {p_lo:>7.4f} {p_hi:>7.4f} {'YES' if equiv else 'NO':>7}")

    # ── Section 6: TOST between human pairs ──────────────────────

    print("\n" + "=" * 80)
    print("SECTION 6: TOST BETWEEN HUMAN PAIRS (margin=+/-0.5)")
    print("Are humans equivalent to EACH OTHER?")
    print("=" * 80)

    for ra, rb in [(a, b) for a, b in pairs if a in human_names and b in human_names]:
        print(f"\n  --- {ra} vs {rb} ---")
        print(f"  {'Field':<23} {'Diff':>6} {'Equiv?':>7}")
        print(f"  {'-'*23} {'-'*6} {'-'*7}")
        for field in ALL_FIELDS:
            a_vals, b_vals = get_paired(raters, ra, rb, field, frames)
            if len(a_vals) < 3:
                continue
            mean_d, _, p_lo, _, p_hi, equiv = tost_test(a_vals, b_vals, margin=0.5)
            print(f"  {LABELS[field]:<23} {mean_d:>+6.2f} {'YES' if equiv else 'NO':>7}")

    # ── Section 7: Bland-Altman (Human mean vs LLM) ──────────────

    if "LLM" in raters:
        print("\n" + "=" * 80)
        print("SECTION 7: BLAND-ALTMAN BIAS (LLM vs Human Mean)")
        print("  Bias = LLM - Human Mean")
        print("  LoA = Limits of Agreement (95%)")
        print("=" * 80)
        print(f"\n  {'Field':<23} {'Bias':>6} {'SD':>6} {'LoA-':>7} {'LoA+':>7} {'p(bias)':>8} {'Sig?'}")
        print(f"  {'-'*23} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*5}")

        for field in ALL_FIELDS:
            llm_vals = []
            human_means = []
            for fn in frames:
                lv = raters["LLM"].get(fn, {}).get(field)
                if lv is None:
                    continue
                h_vals = []
                for hn in human_names:
                    hv = raters[hn].get(fn, {}).get(field)
                    if hv is not None:
                        h_vals.append(hv)
                if h_vals:
                    llm_vals.append(lv)
                    human_means.append(np.mean(h_vals))

            if len(llm_vals) < 3:
                continue

            md, sd, lo, hi, p = bland_altman(llm_vals, human_means)
            sig = "YES" if p < 0.05 else "no"
            print(f"  {LABELS[field]:<23} {md:>+6.2f} {sd:>6.2f} {lo:>+7.2f} {hi:>+7.2f} {p:>8.4f} {sig}")

    # ── Section 8: Summary recommendation ────────────────────────

    print("\n" + "=" * 80)
    print("SECTION 8: METRIC COMPARISON SUMMARY")
    print("Which metrics tell the best story for each dimension?")
    print("=" * 80)

    print(f"\n  {'Field':<23} {'K-Alpha':>8} {'Best K-Quad':>10} {'Best AC1':>9} {'Best AC2':>9} {'Best %+/-1':>10}")
    print(f"  {'-'*23} {'-'*8} {'-'*10} {'-'*9} {'-'*9} {'-'*10}")

    for field in ALL_FIELDS:
        matrix = get_all_rater_matrix(raters, human_names, field, frames)
        alpha = krippendorff_alpha_ordinal(matrix)

        best_kq = -999
        best_ac1 = -999
        best_ac2 = -999
        best_pw1 = 0

        h_pairs = [(a, b) for i, a in enumerate(human_names) for b in human_names[i + 1:]]
        for ra, rb in h_pairs:
            a_vals, b_vals = get_paired(raters, ra, rb, field, frames)
            if not a_vals:
                continue
            kq = cohens_weighted_kappa(a_vals, b_vals, "quadratic")
            ac1 = gwets_ac1(a_vals, b_vals)
            ac2 = gwets_ac2_ordinal(a_vals, b_vals)
            pw1 = pct_within_1(a_vals, b_vals)
            best_kq = max(best_kq, kq)
            best_ac1 = max(best_ac1, ac1)
            best_ac2 = max(best_ac2, ac2)
            best_pw1 = max(best_pw1, pw1)

        print(f"  {LABELS[field]:<23} {alpha:>8.3f} {best_kq:>10.3f} {best_ac1:>9.3f} {best_ac2:>9.3f} {best_pw1:>9.0%}")

    # ── Section 9: The "Defensible Story" tier list ──────────────

    print("\n" + "=" * 80)
    print("SECTION 9: DEFENSIBLE STORY TIER LIST")
    print("=" * 80)

    print("""
  TIER 1 - Strong dimensions (multiple metrics agree):
    Report with: Gwet's AC1/AC2, quadratic kappa, %within-1, TOST

  TIER 2 - Moderate dimensions (some metrics okay):
    Report with: %within-1, Gwet's AC2, collapsed binary agreement
    Acknowledge: chance-corrected metrics are deflated by prevalence

  TIER 3 - Genuinely subjective (all metrics low):
    Strategy: report human-human ceiling, show AI meets/matches ceiling
    Argue: task is inherently subjective; AI is within human variance
""")

    # Classify each dimension
    for field in ALL_FIELDS:
        matrix = get_all_rater_matrix(raters, human_names, field, frames)
        alpha = krippendorff_alpha_ordinal(matrix)

        # Best pairwise human-human metrics
        best_ac2 = -999
        best_pw1 = 0
        for ra, rb in [(a, b) for i, a in enumerate(human_names) for b in human_names[i + 1:]]:
            a_vals, b_vals = get_paired(raters, ra, rb, field, frames)
            if not a_vals:
                continue
            ac2 = gwets_ac2_ordinal(a_vals, b_vals)
            pw1 = pct_within_1(a_vals, b_vals)
            best_ac2 = max(best_ac2, ac2)
            best_pw1 = max(best_pw1, pw1)

        if best_ac2 >= 0.6 and best_pw1 >= 0.85:
            tier = "TIER 1"
        elif best_pw1 >= 0.75:
            tier = "TIER 2"
        else:
            tier = "TIER 3"

        print(f"  {tier}: {LABELS[field]:<23} (alpha={alpha:.3f}, best AC2={best_ac2:.3f}, best %+/-1={best_pw1:.0%})")

    print()


if __name__ == "__main__":
    main()
