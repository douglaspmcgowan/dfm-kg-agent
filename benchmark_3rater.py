"""
Three-rater agreement metrics for the 26-frame human review panel.
Computes metrics across ALL three raters simultaneously, not just pairwise.
"""

import json
from pathlib import Path
import numpy as np

DATA_DIR = Path("reviewer_docs")

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


def load():
    raters = {}
    for name, fname in [
        ("R1", "reviewer_1_review.json"),
        ("R2", "reviewer_2_review.json"),
        ("R4", "reviewer_4_review.json"),
    ]:
        with open(DATA_DIR / fname, "r", encoding="utf-8") as f:
            reviews = json.load(f)
        raters[name] = {r.get("frame_num", i + 1): r for i, r in enumerate(reviews)}
    return raters


def fleiss_kappa(matrix):
    """
    Fleiss' kappa for multiple raters.
    matrix: (n_items, n_categories) - counts of raters per category per item.
    """
    N, k = matrix.shape
    n = matrix.sum(axis=1)[0]  # raters per item (assumes constant)

    # Proportion per category
    p_j = matrix.sum(axis=0) / (N * n)

    # Per-item agreement
    P_i = (matrix ** 2).sum(axis=1) - n
    P_i = P_i / (n * (n - 1))

    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def main():
    raters = load()
    frames = sorted(set().union(*(set(raters[r].keys()) for r in raters)))
    names = ["R1", "R2", "R4"]

    print("=" * 80)
    print("THREE-RATER AGREEMENT METRICS (R1, R2, R4)")
    print(f"Frames: {len(frames)}")
    print("=" * 80)

    # ── Section 1: All-three exact and within-1 ──────────────────

    print("\n--- Percent Agreement: All Three Raters ---")
    print(f"  {'Field':<23} {'All 3 Exact':>12} {'All 3 +/-1':>12} {'Majority':>10} {'Mean Range':>11}")
    print(f"  {'-'*23} {'-'*12} {'-'*12} {'-'*10} {'-'*11}")

    for field in ALL_FIELDS:
        exact_3 = 0
        within1_3 = 0
        majority = 0
        ranges = []
        n_valid = 0

        for fn in frames:
            vals = []
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    vals.append(v)
            if len(vals) < 3:
                continue
            n_valid += 1

            # All 3 exact
            if vals[0] == vals[1] == vals[2]:
                exact_3 += 1

            # All 3 within 1 of each other (max - min <= 1)
            if max(vals) - min(vals) <= 1:
                within1_3 += 1

            # Majority: at least 2 of 3 agree exactly
            from collections import Counter
            counts = Counter(vals)
            if counts.most_common(1)[0][1] >= 2:
                majority += 1

            ranges.append(max(vals) - min(vals))

        if n_valid == 0:
            continue

        mean_range = np.mean(ranges)
        print(f"  {LABELS[field]:<23} {exact_3/n_valid:>11.0%} {within1_3/n_valid:>11.0%} {majority/n_valid:>9.0%} {mean_range:>11.2f}")

    # ── Section 2: Overall Judgment (3-rater) ────────────────────

    print("\n--- Overall Judgment: Three-Rater Agreement ---")
    exact_3 = 0
    majority_keep = 0
    majority_revise = 0
    n_valid = 0
    for fn in frames:
        vals = []
        for rn in names:
            v = raters[rn].get(fn, {}).get("overall")
            if v:
                vals.append(v)
        if len(vals) < 3:
            continue
        n_valid += 1
        if vals[0] == vals[1] == vals[2]:
            exact_3 += 1
        counts = Counter(vals)
        winner = counts.most_common(1)[0]
        if winner[1] >= 2:
            if winner[0] == "Keep":
                majority_keep += 1
            elif winner[0] == "Revise":
                majority_revise += 1

    print(f"  All 3 agree:     {exact_3}/{n_valid} ({exact_3/n_valid:.0%})")
    print(f"  Majority Keep:   {majority_keep}/{n_valid} ({majority_keep/n_valid:.0%})")
    print(f"  Majority Revise: {majority_revise}/{n_valid} ({majority_revise/n_valid:.0%})")
    print(f"  No majority:     {n_valid - majority_keep - majority_revise}/{n_valid}")

    # ── Section 3: Collapsed binary (3-rater) ────────────────────

    print("\n--- Collapsed Binary: All Three Raters (4-5=Correct, 1-3=Incorrect) ---")
    print(f"  {'Field':<23} {'All 3 Agree':>12} {'Majority Correct':>17} {'Majority Incorrect':>19}")
    print(f"  {'-'*23} {'-'*12} {'-'*17} {'-'*19}")

    for field in METADATA_FIELDS:
        all_agree = 0
        maj_correct = 0
        maj_incorrect = 0
        n_valid = 0

        for fn in frames:
            vals = []
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    vals.append(1 if v >= 4 else 0)
            if len(vals) < 3:
                continue
            n_valid += 1

            if vals[0] == vals[1] == vals[2]:
                all_agree += 1
            s = sum(vals)
            if s >= 2:
                maj_correct += 1
            elif s <= 1:
                maj_incorrect += 1

        if n_valid == 0:
            continue
        print(f"  {LABELS[field]:<23} {all_agree/n_valid:>11.0%} {maj_correct/n_valid:>16.0%} {maj_incorrect/n_valid:>18.0%}")

    # ── Section 4: Fleiss' Kappa ─────────────────────────────────

    print("\n--- Fleiss' Kappa (3 raters) ---")
    print(f"  {'Field':<23} {'Fleiss K':>9} {'Interp'}")
    print(f"  {'-'*23} {'-'*9} {'-'*15}")

    for field in ALL_FIELDS:
        # Determine categories
        all_vals = set()
        for fn in frames:
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    all_vals.add(v)
        cats = sorted(all_vals)
        if len(cats) <= 1:
            continue
        cat_idx = {c: i for i, c in enumerate(cats)}

        # Build matrix
        mat = np.zeros((len(frames), len(cats)))
        valid_rows = []
        for i, fn in enumerate(frames):
            row_count = 0
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    mat[i, cat_idx[v]] += 1
                    row_count += 1
            if row_count == 3:
                valid_rows.append(i)

        if not valid_rows:
            continue
        mat = mat[valid_rows]

        fk = fleiss_kappa(mat)
        interp = (
            "Substantial" if fk >= 0.61 else
            "Moderate" if fk >= 0.41 else
            "Fair" if fk >= 0.21 else
            "Slight" if fk >= 0.0 else
            "Poor"
        )
        print(f"  {LABELS[field]:<23} {fk:>9.3f} {interp}")

    # Fleiss on collapsed binary
    print("\n--- Fleiss' Kappa on Collapsed Binary (4-5 vs 1-3) ---")
    print(f"  {'Field':<23} {'Fleiss K':>9}")
    print(f"  {'-'*23} {'-'*9}")

    for field in METADATA_FIELDS:
        mat = np.zeros((len(frames), 2))  # 0=incorrect, 1=correct
        valid_rows = []
        for i, fn in enumerate(frames):
            row_count = 0
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    mat[i, 1 if v >= 4 else 0] += 1
                    row_count += 1
            if row_count == 3:
                valid_rows.append(i)
        mat = mat[valid_rows]
        if len(mat) == 0:
            continue
        fk = fleiss_kappa(mat)
        print(f"  {LABELS[field]:<23} {fk:>9.3f}")

    # ── Section 5: Average pairwise vs multi-rater comparison ────

    print("\n--- Average Pairwise %Exact vs All-Three %Exact ---")
    print(f"  {'Field':<23} {'Avg Pairwise':>13} {'All 3 Exact':>12} {'Gap':>6}")
    print(f"  {'-'*23} {'-'*13} {'-'*12} {'-'*6}")

    pairs = [("R1", "R2"), ("R1", "R4"), ("R2", "R4")]
    for field in ALL_FIELDS:
        pair_pcts = []
        for ra, rb in pairs:
            matches = 0
            total = 0
            for fn in frames:
                va = raters[ra].get(fn, {}).get(field)
                vb = raters[rb].get(fn, {}).get(field)
                if va is not None and vb is not None:
                    total += 1
                    if va == vb:
                        matches += 1
            if total > 0:
                pair_pcts.append(matches / total)

        exact_3 = 0
        n_valid = 0
        for fn in frames:
            vals = []
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    vals.append(v)
            if len(vals) == 3:
                n_valid += 1
                if vals[0] == vals[1] == vals[2]:
                    exact_3 += 1

        avg_pw = np.mean(pair_pcts) if pair_pcts else 0
        all3 = exact_3 / n_valid if n_valid else 0
        gap = avg_pw - all3
        print(f"  {LABELS[field]:<23} {avg_pw:>12.0%} {all3:>11.0%} {gap:>+5.0%}")

    print()


if __name__ == "__main__":
    main()
