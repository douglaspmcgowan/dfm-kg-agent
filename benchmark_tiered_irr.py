"""
IRR on the 3-tier quality grouping: Good / Acceptable / Unsatisfactory.

Content fields (1-3): 3=Good, 2=Acceptable, 1=Unsatisfactory
Metadata fields (1-5): 4-5=Good, 3=Acceptable, 1-2=Unsatisfactory
"""

import json
from pathlib import Path
from collections import Counter
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

TIER_NAMES = ["Good", "Acceptable", "Unsatisfactory"]


def tier(value, field):
    if field in CONTENT_FIELDS:
        if value == 3: return "Good"
        elif value == 2: return "Acceptable"
        else: return "Unsatisfactory"
    else:
        if value >= 4: return "Good"
        elif value == 3: return "Acceptable"
        else: return "Unsatisfactory"


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
    N, k = matrix.shape
    n = matrix.sum(axis=1)[0]
    p_j = matrix.sum(axis=0) / (N * n)
    P_i = (matrix ** 2).sum(axis=1) - n
    P_i = P_i / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def gwet_ac1_multi(matrix):
    """Gwet's AC1 for multiple raters (nominal, 3 categories)."""
    N, q = matrix.shape
    n = matrix.sum(axis=1)[0]
    # Observed agreement
    P_i = (matrix ** 2).sum(axis=1) - n
    P_i = P_i / (n * (n - 1))
    P_a = P_i.mean()
    # Chance agreement for AC1: uniform distribution
    P_e = 1.0 / q
    if P_e == 1.0:
        return 1.0
    return (P_a - P_e) / (1 - P_e)


def gwet_ac2_multi(matrix, weights=None):
    """Gwet's AC2 for multiple raters with ordinal weights."""
    N, q = matrix.shape
    n = int(matrix.sum(axis=1)[0])

    # Category proportions
    pi_k = matrix.sum(axis=0) / (N * n)

    # Ordinal weights (quadratic)
    if weights is None:
        weights = np.zeros((q, q))
        for k in range(q):
            for l in range(q):
                weights[k, l] = 1.0 - ((k - l) / (q - 1)) ** 2

    # Observed weighted agreement
    Pa = 0.0
    for i in range(N):
        s = 0.0
        for k in range(q):
            for l in range(q):
                s += weights[k, l] * matrix[i, k] * matrix[i, l]
        s = (s - n) / (n * (n - 1))
        Pa += s
    Pa /= N

    # Expected agreement under AC2
    Pe = 0.0
    for k in range(q):
        for l in range(q):
            Pe += weights[k, l] * pi_k[k] * pi_k[l]

    if Pe == 1.0:
        return 1.0
    return (Pa - Pe) / (1 - Pe)


def main():
    raters = load()
    frames = sorted(set().union(*(set(raters[r].keys()) for r in raters)))
    names = ["R1", "R2", "R4"]
    tier_idx = {"Good": 0, "Acceptable": 1, "Unsatisfactory": 2}

    print("=" * 80)
    print("IRR ON 3-TIER QUALITY GROUPING (Good / Acceptable / Unsatisfactory)")
    print("Content (1-3): 3=Good, 2=Acceptable, 1=Unsatisfactory")
    print("Metadata (1-5): 4-5=Good, 3=Acceptable, 1-2=Unsatisfactory")
    print("=" * 80)

    # ---- Per-field 3-rater metrics ----

    print(f"\n  {'Field':<23} {'All3 Exact':>10} {'All3 Adj':>9} {'Majority':>9} {'Fleiss K':>9} {'AC1':>7} {'AC2':>7}")
    print(f"  {'-'*23} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*7}")

    for field in ALL_FIELDS:
        # Build rating matrix: (n_frames, 3 tiers)
        mat_rows = []
        exact3 = 0
        adj3 = 0
        maj = 0
        n_valid = 0

        for fn in frames:
            tiers = []
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    tiers.append(tier(v, field))
            if len(tiers) < 3:
                continue
            n_valid += 1

            row = [0, 0, 0]
            for t in tiers:
                row[tier_idx[t]] += 1
            mat_rows.append(row)

            # All 3 exact tier
            if tiers[0] == tiers[1] == tiers[2]:
                exact3 += 1

            # All 3 adjacent (within 1 tier step)
            tier_vals = [tier_idx[t] for t in tiers]
            if max(tier_vals) - min(tier_vals) <= 1:
                adj3 += 1

            # Majority
            c = Counter(tiers)
            if c.most_common(1)[0][1] >= 2:
                maj += 1

        if n_valid == 0:
            continue

        mat = np.array(mat_rows, dtype=float)
        fk = fleiss_kappa(mat)
        ac1 = gwet_ac1_multi(mat)
        ac2 = gwet_ac2_multi(mat)

        print(f"  {LABELS[field]:<23} {exact3/n_valid:>9.0%} {adj3/n_valid:>8.0%} {maj/n_valid:>8.0%} {fk:>9.3f} {ac1:>7.3f} {ac2:>7.3f}")

    # ---- Grouped dimensions ----

    print("\n" + "=" * 80)
    print("GROUPED DIMENSIONS")
    print("=" * 80)

    groups = [
        ("Content (EA + TK + Clarity)", CONTENT_FIELDS),
        ("Content excl TK (EA + Clarity)", ["extraction_accuracy", "clarity"]),
        ("Metadata (6 dimensions)", METADATA_FIELDS),
        ("All 9 dimensions", ALL_FIELDS),
    ]

    for group_name, group_fields in groups:
        mat_rows = []
        exact3 = 0
        adj3 = 0
        maj = 0
        n_valid = 0

        for field in group_fields:
            for fn in frames:
                tiers = []
                for rn in names:
                    v = raters[rn].get(fn, {}).get(field)
                    if v is not None:
                        tiers.append(tier(v, field))
                if len(tiers) < 3:
                    continue
                n_valid += 1

                row = [0, 0, 0]
                for t in tiers:
                    row[tier_idx[t]] += 1
                mat_rows.append(row)

                if tiers[0] == tiers[1] == tiers[2]:
                    exact3 += 1
                tier_vals = [tier_idx[t] for t in tiers]
                if max(tier_vals) - min(tier_vals) <= 1:
                    adj3 += 1
                c = Counter(tiers)
                if c.most_common(1)[0][1] >= 2:
                    maj += 1

        mat = np.array(mat_rows, dtype=float)
        fk = fleiss_kappa(mat)
        ac1 = gwet_ac1_multi(mat)
        ac2 = gwet_ac2_multi(mat)

        print(f"\n  {group_name}")
        print(f"    Items:      {n_valid}")
        print(f"    All3 Exact: {exact3/n_valid:.0%}  ({exact3}/{n_valid})")
        print(f"    All3 Adj:   {adj3/n_valid:.0%}  ({adj3}/{n_valid})")
        print(f"    Majority:   {maj/n_valid:.0%}  ({maj}/{n_valid})")
        print(f"    Fleiss K:   {fk:.3f}")
        print(f"    AC1:        {ac1:.3f}")
        print(f"    AC2:        {ac2:.3f}")

    # ---- Tier distribution per field (for context) ----

    print("\n" + "=" * 80)
    print("TIER DISTRIBUTION (pooled across 3 raters)")
    print("=" * 80)

    print(f"\n  {'Field':<23} {'Good':>7} {'Accept':>7} {'Unsat':>7} {'n':>5}")
    print(f"  {'-'*23} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")

    for field in ALL_FIELDS:
        counts = Counter()
        total = 0
        for fn in frames:
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    counts[tier(v, field)] += 1
                    total += 1
        g = counts.get("Good", 0)
        a = counts.get("Acceptable", 0)
        u = counts.get("Unsatisfactory", 0)
        print(f"  {LABELS[field]:<23} {g/total:>6.0%} {a/total:>6.0%} {u/total:>6.0%} {total:>5}")

    # ---- Overall judgment on tiers ----

    print("\n" + "=" * 80)
    print("OVERALL JUDGMENT IRR")
    print("=" * 80)

    mat_rows = []
    n_valid = 0
    exact3 = 0
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
            exact3 += 1
        row = [0, 0]  # Keep, Revise
        for v in vals:
            row[0 if v == "Keep" else 1] += 1
        mat_rows.append(row)

    mat = np.array(mat_rows, dtype=float)
    fk = fleiss_kappa(mat)
    # AC1 for binary
    N, q = mat.shape
    n = int(mat.sum(axis=1)[0])
    P_i = (mat ** 2).sum(axis=1) - n
    P_i = P_i / (n * (n - 1))
    P_a = P_i.mean()
    P_e = 1.0 / q
    ac1 = (P_a - P_e) / (1 - P_e)

    print(f"  All 3 exact:    {exact3}/{n_valid} ({exact3/n_valid:.0%})")
    print(f"  Fleiss Kappa:   {fk:.3f}")
    print(f"  AC1:            {ac1:.3f}")

    print()


if __name__ == "__main__":
    main()
