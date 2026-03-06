"""
Dataset quality analysis: show the distribution of ratings across all reviewers.
Collapse to Good / Acceptable / Unsatisfactory tiers.

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


def tier(value, field):
    """Classify a rating into Good / Acceptable / Unsatisfactory."""
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


def main():
    raters = load()
    frames = sorted(set().union(*(set(raters[r].keys()) for r in raters)))
    names = ["R1", "R2", "R4"]

    # Determine which frames are Keep vs Revise (by majority)
    frame_judgment = {}
    for fn in frames:
        judgments = []
        for rn in names:
            j = raters[rn].get(fn, {}).get("overall")
            if j:
                judgments.append(j)
        keep_count = sum(1 for j in judgments if j == "Keep")
        frame_judgment[fn] = "Keep" if keep_count >= 2 else "Revise"

    keep_frames = [fn for fn in frames if frame_judgment[fn] == "Keep"]
    revise_frames = [fn for fn in frames if frame_judgment[fn] == "Revise"]

    print("=" * 80)
    print("DATASET QUALITY REPORT")
    print(f"Total frames: {len(frames)}  |  Raters: {len(names)}")
    print(f"Majority Keep: {len(keep_frames)}  |  Majority Revise: {len(revise_frames)}")
    print("=" * 80)

    # ── Section 1: Overall quality distribution ──────────────────

    print("\n" + "=" * 80)
    print("SECTION 1: QUALITY DISTRIBUTION (All ratings pooled, 3 raters x 26 frames)")
    print("Content (1-3): 3=Good, 2=Acceptable, 1=Unsatisfactory")
    print("Metadata (1-5): 4-5=Good, 3=Acceptable, 1-2=Unsatisfactory")
    print("=" * 80)

    print(f"\n  {'Field':<23} {'Good':>7} {'Accept':>7} {'Unsat':>7} {'Total':>6} {'Good+Acc':>9}")
    print(f"  {'-'*23} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*9}")

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
        ga_pct = (g + a) / total if total else 0
        print(f"  {LABELS[field]:<23} {g/total:>6.0%} {a/total:>6.0%} {u/total:>6.0%} {total:>6} {ga_pct:>8.0%}")

    # ── Section 2: Per-rater breakdown ───────────────────────────

    print("\n" + "=" * 80)
    print("SECTION 2: PER-RATER QUALITY DISTRIBUTION")
    print("=" * 80)

    for rn in names:
        print(f"\n  --- {rn} ---")
        print(f"  {'Field':<23} {'Good':>7} {'Accept':>7} {'Unsat':>7}")
        print(f"  {'-'*23} {'-'*7} {'-'*7} {'-'*7}")
        for field in ALL_FIELDS:
            counts = Counter()
            total = 0
            for fn in frames:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    counts[tier(v, field)] += 1
                    total += 1
            if total == 0:
                continue
            g = counts.get("Good", 0)
            a = counts.get("Acceptable", 0)
            u = counts.get("Unsatisfactory", 0)
            print(f"  {LABELS[field]:<23} {g/total:>6.0%} {a/total:>6.0%} {u/total:>6.0%}")

    # ── Section 3: Keep frames vs Revise frames ──────────────────

    print("\n" + "=" * 80)
    print("SECTION 3: QUALITY BY JUDGMENT (Majority Keep vs Majority Revise)")
    print("=" * 80)

    for subset_name, subset_frames in [("KEEP frames", keep_frames), ("REVISE frames", revise_frames)]:
        print(f"\n  --- {subset_name} (n={len(subset_frames)}) ---")
        print(f"  {'Field':<23} {'Good':>7} {'Accept':>7} {'Unsat':>7} {'Good+Acc':>9}")
        print(f"  {'-'*23} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")

        for field in ALL_FIELDS:
            counts = Counter()
            total = 0
            for fn in subset_frames:
                for rn in names:
                    v = raters[rn].get(fn, {}).get(field)
                    if v is not None:
                        counts[tier(v, field)] += 1
                        total += 1
            if total == 0:
                continue
            g = counts.get("Good", 0)
            a = counts.get("Acceptable", 0)
            u = counts.get("Unsatisfactory", 0)
            ga_pct = (g + a) / total if total else 0
            print(f"  {LABELS[field]:<23} {g/total:>6.0%} {a/total:>6.0%} {u/total:>6.0%} {ga_pct:>8.0%}")

    # ── Section 4: Frame-level quality summary ───────────────────

    print("\n" + "=" * 80)
    print("SECTION 4: FRAME-LEVEL QUALITY (across all 3 raters)")
    print("A frame is 'clean' if no rater gave Unsatisfactory on any metadata dimension")
    print("=" * 80)

    clean_frames = 0
    one_unsat = 0
    multi_unsat = 0

    for fn in frames:
        unsat_count = 0
        for field in METADATA_FIELDS:
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None and tier(v, field) == "Unsatisfactory":
                    unsat_count += 1
        if unsat_count == 0:
            clean_frames += 1
        elif unsat_count == 1:
            one_unsat += 1
        else:
            multi_unsat += 1

    print(f"\n  No Unsatisfactory ratings:  {clean_frames}/{len(frames)} ({clean_frames/len(frames):.0%})")
    print(f"  1 Unsatisfactory rating:   {one_unsat}/{len(frames)} ({one_unsat/len(frames):.0%})")
    print(f"  2+ Unsatisfactory ratings: {multi_unsat}/{len(frames)} ({multi_unsat/len(frames):.0%})")

    # Also show for content
    print(f"\n  Content dimensions only (no rater gave 1):")
    clean_content = 0
    for fn in frames:
        any_unsat = False
        for field in CONTENT_FIELDS:
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None and v == 1:
                    any_unsat = True
        if not any_unsat:
            clean_content += 1
    print(f"  Clean content: {clean_content}/{len(frames)} ({clean_content/len(frames):.0%})")

    # ── Section 5: The "Revise" frames - what's actually wrong? ──

    print("\n" + "=" * 80)
    print("SECTION 5: WHAT'S WRONG WITH THE 'REVISE' FRAMES?")
    print("For each Revise frame, which dimensions got Unsatisfactory ratings?")
    print("=" * 80)

    for fn in revise_frames:
        print(f"\n  Frame {fn} (Judgment: {frame_judgment[fn]})")
        issues = []
        for field in ALL_FIELDS:
            ratings = []
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    ratings.append((rn, v, tier(v, field)))

            tiers = [t for _, _, t in ratings]
            if "Unsatisfactory" in tiers:
                detail = ", ".join(f"{rn}={v}" for rn, v, _ in ratings)
                issues.append(f"    {LABELS[field]:<23} [{detail}]")
            elif "Acceptable" in tiers and "Good" not in tiers:
                detail = ", ".join(f"{rn}={v}" for rn, v, _ in ratings)
                issues.append(f"    {LABELS[field]:<23} [{detail}]  (all Acceptable)")

        if issues:
            for line in issues:
                print(line)
        else:
            print("    (No Unsatisfactory ratings - disagreement is on Keep/Revise threshold)")

    # ── Section 6: Summary stats ─────────────────────────────────

    print("\n" + "=" * 80)
    print("SECTION 6: HEADLINE NUMBERS")
    print("=" * 80)

    # Total ratings pooled
    total_ratings = 0
    total_good = 0
    total_acceptable = 0
    total_unsat = 0

    for field in ALL_FIELDS:
        for fn in frames:
            for rn in names:
                v = raters[rn].get(fn, {}).get(field)
                if v is not None:
                    total_ratings += 1
                    t = tier(v, field)
                    if t == "Good": total_good += 1
                    elif t == "Acceptable": total_acceptable += 1
                    else: total_unsat += 1

    print(f"\n  Total individual ratings:  {total_ratings}")
    print(f"  Good:                     {total_good} ({total_good/total_ratings:.1%})")
    print(f"  Acceptable:               {total_acceptable} ({total_acceptable/total_ratings:.1%})")
    print(f"  Unsatisfactory:           {total_unsat} ({total_unsat/total_ratings:.1%})")
    print(f"  Good + Acceptable:        {total_good + total_acceptable} ({(total_good + total_acceptable)/total_ratings:.1%})")

    # Content vs metadata
    for label, fields in [("Content", CONTENT_FIELDS), ("Metadata", METADATA_FIELDS)]:
        t_r, t_g, t_a, t_u = 0, 0, 0, 0
        for field in fields:
            for fn in frames:
                for rn in names:
                    v = raters[rn].get(fn, {}).get(field)
                    if v is not None:
                        t_r += 1
                        t = tier(v, field)
                        if t == "Good": t_g += 1
                        elif t == "Acceptable": t_a += 1
                        else: t_u += 1
        print(f"\n  {label} ratings:  Good={t_g/t_r:.0%}  Acceptable={t_a/t_r:.0%}  Unsatisfactory={t_u/t_r:.0%}  (n={t_r})")

    print()


if __name__ == "__main__":
    main()
