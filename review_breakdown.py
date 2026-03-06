"""Quick breakdown of Revise and Remove classes from reviews_all.json."""
import json
import numpy as np
from collections import Counter

d = json.load(open("reviews_all.json", "r", encoding="utf-8"))
reviews = d["reviews"]
valid = [r for r in reviews if "error" not in r]

FIELDS = [
    "extraction_accuracy", "tribal_knowledge", "clarity",
    "frame_type_rating", "post_role_rating", "scope_rating",
    "applicability_rating", "main_point_quality", "epistemic_stance_rating",
]

THRESHOLDS = {
    "scope_rating": 2, "main_point_quality": 3, "extraction_accuracy": 1,
    "frame_type_rating": 2, "applicability_rating": 2, "epistemic_stance_rating": 2,
}

# Also compute Keep stats for comparison
for label in ["Keep", "Revise", "Remove"]:
    subset = [r for r in valid if r["overall"] == label]
    print(f"\n{'='*65}")
    print(f"  {label.upper()} (n={len(subset)}, {100*len(subset)/len(valid):.1f}% of corpus)")
    print(f"{'='*65}")
    if not subset:
        continue

    # Mean ratings
    dash = "-"
    print(f"  {'Dimension':<28s}  {'Mean':>6s} {'SD':>6s} {'Min':>4s} {'Max':>4s}")
    print(f"  {dash*28}  {dash*6} {dash*6} {dash*4} {dash*4}")
    for f in FIELDS:
        vals = [r[f] for r in subset if f in r]
        print(f"  {f:<28s}  {np.mean(vals):>6.2f} {np.std(vals):>6.2f} {min(vals):>4d} {max(vals):>4d}")

    # Fields at/below revision threshold
    if label in ("Revise", "Remove"):
        print()
        print("  Fields at/below revision threshold:")
        for f, t in THRESHOLDS.items():
            count = len([r for r in subset if r.get(f, 99) <= t])
            print(f"    {f:<28s} <= {t}: {count:>4d} ({100*count/len(subset):.1f}%)")

    # Frame type breakdown
    print()
    ft = Counter()
    for r in subset:
        meta = r.get("_meta", {})
        ft[meta.get("frame_type", "?")] += 1
    print("  By frame_type:")
    for k, c in ft.most_common():
        print(f"    {k:<25s} {c:>4d} ({100*c/len(subset):.1f}%)")

    # Scope breakdown
    print()
    scope = Counter()
    for r in subset:
        meta = r.get("_meta", {})
        scope[meta.get("scope", "?")] += 1
    print("  By scope:")
    for k, c in scope.most_common(10):
        print(f"    {k:<25s} {c:>4d} ({100*c/len(subset):.1f}%)")

# Comment theme analysis on Revise frames
print(f"\n{'='*65}")
print("  REVISE COMMENT THEMES")
print(f"{'='*65}")
revise = [r for r in valid if r["overall"] == "Revise"]
keywords = {
    "scope": ["scope", "domain"],
    "main_point": ["main point", "main_point", "summary", "overstat", "adds info"],
    "frame_type": ["frame type", "frame_type", "heuristic", "observation", "should be"],
    "applicability": ["applicability", "universal", "context_dependent", "situational"],
    "source_quote": ["quote", "source", "verbatim", "not in the quote"],
    "epistemic": ["epistemic", "stance", "hedged", "tentative"],
}
theme_counts = Counter()
for r in revise:
    comment = r.get("comments", "").lower()
    for theme, kws in keywords.items():
        if any(kw in comment for kw in kws):
            theme_counts[theme] += 1

for theme, count in theme_counts.most_common():
    print(f"  {theme:<25s} {count:>4d} ({100*count/len(revise):.1f}%)")
