"""
Site-level diagnostics for Stack Exchange extraction results.
Cross-references extractions_se_5000.json with the input posts file
to show per-site acceptance rates, frame counts, and skip patterns.
"""

import sys
import io
import json
from collections import defaultdict

# Fix Windows cp1252 encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

EXTRACTIONS_FILE = "extractions_se_5000.json"
INPUT_FILE = "src/scraping_logic/stackexchange_posts_5000.json"

with open(EXTRACTIONS_FILE, "r", encoding="utf-8") as f:
    extractions = json.load(f)["results"]

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    input_posts = json.load(f)

# ---------------------------------------------------------------------------
# Count input posts per site
# ---------------------------------------------------------------------------

input_counts = defaultdict(int)
for post in input_posts:
    input_counts[post["site"]] += 1

# ---------------------------------------------------------------------------
# Tally extraction results per site
# ---------------------------------------------------------------------------

site_stats = defaultdict(lambda: {
    "extracted": 0,       # in extractions file at all
    "accepted": 0,        # has_relevant_content=True AND len(frames) > 0
    "skipped": 0,         # not accepted (no relevant content or no frames)
    "total_frames": 0,
    "skip_reasons": defaultdict(int),
})

for result in extractions:
    site = result.get("se_site", "unknown")
    stats = site_stats[site]
    stats["extracted"] += 1

    has_content = result.get("has_relevant_content", False)
    frames = result.get("frames", [])
    n_frames = len(frames)

    if has_content and n_frames > 0:
        stats["accepted"] += 1
        stats["total_frames"] += n_frames
    else:
        stats["skipped"] += 1
        reason = result.get("skip_reason") or ("no_frames" if has_content else "not_relevant")
        stats["skip_reasons"][reason] += 1

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

all_sites = sorted(set(list(input_counts.keys()) + list(site_stats.keys())))

# Header
print("=" * 100)
print("SITE-LEVEL EXTRACTION DIAGNOSTICS")
print(f"  Extractions file : {EXTRACTIONS_FILE}")
print(f"  Input posts file : {INPUT_FILE}")
print(f"  Total input posts: {len(input_posts)}")
print(f"  Total extracted  : {len(extractions)}")
print("=" * 100)
print()

# Column headers
hdr = (
    f"{'Site':<20s}"
    f"{'Input':>8s}"
    f"{'Extracted':>11s}"
    f"{'Accepted':>10s}"
    f"{'Skipped':>9s}"
    f"{'Accept%':>9s}"
    f"{'Frames':>8s}"
    f"{'Avg/Post':>10s}"
)
print(hdr)
print("-" * len(hdr))

totals = {
    "input": 0, "extracted": 0, "accepted": 0, "skipped": 0,
    "total_frames": 0,
}

for site in all_sites:
    inp = input_counts.get(site, 0)
    stats = site_stats.get(site, {"extracted": 0, "accepted": 0, "skipped": 0, "total_frames": 0})
    ext = stats["extracted"]
    acc = stats["accepted"]
    skp = stats["skipped"]
    frames = stats["total_frames"]
    rate = (acc / ext * 100) if ext > 0 else 0.0
    avg = (frames / acc) if acc > 0 else 0.0

    print(
        f"{site:<20s}"
        f"{inp:>8d}"
        f"{ext:>11d}"
        f"{acc:>10d}"
        f"{skp:>9d}"
        f"{rate:>8.1f}%"
        f"{frames:>8d}"
        f"{avg:>10.2f}"
    )

    totals["input"] += inp
    totals["extracted"] += ext
    totals["accepted"] += acc
    totals["skipped"] += skp
    totals["total_frames"] += frames

# Totals row
print("-" * len(hdr))
t = totals
rate = (t["accepted"] / t["extracted"] * 100) if t["extracted"] > 0 else 0.0
avg = (t["total_frames"] / t["accepted"]) if t["accepted"] > 0 else 0.0
print(
    f"{'TOTAL':<20s}"
    f"{t['input']:>8d}"
    f"{t['extracted']:>11d}"
    f"{t['accepted']:>10d}"
    f"{t['skipped']:>9d}"
    f"{rate:>8.1f}%"
    f"{t['total_frames']:>8d}"
    f"{avg:>10.2f}"
)

# ---------------------------------------------------------------------------
# Coverage: how many input posts were NOT extracted at all?
# ---------------------------------------------------------------------------

print()
print("=" * 100)
print("COVERAGE (input posts vs extracted posts)")
print("=" * 100)
print()

not_extracted = {}
for site in all_sites:
    inp = input_counts.get(site, 0)
    ext = site_stats.get(site, {"extracted": 0})["extracted"]
    diff = inp - ext
    pct = (ext / inp * 100) if inp > 0 else 0.0
    not_extracted[site] = diff
    print(f"  {site:<20s}  {ext:>5d} / {inp:>5d} extracted  ({pct:>5.1f}%)   {diff:>5d} not in extractions")

total_diff = totals["input"] - totals["extracted"]
total_pct = (totals["extracted"] / totals["input"] * 100) if totals["input"] > 0 else 0.0
print(f"  {'TOTAL':<20s}  {totals['extracted']:>5d} / {totals['input']:>5d} extracted  ({total_pct:>5.1f}%)   {total_diff:>5d} not in extractions")

# ---------------------------------------------------------------------------
# Skip reason breakdown per site
# ---------------------------------------------------------------------------

print()
print("=" * 100)
print("SKIP REASON BREAKDOWN (posts with no frames)")
print("=" * 100)
print()

for site in all_sites:
    stats = site_stats.get(site)
    if not stats or not stats.get("skip_reasons"):
        continue
    reasons = stats["skip_reasons"]
    total_skipped = stats["skipped"]
    print(f"  {site} ({total_skipped} skipped):")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / total_skipped * 100 if total_skipped > 0 else 0
        print(f"    {reason:<40s}  {count:>5d}  ({pct:>5.1f}%)")
    print()

# ---------------------------------------------------------------------------
# Frame type distribution per site
# ---------------------------------------------------------------------------

print("=" * 100)
print("FRAME TYPE DISTRIBUTION PER SITE")
print("=" * 100)
print()

site_frame_types = defaultdict(lambda: defaultdict(int))
all_frame_types = set()

for result in extractions:
    site = result.get("se_site", "unknown")
    for frame in result.get("frames", []):
        ft = frame.get("frame_type", "unknown")
        site_frame_types[site][ft] += 1
        all_frame_types.add(ft)

sorted_types = sorted(all_frame_types)

# Header
type_hdr = f"{'Site':<20s}" + "".join(f"{ft:>14s}" for ft in sorted_types) + f"{'TOTAL':>10s}"
print(type_hdr)
print("-" * len(type_hdr))

grand_totals = defaultdict(int)

for site in all_sites:
    counts = site_frame_types.get(site, {})
    row_total = sum(counts.values())
    row = f"{site:<20s}"
    for ft in sorted_types:
        c = counts.get(ft, 0)
        grand_totals[ft] += c
        row += f"{c:>14d}"
    row += f"{row_total:>10d}"
    print(row)

# Grand total row
print("-" * len(type_hdr))
gt_total = sum(grand_totals.values())
row = f"{'TOTAL':<20s}"
for ft in sorted_types:
    row += f"{grand_totals[ft]:>14d}"
row += f"{gt_total:>10d}"
print(row)

print()
print("Done.")
