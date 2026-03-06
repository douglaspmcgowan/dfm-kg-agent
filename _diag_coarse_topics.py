"""Diagnostics: what fine-grained topics and subjects live inside each coarse cluster."""

import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load data
with open("topic_model_output.json", encoding="utf-8") as f:
    fine = json.load(f)
with open("topic_model_output_coarse.json", encoding="utf-8") as f:
    coarse = json.load(f)
with open("extractions_se_all.json", encoding="utf-8") as f:
    ext = json.load(f)

# frame_id -> fine topic label (cleaned)
fine_map = {}
for a in fine["frame_assignments"]:
    raw = a["topic_label"]
    clean = raw.split("_", 1)[1] if "_" in raw else raw
    fine_map[a["frame_id"]] = clean

# frame_id -> coarse topic id
coarse_map = {a["frame_id"]: a["topic_id"] for a in coarse["frame_assignments"]}

# frame_id -> subject + se_site
subject_map = {}
site_map = {}
for result in ext["results"]:
    if not result.get("has_relevant_content"):
        continue
    for i, frame in enumerate(result["frames"]):
        fid = f"{result['post_id']}-{i}"
        subject_map[fid] = frame.get("subject", "")
        site_map[fid] = result.get("se_site", "")

# Coarse topic labels
coarse_labels = {}
for t in coarse["topics"]:
    raw = t["label"]
    clean = raw.split("_", 1)[1] if "_" in raw else raw
    coarse_labels[t["topic_id"]] = clean

# Process each coarse topic
for t in coarse["topics"]:
    tid = t["topic_id"]
    label = coarse_labels.get(tid, "Outliers" if tid == -1 else f"Topic {tid}")
    fids = [a["frame_id"] for a in coarse["frame_assignments"] if a["topic_id"] == tid]

    # Fine-grained topic breakdown
    fine_counts = Counter(fine_map.get(fid, "(fine outlier)") for fid in fids)

    # Subject frequency
    subj_counts = Counter(subject_map.get(fid, "") for fid in fids if subject_map.get(fid))

    # SE site breakdown
    site_counts = Counter(site_map.get(fid, "") for fid in fids if site_map.get(fid))

    print(f"\n{'=' * 75}")
    print(f"COARSE TOPIC {tid}: {label}  ({t['n_frames']} frames)")
    print(f"{'=' * 75}")

    # Scope distribution
    scopes = t.get("scope_distribution", {})
    scope_str = ", ".join(f"{k}={v}" for k, v in list(scopes.items())[:6])
    print(f"  Scopes: {scope_str}")

    # Frame type distribution
    ftypes = t.get("frame_type_distribution", {})
    ftype_str = ", ".join(f"{k}={v}" for k, v in list(ftypes.items())[:6])
    print(f"  Types:  {ftype_str}")

    # SE sites
    site_str = ", ".join(f"{k}={v}" for k, v in site_counts.most_common(5))
    print(f"  Sites:  {site_str}")

    # Fine-topic sub-clusters
    print(f"\n  Fine-grained sub-topics ({len(fine_counts)} from 40-topic model):")
    for fl, cnt in fine_counts.most_common(12):
        pct = cnt * 100 / t["n_frames"]
        bar = "#" * max(1, int(pct / 2))
        print(f"    {cnt:4d} ({pct:4.1f}%) {bar} {fl}")
    if len(fine_counts) > 12:
        remaining = sum(c for _, c in fine_counts.most_common()[12:])
        print(f"    {remaining:4d}        ... and {len(fine_counts) - 12} more sub-topics")

    # Top subjects
    print(f"\n  Top subjects ({len(subj_counts)} unique):")
    for s, cnt in subj_counts.most_common(8):
        print(f"    [{cnt:2d}x] {s[:72]}")
    if len(subj_counts) > 8:
        print(f"          ... and {len(subj_counts) - 8} more")

# Summary table
print(f"\n\n{'=' * 75}")
print("SUMMARY")
print(f"{'=' * 75}")
print(f"{'ID':>3}  {'Frames':>6}  {'Sub-topics':>10}  {'Unique Subj':>11}  Label")
print(f"{'-'*3}  {'-'*6}  {'-'*10}  {'-'*11}  {'-'*30}")
for t in coarse["topics"]:
    tid = t["topic_id"]
    label = coarse_labels.get(tid, "Outliers")
    fids = [a["frame_id"] for a in coarse["frame_assignments"] if a["topic_id"] == tid]
    n_fine = len(set(fine_map.get(fid, "") for fid in fids))
    n_subj = len(set(subject_map.get(fid, "") for fid in fids if subject_map.get(fid)))
    print(f"{tid:3d}  {t['n_frames']:6d}  {n_fine:10d}  {n_subj:11d}  {label}")
