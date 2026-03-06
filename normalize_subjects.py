"""
Normalize free-text subject fields across extracted knowledge frames.

Embeds all unique subjects via text-embedding-3-large, clusters by cosine
similarity using AgglomerativeClustering, picks a canonical name per cluster,
and outputs a mapping file for downstream graph/search use.

Usage:
    python normalize_subjects.py                              # default threshold
    python normalize_subjects.py --threshold 0.15             # custom distance threshold
    python normalize_subjects.py --review                     # print clusters for inspection
    python normalize_subjects.py --threshold 0.12 --review    # tighter clustering + review

Output files:
    subject_clusters.json       -- cluster definitions + subject-to-canonical mapping
    subject_embeddings.npz      -- cached embeddings (reused on re-runs)

Requires: pip install scikit-learn numpy openai
"""

import json
import sys
import os
import time
import argparse

# Fix Windows cp1252 terminal encoding for unicode subject strings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from collections import Counter

import numpy as np
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = os.getenv("INPUT_FILE", "extractions_se_all.json")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
BATCH_SIZE = 2048
CACHE_FILE = "subject_embeddings.npz"
OUTPUT_FILE = "subject_clusters.json"
BLOCKLIST_FILE = "subject_blocklist.json"
DEFAULT_THRESHOLD = 0.15  # cosine distance; 0.15 = 0.85 cosine similarity

openai_client = OpenAI()

# ---------------------------------------------------------------------------
# Data loading (mirrors build_embeddings.py)
# ---------------------------------------------------------------------------

def load_frames(path):
    """Load frames from SE extractions JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    frames = []
    for result in data["results"]:
        if not result.get("has_relevant_content"):
            continue
        for i, frame in enumerate(result["frames"]):
            frame_id = f"{result['post_id']}-{i}"
            frames.append({
                "id": frame_id,
                "frame": frame,
                "post_meta": result,
            })
    return frames


# ---------------------------------------------------------------------------
# Subject extraction
# ---------------------------------------------------------------------------

def extract_unique_subjects(frames):
    """
    Extract unique subjects with metadata.

    Returns:
        subjects_list: sorted list of unique subject strings
        subject_meta:  dict mapping subject -> {frame_ids, scopes, frame_types, count}
    """
    subject_meta = {}

    for entry in frames:
        subject = (entry["frame"].get("subject") or "").strip()
        if not subject:
            continue

        frame_id = entry["id"]
        scope = entry["frame"].get("scope") or "unknown"
        frame_type = entry["frame"].get("frame_type") or "unknown"

        if subject not in subject_meta:
            subject_meta[subject] = {
                "frame_ids": [],
                "scopes": set(),
                "frame_types": set(),
                "count": 0,
            }

        subject_meta[subject]["frame_ids"].append(frame_id)
        subject_meta[subject]["scopes"].add(scope)
        subject_meta[subject]["frame_types"].add(frame_type)
        subject_meta[subject]["count"] += 1

    # Stable sorted order for reproducibility and cache validity
    subjects_list = sorted(subject_meta.keys())
    return subjects_list, subject_meta


# ---------------------------------------------------------------------------
# Embedding (with .npz cache)
# ---------------------------------------------------------------------------

def embed_subjects(subjects_list):
    """
    Embed all unique subjects. Cache to .npz for fast re-runs.
    Returns numpy array of shape (N, 3072).
    """
    # Check cache validity: subjects must match exactly
    if os.path.exists(CACHE_FILE):
        cached = np.load(CACHE_FILE, allow_pickle=True)
        cached_subjects = list(cached["subjects"])
        if cached_subjects == subjects_list:
            print(f"Loaded cached embeddings from {CACHE_FILE} "
                  f"({len(subjects_list)} subjects)")
            return cached["embeddings"]
        else:
            print(f"Cache stale ({len(cached_subjects)} cached vs "
                  f"{len(subjects_list)} current) -> re-embedding")

    print(f"Embedding {len(subjects_list)} subjects with {EMBEDDING_MODEL}...")
    t0 = time.time()

    vectors = []
    for start in range(0, len(subjects_list), BATCH_SIZE):
        batch = subjects_list[start:start + BATCH_SIZE]
        response = openai_client.embeddings.create(
            input=batch, model=EMBEDDING_MODEL
        )
        vectors.extend([item.embedding for item in response.data])
        done = start + len(batch)
        if done < len(subjects_list):
            print(f"  Embedded {done}/{len(subjects_list)}")

    embeddings = np.array(vectors, dtype=np.float32)

    # Save cache
    np.savez(
        CACHE_FILE,
        embeddings=embeddings,
        subjects=np.array(subjects_list, dtype=object),
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Cached to {CACHE_FILE}")

    return embeddings


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def load_blocklist(path, subjects_list):
    """
    Load blocklist of subject pairs that should never be in the same cluster.
    Returns list of (idx_a, idx_b) index pairs into subjects_list.
    """
    if not os.path.exists(path):
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pairs = data.get("blocked_pairs", [])
    subject_to_idx = {s: i for i, s in enumerate(subjects_list)}

    blocked = []
    skipped = 0
    for pair in pairs:
        if len(pair) != 2:
            continue
        a, b = pair[0], pair[1]
        if a in subject_to_idx and b in subject_to_idx:
            blocked.append((subject_to_idx[a], subject_to_idx[b]))
        else:
            skipped += 1

    if blocked:
        print(f"Loaded {len(blocked)} blocked pairs from {path}"
              + (f" ({skipped} skipped - not found)" if skipped else ""))
    return blocked


def cluster_subjects(embeddings, threshold, blocked_pairs=None):
    """
    Cluster subject embeddings by cosine distance.
    Blocked pairs have their distance forced to maximum before clustering.
    Returns array of cluster labels (one per subject).
    """
    print(f"Computing cosine distance matrix "
          f"({len(embeddings)} x {len(embeddings)})...")
    distance_matrix = cosine_distances(embeddings)

    # Force blocked pairs apart so they never merge
    if blocked_pairs:
        for idx_a, idx_b in blocked_pairs:
            distance_matrix[idx_a, idx_b] = 2.0
            distance_matrix[idx_b, idx_a] = 2.0

    print(f"Running AgglomerativeClustering "
          f"(threshold={threshold}, linkage=average)...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    n_clusters = len(set(labels))
    print(f"  Found {n_clusters} clusters")

    return labels


# ---------------------------------------------------------------------------
# Canonical name selection
# ---------------------------------------------------------------------------

def pick_canonical_name(members):
    """
    Pick the canonical name for a cluster of subjects.
    Heuristic: shortest subject string. Tiebreaker: alphabetically first.
    """
    return min(members, key=lambda s: (len(s), s))


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_output(subjects_list, subject_meta, labels):
    """
    Build the output data structure.
    Returns dict ready for JSON serialization.
    """
    # Group subjects by cluster label
    clusters_by_label = {}
    for i, subject in enumerate(subjects_list):
        label = int(labels[i])
        if label not in clusters_by_label:
            clusters_by_label[label] = []
        clusters_by_label[label].append(subject)

    # Build cluster records
    clusters = []
    subject_to_canonical = {}

    for label in sorted(clusters_by_label.keys()):
        members = clusters_by_label[label]
        canonical = pick_canonical_name(members)

        # Aggregate metadata across all members
        all_frame_ids = []
        all_scopes = set()
        all_frame_types = set()
        total_count = 0
        for subj in members:
            meta = subject_meta[subj]
            all_frame_ids.extend(meta["frame_ids"])
            all_scopes |= meta["scopes"]
            all_frame_types |= meta["frame_types"]
            total_count += meta["count"]

        cluster_record = {
            "cluster_id": label,
            "canonical_name": canonical,
            "member_count": len(members),
            "frame_count": total_count,
            "members": sorted(members),
            "scopes": sorted(all_scopes),
            "frame_types": sorted(all_frame_types),
            "frame_ids": sorted(all_frame_ids),
        }
        clusters.append(cluster_record)

        for subj in members:
            subject_to_canonical[subj] = canonical

    # Sort: multi-member first (desc by size), then singletons alphabetically
    clusters.sort(key=lambda c: (-c["member_count"], c["canonical_name"]))

    output = {
        "metadata": {
            "total_subjects": len(subjects_list),
            "total_clusters": len(clusters),
            "singleton_clusters": sum(
                1 for c in clusters if c["member_count"] == 1
            ),
            "multi_member_clusters": sum(
                1 for c in clusters if c["member_count"] > 1
            ),
            "threshold": None,  # set by caller
            "embedding_model": EMBEDDING_MODEL,
            "input_file": None,  # set by caller
        },
        "clusters": clusters,
        "subject_to_canonical": subject_to_canonical,
    }

    return output


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(output):
    """Print summary statistics to console."""
    meta = output["metadata"]
    clusters = output["clusters"]

    print(f"\n{'=' * 60}")
    print(f"SUBJECT NORMALIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total unique subjects : {meta['total_subjects']}")
    print(f"Total clusters        : {meta['total_clusters']}")
    print(f"Singleton clusters    : {meta['singleton_clusters']}")
    print(f"Multi-member clusters : {meta['multi_member_clusters']}")
    print(f"Distance threshold    : {meta['threshold']}")
    print(f"Cosine similarity     : {1.0 - meta['threshold']:.2f}")
    print(f"Embedding model       : {meta['embedding_model']}")

    # Largest clusters
    multi = [c for c in clusters if c["member_count"] > 1]
    if multi:
        show = min(len(multi), 20)
        print(f"\nLargest clusters (top {show} of {len(multi)}):")
        for c in multi[:show]:
            print(f"\n  [{c['member_count']} members] canonical: {c['canonical_name']!r}")
            for m in sorted(c["members"]):
                marker = " *" if m == c["canonical_name"] else "  "
                print(f"     {marker} {m!r}")
    else:
        print(f"\nNo multi-member clusters at threshold {meta['threshold']}.")
        print(f"  Try increasing --threshold (e.g. 0.20 for 0.80 cosine similarity)")

    # Cluster size distribution
    sizes = [c["member_count"] for c in clusters]
    size_dist = Counter(sizes)
    print(f"\nCluster size distribution:")
    for size in sorted(size_dist.keys()):
        print(f"  Size {size:3d}: {size_dist[size]:5d} clusters")

    print(f"{'=' * 60}")


def print_review(output):
    """Print all multi-member clusters for manual inspection."""
    clusters = output["clusters"]
    multi = [c for c in clusters if c["member_count"] > 1]

    if not multi:
        print("No multi-member clusters to review.")
        return

    print(f"\n{'=' * 60}")
    print(f"CLUSTER REVIEW ({len(multi)} multi-member clusters)")
    print(f"{'=' * 60}")

    for i, c in enumerate(multi, 1):
        print(f"\n--- Cluster {i}/{len(multi)} ---")
        print(f"  Canonical: {c['canonical_name']!r}")
        print(f"  Members ({c['member_count']}):")
        for m in sorted(c["members"]):
            marker = " *" if m == c["canonical_name"] else "  "
            print(f"   {marker} {m!r}")
        print(f"  Scopes: {', '.join(c['scopes'])}")
        print(f"  Frame types: {', '.join(c['frame_types'])}")
        print(f"  Total frames: {c['frame_count']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Normalize subject fields across extracted knowledge frames."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=(
            f"Cosine distance threshold for clustering "
            f"(default: {DEFAULT_THRESHOLD}). "
            f"Lower = tighter clusters (fewer merges). "
            f"0.15 = 0.85 cosine similarity."
        ),
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Print all multi-member clusters for manual inspection.",
    )
    parser.add_argument(
        "--input", default=INPUT_FILE, dest="input_file",
        help=f"Extractions JSON file (default: {INPUT_FILE})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"ERROR: {args.input_file} not found")
        sys.exit(1)

    # Step 1: Load frames
    print(f"Loading frames from {args.input_file}...")
    frames = load_frames(args.input_file)
    print(f"  Loaded {len(frames)} frames")

    # Step 2: Extract unique subjects
    subjects_list, subject_meta = extract_unique_subjects(frames)
    print(f"  Found {len(subjects_list)} unique subjects")

    if len(subjects_list) < 2:
        print("Not enough subjects to cluster.")
        sys.exit(0)

    # Step 3: Embed subjects (with caching)
    embeddings = embed_subjects(subjects_list)

    # Step 4: Load blocklist and cluster
    blocked_pairs = load_blocklist(BLOCKLIST_FILE, subjects_list)
    labels = cluster_subjects(embeddings, args.threshold, blocked_pairs)

    # Step 5: Build output
    output = build_output(subjects_list, subject_meta, labels)
    output["metadata"]["threshold"] = args.threshold
    output["metadata"]["input_file"] = args.input_file

    # Step 6: Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_FILE}")

    # Step 7: Print summary
    print_summary(output)

    # Step 8: Review mode
    if args.review:
        print_review(output)


if __name__ == "__main__":
    main()
