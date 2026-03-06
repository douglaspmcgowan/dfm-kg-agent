"""
Group the existing 40 fine-grained BERTopic topics into ~8 super-topics
by computing topic centroids in the embedding space and clustering them
with AgglomerativeClustering.

This preserves the original 40-topic assignments and adds a higher-level
grouping layer on top, rather than re-running HDBSCAN from scratch.

Usage:
    python group_topics.py                        # default: 8 groups
    python group_topics.py --n-groups 6           # 6 groups
    python group_topics.py --n-groups 10          # 10 groups
    python group_topics.py --review               # print detailed review

Output:
    topic_model_output_grouped.json  -- original structure + super_topics + updated frame_assignments
"""

import json
import sys
import os
import argparse
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_PATH = "./qdrant_data"
KNOWLEDGE_COLLECTION = "frames_knowledge"
TOPIC_FILE = "topic_model_output.json"
EXTRACTIONS_FILE = "extractions_se_all.json"
OUTPUT_FILE = "topic_model_output_grouped.json"
LLM_MODEL = "gpt-4.1-mini"

openai_client = OpenAI()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_topic_data(topic_path):
    """Load the existing 40-topic model output."""
    with open(topic_path, encoding="utf-8") as f:
        return json.load(f)


def load_extractions(path):
    """Load frame metadata keyed by frame_id."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    frames = {}
    for result in data["results"]:
        if not result.get("has_relevant_content"):
            continue
        for i, frame in enumerate(result["frames"]):
            fid = f"{result['post_id']}-{i}"
            frames[fid] = {
                "frame_type": frame.get("frame_type", ""),
                "scope": frame.get("scope", ""),
                "subject": frame.get("subject", ""),
                "main_point": frame.get("main_point", ""),
                "se_site": result.get("se_site", ""),
            }
    return frames


def pull_embeddings():
    """Pull all vectors + frame_ids from Qdrant."""
    qdrant = QdrantClient(path=QDRANT_PATH)
    all_points = []
    offset = None

    while True:
        points, next_offset = qdrant.scroll(
            collection_name=KNOWLEDGE_COLLECTION,
            limit=500,
            offset=offset,
            with_payload=["frame_id"],
            with_vectors=True,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    qdrant.close()

    fid_to_vec = {}
    for p in all_points:
        fid = p.payload.get("frame_id", "")
        fid_to_vec[fid] = np.array(p.vector, dtype=np.float32)

    print(f"Pulled {len(fid_to_vec)} vectors from Qdrant")
    return fid_to_vec


# ---------------------------------------------------------------------------
# Compute topic centroids
# ---------------------------------------------------------------------------

def compute_centroids(topic_data, fid_to_vec):
    """Compute mean embedding vector for each of the 40 topics."""
    # Build topic_id -> list of frame_ids (excluding outliers)
    topic_frames = {}
    for a in topic_data["frame_assignments"]:
        tid = a["topic_id"]
        if tid == -1:
            continue
        topic_frames.setdefault(tid, []).append(a["frame_id"])

    # Compute centroids
    centroids = {}
    for tid, fids in sorted(topic_frames.items()):
        vecs = [fid_to_vec[fid] for fid in fids if fid in fid_to_vec]
        if vecs:
            centroids[tid] = np.mean(vecs, axis=0)

    print(f"Computed centroids for {len(centroids)} topics")
    return centroids


# ---------------------------------------------------------------------------
# Cluster centroids into super-groups
# ---------------------------------------------------------------------------

def cluster_centroids(centroids, n_groups, topic_data):
    """
    Cluster topic centroids into n_groups using size-weighted KMeans.

    Uses KMeans on L2-normalized centroids (equivalent to cosine similarity)
    which produces more balanced groups than agglomerative clustering.
    Tiny niche topics (< 15 frames) are assigned to their nearest group
    after clustering the larger topics, preventing them from dominating.
    """
    topic_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[tid] for tid in topic_ids])

    # Topic sizes for size-aware decisions
    topic_sizes = {}
    for t in topic_data["topics"]:
        if t["topic_id"] != -1:
            topic_sizes[t["topic_id"]] = t["n_frames"]

    # Separate substantial topics from tiny niche ones
    MIN_SIZE_FOR_CLUSTERING = 50
    substantial = [(i, tid) for i, tid in enumerate(topic_ids)
                   if topic_sizes.get(tid, 0) >= MIN_SIZE_FOR_CLUSTERING]
    tiny = [(i, tid) for i, tid in enumerate(topic_ids)
            if topic_sizes.get(tid, 0) < MIN_SIZE_FOR_CLUSTERING]

    print(f"  Substantial topics (>={MIN_SIZE_FOR_CLUSTERING} frames): {len(substantial)}")
    print(f"  Tiny niche topics: {len(tiny)}")

    if len(substantial) <= n_groups:
        # Fewer substantial topics than groups -- each gets its own, tiny ones assigned to nearest
        n_groups = len(substantial)
        print(f"  Adjusted to {n_groups} groups (limited by substantial topics)")

    # L2-normalize for cosine-equivalent KMeans
    normed = normalize(centroid_matrix, norm="l2")

    # Cluster only substantial topics
    sub_indices = [i for i, _ in substantial]
    sub_matrix = normed[sub_indices]

    kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=50)
    sub_labels = kmeans.fit_predict(sub_matrix)

    # Assign substantial topics
    topic_to_group = {}
    for (_, tid), label in zip(substantial, sub_labels):
        topic_to_group[tid] = int(label)

    # Assign tiny topics to nearest cluster center
    if tiny:
        centers = kmeans.cluster_centers_  # already normalized
        for i, tid in tiny:
            vec = normed[i].reshape(1, -1)
            dists = cosine_distances(vec, centers)[0]
            nearest = int(np.argmin(dists))
            topic_to_group[tid] = nearest

    # Print distance stats
    dist_matrix = cosine_distances(centroid_matrix)
    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    print(f"  Centroid distances: min={upper.min():.3f}, max={upper.max():.3f}, "
          f"mean={upper.mean():.3f}, median={np.median(upper):.3f}")

    return topic_to_group


# ---------------------------------------------------------------------------
# LLM labeling for super-groups
# ---------------------------------------------------------------------------

def label_super_groups(groups, topic_data, frame_meta):
    """Use LLM to generate a short label for each super-group based on its sub-topics."""
    # Build topic_id -> clean label
    topic_labels = {}
    for t in topic_data["topics"]:
        raw = t["label"]
        clean = raw.split("_", 1)[1] if "_" in raw else raw
        topic_labels[t["topic_id"]] = clean

    # Build topic_id -> top subjects
    topic_subjects = {}
    for t in topic_data["topics"]:
        if t["topic_id"] == -1:
            continue
        subjects = [f["subject"] for f in t.get("top_frames", [])]
        topic_subjects[t["topic_id"]] = subjects

    labels = {}
    for group_id, topic_ids in sorted(groups.items()):
        # Gather info about this group
        sub_labels = [topic_labels.get(tid, f"Topic {tid}") for tid in topic_ids]
        sub_subjects = []
        for tid in topic_ids:
            sub_subjects.extend(topic_subjects.get(tid, []))

        # Gather scope distribution
        scopes = Counter()
        for tid in topic_ids:
            for t in topic_data["topics"]:
                if t["topic_id"] == tid:
                    for scope, count in t.get("scope_distribution", {}).items():
                        scopes[scope] += count

        top_scopes = [s for s, _ in scopes.most_common(3)]

        prompt = (
            f"These {len(topic_ids)} sub-topics belong to one cluster of machining/engineering knowledge:\n"
            f"  Sub-topics: {', '.join(sub_labels)}\n"
            f"  Sample subjects: {', '.join(sub_subjects[:10])}\n"
            f"  Top scopes: {', '.join(top_scopes)}\n\n"
            f"Write a concise 2-4 word label that captures the shared theme. "
            f"Return ONLY the label, nothing else."
        )

        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )
        label = response.choices[0].message.content.strip().strip('"').strip("'")
        labels[group_id] = label
        print(f"  Group {group_id}: {label} ({len(topic_ids)} sub-topics: {', '.join(sub_labels[:4])}{'...' if len(sub_labels) > 4 else ''})")

    return labels


# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------

def build_grouped_output(topic_data, topic_to_group, group_labels, frame_meta, n_groups):
    """Build output JSON with super-topic layer added."""

    # Clean topic labels
    clean_topic_labels = {}
    for t in topic_data["topics"]:
        raw = t["label"]
        clean = raw.split("_", 1)[1] if "_" in raw else raw
        clean_topic_labels[t["topic_id"]] = clean

    # Invert: group_id -> [topic_ids]
    groups = {}
    for tid, gid in topic_to_group.items():
        groups.setdefault(gid, []).append(tid)

    # Build super_topics list
    super_topics = []
    for gid in sorted(groups.keys()):
        tids = sorted(groups[gid])
        sub_topic_records = []
        total_frames = 0

        # Aggregate scope and frame type distributions
        agg_scopes = Counter()
        agg_types = Counter()

        for tid in tids:
            for t in topic_data["topics"]:
                if t["topic_id"] == tid:
                    sub_topic_records.append({
                        "topic_id": tid,
                        "label": clean_topic_labels.get(tid, f"Topic {tid}"),
                        "n_frames": t["n_frames"],
                    })
                    total_frames += t["n_frames"]
                    for scope, cnt in t.get("scope_distribution", {}).items():
                        agg_scopes[scope] += cnt
                    for ft, cnt in t.get("frame_type_distribution", {}).items():
                        agg_types[ft] += cnt

        super_topics.append({
            "group_id": gid,
            "label": group_labels.get(gid, f"Group {gid}"),
            "n_frames": total_frames,
            "n_sub_topics": len(tids),
            "sub_topics": sub_topic_records,
            "scope_distribution": dict(agg_scopes.most_common()),
            "frame_type_distribution": dict(agg_types.most_common()),
        })

    # Sort by frame count descending
    super_topics.sort(key=lambda x: -x["n_frames"])

    # Build frame assignments with group info
    frame_assignments = []
    for a in topic_data["frame_assignments"]:
        tid = a["topic_id"]
        gid = topic_to_group.get(tid, -1)  # outliers get -1

        frame_assignments.append({
            "frame_id": a["frame_id"],
            "topic_id": tid,
            "topic_label": clean_topic_labels.get(tid, "Outlier"),
            "group_id": gid,
            "group_label": group_labels.get(gid, "Outlier") if gid != -1 else "Outlier",
        })

    # Count outlier frames
    n_outliers = sum(1 for a in frame_assignments if a["group_id"] == -1)

    output = {
        "metadata": {
            "total_frames": topic_data["metadata"]["total_frames"],
            "total_super_topics": len(super_topics),
            "total_sub_topics": topic_data["metadata"]["total_topics"],
            "outlier_count": n_outliers,
            "outlier_pct": round(n_outliers * 100 / topic_data["metadata"]["total_frames"], 1),
            "n_groups_requested": n_groups,
            "grouping_method": "agglomerative_clustering_cosine_centroids",
            "labeling_model": LLM_MODEL,
            "source_topic_file": TOPIC_FILE,
        },
        "super_topics": super_topics,
        "topics": topic_data["topics"],  # preserve original 40 topics
        "frame_assignments": frame_assignments,
    }

    return output


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(output):
    """Print summary of grouped topics."""
    meta = output["metadata"]
    print(f"\n{'=' * 70}")
    print(f"SUPER-TOPIC GROUPING SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total frames          : {meta['total_frames']}")
    print(f"Super-topics          : {meta['total_super_topics']}")
    print(f"Sub-topics (original) : {meta['total_sub_topics']}")
    print(f"Outliers              : {meta['outlier_count']} ({meta['outlier_pct']}%)")
    print(f"Grouping method       : {meta['grouping_method']}")
    print()

    for st in output["super_topics"]:
        sub_labels = [s["label"] for s in st["sub_topics"]]
        sub_sizes = [f"{s['label']} ({s['n_frames']})" for s in st["sub_topics"]]

        scopes = st.get("scope_distribution", {})
        top_scopes = ", ".join(f"{k}={v}" for k, v in list(scopes.items())[:4])

        ftypes = st.get("frame_type_distribution", {})
        top_types = ", ".join(f"{k}={v}" for k, v in list(ftypes.items())[:4])

        print(f"  [{st['n_frames']:4d} frames] {st['label']} ({st['n_sub_topics']} sub-topics)")
        print(f"           Sub-topics: {', '.join(sub_sizes)}")
        print(f"           Scopes: {top_scopes}")
        print(f"           Types:  {top_types}")
        print()

    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Group existing 40 BERTopic topics into super-topic clusters"
    )
    parser.add_argument(
        "--n-groups", type=int, default=8,
        help="Number of super-topic groups (default: 8)",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Print detailed review of each group",
    )
    args = parser.parse_args()

    # Load data
    print("Loading topic model ...")
    topic_data = load_topic_data(TOPIC_FILE)
    print(f"  {topic_data['metadata']['total_topics']} topics, "
          f"{topic_data['metadata']['total_frames']} frames")

    print("Loading frame metadata ...")
    frame_meta = load_extractions(EXTRACTIONS_FILE)
    print(f"  {len(frame_meta)} frames")

    # Pull embeddings
    fid_to_vec = pull_embeddings()

    # Compute topic centroids
    centroids = compute_centroids(topic_data, fid_to_vec)

    # Cluster centroids
    print(f"\nClustering {len(centroids)} topic centroids into {args.n_groups} groups ...")
    topic_to_group = cluster_centroids(centroids, args.n_groups, topic_data)

    # LLM label each group
    print("\nLabeling super-groups with LLM ...")
    # Invert mapping
    groups_inv = {}
    for tid, gid in topic_to_group.items():
        groups_inv.setdefault(gid, []).append(tid)
    group_labels = label_super_groups(groups_inv, topic_data, frame_meta)

    # Build output
    out_path = args.output or OUTPUT_FILE
    output = build_grouped_output(topic_data, topic_to_group, group_labels, frame_meta, args.n_groups)

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Summary
    print_summary(output)


if __name__ == "__main__":
    main()
