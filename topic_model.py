"""
Discover high-level topics across extracted knowledge frames using BERTopic.

Uses pre-computed embeddings from Qdrant (frames_knowledge collection),
UMAP for dimensionality reduction, HDBSCAN for clustering, c-TF-IDF for
keyword extraction, and GPT for human-readable topic labels.

Usage:
    python topic_model.py                          # default (min_cluster_size=5)
    python topic_model.py --min-cluster-size 10    # larger clusters, fewer topics
    python topic_model.py --reduce-topics 30       # merge down to 30 topics
    python topic_model.py --review                 # print all topics with labels

Output:
    topic_model_output.json  -- topics, labels, frame assignments, metadata

Requires: pip install bertopic umap-learn
"""

import json
import sys
import os
import time
import argparse
from collections import Counter

# Fix Windows cp1252 terminal encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from qdrant_client import QdrantClient
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from bertopic import BERTopic
from bertopic.representation import OpenAI as OpenAIRepresentation
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = os.getenv("INPUT_FILE", "extractions_se_all.json")
QDRANT_PATH = "./qdrant_data"
KNOWLEDGE_COLLECTION = "frames_knowledge"
OUTPUT_FILE = "topic_model_output.json"

LLM_MODEL = "gpt-4.1-mini"

# Domain stop words that appear across nearly all topics and dilute keywords
DOMAIN_STOP_WORDS = [
    "cnc", "machine", "machining", "tool", "tools", "cut", "cutting",
    "use", "using", "used", "make", "making", "work", "works", "working",
    "like", "just", "get", "got", "good", "better", "best", "need",
    "way", "one", "would", "will", "also", "much", "may", "well",
    "can", "don", "doesn", "isn", "won", "didn", "shouldn", "ve",
    "really", "thing", "things", "going", "want", "know", "think",
    "said", "say", "sure", "right", "lot", "bit", "even", "actually",
    "point", "case", "set", "run", "running",
]

openai_client = OpenAI()

# ---------------------------------------------------------------------------
# Data loading
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
# Pull embeddings from Qdrant
# ---------------------------------------------------------------------------

def pull_embeddings_from_qdrant(frames):
    """
    Pull pre-computed embeddings from Qdrant frames_knowledge collection.
    Returns numpy array of shape (N, 3072) aligned to frames list order.
    """
    qdrant = QdrantClient(path=QDRANT_PATH)

    # Build frame_id -> index mapping for ordering
    frame_id_to_idx = {f["id"]: i for i, f in enumerate(frames)}

    # Scroll all points with vectors
    all_points = []
    offset = None
    batch_size = 500

    while True:
        points, next_offset = qdrant.scroll(
            collection_name=KNOWLEDGE_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=["frame_id"],
            with_vectors=True,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    qdrant.close()

    # Map Qdrant vectors to frame order
    dim = len(all_points[0].vector)
    embeddings = np.zeros((len(frames), dim), dtype=np.float32)
    matched = 0

    for point in all_points:
        fid = point.payload.get("frame_id", "")
        if fid in frame_id_to_idx:
            embeddings[frame_id_to_idx[fid]] = point.vector
            matched += 1

    print(f"Pulled {matched}/{len(frames)} embeddings from Qdrant "
          f"({dim}-dim vectors)")

    if matched < len(frames):
        missing = len(frames) - matched
        print(f"  WARNING: {missing} frames have no embedding in Qdrant")

    return embeddings


# ---------------------------------------------------------------------------
# Document text builder (for c-TF-IDF keyword extraction)
# ---------------------------------------------------------------------------

def build_doc_texts(frames):
    """
    Build document texts for c-TF-IDF by concatenating subject + main_point.
    Subject provides topical anchor; main_point provides the knowledge claim.
    """
    docs = []
    for entry in frames:
        frame = entry["frame"]
        subject = (frame.get("subject") or "").strip()
        main_point = (frame.get("main_point") or "").strip()

        if subject and main_point:
            docs.append(f"{subject}. {main_point}")
        elif main_point:
            docs.append(main_point)
        elif subject:
            docs.append(subject)
        else:
            docs.append("")
    return docs


# ---------------------------------------------------------------------------
# BERTopic fitting
# ---------------------------------------------------------------------------

def fit_topic_model(docs, embeddings, min_cluster_size=5, min_samples=2,
                    reduce_to=None):
    """
    Fit BERTopic with pre-computed embeddings.

    Uses sklearn.cluster.HDBSCAN (no standalone hdbscan package needed),
    UMAP for dimensionality reduction, CountVectorizer with domain stop words,
    and OpenAI representation model for LLM-generated topic labels.
    """
    # UMAP: reduce 3072-dim to 5-dim for HDBSCAN
    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN clustering (sklearn built-in)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    # CountVectorizer with English + domain stop words, unigrams + bigrams
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    all_stop_words = list(ENGLISH_STOP_WORDS) + DOMAIN_STOP_WORDS
    vectorizer_model = CountVectorizer(
        stop_words=all_stop_words,
        ngram_range=(1, 2),
        min_df=2,
    )

    # LLM representation model for topic labels
    representation_model = OpenAIRepresentation(
        client=openai_client,
        model=LLM_MODEL,
        nr_docs=6,
        doc_length=200,
        tokenizer="char",
        chat=True,
    )

    print(f"Fitting BERTopic (min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples})...")
    t0 = time.time()

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=reduce_to,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    elapsed = time.time() - t0
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = topics.count(-1)
    print(f"  Done in {elapsed:.1f}s: {n_topics} topics, "
          f"{n_outliers} outliers ({n_outliers*100//len(docs)}%)")

    return topic_model, topics, probs


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_output(topic_model, topics, frames, min_cluster_size, reduce_to):
    """Build JSON-serializable output structure."""
    topic_info = topic_model.get_topic_info()

    # Build topic records
    topic_records = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            label = "Outliers"
            keywords = []
        else:
            label = row.get("Name", f"Topic_{tid}")
            # Clean up BERTopic's default "N_keyword1_keyword2" format
            # if LLM label is present it replaces this
            topic_words = topic_model.get_topic(tid)
            keywords = [w for w, _ in topic_words[:10]] if topic_words else []

        # Find frames in this topic
        topic_frame_indices = [i for i, t in enumerate(topics) if t == tid]

        # Scope distribution
        scopes = Counter()
        frame_types = Counter()
        for idx in topic_frame_indices:
            scope = frames[idx]["frame"].get("scope") or "unknown"
            ftype = frames[idx]["frame"].get("frame_type") or "unknown"
            scopes[scope] += 1
            frame_types[ftype] += 1

        # Top representative frames (by main_point, up to 5)
        top_frames = []
        for idx in topic_frame_indices[:5]:
            top_frames.append({
                "frame_id": frames[idx]["id"],
                "subject": frames[idx]["frame"].get("subject", ""),
                "main_point": frames[idx]["frame"].get("main_point", ""),
            })

        topic_records.append({
            "topic_id": int(tid),
            "label": label,
            "keywords": keywords,
            "n_frames": len(topic_frame_indices),
            "scope_distribution": dict(scopes.most_common()),
            "frame_type_distribution": dict(frame_types.most_common()),
            "top_frames": top_frames,
        })

    # Sort: outliers last, then by frame count descending
    topic_records.sort(
        key=lambda t: (t["topic_id"] == -1, -t["n_frames"])
    )

    # Frame assignments
    frame_assignments = []
    topic_label_map = {t["topic_id"]: t["label"] for t in topic_records}
    for i, frame_entry in enumerate(frames):
        tid = topics[i]
        frame_assignments.append({
            "frame_id": frame_entry["id"],
            "topic_id": int(tid),
            "topic_label": topic_label_map.get(tid, f"Topic_{tid}"),
        })

    n_outliers = sum(1 for t in topics if t == -1)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)

    output = {
        "metadata": {
            "total_frames": len(frames),
            "total_topics": n_topics,
            "outlier_count": n_outliers,
            "outlier_pct": round(n_outliers * 100 / len(frames), 1),
            "min_cluster_size": min_cluster_size,
            "reduce_to": reduce_to,
            "llm_model": LLM_MODEL,
            "embedding_source": f"Qdrant {KNOWLEDGE_COLLECTION}",
        },
        "topics": topic_records,
        "frame_assignments": frame_assignments,
    }

    return output


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(output):
    """Print summary statistics."""
    meta = output["metadata"]
    topics = output["topics"]

    print(f"\n{'=' * 60}")
    print(f"TOPIC MODEL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total frames        : {meta['total_frames']}")
    print(f"Total topics        : {meta['total_topics']}")
    print(f"Outliers            : {meta['outlier_count']} "
          f"({meta['outlier_pct']}%)")
    print(f"Min cluster size    : {meta['min_cluster_size']}")
    print(f"LLM label model     : {meta['llm_model']}")

    # Topic size distribution
    non_outlier = [t for t in topics if t["topic_id"] != -1]
    if non_outlier:
        sizes = [t["n_frames"] for t in non_outlier]
        print(f"\nTopic sizes: min={min(sizes)}, max={max(sizes)}, "
              f"median={sorted(sizes)[len(sizes)//2]}, "
              f"mean={sum(sizes)/len(sizes):.1f}")

    # Top 10 largest topics
    show = min(len(non_outlier), 10)
    print(f"\nLargest topics (top {show}):")
    for t in non_outlier[:show]:
        kw = ", ".join(t["keywords"][:5])
        print(f"  [{t['n_frames']:3d} frames] {t['label']}")
        print(f"             keywords: {kw}")

    print(f"{'=' * 60}")


def print_review(output):
    """Print all topics for manual inspection."""
    topics = output["topics"]
    non_outlier = [t for t in topics if t["topic_id"] != -1]

    if not non_outlier:
        print("No topics to review.")
        return

    print(f"\n{'=' * 60}")
    print(f"TOPIC REVIEW ({len(non_outlier)} topics)")
    print(f"{'=' * 60}")

    for i, t in enumerate(non_outlier, 1):
        print(f"\n--- Topic {i}/{len(non_outlier)} (id={t['topic_id']}) ---")
        print(f"  Label: {t['label']}")
        print(f"  Frames: {t['n_frames']}")
        print(f"  Keywords: {', '.join(t['keywords'][:8])}")

        # Scope distribution
        if t["scope_distribution"]:
            top_scopes = list(t["scope_distribution"].items())[:3]
            scope_str = ", ".join(f"{s}={n}" for s, n in top_scopes)
            print(f"  Scopes: {scope_str}")

        # Frame type distribution
        if t["frame_type_distribution"]:
            top_types = list(t["frame_type_distribution"].items())[:3]
            type_str = ", ".join(f"{ft}={n}" for ft, n in top_types)
            print(f"  Frame types: {type_str}")

        # Representative frames
        if t["top_frames"]:
            print(f"  Representative frames:")
            for rf in t["top_frames"][:3]:
                mp = rf["main_point"][:120] + "..." if len(rf["main_point"]) > 120 else rf["main_point"]
                print(f"    - [{rf['frame_id']}] {mp}")

    # Outlier info
    outlier = [t for t in topics if t["topic_id"] == -1]
    if outlier:
        o = outlier[0]
        print(f"\n--- Outliers ({o['n_frames']} frames) ---")
        if o["top_frames"]:
            print(f"  Sample outlier frames:")
            for rf in o["top_frames"][:3]:
                mp = rf["main_point"][:120] + "..." if len(rf["main_point"]) > 120 else rf["main_point"]
                print(f"    - [{rf['frame_id']}] {mp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Topic modeling over extracted knowledge frames using BERTopic."
    )
    parser.add_argument(
        "--input", default=INPUT_FILE, dest="input_file",
        help=f"Extractions JSON file (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=5,
        help="HDBSCAN min_cluster_size (default: 5). Higher = fewer, larger topics.",
    )
    parser.add_argument(
        "--min-samples", type=int, default=2,
        help="HDBSCAN min_samples (default: 2).",
    )
    parser.add_argument(
        "--reduce-topics", type=int, default=None,
        help="Merge topics down to this count after initial clustering.",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output JSON file (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Print all topics for manual inspection.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"ERROR: {args.input_file} not found")
        sys.exit(1)

    # Step 1: Load frames
    print(f"Loading frames from {args.input_file}...")
    frames = load_frames(args.input_file)
    print(f"  Loaded {len(frames)} frames")

    if len(frames) < 2:
        print("Not enough frames for topic modeling.")
        sys.exit(0)

    # Step 2: Pull embeddings from Qdrant
    print(f"\nPulling embeddings from Qdrant ({KNOWLEDGE_COLLECTION})...")
    embeddings = pull_embeddings_from_qdrant(frames)

    # Step 3: Build document texts for c-TF-IDF
    docs = build_doc_texts(frames)
    avg_len = sum(len(d) for d in docs) // len(docs)
    print(f"Built {len(docs)} document texts (avg {avg_len} chars)")

    # Step 4: Fit BERTopic
    topic_model, topics, probs = fit_topic_model(
        docs, embeddings,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        reduce_to=args.reduce_topics,
    )

    # Step 5: Build output
    output = build_output(
        topic_model, topics, frames,
        min_cluster_size=args.min_cluster_size,
        reduce_to=args.reduce_topics,
    )

    # Step 6: Save
    out_path = args.output or OUTPUT_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Step 7: Print summary
    print_summary(output)

    # Step 8: Review mode
    if args.review:
        print_review(output)


if __name__ == "__main__":
    main()
