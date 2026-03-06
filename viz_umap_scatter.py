"""
UMAP scatter plot of knowledge frame embeddings, colored by BERTopic cluster.

Pulls pre-computed 3072-dim vectors from Qdrant (frames_knowledge),
reduces to 2D with UMAP, merges with topic assignments + frame metadata,
and produces an interactive Plotly scatter plot saved as HTML + static PNG.

Usage:
    python viz_umap_scatter.py                     # default: color by topic
    python viz_umap_scatter.py --color frame_type  # color by frame_type
    python viz_umap_scatter.py --color scope       # color by scope domain

Output:
    umap_scatter_<color>.html   -- interactive Plotly (hover for details)
    umap_scatter_<color>.png    -- static image for paper
"""

import json
import sys
import os
import argparse
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient
from umap import UMAP
import plotly.express as px
import plotly.io as pio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_PATH = "./qdrant_data"
KNOWLEDGE_COLLECTION = "frames_knowledge"
TOPIC_FILE = "topic_model_output.json"
EXTRACTIONS_FILE = "extractions_se_all.json"

# UMAP parameters (match topic_model.py settings for consistency)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_SEED = 42
UMAP_CACHE_FILE = "umap_coords_cache.npz"

# Plot sizing
PLOT_WIDTH = 1400
PLOT_HEIGHT = 900
POINT_SIZE = 3
OUTLIER_OPACITY = 0.25
CLUSTER_OPACITY = 0.75

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_extractions(path):
    """Load frames from extractions JSON, return dict keyed by frame_id."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    frames = {}
    for result in data["results"]:
        if not result.get("has_relevant_content"):
            continue
        for i, frame in enumerate(result["frames"]):
            frame_id = f"{result['post_id']}-{i}"
            frames[frame_id] = {
                "frame_type": frame.get("frame_type", ""),
                "scope": frame.get("scope", ""),
                "subject": frame.get("subject", ""),
                "main_point": frame.get("main_point", ""),
                "applicability": frame.get("applicability", ""),
                "epistemic_stance": frame.get("epistemic_stance", ""),
                "post_role": frame.get("post_role", ""),
                "se_site": result.get("se_site", ""),
            }
    return frames


def load_topic_assignments(path):
    """Load topic assignments and labels from topic model output.

    Supports both standard topic_model_output.json and grouped
    topic_model_output_grouped.json (which has super_topics + group_id).
    When a grouped file is detected, uses group_id/group_label for coloring.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Detect grouped format
    is_grouped = "super_topics" in data

    if is_grouped:
        # Use super-topic groups for coloring
        group_labels = {}
        for st in data["super_topics"]:
            group_labels[st["group_id"]] = st["label"]

        assignments = {}
        for a in data["frame_assignments"]:
            gid = a.get("group_id", -1)
            assignments[a["frame_id"]] = {
                "topic_id": gid,
                "topic_label": group_labels.get(gid, "Outlier") if gid != -1 else "Outlier",
            }

        return assignments, group_labels, data["metadata"]

    else:
        # Standard format -- use topic_id/topic_label
        topic_labels = {}
        for t in data["topics"]:
            # Labels are like "0_Precision hole-making" -- strip the numeric prefix
            raw_label = t["label"]
            clean = raw_label.split("_", 1)[1] if "_" in raw_label else raw_label
            topic_labels[t["topic_id"]] = clean

        assignments = {}
        for a in data["frame_assignments"]:
            assignments[a["frame_id"]] = {
                "topic_id": a["topic_id"],
                "topic_label": topic_labels.get(a["topic_id"], f"Topic {a['topic_id']}"),
            }

        return assignments, topic_labels, data["metadata"]


def pull_embeddings():
    """Pull all vectors + frame_ids from Qdrant frames_knowledge collection."""
    qdrant = QdrantClient(path=QDRANT_PATH)

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

    frame_ids = [p.payload.get("frame_id", "") for p in all_points]
    vectors = np.array([p.vector for p in all_points], dtype=np.float32)

    print(f"Pulled {len(frame_ids)} vectors ({vectors.shape[1]}-dim) from Qdrant")
    return frame_ids, vectors


# ---------------------------------------------------------------------------
# UMAP reduction
# ---------------------------------------------------------------------------

def reduce_umap(vectors, frame_ids, use_cache=True):
    """Reduce high-dim vectors to 2D with UMAP. Caches result to disk."""
    if use_cache and os.path.exists(UMAP_CACHE_FILE):
        cached = np.load(UMAP_CACHE_FILE, allow_pickle=True)
        cached_ids = list(cached["frame_ids"])
        if len(cached_ids) == len(frame_ids) and cached_ids == frame_ids:
            print(f"Loaded cached UMAP coords ({cached['coords'].shape})")
            return cached["coords"]
        print("Cache exists but frame IDs changed -- recomputing UMAP ...")

    print(f"Running UMAP ({vectors.shape[0]} points, {vectors.shape[1]}d -> 2d) ...")
    reducer = UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_SEED,
        low_memory=True,
    )
    coords = reducer.fit_transform(vectors)
    print(f"UMAP complete: {coords.shape}")

    # Cache for fast re-coloring
    np.savez(UMAP_CACHE_FILE, coords=coords, frame_ids=np.array(frame_ids, dtype=object))
    print(f"Cached UMAP coords to {UMAP_CACHE_FILE}")

    return coords


# ---------------------------------------------------------------------------
# Build scatter data
# ---------------------------------------------------------------------------

def build_scatter_data(frame_ids, coords, topic_assignments, frame_meta, color_by):
    """Merge UMAP coordinates with metadata into a list of dicts for Plotly."""
    rows = []
    for i, fid in enumerate(frame_ids):
        topic_info = topic_assignments.get(fid, {"topic_id": -1, "topic_label": "Outlier"})
        meta = frame_meta.get(fid, {})

        topic_id = topic_info["topic_id"]
        is_outlier = topic_id == -1

        row = {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "frame_id": fid,
            "topic_id": topic_id,
            "topic_label": topic_info["topic_label"] if not is_outlier else "Outlier",
            "frame_type": meta.get("frame_type", ""),
            "scope": meta.get("scope", "").replace("_", " "),
            "subject": (meta.get("subject", "") or "")[:60],
            "main_point": (meta.get("main_point", "") or "")[:100],
            "applicability": meta.get("applicability", ""),
            "epistemic_stance": meta.get("epistemic_stance", "").replace("_", " "),
            "se_site": meta.get("se_site", ""),
            "is_outlier": is_outlier,
        }

        # The color column value
        if color_by == "topic":
            row["color"] = row["topic_label"]
        elif color_by == "frame_type":
            row["color"] = row["frame_type"]
        elif color_by == "scope":
            row["color"] = row["scope"]
        elif color_by == "se_site":
            row["color"] = row["se_site"]
        elif color_by == "applicability":
            row["color"] = row["applicability"]
        else:
            row["color"] = row["topic_label"]

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------

# Qualitative palette for topics (enough distinct colors for 39 topics + outlier)
TOPIC_COLORS = (
    px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
    + px.colors.qualitative.Alphabet
)

# Manually defined palettes for categorical fields
FRAME_TYPE_COLORS = {
    "heuristic": "#2196F3",
    "observation": "#4CAF50",
    "risk": "#F44336",
    "principle": "#FF9800",
    "workaround": "#9C27B0",
    "comparison": "#00BCD4",
    "case": "#795548",
    "other": "#9E9E9E",
}

SCOPE_COLORS = {
    "machining process": "#2196F3",
    "tooling": "#F44336",
    "design geometry": "#4CAF50",
    "material": "#FF9800",
    "machine capability": "#9C27B0",
    "programming": "#00BCD4",
    "shop operations": "#795548",
    "quality": "#E91E63",
    "cost": "#607D8B",
    "other": "#9E9E9E",
}


def make_figure(rows, color_by, topic_metadata):
    """Create Plotly scatter figure."""
    import pandas as pd

    df = pd.DataFrame(rows)

    # Sort so outliers are drawn first (behind cluster points)
    df = df.sort_values("is_outlier", ascending=False).reset_index(drop=True)

    # Set opacity: outliers are faded
    df["opacity"] = df["is_outlier"].apply(
        lambda x: OUTLIER_OPACITY if x else CLUSTER_OPACITY
    )

    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "Subject: %{customdata[1]}<br>"
        "Type: %{customdata[2]} | Scope: %{customdata[3]}<br>"
        "Topic: %{customdata[4]}<br>"
        "Stance: %{customdata[5]}<br>"
        "%{customdata[6]}"
        "<extra></extra>"
    )

    customdata = df[
        ["frame_id", "subject", "frame_type", "scope", "topic_label", "epistemic_stance", "main_point"]
    ].values

    # Choose color scheme
    if color_by == "frame_type":
        color_map = FRAME_TYPE_COLORS
        title_suffix = "Frame Type"
    elif color_by == "scope":
        color_map = SCOPE_COLORS
        title_suffix = "Scope Domain"
    elif color_by == "topic":
        # Build dynamic color map for topics
        unique_labels = df[~df["is_outlier"]]["color"].unique()
        color_map = {}
        for i, label in enumerate(sorted(unique_labels)):
            color_map[label] = TOPIC_COLORS[i % len(TOPIC_COLORS)]
        color_map["Outlier"] = "#D3D3D3"
        title_suffix = "BERTopic Cluster"
    else:
        color_map = None
        title_suffix = color_by.replace("_", " ").title()

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="color",
        color_discrete_map=color_map,
        custom_data=["frame_id", "subject", "frame_type", "scope", "topic_label", "epistemic_stance", "main_point"],
        title=f"UMAP Projection of {len(df):,} Knowledge Frames -- Colored by {title_suffix}",
        labels={"x": "UMAP 1", "y": "UMAP 2", "color": title_suffix},
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
    )

    fig.update_traces(
        marker=dict(size=POINT_SIZE, line=dict(width=0)),
        hovertemplate=hover_template,
    )

    # Apply per-point opacity (outliers faded)
    for trace in fig.data:
        trace_name = trace.name
        if trace_name == "Outlier":
            trace.marker.opacity = OUTLIER_OPACITY
        else:
            trace.marker.opacity = CLUSTER_OPACITY

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12),
        title_font_size=16,
        legend=dict(
            title_text=title_suffix,
            font=dict(size=10),
            itemsizing="constant",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Add annotation with stats (handle both grouped and standard metadata)
    n_topics = topic_metadata.get("total_super_topics") or topic_metadata.get("total_topics", "?")
    n_outliers = topic_metadata["outlier_count"]
    outlier_pct = topic_metadata["outlier_pct"]

    fig.add_annotation(
        text=f"{len(df):,} frames | {n_topics} topics | {n_outliers} outliers ({outlier_pct:.0f}%)",
        xref="paper", yref="paper",
        x=0.01, y=-0.02,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UMAP scatter plot of knowledge frames")
    parser.add_argument(
        "--color", default="topic",
        choices=["topic", "frame_type", "scope", "se_site", "applicability"],
        help="Field to color points by (default: topic)",
    )
    parser.add_argument("--topics", default=TOPIC_FILE,
                        help=f"Topic model output JSON (default: {TOPIC_FILE})")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML output")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG output")
    parser.add_argument("--no-cache", action="store_true", help="Force UMAP recompute")
    args = parser.parse_args()

    # Load data
    print(f"Loading topic assignments from {args.topics} ...")
    topic_assignments, topic_labels, topic_metadata = load_topic_assignments(args.topics)
    print(f"  {len(topic_assignments)} frame assignments, {len(topic_labels)} topics")

    print("Loading frame metadata ...")
    frame_meta = load_extractions(EXTRACTIONS_FILE)
    print(f"  {len(frame_meta)} frames loaded")

    # Pull embeddings from Qdrant
    frame_ids, vectors = pull_embeddings()

    # UMAP reduction (cached after first run)
    coords = reduce_umap(vectors, frame_ids, use_cache=not args.no_cache)

    # Build scatter data
    rows = build_scatter_data(frame_ids, coords, topic_assignments, frame_meta, args.color)

    # Create figure
    fig = make_figure(rows, args.color, topic_metadata)

    # Save outputs -- include topic file label if non-default
    if args.topics != TOPIC_FILE:
        topic_tag = os.path.splitext(os.path.basename(args.topics))[0].replace("topic_model_output", "").strip("_") or "custom"
        base = f"umap_scatter_{args.color}_{topic_tag}"
    else:
        base = f"umap_scatter_{args.color}"

    if not args.no_html:
        html_path = f"{base}.html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"Saved interactive plot: {html_path}")

    if not args.no_png:
        png_path = f"{base}.png"
        try:
            fig.write_image(png_path, scale=2)
            print(f"Saved static image: {png_path}")
        except Exception as e:
            print(f"PNG export failed ({e}) -- install kaleido: pip install kaleido")
            print("HTML output is still available.")

    print("Done.")


if __name__ == "__main__":
    main()
