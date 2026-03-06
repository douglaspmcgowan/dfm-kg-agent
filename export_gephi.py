"""
Export the provenance graph to GEXF format for Gephi visualization.

Two modes:
  --slim (default): Frame nodes + semantic edges only, stripped of heavy text
                    attributes. Small file, won't crash Gephi.
  --full:           All nodes and edges with all attributes.

Topic coloring:
  --topics FILE   : Color Frame nodes by topic cluster from a topic model output
                    JSON (supports both standard and grouped formats).
                    Adds topic_label attribute + Gephi viz:color.

Usage:
    python export_gephi.py                                         # slim, no topic colors
    python export_gephi.py --topics topic_model_output_grouped.json  # slim + topic colors
    python export_gephi.py --full --topics topic_model_output.json   # full + topic colors
    python export_gephi.py --output foo.gexf
"""

import json
import pickle
import argparse
import re
from pathlib import Path

import networkx as nx

BASE_DIR   = Path(__file__).parent
PICKLE_IN  = BASE_DIR / "provenance_graph.gpickle"

# XML 1.0 illegal characters
_XML_ILLEGAL = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F"
    r"\uD800-\uDFFF\uFFFE\uFFFF]"
)

# Semantic edge types (cross-cluster bridges)
SEMANTIC_EDGE_TYPES = {"CONTRADICTS", "MITIGATES", "SPECIFIES", "CAUSAL_LINK"}

# Lightweight attributes to keep per node type in slim mode
SLIM_ATTRS = {
    "Frame": {"node_type", "frame_type", "scope", "subject", "applicability",
              "epistemic_stance", "post_role", "topic_label"},
    "Post":  {"node_type", "post_type", "post_score", "se_site", "is_accepted_answer"},
    "Thread": {"node_type", "thread_title", "se_site", "post_count"},
    "Author": {"node_type", "author_username", "reputation", "credibility_tier"},
}

SLIM_EDGE_ATTRS = {"edge_type", "confidence"}


def clean_for_xml(val):
    """Strip characters illegal in XML 1.0."""
    if isinstance(val, str):
        return _XML_ILLEGAL.sub("", val)
    return val


def sanitize_attrs(data):
    """Clean attribute dict for XML serialization."""
    for key, val in list(data.items()):
        if val is None:
            data[key] = ""
        elif isinstance(val, bool):
            data[key] = str(val)
        elif isinstance(val, float):
            data[key] = round(val, 4)
        elif isinstance(val, (list, dict)):
            data[key] = str(val)
        else:
            data[key] = clean_for_xml(val)


def build_slim_graph(G):
    """
    Build a slim copy of the full graph: all nodes and edges, but with
    heavy text attributes stripped (source_quote, main_point, reasoning, etc.).
    Adds semantic_degree attribute to Frame nodes for sizing/coloring.
    """
    S = nx.MultiDiGraph()

    # Copy all nodes with only lightweight attributes
    for node_id, data in G.nodes(data=True):
        nt = data.get("node_type", "unknown")
        keep = SLIM_ATTRS.get(nt, {"node_type"})
        slim_data = {k: v for k, v in data.items() if k in keep}
        S.add_node(node_id, **slim_data)

    # Copy all edges with only lightweight attributes
    for u, v, k, data in G.edges(data=True, keys=True):
        slim_edge = {k2: v2 for k2, v2 in data.items() if k2 in SLIM_EDGE_ATTRS}
        S.add_edge(u, v, **slim_edge)

    # Add semantic_degree to Frame nodes
    for node_id, data in S.nodes(data=True):
        if data.get("node_type") == "Frame":
            sem_deg = sum(
                1 for _, _, ed in S.edges(node_id, data=True)
                if ed.get("edge_type") in SEMANTIC_EDGE_TYPES
            ) + sum(
                1 for _, _, ed in S.in_edges(node_id, data=True)
                if ed.get("edge_type") in SEMANTIC_EDGE_TYPES
            )
            S.nodes[node_id]["semantic_degree"] = sem_deg

    return S


# Distinct color palette for topic clusters (RGB tuples)
TOPIC_PALETTE = [
    (31, 119, 180),    # blue
    (255, 127, 14),    # orange
    (44, 160, 44),     # green
    (214, 39, 40),     # red
    (148, 103, 189),   # purple
    (140, 86, 75),     # brown
    (227, 119, 194),   # pink
    (127, 127, 127),   # gray
    (188, 189, 34),    # olive
    (23, 190, 207),    # cyan
    (174, 199, 232),   # light blue
    (255, 187, 120),   # light orange
    (152, 223, 138),   # light green
    (255, 152, 150),   # light red
    (197, 176, 213),   # light purple
    (196, 156, 148),   # light brown
    (247, 182, 210),   # light pink
    (199, 199, 199),   # light gray
    (219, 219, 141),   # light olive
    (158, 218, 229),   # light cyan
    (57, 59, 121),     # dark blue
    (82, 84, 163),     # indigo
    (107, 110, 207),   # slate blue
    (156, 158, 222),   # periwinkle
    (99, 121, 57),     # dark olive
    (140, 162, 82),    # moss green
    (181, 207, 107),   # lime
    (206, 219, 156),   # pale lime
    (132, 60, 57),     # dark red
    (173, 73, 74),     # brick
    (214, 97, 107),    # salmon
    (231, 150, 156),   # rose
    (123, 65, 115),    # plum
    (165, 81, 148),    # magenta
    (206, 109, 189),   # orchid
    (222, 158, 214),   # lavender
    (86, 86, 86),      # dark gray
    (115, 115, 115),   # medium gray
    (186, 186, 186),   # silver
    (224, 224, 224),   # near-white
]

OUTLIER_COLOR = (211, 211, 211)  # light gray for outliers


def load_topic_colors(topic_path):
    """
    Load topic assignments from a topic model JSON and return:
      frame_id -> (topic_label, (r, g, b))

    Supports both standard (topic_model_output.json) and grouped
    (topic_model_output_grouped.json) formats.
    """
    with open(topic_path, encoding="utf-8") as f:
        data = json.load(f)

    is_grouped = "super_topics" in data

    if is_grouped:
        # Use group_id / group_label
        group_labels = {}
        for st in data["super_topics"]:
            group_labels[st["group_id"]] = st["label"]

        # Assign colors by group_id
        group_ids = sorted(group_labels.keys())
        group_color = {gid: TOPIC_PALETTE[i % len(TOPIC_PALETTE)]
                       for i, gid in enumerate(group_ids)}

        frame_colors = {}
        for a in data["frame_assignments"]:
            gid = a.get("group_id", -1)
            if gid == -1:
                label = "Outlier"
                color = OUTLIER_COLOR
            else:
                label = group_labels.get(gid, f"Group {gid}")
                color = group_color.get(gid, OUTLIER_COLOR)
            frame_colors[a["frame_id"]] = (label, color)

        n_groups = len(group_labels)
        print(f"  Loaded {len(frame_colors)} topic assignments ({n_groups} super-groups)")

    else:
        # Standard format -- use topic_id / topic_label
        topic_labels = {}
        for t in data["topics"]:
            raw = t["label"]
            clean = raw.split("_", 1)[1] if "_" in raw else raw
            topic_labels[t["topic_id"]] = clean

        topic_ids = sorted(tid for tid in topic_labels if tid != -1)
        topic_color = {tid: TOPIC_PALETTE[i % len(TOPIC_PALETTE)]
                       for i, tid in enumerate(topic_ids)}

        frame_colors = {}
        for a in data["frame_assignments"]:
            tid = a["topic_id"]
            if tid == -1:
                label = "Outlier"
                color = OUTLIER_COLOR
            else:
                label = topic_labels.get(tid, f"Topic {tid}")
                color = topic_color.get(tid, OUTLIER_COLOR)
            frame_colors[a["frame_id"]] = (label, color)

        print(f"  Loaded {len(frame_colors)} topic assignments ({len(topic_ids)} topics)")

    return frame_colors


def apply_topic_colors(G, frame_colors):
    """
    Add topic_label attribute and Gephi viz:color to Frame nodes in the graph.
    Non-Frame nodes get a neutral color.
    """
    colored = 0
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "Frame":
            fid = node_id  # Frame node IDs match frame_id format
            if fid in frame_colors:
                label, (r, g, b) = frame_colors[fid]
                data["topic_label"] = label
                data["viz"] = {"color": {"r": r, "g": g, "b": b, "a": 1.0}}
                colored += 1
            else:
                data["topic_label"] = "Unknown"
                data["viz"] = {"color": {"r": 211, "g": 211, "b": 211, "a": 1.0}}
        else:
            # Non-Frame nodes: subtle neutral colors by type
            nt = data.get("node_type", "")
            if nt == "Post":
                data["viz"] = {"color": {"r": 240, "g": 240, "b": 240, "a": 0.6}}
            elif nt == "Thread":
                data["viz"] = {"color": {"r": 220, "g": 220, "b": 220, "a": 0.6}}
            elif nt == "Author":
                data["viz"] = {"color": {"r": 230, "g": 230, "b": 230, "a": 0.6}}
            else:
                data["viz"] = {"color": {"r": 200, "g": 200, "b": 200, "a": 0.6}}

    print(f"  Applied topic colors to {colored} Frame nodes")


def main():
    parser = argparse.ArgumentParser(description="Export provenance graph to GEXF for Gephi")
    parser.add_argument("--output", default=None,
                        help="Output GEXF file path")
    parser.add_argument("--input", default=str(PICKLE_IN),
                        help="Input gpickle file path")
    parser.add_argument("--full", action="store_true",
                        help="Export full graph (all nodes/edges/attributes). Default is slim semantic-only.")
    parser.add_argument("--topics", default=None,
                        help="Topic model output JSON for coloring Frame nodes by topic cluster")
    args = parser.parse_args()

    if args.output is None:
        suffix = "_full" if args.full else "_slim"
        if args.topics:
            suffix += "_topics"
        args.output = str(BASE_DIR / f"provenance_graph{suffix}.gexf")

    print(f"Loading graph from {args.input} ...")
    with open(args.input, "rb") as f:
        G = pickle.load(f)

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    if args.full:
        export_graph = G
        print("\nFull export mode (all nodes/edges/attributes)")
    else:
        export_graph = build_slim_graph(G)
        print(f"\nSlim export mode (Frame nodes + semantic edges only)")
        print(f"  Frame nodes:    {export_graph.number_of_nodes()}")
        print(f"  Semantic edges: {export_graph.number_of_edges()}")

        # Connectivity stats
        isolated = sum(1 for n in export_graph.nodes() if export_graph.degree(n) == 0)
        connected = export_graph.number_of_nodes() - isolated
        print(f"  Connected:      {connected}")
        print(f"  Isolated:       {isolated}")

    # Apply topic colors if requested
    if args.topics:
        print(f"\nLoading topic colors from {args.topics} ...")
        frame_colors = load_topic_colors(args.topics)
        apply_topic_colors(export_graph, frame_colors)

    # Sanitize for XML (skip 'viz' dicts -- Gephi needs them as-is)
    for _, data in export_graph.nodes(data=True):
        viz = data.pop("viz", None)
        sanitize_attrs(data)
        if viz:
            data["viz"] = viz
    for _, _, _, data in export_graph.edges(data=True, keys=True):
        sanitize_attrs(data)

    # Node type summary
    type_counts = {}
    for _, data in export_graph.nodes(data=True):
        nt = data.get("node_type", "unknown")
        type_counts[nt] = type_counts.get(nt, 0) + 1
    print("\n  Node types:")
    for nt, cnt in sorted(type_counts.items()):
        print(f"    {nt:<20}: {cnt}")

    # Edge type summary
    edge_counts = {}
    for _, _, data in export_graph.edges(data=True):
        et = data.get("edge_type", "unknown")
        edge_counts[et] = edge_counts.get(et, 0) + 1
    print("\n  Edge types:")
    for et, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"    {et:<30}: {cnt}")

    # Write GEXF
    nx.write_gexf(export_graph, args.output, encoding="utf-8", prettyprint=True)
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nGEXF written: {args.output} ({size_mb:.1f} MB)")

    print("\nGephi tips:")
    print("  1. File -> Open -> select the .gexf file")
    if args.topics:
        print("  2. Node colors are pre-set by topic cluster (loaded from file)")
        print("     - To switch: Appearance -> Nodes -> Color -> Partition -> topic_label")
        print("     - Or by type: Appearance -> Nodes -> Color -> Partition -> node_type")
    else:
        print("  2. Appearance -> Nodes -> Color -> Partition -> node_type")
    print("  3. Appearance -> Nodes -> Size -> Ranking -> Degree (min 3, max 40)")
    print("  4. Appearance -> Edges -> Color -> Partition -> edge_type")
    print("  5. Layout -> ForceAtlas 2:")
    print("     - Scaling: 2-5 (LOWER = tighter clusters)")
    print("     - Gravity: 5-10")
    print("     - LinLog: ON, Prevent Overlap: OFF initially")
    print("     - Run 60+ seconds, then Stop")


if __name__ == "__main__":
    main()
