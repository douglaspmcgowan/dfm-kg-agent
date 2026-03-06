"""
Build a provenance graph from DFM extraction data (Stack Exchange).

Constructs a NetworkX directed multigraph with four node types:
  Thread, Post, Author, Frame

and six edge types:
  THREAD_CONTAINS_POST, POST_AUTHORED_BY, POST_CONTAINS_FRAME,
  SE_ANSWERS, FRAME_RELATED_TO, FRAME_CONTRADICTED_BY

No LLM calls. Pure reorganization of existing JSON data.

Usage:
    python build_graph.py                  # build + stats
    python build_graph.py --interactive    # build + stats + interactive query mode
"""

import json
import pickle
import argparse
import sys
from pathlib import Path

import networkx as nx

BASE_DIR     = Path(__file__).parent
SE_FILE      = BASE_DIR / "extractions_se_all.json"   # default; overridden by --input
AUTHORS_FILE = BASE_DIR / "se_author_frames.json"
GRAPHML_OUT  = BASE_DIR / "provenance_graph.graphml"
PICKLE_OUT   = BASE_DIR / "provenance_graph.gpickle"


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data():
    """
    Load SE extraction data and author frames.
    Returns (se_data dict, author_frames_by_name dict).
    """
    if not SE_FILE.exists():
        print(f"ERROR: {SE_FILE} not found.")
        sys.exit(1)

    with open(SE_FILE, "r", encoding="utf-8") as f:
        se_data = json.load(f)

    results = se_data.get("results") or []
    print(f"SE data      : {SE_FILE.name} ({len(results)} post results)")

    author_frames_by_name = {}
    if AUTHORS_FILE.exists():
        with open(AUTHORS_FILE, "r", encoding="utf-8") as f:
            af_data = json.load(f)
        for af in af_data.get("author_frames") or []:
            name = af.get("author_username") or ""
            if name:
                author_frames_by_name[name] = af
        print(f"Author frames: {AUTHORS_FILE.name} ({len(author_frames_by_name)} authors)")
    else:
        print(f"Author frames: {AUTHORS_FILE.name} not found -- will fall back to inline data only")

    return se_data, author_frames_by_name


# ─────────────────────────────────────────────
# NODE BUILDERS
# ─────────────────────────────────────────────

def add_thread_nodes(G, se_data):
    """Add one Thread node per unique thread_url. Returns count added."""
    results = se_data.get("results") or []

    # Accumulate thread data across all posts
    threads = {}  # thread_url -> {title, se_site, post_count}
    for r in results:
        url = r.get("thread_url") or ""
        if not url:
            continue
        if url not in threads:
            threads[url] = {
                "thread_title": r.get("thread_title") or "",
                "se_site":      r.get("se_site") or "",
                "post_count":   0,
            }
        threads[url]["post_count"] += 1

    for url, attrs in threads.items():
        G.add_node(
            url,
            node_type    = "Thread",
            thread_title = attrs["thread_title"],
            se_site      = attrs["se_site"],
            post_count   = attrs["post_count"],
        )

    return len(threads)


def add_post_nodes(G, se_data):
    """
    Add one Post node per result. Returns post_lookup dict (post_id -> result).
    """
    results = se_data.get("results") or []
    post_lookup = {}

    for r in results:
        pid = r.get("post_id") or ""
        if not pid:
            continue

        tags = r.get("se_tags") or []
        tags_str = ",".join(tags) if tags else ""

        is_acc = r.get("is_accepted_answer")
        is_acc_str = str(is_acc) if is_acc is not None else ""

        skip = r.get("skip_reason") or ""

        G.add_node(
            pid,
            node_type           = "Post",
            post_type           = r.get("post_type") or "",
            post_position       = r.get("post_position") if r.get("post_position") is not None else -1,
            post_date           = r.get("post_date") or "",
            post_score          = r.get("post_score") if r.get("post_score") is not None else 0,
            is_accepted_answer  = is_acc_str,
            se_tags             = tags_str,
            se_site             = r.get("se_site") or "",
            has_relevant_content = str(r.get("has_relevant_content", False)),
            skip_reason         = skip,
        )
        post_lookup[pid] = r

    return post_lookup


def add_author_nodes(G, se_data, author_frames_by_name):
    """
    Add one Author node per unique username.
    Primary source: author_frames_by_name (rich SE data).
    Fallback: inline result fields for authors not in primary source.
    Returns count added.
    """
    results = se_data.get("results") or []

    # Collect all usernames seen in results (for fallback detection)
    inline_by_name = {}
    for r in results:
        name = r.get("author_username") or ""
        if name and name not in inline_by_name:
            inline_by_name[name] = {
                "author_reputation":     r.get("author_reputation"),
                "author_credibility_tier": r.get("author_credibility_tier") or "low",
            }

    # Combine: primary source first, then any missing names from inline
    all_names = set(author_frames_by_name.keys()) | set(inline_by_name.keys())
    count = 0

    for name in all_names:
        node_id = f"author:{name}"

        if name in author_frames_by_name:
            af = author_frames_by_name[name]
            rep = af.get("reputation")
            rep_int = rep if rep is not None else -1
            ar = af.get("accept_rate")
            ar_int = ar if ar is not None else -1
            G.add_node(
                node_id,
                node_type        = "Author",
                author_username  = name,
                reputation       = rep_int,
                credibility_tier = af.get("author_credibility_tier") or "low",
                user_type        = af.get("user_type") or "",
                accept_rate      = ar_int,
                profile_link     = af.get("profile_link") or "",
            )
        else:
            # Fallback from inline extraction data
            inline = inline_by_name.get(name, {})
            rep = inline.get("author_reputation")
            rep_int = rep if rep is not None else -1
            G.add_node(
                node_id,
                node_type        = "Author",
                author_username  = name,
                reputation       = rep_int,
                credibility_tier = inline.get("author_credibility_tier") or "low",
                user_type        = "",
                accept_rate      = -1,
                profile_link     = "",
            )

        count += 1

    return count


def add_frame_nodes(G, se_data):
    """
    Add one Frame node per extracted frame. Frame ID = "{post_id}-{i}".
    Matches build_embeddings.py line 59 convention.
    Returns count added.
    """
    results = se_data.get("results") or []
    count = 0

    for r in results:
        if not r.get("has_relevant_content"):
            continue
        pid = r.get("post_id") or ""
        for i, frame in enumerate(r.get("frames") or []):
            frame_id = f"{pid}-{i}"
            G.add_node(
                frame_id,
                node_type        = "Frame",
                frame_type       = frame.get("frame_type") or "",
                scope            = frame.get("scope") or "",
                subject          = frame.get("subject") or "",
                applicability    = frame.get("applicability") or "",
                main_point       = frame.get("main_point") or "",
                source_quote     = frame.get("source_quote") or "",
                epistemic_stance = frame.get("epistemic_stance") or "",
                post_role        = frame.get("post_role") or "",
                example_value    = frame.get("example_value") or "",
                example_context  = frame.get("example_context") or "",
                thread_context   = frame.get("thread_context") or "",
                validated        = str(frame.get("validated")) if frame.get("validated") is not None else "",
            )
            count += 1

    return count


# ─────────────────────────────────────────────
# EDGE BUILDER
# ─────────────────────────────────────────────

def add_edges(G, se_data, post_lookup):
    """
    Add all six edge types. Returns dict of edge_type -> count.
    Always checks G.has_node() before adding an edge.
    """
    results = se_data.get("results") or []

    counts = {
        "THREAD_CONTAINS_POST":  0,
        "POST_AUTHORED_BY":      0,
        "POST_CONTAINS_FRAME":   0,
        "SE_ANSWERS":            0,
        "FRAME_RELATED_TO":      0,
        "FRAME_CONTRADICTED_BY": 0,
    }

    # Build post_id -> [frame_ids] index for FRAME_RELATED_TO resolution
    post_frames_index = {}
    for r in results:
        pid = r.get("post_id") or ""
        if not pid:
            continue
        ids = [f"{pid}-{i}" for i in range(len(r.get("frames") or []))]
        post_frames_index[pid] = ids

    for r in results:
        pid   = r.get("post_id") or ""
        url   = r.get("thread_url") or ""
        uname = r.get("author_username") or ""

        # 1. THREAD_CONTAINS_POST
        if url and pid and G.has_node(url) and G.has_node(pid):
            G.add_edge(url, pid, edge_type="THREAD_CONTAINS_POST")
            counts["THREAD_CONTAINS_POST"] += 1

        # 2. POST_AUTHORED_BY
        author_node = f"author:{uname}" if uname else ""
        if pid and author_node and G.has_node(pid) and G.has_node(author_node):
            G.add_edge(pid, author_node, edge_type="POST_AUTHORED_BY")
            counts["POST_AUTHORED_BY"] += 1

        # 3. POST_CONTAINS_FRAME
        for i, frame in enumerate(r.get("frames") or []):
            frame_id = f"{pid}-{i}"
            if G.has_node(pid) and G.has_node(frame_id):
                G.add_edge(pid, frame_id, edge_type="POST_CONTAINS_FRAME")
                counts["POST_CONTAINS_FRAME"] += 1

        # 4. SE_ANSWERS (answer post -> question post)
        q_ref = r.get("question_id_ref")
        if q_ref and pid and G.has_node(pid) and G.has_node(q_ref):
            G.add_edge(pid, q_ref, edge_type="SE_ANSWERS")
            counts["SE_ANSWERS"] += 1

        # 5. FRAME_RELATED_TO and 6. FRAME_CONTRADICTED_BY
        for i, frame in enumerate(r.get("frames") or []):
            frame_id = f"{pid}-{i}"
            if not G.has_node(frame_id):
                continue

            # FRAME_RELATED_TO: related_frames lists post_ids; link to all frames in those posts
            for related_pid in (frame.get("related_frames") or []):
                for target_fid in post_frames_index.get(related_pid, []):
                    if G.has_node(target_fid) and target_fid != frame_id:
                        G.add_edge(frame_id, target_fid, edge_type="FRAME_RELATED_TO")
                        counts["FRAME_RELATED_TO"] += 1

            # FRAME_CONTRADICTED_BY: contradicted_by lists frame IDs directly
            for contra_fid in (frame.get("contradicted_by") or []):
                if G.has_node(contra_fid):
                    G.add_edge(frame_id, contra_fid, edge_type="FRAME_CONTRADICTED_BY")
                    counts["FRAME_CONTRADICTED_BY"] += 1

    return counts


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────

def print_stats(G, edge_counts):
    """Print node counts, edge counts, connected components, and distributions."""
    # Node counts by type
    type_counts = {}
    for _, data in G.nodes(data=True):
        nt = data.get("node_type", "unknown")
        type_counts[nt] = type_counts.get(nt, 0) + 1

    print(f"\n{'='*60}")
    print(f"GRAPH STATS")
    print(f"{'='*60}")
    print(f"Total nodes : {G.number_of_nodes()}")
    for nt, cnt in sorted(type_counts.items()):
        print(f"  {nt:<20}: {cnt}")

    print(f"\nTotal edges : {G.number_of_edges()}")
    for et, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"  {et:<30}: {cnt}")

    # Connected components on undirected view
    und = G.to_undirected()
    components = list(nx.connected_components(und))
    sizes = sorted([len(c) for c in components], reverse=True)
    import statistics
    median_size = statistics.median(sizes) if sizes else 0
    print(f"\nConnected components (undirected view):")
    print(f"  Count   : {len(components)}")
    print(f"  Largest : {sizes[0] if sizes else 0} nodes")
    print(f"  Smallest: {sizes[-1] if sizes else 0} nodes")
    print(f"  Median  : {median_size:.0f} nodes")

    # Frame type distribution
    frame_types = {}
    for _, data in G.nodes(data=True):
        if data.get("node_type") == "Frame":
            ft = data.get("frame_type") or "unknown"
            frame_types[ft] = frame_types.get(ft, 0) + 1
    if frame_types:
        print(f"\nFrame types:")
        for ft, cnt in sorted(frame_types.items(), key=lambda x: -x[1]):
            print(f"  {ft:<20}: {cnt}")

    # Author credibility tier distribution
    tiers = {}
    for _, data in G.nodes(data=True):
        if data.get("node_type") == "Author":
            t = data.get("credibility_tier") or "unknown"
            tiers[t] = tiers.get(t, 0) + 1
    if tiers:
        print(f"\nAuthor credibility tiers:")
        for t, cnt in sorted(tiers.items()):
            print(f"  {t:<10}: {cnt}")

    # Top 5 most active authors (by POST_AUTHORED_BY edges)
    author_post_counts = {}
    for src, dst, data in G.edges(data=True):
        if data.get("edge_type") == "POST_AUTHORED_BY":
            uname = G.nodes[dst].get("author_username", dst)
            author_post_counts[uname] = author_post_counts.get(uname, 0) + 1
    if author_post_counts:
        top5 = sorted(author_post_counts.items(), key=lambda x: -x[1])[:5]
        print(f"\nTop 5 most active authors (by posts):")
        for name, cnt in top5:
            print(f"  {name:<30}: {cnt} posts")

    print(f"{'='*60}")


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

import re as _re
_XML_ILLEGAL = _re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F"
    r"\uD800-\uDFFF\uFFFE\uFFFF]"
)


def _clean_xml(val):
    """Strip characters illegal in XML 1.0 from a string."""
    if not isinstance(val, str):
        return val
    return _XML_ILLEGAL.sub("", val)


def save_graph(G):
    """
    Sanitize attributes for GraphML (no None, no lists, no nested objects,
    no XML-illegal control characters), then write GraphML and pickle files.
    """
    # Sanitize node attrs
    for node_id, data in G.nodes(data=True):
        for key, val in list(data.items()):
            if val is None:
                G.nodes[node_id][key] = ""
            elif isinstance(val, bool):
                G.nodes[node_id][key] = str(val)
            elif isinstance(val, str):
                G.nodes[node_id][key] = _clean_xml(val)

    for u, v, data in G.edges(data=True):
        for key, val in list(data.items()):
            if val is None:
                data[key] = ""
            elif isinstance(val, bool):
                data[key] = str(val)
            elif isinstance(val, str):
                data[key] = _clean_xml(val)

    nx.write_graphml(G, str(GRAPHML_OUT), encoding="utf-8")
    print(f"GraphML saved: {GRAPHML_OUT}")

    with open(PICKLE_OUT, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pickle saved : {PICKLE_OUT}")


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

def interactive_mode(G):
    """
    Interactive frame/node query loop.
    Enter a frame_id (or partial) to walk the provenance graph.
    """
    print("\nInteractive query mode. Enter a frame_id (or partial string) to inspect.")
    print("Commands: 'quit' or 'q' to exit, 'nodes' to list all Frame node IDs.\n")

    all_frame_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Frame"]

    while True:
        try:
            raw = input("frame_id> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if raw.lower() in ("quit", "q", "exit"):
            print("Exiting.")
            break

        if raw.lower() == "nodes":
            for fid in sorted(all_frame_ids):
                print(f"  {fid}")
            continue

        if not raw:
            continue

        # Exact match first; otherwise partial
        if G.has_node(raw):
            candidates = [raw]
        else:
            candidates = [fid for fid in all_frame_ids if raw in fid]

        if not candidates:
            print(f"  No frame found matching '{raw}'")
            continue

        if len(candidates) > 10:
            print(f"  {len(candidates)} matches -- showing first 10. Be more specific.")
            candidates = candidates[:10]

        for frame_id in candidates:
            _show_frame(G, frame_id)


def _show_frame(G, frame_id):
    """Print provenance walk for a single frame node."""
    if not G.has_node(frame_id):
        print(f"  Node not found: {frame_id}")
        return

    data = G.nodes[frame_id]
    print(f"\n--- Frame: {frame_id} ---")
    print(f"  frame_type      : {data.get('frame_type')}")
    print(f"  scope           : {data.get('scope')}")
    print(f"  subject         : {data.get('subject')}")
    print(f"  epistemic_stance: {data.get('epistemic_stance')}")
    mp = (data.get("main_point") or "")[:200]
    print(f"  main_point      : {mp}")

    # Walk predecessors to find parent Post
    parent_posts = [
        pred for pred in G.predecessors(frame_id)
        if G.nodes[pred].get("node_type") == "Post"
    ]
    for post_id in parent_posts:
        pd = G.nodes[post_id]
        print(f"\n  Parent Post: {post_id}")
        print(f"    post_type : {pd.get('post_type')}")
        print(f"    post_score: {pd.get('post_score')}")
        print(f"    post_date : {pd.get('post_date')}")

        # Walk post -> thread
        parent_threads = [
            pred for pred in G.predecessors(post_id)
            if G.nodes[pred].get("node_type") == "Thread"
        ]
        for thread_url in parent_threads:
            td = G.nodes[thread_url]
            print(f"\n  Parent Thread: {td.get('thread_title', thread_url)}")
            print(f"    url       : {thread_url}")
            print(f"    se_site   : {td.get('se_site')}")

        # Walk post -> author
        author_nodes = [
            succ for succ in G.successors(post_id)
            if G.nodes[succ].get("node_type") == "Author"
        ]
        for author_node in author_nodes:
            ad = G.nodes[author_node]
            print(f"\n  Author: {ad.get('author_username')}")
            rep = ad.get("reputation", -1)
            print(f"    reputation      : {rep if rep != -1 else 'unknown'}")
            print(f"    credibility_tier: {ad.get('credibility_tier')}")
            print(f"    user_type       : {ad.get('user_type')}")

        # SE_ANSWERS: does this post answer a question?
        answers_edges = [
            (succ, G.edges[post_id, succ, 0].get("edge_type"))
            for succ in G.successors(post_id)
            if any(
                G.edges[post_id, succ, k].get("edge_type") == "SE_ANSWERS"
                for k in G[post_id][succ]
            )
        ]
        for q_post_id, _ in answers_edges:
            qd = G.nodes[q_post_id]
            print(f"\n  Answers question: {q_post_id}")
            q_tags = qd.get("se_tags") or ""
            print(f"    tags: {q_tags}")

    # FRAME_RELATED_TO neighbors
    related = [
        succ for succ in G.successors(frame_id)
        if any(
            G.edges[frame_id, succ, k].get("edge_type") == "FRAME_RELATED_TO"
            for k in G[frame_id][succ]
        )
    ]
    if related:
        print(f"\n  Related frames ({len(related)}):")
        for r in related[:5]:
            print(f"    {r}: {G.nodes[r].get('subject', '')}")

    # FRAME_CONTRADICTED_BY neighbors
    contra = [
        succ for succ in G.successors(frame_id)
        if any(
            G.edges[frame_id, succ, k].get("edge_type") == "FRAME_CONTRADICTED_BY"
            for k in G[frame_id][succ]
        )
    ]
    if contra:
        print(f"\n  Contradicted by ({len(contra)}):")
        for c in contra[:5]:
            print(f"    {c}: {G.nodes[c].get('subject', '')}")

    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    global SE_FILE

    parser = argparse.ArgumentParser(
        description="Build DFM provenance graph from SE extraction data."
    )
    parser.add_argument(
        "--input",
        default=str(SE_FILE),
        help=f"Extractions JSON file (default: {SE_FILE.name})",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="After building, enter interactive query mode.",
    )
    args = parser.parse_args()

    SE_FILE = Path(args.input)

    print("Building provenance graph...")
    print(f"  SE file    : {SE_FILE}")
    print(f"  Authors    : {AUTHORS_FILE}")
    print(f"  GraphML out: {GRAPHML_OUT}")
    print(f"  Pickle out : {PICKLE_OUT}")
    print()

    se_data, author_frames_by_name = load_data()

    G = nx.MultiDiGraph()

    n_threads = add_thread_nodes(G, se_data)
    print(f"Thread nodes : {n_threads}")

    post_lookup = add_post_nodes(G, se_data)
    print(f"Post nodes   : {len(post_lookup)}")

    n_authors = add_author_nodes(G, se_data, author_frames_by_name)
    print(f"Author nodes : {n_authors}")

    n_frames = add_frame_nodes(G, se_data)
    print(f"Frame nodes  : {n_frames}")

    print(f"\nTotal nodes  : {G.number_of_nodes()}")

    edge_counts = add_edges(G, se_data, post_lookup)
    print(f"Total edges  : {G.number_of_edges()}")
    for et, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"  {et:<30}: {cnt}")

    print_stats(G, edge_counts)
    save_graph(G)

    if args.interactive:
        interactive_mode(G)


if __name__ == "__main__":
    main()
