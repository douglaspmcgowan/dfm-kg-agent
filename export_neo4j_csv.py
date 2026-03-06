"""
Export the provenance graph to CSV files + a Cypher import script for Neo4j.

Reads provenance_graph.gpickle (NetworkX MultiDiGraph) and writes:
  neo4j_import/
    nodes_threads.csv
    nodes_posts.csv
    nodes_authors.csv
    nodes_frames.csv
    edges_thread_contains_post.csv
    edges_post_authored_by.csv
    edges_post_contains_frame.csv
    edges_se_answers.csv
    edges_frame_related_to.csv
    edges_frame_contradicted_by.csv
    neo4j_import.cypher

Usage:
    python export_neo4j_csv.py
"""

import csv
import pickle
import sys
from pathlib import Path

import networkx as nx

BASE_DIR   = Path(__file__).parent
PICKLE_IN  = BASE_DIR / "provenance_graph.gpickle"
OUTPUT_DIR = BASE_DIR / "neo4j_import"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_str(val):
    """Convert a value to a string safe for CSV. None -> empty string."""
    if val is None:
        return ""
    return str(val)


def load_graph():
    if not PICKLE_IN.exists():
        print(f"ERROR: {PICKLE_IN} not found. Run build_graph.py first.")
        sys.exit(1)
    with open(PICKLE_IN, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def nodes_by_type(G, node_type):
    """Yield (node_id, attrs_dict) for all nodes of given type."""
    for nid, data in G.nodes(data=True):
        if data.get("node_type") == node_type:
            yield nid, data


def edges_by_type(G, edge_type):
    """Yield (source, target) for all edges of given type."""
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == edge_type:
            yield u, v


def write_csv(filepath, fieldnames, rows):
    """Write a list of dicts to a CSV file."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  {filepath.name}: {len(rows)} rows")


# ─────────────────────────────────────────────
# NODE EXPORTERS
# ─────────────────────────────────────────────

def export_threads(G):
    fields = ["_neo4j_id", "thread_title", "post_count", "se_site"]
    rows = []
    for nid, data in nodes_by_type(G, "Thread"):
        rows.append({
            "_neo4j_id":    safe_str(nid),
            "thread_title": safe_str(data.get("thread_title")),
            "post_count":   safe_str(data.get("post_count")),
            "se_site":      safe_str(data.get("se_site")),
        })
    write_csv(OUTPUT_DIR / "nodes_threads.csv", fields, rows)
    return len(rows)


def export_posts(G):
    fields = [
        "_neo4j_id", "post_type", "post_position", "post_date",
        "post_score", "is_accepted_answer", "se_tags", "se_site",
        "has_relevant_content", "skip_reason",
    ]
    rows = []
    for nid, data in nodes_by_type(G, "Post"):
        rows.append({
            "_neo4j_id":          safe_str(nid),
            "post_type":          safe_str(data.get("post_type")),
            "post_position":      safe_str(data.get("post_position")),
            "post_date":          safe_str(data.get("post_date")),
            "post_score":         safe_str(data.get("post_score")),
            "is_accepted_answer": safe_str(data.get("is_accepted_answer")),
            "se_tags":            safe_str(data.get("se_tags")),
            "se_site":            safe_str(data.get("se_site")),
            "has_relevant_content": safe_str(data.get("has_relevant_content")),
            "skip_reason":        safe_str(data.get("skip_reason")),
        })
    write_csv(OUTPUT_DIR / "nodes_posts.csv", fields, rows)
    return len(rows)


def export_authors(G):
    fields = [
        "_neo4j_id", "author_username", "reputation",
        "credibility_tier", "user_type", "accept_rate", "profile_link",
    ]
    rows = []
    for nid, data in nodes_by_type(G, "Author"):
        rows.append({
            "_neo4j_id":       safe_str(nid),
            "author_username": safe_str(data.get("author_username")),
            "reputation":      safe_str(data.get("reputation")),
            "credibility_tier": safe_str(data.get("credibility_tier")),
            "user_type":       safe_str(data.get("user_type")),
            "accept_rate":     safe_str(data.get("accept_rate")),
            "profile_link":    safe_str(data.get("profile_link")),
        })
    write_csv(OUTPUT_DIR / "nodes_authors.csv", fields, rows)
    return len(rows)


def export_frames(G):
    fields = [
        "_neo4j_id", "frame_type", "scope", "subject", "applicability",
        "main_point", "source_quote", "epistemic_stance", "post_role",
        "example_value", "example_context", "thread_context", "validated",
    ]
    rows = []
    for nid, data in nodes_by_type(G, "Frame"):
        rows.append({
            "_neo4j_id":       safe_str(nid),
            "frame_type":      safe_str(data.get("frame_type")),
            "scope":           safe_str(data.get("scope")),
            "subject":         safe_str(data.get("subject")),
            "applicability":   safe_str(data.get("applicability")),
            "main_point":      safe_str(data.get("main_point")),
            "source_quote":    safe_str(data.get("source_quote")),
            "epistemic_stance": safe_str(data.get("epistemic_stance")),
            "post_role":       safe_str(data.get("post_role")),
            "example_value":   safe_str(data.get("example_value")),
            "example_context": safe_str(data.get("example_context")),
            "thread_context":  safe_str(data.get("thread_context")),
            "validated":       safe_str(data.get("validated")),
        })
    write_csv(OUTPUT_DIR / "nodes_frames.csv", fields, rows)
    return len(rows)


# ─────────────────────────────────────────────
# EDGE EXPORTERS
# ─────────────────────────────────────────────

EDGE_TYPES = [
    ("THREAD_CONTAINS_POST",  "edges_thread_contains_post.csv"),
    ("POST_AUTHORED_BY",      "edges_post_authored_by.csv"),
    ("POST_CONTAINS_FRAME",   "edges_post_contains_frame.csv"),
    ("SE_ANSWERS",            "edges_se_answers.csv"),
    ("FRAME_RELATED_TO",      "edges_frame_related_to.csv"),
    ("FRAME_CONTRADICTED_BY", "edges_frame_contradicted_by.csv"),
]


def export_edges(G):
    """Export all edge types. Returns dict of edge_type -> count."""
    counts = {}
    for edge_type, filename in EDGE_TYPES:
        rows = [{"_source": u, "_target": v} for u, v in edges_by_type(G, edge_type)]
        write_csv(OUTPUT_DIR / filename, ["_source", "_target"], rows)
        counts[edge_type] = len(rows)
    return counts


# ─────────────────────────────────────────────
# CYPHER SCRIPT GENERATOR
# ─────────────────────────────────────────────

def generate_cypher():
    """Write neo4j_import.cypher with LOAD CSV statements using absolute file paths."""
    # Build file:/// URL prefix from the output directory (forward slashes required)
    csv_dir = OUTPUT_DIR.resolve().as_posix()  # e.g. C:/Users/dougl/Documents/dfm_scraping/neo4j_import
    prefix = f"file:///{csv_dir}"              # e.g. file:///C:/Users/dougl/.../neo4j_import

    cypher = f"""// ============================================================
// Neo4j Import Script for DFM Provenance Graph
// Generated by export_neo4j_csv.py
//
// SETUP (do one of these BEFORE running this script):
//
//   Option A -- Change Neo4j's import directory to point at these CSVs:
//     1. In Neo4j Desktop, click three dots on your DBMS -> Settings
//     2. Find the line:  server.directories.import=import
//     3. Change it to:   server.directories.import={csv_dir}
//     4. Restart the database
//
//   Option B -- Copy CSVs into Neo4j's import directory:
//     1. In Neo4j Desktop, click three dots on your DBMS -> Open folder -> Import
//     2. Copy all .csv files from neo4j_import/ into that folder
//     3. Then use the ALTERNATIVE (relative) paths commented below each LOAD CSV
//
// Then open Neo4j Browser (Query tab, or http://localhost:7474) and
// run each section below one at a time (select block + Ctrl+Enter).
// ============================================================


// --- Constraints (run first) ---

CREATE CONSTRAINT thread_id IF NOT EXISTS FOR (t:Thread) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT frame_id IF NOT EXISTS FOR (f:Frame) REQUIRE f.id IS UNIQUE;


// --- Nodes: Threads ---

LOAD CSV WITH HEADERS FROM '{prefix}/nodes_threads.csv' AS row
// Alternative if CSVs are in Neo4j's import dir: FROM 'file:///nodes_threads.csv'
CREATE (t:Thread {{
  id:        row._neo4j_id,
  title:     row.thread_title,
  postCount: toInteger(row.post_count),
  seSite:    row.se_site
}});


// --- Nodes: Posts ---

LOAD CSV WITH HEADERS FROM '{prefix}/nodes_posts.csv' AS row
// Alternative if CSVs are in Neo4j's import dir: FROM 'file:///nodes_posts.csv'
CREATE (p:Post {{
  id:                row._neo4j_id,
  postType:          row.post_type,
  position:          toInteger(row.post_position),
  date:              row.post_date,
  score:             toInteger(row.post_score),
  isAcceptedAnswer:  row.is_accepted_answer,
  seTags:            row.se_tags,
  seSite:            row.se_site,
  hasRelevantContent: row.has_relevant_content,
  skipReason:        row.skip_reason
}});


// --- Nodes: Authors ---

LOAD CSV WITH HEADERS FROM '{prefix}/nodes_authors.csv' AS row
// Alternative if CSVs are in Neo4j's import dir: FROM 'file:///nodes_authors.csv'
CREATE (a:Author {{
  id:              row._neo4j_id,
  username:        row.author_username,
  reputation:      toInteger(row.reputation),
  credibilityTier: row.credibility_tier,
  userType:        row.user_type,
  acceptRate:      toInteger(row.accept_rate),
  profileLink:     row.profile_link
}});


// --- Nodes: Frames ---

LOAD CSV WITH HEADERS FROM '{prefix}/nodes_frames.csv' AS row
// Alternative if CSVs are in Neo4j's import dir: FROM 'file:///nodes_frames.csv'
CREATE (f:Frame {{
  id:              row._neo4j_id,
  frameType:       row.frame_type,
  scope:           row.scope,
  subject:         row.subject,
  applicability:   row.applicability,
  mainPoint:       row.main_point,
  sourceQuote:     row.source_quote,
  epistemicStance: row.epistemic_stance,
  postRole:        row.post_role,
  exampleValue:    row.example_value,
  exampleContext:  row.example_context,
  threadContext:   row.thread_context,
  validated:       row.validated
}});


// --- Edges: Thread CONTAINS Post ---

LOAD CSV WITH HEADERS FROM '{prefix}/edges_thread_contains_post.csv' AS row
MATCH (t:Thread {{id: row._source}})
MATCH (p:Post {{id: row._target}})
CREATE (t)-[:CONTAINS]->(p);


// --- Edges: Post AUTHORED_BY Author ---

LOAD CSV WITH HEADERS FROM '{prefix}/edges_post_authored_by.csv' AS row
MATCH (p:Post {{id: row._source}})
MATCH (a:Author {{id: row._target}})
CREATE (p)-[:AUTHORED_BY]->(a);


// --- Edges: Post CONTAINS Frame ---

LOAD CSV WITH HEADERS FROM '{prefix}/edges_post_contains_frame.csv' AS row
MATCH (p:Post {{id: row._source}})
MATCH (f:Frame {{id: row._target}})
CREATE (p)-[:CONTAINS]->(f);


// --- Edges: Post ANSWERS Post ---

LOAD CSV WITH HEADERS FROM '{prefix}/edges_se_answers.csv' AS row
MATCH (a:Post {{id: row._source}})
MATCH (q:Post {{id: row._target}})
CREATE (a)-[:ANSWERS]->(q);


// --- Edges: Frame RELATED_TO Frame (may be empty) ---

LOAD CSV WITH HEADERS FROM '{prefix}/edges_frame_related_to.csv' AS row
MATCH (f1:Frame {{id: row._source}})
MATCH (f2:Frame {{id: row._target}})
CREATE (f1)-[:RELATED_TO]->(f2);


// --- Edges: Frame CONTRADICTED_BY Frame (may be empty) ---

LOAD CSV WITH HEADERS FROM '{prefix}/edges_frame_contradicted_by.csv' AS row
MATCH (f1:Frame {{id: row._source}})
MATCH (f2:Frame {{id: row._target}})
CREATE (f1)-[:CONTRADICTED_BY]->(f2);


// --- Verification queries ---
// Run these after import to check counts:

// MATCH (n) RETURN labels(n)[0] AS label, count(n) ORDER BY label;
// MATCH ()-[r]->() RETURN type(r) AS relType, count(r) ORDER BY relType;
"""

    cypher_path = OUTPUT_DIR / "neo4j_import.cypher"
    with open(cypher_path, "w", encoding="utf-8") as f:
        f.write(cypher)
    print(f"  {cypher_path.name}: Cypher import script written")
    print(f"  CSV path prefix: {prefix}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("Exporting provenance graph to Neo4j CSV format...")
    print(f"  Input : {PICKLE_IN}")
    print(f"  Output: {OUTPUT_DIR}/")
    print()

    G = load_graph()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Export nodes
    print("\nExporting nodes:")
    n_threads = export_threads(G)
    n_posts   = export_posts(G)
    n_authors = export_authors(G)
    n_frames  = export_frames(G)

    # Export edges
    print("\nExporting edges:")
    edge_counts = export_edges(G)

    # Generate Cypher script
    print("\nGenerating Cypher script:")
    generate_cypher()

    # Summary
    print(f"\n{'='*50}")
    print(f"EXPORT SUMMARY")
    print(f"{'='*50}")
    print(f"Nodes: {n_threads + n_posts + n_authors + n_frames} total")
    print(f"  Thread: {n_threads}")
    print(f"  Post:   {n_posts}")
    print(f"  Author: {n_authors}")
    print(f"  Frame:  {n_frames}")
    print(f"\nEdges: {sum(edge_counts.values())} total")
    for et, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"  {et}: {cnt}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Copy CSV files to Neo4j's import/ directory")
    print(f"  2. Open Neo4j Browser (http://localhost:7474)")
    print(f"  3. Run sections from neo4j_import.cypher one at a time")


if __name__ == "__main__":
    main()
