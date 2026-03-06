"""
Push the provenance graph directly to a running Neo4j instance.

Reads provenance_graph.gpickle and creates all nodes and edges using
the neo4j Python driver with batched transactions.

Environment variables:
    NEO4J_URI      - default: bolt://localhost:7687
    NEO4J_USER     - default: neo4j
    NEO4J_PASSWORD - default: password
    NEO4J_DATABASE - default: neo4j
    BATCH_SIZE     - default: 500

Usage:
    python push_to_neo4j.py                # import graph
    python push_to_neo4j.py --clear        # wipe database first, then import
    python push_to_neo4j.py --verify-only  # just run verification queries
"""

import os
import pickle
import sys
import argparse
from pathlib import Path

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j Python driver not installed.")
    print("Install it with: pip install neo4j")
    sys.exit(1)

BASE_DIR   = Path(__file__).parent
PICKLE_IN  = BASE_DIR / "provenance_graph.gpickle"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "500"))


# ─────────────────────────────────────────────
# GRAPH LOADING
# ─────────────────────────────────────────────

def load_graph():
    if not PICKLE_IN.exists():
        print(f"ERROR: {PICKLE_IN} not found. Run build_graph.py first.")
        sys.exit(1)
    with open(PICKLE_IN, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def safe_val(val):
    """Convert None to empty string for Neo4j properties."""
    if val is None:
        return ""
    return val


def batches(items, size):
    """Yield successive chunks of given size from items list."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def nodes_by_type(G, node_type):
    """Return list of (node_id, attrs_dict) for all nodes of given type."""
    return [
        (nid, data) for nid, data in G.nodes(data=True)
        if data.get("node_type") == node_type
    ]


def edges_by_type(G, edge_type):
    """Return list of (source, target) for all edges of given type."""
    return [
        (u, v) for u, v, data in G.edges(data=True)
        if data.get("edge_type") == edge_type
    ]


# ─────────────────────────────────────────────
# SCHEMA SETUP
# ─────────────────────────────────────────────

def create_constraints(session):
    """Create uniqueness constraints for each node label."""
    constraints = [
        "CREATE CONSTRAINT thread_id IF NOT EXISTS FOR (t:Thread) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT frame_id IF NOT EXISTS FOR (f:Frame) REQUIRE f.id IS UNIQUE",
    ]
    for cypher in constraints:
        session.run(cypher)
    print("Created uniqueness constraints")


# ─────────────────────────────────────────────
# NODE CREATORS
# ─────────────────────────────────────────────

def create_thread_nodes(session, G):
    nodes = nodes_by_type(G, "Thread")
    query = """
    UNWIND $batch AS row
    CREATE (t:Thread {
        id:        row.id,
        title:     row.thread_title,
        postCount: row.post_count,
        seSite:    row.se_site
    })
    """
    total = 0
    for batch in batches(nodes, BATCH_SIZE):
        params = [{
            "id":           str(nid),
            "thread_title": safe_val(data.get("thread_title")),
            "post_count":   data.get("post_count", 0),
            "se_site":      safe_val(data.get("se_site")),
        } for nid, data in batch]
        session.run(query, batch=params)
        total += len(batch)
    print(f"  Thread nodes: {total}")
    return total


def create_post_nodes(session, G):
    nodes = nodes_by_type(G, "Post")
    query = """
    UNWIND $batch AS row
    CREATE (p:Post {
        id:                row.id,
        postType:          row.post_type,
        position:          row.post_position,
        date:              row.post_date,
        score:             row.post_score,
        isAcceptedAnswer:  row.is_accepted_answer,
        seTags:            row.se_tags,
        seSite:            row.se_site,
        hasRelevantContent: row.has_relevant_content,
        skipReason:        row.skip_reason
    })
    """
    total = 0
    for batch in batches(nodes, BATCH_SIZE):
        params = [{
            "id":                  str(nid),
            "post_type":           safe_val(data.get("post_type")),
            "post_position":       data.get("post_position", -1),
            "post_date":           safe_val(data.get("post_date")),
            "post_score":          data.get("post_score", 0),
            "is_accepted_answer":  safe_val(data.get("is_accepted_answer")),
            "se_tags":             safe_val(data.get("se_tags")),
            "se_site":             safe_val(data.get("se_site")),
            "has_relevant_content": safe_val(data.get("has_relevant_content")),
            "skip_reason":         safe_val(data.get("skip_reason")),
        } for nid, data in batch]
        session.run(query, batch=params)
        total += len(batch)
    print(f"  Post nodes:   {total}")
    return total


def create_author_nodes(session, G):
    nodes = nodes_by_type(G, "Author")
    query = """
    UNWIND $batch AS row
    CREATE (a:Author {
        id:              row.id,
        username:        row.author_username,
        reputation:      row.reputation,
        credibilityTier: row.credibility_tier,
        userType:        row.user_type,
        acceptRate:      row.accept_rate,
        profileLink:     row.profile_link
    })
    """
    total = 0
    for batch in batches(nodes, BATCH_SIZE):
        params = [{
            "id":               str(nid),
            "author_username":  safe_val(data.get("author_username")),
            "reputation":       data.get("reputation", -1),
            "credibility_tier": safe_val(data.get("credibility_tier")),
            "user_type":        safe_val(data.get("user_type")),
            "accept_rate":      data.get("accept_rate", -1),
            "profile_link":     safe_val(data.get("profile_link")),
        } for nid, data in batch]
        session.run(query, batch=params)
        total += len(batch)
    print(f"  Author nodes: {total}")
    return total


def create_frame_nodes(session, G):
    nodes = nodes_by_type(G, "Frame")
    query = """
    UNWIND $batch AS row
    CREATE (f:Frame {
        id:              row.id,
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
    })
    """
    total = 0
    for batch in batches(nodes, BATCH_SIZE):
        params = [{
            "id":               str(nid),
            "frame_type":       safe_val(data.get("frame_type")),
            "scope":            safe_val(data.get("scope")),
            "subject":          safe_val(data.get("subject")),
            "applicability":    safe_val(data.get("applicability")),
            "main_point":       safe_val(data.get("main_point")),
            "source_quote":     safe_val(data.get("source_quote")),
            "epistemic_stance": safe_val(data.get("epistemic_stance")),
            "post_role":        safe_val(data.get("post_role")),
            "example_value":    safe_val(data.get("example_value")),
            "example_context":  safe_val(data.get("example_context")),
            "thread_context":   safe_val(data.get("thread_context")),
            "validated":        safe_val(data.get("validated")),
        } for nid, data in batch]
        session.run(query, batch=params)
        total += len(batch)
    print(f"  Frame nodes:  {total}")
    return total


# ─────────────────────────────────────────────
# EDGE CREATORS
# ─────────────────────────────────────────────

EDGE_CONFIG = [
    # (edge_type, relationship_name, source_label, target_label)
    ("THREAD_CONTAINS_POST",  "CONTAINS",        "Thread", "Post"),
    ("POST_AUTHORED_BY",      "AUTHORED_BY",      "Post",   "Author"),
    ("POST_CONTAINS_FRAME",   "CONTAINS",         "Post",   "Frame"),
    ("SE_ANSWERS",            "ANSWERS",           "Post",   "Post"),
    ("FRAME_RELATED_TO",      "RELATED_TO",        "Frame",  "Frame"),
    ("FRAME_CONTRADICTED_BY", "CONTRADICTED_BY",   "Frame",  "Frame"),
]


def create_edges(session, G):
    """Create all edge types. Returns dict of edge_type -> count."""
    counts = {}

    for edge_type, rel_name, src_label, tgt_label in EDGE_CONFIG:
        edges = edges_by_type(G, edge_type)
        if not edges:
            counts[edge_type] = 0
            continue

        query = f"""
        UNWIND $batch AS row
        MATCH (s:{src_label} {{id: row.source}})
        MATCH (t:{tgt_label} {{id: row.target}})
        CREATE (s)-[:{rel_name}]->(t)
        """

        total = 0
        for batch in batches(edges, BATCH_SIZE):
            params = [{"source": str(u), "target": str(v)} for u, v in batch]
            session.run(query, batch=params)
            total += len(batch)

        counts[edge_type] = total
        print(f"  {rel_name} ({src_label}->{tgt_label}): {total}")

    return counts


# ─────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────

def verify(session):
    """Run verification queries and print results."""
    print(f"\n{'='*50}")
    print("VERIFICATION")
    print(f"{'='*50}")

    # Node counts by label
    result = session.run(
        "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label"
    )
    print("\nNodes by label:")
    total_nodes = 0
    for record in result:
        print(f"  {record['label']:<10}: {record['cnt']}")
        total_nodes += record['cnt']
    print(f"  {'TOTAL':<10}: {total_nodes}")

    # Edge counts by type
    result = session.run(
        "MATCH ()-[r]->() RETURN type(r) AS relType, count(r) AS cnt ORDER BY relType"
    )
    print("\nRelationships by type:")
    total_edges = 0
    for record in result:
        print(f"  {record['relType']:<20}: {record['cnt']}")
        total_edges += record['cnt']
    print(f"  {'TOTAL':<20}: {total_edges}")

    # Sample query: frames from high-credibility authors
    result = session.run("""
        MATCH (f:Frame)<-[:CONTAINS]-(p:Post)-[:AUTHORED_BY]->(a:Author)
        WHERE a.reputation > 10000
        RETURN f.subject AS subject, a.username AS author, a.reputation AS rep
        ORDER BY a.reputation DESC
        LIMIT 5
    """)
    records = list(result)
    if records:
        print("\nSample: frames from high-rep authors:")
        for r in records:
            print(f"  {r['author']} (rep {r['rep']}): {r['subject']}")

    print(f"{'='*50}")


# ─────────────────────────────────────────────
# CLEAR DATABASE
# ─────────────────────────────────────────────

def clear_database(session):
    """Delete all nodes and relationships."""
    # Delete in batches to avoid memory issues on large graphs
    print("Clearing database...")
    result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
    record = result.single()
    deleted = record["deleted"] if record else 0
    print(f"  Deleted {deleted} nodes (and all their relationships)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Push DFM provenance graph to Neo4j."
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Wipe the database before importing (asks for confirmation).",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only run verification queries, don't import.",
    )
    args = parser.parse_args()

    print(f"Neo4j connection:")
    print(f"  URI:      {NEO4J_URI}")
    print(f"  User:     {NEO4J_USER}")
    print(f"  Database: {NEO4J_DATABASE}")
    print()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # Test connection
        driver.verify_connectivity()
        print("Connected to Neo4j successfully.\n")
    except Exception as e:
        print(f"ERROR: Could not connect to Neo4j: {e}")
        print(f"\nMake sure Neo4j is running and the connection details are correct.")
        print(f"Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars if needed.")
        sys.exit(1)

    with driver.session(database=NEO4J_DATABASE) as session:
        if args.verify_only:
            verify(session)
            driver.close()
            return

        if args.clear:
            confirm = input("This will DELETE ALL data in the database. Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                driver.close()
                return
            clear_database(session)
            print()

        G = load_graph()

        print("\nCreating constraints...")
        create_constraints(session)

        print("\nCreating nodes:")
        create_thread_nodes(session, G)
        create_post_nodes(session, G)
        create_author_nodes(session, G)
        create_frame_nodes(session, G)

        print("\nCreating edges:")
        edge_counts = create_edges(session, G)

        verify(session)

    driver.close()
    print("\nDone. Neo4j connection closed.")


if __name__ == "__main__":
    main()
