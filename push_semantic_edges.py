"""
Push semantic edges from semantic_edges.json to a running Neo4j instance.

Reads LLM-classified frame-to-frame relationships and creates
CAUSAL_LINK, SPECIFIES, MITIGATES, and CONTRADICTS edges.
Assumes the provenance graph (nodes + structural edges) is already loaded.

Environment variables:
    NEO4J_URI      - default: bolt://localhost:7687
    NEO4J_USER     - default: neo4j
    NEO4J_PASSWORD - default: password
    NEO4J_DATABASE - default: neo4j
    BATCH_SIZE     - default: 500

Usage:
    python push_semantic_edges.py                  # import semantic edges
    python push_semantic_edges.py --clear-semantic  # delete semantic edges first, then reimport
    python push_semantic_edges.py --verify-only     # just check counts
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j Python driver not installed.")
    print("Install it with: pip install neo4j")
    sys.exit(1)

BASE_DIR   = Path(__file__).parent
EDGES_FILE = BASE_DIR / "semantic_edges.json"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "500"))

SEMANTIC_LABELS = ["CAUSAL_LINK", "SPECIFIES", "MITIGATES", "CONTRADICTS"]


def batches(items, size):
    """Yield successive chunks of given size from items list."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def load_and_expand():
    """
    Load semantic_edges.json and expand into directed (source, target, label,
    confidence, reasoning) tuples.  Mirrors infer_edges.py add_semantic_edges().
    """
    if not EDGES_FILE.exists():
        print(f"ERROR: {EDGES_FILE} not found. Run infer_edges.py first.")
        sys.exit(1)

    with open(EDGES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("edges", [])
    print(f"Loaded {len(raw)} edges from {EDGES_FILE.name}")
    print(f"  Model: {data.get('model')}, top_k: {data.get('top_k')}")

    directed = []
    for e in raw:
        label      = e["label"]
        direction  = e["direction"]
        confidence = e["confidence"]
        reasoning  = e["reasoning"]
        fa, fb     = e["frame_a"], e["frame_b"]

        if label == "CONTRADICTS" or direction == "bidirectional":
            directed.append((fa, fb, label, confidence, reasoning))
            directed.append((fb, fa, label, confidence, reasoning))
        elif direction == "B_to_A":
            directed.append((fb, fa, label, confidence, reasoning))
        else:  # A_to_B or "none"
            directed.append((fa, fb, label, confidence, reasoning))

    print(f"Expanded to {len(directed)} directed relationships")
    return directed


def create_semantic_edges(session, directed):
    """Group by label and push each group with batched UNWIND."""
    groups = defaultdict(list)
    for source, target, label, confidence, reasoning in directed:
        groups[label].append({
            "source":     source,
            "target":     target,
            "confidence": confidence,
            "reasoning":  reasoning,
        })

    counts = {}
    for label, edges in groups.items():
        query = f"""
        UNWIND $batch AS row
        MATCH (s:Frame {{id: row.source}})
        MATCH (t:Frame {{id: row.target}})
        CREATE (s)-[:{label} {{confidence: row.confidence, reasoning: row.reasoning}}]->(t)
        """
        total = 0
        for batch in batches(edges, BATCH_SIZE):
            result = session.run(query, batch=batch)
            summary = result.consume()
            total += summary.counters.relationships_created

        skipped = len(edges) - total
        counts[label] = total
        msg = f"  {label}: {total} created"
        if skipped:
            msg += f" ({skipped} skipped -- frame not found)"
        print(msg)

    return counts


def clear_semantic(session):
    """Delete only the 4 semantic relationship types."""
    for label in SEMANTIC_LABELS:
        result = session.run(
            f"MATCH ()-[r:{label}]->() DELETE r RETURN count(r) AS deleted"
        )
        record = result.single()
        deleted = record["deleted"] if record else 0
        print(f"  Deleted {deleted} {label} relationships")


def verify(session):
    """Print semantic edge counts."""
    print(f"\n{'='*50}")
    print("SEMANTIC EDGE VERIFICATION")
    print(f"{'='*50}")

    result = session.run("""
        MATCH ()-[r]->()
        WHERE type(r) IN $labels
        RETURN type(r) AS relType, count(r) AS cnt
        ORDER BY cnt DESC
    """, labels=SEMANTIC_LABELS)

    total = 0
    for record in result:
        print(f"  {record['relType']:<15}: {record['cnt']}")
        total += record['cnt']
    print(f"  {'TOTAL':<15}: {total}")

    # Symmetry check for CONTRADICTS
    result = session.run("""
        MATCH (a:Frame)-[:CONTRADICTS]->(b:Frame)
        WHERE NOT (b)-[:CONTRADICTS]->(a)
        RETURN count(*) AS one_way
    """)
    record = result.single()
    one_way = record["one_way"] if record else 0
    if one_way == 0:
        print("\n  CONTRADICTS symmetry: OK (all bidirectional)")
    else:
        print(f"\n  WARNING: {one_way} one-way CONTRADICTS edges found")

    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Push semantic edges from semantic_edges.json to Neo4j."
    )
    parser.add_argument(
        "--clear-semantic", action="store_true",
        help="Delete existing semantic edges before importing.",
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

        if args.clear_semantic:
            print("Clearing existing semantic edges...")
            clear_semantic(session)
            print()

        directed = load_and_expand()

        print("\nCreating semantic edges:")
        counts = create_semantic_edges(session, directed)

        total = sum(counts.values())
        print(f"\nTotal relationships created: {total}")

        verify(session)

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
