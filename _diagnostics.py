"""One-off diagnostics script for pipeline stats."""
import json, pickle, os
from collections import Counter

print("=" * 65)
print("DFM KNOWLEDGE GRAPH PIPELINE DIAGNOSTICS")
print("=" * 65)

# 1. Posts scraped
print("\n--- SCRAPING ---")
for label, path in [
    ("v1 (original)", "src/scraping_logic/stackexchange_posts_15000.json"),
    ("v3 (expanded)", "src/scraping_logic/stackexchange_posts_15000_v3.json"),
]:
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        sites = Counter(r.get("site") for r in data)
        print(f"  {label}: {len(data)} questions")
        for s, c in sites.most_common():
            print(f"    {s}: {c}")

# 2. Posts filtered (questions + answers)
print("\n--- FILTERING ---")
total_filtered_questions = 0
total_filtered_posts = 0
for label, path in [
    ("v1 filtered", "src/scraping_logic/stackexchange_posts_filtered.json"),
    ("v3 new filtered", "src/scraping_logic/stackexchange_posts_v3_filtered.json"),
]:
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        n_questions = len(data)
        n_answers = sum(len(q.get("answers", [])) for q in data)
        n_posts = n_questions + n_answers
        sites = Counter(r.get("site") for r in data)
        print(f"  {label}: {n_questions} questions + {n_answers} answers = {n_posts} posts")
        for s, c in sites.most_common():
            print(f"    {s}: {c} questions")
        total_filtered_questions += n_questions
        total_filtered_posts += n_posts

print(f"  TOTAL: {total_filtered_questions} questions, {total_filtered_posts} posts")

# 3. Extraction inputs (normalized threads)
print("\n--- NORMALIZATION (extraction inputs) ---")
for label, path in [
    ("v1 normalized", "se_posts_normalized.json"),
    ("v3 new normalized", "se_posts_normalized_v3_new.json"),
]:
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        total_posts = sum(len(t.get("posts", [])) for t in data)
        print(f"  {label}: {len(data)} threads, {total_posts} individual posts")

# 4. Extractions
print("\n--- EXTRACTION ---")
for label, path in [
    ("v1 extractions", "extractions_se_5000.json"),
    ("v3 new extractions", "extractions_se_v3_new.json"),
    ("ALL merged", "extractions_se_all.json"),
]:
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        results = data.get("results", [])
        relevant = [r for r in results if r.get("has_relevant_content")]
        skipped = [r for r in results if not r.get("has_relevant_content")]
        frames = sum(len(r.get("frames", [])) for r in relevant)
        threads = len(set(r.get("thread_url", "") for r in results if r.get("thread_url")))
        sites = Counter(r.get("se_site", "?") for r in results)
        print(f"  {label}:")
        print(f"    Posts sent to LLM: {len(results)} (from {threads} threads)")
        print(f"    Relevant: {len(relevant)}, Skipped: {len(skipped)}")
        print(f"    Frames extracted: {frames}")
        for s, c in sites.most_common():
            print(f"      {s}: {c} posts")
        ftypes = Counter()
        for r in relevant:
            for f in r.get("frames", []):
                ftypes[f.get("frame_type", "?")] += 1
        for ft, c in ftypes.most_common():
            print(f"      {ft}: {c}")

# 5. Embeddings
print("\n--- EMBEDDINGS (Qdrant) ---")
try:
    from qdrant_client import QdrantClient
    qdrant = QdrantClient(path="./qdrant_data")
    for name in ["frames_knowledge", "frames_language", "frames_metadata"]:
        info = qdrant.get_collection(name)
        print(f"  {name}: {info.points_count} points")
    qdrant.close()
except Exception as e:
    print(f"  Error: {e}")

# 6. Provenance graph
print("\n--- PROVENANCE GRAPH ---")
import networkx as nx
gpath = "provenance_graph.gpickle"
if os.path.exists(gpath):
    with open(gpath, "rb") as f:
        G = pickle.load(f)
    type_counts = Counter()
    for _, d in G.nodes(data=True):
        type_counts[d.get("node_type", "?")] += 1
    print(f"  Total nodes: {G.number_of_nodes()}")
    for nt, c in type_counts.most_common():
        print(f"    {nt}: {c}")

    edge_counts = Counter()
    for u, v, d in G.edges(data=True):
        edge_counts[d.get("edge_type", "?")] += 1
    print(f"  Total edges: {G.number_of_edges()}")
    for et, c in edge_counts.most_common():
        print(f"    {et}: {c}")

    semantic_types = {"CONTRADICTS", "MITIGATES", "SPECIFIES", "CAUSAL_LINK"}
    structural = sum(c for et, c in edge_counts.items() if et not in semantic_types)
    semantic = sum(c for et, c in edge_counts.items() if et in semantic_types)
    print(f"  Structural edges: {structural}")
    print(f"  Semantic edges:   {semantic}")

# 7. Semantic edges file
print("\n--- SEMANTIC EDGES (semantic_edges.json) ---")
epath = "semantic_edges.json"
if os.path.exists(epath):
    se = json.load(open(epath, encoding="utf-8"))
    total_classified = se.get("total_classified", "?")
    total_edges = se.get("total_edges", "?")
    print(f"  Total pairs classified: {total_classified}")
    print(f"  Non-UNRELATED edges: {total_edges}")
    elabels = Counter()
    for e in se.get("edges", []):
        elabels[e.get("label", "?")] += 1
    for lab, c in elabels.most_common():
        print(f"    {lab}: {c}")
else:
    print("  File not found")

# 8. Subject normalization
print("\n--- SUBJECT NORMALIZATION ---")
sc_path = "subject_clusters.json"
if os.path.exists(sc_path):
    sc = json.load(open(sc_path, encoding="utf-8"))
    meta = sc.get("metadata", {})
    print(f"  Total unique subjects : {meta.get('total_subjects', '?')}")
    print(f"  Total clusters        : {meta.get('total_clusters', '?')}")
    print(f"  Multi-member clusters : {meta.get('multi_member_clusters', '?')}")
    print(f"  Singleton clusters    : {meta.get('singleton_clusters', '?')}")
    print(f"  Distance threshold    : {meta.get('threshold', '?')}")
    print(f"  Embedding model       : {meta.get('embedding_model', '?')}")
else:
    print("  subject_clusters.json not found")

bl_path = "subject_blocklist.json"
if os.path.exists(bl_path):
    bl = json.load(open(bl_path, encoding="utf-8"))
    print(f"  Blocked pairs         : {len(bl.get('blocked_pairs', []))}")

# 9. Topic model
print("\n--- TOPIC MODEL (BERTopic) ---")
tm_path = "topic_model_output.json"
if os.path.exists(tm_path):
    tm = json.load(open(tm_path, encoding="utf-8"))
    meta = tm.get("metadata", {})
    print(f"  Total frames          : {meta.get('total_frames', '?')}")
    print(f"  Total topics          : {meta.get('total_topics', '?')}")
    print(f"  Outlier frames        : {meta.get('outlier_count', '?')} ({meta.get('outlier_pct', '?')}%)")
    print(f"  Min cluster size      : {meta.get('min_cluster_size', '?')}")
    print(f"  LLM label model       : {meta.get('llm_model', '?')}")
    print(f"  Embedding source      : {meta.get('embedding_source', '?')}")
    topics = [t for t in tm.get("topics", []) if t.get("topic_id") != -1]
    if topics:
        sizes = [t["n_frames"] for t in topics]
        print(f"  Topic sizes           : min={min(sizes)}, max={max(sizes)}, "
              f"median={sorted(sizes)[len(sizes)//2]}")
        print(f"  Top 5 topics:")
        for t in topics[:5]:
            print(f"    [{t['n_frames']:3d} frames] {t['label']}")
else:
    print("  topic_model_output.json not found")

# 10. Note about v1 extraction source
print("\n--- NOTES ---")
if os.path.exists("extractions_se_5000.json"):
    data = json.load(open("extractions_se_5000.json", encoding="utf-8"))
    v1_threads = len(set(r.get("thread_url", "") for r in data.get("results", []) if r.get("thread_url")))
    print(f"  v1 extraction covered {v1_threads} threads (728 threads / 1447 posts)")
    print(f"  v1 GPT filter (456 questions) was applied separately")
    print(f"  v1 extraction includes posts NOT in the GPT-filtered set")

# 11. Revised extraction comparison
print("\n--- REVISION COMPARISON ---")
# Look for revision changelog files
import glob as glob_mod
changelogs = sorted(glob_mod.glob("revision_changelog*.json"))
if changelogs:
    for cl_path in changelogs:
        cl = json.load(open(cl_path, encoding="utf-8"))
        meta = cl.get("metadata", {})
        revisions = cl.get("revisions", [])

        print(f"  {cl_path}:")
        print(f"    Source: {meta.get('source_file', '?')}")
        print(f"    Reviews: {meta.get('review_file', '?')}")
        print(f"    Frames reviewed: {meta.get('total_frames_reviewed', '?')}")
        print(f"    Flagged Revise: {meta.get('frames_flagged_revise', '?')}")
        print(f"    Passed threshold: {meta.get('frames_passing_threshold', '?')}")
        print(f"    Actually revised: {meta.get('frames_actually_revised', '?')}")
        print(f"    Kept by reviser: {meta.get('frames_kept_by_reviser', '?')}")
        print(f"    Errors: {meta.get('errors', '?')}")

        # Field change counts
        field_counts = Counter()
        for rev in revisions:
            if rev.get("action") == "revise":
                for ch in rev.get("actual_changes", []):
                    field_counts[ch["field_name"]] += 1
        if field_counts:
            print(f"    Fields changed: "
                  + ", ".join(f"{f}={c}" for f, c in field_counts.most_common()))

    # Compare original vs revised extraction distributions
    orig_path = "extractions_se_all.json"
    revised_paths = sorted(glob_mod.glob("extractions_revised*.json"))
    if os.path.exists(orig_path) and revised_paths:
        orig_data = json.load(open(orig_path, encoding="utf-8"))
        orig_results = orig_data.get("results", [])

        for rev_path in revised_paths:
            rev_data = json.load(open(rev_path, encoding="utf-8"))
            rev_results = rev_data.get("results", [])

            print(f"\n  Distribution comparison: {orig_path} vs {rev_path}")
            for field in ["frame_type", "scope", "applicability", "epistemic_stance"]:
                orig_dist = Counter()
                rev_dist = Counter()
                for r in orig_results:
                    for f in r.get("frames", []):
                        orig_dist[f.get(field, "?")] += 1
                for r in rev_results:
                    for f in r.get("frames", []):
                        rev_dist[f.get(field, "?")] += 1

                # Only show values that changed
                all_vals = sorted(set(list(orig_dist.keys()) + list(rev_dist.keys())))
                diffs = [(v, orig_dist.get(v, 0), rev_dist.get(v, 0))
                         for v in all_vals
                         if orig_dist.get(v, 0) != rev_dist.get(v, 0)]

                if diffs:
                    print(f"    {field}:")
                    for val, o, r in diffs:
                        d = r - o
                        sign = "+" if d > 0 else ""
                        print(f"      {val:<28} {o:>5} -> {r:>5} ({sign}{d})")
else:
    print("  No revision changelogs found")

print("\n" + "=" * 65)
