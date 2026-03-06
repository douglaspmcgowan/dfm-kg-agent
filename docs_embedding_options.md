# Embedding Space Options for DFM Knowledge Frames

Options considered for building a searchable vector space over extracted KnowledgeFrames, enabling semantic retrieval for DFM feedback systems.

## Background

The DFM extraction pipeline produces structured KnowledgeFrames from CNC machining forum posts. Each frame contains:
- **source_quote**: verbatim machinist language from the forum post
- **main_point**: distilled, transferable knowledge claim (2-3 sentences)
- **subject**: concise noun phrase concept anchor (e.g., "local variable volatility")
- **example_context**: real-world scoping for non-universal claims
- Structured metadata: frame_type, scope, applicability, epistemic_stance, post_role, author_credibility_tier, etc.

The goal: build a system where an LLM (or human) can ask a natural language question and retrieve the most relevant frames.

---

## Concepts

### What is an Embedding?
An embedding turns text into a list of numbers (a "vector") that captures its meaning. Texts with similar meanings produce similar vectors. Similarity is measured by **cosine similarity** (1.0 = identical meaning, 0.0 = unrelated).

### What Does the Embedding Model See?
Only the text you pass to it. Structured metadata fields (frame_type, scope, etc.) stored alongside the vector have **zero influence on the vector itself** unless you include them in the text. The vector search and metadata filtering are independent operations.

---

## Option A: OpenAI Embeddings + Local FAISS Index

**FAISS** = "Facebook AI Similarity Search" -- a library that stores vectors and finds the most similar ones to a query.

### How It Works
1. Concatenate key text fields per frame into one string
2. Send all texts to OpenAI's `text-embedding-3-small` API -> get back 1,536-dimensional vectors
3. Store vectors in a FAISS index file (`.index`)
4. At query time: embed the question, ask FAISS for the N closest vectors

### Pros
- Zero new accounts/services -- just `pip install faiss-cpu` + existing OpenAI key
- Cheap (pennies for 209 frames), fast (<1ms queries), transparent (~60-80 lines of code)
- Portable -- just a `.index` file and a `.json` mapping

### Cons
- Requires OpenAI API for every operation (no offline use)
- **No metadata filtering** -- can't say "only show me risk frames about tooling." Must post-filter manually, which gets painful at scale
- Must maintain separate mapping file from index positions -> frame data
- If OpenAI retires the model, vectors become orphaned

### Best For
Quick prototype where metadata filtering isn't needed.

---

## Option B: OpenAI Embeddings + ChromaDB (Local)

**ChromaDB** = a local vector database that stores vectors AND structured metadata together, enabling combined queries.

### How It Works
Same embedding process as A, but stored in ChromaDB "collections" with metadata. Queries combine vector similarity + metadata filters in one call.

### Key Advantage Over Option A
```python
# "Find heuristics about programming similar to this question"
results = collection.query(
    query_texts=["what happens to variables?"],
    where={"$and": [{"frame_type": "heuristic"}, {"scope": "programming"}]}
)
```

### Pros
- Everything from A, plus built-in metadata filtering on all schema fields
- Documents stored alongside vectors (no separate mapping file)
- Incremental updates via `.upsert()`

### Cons
- Heavy dependency, occasionally has install issues on newer Python versions
- **Incompatible with Python 3.14** (Pydantic v1 dependency + hnswlib build failures)
- Still requires OpenAI API for embeddings

### Best For
Projects on Python 3.13 or earlier with rich structured metadata.

### Status: NOT VIABLE for this project (Python 3.14 incompatibility)

---

## Option B-alt: OpenAI Embeddings + Qdrant (Local) -- SELECTED

**Qdrant** = a vector search engine with a Python client that supports local file-based storage. Same capabilities as ChromaDB but with Python 3.14 support.

### How It Works
Same as Option B conceptually. Qdrant stores "points" with vectors + "payloads" (metadata). Supports rich filtering with `FieldCondition`, `MatchValue`, `MatchAny`, `Range`, etc.

### Key Advantage
- Explicitly supports Python 3.14
- Same metadata filtering capabilities as ChromaDB
- Local mode: `QdrantClient(path="./qdrant_data")` -- persistent file storage, no server

### Pros
- Python 3.14 + Windows compatible
- Rich metadata filtering (equal, in-list, range, boolean)
- Payload (metadata) stored with vectors -- single source of truth
- Clean API with Pydantic models for filters
- Lightweight install (`pip install qdrant-client`)

### Cons
- Still requires OpenAI API for embeddings
- Slightly more verbose filter syntax than ChromaDB
- Smaller community than ChromaDB (though Qdrant itself is well-maintained)

### Best For
This project. Replaces ChromaDB with identical capabilities and Python 3.14 support.

---

## Option C: Local Sentence-Transformers + FAISS

Use an open-source embedding model (e.g., `all-MiniLM-L6-v2`) running locally instead of OpenAI's API.

### Pros
- Fully offline, free, reproducible
- Fast inference on CPU for small datasets

### Cons
- Heavy install (~2GB for torch + sentence-transformers)
- Lower quality on domain-specific CNC machining text
- No metadata filtering (same as Option A)
- May need fine-tuning for machinist jargon

### Best For
Offline-first workflows, reproducibility for a research paper.

---

## Option D: Multi-Vector Approach -- SELECTED (combined with B-alt)

Instead of one embedding per frame, create **multiple vectors** per frame, each capturing a different aspect of the knowledge.

### The Three Vectors

| Vector | Source Text | What It Captures |
|--------|-----------|-----------------|
| **Knowledge** | subject + main_point + example_context | Distilled, transferable insight in clean language |
| **Language** | source_quote | Raw machinist terminology and shop talk |
| **Metadata** | frame_type, scope, applicability, etc. as natural language | What kind of knowledge this is (categorical profile) |

### Why Multiple Vectors?
The same knowledge can be queried in different language registers:
- **Engineer**: "What are the risks of using local variables for persistent data storage?"
- **Machinist**: "why do my #100s keep going blank after M30"
- **Categorical**: "show me heuristics about programming"

A single vector can't match all three well. Multiple vectors give each frame multiple "entry points."

### Score Combination
Results from each vector set are merged by frame ID:
- **Max score** (two-vector default): `combined = max(knowledge_score, language_score)`
- **Weighted average** (three-vector): `combined = 0.5*K + 0.3*L + 0.2*M` (tunable)

### Metadata Vector Details
Embeds structured fields as natural language: `"frame type: risk. scope: design geometry. applicability: context dependent. post role: solution. epistemic stance: direct neutral"`. Includes: frame_type, post_role, scope, subject, applicability, epistemic_stance, extraction_rationale, validation status, validation_notes, related frame count, contradiction status.

### Pros
- Catches both formal and informal queries
- Diagnostic value: see which vector "won" for each result
- Minimal extra cost (2-3x embeddings at pennies total)

### Cons
- Merge logic adds complexity (weight tuning needed)
- Harder to evaluate without retrieval ground truth
- Short source_quotes (~30 chars) produce vague language vectors

---

## Option E: Graph + Vector Hybrid (Neo4j Vector Index)

Store frames as nodes in a Neo4j knowledge graph with embedding vectors as node properties. Combine vector similarity with Cypher graph traversal.

### Key Advantage: Multi-Hop Reasoning
```cypher
MATCH (a:Author {credibility_tier: "high"})
      -[:AUTHORED]->(p:Post)
      -[:EXTRACTED_FROM]->(f:Frame {frame_type: "heuristic"})
      -[:HAS_SUBJECT]->(s:Subject)
      -[:RELATED_TO]->(s2:Subject {name: "local variable volatility"})
RETURN f, a.username, s.name
```

### Pros
- Relationships as first-class data (contradictions, related frames, author networks)
- Subject-level concept navigation
- Multi-hop context for LLM reasoning (risk -> workaround -> counterargument)
- Aligns with project roadmap Step 10 (KG-Augmented RAG)
- Existing `build_neo4j.py` infrastructure

### Cons
- Most complex to build (200-400 lines + Cypher learning curve)
- Graph quality depends on relationship curation (mostly unpopulated currently)
- Neo4j Aura is cloud-hosted (latency, no offline)
- Harder to iterate on during experimental phase
- Two external dependencies (OpenAI + Neo4j)

### Best For
The eventual production DFM feedback system, once relationship data is curated. Premature for current phase.

---

## Decision Matrix

| Dimension | A (FAISS) | B (Chroma) | B-alt (Qdrant) | C (Local) | D (Multi-Vec) | E (Neo4j) |
|-----------|-----------|------------|----------------|-----------|----------------|-----------|
| Py 3.14 compat | Yes | **No** | Yes | Uncertain | Inherits | Yes |
| Metadata filtering | None | Built-in | Built-in | None | Inherits | Built-in + graph |
| Offline capable | No | No | No | Yes | Depends | No |
| Multi-hop reasoning | No | No | No | No | No | Yes |
| Complexity | Low | Medium | Medium | Medium | +Layer | High |
| Best at 209 frames | Adequate | N/A | Good | Adequate | Good | Overkill |

## Selected: Option B-alt + D (Qdrant with Multi-Vector)

Three Qdrant collections with two-vector default mode and optional three-vector mode. See `docs_embedding_tutorial.md` for implementation details.
