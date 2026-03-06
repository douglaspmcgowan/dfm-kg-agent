# DFM Knowledge Graph: Architecture & Current State
*Updated March 2, 2026*

---

## 1. Research Question

**How can undocumented and tribal machining knowledge be collected and structured to improve the available knowledge base for AI-powered DFM interventions?**

The core claim: an extraction pipeline + structured knowledge graph + multi-vector retrieval can surface practical manufacturing constraints that are absent from standard DFM handbooks, and an agentic RAG layer can synthesize those constraints into actionable design feedback.

---

## 2. System Architecture

### Data Sources

| Source | Scraped | Filtered | Extracted | Frames | Status |
|---|---|---|---|---|---|
| PracticalMachinist.com | ~1,197 posts | — | ~209 (partial) | ~209 | Archived in `archive/`; extraction partially run |
| Stack Exchange (5 sites) | 12,805 questions | 639 questions (1,399 posts) | 1,936 posts (911 threads) | 3,879 | Full pipeline complete |

**SE sites:** engineering (1,792), electronics (7,396), robotics (1,628), mechanics (1,496), woodworking (493). The v3 expanded scrape used per-keyword API calls; GPT-5-mini relevance filtering kept 639 questions (6.4% acceptance; engineering 54%, electronics 1.4%). `filter_with_gpt.py` supports `--skip-ids-from` for incremental filtering.

Both sources share an identical schema (`FrameType`, `ScopeDomain`, `EpistemicStance`, etc.) defined in `schema_se.py`. The PM pipeline scripts are archived in `archive/`.

---

## 3. The Knowledge Frame

The atomic unit. Each frame captures **one transferable engineering claim** from a single post, with traceability back to the source text.

**Eight frame types:** `risk`, `heuristic`, `principle`, `observation`, `comparison`, `workaround`, `case`, `other`

**Key fields:**
- `source_quote` — verbatim substring from the post (anti-hallucination anchor)
- `main_point` — 2-3 sentence distillation of the claim
- `subject` — noun phrase (graph node anchor)
- `scope` — `design_geometry`, `machining_process`, `material`, `tooling`, `programming`, `shop_operations`, `quality`, `cost`, `machine_capability`
- `applicability` — `universal` / `context_dependent` / `situational`
- `epistemic_stance` — `absolute_authoritative` / `direct_neutral` / `hedged_emphatic` / `tentative_subjective` / `narrative_implied`
- `example_context` — required when `applicability != universal`

---

## 4. Multi-Vector Embedding (BUILT)

Three Qdrant collections — each frame has three vectors over different text surfaces:

| Collection | Text Embedded | Purpose |
|---|---|---|
| `frames_knowledge` | `{subject}: {main_point} {example_context}` | Semantic content matching |
| `frames_language` | `source_quote` (raw shop talk) | Style/register matching |
| `frames_metadata` | Structured fields as NL text | Categorical profile matching |

**Model:** `text-embedding-3-large` (3,072 dimensions, cosine similarity)
**Index:** Local Qdrant store at `./qdrant_data/` — 3,879 frames indexed. `build_embeddings.py` supports `--append` for incremental updates without re-embedding existing frames.

**Retrieval (`search_frames.py`):** Multi-vector search with max-score (2-vector) or weighted merge (3-vector). Rich filter syntax: `type:risk scope:tooling cred:high score>5 rep>1000 tag:aluminum`.

**Note on the metadata vector:** The filter syntax in `search_frames.py` already handles categorical queries (`type:`, `scope:`, `applicability:`, etc.) via Qdrant payload filters — no embedding needed. The metadata collection's marginal value is similarity over categorical *combinations*. Worth auditing empirically before the next full index rebuild.

---

## 5. Provenance Graph (BUILT)

A `networkx.MultiDiGraph` with four node types, six structural edge types, and four semantic edge types.

**Script:** `build_graph.py --input extractions_se_all.json` | **Output:** `provenance_graph.gpickle`, `provenance_graph.graphml`

### Nodes

| Type | Count | ID Format |
|---|---|---|
| Thread | 911 | `thread_url` |
| Post | 1,936 | `post_id` (e.g. `se-engineering-51725-a51726`) |
| Author | 1,138 | `author:{username}` |
| Frame | 3,879 | `{post_id}-{i}` (matches Qdrant `frame_id`) |
| **Total** | **7,864** | |

### Structural Edges

| Edge Type | Count | Meaning |
|---|---|---|
| `THREAD_CONTAINS_POST` | 1,936 | Thread -> Post |
| `POST_AUTHORED_BY` | 1,936 | Post -> Author |
| `POST_CONTAINS_FRAME` | 3,879 | Post -> Frame |
| `SE_ANSWERS` | 1,025 | Answer post -> Question post |
| **Total structural** | **8,776** | |

### Semantic Edges (inferred by `infer_edges.py`)

| Edge Type | Count | Meaning |
|---|---|---|
| `SPECIFIES` | 2,111 | One frame adds a condition scoping the other |
| `MITIGATES` | 1,206 | One frame solves/prevents a risk in the other |
| `CAUSAL_LINK` | 625 | One frame explains the physical cause of the other |
| `CONTRADICTS` | 330 | Opposing guidance for the same situation (bidirectional) |
| **Total semantic** | **4,272** | |

**Total edges:** 13,048 (8,776 structural + 4,272 semantic)

**Topology:** 325 connected components on the undirected view. Largest component: 5,722 nodes. Median component size: 4 nodes.

**Author data:** `se_author_frames.json` — 562 unique SE authors with reputation, user_type, credibility tier (high/medium/low). Generated by `make_se_author_frames.py` (no LLM; pure data extraction).

**Frame ID convention:** `{post_id}-{i}` is shared between the graph and the Qdrant payload `frame_id` field — this is the cross-system join key.

---

## 6. Semantic Edge Inference (COMPLETE)

**Script:** `infer_edges.py --resume` | **Output:** `semantic_edges.json`, updated graph files

### Approach: Similarity-Triggered Classification

1. For each Frame node, query `frames_knowledge` (Qdrant) for top-5 neighbors using the **stored vector** (no re-embedding cost)
2. Deduplicate pairs (frozenset dedup), skip same-post pairs (structural edges cover those)
3. Call GPT-5 with a structured output schema to classify each unique pair
4. Add directed edges to the graph; save `semantic_edges.json`

### Results

| | |
|---|---|
| Total frames | 3,879 |
| Candidate pairs gathered | 10,229 (after dedup; 5,639 same-post skipped) |
| Pairs classified by GPT-5 | 10,389 |
| Non-UNRELATED edges | 4,107 |
| SPECIFIES | 2,111 |
| MITIGATES | 1,206 |
| CAUSAL_LINK | 625 |
| CONTRADICTS | 165 (330 in graph — bidirectional) |

`--resume` supports incremental runs: checkpoint (`infer_edges_checkpoint.json`) tracks classified pairs, so new frames can be added and only new pairs get classified.

---

## 7. Subject Normalization (COMPLETE)

**Script:** `normalize_subjects.py --threshold 0.30` | **Output:** `subject_clusters.json`, `subject_embeddings.npz`

Clusters semantically similar free-text `subject` fields into canonical groups. The 3,879 frames have 3,870 near-unique subject strings; normalization collapses these into 3,292 clusters (428 multi-member, 2,864 singletons).

**Approach:** Embed all unique subjects with `text-embedding-3-large`, compute pairwise cosine distance, cluster with `AgglomerativeClustering` (average linkage, precomputed metric). Canonical name = shortest string in each cluster.

**Blocklist:** `subject_blocklist.json` contains 10 blocked pairs of subjects that are semantically distinct but have similar embeddings (e.g., "condition for 0% thread engagement" vs "condition for 100% thread engagement"). Blocked pairs have their distance forced to 2.0 before clustering.

| | |
|---|---|
| Unique subjects | 3,870 |
| Clusters (threshold 0.30) | 3,292 |
| Multi-member clusters | 428 |
| Singleton clusters | 2,864 |
| Blocked pairs | 10 |

---

## 8. Topic Modeling (COMPLETE)

**Script:** `topic_model.py --reduce-topics 40` | **Output:** `topic_model_output.json`

Discovers high-level topics across frames using BERTopic with LLM-generated labels.

**Approach:**
1. Pull pre-computed embeddings from Qdrant `frames_knowledge` (3,879 x 3072 vectors)
2. UMAP dimensionality reduction (3072-dim -> 5-dim, cosine metric)
3. HDBSCAN clustering (`sklearn.cluster.HDBSCAN`, min_cluster_size=5)
4. c-TF-IDF keyword extraction from `subject + ". " + main_point` text, with domain stop word filtering
5. LLM topic labeling via `bertopic.representation.OpenAI` with gpt-4.1-mini
6. Topic reduction from 319 initial topics to 39 merged topics

| | |
|---|---|
| Total frames | 3,879 |
| Topics | 39 |
| Outlier frames | 619 (16%) |
| Largest topic | 410 frames (Precision hole-making) |
| Smallest topic | 5 frames (Mach4 PoKeys Connectivity) |
| LLM label model | gpt-4.1-mini |

---

## 9. Extraction Pipeline (MATURE)

**Scripts:** `extract.py` (PracticalMachinist), `extract_se.py` (Stack Exchange)

Key features:
- **Anti-hallucination filtering:** fuzzy substring match (`source_quote` must appear in post text at ≥85% similarity; wider window for long quotes)
- **Consolidation:** two-phase dedup — deterministic rules on quote similarity, then LLM merge review
- **Epistemic stance overrides:** heuristic post-processing corrects hedge/strong keyword combinations
- **Retry logic:** re-runs extraction when frames exist but all fail filtering
- **Resume support:** checkpoint-based incremental runs for large corpora
- **Parallel extraction:** configurable `PARALLEL_REQUESTS` (default 10)

**Model recommendation:** `gpt-4.1-mini` for extraction (best consistency in tested mini-class models: 11% frame-count variability, 100% `example_context` fill, verbatim quotes). `gpt-5` for semantic edge inference (the CONTRADICTS/MITIGATES distinction requires stronger reasoning).

**Evaluation infrastructure:** `assess.py` (metrics suite), `post_browser.py` (gold-standard test selection), 50+ Grok iteration cycles + 11 vanilla cycles in `iterations/` / `iterations_grok/`.

---

## 10. Remaining Work

### Phase 2: Graph + Semantic Edges (COMPLETE)
- [x] Structural provenance graph (`build_graph.py`)
- [x] Author frame generation (`make_se_author_frames.py`)
- [x] Semantic edge inference (`infer_edges.py`) — 4,107 edges classified
- [x] Multi-vector embeddings (`build_embeddings.py`) — 3,879 frames x 3 collections
- [x] Neo4j push scripts (`push_to_neo4j.py`, `push_semantic_edges.py`)
- [x] Subject normalization (`normalize_subjects.py`) — 3,870 subjects -> 3,292 clusters
- [x] Topic modeling (`topic_model.py`) — 39 topics via BERTopic + LLM labeling

### Phase 3: Agentic RAG (READY)
The actual research contribution. `dfm_agent.py` is built and ready to run. Core loop:

1. **Input:** Designer submits a CAD requirement or question (e.g., "I need a 1mm wall in 6061-T6 aluminum, will this cause problems?")
2. **Retrieve:** Query `frames_knowledge` + `frames_language` with the design question. Apply filters (`type:risk,heuristic`) to prioritize actionable frames
3. **Traverse:** Use retrieved `frame_id` values as entry points into the graph. Walk `MITIGATES` edges to find solutions, `CONTRADICTS` to surface disagreements, `SE_ANSWERS` + `POST_AUTHORED_BY` to weight by author credibility
4. **Synthesize:** LLM generates a DFM report grounded in retrieved frames, citing `source_quote` for traceability
5. **Output:** Structured report with risk flags, heuristics, contradicting expert opinions, and links back to source posts

### Phase 4: Evaluation
- Compare system output against: standard LLM baseline (no retrieval), DFM handbooks, ground-truth annotations
- Metrics: claim grounding rate (% of report claims backed by a source_quote), coverage of expert-identified risks, hallucination rate

---

## 11. File Map

```
# --- Current SE KG Pipeline ---
extract_se.py               # SE extraction pipeline (GPT-5 recommended)
schema_se.py                # Frame schema (Pydantic v2, 8 frame types)
normalize_se.py             # SE HTML -> plain text thread normalization
build_graph.py              # Build provenance graph (--input for file selection)
build_embeddings.py         # Build/append 3-collection Qdrant index (--append mode)
search_frames.py            # Interactive multi-vector search CLI
infer_edges.py              # Semantic edge inference (--resume for incremental)
normalize_subjects.py       # Subject normalization (--threshold, --review)
topic_model.py              # BERTopic topic modeling (--reduce-topics, --review)
dfm_agent.py                # Agentic RAG for DFM feedback synthesis
push_to_neo4j.py            # Push structural graph to Neo4j (--clear to wipe first)
push_semantic_edges.py      # Push semantic edges to Neo4j (--clear-semantic)
export_neo4j_csv.py         # Export CSVs + Cypher for manual Neo4j import
make_se_author_frames.py    # Generate author profiles from SE data
_diagnostics.py             # Pipeline statistics and counts

# --- Production Data ---
extractions_se_all.json     # Merged SE extractions (3,879 frames -- PRECIOUS)
extractions_se_5000.json    # v1 SE extractions (2,839 frames)
extractions_se_v3_new.json  # v3 new SE extractions (1,040 frames)
se_author_frames.json       # 562 SE author profiles
provenance_graph.gpickle    # Current graph (7,864 nodes, 13,048 edges)
provenance_graph.graphml    # Current graph (GraphML export)
semantic_edges.json         # 4,107 semantic edges (GPT-5 classified)
infer_edges_checkpoint.json # Checkpoint for incremental edge inference
subject_clusters.json       # 3,292 subject clusters (normalize_subjects.py)
subject_blocklist.json      # 10 blocked subject pairs
subject_embeddings.npz      # Cached subject embeddings (3,870 x 3072)
topic_model_output.json     # 39 topics with LLM labels (topic_model.py)
qdrant_data/                # Local Qdrant vector index (3,879 frames x 3)
neo4j_import/               # CSV exports for Neo4j
se_posts_normalized_v3_new.json  # v3 normalized extraction input

# --- SE Scraping ---
src/scraping_logic/         # SE scraper, GPT filter, all raw/filtered post data

# --- Archive ---
archive/                    # PM pipeline, old experiments, iterations, test files
```
