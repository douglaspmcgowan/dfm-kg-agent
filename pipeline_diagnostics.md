# DFM Knowledge Graph -- Pipeline Diagnostics & Architecture

Generated: 2026-02-28

## 1. Research Question

How can undocumented and tribal machining knowledge be collected and structured to improve the available knowledge base for AI-powered design-for-manufacturability (DFM) interventions?

## 2. Pipeline Overview

```
SE API Scraping          stackexchange_scraper.py
       |
       v
GPT Relevance Filter     filter_with_gpt.py
       |
       v
HTML Normalization       normalize_se.py + make_se_author_frames.py
       |
       v
LLM Frame Extraction     extract_se.py + schema_se.py
       |
       v
   +---+---+---+
   |       |   |
   v       v   v
Graph   Embed  Semantic Edges
build_  build_ infer_
graph   embed  edges
   |       |   |
   v       v   v
   +---+---+---+
       |
       v
  Neo4j / Qdrant / Gephi
  push_to_neo4j.py
  push_semantic_edges.py
  export_neo4j_csv.py
  export_gephi.py
       |
       v
  Query & Synthesis
  search_frames.py
  dfm_agent.py
       |
       v
  Human Evaluation
  generate_reviewer_docs.py
  llm_reviewer.py
```

## 3. Data Volumes by Pipeline Stage

| Stage | Count | Notes |
|---|---|---|
| Questions scraped | 12,805 | Across 5 SE sites (v3 combined) |
| Questions sent to GPT filter | 12,805 | |
| Questions passing filter | 911 | 728 (v1) + 183 (v3) |
| Total posts (questions + answers) | 1,936 | Across filtered threads |
| Posts with extracted frames | 1,203 | 62% of posts had extractable knowledge |
| Posts with no relevant content | 733 | Questions, thin answers, reactions |
| **Frames extracted** | **3,879** | Production total (extractions_se_all.json) |
| Embedding vectors | 11,637 | 3,879 x 3 collections |
| Graph nodes | 7,864 | Post + Author |
| Structural graph edges | 13,048 | |
| Pairs classified for semantic edges | 10,389 | top-k=5 per frame |
| Semantic edges kept (non-UNRELATED) | 4,107 | 39.5% of classified pairs |

### Questions Scraped by Site

| Site | Questions Scraped | Questions Filtered | % Passing |
|---|---|---|---|
| electronics.stackexchange.com | 7,396 | 93 | 1.3% |
| engineering.stackexchange.com | 1,792 | 688 | 38.4% |
| robotics.stackexchange.com | 1,628 | 74 | 4.5% |
| mechanics.stackexchange.com | 1,496 | 13 | 0.9% |
| woodworking.stackexchange.com | 493 | 35 | 7.1% |

### Frame Type Distribution

| Frame Type | Count | % of Total |
|---|---|---|
| heuristic | 1,358 | 35.0% |
| observation | 1,327 | 34.2% |
| risk | 468 | 12.1% |
| principle | 410 | 10.6% |
| workaround | 154 | 4.0% |
| comparison | 147 | 3.8% |
| case | 15 | 0.4% |

### Scope Distribution

| Scope | Count | % of Total |
|---|---|---|
| design_geometry | 1,177 | 30.3% |
| machining_process | 605 | 15.6% |
| machine_capability | 533 | 13.7% |
| other | 428 | 11.0% |
| quality | 252 | 6.5% |
| programming | 241 | 6.2% |
| tooling | 221 | 5.7% |
| material | 165 | 4.3% |
| cost | 131 | 3.4% |
| shop_operations | 126 | 3.2% |

### Applicability Distribution

| Applicability | Count | % of Total |
|---|---|---|
| context_dependent | 2,530 | 65.2% |
| universal | 1,310 | 33.8% |
| situational | 39 | 1.0% |

### Epistemic Stance Distribution

| Epistemic Stance | Count | % of Total |
|---|---|---|
| direct_neutral | 2,597 | 67.0% |
| tentative_subjective | 1,028 | 26.5% |
| hedged_emphatic | 171 | 4.4% |
| absolute_authoritative | 69 | 1.8% |
| narrative_implied | 14 | 0.4% |

### Semantic Edge Distribution

| Edge Type | Count | % of Total |
|---|---|---|
| SPECIFIES | 2,111 | 51.4% |
| MITIGATES | 1,206 | 29.4% |
| CAUSAL_LINK | 625 | 15.2% |
| CONTRADICTS | 165 | 4.0% |

| Confidence | Count | % of Total |
|---|---|---|
| medium | 2,510 | 61.1% |
| high | 1,594 | 38.8% |
| low | 3 | 0.1% |

### Embedding Collections (Qdrant)

| Collection | Vectors | Embedding Source |
|---|---|---|
| frames_knowledge | 3,879 | subject + main_point + example_context + example_value |
| frames_language | 3,879 | source_quote (raw text) |
| frames_metadata | 3,879 | frame_type + scope + applicability + epistemic_stance + post_role |

All collections use OpenAI text-embedding-3-large (3,072 dimensions).

---

## 4. Script Architecture Reference

### 4.1 Scraping Layer

#### stackexchange_scraper.py
- **Purpose:** Fetch machining-related questions and answers from SE API
- **Input:** SE API (5 sites: engineering, mechanics, robotics, woodworking, electronics)
- **Output:** `stackexchange_posts_*.json`
- **Logic:** Keyword-chunked API queries (max 3 single-word OR terms per query; multi-word phrases isolated into their own queries), backoff header handling, keyed requests (10k/day quota)

#### filter_with_gpt.py
- **Purpose:** Score each question+answers for relevance to DFM/machining scope
- **Input:** `stackexchange_posts_*.json`, scope definition, `OPENAI_API_KEY`
- **Output:** `stackexchange_posts_*_filtered.json`
- **Logic:** GPT-5-mini scores each post 1-10 against scope definition; keeps >= 7/10; supports concurrent API calls (default 10), resume via `--skip-ids-from`

### 4.2 Normalization Layer

#### normalize_se.py
- **Purpose:** Convert filtered SE JSON (HTML bodies) into extraction-ready thread structure
- **Input:** `stackexchange_posts_*_filtered.json`
- **Output:** `se_posts_normalized_*.json`
- **Logic:** Groups posts by thread_id; converts HTML to plain text (BeautifulSoup); maps SE fields to extraction schema

#### make_se_author_frames.py
- **Purpose:** Extract author metadata into credibility-tiered records
- **Input:** Filtered SE posts
- **Output:** `se_author_frames.json`
- **Logic:** Derives credibility_tier from reputation (high >= 10k, medium >= 1k, low < 1k)

### 4.3 Extraction Layer

#### schema_se.py
- **Purpose:** Pydantic v2 schema for frame extraction (field descriptions ARE the LLM prompt)
- **Defines:** FrameType (8), ScopeDomain (10), Applicability (3), PostRole (9), EpistemicStance (5), KnowledgeFrame (23 fields), ExtractionOutput (LLM-facing), SEPostExtractionResult (output)
- **Design:** Field descriptions travel to LLM via structured output API; system prompt covers behavioral rules only

#### extract_se.py
- **Purpose:** Core extraction pipeline -- GPT structured output per post
- **Input:** `se_posts_normalized_*.json`, `OPENAI_API_KEY`, `OPENAI_MODEL`
- **Output:** `extractions_se_*.json`
- **Logic:**
  - Per-thread extraction: question first (no context), then each answer (with truncated question as context)
  - Uses `client.beta.chat.completions.parse()` with `response_format=ExtractionOutput`
  - Post-processing: fuzzy quote matching (SequenceMatcher, 0.85 threshold, +/-10% window for quotes > 80 chars)
  - Epistemic stance overrides: hedge/strong keyword patterns correct LLM misclassifications
  - Optional consolidation: dedup frames above 0.90 similarity threshold
  - Parallel execution: ThreadPoolExecutor (default 10 workers), resumable checkpoints
- **Key env vars:** `OPENAI_MODEL` (default gpt-5-nano), `MAX_POSTS` (default 5), `PARALLEL_REQUESTS` (default 10), `OVERWRITE` (default false)

### 4.4 Graph Construction Layer

#### build_graph.py
- **Purpose:** Build NetworkX provenance graph from extractions
- **Input:** `extractions_se_all.json`, `se_author_frames.json` (optional)
- **Output:** `provenance_graph.gpickle`, `provenance_graph.graphml`
- **Logic:** Directed multigraph with 4 node types (Thread, Post, Author, Frame) and 6 structural edge types (THREAD_CONTAINS_POST, POST_AUTHORED_BY, POST_CONTAINS_FRAME, SE_ANSWERS, FRAME_RELATED_TO, FRAME_CONTRADICTED_BY)

### 4.5 Vector Embedding Layer

#### build_embeddings.py
- **Purpose:** Build Qdrant vector index from extracted frames
- **Input:** `extractions_se_all.json`, `OPENAI_API_KEY`
- **Output:** `qdrant_data/` (3 collections)
- **Logic:** Creates 3 collections with different text embeddings (knowledge, language, metadata); OpenAI text-embedding-3-large (3,072 dim); batch size 2,048

### 4.6 Semantic Inference Layer

#### infer_edges.py
- **Purpose:** Infer semantic relationships between frames using GPT-5 + vector similarity
- **Input:** `provenance_graph.gpickle`, `qdrant_data/`, `OPENAI_API_KEY`
- **Output:** `semantic_edges.json`, updated graph files
- **Logic:** For each frame, fetch top-K similar via Qdrant; classify each pair with GPT-5 structured output (EdgeClassification: CONTRADICTS, MITIGATES, SPECIFIES, CAUSAL_LINK, or UNRELATED); supports resume/checkpoint
- **Stats:** 10,389 pairs classified; 4,107 edges kept (39.5%)

### 4.7 Search & Query Layer

#### search_frames.py
- **Purpose:** Interactive multi-vector search CLI
- **Input:** `qdrant_data/`, `OPENAI_API_KEY`
- **Logic:** Two-vector mode (knowledge + language) or three-vector (+ metadata); configurable weights; query filters (type, scope, cred, accepted, site, tag)

#### dfm_agent.py
- **Purpose:** RAG agent for DFM feedback synthesis
- **Input:** `qdrant_data/`, `provenance_graph.gpickle`, `OPENAI_API_KEY`
- **Output:** Structured DFM report
- **Logic:** Query decomposition into sub-aspects; vector retrieval; semantic edge traversal (configurable hops); frame ranking by relevance x credibility x edge_confidence; GPT synthesis into V1 (flat) or V2 (discipline-organized with reasoning chains) report format

### 4.8 Export Layer

#### push_to_neo4j.py
- **Purpose:** Push provenance graph to Neo4j
- **Input:** `provenance_graph.gpickle`, Neo4j connection
- **Logic:** Batched transactions (500/batch); creates all node + structural edge types

#### push_semantic_edges.py
- **Purpose:** Push semantic edges to Neo4j
- **Input:** `semantic_edges.json`, Neo4j connection
- **Logic:** Creates directional semantic edges with confidence and reasoning attributes

#### export_neo4j_csv.py
- **Purpose:** CSV export for Neo4j bulk import
- **Output:** `neo4j_import/` directory with per-type CSVs + Cypher import script

#### export_gephi.py
- **Purpose:** GEXF export for Gephi network visualization
- **Output:** `.gexf` file (slim mode: frames + semantic edges only; full mode: all nodes/edges)

### 4.9 Evaluation Layer

#### generate_reviewer_docs.py
- **Purpose:** Generate stratified-sampled markdown for human frame evaluation
- **Input:** `extractions_se_all.json`, random seed
- **Output:** `reviewer_docs/reviewer_1.md` through `reviewer_N.md`
- **Logic:** Post-based stratified sampling by dominant frame_type per post; proportional round-robin draw until target frame count reached; PII stripping (author, URLs, dates, reputation); self-contained reviewer instructions with rubric tables

#### llm_reviewer.py
- **Purpose:** Automated GPT-5 reviewer using same rubric as human reviewers
- **Input:** `extractions_se_all.json`, seed (same as human reviewers), `OPENAI_API_KEY`
- **Output:** `reviewer_docs/llm_review.json`, `reviewer_docs/reviewer_llm.md`
- **Logic:** Same `stratified_sample_v2` for identical frame selection; FrameReview Pydantic schema for structured output; parallel API calls (default 10 workers)

### 4.10 Diagnostics

#### _diagnostics.py
- **Purpose:** Pipeline-wide statistics (counts at each stage)

#### _site_diagnostics.py
- **Purpose:** Per-site extraction acceptance rates and skip patterns

---

## 5. Controlled Vocabularies

### Frame Types (8)

| Type | Definition |
|---|---|
| risk | Condition(s) leading to failure mode or operational limitation |
| heuristic | Personal practice/preference with rationale, or directed advice with rationale |
| principle | Causal explanation of WHY something happens |
| observation | Empirical/factual/definitional claim -- no action, no causal mechanism |
| comparison | Explicitly weighs two or more options against each other |
| workaround | Non-standard fix for a stated limitation |
| case | Specific real-world job or incident narrative |
| other | Explain in extraction_rationale |

### Scope Domains (10)

| Scope | Coverage |
|---|---|
| design_geometry | Part features, tolerances, wall thickness, radii, hole sizing |
| machining_process | Toolpaths, speeds/feeds, fixturing, operations, sequencing |
| material | Material behavior, machinability, selection, stock availability |
| tooling | Tool selection, geometry, wear, availability, breakage |
| programming | CNC code, macros, CAM strategy, control-specific behavior |
| shop_operations | Scheduling, machine availability, personnel, capacity |
| quality | Inspection, surface finish, tolerances, measurement |
| cost | Cost drivers, quoting, setup time, efficiency, batch decisions |
| machine_capability | Specific machine/control features, limits, configurations |
| other | Explain in extraction_rationale |

### Applicability (3)

| Level | Definition |
|---|---|
| universal | True across virtually all instances of the subject |
| context_dependent | True under specific conditions (material, brand, operation) |
| situational | Specific to a particular shop/machine configuration |

### Epistemic Stance (5)

| Stance | Definition |
|---|---|
| absolute_authoritative | Objective fact, strong language (must, always, never), no hedges |
| direct_neutral | Stated as fact, technical/tutorial tone, no emphasis or hedges |
| hedged_emphatic | Personal hedges + strong language (in my experience... hate/required) |
| tentative_subjective | Weak/uncertain language (might, could, maybe, I think, usually) |
| narrative_implied | Not directly stated, inferred from story or results |

### Post Roles (9)

| Role | Definition |
|---|---|
| question | OP asking for help or information |
| solution | Direct answer to a problem posed in the thread |
| addition | Reply that adds supplementary info, tips, alternatives |
| general_advice | Unprompted guidance not tied to a specific problem |
| tutorial | Teaching-oriented post explaining concepts or techniques |
| war_story | Narrative of a specific past job with struggle/challenge/surprise |
| debate | Disagreement between users |
| reaction | Reply that endorses, acknowledges, or thanks without substantive knowledge |
| other | |

---

## 6. Human Evaluation Study Design

### Sampling Method

- **Strategy:** Post-based stratified sampling by frame_type
- **Assignment rule:** Each post assigned to the frame_type of its most frequent frame (ties broken by first occurrence)
- **Draw method:** Proportional round-robin across strata until cumulative frames >= 25
- **Overshoot policy:** If the last post pushes past 25, include it anyway (at least 25 frames guaranteed)

### Sample (seed=42)

| Metric | Value |
|---|---|
| Target frames per reviewer | 25 |
| Posts sampled | 7 |
| **Actual frames in review** | **26** |
| Reviewers (human) | 4 |
| Reviewers (LLM) | 1 (GPT-5) |

### Stratum Breakdown

| Stratum | Corpus Posts (%) | Sampled Posts | Sampled Frames |
|---|---|---|---|
| observation | 482 (40.1%) | 3 | 10 |
| heuristic | 425 (35.3%) | 2 | 9 |
| principle | 123 (10.2%) | 1 | 5 |
| risk | 112 (9.3%) | 1 | 2 |
| workaround | 29 (2.4%) | 0 | 0 |
| comparison | 28 (2.3%) | 0 | 0 |
| case | 4 (0.3%) | 0 | 0 |

### Rating Rubric

**Quality ratings (1-3 scale):**

| Rating | Extraction Accuracy | Tribal Knowledge | Clarity |
|---|---|---|---|
| 1 | Quote does not support main point | Generic textbook knowledge | Poorly written, ambiguous |
| 2 | Mostly captures, minor issues | Moderately specialized | Understandable but imprecise |
| 3 | Accurately and completely captures | Genuine tribal knowledge | Clear, precise, actionable |

**Classification ratings (1-5 scale):**

| Rating | Meaning |
|---|---|
| 1 | Completely wrong |
| 2 | Poor -- a different label would clearly be better |
| 3 | Acceptable -- reasonable but not ideal |
| 4 | Good -- correct and appropriate |
| 5 | Excellent -- could not be improved |

**Overall judgment:** Keep / Revise / Remove

---

## 7. LLM Review Results (GPT-5, seed=42)

### Overall Verdict

| Verdict | Count | % |
|---|---|---|
| Keep | 12 | 46.2% |
| Revise | 14 | 53.8% |
| Remove | 0 | 0.0% |

### Quality Ratings (1-3 Scale)

| Metric | Mean | Score=1 | Score=2 | Score=3 |
|---|---|---|---|---|
| Extraction Accuracy | 2.50 | 0 | 13 | 13 |
| Tribal Knowledge | 1.73 | 8 | 17 | 1 |
| Clarity | 2.92 | 0 | 2 | 24 |

### Classification Ratings (1-5 Scale)

| Metric | Mean | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Frame Type | 4.38 | 0 | 4 | 1 | 2 | 19 |
| Post Role | 4.88 | 0 | 0 | 0 | 3 | 23 |
| Scope | 3.62 | 1 | 6 | 4 | 6 | 9 |
| Applicability | 4.15 | 1 | 3 | 2 | 5 | 15 |
| Main Point Quality | 4.31 | 0 | 0 | 3 | 12 | 11 |
| Epistemic Stance | 4.77 | 0 | 0 | 1 | 4 | 21 |

### Revise Severity Analysis

Of the 14 frames marked Revise:

| Severity | Count | Definition |
|---|---|---|
| Critical | 12 | At least one classification score <= 2 |
| Minor | 2 | All scores >= 3, minor phrasing concerns only |

### Labels Flagged as Incorrect (score <= 2 in Revise frames)

| Label | Times Flagged | Most Common Issue |
|---|---|---|
| Scope | 5 | Frame spans two domains but only one label allowed |
| Frame Type | 3 | Boundary between observation/principle/risk |
| Applicability | 3 | context_dependent vs. universal disagreement |
| Post Role | 0 | |
| Epistemic Stance | 0 | |
| Main Point Quality | 0 | |

### Key Findings

1. **No frames warranted removal.** GPT-5 found all 26 frames captured real knowledge worth keeping or refining.

2. **Clarity is high** (mean 2.92/3). The extraction pipeline produces well-written, understandable main points.

3. **Post role and epistemic stance labels are reliable.** Mean ratings of 4.88 and 4.77 respectively; zero frames scored <= 2 on either.

4. **Scope is the weakest label** (mean 3.62). Six frames had scope scores <= 2, typically because the knowledge spans two domains (e.g., material + geometry) but the schema allows only one.

5. **Tribal knowledge scores are low** (mean 1.73). Most frames capture moderately specialized (2) or textbook-level (1) knowledge. Only 1 of 26 frames scored 3 (genuine tribal knowledge). This may reflect the SE sites' bias toward standard engineering Q&A rather than undocumented shop knowledge.

6. **Extraction accuracy is split** (mean 2.50). Half of frames perfectly capture the quote; the other half have minor omissions or additions not present in the source text.

---

## 8. File Inventory

### Production Data (do not hand-edit)

| File | Size | Contents |
|---|---|---|
| extractions_se_all.json | 8.6 MB | 1,936 posts, 3,879 frames (merged production) |
| extractions_se_5000.json | 6.3 MB | 1,447 posts, 2,839 frames (component) |
| extractions_se_v3_new.json | 2.3 MB | 489 posts, 1,040 frames (component) |
| provenance_graph.gpickle | -- | 7,864 nodes, 13,048 edges |
| semantic_edges.json | -- | 4,107 semantic edges |
| qdrant_data/ | -- | 11,637 vectors across 3 collections |

### Evaluation Outputs

| File | Contents |
|---|---|
| reviewer_docs/reviewer_1.md - reviewer_4.md | Human reviewer evaluation forms (26 frames each) |
| reviewer_docs/reviewer_llm.md | GPT-5 automated review (filled in) |
| reviewer_docs/llm_review.json | GPT-5 ratings as structured JSON |

### Regenerable (safe to rebuild)

| File | Rebuilt By |
|---|---|
| provenance_graph.gpickle | `python build_graph.py --input extractions_se_all.json` |
| provenance_graph.graphml | `python build_graph.py --input extractions_se_all.json` |
| semantic_edges.json | `python infer_edges.py` |
| qdrant_data/ | `python build_embeddings.py extractions_se_all.json` |
| neo4j_import/ | `python export_neo4j_csv.py` |
