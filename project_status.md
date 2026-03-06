# DFM Knowledge Graph -- Project Status Update

**Date:** 2026-02-26
**Research question:** How can undocumented and tribal machining knowledge be collected and structured to improve the available knowledge base for AI-powered DFM interventions?

---

## Pipeline Overview

```
Stack Exchange API -> stackexchange_scraper.py -> normalize_se.py -> filter_with_gpt.py
    -> extract_se.py (LLM structured extraction) -> extractions_se_*.json (3,879 frames)
    -> build_embeddings.py (3 Qdrant collections, 11,637 vectors)
    -> build_graph.py (NetworkX provenance graph)
    -> infer_edges.py (GPT-5 semantic edge classification)
    -> push_to_neo4j.py / push_semantic_edges.py (Neo4j graph database)
    -> dfm_agent.py V2 (query decomposition + multi-hop RAG + edge-aware synthesis)
```

---

## Component Status

| Component | Script | Status | Output |
|-----------|--------|--------|--------|
| SE Scraping | `stackexchange_scraper.py` | COMPLETE | 12,805 questions -> 639 filtered -> 1,936 posts |
| SE Normalization | `normalize_se.py` | COMPLETE | `se_posts_normalized_v3_new.json` |
| SE Filtering | `filter_with_gpt.py` | COMPLETE | LLM relevance filtering applied |
| SE Extraction | `extract_se.py` | COMPLETE | 3,879 frames from 1,936 posts |
| SE Author Profiles | `make_se_author_frames.py` | COMPLETE | 1,138 authors (`se_author_frames.json`) |
| PM Scraping | `scraper.py` | COMPLETE | ~1,197 posts (`posts_enriched.json`) |
| PM Extraction | `extract.py` | PARTIAL | ~209 frames from test runs only; full run not executed |
| Embeddings | `build_embeddings.py` | COMPLETE | 3 Qdrant collections, 11,637 vectors (3,879 x 3) |
| Provenance Graph | `build_graph.py` | COMPLETE | 7,864 nodes, 13,048 edges (`provenance_graph.gpickle`) |
| Semantic Edge Inference | `infer_edges.py` | COMPLETE | 4,272 edges classified (`semantic_edges.json`) |
| Neo4j (provenance) | `push_to_neo4j.py` | COMPLETE | Provenance graph loaded into Neo4j Desktop |
| Neo4j (semantic) | `push_semantic_edges.py` | BUILT, NOT YET RUN | Script ready; needs execution |
| Neo4j CSV Export | `export_neo4j_csv.py` | COMPLETE | 10 CSVs + Cypher script in `neo4j_import/` |
| RAG Agent V2 | `dfm_agent.py` | **TESTED** | Query decomposition + multi-hop traversal + V2 synthesis |
| Report Schemas | `dfm_report_schema.py` | **NEW** | V1 + V2 output schemas and prompts |
| Frame Search | `search_frames.py` | BUILT | Interactive 2- or 3-vector Qdrant search |
| Iteration Loop | `iterate.py` | STALE | NOT updated for current schema (Feb 2026 refactor) |

---

## Graph Statistics

### Provenance Graph (NetworkX / Neo4j)

**Nodes (7,864):**
- Thread: 911
- Post: 1,936
- Author: 1,138
- Frame: 3,879

**Structural Edges (8,776):**
- THREAD_CONTAINS_POST: 1,936
- POST_AUTHORED_BY: 1,936
- POST_CONTAINS_FRAME: 3,879
- SE_ANSWERS: 1,025

### Semantic Edges (inferred via GPT-5, top_k=5)

**4,272 directed edges (from raw 2,507 classified pairs, expanded for directionality):**
- SPECIFIES: 2,111 (one frame narrows/scopes another)
- MITIGATES: 1,206 (one frame solves a risk in another)
- CAUSAL_LINK: 625 (one frame explains the physics behind another)
- CONTRADICTS: 330 (opposing guidance, bidirectional)

**Confidence distribution:** ~60% medium, ~38% high, ~2% low

---

## DFM Agent V2

### Architecture (7-phase upgrade, implemented 2026-02-26)

```
User Query
    |
    v
[1. Query Decomposition] -- LLM breaks broad query into 3-6 engineering sub-aspects
    |                        (auto for queries > 5 words, skippable with --no-decompose)
    v
[2. Multi-Aspect Retrieval] -- embed each sub-aspect + original query
    |                          retrieve top-K from frames_knowledge + frames_language
    |                          merge by max score across all sub-aspects
    v
[3. Multi-Hop Graph Traversal] -- bidirectional (successors + predecessors)
    |                              follows CAUSAL_LINK, MITIGATES, SPECIFIES, CONTRADICTS
    |                              captures edge confidence + reasoning metadata
    |                              configurable hops (default 2)
    v
[4. Enhanced Ranking] -- base_score * credibility * distance * accepted * edge_confidence
    |                     * frame_type_bonus * epistemic_weight * post_score_bonus
    v
[5. Edge-Aware Formatting] -- each frame block includes edge relationships, path chains,
    |                          epistemic stance, example context, SE tags
    v
[6. Two-Tier Synthesis] -- Evidence tier: strict citation, verbatim quotes
    |                       Inference tier: first-principles engineering rationale (labeled)
    v
[7. Discipline-Organized Report] -- grouped by scope (design_geometry, machining_process, etc.)
                                     reasoning chains, design rules, contradictions, knowledge gaps
```

### Report Formats

| Format | Flag | Output Structure |
|--------|------|-----------------|
| V1 (flat) | `--format v1` | summary, risks[], recommended_practices[], contradictions[], knowledge_gaps |
| V2 (discipline) | `--format v2` (default) | summary, disciplines[{first_principles, evidence[], reasoning_chains[]}], design_rules[], contradictions[], knowledge_gaps[] |

### CLI Options

```
python dfm_agent.py "query"                     # V2 report (default)
python dfm_agent.py "query" --format v1         # V1 backward compat
python dfm_agent.py "query" --verbose           # show retrieval/traversal details
python dfm_agent.py "query" --save              # save JSON to dfm_reports/
python dfm_agent.py "query" --max-hops 3        # deeper graph traversal
python dfm_agent.py "query" --no-decompose      # skip query decomposition
python dfm_agent.py "query" --max-frames 15     # limit frames to synthesis
python dfm_agent.py --interactive               # REPL mode
```

### Test Results (2026-02-26)

**V1 test** ("minimum wall thickness for aluminum CNC parts"):
- 18 seed frames, 122 traversed, 10 selected for synthesis
- 2 risks, 5 recommended practices, specific knowledge gaps
- GPT-5 synthesis succeeded in ~30s

**V2 test** ("what are the risks of thin walls in aluminum"):
- 6 sub-aspects generated (buckling, chatter, residual stress, fixturing, burr, threads)
- 96 seed frames from 7 queries, 339 traversed, 15 selected
- 5 discipline sections (design_geometry, machining_process, material, quality, other)
- 12 design rules, 7 specific knowledge gaps
- GPT-5 synthesis succeeded in ~90s

---

## Embeddings (Qdrant)

Three collections, each with 3,072-dim vectors (text-embedding-3-large, cosine distance):

| Collection | Embeds | Purpose |
|------------|--------|---------|
| `frames_knowledge` | subject + main_point + example_context + example_value + thread_context | Semantic DFM retrieval |
| `frames_language` | source_quote (raw shop talk) | Linguistic/jargon matching |
| `frames_metadata` | structured fields as natural language | Filtering by frame properties |

---

## Schema (Pydantic v2)

**8 frame types:** risk, heuristic, principle, case, workaround, observation, comparison, other

**Key fields per frame:** source_quote (verbatim), main_point, frame_type, scope (ScopeDomain), subject, applicability, epistemic_stance, example_value, example_context, extraction_rationale

**ScopeDomain:** design_geometry, machining_process, material, tooling, programming, shop_operations, quality, cost, machine_capability, other

**EpistemicStance:** absolute_authoritative, direct_neutral, hedged_emphatic, tentative_subjective, narrative_implied

---

## Neo4j Setup

- Neo4j Desktop 2026.01.4, database ID `6d9aaadf-111f-...`
- Connection: `bolt://localhost:7687` (Python driver) or `neo4j://127.0.0.1:7687` (Browser)
- Browser: `http://localhost:7474`
- Import directory redirected to `C:/Users/dougl/Documents/dfm_scraping/neo4j_import`
- Env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`

---

## Model Benchmarks (5-post test, 2 runs each)

| Model | Frames | Variability | EC Fill | Verbatim | Recommended |
|-------|--------|-------------|---------|----------|-------------|
| gpt-5-nano | 11-20 | 45% | 82-100% | Inconsistent | No |
| gpt-5-mini | 30-37 | 19% | ~100% | Paraphrases | Marginal |
| gpt-4.1-mini | 16-18 | 11% | 100% | Yes | Best mini-class |
| gpt-4.1 | 18-19 | -- | 100% | Yes | Best overall quality |
| gpt-5 | 22-23 | -- | 100% | Yes | Richest type diversity |
| gpt-4o-mini | 9 | -- | 60% | -- | No |

---

## Key Files

| File | Role | Safety |
|------|------|--------|
| `extractions_se_5000.json` | SE production extractions (batch 1) | Precious -- OVERWRITE protection |
| `extractions_se_v3_new.json` | SE production extractions (batch 2) | Precious |
| `extractions_se_all.json` | Full merged SE extractions | Precious |
| `semantic_edges.json` | Inferred semantic relationships | Regenerable via `infer_edges.py` |
| `provenance_graph.gpickle` | NetworkX graph (7,864 nodes, 13,048 edges) | Regenerable via `build_graph.py` |
| `se_author_frames.json` | Author metadata | Regenerable |
| `posts_enriched.json` | PM scraped posts | Do NOT modify |
| `schema_se.py` | Frame schema definitions | Field descriptions ARE the prompt |
| `dfm_agent.py` | RAG agent V2 | Main agent code |
| `dfm_report_schema.py` | V1 + V2 output schemas and prompts | Agent dependency |

---

## What's Next

1. **Run `push_semantic_edges.py`** -- load 4,272 semantic edges into Neo4j (script built, not yet executed)
2. **Verify Neo4j data** -- confirm all nodes + edges loaded correctly, explore graph visually
3. **Full PM extraction** -- run `extract.py` on all 1,197 PracticalMachinist posts (currently only test runs)
4. **Update `iterate.py`** -- adapt for current schema (field names changed in Feb 2026 refactor)
5. **Cross-source retrieval** -- test `dfm_agent.py` with unified SE + PM knowledge base
6. **Agent quality evaluation** -- systematic comparison of V1 vs V2 reports across diverse queries
7. **Parameter exploration** -- consider top_k=10 for semantic edge inference (current top_k=5; diminishing returns likely beyond k=10)
