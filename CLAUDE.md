# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DFM Knowledge Graph extraction system: scrapes CNC machining forum posts (PracticalMachinist.com, Stack Exchange engineering/electronics/robotics/mechanics/woodworking sites) and uses LLM structured extraction to convert tribal machining knowledge into a queryable knowledge graph for design-for-manufacturability (DFM) feedback systems.

**Research question:** How can undocumented and tribal machining knowledge be collected and structured to improve the available knowledge base for AI-powered DFM interventions?

## Running Code

Activate the virtual environment before running any Python:
```
.\venv\Scripts\Activate.ps1
```

### Current SE KG Pipeline

```
SE API -> stackexchange_scraper.py -> stackexchange_posts_*.json
  -> filter_with_gpt.py -> stackexchange_posts_*_filtered.json
  -> normalize_se.py -> se_posts_normalized_*.json
  -> extract_se.py -> extractions_se_*.json
  -> build_graph.py -> provenance_graph.gpickle
  -> build_embeddings.py -> qdrant_data/
  -> infer_edges.py -> semantic_edges.json
  -> normalize_subjects.py -> subject_clusters.json
  -> topic_model.py -> topic_model_output.json
  -> push_to_neo4j.py + push_semantic_edges.py -> Neo4j
  -> dfm_agent.py -> DFM reports
```

### Key Scripts

- `python extract_se.py` -- SE extraction pipeline (set `OPENAI_MODEL`, `INPUT_FILE`, `MAX_POSTS`, `OUTPUT_FULL_FILE`)
- `python normalize_se.py --input <file> --output <file>` -- convert SE HTML to plain text threads
- `python build_graph.py --input extractions_se_all.json` -- build provenance graph
- `python build_embeddings.py [file] [--append]` -- build or append to Qdrant vector index
- `python infer_edges.py [--resume] [--limit N]` -- semantic edge inference with GPT-5
- `python normalize_subjects.py [--threshold 0.30] [--review]` -- cluster similar subjects into canonical names
- `python topic_model.py [--reduce-topics 40] [--review]` -- BERTopic topic modeling with LLM labels
- `python search_frames.py` -- interactive multi-vector search CLI
- `python dfm_agent.py "query"` -- RAG agent for DFM feedback
- `python push_to_neo4j.py [--clear]` -- push structural graph to Neo4j
- `python push_semantic_edges.py [--clear-semantic]` -- push semantic edges to Neo4j
- `python _diagnostics.py` -- pipeline statistics and counts

### SE Scraping (in src/scraping_logic/)

- `python src/scraping_logic/stackexchange_scraper.py` -- scrape SE API
- `python src/scraping_logic/filter_with_gpt.py` -- GPT relevance filtering (`--skip-ids-from` for incremental)

### Archived PM Pipeline (in archive/)

Old PracticalMachinist pipeline scripts: `scraper.py`, `extract.py`, `schema.py`, `assess.py`, `iterate.py`, `post_browser.py` and associated data files.

### Environment Variables

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (extraction model, default: gpt-5-nano; recommend gpt-5 for production)
- `MAX_POSTS` -- default 5, set to 0 for all posts
- `INPUT_FILE` -- default: se_posts_normalized_v3_new.json
- `OUTPUT_FULL_FILE` -- default: extractions.json (set to new filename for SE runs)
- `OVERWRITE` -- default: false; refuses to overwrite extractions.json unless set to true
- `PARALLEL_REQUESTS` -- default: 10 (concurrent API calls)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` -- for Neo4j push scripts

## Core Extraction Target: KnowledgeFrame

Eight frame types in `schema_se.py` (Pydantic v2):
- **risk** -- condition(s) -> failure mode or operational limitation
- **heuristic** -- personal practice/preference with rationale, OR directed advice with rationale
- **principle** -- causal explanation WHY something happens
- **case** -- specific real-world job or incident narrative
- **workaround** -- non-standard fix for a stated limitation
- **observation** -- empirical/factual/definitional claim -- no action, no causal mechanism
- **comparison** -- weighs two or more options/approaches against each other
- **other** -- explain in `extraction_rationale`

Critical fields: `source_quote` (must be verbatim from post), `main_point`, `frame_type`, `scope` (ScopeDomain), `subject`, `applicability`, `epistemic_stance`, `example_value` (illustrative numbers), `example_context` (required when applicability != universal), `extraction_rationale` (required when frame_type or scope == other).

## Current Data Volumes

| Stage | Count |
|---|---|
| Questions scraped | 12,805 across 5 SE sites |
| Questions filtered | 639 (1,399 posts with answers) |
| Posts sent to LLM | 1,936 (from 911 threads) |
| Frames extracted | 3,879 |
| Embedding vectors | 11,637 (3,879 x 3 collections) |
| Graph nodes | 7,864 (Frame 3,879 + Post 1,936 + Author 1,138 + Thread 911) |
| Structural edges | 8,776 |
| Semantic edges | 4,272 (SPECIFIES 2,111 + MITIGATES 1,206 + CAUSAL_LINK 625 + CONTRADICTS 330) |
| Subject clusters | 3,292 (from 3,870 unique subjects, threshold 0.30, 10 blocked pairs) |
| Topics (BERTopic) | 39 topics + 619 outliers (16%), LLM-labeled via gpt-4.1-mini |

## File Safety

- `extractions_se_all.json` is the merged production output -- treat as precious
- `extractions_se_5000.json` and `extractions_se_v3_new.json` are the component extraction files
- `provenance_graph.gpickle` and `semantic_edges.json` are rebuilt by scripts -- safe to regenerate
- When testing extraction with MAX_POSTS > 5, set `OUTPUT_FULL_FILE` to a new filename

## Guardrails

- **Do NOT hand-edit** large JSON data files (`extractions*.json`, `se_posts_normalized*.json`)
- **Do NOT modify** contents of `archive/` (old pipeline preserved for reference)
- **Do NOT modify** SE scraping data in `src/scraping_logic/` (read-only post files)
- **Never delete files** -- move to `./trash/` if removal is necessary
- **Create `.bak` backups** before making significant changes to any file (no git in this repo)
- Keep existing Python formatting style; do not add new formatting/lint tooling unless asked

## Making Changes

When modifying extraction behavior:
- Prefer small, targeted edits to `Field(description=...)` in `schema_se.py` or specific sections in `SYSTEM_PROMPT` (in `extract_se.py`)
- Explain why the change addresses a concrete failure pattern
- Preserve the overall structure and logic of the current pipeline

## Reference Documentation

- `dfm_kg_architecture_v2.md` -- comprehensive architecture reference (data sources, embeddings, graph, semantic edges, remaining work)
- `dfm_kg_project_context.md` -- project reference (pipeline details, frame types, schema fields, known issues)
- `pastable_proj_context.md` -- quick reference for code editors
- `docs_embedding_options.md` -- embedding approach evaluation
- `docs_embedding_tutorial.md` -- embedding implementation guide
