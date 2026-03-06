# Embedding Search Tutorial: Qdrant + Multi-Vector

Step-by-step implementation guide for the frame embedding search system. Uses Qdrant (local vector database) with a multi-vector approach (Option B-alt + D from `docs_embedding_options.md`).

## Architecture Overview

```
extractions.json (209 frames)
  |
  |-- For each frame, build THREE texts:
  |     KNOWLEDGE = subject + main_point + example_context
  |     LANGUAGE  = source_quote
  |     METADATA  = frame_type, scope, applicability, etc. as natural language
  |
  |-- Send all texts to OpenAI text-embedding-3-small
  |     -> 209 x 3 = 627 vectors (1,536 dimensions each)
  |
  |-- Store in Qdrant as three collections:
  |     frames_knowledge  (knowledge vectors + full payload)
  |     frames_language   (language vectors + full payload)
  |     frames_metadata   (metadata vectors + full payload)
  |     All three share the same payload (metadata) per frame
  |
  QUERY:
  |-- User types a question
  |-- Embed question via OpenAI -> 1 query vector
  |-- Search relevant collections -> ranked results
  |-- Merge results by frame ID using max-score or weighted average
  |-- Display results with score breakdown
```

## Prerequisites

- Python 3.14+ (tested on 3.14.3)
- OpenAI API key set in environment (`OPENAI_API_KEY`)
- `extractions.json` from the extraction pipeline

## Step 1: Install

```powershell
.\venv\Scripts\pip.exe install qdrant-client
```

This installs `qdrant-client` (the Python client with embedded local storage). No server needed. You already have `openai` and `numpy`.

Note: ChromaDB was the original plan but is **incompatible with Python 3.14** due to a Pydantic v1 dependency. Qdrant provides the same capabilities.

## Step 2: Understand the Three Vectors

Each frame gets embedded three ways. Different vectors match different query styles.

### Vector 1: Knowledge (subject + main_point + example_context)

The distilled, transferable knowledge in clean technical language:
```
"local variable volatility: Local variables on CNC controls reset to null
when the program ends, meaning any data stored in them is lost between
program runs. Fanuc controls with custom macro B"
```

**Best matches queries like:** "What are the risks of using local variables for persistent data storage across program cycles?"

### Vector 2: Language (source_quote)

The raw machinist shop talk, verbatim from the forum:
```
"When the M30 or M02 is reached, all of the local variables
(#1 thru #33) will be cleared to vacant"
```

**Best matches queries like:** "why do my #100s keep going blank after M30"

### Vector 3: Metadata (structured fields as natural language)

The categorical profile of the frame:
```
"frame type: risk. post role: tutorial. scope: programming.
subject: local variable volatility. applicability: universal.
epistemic stance: direct neutral. extraction rationale: none.
validated: not yet reviewed. related frames: 4.
no known contradictions"
```

**Best matches queries like:** "show me validated heuristics about tooling"

### Why Not Just One Vector?

A single vector that concatenates everything produces a "blended" embedding that's decent at matching all query styles but optimal for none. Two or three vectors preserve the distinct signals. In testing, the language vector frequently "wins" for jargon-heavy queries while the knowledge vector wins for conceptual queries.

## Step 3: How Metadata Filtering Works (Separate from Embeddings)

Metadata fields like `frame_type`, `scope`, `applicability`, etc. are stored in the Qdrant "payload" alongside each vector. These enable **exact structured filtering** independent of the vector search:

```python
# "Find heuristics about programming" -- exact category match
query_filter = models.Filter(must=[
    models.FieldCondition(key="frame_type", match=models.MatchValue(value="heuristic")),
    models.FieldCondition(key="scope", match=models.MatchValue(value="programming")),
])

# "Find risk or heuristic frames from high-credibility authors"
query_filter = models.Filter(must=[
    models.FieldCondition(key="frame_type", match=models.MatchAny(any=["risk", "heuristic"])),
    models.FieldCondition(key="author_credibility_tier", match=models.MatchValue(value="high")),
])
```

This is different from the metadata VECTOR (Vector 3), which provides fuzzy semantic matching on frame categories. The exact filters are always available regardless of which vectors you search.

## Step 4: Build the Index

```powershell
.\venv\Scripts\python.exe build_embeddings.py
```

Output:
```
Loaded 209 frames from extractions.json
Embedding 209 frames across 3 collections...
  Knowledge texts: avg 317 chars
  Language texts:  avg 130 chars
  Metadata texts:  avg 287 chars
  Knowledge embeddings done (2.6s)
  Language embeddings done (3.5s)
  Metadata embeddings done (4.3s)
  Knowledge collection stored
  Language collection stored
  Metadata collection stored

Stored 209 frames in 3 collections (9.0s total)
Database saved to ./qdrant_data/
  Total vectors: 627
```

This creates a `./qdrant_data/` folder (~5-10MB). That folder IS your database. Back it up like any other data file.

To rebuild from a different extractions file:
```powershell
.\venv\Scripts\python.exe build_embeddings.py extractions_test.json
```

## Step 5: Search Interactively

```powershell
# Default: two-vector mode (knowledge + language)
.\venv\Scripts\python.exe search_frames.py

# Three-vector mode (includes metadata vector)
.\venv\Scripts\python.exe search_frames.py --three-vector

# Single collection only
.\venv\Scripts\python.exe search_frames.py --single knowledge
```

### Query Syntax

**Plain question:**
```
Query> what happens to variables when a program ends
```

**With metadata filters:**
```
Query> type:risk what can go wrong with macros
Query> scope:tooling best practices
Query> cred:high experienced advice on macro variables
Query> type:heuristic,risk scope:programming variable usage
```

**Filter prefixes:**

| Prefix | Field | Example Values |
|--------|-------|---------------|
| `type:` | frame_type | risk, heuristic, observation, principle, workaround, comparison, case, other |
| `scope:` | scope | programming, tooling, design_geometry, machining_process, material, machine_capability, shop_operations, quality, cost |
| `cred:` | author_credibility_tier | high, medium, low |
| `role:` | post_role | tutorial, solution, addition, general_advice, war_story, debate, question, reaction |
| `stance:` | epistemic_stance | absolute_authoritative, direct_neutral, hedged_emphatic, tentative_subjective, narrative_implied |
| `applicability:` | applicability | universal, context_dependent, situational |

Comma-separate for multi-value: `type:risk,heuristic`

### Interactive Commands

| Command | What It Does |
|---------|-------------|
| `:help` | Show query syntax help |
| `:mode 2` | Switch to two-vector mode (knowledge + language) |
| `:mode 3` | Switch to three-vector mode (knowledge + language + metadata) |
| `:top N` | Show top N results (default 5) |
| `:weights K L M` | Set merge weights for three-vector mode (e.g., `:weights 0.5 0.3 0.2`) |
| `:detail` | Toggle detailed output (shows full knowledge text, quote, metadata text) |
| `:stats` | Show collection sizes and current settings |
| `quit` | Exit |

## Step 6: Understanding Results

### Two-Vector Mode (Default)
```
  #1  [0.544]  best match: language
      Scores: K:0.520  L:0.544
      ID: post-1187971-1
      Type: observation  |  Scope: programming  |  Applic: context_dependent
      Subject: macro variable persistence on power cycle
      Author: Tonytn36 (high)
      Stance: direct_neutral  |  Role: addition
```

- **[0.544]** = final score (max of K and L scores)
- **best match: language** = the source_quote vector matched better than the main_point vector
- **K:0.520  L:0.544** = individual vector scores (cosine similarity, 0-1)
- The metadata fields (Type, Scope, etc.) come from the structured payload, not the vector

### Three-Vector Mode
```
  #1  [0.524]  best match: knowledge
      Scores: K:0.672  L:0.627
```

- **[0.524]** = weighted score: 0.5*0.672 + 0.3*0.627 + 0.2*0.0 = 0.524
- Weights are tunable via `:weights` command

### What "best match" Tells You

| best match | Meaning |
|------------|---------|
| knowledge | Frame found because its distilled `main_point` matched the query's concept |
| language | Frame found because its raw `source_quote` matched the query's phrasing/jargon |
| metadata | Frame found because its categorical profile (frame type, scope, etc.) matched the query |

This is diagnostic information: if most results come via "language," your query was phrased in machinist terms. If most come via "knowledge," your query was more formal/conceptual.

## Step 7: Merge Strategies

### Two-Vector: Max Score (Default)
```
combined = max(knowledge_score, language_score)
```
A frame ranks high if EITHER vector matches. Forgiving, broad recall.

### Three-Vector: Weighted Average
```
combined = 0.5 * knowledge + 0.3 * language + 0.2 * metadata
```
Each vector contributes proportionally. Tune weights based on results:
- If knowledge vector dominates too much: lower K weight, raise L
- If metadata vector adds noise: set M weight to 0 (effectively two-vector mode)
- If you want categorical queries to work better: raise M weight

### To Ignore Metadata Vector
Set its weight to zero:
```
:weights 0.6 0.4 0.0
```
This runs three searches but only scores on knowledge + language.

## File Reference

| File | Purpose |
|------|---------|
| `build_embeddings.py` | Loads extractions, embeds frames, creates Qdrant collections |
| `search_frames.py` | Interactive search CLI with mode/filter/weight controls |
| `./qdrant_data/` | Persistent Qdrant database folder (created by build script) |
| `docs_embedding_options.md` | All options considered (A through E) |
| `docs_embedding_tutorial.md` | This file |

## What To Tweak Later

1. **Embedding model**: Change `EMBEDDING_MODEL` in both scripts. Options: `text-embedding-3-large` (3,072 dims, better quality, 2x cost) -- also update `EMBEDDING_DIM`
2. **Merge weights**: Experiment with `:weights` interactively, then hardcode winners
3. **Feed results to an LLM**: Take top N frames, format as context, pass to GPT for reasoning (the RAG pattern)
4. **Build retrieval test set**: Create sample queries with known-correct frames to measure precision/recall
5. **Migrate to Neo4j**: When relationship data is curated, add vector indexes to graph nodes (Option E)
