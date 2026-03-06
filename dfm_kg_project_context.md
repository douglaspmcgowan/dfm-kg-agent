# DFM Knowledge Graph — Project Context

*Forum → Frame → Knowledge Graph pipeline for machining design-for-manufacturability heuristics*

**Data sources:** PracticalMachinist.com · CNCzone · Reddit r/CNC · r/CAM · r/Machinists  
**Current status:** Scraper working · Extraction iterating · KG construction planned

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Full Pipeline](#2-full-pipeline)
3. [Scraper](#3-scraper)
4. [Frame Extraction](#4-frame-extraction)
5. [Schema](#5-schema)
6. [Prompt Logic & Extraction Rules](#6-prompt-logic--extraction-rules)
7. [Known Issues](#7-known-issues)
8. [Roadmap](#8-roadmap)
9. [Automated Eval Loop](#9-automated-eval-loop)
10. [Key Files](#10-key-files)

---

## 1. Project Overview

**Research question:** How can undocumented and tribal machining knowledge be collected and structured to improve the available knowledge base for AI-powered DFM interventions?

Machinists accumulate enormous practical knowledge about what designs are easy or hard to machine — but this knowledge lives in forum posts, not textbooks. The goal is to extract it, structure it, and make it queryable for DFM (design for manufacturability) feedback systems.

**Novelty:**
- Advances DFM beyond rule-based approaches to include shop context and practical experience
- Builds a structured, AI-readable knowledge base of DFM heuristics, risks, and principles
- Establishes a general NLP method for capturing specialized design knowledge from forums

**Tech stack:**

| Component | Tool | Status |
|---|---|---|
| Scraper | Python / BeautifulSoup / requests (`scraper.py`, `config.py`) | ✅ Working |
| Frame extraction | OpenAI structured outputs (`extract.py`, `schema.py`) | 🔄 Iterating |
| Schema validation | Pydantic v2 | ✅ Working |
| KG store | TBD (Neo4j likely) | 📋 Planned |
| Human review | Label Studio or spreadsheet | 📋 Planned |
| Eval loop | Python harness (designed, not yet built) | 📋 Planned |

---

## 2. Full Pipeline

```
Forum pages
    ↓
scraper.py  (Pass 1A → thread URLs, Pass 1B → posts, Pass 2 → author profiles)
    ↓
posts_enriched.json
    ↓
extract.py  (quote stripping → LLM extraction → filtering → consolidation)
    ↓
extractions.json
    ↓
Human review (Label Studio / spreadsheet)
    ↓
Validated frame database
    ↓
Entity resolution  ("6061" = "AL6061" = "6061-T6" → canonical node)
    ↓
KG construction  (frames → nodes/edges)
    ↓
DFM feedback system  (KG-Augmented RAG)
```

**Knowledge structuring layers (complementary, not alternatives):**

- **Raw text / document store** — trivial archive, nothing lost, but no structure
- **Vector embeddings** — secondary retrieval layer, fuzzy semantic search, good fallback for KG gaps
- **Frames (primary extraction target)** — structured templates, easier extraction target than KG directly
- **Knowledge Graph (primary research contribution)** — multi-hop reasoning, explicit causality, traceable
- **Cases** — secondary artifact, good for human validation and training data

---

## 3. Scraper

**File:** `scraper.py` · **Config:** `config.py`

### Two-Pass Architecture

**Pass 1A — Category crawl:** Walk category listing pages, collect thread URLs and author profile URLs. XenForo's `a.username` element carries the full `/forum/members/name.id/` href. The numeric member ID is opaque — it cannot be derived from the display name. Always read it from the href.

**Pass 1B — Thread scrape:** Scrape all posts per thread. Per post:
- `id`, `thread_url`, `thread_title`, `post_position`
- `author`, `author_profile_url`, `date`
- `text`, `text_length`
- `post_reaction_count`
- `quoted_usernames` (list of usernames quoted in this post)
- `is_op_followup` (bool — OP returning after a solution post)

**Pass 2 — Author profiles:** For each unique author, fetch `/forum/members/username.id/`. Extracts:
- `post_count` — label is **"Posts"**, NOT "Messages"
- `reaction_score`
- `join_date` — lives in `dl.pairs--inline`, NOT `dl.pairs--rows`
- `rank` — badge text from `span.userTitle`

Derives `author_credibility_tier` and saves to `authors.json`.

**Merge:** Author data injected into each post → `posts_enriched.json`. This is the input to `extract.py`.

### Credibility Tier Rules

```
high   = post_count > 1000  AND  senior/diamond/gold/veteran/master rank
medium = post_count >= 200  OR   senior rank
low    = everything else
```

Store both raw numbers and derived tier. Raw numbers let you revisit thresholds later.

### Thread Endorsement Signals (in order of reliability)

1. `post_reaction_count` — explicit, always present
2. `is_op_followup` — OP returning after a solution post
3. `quoted_usernames` — this post was quoted by a subsequent post
4. Best-answer marker (XenForo plugin — not always present)
5. "As [username] said above..." pattern

### Key Config (`config.py`)

```python
MAX_THREADS        = 50
MAX_POSTS          = 250
MAX_CATEGORY_PAGES = 10
MIN_POST_LENGTH    = 50
REQUEST_DELAY      = 2.0  # seconds between requests
PROFILE_DELAY      = 1.5
```

---

## 4. Frame Extraction

**File:** `extract.py`

### What It Does

1. Loads `posts_enriched.json`
2. For each post: strips quoted/replied text, calls OpenAI structured outputs API with `PostExtractionResult` schema
3. Filters output frames: drop missing quotes, drop hallucinated quotes (fuzzy match), drop missing main_point
4. Consolidation: dedup near-identical frames
5. Writes `extractions_test.json` (≤5 posts) or `extractions.json` (full run)

### Quote Stripping Patterns (applied in order)

```python
[quote]...[/quote]                          # BBCode
> lines                                      # Email-style
Originally posted by X:\n...                # Legacy format
Username said: ... Click to expand...        # XenForo
Username said:\n... (to next blank line)    # XenForo no-expand
```

### Post-Extraction Filtering

Every frame passes three checks before being kept:

1. **Missing quote** — `source_quote` is empty → drop
2. **Quote mismatch** — `source_quote` doesn't fuzzy-match (≥85%) the post text → drop (catches hallucinated quotes)
3. **Missing main_point** → drop

Additionally: if `source_quote` contains example markers ("for brevity", "let's say", "for an example", etc.), any `condition_value` is moved to `example_value` automatically.

### Consolidation (two phases)

**Phase 1 — Rules (deterministic):**
- Clear `condition_value` if it duplicates an `example_value`
- Flag source_quote pairs with ≥90% similarity for LLM review

**Phase 2 — LLM merge (only on flagged pairs):**
- MERGE → combine into one frame
- KEEP_BOTH → genuinely distinct claims from same quote
- KEEP_FIRST / KEEP_SECOND → one is more complete

### Environment Variables

```bash
OPENAI_API_KEY           # required
OPENAI_MODEL             # default: gpt-4o-mini
INPUT_FILE               # default: posts_enriched.json
MAX_POSTS                # default: 5  (0 = all)
SLEEP_SECONDS            # default: 0.2
OUTPUT_TEST_FILE         # default: extractions_test.json
OUTPUT_FULL_FILE         # default: extractions.json
DEBUG_FILTERING=true     # verbose filter output
ENABLE_CONSOLIDATION     # default: true
CONSOLIDATION_THRESHOLD  # default: 0.90
```

### Run Commands

```bash
python extract.py                             # 5 posts (default test)
MAX_POSTS=0 python extract.py                 # all posts
MAX_POSTS=20 python extract.py                # 20 posts
DEBUG_FILTERING=true python extract.py        # verbose output
ENABLE_CONSOLIDATION=false python extract.py  # skip dedup
```

---

## 5. Schema

**File:** `schema.py` — Pydantic v2. Field names mirror `posts_enriched.json` exactly.

### Frame Types

| Type | Description |
|---|---|
| `risk` | Conditions that produce a failure mode — what goes wrong and why |
| `heuristic` | Rule of thumb from experience — "I do X because Y", personal practices with rationale |
| `principle` | Physical/mechanical truth explaining WHY — causal, not definitional |
| `case` | Specific real job or problem with context and resolution |
| `workaround` | Non-standard fix for a constraint-driven problem |
| `recommendation` | Explicit advice about what should be done (design, process, or tooling) |
| `observation` | Empirical fact about how something behaves — no action, no causal claim |
| `other` | Genuine value that doesn't fit above — always explain in `validation_notes` |

### KnowledgeFrame Fields

| Field | Required | Description |
|---|---|---|
| `frame_type` | ✅ | One of 8 types above |
| `source_quote` | ✅ | **Non-negotiable.** Verbatim text from the post. Must be a recognizable substring. |
| `main_point` | ✅ | Core claim in 1–2 sentences. Content depends on `frame_type`. |
| `post_role` | ✅ | `question` / `solution` / `war_story` / `debate` / `general_advice` / `tutorial` / `other` |
| `directness` | ✅ | `explicit` / `hedged` / `implied` |
| `expression_strength` | ✅ | `strong` / `moderate` / `weak` |
| `scope` | optional | Free-text: what this frame applies to. Name features, materials, ops, machine types. |
| `subject` | optional | Concise noun phrase — primary anchor for KG node construction. |
| `condition_value` | optional | Real asserted threshold/parameter. NOT toy values. |
| `example_value` | optional | Illustrative toy value only — prevents polluting parametric data. |
| `author_username` | optional | Foreign key to AuthorFrame (keyed by username). |
| `source_url` | optional | `thread_url` from scraper. |
| `post_date` | optional | ISO string from scraper. |
| `post_reaction_count` | optional | Endorsement signal. |
| `thread_endorsement` | optional | Derived from `quoted_usernames` + `is_op_followup`. Never LLM-filled. |
| `contradicted_by` | optional | Frame IDs. Populated during human review. |
| `related_frames` | optional | Frame IDs. Populated during human review. |
| `validated` | optional | Boolean. Null until reviewed. |
| `validation_notes` | optional | Reviewer notes. Contradictions are findings — note what context might resolve them. |

### AuthorFrame Fields

One record per unique author. Keyed by username in `authors.json`. All field names match scraper output exactly.

| Field | Notes |
|---|---|
| `author_username` | Primary key |
| `post_count` | From profile, label "Posts" (not "Messages") |
| `reaction_score` | From profile |
| `join_date` | From `dl.pairs--inline` (not `pairs--rows`) |
| `rank` | Badge text e.g. "Diamond", "Senior Member" |
| `author_credibility_tier` | Derived: `high` / `medium` / `low` |
| `profile_fetch_failed` | True if Pass 2 failed — all fields will be None |

### PostExtractionResult Fields

Wrapper for one post's extraction output. Post identity and all enriched author fields from scraper are at the top level alongside `frames`.

### Critical Field Notes

> **`source_quote` is non-negotiable.** The fuzzy-match filter drops frames where the quote can't be found in the post. Primary defense against hallucination. If no verbatim quote supports the claim, do not extract the frame.

> **`condition_value` vs `example_value`:** "For brevity, let's say 10 holes" → `example_value: "10 holes"`, not `condition_value`. The `EXAMPLE_MARKERS` list in `extract.py` automatically rescues misclassified values at filter time.

> **`thread_endorsement` is never LLM-filled.** It's derived programmatically from `is_op_followup` and `quoted_usernames` in `extract.py`.

> **`directness` and `expression_strength` are independent dimensions.** "I typically define my fields this way" → `hedged` + `moderate`. "I, by policy, never use them" → `explicit` + `strong`.

---

## 6. Prompt Logic & Extraction Rules

### Priority Target: Experiential Content

The knowledge graph is built on what machinists have **learned from experience** — not on definitions of what things are.

**High value — extract these:**
- "I never use X because..." (heuristic with rationale)
- "X will get you in trouble if..." (risk)
- "These are NOT standardized — you must check your manual" (observation with practical implication)
- "My method is X to prevent Y" (heuristic)

**Low value — skip or deprioritize:**
- Plain definitions: "X is a variable that holds Y" — encyclopedic, not experiential
- Tutorial scaffolding: "Next up on our list is..."
- Author's writing process: "I'm thinking this through as I go"

When a post contains both a definition AND the author's experience/opinion about it, **extract the experience, not the definition.** The definition is the setup; the experience is the knowledge.

### Key Type Distinctions

**heuristic vs observation** — the most common misclassification:

```
"Local variables reset to null on program end"
→ observation  (factual property, no personal action)

"I never use local variables — they reset to null on program end"
→ heuristic  (personal practice + the reason behind it)
```

The `heuristic` type covers: "I do X", "I never use X", "I always X", "my policy is X", "my method is X". The rationale is part of the frame — capture "I do X **because** Y" together in `main_point`.

**principle vs observation:**

```
"Work hardening increases surface hardness with each pass"
→ principle  (causal: X causes Y)

"6061 galls on threads without lubrication"
→ observation  (factual claim, no causal mechanism given)
```

**principle — DO NOT use for plain definitions:**

A definition describes what something *is*. A principle explains *why* something behaves as it does. "X is a variable that does Y" is not a principle — skip it or use `observation`. Only use `principle` when there is a causal explanation.

### Social Noise — Do Not Extract

These must never produce frames. They carry no transferable knowledge:

- Greetings, thanks, sign-offs
- Agreement without substance ("Yep, exactly right")
- Author narrating their writing process ("I'm writing this as I go", "I can't think of anything else right now")
- Requests for feedback ("Let me know if I missed anything")
- Statements that something is being used as an example ("I'm just using this as an example")
- Unanswered questions

If a post contains only social noise after quote-stripping, return `has_relevant_content: false`.

### Epistemic Metadata

Two dimensions, rated **independently**:

| Example language | directness | expression_strength |
|---|---|---|
| "I, by policy, never use them for anything" | explicit | strong |
| "I typically define my fields this way" | **hedged** | moderate |
| "I like to use pass-through variables" | **hedged** | moderate |
| "You will cause chatter if you do this" | explicit | strong |
| "This might work in some situations" | explicit | weak |

> **"I typically" is hedged, not explicit.** This was consistently misclassified in early runs.

### condition_value vs example_value

```
"Don't go deeper than 3x diameter"
→ condition_value: "depth > 3x diameter"   (real assertion)

"For brevity, let's limit to 10 holes"
→ example_value: "10 holes"                (illustrative only)
```

Toy values must NOT go into `condition_value`. The `EXAMPLE_MARKERS` filter in `extract.py` catches most cases automatically, but the LLM should classify correctly in the first place.

---

## 7. Known Issues

### 🔴 High — Hallucination on near-empty posts

`post-988671` had its quoted content stripped, leaving only "I'm just using this as an example of how to lay out a program." The model extracted a frame with `source_quote` matching that line but invented a `main_point` about "British controls" that doesn't exist anywhere in the post.

The fuzzy-match filter catches quotes that don't exist in the post, but cannot catch `main_point` content that isn't supported by the quote. A minimum-content threshold (skip post if non-quoted text < N characters) would prevent this.

### 🔴 High — Missing experiential layer (post-988538)

The model repeatedly returns dry definitional `principle` frames describing what each variable type IS, and misses the high-value content: the "NOT standardized" observation about system variables, and the "I never use local variables because they reset" heuristic with its rationale. These are the gold-standard frames for this post and have been elusive across multiple iterations.

The PRIORITY TARGET section in the prompt directly addresses this but hasn't been tested yet.

### 🟡 Medium — Social noise leakage (post-988556)

"That's all I can think of right now" and "Let me know if I missed anything" both extracted as frames in multiple runs despite the social noise rule in the prompt. The latest prompt adds these as explicit example strings rather than just abstract rules.

### 🟡 Medium — Design decision recall (post-988520)

The design decisions Tony walks through (tool-call choice + rationale, multiples choice, angle input) are not well-captured. These are "I decided X because Y" patterns in a tutorial context. They fit `heuristic` but the model tends to skip them as tutorial scaffolding. The heuristic definition now explicitly covers this pattern.

### 🟢 Low — post_role classification

Posts in a tutorial thread that are genuine tutorial content keep being classified as `general_advice`. Minor — doesn't affect knowledge content.

---

## 8. Roadmap

| # | Step | Status |
|---|---|---|
| 1 | Scraper — PracticalMachinist two-pass | ✅ Done |
| 2 | Schema alignment with scraper output | ✅ Done |
| 3 | Prompt iteration — extraction quality | 🔄 Active |
| 4 | Gold standard — 100–200 manually annotated frames | 📋 Next |
| 5 | Automated eval loop | 📋 Planned |
| 6 | Full scrape — CNC Machining category | 📋 Planned |
| 7 | Human review — 500–1000 frames minimum | 📋 Planned |
| 8 | Entity resolution | 📋 Planned |
| 9 | KG construction — frames → nodes/edges | 📋 Planned |
| 10 | DFM feedback proof of concept | 📋 Planned |

### Step 4 — Gold Standard

Manually annotate 100–200 frames across 20–30 diverse posts. This is the validation baseline required before the automated eval loop can produce meaningful metrics, and the foundation for the research methods section. Without it you're optimizing against vibes.

### Step 7 — Human Review Notes

Target: review 500–1000 frames minimum to produce a reportable quality metric. Contradictions found during review are **findings, not noise** — note what context might resolve them (different machine type, different material, different shop context). Use `validation_notes` for this.

### Step 9 — Frame → KG Mapping

How frame types map to graph components:

| Frame type | KG role |
|---|---|
| `risk` | Directed causal edge with condition |
| `heuristic` | Conditional rule node with JUSTIFIED_BY edge |
| `principle` | Property edge or mechanism node |
| `case` | Case node with RESOLVED_BY edge |
| `workaround` | Node with WORKAROUND_FOR edge |
| `recommendation` | Recommendation node with APPLIES_TO edge |
| `observation` | Node property or edge attribute |

### Step 10 — AI Connection Pattern

- **Pattern 1 (proof of concept):** Text-to-Cypher — LLM translates natural language to graph query
- **Pattern 2 (full system):** KG-Augmented RAG — graph traversal + vector search in parallel
- **Pattern 3 (future work):** KG as agent tool — dynamic multi-hop reasoning
- **Pattern 4 (Goal 3):** Fine-tuning on KG-derived data

---

## 9. Automated Eval Loop

Designed for when you want to iterate the extraction prompt automatically. Requires a gold standard.

### Structure

```
Run extraction (current prompt)
    → Score vs gold standard (recall + precision)
    → Classify failures by type
    → Stronger model revises prompt
    → Repeat (cap at 3–5 iterations)
```

### Scoring

```
score = 0.6 × recall + 0.4 × precision
```

Recall weighted higher because missing real frames is worse than occasional noise. Adjust as needed.

### Failure Categories

```
missed_frame     # gold frame not found in output — recall failure
noise_frame      # extracted frame with no knowledge value
wrong_type       # correct content, wrong frame_type
hallucination    # main_point not supported by source_quote
quote_mismatch   # source_quote doesn't appear in post text
```

### Gold Standard Comparison

String matching can't find "I never use local variables" as a match for a gold frame quoting "I, just by policy, never use them for anything." Use an LLM call (semantic match) to check if the same knowledge was captured.

### Model Split

- `gpt-4o-mini` for extraction — cheap, runs many times per iteration
- `gpt-4o` for the judge and prompt-updater — stronger, only runs once per iteration

### Manual Loop (current approach)

Until the gold standard exists:

1. Run `python extract.py` locally → `extractions_test.json`
2. Upload `extractions_test.json` to Claude (with `test_posts.json` if posts changed)
3. Claude reads both, identifies specific failures, proposes targeted prompt changes
4. Download updated `schema.py` + `extract.py`, replace, run again

> Keep `test_posts.json` stable between iterations. Changing input while changing code makes it impossible to attribute what caused any change.

---

## 10. Key Files

| File | Description |
|---|---|
| `scraper.py` | Two-pass forum scraper |
| `config.py` | Scraper config — selectors, limits, credibility tier thresholds |
| `schema.py` | Pydantic schema — `AuthorFrame`, `KnowledgeFrame`, `PostExtractionResult` |
| `extract.py` | Extraction pipeline — quote stripping, LLM call, filtering, consolidation |
| `posts_enriched.json` | Scraper output — posts + author data merged |
| `authors.json` | Author profiles keyed by username |
| `extractions_test.json` | Extraction output for test runs (≤5 posts) |
| `extractions.json` | Extraction output for full runs |
| `test_posts.json` | Fixed test set — do not change between prompt iterations |
