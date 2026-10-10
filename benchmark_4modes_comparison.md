# DFM Agent: Four Analysis Modes -- Benchmark Comparison

**Date:** 2026-03-13
**Queries tested:** 2 (thin-walled aluminum parts, climb vs conventional milling)
**Pipeline:** Qdrant vector retrieval -> NetworkX graph traversal -> GPT synthesis

---

## Mode Definitions

| Mode | Model | Reasoning | Pipeline Steps | Backfill |
|---|---|---|---|---|
| **Flash Mode** | gpt-4.1-mini | None | retrieve -> synthesize | No |
| **Quick Search** | gpt-4.1 | None (non-reasoning model) | decompose -> retrieve -> traverse -> rank -> synthesize -> backfill | Yes |
| **Efficient Thinking** | gpt-5 | reasoning_effort="low" | decompose -> retrieve -> traverse -> rank -> synthesize -> backfill | Yes |
| **Deep Reasoning** | gpt-5 | Default (medium) | decompose -> retrieve -> traverse -> rank -> synthesize -> backfill | Yes |

Flash Mode skips query decomposition, graph traversal, ranking, and backfill -- it retrieves frames and synthesizes directly.

---

## Quantitative Comparison

### Performance & Cost (averaged across 2 queries)

| Metric | Flash | Quick Search | Efficient Thinking | Deep Reasoning |
|---|---|---|---|---|
| **Total time (s)** | 42 | 89 | 202 | 297 |
| **Synthesis time (s)** | 36 | 62 | 116 | 128 |
| **Backfill time (s)** | -- | 16 | 61 | 135 |
| **Decompose time (s)** | -- | 3.4 | 18.6 | 27.2 |
| **Cost (USD)** | $0.006 | $0.050 | $0.098 | $0.163 |
| **Total tokens** | 7,844 | 11,952 | 16,115 | 22,428 |
| **Completion tokens** | 2,090 | 4,281 | 8,277 | 14,790 |
| **API calls** | 2 | 10 | 10 | 10 |

**Key takeaways:**
- Flash Mode is **7x cheaper** and **7x faster** than Deep Reasoning
- Quick Search is the best cost/quality tradeoff at ~$0.05 per query
- The jump from Efficient to Deep is modest in time (+47%) but large in token output (+79%)
- Backfill time scales dramatically with reasoning effort (16s -> 61s -> 135s)

### Report Richness (averaged across 2 queries)

| Metric | Flash | Quick Search | Efficient Thinking | Deep Reasoning |
|---|---|---|---|---|
| **Sub-aspects decomposed** | 0 | 6 | 6 | 6 |
| **Frames cited** | 11.5 | 19.5 | 24.5 | 23.5 |
| **Design rules** | 6 | 6.5 | 9 | 7 |
| **Disciplines covered** | 3 | 4.5 | 7 | 5.5 |
| **Reasoning chains** | 3.5 | 5 | 5.5 | 5 |
| **Evidence claims** | 11 | 17.5 | 22 | 18 |
| **Formulas** | 2 | 4 | 7 | 6 |
| **Knowledge gaps** | 3 | 4 | 5.5 | 7 |
| **Contradictions found** | 0 | 0 | 0 | 1.5 |
| **Backfill answers** | 0 | 4 | 5.5 | 7 |
| **Summary length (words)** | 85 | 99 | 49 | 93 |

**Key takeaways:**
- Flash cites ~50% fewer frames but still produces usable reports
- Efficient Thinking produces the *most* design rules, evidence, and formulas -- more than Deep Reasoning
- Deep Reasoning is uniquely good at finding **contradictions** (1.5 avg vs 0 for all others)
- Deep Reasoning identifies the most **knowledge gaps** (7 avg) and generates the most **backfill**
- Efficient Thinking's summary is paradoxically the shortest (49 words) -- more concise/dense

---

## Qualitative Comparison

### Query 1: "What should I consider when designing thin-walled aluminum parts for CNC machining?"

#### Flash Mode (38s, $0.006)
- **Disciplines:** design_geometry, machining_process, tooling (3)
- **Strengths:** Fast, actionable rules. Gets the key points (wall H/t ratio, drill before hollowing, simulate toolpaths)
- **Weaknesses:** No query decomposition -- treats it as a single monolithic question. No backfill for gaps. Rules are more generic ("start low and iterate")
- **Design rules highlight:** "Sequence drilling in solid stock before hollowing out interiors"
- **Summary quality:** Solid but surface-level. Lists considerations without deep causal reasoning

#### Quick Search (101s, $0.059)
- **Disciplines:** design_geometry, machining_process, material, tooling, programming, quality (6)
- **Strengths:** Best balance. Decomposes into 6 sub-aspects (deflection, thermal, residual stress, tool access, finish, fixturing). Provides specific ratios ("H/t below 10:1 for precision, 20:1 for general"). Covers thermal expansion at service temperature
- **Weaknesses:** No contradictions detected. Rules sometimes overlap
- **Design rules highlight:** "Keep wall height-to-thickness ratio below 10:1 for precision, or 20:1 for general use"
- **Summary quality:** Comprehensive, well-structured, 110 words. Good for practitioners

#### Efficient Thinking (223s, $0.096)
- **Disciplines:** design_geometry, machining_process, material, tooling, programming, quality (6)
- **Strengths:** Most design rules (10) and evidence claims (21). Provides mathematical formulas (deflection-based thickness: t = cubert(12PL^3/48Ed_max*w)). Covers sacrificial tabs, bonded fixturing. Highly technical
- **Weaknesses:** Summary is very terse (47 words). Some rules feel like padding. No contradictions
- **Design rules highlight:** "Apply the deflection-based thickness relation t = cubert(12 P L^3 / (48 E d_max w)) on drawings as a design check"
- **Summary quality:** Dense and expert-level -- assumes reader has machining background

#### Deep Reasoning (297s, $0.19)
- **Disciplines:** machining_process, tooling, material, design_geometry, programming, quality (6)
- **Strengths:** Finds **2 contradictions**. Most knowledge gaps (8) with detailed backfill. Best at identifying what the KG *doesn't* know. Reasoning chains are more causal. Covers unique topics (built-up edge, chip evacuation paths, stress relief)
- **Weaknesses:** Fewer design rules (8) than Efficient. Fewer evidence claims. Most expensive and slowest
- **Design rules highlight:** "When designing thin internal cavities/slots, provide chip-evacuation paths or reliefs because chip packing drives heat and instability"
- **Summary quality:** Practitioner-friendly, 84 words. Good balance of depth and accessibility

### Query 2: "How do I choose between climb and conventional milling?"

#### Flash Mode (47s, $0.005)
- **Disciplines:** machining_process, machine_capability, tooling (3)
- **Strengths:** Gets the fundamental tradeoff right (climb = better finish but needs stiff machine). Fast answer
- **Weaknesses:** Only 4 rules, 10 frames cited. Misses chip formation, tool wear, material-specific guidance
- **Design rules highlight:** "Use climb milling for roughing passes and conventional milling for final profile cuts"

#### Quick Search (76s, $0.040)
- **Disciplines:** machining_process, tooling, material (3)
- **Strengths:** Decomposes into 6 sub-aspects. Better material coverage (FR-4, corrosion-prone). Specific on coolant concerns ("avoid water-based coolants for corrosion-prone materials")
- **Weaknesses:** Some tangential rules (coolant choice). 5 rules vs 4 for Flash -- not a huge jump
- **Design rules highlight:** "For high-wear or abrasive materials like FR-4, select tool geometries and materials suited for rapid wear"

#### Efficient Thinking (180s, $0.101)
- **Disciplines:** machine_capability, machining_process, quality, tooling, programming, design_geometry, material, vibration/stability (8!)
- **Strengths:** By far the broadest coverage. 8 disciplines (created a custom "vibration/stability" discipline). 25 frames cited, 9 formulas, 23 evidence claims. Most thorough technical analysis. Includes deburring, edge-break specifications, DRC enforcement
- **Weaknesses:** Breadth may be excessive for a simple climb/conventional question. Some rules are general DFM rather than climb/conventional-specific
- **Design rules highlight:** "For thin spans and walls, size thickness by a deflection criterion (not just strength)"

#### Deep Reasoning (296s, $0.139)
- **Disciplines:** machining_process, machine_capability, tooling, material, quality (5)
- **Strengths:** Finds **1 contradiction**. Sub-aspects include unique angle: "hard scale or abrasive skin removal." Most nuanced summary (102 words). Practical decision framework ("use climb on backlash-free rigid setups... favor conventional when backlash or structural flexibility could let the cutter pull into the work")
- **Weaknesses:** Fewer evidence claims than Efficient (20 vs 23). Fewer formulas (5 vs 9)
- **Design rules highlight:** "On drawings for features made on manual or backlash-prone equipment, dimension from a consistent primary datum"

---

## Summary: When to Use Each Mode

| Mode | Best For | Avoid When |
|---|---|---|
| **Flash** | Quick sanity checks, simple questions, demos, cost-sensitive usage | Complex multi-faceted questions, need for completeness |
| **Quick Search** | Daily engineering use, good depth at reasonable cost | Budget is unlimited, or need to find contradictions |
| **Efficient Thinking** | Maximum breadth and technical detail, formula-heavy analysis | Time-constrained, or question is simple enough for Quick Search |
| **Deep Reasoning** | Finding contradictions, identifying gaps, nuanced decision frameworks | Cost-sensitive, simple questions, when speed matters |

### Cost-Effectiveness Ranking (quality per dollar)
1. **Quick Search** -- best value. 3x the depth of Flash for 8x the cost, but still under $0.06
2. **Efficient Thinking** -- most raw output per dollar, but diminishing returns on quality
3. **Flash Mode** -- unbeatable for speed/cost, sufficient for simple questions
4. **Deep Reasoning** -- only justified when contradiction detection or exhaustive gap analysis matters

### Unique Capabilities by Mode
- **Flash only:** No decomposition -- answers as a single coherent thought (sometimes better for simple questions)
- **Quick Search only:** Best summary quality -- not too short, not too long
- **Efficient Thinking only:** Creates custom discipline categories beyond the standard set. Most formulas
- **Deep Reasoning only:** Reliably finds contradictions in the evidence base. Most thorough gap analysis
