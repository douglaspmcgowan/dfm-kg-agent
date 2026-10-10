# Ablation Findings for IDETC §4.4

Run date: 2026-05-05. Model: gpt-5 (default reasoning effort) for all
configurations. Source data: `benchmark_ablation_results.json`,
`benchmark_ablation_summary.md`. Total run cost: ~$1.70.

---

## Summary table (drop into paper as Table 16)

| Configuration            | Retrieval                                       | Synthesis context                      | Avg latency | Avg cost | Cited frames | Refusal on out-of-scope (n=2) |
| ------------------------ | ----------------------------------------------- | -------------------------------------- | ----------- | -------- | ------------ | ----------------------------- |
| **A. LLM-only baseline** | none                                            | gpt-5, no context                      | 42 s        | \$0.024  | 0.0          | **0/2**                       |
| **B. LLM + RAG**         | top-12 vector hits, no graph                    | retrieved frames in context            | 17 s        | \$0.013  | 8.4          | **0/2**                       |
| **C. Full pipeline**     | decompose → multi-vector → 2-hop graph → ranked | structured DFM report w/ gap detection | 196 s       | \$0.174  | 8.2          | **2/2**                       |

8 queries × 3 configs = 24 runs. Out-of-scope queries: sheet-metal bend
radius (Q7) and FDM overhang rules (Q8) — neither domain is present in
the corpus. Refusal is detected via explicit "no corpus evidence" /
"knowledge gap" phrasing in the answer.

---

## Refusal behavior by coverage tier

| Coverage               | A (LLM only) | B (LLM+RAG) | C (Full pipeline) |
| ---------------------- | ------------ | ----------- | ----------------- |
| Well-covered (n=3)     | 0/3          | 0/3         | 0/3               |
| Sparse (n=3)           | 0/3          | 0/3         | 1/3               |
| **Out-of-scope (n=2)** | **0/2**      | **0/2**     | **2/2**           |

C is the only configuration that distinguishes "I have no evidence" from
"I have an answer." A and B produce confident, fluent answers for
sheet-metal and FDM questions despite neither domain existing in the
corpus.

---

## Draft §4.4 paragraphs

```text
4.4 Ablation: Pipeline Component Contribution

To isolate the contribution of each pipeline component, we ran three
configurations against eight design questions chosen to span varying
corpus coverage: three well-covered (thin-walled aluminum, climb vs.
conventional milling, deep pockets), three sparse (chamfer cost
impact, shop-floor traceability, tolerance stack-up case), and two
intentionally out-of-scope (sheet-metal bend radius, FDM overhang
support rules — neither domain is present in our corpus). All three
configurations used GPT-5 at default reasoning effort.

Configuration A (LLM-only) issued the question to GPT-5 with no
retrieval. Configuration B (LLM+RAG) added vector retrieval over the
frames_knowledge collection but no graph traversal or query
decomposition. Configuration C (the full pipeline) used the agent
described in Section 4.3: query decomposition, multi-vector retrieval,
two-hop semantic-edge traversal, ranking, and structured synthesis
with explicit knowledge-gap detection. Results are summarized in
Table 16.

The configurations differ most clearly in their handling of
out-of-scope queries (Table 17). Configuration A produced 17–26
confident design rules per query regardless of whether the corpus
contained relevant evidence; on the out-of-scope queries, those rules
were drawn entirely from the model's parametric knowledge without any
grounding signal to the user. Configuration B improved grounding in
the well-covered cases (8–12 frames cited per query) but failed to
recognize when retrieved frames were not actually responsive: on Q7
and Q8 it cited 3 to 9 retrieved frames as if they answered the
question, even though the underlying frames concerned PCB fabrication
and CNC machining rather than sheet metal or additive manufacturing.
Configuration C correctly refused both out-of-scope queries (2/2),
using the explicit knowledge-gap channel to report that the corpus
contains no evidence on the subject; it also flagged Q5
(shop-floor traceability) as having insufficient corpus coverage.
Configuration C produced an average of 6.5 explicit knowledge-gap
statements per query, against zero for both A and B.

The cost of this disambiguation capability is substantial. Configuration
C is roughly 12 times more expensive per query than B and 7 times more
expensive than A (\$0.17 vs. \$0.013 and \$0.024 respectively), and
takes about three minutes per query versus 17–42 seconds. The graph
traversal, decomposition, and structured-report synthesis steps each
contribute to this overhead. The trade-off is a system that
distinguishes evidenced advice from absent evidence — a property
necessary for responsible DFM advisory use.

This ablation is illustrative rather than statistically conclusive
(n=8 queries). Future work should extend the comparison to larger
query sets and to additional retrieval baselines, including hybrid
sparse–dense retrievers and an oracle retrieval condition that uses
ground-truth relevance labels.
```

---

## Key qualitative excerpts (for paper appendix or supplemental)

### Q7 — sheet-metal bend radius (out-of-scope)

**Config A (LLM-only) — first 200 chars:**

> "For 16-gauge stainless steel (~1.5 mm, 304/316), in air bending the
> minimum bend radius typically lies between 1.0× and 1.5× the
> material thickness, with 1.5–2× preferred for cosmetic or critical
> structural bends..."

Confident answer. No indication that the corpus has nothing on sheet
metal forming.

**Config B (LLM+RAG) — first 200 chars:**

> "Based on the retrieved frames, here are the most relevant
> guidelines I can extract for bend-related design considerations on
> 16-gauge stainless steel in air bending..."

Cited 9 frames. The retrieved frames concerned PCB edge beveling and
CNC chip evacuation. Treated them as responsive.

**Config C (Full pipeline) — knowledge-gap output:**

> "Knowledge gaps: 1. No direct corpus evidence on sheet-metal air
> bending, 16-gauge stainless steel, or minimum bend radius in this
> KG. 2. No frames discussing K-factor, bend allowance, or springback
> for stainless steel. ... The corpus does not contain frames on this
> subject; the rules above are out-of-scope inferences..."

Explicit refusal via the knowledge_gaps mechanism.

### Q5 — shop-floor traceability (sparse)

This is the only sparse query where Config C declared insufficient
coverage. The corpus contains workflow material on CNC operations but
nothing specifically on traceability labeling. C produced 10 design
rules but flagged 8 explicit knowledge gaps, including "no direct
evidence on barcode or QR-code labeling on the shop floor" and "no
frames discussing ERP integration with shop-floor part identifiers."
The agent's hybrid behavior — produce rules where ambient knowledge
allows, flag gaps where corpus is silent — is what we expect from a
collaborator-pattern AI.

---

## Caveats and honest limitations

- **n=8 is small.** Treat the per-query numbers as illustrative.
  Do not back claims of "the system always refuses out-of-scope" off
  this size.
- **Refusal heuristic is text-pattern matching.** We detect refusal by
  searching for phrases like "no corpus evidence," "knowledge gap,"
  "outside the scope." A more rigorous future evaluation would parse
  the structured `knowledge_gaps` field and have a human rater confirm
  whether the gap report was actually coverage-correct.
- **All three configs use the same LLM (gpt-5).** The differences are
  in retrieval and synthesis structure, not in model capability.
- **Cost numbers are wall-clock estimates** based on the prices in
  Section 3.6 (gpt-5 at \$1.25 / \$10 per million tokens). Real prices
  shift; treat ±20%.
- **Config A's prompt explicitly says "Do not cite sources you cannot
  verify."** This is why A produces zero fabricated frame IDs. A more
  adversarial prompt (asking for citations) would produce fabricated
  frame IDs. The fabrication-rate column should be read as "with
  reasonable system-prompt hygiene," not "even under adversarial
  prompting."
