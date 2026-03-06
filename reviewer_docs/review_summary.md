# Human & LLM Review: Summary and Analysis

## Study Design

**Objective:** Evaluate the quality of AI-extracted knowledge frames from engineering Stack Exchange posts, using both human domain experts and an LLM-as-judge (GPT-5).

**Population:** 3,879 extracted frames from 1,936 posts across 911 threads (5 SE sites).

**Sample:** 26 frames from 10 posts, drawn via stratified post-based sampling (proportional to frame_type distribution in corpus). Seed-controlled for reproducibility.

**Reviewers:**
- 4 human reviewers (domain experts) -- same 26 frames, same order
- 1 LLM reviewer (GPT-5) -- same 26 frames, structured output via Pydantic schema

---

## Evaluation Rubric

### Part 1: Knowledge Content (1-3 scale)

| Dimension | 1 | 2 | 3 |
|---|---|---|---|
| **Extraction Accuracy** | Misrepresents the source quote | Minor omission/overstatement | Accurately captures the quote |
| **Tribal Knowledge** | Standard textbook knowledge | Moderately specialized | Genuine experiential knowledge |
| **Clarity** | Ambiguous or hard to understand | Understandable but vague | Clear, precise, actionable |

### Part 2: Metadata Accuracy (1-5 scale)

| Dimension | What it rates |
|---|---|
| **Frame Type** | risk, heuristic, principle, observation, comparison, workaround, case, other |
| **Post Role** | question, solution, addition, general_advice, tutorial, war_story, debate, reaction |
| **Scope** | design_geometry, machining_process, material, tooling, programming, shop_operations, quality, cost, machine_capability, other |
| **Applicability** | universal, context_dependent, situational |
| **Main Point Quality** | Standalone knowledge claim quality (1-5) |
| **Epistemic Stance** | absolute_authoritative, direct_neutral, hedged_emphatic, tentative_subjective, narrative_implied |

### Overall Judgment
- **Keep** -- accurate, useful, correctly classified
- **Revise** -- captures real knowledge but needs label/main point corrections
- **Remove** -- wrong, trivial, or not meaningful

---

## Frame Sample Composition

| Post Group | Thread Topic | Post Score | Frames | Domain |
|---|---|---|---|---|
| 1 | PCB rounded/fillet edges | 10 | 3 | Electronics / PCB fabrication |
| 2 | Best material for project | 2 | 3 | Manufacturing / materials |
| 3 | 3D tube bending process | 1 | 5 | Mechanical / bending |
| 4 | EDM electrode material | 1 | 3 | Machining / EDM |
| 5 | CNC coolant automation | 0 | 6 | Shop operations / coolant |
| 6 | USB disconnects (CNC) | 0 | 2 | Electronics / CNC control |
| 7 | Spur gear meshing | 0 | 4 | Mechanical / gears |
| **Total** | | | **26** | |

**Frame type distribution in sample:**
- observation: 12 (46%)
- heuristic: 6 (23%)
- principle: 3 (12%)
- risk: 3 (12%)
- workaround: 1 (4%)
- case: 1 (4%)

---

## GPT-5 LLM Review Results (n=26 frames)

### Overall Judgments

| Judgment | Count | % |
|---|---|---|
| Keep | 12 | 46% |
| Revise | 14 | 54% |
| Remove | 0 | 0% |

### Knowledge Content Ratings (1-3 scale)

| Metric | Mean | Median | SD | 1 | 2 | 3 |
|---|---|---|---|---|---|---|
| Extraction Accuracy | 2.50 | 2.5 | 0.51 | 0 (0%) | 13 (50%) | 13 (50%) |
| Tribal Knowledge | 1.73 | 2 | 0.53 | 8 (31%) | 17 (65%) | 1 (4%) |
| Clarity | 2.92 | 3 | 0.27 | 0 (0%) | 2 (8%) | 24 (92%) |

### Metadata Accuracy Ratings (1-5 scale)

| Metric | Mean | Median | SD | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| Frame Type | 4.38 | 5 | 1.13 | 0 | 4 | 1 | 2 | 19 |
| Post Role | 4.88 | 5 | 0.33 | 0 | 0 | 0 | 3 | 23 |
| Scope | 3.62 | 4 | 1.30 | 1 | 6 | 4 | 6 | 9 |
| Applicability | 4.15 | 5 | 1.22 | 1 | 3 | 2 | 5 | 15 |
| Main Point Quality | 4.31 | 4 | 0.68 | 0 | 0 | 3 | 12 | 11 |
| Epistemic Stance | 4.77 | 5 | 0.51 | 0 | 0 | 1 | 4 | 21 |

### Key Findings from LLM Review

1. **No frames removed** -- the LLM judged all 26 as containing real knowledge, with ~half needing revisions.

2. **Strongest metadata:** Post Role (4.88) and Epistemic Stance (4.77) are the most accurately classified fields. The extraction system reliably identifies how posts function in discussions and the certainty level of claims.

3. **Weakest metadata:** Scope (3.62) is the most commonly criticized field. The LLM flagged scope labels as too narrow or misaligned in 11/26 frames (42%).

4. **Extraction accuracy is solid:** 50% rated 3 (perfect), 50% rated 2 (minor issues). Common issues: main point adds unstated rationale or slightly overstates the source quote.

5. **Low tribal knowledge scores:** Only 1 frame (4%) rated as genuine tribal knowledge. 65% rated as moderately specialized. The sample includes several textbook-level items (beam theory, gear definitions), pulling the average down.

6. **Clarity is excellent:** 92% of main points rated as clear and actionable.

7. **Applicability miscalibration:** 4 frames had applicability labeled as "universal" that the LLM judged should be "context_dependent" -- the system tends to over-generalize.

---

## Common LLM Reviewer Comments (Themes)

| Theme | Frequency | Examples |
|---|---|---|
| **Scope label too narrow/wrong** | 11 (42%) | "material" for bending mechanics; "quality" for coolant maintenance |
| **Main point adds unstated info** | 9 (35%) | Adds rationale, context, or specifics not in source quote |
| **Frame type should differ** | 5 (19%) | observation -> heuristic; principle -> observation; risk -> observation |
| **Applicability should be context_dependent** | 4 (15%) | Universal claims that depend on material, setup, or context |
| **Textbook, not tribal** | 5 (19%) | Beam theory, gear definitions, basic EDM facts |

---

## How to Discuss in Paper

### Methods Section

> Expert evaluation followed a structured rubric with two tiers: (1) knowledge content quality on a 3-point scale (extraction accuracy, tribal knowledge value, clarity) and (2) metadata classification accuracy on a 5-point scale across six dimensions (frame type, post role, scope, applicability, main point quality, epistemic stance). Each frame also received an overall judgment of Keep, Revise, or Remove.
>
> A stratified sample of 26 frames from 10 posts was drawn proportionally across frame types from the full corpus of 3,879 frames. Four domain experts and one LLM evaluator (GPT-5) independently reviewed the same sample using identical rubrics. The LLM reviewer used structured output parsing (Pydantic schema) to ensure rating format compliance.

### Results Section

> The LLM evaluator judged 46% of frames as ready to keep without changes, 54% as needing revision (primarily scope labels and minor main point adjustments), and 0% as warranting removal. Extraction accuracy was rated highly (M=2.50/3.00), with clarity near-perfect (M=2.92/3.00). Post role (M=4.88/5.00) and epistemic stance (M=4.77/5.00) were the most reliably classified metadata fields, while scope (M=3.62/5.00) emerged as the weakest, with the LLM noting that the fixed scope taxonomy often fails to capture cross-domain or nuanced categorizations.
>
> Tribal knowledge scores were modest (M=1.73/3.00), reflecting that the stratified sample included several textbook-level items (beam theory, gear definitions) alongside genuinely experiential claims. This finding suggests that while the extraction pipeline captures knowledge broadly, it does not preferentially select for tribal content -- a filter or weighting step may be needed downstream.

---

## Human Review Results (3 reviewers x 26 frames)

Reviewers: Reviewer 1, Reviewer 2, Reviewer 4 (Reviewer 3 not returned).

### Human Reviewer Mean Ratings

| Metric | Human Mean | LLM Mean | Diff (LLM - Human) |
|---|---|---|---|
| Extraction Accuracy (1-3) | 2.92 | 2.50 | -0.42 |
| Tribal Knowledge (1-3) | 2.04 | 1.73 | -0.31 |
| Clarity (1-3) | 2.91 | 2.92 | +0.01 |
| Frame Type (1-5) | 4.27 | 4.38 | +0.12 |
| Post Role (1-5) | 4.50 | 4.88 | +0.39 |
| Scope (1-5) | 4.47 | 3.62 | -0.86 |
| Applicability (1-5) | 4.35 | 4.15 | -0.19 |
| Main Point Quality (1-5) | 4.79 | 4.31 | -0.49 |
| Epistemic Stance (1-5) | 4.53 | 4.77 | +0.24 |

---

## Inter-Rater Agreement (Krippendorff's alpha, 4 raters)

| Dimension | Alpha | Interpretation |
|---|---|---|
| Tribal Knowledge | 0.395 | Fair |
| Clarity | 0.322 | Fair |
| Epistemic Stance | 0.243 | Fair |
| Main Point Quality | 0.166 | Slight |
| Scope | 0.151 | Slight |
| Extraction Accuracy | 0.010 | None (ceiling effect) |
| Frame Type | -0.002 | None |
| Applicability | -0.003 | None |
| Post Role | -0.068 | None |
| Overall Judgment | -0.040 | None |

**Interpretation (Krippendorff's guidelines):** alpha > 0.667 = reliable for conclusions; 0.667-0.80 = tentative conclusions; alpha < 0.667 = not reliable. No dimension reaches the reliability threshold, which is consistent with the inherently subjective nature of knowledge frame evaluation and the relatively small sample.

### Best Pairwise Human-Human Agreement

| Pair | Dimension | Kappa | % Agreement |
|---|---|---|---|
| R2 vs R4 | Extraction Accuracy | 0.648 | 96% |
| R1 vs R4 | Tribal Knowledge | 0.570 | 65% |
| R2 vs R4 | Clarity | 0.480 | 92% |
| R2 vs R4 | Main Point Quality | 0.328 | 77% |
| R2 vs R4 | Scope | 0.311 | 42% |

Reviewers 2 and 4 show the strongest human-human agreement across most dimensions.

---

## Human vs. LLM Comparison

### Key Patterns

1. **LLM is harsher on Scope (-0.86):** The largest systematic difference. The LLM frequently flags scope labels as too narrow or misaligned, while human reviewers tend to accept them.

2. **LLM is harsher on Main Point Quality (-0.49) and Extraction Accuracy (-0.42):** The LLM more often notes when the main point adds unstated information or slightly overstates the source quote. Humans rate extraction quality as near-perfect (2.92/3), while the LLM gives 2.50/3.

3. **LLM is more generous on Post Role (+0.39):** The LLM almost universally accepts the system's post role labels (4.88/5), while human reviewers are more critical (4.50/5).

4. **LLM undervalues tribal knowledge (-0.31):** Human reviewers are more willing to classify knowledge as experiential (2.04/3 vs 1.73/3).

5. **Clarity is consensus:** Both humans (2.91/3) and LLM (2.92/3) agree the main points are clearly written.

### LLM-as-Judge Suitability

The LLM reviewer demonstrates:
- **Strengths:** More detailed comments with specific suggestions; consistent application of rubric criteria; near-zero variance in clarity assessment
- **Weaknesses:** Systematically harsher on scope labels; underweights tribal knowledge value; somewhat lenient on post role classification
- **Calibration gap:** The LLM produces longer, more analytical comments but differs from human consensus by ~0.5 points on average across content dimensions

---

## How to Discuss in Paper

### Methods Section

> Expert evaluation followed a structured rubric with two tiers: (1) knowledge content quality on a 3-point scale (extraction accuracy, tribal knowledge value, clarity) and (2) metadata classification accuracy on a 5-point scale across six dimensions (frame type, post role, scope, applicability, main point quality, epistemic stance). Each frame also received an overall judgment of Keep, Revise, or Remove.
>
> A stratified sample of 26 frames from 10 posts was drawn proportionally across frame types from the full corpus of 3,879 frames. Three domain experts and one LLM evaluator (GPT-5) independently reviewed the same sample using identical rubrics. The LLM reviewer used structured output parsing (Pydantic schema) to ensure rating format compliance. Inter-rater reliability was assessed using Krippendorff's alpha for ordinal data across all four raters, with pairwise Cohen's weighted kappa computed for each rater pair.

### Results Section

> Three human reviewers and one LLM evaluator (GPT-5) independently rated 26 frames on nine dimensions. Inter-rater agreement, measured by Krippendorff's alpha across all four raters, was highest for tribal knowledge (alpha=0.395), clarity (alpha=0.322), and epistemic stance (alpha=0.243), indicating fair agreement on these subjective dimensions. Agreement was near zero for extraction accuracy (alpha=0.010), driven by a ceiling effect (humans rated 92% of frames as perfectly accurate). Metadata classification dimensions showed slight-to-no agreement: scope (alpha=0.151), main point quality (alpha=0.166), frame type (alpha=-0.002), applicability (alpha=-0.003), and post role (alpha=-0.068).
>
> Comparing the LLM evaluator to human consensus, the LLM was systematically harsher on scope labels (3.62 vs 4.47, diff=-0.86), main point quality (4.31 vs 4.79, diff=-0.49), and extraction accuracy (2.50 vs 2.92, diff=-0.42), while being more lenient on post role (4.88 vs 4.50, diff=+0.39). Clarity was the sole dimension with near-perfect human-LLM alignment (2.92 vs 2.91). The LLM judged 46% of frames as Keep and 54% as Revise (0% Remove), with revision recommendations primarily targeting scope labels and main point overstatement.
>
> The low inter-rater agreement is consistent with findings in annotation studies of complex, multi-attribute evaluation tasks, where subjective judgment inherently limits concordance. Notably, the pairwise agreement between Reviewers 2 and 4 was substantially higher (kappa=0.648 for extraction accuracy, 96% agreement) than other pairs, suggesting reviewer calibration differences rather than rubric ambiguity as the primary source of disagreement.

---

## Batch LLM Review (100 frames)

A separate batch of 100 randomly sampled frames was reviewed by GPT-5 for broader quality assessment.

| Metric | 26-frame sample | 100-frame batch |
|---|---|---|
| Keep | 46% | 51% |
| Revise | 54% | 49% |
| Remove | 0% | 0% |
| Extraction Accuracy | 2.50 | 2.60 |
| Tribal Knowledge | 1.73 | 1.72 |
| Clarity | 2.92 | 2.99 |
| Frame Type | 4.38 | 4.67 |
| Post Role | 4.88 | 4.91 |
| Scope | 3.62 | 3.90 |
| Main Point Quality | 4.31 | 4.38 |
| Epistemic Stance | 4.77 | 4.64 |

The 100-frame batch shows consistent patterns with the 26-frame sample:
- Scope remains the weakest-classified field (3.90/5)
- Post Role remains the strongest (4.91/5)
- No frames judged as Remove -- the pipeline consistently produces usable extractions
- Tribal knowledge remains modest (1.72/3) across the corpus
