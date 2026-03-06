# Expert Frame Review Rubric
**Project:** AI-Assisted Structuring of Tribal Machining Knowledge for DFM
**Reviewer role:** Domain expert evaluating AI-extracted machining knowledge frames
**Estimated time:** ~30–45 minutes for 25 frames

---

## What is a "Frame"?

A frame is a discrete unit of practical machining knowledge extracted by an AI system from online machining forum posts. Each frame has two parts you are evaluating:

1. **The knowledge content itself** — is it an accurate extraction from the original post?
2. **The metadata labels** — does the AI correctly classify and describe the frame?

Each frame you review will show you:
- The **source quote** — the verbatim text from the original forum post
- The **main point** — the AI's summary of the knowledge claim
- Six **metadata fields** the AI assigned

---

## PART 1: Knowledge Content (1–3 scale)

### 1A. Extraction Accuracy
*Does the main point accurately and faithfully represent what the source quote actually says? Is anything added, distorted, or lost?*

| Score | Meaning |
|-------|---------|
| 3 | Accurate — the main point faithfully captures what the source quote says, nothing added or distorted |
| 2 | Mostly accurate — minor omission, slight overstatement, or small loss of nuance |
| 1 | Inaccurate — the main point misrepresents, contradicts, or significantly distorts the source quote |

---

### 1B. Captures Tribal / Experiential Knowledge
*Does this frame reflect the kind of practical knowledge that isn't in a standard textbook — something learned from experience rather than formal training?*

| Score | Meaning |
|-------|---------|
| 3 | Clearly experiential — unlikely to find this stated this way in a textbook |
| 2 | Partially experiential — some overlap with textbook knowledge but includes a practical nuance |
| 1 | Textbook knowledge — a standard rule or principle readily found in a reference |

---

### 1C. Clarity
*Is the main point clearly and specifically stated — could someone understand the knowledge claim without reading the original post?*

| Score | Meaning |
|-------|---------|
| 3 | Clear and specific — the claim is understandable and complete as written |
| 2 | Understandable but vague — the point comes through but lacks specificity |
| 1 | Unclear or ambiguous — hard to understand without the original post |

---

## PART 2: Metadata Accuracy (1–5 scale)

For each metadata field, rate whether the AI's assigned label is an accurate representation of the frame. A 5 means the label is the best possible choice; a 1 means it is clearly wrong.

---

### 2A. Frame Type
*The AI classified this frame as one of: risk, heuristic, principle, case, workaround, observation, comparison, or other.*

**Definitions for reference:**
- **risk** — a condition that leads to a failure mode or limitation ("X will cause problems if Y")
- **heuristic** — a personal rule, practice, or directed advice with rationale ("I do X because Y")
- **principle** — a causal explanation of WHY something happens (cause → effect)
- **observation** — a factual or empirical claim about how something is or behaves (no action, no mechanism)
- **comparison** — explicitly weighs two or more options against each other
- **workaround** — a non-standard fix for a stated limitation
- **case** — a narrative of a specific real-world job or incident

| Score | Meaning |
|-------|---------|
| 5 | Correct — this is clearly the best type label for this frame |
| 4 | Reasonable — a defensible choice, minor ambiguity |
| 3 | Acceptable — not wrong, but another type would fit better |
| 2 | Poor — a different type would be clearly more appropriate |
| 1 | Wrong — this label does not fit the frame |

---

### 2B. Post Role
*The AI classified the role of the original post in its thread as one of: question, solution, addition, general_advice, tutorial, war_story, debate, or reaction.*

| Score | Meaning |
|-------|---------|
| 5 | Correct — this clearly describes the post's role in context |
| 4 | Reasonable — a defensible choice |
| 3 | Acceptable — not wrong, but another role would fit better |
| 2 | Poor — a different role would be clearly more appropriate |
| 1 | Wrong — this label does not fit |

---

### 2C. Scope / Subject Domain
*The AI assigned a primary knowledge domain: design_geometry, machining_process, material, tooling, programming, shop_operations, quality, cost, machine_capability, or other.*

| Score | Meaning |
|-------|---------|
| 5 | Correct — clearly the right domain for this frame |
| 4 | Reasonable — defensible, minor ambiguity |
| 3 | Acceptable — not wrong, but another domain fits better |
| 2 | Poor — a different domain would be more appropriate |
| 1 | Wrong — this domain does not fit |

---

### 2D. Applicability
*The AI classified how broadly the claim applies as: universal, context_dependent, or situational.*

- **universal** — true regardless of equipment, material, or configuration
- **context_dependent** — true within specific contexts (material type, machine brand, operation)
- **situational** — specific to a particular shop setup or one-off scenario

| Score | Meaning |
|-------|---------|
| 5 | Correct — clearly the right applicability level |
| 4 | Reasonable — defensible |
| 3 | Acceptable — borderline case |
| 2 | Poor — a different level fits better |
| 1 | Wrong — clearly incorrect applicability level |

---

### 2E. Main Point Quality
*Is the main point a well-formed, standalone knowledge claim? Does it accurately state the core lesson from the source quote without adding or removing meaning?*

| Score | Meaning |
|-------|---------|
| 5 | Excellent — precise, complete, well-stated, fully supported by the source quote |
| 4 | Good — accurate with minor room for improvement |
| 3 | Acceptable — captures the claim but is vague, wordy, or misses a nuance |
| 2 | Poor — partially captures the claim but has notable gaps or distortions |
| 1 | Bad — inaccurate, misleading, or not supported by the source quote |

---

### 2F. Epistemic Stance
*The AI classified how the claim is expressed as: absolute_authoritative, direct_neutral, hedged_emphatic, tentative_subjective, or narrative_implied.*

- **absolute_authoritative** — stated as fact with strong language (must, always, never, cannot)
- **direct_neutral** — stated as fact, technical tone, no hedges or emphasis
- **hedged_emphatic** — personal hedges (in my experience) with strong internal language
- **tentative_subjective** — uncertain language (might, usually, I think, seems)
- **narrative_implied** — inferred from a story or described outcome, not directly stated

| Score | Meaning |
|-------|---------|
| 5 | Correct — clearly matches how the claim is expressed in the source quote |
| 4 | Reasonable — defensible |
| 3 | Acceptable — borderline |
| 2 | Poor — a different stance fits better |
| 1 | Wrong — does not match how the claim is expressed |

---

## Overall Judgment

After rating all dimensions, give an overall judgment:

| | Meaning |
|---|---------|
| ✅ Keep | High-quality frame, accurate extraction, metadata mostly correct |
| ⚠️ Revise | Frame has value but needs correction to the main point or metadata |
| ❌ Remove | Frame is a bad extraction, inaccurate, or not useful |

---

## Comments
*Please note anything relevant: corrections to the main point, what metadata label you'd assign instead, missing context, caveats, or anything else useful. Even a brief note helps.*

---

## Submission
Return your completed ratings to **[your email]** by **[deadline date]**.
Questions? Reach out anytime. Thank you — your input directly shapes the paper's evaluation section.
