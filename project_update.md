# Project Update: AI-Structured Machining Knowledge for DFM
**Last updated:** February 2026 | **Submitting to:** ASME IDETC/CIE 2026 (August 23–26, Anaheim, CA)

---

## WHERE I AM RIGHT NOW

### ✅ Completed / In Progress
- **Core concept fully defined:** scrape forums → extract knowledge frames via LLM-as-judge → embed in three embedding spaces → build knowledge graph → enable RAG + agentic reasoning
- **Framing is solid:** two distinct problems addressed (DFM redesign loops + tribal knowledge transfer), both motivated well
- **"Frame" as the core unit of knowledge is defined** conceptually — discrete machining knowledge kernels extracted from forum posts
- **Three embedding space types identified** (exact types need to be documented here as work progresses)
- **Knowledge graph structure designed** with semantic connections between frames
- **Contribution narrative is clear:** process + database + AI-as-collaborator paradigm

### 🔄 In Progress / Partially Done
- Forum scraping — underway, scope and forums TBD/confirm
- LLM-as-judge extraction pipeline — built, running
- Frame extraction — thousands of frames extracted (exact N TBD)
- Embedding spaces — built, specifics need documentation
- Knowledge graph — constructed, details need documentation

---

## WHAT NEEDS TO BE DONE

### Methods (to write/finalize)
- [ ] Document exactly which forums were scraped and why
- [ ] Write up the frame schema in full — what fields does each frame contain?
- [ ] Formally define what the three embedding space types are and the rationale for each
- [ ] Document the knowledge graph schema: node types, edge types, how graph was built
- [ ] Write up the LLM-as-judge workflow in enough detail that it's reproducible
- [ ] Create a **pipeline/system architecture figure** — this is essential for IDETC reviewers

### Results (largely missing — highest priority gap)
- [ ] Report final dataset statistics: number of posts scraped, frames extracted, graph size
- [ ] Run and document **extraction quality evaluation** — need a human eval protocol or automated metric
- [ ] Run **RAG system evaluation** — compare responses to a baseline (vanilla LLM with no knowledge base)
- [ ] Consider a **design task evaluation** — give a designer a DFM task and compare with/without system
- [ ] Pull 2–3 strong example frames to include in the paper as illustrations

### Writing
- [ ] Draft Introduction (motivation + research question)
- [ ] Build out Related Work section — needs actual citations (see gaps below)
- [ ] Draft Methods in full
- [ ] Draft Results once evaluation is done
- [ ] Draft Discussion and Conclusion
- [ ] Write Abstract last

### Logistics
- [ ] Confirm IDETC 2026 submission deadline
- [ ] Choose the right IDETC track/conference (DAC? CIE? — depends on framing)
- [ ] Get paper number assigned after abstract submission

---

## REMAINING GAPS & UNANSWERED QUESTIONS

### Conceptual / Framing
- **Scope of machining:** Does this cover CNC only? Manual machining? Mill/lathe/both/all? This needs to be stated explicitly — reviewers will ask
- **What makes a "frame" valid?** Need a clear quality criterion — what gets included vs. filtered out?
- **AI-as-collaborator paradigm:** This is a strong framing but needs grounding in existing literature — who else has used this framing and how does this paper extend it?
- **Comparison to textbook knowledge:** You mention frames can be compared to traditional textbook knowledge — has this comparison been done? What does it show?

### Methods
- **Three embedding space types:** What are they exactly? What is the justification for choosing those three? What does each one enable that the others don't?
- **LLM-as-judge validation:** How do you know the judge is doing a good job? What's the error rate? Is there a human-in-the-loop check?
- **Knowledge graph edge types:** What defines a "semantic connection" between frames? Is this cosine similarity above a threshold? An LLM-determined relation? Something else?
- **Agentic reasoning over the graph:** How does the agent decide when to use RAG vs. graph traversal? Is there a routing mechanism?
- **Forum terms of service / data ethics:** Scraping forums raises questions reviewers may ask — is this covered?

### Results / Evaluation
- **How will you evaluate the system?** This is the biggest open question. Options:
  - Human raters evaluating frame quality (precision/recall of knowledge extraction)
  - LLM-based evaluation of RAG response quality vs. baseline
  - A controlled design task study (most compelling but hardest)
  - Expert machinist review of a sample of frames
- **What's the baseline?** Vanilla GPT-4 / Claude with no knowledge base? A keyword search over raw forum posts?
- **How do you show the knowledge graph adds value beyond flat RAG?** Need an ablation or comparison

### Literature / Citations Needed
- **DFM computational tools** — foundational papers + recent work; who are the main groups?
- **Tacit knowledge in engineering** — Nonaka is the anchor; what else exists specifically in manufacturing?
- **LLM-as-judge** — cite the original methodology paper
- **RAG in engineering** — what recent papers exist applying RAG to technical/design domains?
- **LLMs in engineering design** — growing IDETC body of work (Ahmed, Cagan, and others); need a thorough lit search here
- **Knowledge graphs in manufacturing** — who has built these before and for what purpose?
- **AI-as-collaborator / second brain** — HCI and design cognition literature; cognitive load theory may be relevant here
- **Forum/community knowledge mining** — is there precedent for extracting structured knowledge from online communities in other domains?

---

## NOTES FOR FUTURE VOICE MEMOS
*Send voice updates on any of the following and this doc will be updated:*
- Progress on frame extraction (how many frames, which forums)
- Details on the three embedding space types
- Knowledge graph schema decisions
- Evaluation approach decisions
- Any new related work you come across
- Methods clarifications (frame schema, LLM-as-judge workflow details)
