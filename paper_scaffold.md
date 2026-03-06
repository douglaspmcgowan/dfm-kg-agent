# Paper Scaffold: AI-Assisted Structuring of Tribal Machining Knowledge for Design for Manufacturability

**Target venue:** ASME IDETC/CIE 2026 | **Track:** DTM-06 (Rethinking Design in the Age of AI) | **Target length:** 8–10 pages (two-column)

---

## ABSTRACT

_(200–400 words — write last)_

- State the problem: DFM workflows are inefficient; practical machining knowledge is undocumented, tribal, and at risk of being lost as the workforce ages
- State the approach: scrape forum posts → extract knowledge frames via LLM-as-judge → embed in three embedding spaces → build knowledge graph → enable RAG + agentic reasoning
- State the contributions: a replicable extraction process, a growing practical knowledge database, and a new AI-as-collaborator design paradigm
- Key results: [TBD — number of frames extracted, evaluation metrics]

---

## 1. INTRODUCTION

_(~0.75–1 page)_

### 1.1 Machining as a Critical Manufacturing Process

- Machining is a foundational and widely used manufacturing process across industries including defense, transportation, and healthcare
- [**FIND SOURCES:** statistics on machining prevalence/industry usage — defense, transportation, healthcare]
- To produce a machinable part, designers must account for machinability constraints that make the machining process both physically possible and feasible in cost and timeline
- This process is broadly called design for manufacturing (DFM); in this paper it is specifically referred to as **design for machining**

### 1.2 The Cost of Late-Stage Design Rework

- Late-stage design rebuilds and reworks due to machinability issues are costly — both to project budgets and timelines
- Early-stage design decisions (where part topologies and system architectures are set) have an outsized influence on total product cost
- [**FIND SOURCES:** early design stage influence on total cost — the classic "80% of cost determined in early design" statistic; late-stage rework cost impacts]
- Therefore, improving design for machinability at the early stage is a high-leverage intervention

### 1.3 Limitations of Traditional DFM Methods

- Traditional DFM methods focus on general geometric heuristics and rule-based assessment of parts
- Outside of these, manufacturability feedback primarily comes from manufacturers or manufacturing engineers late in the design process — not at the early stage where it is most impactful
- [**FIND SOURCES:** DFM is not normally considered in early design stage; machinability issues discovered late in production; cost impact of not considering manufacturability early]
- Manufacturability constraints are frequently not fully considered before a part is sent to production, leading to costly last-minute design changes

### 1.4 The Tribal Knowledge Gap

- Traditional DFM methods are further insufficient because much manufacturability feedback is not based on universal geometric constraints but on **experiential, undocumented knowledge** — referred to here as **tribal knowledge**
- Tribal knowledge includes: specific machine and shop constraints, practical lessons learned over time, and heuristics that machinists did not learn from a textbook but accumulated through experience
- [**FIND SOURCES:** tribal knowledge in manufacturing and engineering; tacit knowledge literature]
- This tribal knowledge is a large factor in machinist feedback but is esoteric and inaccessible to designers, making it difficult to incorporate into early-stage design decisions

### 1.5 Research Question and Approach

- **Research question:** How can we take machining tribal knowledge and structure it in a way that it can be used for future AI-assisted design interventions?
- This paper addresses this question by: scraping online forum posts, using an LLM-as-judge workflow to extract discrete knowledge units ("frames"), embedding them in a structured knowledge space, and building a knowledge graph to enable agentic RAG-based design assistance
- This work also demonstrates a new paradigm for AI in design: **AI as collaborator, not generator** — the designer retains creative and generative agency while the AI provides feedback, surfaces relevant knowledge, and expands the designer's cognitive bandwidth

### 1.6 Paper Roadmap

- Section 2 reviews related work; Section 3 details methodology; Section 4 presents results and evaluation; Section 5 discusses implications; Section 6 concludes

---

## 2. RELATED WORK

_(~0.75–1 page)_

### 2.1 Design for Manufacturability (DFM)

- Rule-based DFM approaches and their limitations
- Computational DFM tools
- Gap: formal rules don't capture practical/tribal knowledge
- [**FIND SOURCES:** Sriram, Bracewell, MIT/Stanford DFM literature; recent computational DFM tools]

### 2.2 Tacit and Tribal Knowledge in Engineering

- Nonaka's knowledge creation theory (tacit vs. explicit knowledge)
- Tacit knowledge in manufacturing contexts
- Challenges of knowledge transfer as workforce ages
- [**FIND SOURCES:** Nonaka; tacit knowledge in manufacturing; aging workforce / knowledge transfer literature]

### 2.3 Knowledge Representation and Knowledge Graphs

- Knowledge graphs in manufacturing and engineering design
- Ontologies for machining/manufacturing domains
- [**FIND SOURCES:** knowledge graphs in manufacturing; manufacturing ontologies]

### 2.4 LLMs for Knowledge Extraction and Engineering Design

- LLM-based information extraction from unstructured text
- LLMs in engineering design contexts
- LLM-as-judge workflows for automated evaluation and extraction
- [**FIND SOURCES:** growing IDETC/JMD body of work — Ahmed, Cagan, and others; LLM-as-judge original methodology paper]

### 2.5 Retrieval Augmented Generation (RAG) in Technical Domains

- RAG fundamentals and architecture
- RAG applied to engineering and technical knowledge domains
- [**FIND SOURCES:** RAG foundational paper (Lewis et al.); RAG in engineering/technical contexts]

### 2.6 AI as Collaborator in Design

- AI as generator vs. AI as collaborator / "second brain"
- Cognitive bandwidth augmentation and cognitive load theory in design
- HCI and design cognition literature on AI-assisted design
- [**FIND SOURCES:** cognitive load theory; AI-assisted design HCI literature; design cognition and AI]

---

## 3. METHODOLOGY

_(~2.5–3 pages)_

### 3.1 Overview and System Architecture

- High-level pipeline: data collection → knowledge extraction → knowledge representation → agentic interface
- [**FIGURE NEEDED:** system architecture / pipeline diagram — essential for IDETC reviewers]

### 3.2 Data Collection

- Source: online machining forum posts [**SPECIFY WHICH FORUMS:** Practical Machinist, MachinistForum, Reddit r/machining, etc.]
- Scraping approach and scope
- Data cleaning and preprocessing
- Dataset statistics: number of posts, threads, time range, topics covered

### 3.3 Knowledge Extraction: Frame Generation

- Definition of a **"frame"** — a discrete machining knowledge kernel extracted from forum discourse
- Frame schema: fields contained in each frame [**DEFINE EXPLICITLY — e.g., knowledge claim, context, constraints, source post, confidence, topic tags**]
- LLM-as-judge workflow: how frames are extracted and validated from raw post text
- Quality control: how bad or low-confidence extractions are filtered
- Output: N frames extracted from M posts [**FILL IN**]

### 3.4 Knowledge Representation

#### 3.4.1 Embedding Spaces

- Three embedding spaces, each structured by a different data type [**SPECIFY THE THREE TYPES AND RATIONALE**]
- Why three spaces? What does each capture that the others don't?
- Embedding model(s) used

#### 3.4.2 Knowledge Graph

- Node types: frames, authors, posts, topics [**CONFIRM SCHEMA**]
- Edge types: semantic similarity, discourse relations, authorship, post membership
- Graph construction process
- Graph statistics: nodes, edges, connectivity [**FILL IN**]

### 3.5 Agentic Interface Design

- RAG pipeline: how frames are retrieved for a given design query
- Knowledge graph reasoning: how the agent traverses discourse/author/post structure
- Interaction model: how a designer queries the system

---

## 4. RESULTS / EVALUATION

_(~2–2.5 pages)_

### 4.1 Dataset and Extraction Statistics

- Total frames extracted, posts processed, forums covered
- Distribution across topics, forums, authors
- Qualitative comparison to traditional machining textbook coverage — what does the forum corpus capture that textbooks don't?

### 4.2 Frame Quality Evaluation

- Designer & machinist review of a random sample of frames
- Rating dimensions: accuracy, usefulness, authenticity to machinist knowledge
- LLM-as-judge agreement metrics
	- need to develop this more
- 2–3 illustrative example frames [**INCLUDE IN PAPER**]

### 4.3 Knowledge Graph Analysis

- Graph topology metrics (nodes, edges, clustering, path lengths)
- Example semantic clusters — do related concepts group meaningfully?
- Example graph traversals showing connected knowledge chains

### 4.4 RAG System Evaluation

- Retrieval precision on a curated set of DFM queries (10–20 queries with known answers)
- Comparison: system vs. vanilla LLM (GPT-4 / Claude with no knowledge base) — blind-rated by expert
- Ablation: RAG with knowledge graph vs. flat vector search only
- Example DFM queries and system responses [**INCLUDE 2–3 IN PAPER**]

---

## 5. DISCUSSION

_(~0.5 page)_

- What does this enable that wasn't possible before?
- AI-as-collaborator paradigm: implications for design practice — designer retains generative agency, AI expands cognitive bandwidth for constraint checking and knowledge retrieval
- Limitations: machining scope (CNC only? manual? mill/lathe?), coverage gaps in forums, LLM extraction errors
- Generalizability: this pipeline applies beyond machining to other tacit/tribal knowledge domains
- Ethical considerations: attribution, forum terms of service, data provenance

---

## 6. CONCLUSION

_(~0.5 page)_

- Restate the problem: DFM is hampered by inaccessible tribal knowledge and late-stage feedback loops
- Summarize contributions:
    1. A process for transforming unstructured forum text into structured, AI-ready knowledge
    2. A growing database of practical machining frames comparable to and extending textbook knowledge
    3. A new AI-as-collaborator design paradigm where AI expands cognitive bandwidth without replacing designer agency
- Future work: expanding forum coverage, adding modalities (video, technical manuals), integrating into CAD tools, live designer study with formal user evaluation

---

## ACKNOWLEDGMENTS

_(optional, 1–2 sentences)_

---

## REFERENCES

_(target ~20–30 citations)_

Key areas needing coverage:

- Machining industry prevalence (defense, transportation, healthcare)
- Early design stage cost influence ("80% of cost locked in early design")
- Late-stage rework cost impact
- DFM foundational papers and computational tools
- Machinability not considered in early design stage
- Tacit/tribal knowledge (Nonaka + manufacturing-specific)
- Knowledge graphs in manufacturing
- LLM-as-judge methodology (original paper)
- RAG (Lewis et al. + engineering applications)
- LLMs in engineering design (recent IDETC/JMD papers)
- AI-as-collaborator / cognitive augmentation
- Forum/community knowledge mining

---

## SOURCE-FINDING PROMPTS

_Use these in Perplexity or Elicit to find citations for the claims above:_

- "machining industry market size defense transportation healthcare manufacturing"
- "early stage design decisions influence total product cost percentage"
- "late stage design rework manufacturing cost impact"
- "design for manufacturability not considered early design stage empirical study"
- "tribal knowledge tacit knowledge manufacturing engineering"
- "Nonaka tacit knowledge creation theory"
- "LLM as judge evaluation methodology"
- "retrieval augmented generation RAG Lewis 2020"
- "large language models engineering design IDETC"
- "knowledge graph manufacturing ontology"
- "AI design collaboration cognitive load designer support"

---

## APPENDIX _(if needed)_

- Extended frame examples
- Full graph schema and ontology
- Prompt templates used in LLM-as-judge extraction workflow