# State of the Art: Manufacturing Knowledge Transfer & Capture
*Literature summary for related works — DFM tribal knowledge paper*

---

## Overview

Manufacturing knowledge transfer and capture sits at the intersection of knowledge management theory, industrial engineering, and, increasingly, AI/ML-assisted systems. The core challenge is that a large fraction of manufacturing expertise is *tacit* — embodied in practitioners, not documented — and is at risk of loss as experienced workforces age out. The literature addresses this from several angles: theoretical foundations of tacit knowledge, practical elicitation methods, ontology and knowledge graph approaches, and emerging AI-assisted capture pipelines.

---

## 1. Foundational Theory: Tacit Knowledge and the SECI Model

### State of the Art Summary
The theoretical backbone of this space is Nonaka and Takeuchi's knowledge creation model, which distinguishes tacit (experiential, hard to articulate) from explicit (codified, transferable) knowledge and describes a cycle — the SECI model — through which tacit knowledge becomes organizational knowledge. In manufacturing, this framework motivates the design of knowledge capture systems: the goal is to externalize tacit practitioner knowledge into explicit, AI-ready representations. This research's LLM extraction pipeline is, in effect, a scaled, automated externalization mechanism.

**Key claims from this literature:**
- Tacit knowledge is "sticky" — it resists transfer without social interaction or direct experience
- Socialization (person-to-person knowledge sharing, including in online forums) is the primary pathway for tacit knowledge transmission
- The gap between tacit and explicit knowledge is especially pronounced in skilled trades and manufacturing
- Most organizations have no systematic mechanism for capturing tacit knowledge before workers retire

### Sources

**Nonaka, I. & Takeuchi, H. (1995). *The Knowledge-Creating Company*. Oxford University Press.**
- The canonical theoretical text. Introduces the SECI model: Socialization → Externalization → Combination → Internalization. Distinguishes tacit (know-how, mental models) from explicit (documents, databases). Required citation for any paper invoking tacit knowledge.
- No DOI (book). Cite as: Nonaka, I., & Takeuchi, H. (1995). *The Knowledge-Creating Company: How Japanese Companies Create the Dynamics of Innovation*. Oxford University Press.
- Overview of the SECI model: https://en.wikipedia.org/wiki/SECI_model_of_knowledge_dimensions

---

## 2. Tribal Knowledge in Manufacturing: Crisis and Cost

### State of the Art Summary
Recent industry and academic literature documents that "tribal knowledge" loss is a measurable, costly problem in manufacturing — not just a theoretical concern. As baby boomers retire, the knowledge embedded in experienced workers walks out the door with them. This creates real business risk: production failures, quality escapes, and costly rediscovery of lessons already learned. The literature in this area tends to be applied and practitioner-facing, but provides essential motivation for systematic capture efforts.

**Key claims from this literature:**
- 25% of the U.S. manufacturing workforce is over 55; 10,000 baby boomers retire daily
- Tribal knowledge loss costs organizations an estimated $47M/year on average
- Major industrial failures have been attributed to lost tacit knowledge (e.g., NASA Saturn V, nuclear warhead components, Plant Vogtle)
- Only ~10% of organizational knowledge is explicit; ~60% is tacit and ~30% implicit — meaning the vast majority of manufacturing knowledge is vulnerable
- Existing enterprise systems (ERP, MES, PLM) are designed for explicit structured data and fundamentally cannot capture tacit practitioner knowledge

### Sources

**Manual.to (2025). "The Tribal Knowledge Crisis in Manufacturing." Industry whitepaper.**
- Documents workforce aging crisis with concrete statistics: 25% of manufacturing workforce over 55, estimated $47M/year in knowledge loss costs per organization. Includes historical examples: U.S. nuclear warhead component ($69M, 5 years to re-learn), Plant Vogtle nuclear plant ($21B over budget, 7 years late), NASA Saturn V (blueprints existed but practical knowledge was scattered). Distinguishes explicit (10%), tacit (60%), and implicit (30%) knowledge.
- URL: https://manual.to/the-tribal-knowledge-crisis-in-manufacturing/

---

## 3. Tacit Knowledge Elicitation Methods

### State of the Art Summary
Once the problem is established theoretically, the question becomes: how do you actually get tacit knowledge out of people's heads? The literature explores a range of elicitation methods, from structured interviews and cognitive task analysis to more recent AI-assisted conversational approaches. Key findings: (1) most methods are labor-intensive and don't scale; (2) direct questioning often fails because practitioners can't articulate what they know; (3) observation and work shadowing are more effective but expensive; (4) AI-assisted systems that prompt workers through structured reflection show promise.

**Key claims from this literature:**
- Direct questioning ("what do you know about X?") is unreliable — experts have difficulty articulating tacit knowledge on demand
- Knowledge elicitation is more effective when embedded in work practice rather than separated from it
- Ontology-based elicitation systems can provide structured scaffolding that helps practitioners articulate what they know
- Conversational AI (voice/text) can serve as a scalable knowledge elicitation partner for shop-floor workers
- There is a significant gap between what ERP systems capture and what practitioners actually know

### Sources

**Fenoglio, E., Kazim, E., Latapie, H., & Koshiyama, A. (2022). "Tacit Knowledge Elicitation Process for Industry 4.0." *Discover Artificial Intelligence*, 2(1).**
- Closest structural analog to this research in the literature. Uses ontologies and knowledge graphs to capture tacit knowledge from aging manufacturing workers motivated by the Industry 4.0 transition. Key quote: tribal knowledge "is stored within the heads of the experienced workforce never transformed into the company knowledge base, and quantifiable only indirectly as a loss when senior workers leave." Uses domain ontologies, reasoning techniques, and operational knowledge graphs. Describes role-playing game approach with human/virtual agents for knowledge elicitation.
- DOI: https://doi.org/10.1007/s44163-022-00020-w
- Full PDF: https://assets-eu.researchsquare.com/files/rs-1146087/v1_covered.pdf

**Rapos, V., et al. (2023). "Tacit Knowledge Elicitation for Shop-Floor Workers with an Intelligent Assistant." *ACM CHI 2023*.**
- Addresses manufacturing line workers specifically using conversational AI to surface tacit knowledge through structured reflection. System interacts via voice/text and generates visualizations of shop floor data. Useful framing comparison: works with workers directly vs. this research's approach of mining forum discourse produced organically.
- DOI: https://dl.acm.org/doi/10.1145/3544549.3585755

---

## 4. Knowledge Representation: Ontologies and Knowledge Graphs in Manufacturing

### State of the Art Summary
A parallel literature addresses how manufacturing knowledge should be *represented* once captured. Ontologies (formal, machine-readable concept hierarchies) and knowledge graphs (ontologies instantiated with specific facts and relationships) have become the dominant paradigm for structured manufacturing knowledge representation. The state of the art features domain ontologies for machining, assembly, and process planning, as well as knowledge graphs that connect entities across multiple knowledge domains. Key limitation: most existing approaches assume structured data sources; capturing knowledge from unstructured text (like forum discourse) remains underexplored.

**Key claims from this literature:**
- Manufacturing ontologies can formalize process constraints, material properties, machine capabilities, and geometric requirements
- Knowledge graphs enable multi-hop reasoning that flat vector databases cannot support
- Most KG-based approaches in manufacturing assume structured inputs (engineering documents, sensor data, CAD attributes) — unstructured text sources are underserved
- Semantic similarity, causal links, and contradiction relationships between knowledge units require explicit modeling

### Sources

**Sanya, R., & Shehab, E. (2023). "Towards a Knowledge Graph Framework for Ad Hoc Analysis in Manufacturing." *Journal of Intelligent Manufacturing*, 34(8).**
- Reviews knowledge graph applications across manufacturing. Notes "a lot of work is being done representing manufacturing knowledge on a more general level, through the development of domain ontologies." Explicitly identifies gap in capturing knowledge from unstructured data sources — positions this research's forum-mining approach well.
- DOI: https://doi.org/10.1007/s10845-023-02319-6

**Li, Z., et al. (2024). "Knowledge Graph-Based Multi-Granularity Tacit Design Knowledge Reuse." *Journal of Computational Design and Engineering*, 12(1).**
- Directly addresses tacit design knowledge (TDK) in product development using knowledge graphs for design knowledge reuse. Establishes case-based reasoning + ontology approach. Notes that existing KG approaches assume structured sources — this research can position itself as handling unstructured forum discourse.
- DOI: https://doi.org/10.1093/jcde/qwae108
- URL: https://academic.oup.com/jcde/article/12/1/53/7917624

---

## 5. LLM-Assisted Knowledge Extraction from Unstructured Text

### State of the Art Summary
Large language models have enabled a step-change in the ability to extract structured knowledge from unstructured text at scale. Recent work explores LLM-based entity recognition, relation extraction, and knowledge graph construction directly from documents, replacing hand-crafted rule-based NLP pipelines. In engineering design specifically, early work has demonstrated LLM-based extraction from technical documents and patent corpora. The LLM-as-judge methodology — using a strong LLM as an automated quality evaluator — has become a key technique for scalable, human-comparable extraction validation.

**Key claims from this literature:**
- LLMs outperform traditional NLP pipelines for knowledge extraction from domain text, especially for complex, context-dependent relationships
- LLM-based knowledge graph construction has reached production maturity, with validated pipelines achieving high precision on structured extraction tasks
- LLM-as-judge achieves >80% agreement with human evaluators — comparable to inter-human rater agreement — making it a viable alternative for large-scale quality validation
- Extraction from informal/conversational text (forums, chat logs) remains less studied than extraction from formal documents

### Sources

**Zheng, L., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023*.**
- The paper that formalized LLM-as-judge methodology. Demonstrates GPT-4 as judge achieves >80% agreement with human preferences — matching inter-human rater agreement. Also documents known failure modes (position bias, verbosity bias, self-enhancement bias) and mitigation strategies. Required citation for any use of LLM-as-judge for extraction validation.
- ArXiv: https://arxiv.org/abs/2306.05685

**Qiu, J., & Jin, Y. (2023). "Document Understanding-Based Design Support: Language Model Based Design Knowledge Extraction." *ASME IDETC-CIE 2023*.**
- Published at the target venue (ASME IDETC). Uses BERT fine-tuning to extract design knowledge from formal engineering documents. This research extends this to informal, unstructured forum text with graph-based knowledge representation.
- DOI: https://doi.org/10.1115/IDETC-CIE2023/V03AT03A013
- URL: https://asmedigitalcollection.asme.org/IDETC-CIE/proceedings-abstract/IDETC-CIE2023/87301/V03AT03A013/1170428

**Pan, S., et al. (2025). "LLM-Empowered Knowledge Graph Construction: A Survey." ArXiv preprint.**
- Comprehensive survey on LLM-based KG construction covering entity recognition, relation extraction, and schema-free approaches. Documents production maturity reached in 2024–2025. Useful for positioning the knowledge graph construction methodology.
- ArXiv: https://arxiv.org/pdf/2510.20345

**Mustapha, K.B. (2025). "A Survey of Emerging Applications of Large Language Models for Problems in Mechanics, Product Design, and Manufacturing." *Advanced Engineering Informatics*, 64.**
- Survey on LLM applications in engineering and manufacturing. Addresses implicit and tacit knowledge challenges: "domain expertise resides within experts in a form that is implicit or tacit... posed a formidable challenge in terms of explicit articulation and encoding into deep learning models." Good citation for motivating LLM-based approaches to tacit knowledge.
- DOI: https://doi.org/10.1016/j.aei.2025.103066

---

## 6. LLM-as-Judge Applied to Engineering Design Evaluation

### State of the Art Summary
A very recent and thin literature applies LLM-as-judge specifically to engineering design contexts — evaluating design ideas, design compliance with rules, or alignment between design documents and requirements. This work validates that LLMs can serve as scalable, human-comparable evaluators for design-domain content, which supports using the same methodology for manufacturing knowledge frame validation. The application of LLM-as-judge to knowledge *extraction* validation (rather than design *output* evaluation) is still largely novel.

**Key claims from this literature:**
- LLMs can assess design idea novelty and usefulness with alignment to human expert judgments comparable to inter-rater human agreement
- LLM evaluations of design concept quality sometimes fall within human variance on Likert-scale tasks, but scalar scores can obscure reasoning
- Existing work focuses on evaluating design *outputs* (ideas, CAD models, rule compliance) — not on validating extracted knowledge units from practitioner discourse
- Structured prompts with rubric-based criteria significantly improve LLM-human alignment in design evaluation tasks

### Sources

**Zhang, J., Han, J., & Ahmed-Kristensen, S. (2025). "Exploring the Use of LLMs to Evaluate Design Creativity." *Proceedings of the Design Society*, Volume 5: ICED25.**
- Uses LLMs to assess novelty and usefulness of design ideas against human expert judgments. Develops a structured prompt-based evaluation procedure with 7-point rubric. Demonstrates improved alignment between LLM and human expert assessments. Closest precedent in design research for the LLM-as-judge approach used in this paper's frame validation.
- URL: https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/exploring-the-use-of-llms-to-evaluate-design-creativity/B7D16E2612A33C0A9DE1B1922E054639

**Doris, A.C., et al. (2024). "DesignQA: Benchmarking Multimodal LLMs on Questions Grounded in Engineering Documentation." *ASME IDETC-CIE 2024* (DAC track). Also published in *JCISE* (2025).**
- Uses LLMs to evaluate comprehension and compliance with engineering design rules from technical documentation. Involves automated scoring of design-rule-related questions — a close parallel to LLM-based extraction and evaluation in engineering contexts.
- IDETC DOI: https://doi.org/10.1115/DETC2024-139024
- JCISE URL: https://asmedigitalcollection.asme.org/computingengineering/article/25/2/021009/1210215

---

## 7. Positioning Summary for Related Works Section

| Gap | This Research's Response |
|-----|--------------------------|
| Nonaka describes externalization as the key knowledge creation challenge | This pipeline automates the externalization step via LLM extraction |
| Tacit knowledge elicitation methods don't scale | Forum discourse provides organic, large-scale tacit knowledge signal; LLM extraction scales to thousands of posts |
| KG approaches assume structured sources | This research handles unstructured forum discourse as the knowledge source |
| LLM-as-judge validated in NLP and design output evaluation | This work extends it to knowledge extraction validation from informal practitioner text |
| Existing DFM knowledge is geometric/rule-based | Forum-extracted frames capture experiential, context-dependent constraints that formal DFM rules miss |

---

*Last updated: February 2026. For source-finding, use queries: "tacit knowledge manufacturing elicitation", "tribal knowledge transfer Industry 4.0", "knowledge graph manufacturing unstructured text", "LLM as judge engineering design", "Nonaka SECI model manufacturing".*
