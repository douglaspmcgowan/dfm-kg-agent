# Citation Master Overview
## IDETC 2026 — AI-Assisted Structuring of Tribal Machining Knowledge for DFM

**Purpose:** Triage all sources across all literature files and gap resolution sessions. Three tiers: Must Cite, Nice-to-Have, Skip/Deprioritize, followed by a gap status summary. See individual cluster files (01-06) and the gaps document (08) for full detail.

**Last updated:** 2026-03-02 — reflects gap resolution session (08_gaps_and_actions_updated.md) and two new papers added from external review (Wang et al. JMD 2023, Li et al. JCISE 2025).

---

## MUST CITE (~29 sources)

These earn a place because they are (a) foundational and expected by reviewers, (b) directly analogous to your method, (c) from your own lab/advisor, (d) provide the key claim you need and nothing else does, or (e) newly identified as essential through gap resolution.

| Short Ref | Cluster | Why It's Essential |
|---|---|---|
| Nonaka & Takeuchi (1995) | 2 | Tacit/explicit knowledge distinction and SECI model. Unavoidable. |
| Boothroyd & Dewhurst (1994) | 1 | Foundational DFM/DFMA methodology. Multi-use: Gaps 1, 2, 3, 7. |
| Ehrlenspiel, Kiewert & Lindemann (2007) | 1 | NEW (Gap 1). Academic source for the 70-80% cost-at-design-stage figure, p.32. Springer/ASME. |
| O'Driscoll (2002) | 1 | NEW (Gap 2). Empirically documents late-stage DFM as industry norm. J. Materials Processing Tech. |
| Gupta, Regli, Das & Nau (1997) | 1 | NEW (Gaps 2, 7, 8). Automated DFM survey; over-the-wall framing. Multi-use across Intro and Sec 2.1. |
| Lewis et al. (2020) | 5 | The RAG paper. Every RAG system cites this. Non-negotiable. |
| Siddharth & Luo (2024) | 5 | RAG for engineering design from patent corpora. Most direct RAG pipeline analog prior to Li et al. |
| Li et al. (2025) JCISE | 5 | NEW. LLM + KG pipeline for RAG over unstructured manufacturing data. Direct methodological parallel; published in Ahmed JCISE special issue. Confirm full DOI before submission. |
| Zheng et al. (2023) MT-Bench | 4 | Foundational LLM-as-judge paper. Cite every time you mention the method. |
| Liu et al. (2023) G-Eval | 4 | Rubric-guided LLM evaluation; grounds your 23-field schema design. Also covers Gap 6. |
| Szymanski et al. (2024) IUI | 4 | LLM-as-judge drops to 60-68% in expert domains. Justifies two-tier evaluation. Single most important evaluation-design citation. |
| Edwards et al. (2025) IDETC | 4 | AI judges in engineering design, expert equivalence framework. IDETC-community validation of LLM-as-judge. |
| Qiu & Jin (2023) IDETC | 4 | Direct IDETC precedent for LLM design knowledge extraction. Reviewers will know this. |
| Kernan Freire et al. (2024) Frontiers AI | 2/4 | Closest thematic neighbor: LLMs for manufacturing tacit knowledge. Most important contrast paper. |
| Mustapha (2025) Adv. Eng. Informatics | 4 | LLM survey in manufacturing; directly states tacit/implicit knowledge is a gap. |
| Sweller (1988) | 6 | Cognitive load theory. Expected foundational citation for AI-as-collaborator framing. |
| Zimmerer & Matthiesen (2021) | 6 | Cognitive load reduces performance in engineering design specifically. Core empirical claim. |
| Clemen Lavrsen & Daalhuizen (2024) | 6 | Bridges CLT into design practice. Saves you from doing that bridge yourself. |
| Chong et al. (2023) JMD | 6 | AI-as-collaborator empirically grounded; designer agency; same advisor (Kosa). |
| Zhang et al. (2021) Design Studies | 6 | AI harming high-performing teams -- the contrast that motivates your paradigm. |
| Wang, Grandi & Goucher-Lambert (2023) JMD | 3 | UPDATED. Embedding experiential design knowledge in interactive KGs; same lab. Most direct KG precedent. Confirm relationship to Goridkov 2022 IDETC before submission. |
| Sanya & Shehab (2023) | 3 | KG in manufacturing; explicitly identifies gap in unstructured text sources. |
| Jia, Zhang & Saad (2024) | 3 | Tacit design knowledge + KG from CAD files. Contrast: structured artifacts vs. your forum text. |
| Bolanos et al. (2025) IDETC | 3 | RECALL-MM; same lab. Validates paradigm of turning practitioner corpora into AI-ready design support. |
| Fenoglio et al. (2022) | 2 | Pipeline analog: tacit knowledge elicitation + KG. Also covers Gap 4 peer-reviewed knowledge loss grounding. |
| Sun et al. (2024) JED | 2 | Contemporary review of tacit knowledge management. Also covers Gap 5 forum data justification. |
| Mamykina et al. (2011) CHI | -- | NEW (Gap 5). SE knowledge quality; 92% answer rate, 11-min median. One sentence in Sec 3.2. |
| Minsky (1975) | -- | NEW (Gap 6). Frames as knowledge representation structures. One sentence in Sec 3.3. |
| Deloitte & MI (2024) | -- | NEW (Gap 4). 3.8M workforce figure. Replaces Manual.to. Introduction motivation only; label as industry report. |

---

## NICE-TO-HAVE (use if page count allows; drop if tight)

| Short Ref | Cluster | Value if Included | Can Drop Because |
|---|---|---|---|
| Chong et al. (2022) CHB | 6 | Strengthens Chong 2023 on AI over-reliance mechanism | Chong 2023 alone is sufficient |
| Song et al. (2022) JMD | 6 | AI benefits strongest in high-complexity tasks -- maps to DFM | Unverified author list; Chong covers the empirical ground |
| Xu et al. (2024) JMD | 6 | Human adaptability drives hybrid team performance | Different setup (AI teammate vs. knowledge base); easy to miscite |
| Frich et al. (2024) CHI | 6 | AI-augmented brainwriting; humans-first yields better outcomes | Chong + Zhang already establish generator/collaborator distinction |
| Doris et al. (2025) JCISE DesignQA | 4 | LLMs struggle with engineering doc extraction; motivates your pipeline | Paper reports positive results; this is failure analysis; use lightly |
| Zhao et al. (2024) J. Manuf. Sys. | 4 | LLM NER for additive manufacturing; extraction precedent | Additive is not machining; Kernan Freire + Qiu & Jin cover this ground |
| Polanyi (1966) | 2 | Philosophical origin of tacit knowledge | Nonaka & Takeuchi subsumes it in most engineering papers |
| Buhrig-Polaczek et al. (2025) | 5 | GraphRAG for manufacturing; multi-hop questions benefit from graph retrieval | Lewis + Siddharth + Li et al. cover RAG positioning; add only to defend graph-augmented retrieval in Sec 3.5 |
| Sarica, Luo & Wood (2020) TechNet | 3 | Patent-mining to semantic KG at scale; methodological analog | Wang et al. + Sanya + Jia cover the KG ground more relevantly |
| Pan et al. (2024) JED KG survey | 3 | Broad framing for multimodal KG in engineering design | Already covered by Sanya & Shehab + Jia |
| Onarheim (2012) | 6 | Constraints fuel creativity at optimal levels -- for Discussion | Strong Discussion point; not needed in Related Works |
| Saravi et al. (2008) Springer | 1 | Third peer-reviewed citation for 70-80% cost claim | Ehrlenspiel + Boothroyd already cover it |
| Chang (2011) J. Engineering Design | 1 | Best available empirical source for rework cost; adjacent domain (construction) | Adjacent domain weakens the claim; frame appropriately if used |
| Grand View Research (2023) | -- | CNC market size for Introduction motivation sentence | Industry report; one sentence only; label explicitly |
| Deloitte & MI (2022) | -- | Backup corroboration for workforce claim | 2024 report alone is sufficient |
| Khattab & Zaharia (2020) ColBERT | -- | Multi-vector representation justification (Gap 9) | Only if reviewer pushes on three-collection design; prose rationale usually sufficient |
| IJAMT (2022) automated manufacturability | 1 | Contemporary confirmation DFM tools remain rule-based | Nice support for Gap 7; confirm first author before citing |
| Wan et al. (2025) Computers | 4 | Systematic review of LLMs in digital manufacturing | One more survey on top of Mustapha; diminishing returns |

---

## SKIP / DEPRIORITIZE

| Short Ref | Why to Skip |
|---|---|
| Manual.to (2025) whitepaper | Replaced by Deloitte & MI (2024) for workforce figure. Do not cite. |
| CLEVR, Fictiv, Goddard, LibreTexts | Industry/educational content. Not citable in IDETC. |
| CoLab, bananaz industry guides | Practitioner benchmarks. Not citable in academic related works. |
| Kosmyna et al. (2025) -- Your Brain on ChatGPT | Preprint on essay writing, not engineering design. AI-as-feedback-agent framing addresses it if raised. |
| Clark & Chalmers (1998) -- Extended Mind | Too far from engineering design for IDETC. Sweller + Clemen Lavrsen cover the claims. |
| Singh Medium -- LLM as Second Brain | Not academic. |
| Grinschgl et al. (2025) Frontiers Psych | Too general; not engineering-design specific. |
| Favero et al. (2025) | Secondary citation in another paper; not worth tracking down. |
| Leon-Dominguez (2024) | Same -- secondary citation in a Frontiers piece. |
| Li et al. (2024) arXiv LLM survey | Duplicates Mustapha (2025). |
| Jiang et al. (2024) arXiv LLM-as-judge survey | Zheng + G-Eval + Szymanski cover the territory you need. |
| Chen et al. (2024) arXiv Humans or LLMs as Judge | Szymanski alone is sufficient for expert domain limitation argument. |
| Picard et al. (2025) AI Review | VLMs vs. your text system is an imperfect match. |
| Jadhav & Farimani (2024) LLM as Mechanical Designer | Adjacent framing; not directly connected to your method. |
| KG Fusion arXiv:2506.13026 | Preprint; Wang et al. (Machines 2025) is similar and published. |
| Jeon et al. (2025) ChatCNC | RAG for CNC monitoring, not design feedback. Application overlap is superficial. |
| El Hassani et al. (2024) FMEA | LLMs for FMEA; tangential to DFM frame. |
| Siddharth & Jin (2025) RAG vs CoT | Siddharth & Luo (2024) is the more essential of the two Siddharth papers. |
| Sweller et al. (1998) extended CLT | The 1988 paper is sufficient. |
| JCISE Special Issue editorial | Not a content citation. |
| Calpin & Menold (2023) | Cited within Clemen Lavrsen -- do not track down separately. |
| Sun et al. (2014) | Cited within other papers -- do not track down separately. |
| Rapos et al. (2023) | Alternate listing for Kernan Freire -- same paper. Do not double-list. |
| Goridkov et al. (2022) IDETC | May be the same paper as Wang et al. (2023) JMD. Confirm before deciding whether to cite both or just the JMD version. |

---

## GAP STATUS SUMMARY

All 9 gaps resolved or addressed. See 08_gaps_and_actions_updated.md for full citation stacks and recommended sentences.

| Gap | Claim | Status | Primary Source(s) |
|-----|-------|--------|-------------------|
| 1 | 70-80% of cost locked at design stage | Resolved | Ehrlenspiel et al. (2007) p.32 |
| 2 | DFM not applied early | Resolved | O'Driscoll (2002); Gupta et al. (1997) |
| 3 | Machining industry prevalence | Resolved (industry report) | Grand View Research (2023) |
| 4 | Workforce crisis -- wrong source | Resolved | Deloitte & MI (2024); Fenoglio (2022) for peer-reviewed angle |
| 5 | Forum data legitimacy | Resolved | Mamykina et al. (2011); Sun et al. (2024) |
| 6 | Frame schema -- no theoretical grounding | Resolved | Minsky (1975); Liu et al./G-Eval (2023) |
| 7 | Computational DFM tools -- no citation | Resolved | Gupta et al. (1997); Boothroyd & Dewhurst (1994); IJAMT (2022) |
| 8 | Late-stage rework cost -- machining-specific | Partial | Chang (2011) adjacent domain; Ehrlenspiel + Gupta for framing. No machining-specific empirical study exists -- generalize the claim. |
| 9 | Multi-vector embedding -- no justification | Resolved (prose) | Prose rationale sufficient; ColBERT backup only if reviewer pushes |

---

## Summary Count

| Tier | Count |
|---|---|
| Must Cite | ~29 |
| Nice-to-Have | ~18 |
| Skip/Deprioritize | ~24 |
| Gaps Resolved | 8 of 9 |

Your target of 20-30 citations is achievable from the Must Cite tier alone. At 29 must-cites you are at the upper edge -- when writing, treat the bottom five (Mamykina, Minsky, Deloitte, Gupta, O'Driscoll) as high-priority single-sentence adds rather than developed discussion, keeping the total bibliography tight.

Two pre-submission checks: (1) Confirm whether Wang et al. (JMD 2023) and Goridkov et al. (IDETC 2022) are the same paper or distinct. (2) Confirm full DOI and author list for Li et al. (JCISE 2025) from the published paper.
