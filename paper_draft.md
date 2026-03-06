# Introduction

In mechanical product design, accounting for manufacturing constraints is an essential part of the design process, and has significant impacts on the later stages of product development. Over X% of product development costs are estimated to come from the early stages of design, and late-stage rebuilds risk overextending project timelines (citation here). In spite of this risk however, it is still common to place the majority of design for manufacturing work in the later stages of the development process (citation here). Our work specifically focuses on machining. Early-stage design for manufacturing (DFM) remains difficult, as the rule-based geometric constraints common in most forms of DFM do not adequately capture the numerous considerations needed to design a manufacturable part. These heuristics guard against common universal failures, but a large part of DFM failures remain unaddressed (citation here). One cause of this is the tacit nature of a large portion of the knowledge and skills in the manufacturing workforce (citation here). This is especially true for machining experts, who possess large amounts of experiential knowledge that is either explicit but undocumented or purely tacit. This tribal knowledge plays a crucial role in the DFM feedback process, but is difficult to know ahead of time for designers. It is also difficult to document and structure for digital systems to communicate.

Current research in design for machining focuses on automating the geometric assessment using computational tools and AI integrations. This, while helpful for accelerating the implementation of fundamental design heuristics, is not effective for involving nuanced contextual information. When planning the manufacturing of a product, the context of the manufacturer's capabilities, equipment, and past experiences are important. This varies across organizations, design settings and times, and this context is often a large part of the design feedback that requires costly rebuilds. Furthermore, when a design cycle is completed, the vital lessons learned from the design-manufacturing tradeoff often lack structured documentation. This results in humans or machines who weren't a direct part of the process missing valuable insights.

Thus, both on the parts of the designer and the manufacturer, there exists a need to capture experiential and tacit knowledge in an efficient manner. There is also a need for this knowledge to be structured in a way that enables future AI systems to use this knowledge for better design interventions.

When designing a product, there are constraints from multiple disciplines that are important to consider such as manufacturability, assembly and sustainability, but it is difficult to factor all of these in, especially at the early stage. Furthermore, many of the designers in the early-stage process do not possess expertise in these areas, which presents an opportunity for AI to assist in the design process. While much research is conducted on AI's ability to generate concepts, there remains an opportunity to sharpen AI's ability to function as a second brain for the designer, enabling thought processes and considerations beyond the designer's own cognitive capacity.

This work provides a foundation for this AI capability in design by demonstrating a method for processing unstructured data and providing a dataset for machining considerations.

We answer the research question:
\> How can tacit knowledge in machining be structured into a scalable and machine-readable information system for better AI-in-design interventions?

To address these questions, we develop a multi-vector embedded space and a knowledge graph constructed from 1,936 design and machining forum posts from 1,136 different authors. Forums contain discourse with valuable practical insights that are not always captured in traditional DFM textbooks. With the forum post data, we demonstrate a framework for extracting specific pieces of knowledge, referred to as frames in this paper. LLM-as-a-judge and human review is used to verify the accuracy and utility of the frames, and the method used to construct the knowledge base is shown to be versatile in potential future applications.

This paper first covers related work in Section 2. This is followed by a description of our methods and a discussion of the results and future opportunities in Section 4.

# Section 2:

This section contains a review of tacit knowledge, DFM practices, AI in DFM research, and LLM-as-a-judge for knowledge verification.

### DFM SOTA

Existing research highlights the limits of design for manufacturing, especially in the modality of knowledge transfer and communication between designer and machinist. Vital information on practical manufacturing constraints that may seem obvious to the manufacturer is often not understood by the designer. This occurs for a variety of reasons, including organization growth further separating designer and machinist, unorganized manufacturing knowledge documentation, and an incompatibility between the traditional rigidity of DFM heuristics and the fluid nature of a real-world manufacturing setting (cite). This results in extremely large shifts in manufacturing requirements stemming from small changes in early stage design.

Additionally, a recent review exposes the divergence of recommended DFM methods, across various design and research contexts. This demonstrates the need for a generalizable knowledge transfer methods for designers and manufacturers in these diverse contexts. Each project has a unique gap for DFM to fill, and the information needed to fill that gap cannot be found in a singular group of textbooks or research papers.

Even if there was a consistent DFM framework, research shows that designers

### Addressing the Knowledge Gap

Earlier research into this same knowledge gap highlighted key causes that remain today: lack of extra-disciplinary knowledge, organizational rigidity, and insufficient integration of manufacturing data during the design phase. While LLMs have enabled unprecedented access to information, the fact remains that the cognitive bandwidth of a single team of designers is limited. This restricts the amount of information that this team can process and consider while drafting concepts, especially when seeking to develop innovative and creative solutions. The solution is not simply adding or removing constraints, but balancing the presentation of these constraints to improve design outcomes while allowing for freeform creation.

Making this knowledge available to designers in a better way first requires capturing it. While existing work has improved the automation of rule-based DFM feedback, these design heuristics capture only a small part of the manufacturing knowledge that goes into expert design assessment. In general, there is a great part of knowledge and reasoning that can be classified as tacit or tribal, in that it is important and often referenced, but rarely explicitly stated and methodically documented. This makes the transfer of knowledge difficult, from expert to junior machinist, and especially from machinist to designer. The communication gap between expert designer and expert manufacturer is multifaceted, but one impactful part of it is the gap in knowledge.

The knowledge creation theory written about in 1995 captures this, showing the hands-on practice typically involved in acquiring it. Both the learning and the documentation are formal and unstructured, which allows for agility in developing new practices, workarounds, and adjustments in a dynamic environment. However, this dynamic nature of the manufacturing environment also makes it difficult to standardize this knowledge, which keeps much of this tribal knowledge from becoming explicit knowledge. This is one of the contributing factors to missing early-stage design feedback; there often is no official place for it to be found, except through extended communication and manual feedback. Moreover, this knowledge is not only uncommunicated; it may also be lost.
Recent research shows a verified risk of losing large amounts of tribal knowledge due to the aging workforce and transitioning industry requirements. If this knowledge is not captured, it may not be accessible in the future.

Even now, in the early stage of design, incorporating expert tribal knowledge is difficult, as there are often multiple different angles of expertise to consider each design from. Furthermore, tribal knowledge is highly contextualized, and it can be difficult to explain when the knowledge factors in to a decision. When converging on a design, teams often focus on the user requirements and needs, and lack the bandwidth to conduct an expert-level review in every relevant discipline. Current methods of using LLMs to broaden & accelerate research and feedback risk hallucination and overgeneralization because the lack of machine-readable tribal knowledge.

### Knowledge Collection SOTA

Current research in the machining knowledge transfer space uses knowledge graphs and ontological structures to map knowledge from structured documents or 3D files. These methods allow for better machine-readable natural language structures that AI can reference. In processing this data, LLMs are used to read and structure the data, while also collecting data from workers directly. Related work also identifies the gap in this process, as it only captures the structured aspects of the design and manufacturing process, with limited exploration into the tacit, informal decision-making and experiential learning process.

Earlier research presented NLP techniques to extract data from structured documents and create knowledge graphs. Multiple agents would collaborate on the data analysis to mimic the social nature of tacit knowledge transfer. By pulling from conversational data and recording the stream of discourse in the knowledge graph, our work captures this directly. Other work focused on computational design applications, analyzing CAD files and engineering drawings to infer design decisions.

Related work has focused on capturing experiential learning from product teardowns into a knowledge graph. A paper by Goridkov et. el. conducted a study to collect data on this process, mentioning a future potential for scalability. Our work demonstrates a streamlined method of capturing experiential knowledge by using existing forum data to capture both tribal and experiential knowledge. This also allows for collection of a larger and more diverse dataset, enabling future agentic systems to use a more versatile knowledge base.

Other related work focuses on developing multimodal datasets of specific design failures, using the dataset to form an interactive embedded space and a predictive model for potential design failures. The embedded space was used to show designers related product failures during the ideation stage, and the predictive model was shown to reach near-expert reliability. Our work expands on this by combining an embedded space with a semantic knowledge graph, enabling a two-fold source of historical and experiential data.

### LLM-as-a-judge

In assessing the quality of these frames, both human review and LLM-as-a-judge methods have been shown to be useful. While human review involves having engineers with machining expertise review knowledge extractions or graph reasoning outputs for quality, the LLM-as-a-judge automates this process. While 80% agreement can be found in general areas, this agreement does lessen when considering expert domains. There is also a demonstrated bias toward lay user preferences due to biases in LLM training data. This shows that in niche domains, human review is still needed to verify accuracy. However, it also verifies that in niche domains a general language model is insufficient for providing expert insight into a specific context, thus confirming the need for more specific knowledge capture.

In disciplines where crowdsourcing expertise is difficult but expert review is still needed, LLM-as-a-judge methods have been developed to ensure the automated review is still valid. Multi-modal reasoning models are typically used to review designs, as the chain-of-thought reasoning has been shown to provide more reliable results. Known weaknesses like flattery, position, and verbosity biases can be mitigated with prompt engineering, randomized ordering. To verify the LLM's accuracy, expert-grounded measurements such as gold standard comparisons, inter-rater reliability, equivalence testing, and top-set overlap are essential.

### 2.6 Cognitive Load in Design, AI as a generator

Our work's role in future AI-for-design systems has backing from previous work as well, which shows that AI's role as a feedback provider gives the designer agency that translates to better outcomes. While much research has been conducted on AI's role in design and concept generation, there remains a significant opportunity for AI to grow in its role as a collaborator that preserves the designer's decision-making agency.
During this decision making, the designer's cognitive load has been shown to have a direct affect on performance. In the case of DFM, considering multiple external constraints with an expert level of detail can require extra cognitive load. This load must be balanced as design problem spaces become more complex, as early-stage tasks like exploring the design space are already cognitively demanding. By supplying information and helping the designer reason over knowledge that they would normally not have access to, future applications of this dataset can enable better early-stage design outcomes. Specifically, AI interventions have been shown to be particularly valuable when the task complexity exceeds what a designer can process by themselves. When considering multiple different manufacturing constraints in mechanical design, this is certainly the case.

This paper demonstrates its novelty in that it combines pioneering work from several different domains. By adapting a novel knowledge-capture methodology to a high-demand discipline of design for manufacturing, this work contributes both a valuable knowledge base and structuring process. The applications and utility of this work is made clear by the recent research demonstrating the validity of LLM-as-a-judge and AI's role as a collaborator in product design.

------------------------------------------------------------------------

### Section 3

- This section describes a four-stage pipeline for transforming unstructured online forum discourse into a structured, machine-queryable DFM knowledge base: (1) data collection and filtering, (2) knowledge frame extraction, (3) multi-vector embedding, and (4) knowledge graph construction. The pipeline preserves the nuance of forum discourse while producing structured knowledge representations that an AI agent can reason over, in real time, during design tasks.

#### 3.1 Web Scraping

To collect the data, we scraped forum threads from Stack Exchange. (make some statement about this being legal?). Since there are a wide variety of conversations held on Stack Exchange, we developed a system to full the most relevant discussion threads. To capture both design-oriented posts and machining-process posts with potential design implications, a list of approximately 20 keywords was developed spanning two complementary categories: keywords describing design decisions and constraints, and keywords describing machining problems or failure modes. This dual-category approach was intentional. We wanted to learn about machining concerns that were seemingly unrelated on the surface, but potentially contained important considerations.

We ran searches with these keywords across five different Stack Exchange sites: Engineering, Mechanics, Woodworking, Electronics, and Robotics.

With the Stack Exchange API, we scraped 12,805 raw threads, each containing multiple posts. Using GPT-5, we filtered out questions without relevant content, such as off-topic or non-technical discussion, and distilled the dataset down to 1,936 posts.

To create the knowledge base, we developed a framework for extracting knowledge from the forum data. Raw forum posts have valuable lessons and insight, but in a format that is difficult for a machine to reason over. There is informal language, narrative structure, implicit assumptions, and high-context conversation.

To take advantage of this richness in discourse, we introduce the concept of a knowledge frame: one distinct unit of design or manufacturing knowledge that is extracted from a post by an LLM with reasoning capabilities. Each post often contains knowledge from which multiple different frames can be extracted. For example, a machinist recounting a story of a build failure could include universal lessons, case-based observations, and clever workarounds in one post.

A 23-field schema was developed for each frame, shown in Figure X. The schema served to both constrain the LLM's extraction output and structure the required metadata for future analysis.

The fields were intentionally structured to capture various knowledge facets. The frame_type classifies what type of claim the frame is making, while the post_role classifies the post's main purpose in the discussion and the epistemic_stance classifies how the post author views the claim being made. These fields help the LLM classify the frame and guide the main_point field, which captures the crux of the frame. The scope, subject, and applicability fields force the LLM to specify where the frame is useful. To force a separation between hypothetical values and discrete heuristics, the example_value and example_context fields have targeted prompts. Finally, the extraction_rationale contains any extra reasoning for the LLM to explain a potentially confusing frame classification.

When prompting the LLM, we used a few shot approach, embedding examples for each schema field individually. Our prompt encouraged the LLM to extract multiple frames per post to prevent the LLM from collapsing a multi-faceted statement into only a single frame. After this, there was a multi-part filteirng step.

The first filtering step used few-shot prompting to check the epistemic_stance field. This field is especially important for categorizing the credibility and nature of the frame, and extra steps were needed to ensure the nuance in implicit communication was accurately captured. The second filtering step focused on analyzing the source_quote field to determine if duplicate frames were extracted. Any frames with similar quote citations were reanalyzed, with the LLM deciding whether to merge the frames, keep both frames, or delete one of them. Any frames with inaccurate source_quotes were deleted, and if a post was flagged to have relevant content, but no frames were extracted, the system was instructed to retry extraction.

Before using the schema for full frame extraction, we iterated on the schema and extraction scripts. Out of a total 192 iterations, there were 56 human iterations and 146 LLM iterations on the code set. For the human iterations, we manually reviewed the extraction results from a set of diverse test frames. Based on the results, we adjusted the schema and extraction script to better capture the knowledge in the post. In addition, we applied LLM-as-a-judge evaluation methods to the tests as well. For these called GPT-5 to assess the schema and extraction script alongside the post and its extractions, and instructed it to make direct, surgical changes to the scripts before iterating again. We installed diagnostics and stop values to prevent the LLM from oscillating between code changes, and would manually review the results every 6-10 iterations.

The LLM-as-a-judge iteration involved in context learning (ICL) based on gold-standard examples, which helped schema convergence. After iterating on a schema that could successfully extract frames from posts in multiple different design or manufacturing disciplines, we executed the full extraction. Out of the 1,936 posts, 1,203 posts had relevant content and 3,879 frames were extracted, with an average of 3.22 frames per post.

Once the frames were extracted, a 3-vector embedded space was constructed. The first vector embeds the extracted knowledge content, pulling from the subject, main point, example context, example value and thread context. This supports semantic retrieval for DFM knowledge and advice, enabling the most relevant knowledge to be found, independent of how it was communicated by the author of the original post. The second vector was based entirely on the source quote, to capture the semantic meaning of the natural language used by the author based on linguistic similarity. This helps with queries phrased with jargon or informal speech. The third vector is an auxiliary metadata vector, based on the metadata rendered as natural language which includes: frame type, scope, epistemic stance, and post role. This supports property-based querying of frames, especially queries based on knowledge type.

A script that supports single-vector and multi-vector search was used to test this embedded space. Figure X shows a two-dimensional approximation of the knowledge-content-based embedded space.
(insert commentary on clusters)

Finally, a knowledge graph was structured to capture the structure of the online discourse and the semantic relations between related frames. At first, we built a provenance graph with the authors, threads, posts and frames as node types, and (THREAD_CONTAINS_POST, POST_AUTHORED_BY, POST_CONTAINS_FRAME, AND SE_ANSWERS) as relationships.

To construct the semantic portion of the graph, we found the top 5 nearest neighbors of each frame in the embedded space (based off all 3 vectors). For each unique pair found, we had the LLM infer semantic relationships between the frames that had relevant content. There were 4 node classifications: SPECIFIES, MITIGATES, CASUAL_LINK, and CONTRADICTS. Unrelated node pairs had no edge drawn. This added another 2,507 semantic edges.

All nodes and edges were merged into a unified Neo4j knowledge graph, enabling interactive exploration.

(insert topic modeling info here)

(maybe agentic retrieval info here?)

# 4 Results/Evaluation

This section presents the results of the knowledge extraction process, both through quantitative metrics and qualitative discussion. We first discuss the size and nature of the data collected and created. This is followed by a discussion of the validation steps and opportunities for future work.

Figure X shows the full numerical pipeline. Across the 1,936 posts used, these posts were written by a total of 1,130 different Stack Exchange users, showing a broad scope of input. The topic distribution can be seen in Figure X, with the design topics being shown the most.
(discuss topic modeling)

To assess the quality of the frames, we combined with LLM-as-a-judge methods and human expert review. For the human expert review, we had two professional machinists and two professional designers review 26 randomly sampled frames. The experts reviewed the frames on accuracy, clarity, and usefulness.
These examples were used as a gold standard for the review and as a randomized test for the quality of the frames extraction.

Figure X shows the review statistics for both the humans and the LLM review. Overall, the ratings of the frames' quality was positive, showing that the presented extraction method is effective. (more commentary)

Across the different frame types and topics, there was a (discuss scope distribution)

Our first output from this frame dataset was a multi-vector embedded space, visualized through topic clusters in Figure X. The knowledge-content-focused embedded space reveals insights about the overlap between varying topic clusters, as well as the differences in extraction structure between different frame types
(info about topic clusters and overlaps)
(compare info on different frame types)

In total, the knowledge extraction process resulted in a graph with 5,846-node graph with 8,959 total edges.
(find out about clusters and comment on them)
(find out about topic modeling results)

### 4.1 Case Study/Application

Application: to demonstrate the usefulness of this knowledge, we use the frame embeddings in a retrieval system. By constructing an agentic RAG-based system, we enable a more concrete connection between design advice and experiential commentary. With this system, the model retrieves the 5 nearest frames, and performs a knowledge graph query for each of them, finding another 15 semantically related frames. Based on these frames, it creates an answer to the question, and is prompted to identify gaps where the available frames do not provide coverage. In these gaps, it defaults to normal textbook advice, combining both traditional and tacit knowledge. The model starts with a concise response, only referencing the reference frame IDs, but the user can access for more details on the frame citations with the "unpack" keyword.

If the user is confused by the advice, it can respond with a "context" query. This prompts the model to access the provenance graph, and locate the credibility of the author and the original thread context of the frames or claims in question. The model then provides a summary of the existing context to better inform the user.

Figure X depicts this framework.

## 5 Future Applications

This knowledge dataset opens multiple future opportunities for design research. First, it is scalable with new sources of data. Future forum data can be used to expand the knowledge graph to enrich and broaden the knowledge base. Augmenting this dataset with frames from machining reports or manuals would allow for another layer of contextual knowledge in the graph. This would enable an agentic RAG system to be more accurate and versatile.

In addition, the presented examples of knowledge retrieval demonstrate the dataset's potential use, but real-world validation remains an open opportunity. By assessing changes in design outcomes when using this tool in academia and industry, more clarity on the dataset's utility can be found.

Furthermore, visual language models (VLMs) could be used in future studies to provide design feedback based on visual design representations or concept drawings. A comparison between the outputs of the VLM and an LLM with descriptions of the same product could provide insight on the effects of including visual representations alongside text.

Finally, the effects of this automated DFM feedback on concept generation and selection could be analyzed. A study could examine the outcomes of both divergent and convergent concept design sessions with DFM intervention.

It is important to note that the usefulness of this system is focused in the areas of expertise that it provides the LLM. This shows an opportunity to guide the knowledge that the LLM references. Using this methodology, a knowledge graph can be developed that expands the LLM's expertise for a specific design context and places guardrails to reduce hallucination.

to expand on: search forms
- discourse search
- semantic search
- embeddings search
- natural language search
- topic cluster-based search
