# DFM Knowledge Graph Agent

Code and data behind the ASME IDETC 2026 paper *Capturing Tacit Design for Manufacturing Knowledge from Unstructured Discourse*.

The pipeline pulls design-for-manufacturing (DFM) discussions from Stack Exchange and uses an LLM to extract structured "knowledge frames" from them. It then links the frames into a provenance graph, rates them with an LLM judge checked against a human panel, and serves them through a retrieval-augmented agent.

## Pipeline

| Stage | Scripts | Output |
|---|---|---|
| 1. Normalize Stack Exchange posts | `normalize_se.py` | extraction-ready threads |
| 2. Extract knowledge frames | `extract_se.py`, `schema_se.py` | frame JSON |
| 3. Clean subjects | `normalize_subjects.py`, `subject_blocklist.json` | normalized frames |
| 4. Embed and index | `build_embeddings.py`, `search_frames.py` | Qdrant index (`qdrant_data/`, Git LFS) |
| 5. Build the graph | `build_graph.py`, `infer_edges.py` | `provenance_graph.gpickle`, `semantic_edges.json` |
| 6. Topics | `topic_model.py`, `group_topics.py`, `viz_umap_scatter.py` | `topic_model_output*.json`, UMAP plots |
| 7. LLM-as-a-judge review | `review_all.py`, `llm_reviewer.py`, `llm_batch_review.py`, `schema_review.py`, `reviewer_rubric.md` | `reviews_all.json` |
| 8. Revise flagged frames | `revise_frames.py` | `revision_changelog_reviews_all.json` |
| 9. Human panel and agreement | `generate_reviewer_docs.py`, `analyze_reviews.py`, `benchmark_irr.py`, `benchmark_tiered_irr.py`, `benchmark_3rater.py` | `reviewer_docs/` |
| 10. Agent and app | `dfm_agent.py`, `dfm_report_schema.py`, `streamlit_app.py` | `dfm_reports/` (JSON reports from `dfm_agent.py`); the Streamlit app writes token and feedback logs |
| Export | `export_neo4j_csv.py` | `neo4j_import/` |
| Export | `export_gephi.py` | `provenance_graph*.gexf` |
| Upload | `push_to_neo4j.py`, `push_to_qdrant_cloud.py` | uploads to Neo4j and Qdrant Cloud; writes no repository files |

## Run

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp secrets.toml.example .streamlit/secrets.toml   # add your own OpenAI / Qdrant keys
streamlit run streamlit_app.py
```

Clone with Git LFS installed (`git lfs install`) to get the Qdrant index.

## Data and attribution

Source posts come from five Stack Exchange sites (Engineering, Electrical Engineering (`electronics`), Robotics, Woodworking, and Motor Vehicle Maintenance & Repair (`mechanics`)) via the Stack Exchange API. Raw post dumps and raw extraction files are not in this repo. The repo holds the derived frames: `provenance_graph.gpickle` and the Qdrant index cover all 3,879 frames, and `neo4j_import/` is an earlier CSV export.

Stack Exchange content is licensed under [CC BY-SA](https://stackoverflow.com/help/licensing): 2.5 for posts before 2011-04-08, 3.0 until 2018-05-02, and 4.0 after that. Derived frames, including verbatim `source_quote` fields, are shared under the same license.

- **Source:** every frame links to its post. Post IDs map to URLs: `se-engineering-51725` → `https://engineering.stackexchange.com/q/51725`, and `se-engineering-51725-a51726` → `https://engineering.stackexchange.com/a/51726` (the middle part is the site subdomain, e.g. `electronics.stackexchange.com`).
- **Authors:** usernames and profile links are in the Author nodes (`neo4j_import/nodes_authors.csv`).
- **Changes:** frames are LLM extractions and paraphrases of the posts; only `source_quote` is verbatim.

Human-panel ratings in `reviewer_docs/` are labeled by rater number. We found no direct identifiers in them, but that is not a guarantee of anonymity. `scripts/compute_irr_supplemental.py` reproduces the agreement statistics from them and the original 26-frame LLM-judge vs pooled-human comparison (Frame 27 has no LLM observation), and `scripts/ablation_token_accounting.py` reproduces the ablation token totals.

## Citation

> McGowan, D. P., and Goucher-Lambert, K. "Capturing Tacit Design for Manufacturing Knowledge from Unstructured Discourse." *Proc. ASME IDETC/CIE 2026*.
