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
| 10. Agent and app | `dfm_agent.py`, `dfm_report_schema.py`, `streamlit_app.py` | `eval_results/` |
| Export | `export_gephi.py`, `export_neo4j_csv.py`, `push_to_neo4j.py`, `push_to_qdrant_cloud.py` | `neo4j_import/` |

## Run

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp secrets.toml.example .streamlit/secrets.toml   # add your own OpenAI / Qdrant keys
streamlit run streamlit_app.py
```

Clone with Git LFS installed (`git lfs install`) to get the Qdrant index.

## Data

Source posts come from Stack Exchange and are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); derived frames keep links to the original posts. Human-panel ratings in `reviewer_docs/` are anonymized by rater number.

## Citation

> McGowan, D. P., and Goucher-Lambert, K. "Capturing Tacit Design for Manufacturing Knowledge from Unstructured Discourse." *Proc. ASME IDETC/CIE 2026*.
