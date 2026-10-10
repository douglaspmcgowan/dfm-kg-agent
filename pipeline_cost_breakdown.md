# Pipeline Cost Breakdown (IDETC §3.6.2 / Table 14)

_Pricing snapshot: 2026-05 — verify before submission_

## Per-stage cost

| Stage | Model | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|
| Relevance filter | `gpt-5-mini` | 12,805 | 11,524,500 | 384,150 | $3.65 |
| Frame extraction | `gpt-4.1-mini` | 1,936 | 6,582,400 | 3,484,800 | $8.21 |
| Frame review | `gpt-5` | 3,879 | 5,430,600 | 1,357,650 | $20.36 |
| Edge inference | `gpt-5` | 4,272 | 5,126,400 | 854,400 | $14.95 |
| Embeddings | `text-embedding-3-large` | 11,637 | 6,982,200 | 0 | $0.91 |
| **Total** | | | | | **$48.08** |

## Notes

- **Relevance filter** — Per-thread relevance binary classification
- **Frame extraction** — Multi-frame structured output, Pydantic schema
- **Frame review** — Calibrated rubric review (paper used gpt-5; gpt-4.1 recommended)
- **Edge inference** — Pairwise semantic relationship classifier
- **Embeddings** — 3 vectors per frame: knowledge + language + metadata

## Source data

Volumes from the live pipeline:
- 12,805 raw threads scraped
- 1,936 posts sent to LLM extraction
- 3,879 knowledge frames extracted
- 4,272 candidate frame pairs scored for semantic edges
- 11,637 embedding vectors (3 collections × 3,879 frames)

Per-call token estimates are midpoints from observed runs. Replace with real usage logs (`*_usage.json`) for the final paper.
