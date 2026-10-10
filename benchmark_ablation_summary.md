# Ablation Results (IDETC §4.4)

Model: gpt-5  Run: full pipeline ablation across 8 queries × 3 configs

## Per-config averages (n=8 queries each)

| Config | Latency (s) | Cost (USD) | Tokens | Cited frames | Design rules | Contradictions | Gaps | Refusal on out-of-scope | Fabricated frame IDs |
|---|---|---|---|---|---|---|---|---|---|
| A_llm_only | 42 | $0.024 | 2459 | 0.0 | 21.8 | 0.0 | 0.0 | 0/2 | 0 |
| B_llm_rag | 17 | $0.013 | 1633 | 8.4 | 3.5 | 0.0 | 0.0 | 0/2 | 0 |
| C_full_pipeline | 196 | $0.174 | 24582 | 8.2 | 8.2 | 0.0 | 6.5 | 2/2 | 0 |

## Per-query results

| Query | Coverage | Config | Latency | Cost | Cited | Rules | Gaps | Refusal? |
|---|---|---|---|---|---|---|---|---|
| Q1 | well-covered | A_llm_only | 34s | $0.022 | 0 | 23 | 0 | no |
| Q1 | well-covered | B_llm_rag | 16s | $0.016 | 11 | 4 | 0 | no |
| Q1 | well-covered | C_full_pipeline | 169s | $0.172 | 9 | 9 | 8 | no |
| Q2 | well-covered | A_llm_only | 39s | $0.026 | 0 | 21 | 0 | no |
| Q2 | well-covered | B_llm_rag | 13s | $0.012 | 12 | 5 | 0 | no |
| Q2 | well-covered | C_full_pipeline | 147s | $0.155 | 8 | 8 | 5 | no |
| Q3 | well-covered | A_llm_only | 30s | $0.025 | 0 | 24 | 0 | no |
| Q3 | well-covered | B_llm_rag | 23s | $0.015 | 0 | 7 | 0 | no |
| Q3 | well-covered | C_full_pipeline | 194s | $0.186 | 9 | 9 | 6 | no |
| Q4 | sparse | A_llm_only | 63s | $0.025 | 0 | 21 | 0 | no |
| Q4 | sparse | B_llm_rag | 23s | $0.013 | 11 | 4 | 0 | no |
| Q4 | sparse | C_full_pipeline | 176s | $0.154 | 8 | 8 | 5 | no |
| Q5 | sparse | A_llm_only | 37s | $0.018 | 0 | 26 | 0 | no |
| Q5 | sparse | B_llm_rag | 17s | $0.013 | 11 | 6 | 0 | no |
| Q5 | sparse | C_full_pipeline | 233s | $0.186 | 10 | 10 | 8 | yes |
| Q6 | sparse | A_llm_only | 53s | $0.023 | 0 | 25 | 0 | no |
| Q6 | sparse | B_llm_rag | 12s | $0.010 | 10 | 0 | 0 | no |
| Q6 | sparse | C_full_pipeline | 176s | $0.164 | 10 | 10 | 5 | no |
| Q7 | out-of-scope | A_llm_only | 35s | $0.021 | 0 | 17 | 0 | no |
| Q7 | out-of-scope | B_llm_rag | 14s | $0.013 | 9 | 2 | 0 | no |
| Q7 | out-of-scope | C_full_pipeline | 273s | $0.212 | 6 | 6 | 8 | yes |
| Q8 | out-of-scope | A_llm_only | 42s | $0.030 | 0 | 17 | 0 | no |
| Q8 | out-of-scope | B_llm_rag | 16s | $0.012 | 3 | 0 | 0 | no |
| Q8 | out-of-scope | C_full_pipeline | 203s | $0.165 | 6 | 6 | 7 | yes |

## Refusal behavior by coverage tier

| Coverage | Config A (LLM only) | Config B (LLM+RAG) | Config C (Full) |
|---|---|---|---|
| well-covered | 0/3 | 0/3 | 0/3 |
| sparse | 0/3 | 0/3 | 1/3 |
| out-of-scope | 0/2 | 0/2 | 2/2 |

