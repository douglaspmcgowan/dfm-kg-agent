# Ablation Results (IDETC §4.4)

Model: gpt-5    Run: 2026-05-05 13:17

## Per-config averages

| Config | Latency (s) | Cost (USD) | Tokens | Cited frames | Rules | Disciplines | Fabricated frame IDs | Refusals (out-of-scope correct) |
|---|---|---|---|---|---|---|---|---|
| C_full_pipeline | 196.5 | $0.174 | 24582 | 8.2 | 8.2 | 0.0 | 0 total | 2/2 |

## Per-query

| Query | Coverage | Config | Latency | Cost | Cited | Rules | Refusal? | Fabricated |
|---|---|---|---|---|---|---|---|---|
| Q1 | well-covered | C_full_pipeline | 169.1s | $0.172 | 9 | 9 | no | 0 |
| Q2 | well-covered | C_full_pipeline | 147.1s | $0.155 | 8 | 8 | no | 0 |
| Q3 | well-covered | C_full_pipeline | 194.5s | $0.186 | 9 | 9 | no | 0 |
| Q4 | sparse | C_full_pipeline | 175.9s | $0.154 | 8 | 8 | no | 0 |
| Q5 | sparse | C_full_pipeline | 233.2s | $0.186 | 10 | 10 | yes | 0 |
| Q6 | sparse | C_full_pipeline | 176.5s | $0.164 | 10 | 10 | no | 0 |
| Q7 | out-of-scope | C_full_pipeline | 272.8s | $0.212 | 6 | 6 | yes | 0 |
| Q8 | out-of-scope | C_full_pipeline | 202.6s | $0.165 | 6 | 6 | yes | 0 |
