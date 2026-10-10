# Judge-Model Disagreement Analysis — gpt-5 vs gpt-5.4

Method: full-corpus judge-model ablation following Tsaneva et al. (IPM 2025) disagreement-triggered hybrid review.

## Judges
- **Judge 1**: gpt-5 (reasoning=default/medium), 3879 frames
- **Judge 2**: gpt-5.4 (reasoning=medium), 1150 frames

## Headline numbers

- Frames compared: **1,150**
- Full agreement (no metric disagreement, same overall verdict): **412 (35.8%)**
- Any disagreement (hard or soft on any dimension): **738 (64.2%)**
- Hard disagreement (different Keep/Revise/Remove verdict): **370 (32.2%)**

## Per-dimension agreement

| Metric | n | Pearson r | Exact agreement | Within-1 agreement | Mean abs diff | J1 mean | J2 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| extraction_accuracy | 1150 | 0.330 | 81.3% | 99.9% | 0.19 | 2.92 | 2.78 |
| tribal_knowledge | 1150 | 0.601 | 72.0% | 99.9% | 0.28 | 1.68 | 1.81 |
| clarity | 1150 | 0.138 | 99.0% | 100.0% | 0.01 | 2.99 | 2.99 |
| frame_type_rating | 1150 | 0.597 | 61.9% | 92.5% | 0.46 | 4.74 | 4.37 |
| post_role_rating | 1150 | 0.591 | 86.7% | 98.5% | 0.15 | 4.92 | 4.83 |
| scope_rating | 1150 | 0.632 | 56.3% | 95.5% | 0.49 | 4.36 | 4.12 |
| applicability_rating | 1150 | 0.586 | 57.5% | 85.7% | 0.61 | 4.53 | 4.07 |
| main_point_quality | 1150 | 0.428 | 52.2% | 97.3% | 0.51 | 4.62 | 4.17 |
| epistemic_stance_rating | 1150 | 0.624 | 63.7% | 88.9% | 0.50 | 4.65 | 4.27 |

## Overall verdict confusion matrix

Rows: judge 1 (gpt-5). Columns: judge 2 (gpt-5.4). Cells show frame counts.

| j1 \ j2 | Keep | Revise | Remove |
|---|---:|---:|---:|
| Keep | 592 | 320 | 9 |
| Revise | 39 | 188 | 1 |
| Remove | 0 | 1 | 0 |

## Disagreement by frame type

| Frame type | n | Hard disagree | % hard | Any disagree | % any |
|---|---:|---:|---:|---:|---:|
| heuristic | 408 | 146 | 35.8% | 276 | 67.6% |
| observation | 399 | 118 | 29.6% | 250 | 62.7% |
| risk | 134 | 46 | 34.3% | 85 | 63.4% |
| principle | 115 | 37 | 32.2% | 73 | 63.5% |
| workaround | 48 | 13 | 27.1% | 28 | 58.3% |
| comparison | 43 | 8 | 18.6% | 23 | 53.5% |
| case | 3 | 2 | 66.7% | 3 | 100.0% |

## Implication for the paper

Following the Tsaneva et al. (IPM 2025) disagreement-triggered method, the **370 frames with hard verdict disagreement** are the candidate set for targeted human review. This represents 32.2% of the corpus — a tractable subset compared to the 3,879 total frames, suitable for a focused expert pass.
