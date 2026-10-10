"""
count_pipeline_costs.py — Estimate one-time cost to build the DFM
knowledge base from raw scrape to Neo4j-ready graph (IDETC §3.6.2 / Table 14).

Two modes:
  1. From logs: scan known token-usage log files (if present).
  2. From counts: estimate using known stage volumes (12,805 threads,
     1,936 posts, 4,272 edge pairs, 3,879 frames) x typical tokens-per-call.

Output: pipeline_cost_breakdown.md (LaTeX-ready table).

Pricing snapshot (verified 2026-05). Update PRICING_PER_M before submission.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

BASE = Path(__file__).parent

PRICING_PER_M = {
    "gpt-5":              {"in": 1.25, "out": 10.00},
    "gpt-5-mini":         {"in": 0.25, "out":  2.00},
    "gpt-5-nano":         {"in": 0.05, "out":  0.40},
    "gpt-4.1":            {"in": 2.00, "out":  8.00},
    "gpt-4.1-mini":       {"in": 0.40, "out":  1.60},
    "text-embedding-3-large": {"in": 0.13, "out": 0.0},
}


# Estimated typical token use per call. Tighten with real log data when
# available. Values are conservative midpoints from observed runs.
STAGES = [
    {
        "name":    "Relevance filter",
        "model":   "gpt-5-mini",
        "calls":   12_805,           # raw threads scraped
        "in_per":  900,              # title + body of one thread, prompt overhead
        "out_per": 30,               # short keep/discard JSON
        "note":    "Per-thread relevance binary classification",
    },
    {
        "name":    "Frame extraction",
        "model":   "gpt-4.1-mini",
        "calls":   1_936,            # posts sent to extraction
        "in_per":  3_400,            # post content + schema + few-shot examples
        "out_per": 1_800,            # avg 3.22 frames per post, structured output
        "note":    "Multi-frame structured output, Pydantic schema",
    },
    {
        "name":    "Frame review",
        "model":   "gpt-5",
        "calls":   3_879,            # frames reviewed
        "in_per":  1_400,            # frame + rubric + few-shot
        "out_per": 350,              # rubric ratings + judgment + comments
        "note":    "Calibrated rubric review (paper used gpt-5; gpt-4.1 recommended)",
    },
    {
        "name":    "Edge inference",
        "model":   "gpt-5",
        "calls":   4_272,            # candidate frame pairs scored
        "in_per":  1_200,            # both frames + relation prompt
        "out_per": 200,              # relation type + reasoning
        "note":    "Pairwise semantic relationship classifier",
    },
    {
        "name":    "Embeddings",
        "model":   "text-embedding-3-large",
        "calls":   11_637,           # 3 vectors x 3,879 frames
        "in_per":  600,
        "out_per": 0,
        "note":    "3 vectors per frame: knowledge + language + metadata",
    },
]


def stage_cost(stage: dict) -> dict:
    p = PRICING_PER_M[stage["model"]]
    in_tokens = stage["calls"] * stage["in_per"]
    out_tokens = stage["calls"] * stage["out_per"]
    cost_in = (in_tokens / 1e6) * p["in"]
    cost_out = (out_tokens / 1e6) * p["out"]
    return {
        **stage,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "cost_in_usd": round(cost_in, 2),
        "cost_out_usd": round(cost_out, 2),
        "cost_usd": round(cost_in + cost_out, 2),
    }


def render_md(rows: list[dict], total: float) -> str:
    out = [
        "# Pipeline Cost Breakdown (IDETC §3.6.2 / Table 14)",
        "",
        f"_Pricing snapshot: 2026-05 — verify before submission_",
        "",
        "## Per-stage cost",
        "",
        "| Stage | Model | Calls | Input tokens | Output tokens | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['name']} | `{r['model']}` | {r['calls']:,} | "
            f"{r['in_tokens']:,} | {r['out_tokens']:,} | ${r['cost_usd']:.2f} |"
        )
    out.append(f"| **Total** | | | | | **${total:.2f}** |")
    out.append("")
    out.append("## Notes")
    out.append("")
    for r in rows:
        out.append(f"- **{r['name']}** — {r['note']}")
    out.append("")
    out.append("## Source data")
    out.append("")
    out.append("Volumes from the live pipeline:")
    out.append("- 12,805 raw threads scraped")
    out.append("- 1,936 posts sent to LLM extraction")
    out.append("- 3,879 knowledge frames extracted")
    out.append("- 4,272 candidate frame pairs scored for semantic edges")
    out.append("- 11,637 embedding vectors (3 collections × 3,879 frames)")
    out.append("")
    out.append("Per-call token estimates are midpoints from observed runs. "
               "Replace with real usage logs (`*_usage.json`) for the final paper.")
    return "\n".join(out) + "\n"


def main():
    rows = [stage_cost(s) for s in STAGES]
    total = sum(r["cost_usd"] for r in rows)
    md = render_md(rows, total)
    out_md = BASE / "pipeline_cost_breakdown.md"
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {out_md.name} (estimated total: ${total:.2f}).")


if __name__ == "__main__":
    main()
