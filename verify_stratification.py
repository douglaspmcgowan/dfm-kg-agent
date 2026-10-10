"""
verify_stratification.py — Audit the 26-frame human-review sample's
stratification across frame_type, scope, and SE site (IDETC §4.2,
addresses R1c and R1d).

Sources:
  - Panel sample: 26 frames embedded as `FRAMES` list in `llm_review_v3.py`.
  - Corpus: 3,879 frames flattened from `extractions_se_all.json`
    (top-level "results" -> per-post "frames").

Output:
  stratification_report.md     LaTeX-ready table for §4.2 / Table 15

Usage:
    python verify_stratification.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent
EXTRACTIONS = BASE / "extractions_se_all.json"
REPORT_OUT = BASE / "stratification_report.md"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load_corpus() -> list[dict]:
    raw = json.loads(EXTRACTIONS.read_text(encoding="utf-8"))
    posts = raw["results"]
    out = []
    for post in posts:
        site = post.get("se_site", "") or "unknown"
        pid = post.get("post_id", "")
        pscore = post.get("post_score", 0) or 0
        for fr in post.get("frames", []) or []:
            f = dict(fr)
            f["_post_id"] = pid
            f["_post_score"] = pscore
            f["_se_site"] = site
            out.append(f)
    return out


def load_panel() -> list[dict]:
    sys.path.insert(0, str(BASE))
    from llm_review_v3 import FRAMES  # type: ignore
    return list(FRAMES)


def attach_site_to_panel(panel: list[dict], corpus: list[dict]) -> list[dict]:
    """Look up SE site for each panel frame by source_quote match against corpus."""
    by_quote: dict[str, list[dict]] = {}
    for f in corpus:
        q = norm(f.get("source_quote", ""))
        if q:
            by_quote.setdefault(q, []).append(f)

    enriched = []
    for pf in panel:
        pf = dict(pf)
        q = norm(pf.get("source_quote", ""))
        hits = by_quote.get(q, [])
        if not hits and q and len(q) > 30:
            for cq, frs in by_quote.items():
                if q in cq or cq in q:
                    hits = frs
                    break
        if hits:
            pf["_se_site"] = hits[0].get("_se_site", "unknown")
            pf["_post_id"] = hits[0].get("_post_id", "")
            pf["_corpus_match"] = True
        else:
            # Fall back to inferring site from `tags` field on panel
            tags = (pf.get("tags") or "").lower()
            site = "unknown"
            if "pcb" in tags or "electronics" in tags:
                site = "electronics"
            elif "robot" in tags:
                site = "robotics"
            elif "wood" in tags:
                site = "woodworking"
            elif "mechanic" in tags:
                site = "mechanics"
            else:
                site = "engineering"  # default for non-tagged panel items
            pf["_se_site"] = site
            pf["_corpus_match"] = False
        enriched.append(pf)
    return enriched


def fmt_dist(items: list[dict], key: str, n_total: int, top: int = 12) -> str:
    c = Counter((i.get(key) or "(missing)") for i in items)
    parts = [f"{k} {v} ({100*v/n_total:.1f}%)" for k, v in c.most_common(top)]
    return ", ".join(parts) or "—"


def render(panel: list[dict], corpus: list[dict]) -> str:
    n_p = len(panel)
    n_c = len(corpus)

    threads = {pf.get("thread") or pf.get("_post_id") for pf in panel}
    posts = {pf.get("_post_id") for pf in panel if pf.get("_post_id")}
    score_vals = [pf.get("post_score") or pf.get("_post_score") or 0 for pf in panel]
    score_min, score_max = (min(score_vals), max(score_vals)) if score_vals else (0, 0)

    matched = sum(1 for pf in panel if pf.get("_corpus_match"))

    md = [
        "# 26-Frame Sample Stratification (IDETC §4.2 / Table 15)",
        "",
        f"Sample size: **{n_p}** frames    Corpus: **{n_c}** frames    "
        f"Threads: **{len(threads)}**    Distinct posts: **{len(posts)}**    "
        f"Post-score range: **{score_min} to {score_max}**    "
        f"Source-quote matches: **{matched}/{n_p}**",
        "",
        "## Distributions",
        "",
        f"| Stratum | Sample (n={n_p}) | Corpus (n={n_c}) |",
        "|---|---|---|",
        f"| Frame type | {fmt_dist(panel, 'frame_type', n_p)} | {fmt_dist(corpus, 'frame_type', n_c)} |",
        f"| Scope | {fmt_dist(panel, 'scope', n_p)} | {fmt_dist(corpus, 'scope', n_c)} |",
        f"| Stack Exchange site | {fmt_dist(panel, '_se_site', n_p)} | {fmt_dist(corpus, '_se_site', n_c)} |",
        f"| Epistemic stance | {fmt_dist(panel, 'epistemic_stance', n_p)} | {fmt_dist(corpus, 'epistemic_stance', n_c)} |",
        f"| Applicability | {fmt_dist(panel, 'applicability', n_p)} | {fmt_dist(corpus, 'applicability', n_c)} |",
        "",
        "## Coverage check",
        "",
    ]

    for key in ("frame_type", "scope", "_se_site"):
        s_set = {(f.get(key) or "(missing)") for f in panel}
        c_set = {(f.get(key) or "(missing)") for f in corpus}
        missing = c_set - s_set
        md.append(f"- **{key.lstrip('_')}**: sample covers {len(s_set)} of {len(c_set)} corpus values")
        if missing and len(missing) <= 6:
            md.append(f"  - Not represented in sample: {sorted(missing)}")

    md.append("")
    md.append("## Sample frame list")
    md.append("")
    md.append("| # | Thread | Post score | frame_type | scope | site | matched |")
    md.append("|---|---|---|---|---|---|---|")
    for pf in sorted(panel, key=lambda f: f.get("num", 0)):
        thread = (pf.get("thread") or "")[:50]
        md.append(
            f"| {pf.get('num', '?')} | {thread} | "
            f"{pf.get('post_score', pf.get('_post_score', '?'))} | "
            f"{pf.get('frame_type', '?')} | {pf.get('scope', '?')} | "
            f"{pf.get('_se_site', '?')} | "
            f"{'yes' if pf.get('_corpus_match') else 'inferred'} |"
        )
    md.append("")
    return "\n".join(md) + "\n"


def main():
    corpus = load_corpus()
    panel = load_panel()
    panel = attach_site_to_panel(panel, corpus)

    md = render(panel, corpus)
    REPORT_OUT.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {REPORT_OUT.name}.")


if __name__ == "__main__":
    main()
