"""
Compare two LLM-judge runs over the same corpus and flag disagreements.

Implements the disagreement-triggered hybrid review method from
Tsaneva et al. (IPM 2025): two LLM judges score the corpus; frames where
they disagree are the candidates for targeted human review.

Inputs (default paths):
  reviews_all.json         (judge 1: gpt-5, medium reasoning)
  reviews_all_gpt54.json   (judge 2: gpt-5.4, medium reasoning)

Outputs:
  disagreements.json       per-frame disagreement records
  disagreement_summary.md  paper-ready summary statistics

Disagreement definition:
  Hard: different `overall` verdict (Keep / Revise / Remove)
  Soft on 1-3 scale: difference >= 1
  Soft on 1-5 scale: difference >= 2
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


SCALE_3 = ("extraction_accuracy", "tribal_knowledge", "clarity")
SCALE_5 = ("frame_type_rating", "post_role_rating", "scope_rating",
           "applicability_rating", "main_point_quality", "epistemic_stance_rating")
ALL_NUMERIC = SCALE_3 + SCALE_5
OVERALL = "overall"


def load_reviews(path: str) -> dict:
    """Load reviews and index by (post_id, frame_index)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    reviews = data.get("reviews", data) if isinstance(data, dict) else data
    indexed = {}
    for r in reviews:
        if "error" in r:
            continue
        key = (r.get("post_id"), r.get("frame_index"))
        if key[0] is None or key[1] is None:
            continue
        indexed[key] = r
    return indexed, data.get("metadata", {}) if isinstance(data, dict) else {}


def pearson(xs, ys):
    """Pearson correlation; returns None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    den = (sx2 * sy2) ** 0.5
    return num / den if den > 0 else None


def diff_record(r1: dict, r2: dict) -> dict:
    """Compute disagreement record for one frame across both judges."""
    rec = {
        "post_id": r1.get("post_id"),
        "frame_index": r1.get("frame_index"),
        "global_idx": r1.get("global_idx"),
        "frame_type": r1.get("_meta", {}).get("frame_type"),
        "scope": r1.get("_meta", {}).get("scope"),
        "thread_title": r1.get("_meta", {}).get("thread_title"),
        "source_quote_preview": r1.get("_meta", {}).get("source_quote_preview"),
        "judge1_overall": r1.get(OVERALL),
        "judge2_overall": r2.get(OVERALL),
        "hard_disagree": r1.get(OVERALL) != r2.get(OVERALL),
        "metric_diffs": {},
        "soft_disagree_count": 0,
    }
    for field in ALL_NUMERIC:
        v1, v2 = r1.get(field), r2.get(field)
        if v1 is None or v2 is None:
            continue
        d = v1 - v2
        rec["metric_diffs"][field] = {"judge1": v1, "judge2": v2, "diff": d}
        threshold = 1 if field in SCALE_3 else 2
        if abs(d) >= threshold:
            rec["soft_disagree_count"] += 1
    rec["any_disagreement"] = rec["hard_disagree"] or rec["soft_disagree_count"] > 0
    rec["judge1_comments"] = r1.get("comments", "")[:300]
    rec["judge2_comments"] = r2.get("comments", "")[:300]
    return rec


def summarize(records: list[dict], meta1: dict, meta2: dict) -> dict:
    """Compute aggregate statistics for the response letter and paper appendix."""
    n = len(records)
    n_hard = sum(1 for r in records if r["hard_disagree"])
    n_any = sum(1 for r in records if r["any_disagreement"])
    n_full_agree = n - n_any

    # Per-metric Pearson r and exact / +-1 agreement
    per_metric = {}
    for field in ALL_NUMERIC:
        xs, ys = [], []
        for r in records:
            md = r["metric_diffs"].get(field)
            if md is not None:
                xs.append(md["judge1"])
                ys.append(md["judge2"])
        if not xs:
            continue
        exact = sum(1 for x, y in zip(xs, ys) if x == y) / len(xs)
        within1 = sum(1 for x, y in zip(xs, ys) if abs(x - y) <= 1) / len(xs)
        per_metric[field] = {
            "n": len(xs),
            "judge1_mean": mean(xs),
            "judge2_mean": mean(ys),
            "judge1_sd": stdev(xs) if len(xs) > 1 else 0,
            "judge2_sd": stdev(ys) if len(ys) > 1 else 0,
            "pearson_r": pearson(xs, ys),
            "exact_agreement_pct": exact * 100,
            "within_1_agreement_pct": within1 * 100,
            "mean_abs_diff": mean(abs(x - y) for x, y in zip(xs, ys)),
        }

    # Overall verdict confusion matrix
    confusion = defaultdict(int)
    for r in records:
        confusion[(r["judge1_overall"], r["judge2_overall"])] += 1
    confusion = {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())}

    # Disagreement breakdown by frame_type
    by_frame_type = defaultdict(lambda: {"n": 0, "hard": 0, "any": 0})
    for r in records:
        ft = r["frame_type"] or "unknown"
        by_frame_type[ft]["n"] += 1
        if r["hard_disagree"]:
            by_frame_type[ft]["hard"] += 1
        if r["any_disagreement"]:
            by_frame_type[ft]["any"] += 1

    return {
        "n_frames_compared": n,
        "n_hard_disagreements": n_hard,
        "hard_disagreement_pct": 100 * n_hard / n if n else 0,
        "n_any_disagreement": n_any,
        "any_disagreement_pct": 100 * n_any / n if n else 0,
        "n_full_agreement": n_full_agree,
        "full_agreement_pct": 100 * n_full_agree / n if n else 0,
        "overall_confusion": confusion,
        "per_metric": per_metric,
        "by_frame_type": dict(by_frame_type),
        "judge1_meta": meta1,
        "judge2_meta": meta2,
    }


def write_summary_md(summary: dict, path: Path):
    """Paper-ready markdown summary."""
    s = summary
    lines = []
    lines.append("# Judge-Model Disagreement Analysis — gpt-5 vs gpt-5.4")
    lines.append("")
    lines.append(f"Method: full-corpus judge-model ablation following Tsaneva et al. (IPM 2025) "
                 "disagreement-triggered hybrid review.")
    lines.append("")
    lines.append("## Judges")
    j1 = s["judge1_meta"]
    j2 = s["judge2_meta"]
    lines.append(f"- **Judge 1**: {j1.get('model','?')} (reasoning={j1.get('reasoning_effort','default/medium')}), {j1.get('reviewed','?')} frames")
    lines.append(f"- **Judge 2**: {j2.get('model','?')} (reasoning={j2.get('reasoning_effort','?')}), {j2.get('reviewed','?')} frames")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Frames compared: **{s['n_frames_compared']:,}**")
    lines.append(f"- Full agreement (no metric disagreement, same overall verdict): **{s['n_full_agreement']:,} ({s['full_agreement_pct']:.1f}%)**")
    lines.append(f"- Any disagreement (hard or soft on any dimension): **{s['n_any_disagreement']:,} ({s['any_disagreement_pct']:.1f}%)**")
    lines.append(f"- Hard disagreement (different Keep/Revise/Remove verdict): **{s['n_hard_disagreements']:,} ({s['hard_disagreement_pct']:.1f}%)**")
    lines.append("")
    lines.append("## Per-dimension agreement")
    lines.append("")
    lines.append("| Metric | n | Pearson r | Exact agreement | Within-1 agreement | Mean abs diff | J1 mean | J2 mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for field, m in s["per_metric"].items():
        r = m["pearson_r"]
        r_str = f"{r:.3f}" if r is not None else "n/a"
        lines.append(
            f"| {field} | {m['n']} | {r_str} | "
            f"{m['exact_agreement_pct']:.1f}% | "
            f"{m['within_1_agreement_pct']:.1f}% | "
            f"{m['mean_abs_diff']:.2f} | "
            f"{m['judge1_mean']:.2f} | {m['judge2_mean']:.2f} |"
        )
    lines.append("")
    lines.append("## Overall verdict confusion matrix")
    lines.append("")
    lines.append("Rows: judge 1 (gpt-5). Columns: judge 2 (gpt-5.4). Cells show frame counts.")
    lines.append("")
    verdicts = ["Keep", "Revise", "Remove"]
    lines.append("| j1 \\ j2 | " + " | ".join(verdicts) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(verdicts)) + "|")
    for v1 in verdicts:
        row = [v1]
        for v2 in verdicts:
            row.append(str(s["overall_confusion"].get(f"{v1}->{v2}", 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Disagreement by frame type")
    lines.append("")
    lines.append("| Frame type | n | Hard disagree | % hard | Any disagree | % any |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ft, d in sorted(s["by_frame_type"].items(), key=lambda x: -x[1]["n"]):
        hard_pct = 100 * d["hard"] / d["n"] if d["n"] else 0
        any_pct = 100 * d["any"] / d["n"] if d["n"] else 0
        lines.append(f"| {ft} | {d['n']} | {d['hard']} | {hard_pct:.1f}% | {d['any']} | {any_pct:.1f}% |")
    lines.append("")
    lines.append("## Implication for the paper")
    lines.append("")
    lines.append(f"Following the Tsaneva et al. (IPM 2025) disagreement-triggered method, "
                 f"the **{s['n_hard_disagreements']:,} frames with hard verdict disagreement** "
                 "are the candidate set for targeted human review. This represents "
                 f"{s['hard_disagreement_pct']:.1f}% of the corpus — a tractable subset "
                 "compared to the 3,879 total frames, suitable for a focused expert pass.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge1", default="reviews_all.json", help="Reference judge (gpt-5)")
    parser.add_argument("--judge2", default="reviews_all_gpt54.json", help="Ablation judge (gpt-5.4)")
    parser.add_argument("--output-json", default="judge_disagreements.json")
    parser.add_argument("--output-md", default="judge_disagreement_summary.md")
    parser.add_argument("--flagged-out", default="flagged_for_human_review.json")
    parser.add_argument("--flag-threshold", type=int, default=2,
                        help="Frame is flagged if hard_disagree OR soft_disagree_count >= threshold")
    args = parser.parse_args()

    print(f"Loading judge 1: {args.judge1}")
    r1, meta1 = load_reviews(args.judge1)
    print(f"  loaded {len(r1)} reviews")
    print(f"Loading judge 2: {args.judge2}")
    r2, meta2 = load_reviews(args.judge2)
    print(f"  loaded {len(r2)} reviews")

    common_keys = sorted(set(r1) & set(r2))
    print(f"Common (post_id, frame_index) keys: {len(common_keys)}")
    only_j1 = set(r1) - set(r2)
    only_j2 = set(r2) - set(r1)
    if only_j1:
        print(f"  WARNING: {len(only_j1)} keys only in judge 1")
    if only_j2:
        print(f"  WARNING: {len(only_j2)} keys only in judge 2")

    records = [diff_record(r1[k], r2[k]) for k in common_keys]
    summary = summarize(records, meta1, meta2)

    # Sort flagged frames by severity
    flagged = [r for r in records
               if r["hard_disagree"] or r["soft_disagree_count"] >= args.flag_threshold]
    flagged.sort(key=lambda r: (-int(r["hard_disagree"]), -r["soft_disagree_count"]))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote per-frame records: {args.output_json}")

    with open(args.flagged_out, "w", encoding="utf-8") as f:
        json.dump({
            "n_flagged": len(flagged),
            "criteria": f"hard_disagree OR soft_disagree_count >= {args.flag_threshold}",
            "frames": flagged,
        }, f, indent=2, ensure_ascii=False)
    print(f"Wrote flagged frames ({len(flagged)} candidates): {args.flagged_out}")

    write_summary_md(summary, Path(args.output_md))
    print(f"Wrote summary markdown: {args.output_md}")

    print()
    print("HEADLINE NUMBERS")
    print(f"  Compared:        {summary['n_frames_compared']:,}")
    print(f"  Full agreement:  {summary['n_full_agreement']:,} ({summary['full_agreement_pct']:.1f}%)")
    print(f"  Any disagree:    {summary['n_any_disagreement']:,} ({summary['any_disagreement_pct']:.1f}%)")
    print(f"  Hard disagree:   {summary['n_hard_disagreements']:,} ({summary['hard_disagreement_pct']:.1f}%)")
    print(f"  Flagged (thr={args.flag_threshold}): {len(flagged):,}")


if __name__ == "__main__":
    main()
