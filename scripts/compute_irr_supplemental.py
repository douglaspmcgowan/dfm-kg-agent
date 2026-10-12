"""Compute n=26 and n=27 inter-rater reliability for the IDETC 2026 reviewer panel.

Inputs are repository-relative by default:

  reviewer_docs/reviewer_1_review.json   original slot r1
  reviewer_docs/reviewer_2_review.json   original slot r2
  reviewer_docs/reviewer_4_review.json   original slot r4
  reviewer_docs/frame27_supplemental_ratings.json   Frame 27 (n=27 only)

n=26 is the original panel. n=27 appends Frame 27 from the supplemental
records. Those records are mapped onto the original slots by position; that
constant-strictness mapping is a disclosed assumption, not an identity claim.

Missing data
  A rating that is absent or null is never imputed. For each dimension:
  * AC1 and pairwise agreement use complete-case items: an item enters a
    dimension only when every rater has a value for it.
  * Per-rater proportions (tier distributions, Good rates) use each rater's
    observed ratings as the denominator.
  * Exact all-three agreement is a cross-rater measure, so it uses the same
    complete-case items as AC1 and pairwise agreement.
  Every missing rating is listed with its slot, frame number and field.

Categories
  Raw 1-3 dimensions use categories {1, 2, 3}.
  Raw 1-5 dimensions collapse to {0, 1, 2} = Poor (1-2), Acceptable (3),
  Good (4-5). Gwet's AC1 uses K = 3 categories in both cases, and the chance
  term includes every category, including ones that never occur.

LLM judge comparison (n=26 only)
  reviewer_docs/llm_review.json holds the LLM judge's ratings for the original
  26 frames. Records are aligned to the human slots by frame_num, and the
  alignment is validated (same frame set, no duplicates) before use. Frame 27
  has no LLM observation in this source, so the comparison is n=26 and never
  fabricates a Frame 27 LLM value. Per field, only frames where the LLM and
  all three human slots have a value enter the means; the pooled human mean
  is over every human rating on those frames. Frames dropped for a missing
  value are listed.

Usage
  python scripts/compute_irr_supplemental.py
  python scripts/compute_irr_supplemental.py --input-dir reviewer_docs --output reviewer_docs/irr_metrics.json
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "reviewer_docs"
DEFAULT_OUTPUT = REPO_ROOT / "reviewer_docs" / "irr_metrics.json"

SLOT_FILES = {
    "r1": "reviewer_1_review.json",
    "r2": "reviewer_2_review.json",
    "r4": "reviewer_4_review.json",
}
FRAME27_FILE = "frame27_supplemental_ratings.json"
LLM_FILE = "llm_review.json"

FIELDS_1TO3 = ["extraction_accuracy", "tribal_knowledge", "clarity"]
FIELDS_1TO5 = [
    "frame_type_rating", "post_role_rating", "scope_rating",
    "applicability_rating", "main_point_quality", "epistemic_stance_rating",
]
FIELDS = FIELDS_1TO3 + FIELDS_1TO5
OVERALL_LABELS = ["Keep", "Revise", "Remove"]
K_CATEGORIES = 3


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def collapse_5(v):
    """Collapse a 1-5 rating to Poor (0) / Acceptable (1) / Good (2)."""
    if v <= 2:
        return 0
    if v == 3:
        return 1
    return 2


def category(field, value):
    """Map a raw rating to its analysis category for this field."""
    return collapse_5(value) if field in FIELDS_1TO5 else value


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_panel(input_dir, include_frame27):
    """Return ({slot: [record, ...]}, [missing-record descriptions]).

    Every panel slot holds one record per frame, in frame order. A record is
    kept even when some fields are missing; missing fields are reported.
    """
    panel = {}
    for slot, name in SLOT_FILES.items():
        panel[slot] = load_json(input_dir / name)
    if include_frame27:
        supp = load_json(input_dir / FRAME27_FILE)
        for slot in SLOT_FILES:
            rec = dict(supp["ratings"][slot])
            rec["frame_num"] = supp["frame_num"]
            panel[slot] = panel[slot] + [rec]
    missing = []
    for slot, records in panel.items():
        for rec in records:
            for field in FIELDS + ["overall"]:
                if rec.get(field) is None:
                    missing.append({"slot": slot, "frame_num": rec.get("frame_num"), "field": field})
    return panel, missing


def complete_items(panel, field):
    """Indices of frames where every slot has a non-null value for field."""
    raters = list(panel)
    n_frames = len(panel[raters[0]])
    return [i for i in range(n_frames)
            if all(panel[r][i].get(field) is not None for r in raters)]


def pairwise_agreement(panel, field):
    """Pairwise percent agreement per rater pair, on complete-case items."""
    raters = list(panel)
    items = complete_items(panel, field)
    result = {}
    for a, b in combinations(raters, 2):
        agree = 0
        for i in items:
            va = category(field, panel[a][i][field])
            vb = category(field, panel[b][i][field])
            agree += va == vb
        result[f"{a}_vs_{b}"] = {
            "agree": agree,
            "pairs": len(items),
            "pct": agree / len(items) if items else float("nan"),
        }
    mean_pct = mean([v["pct"] for v in result.values()])
    return result, mean_pct


def gwet_ac1(panel, field):
    """Gwet's AC1 over complete-case items with K = 3 fixed categories.

    Pa is the mean over items of the fraction of rater pairs that agree.
    Pi_k is the marginal share of category k over all ratings in the items.
    Pe = sum_k Pi_k (1 - Pi_k) / (K - 1), with K = 3.
    The categories are {1, 2, 3} for raw 1-3 fields and {0, 1, 2} for
    collapsed 1-5 fields.
    """
    raters = list(panel)
    items = complete_items(panel, field)
    if not items:
        return float("nan"), 0
    cats = [0, 1, 2] if field in FIELDS_1TO5 else [1, 2, 3]
    pa_items = []
    counts = {k: 0 for k in cats}
    for i in items:
        vs = [category(field, panel[r][i][field]) for r in raters]
        pairs = list(combinations(vs, 2))
        pa_items.append(sum(1 for a, b in pairs if a == b) / len(pairs))
        for v in vs:
            counts[v] += 1
    total = sum(counts.values())
    pa = mean(pa_items)
    pe = sum((counts[k] / total) * (1 - counts[k] / total) for k in cats) / (K_CATEGORIES - 1)
    ac1 = (pa - pe) / (1 - pe) if pe < 1 else float("nan")
    return ac1, len(items)


def tier_distribution(panel, field):
    """Per-rater counts of every raw value, over that rater's observed ratings."""
    out = {}
    for slot, records in panel.items():
        observed = [r[field] for r in records if r.get(field) is not None]
        dist = {}
        for v in observed:
            dist[str(v)] = dist.get(str(v), 0) + 1
        out[slot] = {"observed": len(observed), "counts": dict(sorted(dist.items()))}
    return out


def good_rate(panel, field):
    """Per-rater share of observed ratings in the Good tier (raw 3 for 1-3; 4-5 for 1-5)."""
    out = {}
    for slot, records in panel.items():
        observed = [r[field] for r in records if r.get(field) is not None]
        if field in FIELDS_1TO3:
            good = sum(1 for v in observed if v == 3)
        else:
            good = sum(1 for v in observed if v >= 4)
        out[slot] = {"good": good, "observed": len(observed),
                     "rate": good / len(observed) if observed else float("nan")}
    return out


def exact_agreement(panel, field):
    """Share of complete-case items where all raters gave the identical raw value."""
    items = complete_items(panel, field)
    raters = list(panel)
    same = sum(1 for i in items if len({panel[r][i][field] for r in raters}) == 1)
    return {"all_three_exact": same, "items": len(items),
            "pct": same / len(items) if items else float("nan")}


def overall_summary(panel):
    """Per-rater overall distribution, majority verdicts and per-rater Keep rate."""
    raters = list(panel)
    n_frames = len(panel[raters[0]])
    per_rater = {}
    for slot in raters:
        dist = {lab: 0 for lab in OVERALL_LABELS}
        for r in panel[slot]:
            label = str(r["overall"]).title()
            dist[label] = dist.get(label, 0) + 1
        observed = sum(dist.values())
        per_rater[slot] = {"observed": observed, "counts": dist,
                           "keep_rate": dist["Keep"] / observed if observed else float("nan")}
    majority = {lab: 0 for lab in OVERALL_LABELS}
    revise_frames = []
    for i in range(n_frames):
        vs = [str(panel[r][i]["overall"]).title() for r in raters]
        counts = {v: vs.count(v) for v in set(vs)}
        winner = max(counts, key=lambda k: (counts[k], -OVERALL_LABELS.index(k)))
        majority[winner] = majority.get(winner, 0) + 1
        if winner == "Revise":
            revise_frames.append(panel[raters[0]][i]["frame_num"])
    return {"per_rater": per_rater, "majority": majority, "majority_revise_frames": revise_frames}


def align_llm(panel, llm_records):
    """Align LLM records to the human panel by frame_num; raise on any mismatch.

    Returns (llm_by_index, mapping) where llm_by_index[i] is the LLM record for
    the panel frame at position i and mapping lists each frame_num pair.
    """
    raters = list(panel)
    llm_by_frame = {}
    for rec in llm_records:
        fn = rec.get("frame_num")
        if fn in llm_by_frame:
            raise ValueError(f"duplicate LLM frame_num {fn}")
        llm_by_frame[fn] = rec
    aligned, mapping = [], []
    for i, rec in enumerate(panel[raters[0]]):
        fn = rec["frame_num"]
        for r in raters:
            if panel[r][i]["frame_num"] != fn:
                raise ValueError(f"human slots disagree on frame_num at position {i}")
        if fn not in llm_by_frame:
            raise ValueError(f"no LLM record for frame_num {fn}")
        aligned.append(llm_by_frame[fn])
        mapping.append({"frame_num": fn, "llm_frame_num": llm_by_frame[fn]["frame_num"]})
    extra = sorted(set(llm_by_frame) - {m["frame_num"] for m in mapping})
    if extra:
        raise ValueError(f"LLM frames with no human counterpart: {extra}")
    return aligned, mapping


def llm_vs_human(panel, llm_records):
    """LLM judge vs pooled human raters: per-field means, deltas and denominators."""
    raters = list(panel)
    aligned, mapping = align_llm(panel, llm_records)
    out = {"n_frames": len(aligned), "human_slots": raters,
           "frame_mapping": mapping, "fields": {}}
    for field in FIELDS:
        used, dropped = [], []
        for i, rec in enumerate(aligned):
            vals = [panel[r][i].get(field) for r in raters]
            if rec.get(field) is None or any(v is None for v in vals):
                dropped.append(rec["frame_num"])
            else:
                used.append(i)
        llm_vals = [aligned[i][field] for i in used]
        human_vals = [panel[r][i][field] for i in used for r in raters]
        llm_mean, human_mean = mean(llm_vals), mean(human_vals)
        out["fields"][field] = {
            "frames_used": len(used),
            "llm_ratings": len(llm_vals),
            "pooled_human_ratings": len(human_vals),
            "frames_dropped_missing": dropped,
            "llm_mean": llm_mean,
            "pooled_human_mean": human_mean,
            "llm_minus_human": llm_mean - human_mean,
        }
    llm_overall = [str(r["overall"]).title() for r in aligned if r.get("overall") is not None]
    keep = sum(1 for v in llm_overall if v == "Keep")
    maj = overall_summary(panel)["majority"]
    out["overall"] = {
        "llm_keep": keep, "llm_observed": len(llm_overall),
        "llm_keep_rate": keep / len(llm_overall) if llm_overall else float("nan"),
        "human_majority_keep": maj["Keep"], "human_majority_frames": len(aligned),
        "human_majority_keep_rate": maj["Keep"] / len(aligned),
    }
    out["frame27_llm_observation"] = "unavailable in this source; comparison is n=26 only"
    return out


def compute(panel):
    metrics = {"n_frames": len(panel["r1"]), "raters": list(panel), "fields": {}}
    for field in FIELDS:
        pairwise, mean_pct = pairwise_agreement(panel, field)
        ac1, n_items = gwet_ac1(panel, field)
        metrics["fields"][field] = {
            "scale": "1-3" if field in FIELDS_1TO3 else "1-5 collapsed to 3 tiers",
            "items_complete_case": n_items,
            "ac1": ac1,
            "pairwise": pairwise,
            "mean_pairwise_pct": mean_pct,
            "exact_agreement": exact_agreement(panel, field),
            "tier_distribution": tier_distribution(panel, field),
            "good_rate": good_rate(panel, field),
        }
    metrics["overall"] = overall_summary(panel)
    return metrics


def fmt(x):
    return "nan" if x != x else f"{x:.6f}"


def print_report(label, metrics, missing):
    print(f"\n=== {label}: {metrics['n_frames']} frames, raters {', '.join(metrics['raters'])} ===")
    print(f"{'field':<26}{'n_items':>8}{'AC1':>12}{'meanPairwise':>14}{'allExact':>12}")
    for field in FIELDS:
        f = metrics["fields"][field]
        ex = f["exact_agreement"]
        print(f"{field:<26}{f['items_complete_case']:>8}{fmt(f['ac1']):>12}"
              f"{fmt(f['mean_pairwise_pct']):>14}{ex['all_three_exact']}/{ex['items']:<6}")
    ov = metrics["overall"]
    print(f"\nMajority verdicts: {ov['majority']}  Revise frames: {ov['majority_revise_frames']}")
    for slot, v in ov["per_rater"].items():
        print(f"  {slot}: observed={v['observed']} counts={v['counts']} keep_rate={fmt(v['keep_rate'])}")
    print("Per-rater Good rate (raw 3 for 1-3 fields; 4-5 for 1-5 fields):")
    for field in FIELDS:
        gr = metrics["fields"][field]["good_rate"]
        cells = "  ".join(f"{s}={g['good']}/{g['observed']}" for s, g in gr.items())
        print(f"  {field:<26}{cells}")
    print(f"Missing ratings in this panel: {len([m for m in missing])}")
    for m in missing:
        print(f"  slot={m['slot']} frame={m['frame_num']} field={m['field']}")


def print_llm_comparison(cmp_):
    print(f"\n--- LLM judge vs pooled human (n={cmp_['n_frames']}; Frame 27 LLM observation unavailable) ---")
    print(f"{'field':<26}{'frames':>7}{'llm':>10}{'human':>10}{'delta':>10}  dropped")
    for field in FIELDS:
        f = cmp_["fields"][field]
        print(f"{field:<26}{f['frames_used']:>7}{fmt(f['llm_mean']):>10}"
              f"{fmt(f['pooled_human_mean']):>10}{fmt(f['llm_minus_human']):>10}  {f['frames_dropped_missing']}")
    o = cmp_["overall"]
    print(f"LLM Keep {o['llm_keep']}/{o['llm_observed']}; human majority Keep "
          f"{o['human_majority_keep']}/{o['human_majority_frames']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Aggregate JSON for the paper builder; pass '-' to skip writing.")
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    results = {}
    print("REVIEWER IRR: n=26 vs n=27")
    for label, include27 in [("n=26 original panel", False), ("n=27 original + Frame 27", True)]:
        panel, missing = load_panel(input_dir, include27)
        metrics = compute(panel)
        metrics["missing_ratings"] = missing
        results["n26" if not include27 else "n27"] = metrics
        print_report(label, metrics, missing)
        if not include27:
            llm_path = input_dir / LLM_FILE
            if llm_path.is_file():
                cmp_ = llm_vs_human(panel, load_json(llm_path))
                results["llm_vs_human_n26"] = cmp_
                print_llm_comparison(cmp_)

    if str(args.output) != "-":
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
