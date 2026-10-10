"""Compute n=26 and n=27 IRR metrics for the IDETC 2026 reviewer panel.

n=26: original panel (reviewer_1_review.json, reviewer_2_review.json,
reviewer_4_review.json) — the 3 raters who completed the original sample.

n=27: appends Frame 27 (the supplemental "comparison"-type frame that
closes the post-level stratification gap). Frame 27 has 3 supplemental
ratings collected from the IDETC Supplemental Review Folder gdocs:

  Supplemental Reviewer 2  → maps to original slot reviewer_2_review.json
  Supplemental Reviewer 3  → mapped here to original slot reviewer_4_review.json
                              (was the third rater in the original 3-panel)
  Unnumbered Supplemental  → mapped here to original slot reviewer_1_review.json
                              (Doug's other slot)
  Supplemental Reviewer 4  → blank as of pull time; not used

Frame 27 ratings (from gdocs):

  Slot reviewer_1 (= unnumbered):  EA=3 TK=3 CL=3 FT=3 PR=4 SC=5 AP=5 MP=5 ES=5  Keep
  Slot reviewer_2 (= Supp R2):     EA=3 TK=1 CL=3 FT=5 PR=4 SC=5 AP=3 MP=5 ES=5  Keep
  Slot reviewer_4 (= Supp R3):     EA=3 TK=2 CL=3 FT=4 PR=5 SC=4 AP=4 MP=5 ES=5  Keep

This mapping is one interpretation — Doug should sanity-check the
reviewer-identity-to-slot mapping when he writes the appendix. Either
way, the 3 raters on Frame 27 are a real 3-rater set and the IRR
numbers below are computed honestly from the actual ratings.
"""
import json
from itertools import combinations
from pathlib import Path

DFM = Path(r"G:\My Drive\UC Berkeley\Research\dfm_scraping\reviewer_docs")
RDOCS = {
    "r1": DFM / "reviewer_1_review.json",
    "r2": DFM / "reviewer_2_review.json",
    "r4": DFM / "reviewer_4_review.json",
}

FIELDS_1to3 = ["extraction_accuracy", "tribal_knowledge", "clarity"]
FIELDS_1to5 = [
    "frame_type_rating", "post_role_rating", "scope_rating",
    "applicability_rating", "main_point_quality", "epistemic_stance_rating",
]
FIELDS = FIELDS_1to3 + FIELDS_1to5

# Frame 27 (the supplemental comparison frame). 3 raters who completed it.
FRAME27 = {
    "r1": {  # unnumbered supplemental doc
        "frame_num": 27, "extraction_accuracy": 3, "tribal_knowledge": 3,
        "clarity": 3, "frame_type_rating": 3, "post_role_rating": 4,
        "scope_rating": 5, "applicability_rating": 5, "main_point_quality": 5,
        "epistemic_stance_rating": 5, "overall": "Keep",
    },
    "r2": {  # Supplemental Reviewer 2
        "frame_num": 27, "extraction_accuracy": 3, "tribal_knowledge": 1,
        "clarity": 3, "frame_type_rating": 5, "post_role_rating": 4,
        "scope_rating": 5, "applicability_rating": 3, "main_point_quality": 5,
        "epistemic_stance_rating": 5, "overall": "Keep",
    },
    "r4": {  # Supplemental Reviewer 3 (mapped here to slot r4)
        "frame_num": 27, "extraction_accuracy": 3, "tribal_knowledge": 2,
        "clarity": 3, "frame_type_rating": 4, "post_role_rating": 5,
        "scope_rating": 4, "applicability_rating": 4, "main_point_quality": 5,
        "epistemic_stance_rating": 5, "overall": "Keep",
    },
}

# -------- Helpers --------

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

def collapse_5(v):
    """Collapse 1–5 scale to Poor (1–2) / Acceptable (3) / Good (4–5)."""
    if v <= 2: return 0
    if v == 3: return 1
    return 2

def pairwise_pct_agreement(ratings_by_rater, field, scale):
    """Mean pairwise percent agreement.

    scale='1-3' → raw 3-level compare.
    scale='1-5' → collapse to 3 bins (Poor/Acceptable/Good), then compare.
    """
    raters = list(ratings_by_rater.keys())
    n_frames = len(ratings_by_rater[raters[0]])
    pairs = list(combinations(raters, 2))
    agreements = {p: 0 for p in pairs}
    for i in range(n_frames):
        for a, b in pairs:
            va = ratings_by_rater[a][i][field]
            vb = ratings_by_rater[b][i][field]
            if scale == "1-5":
                va, vb = collapse_5(va), collapse_5(vb)
            if va == vb:
                agreements[(a, b)] += 1
    return {f"{a}_vs_{b}": agreements[(a, b)] / n_frames for a, b in pairs}, \
           mean([agreements[p] / n_frames for p in pairs])

def gwet_ac1(ratings_by_rater, field, scale):
    """Gwet's AC1 (multi-rater, single field). Treats each rater equally.

    Uses the standard nominal-category AC1 formula:
      AC1 = (Pa - Pe) / (1 - Pe)
    where Pa is observed pairwise agreement across all rater pairs/items
    and Pe is the chance-agreement term:
      Pe = (1/(K-1)) * sum_k pi_k * (1 - pi_k)
    with pi_k = marginal proportion of category k aggregated across raters.
    K = number of categories that actually appear.
    """
    raters = list(ratings_by_rater.keys())
    n_frames = len(ratings_by_rater[raters[0]])
    # Build category mapping
    def cat(v):
        if scale == "1-5":
            return collapse_5(v)
        return v

    # observed agreement: mean over items of pairwise agreement proportion
    pa_items = []
    for i in range(n_frames):
        vs = [cat(ratings_by_rater[r][i][field]) for r in raters]
        pairs = list(combinations(vs, 2))
        if not pairs:
            continue
        agree = sum(1 for a, b in pairs if a == b)
        pa_items.append(agree / len(pairs))
    Pa = mean(pa_items)

    # marginal pi_k (assume fixed category set of size 3 for tier-collapsed,
    # or the observed set otherwise)
    counts = {}
    for r in raters:
        for i in range(n_frames):
            v = cat(ratings_by_rater[r][i][field])
            counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values())
    # Fixed K=3 for tier-collapsed scales (Poor / Acceptable / Good),
    # matching the paper's convention. Add zero-count categories.
    K_fixed = 3
    if scale == "1-5" or scale == "1-3":
        for k in range(K_fixed):
            counts.setdefault(k, 0)
    pis = {k: counts[k] / total for k in counts}
    K = max(K_fixed, len(pis))
    if K <= 1:
        return 1.0 if Pa == 1.0 else float("nan")
    Pe = sum(pi * (1 - pi) for pi in pis.values()) / (K - 1)
    return (Pa - Pe) / (1 - Pe) if Pe < 1 else float("nan")

def load_panel(extra_frame=None):
    panel = {}
    for slot, path in RDOCS.items():
        with open(path, encoding="utf-8") as f:
            panel[slot] = json.load(f)
        if extra_frame is not None and slot in extra_frame:
            panel[slot] = panel[slot] + [extra_frame[slot]]
    return panel

def overall_distribution(panel):
    out = {}
    for slot, rs in panel.items():
        d = {}
        for r in rs:
            d[r["overall"]] = d.get(r["overall"], 0) + 1
        out[slot] = d
    return out

def majority_overall(panel):
    """For each frame, majority of 3 raters."""
    raters = list(panel.keys())
    n_frames = len(panel[raters[0]])
    counts = {"Keep": 0, "Revise": 0, "Remove": 0}
    revise_frames = []
    for i in range(n_frames):
        vs = [panel[r][i]["overall"].title() for r in raters]
        c = {x: vs.count(x) for x in vs}
        winner = max(c.keys(), key=lambda k: (c[k], -["Keep","Revise","Remove"].index(k)))
        counts[winner] = counts.get(winner, 0) + 1
        if winner == "Revise":
            revise_frames.append(i + 1)
    return counts, revise_frames

def summarize(label, panel):
    print(f"\n=== {label} ===\nN frames = {len(panel['r1'])}, 3 raters\n")
    # means
    print(f"{'Metric':<28} {'R1':>6} {'R2':>6} {'R4':>6} {'Mean':>6} {'SD':>6}")
    for f in FIELDS:
        per_rater = {r: [x[f] for x in panel[r]] for r in panel}
        m_all = mean([v for vs in per_rater.values() for v in vs])
        s_all = std([v for vs in per_rater.values() for v in vs])
        ms = [mean(per_rater[r]) for r in ["r1","r2","r4"]]
        print(f"{f:<28} {ms[0]:>6.2f} {ms[1]:>6.2f} {ms[2]:>6.2f} {m_all:>6.2f} {s_all:>6.2f}")
    print()
    print(f"{'Metric':<28} {'AC1':>7} {'PctAgree':>10}")
    for f in FIELDS:
        scale = "1-3" if f in FIELDS_1to3 else "1-5"
        ac1 = gwet_ac1(panel, f, scale)
        _, pa = pairwise_pct_agreement(panel, f, scale)
        print(f"{f:<28} {ac1:>7.3f} {pa:>10.3f}")
    counts, revise_frames = majority_overall(panel)
    print()
    print("Overall (majority):", counts, " Revise frames:", revise_frames)
    print("Per-rater overall:", overall_distribution(panel))

# Run
print("=" * 70)
print("REVIEWER METRICS: n=26 vs n=27")
print("=" * 70)
summarize("n=26 (original panel)", load_panel())
summarize("n=27 (original + Frame 27 supplemental comparison frame)",
          load_panel(extra_frame=FRAME27))
