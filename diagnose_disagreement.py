"""
Diagnostic deep-dive on gpt-5 vs gpt-5.4 disagreement patterns.
Goal: find the *why* behind the systematic Keep -> Revise shift.
"""
import json
from collections import Counter, defaultdict
from statistics import mean

with open('reviews_all.json', encoding='utf-8') as f:
    j1 = {(r['post_id'], r['frame_index']): r for r in json.load(f)['reviews'] if 'overall' in r}
with open('reviews_all_gpt54_checkpoint.json', encoding='utf-8') as f:
    j2 = {(r['post_id'], r['frame_index']): r for r in json.load(f)['reviews'] if 'overall' in r}

keys = sorted(set(j1) & set(j2))
print(f"N compared: {len(keys)}")

# 1. Where does the disagreement live?
SCALE_3 = ('extraction_accuracy', 'tribal_knowledge', 'clarity')
SCALE_5 = ('frame_type_rating','post_role_rating','scope_rating','applicability_rating','main_point_quality','epistemic_stance_rating')

print("\n=== Per-dimension diff (j2 - j1) ===")
print(f"{'metric':<28} {'mean_diff':>10} {'mean_abs_diff':>14} {'j2_<_j1_%':>12} {'j2_>_j1_%':>12}")
for f in SCALE_3 + SCALE_5:
    diffs = [j2[k][f] - j1[k][f] for k in keys if f in j2[k] and f in j1[k]]
    md = mean(diffs)
    mad = mean(abs(d) for d in diffs)
    lower = 100 * sum(1 for d in diffs if d < 0) / len(diffs)
    higher = 100 * sum(1 for d in diffs if d > 0) / len(diffs)
    print(f"{f:<28} {md:>+10.3f} {mad:>14.3f} {lower:>11.1f}% {higher:>11.1f}%")

# 2. Overall verdict confusion (full matrix)
print("\n=== Overall verdict confusion (j1 rows -> j2 cols) ===")
verdicts = ['Keep','Revise','Remove']
mat = {v1: Counter() for v1 in verdicts}
for k in keys:
    mat[j1[k]['overall']][j2[k]['overall']] += 1
print(f"{'j1\\j2':<10}", *(f"{v:>8}" for v in verdicts), 'total')
for v1 in verdicts:
    row = mat[v1]
    total = sum(row.values())
    print(f"{v1:<10}", *(f"{row[v2]:>8}" for v2 in verdicts), f"{total:>6}")

# 3. For the systematic Keep->Revise shift: what's different about those frames?
shift_records = []
for k in keys:
    if j1[k]['overall'] == 'Keep' and j2[k]['overall'] == 'Revise':
        shift_records.append((k, j1[k], j2[k]))

print(f"\n=== Keep->Revise shift frames: n={len(shift_records)} ===")
print("Frame-type distribution in shift set:")
ft = Counter(r1['_meta']['frame_type'] for _,r1,_ in shift_records)
ft_all = Counter(j1[k]['_meta']['frame_type'] for k in keys)
for f, c in ft.most_common():
    rate = c / ft_all[f] * 100
    print(f"  {f:<15} {c:>4} ({rate:.1f}% of all {f} frames)")

print("\nScope distribution in shift set:")
sc = Counter(r1['_meta']['scope'] for _,r1,_ in shift_records)
sc_all = Counter(j1[k]['_meta']['scope'] for k in keys)
for f, c in sc.most_common():
    rate = c / sc_all[f] * 100
    print(f"  {f:<22} {c:>4} ({rate:.1f}% of all {f} frames)")

# 4. Which metric ratings differ most on the shift set?
print(f"\n=== Metric-rating shifts among Keep->Revise frames (j2 - j1) ===")
print(f"{'metric':<28} {'mean_diff':>10} {'pct_lower_in_j2':>16}")
for f in SCALE_3 + SCALE_5:
    diffs = [r2[f]-r1[f] for _,r1,r2 in shift_records if f in r1 and f in r2]
    lower = 100*sum(1 for d in diffs if d<0)/len(diffs)
    print(f"{f:<28} {mean(diffs):>+10.3f} {lower:>15.1f}%")

# 5. Look at gpt-5.4 comments to figure out WHY
print(f"\n=== Sample gpt-5.4 comments on Keep->Revise shift frames (first 8) ===")
for i, (k, r1, r2) in enumerate(shift_records[:8]):
    print(f"\n[{i+1}] {k[0]}:{k[1]}  ft={r1['_meta']['frame_type']}  scope={r1['_meta']['scope']}")
    print(f"    quote: {r1['_meta']['source_quote_preview']}")
    print(f"    main:  {r1['_meta']['main_point_preview']}")
    print(f"    gpt-5 said Keep   ({r1.get('comments','')[:200]})")
    print(f"    gpt-5.4 said Revise ({r2.get('comments','')[:280]})")

# 6. Reverse direction: where does j2 actually agree more strongly than j1?
revise_to_keep = [(k,j1[k],j2[k]) for k in keys if j1[k]['overall']=='Revise' and j2[k]['overall']=='Keep']
print(f"\n=== Reverse shift (j1=Revise, j2=Keep): n={len(revise_to_keep)} ===")
print("(Mostly empty if gpt-5.4 is uniformly stricter)")
