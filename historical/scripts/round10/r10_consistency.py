"""Round-10: are true evidence hands mutually more consistent than false candidates?

Within each pair's top-12 (r7 score), compare average pairwise similarity of
feature vectors: evidence-evidence pairs vs evidence-false and false-false pairs.
"""
import json
import numpy as np
import pandas as pd
from itertools import combinations

DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"

n = pd.read_csv(f"{DST}/r6_narrow_candidates_v2.csv")
# r7 scores from geometry file if available
geo = pd.read_csv(f"{R8}/r9_geometry_r7_per_pair.csv")
# use u_r5b ranking as base (r7 per-row z not saved); recompute rank within top12 by u_r5b
n = n.sort_values(["slot", "rank_u_r5b"]).reset_index(drop=True)
feat_cols = [c for c in n.columns if c.startswith(("z_", "ev_", "wit_", "o_", "DS_", "S_", "Q_", "tpl_"))]
X = n[feat_cols].astype(float).values
mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
X = np.clip((X - mu) / sd, -6, 6)

top12 = n[n.rank_u_r5b <= 12].copy()
Xt = X[top12.index.values]
res = {"pairs": 0, "ee": [], "ef": [], "ff": []}
for slot, g in top12.groupby("slot"):
    idx = g.index.values
    y = g.ev.values.astype(bool)
    if y.sum() < 2:
        continue
    ee, ef, ff = [], [], []
    for a, b in combinations(range(len(idx)), 2):
        d = float(np.mean((Xt[a] - Xt[b]) ** 2))
        sim = -d
        if y[a] and y[b]:
            ee.append(sim)
        elif y[a] or y[b]:
            ef.append(sim)
        else:
            ff.append(sim)
    if ee and ef:
        res["pairs"] += 1
        res["ee"].append(np.mean(ee))
        res["ef"].append(np.mean(ef))
        if ff:
            res["ff"].append(np.mean(ff))

import numpy as np
ee = np.array(res["ee"]); ef = np.array(res["ef"])
print(json.dumps({
    "pairs_used": int(res["pairs"]),
    "mean_similarity_evidence_evidence": round(float(ee.mean()), 4),
    "mean_similarity_evidence_false": round(float(ef.mean()), 4),
    "diff_ee_minus_ef": round(float((ee - ef).mean()), 4),
    "frac_pairs_ee_gt_ef": round(float((ee > ef).mean()), 4),
    "ttest_like_ratio": round(float((ee - ef).mean() / (np.std(ee - ef) / np.sqrt(len(ee)) + 1e-9)), 2),
}, indent=2))
