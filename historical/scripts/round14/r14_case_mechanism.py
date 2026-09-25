"""Round-14: paired mechanism analysis of 60 hard cases.

For each case (false positive in top-5 vs true evidence at rank 6-12), compare
witness feature vectors and simple record summaries. Outputs ranked discriminators.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

C60 = "/home/thisray/projects/260916_Kaggle_Poker_workers/round14-research-20260918/pkg/results/cases/priority_60_distinct_pair_cases.csv"
EXP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round14_cases60_20260918"
OUT = f"{EXP}/mechanism"
import os
os.makedirs(OUT, exist_ok=True)

cases = pd.read_csv(C60)
meta = pd.read_csv(f"{EXP}/meta.csv")
x = np.load(f"{EXP}/witness_views.npy", mmap_mode="r")
names = json.loads(open(f"{EXP}/feature_names.json").read())
dim1 = x.shape[-1]
nv = len(names)
if nv * 2 == dim1:
    nm = names + ["v1_" + n for n in names]
elif nv == dim1:
    nm = names
else:
    nm = [f"f{i}" for i in range(dim1)]
mi = meta.set_index(["slot", "hand_id"])
print("cases", len(cases), "meta", meta.shape, "feat", dim1, "names", nv)

rows = []
for r in cases.itertuples():
    try:
        ip = mi.loc[(r.slot, r.rejected_positive_hand_id)]
        ineg = mi.loc[(r.slot, r.selected_negative_hand_id)]
    except KeyError:
        continue
    if isinstance(ip, pd.DataFrame): ip = ip.iloc[0]
    if isinstance(ineg, pd.DataFrame): ineg = ineg.iloc[0]
    ipos = meta.index[(meta.slot == r.slot) & (meta.hand_id == r.rejected_positive_hand_id)][0]
    ineg_i = meta.index[(meta.slot == r.slot) & (meta.hand_id == r.selected_negative_hand_id)][0]
    a = np.asarray(x[ipos]).mean(0); b = np.asarray(x[ineg_i]).mean(0)
    rows.append(a - b)
D = np.vstack(rows)
print("diffs", D.shape)
D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
mean = D.mean(0); sd = D.std(0) + 1e-9
t = mean / (sd / np.sqrt(len(D)))
order = np.argsort(-np.abs(t))
res = []
for j in order[:40]:
    res.append({"feature": nm[j] if j < len(nm) else str(j), "t": float(t[j]),
                "mean_diff_positive_minus_negative": float(mean[j]),
                "n_nonzero": int((np.abs(D[:, j]) > 1e-9).sum())})
print(json.dumps(res[:25], indent=2))
pd.DataFrame(res).to_csv(f"{OUT}/paired_witness_discriminators.csv", index=False)

# raw record summaries
summ = {"positive": [], "negative": []}
with open(f"{EXP}/raw_cases.jsonl") as f:
    for line in f:
        d = json.loads(line)
        key = (d["slot"], d["hand_id"])
        grp = None
        for r in cases.itertuples():
            if (r.slot, r.rejected_positive_hand_id) == key: grp = "positive"
            if (r.slot, r.selected_negative_hand_id) == key: grp = "negative"
        if grp is None: continue
        recs = d["records"]
        pair = set(d["seats"])
        n_act = len(recs)
        n_post = sum(1 for r in recs if r.get("st", 0) > 0)
        n_fold = sum(1 for r in recs if r.get("act") == 0)
        n_aggr = sum(1 for r in recs if r.get("aggr"))
        n_hu = sum(1 for r in recs if sum(r.get("alive", [])) == 2)
        max_pot = max([r.get("pot", 0) for r in recs] or [0])
        max_sur = max([r.get("raise_proxy", 0) for r in recs] or [0])
        pe = [r.get("pair_equity_exact") for r in recs if r.get("pair_equity_exact") is not None]
        summ[grp].append({"slot": d["slot"], "hand": d["hand_id"], "n_act": n_act, "n_post": n_post,
                          "n_fold": n_fold, "n_aggr": n_aggr, "n_hu": n_hu, "max_pot": max_pot,
                          "max_raise_proxy": max_sur, "n_pair_eq": len(pe)})
for g in summ:
    df = pd.DataFrame(summ[g])
    print(f"== {g} ({len(df)}) means:", df.drop(columns=['slot','hand']).mean().round(3).to_dict())
json.dump({g: {k: float(np.mean([r[k] for r in summ[g]])) for k in summ[g][0] if k not in ('slot','hand')} for g in summ},
          open(f"{OUT}/record_summary.json", "w"), indent=2)
print("done")
