"""R4-G2: the evidence list is a concatenation of two chronologically sorted sub-lists (<= 2 ascending runs in 369/372 pairs). What distinguishes sub-list A (first run) from B (second run)?"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
ev = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet").sort_values(["pair_id", "evidence_rank"]).reset_index(drop=True)
def label_runs(ch):
    out = []; r = 0
    for i, c in enumerate(ch):
        if i > 0 and c < ch[i - 1]: r += 1
        out.append(r)
    return out
ev["run"] = np.concatenate([label_runs(g.chron.values) for _, g in ev.groupby("pair_id", sort=False)]); ev["nruns"] = ev.groupby("pair_id").run.transform("max") + 1
t5 = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"}); role = pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"); ker = pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet")
f = t5.merge(role.drop(columns=["ts", "rs", "ss"], errors="ignore"), on=["slot", "h"]).merge(ker, on=["slot", "h"])
m = ev.merge(f, on="h", how="left", suffixes=("", "_f")); m = m[m.fam == m.behavior_family]
num = [c for c in f.columns if c not in ("slot", "h", "fam", "zone", "ev") and f[c].dtype != object]
for fam, g in m.groupby("behavior_family"):
    two = g[g.nruns == 2]; y = (two.run == 1).astype(int)
    if y.nunique() < 2: continue
    au = {c: roc_auc_score(y, two[c].astype(float).fillna(0)) for c in num if two[c].nunique() > 1}; s = pd.Series(au).sort_values()
    print(f"===== {fam}: 2-run pairs {two.pair_id.nunique()}, hands A {int((y == 0).sum())} B {int((y == 1).sum())}"); print(pd.concat([s.head(12), s.tail(12)]).round(3).to_string())
m.to_parquet(f"{O}/r4/g2_sublists.parquet")
