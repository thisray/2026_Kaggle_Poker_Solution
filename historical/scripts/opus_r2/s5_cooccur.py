"""Excess co-seating: do colluders share tables more than their activity predicts? Also session-structure features."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
T = pd.read_parquet(f"{OUT}/m15_v6base_r2_train_oof.parquet"); d = T[T.src == "devsub11"].copy()
t = pd.read_parquet(f"{OUT}/ptab_devsub11.parquet", columns=["p_lo", "p_hi", "n", "lo_hands", "hi_hands", "pool"])
t["key"] = t.p_lo * 12000 + t.p_hi
d = d.merge(t[["key", "lo_hands", "hi_hands"]], on="key", how="left")
N = 2000.0  # hands per pool in a devsub mask (approx)
d["expected"] = d.lo_hands * d.hi_hands / N * (5.0 / 29.0) * (29.0 / 5.0)  # independence baseline scale-free
d["ratio"] = d.n / (d.lo_hands * d.hi_hands / N)
lab = d[d.label >= 0]
print("excess co-seating ratio (n / (a*b/N)) quantiles:")
for nm, g in [("labelled pos", d[d.y == 1]), ("labelled neg", d[(d.label == 0)]), ("U non-hidden", d[(d.label == -1) & ~d.hid]), ("U hidden", d[d.hid])]:
    print(f"  {nm:14s} n={len(g):6d}  median {g.ratio.median():.3f}  p90 {g.ratio.quantile(0.9):.3f}  mean {g.ratio.mean():.3f}")
print("AUC(ratio) pos vs neg (labelled):", round(roc_auc_score(lab.y, lab.ratio), 4), "  pos vs population:", round(roc_auc_score(d.y[~d.hid], d.ratio[~d.hid]), 4))
# residual information beyond the P model: AUC of ratio among top-3000 by oof
top = d[~d.hid].sort_values("oof", ascending=False).head(3000)
print("within top-3000 by P: AUC(ratio) pos vs rest", round(roc_auc_score(top.y, top.ratio), 4), " AUC(oof)", round(roc_auc_score(top.y, top.oof), 4))
