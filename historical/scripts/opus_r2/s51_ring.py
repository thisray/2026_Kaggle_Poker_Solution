"""Collusion rings: are lower-ranked pairs that share a player with a confident top pair more often positive (ring members)?"""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as APS
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
oof = pd.read_parquet(f"{OUT}/m15_v6base_r2_train_oof.parquet")
for src in ["devsub11", "devsub12"]:
    t = oof[oof.src == src].copy(); t["p_lo"] = (t.key // 12000).astype(int); t["p_hi"] = (t.key % 12000).astype(int)
    t = t.sort_values("oof", ascending=False).reset_index(drop=True); t["rk"] = np.arange(1, len(t) + 1)
    for K in [150, 300]:
        topp = set(t.p_lo[:K]) | set(t.p_hi[:K])
        t["touch"] = t.p_lo.isin(topp) | t.p_hi.isin(topp)
        for lo, hi in [(K, 600), (600, 2000), (2000, 10000), (10000, 60000)]:
            m = (t.rk > lo) & (t.rk <= hi) & ~t.hid
            a = t[m & t.touch]; b = t[m & ~t.touch]
            print(f"{src} K={K} rank ({lo},{hi}]: touching top-{K} players: n {len(a):5d} pos {a.y.mean():.4f} | not touching: n {len(b):6d} pos {b.y.mean():.4f}")
    # promotion test: add bonus to touching pairs beyond rank 600
    clean = ~t.hid; base = APS(t.y[clean], t.oof[clean])
    topp = set(t.p_lo[:300]) | set(t.p_hi[:300]); t["touch"] = t.p_lo.isin(topp) | t.p_hi.isin(topp)
    for f in [1.5, 3.0, 10.0]:
        s2 = np.where(t.touch & (t.rk > 300), np.minimum(t.oof * f, t.oof.iloc[299]), t.oof)
        print(f"   promote touching x{f}: AP_clean {APS(t.y[clean], s2[clean]):.4f} (base {base:.4f})")
