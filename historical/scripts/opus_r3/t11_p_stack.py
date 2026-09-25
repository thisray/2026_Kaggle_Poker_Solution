"""Do the script-signature counts add P information beyond the deployed pair model?  Fold-wise (pool folds of the OOF) residual
stacker: logistic regression on [logit(oof), signature features]; AP over the full devsub population (hidden positives
excluded, as AP_nohid), per devsub and per family tail."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as aps
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
C = pd.read_parquet(f"{OUT}/t10_dump_counts.parquet").rename(columns={"n": "nh"})
for tag in ["m15_v6ens_base", "m15_v6ens_cat"]:
    d = pd.read_parquet(f"{OUT}/{tag}_train_oof.parquet"); d = d[d.src.str.startswith("devsub")].copy()
    lo = d.key // 12000; hi = d.key % 12000
    d["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
    d = d.merge(C, on=["slot", "src"], how="left")
    d["dmax"] = np.maximum(d.dBA, d.dAB); d["dmin"] = np.minimum(d.dBA, d.dAB); d["dexc"] = d.dmax - d.dmin
    d["lo"] = np.log(np.clip(d.oof, 1e-6, 1 - 1e-6) / (1 - np.clip(d.oof, 1e-6, 1 - 1e-6)))
    d["ln"] = np.log(d.nh.clip(1)); d["r_dmax"] = d.dmax / d.nh.clip(1); d["r_hucd"] = d.hucd / d.nh.clip(1); d["r_ciop"] = d.ciop / d.nh.clip(1)
    print(f"== {tag}")
    pos = d[d.y == 1]; neg = d[(d.y == 0) & ~d.hid]
    for fam, g in pos.groupby("fam"):
        print(f"   {fam:22s} mean dmax {g.dmax.mean():.2f} dmin {g.dmin.mean():.2f} hucd {g.hucd.mean():.2f} ciop {g.ciop.mean():.2f} | n {g.nh.mean():.0f}")
    print(f"   {'negatives':22s} mean dmax {neg.dmax.mean():.2f} dmin {neg.dmin.mean():.2f} hucd {neg.hucd.mean():.2f} ciop {neg.ciop.mean():.2f} | n {neg.nh.mean():.0f}")
    for feats in (["dexc", "dmin"], ["dexc", "dmin", "hucd", "ciop", "ln"], ["r_dmax", "r_hucd", "r_ciop", "dexc", "dmin", "hucd", "ciop", "ln"]):
        for src, g in d.groupby("src"):
            g = g[~g.hid].copy(); new = np.zeros(len(g))
            for f in range(5):
                tr = g.fold != f; va = g.fold == f
                X = g[["lo"] + feats].values
                m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr.values], g.y.values[tr.values])
                new[va.values] = m.decision_function(X[va.values])
            a0 = aps(g.y, g.lo); a1 = aps(g.y, new)
            per = []
            for f in range(5):
                mm = (g.fold == f).values; per.append(round(aps(g.y.values[mm], new[mm]) - aps(g.y.values[mm], g.lo.values[mm]), 4))
            print(f"   {src} feats={feats}: AP_nohid {a0:.4f} -> {a1:.4f} ({a1 - a0:+.4f}); per-fold {per}")
