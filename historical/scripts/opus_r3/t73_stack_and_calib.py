"""R3-P14: can a fitted stack or an exposure calibration beat the parameter-free group fusion?
(a) nested stacking: ridge/logistic weights on the model OOF scores, fitted on four pool folds and applied to the
    fifth, per subsample; (b) exposure calibration: rank-normalise scores inside shared-hand bins before fusing.
Both are compared with grp_logit and the deployed ranking on the official AP (hidden positives dropped)."""
import numpy as np, pandas as pd, glob, os, json
from sklearn.linear_model import LogisticRegression, Ridge
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/m*_train_oof.parquet"))]
names = [n for n in names if os.path.exists(f"{O}/{n}_eval_scores.parquet")]
def group_of(n):
    if "cat" in n: return "cat"
    if any(k in n for k in ("goss", "dart", "extra")): return "lgb_alt"
    if n.startswith("m5"): return "old"
    return "lgb_gbdt"
out = {}
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); M = base[base.src == src][["key", "pool", "y", "hid", "n"]].set_index("key")
    for n in names:
        d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": n}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    mods = [c for c in M.columns if c not in ("pool", "y", "hid", "n") and M[c].notna().mean() > 0.9]
    keep = (~M.hid.astype(bool) | (M.y == 1)).values; y = M.y.values.astype(int); pools = M.pool.values
    lg = {m: np.log(np.clip(M[m].values, 1e-9, 1 - 1e-9) / (1 - np.clip(M[m].values, 1e-9, 1 - 1e-9))) for m in mods}
    G = {}
    for m in mods: G.setdefault(group_of(m), []).append(m)
    grp = np.nanmean(np.stack([np.nanmean(np.stack([lg[m] for m in v]), 0) for v in G.values()]), 0)
    X = np.stack([lg[m] for m in mods], 1); X = (X - X.mean(0)) / X.std(0)
    up = np.unique(pools); rs = np.random.default_rng(5); fold = {p: i % 5 for i, p in enumerate(rs.permutation(up))}
    fv = np.array([fold[p] for p in pools])
    stack_lr = np.zeros(len(X)); stack_rg = np.zeros(len(X))
    for f in range(5):
        tr = (fv != f) & keep; te = fv == f
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(X[tr], y[tr]); stack_lr[te] = lr.decision_function(X[te])
        rg = Ridge(alpha=50.0).fit(X[tr], y[tr]); stack_rg[te] = rg.predict(X[te])
    nb = pd.qcut(M.n.values, 5, labels=False, duplicates="drop")
    cal = np.zeros(len(X))
    for b in np.unique(nb):
        m_ = nb == b; cal[m_] = pd.Series(grp[m_]).rank(pct=True).values
    res = {"deployed": ap(y[keep], (0.5 * M["m15_v6ens_base"].rank(pct=True) + 0.25 * M["m15_v6ens_cat"].rank(pct=True) + 0.25 * M["m15_v6ens_cat11"].rank(pct=True)).values[keep]),
           "grp_logit": ap(y[keep], grp[keep]), "stack_logistic": ap(y[keep], stack_lr[keep]), "stack_ridge": ap(y[keep], stack_rg[keep]),
           "grp_logit_exposure_calibrated": ap(y[keep], cal[keep])}
    out[src] = {k: round(v, 5) for k, v in res.items()}
    print(src, json.dumps(out[src]), flush=True)
print("means:", {k: round(float(np.mean([out[s][k] for s in out])), 5) for k in out["devsub11"]})
json.dump(out, open(f"{O}/r3/t73_stack_calib.json", "w"), indent=1)
