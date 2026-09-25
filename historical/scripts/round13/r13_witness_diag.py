"""Round-13 witness diagnostics: (1) per-feature AUC on hard top-12 cases,
(2) incremental value when combined with moments in the CatBoost residual."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R13 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round13_witness_20260918"
R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
OUT = Path(f"{R13}/diag"); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/round11-research-20260918/code")
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917/code")
from r11_bridge_simple import SCORES, build_x, dense  # noqa: E402
from metrics import per_pair  # noqa: E402

meta = pd.read_csv(f"{R13}/dev/meta.csv")
xw = np.load(f"{R13}/dev/witness_views.npy", mmap_mode="r")
fnames = json.loads(Path(f"{R13}/dev/feature_names.json").read_text())
print("witness", xw.shape, "names", len(fnames))

d = pd.read_csv(f"{R8}/dev_pack/meta.csv")
key = ["slot", "hand_id"]
m = d.merge(meta[key], on=key, how="inner")
idx = m.index.to_numpy()
order = pd.MultiIndex.from_frame(d[key]).get_indexer(pd.MultiIndex.from_frame(meta[key]))
assert (order >= 0).all()
xw = np.asarray(xw[order])  # align to dev_pack row order
d["_rk"] = d.groupby("slot")["u_r5b"].rank(ascending=False, method="first")

# ---- (1) per-feature AUC on hard top-12 evidence vs false
sel = (d._rk <= 12).to_numpy()
yv = d.ev.to_numpy(int)[sel]
aucs = []
for vi in range(2):
    X = xw[sel, vi, :]
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.ptp(col) == 0:
            continue
        try:
            a = roc_auc_score(yv, col)
        except ValueError:
            continue
        aucs.append((abs(a - 0.5) + 0.5, a, vi, j, fnames[j] if j < len(fnames) else str(j)))
aucs.sort(reverse=True)
print("top witness features by |AUC-0.5| on top-12:")
for row in aucs[:20]:
    print("  auc=%.4f view=%d %s" % (row[1], row[2], row[4]))

# ---- (2) combined CatBoost residual: moments vs moments+witness
mom = np.load(f"{R8}/dev_pack/tab.npy", mmap_mode="r")
xm = np.nan_to_num(np.c_[mom.mean(1), mom.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
raw = dense(d, SCORES); mu = raw.mean(0); sd = raw.std(0) + 1e-9
xc = build_x(d, xm, mu, sd)  # moments + scores + consistency (deployed feature set)
y = d.ev.to_numpy(int); b = d.u_r5b.to_numpy(float); folds = sorted(d.fold.unique())
# screen witness features on full data (constant removal only)
keepw = np.ptp(xw.reshape(-1, xw.shape[-1]), axis=0) > 0
xw_k = np.nan_to_num(xw[:, :, keepw], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
print("kept witness features", int(keepw.sum()))

params = dict(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30, random_seed=71,
              thread_count=16, verbose=False, allow_writing_files=False)


def run(tag, use_witness):
    delta = np.zeros(len(d))
    for f in folds:
        tr = d.fold.to_numpy() != f; va = ~tr
        cal = LogisticRegression(C=10, max_iter=1000).fit(b[tr][:, None], y[tr])
        sl, ic = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        w = 1.0 / d.m_p.to_numpy(float)
        if use_witness:
            Xtr = np.c_[np.repeat(xc[tr], 2, axis=0), xw_k[tr].reshape(-1, xw_k.shape[-1])]
            ytr = np.repeat(y[tr], 2); btr = np.repeat(sl * b[tr] + ic, 2); wtr = np.repeat(w[tr], 2)
            Xva = np.c_[np.repeat(xc[va], 2, axis=0), xw_k[va].reshape(-1, xw_k.shape[-1])]
        else:
            Xtr, ytr, btr, wtr = xc[tr], y[tr], sl * b[tr] + ic, w[tr]
            Xva = xc[va]
        mdl = CatBoostClassifier(**params)
        mdl.fit(Pool(Xtr, ytr, baseline=btr, weight=wtr))
        p = mdl.predict(Xva, prediction_type="RawFormulaVal")
        if use_witness:
            p = p.reshape(-1, 2).mean(1)
        delta[va] = p / sl
    z = b + 0.25 * delta
    e = float(per_pair(d, z).E.mean())
    print(f"[{tag}] E={e:.6f}", flush=True)
    return e, z


res = {}
res["moments_only"], _ = run("moments_only", False)
res["moments+witness"], zw = run("moments+witness", True)
json.dump({"base": res, "top_auc": [(float(r[1]), int(r[2]), r[4]) for r in aucs[:50]]},
          open(OUT / "r13_witness_diag.json", "w"), indent=2)
np.save(OUT / "z_witness_combo.npy", zw)
print("done")
