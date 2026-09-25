"""Round-10 E3: deployable score-only rankers + blend with r7 combo scores.

Features available for eval from eval_candidates.csv: 11 score columns.
Add within-pair rank/gap features per score; train LightGBM lambdarank (trunc 5)
and CatBoost; measure dev OOF AP@5; find best deployable blend with r7 (z_cat).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917/code")
from metrics import per_pair

R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
OUT = Path(f"{R8}/r10_e3"); OUT.mkdir(parents=True, exist_ok=True)
SCORES = ['sc_r5', 'u0', 'u_rr', 'u_r5b', 'lin_contrib', 'nn_contrib', 't1_score',
          's1_stage1', 'gen_logit', 'gen_rank_pct', 'rank_u_r5b']
d = pd.read_csv(f"{R8}/dev_pack/meta.csv")
y = d.ev.to_numpy(int)
folds = sorted(d.fold.unique())

feat = {}
for c in SCORES:
    v = d[c].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    feat[c] = v
    r = d.assign(_v=v).groupby("slot")["_v"].rank(ascending=False, method="average").to_numpy()
    feat[c + "_rk"] = r
    g = d.assign(_v=v, _r=r)
    mx = g.groupby("slot")["_v"].transform("max").to_numpy()
    feat[c + "_gap1"] = mx - v
X11 = np.nan_to_num(np.column_stack(list(feat.values())), nan=0., posinf=0., neginf=0.)
print("score-only feature dim", X11.shape)

order = np.argsort(d.slot.to_numpy(), kind='stable')
ds = d.iloc[order]; xs = X11[order]; ys = y[order]
z_lr11 = np.full(len(d), np.nan)
for f in folds:
    tr = ds.fold.to_numpy() != f; va = ds.fold.to_numpy() == f
    gtr = ds[tr].groupby('slot', sort=False).size().to_numpy()
    mdl = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.04, num_leaves=31,
                         min_child_samples=20, subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                         lambdarank_truncation_level=5, label_gain=[0, 1],
                         random_state=71, n_jobs=16, verbose=-1)
    mdl.fit(xs[tr], ys[tr], group=gtr)
    z_lr11[order[va]] = mdl.predict(xs[va])
E_lr11 = float(per_pair(d, z_lr11).E.mean())
print(f"[lambdarank score-only] E={E_lr11:.6f}", flush=True)

b = d.u_r5b.to_numpy()
cal = LogisticRegression(C=10, max_iter=1000).fit(b[:, None], y)
sl, ic = float(cal.coef_[0, 0]), float(cal.intercept_[0])
z_c11 = np.full(len(d), np.nan)
for f in folds:
    va = d.fold.to_numpy() == f
    m = CatBoostClassifier(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30,
                           random_seed=71, thread_count=16, verbose=False, allow_writing_files=False)
    m.fit(Pool(X11[~va], y[~va], baseline=sl * b[~va] + ic, weight=1 / d.m_p.to_numpy()[~va]))
    z_c11[va] = b[va] + 0.25 * m.predict(X11[va], prediction_type='RawFormulaVal') / sl
E_c11 = float(per_pair(d, z_c11).E.mean())
print(f"[catboost score-only] E={E_c11:.6f}", flush=True)

z_cat = np.load(f"{R8}/r10_e2/z_cat.npy")          # r7-like combo (needs pack; eval already scored)
z_lr = np.load(f"{R8}/r10_e2/z_lr.npy")            # full-feature lambdarank (not deployable)

def pair_rank(v):
    r = np.full(len(v), np.nan)
    for slot, g in d.assign(_v=v).groupby("slot", sort=False):
        r[g.index] = rankdata(-g._v.to_numpy(), method="average")
    return r

res = {"lr11": E_lr11, "cat11": E_c11,
       "z_cat": float(per_pair(d, z_cat).E.mean()), "z_lr_full": float(per_pair(d, z_lr).E.mean())}
rc = pair_rank(z_cat); r11 = pair_rank(z_lr11); rfull = pair_rank(z_lr)
best = (None, -1.0)
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    z = -(w * rc + (1 - w) * r11)
    E = float(per_pair(d, z).E.mean())
    print(f"[blend r7 x lr11] w_r7={w} E={E:.6f}", flush=True)
    if E > best[1]:
        best = ("r7+lr11", w, E)
for name, rk in [("r7+lrfull", rfull)]:
    for w in [0.3, 0.4, 0.5]:
        z = -(w * rc + (1 - w) * rk)
        E = float(per_pair(d, z).E.mean())
        print(f"[blend {name}] w={w} E={E:.6f}", flush=True)
res["blend"] = {"kind": best[0], "w": best[1], "E": best[2]}
np.save(OUT / "z_lr11.npy", z_lr11); np.save(OUT / "z_c11.npy", z_c11)
json.dump(res, open(OUT / "r10_e3.json", "w"), indent=2)
print(json.dumps(res, indent=2))
