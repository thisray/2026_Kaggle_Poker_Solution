"""Round-10 E2: LambdaRank (NDCG@5 truncation) + rank blending vs CatBoost combo.

Metrics context: r7 AP=0.6848 with hits@5/m=0.7437 -> oracle-order AP=0.7437,
so ordering loss 0.059 is the target. LightGBM lambdarank with truncation=5
optimizes a position-weighted objective aligned with AP@5.
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
OUT = Path(f"{R8}/r10_e2"); OUT.mkdir(parents=True, exist_ok=True)
SCORES = ['sc_r5', 'u0', 'u_rr', 'u_r5b', 'lin_contrib', 'nn_contrib', 't1_score',
          's1_stage1', 'gen_logit', 'gen_rank_pct', 'rank_u_r5b']
IDS = ['slot', 'pool', 'pair_player_lo', 'pair_player_hi', 'hand_id', 'fold', 'ev', 'm_p',
       'top5_pick', 'h_p']

d = pd.read_csv(f"{R8}/dev_pack/meta.csv")
t = np.load(f"{R8}/dev_pack/tab.npy", mmap_mode='r')
x_mom = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
s = d[SCORES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
x_base = np.c_[x_mom, s]
EXTRA = [c for c in d.columns if c not in IDS + SCORES]
x_extra = np.nan_to_num(d[EXTRA].to_numpy(float), nan=0., posinf=30., neginf=-30.).clip(-30, 30)
# consistency features (recomputed compactly)
det = np.nan_to_num(d[EXTRA].to_numpy(float), nan=0., posinf=0., neginf=0.)
mu, sd = det.mean(0), det.std(0) + 1e-9
zdet = np.clip((det - mu) / sd, -6, 6)
cons = np.zeros((len(d), 3), np.float32)
for slot, g in d.groupby("slot", sort=False):
    idx = g.index.to_numpy(); X = zdet[idx]
    sim = X @ X.T / (np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(X, axis=1)[None, :] + 1e-9)
    np.fill_diagonal(sim, -np.inf)
    for r_ in range(len(idx)):
        v = np.sort(sim[r_])[-3:]
        cons[idx[r_]] = [float(np.mean(v)), float(np.max(sim[r_])), float(np.mean(np.sort(sim[r_])[:5]))]
x_all = np.c_[x_base, x_extra, cons]
y = d.ev.to_numpy(int)
b = d.u_r5b.to_numpy()
folds = sorted(d.fold.unique())
groups = d.groupby("slot", sort=False).size().to_numpy()

# reference: CatBoost combo (variant D-like)
cal = LogisticRegression(C=10, max_iter=1000).fit(b[:, None], y)
sl, ic = float(cal.coef_[0, 0]), float(cal.intercept_[0])
z_cat = np.full(len(d), np.nan)
for f in folds:
    va = d.fold.to_numpy() == f
    m = CatBoostClassifier(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30,
                           random_seed=71, thread_count=16, verbose=False, allow_writing_files=False)
    m.fit(Pool(x_all[~va], y[~va], baseline=sl * b[~va] + ic, weight=1 / d.m_p.to_numpy()[~va]))
    z_cat[va] = b[va] + 0.25 * m.predict(x_all[va], prediction_type='RawFormulaVal') / sl
E_cat = float(per_pair(d, z_cat).E.mean())
print(f"[cat D-like] E={E_cat:.6f}", flush=True)

# lambdarank, group-safe: contiguous by slot
order = np.argsort(d.slot.to_numpy(), kind='stable')
ds = d.iloc[order]; xs = np.nan_to_num(x_all[order], nan=0., posinf=30., neginf=-30.)
ys = y[order]
z_lr = np.full(len(d), np.nan)
for f in folds:
    tr = ds.fold.to_numpy() != f; va = ds.fold.to_numpy() == f
    gtr = ds[tr].groupby('slot', sort=False).size().to_numpy()
    mdl = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.04, num_leaves=31,
                         min_child_samples=20, subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                         lambdarank_truncation_level=5, label_gain=[0, 1], eval_at=[5],
                         random_state=71, n_jobs=16, verbose=-1)
    mdl.fit(xs[tr], ys[tr], group=gtr)
    z_lr[order[va]] = mdl.predict(xs[va])
E_lr = float(per_pair(d, z_lr).E.mean())
print(f"[lambdarank] E={E_lr:.6f}", flush=True)

# pairwise swap-weighted objective via LambdaRank on truncated candidate lists is above;
# now rank blends (per-pair ranks, then average)
def pair_rank(v):
    r = np.full(len(v), np.nan)
    for slot, g in d.assign(_v=v).groupby("slot", sort=False):
        r[g.index] = rankdata(-g._v.to_numpy(), method="average")
    return r

best = (None, -1.0)
res = {"cat": E_cat, "lambdarank": E_lr}
rc = pair_rank(z_cat); rl = pair_rank(z_lr)
for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    z = -(w * rc + (1 - w) * rl)
    E = float(per_pair(d, z).E.mean())
    print(f"[blend w={w}] E={E:.6f}", flush=True)
    if E > best[1]:
        best = (w, E)
res["blend_best"] = {"w_cat": best[0], "E": best[1]}
print(json.dumps(res, indent=2))
json.dump(res, open(OUT / "r10_e2.json", "w"), indent=2)
np.save(OUT / "z_cat.npy", z_cat); np.save(OUT / "z_lr.npy", z_lr)
