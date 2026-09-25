"""Round-12 follow-up: truncation level sweep for the Lambdarank + blends."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata

CODE = "/home/thisray/projects/260916_Kaggle_Poker_workers/round11-research-20260918/code"
R8CODE = "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917/code"
sys.path.insert(0, CODE); sys.path.insert(0, R8CODE)
from r11_bridge_simple import load_pack, build_x, dense, SCORES, predict_one  # noqa: E402
from metrics import per_pair  # noqa: E402

S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
OUT = S / "r12_trunc"; OUT.mkdir(parents=True, exist_ok=True)

d, mom = load_pack(f"{R8}/dev_pack")
y = d.ev.to_numpy(int); folds = sorted(d.fold.unique())
raw = dense(d, SCORES); mu = raw.mean(0); sd = raw.std(0) + 1e-9
x = build_x(d, mom, mu, sd)
order = np.argsort(d.slot.to_numpy(), kind="stable")
ds = d.iloc[order]; xs = x[order]; ys = y[order]

base = dict(objective="lambdarank", n_estimators=500, learning_rate=0.04, num_leaves=31,
            min_child_samples=20, colsample_bytree=0.5, reg_lambda=2., label_gain=[0, 1],
            n_jobs=12, verbose=-1, subsample=1., subsample_freq=0, deterministic=True, force_col_wise=True)


def ranker_oof(trunc, seed, rounds=500):
    z = np.full(len(d), np.nan)
    for f in folds:
        tr = ds.fold.to_numpy() != f
        gtr = ds[tr].groupby("slot", sort=False).size().to_numpy()
        m = lgb.LGBMRanker(**base, random_state=seed, lambdarank_truncation_level=trunc)
        m.fit(xs[tr], ys[tr], group=gtr)
        z[order[ds.fold.to_numpy() == f]] = m.predict(xs[ds.fold.to_numpy() == f])
    return z


def E_of(z):
    return float(per_pair(d, z).E.mean())


def rank_of(v):
    out = np.empty(len(v))
    for ix in d.groupby("slot", sort=False).indices.values():
        out[ix] = rankdata(-np.asarray(v)[ix], method="average")
    return out


z_cat = np.full(len(d), np.nan)
for f in folds:
    va = d.fold.to_numpy() == f
    z_cat[va] = predict_one(d.loc[va].reset_index(drop=True), mom[va], S / "ranker_simple" / f"fold_{f}")[0]

res = {"cat": E_of(z_cat)}
zs = {}
for trunc in [10, 12, 15, 20]:
    for seed in [71, 99]:
        key = f"t{trunc}_s{seed}"
        z = ranker_oof(trunc, seed)
        zs[key] = z
        res[key] = E_of(z)
        print(f"{key}: E={res[key]:.6f}", flush=True)

rc = rank_of(z_cat)
best = (None, -1)
for key, z in zs.items():
    for w in [0.0, 0.3, 0.4, 0.5, 0.7]:
        zz = -(w * rc + (1 - w) * rank_of(z))
        e = E_of(zz)
        res[f"blend_{key}_w{w}"] = e
        if e > best[1]:
            best = (f"{key}_w{w}", e)
print("best blend:", best)
res["best"] = best
json.dump(res, open(OUT / "r12_trunc.json", "w"), indent=2)
for k, v in zs.items():
    np.save(OUT / f"z_{k}.npy", v)
np.save(OUT / "z_cat.npy", z_cat)
print("done")
