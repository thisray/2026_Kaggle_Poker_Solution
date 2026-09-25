"""Round-12: ranker ensemble sweep (seeds x truncation) on frozen-upstream dev CV."""
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
from r11_bridge_simple import load_pack, build_x, dense, SCORES, fit_one, predict_one  # noqa: E402
from metrics import per_pair  # noqa: E402

S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
OUT = S / "r12_sweep"; OUT.mkdir(parents=True, exist_ok=True)

d, mom = load_pack(f"{R8}/dev_pack")
y = d.ev.to_numpy(int); folds = sorted(d.fold.unique())
raw = dense(d, SCORES); mu = raw.mean(0); sd = raw.std(0) + 1e-9
x = build_x(d, mom, mu, sd)
b0 = d.u_r5b.to_numpy(float)
order = np.argsort(d.slot.to_numpy(), kind="stable")
ds = d.iloc[order]; xs = x[order]; ys = y[order]


def ranker_oof(make_model):
    z = np.full(len(d), np.nan)
    for f in folds:
        tr = ds.fold.to_numpy() != f
        gtr = ds[tr].groupby("slot", sort=False).size().to_numpy()
        m = make_model()
        m.fit(xs[tr], ys[tr], group=gtr)
        z[order[ds.fold.to_numpy() == f]] = m.predict(xs[ds.fold.to_numpy() == f])
    return z


def E_of(z):
    return float(per_pair(d, z).E.mean())


def rank_of(v):
    out = np.empty(len(v))
    for ix in d.groupby("slot", sort=False).indices.values():
        out[ix] = rankdata(-np.asarray(v)[ix], method="average")
    return out  # 1 = best within pair


base = dict(objective="lambdarank", n_estimators=500, learning_rate=0.04, num_leaves=31,
            min_child_samples=20, colsample_bytree=0.5, reg_lambda=2., label_gain=[0, 1],
            n_jobs=12, verbose=-1, subsample=1., subsample_freq=0, deterministic=True, force_col_wise=True)
variants = {}
for seed in [71, 17, 42, 99]:
    variants[f"lr_s{seed}_t5"] = make = (lambda sd: (lambda: lgb.LGBMRanker(**base, random_state=sd, lambdarank_truncation_level=5)))(seed)
for trunc in [8, 12]:
    variants[f"lr_s71_t{trunc}"] = (lambda tr: (lambda: lgb.LGBMRanker(**base, random_state=71, lambdarank_truncation_level=tr)))(trunc)

zs = {}
for name, mk in variants.items():
    z = ranker_oof(mk)
    zs[name] = z
    print(f"{name}: E={E_of(z):.6f}", flush=True)

# CatBoost reference (fold models already saved)
z_cat = np.full(len(d), np.nan)
for f in folds:
    va = d.fold.to_numpy() == f
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    # reuse saved fold model via predict_one on the val rows
    zc = predict_one(d.loc[va].reset_index(drop=True), mom[va], S / "ranker_simple" / f"fold_{f}")[0]
    z_cat[va] = zc
print(f"cat_reference: E={E_of(z_cat):.6f}", flush=True)

res = {"variants": {k: E_of(v) for k, v in zs.items()}, "cat": E_of(z_cat)}
# random-subset ensembles for seed averaging evidence
import itertools
combos = [("lr_s71_t5", "lr_s17_t5"), ("lr_s71_t5", "lr_s17_t5", "lr_s42_t5"),
          ("lr_s71_t5", "lr_s17_t5", "lr_s42_t5", "lr_s99_t5")]
zr = {k: rank_of(v) for k, v in zs.items()}
for combo in combos:
    m = np.mean([zr[k] for k in combo], axis=0)
    e = float(per_pair(d, -m).E.mean())
    res["ens_" + "_".join(combo)] = e
    print(f"ensemble {combo}: E={e:.6f}", flush=True)

best = max(res.items(), key=lambda kv: kv[1])
print("best:", best)
json.dump(res, open(OUT / "r12_sweep.json", "w"), indent=2)
np.save(OUT / "z_cat_ref.npy", z_cat)
for k, v in zs.items():
    np.save(OUT / f"z_{k}.npy", v)
print("done")
