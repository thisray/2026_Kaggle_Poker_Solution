"""Round-4: nested decoder selection for the family-routed evidence model."""
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
import handdesc as HD
import orient as OR
import withinfeat as WF

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
GEN = "m19w10"
HFMOD = "handfeat2"
TEMPLATE = "m25e1_handfeat2_m19w10_oof.parquet"
TPOW = 2.0
HF = importlib.import_module(HFMOD)
t0 = time.time()
base = pd.read_parquet(f"{OUT}/{GEN}_handscores.parquet")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet")
members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"):
    members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{OUT}/np/s_player.npy")
P = base[base.pos & (base.phase == 0)].sort_values(["sl", "ts"]).reset_index(drop=True)
last = P[P.ev].groupby("sl").ts.max()
P["win"] = P.ts <= P.sl.map(last)
sl = P.sl.values
h = P.h.values
plo = members[sl // 900, (sl % 900) // 30]
phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1)
sb = np.argmax(sp[h] == phi[:, None], axis=1)
F = pd.concat([HF.features(h, sa, sb), HD.descriptors(h, sa, sb, "dec_probs_v1.npy")], axis=1)
O = OR.features(sl * 2, h, sa, sb, P.s.values)
Z = WF.pair_z(pd.concat([F, O], axis=1), sl.astype(np.int64), list(F.columns) + list(O.columns))
X = pd.concat([F, O, Z], axis=1)
X["gen_logit"] = np.log(np.clip(P.s.values, 1e-6, 1 - 1e-6) / (1 - np.clip(P.s.values, 1e-6, 1 - 1e-6)))
X["gen_rank_pct"] = P.groupby("sl").s.rank(pct=True).values
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
for i, fm in enumerate(FAMS):
    X[f"fam_{i}"] = (P.fam.values == fm).astype(np.float32)
ws = pd.read_parquet(f"{OUT}/{TEMPLATE}").set_index(["sl", "h"]).sc_fam.reindex(
    pd.MultiIndex.from_arrays([P.sl.values, P.h.values])).values
num = [c for c in X.columns if not c.startswith(("z_", "fam_", "gen_", "tpl_"))]
V = X[num].values.astype(np.float64)
V = np.clip((V - V.mean(0)) / (V.std(0) + 1e-6), -5, 5)
w = np.clip(ws, 0, 1) ** TPOW
order = np.argsort(sl, kind="stable")
g = sl[order]
starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
counts = np.diff(np.r_[starts, len(g)])
Vo = V[order]
wo = w[order]
SW = np.add.reduceat(wo, starts)
SV = np.add.reduceat(Vo * wo[:, None], starts, axis=0)
gid = np.repeat(np.arange(len(starts)), counts)
Tm = (SV[gid] - Vo * wo[:, None]) / np.maximum(SW[gid] - wo, 1e-6)[:, None]
dist = np.sqrt(((Vo - Tm) ** 2).mean(1))
cos = (Vo * Tm).sum(1) / (np.linalg.norm(Vo, axis=1) * np.linalg.norm(Tm, axis=1) + 1e-6)
tpl_dist = np.empty(len(P))
tpl_cos = np.empty(len(P))
tpl_mass = np.empty(len(P))
tpl_dist[order] = dist
tpl_cos[order] = cos
tpl_mass[order] = SW[gid] - wo
X["tpl_dist"] = tpl_dist.astype(np.float32)
X["tpl_cos"] = tpl_cos.astype(np.float32)
X["tpl_mass"] = tpl_mass.astype(np.float32)
y = P.ev.astype(int).values
win = P.win.values
fold = fold_of_pool[sl // 900]
cols = list(X.columns)
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=12, seed=4)
P["sc"] = 0.0
for fm in FAMS:
    mask_f = (P.fam.values == fm)
    oof_f = np.zeros(len(P))
    for f in range(5):
        tr = win & (fold != f) & mask_f
        va = (fold == f) & mask_f
        m = lgb.train(params, lgb.Dataset(X.loc[tr, cols], y[tr]), num_boost_round=600)
        oof_f[va] = m.predict(X.loc[va, cols])
    P.loc[mask_f, "sc"] = oof_f[mask_f]
P.to_parquet(f"{OUT}/m25famsep4_oof.parquet")
print("trained", round(time.time() - t0, 1), flush=True)


def map5(df, col):
    aps = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev])
        g = g.sort_values(col, ascending=False)
        top = g.h.values[:5]
        hits = 0
        s = 0.0
        for i, hh in enumerate(top):
            if hh in rel:
                hits += 1
                s += hits / (i + 1)
        aps.append(s / min(5, max(len(rel), 1)))
    return float(np.mean(aps))


S = P.sc.values
cum = P.groupby("sl").sc.cumsum().values - S
tp = P.groupby("sl").ts.rank(pct=True).values
folds = (P.groupby("sl").ngroup() % 5).values
res = {f"K{k}_a{a}": S * poisson.cdf(k, cum) * np.exp(-a * tp) for k in [3, 4, 5, 6] for a in [0.0, 0.25, 0.5]}
out = {name: round(map5(P.assign(y=yv), "y"), 4) for name, yv in res.items()}
nested = []
for f in range(5):
    tr, va = folds != f, folds == f
    Dt, Dv = P[tr].copy(), P[va].copy()
    best, bestv = None, -1.0
    for c in res:
        Dt["y"] = res[c][tr]
        v = map5(Dt, "y")
        if v > bestv:
            bestv, best = v, c
    Dv = Dv.copy()
    Dv["y"] = res[best][va]
    nested.append(map5(Dv, "y"))
out["nested_mean"] = round(float(np.mean(nested)), 4)
out["nested_picks"] = []
# report chosen config per fold
for f in range(5):
    tr = folds != f
    Dt = P[tr].copy()
    best, bestv = None, -1.0
    for c in res:
        Dt["y"] = res[c][tr]
        v = map5(Dt, "y")
        if v > bestv:
            bestv, best = v, c
    out["nested_picks"].append(best)
out["x3_a0.25"] = round(map5(P.assign(y=res["K3_a0.25"]), "y"), 4)
out["x4_a0"] = round(map5(P.assign(y=res["K4_a0.0"]), "y"), 4)
print(json.dumps(out, indent=2))
json.dump(out, open(f"{OUT}/m25famsep4_decoder.json", "w"), indent=2)
