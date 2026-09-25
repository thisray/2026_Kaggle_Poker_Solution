"""Round-6: loss decomposition (recall/selection/order) for r4 model, R5 base, and r5b combo.

Also saves the R5-base and r5b dev OOF scores for later use.
"""
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
from scipy.optimize import minimize
import handdesc as HD
import orient as OR
import withinfeat as WF
import r5feat as R5

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
SEED = 4
TOPK, L2 = 12, 1.0
HF = importlib.import_module("handfeat2")
t0 = time.time()
base = pd.read_parquet(f"{OUT}/m19w10_handscores.parquet")
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
sl, h = P.sl.values, P.h.values
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
E5 = R5.features(h, sa, sb)
X = pd.concat([X, E5], axis=1)
ws = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet").set_index(["sl", "h"]).sc_fam.reindex(
    pd.MultiIndex.from_arrays([sl, h])).values
num = [c for c in X.columns if not c.startswith(("z_", "fam_", "gen_", "tpl_", "e_", "ev_", "wit_"))]
V = X[num].values.astype(np.float64)
V = np.clip((V - V.mean(0)) / (V.std(0) + 1e-6), -5, 5)
w = np.clip(ws, 0, 1) ** 2
order = np.argsort(sl, kind="stable")
g = sl[order]
st = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
cnt = np.diff(np.r_[st, len(g)])
Vo, wo = V[order], w[order]
SW = np.add.reduceat(wo, st)
SV = np.add.reduceat(Vo * wo[:, None], st, axis=0)
gid = np.repeat(np.arange(len(st)), cnt)
Tm = (SV[gid] - Vo * wo[:, None]) / np.maximum(SW[gid] - wo, 1e-6)[:, None]
dist = np.sqrt(((Vo - Tm) ** 2).mean(1))
cos = (Vo * Tm).sum(1) / (np.linalg.norm(Vo, axis=1) * np.linalg.norm(Tm, axis=1) + 1e-6)
td, tc, tm = np.empty(len(X), np.float32), np.empty(len(X), np.float32), np.empty(len(X), np.float32)
td[order], tc[order], tm[order] = dist, cos, SW[gid] - wo
X["tpl_dist"], X["tpl_cos"], X["tpl_mass"] = td, tc, tm
y = P.ev.astype(int).values
win = P.win.values
fold = fold_of_pool[sl // 900]
cols = list(X.columns)
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=12, seed=SEED)
P["sc_r5"] = 0.0
for fm in FAMS:
    mask_f = (P.fam.values == fm)
    oof_f = np.zeros(len(P))
    for f in range(5):
        tr = win & (fold != f) & mask_f
        va = (fold == f) & mask_f
        m = lgb.train(params, lgb.Dataset(X.loc[tr, cols], y[tr]), num_boost_round=600)
        oof_f[va] = m.predict(X.loc[va, cols])
    P.loc[mask_f, "sc_r5"] = oof_f[mask_f]
cum = P.groupby("sl").sc_r5.cumsum() - P.sc_r5
P["u_r5"] = np.log(np.clip(P.sc_r5 * poisson.cdf(3, cum) * np.exp(-0.25 * P.groupby("sl").ts.rank(pct=True)), 1e-9, None))
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
T = pd.read_parquet(f"{OUT}/m25t1_handfeat2_m19w10_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
S = pd.read_parquet(f"{OUT}/seqwithin_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
assert (T.sl.values == sl).all() and (S.sl.values == sl).all()
nn = lg(S.nn_cal.values)

feat_cols = [c for c in X.columns if c.startswith(("ev_", "wit_", "o_", "DS_", "S_", "Q_")) and not c.startswith("z_")]
feat_cols = list(dict.fromkeys(feat_cols))[:120]
Fv = X[feat_cols].values.astype(np.float64)
fmu, fsd = Fv.mean(0), Fv.std(0) + 1e-6
Fv = np.clip((Fv - fmu) / fsd, -6, 6)
idx_by_pair = {}
for i, s_ in enumerate(sl):
    idx_by_pair.setdefault(s_, []).append(i)


def ap5_flags(flags, n_g):
    hits, s = 0, 0.0
    for r, z in enumerate(flags[:5], start=1):
        if z:
            hits += 1
            s += hits / r
    return s / min(5, max(int(n_g), 1))


rows = []
for s_, idxs in idx_by_pair.items():
    idxs = np.array(idxs)
    pool = idxs[np.argsort(-P.u_r5.values[idxs])][:TOPK]
    flags = P.ev.values[pool].astype(int)
    n_g = int(flags.sum())
    bap = ap5_flags(flags, n_g)
    for a in range(len(pool)):
        if flags[a] == 0:
            continue
        for b in range(len(pool)):
            if flags[b] == 1 or a == b:
                continue
            fl = flags.copy()
            fl[a], fl[b] = 0, 1
            w_ij = abs(ap5_flags(fl, n_g) - bap)
            if w_ij > 0:
                rows.append((pool[a], pool[b], w_ij))
I = np.array([x[0] for x in rows]); J = np.array([x[1] for x in rows]); W = np.array([x[2] for x in rows])
Df = Fv[I] - Fv[J]
base_delta = P.u_r5.values[I] - P.u_r5.values[J]
fold_i = fold[I]


def loss(beta, Dm, bd, wt):
    return np.mean(wt * np.log1p(np.exp(-(bd + Dm @ beta)))) + L2 * 0.5 * np.sum(beta ** 2) / len(beta)


lin = np.zeros(len(P))
for f in range(5):
    tr = fold_i != f
    r = minimize(loss, np.zeros(Df.shape[1]), args=(Df[tr], base_delta[tr], W[tr]), method="L-BFGS-B", options={"maxiter": 400})
    lin[fold == f] = Fv[fold == f] @ r.x
P["u_rr"] = P.u_r5.values + 0.5 * lin
P["u_r5b"] = P.u_r5.values + 0.5 * lin + 0.1 * nn

f4 = pd.read_parquet(f"{OUT}/m25famsep4_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
assert (f4.sl.values == sl).all()
cum4 = f4.groupby("sl").sc.cumsum() - f4.sc
P["u_r4model"] = np.log(np.clip(f4.sc * poisson.cdf(3, cum4) * np.exp(-0.25 * f4.groupby("sl").ts.rank(pct=True)), 1e-9, None))


def decompose(u, name):
    aps, ts_oracle, ns = [], [], []
    for s_, idxs in idx_by_pair.items():
        idxs = np.array(idxs)
        n_g = int(P.ev.values[idxs].sum())
        if n_g == 0:
            continue
        ordr = idxs[np.argsort(-u[idxs])][:5]
        flags = P.ev.values[ordr].astype(int)
        a = ap5_flags(flags, n_g)
        h_p = int(flags.sum())
        t = h_p / n_g
        aps.append(a); ts_oracle.append(t); ns.append(n_g)
    A, T = float(np.mean(aps)), float(np.mean(ts_oracle))
    return {"E": round(A, 4), "fixed5_oracle": round(T, 4), "order_loss": round(T - A, 4),
            "n_pairs": len(aps)}


res = {"r4_model": decompose(P.u_r4model.values, "r4"),
       "r5_base": decompose(P.u_r5.values, "r5"),
       "r5_rerank": decompose(P.u_rr.values, "rr"),
       "r5b": decompose(P.u_r5b.values, "r5b")}
P[["sl", "h", "ev", "fam", "ts", "sc_r5", "u_r5", "u_rr", "u_r5b"]].to_parquet(f"{DST}/r6_dev_scores.parquet")
print(json.dumps(res, indent=1))
json.dump(res, open(f"{DST}/r6_decomp.json", "w"), indent=2)
