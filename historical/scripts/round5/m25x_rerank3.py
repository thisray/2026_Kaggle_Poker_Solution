"""Round-5 E6: AP@5-weighted comparative rerank on top-12 candidates over the family-routed base.

L_ij = |dAP_ij| * log(1 + exp(-[(u0_i-u0_j) + beta.(x_i-x_j)]))  with L2 shrinkage and pair balancing.
u0 = log(decoded famsep score); final ranking uses u = u0 + beta.x (no second decoder).
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
GEN = "m19w10"
HFMOD = "handfeat2"
TEMPLATE = "m25e1_handfeat2_m19w10_oof.parquet"
TPOW = 2.0
TOPK = 12
L2 = 1.0
USE_R5 = True
SEED = int(os.environ.get("SEED", "4"))
HF = importlib.import_module(HFMOD)
t0 = time.time()

# ---------- base features and family-routed model (same as m25x_fam_r5) ----------
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
E5 = R5.features(h, sa, sb)
X = pd.concat([X, E5], axis=1)
ws = pd.read_parquet(f"{OUT}/{TEMPLATE}").set_index(["sl", "h"]).sc_fam.reindex(
    pd.MultiIndex.from_arrays([P.sl.values, P.h.values])).values
num = [c for c in X.columns if not c.startswith(("z_", "fam_", "gen_", "tpl_", "e_", "ev_", "wit_"))]
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
td = np.empty(len(X), np.float32)
tc = np.empty(len(X), np.float32)
tm = np.empty(len(X), np.float32)
td[order] = dist
tc[order] = cos
tm[order] = SW[gid] - wo
X["tpl_dist"] = td
X["tpl_cos"] = tc
X["tpl_mass"] = tm
y = P.ev.astype(int).values
win = P.win.values
fold = fold_of_pool[sl // 900]
cols = list(X.columns)
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=12, seed=SEED)
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
print("base trained", round(time.time() - t0, 1), flush=True)
cum = P.groupby("sl").sc.cumsum() - P.sc
P["dec"] = P.sc * poisson.cdf(3, cum) * np.exp(-0.25 * P.groupby("sl").ts.rank(pct=True))
P["u0"] = np.log(np.clip(P.dec.values, 1e-9, None))


def ap5_from_flags(flags, n_g=None):
    if n_g is None:
        n_g = int(sum(flags))
    hits = 0
    s = 0.0
    for r, z in enumerate(flags[:5], start=1):
        if z:
            hits += 1
            s += hits / r
    return s / min(5, max(int(n_g), 1))


# pairwise training rows
feat_cols = [c for c in X.columns if c.startswith(("ev_", "wit_", "o_", "DS_", "S_", "Q_")) and not c.startswith("z_")]
feat_cols = list(dict.fromkeys(feat_cols))[:120]
Fv = X[feat_cols].values.astype(np.float64)
mu = Fv.mean(0)
sd = Fv.std(0) + 1e-6
Fv = np.clip((Fv - mu) / sd, -6, 6)
idx_by_pair = {}
for i, s_ in enumerate(P.sl.values):
    idx_by_pair.setdefault(s_, []).append(i)
rows = []
for s_, idxs in idx_by_pair.items():
    idxs = np.array(idxs)
    ordr = idxs[np.argsort(-P.dec.values[idxs])][:TOPK]
    flags = P.ev.values[ordr].astype(int)
    n_g = int(flags.sum())
    base_ap = ap5_from_flags(flags, n_g)
    for a in range(len(ordr)):
        for b in range(len(ordr)):
            if flags[a] == 1 and flags[b] == 0:
                fl = flags.copy()
                fl[a], fl[b] = 0, 1
                new_ap = ap5_from_flags(fl, n_g)
                w_ij = abs(new_ap - base_ap)
                if w_ij <= 0:
                    continue
                rows.append((ordr[a], ordr[b], w_ij))
print("pairwise rows", len(rows), flush=True)
I = np.array([r[0] for r in rows])
J = np.array([r[1] for r in rows])
W = np.array([r[2] for r in rows])
Df = Fv[I] - Fv[J]
base_delta = P.u0.values[I] - P.u0.values[J]
fold_i = fold[I]
res = {}
P["u_lin"] = P.u0.values


def eval_rank(u):
    P["u_tmp"] = u
    aps = []
    for s_, idxs in idx_by_pair.items():
        idxs = np.array(idxs)
        ordr = idxs[np.argsort(-P.u_tmp.values[idxs])][:5]
        flags = P.ev.values[ordr].astype(int)
        aps.append(ap5_from_flags(flags, int(P.ev.values[idxs].sum())))
    return round(float(np.mean(aps)), 4)


res["base_dec_top5"] = eval_rank(P.u0.values)
oof_u = P.u0.values.copy()


def loss(beta, Dm, base_d, wt):
    z = base_d + Dm @ beta
    return np.mean(wt * np.log1p(np.exp(-z))) + L2 * 0.5 * np.sum(beta ** 2) / len(beta)


for f in range(5):
    tr = fold_i != f
    if tr.sum() == 0:
        continue
    Dtr = Df[tr]
    b0 = np.zeros(Dtr.shape[1])
    r = minimize(loss, b0, args=(Dtr, base_delta[tr], W[tr]), method="L-BFGS-B",
                 options={"maxiter": 300})
    beta = r.x
    va_idx = np.where((fold == f))[0]
    oof_u[va_idx] = P.u0.values[va_idx] + Fv[va_idx] @ beta / max(1e-9, 1.0)
    print("fold", f, "beta norm", round(float(np.linalg.norm(beta)), 4), flush=True)
P["u_rr"] = oof_u
res["rerank_oof_top5"] = eval_rank(oof_u)
S = pd.read_parquet(f"{OUT}/seqwithin_oof.parquet").sort_values(["sl","ts"]).reset_index(drop=True)
assert (S.sl.values == sl).all()
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
nn = lg(S.nn_cal.values)
for s in [0.25, 0.5, 0.75, 1.0]:
    res[f"rerank_s{s}"] = eval_rank(P.u0.values + s * (oof_u - P.u0.values))
for a in [0.1, 0.25]:
    res[f"rerank_s0.5_nn{a}"] = eval_rank(P.u0.values + 0.5 * (oof_u - P.u0.values) + a * nn)
    res[f"nn{a}_only"] = eval_rank(P.u0.values + a * nn)
print(json.dumps(res, indent=2))
json.dump(res, open(f"{OUT}/m25rerank_result.json", "w"), indent=2)
