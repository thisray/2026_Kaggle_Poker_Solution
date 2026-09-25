"""Round-6: same-hand alternative-partner contrast features (ChatGPT relational route) on dev."""
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import handdesc as HD
import orient as OR
import withinfeat as WF
import r5feat as R5

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
SEED = int(os.environ.get("SEED", "4"))
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
P["u0"] = np.log(np.clip(P.sc_r5 * poisson.cdf(3, cum) * np.exp(-0.25 * P.groupby("sl").ts.rank(pct=True)), 1e-9, None))
print("base trained", round(time.time() - t0, 1), flush=True)

# ---- alternative-partner contrast features for candidate rows
idx_by_pair = {}
for i, s_ in enumerate(sl):
    idx_by_pair.setdefault(s_, []).append(i)
cand_idx = []
for s_, idxs in idx_by_pair.items():
    idxs = np.array(idxs)
    cand_idx.append(idxs[np.argsort(-P.u0.values[idxs])][:TOPK])
cand_idx = np.concatenate(cand_idx)
CH = len(cand_idx)
hh = h[cand_idx]
pairs = [(a, b) for a in range(6) for b in range(a + 1, 6)]
alt_h = np.repeat(hh, len(pairs))
alt_a = np.tile([p[0] for p in pairs], CH)
alt_b = np.tile([p[1] for p in pairs], CH)
A5 = R5.features(alt_h, alt_a, alt_b)
print("alt features", A5.shape, round(time.time() - t0, 1), flush=True)
SYM = ["ev_partner_concession", "ev_both_aggressive", "ev_outsider_fold_after_pair_pressure",
       "ev_total_pair_aggr", "ev_total_out_aggr", "wit_r2c_max", "wit_c2f_sum", "wit_f2c_sum",
       "ev_first_pressure_pair", "ev_middle_hu_transition"]
Av = A5[SYM].values.reshape(CH, 15, len(SYM))
# focal position
foc_a = sa[cand_idx]
foc_b = sb[cand_idx]
focal_pos = np.array([pairs.index((min(a, b), max(a, b))) for a, b in zip(foc_a, foc_b)])
overlap = np.zeros((CH, 15), bool)
disjoint = np.zeros((CH, 15), bool)
for i in range(CH):
    fa, fb = foc_a[i], foc_b[i]
    for j, (x, y) in enumerate(pairs):
        s1 = {x, y}
        shared = len(s1 & {fa, fb})
        overlap[i, j] = shared == 1
        disjoint[i, j] = shared == 0
foc = Av[np.arange(CH), focal_pos, :]
ov_max = np.full_like(foc, 0.0)
ov_mean = np.full_like(foc, 0.0)
dj_max = np.full_like(foc, 0.0)
for i in range(CH):
    ov = Av[i][overlap[i]]
    dj = Av[i][disjoint[i]]
    ov_max[i] = ov.max(0) if len(ov) else 0
    ov_mean[i] = ov.mean(0) if len(ov) else 0
    dj_max[i] = dj.max(0) if len(dj) else 0
CT = np.concatenate([foc - ov_max, foc - ov_mean, foc - dj_max], axis=1)
ct_names = [f"ct_ovmax_{c}" for c in SYM] + [f"ct_ovmean_{c}" for c in SYM] + [f"ct_djmax_{c}" for c in SYM]
print("contrast", CT.shape, flush=True)

# discriminative check on candidates
evc = P.ev.values[cand_idx]
auc = {}
for j, nm in enumerate(ct_names):
    x = CT[:, j]
    if x.std() > 1e-9:
        auc[nm] = round(float(roc_auc_score(evc, x)), 4)
res = {"contrast_auc_top12": dict(sorted(auc.items(), key=lambda kv: -abs(kv[1] - 0.5))[:8])}

# add contrast to rerank features
feat_cols = [c for c in X.columns if c.startswith(("ev_", "wit_", "o_", "DS_", "S_", "Q_")) and not c.startswith("z_")]
feat_cols = list(dict.fromkeys(feat_cols))[:120]
Fv = X[feat_cols].values.astype(np.float64)
mu, sd = Fv.mean(0), Fv.std(0) + 1e-6
Fv = np.clip((Fv - mu) / sd, -6, 6)
cCT = np.zeros((len(P), CT.shape[1]))
cCT[cand_idx] = CT
cCT = np.clip((cCT - cCT[cand_idx].mean(0)) / (cCT[cand_idx].std(0) + 1e-6), -6, 6)


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
    pool = idxs[np.argsort(-P.u0.values[idxs])][:TOPK]
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
fold_i = fold[I]


def fit_eval(All, label):
    Dm = All[I] - All[J]
    base_delta = P.u0.values[I] - P.u0.values[J]

    def loss(beta, Dmm, bd, wt):
        return np.mean(wt * np.log1p(np.exp(-(bd + Dmm @ beta)))) + L2 * 0.5 * np.sum(beta ** 2) / len(beta)

    lin = np.zeros(len(P))
    for f in range(5):
        tr = fold_i != f
        r = minimize(loss, np.zeros(Dm.shape[1]), args=(Dm[tr], base_delta[tr], W[tr]),
                     method="L-BFGS-B", options={"maxiter": 400})
        lin[fold == f] = All[fold == f] @ r.x
    out = {}
    for s in [0.25, 0.5, 0.75]:
        u = P.u0.values + s * lin
        aps = []
        for s_, idxs in idx_by_pair.items():
            idxs = np.array(idxs)
            ordr = idxs[np.argsort(-u[idxs])][:5]
            aps.append(ap5_flags(P.ev.values[ordr].astype(int), int(P.ev.values[idxs].sum())))
        out[f"scale{s}"] = round(float(np.mean(aps)), 4)
    print(label, json.dumps(out), flush=True)
    return out


res["rerank_base"] = fit_eval(Fv, "base")
res["rerank_ct"] = fit_eval(np.concatenate([Fv, cCT], axis=1)[:], "with_contrast")
print(json.dumps(res, indent=2))
json.dump(res, open(f"{DST}/r6_altpartner_seed{SEED}.json", "w"), indent=2)
