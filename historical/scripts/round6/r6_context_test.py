"""Round-6: ChatGPT context-interaction features on top of the r5b rerank (dev, seed 4)."""
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
P["u0"] = np.log(np.clip(P.sc_r5 * poisson.cdf(3, cum) * np.exp(-0.25 * P.groupby("sl").ts.rank(pct=True)), 1e-9, None))
print("base trained", round(time.time() - t0, 1), flush=True)

# ---- context features (ChatGPT ContextFeatures 'context' mode, compact version)
Zfeat = [c for c in X.columns if c.startswith("z_")]
Zv = X[Zfeat].values.astype(np.float64)
wctx = np.clip(P.sc_r5.values, 0, 1) ** 2
codes, _ = pd.factorize(sl, sort=False)
total = np.bincount(codes, weights=wctx, minlength=codes.max() + 1)
sums = np.zeros((len(total), Zv.shape[1]))
np.add.at(sums, codes, Zv * wctx[:, None])
mass = np.maximum(0, total[codes] - wctx)
CTX = np.divide(sums[codes] - Zv * wctx[:, None], mass[:, None], out=np.zeros_like(Zv), where=mass[:, None] > 1e-10)
# pick top-8 semantic interaction dims from r5 fam model importance
imp = None
for fi, fm in enumerate(FAMS):
    b = lgb.Booster(model_file=f"{OUT}/r5fam_{fi}_{fm}_f0.txt")
    g_ = pd.Series(b.feature_importance("gain"), index=b.feature_name())
    imp = g_ if imp is None else imp + g_
top8 = [c for c in imp.sort_values(ascending=False).index if c in Zfeat][:8]
idx8 = [Zfeat.index(c) for c in top8]
cross = (Zv[:, idx8, None] * CTX[:, None, idx8]).reshape(len(Zv), len(idx8) ** 2)
diff = Zv - CTX
massc = np.log1p(mass)[:, None]
CTXBLOCK = np.concatenate([cross, diff, massc], axis=1)
ctx_names = [f"x_{a}_{b}" for a in top8 for b in top8] + [f"d_{c}" for c in Zfeat] + ["ctx_mass"]
print("context block", CTXBLOCK.shape, flush=True)

feat_cols = [c for c in X.columns if c.startswith(("ev_", "wit_", "o_", "DS_", "S_", "Q_")) and not c.startswith("z_")]
feat_cols = list(dict.fromkeys(feat_cols))[:120]
Fv = X[feat_cols].values.astype(np.float64)


def make_std(M):
    mu, sd = M.mean(0), M.std(0) + 1e-6
    return np.clip((M - mu) / sd, -6, 6)


Fv = make_std(Fv)
Cv = make_std(CTXBLOCK)
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


# candidate pool: top-12 by u0 -> same as rerank setup
cand_idx = []
for s_, idxs in idx_by_pair.items():
    idxs = np.array(idxs)
    pool = idxs[np.argsort(-P.u0.values[idxs])][:TOPK]
    cand_idx.append(pool)
cand_idx = np.concatenate(cand_idx)
cand_mask = np.zeros(len(P), bool)
cand_mask[cand_idx] = True


def build_pairs(pool_mask):
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
    return rows


rows = build_pairs(cand_mask)
I = np.array([x[0] for x in rows]); J = np.array([x[1] for x in rows]); W = np.array([x[2] for x in rows])
fold_i = fold[I]


def fit_eval(Dm, label):
    base_delta = P.u0.values[I] - P.u0.values[J]

    def loss(beta, Dmm, bd, wt):
        return np.mean(wt * np.log1p(np.exp(-(bd + Dmm @ beta)))) + L2 * 0.5 * np.sum(beta ** 2) / len(beta)

    lin = np.zeros(len(P))
    for f in range(5):
        tr = fold_i != f
        r = minimize(loss, np.zeros(Dm.shape[1]), args=(Dm[tr], base_delta[tr], W[tr]),
                     method="L-BFGS-B", options={"maxiter": 400})
        lin[fold == f] = Dm[fold == f] @ r.x
    best = {}
    for s in [0.25, 0.5, 0.75]:
        u = P.u0.values + s * lin
        aps = []
        for s_, idxs in idx_by_pair.items():
            idxs = np.array(idxs)
            ordr = idxs[np.argsort(-u[idxs])][:5]
            aps.append(ap5_flags(P.ev.values[ordr].astype(int), int(P.ev.values[idxs].sum())))
        best[f"scale{s}"] = round(float(np.mean(aps)), 4)
    print(label, json.dumps(best), flush=True)
    return best


res = {}
res["linear_rerank"] = fit_eval(Fv[I] - Fv[J], "linear")
res["context_rerank"] = fit_eval(np.concatenate([Fv, Cv], axis=1)[I] - np.concatenate([Fv, Cv], axis=1)[J], "context")

# placebo: shuffle context block across pairs
rng = np.random.RandomState(7)
pair_of_row = {i: s_ for s_, idxs in idx_by_pair.items() for i in idxs}
pairs = list(idx_by_pair)
perm = rng.permutation(len(pairs))
remap = {pairs[i]: pairs[perm[i]] for i in range(len(pairs))}
Csh = np.zeros_like(Cv)
for i in range(len(Cv)):
    Csh[i] = Cv[idx_by_pair[remap[pair_of_row[i]]][0]]
res["context_shuffled"] = fit_eval(np.concatenate([Fv, Csh], axis=1)[I] - np.concatenate([Fv, Csh], axis=1)[J], "shuffled")
print(json.dumps(res, indent=2))
json.dump(res, open(f"{DST}/r6_context_test.json", "w"), indent=2)
