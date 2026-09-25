"""Train and save the r5 deployment artifacts: family-routed models with R5 features + rerank beta."""
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

OUT = os.environ["POKER_WORK_DIR"]
GEN = "m19w10"
HFMOD = "handfeat2"
TEMPLATE = os.environ.get("R5_TEMPLATE", "m25_handfeat2_m19w10_oof.parquet")
TPOW = 2.0
SEED = 4
L2 = 1.0
RERANK_SCALE = 0.5
TOPK = 12
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
E5 = R5.features(h, sa, sb)
X = pd.concat([X, E5], axis=1)
ws = pd.read_parquet(f"{OUT}/{TEMPLATE}").set_index(["sl", "h"]).sc_fam.reindex(
    pd.MultiIndex.from_arrays([P.sl.values, P.h.values])).values
num = [c for c in X.columns if not c.startswith(("z_", "fam_", "gen_", "tpl_", "e_", "ev_", "wit_"))]
V = X[num].values.astype(np.float64)
mu = V.mean(0)
sd = V.std(0) + 1e-6
V = np.clip((V - mu) / sd, -5, 5)
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
        m.save_model(f"{OUT}/r5fam_{FAMS.index(fm)}_{fm}_f{f}.txt")
    P.loc[mask_f, "sc"] = oof_f[mask_f]
    print("saved family", fm, round(time.time() - t0, 1), flush=True)
np.savez(f"{OUT}/r5_template_stats.npz", mu=mu, sd=sd, num=np.array(num))
open(f"{OUT}/r5_feature_cols.txt", "w").write("\n".join(cols))
if os.environ.get("POKER_BUILD_T1") == "1":
    t1_cols = [c for c in cols if not c.startswith(("ev_", "wit_"))]
    t1_params = dict(params, num_threads=8)
    P["t1"] = 0.0
    for f in range(5):
        tr = win & (fold != f)
        va = fold == f
        model = lgb.train(
            t1_params, lgb.Dataset(X.loc[tr, t1_cols], y[tr]), num_boost_round=600
        )
        P.loc[va, "t1"] = model.predict(X.loc[va, t1_cols])
        model.save_model(f"{OUT}/m25t1_handfeat2_fam_m19w10_f{f}.txt")
    P[["sl", "h", "ev", "fam", "ts", "s", "t1"]].rename(
        columns={"t1": "sc_fam"}
    ).to_parquet(f"{OUT}/m25t1_handfeat2_m19w10_oof.parquet", index=False)
    with open(f"{OUT}/m25t1_handfeat2_m19w10_cols.json", "w") as stream:
        json.dump({"fam": t1_cols}, stream)
cum = P.groupby("sl").sc.cumsum() - P.sc
P["dec"] = P.sc * poisson.cdf(3, cum) * np.exp(-0.25 * P.groupby("sl").ts.rank(pct=True))
P["u0"] = np.log(np.clip(P.dec.values, 1e-9, None))

# rerank features and beta
feat_cols = [c for c in X.columns if c.startswith(("ev_", "wit_", "o_", "DS_", "S_", "Q_")) and not c.startswith("z_")]
feat_cols = list(dict.fromkeys(feat_cols))[:120]
Fv = X[feat_cols].values.astype(np.float64)
fmu = Fv.mean(0)
fsd = Fv.std(0) + 1e-6
Fv = np.clip((Fv - fmu) / fsd, -6, 6)
idx_by_pair = {}
for i, s_ in enumerate(sl):
    idx_by_pair.setdefault(s_, []).append(i)


def ap5_flags(flags, n_g):
    hits = 0
    s = 0.0
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
    base_ap = ap5_flags(flags, n_g)
    for a in range(len(pool)):
        if flags[a] == 0:
            continue
        for b in range(len(pool)):
            if flags[b] == 1 or a == b:
                continue
            fl = flags.copy()
            fl[a], fl[b] = 0, 1
            w_ij = abs(ap5_flags(fl, n_g) - base_ap)
            if w_ij > 0:
                rows.append((pool[a], pool[b], w_ij))
I = np.array([r[0] for r in rows])
J = np.array([r[1] for r in rows])
W = np.array([r[2] for r in rows])
Df = Fv[I] - Fv[J]
base_delta = P.u0.values[I] - P.u0.values[J]


def loss(beta, Dm, base_d, wt):
    z = base_d + Dm @ beta
    return np.mean(wt * np.log1p(np.exp(-z))) + L2 * 0.5 * np.sum(beta ** 2) / len(beta)


r = minimize(loss, np.zeros(Df.shape[1]), args=(Df, base_delta, W), method="L-BFGS-B",
             options={"maxiter": 400})
beta = r.x
np.savez(f"{OUT}/r5_rerank_beta.npz", beta=beta, fmu=fmu, fsd=fsd,
         feat_cols=np.array(feat_cols), scale=RERANK_SCALE)
if os.environ.get("POKER_BUILD_T1") == "1":
    lin_oof = np.zeros(len(P), dtype=np.float64)
    fold_i = fold[I]
    for f in range(5):
        tr = fold_i != f
        fit = minimize(
            loss, np.zeros(Df.shape[1]),
            args=(Df[tr], base_delta[tr], W[tr]),
            method="L-BFGS-B", options={"maxiter": 400},
        )
        va = fold == f
        lin_oof[va] = Fv[va] @ fit.x
    P["lin"] = lin_oof
    P["u"] = P.u0 + RERANK_SCALE * P.lin
    selected = P.sort_values(["sl", "u"], ascending=[True, False]).groupby("sl").head(20).index
    dev = P.loc[selected, ["sl", "h", "s", "sc", "t1", "u0", "lin", "ev"]].copy()
    dev = dev.rename(columns={"sl": "slot", "s": "s1"})
    dev["pool"] = dev.slot.to_numpy() // 900
    dev["fold"] = fold[selected]
    dev["m_p"] = P.groupby("sl").ev.sum().reindex(dev.slot).to_numpy()
    hand_index = pd.read_parquet(f"{OUT}/np/hand_index.parquet").set_index("hi")
    player_index = pd.read_parquet(f"{OUT}/np/player_index.parquet").set_index("pi")
    player_map = dict(zip(player_index.player_id, player_index.index))
    local_index = loc.set_index("player_gi")
    labels = pd.read_csv(
        f"{os.environ['POKER_DATA_DIR']}/development_labels.csv", dtype={"pair_id": str}
    )
    low = np.minimum(labels.player_1.map(player_map), labels.player_2.map(player_map))
    high = np.maximum(labels.player_1.map(player_map), labels.player_2.map(player_map))
    labels["slot"] = (
        local_index.pool.loc[low].to_numpy() * 900
        + local_index.local.loc[low].to_numpy() * 30
        + local_index.local.loc[high].to_numpy()
    )
    dev["pair_id"] = dev.slot.map(labels.set_index("slot").pair_id)
    dev["hand_id"] = hand_index.hand_id.reindex(dev.h).to_numpy()
    dev["pair_player_lo"] = player_index.player_id.reindex(plo[selected]).to_numpy()
    dev["pair_player_hi"] = player_index.player_id.reindex(phi[selected]).to_numpy()
    dev["gen_rank_all"] = X.loc[selected, "gen_rank_pct"].to_numpy()
    extras = [
        "o_lost_dr", "o_flow_dr", "tpl_cos", "wit_r2c_max", "o_dir_agree",
        "facing_mx", "S_sur_aggr_act_mx", "tpl_mass", "eq_fold_to_mx",
        "P_pos_mx", "o_net", "flow_mx", "z_o_contrib",
    ]
    missing = sorted(set(extras) - set(X))
    if missing:
        raise ValueError(f"Missing development gameplay extras: {missing}")
    for name in extras:
        dev[name] = X.loc[selected, name].to_numpy()
    if dev[["pair_id", "hand_id", "pair_player_lo", "pair_player_hi"]].isna().any().any():
        raise ValueError("Development candidate IDs are incomplete")
    dev.to_parquet(f"{OUT}/r5_dev_candidates.parquet", index=False)
print("beta norm", round(float(np.linalg.norm(beta)), 4), "rows", len(rows))
print(json.dumps({"families_saved": 3, "beta_norm": round(float(np.linalg.norm(beta)), 4)},
                 indent=1))
