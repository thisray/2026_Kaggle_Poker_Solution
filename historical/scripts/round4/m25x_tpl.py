"""Round-4 experiment: template-feature variants + hard-negative mining for the within-pair ranker.

Base: m25e1 harness (reproduces exactly). Env controls:
  TEMPLATE  : parquet name with stage-2 OOF for template weights ("" => use stage-1 s)
  TPOW      : power of the weight (default 2)
  TCOL      : "all" | "sur" (feature subset used for template distance)
  HNM_A     : hard-negative weight multiplier (0=off)
  HNM_TAU   : score threshold for hard negatives
  HNM_SRC   : parquet with OOF scores for HNM (default m25e1...)
"""
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
GEN = os.environ.get("GEN", "m19w10")
HFMOD = os.environ.get("HFMOD", "handfeat2")
DPROBS = os.environ.get("DPROBS", "dec_probs_v1.npy")
TEMPLATE = os.environ.get("TEMPLATE", "")
TPOW = float(os.environ.get("TPOW", "2"))
TCOL = os.environ.get("TCOL", "all")
HNM_A = float(os.environ.get("HNM_A", "0"))
HNM_TAU = float(os.environ.get("HNM_TAU", "0.8"))
HNM_SRC = os.environ.get("HNM_SRC", "m25p2_handfeat2_m19w10_oof.parquet")
RUNTAG = os.environ.get("RUNTAG", "tplx")
t0 = time.time()
HF = importlib.import_module(HFMOD)

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

F = pd.concat([HF.features(h, sa, sb), HD.descriptors(h, sa, sb, DPROBS)], axis=1)
O = OR.features(sl * 2, h, sa, sb, P.s.values)
Z = WF.pair_z(pd.concat([F, O], axis=1), sl.astype(np.int64), list(F.columns) + list(O.columns))
X = pd.concat([F, O, Z], axis=1)
X["gen_logit"] = np.log(np.clip(P.s.values, 1e-6, 1 - 1e-6) / (1 - np.clip(P.s.values, 1e-6, 1 - 1e-6)))
X["gen_rank_pct"] = P.groupby("sl").s.rank(pct=True).values
for i, fm in enumerate(["directed_transfer", "soft_play", "coordinated_isolation"]):
    X[f"fam_{i}"] = (P.fam.values == fm).astype(np.float32)

if TEMPLATE:
    ws = pd.read_parquet(f"{OUT}/{TEMPLATE}").set_index(["sl", "h"]).sc_fam.reindex(
        pd.MultiIndex.from_arrays([P.sl.values, P.h.values])).values
else:
    ws = P.s.values.astype(np.float64)

if TEMPLATE or TPOW != 99:
    num = [c for c in X.columns if not c.startswith(("z_", "fam_", "gen_", "tpl_"))]
    if TCOL == "sur":
        num = [c for c in num if c.startswith(("S_", "Q_"))]
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
    tpl_dist = np.empty(len(P))
    tpl_cos = np.empty(len(P))
    tpl_mass = np.empty(len(P))
    tpl_dist[order] = dist
    tpl_cos[order] = cos
    tpl_mass[order] = SW[gid] - wo
    X["tpl_dist"] = tpl_dist.astype(np.float32)
    X["tpl_cos"] = tpl_cos.astype(np.float32)
    X["tpl_mass"] = tpl_mass.astype(np.float32)
    print("template features added; weights from", TEMPLATE if TEMPLATE else "stage1 s", "pow", TPOW, "cols", TCOL, len(num), flush=True)

y = P.ev.astype(int).values
win = P.win.values.copy()
fold = fold_of_pool[sl // 900]
wts = np.ones(len(P))
if HNM_A > 0:
    hs = pd.read_parquet(f"{OUT}/{HNM_SRC}").set_index(["sl", "h"]).sc_fam.reindex(
        pd.MultiIndex.from_arrays([P.sl.values, P.h.values])).values
    hn = (hs > HNM_TAU) & (y == 0) & win
    wts = np.where(hn, HNM_A, 1.0)
    print("hard negatives upweighted:", int(hn.sum()), "A", HNM_A, flush=True)

print(HFMOD, "features", X.shape, round(time.time() - t0, 1), flush=True)


def map5(df, col):
    out = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev])
        top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0
        ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel:
                hits += 1
                ssum += hits / (i + 1)
        out.append((g.fam.iloc[0], ssum / min(5, max(len(rel), 1))))
    r = pd.DataFrame(out, columns=["fam", "ap"])
    return round(r.ap.mean(), 4), r.groupby("fam").ap.mean().round(4).to_dict()


params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=8, seed=int(os.environ.get("SEED", "4")))
cols = list(X.columns)
oof = np.zeros(len(P))
for f in range(5):
    tr = win & (fold != f)
    va = fold == f
    m = lgb.train(params, lgb.Dataset(X.loc[tr, cols], y[tr], weight=wts[tr]), num_boost_round=600)
    oof[va] = m.predict(X.loc[va, cols])
    m.save_model(f"{OUT}/m25{RUNTAG}_{HFMOD}_f{f}.txt")
P["sc"] = oof
P["cum"] = P.groupby("sl")["sc"].cumsum() - P["sc"]
P["x4"] = P["sc"] * poisson.cdf(4, P["cum"])
P["x3"] = P["sc"] * poisson.cdf(3, P["cum"]) * np.exp(-0.25 * P.groupby("sl").ts.rank(pct=True))
m_raw = map5(P, "sc")
m_x4 = map5(P, "x4")
m_x3 = map5(P, "x3")
print(f"RESULT {RUNTAG} TEMPLATE={TEMPLATE} TPOW={TPOW} TCOL={TCOL} HNM={HNM_A}")
print("raw", m_raw[0], "fam", m_raw[1])
print("x4 ", m_x4[0], "fam", m_x4[1])
print("x3 ", m_x3[0], "fam", m_x3[1])
json.dump({"raw": m_raw, "x4": m_x4, "x3": m_x3, "template": TEMPLATE, "tpow": TPOW,
           "tcol": TCOL, "hnm_a": HNM_A, "n_rows": int(len(P))},
          open(f"{OUT}/m25{RUNTAG}_{HFMOD}_result.json", "w"), indent=2)
P[["sl", "h", "ev", "fam", "ts", "s", "sc"]].to_parquet(f"{OUT}/m25{RUNTAG}_{HFMOD}_oof.parquet")
