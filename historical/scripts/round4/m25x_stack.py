"""Round-4 experiment: stacked multi-view features + tail mask for the within-pair ranker.

Same folds/params as m25_within_exp.py (opus worker). Read-only on existing artifacts.
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
STACK = os.environ.get("STACK", "")          # e.g. "nn,e5,m21a" or "all" or ""
TAILW = os.environ.get("TAILW", "0")         # weight for tail-segment negatives (0 = off)
RUNTAG = os.environ.get("RUNTAG", "stack")
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

VIEWS = {
    "t1": "m25t1_handfeat2_m19w10_oof.parquet",
    "e1": "m25e1_handfeat2_m19w10_oof.parquet",
    "e2": "m25e2_handfeat4_m19w10_oof.parquet",
    "e4": "m25e4_handfeat4_m19w10_oof.parquet",
    "e5": "m25e5_handfeat5_m19w10_oof.parquet",
    "p1": "m25p1_handfeat2_m19w10_oof.parquet",
    "p2": "m25p2_handfeat2_m19w10_oof.parquet",
    "p3": "m25p3_handfeat2_m19w10_oof.parquet",
    "m21a": "m25_handfeat2_m21a_oof.parquet",
    "nb": "m25nb_handfeat2_m19w10_oof.parquet",
}
stack_names = [] if STACK in ("", "none") else (list(VIEWS) + ["nn"] if STACK == "all" else STACK.split(","))
if stack_names:
    for nm in stack_names:
        if nm == "nn":
            S = pd.read_parquet(f"{OUT}/seqwithin_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
            assert (S.sl.values == sl).all() and (S.h.values == h).all()
            X["st_nn"] = S.nn_cal.values
            X["st_nn_rk"] = pd.Series(S.nn_cal.values).groupby(sl).rank(pct=True).values
        else:
            d = pd.read_parquet(f"{OUT}/{VIEWS[nm]}").sort_values(["sl", "ts"]).reset_index(drop=True)
            assert (d.sl.values == sl).all() and (d.h.values == h).all()
            X[f"st_{nm}"] = d.sc_fam.values
            X[f"st_{nm}_rk"] = pd.Series(d.sc_fam.values).groupby(sl).rank(pct=True).values
    print("stacked:", stack_names, flush=True)

y = P.ev.astype(int).values
win = P.win.values.copy()
fold = fold_of_pool[sl // 900]
wts = np.ones(len(P))
if float(TAILW) > 0:
    ev_last = P[P.ev].groupby("sl").ts.max()
    n_ev = P[P.ev].groupby("sl").size()
    short = n_ev[n_ev < 5].index
    tail = (~win) & P.sl.isin(short).values
    win = win | tail
    wts = np.where(tail, float(TAILW), 1.0)
    print("tail rows:", int(tail.sum()), "short pairs:", len(short), flush=True)

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
print(f"RESULT {RUNTAG} stack={STACK} tailw={TAILW}")
print("raw", m_raw[0], "fam", m_raw[1])
print("x4 ", m_x4[0], "fam", m_x4[1])
print("x3 ", m_x3[0], "fam", m_x3[1])
json.dump({"raw": m_raw, "x4": m_x4, "x3": m_x3, "stack": stack_names, "tailw": TAILW,
           "n_rows": int(len(P))},
          open(f"{OUT}/m25{RUNTAG}_{HFMOD}_result.json", "w"), indent=2)
P[["sl", "h", "ev", "fam", "ts", "s", "sc"]].to_parquet(f"{OUT}/m25{RUNTAG}_{HFMOD}_oof.parquet")
