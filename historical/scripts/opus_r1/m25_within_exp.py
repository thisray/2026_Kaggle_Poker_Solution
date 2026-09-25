"""Fast experiment runner for the within-pair ranker (variants of features, family context, selection scale, capacity)."""
import numpy as np, pandas as pd, lightgbm as lgb, os, time, sys, importlib
from scipy.stats import poisson
import handdesc as HD, orient as OR, withinfeat as WF
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; GEN = os.environ.get("GEN", "m19w10")
HFMOD = os.environ.get("HFMOD", "handfeat2"); HF = importlib.import_module(HFMOD); DPROBS = os.environ.get("DPROBS", "dec_probs_v1.npy")
t0 = time.time()
base = pd.read_parquet(f"{OUT}/{GEN}_handscores.parquet")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{OUT}/np/s_player.npy")
P = base[base.pos & (base.phase == 0)].sort_values(["sl", "ts"]).reset_index(drop=True)
last = P[P.ev].groupby("sl").ts.max(); P["win"] = P.ts <= P.sl.map(last)
sl = P.sl.values; h = P.h.values
plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
F = pd.concat([HF.features(h, sa, sb), HD.descriptors(h, sa, sb, DPROBS)], axis=1)
O = OR.features(sl * 2, h, sa, sb, P.s.values)
Z = WF.pair_z(pd.concat([F, O], axis=1), sl.astype(np.int64), list(F.columns) + list(O.columns))
X = pd.concat([F, O, Z], axis=1)
X["gen_logit"] = np.log(np.clip(P.s.values, 1e-6, 1 - 1e-6) / (1 - np.clip(P.s.values, 1e-6, 1 - 1e-6)))
X["gen_rank_pct"] = P.groupby("sl").s.rank(pct=True).values
for i, fm in enumerate(["directed_transfer", "soft_play", "coordinated_isolation"]): X[f"fam_{i}"] = (P.fam.values == fm).astype(np.float32)
y = P.ev.astype(int).values; win = P.win.values; fold = fold_of_pool[sl // 900]
print(HFMOD, "features", X.shape, round(time.time() - t0, 1), flush=True)
def map5(df, col):
    out = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        out.append((g.fam.iloc[0], ssum / min(5, max(len(rel), 1))))
    r = pd.DataFrame(out, columns=["fam", "ap"]); return round(r.ap.mean(), 4), r.groupby("fam").ap.mean().round(4).to_dict()
base_params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=4)
nofam = [c for c in X.columns if not c.startswith("fam_")]
runs = {
  "base": (base_params, nofam, 600),
  "fam": (base_params, list(X.columns), 600),
  "fam_deep": (dict(base_params, num_leaves=31, min_data_in_leaf=30, learning_rate=0.02), list(X.columns), 1000),
}
sel = os.environ.get("RUNS", "base,fam,fam_deep").split(",")
for name in sel:
    params, cols, rounds = runs[name]
    oof = np.zeros(len(P))
    for f in range(5):
        tr = win & (fold != f); va = fold == f
        m = lgb.train(params, lgb.Dataset(X.loc[tr, cols], y[tr]), num_boost_round=rounds); oof[va] = m.predict(X.loc[va, cols])
        m.save_model(f"{OUT}/m25_{HFMOD}_{name}_{GEN}_f{f}.txt")
    P["sc_" + name] = oof
    res = []
    for sc in [0.5, 1.0, 2.0]:
        P["cum"] = P.groupby("sl")["sc_" + name].cumsum() - P["sc_" + name]; P["x"] = P["sc_" + name] * poisson.cdf(4, P.cum * sc); res.append((f"plt5x{sc}", map5(P, "x")[0]))
    for a in [0.25, 0.5, 1.0]:
        P["e"] = P["sc_" + name] * np.exp(-a * P.groupby("sl").ts.rank(pct=True)); res.append((f"exp{a}", map5(P, "e")[0]))
    P["cum"] = P.groupby("sl")["sc_" + name].cumsum() - P["sc_" + name]; P["x"] = P["sc_" + name] * poisson.cdf(4, P.cum)
    print(f"{name:9s} raw {map5(P, "sc_" + name)[0]} fam-breakdown(plt5) {map5(P, 'x')[1]} | " + " ".join(f"{k}={v}" for k, v in res) + f" t={time.time()-t0:.0f}", flush=True)
P[["sl", "h", "ev", "fam", "ts", "s"] + ["sc_" + n for n in sel]].to_parquet(f"{OUT}/m25_{HFMOD}_{GEN}_oof.parquet")
json_cols = {n: runs[n][1] for n in sel}
import json; json.dump(json_cols, open(f"{OUT}/m25_{HFMOD}_{GEN}_cols.json", "w"))
