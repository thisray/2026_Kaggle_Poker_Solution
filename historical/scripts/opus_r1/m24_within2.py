"""Within-pair ranker v2: hand features + descriptors + oriented roles + within-pair z-scores + general score; binary vs lambdarank."""
import numpy as np, pandas as pd, lightgbm as lgb, os, time
from scipy.stats import poisson
import handfeat2 as HF2, handdesc as HD, orient as OR, withinfeat as WF
OUT = HF2.OUT; GEN = os.environ.get("GEN", "m19w10")
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
F = pd.concat([HF2.features(h, sa, sb), HD.descriptors(h, sa, sb, "dec_probs_v1.npy")], axis=1)
O = OR.features(sl * 2, h, sa, sb, P.s.values)
Z = WF.pair_z(pd.concat([F, O], axis=1), sl.astype(np.int64), list(F.columns) + list(O.columns))
X = pd.concat([F, O, Z], axis=1)
X["gen_logit"] = np.log(np.clip(P.s.values, 1e-6, 1 - 1e-6) / (1 - np.clip(P.s.values, 1e-6, 1 - 1e-6)))
X["gen_rank_pct"] = P.groupby("sl").s.rank(pct=True).values
print("features", X.shape, "time", round(time.time() - t0, 1), flush=True)
y = P.ev.astype(int).values; win = P.win.values; fold = fold_of_pool[sl // 900]
def map5(df, col):
    out = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        out.append((g.fam.iloc[0], ssum / min(5, max(len(rel), 1))))
    r = pd.DataFrame(out, columns=["fam", "ap"]); return round(r.ap.mean(), 4), r.groupby("fam").ap.mean().round(4).to_dict()
variants = {
    "bin_all": (dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=6, seed=4), list(X.columns)),
    "bin_noz": (dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=6, seed=4), [c for c in X.columns if not c.startswith("z_")]),
    "rank_all": (dict(objective="lambdarank", eval_at=[5], lambdarank_truncation_level=20, learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=6, seed=4), list(X.columns)),
}
for name, (params, cols) in variants.items():
    oof = np.zeros(len(P))
    for f in range(5):
        tr = win & (fold != f); va = fold == f
        if params["objective"] == "lambdarank":
            grp = pd.Series(sl[tr]).groupby(sl[tr], sort=False).size().values
            m = lgb.train(params, lgb.Dataset(X.loc[tr, cols], y[tr], group=grp), num_boost_round=500)
        else:
            m = lgb.train(params, lgb.Dataset(X.loc[tr, cols], y[tr]), num_boost_round=600)
        oof[va] = m.predict(X.loc[va, cols])
        m.save_model(f"{OUT}/m24_{name}_{GEN}_f{f}.txt")
    if params["objective"] == "lambdarank":
        oof = 1 / (1 + np.exp(-oof))
    P[name] = oof
    P["cum"] = P.groupby("sl")[name].cumsum() - P[name]
    P["x"] = P[name] * poisson.cdf(4, P.cum); P["e"] = P[name] * np.exp(-0.5 * P.groupby("sl").ts.rank(pct=True))
    P["g"] = np.sqrt(P[name] * P.s); P["cumg"] = P.groupby("sl").g.cumsum() - P.g; P["gx"] = P.g * poisson.cdf(4, P.cumg); P["ge"] = P.g * np.exp(-0.5 * P.groupby("sl").ts.rank(pct=True))
    print(f"{name:9s} raw {map5(P, name)[0]} plt5 {map5(P, 'x')} exp {map5(P, 'e')[0]} | x gen: plt5 {map5(P, 'gx')[0]} exp {map5(P, 'ge')[0]}  t={time.time()-t0:.0f}", flush=True)
P[["sl", "h", "ev", "fam", "ts", "s"] + list(variants)].to_parquet(f"{OUT}/m24_within_{GEN}_oof.parquet")
import json; json.dump({"cols_all": list(X.columns)}, open(f"{OUT}/m24_cols.json", "w"))
