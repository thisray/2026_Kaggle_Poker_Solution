"""Within-pair ranker trained only on positive-pair windows (evidence vs earlier non-evidence), combined with a general detector."""
import numpy as np, pandas as pd, lightgbm as lgb, os, time
from scipy.stats import poisson
import handfeat2 as HF2, handdesc as HD
OUT = HF2.OUT; GEN = os.environ.get("GEN", "m19w10")
t0 = time.time()
base = pd.read_parquet(f"{OUT}/{GEN}_handscores.parquet")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{OUT}/np/s_player.npy")
P = base[base.pos & (base.phase == 0)].copy()
last = P[P.ev].groupby("sl").ts.max(); P["win"] = P.ts <= P.sl.map(last)
sl = P.sl.values; h = P.h.values
plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
X = pd.concat([HF2.features(h, sa, sb), HD.descriptors(h, sa, sb, "dec_probs_v1.npy")], axis=1)
X["gen_logit"] = np.log(np.clip(P.s.values, 1e-6, 1 - 1e-6) / (1 - np.clip(P.s.values, 1e-6, 1 - 1e-6)))
y = P.ev.astype(int).values; win = P.win.values; fold = fold_of_pool[sl // 900]
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4, seed=4)
for use_gen in [False, True]:
    cols = [c for c in X.columns if use_gen or c != "gen_logit"]
    oof = np.zeros(len(P))
    for f in range(5):
        tr = win & (fold != f); va = fold == f
        m = lgb.train(params, lgb.Dataset(X.loc[tr, cols], y[tr]), num_boost_round=500)
        oof[va] = m.predict(X.loc[va, cols])
        if use_gen: m.save_model(f"{OUT}/m23_within_gen_{GEN}_f{f}.txt")
    P["w_gen" if use_gen else "w"] = oof
def map5(df, col):
    out = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        out.append((g.fam.iloc[0], ssum / min(5, max(len(rel), 1))))
    r = pd.DataFrame(out, columns=["fam", "ap"]); return round(r.ap.mean(), 4), r.groupby("fam").ap.mean().round(4).to_dict()
P = P.sort_values(["sl", "ts"])
P["gen"] = P.s
P["prod"] = P.gen * P.w; P["prod_sqrt"] = np.sqrt(P.gen * P.w)
for col in ["gen", "w", "w_gen", "prod", "prod_sqrt"]:
    P["cum"] = P.groupby("sl")[col].cumsum() - P[col]
    P["x"] = P[col] * poisson.cdf(4, P.cum); P["e"] = P[col] * np.exp(-0.5 * P.groupby("sl").ts.rank(pct=True))
    print(f"{col:10s} raw {map5(P, col)[0]}  plt5 {map5(P, 'x')}  exp {map5(P, 'e')}", flush=True)
print("time", time.time() - t0)
