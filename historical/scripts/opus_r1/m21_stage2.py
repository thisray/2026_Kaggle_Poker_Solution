"""Stage-2 candidate-hand detector: stage-1 features + OOF score + pair-role oriented features; in-pair hard negatives."""
import numpy as np, pandas as pd, time, lightgbm as lgb, os
from scipy.stats import poisson
import handfeat2 as HF2, handdesc as HD, orient as OR
OUT = HF2.OUT
S1TAG = os.environ.get("S1", "m17"); TAG = os.environ.get("MTAG", "m21"); PROBS = os.environ.get("PROBS", "dec_probs_v1.npy"); HFMOD = os.environ.get("HFMOD", "handfeat2")
HFX = __import__(HFMOD)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
base = pd.read_parquet(f"{OUT}/{S1TAG}_handscores.parquet")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{OUT}/np/s_player.npy")
sl = base.sl.values; h = base.h.values; phase = base.phase.values
plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
key = sl * 2 + phase   # roles estimated within the same phase of the same pair
X = pd.concat([HFX.features(h, sa, sb), HD.descriptors(h, sa, sb, PROBS), OR.features(key, h, sa, sb, base.s.values)], axis=1)
X["s1_logit"] = np.log(np.clip(base.s.values, 1e-6, 1 - 1e-6) / (1 - np.clip(base.s.values, 1e-6, 1 - 1e-6)))
log("features", X.shape)
ev = base.ev.values; pos_bag = base.pos.values & (phase == 0); fold = fold_of_pool[sl // 900]
last = base[base.ev].groupby("sl").ts.max()
before_last = pos_bag & (~ev) & (base.ts.values < pd.Series(sl).map(last).fillna(-1).values)
trainable = ev | (~pos_bag) | before_last
y = ev.astype(int)
HNW = float(os.environ.get("HNW", "1"))
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=30, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=12, seed=21)
oof = np.zeros(len(y))
for f in range(5):
    tr = trainable & (fold != f)
    w = np.where(y[tr] == 1, 1.0, y[tr].sum() / (len(y[tr]) - y[tr].sum()) * 3)
    w = np.where(before_last[tr], w * HNW, w)
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w), num_boost_round=600)
    m.save_model(f"{OUT}/{TAG}_hand_f{f}.txt")
    va = fold == f; oof[va] = m.predict(X[va])
log("trained")
D = base[["sl", "h", "ev", "fam", "pos", "phase", "ts"]].copy(); D["s"] = oof; D["s1"] = base.s.values
P = D[D.pos].sort_values(["sl", "ts"]).copy()
def map5(df, col):
    out = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        out.append((g.fam.iloc[0], ssum / min(5, max(len(rel), 1))))
    r = pd.DataFrame(out, columns=["fam", "ap"]); return round(r.ap.mean(), 4), r.groupby("fam").ap.mean().round(4).to_dict()
for col in ["s1", "s"]:
    P["cum"] = P.groupby("sl")[col].cumsum() - P[col]; P[col + "_plt5"] = P[col] * poisson.cdf(4, P.cum)
    print(col, "raw", map5(P, col), "plt5", map5(P, col + "_plt5"), flush=True)
D.to_parquet(f"{OUT}/{TAG}_handscores.parquet")
imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False); print(imp.head(20).round(0).to_string())
