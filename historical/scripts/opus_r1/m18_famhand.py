"""Family-specific candidate-hand detectors with in-pair hard negatives and sequence descriptors; routed by family (true for dev OOF eval)."""
import numpy as np, pandas as pd, time, lightgbm as lgb, os
from sklearn.metrics import average_precision_score
from scipy.stats import poisson
import handfeat as HF, handfeat2 as HF2, pairindex as PI, handdesc as HD
OUT = HF.OUT; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
TAG = os.environ.get("MTAG", "m18"); PROBS = os.environ.get("PROBS", "dec_probs_v1.npy")
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
base = pd.read_parquet(f"{OUT}/m16_handscores.parquet")  # sl, h, s, ev, rk, fam, pos, phase, ts (same universe as m6/m16)
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{OUT}/np/s_player.npy")
sl = base.sl.values; h = base.h.values
plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
X = pd.concat([HF2.features(h, sa, sb), HD.descriptors(h, sa, sb, PROBS)], axis=1); log("features", X.shape)
ev = base.ev.values; pos_bag = base.pos.values & (base.phase.values == 0); fam = base.fam.values; fold = fold_of_pool[sl // 900]
last = base[base.ev].groupby("sl").ts.max()
before_last = pos_bag & (~ev) & (base.ts.values < pd.Series(sl).map(last).fillna(-1).values)
neg_easy = ~pos_bag
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=30, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=12, seed=9)
S = np.zeros((len(base), 3), np.float32)
for fi, fm in enumerate(FAMS):
    y = (ev & (fam == fm)).astype(int)
    trainable = y.astype(bool) | neg_easy | before_last
    for f in range(5):
        tr = trainable & (fold != f)
        w = np.where(y[tr] == 1, 1.0, y[tr].sum() / (len(y[tr]) - y[tr].sum()) * 3)
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w), num_boost_round=500)
        m.save_model(f"{OUT}/{TAG}_{fm}_f{f}.txt")
        va = fold == f; S[va, fi] = m.predict(X[va])
    log(fm, "trained")
D = base[["sl", "h", "ev", "fam", "pos", "phase", "ts"]].copy()
for fi, fm in enumerate(FAMS): D["s_" + fm[:2]] = S[:, fi]
D["s_route"] = np.where(fam == FAMS[0], S[:, 0], np.where(fam == FAMS[1], S[:, 1], np.where(fam == FAMS[2], S[:, 2], S.max(1))))
D["s_max"] = S.max(1)
D["s_m16"] = base.s.values
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
for col in ["s_m16", "s_max", "s_route"]:
    P["cum"] = P.groupby("sl")[col].cumsum() - P[col]
    P[col + "_plt5"] = P[col] * poisson.cdf(4, P.cum)
    P[col + "_exp"] = P[col] * np.exp(-0.5 * P.groupby("sl").ts.rank(pct=True))
    print(col, "raw", map5(P, col), "plt5", map5(P, col + "_plt5"), "exp0.5", map5(P, col + "_exp"), flush=True)
D.to_parquet(f"{OUT}/{TAG}_handscores.parquet")
