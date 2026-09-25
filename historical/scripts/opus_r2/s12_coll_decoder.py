"""R2-E3: strong TIME-AGNOSTIC collusion detector on the r11 candidate pack (moments + time-free scores), trained on
in-window candidates only (labels complete there; post-window hands are unlabelled collusion/non-collusion), then exact
Poisson-binomial 'first-5 collusion' decoding over the pair's candidates in time order.  Nested evaluation."""
import numpy as np, pandas as pd, itertools, lightgbm as lgb, os, json
from sklearn.isotonic import IsotonicRegression
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"; PK = f"{A}/round8_raw_20260917/dev_pack"
meta = pd.read_csv(f"{PK}/meta.csv"); t = np.load(f"{PK}/tab.npy", mmap_mode="r")
mom = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30).astype(np.float32)
TA = ["sc_r5", "lin_contrib", "nn_contrib", "t1_score", "s1_stage1", "gen_logit", "gen_rank_pct"]
n = pd.read_csv(f"{A}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "hand_ts"])
meta = meta.merge(n, on=["slot", "hand_id"], how="left"); assert meta.hand_ts.notna().all()
r11 = pd.read_parquet(f"{A}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "rs_blend"]]
meta = meta.merge(r11, on=["slot", "hand_id"], how="left"); assert meta.rs_blend.notna().all()
last = meta[meta.ev == 1].groupby("slot").hand_ts.max(); meta["post"] = meta.hand_ts > meta.slot.map(last)
X = np.c_[mom, meta[TA].to_numpy(float)].astype(np.float32); y = meta.ev.to_numpy(int); fold = meta.fold.to_numpy()
print("rows", len(meta), "in-window", int((~meta.post).sum()), "post", int(meta.post.sum()), "feat", X.shape[1], flush=True)
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.3, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=7)
TRAIN_ALL = os.environ.get("TRAIN_ALL", "0") == "1"      # control: train on all candidates (post-window as negatives)
oof = np.zeros(len(meta))
for f in range(5):
    tr = (fold != f) & ((~meta.post.values) | TRAIN_ALL); te = fold == f
    mdl = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=700); oof[te] = mdl.predict(X[te])
meta["p_raw"] = oof
q = np.zeros(len(meta))
for f in range(5):
    tr = (fold != f) & (~meta.post.values); te = fold == f
    q[te] = IsotonicRegression(y_min=1e-4, y_max=0.999, out_of_bounds="clip").fit(oof[tr], y[tr]).predict(oof[te])
meta["q"] = q
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
def Ef(df, col): return float(np.mean([ap5(g, col) for _, g in df.groupby("slot")]))
meta = meta.sort_values(["slot", "hand_ts"]).reset_index(drop=True)
slots = meta.slot.values; starts = np.r_[0, np.flatnonzero(slots[1:] != slots[:-1]) + 1, len(meta)]
def cb_table(qv):
    out = np.zeros((len(qv), 7))
    for a, b in zip(starts[:-1], starts[1:]):
        dist = np.zeros(8); dist[0] = 1.0
        for i in range(a, b):
            out[i] = np.cumsum(dist[:7]); p = qv[i]; nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[7] += dist[7] * p; dist = nd
    return out
CB = cb_table(meta.q.values)
res = {"base_r11": Ef(meta, "rs_blend"), "p_raw_alone": Ef(meta, "p_raw")}
iw = meta[~meta.post]; res["p_raw_alone_inwindow_only(oracle-window)"] = Ef(iw, "p_raw"); res["r11_inwindow_only(oracle-window)"] = Ef(iw, "rs_blend")
for K in [3, 4, 5, 6, 7]:
    meta[f"dec{K}"] = np.log(meta.q) + np.log(np.clip(CB[:, K - 1], 1e-12, 1)); res[f"dec_K{K}"] = Ef(meta, f"dec{K}")
print(json.dumps({k: round(v, 6) for k, v in res.items()}, indent=1), flush=True)
meta["rn"] = meta.groupby("slot").rs_blend.rank(pct=True)
meta["dn"] = {}
grid = list(itertools.product([4, 5, 6, 7], [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]))
cache = {}
for K, w in grid:
    meta["tmp"] = meta.rn + w * meta.groupby("slot")[f"dec{K}"].rank(pct=True)
    cache[(K, w)] = {f: Ef(meta[meta.fold == f], "tmp") for f in range(5)}
bf = {f: Ef(meta[meta.fold == f], "rs_blend") for f in range(5)}
nest = []
for f in range(5):
    best = max(grid, key=lambda g: np.mean([cache[g][x] - bf[x] for x in range(5) if x != f])); nest.append((f, best, round(cache[best][f] - bf[f], 5)))
print("nested rank-blend r11 + decoder:", nest, "mean", round(np.mean([t[2] for t in nest]), 6))
print("optimistic top:", sorted(((g, round(np.mean([cache[g][x] - bf[x] for x in range(5)]), 5)) for g in grid), key=lambda t: -t[1])[:6])
meta.drop(columns=["dn"]).to_parquet(f"{OUT}/s12_coll_decoder{'_trainall' if TRAIN_ALL else ''}.parquet")
