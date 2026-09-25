"""Predict each pair's evidence-window end (time percentile of the last evidence hand) from the time profile of the
candidate scores, then softly demote candidates beyond the predicted end.  Everything nested by fold."""
import numpy as np, pandas as pd, lightgbm as lgb, itertools
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
t = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"}); d = d.merge(t, on=["slot", "hand_id"])
n = pd.read_csv(f"{A_}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "ts_pct_in_pair", "family"]); d = d.merge(n, on=["slot", "hand_id"])
d["r15"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
# pair-level features from the candidate list (no labels): time profile of the top-scored candidates
rows = []
for sl, g in d.groupby("slot"):
    g = g.sort_values("r15", ascending=False); tp = g.ts_pct_in_pair.values; sc = g.r15.values
    f = {"slot": sl, "fold": g.fold.iloc[0], "fam": g.family.iloc[0]}
    for k in [1, 3, 5, 8]: f[f"t_top{k}_max"] = tp[:k].max(); f[f"t_top{k}_mean"] = tp[:k].mean()
    f["t_5th_by_time_among_top8"] = np.sort(tp[:8])[4]; f["t_4th_by_time_among_top6"] = np.sort(tp[:6])[3]
    f["n_early_top10"] = (tp[:10] < 0.5).sum(); f["score_gap_5_6"] = sc[4] - sc[5]
    ev_t = g[g.ev == 1].ts_pct_in_pair; f["y"] = ev_t.max() if len(ev_t) else np.nan
    rows.append(f)
P = pd.DataFrame(rows); P["fam_c"] = P.fam.astype("category").cat.codes
X = [c for c in P.columns if c.startswith(("t_", "n_", "score_"))] + ["fam_c"]
params = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=15, feature_fraction=0.8, lambda_l2=5.0, verbose=-1, num_threads=2, seed=1)
P["pred"] = np.nan
for f in range(5):
    tr = P.fold != f; m = lgb.train(params, lgb.Dataset(P.loc[tr, X], P.loc[tr, "y"]), num_boost_round=300); P.loc[~tr, "pred"] = m.predict(P.loc[~tr, X])
print("window-end prediction: corr(pred, true) =", round(np.corrcoef(P.pred, P.y)[0, 1], 3), " MAE", round((P.pred - P.y).abs().mean(), 3), " baseline MAE (mean)", round((P.y - P.y.mean()).abs().mean(), 3))
d = d.merge(P[["slot", "pred"]], on="slot")
grid = list(itertools.product([0.0, 0.1, 0.2, 0.3, 0.5, 1.0], [0.0, 0.05, 0.1, 0.2]))
cache = {}
for lam, margin in grid:
    d["tmp"] = d.r15 - lam * np.clip(d.ts_pct_in_pair - (d.pred + margin), 0, None)
    cache[(lam, margin)] = d.groupby("slot").apply(lambda g: ap5(g, "tmp"))
meta = d.groupby("slot").fold.first(); tot = []; ch = []
for f in range(5):
    trs = meta.index[meta != f]; tes = meta.index[meta == f]; b = max(grid, key=lambda g: cache[g].loc[trs].mean()); ch.append(b); tot += list(cache[b].loc[tes])
print("E r15", round(cache[(0.0, 0.0)].mean(), 6), " nested window-demotion", round(np.mean(tot), 6), " delta", round(np.mean(tot) - cache[(0.0, 0.0)].mean(), 6), " chosen", ch)
print("optimistic best", max(grid, key=lambda g: cache[g].mean()), round(max(cache[g].mean() for g in grid) - cache[(0.0, 0.0)].mean(), 6))
# oracle with the true window end (upper bound of this mechanism)
d = d.merge(P[["slot", "y"]], on="slot"); d["tmp"] = d.r15 - 10 * (d.ts_pct_in_pair > d.y + 1e-9)
print("oracle (true window end) on r15:", round(d.groupby("slot").apply(lambda g: ap5(g, "tmp")).mean(), 6))
