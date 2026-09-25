"""Ordering-only lever: keep the r15 top-5 SET per pair fixed, re-ORDER it with a nested classifier trained on
'is this pick a hit' using pick-level features (score margins, rank, time order among the 5, time pct, time-agnostic
p_raw, family).  Oracle (perfect order of the same set) bounds the gain."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend", "u_r5b", "z_cat", "z_lr"]]
t = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"}); d = d.merge(t, on=["slot", "hand_id"])
s12 = pd.read_parquet(f"{OUT}/s12_coll_decoder.parquet")[["slot", "hand_id", "p_raw", "hand_ts"]]; d = d.merge(s12, on=["slot", "hand_id"])
fam = pd.read_csv(f"{A_}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family"]); d = d.merge(fam, on=["slot", "hand_id"])
R = lambda c: d.groupby("slot")[c].rank(pct=True)
d["r15"] = 0.6 * R("tab") + 0.4 * R("rs_blend"); d["rk"] = d.groupby("slot").r15.rank(ascending=False, method="first")
T = d[d.rk <= 5].copy()
T["t_order5"] = T.groupby("slot").hand_ts.rank(); T["tpct"] = d.loc[T.index].groupby("slot").hand_ts.rank(pct=True) if False else R("hand_ts").loc[T.index]
T["tab_r"] = T.groupby("slot").tab.rank(ascending=False); T["rs_r"] = T.groupby("slot").rs_blend.rank(ascending=False)
T["praw_r"] = T.groupby("slot").p_raw.rank(ascending=False); T["gap_to_6th"] = T.r15 - T.slot.map(d[d.rk == 6].set_index("slot").r15)
T["fam_dt"] = (T.family == "directed_transfer").astype(int); T["fam_sp"] = (T.family == "soft_play").astype(int)
F = ["rk", "r15", "tab_r", "rs_r", "praw_r", "p_raw", "u_r5b", "z_cat", "z_lr", "t_order5", "tpct", "gap_to_6th", "fam_dt", "fam_sp"]
def ap_order(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
T["base_order"] = -T.rk; T["oracle"] = T.ev * 10 - T.rk * 0.01
print("E base order", round(T.groupby("slot").apply(lambda g: ap_order(g, "base_order")).mean(), 6), " oracle order (same set)", round(T.groupby("slot").apply(lambda g: ap_order(g, "oracle")).mean(), 6))
for name, mk in [("logistic", lambda: LogisticRegression(C=1.0, max_iter=3000)), ("lgbm", None)]:
    pred = np.zeros(len(T))
    for f in range(5):
        tr = (T.fold != f).values; te = (T.fold == f).values
        if mk is not None:
            m = mk().fit(T.loc[tr, F], T.loc[tr, "ev"]); pred[te] = m.predict_proba(T.loc[te, F])[:, 1]
        else:
            m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=30, lambda_l2=10.0, feature_fraction=0.9, verbose=-1, num_threads=2, seed=1), lgb.Dataset(T.loc[tr, F], T.loc[tr, "ev"]), num_boost_round=200); pred[te] = m.predict(T.loc[te, F])
    T["new"] = pred
    e_new = T.groupby("slot").apply(lambda g: ap_order(g, "new")); e_base = T.groupby("slot").apply(lambda g: ap_order(g, "base_order"))
    meta = T.groupby("slot").fold.first()
    print(f"{name}: E reordered {e_new.mean():.6f} (delta {e_new.mean() - e_base.mean():+.6f}); per-fold deltas {[round((e_new[meta == f] - e_base[meta == f]).mean(), 4) for f in range(5)]}")
