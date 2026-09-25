"""Explicit 'how many planted-looking hands came earlier' features for the within-pair ranker (nested LambdaRank):
counts of EARLIER / LATER candidates whose time-agnostic collusion probability (s12 p_raw, OOF) exceeds thresholds,
cumulative p_raw before, and time percentile; compared against the same LambdaRank without them and the fixed r15 blend."""
import numpy as np, pandas as pd, lightgbm as lgb
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend", "u_r5b"]]
t = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"}); d = d.merge(t, on=["slot", "hand_id"])
s12 = pd.read_parquet(f"{OUT}/s12_coll_decoder.parquet")[["slot", "hand_id", "p_raw", "hand_ts"]]; d = d.merge(s12, on=["slot", "hand_id"])
fam = pd.read_csv(f"{A_}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family"]); d = d.merge(fam, on=["slot", "hand_id"])
d = d.sort_values(["slot", "hand_ts"]).reset_index(drop=True)
R = lambda c: d.groupby("slot")[c].rank(pct=True)
d["r15"] = 0.6 * R("tab") + 0.4 * R("rs_blend"); d["r_r15"] = R("r15"); d["r_tab"] = R("tab"); d["r_rs"] = R("rs_blend"); d["r_praw"] = R("p_raw")
d["tpct"] = R("hand_ts"); d["fam_dt"] = (d.family == "directed_transfer").astype(int); d["fam_sp"] = (d.family == "soft_play").astype(int)
for thr in [0.3, 0.5, 0.8]:
    hi = (d.p_raw > thr).astype(int)
    d[f"n_before_{thr}"] = d.assign(x=hi).groupby("slot").x.cumsum() - hi
    d[f"n_after_{thr}"] = d.assign(x=hi).groupby("slot").x.transform("sum") - d.assign(x=hi).groupby("slot").x.cumsum()
d["cum_praw_before"] = d.groupby("slot").p_raw.cumsum() - d.p_raw
# 'order among strong candidates': rank of time among the pair's top-8 by r15
d["top8"] = d.groupby("slot").r15.rank(ascending=False, method="first") <= 8
d["time_rank_in_top8"] = np.where(d.top8, d[d.top8].groupby("slot").hand_ts.rank().reindex(d.index).fillna(0), 0)
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
params = dict(objective="lambdarank", learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, feature_fraction=0.9, lambda_l2=10.0, lambdarank_truncation_level=5, verbose=-1, num_threads=2, seed=3)
def nested(cols, rounds=150, seeds=(3, 5, 7)):
    tot = np.zeros(len(d))
    for sd in seeds:
        p = dict(params); p["seed"] = sd
        for f in range(5):
            tr = (d.fold != f).values; te = (d.fold == f).values
            dt = d[tr].sort_values("slot", kind="mergesort"); grp = dt.groupby("slot", sort=False).size().values
            m = lgb.train(p, lgb.Dataset(dt[cols], dt.ev, group=grp), num_boost_round=rounds); tot[te] += m.predict(d.loc[te, cols]) / len(seeds)
    d["tmp"] = tot; return d.groupby("slot").apply(lambda g: ap5(g, "tmp"))
base_cols = ["r_r15", "r_tab", "r_rs", "fam_dt", "fam_sp"]
cnt_cols = ["tpct", "r_praw", "cum_praw_before", "time_rank_in_top8"] + [c for c in d.columns if c.startswith(("n_before_", "n_after_"))]
e_fix = d.groupby("slot").apply(lambda g: ap5(g, "r15")); e_ctl = nested(base_cols); e_new = nested(base_cols + cnt_cols)
fam_of = d.groupby("slot").family.first(); meta = d.groupby("slot").fold.first()
print(f"E fixed r15 {e_fix.mean():.6f} | lambdarank control {e_ctl.mean():.6f} | + count/time features {e_new.mean():.6f}  (delta vs control {e_new.mean() - e_ctl.mean():+.6f}, vs fixed {e_new.mean() - e_fix.mean():+.6f})")
print("per fold (fixed, new):", [(round(e_fix[meta == f].mean(), 4), round(e_new[meta == f].mean(), 4)) for f in range(5)])
print("per family (fixed -> new):", {f: (round(e_fix[fam_of == f].mean(), 4), round(e_new[fam_of == f].mean(), 4)) for f in fam_of.unique()})
