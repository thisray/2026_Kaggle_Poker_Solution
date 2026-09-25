"""Nested meta-ranker over existing OOF signals: TabICL (round15), r11 cat/lambdarank/blend, time-agnostic detector p_raw (R2-E3),
time features and the exact first-5 prior.  Compare with the fixed r15 rank blend (0.6 TabICL + 0.4 r11 blend)."""
import numpy as np, pandas as pd, lightgbm as lgb, itertools
from sklearn.linear_model import LogisticRegression
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")
t = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"})
s12 = pd.read_parquet(f"{OUT}/s12_coll_decoder.parquet")[["slot", "hand_id", "p_raw", "q", "hand_ts", "dec4", "dec5"]]
d = d.merge(t, on=["slot", "hand_id"]).merge(s12, on=["slot", "hand_id"]); assert len(d) == 7440, len(d)
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
def Ef(df, col): return float(np.mean([ap5(g, col) for _, g in df.groupby("slot")]))
R = lambda c: d.groupby("slot")[c].rank(pct=True)
for c in ["tab", "rs_blend", "z_cat", "z_lr", "p_raw", "u_r5b", "dec4", "dec5"]: d["r_" + c] = R(c)
d["tpct"] = d.groupby("slot").hand_ts.rank(pct=True)
d["r15"] = 0.6 * d.r_tab + 0.4 * d.r_rs_blend
print("E: r11", round(Ef(d, "rs_blend"), 6), " tabicl", round(Ef(d, "tab"), 6), " r15 fixed blend", round(Ef(d, "r15"), 6), " p_raw", round(Ef(d, "p_raw"), 6))
F = ["r_tab", "r_rs_blend", "r_z_cat", "r_z_lr", "r_p_raw", "r_u_r5b", "r_dec4", "tpct"]
res = {}
# (1) nested logistic on ranks
oof = np.zeros(len(d))
for f in range(5):
    tr = d.fold != f; te = d.fold == f
    m = LogisticRegression(C=1.0, max_iter=2000).fit(d.loc[tr, F], d.loc[tr, "ev"]); oof[te.values] = m.decision_function(d.loc[te, F])
d["m_lr"] = oof; res["nested logistic(ranks)"] = Ef(d, "m_lr")
# (2) nested small lambdarank
params = dict(objective="lambdarank", learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, feature_fraction=0.8, lambda_l2=10.0, lambdarank_truncation_level=5, verbose=-1, num_threads=2, seed=3)
oof2 = np.zeros(len(d))
for f in range(5):
    tr = (d.fold != f).values; te = (d.fold == f).values
    dt = d[tr].sort_values("slot"); grp = dt.groupby("slot", sort=False).size().values
    m = lgb.train(params, lgb.Dataset(dt[F], dt.ev, group=grp), num_boost_round=150); oof2[te] = m.predict(d.loc[te, F])
d["m_lgb"] = oof2; res["nested lambdarank(ranks)"] = Ef(d, "m_lgb")
# (3) nested weight search over 3-way rank blends (tab, r11 blend, p_raw) + time penalty
grid = [(a, b, c) for a in np.arange(0, 1.01, 0.1) for b in np.arange(0, 1.01, 0.1) for c in [0, 0.1, 0.2, 0.3] if a + b <= 1.0001]
cache = {}
for a, b, c in grid:
    d["tmp"] = a * d.r_tab + b * d.r_rs_blend + (1 - a - b) * d.r_p_raw - c * d.tpct
    cache[(a, b, c)] = d.groupby("slot").apply(lambda g: ap5(g, "tmp"))
meta = d.groupby("slot").fold.first(); tot = []
for f in range(5):
    trs = meta.index[meta != f]; tes = meta.index[meta == f]
    best = max(grid, key=lambda g: cache[g].loc[trs].mean()); tot += list(cache[best].loc[tes]); print("   fold", f, "chosen", tuple(round(x, 2) for x in best))
res["nested 3-way blend + time"] = float(np.mean(tot))
for k, v in res.items(): print(f"{k:28s} E {v:.6f}  (vs r15 fixed {Ef(d, 'r15'):.6f})")
