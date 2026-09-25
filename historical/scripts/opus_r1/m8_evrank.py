import numpy as np, pandas as pd, lightgbm as lgb, time, sys
from evrank import rank_features
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
SRC = sys.argv[1] if len(sys.argv) > 1 else "m6_handscores.parquet"
D0 = pd.read_parquet(f"{OUT}/{SRC}")
neg = D0[(D0.phase == 0) & (~D0.pos)]
q99, q999 = np.quantile(neg.s, [0.99, 0.999])
print("neg quantiles", q99, q999)
P = D0[D0.pos].copy()
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
P, feats = rank_features(P, q99, q999)
P["fold"] = fold_of_pool[P.sl.values // 900]
def map5(df, col):
    out = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        out.append((g.fam.iloc[0], ssum / min(5, max(len(rel), 1))))
    r = pd.DataFrame(out, columns=["fam", "ap"]); return round(r.ap.mean(), 4), r.groupby("fam").ap.mean().round(4).to_dict()
print("baseline s:", map5(P, "s")); print("baseline s*P(<5 before):", map5(P, "s_x_plt5"))
P = P.sort_values(["sl", "t_idx"]).reset_index(drop=True)
oof = np.zeros(len(P))
params = dict(objective="lambdarank", metric="map", eval_at=[5], learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=8, seed=1)
for f in range(5):
    tr = P[P.fold != f]; va = P.fold == f
    grp = tr.groupby("sl", sort=False).size().values
    m = lgb.train(params, lgb.Dataset(tr[feats], tr.ev.astype(int), group=grp), num_boost_round=300)
    oof[va.values] = m.predict(P.loc[va, feats])
P["rank_score"] = oof
print("lambdarank OOF:", map5(P, "rank_score"))
params_b = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8, seed=1)
oofb = np.zeros(len(P))
for f in range(5):
    tr = P.fold != f; va = P.fold == f
    m = lgb.train(params_b, lgb.Dataset(P.loc[tr, feats], P.ev[tr].astype(int)), num_boost_round=300)
    oofb[va.values] = m.predict(P.loc[va, feats])
P["bin_score"] = oofb
print("binary stage-2 OOF:", map5(P, "bin_score"))
# no-time ablation: only detection-derived non-temporal features
nt = ["s", "logit", "s_rank", "s_rank_pct", "s_rel_max", "n", "tot_hi999", "tot_hi99"]
oofn = np.zeros(len(P))
for f in range(5):
    tr = P[P.fold != f]; va = P.fold == f
    grp = tr.groupby("sl", sort=False).size().values
    m = lgb.train(params, lgb.Dataset(tr[nt], tr.ev.astype(int), group=grp), num_boost_round=300)
    oofn[va.values] = m.predict(P.loc[va, nt])
P["nt_score"] = oofn
print("lambdarank no-time OOF:", map5(P, "nt_score"))
full = lgb.train(params, lgb.Dataset(P[feats], P.ev.astype(int), group=P.groupby("sl", sort=False).size().values), num_boost_round=300)
full.save_model(f"{OUT}/m8_evrank_full.txt")
fulln = lgb.train(params, lgb.Dataset(P[nt], P.ev.astype(int), group=P.groupby("sl", sort=False).size().values), num_boost_round=300)
fulln.save_model(f"{OUT}/m8_evrank_notime_full.txt")
import json; json.dump({"q99": float(q99), "q999": float(q999), "feats": feats, "notime_feats": nt}, open(f"{OUT}/m8_evrank_meta.json", "w"))
