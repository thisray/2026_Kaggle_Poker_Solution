"""Transductive expansion of the fourth-family set: learn 'pattern-2 seed' vs (normal + known-family) from pair features that
EXCLUDE the information-dependence statistic, then check whether high-scoring non-seed pairs show elevated p2 (independent confirmation)."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
t = pd.read_parquet(f"{OUT}/ptab_eval.parquet"); t4 = pd.read_parquet(f"{OUT}/ptab4_eval.parquet")
t = pd.concat([t, t4[[c for c in t4.columns if c.startswith("Q_")]]], axis=1)
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values
e = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet").merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"), on="slot")
e["p2"] = np.minimum(e.za0 - e.zf0, e.za1 - e.zf1)
e = e.sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); e["rk"] = np.arange(1, len(e) + 1)
d = e[["slot", "pair_id", "p2", "rk", "n_is"]].merge(t, on="slot", how="inner")
feat = [c for c in t.columns if c not in ("slot", "pool", "p_lo", "p_hi", "key") and t[c].dtype.kind in "fi"]
print("pairs", len(d), "features", len(feat), flush=True)
seed = (d.p2 > 4).values
neg_norm = ((d.rk > 5000) & (d.p2 < 2)).values; neg_known = ((d.rk <= 450) & (d.p2 < 1)).values
rng = np.random.default_rng(0); nn = np.flatnonzero(neg_norm); nn = rng.choice(nn, 30000, replace=False)
idx = np.r_[np.flatnonzero(seed), nn, np.flatnonzero(neg_known)]; y = np.r_[np.ones(seed.sum()), np.zeros(len(nn) + neg_known.sum())]
w = np.r_[np.ones(seed.sum()) * 20, np.ones(len(nn)), np.ones(neg_known.sum()) * 5]
X = d[feat].to_numpy(np.float32)
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4, seed=3)
fold = rng.integers(0, 5, len(idx)); oof = np.zeros(len(idx)); full_pred = np.zeros(len(d))
for f in range(5):
    tr = fold != f; m = lgb.train(params, lgb.Dataset(X[idx[tr]], y[tr], weight=w[tr]), num_boost_round=400)
    oof[~tr] = m.predict(X[idx[~tr]]); full_pred += m.predict(X) / 5
print("CV AUC seed vs normal:", round(roc_auc_score(y[(y == 1) | np.isin(idx, nn)], oof[(y == 1) | np.isin(idx, nn)]), 4),
      " seed vs known-family:", round(roc_auc_score(y[(y == 1) | np.isin(idx, np.flatnonzero(neg_known))], oof[(y == 1) | np.isin(idx, np.flatnonzero(neg_known))]), 4), flush=True)
d["f4"] = full_pred
imp = pd.Series(m.feature_importance("gain"), index=feat).sort_values(ascending=False); print(imp.head(15).round(0).to_string())
# independent confirmation: among NON-seed pairs, is p2 elevated in the high-f4 tail?  (p2 was not a feature)
ns = d[~seed].sort_values("f4", ascending=False)
for k in [50, 100, 200, 400, 1000]:
    top = ns.head(k); print(f"non-seed top-{k} by f4: mean p2 {top.p2.mean():.3f}  frac p2>2 {(top.p2 > 2).mean():.3f}  frac p2>3 {(top.p2 > 3).mean():.3f}  r11 rank median {top.rk.median():.0f}  n_is median {top.n_is.median():.0f}")
print(f"all non-seed: mean p2 {ns.p2.mean():.3f} frac p2>2 {(ns.p2 > 2).mean():.4f} frac p2>3 {(ns.p2 > 3).mean():.4f}")
d[["slot", "pair_id", "p2", "rk", "f4", "n_is"]].to_parquet(f"{OUT}/s26_f4_eval.parquet")
