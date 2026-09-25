"""Adversarial validation for novelty: dev positives (exposure-matched devsub11/12, labelled + hidden) vs eval top-600 pairs.
Pairs in eval that look unlike ANY dev positive = candidate novel behaviour.  Is the novelty explained by pattern-2?"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
def load(nm):
    t = pd.read_parquet(f"{OUT}/ptab_{nm}.parquet"); t4 = pd.read_parquet(f"{OUT}/ptab4_{nm}.parquet")
    t = pd.concat([t, t4[[c for c in t4.columns if c.startswith("Q_")]]], axis=1)
    t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values; return t
ref = pd.read_parquet(f"{OUT}/m15_v6_drop_m26_train_oof.parquet")
parts = []
for nm in ["devsub11", "devsub12"]:
    t = load(nm); r = ref[ref.src == nm]; pos_keys = set(r[(r.y == 1) | (r.hid)].key)
    t["key"] = t.p_lo * 12000 + t.p_hi; parts.append(t[t.key.isin(pos_keys)].assign(src=nm))
dpos = pd.concat(parts, ignore_index=True)
te = load("eval"); ev = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv").sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); ev["rk"] = np.arange(1, len(ev) + 1)
te = te.merge(ev[["slot", "pair_id", "rk"]], on="slot"); etop = te[te.rk <= 600].copy()
old = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet"); old["p2"] = np.minimum(old.za0 - old.zf0, old.za1 - old.zf1); etop = etop.merge(old[["slot", "p2"]], on="slot", how="left")
feat = [c for c in te.columns if c not in ("slot", "pool", "p_lo", "p_hi", "key", "pair_id", "rk") and te[c].dtype.kind in "fi" and c in dpos.columns and c != "n"]
X = np.r_[dpos[feat].to_numpy(np.float32), etop[feat].to_numpy(np.float32)]; y = np.r_[np.zeros(len(dpos)), np.ones(len(etop))]
grp = np.r_[dpos.pool.values, etop.pool.values]; fold = grp % 5
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4, seed=1)
oof = np.zeros(len(y))
for f in range(5):
    tr = fold != f; m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=300); oof[~tr] = m.predict(X[~tr])
print("adversarial AUC (dev positives vs eval top-600):", round(roc_auc_score(y, oof), 4), " n dev pos", len(dpos), " n eval top", len(etop))
etop["adv"] = oof[len(dpos):]
etop["p2flag"] = etop.p2 > 3
print("eval top-600 by adversarial 'eval-ness' decile: share with p2>3, median rank")
etop["dec"] = pd.qcut(etop.adv, 10, labels=False)
print(etop.groupby("dec").agg(p2share=("p2flag", "mean"), rk_med=("rk", "median"), n=("rk", "size")).round(3).to_string())
imp = pd.Series(m.feature_importance("gain"), index=feat).sort_values(ascending=False); print(imp.head(12).round(0).to_string())
print("AUC without pattern-2 pairs (p2>3 removed from eval side):", round(roc_auc_score(y[np.r_[np.ones(len(dpos), bool), ~etop.p2flag.values]], oof[np.r_[np.ones(len(dpos), bool), ~etop.p2flag.values]]), 4))
etop[["slot", "pair_id", "rk", "p2", "adv"]].to_parquet(f"{OUT}/s29_adv_eval_top600.parquet")
