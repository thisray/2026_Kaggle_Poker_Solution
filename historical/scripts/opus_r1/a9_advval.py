"""Do evaluation-phase high-risk pairs look like dev positives? Family ambiguity and adversarial validation (hint for a hidden 4th family)."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pairfeat import build
from pairfeat2 import build2
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
ev = pd.read_parquet(f"{OUT}/m5_both_eval_scores.parquet"); fam = pd.read_parquet(f"{OUT}/m7_family_eval.parquet").set_index("key")
hi = ev[ev.score > 0.5].copy()
for c in ["p_dt", "p_sp", "p_ci", "family"]: hi[c] = hi.key.map(fam[c])
hi["pmax"] = hi[["p_dt", "p_sp", "p_ci"]].max(axis=1)
print("eval high pairs:", len(hi), " family counts:", hi.family.value_counts().to_dict())
print("max family prob quantiles:", np.round(np.quantile(hi.pmax, [0.05, 0.1, 0.25, 0.5]), 3), " n pmax<0.6:", int((hi.pmax < 0.6).sum()), " <0.8:", int((hi.pmax < 0.8).sum()))
tr = pd.read_parquet(f"{OUT}/m5_both_train_oof.parquet")
dpos = tr[(tr.src == "devsub11") & ((tr.y == 1) | ((tr.label == -1) & (tr.oof > 0.5)))]
print("dev positives (labelled + hidden>0.5):", len(dpos), " labelled family mix:", dpos.fam.value_counts().to_dict())
t_dev = pd.read_parquet(f"{OUT}/ptab_devsub11.parquet"); t_dev["key"] = t_dev.p_lo * 12000 + t_dev.p_hi
t_ev = pd.read_parquet(f"{OUT}/ptab_eval.parquet"); t_ev["key"] = t_ev.p_lo * 12000 + t_ev.p_hi
A = t_dev[t_dev.key.isin(set(dpos.key))]; B = t_ev[t_ev.key.isin(set(hi.key))]
XA = pd.concat([build(A), build2(A)], axis=1); XB = pd.concat([build(B), build2(B)], axis=1)
X = pd.concat([XA, XB], ignore_index=True); y = np.r_[np.zeros(len(XA)), np.ones(len(XB))]
X = X.drop(columns=[c for c in X.columns if c in ("n", "lo_hands", "hi_hands")])
oof = np.zeros(len(y))
for trn, val in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.7, verbose=-1, num_threads=4), lgb.Dataset(X.iloc[trn], y[trn]), 200)
    oof[val] = m.predict(X.iloc[val])
print("adversarial AUC dev-positives vs eval-high:", round(roc_auc_score(y, oof), 4))
imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False).head(12)
print(imp.round(0).to_string())
Bh = hi.set_index("key").loc[B.key.values]
Bh = Bh.assign(adv=oof[len(XA):])
print("eval-high pairs most unlike dev positives (adv score top 15):")
print(Bh.sort_values("adv", ascending=False).head(15)[["pool", "n", "score", "family", "pmax", "adv"]].round(3).to_string())
Bh.sort_values("adv", ascending=False).to_parquet(f"{OUT}/a9_eval_high_adv.parquet")
