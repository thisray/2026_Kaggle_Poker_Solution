"""R4-U8: inside DT right-direction big pots (in-window), what separates listed from unlisted hands? Pair-grouped CV LightGBM + single-feature AUCs."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
u = pd.read_parquet(f"{OUT}/r4/u5_directed_transfer.parquet")
f = pd.read_parquet(f"{OUT}/r3/t58_seq_feats.parquet").rename(columns={"slot": "sl"}).drop(columns=["pa", "pb"])
m = u.merge(f, on=["sl", "h"], how="left"); m["nwin"] = m.groupby("sl").k.transform("max"); m["kfrac"] = m.k / m.nwin
m["k_big"] = m[(m.conR >= 20) & (m.conS >= 20) & (m.dir == 1)].groupby("sl").cumcount(); 
x = m[(m.zone != "post") & m.big & (m.dir == 1)].copy(); print("rows", len(x), "ev", x.ev.mean(), "pairs", x.sl.nunique())
drop = {"sl", "h", "ev", "fam", "zone", "win_end", "ts", "pa", "pb", "sa", "sb", "recvA", "b", "rk", "tab", "top5", "cand", "big", "dir", "nwin", "_x"}
cols = [c for c in x.columns if c not in drop and x[c].dtype != object]
X = x[cols].astype(float); y = x.ev.astype(int).values
aucs = {c: roc_auc_score(y, X[c].fillna(0)) for c in cols}; s = pd.Series(aucs).sort_values(); print(pd.concat([s.head(12), s.tail(12)]).round(3).to_string())
oof = np.zeros(len(x)); imp = np.zeros(len(cols))
for tr, te in GroupKFold(5).split(X, y, x.sl.values):
    mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, verbose=-1).fit(X.iloc[tr], y[tr])
    oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]; imp += mdl.booster_.feature_importance("gain")
print("CV AUC all feats", roc_auc_score(y, oof)); print(pd.Series(imp, cols).sort_values(ascending=False).head(15).round(0).to_string())
print("R15 blend AUC among candidates", roc_auc_score(y[x.b.notna().values], x.b[x.b.notna()]), "cand frac", x.b.notna().mean())
