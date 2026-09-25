import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
F = pd.read_parquet(f"{OUT}/a13_dt_hands.parquet")
last = F[F.ev].groupby("sl").ts.max(); W = F[F.ts <= F.sl.map(last)].copy()
feats = ["flow_dr", "flow_rd", "d_lost_to_r", "d_net", "r_net", "d_contrib", "pot_bb", "d_folded", "d_sd", "r_sd", "nsd", "nboard", "d_fold_to_r", "r_fold_to_d",
         "d_call_to_r", "d_raise_over_r", "r_raise_over_d", "d_eq_fold_to_r", "d_eq_call_to_r", "d_sunk_fold", "hu_streets", "d_pf", "r_pf", "d_eq_last", "r_eq_last"]
y = W.ev.astype(int).values; X = W[feats].values; g = W.sl.values
for depth in [2, 3, 4, 6]:
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, g):
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, class_weight="balanced").fit(X[tr], y[tr]); oof[va] = m.predict_proba(X[va])[:, 1]
    print(f"depth {depth}: within-window AUC {roc_auc_score(y, oof):.4f} AP {average_precision_score(y, oof):.4f}  (GBDT m17 window AP DT ~0.537)")
m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced").fit(X, y)
print(export_text(m, feature_names=feats, show_weights=True))
