import numpy as np, pandas as pd
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score, average_precision_score
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
D = pd.read_parquet(f"{OUT}/m17_handscores.parquet")
P = D[D.pos].sort_values(["sl", "ts"]).copy()
P["cum"] = P.groupby("sl").s.cumsum() - P.s; P["r"] = P.s * poisson.cdf(4, P.cum)
first = P[P.ev].groupby("sl").ts.min(); last = P[P.ev].groupby("sl").ts.max()
P["where"] = np.where(P.ev, "EVIDENCE", np.where(P.ts < P.sl.map(first), "before_first", np.where(P.ts < P.sl.map(last), "between", "after_last")))
top = P.sort_values(["sl", "r"], ascending=[True, False]).groupby("sl").head(5)
print("top-5 picks composition:", top["where"].value_counts().to_dict(), " per family:")
print(pd.crosstab(top.fam, top["where"]))
win = P[P["where"] != "after_last"]
y = win.ev.astype(int)
print("within window (<= last evidence): AUC", round(roc_auc_score(y, win.s), 4), "AP", round(average_precision_score(y, win.s), 4), " n", len(win), "pos", int(y.sum()))
for fam, g in win.groupby("fam"):
    print("   ", fam, "AUC", round(roc_auc_score(g.ev, g.s), 4), "AP", round(average_precision_score(g.ev, g.s), 4))
# how many evidence hands are ranked below some non-evidence hand in the window (per pair)
