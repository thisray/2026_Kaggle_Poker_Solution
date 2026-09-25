"""Mid-rank fourth-family posterior by density ratio: eval mid-rank pairs (BF_hier>2, t27) vs devsub11 null pairs selected the
same way (t28).  Populations are the same size (rank 600-30000 of ~equal pair universes), so f_eval/f_null ~ p/(1-p) * n0/n1;
posterior(F4 | x) = max(0, 1 - f_null/f_eval).  Cross-validated logistic on per-decision-normalised features; also reports the
known-family positive tail check (devsub positives never appear in this rank band)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
R3 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r3"
E = pd.read_parquet(f"{R3}/t27_placebo_ext.parquet"); E = E[E.set == "eval_mid_bf2"].copy()
N = pd.read_parquet(f"{R3}/t28_placebo_null.parquet"); N = N[N.set == "devsub_mid_bf2_null"].copy()
BE = pd.read_parquet(f"{R3}/t17_bf_sub_eval.parquet")[["slot", "pair_id", "rk_r2j2m", "predicted_behavior", "bf", "bf_hier", "nd"]]
E = E.drop(columns=[c for c in ["pair_id", "rk_r2j2m", "predicted_behavior", "bf_hier"] if c in E]).merge(BE, on="slot", how="left")
BV = pd.read_parquet(f"{R3}/t17_bf_sub_devsub11.parquet")[["slot", "bf_hier", "nd"]]
N = N.drop(columns=[c for c in ["bf_hier"] if c in N]).merge(BV, on="slot", how="left")
for d in (E, N):
    d["bh_n"] = d.bf_hier / d.nd; d["Dn"] = d.D / np.sqrt(d.n); d["bO_n"] = d.bfO / d.n; d["bP_n"] = d.bfP / d.n; d["ln"] = np.log(d.nd)
X = pd.concat([E.assign(y=1), N.assign(y=0)], ignore_index=True)
F = ["bh_n", "Dn", "bO_n", "ln", "bf_hier", "D"]
p = np.zeros(len(X))
for seed in range(20):
    p += cross_val_predict(LogisticRegression(C=0.5, max_iter=2000), X[F], X.y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1] / 20
X["p"] = p; n1, n0 = int(X.y.sum()), int((1 - X.y).sum())
X["ratio"] = X.p / (1 - X.p) * n0 / n1                     # f_eval / f_null (equal-size source populations)
X["post"] = np.clip(1 - 1 / X.ratio, 0, 1)
b6 = open(f"{R3}/b6_promoted.txt").read().split(","); b7 = open(f"{R3}/r3_b7_promote.txt").read().split(",")
ev = X[X.y == 1].sort_values("post", ascending=False)
ev["tag"] = np.where(ev.pair_id.isin(b6), "B6", np.where(ev.pair_id.isin(b7), "B7", ""))
pd.set_option("display.width", 220)
print(f"eval {n1} vs null {n0}; implied excess = {n1 - n0}; sum of posteriors over eval = {ev.post.sum():.1f}")
print(ev[["pair_id", "rk_r2j2m", "predicted_behavior", "bf", "bf_hier", "nd", "D", "Dn", "bfO", "p", "post", "tag"]].head(25).round(3).to_string(index=False))
print("null pairs with highest p (false-positive check):", X[X.y == 0].sort_values("p", ascending=False)[["bf_hier", "nd", "D", "Dn", "p"]].head(6).round(3).to_string(index=False))
ev.to_parquet(f"{R3}/t33_mid_posterior.parquet")
