import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
T = pd.read_parquet(f"{OUT}/m5_both_train_oof.parquet")
d = T[T.src == "devsub11"].copy()
d["lab"] = np.where(d.y == 1, "pos", np.where(d.label == 0, "neg", "U"))
d = d.sort_values("oof", ascending=False).reset_index(drop=True)
for k in [100, 200, 300, 364, 400, 500, 600, 800]:
    print(f"top-{k}: ", d.head(k).lab.value_counts().to_dict())
for thr in [0.9, 0.5, 0.2, 0.05]:
    print(f"oof>{thr}:", d[d.oof > thr].lab.value_counts().to_dict())
# per-pool: pools with zero labelled positives -> number of high U
pp = d.groupby("pool").apply(lambda g: pd.Series({"npos": (g.lab == "pos").sum(), "hiU": ((g.lab == "U") & (g.oof > 0.5)).sum(), "hipos": ((g.lab == "pos") & (g.oof > 0.5)).sum()}))
print(pd.crosstab(pp.npos, pp.hiU))
# AP if top-scoring U (oof>0.5) are treated as positives
dd = d.copy(); dd["y2"] = ((dd.lab == "pos") | ((dd.lab == "U") & (dd.oof > 0.5))).astype(int)
print("AP with hiU as positives:", average_precision_score(dd.y2, dd.oof), " AP original:", average_precision_score((dd.lab == "pos").astype(int), dd.oof))
# eval: count of high pairs per pool vs dev (labelled pos + hiU)
ev = pd.read_parquet(f"{OUT}/m5_both_eval_scores.parquet")
ce = (ev.score > 0.5).groupby(ev.pool).sum()
cd = pp.hipos + pp.hiU
print("eval high per pool:", ce.value_counts().sort_index().to_dict())
print("dev (pos+hiU high) per pool:", cd.value_counts().sort_index().to_dict())
print("total eval high:", int(ce.sum()), " total dev high (pos+U):", int(cd.sum()), " labelled pos in devsub11 pop:", int((d.lab == "pos").sum()))
