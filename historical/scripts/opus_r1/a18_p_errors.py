import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
T = pd.read_parquet(f"{OUT}/m5_both_train_oof.parquet"); d = T[T.src == "devsub11"].copy()
hid = d[(d.label == -1) & (d.oof > 0.3)].key
d = d[~d.key.isin(set(hid))].sort_values("oof", ascending=False).reset_index(drop=True)
d["rank"] = np.arange(len(d))
pos = d[d.y == 1]
print("labelled positives:", len(pos), " rank quantiles:", np.quantile(pos["rank"], [0.5, 0.8, 0.9, 0.95, 0.99]).round(0))
low = pos[pos["rank"] > 400]
print("positives ranked beyond 400:", len(low), " family:", low.fam.value_counts().to_dict(), " n (exposure) median:", low.n.median(), " vs all pos median:", pos.n.median())
print(low[["fam", "n", "oof", "rank"]].head(30).to_string())
negs_above = d[(d.y == 0) & (d["rank"] < 380)]
print("non-positive rows in top-380:", len(negs_above), " labelled-neg:", int((negs_above.label == 0).sum()), " U:", int((negs_above.label == -1).sum()), " n median", negs_above.n.median())
print(negs_above[["label", "n", "oof", "rank"]].head(15).to_string())
