"""Planting clock test (dev, true evidence): is planting a constant rate per co-seated hand (absolute clock) or spread over
the phase (relative clock)?  Regress first/last evidence position on the pair's number of co-seated hands n.
Absolute clock: k_first independent of n, pct_first ~ 1/n.  Relative clock: pct_first independent of n, k_first ~ n."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")[["sl", "h", "ev", "fam", "ts"]].sort_values(["sl", "ts"])
M["k"] = M.groupby("sl").cumcount(); M["n"] = M.groupby("sl").sl.transform("size")
E = M[M.ev].copy(); E["rank_in_pair"] = E.groupby("sl").cumcount()
f = E.groupby("sl").agg(n=("n", "first"), m=("k", "size"), k1=("k", "min"), k5=("k", "max"), fam=("fam", "first"))
f["p1"] = (f.k1 + 0.5) / f.n; f["p5"] = (f.k5 + 0.5) / f.n
f["nb"] = pd.qcut(f.n, 4)
print(f.groupby("nb", observed=True).agg(pairs=("n", "size"), n_med=("n", "median"), k1_med=("k1", "median"), k5_med=("k5", "median"),
                                          p1_med=("p1", "median"), p5_med=("p5", "median"), m_mean=("m", "mean")).round(3).to_string())
for col in ["k1", "k5", "p1", "p5"]:
    x = np.log(f.n); y = np.log(f[col] + (1 if col.startswith("k") else 0.01)); b = np.polyfit(x, y, 1)[0]
    print(f"log-log slope of {col} on n: {b:.3f}   (absolute clock: k~0 / p~-1 ; relative clock: k~+1 / p~0)")
# gap between consecutive evidence (absolute) by n
E["gap"] = E.groupby("sl").k.diff(); g = E.merge(f[["nb"]], left_on="sl", right_index=True)
print("median gap between consecutive evidence by n quartile:", g.groupby("nb", observed=True).gap.median().to_dict())
