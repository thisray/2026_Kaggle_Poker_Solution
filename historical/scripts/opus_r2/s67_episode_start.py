"""Episode structure of labelled evidence (dev): where does the first evidence hand fall relative to the start of the
pair's co-seated dev hands? Is there a pre-episode stretch with no planted behaviour (latent scenario activation)?"""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")
M = M.sort_values(["sl", "ts"]).reset_index(drop=True)
M["k"] = M.groupby("sl").cumcount(); M["n"] = M.groupby("sl").sl.transform("size"); M["pct"] = M.k / M.n
E = M[M.ev]
f = E.groupby("sl").agg(first_k=("k", "min"), last_k=("k", "max"), first_pct=("pct", "min"), last_pct=("pct", "max"), m=("k", "size"), n=("n", "first"), fam=("fam", "first"))
print("first evidence index (co-seated hand number) quantiles by family:")
print(f.groupby("fam").first_k.describe(percentiles=[.1, .25, .5, .75, .9]).round(1))
print("first evidence time percentile:")
print(f.groupby("fam").first_pct.describe(percentiles=[.1, .25, .5, .75, .9]).round(3))
print("last evidence time percentile:")
print(f.groupby("fam").last_pct.describe(percentiles=[.1, .25, .5, .75, .9]).round(3))
# gap structure: hands between consecutive evidence (in co-seated hand counts) vs gap before first
E2 = E.copy(); E2["gap"] = E2.groupby("sl").k.diff()
print("median gap between consecutive evidence by family:", E2.groupby("fam").gap.median().to_dict())
print("mean first_k / mean gap:", (f.groupby("fam").first_k.mean() / E2.groupby("fam").gap.mean()).round(2).to_dict())
# under 'episode starts at phase start' + Bernoulli planting with rate r per co-seated hand, first_k ~ Geom(r): mean(first_k) ~ mean(gap)-1
# pre-first stretch: does it contain planted-looking hands at a lower rate than the in-window non-evidence?
M["zone2"] = np.where(M.ev, "ev", np.where(M.k < M.sl.map(f.first_k), "pre", np.where(M.k <= M.sl.map(f.last_k), "in_non", "post")))
print(M.groupby(["fam", "zone2"])[["pos_net", "xfer", "iso"]].mean().round(3))
print(M.groupby(["fam", "zone2"]).size().unstack())
