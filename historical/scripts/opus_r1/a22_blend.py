import numpy as np, pandas as pd, sys
from scipy.stats import poisson, rankdata
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
tags = sys.argv[1].split(",")
Ds = [pd.read_parquet(f"{OUT}/{t}_handscores.parquet") for t in tags]
for d in Ds[1:]: assert (d.h.values == Ds[0].h.values).all() and (d.sl.values == Ds[0].sl.values).all()
D = Ds[0][["sl", "h", "ev", "fam", "pos", "phase", "ts"]].copy()
for t, d in zip(tags, Ds): D[t] = d.s.values
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
D["blend_logit"] = 1 / (1 + np.exp(-np.mean([lg(D[t].values) for t in tags], axis=0)))
D["blend_mean"] = np.mean([D[t].values for t in tags], axis=0)
P = D[D.pos].sort_values(["sl", "ts"]).copy()
def map5(df, col):
    out = []
    for k, g in df.groupby("sl"):
        rel = set(g.h[g.ev]); top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0; ssum = 0.0
        for i, hh in enumerate(top):
            if hh in rel: hits += 1; ssum += hits / (i + 1)
        out.append((g.fam.iloc[0], ssum / min(5, max(len(rel), 1))))
    r = pd.DataFrame(out, columns=["fam", "ap"]); return round(r.ap.mean(), 4), r.groupby("fam").ap.mean().round(4).to_dict()
for col in tags + ["blend_logit", "blend_mean"]:
    P["cum"] = P.groupby("sl")[col].cumsum() - P[col]
    P["x"] = P[col] * poisson.cdf(4, P.cum); P["e"] = P[col] * np.exp(-0.5 * P.groupby("sl").ts.rank(pct=True))
    print(f"{col:12s} plt5 {map5(P, 'x')}  exp {map5(P, 'e')}", flush=True)
