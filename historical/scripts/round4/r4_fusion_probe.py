"""R4 probe: multi-view evidence union/fusion on existing OOF (read-only)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson

A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
VIEWS = {
    "t1": "m25t1_handfeat2_m19w10_oof.parquet",
    "e1": "m25e1_handfeat2_m19w10_oof.parquet",
    "e2": "m25e2_handfeat4_m19w10_oof.parquet",
    "e4": "m25e4_handfeat4_m19w10_oof.parquet",
    "e5": "m25e5_handfeat5_m19w10_oof.parquet",
    "p1": "m25p1_handfeat2_m19w10_oof.parquet",
    "p2": "m25p2_handfeat2_m19w10_oof.parquet",
    "p3": "m25p3_handfeat2_m19w10_oof.parquet",
    "m21a": "m25_handfeat2_m21a_oof.parquet",
    "nb": "m25nb_handfeat2_m19w10_oof.parquet",
}
base = None
for nm, f in VIEWS.items():
    d = pd.read_parquet(f"{A}/{f}").sort_values(["sl", "ts"]).reset_index(drop=True)
    if base is None:
        base = d[["sl", "h", "ev", "fam", "ts", "s"]].copy()
    else:
        assert (d.sl.values == base.sl.values).all() and (d.h.values == base.h.values).all()
    base[nm] = d.sc_fam.values
S = pd.read_parquet(f"{A}/seqwithin_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
assert (S.sl.values == base.sl.values).all() and (S.h.values == base.h.values).all()
base["nn"] = S.nn_cal.values
view_cols = list(VIEWS) + ["nn"]


def map5(df, col):
    aps = []
    for sl, g in df.groupby("sl"):
        rel = set(g.h[g.ev])
        g = g.sort_values(col, ascending=False)
        hits = 0
        s = 0.0
        for i, hh in enumerate(g.h.values[:5]):
            if hh in rel:
                hits += 1
                s += hits / (i + 1)
        aps.append(s / min(5, max(len(rel), 1)))
    return float(np.mean(aps))


def prior(df, col, K=3, a=0.25):
    y = df[col].values * poisson.cdf(K, df.groupby("sl")[col].cumsum() - df[col])
    y = y * np.exp(-a * df.groupby("sl").ts.rank(pct=True).values)
    return df.assign(y=y)


rep = {}
rep["t1_prior"] = round(map5(prior(base, "t1"), "y"), 4)
rep["t1_raw"] = round(map5(base, "t1"), 4)
for k in [5, 8, 12]:
    cov = {}
    for v in view_cols:
        top = base.sort_values(["sl", v], ascending=[True, False]).groupby("sl").head(k)
        g = top.groupby("sl").h.apply(set)
        rel = base[base.ev].groupby("sl").h.apply(set)
        tot = 0
        hit = 0
        for sl in rel.index:
            tot += len(rel[sl])
            hit += len(rel[sl] & g.get(sl, set()))
        cov[v] = round(hit / tot, 4)
    top = []
    for v in view_cols:
        top.append(base.sort_values(["sl", v], ascending=[True, False]).groupby("sl").head(k))
    U = pd.concat(top).groupby(["sl", "h"]).size().reset_index()
    rel = base[base.ev].groupby("sl").h.apply(set)
    tot = 0
    hit = 0
    for sl, mm in U.groupby("sl"):
        hs = set(mm.h)
        r = rel.get(sl, set())
        tot += len(r)
        hit += len(r & hs)
    rep[f"union_top{k}_coverage"] = round(hit / tot, 4)
    rep[f"view_coverage_top{k}"] = cov


def oracle_map5(df, cand):
    aps = []
    for sl, g in df.groupby("sl"):
        rel = set(g.h[g.ev])
        c = cand.get(sl, set())
        hits = len(rel & c)
        k = min(5, max(len(rel), 1))
        s = 0.0
        hh = 0
        for i in range(min(5, hits)):
            hh += 1
            s += hh / (i + 1)
        aps.append(s / k)
    return float(np.mean(aps))


top8 = {v: base.sort_values(["sl", v], ascending=[True, False]).groupby("sl").head(8) for v in view_cols}
U8 = pd.concat(top8.values()).groupby(["sl", "h"]).size().reset_index()
cand8 = U8.groupby("sl").h.apply(set)
rep["union_top8_oracle_map5"] = round(oracle_map5(base, cand8), 4)
top12 = {v: base.sort_values(["sl", v], ascending=[True, False]).groupby("sl").head(12) for v in view_cols}
U12 = pd.concat(top12.values()).groupby(["sl", "h"]).size().reset_index()
cand12 = U12.groupby("sl").h.apply(set)
rep["union_top12_oracle_map5"] = round(oracle_map5(base, cand12), 4)

R = base.copy()
for v in view_cols:
    R["r_" + v] = base.groupby("sl")[v].rank(pct=True)
combos = [["t1", "nn"], ["t1", "m21a"], ["t1", "e5"], ["t1", "nn", "m21a"],
          ["t1", "e5", "m21a", "nn"], ["t1", "e2", "e5", "m21a", "nn"], view_cols]
for combo in combos:
    col = "fuse_" + "_".join(combo)
    R[col] = R[["r_" + v for v in combo]].mean(axis=1)
    rep["pct_" + "+".join(combo)] = round(map5(prior(R, col), "y"), 4)
for combo in [["t1", "nn"], ["t1", "e5", "m21a", "nn"], view_cols]:
    col = "rrf_" + "_".join(combo)
    y = np.zeros(len(R))
    for v in combo:
        rk = R.groupby("sl")[v].rank(ascending=False, method="first").values
        y += 1.0 / (60 + rk)
    R[col] = y
    rep["rrf_" + "+".join(combo)] = round(map5(prior(R, col), "y"), 4)
for v in view_cols:
    if v != "t1":
        rep[f"{v}_prior"] = round(map5(prior(base, v), "y"), 4)
with open(f"{OUT}/r4_fusion_probe.json", "w") as f:
    json.dump(rep, f, indent=2)
print(json.dumps(rep, indent=2))
