"""Round-5 E1+E2: candidate-union diagnosis with family-routed scores + scale-safe fusion re-test."""
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson

A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"

F = pd.read_parquet(f"{A}/m25famsep4_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
views = {"famsep": F.sc.values}
for nm, f in {
    "t1": "m25t1_handfeat2_m19w10_oof.parquet",
    "p2": "m25p2_handfeat2_m19w10_oof.parquet",
    "e5": "m25e5_handfeat5_m19w10_oof.parquet",
    "m21a": "m25_handfeat2_m21a_oof.parquet",
}.items():
    d = pd.read_parquet(f"{A}/{f}").sort_values(["sl", "ts"]).reset_index(drop=True)
    assert (d.sl.values == F.sl.values).all() and (d.h.values == F.h.values).all()
    views[nm] = d.sc_fam.values
S = pd.read_parquet(f"{A}/seqwithin_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
assert (S.sl.values == F.sl.values).all()
views["nn"] = S.nn_cal.values
D = pd.DataFrame({"sl": F.sl.values, "h": F.h.values, "ev": F.ev.values, "fam": F.fam.values, "ts": F.ts.values})
for k, v in views.items():
    D[k] = v
rep = {}


def decode(df, col):
    cum = df.groupby("sl")[col].cumsum() - df[col]
    return df[col] * poisson.cdf(3, cum) * np.exp(-0.25 * df.groupby("sl").ts.rank(pct=True))


def map5(df, col):
    aps = []
    for _, g in df.groupby("sl"):
        rel = set(g.h[g.ev])
        top = g.sort_values(col, ascending=False).h.values[:5]
        hits = 0
        s = 0.0
        for i, hh in enumerate(top):
            if hh in rel:
                hits += 1
                s += hits / (i + 1)
        aps.append(s / min(5, max(len(rel), 1)))
    return round(float(np.mean(aps)), 4)


def oracle_map5(df, cand):
    aps = []
    for sl, g in df.groupby("sl"):
        rel = set(g.h[g.ev])
        hits = len(rel & cand.get(sl, set()))
        k = min(5, max(len(rel), 1))
        s = 0.0
        hh = 0
        for i in range(min(5, hits)):
            hh += 1
            s += hh / (i + 1)
        aps.append(s / k)
    return round(float(np.mean(aps)), 4)


# coverage per view with decoded ranking
for k in list(views):
    D["dec_" + k] = decode(D, k)
    rep[f"map5_{k}_dec"] = map5(D, "dec_" + k)
for k in [5, 8, 12]:
    cov = {}
    for v in views:
        top = D.sort_values(["sl", "dec_" + v], ascending=[True, False]).groupby("sl").head(k)
        g = top.groupby("sl").h.apply(set)
        rel = D[D.ev].groupby("sl").h.apply(set)
        tot = sum(len(r) for r in rel)
        hit = sum(len(r & g.get(sl, set())) for sl, r in rel.items())
        cov[v] = round(hit / tot, 4)
    top = []
    for v in views:
        top.append(D.sort_values(["sl", "dec_" + v], ascending=[True, False]).groupby("sl").head(k))
    U = pd.concat(top).groupby(["sl", "h"]).size().reset_index()
    rel = D[D.ev].groupby("sl").h.apply(set)
    tot = sum(len(r) for r in rel)
    hit = sum(len(r & set(mm.h)) for sl, mm in U.groupby("sl") for r in [rel.get(sl, set())])
    rep[f"union_top{k}_coverage"] = round(hit / tot, 4)
    rep[f"view_coverage_top{k}"] = cov
    cand = U.groupby("sl").h.apply(set)
    rep[f"union_top{k}_oracle_map5"] = oracle_map5(D, cand)

# error decomposition for famsep decoded
D["dec"] = D["dec_famsep"]
top5 = D.sort_values(["sl", "dec"], ascending=[True, False]).groupby("sl").head(5)
first = D[D.ev].groupby("sl").ts.min()
lastE = D[D.ev].groupby("sl").ts.max()
D["where"] = np.where(D.ev, "EVIDENCE", np.where(D.ts < D.sl.map(first), "before_first",
                        np.where(D.ts <= D.sl.map(lastE), "between", "after_last")))
top5w = D.loc[top5.index]
rep["top5_where"] = top5w["where"].value_counts().to_dict()
missed = D[D.ev & ~D.index.isin(top5.index)]
rep["missed_evidence"] = int(len(missed))
# of missed evidence, how many are in union top12 (of other views) => candidate vs discrimination
top12 = {v: set(map(tuple, D.sort_values(["sl", "dec_" + v], ascending=[True, False]).groupby("sl").head(12)[["sl", "h"]].values)) for v in views}
missed_in_union = sum(1 for sl, hh in missed[["sl", "h"]].values if any((sl, hh) in top12[v] for v in views))
rep["missed_in_any_view_top12"] = int(missed_in_union)
mis2 = missed[["sl", "h"]].values
rep["missed_in_famsep_top12"] = int(sum(1 for sl, hh in mis2 if (sl, hh) in top12["famsep"]))
rep["missed_in_famsep_top20"] = int(sum(1 for sl, hh in mis2 if (sl, hh) in set(map(tuple, D.sort_values(["sl", "dec_famsep"], ascending=[True, False]).groupby("sl").head(20)[["sl", "h"]].values))))
rep["missed_in_famsep_top30"] = int(sum(1 for sl, hh in mis2 if (sl, hh) in set(map(tuple, D.sort_values(["sl", "dec_famsep"], ascending=[True, False]).groupby("sl").head(30)[["sl", "h"]].values))))

# E2: scale-safe fusion — decode each view first, then fuse decoded within-pair ranks
r = {}
for combo in [["famsep", "t1"], ["famsep", "nn"], ["famsep", "t1", "e5"], ["famsep", "t1", "nn"]]:
    rk = sum(D.groupby("sl")["dec_" + v].rank(pct=True) for v in combo) / len(combo)
    D["fuse_dec_" + "+".join(combo)] = rk
    rep["fuse_dec_" + "+".join(combo)] = map5(D, "fuse_dec_" + "+".join(combo))
with open(f"{OUT}/r5_diag.json", "w") as f:
    json.dump(rep, f, indent=2)
print(json.dumps(rep, indent=2))
