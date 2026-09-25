"""Round-6: reorder the selected top-5 by alternative signals; measure E."""
import json
import numpy as np
import pandas as pd
from scipy.stats import poisson

A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
S = pd.read_parquet(f"{DST}/r6_dev_scores.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
T = pd.read_parquet(f"{A}/m25t1_handfeat2_m19w10_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
N = pd.read_parquet(f"{A}/seqwithin_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
base = pd.read_parquet(f"{A}/m19w10_handscores.parquet")
P = base[base.pos & (base.phase == 0)].sort_values(["sl", "ts"]).reset_index(drop=True)
assert (T.sl.values == S.sl.values).all() and (N.sl.values == S.sl.values).all() and (P.sl.values == S.sl.values).all()
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
S["t1"] = T.sc_fam.values
S["nn"] = N.nn_cal.values
S["s1"] = P.s.values
S["fams"] = P.fam.values


def ap5_flags(flags, n_g):
    hits, s = 0, 0.0
    for r, z in enumerate(flags[:5], start=1):
        if z:
            hits += 1
            s += hits / r
    return s / min(5, max(int(n_g), 1))


def eval_orders(order_col):
    aps = []
    for s_, g in S.groupby("sl"):
        n_g = int(g.ev.sum())
        if n_g == 0:
            continue
        top5 = g.sort_values("u_r5b", ascending=False).head(5)
        o = top5.sort_values(order_col, ascending=False)
        aps.append(ap5_flags(o.ev.values.astype(int), n_g))
    return round(float(np.mean(aps)), 4)


res = {}
res["current_u_r5b"] = eval_orders("u_r5b")
for c in ["sc_r5", "t1", "nn", "s1"]:
    res["order_by_" + c] = eval_orders(c)
res["order_by_nn_x_t1"] = eval_orders_add = None
S["mix"] = 0.5 * lg(S.nn.values) + 0.5 * lg(S.t1.values)
res["order_by_mix_nn_t1"] = eval_orders("mix")
print(json.dumps(res, indent=2))
json.dump(res, open(f"{DST}/r6_order_probe.json", "w"), indent=2)
