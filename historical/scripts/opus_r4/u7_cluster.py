"""R4-U7: do the listed evidence hands cluster in time / co-seating sessions beyond a homogeneous Bernoulli process? Permutation null inside each pair's window."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
d = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet").sort_values(["sl", "ts", "h"]).reset_index(drop=True)
d["k"] = d.groupby("sl").cumcount()
d["dh"] = d.groupby("sl").h.diff().fillna(1e9); d["dts"] = d.groupby("sl").ts.diff().fillna(1e9)
print("h gap quantiles", d.dh[d.dh < 1e8].quantile([.1, .25, .5, .75, .9, .99]).tolist()); print("ts gap quantiles (s)", d.dts[d.dts < 1e8].quantile([.1, .25, .5, .75, .9, .99]).round(0).tolist())
d["sess"] = (d.dh > 1).groupby(d.sl).cumsum()
print("session length (co-seated run of consecutive h) quantiles", d.groupby(["sl", "sess"]).size().quantile([.1, .25, .5, .75, .9]).tolist())
rng = np.random.RandomState(0)
for fam, g in d[d.zone != "post"].groupby("fam"):
    obs_s = []; nul_s = []; obs_g = []; nul_g = []; obs_a = []; nul_a = []
    for sl, x in g.groupby("sl"):
        ev = x.ev.values; n = int(ev.sum());
        if n < 3: continue
        k = x.k.values; se = x.sess.values; idx = np.flatnonzero(ev)
        obs_s.append(len(set(se[idx]))); obs_g.append(np.mean(np.diff(k[idx]) <= 3)); obs_a.append(np.mean(np.diff(idx) == 1))
        # the last in-window hand is an evidence hand by construction: keep it fixed, permute the others
        ns = []; ng = []; na = []
        for _ in range(200):
            p = np.sort(np.append(rng.choice(len(x) - 1, n - 1, replace=False), len(x) - 1))
            ns.append(len(set(se[p]))); ng.append(np.mean(np.diff(k[p]) <= 3)); na.append(np.mean(np.diff(p) == 1))
        nul_s.append(np.mean(ns)); nul_g.append(np.mean(ng)); nul_a.append(np.mean(na))
    print(f"{fam:24s} distinct sessions obs {np.mean(obs_s):.3f} null {np.mean(nul_s):.3f} | gaps<=3 obs {np.mean(obs_g):.3f} null {np.mean(nul_g):.3f} | adjacent obs {np.mean(obs_a):.3f} null {np.mean(nul_a):.3f}")
