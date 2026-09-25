"""(1) Unbiased post-window collusion test on ALL pair hands (not candidates).  (2) Episodic clustering of evidence in time."""
import numpy as np, pandas as pd
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
S6 = pd.read_parquet(f"{OUT}/s6_situation.parquet")[["sl", "h", "sit"]]
M = M.merge(S6, on=["sl", "h"], how="left")
M["idx"] = M.groupby("sl").cumcount(); M["npair"] = M.groupby("sl").sl.transform("size")
last = M[M.ev].groupby("sl").idx.max(); M["post"] = M.idx > M.sl.map(last)
mp = M.groupby("sl").ev.sum(); M["m_p"] = M.sl.map(mp)
for nm, g in [("in-window non-ev", M[~M.post & ~M.ev]), ("post-window", M[M.post]), ("evidence", M[M.ev])]:
    for sit in [True]:
        x = g[g.sit == sit]
        print(f"{nm:18s} sit={sit} n {len(x):6d}  frac s>0.5 {(x.s > 0.5).mean():.4f}  s>0.8 {(x.s > 0.8).mean():.4f}  s>0.95 {(x.s > 0.95).mean():.4f}")
for full in [5, 4, 3]:
    g = M[(M.m_p == full) & M.post & (M.sit == True)]
    print(f"pairs with m_p={full}: post-window situation hands {len(g)}  frac s>0.8 {(g.s > 0.8).mean():.4f}  (pairs {M[M.m_p == full].sl.nunique()})")
print("fraction of pair hands that are post-window, by m_p:", M.groupby("m_p").post.mean().round(3).to_dict())
# (2) clustering: gaps between consecutive evidence (in situation-hand index) vs permutation null within [first sit, last ev]
rng = np.random.default_rng(0); obs = []; null = []
for sl, g in M[M.sit == True].groupby("sl"):
    g = g.reset_index(drop=True); e = np.flatnonzero(g.ev.values)
    if len(e) < 3: continue
    L = e.max() + 1; obs.append(np.diff(e).mean())
    for _ in range(50):
        r = np.sort(np.r_[rng.choice(L - 1, len(e) - 1, replace=False), L - 1]); null.append(np.diff(r).mean())
print(f"mean gap between consecutive evidence (situation-hand units): observed {np.mean(obs):.3f}  null {np.mean(null):.3f}")
# run-length: are evidence hands adjacent more often than chance
adj = []; adjn = []
for sl, g in M[M.sit == True].groupby("sl"):
    e = g.ev.values.astype(int); L = np.flatnonzero(e).max() + 1 if e.any() else 0
    if L < 3: continue
    x = e[:L]; adj.append((x[1:] * x[:-1]).sum() / max(x.sum() - 1, 1))
    for _ in range(50):
        y = np.r_[rng.permutation(x[:-1]), 1]; adjn.append((y[1:] * y[:-1]).sum() / max(y.sum() - 1, 1))
print(f"P(next situation hand is evidence | this is evidence): observed {np.mean(adj):.3f}  null {np.mean(adjn):.3f}")
