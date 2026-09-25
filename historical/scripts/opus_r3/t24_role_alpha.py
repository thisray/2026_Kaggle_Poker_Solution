"""Substitution activity by street x role (first-acting member of the hand vs the other member), 77 members (t13 arrays)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
a = np.load(f"{OUT}/t13_members_qsub.npy"); M = pd.DataFrame(a[:, :7].astype(np.int64), columns=["slot", "h", "k", "s", "o", "st", "y"])
q0 = np.clip(a[:, 7:11], 1e-6, 1); q0 /= q0.sum(1, keepdims=True); qs = np.clip(a[:, 11:15], 1e-6, 1); qs /= qs.sum(1, keepdims=True)
M["r"] = qs[np.arange(len(M)), M.y] / q0[np.arange(len(M)), M.y]
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy")
first = {}
for (sl, h), g in M.groupby(["slot", "h"]):
    seats = set(g.s) | set(g.o)
    for k in range(off[h], off[h + 1]):
        if a_seat[k] in seats: first[(sl, h)] = a_seat[k]; break
M["first_actor"] = [first[(sl, h)] == s for sl, h, s in zip(M.slot, M.h, M.s)]
tot = 0.0
for (st, fa), g in M.groupby([M.st > 0, "first_actor"]):
    r = g.r.values; f = lambda x: -np.sum(np.log((1 - x) + x * r)); res = minimize_scalar(f, bounds=(1e-4, .9999), method="bounded")
    tot += -res.fun; print(f"street {'post' if st else 'pre '} first_actor={fa}: n {len(g)} alpha {res.x:.3f} LLgain {-res.fun:.1f}")
print("total LL (4 alphas):", round(tot, 1), " vs 2 alphas (street only): 2187.5")
