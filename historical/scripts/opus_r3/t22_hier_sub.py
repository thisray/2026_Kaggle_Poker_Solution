"""Hierarchical activity test under the exact substitution mechanism (77 members, t13 arrays): unit = hand or hand x actor;
L_unit = (1-rho) + rho prod_d [(1-s_st) + s_st r_d],  r_d = q_sub(y)/q0(y).  Compare with per-decision (rho=1)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
a = np.load("/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/t13_members_qsub.npy")
M = pd.DataFrame(a[:, :7].astype(np.int64), columns=["slot", "h", "k", "s", "o", "st", "y"])
q0 = np.clip(a[:, 7:11], 1e-6, 1); q0 /= q0.sum(1, keepdims=True); qs = np.clip(a[:, 11:15], 1e-6, 1); qs /= qs.sum(1, keepdims=True)
r = qs[np.arange(len(M)), M.y] / q0[np.arange(len(M)), M.y]; pre = (M.st.values == 0)
for unit in ("hand", "hand_actor"):
    key = M.slot.values * 10_000_000 + M.h.values
    if unit == "hand_actor": key = key * 8 + M.s.values
    _, inv = np.unique(key, return_inverse=True); nu = inv.max() + 1
    def nll(th, mode):
        rho = 1 / (1 + np.exp(-th[0])); s1 = 1 / (1 + np.exp(-th[1])); s2 = 1 / (1 + np.exp(-th[2]))
        if mode == "dec": rho = 1.0
        s = np.where(pre, s1, s2)
        ld = np.log((1 - s) + s * r); lu = np.bincount(inv, weights=ld, minlength=nu)
        lu = np.log((1 - rho) + rho * np.exp(lu)) if rho < 1 else lu
        return -lu.sum()
    for mode in ("dec", "hier"):
        best = min((minimize(nll, np.array(x0), args=(mode,), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-5, fatol=1e-4)) for x0 in ([5, -0.5, -0.7], [1, 0, 0], [0, 1, 0.5], [-1, 2, 2])), key=lambda z: z.fun)
        th = best.x; rho = 1 / (1 + np.exp(-th[0])) if mode == "hier" else 1.0
        print(f"unit={unit:10s} mode={mode}: LLgain {-best.fun:8.1f}  rho {rho:.3f}  s_pre {1/(1+np.exp(-th[1])):.3f}  s_post {1/(1+np.exp(-th[2])):.3f}  units {nu}")
