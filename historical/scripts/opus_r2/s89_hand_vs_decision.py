"""Model comparison for the fourth family's activity structure: per-HAND latent activity (all member decisions in a hand share
z_h) vs per-DECISION independent activity, same tilt policy (street betas).  Fit both by EM on the 77 member pairs; compare
max log-likelihood (same number of parameters: 4 betas + 1 rate)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
exec(open("s83_hand_tilt.py").read().split("PM = prep(Mx)")[0].replace("@njit(cache=True)", "@njit"))
PM = prep(Mx)
TT = np.array([-1.0, 0.0, 0.0, 1.0])
def dec_llr(P, beta):
    be = beta[P["st"]] * P["e"]; logZ = np.log((P["q"] * np.exp(be[:, None] * TT[None, :])).sum(1)); return be * P["t"] - logZ
def fit_dec(P, iters=60):
    beta = np.array([1.0, 0.5, 0.5, 0.5]); a = 0.3
    for it in range(iters):
        l = dec_llr(P, beta); r = a * np.exp(np.clip(l, -50, 50)); r = r / (r + 1 - a); a = float(np.clip(r.mean(), 1e-4, .999))
        def negll(bv):
            be = bv[P["st"]] * P["e"]; logZ = np.log((P["q"] * np.exp(be[:, None] * TT[None, :])).sum(1)); return -np.sum(r * (be * P["t"] - logZ)) + 0.005 * np.sum(bv ** 2)
        beta = minimize(negll, beta, method="L-BFGS-B").x
    l = dec_llr(P, beta); ll = np.sum(np.log((1 - a) + a * np.exp(np.clip(l, -50, 50))))
    return beta, a, ll
bD, aD, llD = fit_dec(PM); print("per-DECISION mixture: beta", np.round(bD, 3), "alpha", round(aD, 3), "LL gain", round(llD, 1))
bH, pH, llH = fit(PM); print("per-HAND mixture:     beta", np.round(bH, 3), "pi", round(pH, 3), "LL gain", round(llH, 1))
# hybrid: per-hand activity for the first actor's first decision + independent for the rest is not separately fit; report per-street share
print("decisions per hand (members): mean", round(len(Mx) / PM['nh'], 2))
