"""Fourth-family activity structure, hierarchical test.  Hand (or hand x actor) is 'planted' w.p. rho; inside a planted unit each
member decision (partner still active) follows the partner-card tilt w.p. s, else the normal policy q0.
  L_unit = (1-rho) + rho * prod_d [(1-s) + s exp(l_d)],  l_d = b_st e t(a) - log Z_d.
Nests per-decision (rho=1, s=alpha) and per-hand (s=1, rho=pi).  Fit on the 77 member pairs; also on controls."""
import numpy as np, pandas as pd, time, sys
from scipy.optimize import minimize
from scipy.special import logsumexp
exec(open("s83_hand_tilt.py").read().split("c38 = pd.read_parquet")[0].replace("@njit(cache=True)", "@njit"))
t0 = time.time()
def table2(slots):
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    rows = np.zeros((len(H) * 16, 5), np.int64); n = collect(H, S, T, off, a_seat, a_st, Y, rows); rows = rows[:n]
    r, k, s, o, st = rows.T; h = H[r]
    kk = np.argsort(k); pr = np.asarray(P2[k[kk]]); prr = np.empty_like(pr); prr[kk] = pr
    X = pd.DataFrame({"slot": SL[r], "h": h, "st": st, "y": Y[k], "seat": s, "k": k})
    X[["q_f", "q_k", "q_c", "q_a"]] = np.clip(prr, 1e-6, 1)
    X["e"] = np.where(st == 0, pfeq[h, o], eql[h, o])
    return X
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior"])
mids = set(base[base.predicted_behavior == "other_coordination"].pair_id)
msl = c38[c38.pair_id.isin(mids)].slot.values; csl = c38[c38.rk > 5000].sample(3000, random_state=3).slot.values
Mx = table2(msl); Cx = table2(csl)
mu = Cx.groupby("st").e.mean(); Mx["e"] -= Mx.st.map(mu); Cx["e"] -= Cx.st.map(mu)
print(f"[{time.time()-t0:6.1f}s] decisions members {len(Mx)} controls {len(Cx)}", flush=True)
TT = np.array([-1.0, 0.0, 0.0, 1.0])
def prep2(X, unit):
    t = np.select([X.y.values == 3, X.y.values == 0], [1.0, -1.0], 0.0)
    q = X[["q_f", "q_k", "q_c", "q_a"]].to_numpy(); q = q / q.sum(1, keepdims=True)
    key = X.slot.astype(np.int64).values * 10_000_000 + X.h.astype(np.int64).values
    if unit == "hand_actor": key = key * 8 + X.seat.values
    u, inv = np.unique(key, return_inverse=True)
    return dict(t=t, q=q, e=X.e.values.astype(float), st=X.st.values.astype(int), inv=inv, nu=len(u), key=u)
def dec_l(P, beta):
    be = beta[P["st"]] * P["e"]; logZ = np.log((P["q"] * np.exp(be[:, None] * TT[None, :])).sum(1)); return be * P["t"] - logZ
def negll(theta, P, mode):
    beta = theta[:4]; rho = 1 / (1 + np.exp(-theta[4])); s = 1 / (1 + np.exp(-theta[5]))
    if mode == "dec": rho = 1.0
    if mode == "hand": s = 1.0
    l = dec_l(P, beta)
    ld = np.logaddexp(np.log1p(-s) if s < 1 else -np.inf, np.log(s) + l)          # log[(1-s) + s e^l]
    lu = np.bincount(P["inv"], weights=ld, minlength=P["nu"])                        # log prod over the unit
    lu = np.logaddexp(np.log1p(-rho) if rho < 1 else -np.inf, np.log(rho) + lu)
    return -lu.sum() + 0.005 * np.sum(beta ** 2)
def fitm(P, mode, x0=None):
    best = None
    for init in ([1.0, 0.5, 0.5, 0.5, 0.0, 0.0], [20, 2, 2, 2, 1.0, -1.0], [40, 4, 3, 4, 3.0, -0.6], [20, 2, 2, 2, -0.5, 2.0]):
        r = minimize(negll, np.array(init, float), args=(P, mode), method="L-BFGS-B")
        if best is None or r.fun < best.fun: best = r
    th = best.x; rho = 1 / (1 + np.exp(-th[4])); s = 1 / (1 + np.exp(-th[5]))
    if mode == "dec": rho = 1.0
    if mode == "hand": s = 1.0
    return th, rho, s, -best.fun
for nm, X in (("members", Mx), ("controls", Cx)):
    for unit in ("hand", "hand_actor"):
        P = prep2(X, unit)
        # null LL (all q0): 0 by construction of LLR; report LL gains
        for mode in ("dec", "hand", "hier"):
            th, rho, s, ll = fitm(P, mode)
            print(f"[{time.time()-t0:6.1f}s] {nm:8s} unit={unit:10s} mode={mode:4s} LLgain={ll:9.1f} beta={np.round(th[:4],2)} rho={rho:.3f} s={s:.3f} units={P['nu']}", flush=True)
