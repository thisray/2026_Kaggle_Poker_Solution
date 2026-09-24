"""Hand-level conditional-tilt mixture for the fourth family (Round16 idea, implemented on real data).
Every decision d of either member while the partner is still active (all streets):  q_beta(a) = q0(a) exp(b_st e t(a)) / Z,
t(aggr)=+1, t(fold)=-1, else 0; e = partner strength (pf_eq preflop, eq_last postflop), centred per street.
Hand log-LR  l_h = sum_d [b e t(a_d) - log Z_d];  pair likelihood  prod_h [(1-pi) + pi exp(l_h)].
EM on the 77 member pairs (global pi, 4 street betas); controls must give pi ~ 0.  Outputs per-hand activity posterior."""
import os
import numpy as np, pandas as pd, time
from numba import njit
from scipy.optimize import minimize
import pairindex as PI
OUT = os.environ["POKER_WORK_DIR"]; D = f"{OUT}/np"; C = OUT
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); eql = np.asarray(Pt[:, :, PN.index("eq_last")]).astype(np.float32)
@njit(cache=True)
def collect(H, S, T, off, a_seat, a_st, Y, rows):
    n = 0
    for r in range(len(H)):
        h = H[r]; active = np.ones(6, np.int64)
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]
            if s == S[r] or s == T[r]:
                o = T[r] if s == S[r] else S[r]
                if active[o] == 1:
                    rows[n, 0] = r; rows[n, 1] = k; rows[n, 2] = s; rows[n, 3] = o; rows[n, 4] = a_st[k]; n += 1
            if Y[k] == 0: active[s] = 0
    return n
def table(slots):
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    rows = np.zeros((len(H) * 16, 5), np.int64); n = collect(H, S, T, off, a_seat, a_st, Y, rows); rows = rows[:n]
    r, k, s, o, st = rows.T; h = H[r]
    kk = np.argsort(k); pr = np.asarray(P2[k[kk]]); prr = np.empty_like(pr); prr[kk] = pr
    X = pd.DataFrame({"slot": SL[r], "h": h, "st": st, "y": Y[k]})
    X[["q_f", "q_k", "q_c", "q_a"]] = np.clip(prr, 1e-6, 1)
    X["e"] = np.where(st == 0, pfeq[h, o], eql[h, o])
    return X
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
base = pd.read_csv(f"{C}/f4_routed_baseline.csv", usecols=["pair_id", "predicted_behavior"])
mids = set(base[base.predicted_behavior == "other_coordination"].pair_id)
msl = c38[c38.pair_id.isin(mids)].slot.values; csl = c38[c38.rk > 5000].sample(3000, random_state=3).slot.values
Mx = table(msl); Cx = table(csl); log("decisions: members", len(Mx), "controls", len(Cx))
mu = Cx.groupby("st").e.mean(); Mx["e"] -= Mx.st.map(mu); Cx["e"] -= Cx.st.map(mu)
def prep(X):
    t = np.select([X.y.values == 3, X.y.values == 0], [1.0, -1.0], 0.0)
    q = X[["q_f", "q_k", "q_c", "q_a"]].to_numpy(); q = q / q.sum(1, keepdims=True)
    hid = (X.slot.astype(np.int64) * 10_000_000 + X.h.astype(np.int64)).values
    u, inv = np.unique(hid, return_inverse=True)
    return dict(t=t, q=q, e=X.e.values.astype(float), st=X.st.values.astype(int), inv=inv, nh=len(u), hkey=u)
TT = np.array([-1.0, 0.0, 0.0, 1.0])     # t for classes fold, check, call, aggr
def hand_llr(P, beta):
    b = beta[P["st"]]; be = b * P["e"]
    logZ = np.log((P["q"] * np.exp(be[:, None] * TT[None, :])).sum(1))
    l = be * P["t"] - logZ
    return np.bincount(P["inv"], weights=l, minlength=P["nh"])
def fit(P, iters=40):
    beta = np.array([1.0, 0.5, 0.5, 0.5]); pi = 0.3
    for it in range(iters):
        lh = hand_llr(P, beta)
        r = pi * np.exp(np.clip(lh, -50, 50)); r = r / (r + 1 - pi)
        pi = float(np.clip(r.mean(), 1e-4, 0.999))
        w = r[P["inv"]]
        def negll(bv):
            b = bv[P["st"]]; be = b * P["e"]; logZ = np.log((P["q"] * np.exp(be[:, None] * TT[None, :])).sum(1))
            return -np.sum(w * (be * P["t"] - logZ)) + 0.5 * 0.01 * np.sum(bv ** 2)
        beta = minimize(negll, beta, method="L-BFGS-B").x
    lh = hand_llr(P, beta); ll = np.sum(np.log((1 - pi) + pi * np.exp(np.clip(lh, -50, 50))))
    return beta, pi, ll
PM = prep(Mx); PC = prep(Cx)
bM, piM, llM = fit(PM); log("members: beta", np.round(bM, 3), "pi", round(piM, 3), "mixture LL gain", round(llM, 1), "hands", PM["nh"])
# controls with the member betas: pi estimated with beta fixed
lhC = hand_llr(PC, bM); pi = 0.3
for it in range(100):
    r = pi * np.exp(np.clip(lhC, -50, 50)); r = r / (r + 1 - pi); pi = float(np.clip(r.mean(), 1e-5, 0.999))
log("controls with member betas: pi", round(pi, 4), " mean hand LLR", round(float(lhC.mean()), 3), " members mean hand LLR", round(float(hand_llr(PM, bM).mean()), 3))
bC, piC, llC = fit(PC); log("controls own fit: beta", np.round(bC, 3), "pi", round(piC, 3), "LL gain", round(llC, 1))
lhM = hand_llr(PM, bM); post = piM * np.exp(np.clip(lhM, -50, 50)); post = post / (post + 1 - piM)
Hh = pd.DataFrame({"hkey": PM["hkey"], "llr": lhM, "post": post}); Hh["slot"] = Hh.hkey // 10_000_000; Hh["h"] = Hh.hkey % 10_000_000
Hh["ts"] = ts[Hh.h.values]
Hh.to_parquet(f"{OUT}/s83_hand_tilt_posterior.parquet"); log("saved", len(Hh), "member hands; mean posterior", round(Hh.post.mean(), 3))
