"""Fourth family: where does the partner-card tilt live?  Per-decision mixture (alpha, beta) fitted separately by context:
players active at the decision (6 = nobody folded yet), first member decision of the hand vs later, partner acted before or not,
actor = first-acting member vs responder, street.  Members (77) vs controls."""
import numpy as np, pandas as pd, time
from scipy.optimize import minimize
exec(open("s83_hand_tilt.py").read().split("c38 = pd.read_parquet")[0].replace("@njit(cache=True)", "@njit"))
a_pa = np.load(f"{D}/a_players_active.npy")
from numba import njit
@njit
def ctx(H, S, T, off, a_seat, a_st, Y, rows_k, out):
    # for every member decision k (partner active): out = [first member decision in hand?, partner acted before (this street or earlier)?, actor is first-acting member?]
    n = 0
    for r in range(len(H)):
        h = H[r]; act = np.ones(6, np.int64); first_seat = -1; acted = np.zeros(6, np.int64); ndec = 0
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]
            if s == S[r] or s == T[r]:
                if first_seat < 0: first_seat = s
                o = T[r] if s == S[r] else S[r]
                if act[o] == 1:
                    out[n, 0] = 1 if ndec == 0 else 0; out[n, 1] = acted[o]; out[n, 2] = 1 if s == first_seat else 0; rows_k[n] = k; n += 1
                ndec += 1
                acted[s] = 1
            if Y[k] == 0: act[s] = 0
    return n
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior"])
mids = set(base[base.predicted_behavior == "other_coordination"].pair_id)
def build(slots):
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    rows_k = np.zeros(len(H) * 16, np.int64); out = np.zeros((len(H) * 16, 3), np.int64)
    n = ctx(H, S, T, off, a_seat, a_st, Y, rows_k, out); k = rows_k[:n]; o3 = out[:n]
    rows = np.zeros((len(H) * 16, 5), np.int64); n2 = collect(H, S, T, off, a_seat, a_st, Y, rows); rows = rows[:n2]
    assert n2 == n and (rows[:, 1] == k).all()
    r, k, s, o, st = rows.T; h = H[r]
    kk = np.argsort(k); pr = np.asarray(P2[k[kk]]); prr = np.empty_like(pr); prr[kk] = pr
    X = pd.DataFrame({"st": st, "y": Y[k], "pa": a_pa[k], "firstdec": o3[:, 0], "p_acted": o3[:, 1], "first_actor": o3[:, 2]})
    X[["q_f", "q_k", "q_c", "q_a"]] = np.clip(prr, 1e-6, 1); X["e"] = np.where(st == 0, pfeq[h, o], eql[h, o])
    return X
msl = c38[c38.pair_id.isin(mids)].slot.values; csl = c38[c38.rk > 5000].sample(2000, random_state=5).slot.values
Mx = build(msl); Cx = build(csl); mu = Cx.groupby("st").e.mean(); Mx["e"] -= Mx.st.map(mu); Cx["e"] -= Cx.st.map(mu)
TT = np.array([-1.0, 0.0, 0.0, 1.0])
def fit1(X):
    t = np.select([X.y.values == 3, X.y.values == 0], [1.0, -1.0], 0.0); q = X[["q_f", "q_k", "q_c", "q_a"]].to_numpy(); q = q / q.sum(1, keepdims=True); e = X.e.values
    def nll(th):
        b = th[0]; a = 1 / (1 + np.exp(-th[1])); be = b * e
        l = be * t - np.log((q * np.exp(be[:, None] * TT[None, :])).sum(1))
        return -np.sum(np.logaddexp(np.log1p(-a), np.log(a) + l)) + 0.005 * b * b
    best = min((minimize(nll, np.array(x0), method="L-BFGS-B") for x0 in ([1.0, -1.0], [20.0, 0.0], [40.0, -0.7], [4.0, -0.7])), key=lambda r: r.fun)
    return best.x[0], 1 / (1 + np.exp(-best.x[1])), -best.fun, len(X)
pd.set_option("display.width", 200)
rows = []
for nm, X in (("members", Mx), ("controls", Cx)):
    for st in (0, 1):
        Xs = X[(X.st == 0) if st == 0 else (X.st > 0)]
        for gname, gcol in (("pa", "pa"), ("firstdec", "firstdec"), ("p_acted", "p_acted"), ("first_actor", "first_actor")):
            for v, G in Xs.groupby(gcol):
                if len(G) < 150: continue
                b, a, ll, n = fit1(G); rows.append(dict(set=nm, street="pre" if st == 0 else "post", ctx=gname, val=v, n=n, beta=round(b, 2), alpha=round(a, 3), LL=round(ll, 1), LL_per_1k=round(1000 * ll / n, 1)))
R = pd.DataFrame(rows); print(R.to_string(index=False))
