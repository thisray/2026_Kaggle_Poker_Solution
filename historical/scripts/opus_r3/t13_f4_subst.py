"""Fourth-family mechanism, exact version: does the member play the NORMAL policy with the PARTNER's cards?
q_sub(a) = policy_v2(features with the card block [hs1, hs2, cat, pf_eq] taken from the partner's seat), q_max = card block of the
stronger of (own, partner).  Per-decision mixtures (1-a) q0 + a q_alt, fitted per street; LL gain vs the exponential tilt."""
import numpy as np, pandas as pd, time, lightgbm as lgb, sys
from scipy.optimize import minimize_scalar
import m4b_policy_v2 as PV
exec(open("s83_hand_tilt.py").read().split("c38 = pd.read_parquet")[0].replace("@njit(cache=True)", "@njit"))
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
C_ = PV.ctx_counts(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_tc, PV.s_player, PV.h_phase, PV.Y)
EX = np.zeros((PV.N, len(PV.FN2)), np.float32)
PV.extra(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_act, PV.a_amt, PV.a_amt_to, PV.a_tc, PV.a_pot, PV.s_player, PV.h_phase, C_, EX); log("extra feats")
bst = lgb.Booster(model_file=f"{OUT}/policy_v2.txt")
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); HS2 = np.load(f"{OUT}/HS2.npy", mmap_mode="r"); CAT = np.load(f"{OUT}/CAT.npy", mmap_mode="r")
PF = np.asarray(Pt[:, :, 12]).astype(np.float32)
log("HS1 shape", HS1.shape, "CAT shape", CAT.shape)
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior"])
mids = set(base[base.predicted_behavior == "other_coordination"].pair_id)
def rows_for(slots):
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    rows = np.zeros((len(H) * 16, 5), np.int64); n = collect(H, S, T, off, a_seat, a_st, Y, rows); rows = rows[:n]
    r, k, s, o, st = rows.T
    return pd.DataFrame({"slot": SL[r], "h": H[r], "k": k, "s": s, "o": o, "st": st, "y": Y[k]})
def q_alt(R):
    k = R.k.values; h = R.h.values; st = R.st.values; s = R.s.values; o = R.o.values
    order = np.argsort(k); Xo = np.hstack([np.asarray(PV.X1[k[order]]), EX[k[order]]]); inv = np.empty_like(order); inv[order] = np.arange(len(order)); Xo = Xo[inv]
    Xs = Xo.copy(); Xs[:, 14] = HS1[h, st, o]; Xs[:, 15] = HS2[h, st, o]; Xs[:, 16] = CAT[h, st, o]; Xs[:, 17] = PF[h, o]
    q0 = bst.predict(Xo, num_threads=2); qs = bst.predict(Xs, num_threads=2)
    better = np.where(st == 0, PF[h, o] > PF[h, s], HS2[h, st, o] > HS2[h, st, s])
    Xm = np.where(better[:, None], Xs, Xo); qm = bst.predict(Xm, num_threads=2)
    return q0, qs, qm
def mix_ll(q0, qa, y, st):
    p0 = np.clip(q0[np.arange(len(y)), y], 1e-9, 1); pa = np.clip(qa[np.arange(len(y)), y], 1e-9, 1)
    out = {}; tot = 0.0
    for sname, m in (("pre", st == 0), ("post", st > 0)):
        f = lambda a: -np.sum(np.log((1 - a) * p0[m] + a * pa[m]) - np.log(p0[m]))
        r = minimize_scalar(f, bounds=(1e-4, 0.9999), method="bounded"); out[sname] = (round(r.x, 3), round(-r.fun, 1)); tot += -r.fun
    return out, round(tot, 1)
msl = c38[c38.pair_id.isin(mids)].slot.values; csl = c38[c38.rk > 5000].sample(800, random_state=9).slot.values
for nm, sl in (("members", msl), ("controls", csl)):
    R = rows_for(sl); q0, qs, qm = q_alt(R); log(nm, "decisions", len(R))
    chk = np.abs(q0 - np.asarray(P2[R.k.values])).max(); log("  max |q0 - dec_probs_v2| =", round(float(chk), 4))
    y = R.y.values; st = R.st.values
    for tag, qa in (("SUBST partner cards", qs), ("MAX(own,partner)", qm)):
        o, tot = mix_ll(q0, qa, y, st); log(f"  {tag:22s} (alpha, LLgain) {o}  total {tot}")
    np.save(f"{OUT}/t13_{nm}_qsub.npy", np.hstack([R[["slot", "h", "k", "s", "o", "st", "y"]].values.astype(np.float64), q0, qs, qm]))
