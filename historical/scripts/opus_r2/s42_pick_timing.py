"""Timing of the fourth-family evidence picks (r2d_ev, r2e) and density of high-c hands early in the eval phase."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy"); ts = np.load(f"{D}/h_ts.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); MU = float(pfeq.mean())
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi))
@njit(cache=True)
def cfirst(H, S, T, off, a_seat, a_st, Y, P2, pfeq, mu, out):
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1; best = 0.0
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        for d in range(2):
            k1 = ka if d == 0 else kb; k2 = kb if d == 0 else ka; b = T[r] if d == 0 else S[r]
            if k1 < 0 or (k2 >= 0 and k2 < k1): continue
            v = (((1.0 if Y[k1] == 3 else 0.0) - P2[k1, 3]) - ((1.0 if Y[k1] == 0 else 0.0) - P2[k1, 0])) * (pfeq[h, b] - mu)
            if v > best: best = v
        out[r] = best
d = pd.read_csv(f"{C}/r2d_p2comb_other_ev_on_r15.csv", dtype=str); e = pd.read_csv(f"{C}/r2e_p2comb_other_evall_on_r15.csv", dtype=str)
mem = d[d.predicted_behavior == "other_coordination"].pair_id
sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]; sm = sm[sm.pair_id.isin(mem)]
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, sm.slot.values); H, S, T, SL = H[m], S[m], T[m], SL[m]
c = np.zeros(len(H)); cfirst(H, S, T, off, a_seat, a_st, Y, P2, pfeq, MU, c)
X = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "c": c}); X["tpct"] = X.groupby("slot").ts.rank(pct=True); X["order"] = X.groupby("slot").ts.rank()
X = X.merge(sm, on="slot")
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
for nm, sub in [("r2d_ev (first-decision c)", d), ("r2e (all-decision c)", e)]:
    picks = sub[sub.pair_id.isin(mem)].melt(id_vars="pair_id", value_vars=EV).value.map(id2hi)
    P = X[X.h.isin(set(picks))]
    print(f"{nm}: picked-hand tpct median {P.tpct.median():.3f}  mean {P.tpct.mean():.3f}  share in first 20% {(P.tpct <= 0.2).mean():.3f}  mean order {P.order.mean():.1f}")
for thr in [0.05, 0.1, 0.15]:
    g = X[X.c > thr].groupby("slot"); first5 = g.order.apply(lambda s: np.sort(s.values)[:5].max() if len(s) >= 5 else np.nan)
    print(f"c>{thr}: per pair count median {g.size().median():.0f}; order index of the 5th such hand median {first5.median():.1f} (pair hands median {X.groupby('slot').size().median():.0f})")
