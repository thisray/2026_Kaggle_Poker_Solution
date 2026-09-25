"""Shared kernel: pair-level Bayes factor of the hand-level partner-card tilt mixture (parameters fitted in s83 on the 77
eval fourth-family members)."""
import numpy as np
from numba import njit
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
BETA = np.array([20.795, 1.987, 1.673, 1.855]); PI_ = 0.498
def load_inputs():
    off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
    Q0 = np.clip(np.load(f"{OUT}/dec_probs_v2.npy").astype(np.float64), 1e-6, 1.0); Q0 /= Q0.sum(1, keepdims=True)
    Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
    pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float64); eql = np.asarray(Pt[:, :, PN.index("eq_last")]).astype(np.float64)
    MU = np.array([float(pfeq.mean()), float(eql.mean()), float(eql.mean()), float(eql.mean())])
    return off, a_seat, a_st, Y, Q0, pfeq, eql, MU
@njit(cache=True)
def pair_bf(H, S, T, SL, off, a_seat, a_st, Y, Q0, pfeq, eql, BETA, MU, PI_, BF, NH, NA):
    for r in range(len(H)):
        h = H[r]; l = 0.0; nd = 0
        act = np.ones(6, np.int64)
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]
            if s == S[r] or s == T[r]:
                o = T[r] if s == S[r] else S[r]
                if act[o] == 1:
                    st = a_st[k]
                    e = (pfeq[h, o] if st == 0 else eql[h, o]) - MU[st]
                    be = BETA[st] * e
                    z = Q0[k, 0] * np.exp(-be) + Q0[k, 1] + Q0[k, 2] + Q0[k, 3] * np.exp(be)
                    t = 0.0
                    if Y[k] == 3: t = 1.0
                    elif Y[k] == 0: t = -1.0
                    l += be * t - np.log(z); nd += 1
            if Y[k] == 0: act[s] = 0
        sl = SL[r]
        if nd > 0:
            m = l if l > 0 else 0.0
            BF[sl] += m + np.log((1 - PI_) * np.exp(-m) + PI_ * np.exp(l - m)); NA[sl] += 1
        NH[sl] += 1
