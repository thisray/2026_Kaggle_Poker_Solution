"""When 'active' and the partner is weak, what does the member do?  Action mix (fold / check / call / aggressive) of the
first preflop decision by own-strength x partner-strength, members vs controls; plus the fitted active-policy action mix."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32)
@njit(cache=True)
def first_actions(H, S, T, off, a_seat, a_st, Y, pfeq, out):
    n = 0
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        for d in range(2):
            k1 = ka if d == 0 else kb; k2 = kb if d == 0 else ka; a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            if k1 < 0 or (k2 >= 0 and k2 < k1): continue
            out[n, 0] = pfeq[h, a]; out[n, 1] = pfeq[h, b]; out[n, 2] = Y[k1]; n += 1
    return n
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
mids = pd.read_csv(f"{C}/r2d_p2comb_other.csv", usecols=["pair_id", "predicted_behavior"]); mids = set(mids[mids.predicted_behavior == "other_coordination"].pair_id)
OWN = [0, 0.35, 0.45, 0.55, 0.65, 1.0]; PAR = [0, 0.4, 0.5, 0.6, 0.7, 1.0]
for nm, slots in [("member", c38[c38.pair_id.isin(mids)].slot.values), ("control", c38[c38.rk > 5000].sample(3000, random_state=0).slot.values)]:
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T = H[m], S[m], T[m]
    out = np.zeros((2 * len(H), 3)); n = first_actions(H, S, T, off, a_seat, a_st, Y, pfeq, out)
    A = pd.DataFrame(out[:n], columns=["eA", "eB", "y"]); A["own"] = pd.cut(A.eA, OWN); A["par"] = pd.cut(A.eB, PAR)
    for lab, v in [("fold", 0), ("check", 1), ("call", 2), ("aggr", 3)]: A[lab] = (A.y == v).astype(float)
    print(f"===== {nm}")
    for lab in ["fold", "call", "check"]:
        print(f"  P({lab} | own x partner):\n", A.groupby(["own", "par"], observed=True)[lab].mean().unstack().round(3).to_string())
