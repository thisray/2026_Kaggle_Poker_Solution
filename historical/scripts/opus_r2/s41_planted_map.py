"""Planted-raise probability map for the fourth family, estimated WITHIN member pairs from the partner-card contrast:
for the actor's first preflop decision (partner not yet acted), P(aggressive | own-strength bin, partner strong) minus
P(aggressive | same own bin, partner weak) is the part caused by partner information (natural play cannot depend on it)."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy")
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
            k1 = ka if d == 0 else kb; k2 = kb if d == 0 else ka
            a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            if k1 < 0 or (k2 >= 0 and k2 < k1): continue
            out[n, 0] = r; out[n, 1] = pfeq[h, a]; out[n, 2] = pfeq[h, b]; out[n, 3] = 1.0 if Y[k1] == 3 else 0.0; out[n, 4] = 1.0 if Y[k1] == 0 else 0.0; n += 1
    return n
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
mids = pd.read_csv(f"{OUT}/r2_candidates/r2d_p2comb_other.csv", usecols=["pair_id", "predicted_behavior"]); mids = set(mids[mids.predicted_behavior == "other_coordination"].pair_id)
res = {}
for nm, slots in [("member", c38[c38.pair_id.isin(mids)].slot.values), ("control", c38[c38.rk > 5000].sample(3000, random_state=0).slot.values)]:
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T = H[m], S[m], T[m]
    out = np.zeros((2 * len(H), 5)); n = first_actions(H, S, T, off, a_seat, a_st, Y, pfeq, out); A = pd.DataFrame(out[:n], columns=["r", "eA", "eB", "aggr", "fold"])
    A["own"] = pd.cut(A.eA, [0, 0.35, 0.45, 0.55, 0.65, 1.0]); A["partner"] = pd.cut(A.eB, [0, 0.4, 0.5, 0.6, 0.7, 1.0])
    res[nm] = A.groupby(["own", "partner"], observed=True).agg(aggr=("aggr", "mean"), fold=("fold", "mean"), n=("aggr", "size"))
    print(f"== {nm}: P(aggressive | own bin x partner bin)\n", res[nm].aggr.unstack().round(3).to_string(), "\n n per cell:\n", res[nm].n.unstack().to_string(), flush=True)
M = res["member"].aggr.unstack(); Cc = res["control"].aggr.unstack()
base = M.iloc[:, 0]   # member rate when partner weakest (<0.4) as own-style baseline
planted = M.sub(base, axis=0).clip(lower=0).div(M.clip(lower=1e-6))
print("\nplanted fraction among member aggressive first actions (partner-caused share), by own x partner bin:\n", planted.round(2).to_string())
print("\ncontrol partner-dependence check (should be ~flat):\n", Cc.sub(Cc.iloc[:, 0], axis=0).round(3).to_string())
planted.to_pickle(f"{OUT}/s41_planted_map.pkl")
