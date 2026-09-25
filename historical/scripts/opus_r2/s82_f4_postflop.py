"""Fourth family beyond the first preflop decisions: in hands where both members are still active, do members' decisions
depend on the PARTNER's current (omniscient) strength?  Per street, policy-v2 fold/aggression residuals vs partner's
eq_last-style strength proxy; members vs controls.  Also pair-vs-pair confrontation patterns (who folds to whom)."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); eql = np.asarray(Pt[:, :, PN.index("eq_last")]).astype(np.float32)
eqf = np.asarray(Pt[:, :, PN.index("eq_first")]).astype(np.float32)
@njit(cache=True)
def collect(H, S, T, off, a_seat, a_st, Y, rows):
    n = 0
    for r in range(len(H)):
        h = H[r]; active = np.ones(6, np.int64); firstdone = np.zeros(6, np.int64)
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]
            if s == S[r] or s == T[r]:
                o = T[r] if s == S[r] else S[r]
                if firstdone[s] == 1 and active[o] == 1:        # later decision of a member while the partner is still in the hand
                    rows[n, 0] = r; rows[n, 1] = k; rows[n, 2] = s; rows[n, 3] = o; rows[n, 4] = a_st[k]; n += 1
                firstdone[s] = 1
            if Y[k] == 0: active[s] = 0
    return n
def table(slots, tag):
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    rows = np.zeros((len(H) * 12, 5), np.int64); n = collect(H, S, T, off, a_seat, a_st, Y, rows); rows = rows[:n]
    r, k, s, o, st = rows.T; h = H[r]
    kk = np.argsort(k); pr = np.asarray(P2[k[kk]]); prr = np.empty_like(pr); prr[kk] = pr
    X = pd.DataFrame({"slot": SL[r], "h": h, "st": st, "y": Y[k], "pf": prr[:, 0], "pa": prr[:, 3],
                      "e_par_pf": pfeq[h, o], "e_par_last": eql[h, o], "e_own_last": eql[h, s]})
    X["grp"] = tag; return X
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior"])
mids = set(base[base.predicted_behavior == "other_coordination"].pair_id)
Mt = table(c38[c38.pair_id.isin(mids)].slot.values, "member"); Ct = table(c38[c38.rk > 5000].sample(4000, random_state=2).slot.values, "control")
X = pd.concat([Mt, Ct])
X["rf"] = (X.y == 0) - X.pf; X["ra"] = (X.y == 3) - X.pa
for ecol in ["e_par_pf", "e_par_last"]:
    X["e"] = X[ecol] - X[ecol].mean()
    print(f"== partner strength = {ecol}")
    for (g, st), G in X.groupby(["grp", "st"]):
        zf = (G.rf * G.e).sum() / np.sqrt((G.pf * (1 - G.pf) * G.e ** 2).sum() + 1e-9); za = (G.ra * G.e).sum() / np.sqrt((G.pa * (1 - G.pa) * G.e ** 2).sum() + 1e-9)
        print(f"   {g:8s} street {st}: n={len(G):6d}  pooled z_fold {zf:7.2f}  z_aggr {za:7.2f}  (per-decision effect: fold {(G.rf * G.e).mean() / (G.e ** 2).mean():.3f}, aggr {(G.ra * G.e).mean() / (G.e ** 2).mean():.3f})")
X.to_parquet(f"{OUT}/s82_f4_postflop.parquet")
