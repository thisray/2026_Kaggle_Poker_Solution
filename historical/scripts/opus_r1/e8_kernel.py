"""Probability-weighted suppressed/forced action features per ordered seat pair (actor i -> partner j)."""
import numpy as np, time, os, sys
from numba import njit, prange
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
PROBS = sys.argv[1] if len(sys.argv) > 1 else "dec_probs_v2.npy"; TAG = sys.argv[2] if len(sys.argv) > 2 else "v2"
R4_NAMES = ["supp_aggr_act","supp_aggr_act_max","supp_aggr_hu","supp_aggr_hu_max","forced_fold_to","forced_fold_to_max","forced_aggr_mw","forced_aggr_mw_max",
            "forced_call_to","forced_call_to_max","supp_aggr_hu_eqwin","forced_fold_to_eqwin","forced_aggr_mw_weak","n_passive_act","n_fold_to","n_aggr_mw","n_call_to",
            "sur_sum_act","sur_max_act","sur_fold_to_max","sur_aggr_mw_max","sur_pass_hu_max","supp_aggr_pre_partner_open","n_pre_partner_open"]
P4_NAMES = ["sur_sum","sur_max","n_dec","supp_aggr_sum","forced_fold_sum","forced_aggr_sum"]
@njit(parallel=True, cache=True)
def run(nh, off, a_st, a_seat, a_act, a_amt, a_tc, probs, eqm, eqla, HS1, R4, P4):
    for h in prange(nh):
        active = np.ones(6, np.int64); last_aggr = -1; cur = -1; first_raiser_pre = -1
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                cur = st; last_aggr = -1
            i = a_seat[k]; act = a_act[k]; tc = float(a_tc[k]); amt = float(a_amt[k])
            aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
            passive = act == 1 or act == 2 or (act == 5 and amt <= tc)
            y = 0 if act == 0 else (1 if act == 1 else (3 if aggr else 2))
            pf = probs[k, 0]; pk = probs[k, 1]; pc = probs[k, 2]; pa = probs[k, 3]
            sur = -np.log(max(probs[k, y], 1e-6))
            nopp = 0
            for s in range(6):
                if s != i and active[s] == 1: nopp += 1
            P4[h, i, 0] += sur; P4[h, i, 2] += 1
            if sur > P4[h, i, 1]: P4[h, i, 1] = sur
            if passive: P4[h, i, 3] += pa
            if act == 0: P4[h, i, 4] += 1 - pf
            if aggr: P4[h, i, 5] += 1 - pa
            hs = HS1[h, st, i]
            for j in range(6):
                if j == i or active[j] == 0: continue
                R4[h, i, j, 17] += sur
                if sur > R4[h, i, j, 18]: R4[h, i, j, 18] = sur
                if passive:
                    R4[h, i, j, 0] += pa; R4[h, i, j, 13] += 1
                    if pa > R4[h, i, j, 1]: R4[h, i, j, 1] = pa
                    if nopp == 1:
                        R4[h, i, j, 2] += pa
                        if pa > R4[h, i, j, 3]: R4[h, i, j, 3] = pa
                        if sur > R4[h, i, j, 21]: R4[h, i, j, 21] = sur
                        if eqm[k] >= 0.8: R4[h, i, j, 10] += pa
                if aggr and nopp >= 2:
                    R4[h, i, j, 6] += 1 - pa; R4[h, i, j, 15] += 1
                    if 1 - pa > R4[h, i, j, 7]: R4[h, i, j, 7] = 1 - pa
                    if hs < 0.45: R4[h, i, j, 12] += 1 - pa
                    if sur > R4[h, i, j, 20]: R4[h, i, j, 20] = sur
                if st == 0 and first_raiser_pre == j and passive:
                    R4[h, i, j, 22] += pa; R4[h, i, j, 23] += 1
            la = last_aggr
            if la >= 0 and la != i:
                if act == 0:
                    R4[h, i, la, 4] += 1 - pf; R4[h, i, la, 14] += 1
                    if 1 - pf > R4[h, i, la, 5]: R4[h, i, la, 5] = 1 - pf
                    if eqla[k] >= 0.5: R4[h, i, la, 11] += 1 - pf
                    if sur > R4[h, i, la, 19]: R4[h, i, la, 19] = sur
                elif passive:
                    R4[h, i, la, 8] += 1 - pc; R4[h, i, la, 16] += 1
                    if 1 - pc > R4[h, i, la, 9]: R4[h, i, la, 9] = 1 - pc
            if act == 0: active[i] = 0
            if aggr:
                last_aggr = i
                if st == 0 and first_raiser_pre < 0: first_raiser_pre = i
if __name__ == "__main__":
    t0 = time.time()
    L = lambda n: np.load(f"{D}/{n}.npy")
    off = L("a_off"); nh = len(off) - 1
    probs = np.load(f"{OUT}/{PROBS}"); eqm = np.load(f"{OUT}/act_eqm_v1.npy"); eqla = np.load(f"{OUT}/act_eqla_v1.npy"); HS1 = np.load(f"{OUT}/HS1.npy")
    R4 = np.zeros((nh, 6, 6, len(R4_NAMES)), np.float32); P4 = np.zeros((nh, 6, len(P4_NAMES)), np.float32)
    run(nh, off, L("a_st"), L("a_seat"), L("a_act"), L("a_amount"), L("a_to_call"), probs, eqm, eqla, HS1, R4, P4)
    np.save(f"{OUT}/R4_{TAG}.npy", R4); np.save(f"{OUT}/P4_{TAG}.npy", P4)
    open(f"{OUT}/feature_names4_{TAG}.txt", "w").write("R:" + ",".join(R4_NAMES) + "\nP:" + ",".join(P4_NAMES) + "\n")
    print("done", time.time() - t0)
