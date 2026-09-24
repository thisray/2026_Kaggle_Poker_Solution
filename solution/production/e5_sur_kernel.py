"""Surprisal / value-ignoring ordered seat-pair features per hand (actor i -> partner j)."""
import numpy as np, time, os
from numba import njit, prange
OUT = __import__("os").environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
TAG = os.environ.get("TAG", "v1"); PROBS = os.environ.get("PROBS", "dec_probs_v1.npy")
R2_NAMES = ["sur_sum_act","sur_max_act","n_hi3_act","n_hi5_act","sur_fold_to","sur_max_fold_to","sur_pass_hu","sur_max_pass_hu","sur_aggr_act","sur_max_aggr_act",
            "fold_to_strong_hs","fold_to_strong_eq","check_hu_strong_hs","check_hu_strong_eq","weak_aggr_act","xs_fold_to","xs_aggr_act","xs_pass_hu","n_dec_act","n_facing",
            "call_to_dead_eq","xs_call_to","sur_call_to","big_fold_to_strong"]
P2_NAMES = ["sur_sum","sur_max","n_hi3","n_hi5","n_dec","xs_fold","xs_aggr"]
@njit(parallel=True, cache=True)
def run(nh, off, a_st, a_seat, a_act, a_amt, a_tc, a_pot, h_bb, probs, eqm, eqla, HS1, R2, P2):
    for h in prange(nh):
        active = np.ones(6, np.int64); last_aggr = -1; cur = -1; bb = float(h_bb[h])
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                cur = st; last_aggr = -1
            i = a_seat[k]; act = a_act[k]; tc = float(a_tc[k]); amt = float(a_amt[k])
            aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
            passive = act == 1 or act == 2 or (act == 5 and amt <= tc)
            y = 0 if act == 0 else (1 if act == 1 else (3 if aggr else 2))
            p = probs[k, y]
            if p < 1e-6: p = 1e-6
            sur = -np.log(p)
            hs = HS1[h, st, i]
            nopp = 0
            for s in range(6):
                if s != i and active[s] == 1: nopp += 1
            P2[h, i, 0] += sur; P2[h, i, 4] += 1
            if sur > P2[h, i, 1]: P2[h, i, 1] = sur
            if sur > 3: P2[h, i, 2] += 1
            if sur > 5: P2[h, i, 3] += 1
            if tc > 0: P2[h, i, 5] += (1.0 if act == 0 else 0.0) - probs[k, 0]
            P2[h, i, 6] += (1.0 if aggr else 0.0) - probs[k, 3]
            for j in range(6):
                if j == i or active[j] == 0: continue
                R2[h, i, j, 0] += sur
                if sur > R2[h, i, j, 1]: R2[h, i, j, 1] = sur
                if sur > 3: R2[h, i, j, 2] += 1
                if sur > 5: R2[h, i, j, 3] += 1
                R2[h, i, j, 18] += 1
                if aggr:
                    R2[h, i, j, 8] += sur
                    if sur > R2[h, i, j, 9]: R2[h, i, j, 9] = sur
                    if hs < 0.35: R2[h, i, j, 14] += 1
                R2[h, i, j, 16] += (1.0 if aggr else 0.0) - probs[k, 3]
                if nopp == 1 and tc == 0.0 or (nopp == 1 and passive):
                    pass
                if nopp == 1:
                    if passive or act == 1:
                        R2[h, i, j, 6] += sur
                        if sur > R2[h, i, j, 7]: R2[h, i, j, 7] = sur
                    if act == 1:
                        if hs >= 0.8: R2[h, i, j, 12] += 1
                        if eqm[k] >= 0.9: R2[h, i, j, 13] += 1
                    R2[h, i, j, 17] += (1.0 if (passive or act == 1) else 0.0) - probs[k, 1] - probs[k, 2]
            la = last_aggr
            if la >= 0 and la != i:
                R2[h, i, la, 19] += 1
                R2[h, i, la, 15] += (1.0 if act == 0 else 0.0) - probs[k, 0]
                if act == 0:
                    R2[h, i, la, 4] += sur
                    if sur > R2[h, i, la, 5]: R2[h, i, la, 5] = sur
                    if hs >= 0.75: R2[h, i, la, 10] += 1
                    if eqla[k] >= 0.9: R2[h, i, la, 11] += 1
                    if hs >= 0.75 and float(a_pot[k]) / bb >= 10: R2[h, i, la, 23] += 1
                elif passive:
                    R2[h, i, la, 22] += sur
                    R2[h, i, la, 21] += (1.0 if passive else 0.0) - probs[k, 2]
                    if eqla[k] <= 0.1 and eqla[k] >= 0: R2[h, i, la, 20] += 1
            if act == 0: active[i] = 0
            if aggr: last_aggr = i
if __name__ == "__main__":
    t0 = time.time()
    L = lambda n: np.load(f"{D}/{n}.npy")
    off = L("a_off"); nh = len(off) - 1
    args = [L(x) for x in ["a_st","a_seat","a_act","a_amount","a_to_call","a_pot_before"]]
    probs = np.load(f"{OUT}/{PROBS}"); eqm = np.load(f"{OUT}/act_eqm_v1.npy"); eqla = np.load(f"{OUT}/act_eqla_v1.npy"); HS1 = np.load(f"{OUT}/HS1.npy")
    R2 = np.zeros((nh, 6, 6, len(R2_NAMES)), np.float32); P2 = np.zeros((nh, 6, len(P2_NAMES)), np.float32)
    run(nh, off, *args, L("h_bb"), probs, eqm, eqla, HS1, R2, P2)
    print("kernel", time.time() - t0, flush=True)
    np.save(f"{OUT}/R2_{TAG}.npy", R2); np.save(f"{OUT}/P2_{TAG}.npy", P2)
    with open(f"{OUT}/feature_names2_{TAG}.txt", "w") as f:
        f.write("R:" + ",".join(R2_NAMES) + "\nP:" + ",".join(P2_NAMES) + "\n")
    print("saved", time.time() - t0)
