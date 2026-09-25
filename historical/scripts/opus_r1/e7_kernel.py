"""Ordered seat-pair features for information sharing, whipsaw, and refined surprisal counts."""
import numpy as np, time, os
from numba import njit, prange
from pk_eval import eval_cards
from e2_kernel import street_scores
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
R3_NAMES = ["n_aggr_sur4_act","n_fold_sur3_to","n_pass_sur3_hu","n_call_sur3_to","whipsaw","whipsaw_sur",
            "mw_dec_partner_strong","mw_fold_partner_strong","mw_aggr_partner_strong","mw_dec_partner_weak","mw_fold_partner_weak","mw_aggr_partner_weak",
            "mw_dec_opp_strong","mw_fold_opp_strong","mw_aggr_opp_strong","xs_fold_partner_strong","xs_aggr_partner_strong",
            "sum_partner_eq_at_fold","n_fold_mw","sum_partner_eq_at_aggr","n_aggr_mw","outsider_called_then_squeezed","eq_gain_partner_after_fold"]
@njit(parallel=True, cache=True)
def run(nh, off, a_st, a_seat, a_act, a_amt, a_tc, h_board, s_c1, s_c2, probs, K, R3):
    for h in prange(nh):
        rc = np.zeros(13, np.int64); sc = np.zeros(4, np.int64); sm = np.zeros(4, np.int64)
        cards7 = np.zeros(7, np.int64); deck = np.zeros(52, np.int64); used = np.zeros(52, np.int64)
        scores = np.zeros((K, 6), np.int64); board = np.zeros(5, np.int64); nb = 0
        for b in range(5):
            if h_board[h, b] >= 0:
                board[b] = h_board[h, b]; nb += 1
        c1 = np.zeros(6, np.int64); c2 = np.zeros(6, np.int64)
        for s in range(6):
            c1[s] = s_c1[h, s]; c2[s] = s_c2[h, s]
        active = np.ones(6, np.int64); cur = -1; last_aggr = -1; kk = 1
        called_after = np.zeros(6, np.int64)  # outsider seats that called the current last aggression
        eqs = np.zeros(6)
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                cur = st; last_aggr = -1
                for s in range(6): called_after[s] = 0
                nbk = 0 if st == 0 else 2 + st
                if nbk > nb: nbk = nb
                kk = street_scores(c1, c2, board, nbk, K, h * 7 + st + 1000003, scores, rc, sc, sm, cards7, deck, used)
            i = a_seat[k]; act = a_act[k]; tc = float(a_tc[k]); amt = float(a_amt[k])
            aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
            passive = act == 1 or act == 2 or (act == 5 and amt <= tc)
            y = 0 if act == 0 else (1 if act == 1 else (3 if aggr else 2))
            p = probs[k, y]
            if p < 1e-6: p = 1e-6
            sur = -np.log(p)
            nact = 0
            for s in range(6):
                if active[s] == 1: nact += 1
            # multiway equity of every active seat vs all other active seats (including actor)
            for s in range(6):
                eqs[s] = 0.0
            for q in range(kk):
                best = -1; nbest = 0
                for s in range(6):
                    if active[s] == 1:
                        if scores[q, s] > best:
                            best = scores[q, s]; nbest = 1
                        elif scores[q, s] == best:
                            nbest += 1
                for s in range(6):
                    if active[s] == 1 and scores[q, s] == best:
                        eqs[s] += 1.0 / nbest
            for s in range(6):
                eqs[s] /= kk
            nopp = nact - 1
            for j in range(6):
                if j == i or active[j] == 0: continue
                if aggr and sur > 4: R3[h, i, j, 0] += 1
                if nopp == 1 and passive and sur > 3: R3[h, i, j, 2] += 1
                if nopp >= 2:
                    strong = eqs[j] >= 0.5
                    weak = eqs[j] < 0.15
                    if strong:
                        R3[h, i, j, 6] += 1
                        if act == 0: R3[h, i, j, 7] += 1
                        if aggr: R3[h, i, j, 8] += 1
                        R3[h, i, j, 15] += (1.0 if act == 0 else 0.0) - probs[k, 0]
                        R3[h, i, j, 16] += (1.0 if aggr else 0.0) - probs[k, 3]
                    if weak:
                        R3[h, i, j, 9] += 1
                        if act == 0: R3[h, i, j, 10] += 1
                        if aggr: R3[h, i, j, 11] += 1
                    # "any opponent strong" baseline: counted once per decision on each active j when some other opponent (not j) is strong
                    other_strong = False
                    for o in range(6):
                        if o != i and o != j and active[o] == 1 and eqs[o] >= 0.5: other_strong = True
                    if other_strong:
                        R3[h, i, j, 12] += 1
                        if act == 0: R3[h, i, j, 13] += 1
                        if aggr: R3[h, i, j, 14] += 1
                    if act == 0:
                        R3[h, i, j, 17] += eqs[j]; R3[h, i, j, 18] += 1
                    if aggr:
                        R3[h, i, j, 19] += eqs[j]; R3[h, i, j, 20] += 1
            la = last_aggr
            if la >= 0 and la != i:
                if act == 0 and sur > 3: R3[h, i, la, 1] += 1
                if passive and sur > 3: R3[h, i, la, 3] += 1
                if aggr:
                    # whipsaw: re-raise over la while an outsider who already called la's aggression remains active
                    for o in range(6):
                        if o != i and o != la and active[o] == 1 and called_after[o] == 1:
                            R3[h, i, la, 4] += 1; R3[h, i, la, 5] += sur
                            R3[h, i, la, 21] += 1
                            break
                if act == 0 and nopp >= 2:
                    # equity gain of the last aggressor from this fold (partner benefit proxy): la equity before vs after removing i
                    pass
            if act == 0:
                active[i] = 0
            if aggr:
                last_aggr = i
                for s in range(6): called_after[s] = 0
            elif passive and la >= 0 and la != i:
                called_after[i] = 1
if __name__ == "__main__":
    t0 = time.time()
    L = lambda n: np.load(f"{D}/{n}.npy")
    off = L("a_off"); nh = len(off) - 1
    probs = np.load(f"{OUT}/dec_probs_v1.npy")
    R3 = np.zeros((nh, 6, 6, len(R3_NAMES)), np.float32)
    run(nh, off, L("a_st"), L("a_seat"), L("a_act"), L("a_amount"), L("a_to_call"), L("h_board"), L("s_c1"), L("s_c2"), probs, 64, R3)
    print("kernel", time.time() - t0, flush=True)
    np.save(f"{OUT}/R3_v1.npy", R3)
    with open(f"{OUT}/feature_names3_v1.txt", "w") as f:
        f.write("R:" + ",".join(R3_NAMES) + "\nP:\n")
    print("saved", time.time() - t0)
