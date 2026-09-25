"""Paired conditional-entry features (R4-02): both partners entering with junk.

Per (hand, seat): first preflop decision, entry probability from policy v1, entered flag.
Per pair-hand row: joint entry deviation features.
"""
import numpy as np
import pandas as pd
from numba import njit, prange

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
D = f"{OUT}/np"
_c = {}


def load():
    if not _c:
        L = lambda n: np.load(f"{D}/{n}.npy")
        _c["off"] = L("a_off")
        _c["a_st"] = L("a_st")
        _c["a_seat"] = L("a_seat")
        _c["a_act"] = L("a_act")
        _c["a_amt"] = L("a_amount")
        _c["a_tc"] = L("a_to_call")
        _c["probs"] = np.load(f"{OUT}/dec_probs_v1.npy")
        _c["HS1"] = np.load(f"{OUT}/HS1.npy")
        nh = len(_c["off"]) - 1
        entry_k = np.full((nh, 6), -1, np.int64)
        entered = np.zeros((nh, 6), np.float32)
        enter_p = np.zeros((nh, 6), np.float32)
        fold_p = np.zeros((nh, 6), np.float32)
        _first_decision(_c["off"], _c["a_st"], _c["a_seat"], _c["a_act"], _c["a_tc"],
                        _c["probs"], entry_k, entered, enter_p, fold_p)
        _c["entry_k"] = entry_k
        _c["entered"] = entered
        _c["enter_p"] = enter_p
        _c["fold_p"] = fold_p
    return _c


@njit(parallel=True, cache=True)
def _first_decision(off, a_st, a_seat, a_act, a_tc, probs, entry_k, entered, enter_p, fold_p):
    nh = len(off) - 1
    for h in prange(nh):
        seen = np.zeros(6, np.int8)
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0:
                break
            s = a_seat[k]
            if seen[s] == 1:
                continue
            seen[s] = 1
            entry_k[h, s] = k
            tc = a_tc[k]
            act = a_act[k]
            pf = probs[k, 0]
            pc = probs[k, 2]
            pa = probs[k, 3]
            fold_p[h, s] = pf
            enter_p[h, s] = pc + pa
            if tc > 0 and (act == 2 or act == 3 or act == 4 or act == 5):
                entered[h, s] = 1.0


NAMES = ["e_lo_a", "e_lo_b", "e_lo_min", "e_lo_sum", "e_both_entered", "e_min_prob",
         "e_second_lo", "e_second_weak", "e_first_weak", "e_tight_max", "e_lo_diff",
         "e_both_weak_lo", "e_lo_second_x_weak"]


def features(h, sa, sb):
    c = load()
    entry_k = c["entry_k"]
    entered = c["entered"]
    enter_p = c["enter_p"]
    fold_p = c["fold_p"]
    HS1 = c["HS1"]
    n = len(h)
    out = np.zeros((n, len(NAMES)), np.float32)
    _gather(h.astype(np.int64), sa.astype(np.int64), sb.astype(np.int64),
            entry_k, entered, enter_p, fold_p, HS1, out)
    return pd.DataFrame(out, columns=NAMES)


@njit(parallel=True, cache=True)
def _gather(hs, sa, sb, entry_k, entered, enter_p, fold_p, HS1, out):
    for r in prange(len(hs)):
        h = hs[r]
        a = sa[r]
        b = sb[r]
        ka = entry_k[h, a]
        kb = entry_k[h, b]
        va = ka >= 0
        vb = kb >= 0
        ea = entered[h, a] > 0
        eb = entered[h, b] > 0
        loa = 0.0
        lob = 0.0
        if va and ea:
            loa = -np.log(max(enter_p[h, a], 1e-6))
        if vb and eb:
            lob = -np.log(max(enter_p[h, b], 1e-6))
        both = 1.0 if (va and vb and ea and eb) else 0.0
        out[r, 4] = both
        if both > 0:
            out[r, 0] = loa
            out[r, 1] = lob
            out[r, 2] = min(loa, lob)
            out[r, 3] = loa + lob
            out[r, 5] = min(enter_p[h, a], enter_p[h, b])
            out[r, 10] = abs(loa - lob)
            # chronological order of first decisions
            if ka < kb:
                out[r, 6] = lob
                w_first = 1.0 if HS1[h, 0, a] < 0.45 else 0.0
                w_second = 1.0 if HS1[h, 0, b] < 0.45 else 0.0
                out[r, 8] = w_first
                out[r, 7] = w_second
                out[r, 12] = lob * w_second
                if w_first > 0 and w_second > 0:
                    out[r, 11] = min(loa, lob)
            else:
                out[r, 6] = loa
                w_first = 1.0 if HS1[h, 0, b] < 0.45 else 0.0
                w_second = 1.0 if HS1[h, 0, a] < 0.45 else 0.0
                out[r, 8] = w_first
                out[r, 7] = w_second
                out[r, 12] = loa * w_second
                if w_first > 0 and w_second > 0:
                    out[r, 11] = min(loa, lob)
        # tightness of the one who folded at first decision
        ta = 0.0
        tb = 0.0
        if va and not ea:
            ta = -np.log(max(1.0 - fold_p[h, a], 1e-6))
        if vb and not eb:
            tb = -np.log(max(1.0 - fold_p[h, b], 1e-6))
        out[r, 9] = max(ta, tb)
