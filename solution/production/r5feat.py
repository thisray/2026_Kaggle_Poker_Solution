"""Round-5 features: role-tagged event sequence + visible-edit action witnesses.

Adapted from the ChatGPT round-5 design (events.py / witness.py) to the opus replay arrays.
A/B are the pair seats; O outsiders. Features are hand-level aggregates per pair-hand row.
"""
import os
import numpy as np
import pandas as pd
from numba import njit, prange

OUT = os.environ["POKER_WORK_DIR"]
D = f"{OUT}/np"
_c = {}

NAMES = [
    "ev_partner_concession", "ev_partner_fold_after_outsider", "ev_both_aggressive",
    "ev_outsider_fold_after_pair_pressure", "ev_middle_hu_transition",
    "wit_r2c_max", "wit_r2c_sum", "wit_r2k_max", "wit_r2k_sum",
    "wit_c2f_max", "wit_c2f_sum", "wit_f2c_max", "wit_f2c_sum",
    "ev_pair_aggr_while_outsiders", "ev_pair_passive_after_pair_aggr",
    "ev_first_pressure_pair", "ev_first_pressure_out", "ev_total_pair_aggr", "ev_total_out_aggr",
]


def load():
    if not _c:
        L = lambda n: np.load(f"{D}/{n}.npy")
        _c["off"] = L("a_off")
        _c["a_st"] = L("a_st")
        _c["a_seat"] = L("a_seat")
        _c["a_act"] = L("a_act")
        _c["a_amt"] = L("a_amount")
        _c["a_tc"] = L("a_to_call")
        _c["probs"] = np.load(f"{OUT}/" + os.environ.get("R5PROBS", "dec_probs_v1.npy"))
    return _c


@njit(inline="always")
def _vis(pa_src, p_dst, y, src_idx, dst_idx):
    """Visible-edit posterior, one-row rewrite src->dst, rate .6, activation .5.

    Nonzero only when the observed action equals the destination.
    """
    rate = 0.6
    if y == dst_idx:
        alt = p_dst + pa_src * rate
        obs = 0.5 * p_dst + 0.5 * alt
        vm = 0.5 * (alt - p_dst)
        if obs <= 0.0:
            return 0.0
        v = vm / obs
        return v if v < 1.0 else 1.0
    return 0.0


@njit(parallel=True, cache=True)
def _run(hs, sas, sbs, off, a_st, a_seat, a_act, a_amt, a_tc, probs, out):
    for r in prange(len(hs)):
        h = hs[r]
        A = sas[r]
        B = sbs[r]
        folded = np.zeros(6, np.int8)
        allin = np.zeros(6, np.int8)
        aggr_street = np.zeros(6, np.int8)
        cur = -1
        last_aggr = -1
        outsider_exit = 0
        n_pair_aggr = 0
        n_out_aggr = 0
        first_pressure = -2
        prev_out_for_partner = -1
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                cur = st
                last_aggr = -1
                aggr_street[:] = 0
                outsider_exit = 0
            i = a_seat[k]
            act = a_act[k]
            amt = a_amt[k]
            tc = a_tc[k]
            ispair = 1 if (i == A or i == B) else 0
            other = B if i == A else (A if i == B else -1)
            acting_outs = 0
            for s in range(6):
                if s != A and s != B and folded[s] == 0 and allin[s] == 0:
                    acting_outs += 1
            partner_can = 0
            if other >= 0 and folded[other] == 0 and allin[other] == 0:
                partner_can = 1
            toward_partner = 1 if (other >= 0 and last_aggr == other) else 0
            pair_pressure = 1 if (last_aggr == A or last_aggr == B) else 0
            no_pressure = 1 if last_aggr < 0 else 0
            aggr = 1 if (act == 3 or act == 4 or (act == 5 and amt > tc)) else 0
            if first_pressure == -2 and aggr == 1:
                first_pressure = 0 if ispair == 1 else 1
            if ispair == 1:
                if partner_can == 1 and prev_out_for_partner >= 0 and acting_outs == 0 and prev_out_for_partner > 0:
                    out[r, 4] += 1.0
                if partner_can == 1:
                    prev_out_for_partner = acting_outs
                if act == 0 and toward_partner == 1:
                    out[r, 0] += 1.0
                    if outsider_exit == 1:
                        out[r, 1] += 1.0
                if aggr == 1:
                    n_pair_aggr += 1
                    if aggr_street[i] == 0:
                        aggr_street[i] = 1
                        if aggr_street[A] == 1 and aggr_street[B] == 1:
                            if acting_outs > 0:
                                out[r, 2] += 1.0
                    if acting_outs > 0:
                        out[r, 13] += 1.0
                elif (act == 1 or act == 2) and pair_pressure == 1:
                    if aggr_street[A] == 1 or aggr_street[B] == 1:
                        out[r, 14] += 1.0
                y = 0 if act == 0 else (1 if act == 1 else (3 if aggr == 1 else 2))
                pf = probs[k, 0]
                pk = probs[k, 1]
                pc = probs[k, 2]
                pa = probs[k, 3]
                if toward_partner == 1:
                    v = _vis(pa, pc, y, 3, 2)
                    if v > out[r, 5]:
                        out[r, 5] = v
                    out[r, 6] += v
                    v = _vis(pc, pf, y, 2, 0)
                    if v > out[r, 9]:
                        out[r, 9] = v
                    out[r, 10] += v
                    v = _vis(pf, pc, y, 0, 2)
                    if v > out[r, 11]:
                        out[r, 11] = v
                    out[r, 12] += v
                if no_pressure == 1 and acting_outs == 0 and partner_can == 1:
                    v = _vis(pa, pk, y, 3, 1)
                    if v > out[r, 7]:
                        out[r, 7] = v
                    out[r, 8] += v
            else:
                if aggr == 1:
                    n_out_aggr += 1
                if act == 0 and pair_pressure == 1:
                    both_raised = 1 if (aggr_street[A] == 1 and aggr_street[B] == 1) else 0
                    if both_raised == 1:
                        outsider_exit = 1
                        out[r, 3] += 1.0
            if act == 0:
                folded[i] = 1
            if act == 5:
                allin[i] = 1
            if aggr == 1:
                last_aggr = i
        out[r, 15] = 1.0 if first_pressure == 0 else 0.0
        out[r, 16] = 1.0 if first_pressure == 1 else 0.0
        out[r, 17] = float(n_pair_aggr)
        out[r, 18] = float(n_out_aggr)


def features(h, sa, sb):
    c = load()
    out = np.zeros((len(h), len(NAMES)), np.float32)
    _run(h.astype(np.int64), sa.astype(np.int64), sb.astype(np.int64), c["off"], c["a_st"],
         c["a_seat"], c["a_act"], c["a_amt"], c["a_tc"], c["probs"], out)
    return pd.DataFrame(out, columns=NAMES)
