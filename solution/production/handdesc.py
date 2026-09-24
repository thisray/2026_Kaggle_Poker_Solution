"""Pair-relative hand sequence descriptors (numba, row-parallel over pair-hands)."""
import numpy as np, pandas as pd
from numba import njit, prange
OUT = __import__("os").environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
DN = ["fa_pair", "fa_out", "pair_aggr", "out_aggr", "o_fold_after_pair", "pair_fold_to_partner", "pair_reraise", "o_vpip_pre", "max_street",
      "ftp_eq_max", "ftp_hs_max", "ftp_pot_bb", "ftp_sunk_bb", "ftp_street", "checks_hu", "aggr_hu", "outs_at_reraise", "hu_check_pa_sum", "weak_aggr_pair",
      "iso_fold", "partner_wins_uncontested", "o_left_before_pair_fold", "pair_calls_partner", "pair_call_dead", "fold_to_partner_pf", "ftp_p_fold_min", "pair_first_raise_pa"]
@njit(parallel=True, cache=True)
def run(hs, sa, sb, off, a_st, a_seat, a_act, a_amt, a_tc, a_pot, h_bb, eqla, HS1, probs, s_contrib, s_won, s_folded, out):
    for r in prange(len(hs)):
        h = hs[r]; A = sa[r]; B = sb[r]; bb = float(h_bb[h])
        active = np.ones(6, np.int64); total = np.zeros(6)
        first_aggr = -1; pair_aggr = 0; out_aggr = 0; o_fold_after_pair = 0; ftp = 0; reraise = 0; o_vpip = 0; max_st = 0
        ftp_eq = -1.0; ftp_hs = -1.0; ftp_pot = 0.0; ftp_sunk = 0.0; ftp_st = -1; checks_hu = 0; aggr_hu = 0; outs_rr = 0; hu_pa = 0.0; weak_aggr = 0.0
        iso_fold = 0; cur = -1; last_aggr = -1; pair_calls = 0; call_dead = 0; ftp_pmin = 1.0; first_raise_pa = -1.0
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                cur = st; last_aggr = -1
            if st > max_st: max_st = st
            i = a_seat[k]; act = a_act[k]; tc = float(a_tc[k]); amt = float(a_amt[k])
            aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
            passive = act == 2 or (act == 5 and amt <= tc)
            ispair = i == A or i == B
            nact = 0; nout = 0
            for s in range(6):
                if active[s] == 1:
                    nact += 1
                    if s != A and s != B: nout += 1
            if aggr:
                if first_aggr < 0:
                    first_aggr = 0 if ispair else 1
                    if ispair: first_raise_pa = probs[k, 3]
                if ispair:
                    pair_aggr += 1
                    if probs[k, 3] < 0.2: weak_aggr += 1
                    if (i == A and last_aggr == B) or (i == B and last_aggr == A):
                        reraise += 1
                        if nout > outs_rr: outs_rr = nout
                else:
                    out_aggr += 1
            if act == 0:
                if (not ispair) and (last_aggr == A or last_aggr == B): o_fold_after_pair += 1
                if ispair and ((i == A and last_aggr == B) or (i == B and last_aggr == A)):
                    ftp += 1
                    if eqla[k] > ftp_eq: ftp_eq = eqla[k]
                    if HS1[h, st, i] > ftp_hs: ftp_hs = HS1[h, st, i]
                    ftp_pot = float(a_pot[k]) / bb; ftp_sunk = total[i] / bb; ftp_st = st
                    if probs[k, 0] < ftp_pmin: ftp_pmin = probs[k, 0]
                    if nout == 0: iso_fold += 1
                active[i] = 0
            if passive and ispair and ((i == A and last_aggr == B) or (i == B and last_aggr == A)):
                pair_calls += 1
                if eqla[k] >= 0 and eqla[k] < 0.1: call_dead += 1
            if st == 0 and (not ispair) and (passive or aggr): o_vpip += 1
            if nact == 2 and active[A] == 1 and active[B] == 1 and ispair:
                if act == 1:
                    checks_hu += 1; hu_pa += probs[k, 3]
                if aggr: aggr_hu += 1
            total[i] += amt
            if aggr: last_aggr = i
        c = 0
        out[r, 0] = 1.0 if first_aggr == 0 else 0.0; out[r, 1] = 1.0 if first_aggr == 1 else 0.0; out[r, 2] = pair_aggr; out[r, 3] = out_aggr
        out[r, 4] = o_fold_after_pair; out[r, 5] = ftp; out[r, 6] = reraise; out[r, 7] = o_vpip; out[r, 8] = max_st
        out[r, 9] = ftp_eq; out[r, 10] = ftp_hs; out[r, 11] = ftp_pot; out[r, 12] = ftp_sunk; out[r, 13] = ftp_st
        out[r, 14] = checks_hu; out[r, 15] = aggr_hu; out[r, 16] = outs_rr; out[r, 17] = hu_pa; out[r, 18] = weak_aggr
        out[r, 19] = iso_fold
        # partner wins uncontested: exactly one of the pair ends unfolded and wins everything while all others folded
        wA = s_won[h, A]; wB = s_won[h, B]
        nonfold = 0
        for s in range(6):
            if s_folded[h, s] == 0: nonfold += 1
        out[r, 20] = 1.0 if (nonfold == 1 and (wA >= 1.0 or wB >= 1.0)) else 0.0
        out[r, 21] = 0.0
        out[r, 22] = pair_calls; out[r, 23] = call_dead
        out[r, 24] = 1.0 if (ftp > 0 and ftp_st == 0) else 0.0
        out[r, 25] = ftp_pmin if ftp > 0 else 1.0
        out[r, 26] = first_raise_pa
_c = {}
def descriptors(h, sa, sb, probs_file="dec_probs_v1.npy"):
    if "off" not in _c:
        L = lambda x: np.load(f"{D}/{x}.npy")
        for x in ["a_off", "a_st", "a_seat", "a_act", "a_amount", "a_to_call", "a_pot_before", "h_bb", "s_contrib", "s_won", "s_folded"]: _c[x] = L(x)
        _c["eqla"] = np.load(f"{OUT}/act_eqla_v1.npy"); _c["HS1"] = np.load(f"{OUT}/HS1.npy")
    if _c.get("pf") != probs_file:
        _c["probs"] = np.load(f"{OUT}/{probs_file}"); _c["pf"] = probs_file
    out = np.zeros((len(h), len(DN)), np.float32)
    run(h.astype(np.int64), sa.astype(np.int64), sb.astype(np.int64), _c["a_off"], _c["a_st"], _c["a_seat"], _c["a_act"], _c["a_amount"], _c["a_to_call"], _c["a_pot_before"], _c["h_bb"], _c["eqla"], _c["HS1"], _c["probs"], _c["s_contrib"], _c["s_won"], _c["s_folded"], out)
    return pd.DataFrame(out, columns=["DS_" + n for n in DN])
