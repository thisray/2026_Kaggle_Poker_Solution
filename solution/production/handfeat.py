import numpy as np, pandas as pd
from numba import njit, prange
OUT = __import__("os").environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
RN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[0][2:].split(",")
PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
@njit(parallel=True, cache=True)
def gather(hs, ss, ts, R, P, h_pot, h_bb, h_nsd, h_board, out):
    nr = R.shape[3]; npf = P.shape[2]
    for r in prange(len(hs)):
        h = hs[r]; s = ss[r]; t = ts[r]; c = 0
        for f in range(nr):
            a = R[h, s, t, f]; b = R[h, t, s, f]
            out[r, c] = max(a, b); out[r, c + 1] = min(a, b); c += 2
        for f in range(npf):
            a = P[h, s, f]; b = P[h, t, f]
            out[r, c] = max(a, b); out[r, c + 1] = min(a, b); c += 2
        # outsider aggregates
        o_vpip = 0.0; o_fold_to_ab = 0.0; ab_fold_to_o = 0.0; o_aggr = 0.0; o_contrib = 0.0; o_net = 0.0; o_sd = 0.0; o_flow_to_ab = 0.0; ab_flow_to_o = 0.0; o_iso = 0.0; o_eq_fold = 0.0
        for o in range(6):
            if o == s or o == t: continue
            o_vpip += P[h, o, 0]; o_aggr += P[h, o, 2]; o_contrib += P[h, o, 7]; o_net += P[h, o, 8]; o_sd += P[h, o, 9]
            o_fold_to_ab += R[h, o, s, 1] + R[h, o, t, 1]
            ab_fold_to_o += R[h, s, o, 1] + R[h, t, o, 1]
            o_flow_to_ab += R[h, o, s, 7] + R[h, o, t, 7]
            ab_flow_to_o += R[h, s, o, 7] + R[h, t, o, 7]
            o_eq_fold += R[h, o, s, 15] + R[h, o, t, 15]
        out[r, c] = o_vpip; out[r, c + 1] = o_aggr; out[r, c + 2] = o_contrib; out[r, c + 3] = o_net; out[r, c + 4] = o_sd
        out[r, c + 5] = o_fold_to_ab; out[r, c + 6] = ab_fold_to_o; out[r, c + 7] = o_flow_to_ab; out[r, c + 8] = ab_flow_to_o
        out[r, c + 9] = R[h, s, t, 12]; out[r, c + 10] = o_eq_fold
        out[r, c + 11] = h_pot[h] / h_bb[h]; out[r, c + 12] = h_bb[h]; out[r, c + 13] = h_nsd[h]
        nb = 0
        for b in range(5):
            if h_board[h, b] >= 0: nb += 1
        out[r, c + 14] = nb
        # net flow between the two (signed magnitude) and pair net
        out[r, c + 15] = abs(P[h, s, 8] - P[h, t, 8]); out[r, c + 16] = P[h, s, 8] + P[h, t, 8]
def names():
    n = []
    for f in RN: n += [f"{f}_mx", f"{f}_mn"]
    for f in PN: n += [f"P_{f}_mx", f"P_{f}_mn"]
    n += ["o_vpip","o_aggr","o_contrib","o_net","o_sd","o_fold_to_ab","ab_fold_to_o","o_flow_to_ab","ab_flow_to_o","iso_pair","o_eq_fold","pot_bb","bb","nsd","nboard","net_gap","pair_net"]
    return n
_cache = {}
def load():
    if not _cache:
        _cache["R"] = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); _cache["P"] = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
        for k in ["h_pot","h_bb","h_nsd","h_board","s_player","h_ts","h_table","h_phase"]:
            _cache[k] = np.load(f"{D}/{k}.npy")
    return _cache
def pair_hands(p_lo, p_hi, phase=None):
    """return arrays (h, seat_lo, seat_hi, pair_row) for all hands where both players sat"""
    c = load(); sp = c["s_player"]
    # index player -> list of (h, seat)
    if "pl_h" not in c:
        flat = sp.reshape(-1); order = np.argsort(flat, kind="stable")
        c["pl_order"] = order; c["pl_start"] = np.searchsorted(flat[order], np.arange(12001))
    order = c["pl_order"]; start = c["pl_start"]
    H = []; S = []; T = []; RW = []
    for r, (a, b) in enumerate(zip(p_lo, p_hi)):
        ia = order[start[a]:start[a+1]]; ib = order[start[b]:start[b+1]]
        ha = ia // 6; hb = ib // 6
        common, xa, xb = np.intersect1d(ha, hb, assume_unique=True, return_indices=True)
        if phase is not None:
            keep = c["h_phase"][common] == phase; common = common[keep]; xa = xa[keep]; xb = xb[keep]
        H.append(common); S.append(ia[xa] % 6); T.append(ib[xb] % 6); RW.append(np.full(len(common), r))
    return np.concatenate(H), np.concatenate(S), np.concatenate(T), np.concatenate(RW)
def features(h, s, t):
    c = load(); nm = names()
    out = np.zeros((len(h), len(nm)), np.float32)
    gather(h.astype(np.int64), s.astype(np.int64), t.astype(np.int64), c["R"], c["P"], c["h_pot"], c["h_bb"], c["h_nsd"], c["h_board"], out)
    return pd.DataFrame(out, columns=nm)
