"""Pair-level role estimation from stage-1 hand scores and oriented hand features (donor->receiver, squeezer->partner)."""
import numpy as np, pandas as pd
from numba import njit, prange
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
ON = ["o_flow_dr", "o_flow_rd", "o_lost_dr", "o_d_fold_to_r", "o_r_fold_to_d", "o_d_call_to_r", "o_d_raise_over_r", "o_r_raise_over_d", "o_d_eq_fold", "o_d_eq_call",
      "o_d_sunk_fold", "o_d_sd", "o_d_folded", "o_d_eqlast", "o_r_eqlast", "o_d_net", "o_r_net", "o_hu_streets", "o_d_hs_pf", "o_r_hs_pf",
      "o_sq_s", "o_sq_p", "o_p_fold_to_s", "o_s_fold_to_p", "o_s_iso", "o_dir_conf", "o_sq_conf", "o_dir_agree", "o_sq_agree"]
@njit(parallel=True, cache=True)
def oriented(hs, sa, sb, donorA, squeezA, dconf, sconf, R, P, out):
    for r in prange(len(hs)):
        h = hs[r]; A = sa[r]; B = sb[r]
        d = A if donorA[r] else B; rc = B if donorA[r] else A
        s = A if squeezA[r] else B; p = B if squeezA[r] else A
        out[r, 0] = R[h, d, rc, 7]; out[r, 1] = R[h, rc, d, 7]
        dn = -P[h, d, 8]; rn = P[h, rc, 8]
        out[r, 2] = min(max(dn, 0.0), max(rn, 0.0))
        out[r, 3] = R[h, d, rc, 1]; out[r, 4] = R[h, rc, d, 1]; out[r, 5] = R[h, d, rc, 2]; out[r, 6] = R[h, d, rc, 3]; out[r, 7] = R[h, rc, d, 3]
        out[r, 8] = R[h, d, rc, 15]; out[r, 9] = R[h, d, rc, 16]; out[r, 10] = R[h, d, rc, 5]
        out[r, 11] = P[h, d, 9]; out[r, 12] = P[h, d, 5]; out[r, 13] = P[h, d, 16]; out[r, 14] = P[h, rc, 16]
        out[r, 15] = P[h, d, 8]; out[r, 16] = P[h, rc, 8]; out[r, 17] = R[h, d, rc, 13]; out[r, 18] = P[h, d, 12]; out[r, 19] = P[h, rc, 12]
        out[r, 20] = R[h, s, p, 11]; out[r, 21] = R[h, p, s, 11]; out[r, 22] = R[h, p, s, 1]; out[r, 23] = R[h, s, p, 1]; out[r, 24] = R[h, s, p, 12]
        out[r, 25] = dconf[r]; out[r, 26] = sconf[r]
        fl = R[h, A, B, 7] - R[h, B, A, 7]
        out[r, 27] = (1.0 if fl > 0 else (-1.0 if fl < 0 else 0.0)) * (1.0 if donorA[r] else -1.0)
        sq = R[h, A, B, 11] - R[h, B, A, 11]
        out[r, 28] = (1.0 if sq > 0 else (-1.0 if sq < 0 else 0.0)) * (1.0 if squeezA[r] else -1.0)
_c = {}
def features(sl, h, sa, sb, s1):
    """sl: pair slot per row, h/sa/sb: hand and seats, s1: stage-1 score per row. Pair roles estimated within the given rows grouped by slot."""
    if "R" not in _c:
        _c["R"] = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); _c["P"] = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
    R = _c["R"]; P = _c["P"]
    fAB = np.asarray(R[h, sa, sb, 7]); fBA = np.asarray(R[h, sb, sa, 7]); oAB = np.asarray(R[h, sa, sb, 1]); oBA = np.asarray(R[h, sb, sa, 1])
    qAB = np.asarray(R[h, sa, sb, 11]); qBA = np.asarray(R[h, sb, sa, 11])
    w = s1.astype(np.float64) ** 2
    df = pd.DataFrame({"sl": sl, "dflow": ((fAB - fBA) / (np.abs(fAB - fBA) + 1.0) + (oAB - oBA)) * w, "dsq": (qAB - qBA) * w, "w": w})
    g = df.groupby("sl")[["dflow", "dsq", "w"]].sum()
    dscore = pd.Series(sl).map(g.dflow).values; sscore = pd.Series(sl).map(g.dsq).values; wsum = pd.Series(sl).map(g.w).values
    donorA = dscore > 0; squeezA = sscore > 0
    dconf = np.abs(dscore) / (wsum + 1e-3); sconf = np.abs(sscore) / (wsum + 1e-3)
    out = np.zeros((len(h), len(ON)), np.float32)
    oriented(h.astype(np.int64), sa.astype(np.int64), sb.astype(np.int64), donorA, squeezA, dconf.astype(np.float64), sconf.astype(np.float64), R, P, out)
    return pd.DataFrame(out, columns=ON)
