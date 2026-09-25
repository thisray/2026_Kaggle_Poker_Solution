"""Hand-level pair features v3 = v2 (interaction + surprisal v1) + probability-weighted suppressed actions (R4/P4)."""
import numpy as np, pandas as pd
from numba import njit, prange
import handfeat2 as HF2
OUT = HF2.OUT
TAG = "v2"
n4 = open(f"{OUT}/feature_names4_{TAG}.txt").read().split("\n"); RN4 = n4[0][2:].split(","); PN4 = n4[1][2:].split(",")
@njit(parallel=True, cache=True)
def gather4(hs, ss, ts, R4, P4, out):
    nr = R4.shape[3]; npf = P4.shape[2]
    for r in prange(len(hs)):
        h = hs[r]; s = ss[r]; t = ts[r]; c = 0
        for f in range(nr):
            a = R4[h, s, t, f]; b = R4[h, t, s, f]
            out[r, c] = max(a, b); out[r, c + 1] = min(a, b); c += 2
        for f in range(npf):
            a = P4[h, s, f]; b = P4[h, t, f]
            out[r, c] = max(a, b); out[r, c + 1] = min(a, b); c += 2
        # partner-directed share of suppressed/forced mass
        tot_s = P4[h, s, 3] + P4[h, t, 3]; inv_s = R4[h, s, t, 0] + R4[h, t, s, 0]
        out[r, c] = inv_s / (tot_s + 1e-3)
        tot_f = P4[h, s, 4] + P4[h, t, 4]; inv_f = R4[h, s, t, 4] + R4[h, t, s, 4]
        out[r, c + 1] = inv_f / (tot_f + 1e-3)
_c = {}
def names4():
    n = []
    for f in RN4: n += [f"Q_{f}_mx", f"Q_{f}_mn"]
    for f in PN4: n += [f"Q_P_{f}_mx", f"Q_P_{f}_mn"]
    return n + ["Q_supp_share", "Q_forced_fold_share"]
def features(h, s, t):
    if not _c:
        _c["R4"] = np.load(f"{OUT}/R4_{TAG}.npy", mmap_mode="r"); _c["P4"] = np.load(f"{OUT}/P4_{TAG}.npy", mmap_mode="r")
    X = HF2.features(h, s, t)
    nm = names4(); out = np.zeros((len(h), len(nm)), np.float32)
    gather4(h.astype(np.int64), s.astype(np.int64), t.astype(np.int64), _c["R4"], _c["P4"], out)
    return pd.concat([X, pd.DataFrame(out, columns=nm)], axis=1)
