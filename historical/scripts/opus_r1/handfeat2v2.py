"""Hand-level pair features from surprisal tensors (R2/P2) plus v1 (handfeat)."""
import numpy as np, pandas as pd
from numba import njit, prange
import handfeat as HF
OUT = HF.OUT
TAG2 = "v2"
n2 = open(f"{OUT}/feature_names2_{TAG2}.txt").read().split("\n"); RN2 = n2[0][2:].split(","); PN2 = n2[1][2:].split(",")
@njit(parallel=True, cache=True)
def gather2(hs, ss, ts, R2, P2, out):
    nr = R2.shape[3]; npf = P2.shape[2]
    for r in prange(len(hs)):
        h = hs[r]; s = ss[r]; t = ts[r]; c = 0
        for f in range(nr):
            a = R2[h, s, t, f]; b = R2[h, t, s, f]
            out[r, c] = max(a, b); out[r, c + 1] = min(a, b); c += 2
        for f in range(npf):
            a = P2[h, s, f]; b = P2[h, t, f]
            out[r, c] = max(a, b); out[r, c + 1] = min(a, b); c += 2
        # outsider surprisal baseline in the same hand
        osum = 0.0; omax = 0.0
        for o in range(6):
            if o == s or o == t: continue
            osum += P2[h, o, 0]
            if P2[h, o, 1] > omax: omax = P2[h, o, 1]
        out[r, c] = osum; out[r, c + 1] = omax
        # pair-involved vs total surprisal share
        tot = P2[h, s, 0] + P2[h, t, 0]
        inv = R2[h, s, t, 0] + R2[h, t, s, 0]
        out[r, c + 2] = inv / (tot + 1e-3)
_c = {}
def names2():
    n = []
    for f in RN2: n += [f"S_{f}_mx", f"S_{f}_mn"]
    for f in PN2: n += [f"S_P_{f}_mx", f"S_P_{f}_mn"]
    return n + ["S_o_sur_sum", "S_o_sur_max", "S_inv_share"]
def features(h, s, t):
    if not _c:
        _c["R2"] = np.load(f"{OUT}/R2_{TAG2}.npy", mmap_mode="r"); _c["P2"] = np.load(f"{OUT}/P2_{TAG2}.npy", mmap_mode="r")
    X1 = HF.features(h, s, t)
    nm = names2(); out = np.zeros((len(h), len(nm)), np.float32)
    gather2(h.astype(np.int64), s.astype(np.int64), t.astype(np.int64), _c["R2"], _c["P2"], out)
    return pd.concat([X1, pd.DataFrame(out, columns=nm)], axis=1)
