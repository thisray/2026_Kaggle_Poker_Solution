"""handfeat2 + chip-dumping/cooler features (R5), max/min over directions plus the dump-vs-cooler contrast."""
import numpy as np, pandas as pd
from numba import njit, prange
import handfeat2 as HF2
OUT = HF2.OUT
RN5 = open(f"{OUT}/feature_names5_v1.txt").read().split("\n")[0][2:].split(",")
@njit(parallel=True, cache=True)
def gather5(hs, ss, ts, R5, out):
    nr = R5.shape[3]
    for r in prange(len(hs)):
        h = hs[r]; s = ss[r]; t = ts[r]; c = 0
        for f in range(nr):
            a = R5[h, s, t, f]; b = R5[h, t, s, f]
            out[r, c] = max(a, b); out[r, c + 1] = min(a, b); c += 2
        # direction-consistent contrast: weak investing of the bigger investor minus strong investing
        ws = R5[h, s, t, 0] - R5[h, s, t, 3]; wt = R5[h, t, s, 0] - R5[h, t, s, 3]
        out[r, c] = max(ws, wt); out[r, c + 1] = min(ws, wt)
_c = {}
def names():
    n = []
    for f in RN5: n += [f"W_{f}_mx", f"W_{f}_mn"]
    return n + ["W_weak_minus_strong_mx", "W_weak_minus_strong_mn"]
def features(h, s, t):
    if not _c: _c["R5"] = np.load(f"{OUT}/R5_v1.npy", mmap_mode="r")
    X = HF2.features(h, s, t)
    nm = names(); out = np.zeros((len(h), len(nm)), np.float32)
    gather5(h.astype(np.int64), s.astype(np.int64), t.astype(np.int64), _c["R5"], out)
    return pd.concat([X, pd.DataFrame(out, columns=nm)], axis=1)
