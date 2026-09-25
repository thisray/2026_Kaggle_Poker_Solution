"""Generic full-population pair aggregation for an ordered-pair tensor R and seat tensor P."""
import numpy as np, pandas as pd, time, os, sys
from numba import njit, prange
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
RFILE, PFILE, NAMES, OTAG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
t0 = time.time()
R = np.load(f"{OUT}/{RFILE}", mmap_mode="r"); P = np.load(f"{OUT}/{PFILE}", mmap_mode="r")
names = open(f"{OUT}/{NAMES}").read().split("\n"); RN = names[0][2:].split(","); PN = names[1][2:].split(",")
sp = np.load(f"{D}/s_player.npy"); ht = np.load(f"{D}/h_table.npy"); hph = np.load(f"{D}/h_phase.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").sort_values("player_gi"); local = loc.local.values.astype(np.int64)
members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
pstart = np.searchsorted(ht, np.arange(401))
MAXF = set(i for i, n in enumerate(RN) if "max" in n)
maxmask = np.array([1 if i in MAXF else 0 for i in range(len(RN))], np.int64)
pmaxmask = np.array([1 if "max" in n else 0 for n in PN], np.int64)
@njit(parallel=True, cache=True)
def agg(pstart, sp, hph, local, R, P, phase, maxmask, pmaxmask):
    NRr = R.shape[3]; NPp = P.shape[2]
    S = np.zeros((400, 30, 30, NRr)); PR = np.zeros((400, 30, NRr)); PP = np.zeros((400, 30, NPp))
    for pool in prange(400):
        for h in range(pstart[pool], pstart[pool + 1]):
            if hph[h] != phase: continue
            for s in range(6):
                a = local[sp[h, s]]
                for f in range(NPp):
                    if pmaxmask[f] == 1:
                        if P[h, s, f] > PP[pool, a, f]: PP[pool, a, f] = P[h, s, f]
                    else:
                        PP[pool, a, f] += P[h, s, f]
                for t in range(6):
                    if t == s: continue
                    b = local[sp[h, t]]
                    for f in range(NRr):
                        v = R[h, s, t, f]
                        if maxmask[f] == 1:
                            if v > S[pool, a, b, f]: S[pool, a, b, f] = v
                            if v > PR[pool, a, f]: PR[pool, a, f] = v
                        else:
                            S[pool, a, b, f] += v; PR[pool, a, f] += v
    return S, PR, PP
lo_idx, hi_idx = np.triu_indices(30, 1)
pools = np.repeat(np.arange(400), len(lo_idx)); lo = np.tile(lo_idx, 400); hi = np.tile(hi_idx, 400)
for phase, pname in [(0, "dev"), (1, "eval")]:
    S, PR, PP = agg(pstart, sp, hph, local, np.asarray(R), np.asarray(P), phase, maxmask, pmaxmask)
    cols = {"pool": pools, "lo": lo, "hi": hi, "p_lo": members[pools, lo], "p_hi": members[pools, hi]}
    for k, nm in enumerate(RN):
        cols[f"{nm}__lh"] = S[pools, lo, hi, k].astype(np.float32); cols[f"{nm}__hl"] = S[pools, hi, lo, k].astype(np.float32)
        cols[f"{nm}__lo_all"] = PR[pools, lo, k].astype(np.float32); cols[f"{nm}__hi_all"] = PR[pools, hi, k].astype(np.float32)
    for k, nm in enumerate(PN):
        cols[f"P_{nm}__lo_all"] = PP[pools, lo, k].astype(np.float32); cols[f"P_{nm}__hi_all"] = PP[pools, hi, k].astype(np.float32)
    pd.DataFrame(cols).to_parquet(f"{OUT}/pairs_{pname}_{OTAG}.parquet")
    print(pname, "saved", time.time() - t0, flush=True)
