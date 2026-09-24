"""Aggregate ordered seat-pair interactions into full-population unordered player pairs per pool and phase."""
import numpy as np, pandas as pd, time, os
from numba import njit, prange
OUT = __import__("os").environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
TAG = os.environ.get("TAG", "v1")
t0 = time.time()
R = np.load(f"{OUT}/R_{TAG}.npy", mmap_mode="r"); P = np.load(f"{OUT}/P_{TAG}.npy", mmap_mode="r")
names = open(f"{OUT}/feature_names_{TAG}.txt").read().split("\n"); RN = names[0][2:].split(","); PN = names[1][2:].split(",")
sp = np.load(f"{D}/s_player.npy"); ht = np.load(f"{D}/h_table.npy"); hph = np.load(f"{D}/h_phase.npy")
NRr = R.shape[3]; NPp = P.shape[2]
# local player index within pool (0..29), players sorted by global index
nplayers = 12000
pool_of = np.full(nplayers, -1, np.int64)
for h in range(0, len(ht), 1):
    pass
pool_of[sp.reshape(-1)] = np.repeat(ht, 6)
local = np.zeros(nplayers, np.int64); members = np.zeros((400, 30), np.int64)
for pool in range(400):
    m = np.sort(np.where(pool_of == pool)[0]); assert len(m) == 30
    members[pool] = m; local[m] = np.arange(30)
pstart = np.searchsorted(ht, np.arange(401))
# extra joint features per pair-hand
JN = ["n","both_vpip","both_sd","both_end","hu_any","both_contrib_min_bb","ab_flow_max","vpip_hu_conf"]
NJ = len(JN)
@njit(parallel=True, cache=True)
def agg(pstart, sp, hph, local, R, P, NRr, NPp, NJ, phase):
    # pair sums: [400, 30, 30, NRr] ordered (i->j) ; joint [400,30,30,NJ] stored on (lo,hi) ; player sums of R over opponents [400,30,NRr]; player P sums [400,30,NPp]; player hands count
    S = np.zeros((400, 30, 30, NRr)); J = np.zeros((400, 30, 30, NJ)); PR = np.zeros((400, 30, NRr)); PP = np.zeros((400, 30, NPp)); PC = np.zeros((400, 30))
    for pool in prange(400):
        for h in range(pstart[pool], pstart[pool + 1]):
            if hph[h] != phase: continue
            for s in range(6):
                a = local[sp[h, s]]
                PC[pool, a] += 1
                for f in range(NPp): PP[pool, a, f] += P[h, s, f]
                for t in range(6):
                    if t == s: continue
                    b = local[sp[h, t]]
                    for f in range(NRr):
                        S[pool, a, b, f] += R[h, s, t, f]; PR[pool, a, f] += R[h, s, t, f]
                    if s < t:
                        lo = min(a, b); hi = max(a, b)
                        # s maps to a, t maps to b
                        J[pool, lo, hi, 0] += 1
                        bv = P[h, s, 0] * P[h, t, 0]
                        J[pool, lo, hi, 1] += bv
                        J[pool, lo, hi, 2] += P[h, s, 9] * P[h, t, 9]
                        J[pool, lo, hi, 3] += (1 - P[h, s, 5]) * (1 - P[h, t, 5])
                        hu = 1.0 if R[h, s, t, 13] > 0 else 0.0
                        J[pool, lo, hi, 4] += hu
                        J[pool, lo, hi, 5] += min(P[h, s, 7], P[h, t, 7])
                        J[pool, lo, hi, 6] += max(R[h, s, t, 7], R[h, t, s, 7])
                        J[pool, lo, hi, 7] += bv * hu
    return S, J, PR, PP, PC
for phase, pname in [(0, "dev"), (1, "eval")]:
    S, J, PR, PP, PC = agg(pstart, sp, hph, local, np.asarray(R), np.asarray(P), NRr, NPp, NJ, phase)
    print(pname, "agg", time.time() - t0, flush=True)
    rows = []
    lo_idx, hi_idx = np.triu_indices(30, 1)
    pools = np.repeat(np.arange(400), len(lo_idx)); lo = np.tile(lo_idx, 400); hi = np.tile(hi_idx, 400)
    df = pd.DataFrame({"pool": pools, "lo": lo, "hi": hi})
    df["p_lo"] = members[pools, lo]; df["p_hi"] = members[pools, hi]
    for k, nm in enumerate(JN):
        df[nm] = J[pools, lo, hi, k].astype(np.float32)
    for k, nm in enumerate(RN):
        df[f"{nm}__lh"] = S[pools, lo, hi, k].astype(np.float32)
        df[f"{nm}__hl"] = S[pools, hi, lo, k].astype(np.float32)
        # player totals against everyone (for partner-specific baselines)
        df[f"{nm}__lo_all"] = PR[pools, lo, k].astype(np.float32)
        df[f"{nm}__hi_all"] = PR[pools, hi, k].astype(np.float32)
    df["lo_hands"] = PC[pools, lo].astype(np.float32); df["hi_hands"] = PC[pools, hi].astype(np.float32)
    for k, nm in enumerate(PN):
        df[f"P_{nm}__lo_all"] = PP[pools, lo, k].astype(np.float32)
        df[f"P_{nm}__hi_all"] = PP[pools, hi, k].astype(np.float32)
    df.to_parquet(f"{OUT}/pairs_{pname}_{TAG}.parquet")
    print(pname, "saved", df.shape, time.time() - t0, flush=True)
pd.DataFrame({"player_gi": np.arange(nplayers), "pool": pool_of, "local": local}).to_parquet(f"{OUT}/player_local_{TAG}.parquet")
