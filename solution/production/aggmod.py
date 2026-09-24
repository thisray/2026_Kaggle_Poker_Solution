"""Masked full-population pair aggregation of ordered-pair tensors (sum or max per feature)."""
import numpy as np, pandas as pd
from numba import njit, prange
OUT = __import__("os").environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
@njit(parallel=True, cache=True)
def _agg(pstart, sp, mask, local, R, P, maxmask, pmaxmask):
    NRr = R.shape[3]; NPp = P.shape[2]
    S = np.zeros((400, 30, 30, NRr), np.float32); PR = np.zeros((400, 30, NRr), np.float32); PP = np.zeros((400, 30, NPp), np.float32)
    N = np.zeros((400, 30, 30), np.float32); PC = np.zeros((400, 30), np.float32)
    for pool in prange(400):
        for h in range(pstart[pool], pstart[pool + 1]):
            if mask[h] == 0: continue
            for s in range(6):
                a = local[sp[h, s]]
                PC[pool, a] += 1
                for f in range(NPp):
                    if pmaxmask[f] == 1:
                        if P[h, s, f] > PP[pool, a, f]: PP[pool, a, f] = P[h, s, f]
                    else:
                        PP[pool, a, f] += P[h, s, f]
                for t in range(6):
                    if t == s: continue
                    b = local[sp[h, t]]
                    N[pool, a, b] += 1
                    for f in range(NRr):
                        v = R[h, s, t, f]
                        if maxmask[f] == 1:
                            if v > S[pool, a, b, f]: S[pool, a, b, f] = v
                            if v > PR[pool, a, f]: PR[pool, a, f] = v
                        else:
                            S[pool, a, b, f] += v; PR[pool, a, f] += v
    return S, PR, PP, N, PC
_c = {}
def _load():
    if not _c:
        _c["sp"] = np.load(f"{D}/s_player.npy"); ht = np.load(f"{D}/h_table.npy"); _c["ht"] = ht
        _c["pstart"] = np.searchsorted(ht, np.arange(401)); _c["phase"] = np.load(f"{D}/h_phase.npy")
        loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").sort_values("player_gi")
        _c["local"] = loc.local.values.astype(np.int64)
        members = np.zeros((400, 30), np.int64)
        for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
        _c["members"] = members
    return _c
def hand_mask(kind, seed=0, frac=2/3):
    c = _load(); ph = c["phase"]
    if kind == "dev": return (ph == 0).astype(np.int8)
    if kind == "eval": return (ph == 1).astype(np.int8)
    if kind == "devsub":
        rng = np.random.RandomState(seed); m = (ph == 0) & (rng.rand(len(ph)) < frac)
        return m.astype(np.int8)
    raise ValueError(kind)
def aggregate(R, P, rnames, pnames, mask, prefix):
    c = _load()
    maxmask = np.array([1 if "max" in n else 0 for n in rnames], np.int64); pmaxmask = np.array([1 if "max" in n else 0 for n in pnames], np.int64)
    S, PR, PP, N, PC = _agg(c["pstart"], c["sp"], mask, c["local"], R, P, maxmask, pmaxmask)
    lo_idx, hi_idx = np.triu_indices(30, 1)
    pools = np.repeat(np.arange(400), len(lo_idx)); lo = np.tile(lo_idx, 400); hi = np.tile(hi_idx, 400)
    cols = {}
    if prefix == "":
        cols = {"pool": pools, "lo": lo, "hi": hi, "p_lo": c["members"][pools, lo], "p_hi": c["members"][pools, hi],
                "n": N[pools, lo, hi], "lo_hands": PC[pools, lo], "hi_hands": PC[pools, hi]}
    for k, nm in enumerate(rnames):
        cols[f"{prefix}{nm}__lh"] = S[pools, lo, hi, k]; cols[f"{prefix}{nm}__hl"] = S[pools, hi, lo, k]
        cols[f"{prefix}{nm}__lo_all"] = PR[pools, lo, k]; cols[f"{prefix}{nm}__hi_all"] = PR[pools, hi, k]
    for k, nm in enumerate(pnames):
        cols[f"{prefix}P_{nm}__lo_all"] = PP[pools, lo, k]; cols[f"{prefix}P_{nm}__hi_all"] = PP[pools, hi, k]
    return pd.DataFrame(cols)
