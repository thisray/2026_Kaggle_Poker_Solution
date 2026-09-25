"""Enumerate all unordered player pairs per hand (15 per hand) with pool-local pair slots."""
import numpy as np, pandas as pd
from numba import njit, prange
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
@njit(parallel=True, cache=True)
def _enum(hands, sp, ht, local, H, S, T, SLOT):
    for r in prange(len(hands)):
        h = hands[r]; k = 0
        for s in range(6):
            for t in range(s + 1, 6):
                a = local[sp[h, s]]; b = local[sp[h, t]]
                row = r * 15 + k
                H[row] = h
                if a < b:
                    S[row] = s; T[row] = t; SLOT[row] = ht[h] * 900 + a * 30 + b
                else:
                    S[row] = t; T[row] = s; SLOT[row] = ht[h] * 900 + b * 30 + a
                k += 1
def all_pair_hands(phase):
    sp = np.load(f"{D}/s_player.npy"); ht = np.load(f"{D}/h_table.npy"); hph = np.load(f"{D}/h_phase.npy")
    loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").sort_values("player_gi")
    local = loc.local.values.astype(np.int64)
    hands = np.where(hph == phase)[0].astype(np.int64)
    n = len(hands) * 15
    H = np.zeros(n, np.int64); S = np.zeros(n, np.int8); T = np.zeros(n, np.int8); SLOT = np.zeros(n, np.int64)
    _enum(hands, sp, ht.astype(np.int64), local, H, S, T, SLOT)
    return H, S, T, SLOT
def pair_slot(pool, lo, hi):
    return pool * 900 + lo * 30 + hi
