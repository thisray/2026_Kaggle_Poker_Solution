"""Which planted hands does the labeller list?  Test on known families (dev, true evidence):
H-all: every hand where the collusive branch was used is listed (planted hands are a random subset of eligible hands,
       so the first actor's OWN hand-strength distribution in evidence hands = the natural distribution);
H-dev: only hands whose collusive action deviates from what the normal policy would do are listed (strong-own hands,
       which the normal policy already raises, are depleted among evidence raises).
Statistic: first preflop decision of the pair member who acts before the partner; own pf_eq bin distribution and action
distribution, evidence vs in-window non-evidence vs post-window hands."""
import numpy as np, pandas as pd
from numba import njit
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
H = M.h.values; plo = mem[M.sl.values // 900, (M.sl.values % 900) // 30]; phi = mem[M.sl.values // 900, M.sl.values % 30]
spH = np.asarray(sp[H]); S = np.argmax(spH == plo[:, None], axis=1); T = np.argmax(spH == phi[:, None], axis=1)
pf = np.asarray(Pt[H, :, PN.index("pf_eq_rand")]).astype(np.float32)
@njit(cache=True)
def first(H, S, T, off, a_seat, a_st, Y, out):
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        if ka < 0 and kb < 0: out[r, 0] = -1; continue
        if kb < 0 or (ka >= 0 and ka < kb): out[r, 0] = 0; out[r, 1] = Y[ka]
        else: out[r, 0] = 1; out[r, 1] = Y[kb]
O = np.zeros((len(M), 2), np.int64); first(H, S, T, off, a_seat, a_st, Y, O)
ix = np.arange(len(M)); who = O[:, 0]
M["own"] = np.where(who == 0, pf[ix, S], np.where(who == 1, pf[ix, T], np.nan))
M["par"] = np.where(who == 0, pf[ix, T], np.where(who == 1, pf[ix, S], np.nan))
M["y"] = O[:, 1]; M = M[who >= 0].copy()
M["act"] = np.select([M.y == 0, M.y == 3], ["fold", "raise"], "call")
M["ob"] = pd.cut(M.own, [0, .35, .45, .55, .65, 1.01], labels=["<.35", ".35-.45", ".45-.55", ".55-.65", ">.65"])
pd.set_option("display.width", 220)
for fam, F in M.groupby("fam"):
    print(f"== {fam}")
    t = F.groupby("zone").ob.value_counts(normalize=True).unstack().round(3); t["n"] = F.groupby("zone").size(); print("own-strength distribution of first actor:\n", t)
    for z in ["ev", "in_non"]:
        G = F[F.zone == z]; tt = pd.crosstab(G.ob, G.act, normalize="index").round(3); tt["n"] = G.ob.value_counts().sort_index()
        print(f"  action | own bin  ({z}):\n", tt)
    E = F[F.zone == "ev"]
    print("  evidence hands: share with first actor strong (>.65) and raising:", round(((E.own > .65) & (E.act == "raise")).mean(), 3),
          " | natural share strong among first actors (in_non):", round((F[F.zone == 'in_non'].own > .65).mean(), 3))
