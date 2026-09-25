"""Activation vs evidence (known families, dev).  CI members raise junk ~10% of the time in NON-evidence hands (controls
~0.5%) -> collusive activations that are not listed.  What distinguishes listed activations?  For hands in which the first
acting member raised preflop with a weak own hand (own pf_eq<0.5): partner's first preflop action, whether an outsider
folded to the pair's aggression (iso_ofold), whether both stayed to the flop, pair outcome.  Evidence vs non-evidence."""
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
ls = np.asarray(Pt[H, :, PN.index("last_street")]).astype(np.float32)
@njit(cache=True)
def seq(H, S, T, off, a_seat, a_st, Y, out):
    # out: who first (0=S,1=T), first actor's y, partner's first preflop y (-1 none), n outsider preflop folds after first raise,
    #      partner acted after first actor's action (1/0), max street reached by both (min of last streets computed outside)
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        if ka < 0 and kb < 0: out[r, 0] = -1; continue
        if kb < 0 or (ka >= 0 and ka < kb): f = ka; o = kb; out[r, 0] = 0
        else: f = kb; o = ka; out[r, 0] = 1
        out[r, 1] = Y[f]; out[r, 2] = Y[o] if o >= 0 else -1
        nf = 0
        for k in range(f + 1, off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] != S[r] and a_seat[k] != T[r] and Y[k] == 0: nf += 1
        out[r, 3] = nf
O = np.zeros((len(M), 4), np.int64); seq(H, S, T, off, a_seat, a_st, Y, O)
ix = np.arange(len(M)); who = O[:, 0]
M["own"] = np.where(who == 0, pf[ix, S], pf[ix, T]); M["par"] = np.where(who == 0, pf[ix, T], pf[ix, S])
M["y1"] = O[:, 1]; M["y2"] = O[:, 2]; M["out_folds"] = O[:, 3]; M["who"] = who
M["both_flop"] = (np.minimum(ls[ix, S], ls[ix, T]) >= 1)
M = M[who >= 0].copy()
lab = {-1: "none", 0: "fold", 1: "check", 2: "call", 3: "raise"}
M["p_act"] = M.y2.map(lab)
pd.set_option("display.width", 220)
for fam, F in M.groupby("fam"):
    A = F[(F.y1 == 3) & (F.own < 0.5)]   # activation-like: first actor raised a weak hand
    print(f"== {fam}: first actor raised with own<0.5 : ev {int((A.zone=='ev').sum())}, in-window non-ev {int((A.zone=='in_non').sum())}, post {int((A.zone=='post').sum())}")
    for z in ["ev", "in_non", "post"]:
        G = A[A.zone == z]
        if len(G) == 0: continue
        print(f"  [{z:6s}] partner action {G.p_act.value_counts(normalize=True).round(2).to_dict()} | outsider folds mean {G.out_folds.mean():.2f} | both reach flop {G.both_flop.mean():.2f}"
              f" | pair win {G.pair_win.mean():.2f} | iso>0 {(G.iso > 0).mean():.2f} | partner pf_eq mean {G.par.mean():.3f}")
    Wn = F[F.zone != "post"]; Aw = Wn[(Wn.y1 == 3) & (Wn.own < 0.5)]
    print(f"  P(ev | weak raise, in window) = {Aw.ev.mean():.3f};  P(ev | weak raise & partner raise/call) = {Aw[Aw.y2.isin([2,3])].ev.mean():.3f}; P(ev | weak raise & partner fold) = {Aw[Aw.y2 == 0].ev.mean():.3f}")
