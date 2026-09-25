"""CI completion rule: among in-window CI hands where the first-acting member raised preflop with a weak own hand (these are
mostly planted), what separates LISTED from UNLISTED?  Fine-grained sequence features after the trigger raise."""
import numpy as np, pandas as pd
from numba import njit
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
a_amt = np.load(f"{D}/a_amount.npy"); a_to = np.load(f"{D}/a_amount_to.npy"); a_pa = np.load(f"{D}/a_players_active.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); swon = np.load(f"{D}/s_won.npy", mmap_mode="r"); ssd = np.load(f"{D}/s_sd.npy", mmap_mode="r")
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
H = M.h.values; plo = mem[M.sl.values // 900, (M.sl.values % 900) // 30]; phi = mem[M.sl.values // 900, M.sl.values % 30]
spH = np.asarray(sp[H]); S = np.argmax(spH == plo[:, None], axis=1); T = np.argmax(spH == phi[:, None], axis=1)
pf = np.asarray(Pt[H, :, PN.index("pf_eq_rand")]).astype(np.float32); ls = np.asarray(Pt[H, :, PN.index("last_street")]).astype(np.float32)
@njit(cache=True)
def seq(H, S, T, off, a_seat, a_st, Y, a_pa, dealt, out):
    # 0 who first (0=S,1=T,-1 none) | 1 first actor y | 2 partner first y | 3 outsiders folding after trigger (preflop) |
    # 4 outsiders still in when trigger happened (acted-or-pending) | 5 outsiders calling/raising after trigger preflop |
    # 6 hand ended preflop (1/0) | 7 n outsiders alive at end of preflop | 8 partner acted before trigger (1/0) |
    # 9 n outsider voluntary actions (call/raise) before the trigger | 10 players_active at trigger
    for r in range(len(H)):
        h = H[r]; f = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] or a_seat[k] == T[r]:
                f = k; break
        if f < 0: out[r, 0] = -1; continue
        me = a_seat[f]; pa = T[r] if me == S[r] else S[r]
        out[r, 0] = 0 if me == S[r] else 1; out[r, 1] = Y[f]; out[r, 10] = a_pa[f]
        alive = dealt[r].copy(); nvol = 0
        for k in range(off[h], f):
            if Y[k] == 0: alive[a_seat[k]] = 0
            elif a_seat[k] != S[r] and a_seat[k] != T[r] and Y[k] >= 2: nvol += 1
        out[r, 9] = nvol
        n_in = 0
        for s in range(6):
            if alive[s] == 1 and s != S[r] and s != T[r]: n_in += 1
        out[r, 4] = n_in
        pfirst = -1; nf = 0; nc = 0; last_st = 0
        for k in range(f + 1, off[h + 1]):
            last_st = a_st[k]
            if a_st[k] != 0: continue
            s = a_seat[k]
            if s == pa and pfirst < 0: pfirst = Y[k]
            if s != S[r] and s != T[r]:
                if Y[k] == 0: nf += 1; alive[s] = 0
                elif Y[k] >= 2: nc += 1
        out[r, 2] = pfirst; out[r, 3] = nf; out[r, 5] = nc; out[r, 6] = 1 if last_st == 0 else 0
        na = 0
        for s in range(6):
            if alive[s] == 1 and s != S[r] and s != T[r]: na += 1
        out[r, 7] = na
dealt = (spH >= 0).astype(np.int64)
O = np.zeros((len(M), 11), np.int64); seq(H, S, T, off, a_seat, a_st, Y, a_pa, dealt, O)
ix = np.arange(len(M)); who = O[:, 0]
M["own"] = np.where(who == 0, pf[ix, S], pf[ix, T]); M["par"] = np.where(who == 0, pf[ix, T], pf[ix, S])
cols = ["who", "y1", "y2", "o_fold_after", "o_in_at_trig", "o_callraise_after", "end_pre", "o_alive_endpre", "_x", "o_vol_before", "pa_at_trig"]
for j, c in enumerate(cols): M[c] = O[:, j]
wonH = np.asarray(swon[H]); sdH = np.asarray(ssd[H])
out_won = np.zeros(len(M)); out_sd = np.zeros(len(M))
for j in range(6):
    m = (j != S) & (j != T) & (spH[:, j] >= 0)
    out_won += np.where(m, wonH[:, j] > 0, 0); out_sd += np.where(m, sdH[:, j] > 0, 0)
M["out_won"] = out_won > 0; M["out_sd"] = out_sd > 0
M["all_out_folded"] = (~M.out_won) & (~M.out_sd)
M["both_flop"] = np.minimum(ls[ix, S], ls[ix, T]) >= 1
pd.set_option("display.width", 220)
for fam in ["coordinated_isolation", "directed_transfer", "soft_play"]:
    F = M[(M.fam == fam) & (M.zone != "post") & (M.who >= 0)]
    A = F[(F.y1 == 3) & (F.own < 0.5)]
    print(f"== {fam}: weak first raise in window: listed {int(A.ev.sum())} unlisted {int((~A.ev).sum())}")
    feats = ["o_in_at_trig", "o_fold_after", "o_callraise_after", "end_pre", "o_alive_endpre", "o_vol_before", "pa_at_trig", "out_won", "out_sd", "all_out_folded", "pair_win", "both_flop", "xfer", "par"]
    print(A.groupby("ev")[feats].mean().round(3).to_string())
    # crisp candidate rules
    for nm, cond in [("all outsiders in at trigger folded preflop", A.o_alive_endpre == 0),
                     ("no outsider called/raised after trigger", A.o_callraise_after == 0),
                     ("all outsiders folded (no outsider won/sd)", A.all_out_folded),
                     ("o_fold_after >= o_in_at_trig", A.o_fold_after >= A.o_in_at_trig),
                     ("o_fold_after >= 1", A.o_fold_after >= 1), ("o_fold_after >= 2", A.o_fold_after >= 2),
                     ("o_fold_after >= 3", A.o_fold_after >= 3)]:
        print(f"   {nm:45s} recall(listed) {cond[A.ev].mean():.3f}  pass(unlisted) {cond[~A.ev].mean():.3f}")
    print("   o_fold_after x listed:", pd.crosstab(A.o_fold_after, A.ev).to_dict())
    print("   o_in_at_trig x listed:", pd.crosstab(A.o_in_at_trig, A.ev).to_dict())
M.to_parquet(f"{OUT}/t5_dev_seq.parquet")
