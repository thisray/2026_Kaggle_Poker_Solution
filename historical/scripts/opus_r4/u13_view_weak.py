"""R4-U13: view every evidence hand (+ in-window both-voluntary hands) of the weakest-ranked dev positives."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
keys = [int(k) for k in sys.argv[1].split(",")]
d = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet").sort_values(["sl", "ts", "h"]).reset_index(drop=True); d["k"] = d.groupby("sl").cumcount()
t58 = pd.read_parquet(f"{OUT}/r3/t58_seq_feats.parquet")[["slot", "pa", "pb"]].drop_duplicates(); t58["key"] = t58.pa * 12000 + t58.pb
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); a_act = np.load(f"{D}/a_act.npy", mmap_mode="r")
a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_st = np.load(f"{D}/a_st.npy", mmap_mode="r"); c1 = np.load(f"{D}/s_c1.npy", mmap_mode="r"); c2 = np.load(f"{D}/s_c2.npy", mmap_mode="r")
net = np.load(f"{D}/s_net.npy", mmap_mode="r"); bb = np.load(f"{D}/h_bb.npy", mmap_mode="r"); board = np.load(f"{D}/h_board.npy", mmap_mode="r"); con = np.load(f"{D}/s_contrib.npy", mmap_mode="r")
R_ = "23456789TJQKA"; SU = "cdhs"; AC = "fxcbrA"
def card(c): c = int(c); return R_[c // 4] + SU[c % 4] if 0 <= c < 52 else "--"
for key in keys:
    r0 = t58[t58.key == key].iloc[0]; sl = int(r0.slot); pa, pb = int(r0.pa), int(r0.pb); g = d[d.sl == sl]
    print("#" * 30, "key", key, "slot", sl, g.fam.iloc[0], "n co-seated", len(g), "ev idx", g[g.ev].k.tolist())
    for _, r in g.iterrows():
        h = int(r.h); seats = np.asarray(sp[h]); sa = int(np.flatnonzero(seats == pa)[0]); sb = int(np.flatnonzero(seats == pb)[0])
        vol = con[h, sa] > bb[h] and con[h, sb] > bb[h]
        if not (r.ev or (vol and r.zone != "post")): continue
        ks = np.arange(off[h], off[h + 1]); out = []; cur = -1
        for k in ks:
            st = int(a_st[k])
            if st != cur: out.append("|" + "PFTR"[st] + ":"); cur = st
            s = int(a_seat[k]); who = "A" if s == sa else ("B" if s == sb else "o")
            out.append(f"{who}{AC[int(a_act[k])]}" + (f"{a_amt[k] / bb[h]:.0f}" if a_act[k] >= 2 else ""))
        bd = "".join(card(c) for c in np.asarray(board[h]).ravel()[:5])
        print(f"{'EV ' if r.ev else '   '}k={int(r.k):3d} A[{card(c1[h,sa])}{card(c2[h,sa])}] B[{card(c1[h,sb])}{card(c2[h,sb])}] bd[{bd}] netA={net[h,sa]/bb[h]:+.0f} netB={net[h,sb]/bb[h]:+.0f} " + " ".join(out))
