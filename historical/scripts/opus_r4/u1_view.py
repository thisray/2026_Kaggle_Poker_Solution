"""R4-U1: compact one-line-per-hand viewer of a positive pair's co-seated hands (dev)."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
fam = sys.argv[1]; npairs = int(sys.argv[2]); seed = int(sys.argv[3]); mode = sys.argv[4] if len(sys.argv) > 4 else "cell"
d = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet").sort_values(["sl", "ts", "h"]).reset_index(drop=True)
d["k"] = d.groupby("sl").cumcount()
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_st = np.load(f"{D}/a_st.npy", mmap_mode="r")
c1 = np.load(f"{D}/s_c1.npy", mmap_mode="r"); c2 = np.load(f"{D}/s_c2.npy", mmap_mode="r"); net = np.load(f"{D}/s_net.npy", mmap_mode="r"); btn = np.load(f"{D}/h_btn.npy", mmap_mode="r")
bb = np.load(f"{D}/h_bb.npy", mmap_mode="r")
t58 = pd.read_parquet(f"{OUT}/r3/t58_seq_feats.parquet").set_index(["slot", "h"])
R = "23456789TJQKA"; SU = "cdhs"
def card(c): c = int(c); return R[c // 4] + SU[c % 4] if 0 <= c < 52 else "??"
AC = "fxcbrA"
sls = d[d.fam == fam].sl.drop_duplicates().sample(npairs, random_state=seed).values
for sl in sls:
    g = d[d.sl == sl]; pa, pb = t58.loc[sl].iloc[0][["pa", "pb"]].astype(int)
    print("#" * 60, "pair slot", sl, fam, "n", len(g), "ev idx", g[g.ev].k.tolist())
    for _, r in g.iterrows():
        if mode == "cell" and not (r.ev or (r.both_flop and r.zone != "post")): continue
        if mode == "all" and r.k > g[g.ev].k.max() + 3: continue
        h = int(r.h); seats = np.asarray(sp[h]); sa = int(np.flatnonzero(seats == pa)[0]); sb_ = int(np.flatnonzero(seats == pb)[0])
        ks = np.arange(off[h], off[h + 1]); out = []; cur = -1
        for k in ks:
            st = int(a_st[k])
            if st != cur: out.append("|" + "PFTR"[st] + ":"); cur = st
            s = int(a_seat[k]); who = "A" if s == sa else ("B" if s == sb_ else "o")
            amt = a_amt[k] / max(bb[h], 1)
            out.append(f"{who}{AC[int(a_act[k])]}" + (f"{amt:.0f}" if a_act[k] >= 2 else ""))
        print(f"{'EV ' if r.ev else '   '}k={r.k:3d} btn={int(btn[h])} A@{sa}[{card(c1[h,sa])}{card(c2[h,sa])}] B@{sb_}[{card(c1[h,sb_])}{card(c2[h,sb_])}] netA={net[h,sa]/bb[h]:+.0f} netB={net[h,sb_]/bb[h]:+.0f} " + " ".join(out))
