"""R4-U9: view big right-direction pots of DT pairs whose big pots are (a) mostly listed, (b) never listed. Roles: R receiver, S sender."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
m = pd.read_parquet(f"{OUT}/r4/u5_directed_transfer.parquet"); m["bigR"] = m.big & (m.dir == 1)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_st = np.load(f"{D}/a_st.npy", mmap_mode="r")
c1 = np.load(f"{D}/s_c1.npy", mmap_mode="r"); c2 = np.load(f"{D}/s_c2.npy", mmap_mode="r"); bb = np.load(f"{D}/h_bb.npy", mmap_mode="r"); board = np.load(f"{D}/h_board.npy", mmap_mode="r"); stack = np.load(f"{D}/s_stack.npy", mmap_mode="r")
R_ = "23456789TJQKA"; SU = "cdhs"
def card(c): c = int(c); return R_[c // 4] + SU[c % 4] if 0 <= c < 52 else "--"
AC = "fxcbrA"
w = m[(m.zone != "post") & m.bigR]; g = w.groupby("sl").ev.agg(["sum", "size"])
yes = g[(g["size"] >= 3) & (g["sum"] >= g["size"] - 0)].index[:3].tolist(); no = g[(g["size"] >= 4) & (g["sum"] == 0)].index[:4].tolist()
for tag, sls in [("BIG POTS LISTED", yes), ("BIG POTS NEVER LISTED", no)]:
    for sl in sls:
        x = m[(m.sl == sl)]; print("#" * 40, tag, "slot", sl, "ev idx", x[x.ev].k.tolist(), "n", len(x))
        for _, r in x[(x.bigR | x.ev) & (x.zone != "post")].iterrows():
            h = int(r.h); sa, sb = int(r.sa), int(r.sb); rs, ss = (sa, sb) if r.recvA else (sb, sa)
            ks = np.arange(off[h], off[h + 1]); out = []; cur = -1
            for k in ks:
                st = int(a_st[k])
                if st != cur: out.append("|" + "PFTR"[st] + ":"); cur = st
                s = int(a_seat[k]); who = "R" if s == rs else ("S" if s == ss else "o")
                out.append(f"{who}{AC[int(a_act[k])]}" + (f"{a_amt[k] / bb[h]:.0f}" if a_act[k] >= 2 else ""))
            bd = "".join(card(c) for c in np.asarray(board[h]).ravel()[:5])
            print(f"{'EV ' if r.ev else '   '}k={int(r.k):3d} R[{card(c1[h,rs])}{card(c2[h,rs])}] S[{card(c1[h,ss])}{card(c2[h,ss])}] bd[{bd}] stR={stack[h,rs]/bb[h]:.0f} stS={stack[h,ss]/bb[h]:.0f} netR={r.netR:+.0f} netS={r.netS:+.0f} " + " ".join(out))
