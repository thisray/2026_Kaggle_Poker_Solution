"""R4-G7: view SP pairs' evidence in evidence_rank order with run labels (A = first ascending run, B = second), equities and compact action strings; plus unlisted fold-to-partner / showdown hands."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; O = f"{A_}/opus_r1_20260917"; D = f"{O}/np"
f = pd.read_parquet(f"{O}/r4/g3_patterns.parquet"); g2 = pd.read_parquet(f"{O}/r4/g2_sublists.parquet")[["h", "run", "nruns"]]; f = f.merge(g2, on="h", how="left")
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_st = np.load(f"{D}/a_st.npy", mmap_mode="r"); bb = np.load(f"{D}/h_bb.npy")
AC = "fxcbrA"; fam = sys.argv[1]; npairs = int(sys.argv[2]); seed = int(sys.argv[3]); x = f[f.fam == fam]
pick = x[x.nruns == 2].slot.drop_duplicates().sample(npairs, random_state=seed).values
for sl in pick:
    g = x[x.slot == sl].sort_values(["ts", "h"]).reset_index(drop=True); g["k"] = np.arange(len(g)); print("#" * 30, fam, "slot", sl, "n", len(g))
    for _, r in g[g.ev].sort_values("evidence_rank").iterrows():
        h = int(r.h); out = []; cur = -1
        for k in range(off[h], off[h + 1]):
            st = int(a_st[k])
            if st != cur: out.append("|" + "PFTR"[st] + ":"); cur = st
            s_ = int(a_seat[k]); who = "R" if s_ == r.rs else ("S" if s_ == r.ss else "o"); out.append(f"{who}{AC[int(a_act[k])]}" + (f"{a_amt[k] / bb[h]:.0f}" if a_act[k] >= 2 else ""))
        print(f"r{int(r.evidence_rank)} {'A' if r.run == 0 else 'B'} k={int(r.k):3d} eqS={r.k_ps_eq_last:.2f} eqR={r.k_pr_eq_last:.2f} netR={r.x_netR:+.0f} netS={r.x_netS:+.0f} " + " ".join(out))
