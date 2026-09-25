"""R4-G5: view DT pairs with listed B events: every evidence hand (typed A/B) and every unlisted B-pattern / A-pattern hand, in time order, with omniscient equities."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; O = f"{A_}/opus_r1_20260917"; D = f"{O}/np"
f = pd.read_parquet(f"{O}/r4/g3_patterns.parquet"); rows = pd.read_parquet(f"{O}/r4/g4_rows_di.parquet"); f = f.merge(rows[["slot", "h", "isA", "isB"]], on=["slot", "h"])
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_st = np.load(f"{D}/a_st.npy", mmap_mode="r"); bb = np.load(f"{D}/h_bb.npy")
AC = "fxcbrA"; npairs = int(sys.argv[1]); seed = int(sys.argv[2])
g = f.groupby("slot").agg(nA=("isA", "sum"), nB=("isB", "sum")); pick = g[(g.nA < 5) & (g.nB >= 1) & (g.nA + g.nB == 5)].sample(npairs, random_state=seed).index
for sl in pick:
    x = f[f.slot == sl].sort_values(["ts", "h"]).reset_index(drop=True); x["k"] = np.arange(len(x)); print("#" * 30, "slot", sl, "n", len(x), "nA", int(x.isA.sum()), "nB", int(x.isB.sum()))
    for _, r in x[(x.ev) | (x.B_SR == 1) | (x.A_SR == 1) | (x.x_bigR == 1)].iterrows():
        h = int(r.h); out = []; cur = -1
        for k in range(off[h], off[h + 1]):
            st = int(a_st[k]);
            if st != cur: out.append("|" + "PFTR"[st] + ":"); cur = st
            s_ = int(a_seat[k]); who = "R" if s_ == r.rs else ("S" if s_ == r.ss else "o"); out.append(f"{who}{AC[int(a_act[k])]}" + (f"{a_amt[k] / bb[h]:.0f}" if a_act[k] >= 2 else ""))
        tag = ("EV-A" if r.isA else "EV-B") if r.ev else "    "; rk = f"r{int(r.evidence_rank)}" if r.ev else "  "
        print(f"{tag} {rk} k={int(r.k):3d} eqS={r.k_ps_eq_last:.2f} eqR={r.k_pr_eq_last:.2f} netR={r.x_netR:+.0f} netS={r.x_netS:+.0f} patA={int(r.A_SR)} patB={int(r.B_SR)} " + " ".join(out))
