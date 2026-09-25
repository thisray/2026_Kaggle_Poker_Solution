"""Time structure of the fourth-family 'informed raise' hands (per-hand contribution c) within the eval phase."""
import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); ts = np.load(f"{D}/h_ts.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(","); ie = PN.index("pf_eq_rand")
old = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet"); old["p2"] = np.minimum(old.za0 - old.zf0, old.za1 - old.zf1)
seeds = old[old.p2 > 4].slot.values
rng = np.random.default_rng(0); ctrl = rng.choice(old[(old.p2 < 1)].slot.values, 600, replace=False)
H, S, T, SL = PI.all_pair_hands(1)
def contrib(slots):
    m = np.isin(SL, slots); rows = []
    for h, s, t, sl in zip(H[m], S[m], T[m], SL[m]):
        ks = {}
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            ks.setdefault(int(a_seat[k]), k)
        best = 0.0
        for a, b in [(s, t), (t, s)]:
            if a not in ks or (b in ks and ks[b] < ks[a]): continue
            k = ks[a]; p = np.asarray(P2[k]); best = max(best, ((Y[k] == 3) - p[3] - ((Y[k] == 0) - p[0])) * (float(Pt[h, b, ie]) - 0.5))
        rows.append((sl, h, ts[h], best))
    C = pd.DataFrame(rows, columns=["slot", "h", "ts", "c"]); C["tpct"] = C.groupby("slot").ts.rank(pct=True); return C
Cs = contrib(seeds); Cc = contrib(ctrl)
for nm, C in [("seeds", Cs), ("control", Cc)]:
    hi = C[C.c > 0.1]
    print(f"{nm:8s} hands {len(C)}  frac c>0.1 {len(hi) / len(C):.4f}  high-c by time quintile:", hi.groupby(pd.cut(hi.tpct, [0, .2, .4, .6, .8, 1.0])).size().to_dict())
    # run-length clustering of high-c hands within pair (in pair-hand order)
    adj = []
    for sl, g in C.sort_values(["slot", "ts"]).groupby("slot"):
        x = (g.c.values > 0.1).astype(int)
        if x.sum() >= 2: adj.append(((x[1:] * x[:-1]).sum() / (x.sum() - 1), x.mean()))
    a = np.array(adj); print(f"          P(next hand high | high) {a[:, 0].mean():.3f} vs base rate {a[:, 1].mean():.3f}")
