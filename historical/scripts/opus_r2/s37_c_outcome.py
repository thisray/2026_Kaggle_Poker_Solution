"""Outcome of 'informed raise' hands: seeds (confirmed fourth family, top-600 p2>2.5) vs control pairs.  If planted hands differ in
outcome (partner wins / pair net), use it to sharpen the fourth-family evidence rule."""
import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
ie = PN.index("pf_eq_rand"); inet = PN.index("net_bb"); iwon = PN.index("won"); isd = PN.index("sd"); ifold = PN.index("folded")
e = pd.read_parquet(f"{OUT}/s35_q_later_eval.parquet")
seeds = e[(e.rk <= 600) & (e.p2 > 2.5)].slot.values
ctrl = e[(e.rk > 5000)].sample(800, random_state=0).slot.values
H, S, T, SL = PI.all_pair_hands(1)
def rows(slots):
    m = np.isin(SL, slots); out = []
    for h, s, t, sl in zip(H[m], S[m], T[m], SL[m]):
        ks = {}
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            ks.setdefault(int(a_seat[k]), k)
        best = 0.0; who = -1
        for a, b in [(s, t), (t, s)]:
            if a not in ks or (b in ks and ks[b] < ks[a]): continue
            k = ks[a]; p = np.asarray(P2[k]); c = ((Y[k] == 3) - p[3] - ((Y[k] == 0) - p[0])) * (float(Pt[h, b, ie]) - 0.5)
            if c > best: best = c; who = a
        if who < 0: continue
        b = t if who == s else s
        out.append((sl, h, best, float(Pt[h, who, inet]), float(Pt[h, b, inet]), float(Pt[h, b, iwon]), float(Pt[h, who, ifold]), float(Pt[h, b, isd])))
    return pd.DataFrame(out, columns=["slot", "h", "c", "net_actor", "net_partner", "partner_won", "actor_folded", "partner_sd"])
Rs = rows(seeds); Rc = rows(ctrl)
for nm, Rr in [("seeds", Rs), ("control", Rc)]:
    hi = Rr[Rr.c > 0.1]
    print(f"{nm:8s} high-c hands {len(hi):5d}: partner won {hi.partner_won.gt(0).mean():.3f}  actor folded later {hi.actor_folded.gt(0).mean():.3f}  pair net {(hi.net_actor + hi.net_partner).mean():+.2f} bb  partner net {hi.net_partner.mean():+.2f}  partner sd {hi.partner_sd.gt(0).mean():.3f}")
