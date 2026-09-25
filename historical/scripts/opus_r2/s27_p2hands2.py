"""Top per-hand information-dependence contributions for seed pairs: what happens after the 'informed' preflop action?"""
import numpy as np, pandas as pd, sys
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; RAW = f"{A_}/data/raw"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
slots = [int(x) for x in sys.argv[1].split(",")]; nshow = int(sys.argv[2])
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
mu = 0.5
rows = []
for h, s, t, sl in zip(H, S, T, SL):
    ks = {}
    for k in range(off[h], off[h + 1]):
        if a_st[k] != 0: break
        ks.setdefault(int(a_seat[k]), k)
    for a, b in [(s, t), (t, s)]:
        if a not in ks: continue
        if b in ks and ks[b] < ks[a]: continue
        k = ks[a]; p = np.asarray(P2[k]); eB = float(Pt[h, b, PN.index("pf_eq_rand")]); eA = float(Pt[h, a, PN.index("pf_eq_rand")])
        c = ((Y[k] == 3) - p[3] - ((Y[k] == 0) - p[0])) * (eB - mu)
        rows.append((sl, h, a, b, int(Y[k]), p[0], p[3], eA, eB, c))
R = pd.DataFrame(rows, columns=["slot", "h", "a", "b", "y", "pfold", "paggr", "eqA", "eqB", "c"])
Ac = None
for sl in slots:
    g = R[R.slot == sl].sort_values("c", ascending=False).head(nshow)
    hids = [hi2id[x] for x in g.h]
    Ac = pd.read_parquet(f"{RAW}/actions.parquet", filters=[("hand_id", "in", hids)]); Se = pd.read_parquet(f"{RAW}/seats.parquet", filters=[("hand_id", "in", hids)])
    Hh = pd.read_parquet(f"{RAW}/hands.parquet", filters=[("hand_id", "in", hids)])
    print(f"\n######## slot {sl}: hands {int((R.slot == sl).sum())}  sum c {R[R.slot == sl].c.sum():.2f}")
    for r in g.itertuples():
        hid = hi2id[r.h]; s_ = Se[Se.hand_id == hid].sort_values("seat_no"); hh = Hh[Hh.hand_id == hid].iloc[0]; act = Ac[Ac.hand_id == hid].sort_values("action_no")
        seat_of = dict(zip(s_.player_id, s_.seat_no)); pa = s_[s_.seat_no == r.a].player_id.iloc[0]; pb = s_[s_.seat_no == r.b].player_id.iloc[0]; nm = {pa: "A", pb: "B"}
        print(f"  -- c {r.c:+.2f} actor y={r.y} (pfold {r.pfold:.2f} paggr {r.paggr:.2f}) eqA {r.eqA:.2f} eqB {r.eqB:.2f} | board [{hh.board_cards}] pot {hh.final_pot} | " + " ".join(f"s{x.seat_no}{nm.get(x.player_id, '')}:{x.hole_card_1}{x.hole_card_2}({x.net_chips:+d})" for x in s_.itertuples()))
        print("     " + " | ".join(f"{x.street[:2]} s{seat_of[x.player_id]}{nm.get(x.player_id, '')} {x.action}{'' if x.amount == 0 else ' ' + str(x.amount)}" for x in act.itertuples()))
