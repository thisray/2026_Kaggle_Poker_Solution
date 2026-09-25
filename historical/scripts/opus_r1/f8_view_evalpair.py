import numpy as np, pandas as pd, sys
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D_ = f"{OUT}/np"
keys = [int(k) for k in sys.argv[1].split(",")]; topn = int(sys.argv[2])
HS = pd.read_parquet(f"{OUT}/subs/r1_handscores_eval.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
off = np.load(f"{D_}/a_off.npy"); L = lambda x: np.load(f"{D_}/{x}.npy", mmap_mode="r")
a_st, a_seat, a_act, a_amt, a_tc, a_pot, a_pa = [L(x) for x in ["a_st","a_seat","a_act","a_amount","a_to_call","a_pot_before","a_players_active"]]
eqm = np.load(f"{OUT}/act_eqm_v1.npy", mmap_mode="r"); eqla = np.load(f"{OUT}/act_eqla_v1.npy", mmap_mode="r")
probs = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r")
sp = L("s_player"); c1 = L("s_c1"); c2 = L("s_c2"); net = L("s_net"); board = L("h_board"); bb = L("h_bb")
R_ = "23456789TJQKA"; S_ = "cdhs"; cs = lambda c: R_[c >> 2] + S_[c & 3]
ACT = ["fold","check","call","bet","raise","allin"]; ST = ["pre","flop","turn","river"]
for key in keys:
    A = key // 12000; B = key % 12000
    slot = loc.pool.loc[A] * 900 + loc.local.loc[A] * 30 + loc.local.loc[B]
    g = HS[HS.slot == slot].sort_values("s", ascending=False).head(topn)
    print("#" * 20, "pair", key, "slot", slot, "n_hands", int((HS.slot == slot).sum()))
    for _, r in g.iterrows():
        h = int(r.h); seats = list(sp[h]); sA = seats.index(A); sB = seats.index(B); tag = {sA: "A", sB: "B"}; who = lambda s: tag.get(s, f"o{s}")
        brd = " ".join(cs(c) for c in board[h] if c >= 0)
        print(f"-- hand {h} s={r.s:.3f} board=[{brd}] " + " ".join(f"{who(s)}:{cs(c1[h,s])}{cs(c2[h,s])} {net[h,s]/bb[h]:+.1f}" for s in range(6)))
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]; act = a_act[k]; tc = a_tc[k]; amt = a_amt[k]
            y = 0 if act == 0 else (1 if act == 1 else (3 if (act in (3, 4) or (act == 5 and amt > tc)) else 2))
            print(f"   {ST[a_st[k]]:5s} {who(s):3s} {ACT[act]:6s} amt={amt/bb[h]:6.1f} tc={tc/bb[h]:5.1f} pot={a_pot[k]/bb[h]:6.1f} act={a_pa[k]} hs1={HS1[h, a_st[k], s]:.2f} eq={eqm[k]:.2f} p=[{probs[k,0]:.2f} {probs[k,1]:.2f} {probs[k,2]:.2f} {probs[k,3]:.2f}] sur={-np.log(max(probs[k,y],1e-6)):.2f}")
