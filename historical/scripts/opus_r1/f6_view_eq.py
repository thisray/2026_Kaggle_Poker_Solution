import numpy as np, pandas as pd, sys
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
fam = sys.argv[1]; npairs = int(sys.argv[2]); seed = int(sys.argv[3]); mode = sys.argv[4] if len(sys.argv) > 4 else "ev"
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evid = pd.read_csv(f"{RAW}/development_evidence.csv")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
off = np.load(f"{D}/a_off.npy"); L = lambda n: np.load(f"{D}/{n}.npy", mmap_mode="r")
a_st, a_seat, a_act, a_amt, a_tc, a_pot, a_stk, a_pa = [L(x) for x in ["a_st","a_seat","a_act","a_amount","a_to_call","a_pot_before","a_stack_before","a_players_active"]]
eqm = np.load(f"{OUT}/act_eqm_v1.npy", mmap_mode="r"); eqla = np.load(f"{OUT}/act_eqla_v1.npy", mmap_mode="r")
sp = L("s_player"); c1 = L("s_c1"); c2 = L("s_c2"); net = L("s_net"); stack = L("s_stack"); board = L("h_board"); bb = L("h_bb"); btn = L("h_btn"); P = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
R_ = "23456789TJQKA"; S_ = "cdhs"
cs = lambda c: R_[c >> 2] + S_[c & 3] if c >= 0 else ".."
ACT = ["fold","check","call","bet","raise","allin"]; ST = ["pre","flop","turn","river"]
pairs = labels[labels.behavior_family == fam].sample(npairs, random_state=seed)
for _, pr in pairs.iterrows():
    A = pmap[pr.player_1]; B = pmap[pr.player_2]
    hs = [hmap[x] for x in evid[evid.pair_id == pr.pair_id].sort_values("evidence_rank").hand_id]
    print("#" * 30, pr.pair_id, fam)
    for h in hs:
        seats = list(sp[h]); sA = seats.index(A); sB = seats.index(B)
        tag = {sA: "A", sB: "B"}
        who = lambda s: tag.get(s, f"o{s}")
        brd = " ".join(cs(c) for c in board[h] if c >= 0)
        print(f"-- hand {h} bb={bb[h]} btn={btn[h]} board=[{brd}]  " + "  ".join(f"{who(s)}:{cs(c1[h,s])}{cs(c2[h,s])} st={stack[h,s]//bb[h]} net={net[h,s]/bb[h]:+.1f} pfeq={P[h,s,12]:.2f}" for s in range(6)))
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]
            print(f"   {ST[a_st[k]]:5s} {who(s):3s} {ACT[a_act[k]]:6s} amt={a_amt[k]/bb[h]:6.1f} tc={a_tc[k]/bb[h]:5.1f} pot={a_pot[k]/bb[h]:6.1f} act={a_pa[k]}  eq_vs_active={eqm[k]:.2f} eq_vs_lastaggr={eqla[k]:.2f}")
