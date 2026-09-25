"""Print high-scoring NON-evidence hands occurring before the last evidence hand of a positive pair (they fail the exact criterion)."""
import numpy as np, pandas as pd, sys
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D_ = f"{OUT}/np"
fam = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
X = pd.read_parquet(f"{OUT}/a7_nonev_before_high.parquet")
M = pd.read_parquet(f"{OUT}/m6_handscores.parquet")
cand = X[X.fam == fam].sample(n, random_state=seed)
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = {}
for pool, g in loc.groupby("pool"): members[pool] = dict(zip(g.local, g.player_gi))
off = np.load(f"{D_}/a_off.npy"); L = lambda x: np.load(f"{D_}/{x}.npy", mmap_mode="r")
a_st, a_seat, a_act, a_amt, a_tc, a_pot, a_pa = [L(x) for x in ["a_st","a_seat","a_act","a_amount","a_to_call","a_pot_before","a_players_active"]]
eqm = np.load(f"{OUT}/act_eqm_v1.npy", mmap_mode="r"); eqla = np.load(f"{OUT}/act_eqla_v1.npy", mmap_mode="r")
probs = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r")
sp = L("s_player"); c1 = L("s_c1"); c2 = L("s_c2"); net = L("s_net"); board = L("h_board"); bb = L("h_bb")
R_ = "23456789TJQKA"; S_ = "cdhs"; cs = lambda c: R_[c >> 2] + S_[c & 3]
ACT = ["fold","check","call","bet","raise","allin"]; ST = ["pre","flop","turn","river"]
for _, r in cand.iterrows():
    pool = r.sl // 900; A = members[pool][(r.sl % 900) // 30]; B = members[pool][r.sl % 30]; h = int(r.h)
    evs = M[(M.sl == r.sl) & M.ev].sort_values("ts")
    order = "".join("E" if t < r.ts else "" for t in evs.ts)
    seats = list(sp[h]); sA = seats.index(A); sB = seats.index(B); tag = {sA: "A", sB: "B"}; who = lambda s: tag.get(s, f"o{s}")
    brd = " ".join(cs(c) for c in board[h] if c >= 0)
    print(f"-- NON-EV {fam} hand {h} score={r.s:.3f} evidence-before={len(order)} board=[{brd}] " + " ".join(f"{who(s)}:{cs(c1[h,s])}{cs(c2[h,s])} {net[h,s]/bb[h]:+.1f}" for s in range(6)))
    for k in range(off[h], off[h + 1]):
        s = a_seat[k]; act = a_act[k]; tc = a_tc[k]; amt = a_amt[k]
        y = 0 if act == 0 else (1 if act == 1 else (3 if (act in (3, 4) or (act == 5 and amt > tc)) else 2))
        print(f"   {ST[a_st[k]]:5s} {who(s):3s} {ACT[act]:6s} amt={amt/bb[h]:6.1f} tc={tc/bb[h]:5.1f} pot={a_pot[k]/bb[h]:6.1f} act={a_pa[k]} hs1={HS1[h, a_st[k], s]:.2f} eq={eqm[k]:.2f} eqla={eqla[k]:.2f} p=[{probs[k,0]:.2f} {probs[k,1]:.2f} {probs[k,2]:.2f} {probs[k,3]:.2f}] sur={-np.log(max(probs[k,y],1e-6)):.2f}")
