"""Which hand-level events are enriched in confirmed fourth-family member pairs vs control pairs (eval phase)?
The planted 'behavior-specific action' must be present in ~all planted hands (~8% of member hands)."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
L = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = L[0][2:].split(","); PN = L[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); a_act = np.load(f"{D}/a_act.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); MU = float(pfeq.mean())
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
mem_ids = pd.read_csv(f"{OUT}/r2_candidates/r2d_p2comb_other.csv", usecols=["pair_id", "predicted_behavior"]); mem_ids = set(mem_ids[mem_ids.predicted_behavior == "other_coordination"].pair_id)
mem = c38[c38.pair_id.isin(mem_ids)].slot.values; ctrl = c38[c38.rk > 5000].sample(1500, random_state=0).slot.values
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, np.r_[mem, ctrl]); H, S, T, SL = H[m], S[m], T[m], SL[m]; y = np.isin(SL, mem)
ri = lambda n: RN.index(n); pi = lambda n: PN.index(n)
def either(c): return (np.asarray(R[H, S, T, ri(c)]) + np.asarray(R[H, T, S, ri(c)])) > 0
vA = np.asarray(Pt[H, S, pi("vpip")]) > 0; vB = np.asarray(Pt[H, T, pi("vpip")]) > 0
fA = np.asarray(Pt[H, S, pi("pfr")]) > 0; fB = np.asarray(Pt[H, T, pi("pfr")]) > 0
eA = pfeq[H, S]; eB = pfeq[H, T]
wonA = np.asarray(Pt[H, S, pi("won")]) > 0; wonB = np.asarray(Pt[H, T, pi("won")]) > 0
sdA = np.asarray(Pt[H, S, pi("sd")]) > 0; sdB = np.asarray(Pt[H, T, pi("sd")]) > 0
netA = np.asarray(Pt[H, S, pi("net_bb")]); netB = np.asarray(Pt[H, T, pi("net_bb")])
strongB_raiseA = (fA & (eB > 0.6)) | (fB & (eA > 0.6))
weakraise_strongpartner = (fA & (eA < 0.45) & (eB > 0.6)) | (fB & (eB < 0.45) & (eA > 0.6))
ev = {
 "both_vpip": vA & vB, "both_pfr": fA & fB, "raise_over": either("raise_over"), "squeeze": either("squeeze"), "iso_ofold": either("iso_ofold"),
 "fold_to": either("fold_to"), "call_to": either("call_to"), "facing": either("facing"), "hu_streets": either("hu_streets"),
 "aggr_active_both": (np.asarray(R[H, S, T, ri("aggr_active")]) > 0) & (np.asarray(R[H, T, S, ri("aggr_active")]) > 0),
 "raise_with_partner_strong": strongB_raiseA, "weak_raise_partner_strong": weakraise_strongpartner,
 "weakraise_ps & partner_won": ((fA & (eA < 0.45) & (eB > 0.6) & wonB) | (fB & (eB < 0.45) & (eA > 0.6) & wonA)),
 "weakraise_ps & raiser_folded": ((fA & (eA < 0.45) & (eB > 0.6) & ~wonA & ~sdA) | (fB & (eB < 0.45) & (eA > 0.6) & ~wonB & ~sdB)),
 "pair_net>5bb": (netA + netB) > 5, "pair_net<-5bb": (netA + netB) < -5, "one_partner_sd": sdA ^ sdB,
}
rows = []
for k, v in ev.items():
    rm, rc = v[y].mean(), v[~y].mean(); rows.append((k, round(rm, 4), round(rc, 4), round(rm - rc, 4), round(rm / max(rc, 1e-6), 2)))
print(pd.DataFrame(rows, columns=["event", "member_rate", "control_rate", "excess", "enrichment"]).sort_values("excess", ascending=False).to_string(index=False))
print("member hands", int(y.sum()), "control hands", int((~y).sum()))
