"""Fourth family: 'activation vs completion'.  Known-family evidence = activations where the coordinated interaction
unfolded (DT: partner wins; CI: outsiders fold; SP: partner plays).  For the 77 fourth-family members (eval): informed
first decisions (A acts before B) and what happened next (B continues? outsiders fold? pair wins?), and how often each
existing evidence variant's picks are 'completed'."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy"); ts = np.load(f"{D}/h_ts.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); lst = np.asarray(Pt[:, :, PN.index("last_street")]).astype(np.float32)
swon = np.load(f"{D}/s_won.npy", mmap_mode="r"); snet = np.load(f"{D}/s_net.npy", mmap_mode="r")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = dict(zip(hidx.hand_id, hidx.hi))
@njit(cache=True)
def seq(H, S, T, off, a_seat, a_st, Y, out):
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        if ka < 0 and kb < 0: out[r, 0] = -1; continue
        if kb < 0 or (ka >= 0 and ka < kb): f = ka; o = kb; out[r, 0] = 0
        else: f = kb; o = ka; out[r, 0] = 1
        out[r, 1] = Y[f]; out[r, 2] = Y[o] if o >= 0 else -1
        nf = 0
        for k in range(f + 1, off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] != S[r] and a_seat[k] != T[r] and Y[k] == 0: nf += 1
        out[r, 3] = nf
base = pd.read_csv(f"{C}/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]; sm = sm[sm.pair_id.isin(mids)]
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, sm.slot.values); H, S, T, SL = H[m], S[m], T[m], SL[m]
O = np.zeros((len(H), 4), np.int64); seq(H, S, T, off, a_seat, a_st, Y, O)
ix = np.arange(len(H)); who = O[:, 0]
X = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "who": who, "y1": O[:, 1], "y2": O[:, 2], "ofold": O[:, 3]})
X["own"] = np.where(who == 0, pfeq[H, S], pfeq[H, T]); X["par"] = np.where(who == 0, pfeq[H, T], pfeq[H, S])
won = np.asarray(swon[H]); X["pair_win"] = (won[ix, S] > 0) | (won[ix, T] > 0)
X["b_cont"] = X.y2.isin([1, 2, 3]); X["both_flop"] = np.minimum(lst[H, S], lst[H, T]) >= 1
X = X[X.who >= 0].sort_values(["slot", "ts"]).reset_index(drop=True)
X["k"] = X.groupby("slot").cumcount()
inf_raise = (X.y1 == 3) & (X.own < 0.45) & (X.par > 0.6); inf_fold = (X.y1.isin([0])) & (X.own > 0.62) & (X.par < 0.45)
print("fourth-family members: pairs", X.slot.nunique(), " co-seated hands with a first decision", len(X))
for nm, mk in [("clear informed raise (own<.45, par>.6)", inf_raise), ("clear informed fold (own>.62, par<.45)", inf_fold), ("all first-actor raises", X.y1 == 3)]:
    G = X[mk]
    print(f"  {nm:42s} n={len(G):5d} | B continues {G.b_cont.mean():.2f} | outsider folds {G.ofold.mean():.2f} | both flop {G.both_flop.mean():.2f} | pair win {G.pair_win.mean():.2f}")
# controls: same events in low-ranked pairs
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); ctrl = c38[c38.rk > 5000].sample(3000, random_state=0).slot.values
H2, S2, T2, SL2 = PI.all_pair_hands(1); m2 = np.isin(SL2, ctrl); H2, S2, T2, SL2 = H2[m2], S2[m2], T2[m2], SL2[m2]
O2 = np.zeros((len(H2), 4), np.int64); seq(H2, S2, T2, off, a_seat, a_st, Y, O2); ix2 = np.arange(len(H2)); w2 = O2[:, 0]
Z = pd.DataFrame({"who": w2, "y1": O2[:, 1], "y2": O2[:, 2], "ofold": O2[:, 3]})
Z["own"] = np.where(w2 == 0, pfeq[H2, S2], pfeq[H2, T2]); Z["par"] = np.where(w2 == 0, pfeq[H2, T2], pfeq[H2, S2])
won2 = np.asarray(swon[H2]); Z["pair_win"] = (won2[ix2, S2] > 0) | (won2[ix2, T2] > 0); Z["b_cont"] = Z.y2.isin([1, 2, 3])
Z = Z[Z.who >= 0]
Gz = Z[(Z.y1 == 3) & (Z.own < 0.45) & (Z.par > 0.6)]
print(f"  CONTROL clear 'informed' raise (natural)             n={len(Gz):5d} | B continues {Gz.b_cont.mean():.2f} | outsider folds {Gz.ofold.mean():.2f} | pair win {Gz.pair_win.mean():.2f}")
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
pid2sl = sm.set_index("pair_id").slot
Xi = X.set_index(["slot", "h"])
for nm, f in [("c-first (r2j2 base)", "r2j2_lgbcat2_p2comb_other_ev_on_r15.csv"), ("M1", "r2m_M1_clear_raise_on_r2j2.csv"), ("M3", "r2m_M3_both_on_r2j2.csv"),
              ("r2h2", "r2h2ev_on_r2j2.csv"), ("CI-routed r2c", "r2c_p2top600_other.csv")]:
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    rows = []
    for r in sub.itertuples():
        sl = pid2sl[r.pair_id]
        for c in EV:
            hh = id2hi.get(getattr(r, c), -1)
            if (sl, hh) in Xi.index: rows.append(Xi.loc[(sl, hh)])
    R_ = pd.DataFrame(rows)
    print(f"  picks {nm:20s}: n={len(R_):4d} | first-actor raise {(R_.y1 == 3).mean():.2f} | B continues {R_.b_cont.mean():.2f} | pair win {R_.pair_win.mean():.2f} | median k {R_.k.median():.0f}")
X.to_parquet(f"{OUT}/s71_f4_hands.parquet")
