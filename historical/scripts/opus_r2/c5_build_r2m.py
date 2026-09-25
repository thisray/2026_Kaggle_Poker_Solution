"""Fourth-family evidence hypotheses built to be MAXIMALLY DISTINCT (for efficient LB discrimination), all on the r2j2 base
(P-ensemble risk + 77 members + r15 evidence for known families):
  M1 'clear-deviation earliest': earliest 5 hands with a CLEAR informed raise (own pf_eq<0.45, partner pf_eq>0.6, member raised
     at first preflop decision before partner acted); fill with the next earliest informed raises (own<0.55, partner>0.55).
  M2 'premium-fold earliest': earliest 5 hands with the opposite visible deviation (own pf_eq>0.62 folded/called while partner
     pf_eq<0.45); fill with M1 hands.
  M3 'both directions earliest': earliest 5 hands with either clear deviation (union of M1 and M2 cores).
Reports pairwise overlaps with r2d_ev (c-first decoder) and r2h2 (model posterior) picks."""
import numpy as np, pandas as pd, hashlib, json, os
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy"); ts = np.load(f"{D}/h_ts.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = dict(zip(hidx.hand_id, hidx.hi))
@njit(cache=True)
def events(H, S, T, off, a_seat, a_st, Y, pfeq, out):
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        m1 = 0; m1s = 0; m2 = 0
        for d in range(2):
            k1 = ka if d == 0 else kb; k2 = kb if d == 0 else ka; a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            if k1 < 0 or (k2 >= 0 and k2 < k1): continue
            ea = pfeq[h, a]; eb = pfeq[h, b]; y = Y[k1]
            if y == 3 and ea < 0.45 and eb > 0.6: m1 = 1
            if y == 3 and ea < 0.55 and eb > 0.55: m1s = 1
            if (y == 0 or y == 2) and ea > 0.62 and eb < 0.45: m2 = 1
        out[r, 0] = m1; out[r, 1] = m1s; out[r, 2] = m2
base = pd.read_csv(f"{C}/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]; sm = sm[sm.pair_id.isin(mids)]
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, sm.slot.values); H, S, T, SL = H[m], S[m], T[m], SL[m]
E = np.zeros((len(H), 3), np.int64); events(H, S, T, off, a_seat, a_st, Y, pfeq, E)
X = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "m1": E[:, 0], "m1s": E[:, 1], "m2": E[:, 2]}).sort_values(["slot", "ts"])
print("per-pair counts (median): M1 core", X.groupby("slot").m1.sum().median(), " M1 soft", X.groupby("slot").m1s.sum().median(), " M2 core", X.groupby("slot").m2.sum().median(), " pairs", X.slot.nunique())
def pick(g, prim, fill):
    p = g[g[prim] == 1].h.tolist()
    for f in fill:
        if len(p) >= 5: break
        p += [x for x in g[g[f] == 1].h.tolist() if x not in p]
    if len(p) < 5: p += [x for x in g.h.tolist() if x not in p]
    return p[:5]
X["m12"] = ((X.m1 == 1) | (X.m2 == 1)).astype(int)
rules = {"M1_clear_raise": ("m1", ["m1s", "m2"]), "M2_premium_fold": ("m2", ["m1", "m1s"]), "M3_both": ("m12", ["m1s"])}
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
def load_picks(f):
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    return {r.pair_id: set(id2hi.get(x, -1) for x in [getattr(r, c) for c in EV]) for r in sub.itertuples()}
ref = {"r2d_ev": load_picks("r2d_p2comb_other_ev_on_r15.csv"), "r2h2": load_picks("r2h2_p2comb_other_evdev_on_r15.csv"), "r2c_old": load_picks("r2c_p2top600_other.csv")}
pid_of = sm.set_index("slot").pair_id
for nm, (prim, fill) in rules.items():
    out = base.set_index("pair_id").copy(); ov = {k: [] for k in ref}
    for sl, g in X.groupby("slot"):
        pid = pid_of[sl]; p = pick(g, prim, fill); out.loc[pid, EV] = [hi2id[h] for h in p]
        for k in ref: ov[k].append(len(set(p) & ref[k][pid]))
    o = out.reset_index()[base.columns]; path = f"{C}/r2m_{nm}_on_r2j2.csv"; o.to_csv(path, index=False)
    diff = int((o.to_numpy() != base.to_numpy()).any(1).sum())
    print(f"{nm}: rows differing from r2j2 {diff}; mean overlap with " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in ov.items()) + f"; sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
