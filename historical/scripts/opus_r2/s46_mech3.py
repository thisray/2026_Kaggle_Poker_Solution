"""Three-action mechanism model for the fourth family and visible-deviation evidence.
P(a | own, pb) = (1-alpha) * pi0(a | own) + alpha * Q(a | pb), a in {fold, call, raise}; pi0 from control pairs (partner-flat).
Evidence probability of a hand = P(active | a, own, pb) * (1 - pi0(a | own))  [active AND the action differs from the normal one].
Builds r2h2 (first-5 decoder over this probability) and a self-consistency table under the fitted model."""
import numpy as np, pandas as pd, hashlib
from numba import njit
from scipy.optimize import minimize
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); ts = np.load(f"{D}/h_ts.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = dict(zip(hidx.hand_id, hidx.hi))
OWN = np.array([0.35, 0.45, 0.55, 0.65]); PAR = np.array([0.4, 0.5, 0.6, 0.7])
@njit(cache=True)
def first_actions(H, S, T, off, a_seat, a_st, Y, pfeq, out):
    n = 0
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        for d in range(2):
            k1 = ka if d == 0 else kb; k2 = kb if d == 0 else ka; a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            if k1 < 0 or (k2 >= 0 and k2 < k1): continue
            y = Y[k1]; act = 0 if y == 0 else (2 if y == 3 else 1)      # fold / call(or check) / raise
            out[n, 0] = r; out[n, 1] = pfeq[h, a]; out[n, 2] = pfeq[h, b]; out[n, 3] = act; n += 1
    return n
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
mids = pd.read_csv(f"{C}/r2d_p2comb_other.csv", usecols=["pair_id", "predicted_behavior"]); mids = set(mids[mids.predicted_behavior == "other_coordination"].pair_id)
tabs = {}
for nm, slots in [("member", c38[c38.pair_id.isin(mids)].slot.values), ("control", c38[c38.rk > 5000].sample(3000, random_state=0).slot.values)]:
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    out = np.zeros((2 * len(H), 4)); n = first_actions(H, S, T, off, a_seat, a_st, Y, pfeq, out)
    A = pd.DataFrame(out[:n], columns=["r", "eA", "eB", "act"]); A["h"] = H[A.r.astype(int)]; A["slot"] = SL[A.r.astype(int)]
    A["ob"] = np.searchsorted(OWN, A.eA); A["pb"] = np.searchsorted(PAR, A.eB); A["act"] = A.act.astype(int); tabs[nm] = A
Cc = tabs["control"]; pi0 = np.zeros((5, 3))
for ob in range(5):
    v = np.bincount(Cc[Cc.ob == ob].act, minlength=3).astype(float) + 0.5; pi0[ob] = v / v.sum()
M = tabs["member"]; ob = M.ob.values; pb = M.pb.values; act = M.act.values
def unpack(x):
    al = 1 / (1 + np.exp(-x[0])); Z = np.c_[np.zeros(5), x[1:].reshape(5, 2)]; Q = np.exp(Z); Q /= Q.sum(1, keepdims=True); return al, Q
def nll(x):
    al, Q = unpack(x); p = (1 - al) * pi0[ob, act] + al * Q[pb, act]; return -np.sum(np.log(np.clip(p, 1e-12, 1)))
best = None
for s in range(8):
    x0 = np.random.default_rng(s).normal(0, 1, 11); r = minimize(nll, x0, method="L-BFGS-B")
    if best is None or r.fun < best.fun: best = r
al, Q = unpack(best.x)
print("alpha", round(al, 3)); print("active policy Q(action | partner bin) [fold, call, raise]:\n", np.round(Q, 3)); print("pi0(action | own bin):\n", np.round(pi0, 3))
M["post"] = al * Q[pb, act] / ((1 - al) * pi0[ob, act] + al * Q[pb, act])
M["pev"] = M.post * (1 - pi0[ob, act])
Hq = M.groupby(["slot", "h"]).pev.max().reset_index()
H, S, T, SL = PI.all_pair_hands(1); msl = c38[c38.pair_id.isin(mids)].slot.values; mm = np.isin(SL, msl)
allh = pd.DataFrame({"slot": SL[mm], "h": H[mm]}); allh["ts"] = ts[allh.h]
allh = allh.merge(Hq, on=["slot", "h"], how="left").fillna({"pev": 0.0}).sort_values(["slot", "ts"]).reset_index(drop=True)
PHI = 0.75
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
allh["pfk"] = 0.0
for sl, idx in allh.groupby("slot").indices.items(): allh.loc[allh.index[idx], "pfk"] = first_k_prob(allh.pev.values[idx] * PHI)
EV = [f"evidence_hand_{i}" for i in range(1, 6)]; sm = c38.set_index("pair_id").slot; picks = {}
for nm, f in [("r2c_old(CI-routed r11)", "r2c_p2top600_other.csv"), ("r2d_ev(c-first)", "r2d_p2comb_other_ev_on_r15.csv"), ("r2e(c-all)", "r2e_p2comb_other_evall_on_r15.csv"), ("r2h(raise-only model)", "r2h_p2comb_other_evmodel_on_r15.csv")]:
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    picks[nm] = {sm[r.pair_id]: [id2hi.get(x, -1) for x in [getattr(r, c) for c in EV]] for r in sub.itertuples()}
picks["r2h2(3-action deviation model)"] = {sl: g.sort_values("pfk", ascending=False).h.tolist()[:5] for sl, g in allh.groupby("slot")}
rng = np.random.default_rng(0); res = {k: [] for k in picks}
for sl, g in allh.groupby("slot"):
    hs = g.h.values; q = g.pev.values * PHI
    for s in range(300):
        z = rng.random(len(q)) < q; ev = hs[z][:5]
        if len(ev) == 0: continue
        evs = set(ev); mden = min(5, len(ev))
        for k, pk in picks.items():
            hits = 0; ap = 0.0
            for i, hh in enumerate(pk.get(sl, [])[:5]):
                if hh in evs: hits += 1; ap += hits / (i + 1)
            res[k].append(ap / mden)
for k, v in res.items(): print(f"expected AP@5 (3-action deviation model): {k:32s} {np.mean(v):.4f}")
print("mean evidence-probability hands per pair (sum pev):", round(allh.groupby("slot").pev.sum().median(), 2), " pairs", allh.slot.nunique())
base = pd.read_csv(f"{C}/r2d_p2comb_other_on_r15.csv", dtype=str).set_index("pair_id"); ref = pd.read_csv(f"{C}/r2d_p2comb_other_ev_on_r15.csv", dtype=str)
for pid in mids:
    hsel = [hi2id[h] for h in picks["r2h2(3-action deviation model)"][sm[pid]]]; base.loc[pid, EV] = hsel + ["NO_EVIDENCE"] * (5 - len(hsel))
outp = base.reset_index()[ref.columns]; path = f"{C}/r2h2_p2comb_other_evdev_on_r15.csv"; outp.to_csv(path, index=False)
ov = np.mean([len(set(picks["r2h2(3-action deviation model)"][sm[p]]) & set(picks["r2d_ev(c-first)"][sm[p]])) for p in mids])
print("r2h2 written; rows differing vs r2d_ev_on_r15:", int((outp.to_numpy() != ref.to_numpy()).any(1).sum()), " mean overlap with r2d_ev picks", round(ov, 2), " sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
allh.to_parquet(f"{OUT}/s46_mech3_posterior.parquet")
