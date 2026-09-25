"""Model-based evidence for the fourth family.  Mechanism (docs/30 R2-X4): with prob alpha a member's first preflop decision
(before partner acts) is 'active' and follows the partner's cards: raise with prob g(partner strength); otherwise it follows
the normal policy pi0(own strength) (= control pairs).  Fit alpha, g per partner bin on members; posterior that an observed
raise is an active (planted) raise.  Then (i) build the decoder with calibrated q, (ii) self-consistency simulation of the
expected AP@5 of each candidate evidence rule under 'evidence = first 5 planted raises (thinned 75%)'."""
import numpy as np, pandas as pd, json, os
from numba import njit
from scipy.stats import poisson
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); ts = np.load(f"{D}/h_ts.npy")
Y = np.load(f"{OUT}/dec_Y.npy")
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
            out[n, 0] = r; out[n, 1] = pfeq[h, a]; out[n, 2] = pfeq[h, b]; out[n, 3] = 1.0 if Y[k1] == 3 else 0.0; n += 1
    return n
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
mids = pd.read_csv(f"{C}/r2d_p2comb_other.csv", usecols=["pair_id", "predicted_behavior"]); mids = set(mids[mids.predicted_behavior == "other_coordination"].pair_id)
tabs = {}
for nm, slots in [("member", c38[c38.pair_id.isin(mids)].slot.values), ("control", c38[c38.rk > 5000].sample(3000, random_state=0).slot.values)]:
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    out = np.zeros((2 * len(H), 4)); n = first_actions(H, S, T, off, a_seat, a_st, Y, pfeq, out)
    A = pd.DataFrame(out[:n], columns=["r", "eA", "eB", "aggr"]); A["h"] = H[A.r.astype(int)]; A["slot"] = SL[A.r.astype(int)]
    A["ob"] = np.searchsorted(OWN, A.eA); A["pb"] = np.searchsorted(PAR, A.eB); tabs[nm] = A
pi0 = tabs["control"].groupby("ob").aggr.mean().reindex(range(5)).values          # normal policy by own bin (partner-flat in controls)
M = tabs["member"]; mt = M.groupby(["ob", "pb"]).aggr.agg(["mean", "size"])
# fit alpha, g[pb] by least squares on cell rates: rate(ob,pb) = (1-alpha)*pi0[ob] + alpha*g[pb]
from scipy.optimize import minimize
cells = mt.reset_index()
def loss(x):
    al = 1 / (1 + np.exp(-x[0])); g = 1 / (1 + np.exp(-x[1:]))
    pred = (1 - al) * pi0[cells.ob.values] + al * g[cells.pb.values]
    return float(np.sum(cells["size"].values * (cells["mean"].values - pred) ** 2))
res = minimize(loss, np.zeros(6), method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-9))
al = 1 / (1 + np.exp(-res.x[0])); g = 1 / (1 + np.exp(-res.x[1:]))
print("fitted alpha", round(al, 3), " g(partner bin) =", np.round(g, 3), " pi0(own bin) =", np.round(pi0, 3), flush=True)
# posterior that an observed raise is an active raise
M["post"] = np.where(M.aggr > 0, al * g[M.pb] / ((1 - al) * pi0[M.ob] + al * g[M.pb]), 0.0)
Hq = M.groupby(["slot", "h"]).post.max().reset_index(); Hq["ts"] = ts[Hq.h]
# all shared hands of members (hands with no qualifying decision get q=0)
H, S, T, SL = PI.all_pair_hands(1); msl = c38[c38.pair_id.isin(mids)].slot.values; mm = np.isin(SL, msl)
allh = pd.DataFrame({"slot": SL[mm], "h": H[mm]}); allh["ts"] = ts[allh.h]
allh = allh.merge(Hq[["slot", "h", "post"]], on=["slot", "h"], how="left").fillna({"post": 0.0}).sort_values(["slot", "ts"]).reset_index(drop=True)
allh["tpct"] = allh.groupby("slot").ts.rank(pct=True); allh["hand_id"] = allh.h.map(hi2id)
# model-optimal decoder: P(h among first K planted) with thinning phi
PHI = 0.75
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
allh["pfk"] = 0.0
for sl, idx in allh.groupby("slot").indices.items():
    allh.loc[allh.index[idx], "pfk"] = first_k_prob(allh.post.values[idx] * PHI)
# candidate pick sets
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
sm = c38.set_index("pair_id").slot
picks = {}
for nm, f in [("r2c_old(CI-routed r11)", "r2c_p2top600_other.csv"), ("r2d_ev(c-first decoder)", "r2d_p2comb_other_ev_on_r15.csv"), ("r2e(c-all decoder)", "r2e_p2comb_other_evall_on_r15.csv"), ("r2g(earliest c>0.05)", "r2g_p2comb_other_evearly_on_r15.csv")]:
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    picks[nm] = {sm[r.pair_id]: [id2hi.get(x, -1) for x in [getattr(r, c) for c in EV]] for r in sub.itertuples()}
picks["r2h(model-optimal)"] = {sl: g.sort_values("pfk", ascending=False).h.tolist()[:5] for sl, g in allh.groupby("slot")}
# self-consistency simulation
rng = np.random.default_rng(0); NS = 300; res_ap = {k: [] for k in picks}
groups = {sl: g for sl, g in allh.groupby("slot")}
for sl, g in groups.items():
    hs = g.h.values; q = g.post.values * PHI
    for s in range(NS):
        z = rng.random(len(q)) < q; ev = hs[z][:5]
        if len(ev) == 0: continue
        evs = set(ev); m = min(5, len(ev))
        for k, pk in picks.items():
            top = pk.get(sl, [])[:5]; hits = 0; ap = 0.0
            for i, hh in enumerate(top):
                if hh in evs: hits += 1; ap += hits / (i + 1)
            res_ap[k].append(ap / m)
for k, v in res_ap.items(): print(f"expected AP@5 under mechanism model: {k:28s} {np.mean(v):.4f}")
# write r2h candidate (base r2d_on_r15 risk/behavior, evidence = model-optimal)
base = pd.read_csv(f"{C}/r2d_p2comb_other_on_r15.csv", dtype=str).set_index("pair_id"); ref = pd.read_csv(f"{C}/r2d_p2comb_other_ev_on_r15.csv", dtype=str)
for pid in mids:
    sl = sm[pid]; hsel = [hi2id[h] for h in picks["r2h(model-optimal)"][sl]]
    base.loc[pid, EV] = hsel + ["NO_EVIDENCE"] * (5 - len(hsel))
outp = base.reset_index()[ref.columns]; path = f"{C}/r2h_p2comb_other_evmodel_on_r15.csv"; outp.to_csv(path, index=False)
import hashlib
print("r2h written; rows differing from r2d_ev_on_r15:", int((outp.to_numpy() != ref.to_numpy()).any(1).sum()), " sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
allh.to_parquet(f"{OUT}/s44_mech_posterior.parquet")
