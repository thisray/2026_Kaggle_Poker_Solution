"""Eval-side background-information features for the gated candidate table (4,000 pairs x 20 hands):
pev/post_max from the mechanism model of the pair's PREDICTED family (models fitted on all dev positives; fourth family
from s46), plus unsupervised first-decision partner-card contributions c_first/c_dt/c_abs.  Output joins the 24 round15
TabICL features -> eval_input_29.parquet (same schema as dev_input_29)."""
import numpy as np, pandas as pd, json, os
from numba import njit
from scipy.optimize import minimize
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; DST = f"{A_}/opus_r2_tabicl_20260918"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
P2p = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r"); sp_ = np.load(f"{D}/s_player.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32)
OWN = np.array([0.35, 0.45, 0.55, 0.65]); PAR = np.array([0.4, 0.5, 0.6, 0.7])
@njit(cache=True)
def first_actions(H, S, T, SL, off, a_seat, a_st, Y, pfeq, OWN, PAR, out):
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
            y = Y[k1]; act = 0 if y == 0 else (2 if y == 3 else 1)
            out[n, 0] = SL[r]; out[n, 1] = h; out[n, 2] = np.searchsorted(OWN, pfeq[h, a]); out[n, 3] = np.searchsorted(PAR, pfeq[h, b]); out[n, 4] = act; n += 1
    return n
def decisions(phase, slots):
    H, S, T, SL = PI.all_pair_hands(phase); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    out = np.zeros((2 * len(H), 5), np.int64); n = first_actions(H, S, T, SL, off, a_seat, a_st, Y, pfeq, OWN, PAR, out)
    return pd.DataFrame(out[:n], columns=["slot", "h", "ob", "pb", "act"])
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
ctrl = dv[(dv.label == -1) & (dv.oof < 0.02) & (dv.n >= 57)].sample(3000, random_state=0).slot.values
Cc = decisions(0, ctrl); pi0 = np.zeros((5, 3))
for ob in range(5):
    v = np.bincount(Cc[Cc.ob == ob].act, minlength=3).astype(float) + 0.5; pi0[ob] = v / v.sum()
def fit(A):
    ob, pb, act = A.ob.values, A.pb.values, A.act.values
    def unpack(x):
        al = 1 / (1 + np.exp(-x[0])); Z = np.c_[np.zeros(5), x[1:].reshape(5, 2)]; Q = np.exp(Z); Q /= Q.sum(1, keepdims=True); return al, Q
    def nll(x):
        al, Q = unpack(x); p = (1 - al) * pi0[ob, act] + al * Q[pb, act]; return -np.sum(np.log(np.clip(p, 1e-12, 1)))
    best = None
    for s in range(4):
        r = minimize(nll, np.random.default_rng(s).normal(0, 1, 11), method="L-BFGS-B")
        if best is None or r.fun < best.fun: best = r
    return unpack(best.x)
FAM = {"directed_transfer": None, "soft_play": None, "coordinated_isolation": None}
for f in FAM:
    FAM[f] = fit(decisions(0, dv[(dv.label == 1) & (dv.fam == f)].slot.values)); print(f, "alpha", round(FAM[f][0], 3), flush=True)
Q4 = np.clip(np.array([[0.917, 0.075, 0.008], [0.719, 0.211, 0.070], [0.450, 0.205, 0.345], [0.0, 0.124, 0.876], [0.0, 0.0, 1.0]]), 1e-3, None); Q4 /= Q4.sum(1, keepdims=True)
FAM["other_coordination"] = (0.389, Q4)
g = pd.read_csv(f"{A_}/round15_campaign/gated_tabicl_input.csv")
smap = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]
fam_pred = pd.read_csv(f"{OUT}/r2_candidates/r2d_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior"]).merge(smap, on="pair_id")
g = g.merge(fam_pred[["slot", "predicted_behavior"]], on="slot", how="left")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); g["h"] = g.hand_id.map(dict(zip(hidx.hand_id, hidx.hi))).astype(np.int64)
De = decisions(1, g.slot.unique()); De = De.merge(g[["slot", "predicted_behavior"]].drop_duplicates(), on="slot")
De["pev"] = 0.0; De["post"] = 0.0
for f, (al, Q) in FAM.items():
    m = De.predicted_behavior == f
    ob, pb, act = De.loc[m, "ob"].values, De.loc[m, "pb"].values, De.loc[m, "act"].values
    post = al * Q[pb, act] / ((1 - al) * pi0[ob, act] + al * Q[pb, act]); De.loc[m, "post"] = post; De.loc[m, "pev"] = post * (1 - pi0[ob, act])
hp = De.groupby(["slot", "h"]).agg(pev=("pev", "max"), post_max=("post", "max")).reset_index()
g = g.merge(hp, on=["slot", "h"], how="left").fillna({"pev": 0.0, "post_max": 0.0})
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, gg in loc.groupby("pool"): mem[pool, gg.local.values] = gg.player_gi.values
plo = mem[g.slot.values // 900, (g.slot.values % 900) // 30]; phi = mem[g.slot.values // 900, g.slot.values % 30]
cf = np.zeros(len(g)); cd_ = np.zeros(len(g)); ca = np.zeros(len(g))
for i, (h, pa, pbb) in enumerate(zip(g.h.values, plo, phi)):
    s_ = int(np.argmax(sp_[h] == pa)); t_ = int(np.argmax(sp_[h] == pbb)); ks = {}
    for k in range(off[h], off[h + 1]):
        if a_st[k] != 0: break
        ks.setdefault(int(a_seat[k]), k)
    b1 = b2 = b3 = 0.0
    for a, b in [(s_, t_), (t_, s_)]:
        if a not in ks or (b in ks and ks[b] < ks[a]): continue
        k = ks[a]; p = np.asarray(P2p[k]); e = float(pfeq[h, b]) - 0.5; r = ((Y[k] == 3) - p[3]) - ((Y[k] == 0) - p[0])
        b1 = max(b1, r * e); b2 = max(b2, -r * e); b3 = max(b3, abs(r * e))
    cf[i] = b1; cd_[i] = b2; ca[i] = b3
g["c_first"] = cf; g["c_dt"] = cd_; g["c_abs"] = ca
F29 = json.load(open(f"{DST}/features_29.json"))
g[["slot", "hand_id"] + F29].to_parquet(f"{DST}/eval_input_29.parquet"); print("eval_input_29", g.shape, " predicted families:", g.drop_duplicates("slot").predicted_behavior.value_counts().to_dict())
