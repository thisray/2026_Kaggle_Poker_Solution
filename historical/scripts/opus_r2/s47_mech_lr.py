"""Neyman-Pearson pair statistic for the fourth family: log-likelihood ratio of the fitted three-action mixture model
(alpha, Q from s46; pi0 from controls) vs the normal policy, summed over each player's first preflop decisions taken before
the partner acts (both directions).  Null calibration on dev U; eval exceedances by r11 rank bucket; new members."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32)
AL = 0.389
Q = np.array([[0.917, 0.075, 0.008], [0.719, 0.211, 0.070], [0.450, 0.205, 0.345], [0.0, 0.124, 0.876], [0.0, 0.0, 1.0]]); Q = np.clip(Q, 1e-3, None); Q /= Q.sum(1, keepdims=True)
PI0 = np.array([[0.981, 0.018, 0.001], [0.917, 0.074, 0.009], [0.733, 0.180, 0.086], [0.390, 0.220, 0.390], [0.035, 0.050, 0.916]])
OWN = np.array([0.35, 0.45, 0.55, 0.65]); PAR = np.array([0.4, 0.5, 0.6, 0.7])
@njit(cache=True)
def llr(H, S, T, SL, off, a_seat, a_st, Y, pfeq, OWN, PAR, PI0, Q, al, ACC):
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
            ob = np.searchsorted(OWN, pfeq[h, a]); pb = np.searchsorted(PAR, pfeq[h, b])
            p0 = PI0[ob, act]; p1 = (1 - al) * p0 + al * Q[pb, act]
            ACC[SL[r], d, 0] += np.log(p1) - np.log(p0); ACC[SL[r], d, 1] += 1
out = {}
for ph in [0, 1]:
    H, S, T, SL = PI.all_pair_hands(ph); ACC = np.zeros((360000, 2, 2)); llr(H, S, T, SL, off, a_seat, a_st, Y, pfeq, OWN, PAR, PI0, Q, AL, ACC)
    out[ph] = ACC[:, :, 0].sum(1)          # both directions: the mechanism is symmetric
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values); dv["L"] = out[0][dv.slot.values]
nul = dv[(dv.label == -1) & (dv.oof < 0.02) & (dv.n >= 38)].L
thr = {a: float(np.quantile(nul, 1 - a)) for a in [1e-2, 1e-3, 1e-4]}; print("null LLR quantiles:", {k: round(v, 2) for k, v in thr.items()}, " null mean", round(nul.mean(), 2))
for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    g = dv[(dv.label == 1) & (dv.fam == f)].L; print(f"dev {f[:2]} positives: frac > thr(1e-3) {(g > thr[1e-3]).mean():.3f}   > thr(1e-4) {(g > thr[1e-4]).mean():.3f}")
e = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv").sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); e["rk"] = np.arange(1, len(e) + 1); e["L"] = out[1][e.slot.values]
mids = pd.read_csv(f"{C}/r2d_p2comb_other.csv", usecols=["pair_id", "predicted_behavior"]); mids = set(mids[mids.predicted_behavior == "other_coordination"].pair_id); e["mem"] = e.pair_id.isin(mids)
print("the 77 members: LLR median", round(e[e.mem].L.median(), 1), " frac > thr(1e-4)", round((e[e.mem].L > thr[1e-4]).mean(), 3), " min", round(e[e.mem].L.min(), 1))
for lo, hi in [(0, 450), (450, 600), (600, 1000), (1000, 2000), (2000, 5000), (5000, 20000), (20000, 120000)]:
    m = (e.rk > lo) & (e.rk <= hi)
    print(f"({lo},{hi}] LLR>thr(1e-3): {int((m & (e.L > thr[1e-3])).sum())} (null {1e-3 * m.sum():.1f}), not in 77: {int((m & (e.L > thr[1e-3]) & ~e.mem).sum())};  >thr(1e-4): {int((m & (e.L > thr[1e-4])).sum())} (null {1e-4 * m.sum():.2f}), not in 77: {int((m & (e.L > thr[1e-4]) & ~e.mem).sum())}")
new = e[(e.L > thr[1e-4]) & ~e.mem].sort_values("L", ascending=False)
print(new[["pair_id", "rk", "risk_score", "L"]].head(25).round(3).to_string())
e[["slot", "pair_id", "rk", "L", "mem"]].to_parquet(f"{OUT}/s47_mech_llr_eval.parquet")
