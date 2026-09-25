"""Pair-level Bayes factor of the hand-level tilt mixture (s83 parameters) for EVERY pair in dev (null: no fourth family)
and eval:  BF_p = sum_h log[(1-pi) + pi*exp(l_h)],  l_h = sum over member decisions (partner active) of tilted log-LR.
Streaming numba kernel.  Compares eval rank bands with the dev null matched on the number of co-seated hands."""
import numpy as np, pandas as pd, time
from numba import njit, prange
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
Q0 = np.clip(np.load(f"{OUT}/dec_probs_v2.npy").astype(np.float64), 1e-6, 1.0); Q0 /= Q0.sum(1, keepdims=True); log("q0 loaded")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float64); eql = np.asarray(Pt[:, :, PN.index("eq_last")]).astype(np.float64)
BETA = np.array([20.795, 1.987, 1.673, 1.855]); PI_ = 0.498
# centring constants (controls, per street) recomputed quickly from a random sample of decisions with a partner-active pair is
# approximated by the population means of the partner-strength variables:
MU = np.array([float(pfeq.mean()), float(eql.mean()), float(eql.mean()), float(eql.mean())])
@njit(cache=True)
def pair_bf(H, S, T, SL, off, a_seat, a_st, Y, Q0, pfeq, eql, BETA, MU, PI_, BF, NH, NA):
    for r in range(len(H)):
        h = H[r]; active0 = 1; active1 = 1; l = 0.0; nd = 0
        act = np.ones(6, np.int64)
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]
            if s == S[r] or s == T[r]:
                o = T[r] if s == S[r] else S[r]
                if act[o] == 1:
                    st = a_st[k]
                    e = (pfeq[h, o] if st == 0 else eql[h, o]) - MU[st]
                    be = BETA[st] * e
                    z = Q0[k, 0] * np.exp(-be) + Q0[k, 1] + Q0[k, 2] + Q0[k, 3] * np.exp(be)
                    t = 0.0
                    if Y[k] == 3: t = 1.0
                    elif Y[k] == 0: t = -1.0
                    l += be * t - np.log(z); nd += 1
            if Y[k] == 0: act[s] = 0
        sl = SL[r]
        if nd > 0:
            m = l if l > 0 else 0.0
            BF[sl] += m + np.log((1 - PI_) * np.exp(-m) + PI_ * np.exp(l - m)); NA[sl] += 1
        NH[sl] += 1
res = {}
for nm, ph in [("dev", 0), ("eval", 1)]:
    H, S, T, SL = PI.all_pair_hands(ph); BF = np.zeros(400 * 900); NH = np.zeros(400 * 900); NA = np.zeros(400 * 900)
    pair_bf(H, S, T, SL, off, a_seat, a_st, Y, Q0, pfeq, eql, BETA, MU, PI_, BF, NH, NA)
    df = pd.DataFrame({"slot": np.arange(400 * 900), "bf": BF, "n_h": NH, "n_a": NA}); df = df[df.n_h > 0]; res[nm] = df; log(nm, len(df))
    df.to_parquet(f"{OUT}/s84_pair_bf_{nm}.parquet")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
d = res["dev"].merge(dv[["slot", "label", "fam"]], on="slot", how="left")
e = res["eval"].merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"), on="slot"); e["rk"] = e.risk_score.rank(ascending=False)
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior"])
e = e.merge(base, on="pair_id"); e["member"] = e.predicted_behavior == "other_coordination"
print("dev BF by label:", d.groupby(d.label.fillna(-1)).bf.describe()[["count", "mean", "50%", "max"]].round(2).to_dict("index"))
print("dev positives BF by family:", d[d.label == 1].groupby("fam").bf.agg(["mean", "median", "max"]).round(2).to_dict("index"))
print("eval members BF:", e[e.member].bf.describe()[["mean", "50%", "min"]].round(2).to_dict())
# null matched on number of co-seated hands (dev hand counts scaled to eval's 2/3 exposure is not exact -> bin by n_a)
bins = [0, 40, 60, 80, 100, 130, 200, 10000]
d["nb"] = pd.cut(d.n_a, bins); e["nb"] = pd.cut(e.n_a, bins)
dn = d[d.label != 1]
for thr in [5, 10, 20, 40]:
    rate = dn.groupby("nb", observed=True).bf.apply(lambda x: (x > thr).mean())
    row = [f"BF>{thr}"]
    for lo, hi in [(0, 600), (600, 1000), (1000, 2000), (2000, 5000), (5000, 20000), (20000, 1e9)]:
        g = e[(e.rk > lo) & (e.rk <= hi) & (~e.member)]; o = int((g.bf > thr).sum()); x = float(g.nb.map(rate).astype(float).fillna(0).sum())
        row.append(f"{lo}-{int(min(hi, 1e6))}: {o}/{x:.1f}")
    row.append(f"members {int((e[e.member].bf > thr).sum())}/77")
    print(" | ".join(row))
cand = e[(~e.member) & (e.bf > 10)].sort_values("bf", ascending=False)
print(cand[["pair_id", "rk", "predicted_behavior", "bf", "n_a"]].head(30).round(2).to_string())
e.to_parquet(f"{OUT}/s84_eval_bf_joined.parquet"); d.to_parquet(f"{OUT}/s84_dev_bf_joined.parquet")
