"""NEW signal for fourth-family membership: the SECOND member's first preflop decision (after the partner has acted)
depends on the partner's hole cards (s72: B folds strong hands when A is weak).  Neither p2 (first actor only) nor q_later
(excludes each player's first decision) uses it.  Per pair & direction: Rao score z of B's policy-v2 fold / aggression
residuals vs A's preflop equity; r2 = min_dir(z_aggr - z_fold).  Null = dev phase (no fourth family there)."""
import numpy as np, pandas as pd, time
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32)
nh = len(off) - 1
@njit(cache=True)
def first_pre(nh, off, a_seat, a_st, K):
    for h in range(nh):
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            s = a_seat[k]
            if K[h, s] < 0: K[h, s] = k
K = -np.ones((nh, 6), np.int64); first_pre(nh, off, a_seat, a_st, K)
kk = K.ravel(); valid = kk >= 0
pf_fold = np.zeros(nh * 6, np.float32); pf_aggr = np.zeros(nh * 6, np.float32); yf = np.zeros(nh * 6, np.int8); ya = np.zeros(nh * 6, np.int8)
idx = kk[valid]; srt = np.argsort(idx); pr = np.asarray(P2[idx[srt]]); pr_full = np.empty_like(pr); pr_full[srt] = pr
pf_fold[valid] = pr_full[:, 0]; pf_aggr[valid] = pr_full[:, 3]; yf[valid] = (Y[idx] == 0); ya[valid] = (Y[idx] == 3)
pf_fold = pf_fold.reshape(nh, 6); pf_aggr = pf_aggr.reshape(nh, 6); yf = yf.reshape(nh, 6); ya = ya.reshape(nh, 6); log("inputs ready")
mu = float(pfeq.mean())
@njit(cache=True)
def accum(H, S, T, SL, K, pfeq, mu, pf_fold, pf_aggr, yf, ya, ACC):
    for r in range(len(H)):
        h = H[r]
        for d in range(2):
            b = S[r] if d == 0 else T[r]; a = T[r] if d == 0 else S[r]      # b = responder, a = first actor
            ka = K[h, a]; kb = K[h, b]
            if ka < 0 or kb < 0 or kb < ka: continue                          # need a acted first, then b responds
            e = pfeq[h, a] - mu
            pf = pf_fold[h, b]; pa = pf_aggr[h, b]; sl = SL[r]
            ACC[sl, d, 0] += (yf[h, b] - pf) * e; ACC[sl, d, 1] += pf * (1 - pf) * e * e
            ACC[sl, d, 2] += (ya[h, b] - pa) * e; ACC[sl, d, 3] += pa * (1 - pa) * e * e
            ACC[sl, d, 4] += 1
res = {}
for nm, phase in [("dev", 0), ("eval", 1)]:
    H, S, T, SL = PI.all_pair_hands(phase)
    ACC = np.zeros((400 * 900, 2, 5)); accum(H, S, T, SL, K, pfeq, mu, pf_fold, pf_aggr, yf, ya, ACC)
    zf = ACC[:, :, 0] / np.sqrt(ACC[:, :, 1] + 1e-9); za = ACC[:, :, 2] / np.sqrt(ACC[:, :, 3] + 1e-9)
    df = pd.DataFrame({"slot": np.arange(400 * 900), "n_r": ACC[:, 0, 4] + ACC[:, 1, 4], "rzf0": zf[:, 0], "rzf1": zf[:, 1], "rza0": za[:, 0], "rza1": za[:, 1]})
    df = df[df.n_r > 0].copy(); df["r2"] = np.minimum(df.rza0 - df.rzf0, df.rza1 - df.rzf1); df["r2sum"] = (df.rza0 - df.rzf0) + (df.rza1 - df.rzf1)
    df.to_parquet(f"{OUT}/s76_responder_{nm}.parquet"); res[nm] = df; log(nm, len(df))
# ---- analysis
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
d = res["dev"].merge(dv[["slot", "label", "fam", "oof"]], on="slot", how="left")
e = res["eval"].merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"), on="slot"); e["rk"] = e.risk_score.rank(ascending=False)
p2e = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet"); p2e["p2"] = np.minimum(p2e.za0 - p2e.zf0, p2e.za1 - p2e.zf1); e = e.merge(p2e[["slot", "p2"]], on="slot", how="left")
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); e = e.merge(c38[["slot", "q"]] if "q" in c38.columns else c38[["slot"]], on="slot", how="left")
base = pd.read_csv(f"{OUT}/r2_candidates/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior"])
e["member"] = e.pair_id.isin(base[base.predicted_behavior == "other_coordination"].pair_id)
print("dev r2 by label:", d.groupby(d.label.fillna(-1)).r2.describe()[["count", "mean", "std", "50%"]].round(3).to_dict("index"))
print("dev r2 by family (positives):", d[d.label == 1].groupby("fam").r2.agg(["count", "mean", "median"]).round(3).to_dict("index"))
print("eval r2: members", e[e.member].r2.describe()[["count", "mean", "50%"]].round(3).to_dict(), " non-members top600", e[(~e.member) & (e.rk <= 600)].r2.describe()[["count", "mean", "50%"]].round(3).to_dict())
print("corr(p2, r2) among members:", round(e[e.member][["p2", "r2"]].corr().iloc[0, 1], 3), " among all eval:", round(e[["p2", "r2"]].corr().iloc[0, 1], 3))
dn = d[(d.label != 1)]
for thr in [2.0, 2.5, 3.0, 3.5]:
    null_rate = (dn.r2 > thr).mean()
    row = [f"thr {thr}: dev-null rate {null_rate:.5f}"]
    for lo, hi in [(0, 600), (600, 1000), (1000, 2000), (2000, 5000), (5000, 1e9)]:
        g = e[(e.rk > lo) & (e.rk <= hi) & (~e.member)]
        row.append(f"rk{lo}-{int(min(hi, 1e6))}: {int((g.r2 > thr).sum())} obs vs {null_rate * len(g):.1f} exp")
    row.append(f"members {int((e[e.member].r2 > thr).sum())}/77")
    print(" | ".join(row))
e.to_parquet(f"{OUT}/s76_eval_joined.parquet")
