"""R3-F20: re-test mid-rank fourth-family recall with de-memorised partner-card ratios.
For every eval pair ranked <= RMAX by the current candidate and every devsub11 pair ranked <= RMAX by the P-model OOF
(exposure-matched null that contains no labelled fourth family), compute the M3 pair marginal log-likelihood ratio
(substitution vs normal play) with the r9 hyperparameters, using r_v2 out of policy_v2's training rows, r_v1 where only
v2 trained on the decision, and r = 1 where both did. Compare eval vs devsub by rank bins; list eval outliers."""
import os, json
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import expit, logit, logsumexp, roots_jacobi
from numpy.polynomial.hermite_e import hermegauss
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; R3 = f"{OUT}/r3"; C = f"{OUT}/r2_candidates"
RMAX = int(os.environ.get("RMAX", "3000")); CAND = os.environ.get("CAND", "r9_subh.csv")
TH = dict(s_pre=0.5647, s_post=0.5703, mu=0.7966, kappa=4.892, sigma=0.519)        # r9 M3 fit (87 members)
Y = np.load(f"{OUT}/dec_Y.npy"); N = len(Y)
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
P1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r"); X1 = np.load(f"{OUT}/dec_X.npy", mmap_mode="r")
b1 = lgb.Booster(model_file=f"{OUT}/policy_v1.txt")

def gather(P, k):
    o = np.argsort(k); q = np.empty((len(k), 4)); q[o] = np.asarray(P[k[o]]); q = np.clip(q, 1e-6, 1); return q / q.sum(1, keepdims=True)

def clean_rows(nm, slots):
    meta = pd.read_parquet(f"{R3}/sub_meta_{nm}.parquet"); idx = np.flatnonzero(meta.slot.isin(slots).values); m = meta.iloc[idx].reset_index(drop=True)
    qs2 = np.load(f"{R3}/q_sub_{nm}.npy", mmap_mode="r")[idx].astype(np.float64); qs2 = np.clip(qs2, 1e-6, 1); qs2 /= qs2.sum(1, keepdims=True)
    k = m.k.values; y = m.y.values; a = np.arange(len(m))
    r = qs2[a, y] / gather(P2, k)[a, y]
    use1 = in2[k] & ~in1[k]; both = in2[k] & in1[k]
    if use1.any():
        SX = np.load(f"{R3}/sub_X_{nm}.npy", mmap_mode="r"); ii = idx[use1]; oo = np.argsort(ii)
        xs = np.empty((len(ii), 26)); xs[oo] = np.asarray(SX[ii[oo], :26]); qs1 = np.clip(b1.predict(xs, num_threads=2), 1e-6, 1); qs1 /= qs1.sum(1, keepdims=True)
        r[use1] = qs1[np.arange(use1.sum()), y[use1]] / gather(P1, k[use1])[np.arange(use1.sum()), y[use1]]
    r[both] = 1.0; m["r"] = r
    print(f"{nm}: {len(m)} rows for {len(slots)} pairs; v1 {use1.mean():.3f}, uninformative {both.mean():.3f}", flush=True)
    return m

def m3_llr(m):
    x, w = roots_jacobi(40, (1 - TH["mu"]) * TH["kappa"] - 1, TH["mu"] * TH["kappa"] - 1); rho = ((x + 1) / 2).clip(1e-9, 1 - 1e-9); lwr = np.log(w) - logsumexp(np.log(w))
    xu, wu = hermegauss(12); u = TH["sigma"] * xu; lwu = np.log(wu) - logsumexp(np.log(wu))
    m = m.sort_values(["slot", "h"]).reset_index(drop=True)
    s = expit(np.where(m.st.values == 0, logit(TH["s_pre"]), logit(TH["s_post"]))[:, None] + u[None, :])
    t1 = np.log1p(s * (m.r.values.clip(1e-12, None)[:, None] - 1))
    key = m.slot.values * 10_000_000 + m.h.values; hs = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    lf = np.add.reduceat(t1, hs, axis=0); hslot = m.slot.values[hs]
    loc = np.logaddexp(np.log1p(-rho)[None, None, :], np.log(rho)[None, None, :] + lf[:, :, None])
    ps = np.r_[0, np.flatnonzero(np.diff(hslot)) + 1]; lp = np.add.reduceat(loc, ps, axis=0)
    llr = logsumexp(lp + lwu[None, :, None] + lwr[None, None, :], axis=(1, 2))
    nd = np.add.reduceat(np.ones(len(m)), np.r_[0, np.flatnonzero(np.diff(m.slot.values)) + 1])
    return pd.DataFrame({"slot": hslot[ps], "llr": llr, "nhands": np.diff(np.r_[ps, len(hs)]), "ndec": nd})

# eval: current candidate ranking
cand = pd.read_csv(f"{C}/{CAND}", dtype=str); cand["rk"] = cand.risk_score.astype(float).rank(ascending=False, method="first")
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]; ev = cand.merge(e85, on="pair_id"); ev = ev[ev.rk <= RMAX]
E = m3_llr(clean_rows("eval", ev.slot.values)).merge(ev[["slot", "pair_id", "rk", "predicted_behavior"]], on="slot")
# devsub11 null: P-model OOF ranking, labels
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet"); o = o[o.src == "devsub11"].copy(); o["rk"] = o.oof.rank(ascending=False, method="first")
lo = o.key // 12000; hi = o.key % 12000; o["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
o = o[o.rk <= RMAX]; lab = [c for c in o.columns if c in ("y", "label", "family", "fam")]
V = m3_llr(clean_rows("devsub11", o.slot.values)).merge(o[["slot", "rk"] + lab], on="slot")
E["set"] = "eval"; V["set"] = "devsub11"; allp = pd.concat([E, V], ignore_index=True); allp.to_parquet(f"{R3}/t41_mid_clean.parquet")
print("devsub label columns:", lab)
bins = [0, 344, 600, 1000, 2000, RMAX]
for a_, b_ in zip(bins[:-1], bins[1:]):
    e = E[(E.rk > a_) & (E.rk <= b_) & (E.predicted_behavior != "other_coordination")]; v = V[(V.rk > a_) & (V.rk <= b_)]
    q = lambda d, t: int((d.llr > t).sum())
    print(f"rank ({a_},{b_}]: eval n={len(e)} llr>5 {q(e, 5)} >10 {q(e, 10)} >20 {q(e, 20)} max {e.llr.max():.1f} | devsub n={len(v)} llr>5 {q(v, 5)} >10 {q(v, 10)} >20 {q(v, 20)} max {v.llr.max():.1f}")
mem = E[E.predicted_behavior == "other_coordination"]; print(f"members n={len(mem)} llr median {mem.llr.median():.1f} p10 {mem.llr.quantile(.1):.1f}")
top = E[(E.predicted_behavior != "other_coordination") & (E.rk > 344)].nlargest(15, "llr")
print(top[["pair_id", "rk", "predicted_behavior", "llr", "nhands", "ndec"]].round(2).to_string(index=False))
