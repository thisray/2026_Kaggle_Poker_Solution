"""R3-F19: recover partner-card likelihood ratios for member decisions that policy_v2 memorised (its 6M training rows),
using policy_v1 (26 base features, its own RandomState(0) 4M training rows) wherever v1 did not train on the decision.
Writes (k, slot) -> r_v1 plus in-train flags for both models; also reports each model's in/out-of-sample log q0(y)."""
import numpy as np, pandas as pd, lightgbm as lgb
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; R3 = f"{OUT}/r3"; C = f"{OUT}/r2_candidates"
Y = np.load(f"{OUT}/dec_Y.npy"); N = len(Y)
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
P1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
K = np.sort(np.random.default_rng(5).choice(N, 2_000_000, replace=False))
for nm, P, inn in (("v1", P1, in1), ("v2", P2, in2)):
    lq = np.log(np.clip(np.asarray(P[K])[np.arange(len(K)), Y[K]], 1e-6, 1))
    print(f"{nm}: in-train share {inn[K].mean():.3f}; mean log q0(y) in {lq[inn[K]].mean():.4f} out {lq[~inn[K]].mean():.4f}")
base = pd.read_csv(f"{C}/r5_subh.csv", dtype=str); e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]
sl = base.merge(e85, on="pair_id"); sl = sl[sl.predicted_behavior == "other_coordination"].slot.values
meta = pd.read_parquet(f"{R3}/sub_meta_eval.parquet"); idx = np.flatnonzero(meta.slot.isin(sl).values); m = meta.iloc[idx].reset_index(drop=True)
SX = np.load(f"{R3}/sub_X_eval.npy", mmap_mode="r"); Xs = np.asarray(SX[idx, :26]).astype(np.float64)
X1 = np.load(f"{OUT}/dec_X.npy", mmap_mode="r"); k = m.k.values; o_ = np.argsort(k); Xo = np.empty((len(k), 26)); Xo[o_] = np.asarray(X1[k[o_]])
b1 = lgb.Booster(model_file=f"{OUT}/policy_v1.txt")
qo = b1.predict(Xo, num_threads=2); qs = b1.predict(Xs, num_threads=2)
q0s = np.asarray(P1[np.sort(k)])[np.searchsorted(np.sort(k), k)]
print("v1 own-card re-prediction matches stored dec_probs_v1: max abs diff", float(np.abs(qo - q0s).max()))
qo = np.clip(qo, 1e-6, 1); qo /= qo.sum(1, keepdims=True); qs = np.clip(qs, 1e-6, 1); qs /= qs.sum(1, keepdims=True)
y = m.y.values; r1 = qs[np.arange(len(m)), y] / qo[np.arange(len(m)), y]
res = pd.DataFrame({"k": k, "slot": m.slot.values, "r_v1": r1, "in_v1": in1[k], "in_v2": in2[k]})
res.to_parquet(f"{R3}/t40_v1_ratio.parquet")
print(f"member rows {len(res)}; in v2 {res.in_v2.mean():.3f}; in v2 & not v1 {(res.in_v2 & ~res.in_v1).mean():.3f}; in both {(res.in_v2 & res.in_v1).mean():.3f}")
for nm, msk in (("out-v1", ~res.in_v1), ("in-v1", res.in_v1)):
    print(f"mean log r_v1 on {nm} rows: {np.log(res.r_v1[msk]).mean():+.4f} (n {int(msk.sum())})")
