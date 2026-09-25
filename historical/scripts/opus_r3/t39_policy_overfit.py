"""R3-F18 (R19 T11 diagnostic): is the normal policy (policy_v2) overconfident on its own training rows?
m4b_policy_v2.py trains on RandomState(1).choice(N, 6M) rows. Compare mean log q0(y) on training vs non-training
decisions (all decisions and the fourth-family member decisions of the t17 row cache). A small gap means the
partner-card likelihood ratios r = q_sub / q0 are not distorted by in-sample memorisation, so a clean refit is not needed."""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; R3 = f"{OUT}/r3"; C = f"{OUT}/r2_candidates"
Y = np.load(f"{OUT}/dec_Y.npy"); N = len(Y); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
rng = np.random.RandomState(1); samp = np.sort(rng.choice(N, 6_000_000, replace=False)); intrain = np.zeros(N, bool); intrain[samp] = True
r2 = np.random.default_rng(5); K = np.sort(r2.choice(N, 3_000_000, replace=False))
lq = np.log(np.clip(np.asarray(P2[K])[np.arange(len(K)), Y[K]], 1e-6, 1))
print(f"all decisions: in-train {intrain[K].mean():.3f} share; mean log q0(y) in {lq[intrain[K]].mean():.5f} vs out {lq[~intrain[K]].mean():.5f} (gap {lq[intrain[K]].mean() - lq[~intrain[K]].mean():+.5f})")
base = pd.read_csv(f"{C}/r7_subh.csv", dtype=str); e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]
sl = base.merge(e85, on="pair_id"); sl = sl[sl.predicted_behavior == "other_coordination"].slot.values
m = pd.read_parquet(f"{R3}/sub_meta_eval.parquet", columns=["k", "slot", "y"]); m = m[m.slot.isin(sl)]
k = np.sort(m.k.unique()); lqm = np.log(np.clip(np.asarray(P2[k])[np.arange(len(k)), Y[k]], 1e-6, 1))
print(f"member decisions ({len(k)}): in-train share {intrain[k].mean():.3f}; mean log q0(y) in {lqm[intrain[k]].mean():.5f} vs out {lqm[~intrain[k]].mean():.5f} (gap {lqm[intrain[k]].mean() - lqm[~intrain[k]].mean():+.5f})")
rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet", columns=["k", "slot", "r"]); rows = rows[rows.slot.isin(sl)]
it = intrain[rows.k.values]; lr = np.log(rows.r.clip(1e-12, None).values)
print(f"member rows: mean log r in-train {lr[it].mean():+.5f} (n {it.sum()}) vs out {lr[~it].mean():+.5f} (n {(~it).sum()})")
