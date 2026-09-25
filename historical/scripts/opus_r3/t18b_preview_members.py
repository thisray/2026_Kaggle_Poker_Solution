"""Preview: NDw with substitution posteriors (t13 member arrays) vs the LB-verified tilt NDw, 77 members: overlap, posterior sharpness."""
import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
a = np.load(f"{OUT}/t13_members_qsub.npy"); M = pd.DataFrame(a[:, :7].astype(np.int64), columns=["slot", "h", "k", "s", "o", "st", "y"])
q0 = a[:, 7:11]; qs = a[:, 11:15]; q0 = np.clip(q0, 1e-6, 1); q0 /= q0.sum(1, keepdims=True); qs = np.clip(qs, 1e-6, 1); qs /= qs.sum(1, keepdims=True)
r = qs[np.arange(len(M)), M.y] / q0[np.arange(len(M)), M.y]; al = np.where(M.st == 0, 0.362, 0.323)
M["post"] = al * r / ((1 - al) + al * r)
hq = M.groupby(["slot", "h"]).post.apply(lambda p: 1 - np.prod(1 - p.values)).rename("q").reset_index()
old = pd.read_parquet(f"{OUT}/c14_hand_tables_ext.parquet")[["slot", "h", "q_HD"]]
j = hq.merge(old, on=["slot", "h"], how="inner")
print(f"hands {len(j)}: corr(q_sub, q_tilt) {np.corrcoef(j.q, j.q_HD)[0, 1]:.3f}; mean q_sub {j.q.mean():.3f} tilt {j.q_HD.mean():.3f}; share q>0.9: sub {(j.q > .9).mean():.3f} tilt {(j.q_HD > .9).mean():.3f}; share q<0.1: sub {(j.q < .1).mean():.3f} tilt {(j.q_HD < .1).mean():.3f}")
print(f"decision posteriors: share post>0.9 {(M.post > .9).mean():.3f}, <0.1 {(M.post < .1).mean():.3f}; mean |post-alpha| {np.mean(np.abs(M.post - al)):.3f}")
