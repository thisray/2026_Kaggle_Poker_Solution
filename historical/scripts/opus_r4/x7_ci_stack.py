"""R4-X7: CI only - probability-scale fusion logit(tab) + beta * logit(q) with a wider beta grid, with and without the R3 hard listing filter; nested pool-fold CV,
paired against the deployed r13 configuration and against the rank-average used in r15/r16."""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
hmap = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi
t45 = pd.read_parquet(f"{O}/r3/t45_known_e_rerank.parquet"); t45["h"] = t45.hand_id.map(hmap); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
t5 = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"})[["slot", "h", "y1", "pa_at_trig"]]
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
s = pd.read_parquet(f"{O}/r4/x4_rows_co.parquet").reset_index(drop=True); P = np.load(f"{O}/r4/x4_oofp_co.npy"); counts = s.groupby("slot").ev.sum()
c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts", "y1", "pa_at_trig"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one")
c = c.merge(t45[["slot", "h", "tab", "fold"]], on=["slot", "h"], how="left", validate="one_to_one").merge(t5, on=["slot", "h"], how="left").reset_index(drop=True)
viol = (~((c.pa_at_trig == 6) & c.y1.isin([2, 3]))).values.astype(float)
Q = [pd.Series(C.first_k_marginal(s, p), index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(pd.MultiIndex.from_arrays([c.slot, c.h])).values for p in P]
pp = lambda scs: np.mean([C.pair_ap(c, sc, counts).values for sc in scs], axis=0)
idx = counts.index; fo = c.groupby("slot").fold.first().reindex(idx).values; pl = pd.Series(idx // 900, index=idx); up = pl.unique(); rng = np.random.RandomState(4)
boot = lambda d: float(np.mean([np.mean(np.concatenate([pd.Series(d, index=idx)[pl == p_].values for p_ in rng.choice(up, len(up))])) > 0 for _ in range(1000)]))
rank_hard = pp([(j := C.rank_candidates(s, c.drop(columns=["q", "newscore"], errors="ignore"), C.first_k_marginal(s, p), weight=0.5)).newscore.values - 100 * viol for p in P])
print(f"rank-average w0.5 + hard (the r15/r16 CI patch): {rank_hard.mean():.4f}")
grid = {}
for beta in (1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 1e6):
    for hard in (0, 1):
        grid[(beta, hard)] = pp([(lg(c.tab.values) + beta * lg(q) if beta < 1e5 else lg(q)) - 1000 * hard * viol for q in Q])
        print(f"beta {beta:>9} hard {hard}: {grid[(beta, hard)].mean():.4f}  delta vs rank-avg-hard {grid[(beta, hard)].mean() - rank_hard.mean():+.4f}  P>0 {boot(grid[(beta, hard)] - rank_hard):.3f}", flush=True)
nested = np.zeros(len(idx)); picks = []
for f in sorted(set(fo)):
    k = max(grid, key=lambda kk: grid[kk][fo != f].mean()); nested[fo == f] = grid[k][fo == f]; picks.append(k)
print(f"NESTED {nested.mean():.4f} picks {picks}  delta vs rank-avg-hard {nested.mean() - rank_hard.mean():+.4f} P>0 {boot(nested - rank_hard):.3f}")
json.dump({str(k): float(v.mean()) for k, v in grid.items()} | {"nested": float(nested.mean()), "rank_hard": float(rank_hard.mean())}, open(f"{O}/r4/x7_ci_stack.json", "w"), indent=1)
