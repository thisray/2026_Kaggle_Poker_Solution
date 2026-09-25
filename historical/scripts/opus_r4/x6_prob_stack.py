"""R4-X6: probability-scale fusion instead of rank averaging. For the R15 top-20 candidates we have the OOF TabICL probability `tab`, the ranker's within-pair rank `r_rs`
and the event model's first-five marginal q (x4 OOF). score = logit(tab) + beta * logit(q) + gamma * logit(r_rs); (beta, gamma) chosen by nested pool-fold CV.
Baselines: R15 blend, and the deployed-style rank average (1-w) * pct(R15) + w * pct(q)."""
import numpy as np, pandas as pd, sys, json, itertools
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
hmap = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi
t45 = pd.read_parquet(f"{O}/r3/t45_known_e_rerank.parquet"); t45["h"] = t45.hand_id.map(hmap); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
BET = (0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5); GAM = (0, 0.5, 1.0, 2.0, 4.0); res = {}
for fam2, fam, w0 in (("di", "directed_transfer", 0.35), ("so", "soft_play", 0.15), ("co", "coordinated_isolation", 0.5)):
    s = pd.read_parquet(f"{O}/r4/x4_rows_{fam2}.parquet").reset_index(drop=True); P = np.load(f"{O}/r4/x4_oofp_{fam2}.npy"); counts = s.groupby("slot").ev.sum()
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one")
    c = c.merge(t45[["slot", "h", "tab", "r_rs", "b", "fold"]], on=["slot", "h"], how="left", validate="one_to_one").reset_index(drop=True); assert c.tab.notna().all()
    Q = []
    for p in P:
        q = pd.Series(C.first_k_marginal(s, p), index=pd.MultiIndex.from_arrays([s.slot, s.h])); Q.append(q.reindex(pd.MultiIndex.from_arrays([c.slot, c.h])).values)
    fold_of = c.groupby("slot").fold.first(); folds = sorted(fold_of.unique())
    def per_pair(score_list): return np.mean([C.pair_ap(c, sc, counts).values for sc in score_list], axis=0)
    base = per_pair([-c.r.values]); rankavg = per_pair([C.rank_candidates(s, c.drop(columns=["q", "newscore"], errors="ignore"), C.first_k_marginal(s, p), weight=w0).newscore.values for p in P])
    grid = {(b_, g_): per_pair([lg(c.tab.values) + b_ * lg(q) + g_ * lg(np.clip(c.r_rs.values, 0.025, 0.975)) for q in Q]) for b_, g_ in itertools.product(BET, GAM)}
    idx = counts.index; fo = fold_of.reindex(idx).values; tbl = {k: float(v.mean()) for k, v in grid.items()}; best_all = max(tbl, key=tbl.get)
    nested = np.zeros(len(idx)); picks = []
    for f in folds:
        tr = fo != f; k = max(grid, key=lambda kk: grid[kk][tr].mean()); nested[fo == f] = grid[k][fo == f]; picks.append(k)
    pl = pd.Series(idx // 900, index=idx); up = pl.unique(); rng = np.random.RandomState(4)
    def boot(d): d = pd.Series(d, index=idx); return float(np.mean([np.mean(np.concatenate([d[pl == p_].values for p_ in rng.choice(up, len(up))])) > 0 for _ in range(1000)]))
    print(f"{fam2}: R15 {base.mean():.4f} | rank-avg w{w0} {rankavg.mean():.4f} | tab only {grid[(0, 0)].mean():.4f} | tab+ranker best-gamma {max(tbl[(0, g_)] for g_ in GAM):.4f} | best grid {best_all} {tbl[best_all]:.4f} | NESTED {nested.mean():.4f} picks {picks}")
    print(f"     nested - R15 {nested.mean() - base.mean():+.4f} P>0 {boot(nested - base):.3f} | nested - rankavg {nested.mean() - rankavg.mean():+.4f} P>0 {boot(nested - rankavg):.3f}", flush=True)
    res[fam] = dict(R15=float(base.mean()), rankavg=float(rankavg.mean()), nested=float(nested.mean()), best=str(best_all), best_val=tbl[best_all], picks=[str(k) for k in picks])
json.dump(res, open(f"{O}/r4/x6_prob_stack.json", "w"), indent=1)
