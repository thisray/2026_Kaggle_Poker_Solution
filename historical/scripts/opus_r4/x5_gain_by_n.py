"""R4-X5: is the event-model blend gain stable across exposure (number of co-seated hands n)? Eval pairs have ~0.7x the dev exposure, so the low-n stratum is the transfer-relevant one.
Also reports per-pair win/loss counts and a pool bootstrap of the gain."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; TAG = sys.argv[1] if len(sys.argv) > 1 else "x3"
cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
for fam2, w in (("di", 0.35), ("so", 0.15), ("co", 0.65)):
    s = pd.read_parquet(f"{O}/r4/{TAG}_rows_{fam2}.parquet").reset_index(drop=True); P = np.load(f"{O}/r4/{TAG}_oofp_{fam2}.npy"); s["pool"] = s.slot // 900
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
    counts = s.groupby("slot").ev.sum(); n = s.groupby("slot").size(); base = C.pair_ap(c, -c.r, counts)
    new = np.mean([C.pair_ap((j := C.rank_candidates(s, c, C.first_k_marginal(s, p), weight=w)), j.newscore, counts).values for p in P], axis=0); new = pd.Series(new, index=base.index)
    d = new - base; ter = pd.qcut(n.reindex(base.index), 3, labels=["low n", "mid n", "high n"])
    print(f"{fam2} w={w}: R15 {base.mean():.4f} -> {new.mean():.4f} (delta {d.mean():+.4f}); pairs better {int((d > 1e-9).sum())} worse {int((d < -1e-9).sum())}")
    print("   by exposure:", {k: (round(float(base[ter == k].mean()), 4), round(float(new[ter == k].mean()), 4), int((ter == k).sum()), f"n<= {int(n.reindex(base.index)[ter == k].max())}") for k in ["low n", "mid n", "high n"]})
    pools = pd.Series(base.index // 900, index=base.index); up = pools.unique(); rng = np.random.RandomState(3); bs = []
    for _ in range(2000):
        pick = rng.choice(up, len(up)); bs.append(np.mean(np.concatenate([d[pools == p].values for p in pick])))
    print(f"   pool bootstrap: mean {np.mean(bs):+.4f}  P(delta>0) {np.mean(np.array(bs) > 0):.3f}  5%..95% [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}]")
