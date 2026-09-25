import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
m = pd.read_parquet(f"{OUT}/m2_dev_handscores.parquet")
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
ev = pd.read_csv(f"{RAW}/development_evidence.csv"); ev["h"] = ev.hand_id.map(hmap)
m = m.merge(ev[["pair_id","h","evidence_rank"]], on=["pair_id","h"], how="left")
pos = m[m.label == 1].copy()
pos["trank"] = pos.groupby("pair_id").ts.rank(method="first")
rows = []
for pid, g in pos.groupby("pair_id"):
    e = g[g.is_ev]; k = len(e)
    t_last = e.ts.max(); t_first = e.ts.min()
    for thr in [0.5, 0.2, 0.1]:
        hi = g[(~g.is_ev) & (g.hs > thr)]
        before = (hi.ts < t_last).sum(); after = (hi.ts > t_last).sum()
        # expected fraction of non-ev hands before t_last under uniform placement
        nonev = g[~g.is_ev]; frac_before = (nonev.ts < t_last).mean()
        rows.append((pid, g.behavior_family.iloc[0], k, thr, before, after, frac_before, len(g), e.trank.max()))
R = pd.DataFrame(rows, columns=["pair_id","fam","k","thr","before","after","frac_before_all","n","last_ev_trank"])
for thr in [0.5, 0.2, 0.1]:
    r = R[R.thr == thr]
    for fam, g in r.groupby("fam"):
        tot = g.before.sum() + g.after.sum()
        exp_before = (g.frac_before_all * (g.before + g.after)).sum()
        print(f"thr {thr} {fam:22s} high non-ev hands: before last-ev {g.before.sum():4d} after {g.after.sum():4d}  expected-before-if-uniform {exp_before:7.1f}  pairs {len(g)}")
r = R[R.thr == 0.5]
print("position (time rank) of last evidence hand / n:", np.round(np.quantile(r.last_ev_trank / r.n, [0.1,0.25,0.5,0.75,0.9]),3))
# Within evidence: rank vs time for each family (exact match rate)
e = pos[pos.is_ev].copy()
e["t_order"] = e.groupby("pair_id").ts.rank(method="first")
for fam, g in e.groupby("behavior_family"):
    exact = (g.groupby("pair_id").apply(lambda x: (x.evidence_rank.values == x.t_order.values).all())).mean()
    print(fam, "pairs with evidence_rank == chronological order:", round(exact, 3))
# Are evidence hands contiguous in the pair's shared-hand timeline? gaps (in shared-hand index) between consecutive evidence hands
e = e.sort_values(["pair_id","ts"])
e["gap"] = e.groupby("pair_id").trank.diff()
for fam, g in e.groupby("behavior_family"):
    print(fam, "gap (shared-hand index) between consecutive ev hands quantiles:", np.round(np.quantile(g.gap.dropna(), [0.1,0.25,0.5,0.75,0.9]),1), " first ev trank quantiles:", np.round(np.quantile(g.groupby('pair_id').trank.min(), [0.1,0.25,0.5,0.75,0.9]),1))
# real time: session structure; are ev hands clustered in same session (gap in seconds)?
e["dt"] = e.groupby("pair_id").ts.diff()
print("seconds between consecutive ev hands quantiles:", np.round(np.quantile(e.dt.dropna(), [0.1,0.25,0.5,0.75,0.9]),0))
