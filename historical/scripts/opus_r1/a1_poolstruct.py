import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
ev = pd.read_parquet(f"{OUT}/m1_eval_scores.parquet"); dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
e = ev[ev.in_eval].copy()
e["r"] = e.groupby("pool").score.rank(ascending=False, method="first")
g = e.groupby("pool").score
print("eval: per-pool top1 score quantiles", np.round(np.quantile(g.max(), [0.05,0.1,0.25,0.5,0.75]),4))
top2 = e[e.r==2].set_index("pool").score; top1 = g.max(); top3 = e[e.r==3].set_index("pool").score
print("eval: per-pool 2nd score quantiles", np.round(np.quantile(top2, [0.25,0.5,0.75,0.9,0.95]),4))
print("eval: per-pool 3rd score quantiles", np.round(np.quantile(top3, [0.5,0.9,0.95,0.99]),4))
for thr in [0.5, 0.2, 0.05, 0.01]:
    c = (e.score > thr).groupby(e.pool).sum()
    print(f"eval thr {thr}: pools by count of pairs above", c.value_counts().sort_index().to_dict())
# players in top pairs: do they overlap (rings)?
t = e[e.score > 0.05]
pl = pd.concat([t.p_lo, t.p_hi]).value_counts()
print("eval players in >0.05 pairs multiplicity", pl.value_counts().to_dict())
# dev: labelled positives + U
d = dv.copy()
d["pop"] = (d.n >= 57) & ((~d.touch_pos) | (d.label >= 0))
d = d[d["pop"]]
for thr in [0.5, 0.2, 0.05]:
    c1 = ((d.oof > thr) & (d.label == 1)).groupby(d.pool).sum(); c2 = ((d.oof > thr) & (d.label == -1)).groupby(d.pool).sum()
    print(f"dev thr {thr}: labelled-pos above per pool", c1.value_counts().sort_index().to_dict(), " U above per pool", c2.value_counts().sort_index().to_dict())
# pools with 0 labelled positives: U high?
npos = d.groupby("pool").apply(lambda x: (x.label==1).sum())
uhi = d.groupby("pool").apply(lambda x: ((x.label==-1)&(x.oof>0.2)).sum())
print(pd.crosstab(npos, uhi))
# overlap between high dev-U pairs and high eval pairs
ee = ev.set_index("key").score
d["eval_score"] = d.key.map(ee)
hu = d[(d.label == -1) & (d.oof > 0.2)]
print("dev-U high (>0.2):", len(hu), " their eval-phase score quantiles", np.round(np.quantile(hu.eval_score.dropna(), [0.1,0.5,0.9]),5))
he = ev[ev.in_eval & (ev.score > 0.2)]
dvs = dv.set_index("key").oof
print("eval high (>0.2):", len(he), " their dev-phase OOF quantiles", np.round(np.quantile(he.key.map(dvs).dropna(), [0.1,0.5,0.9]),5))
