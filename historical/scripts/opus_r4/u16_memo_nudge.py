"""R4-U16: one-parameter de-memorisation nudge on the R15 blend. For candidates whose most surprising member decision was in policy_v1's training sample,
add lam * (clean surprisal - v1 surprisal as used), clipped at 0 (scores live on the within-pair rank scale 0..1). Full-truth AP@5 denominators, pool-fold nested CV."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
t = pd.read_parquet(f"{OUT}/r4/u6_memo.parquet")
d = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet"); full = d.groupby("sl").ev.sum()
t["gap"] = np.where(t.key_memo1, np.clip(t.key_clean - t.key_s1, 0, None), 0.0); print("gap>0 share", (t.gap > 0).mean().round(3), "mean gap when >0", t.gap[t.gap > 0].mean().round(3))
def ap5(g, col, den):
    g = g.sort_values(col, ascending=False).head(5); hits = g.ev.values.astype(float); prec = np.cumsum(hits) / (np.arange(len(hits)) + 1); return float((prec * hits).sum() / min(5, den))
lams = [0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.15]
for fam, tf in t.groupby("behavior_family"):
    res = {}
    for lam in lams:
        tf = tf.assign(s2=tf.b + lam * tf.gap); res[lam] = tf.groupby("fold").apply(lambda x: np.mean([ap5(g, "s2", full[sl]) for sl, g in x.groupby("slot")]), include_groups=False)
    r = pd.DataFrame(res).T; w = tf.groupby("fold").slot.nunique(); r["wmean"] = (r * w).sum(axis=1) / w.sum(); print(fam); print(r.round(4).to_string())
    folds = list(w.index); nested = [r.loc[max(lams, key=lambda l: np.average([r.loc[l, g] for g in folds if g != f], weights=[w[g] for g in folds if g != f])), f] for f in folds]
    print("  nested", round(np.average(nested, weights=w.values), 5), "baseline", round(r.loc[0, "wmean"], 5))
