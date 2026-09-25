"""R4-U10: one-parameter pair-context nudge. Hand type = big pot (both members >= TH bb). Context = score-weighted share of big hands among the pair's OTHER candidates.
score' = rank-score + lam * match, match = +(c - cbar) for big hands and -(c - cbar) for small hands. Full-truth AP@5 denominators. Pool-fold CV for lam."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
fam = sys.argv[1]; TH = float(sys.argv[2]) if len(sys.argv) > 2 else 20
m = pd.read_parquet(f"{OUT}/r4/u5_{fam}.parquet"); full = m.groupby("sl").ev.sum()
c = m[m.cand].copy(); c["big"] = ((c.conR >= TH) & (c.conS >= TH)).astype(float)
t45 = pd.read_parquet(f"{OUT}/r3/t45_known_e_rerank.parquet")[["slot", "fold"]].drop_duplicates().rename(columns={"slot": "sl"}); c = c.merge(t45, on="sl")
def ap5(g, col, den):
    g = g.sort_values(col, ascending=False).head(5); hits = g.ev.values.astype(float); prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / min(5, den))
def score(df, col): return np.mean([ap5(g, col, full[sl]) for sl, g in df.groupby("sl")])
print("pairs", c.sl.nunique(), "baseline E (full denominators)", round(score(c, "b"), 5))
for wpow in (1.0, 2.0, 4.0):
    ctx = []
    for sl, g in c.groupby("sl"):
        w = (g.b.values ** wpow); bg = g.big.values; tot = w.sum() - w; ctx.append(pd.Series((np.dot(w, bg) - w * bg) / np.maximum(tot, 1e-9), index=g.index))
    c["ctx"] = pd.concat(ctx); cbar = c.ctx.mean(); c["match"] = np.where(c.big == 1, c.ctx - cbar, cbar - c.ctx)
    res = {}
    for lam in (0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        c["s2"] = c.b + lam * c.match; res[lam] = [score(c[c.fold == f], "s2") for f in sorted(c.fold.unique())]
    r = pd.DataFrame(res).T; r["mean"] = r.mean(axis=1); print("wpow", wpow); print(r.round(4).to_string())
    # nested: choose lam on the other folds
    folds = sorted(c.fold.unique()); nested = []
    for f in folds:
        best = max(res, key=lambda l: np.mean([res[l][j] for j, ff in enumerate(folds) if ff != f])); nested.append(res[best][folds.index(f)])
    print("  nested CV E", round(np.mean(nested), 5), "vs baseline", round(np.mean(res[0]), 5))
