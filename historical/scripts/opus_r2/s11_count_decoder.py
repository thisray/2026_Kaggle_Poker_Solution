"""Collusion-count decoder: evidence = first 5 COLLUSION hands.  q_h = P(collusion | time-agnostic m19 score), calibrated on
in-window hands only (complete labels there); P(first5_h) = P(#collusion hands before h <= 4) via Poisson-binomial over ALL
earlier pair hands.  Oracles bound the time-handling headroom.  Nested over folds for every tuned knob."""
import numpy as np, pandas as pd, itertools
from sklearn.isotonic import IsotonicRegression
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"
d = pd.read_parquet(f"{A}/round11_scoped/dev_oof_aligned.parquet")
n = pd.read_csv(f"{A}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family", "hand_ts"])
d = d.merge(n, on=["slot", "hand_id"], how="left")
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi)); d["h"] = d.hand_id.map(hmap).astype(np.int64)
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet").rename(columns={"sl": "slot"})
fold_of = d.groupby("slot").fold.first(); M = M[M.slot.isin(fold_of.index)].copy(); M["fold"] = M.slot.map(fold_of).astype(int)
last = M[M.ev].groupby("slot").ts.max(); M["win"] = M.ts <= M.slot.map(last)
M = M.sort_values(["slot", "ts"]).reset_index(drop=True)
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
def Ef(df, col): return float(np.mean([ap5(g, col) for _, g in df.groupby("slot")]))
d["post"] = d.hand_ts > d.slot.map(d[d.ev == 1].groupby("slot").hand_ts.max())
base = Ef(d, "rs_blend"); print("base E", round(base, 6))
d["o1"] = d.rs_blend - 100 * d.post; print("ORACLE demote post-window candidates:", round(Ef(d, "o1"), 6))
# q calibration (per outer fold, fitted on other folds' in-window hands), optionally per family
def pb_before(q, starts):
    """P(count of collusion before h <= k) for k=0..5, per hand (Poisson-binomial over earlier hands of the same pair)."""
    out = np.zeros((len(q), 6))
    for a, b in zip(starts[:-1], starts[1:]):
        dist = np.zeros(7); dist[0] = 1.0
        for i in range(a, b):
            out[i] = np.cumsum(dist[:6])
            p = q[i]; nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[6] += dist[6] * p; dist = nd
    return out
slots = M.slot.values; starts = np.r_[0, np.flatnonzero(slots[1:] != slots[:-1]) + 1, len(M)]
Q = np.zeros(len(M)); famwise = True
for f in range(5):
    tr = (M.fold != f) & M.win; te = M.fold == f
    for fm in M.fam.unique():
        a = tr & (M.fam == fm); b = te & (M.fam == fm)
        ir = IsotonicRegression(y_min=1e-4, y_max=0.999, out_of_bounds="clip").fit(M.s[a], M.ev[a]); Q[b] = ir.predict(M.s[b])
M["q"] = Q
C = pb_before(Q, starts)
for k in range(6): M[f"cb{k}"] = C[:, k]
print("mean q in-window", M.q[M.win].mean().round(4), " post-window", M.q[~M.win].mean().round(4), " ev", M.q[M.ev].mean().round(4))
print("sum q per pair (all dev hands) median", M.groupby("slot").q.sum().median().round(2), " in-window", M[M.win].groupby("slot").q.sum().median().round(2))
d = d.merge(M[["slot", "h", "q"] + [f"cb{k}" for k in range(6)]], on=["slot", "h"], how="left")
print("candidate coverage", d.q.notna().mean())
d["rn"] = d.groupby("slot").rs_blend.rank(pct=True)
for K in [3, 4, 5]:
    d["dec"] = np.log(d.q) + np.log(np.clip(d[f"cb{K - 1}"], 1e-9, 1)); print(f"pure decoder K={K}: E {Ef(d, 'dec'):.6f}")
grid = list(itertools.product([3, 4, 5, 6], [0.0, 0.02, 0.05, 0.1, 0.2, 0.4], [0.0, 0.5, 1.0]))
cache = {}
for K, beta, gq in grid:
    d["tmp"] = d.rn + beta * (np.log(np.clip(d[f"cb{min(K, 6) - 1}"], 1e-9, 1)) + gq * np.log(d.q))
    cache[(K, beta, gq)] = {f: Ef(d[d.fold == f], "tmp") for f in range(5)}
bf = {f: Ef(d[d.fold == f], "rs_blend") for f in range(5)}
nest = []
for f in range(5):
    best = max(grid, key=lambda g: np.mean([cache[g][x] - bf[x] for x in range(5) if x != f])); nest.append((f, best, round(cache[best][f] - bf[f], 5)))
print("nested:", nest, "mean", round(np.mean([t[2] for t in nest]), 6))
top = sorted(((g, np.mean([cache[g][x] - bf[x] for x in range(5)])) for g in grid), key=lambda t: -t[1])[:5]; print("optimistic top:", top)
d.to_parquet(f"{OUT}/s11_count_decoder_cands.parquet")
