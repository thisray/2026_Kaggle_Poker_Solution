"""(A) Is the collusion mechanism pair-specific? Overdispersion of per-pair evidence-hand channel rates.
(B) Pseudo-relevance feedback: re-rank candidates by similarity to the pair's most confident hands (nested)."""
import numpy as np, pandas as pd, itertools
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"; D_ = f"{OUT}/np"
RN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[0][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
IND = ["facing", "fold_to", "call_to", "raise_over", "aggr_active", "passive_active", "squeeze", "iso_ofold", "hu_streets", "hu_noaggr", "check_hu", "bet_hu", "fold_hu_to"]
ci = [RN.index(c) for c in IND]
def rel(sl, h):
    plo = mem[sl // 900, (sl % 900) // 30]; phi = mem[sl // 900, sl % 30]
    sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
    ab = np.stack([np.asarray(R[h, sa, sb, c]) for c in ci], 1) > 0; ba = np.stack([np.asarray(R[h, sb, sa, c]) for c in ci], 1) > 0
    return ab, ba
# ---------- (A) overdispersion on evidence hands, orientation canonicalised per pair by majority 'facing' direction
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet"); E_ = M[M.ev].reset_index(drop=True)
ab, ba = rel(E_.sl.values, E_.h.values)
fdir = pd.Series(ab[:, 0].astype(int) - ba[:, 0].astype(int)).groupby(E_.sl.values).transform("sum").values >= 0
x = np.where(fdir[:, None], ab, ba); y = np.where(fdir[:, None], ba, ab)   # x: canonical 'A faces B' side
V = np.c_[x, y].astype(float); names = [f"x_{c}" for c in IND] + [f"y_{c}" for c in IND]
print("ev hands", len(E_), "pairs", E_.sl.nunique())
for fm in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    m = (E_.fam == fm).values; g = pd.DataFrame(V[m], columns=names); g["sl"] = E_.sl.values[m]
    agg = g.groupby("sl"); n = agg.size().values; out = []
    for c in names:
        k = agg[c].sum().values; p = k.sum() / n.sum()
        if p < 0.03 or p > 0.97: continue
        chi = (((k - n * p) ** 2) / (n * p * (1 - p))).sum(); dfree = len(n) - 1
        out.append((c, round(p, 3), round(chi / dfree, 2)))
    out.sort(key=lambda t: -t[2]); print(fm[:2], "dispersion index (1=homogeneous):", out[:10])
# ---------- (B) PRF on r11 OOF candidate lists
d = pd.read_parquet(f"{A}/round11_scoped/dev_oof_aligned.parquet")
hidx = pd.read_parquet(f"{D_}/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi)); d["h"] = d.hand_id.map(hmap).astype(np.int64)
ab2, ba2 = rel(d.slot.values, d.h.values)
d["r"] = d.groupby("slot").rs_blend.rank(ascending=False, method="first")
d = d.sort_values(["slot", "r"]).reset_index(drop=True); ab2, ba2 = None, None
ab2, ba2 = rel(d.slot.values, d.h.values)
F = np.c_[ab2, ba2].astype(float)
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
def Ef(df, col): return float(np.mean([ap5(g, col) for _, g in df.groupby("slot")]))
slots = d.slot.values; starts = np.r_[0, np.flatnonzero(slots[1:] != slots[:-1]) + 1, len(d)]
def prf_sim(k):
    sim = np.zeros(len(d))
    for a, b in zip(starts[:-1], starts[1:]):
        Fg = F[a:b]; top = Fg[:k]
        for i in range(b - a):
            proto = (top.sum(0) - (Fg[i] if i < k else 0)) / (k - (1 if i < k else 0))
            den = np.linalg.norm(Fg[i]) * np.linalg.norm(proto)
            sim[a + i] = Fg[i] @ proto / den if den > 0 else 0
    return sim
d["base_rn"] = 1 - (d.r - 1) / 19
print("base E", round(Ef(d, "rs_blend"), 6))
sims = {k: prf_sim(k) for k in [1, 2, 3]}
grid = list(itertools.product([1, 2, 3], [0.02, 0.05, 0.1, 0.2, 0.4]))
cache = {}
for k, w in grid:
    d["tmp"] = d.base_rn + w * sims[k]
    cache[(k, w)] = {f: Ef(d[d.fold == f], "tmp") for f in range(5)}
base_f = {f: Ef(d[d.fold == f], "rs_blend") for f in range(5)}
nested = []
for f in range(5):
    tr = [g for g in range(5) if g != f]
    best = max(grid, key=lambda kw: np.mean([cache[kw][g] - base_f[g] for g in tr]))
    nested.append((f, best, cache[best][f] - base_f[f]))
print("nested PRF deltas:", nested, " mean", round(np.mean([t[2] for t in nested]), 6))
print("full-grid (optimistic) best:", max(((kw, np.mean([cache[kw][f] - base_f[f] for f in range(5)])) for kw in grid), key=lambda t: t[1]))
