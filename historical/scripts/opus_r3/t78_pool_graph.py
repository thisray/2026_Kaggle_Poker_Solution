"""R3-P16: two structural priors on the pair ranking that no round has tested.

(1) POOL CENTRING. Every positive pair is inside one pool and the per-pool count of labelled positives is
    Poisson-like (155/63/18/8/1 pools with 1/2/3/4/5; 155 with none; lambda ~ 0.93). A constant expected count per
    pool means a pool-level shift of the score is NOISE, so centring the score inside its pool should help.
(2) PLAYER GRAPH. 47 players sit in two labelled positive pairs and 2 in three. If a member's OTHER pairs score high,
    that is transductive evidence for this pair. Measured as a lift first, then as a rank blend.

Everything is scored with the same clean AP (hidden positives dropped) on the two exposure-matched dev subsamples,
with a pool bootstrap for the paired difference.
"""
import numpy as np, pandas as pd, glob, os, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/m*_train_oof.parquet"))]
names = [n for n in names if os.path.exists(f"{O}/{n}_eval_scores.parquet")]
K = {}
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); M = base[base.src == src][["key", "pool", "y", "hid"]].set_index("key")
    for n in names:
        d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": n}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    K[src] = M
avail = [n for n in names if all(n in K[s] and K[s][n].notna().mean() > 0.9 for s in K)]
print(f"models {len(avail)}", flush=True)
def fuse(M, mods):
    V = [((M[m] - M[m].mean()) / M[m].std()).values for m in mods]
    return np.nanmean(np.stack(V), 0)
res = {}
for src, M in K.items():
    y = M.y.values.astype(int); hid = M.hid.astype(bool).values; pool = M.pool.values
    keys = M.index.values.astype(np.int64); plo = keys // 12000; phi = keys % 12000
    f = fuse(M, avail); msk = ~hid | (y == 1)
    d = pd.DataFrame(dict(f=f, pool=pool, plo=plo, phi=phi))
    # (1) pool centring, three strengths
    g = d.groupby("pool").f
    mu = g.transform("mean").values; sd = g.transform("std").replace(0, np.nan).fillna(1.0).values
    cen = f - mu; z = (f - mu) / sd
    half = f - 0.5 * mu
    # (2) player graph: best OTHER pair score of each member (leave-one-out max over pairs sharing a player)
    s = pd.Series(f)
    best = {}
    for col in ("plo", "phi"):
        for p, gg in d.groupby(col):
            v = np.sort(gg.f.values)[::-1]
            top1, top2 = v[0], (v[1] if len(v) > 1 else -np.inf)
            for i, fv in zip(gg.index, gg.f.values):
                o = top2 if fv >= top1 else top1
                best[(col, i)] = o
    bl = np.array([best[("plo", i)] for i in d.index]); bh = np.array([best[("phi", i)] for i in d.index])
    nb = np.maximum(bl, bh)                                   # best other pair containing either member
    rf = pd.Series(f).rank(pct=True).values; rn = pd.Series(nb).rank(pct=True).values
    out = dict(base=ap(y[msk], f[msk]), pool_centre=ap(y[msk], cen[msk]), pool_z=ap(y[msk], z[msk]),
               pool_half=ap(y[msk], half[msk]), graph_10=ap(y[msk], (rf + 0.10 * rn)[msk]),
               graph_05=ap(y[msk], (rf + 0.05 * rn)[msk]), graph_20=ap(y[msk], (rf + 0.20 * rn)[msk]))
    # graph lift: P(positive | a member is in another LABELLED positive pair)
    posmask = y == 1
    inpos = set(plo[posmask]) | set(phi[posmask])
    share = np.array([ (a in inpos) or (b in inpos) for a, b in zip(plo, phi) ])
    # exclude the pair's own positive status by removing its own contribution
    sh2 = np.zeros(len(d), bool)
    cnt = pd.Series(np.concatenate([plo[posmask], phi[posmask]])).value_counts()
    for i, (a, b) in enumerate(zip(plo, phi)):
        ca = cnt.get(a, 0) - (1 if posmask[i] else 0); cb = cnt.get(b, 0) - (1 if posmask[i] else 0)
        sh2[i] = (ca > 0) or (cb > 0)
    base_rate = float(posmask[msk].mean()); cond = float(posmask[msk & sh2].mean()) if (msk & sh2).sum() else float("nan")
    out["graph_lift"] = dict(base_rate=round(base_rate, 5), cond_rate=round(cond, 5),
                             lift=round(cond / base_rate, 3) if base_rate else None, n_cond=int((msk & sh2).sum()))
    # bootstrap the best pool variant
    up = np.unique(pool); rs = np.random.default_rng(7); dd = []
    bestv = max(("pool_centre", "pool_z", "pool_half"), key=lambda k: out[k]); bv = dict(pool_centre=cen, pool_z=z, pool_half=half)[bestv]
    for _ in range(200):
        pick = rs.choice(up, len(up), replace=True); idx = np.concatenate([np.flatnonzero((pool == p) & msk) for p in pick])
        dd.append(ap(y[idx], bv[idx]) - ap(y[idx], f[idx]))
    out["boot_" + bestv] = dict(mean=round(float(np.mean(dd)), 5), p_gt0=round(float(np.mean(np.array(dd) > 0)), 3))
    res[src] = out
    print(src, json.dumps(out), flush=True)
json.dump(res, open(f"{O}/r3/t78_pool_graph.json", "w"), indent=1)
