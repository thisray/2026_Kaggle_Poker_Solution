"""R3-P29: fusion functional form. AP only cares about the head of the ranking, but equal-weight z averaging treats
the whole distribution equally and lets one model's outlier move a pair. Alternatives tested under the same protocol:
robust centre (median / trimmed mean of z), top-heavy transforms (-log(1-pct), pct^k), and vote counts (how many
models put the pair in their own top-K)."""
import numpy as np, pandas as pd, glob, os, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/m*_train_oof.parquet"))]
names = [n for n in names if os.path.exists(f"{O}/{n}_eval_scores.parquet")]
out = {}
V = {}
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); M = base[base.src == src][["key", "pool", "y", "hid"]].set_index("key")
    for n in names:
        d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": n}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    sh = [n for n in names if M[n].notna().mean() > 0.9]
    y = M.y.values.astype(int); hid = M.hid.astype(bool).values; msk = ~hid | (y == 1); pools = M.pool.values
    Z = np.stack([((M[n] - M[n].mean()) / M[n].std()).values for n in sh])
    P = np.stack([M[n].rank(pct=True).values for n in sh])
    N = len(y)
    f = {}
    f["mean_z"] = np.nanmean(Z, 0)
    f["median_z"] = np.nanmedian(Z, 0)
    f["trimmed_z"] = np.nanmean(np.sort(Z, 0)[len(sh)//10: len(sh) - len(sh)//10], 0)
    f["mean_pct"] = np.nanmean(P, 0)
    f["neglog_tail"] = np.nanmean(-np.log(np.clip(1 - P + 1.0 / N, 1e-9, 1)), 0)
    for k in (8, 32):
        f[f"pct_pow{k}"] = np.nanmean(P ** k, 0)
    for K in (300, 600, 1200):
        f[f"votes_top{K}"] = np.nanmean((P >= 1 - K / N).astype(float), 0) + 1e-9 * f["mean_z"]
    f["mean_z_plus_votes"] = (f["mean_z"] - f["mean_z"].mean()) / f["mean_z"].std() + 2.0 * f["votes_top600"]
    out[src] = {k: round(ap(y[msk], v[msk]), 5) for k, v in f.items()}
    V[src] = (y, msk, pools, f)
    print(src, json.dumps(out[src]), flush=True)
mean = {k: np.mean([out[s][k] for s in out]) for k in out["devsub11"]}
rank = sorted(mean.items(), key=lambda kv: -kv[1])
print("\nby mean of the two subsamples:")
for k, v in rank: print(f"   {k:18s} {v:.5f}  ({v - mean['mean_z']:+.5f} vs mean_z)")
bestk = rank[0][0]
if bestk != "mean_z":
    for src in ("devsub11", "devsub12"):
        y, msk, pools, f = V[src]; up = np.unique(pools); idx = {p: np.flatnonzero((pools == p) & msk) for p in up}
        rs = np.random.default_rng(7); d = []
        for _ in range(300):
            ii = np.concatenate([idx[p] for p in rs.choice(up, len(up))]); d.append(ap(y[ii], f[bestk][ii]) - ap(y[ii], f["mean_z"][ii]))
        print(f"  bootstrap {src}: {bestk} vs mean_z {np.mean(d):+.5f} P {np.mean(np.array(d) > 0):.3f}")
json.dump(out, open(f"{O}/r3/t97_fusion_form.json", "w"), indent=1)
