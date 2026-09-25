"""R3-P10: deployed pair ranking (rank average of v6ens_base + v6ens_cat + v6ens_cat11, as built by c4_build_r2j)
versus parameter-free fusions over every available pair model. Dev OOF, official AP with hidden positives dropped,
both exposure-matched subsamples, plus a pool-bootstrap of the difference."""
import numpy as np, pandas as pd, glob, os, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
files = sorted(glob.glob(f"{O}/m*_train_oof.parquet"))
res = {}
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); b = base[base.src == src][["key", "pool", "y", "hid", "fam"]].set_index("key")
    M = b.copy()
    for f in files:
        nm = os.path.basename(f).replace("_train_oof.parquet", "")
        d = pd.read_parquet(f)
        if "src" not in d or "oof" not in d: continue
        d = d[d.src == src][["key", "oof"]].rename(columns={"oof": nm}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    mods = [c for c in M.columns if c not in ("pool", "y", "hid", "fam") and M[c].notna().mean() > 0.9]
    K = M[~M.hid.astype(bool) | (M.y == 1)].copy(); y = K.y.values.astype(int)
    z = lambda c: (K[c] - K[c].mean()) / K[c].std()
    rk = lambda c: K[c].rank(pct=True)
    DEP = [c for c in ("m15_v6ens_base", "m15_v6ens_cat", "m15_v6ens_cat11") if c in mods]
    dep = 0.5 * rk(DEP[0]) + sum(0.5 / max(len(DEP) - 1, 1) * rk(c) for c in DEP[1:])
    good = [m for m in mods if ap(y, K[m].fillna(K[m].min()).values) > 0.96]
    cand = {"deployed_3model_rank": dep.values,
            "mean_z_all": np.nanmean(np.stack([z(c) for c in mods]), 0),
            "mean_z_good": np.nanmean(np.stack([z(c) for c in good]), 0),
            "mean_rankpct_good": np.nanmean(np.stack([rk(c) for c in good]), 0),
            "mean_logit_good": np.nanmean(np.stack([np.log(np.clip(K[c], 1e-9, 1 - 1e-9) / (1 - np.clip(K[c], 1e-9, 1 - 1e-9))) for c in good]), 0)}
    aps = {k: round(ap(y, v), 5) for k, v in cand.items()}
    print(f"== {src}: models {len(mods)}, good {len(good)}, positives {int(y.sum())}"); print("   ", json.dumps(aps))
    # pool bootstrap of the best fusion minus deployed
    best = max((k for k in aps if k != "deployed_3model_rank"), key=lambda k: aps[k])
    pools = K.pool.values; up = np.unique(pools); rs = np.random.default_rng(7); d = []
    for _ in range(200):
        pick = rs.choice(up, len(up), replace=True); idx = np.concatenate([np.flatnonzero(pools == p) for p in pick])
        d.append(ap(y[idx], cand[best][idx]) - ap(y[idx], cand["deployed_3model_rank"][idx]))
    print(f"    best fusion {best}: delta {aps[best] - aps['deployed_3model_rank']:+.5f}; pool bootstrap mean {np.mean(d):+.5f}, "
          f"P(>0) {np.mean(np.array(d) > 0):.3f}, 5-95% [{np.percentile(d, 5):+.5f},{np.percentile(d, 95):+.5f}]")
    res[src] = dict(aps=aps, best=best, boot_mean=float(np.mean(d)), boot_p=float(np.mean(np.array(d) > 0)), models=mods, good=good)
json.dump(res, open(f"{O}/r3/t68_fusion_vs_deployed.json", "w"), indent=1)
