"""R3-P28: weight scan for the grouped fusion (1-W)*all_z(shared models) + W*aug_z(augmented models).

R4 established the augmentation gain with 5 LightGBM-gbdt configs and chose W=0.7. This re-runs the same protocol with
whatever augmented models exist, so that the extra learner families (dart / extra_trees / feature view / PU=none /
deep / m9 MIL) can be included. Scored with AP_clean on both exposure-matched dev subsamples, with a paired pool
bootstrap of the best W against the deployed three-model rank average and against R4's W=0.7 over its own five models.
Usage: AUG=tag1,tag2,... python t96_wscan.py
"""
import numpy as np, pandas as pd, glob, os, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
AUG = [t for t in os.environ.get("AUG", "").split(",") if t]
R4FIVE = ["q2_pos_a5", "q2_drop_d5", "q2_pos_h5", "q2_pos_l5", "q2_pos_i5"]
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
names = [os.path.basename(f).replace("_train_oof.parquet", "") for f in sorted(glob.glob(f"{O}/m*_train_oof.parquet"))]
names = [n for n in names if os.path.exists(f"{O}/{n}_eval_scores.parquet")]
WS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 1.0]
res = {}
for src in ("devsub11", "devsub12"):
    base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet"); M = base[base.src == src][["key", "pool", "y", "hid"]].set_index("key")
    for n in names:
        d = pd.read_parquet(f"{O}/{n}_train_oof.parquet"); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": n}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    got = []
    for t in AUG:
        p = f"{O}/r4/m15_{t}_train_oof.parquet"
        if not os.path.exists(p): continue
        d = pd.read_parquet(p); d = d[d.src == src][["key", "oof"]].rename(columns={"oof": "A_" + t}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left"); got.append(t)
    shared = [n for n in names if M[n].notna().mean() > 0.9]
    y = M.y.values.astype(int); hid = M.hid.astype(bool).values; msk = ~hid | (y == 1); pools = M.pool.values
    def z(cols): return np.nanmean(np.stack([((M[c] - M[c].mean()) / M[c].std()).values for c in cols]), 0)
    allz = z(shared)
    augz = z(["A_" + t for t in got if M["A_" + t].notna().mean() > 0.9])
    r4z = z(["A_" + t for t in R4FIVE if "A_" + t in M and M["A_" + t].notna().mean() > 0.9])
    dep = 0.5 * M.m15_v6ens_base.rank(pct=True).values + 0.25 * M.m15_v6ens_cat.rank(pct=True).values + 0.25 * M.m15_v6ens_cat11.rank(pct=True).values
    row = {"shared_models": len(shared), "aug_models": len(got), "aug": got,
           "deployed": round(ap(y[msk], dep[msk]), 5), "r4_w07": round(ap(y[msk], (0.3 * ((allz - allz.mean()) / allz.std()) + 0.7 * ((r4z - r4z.mean()) / r4z.std()))[msk]), 5)}
    zs = lambda v: (v - v.mean()) / v.std()
    for w in WS: row[f"W{w}"] = round(ap(y[msk], ((1 - w) * zs(allz) + w * zs(augz))[msk]), 5)
    res[src] = row
    print(src, json.dumps({k: v for k, v in row.items() if k != "aug"}), flush=True)
    res[src + "_vec"] = dict(y=y, msk=msk, pools=pools, allz=allz, augz=augz, dep=dep, r4z=r4z)
best = max(WS, key=lambda w: np.mean([res[s][f"W{w}"] for s in ("devsub11", "devsub12")]))
print(f"\nbest W = {best} (mean {np.mean([res[s][f'W{best}'] for s in ('devsub11','devsub12')]):.5f}; "
      f"R4 W=0.7 on its five {np.mean([res[s]['r4_w07'] for s in ('devsub11','devsub12')]):.5f}; "
      f"deployed {np.mean([res[s]['deployed'] for s in ('devsub11','devsub12')]):.5f})")
for src in ("devsub11", "devsub12"):
    v = res[src + "_vec"]; y, msk, pools = v["y"], v["msk"], v["pools"]; up = np.unique(pools)
    zs = lambda a: (a - a.mean()) / a.std()
    cur = (1 - best) * zs(v["allz"]) + best * zs(v["augz"]); r4 = 0.3 * zs(v["allz"]) + 0.7 * zs(v["r4z"])
    rs = np.random.default_rng(11); d1 = []; d2 = []
    idx_by = {p: np.flatnonzero((pools == p) & msk) for p in up}
    for _ in range(300):
        ii = np.concatenate([idx_by[p] for p in rs.choice(up, len(up))])
        d1.append(ap(y[ii], cur[ii]) - ap(y[ii], v["dep"][ii])); d2.append(ap(y[ii], cur[ii]) - ap(y[ii], r4[ii]))
    print(f"  {src}: vs deployed {np.mean(d1):+.5f} P {np.mean(np.array(d1) > 0):.3f} | vs R4 W=0.7 {np.mean(d2):+.5f} P {np.mean(np.array(d2) > 0):.3f}")
json.dump({k: v for k, v in res.items() if not k.endswith("_vec")}, open(f"{O}/r3/t96_wscan.json", "w"), indent=1)
