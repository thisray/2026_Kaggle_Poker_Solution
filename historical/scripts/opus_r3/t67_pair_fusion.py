"""R3-P9: is the deployed pair ranking losing weak positives that some component model does find?
Dev OOF of every available pair model, official AP (hidden positives dropped), per-model ranks of the positives the
ensemble puts beyond rank 400, and parameter-free fusions (mean rank, median rank, best rank, logit mean)."""
import numpy as np, pandas as pd, glob, json, os
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); r = y[o]; tp = np.cumsum(r); k = np.arange(1, len(r) + 1)
    return float(np.sum(tp / k * r) / max(int(y.sum()), 1))
files = sorted(glob.glob(f"{O}/m*_train_oof.parquet"))
base = pd.read_parquet(f"{O}/m15_v6ens_base_train_oof.parquet")
out = {}
for src in ("devsub11", "devsub12"):
    b = base[base.src == src][["key", "y", "hid", "fam", "n", "oof"]].rename(columns={"oof": "m15_v6ens_base"})
    M = b.set_index("key")
    for f in files:
        nm = os.path.basename(f).replace("_train_oof.parquet", "")
        if nm == "m15_v6ens_base": continue
        d = pd.read_parquet(f)
        if "src" not in d or "oof" not in d: continue
        d = d[d.src == src][["key", "oof"]].rename(columns={"oof": nm}).set_index("key")
        if len(d) and d.index.is_unique: M = M.join(d, how="left")
    mods = [c for c in M.columns if c not in ("y", "hid", "fam", "n")]
    keep = ~M.hid.astype(bool) | (M.y == 1); K = M[keep]
    y = K.y.values.astype(int)
    aps = {m: round(ap(y, K[m].fillna(K[m].min()).values), 5) for m in mods if K[m].notna().mean() > 0.9}
    mods = [m for m in aps]
    R = pd.DataFrame({m: K[m].rank(ascending=False, method="first") for m in mods})
    fus = {"mean_rank": -R.mean(1).values, "median_rank": -R.median(1).values, "best_rank": -R.min(1).values,
           "mean_z": np.nanmean(np.stack([(K[m] - K[m].mean()) / K[m].std() for m in mods]), 0),
           "mean_logit": np.nanmean(np.stack([np.log(np.clip(K[m], 1e-9, 1 - 1e-9) / (1 - np.clip(K[m], 1e-9, 1 - 1e-9))) for m in mods]), 0)}
    faps = {k: round(ap(y, v), 5) for k, v in fus.items()}
    out[src] = dict(models=aps, fusions=faps, n_models=len(mods))
    print(f"== {src}: {len(mods)} models, positives {int(y.sum())}")
    print("   per model AP:", json.dumps(dict(sorted(aps.items(), key=lambda kv: -kv[1]))))
    print("   fusions:", json.dumps(faps))
    ens_rank = K["m15_v6ens_base"].rank(ascending=False, method="first")
    missed = K[(K.y == 1) & (ens_rank > 400)]
    if len(missed):
        rr = pd.DataFrame({m: K[m].rank(ascending=False, method="first") for m in mods}).loc[missed.index]
        print(f"   positives beyond ensemble rank 400: {len(missed)}; best rank across models:")
        print(pd.DataFrame({"fam": missed.fam, "n": missed.n, "ens_rank": ens_rank.loc[missed.index].astype(int),
                            "best_model_rank": rr.min(1).astype(int), "best_model": rr.idxmin(1)}).to_string())
json.dump(out, open(f"{O}/r3/t67_pair_fusion.json", "w"), indent=1)
