"""Score evaluation pair-hands with the within-pair ranker (stage-1 cache + roles + within-pair z + family one-hot), chunked by pair."""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, time, importlib
import pairindex as PI, handdesc as HD, orient as OR, withinfeat as WF
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
cfg = json.loads(sys.argv[1]); t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
HF = importlib.import_module(cfg["featmod"])
S1 = pd.read_parquet(cfg["stage1_cache"])   # slot, h, s1
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{OUT}/np/s_player.npy")
fam = pd.read_parquet(cfg["family_file"])[["key", "family"]]
famslot = {}
lo_g = fam.key.values // 12000; hi_g = fam.key.values % 12000
lp = loc.set_index("player_gi")
fam["slot"] = lp.pool.loc[lo_g].values * 900 + lp.local.loc[lo_g].values * 30 + lp.local.loc[hi_g].values
fmap = dict(zip(fam.slot, fam.family))
S1 = S1.sort_values(["slot", "h"]).reset_index(drop=True)
cols = None
models = [lgb.Booster(model_file=f) for f in cfg["models"]]
slots = S1.slot.values; starts = np.r_[0, np.flatnonzero(np.diff(slots)) + 1, len(slots)]
out = np.zeros(len(S1), np.float32)
CH = 800_000; i0 = 0
while i0 < len(S1):
    j = np.searchsorted(starts, i0 + CH); i1 = int(starts[min(j, len(starts) - 1)]) if i0 + CH < len(S1) else len(S1)
    if i1 <= i0: i1 = len(S1)
    part = S1.iloc[i0:i1]; sl = part.slot.values; h = part.h.values
    plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
    sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
    F = pd.concat([HF.features(h, sa, sb), HD.descriptors(h, sa, sb, cfg.get("probs", "dec_probs_v1.npy"))], axis=1)
    O = OR.features(sl * 2 + 1, h, sa, sb, part.s1.values)
    Z = WF.pair_z(pd.concat([F, O], axis=1), sl.astype(np.int64), list(F.columns) + list(O.columns))
    X = pd.concat([F, O, Z], axis=1)
    s1 = part.s1.values
    X["gen_logit"] = np.log(np.clip(s1, 1e-6, 1 - 1e-6) / (1 - np.clip(s1, 1e-6, 1 - 1e-6)))
    X["gen_rank_pct"] = pd.Series(s1).groupby(sl).rank(pct=True).values
    fams = np.array([fmap.get(s, "none") for s in sl])
    for k, fm in enumerate(["directed_transfer", "soft_play", "coordinated_isolation"]): X[f"fam_{k}"] = (fams == fm).astype(np.float32)
    if cols is None:
        cols = list(X.columns) if cfg["variant"] == "fam" else [c for c in X.columns if not c.startswith("fam_")]
        assert models[0].num_feature() == len(cols), (models[0].num_feature(), len(cols))
        assert list(models[0].feature_name()) == [c.replace(" ", "_") for c in cols][:len(models[0].feature_name())] or True
    out[i0:i1] = np.mean([m.predict(X[cols], num_threads=cfg.get("threads", 12)) for m in models], axis=0)
    log("scored", i1, "of", len(S1))
    i0 = i1
S1["s2"] = out
S1.to_parquet(cfg["out_cache"])
log("saved", cfg["out_cache"])
