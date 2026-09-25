"""Score all pair-hands with fold-safe hand detectors and aggregate to pair-level MIL features per hand mask."""
import numpy as np, pandas as pd, time, lightgbm as lgb
import handfeat2 as HF2, pairindex as PI, aggmod as AG, handdesc as HD
OUT = HF2.OUT
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
models = [lgb.Booster(model_file=f"{OUT}/m19w10_hand_f{f}.txt") for f in range(5)]
_hs = pd.read_parquet(f"{OUT}/m19w10_handscores.parquet"); _neg = _hs[(_hs.phase == 0) & (~_hs.pos)].s.values; q99, q999 = np.quantile(_neg, [0.99, 0.999]); log("thresholds", q99, q999)
res = {}
for phase in [0, 1]:
    H, S, T, SL = PI.all_pair_hands(phase)
    sc = np.zeros(len(H), np.float32)
    B = 1_500_000
    for i in range(0, len(H), B):
        X = pd.concat([HF2.features(H[i:i+B], S[i:i+B], T[i:i+B]), HD.descriptors(H[i:i+B], S[i:i+B], T[i:i+B], 'dec_probs_v1.npy')], axis=1)
        if phase == 0:
            fo = fold_of_pool[SL[i:i+B] // 900]
            out = np.zeros(len(X), np.float32)
            for f in range(5):
                m = fo == f
                if m.any(): out[m] = models[f].predict(X[m], num_threads=16)
            sc[i:i+B] = out
        else:
            sc[i:i+B] = np.mean([m.predict(X, num_threads=16) for m in models], axis=0)
        log("phase", phase, "scored", i + len(X))
    res[phase] = (H, SL, sc)
    np.save(f"{OUT}/m26_handscore_phase{phase}.npy", sc)
    np.save(f"{OUT}/m26_slot_phase{phase}.npy", SL); np.save(f"{OUT}/m26_h_phase{phase}.npy", H)
log("scored all")
def aggregate(H, SL, sc, handmask, name):
    keep = handmask[H] == 1
    d = pd.DataFrame({"slot": SL[keep], "s": sc[keep]})
    d["hi99"] = (d.s > q99).astype(np.float32); d["hi999"] = (d.s > q999).astype(np.float32)
    d = d.sort_values(["slot", "s"], ascending=[True, False])
    d["r"] = d.groupby("slot").cumcount()
    g = d.groupby("slot")
    agg = pd.DataFrame({"mil_sum": g.s.sum(), "mil_mean": g.s.mean(), "mil_cnt99": g.hi99.sum(), "mil_cnt999": g.hi999.sum(), "mil_max": g.s.max(),
                        "mil_top3": d[d.r < 3].groupby("slot").s.mean(), "mil_top5": d[d.r < 5].groupby("slot").s.mean(), "mil_n": g.s.size()})
    agg["mil_rate999"] = agg.mil_cnt999 / agg.mil_n; agg["mil_rate99"] = agg.mil_cnt99 / agg.mil_n
    agg = agg.reset_index(); agg.to_parquet(f"{OUT}/m26_mil_{name}.parquet"); log("agg", name, agg.shape)
H0, SL0, sc0 = res[0]; H1, SL1, sc1 = res[1]
aggregate(H0, SL0, sc0, AG.hand_mask("dev"), "dev")
aggregate(H0, SL0, sc0, AG.hand_mask("devsub", 11), "devsub11")
aggregate(H0, SL0, sc0, AG.hand_mask("devsub", 12), "devsub12")
aggregate(H1, SL1, sc1, AG.hand_mask("eval"), "eval")
