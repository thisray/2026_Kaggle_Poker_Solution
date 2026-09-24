"""Two-stage evidence pipeline for evaluation pairs + risk/behavior assembly + legality checks."""
import numpy as np, pandas as pd, time, sys, json, hashlib, lightgbm as lgb, importlib
import pairindex as PI, handdesc as HD, orient as OR
from scipy.stats import poisson
OUT = __import__("os").environ["POKER_WORK_DIR"]; RAW = __import__("os").environ["POKER_DATA_DIR"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
cfg = json.loads(sys.argv[1])
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
hand_ids = pd.read_parquet(f"{OUT}/np/hand_index.parquet").sort_values("hi").hand_id.values
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
lo = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values; hi = np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
evalp["key"] = lo * 12000 + hi
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
evalp["slot"] = PI.pair_slot(loc.pool.loc[lo].values, loc.local.loc[lo].values, loc.local.loc[hi].values)
sub = evalp.merge(pd.read_parquet(cfg["risk_file"])[["key", "score"]], on="key", how="left"); assert sub.score.notna().all()
sub["risk_score"] = sub.score.rank(method="average", pct=True)
fm = pd.read_parquet(cfg["family_file"])[["key", "family"]]; sub = sub.merge(fm, on="key", how="left"); sub["predicted_behavior"] = sub.family.fillna("none")
if cfg.get("hand_cache") and __import__("os").path.exists(cfg["hand_cache"]):
    D0 = pd.read_parquet(cfg["hand_cache"]); log("loaded hand cache", len(D0))
else:
    He, Se, Te, SLe = PI.all_pair_hands(1)
    keep = np.isin(SLe, sub.slot.values); He, Se, Te, SLe = He[keep], Se[keep], Te[keep], SLe[keep]
    HF1 = importlib.import_module(cfg["s1_featmod"]); HF2m = importlib.import_module(cfg.get("s2_featmod", cfg["s1_featmod"]))
    m1 = [lgb.Booster(model_file=f) for f in cfg["s1_models"]]
    m2 = [lgb.Booster(model_file=f) for f in cfg.get("s2_models", [])]
    s1 = np.zeros(len(He), np.float32); s2 = np.zeros(len(He), np.float32)
    B = 1_000_000
    for i in range(0, len(He), B):
        hh, ss, tt = He[i:i+B], Se[i:i+B], Te[i:i+B]
        X1 = pd.concat([HF1.features(hh, ss, tt), HD.descriptors(hh, ss, tt, cfg.get("s1_probs", "dec_probs_v1.npy"))], axis=1) if cfg.get("s1_desc", True) else HF1.features(hh, ss, tt)
        s1[i:i+B] = np.mean([m.predict(X1, num_threads=16) for m in m1], axis=0)
        log("stage1", i + len(hh))
    if m2:
        # roles estimated per pair over all its evaluation hands (needs full stage-1 vector)
        for i in range(0, len(He), B):
            hh, ss, tt = He[i:i+B], Se[i:i+B], Te[i:i+B]
            X2 = pd.concat([HF2m.features(hh, ss, tt), HD.descriptors(hh, ss, tt, cfg.get("s2_probs", "dec_probs_v1.npy"))], axis=1)
            O = OR.features(SLe * 2 + 1, He, Se, Te, s1) if i == 0 else O
            X2 = pd.concat([X2, O.iloc[i:i+B].reset_index(drop=True)], axis=1)
            X2["s1_logit"] = np.log(np.clip(s1[i:i+B], 1e-6, 1 - 1e-6) / (1 - np.clip(s1[i:i+B], 1e-6, 1 - 1e-6)))
            s2[i:i+B] = np.mean([m.predict(X2, num_threads=16) for m in m2], axis=0)
            log("stage2", i + len(hh))
    D0 = pd.DataFrame({"slot": SLe, "h": He, "s1": s1, "s2": s2 if m2 else s1})
    if cfg.get("hand_cache"): D0.to_parquet(cfg["hand_cache"])
ts_all = np.load(f"{OUT}/np/h_ts.npy")
samp = pd.read_csv(f"{RAW}/sample_submission.csv", usecols=["pair_id"])
for var in cfg["variants"]:
    D = D0.copy(); D["s_raw"] = D[var.get("score_col", "s2")]
    D["ts"] = ts_all[D.h.values]; D = D.sort_values(["slot", "ts"]).reset_index(drop=True)
    mode = var.get("mode", "plt5")
    if mode == "plt5":
        D["cum"] = D.groupby("slot").s_raw.cumsum() - D.s_raw; D["s"] = D.s_raw * poisson.cdf(4, D.cum * var.get("scale", 1.0))
    elif mode == "exp":
        D["s"] = D.s_raw * np.exp(-var.get("a", 0.5) * D.groupby("slot").ts.rank(pct=True))
    else:
        D["s"] = D.s_raw
    D = D.sort_values(["slot", "s"], ascending=[True, False]); D["r"] = D.groupby("slot").cumcount()
    top = D[D.r < 5].copy(); top["hid"] = hand_ids[top.h.values]
    wide = top.pivot(index="slot", columns="r", values="hid")
    sv = sub.copy()
    for r in range(5):
        sv[f"evidence_hand_{r+1}"] = sv.slot.map(wide[r]).fillna("NO_EVIDENCE") if r in wide.columns else "NO_EVIDENCE"
    out = sv[["pair_id", "risk_score", "predicted_behavior"] + [f"evidence_hand_{i}" for i in range(1, 6)]]
    assert len(out) == len(samp) == out.pair_id.nunique() and set(out.pair_id) == set(samp.pair_id)
    assert out.risk_score.between(0, 1).all()
    cells = out[[f"evidence_hand_{i}" for i in range(1, 6)]].values
    assert all(len([x for x in row if x != "NO_EVIDENCE"]) == len(set(x for x in row if x != "NO_EVIDENCE")) for row in cells)
    out.to_csv(var["out"], index=False)
    log("wrote", var["out"], hashlib.sha256(open(var["out"], "rb").read()).hexdigest(), "no_evidence", int((cells == "NO_EVIDENCE").sum()))
