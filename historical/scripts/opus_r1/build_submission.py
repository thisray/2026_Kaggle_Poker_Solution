"""Assemble a legal submission: risk from pair model, behavior from family model, evidence from hand model over eval-phase shared hands."""
import numpy as np, pandas as pd, time, sys, json, hashlib, lightgbm as lgb
import handfeat2 as HF2, pairindex as PI
OUT = HF2.OUT; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
cfg = json.loads(sys.argv[1])
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet").sort_values("hi"); hand_ids = hidx.hand_id.values
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
evalp["key"] = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)) * 12000 + np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap))
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
lo = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values; hi = np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
evalp["slot"] = PI.pair_slot(loc.pool.loc[lo].values, loc.local.loc[lo].values, loc.local.loc[hi].values)
# risk
rs = pd.read_parquet(cfg["risk_file"])[["key", "score"]]
sub = evalp.merge(rs, on="key", how="left")
assert sub.score.notna().all(), "missing risk"
sub["risk_score"] = sub.score.rank(method="average", pct=True).clip(0, 1)
# behavior
if cfg.get("family_file"):
    fm = pd.read_parquet(cfg["family_file"])[["key", "family"]]
    sub = sub.merge(fm, on="key", how="left"); sub["predicted_behavior"] = sub.family.fillna("none")
else:
    sub["predicted_behavior"] = "none"
# evidence
He, Se, Te, SLe = PI.all_pair_hands(1)
keep = np.isin(SLe, sub.slot.values)
He, Se, Te, SLe = He[keep], Se[keep], Te[keep], SLe[keep]
log("eval pair-hands for submission", len(He))
models = [lgb.Booster(model_file=f) for f in cfg["hand_models"]]
scores = np.zeros(len(He), np.float32)
B = 1_500_000
for i in range(0, len(He), B):
    X = HF2.features(He[i:i+B], Se[i:i+B], Te[i:i+B])
    scores[i:i+B] = np.mean([m.predict(X, num_threads=18) for m in models], axis=0)
    log("scored", i + len(X))
D0 = pd.DataFrame({"slot": SLe, "h": He, "s": scores})
D0.to_parquet(cfg["score_cache"]) if cfg.get("score_cache") else None
from scipy.stats import poisson
ts_all = np.load(f"{OUT}/np/h_ts.npy")
samp = pd.read_csv(f"{RAW}/sample_submission.csv", usecols=["pair_id"])
for var in cfg["variants"]:
    D = D0.copy(); D["s_raw"] = D.s
    if var.get("selgen"):
        from selmodel import pair_loglik_and_post, calib
        prm = var["selgen"]
        D["ts"] = ts_all[D.h.values]
        D = D.sort_values(["slot", "ts"]).reset_index(drop=True)
        q = calib(D.s_raw.values, prm["a"], prm["b"], prm["p"])
        starts = np.r_[0, np.flatnonzero(np.diff(D.slot.values)) + 1, len(D)]
        post = np.zeros(len(D))
        dummy = np.zeros(0, np.int64)
        for i in range(len(starts) - 1):
            a0, a1 = starts[i], starts[i + 1]
            _, pp = pair_loglik_and_post(q[a0:a1], np.zeros(a1 - a0, np.int64), 5)
            post[a0:a1] = pp
        D["s"] = post
    if var.get("plt5", False):
        D["ts"] = ts_all[D.h.values]
        D = D.sort_values(["slot", "ts"]).reset_index(drop=True)
        D["cum_before"] = D.groupby("slot").s_raw.cumsum() - D.s_raw
        D["s"] = D.s_raw * poisson.cdf(4, D.cum_before * var.get("plt5_scale", 1.0))
    D = D.sort_values(["slot", "s"], ascending=[True, False])
    D["r"] = D.groupby("slot").cumcount()
    top = D[D.r < 5].copy(); top["hid"] = hand_ids[top.h.values]
    wide = top.pivot(index="slot", columns="r", values="hid")
    sv = sub.copy()
    for r in range(5):
        sv[f"evidence_hand_{r+1}"] = sv.slot.map(wide[r]).fillna("NO_EVIDENCE") if r in wide.columns else "NO_EVIDENCE"
    cols = ["pair_id", "risk_score", "predicted_behavior"] + [f"evidence_hand_{i}" for i in range(1, 6)]
    out = sv[cols]
    assert len(out) == len(samp) == out.pair_id.nunique() and set(out.pair_id) == set(samp.pair_id)
    assert out.risk_score.between(0, 1).all() and np.isfinite(out.risk_score).all()
    ev_cells = out[[f"evidence_hand_{i}" for i in range(1, 6)]].values
    dup = sum(len([x for x in row if x != "NO_EVIDENCE"]) != len(set(x for x in row if x != "NO_EVIDENCE")) for row in ev_cells)
    assert dup == 0
    assert set(out.predicted_behavior.unique()) <= {"none", "directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"}
    path = var["out"]; out.to_csv(path, index=False)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    log("wrote", path, "sha256", sha, "no_evidence cells", int((ev_cells == "NO_EVIDENCE").sum()), "behavior", out.predicted_behavior.value_counts().to_dict())
