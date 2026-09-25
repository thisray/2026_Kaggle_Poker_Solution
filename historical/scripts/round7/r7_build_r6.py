"""Round-7: build r6 candidate = r5b + 11-score CatBoost residual (scale 0.25, 3 seeds)."""
import hashlib
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
FEATURES = ['sc_r5', 'u0', 'u_rr', 'u_r5b', 'lin_contrib', 'nn_contrib', 't1_score',
            's1_stage1', 'gen_logit', 'gen_rank_pct', 'rank_u_r5b']
SCALE = 0.25
SEEDS = [71, 72, 73]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# ---- train residual on the dev narrow table (all pairs)
d = pd.read_csv(f"{DST}/r6_narrow_candidates.csv").sort_values(["slot", "rank_u_r5b"]).reset_index(drop=True)
x = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
y = d.ev.to_numpy(int)
b = d.u_r5b.to_numpy(float)
m = d.m_p.to_numpy(float)
cal = LogisticRegression(C=10, max_iter=1000).fit(b[:, None], y)
slope, intercept = float(cal.coef_[0, 0]), float(cal.intercept_[0])
models = []
for seed in SEEDS:
    model = CatBoostClassifier(iterations=250, depth=3, learning_rate=.035, l2_leaf_reg=30,
                               thread_count=12, random_seed=seed, verbose=False, allow_writing_files=False)
    model.fit(Pool(x, y, baseline=slope * b + intercept, weight=1 / m))
    models.append(model)
print("residual trained; slope", round(slope, 4), "intercept", round(intercept, 4))

# ---- dev sanity: in-sample correction magnitude
dev_raw = np.mean([mm.predict(x, prediction_type='RawFormulaVal') / slope for mm in models], axis=0)
print("dev correction percentiles", np.round(np.quantile(0.25 * dev_raw, [0.01, 0.5, 0.99]), 4))
dev_corr = np.abs(0.25 * dev_raw)
print("dev |correction| mean", round(float(dev_corr.mean()), 4))

# ---- eval candidates
c = pd.read_parquet(f"{DST}/r5_candidates.parquet")
nn = pd.read_parquet(f"{DST}/r5cand_nn.parquet")[["slot", "h", "nn_cal", "lg_nn_cal"]]
c = c.merge(nn, on=["slot", "h"], how="left")
assert c.nn_cal.notna().mean() > 0.999, c.nn_cal.notna().mean()
c["nn_cal"] = c.nn_cal.fillna(0.5)
c["lg_nn_cal"] = c.lg_nn_cal.fillna(0.0)
c["u_rr"] = c.u0 + 0.5 * c.lin
c["nn_contrib"] = 0.1 * c.lg_nn_cal
c["u_r5b"] = c.u_rr + c.nn_contrib
c["lin_contrib"] = 0.5 * c.lin
c["rank_u_r5b"] = c.groupby("slot")["u_r5b"].rank(ascending=False, method="first")
c["gen_logit"] = lg(c.s1.values)
c["gen_rank_pct"] = c.gen_rank_all
ce = c.rename(columns={"sc": "sc_r5", "t1": "t1_score", "s1": "s1_stage1"})
xe = ce[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
be = ce.u_r5b.to_numpy(float)
raw = np.mean([mm.predict(xe, prediction_type='RawFormulaVal') / slope for mm in models], axis=0)
z = be + SCALE * raw
print("eval correction percentiles", np.round(np.quantile(SCALE * raw, [0.01, 0.5, 0.99]), 6))
ce = ce.assign(z=z)
ce = ce.sort_values(["slot", "z"], ascending=[True, False])
ce["r"] = ce.groupby("slot").cumcount()
top = ce[ce.r < 5]
hand_ids = pd.read_parquet(f"{OUT}/np/hand_index.parquet").sort_values("hi").hand_id.values
top = top.assign(hid=hand_ids[top.h.values])
wide = top.pivot(index="slot", columns="r", values="hid")

# ---- assemble submission
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet")
pmap = dict(zip(pidx.player_id, pidx.pi))
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
lo = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
hi = np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
evalp["key"] = lo * 12000 + hi
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
evalp["slot"] = (loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values)
risk = pd.read_parquet(f"{OUT}/m15_v6_drop_m26_eval_scores.parquet")[["key", "score"]]
fam = pd.read_parquet(f"{OUT}/m7_family_eval.parquet")[["key", "family"]]
sub = evalp.merge(risk, on="key", how="left").merge(fam, on="key", how="left")
assert sub.score.notna().all() and sub.family.notna().all()
sub["risk_score"] = sub.score.rank(method="average", pct=True)
sub["predicted_behavior"] = sub.family
sv = sub.copy()
for r in range(5):
    sv[f"evidence_hand_{r+1}"] = sv.slot.map(wide[r]).fillna("NO_EVIDENCE") if r in wide.columns else "NO_EVIDENCE"
out = sv[["pair_id", "risk_score", "predicted_behavior"] + [f"evidence_hand_{i}" for i in range(1, 6)]]
samp = pd.read_csv(f"{RAW}/sample_submission.csv", usecols=["pair_id"])
assert len(out) == len(samp) == out.pair_id.nunique() and set(out.pair_id) == set(samp.pair_id)
assert out.risk_score.between(0, 1).all()
cells = out[[f"evidence_hand_{i}" for i in range(1, 6)]].values
assert all(len([x for x in row if x != "NO_EVIDENCE"]) == len(set(x for x in row if x != "NO_EVIDENCE")) for row in cells)
no_ev = int((cells == "NO_EVIDENCE").sum())
path = f"{DST}/r6_submission.csv"
out.to_csv(path, index=False)
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
print("wrote", path, "rows", len(out), "no_evidence", no_ev, "sha256", sha)
json.dump({"rows": len(out), "no_evidence": no_ev, "sha256": sha, "scale": SCALE, "seeds": SEEDS,
           "slope": slope, "intercept": intercept,
           "evidence": "r5b (u0+0.5lin+0.1nn) + 11-score CatBoost residual (scale 0.25, seeds 71/72/73)"},
          open(f"{DST}/r6_submission_receipt.json", "w"), indent=2)
