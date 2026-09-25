"""Step C of r5b: assemble submission with u = u0 + 0.5*lin + 0.1*lg(nn_cal)."""
import hashlib
import json

import numpy as np
import pandas as pd

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
A_SCALE, A_NN = 0.5, 0.1

cand = pd.read_parquet(f"{DST}/r5_candidates.parquet")
nn = pd.read_parquet(f"{DST}/r5cand_nn.parquet")
C = cand.merge(nn[["slot", "h", "lg_nn_cal"]], on=["slot", "h"], how="left")
assert C.lg_nn_cal.notna().all()
C["uf"] = C.u0 + A_SCALE * C.lin + A_NN * C.lg_nn_cal
C = C.sort_values(["slot", "uf"], ascending=[True, False])
C["r"] = C.groupby("slot").cumcount()
top = C[C.r < 5]
hand_ids = pd.read_parquet(f"{OUT}/np/hand_index.parquet").sort_values("hi").hand_id.values
top["hid"] = hand_ids[top.h.values]
wide = top.pivot(index="slot", columns="r", values="hid")

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
path = f"{DST}/r5b_submission.csv"
out.to_csv(path, index=False)
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
print("wrote", path, "rows", len(out), "no_evidence", no_ev, "sha256", sha)
json.dump({"rows": len(out), "no_evidence": no_ev, "sha256": sha, "a_scale": A_SCALE, "a_nn": A_NN,
           "evidence": "family-routed r5fam(R5) + x3 + rerank(0.5) + nn(0.1)"},
          open(f"{DST}/r5b_submission_receipt.json", "w"), indent=2)
