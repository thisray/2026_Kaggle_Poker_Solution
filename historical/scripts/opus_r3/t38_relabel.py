"""R3-F16: return fourth-family members whose partner-card substitution evidence looks like normal play back to their
family-model label (rank unchanged), with the family evidence path of t21 DEMOTE (R15 scored top-5).

Env: CAND (input candidate in r2_candidates), OUTNAME, RELAB = "pair:family,pair:family,..." (family = m7 label or 'm7')."""
import os, json, hashlib
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"
CAND, OUTNAME = os.environ["CAND"], os.environ["OUTNAME"]
RELAB = [x.split(":") for x in os.environ["RELAB"].split(",") if x]
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
base = pd.read_csv(f"{C}/{CAND}", dtype=str); out = base.set_index("pair_id").copy()
m7 = pd.read_parquet(f"{OUT}/m7_family_eval.parquet"); e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
lo = m7.key // 12000; hi = m7.key % 12000; m7["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
fam_of = m7.merge(e85, on="slot").set_index("pair_id").family
sc = pd.read_csv(f"{A_}/round15_campaign/scored_tabicl_rank_blend.csv")
done = []
for pid, fam in RELAB:
    assert out.loc[pid, "predicted_behavior"] == "other_coordination", pid
    fam = fam_of[pid] if fam == "m7" else fam; assert fam == fam_of[pid], (pid, fam, fam_of[pid])
    g = sc[sc.pair_id == pid].sort_values("score", ascending=False, kind="mergesort"); assert len(g) >= 5, pid
    out.loc[pid, "predicted_behavior"] = fam; out.loc[pid, EVC] = g.hand_id.values[:5]; done.append((pid, fam))
o = out.reset_index()[base.columns]; path = f"{C}/{OUTNAME}.csv"; o.to_csv(path, index=False)
assert (o.risk_score.values == base.risk_score.values).all()
rec = dict(file=OUTNAME + ".csv", cand=CAND, relabel=done, sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
           behavior_changed=int((o.predicted_behavior.values != base.predicted_behavior.values).sum()),
           evidence_rows_changed=int((o[EVC].values != base[EVC].values).any(1).sum()), n_other=int((o.predicted_behavior == "other_coordination").sum()))
json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec))
