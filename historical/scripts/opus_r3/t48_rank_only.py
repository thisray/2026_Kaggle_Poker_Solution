"""R3: P-only probe - take the risk ranking of the R9 package (new8 + B6 promoted) but keep every behaviour label and
evidence cell of the base candidate. Isolates the pure pair-AP effect of the promotions."""
import json, hashlib, os
import pandas as pd, numpy as np
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"
BASE = os.environ.get("BASE", "r10_ci.csv"); RANKS = os.environ.get("RANKS", "r9_subh.csv"); OUTNAME = os.environ["OUTNAME"]
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
b = pd.read_csv(f"{C}/{BASE}", dtype=str, keep_default_na=False); r = pd.read_csv(f"{C}/{RANKS}", dtype=str, keep_default_na=False).set_index("pair_id")
out = b.copy(); out["risk_score"] = out.pair_id.map(r.risk_score)
assert out.risk_score.notna().all()
o = out[b.columns]; path = f"{C}/{OUTNAME}.csv"; o.to_csv(path, index=False)
assert (o.predicted_behavior.values == b.predicted_behavior.values).all() and (o[EVC].values == b[EVC].values).all()
ra = b.risk_score.astype(float).rank(ascending=False, method="first"); rb = o.risk_score.astype(float).rank(ascending=False, method="first")
moved = (ra != rb); big = (rb - ra).abs() > 20
rec = dict(file=OUTNAME + ".csv", base=BASE, ranks_from=RANKS, sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
           rank_changed=int(moved.sum()), moved_far=int(big.sum()), moved_up=b.pair_id[(rb - ra) < -20].tolist(),
           behavior_changed=0, evidence_rows_changed=0)
json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec)[:700])
