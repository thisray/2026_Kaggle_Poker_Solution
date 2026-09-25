"""R3: apply the R18(+hard) CI evidence patch to the LB-verified NDw base without touching P/B.
The patch was built on the r3_main CI routing; keep only the pairs that NDw also routes to CI."""
import json, hashlib
import pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
base = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str, keep_default_na=False)
patch = pd.read_csv(f"{OUT}/r18/main/patch_r18hard.csv", dtype=str, keep_default_na=False)
b = base.set_index("pair_id"); ci = set(b.index[b.predicted_behavior == "coordinated_isolation"])
keep = patch[patch.pair_id.isin(ci)].copy(); drop = patch[~patch.pair_id.isin(ci)]
print(f"NDw: other {int((b.predicted_behavior == 'other_coordination').sum())}, CI {len(ci)}; patch rows {len(patch)}, applied {len(keep)}, skipped (not CI in NDw) {len(drop)} {drop.pair_id.tolist()[:12]}")
out = b.copy(); out.loc[keep.pair_id, EVC] = keep.set_index("pair_id")[EVC].values
o = out.reset_index()[base.columns]; path = f"{C}/r10_ci.csv"; o.to_csv(path, index=False)
assert (o.risk_score.values == base.risk_score.values).all() and (o.predicted_behavior.values == base.predicted_behavior.values).all()
rec = dict(file="r10_ci.csv", base="r2n_NDw_on_r2j2m.csv", patch="r18/main/patch_r18hard.csv", applied=len(keep), skipped=drop.pair_id.tolist(),
           sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), evidence_rows_changed=int((o[EVC].values != base[EVC].values).any(1).sum()))
json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec)[:400])
