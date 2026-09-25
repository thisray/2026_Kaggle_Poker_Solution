"""R4-Y5: apply R4 evidence patches (DT clock/role event blend, CI clock/role event blend + hard filter) to a base candidate. P/B (risk, behaviour) are untouched.
Usage: python y5_build_candidates.py <base csv in r2_candidates> <out name> <patch csv> [<patch csv> ...]"""
import json, hashlib, sys, os
import pandas as pd
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; C = f"{O}/r2_candidates"; DST = f"{O}/r4/cand"; os.makedirs(DST, exist_ok=True)
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]; base_name, out_name, patches = sys.argv[1], sys.argv[2], sys.argv[3:]
base = pd.read_csv(f"{C}/{base_name}", dtype=str, keep_default_na=False); out = base.set_index("pair_id").copy(); rec = dict(file=out_name + ".csv", base=base_name, patches=[])
fam_of = {"dt": "directed_transfer", "ci": "coordinated_isolation", "sp": "soft_play", "f4": "other_coordination"}
for p in patches:
    pt = pd.read_csv(p, dtype=str, keep_default_na=False); fam = fam_of[[k for k in fam_of if f"_{k}_" in os.path.basename(p)][0]]
    ok = pt.pair_id.isin(out.index[out.predicted_behavior == fam]); keep = pt[ok]
    before = out.loc[keep.pair_id, EVC].values.copy(); out.loc[keep.pair_id, EVC] = keep.set_index("pair_id")[EVC].values
    rec["patches"].append(dict(patch=os.path.basename(p), family=fam, rows=len(pt), applied=int(ok.sum()), skipped_not_routed=int((~ok).sum()),
                               pairs_with_changed_set=int(sum(set(a) != set(b) for a, b in zip(before, out.loc[keep.pair_id, EVC].values)))))
o = out.reset_index()[base.columns]; path = f"{DST}/{out_name}.csv"; o.to_csv(path, index=False)
assert (o.risk_score.values == base.risk_score.values).all() and (o.predicted_behavior.values == base.predicted_behavior.values).all() and (o.pair_id.values == base.pair_id.values).all()
rec.update(sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), evidence_rows_changed=int((o[EVC].values != base[EVC].values).any(1).sum()), risk_identical=True, behavior_identical=True)
json.dump(rec, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec, indent=1))
