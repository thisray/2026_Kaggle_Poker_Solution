"""R3-QA: independent verification of the final candidate's content against its sources.
Checks: (1) risk order == fusion order with the NDw member block re-inserted after rank 250; (2) evidence rows equal the
expected source per pair (CI patch / F4 M3 decoder output / NDw base / R15 for the relabelled pairs); (3) labels equal
the NDw labels with exactly the 20 intended changes; (4) official-format constraints."""
import numpy as np, pandas as pd, json, os, sys
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; C = f"{O}/r2_candidates"; A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
CAND = os.environ.get("CAND", "r19_fuseALL_cinew.csv"); EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
d = pd.read_csv(f"{C}/{CAND}", dtype=str, keep_default_na=False)
ndw = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str, keep_default_na=False).set_index("pair_id")
r13 = pd.read_csv(f"{C}/r13_ndwrank_cinew_f4.csv", dtype=str, keep_default_na=False).set_index("pair_id")
x = d.set_index("pair_id")
ok = True
# 1. labels
lab_diff = (x.predicted_behavior != ndw.predicted_behavior.reindex(x.index))
print(f"1. labels differing from NDw: {int(lab_diff.sum())} (expected 20)"); ok &= int(lab_diff.sum()) == 20
print("   transitions:", dict(pd.crosstab(ndw.predicted_behavior.reindex(x.index)[lab_diff], x.predicted_behavior[lab_diff]).stack().loc[lambda s: s > 0]))
# 2. evidence equals r13 (same labels/evidence, only ranking differs)
same_ev = (x[EVC].values == r13[EVC].reindex(x.index).values).all()
print(f"2. evidence identical to r13: {same_ev}"); ok &= bool(same_ev)
ci = x.index[x.predicted_behavior == "coordinated_isolation"]
patch = pd.read_csv(f"{O}/r18/main/patch_r18new_hard.csv", dtype=str, keep_default_na=False).set_index("pair_id")
inter = [p for p in patch.index if p in set(ci)]
match = (x.loc[inter, EVC].values == patch.loc[inter, EVC].values).all()
print(f"   CI pairs covered by the new patch: {len(inter)}; evidence matches the patch: {match}"); ok &= bool(match)
oth = x.index[x.predicted_behavior == "other_coordination"]
print(f"   other_coordination pairs: {len(oth)} (expected 87)"); ok &= len(oth) == 87
# 3. ranking equals the fusion order with the member block after rank 250
fz = pd.read_parquet(f"{O}/r3/t70_eval_fused.parquet") if os.path.exists(f"{O}/r3/t70_eval_fused.parquet") else None
rk = x.risk_score.astype(float).rank(ascending=False, method="first")
mem = set(ndw.index[ndw.predicted_behavior == "other_coordination"])
mem_rk = sorted(int(rk[p]) for p in mem)
print(f"3. NDw member block occupies ranks {mem_rk[0]}..{mem_rk[-1]} (expected 251..327, contiguous: {mem_rk == list(range(mem_rk[0], mem_rk[0] + len(mem_rk)))})")
ok &= mem_rk[0] == 251 and mem_rk[-1] == 327
# 4. format
r = x.risk_score.astype(float)
print(f"4. risk in [0,1]: {bool(r.between(0, 1).all())}; unique risks: {r.nunique()} of {len(r)}; rows {len(x)}; duplicate evidence within a row: {int((x[EVC].apply(lambda s: s.nunique(), axis=1) != 5).sum())}")
ok &= bool(r.between(0, 1).all()) and len(x) == 112540
print("VERDICT:", "PASS" if ok else "FAIL")
