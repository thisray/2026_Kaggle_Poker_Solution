"""R3-F22: do fourth-family members also substitute partner cards in the DEV phase (same player pairs, earlier hands)?
Dev and eval share all 400 pools and ~91k of ~122k devsub pair slots. Compute the clean M3 pair LLR (t41 method) on the
devsub11 rows of (a) the eval member slots, (b) a random sample of eval-pair slots ranked 344-3000 (reference)."""
import numpy as np, pandas as pd, sys
sys.argv = ["t41"]; src = open("t41_mid_clean.py").read().split("# eval: current candidate ranking")[0]; exec(src)
cand = pd.read_csv(f"{C}/r9_subh.csv", dtype=str); cand["rk"] = cand.risk_score.astype(float).rank(ascending=False, method="first")
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]; ev = cand.merge(e85, on="slot" if "slot" in cand else "pair_id")
mem = ev[ev.predicted_behavior == "other_coordination"]; ref = ev[(ev.rk > 344) & (ev.rk <= 3000)].sample(400, random_state=1)
meta = pd.read_parquet(f"{R3}/sub_meta_devsub11.parquet", columns=["slot"]); have = set(meta.slot.unique())
print(f"member slots with devsub11 rows: {sum(s in have for s in mem.slot)}/{len(mem)}; reference {sum(s in have for s in ref.slot)}/{len(ref)}")
out = []
for nm, g in (("members", mem), ("ref_344_3000", ref)):
    sl = [s for s in g.slot if s in have]
    L = m3_llr(clean_rows("devsub11", np.array(sl))); L["set"] = nm; out.append(L.merge(g[["slot", "pair_id", "rk"]], on="slot"))
R = pd.concat(out); R.to_parquet(f"{R3}/t43_member_devphase.parquet")
print(R.groupby("set").llr.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
ev41 = pd.read_parquet(f"{R3}/t41_mid_clean.parquet"); ev41 = ev41[ev41.set == "eval"][["slot", "llr"]].rename(columns={"llr": "llr_eval"})
j = R[R.set == "members"].merge(ev41, on="slot"); print("members: corr(dev LLR, eval LLR) =", round(float(np.corrcoef(j.llr, j.llr_eval)[0, 1]), 3))
print(j.sort_values("llr")[["pair_id", "rk", "llr_eval", "llr", "nhands", "ndec"]].round(2).head(10).to_string(index=False))
