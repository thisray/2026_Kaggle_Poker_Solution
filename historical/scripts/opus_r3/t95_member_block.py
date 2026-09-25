"""R3-P27: how big a bet is the fourth-family member block's hard insertion at rank 251?

The candidate's ranking is: non-member pairs by fused score, then the 87 member pairs, then the rest. Nobody has
measured where those 87 pairs would sit on the pair model's own ranking. If they sit near the top anyway, the
insertion is cosmetic; if they sit deep, the candidate is betting a large amount of AP on the fourth-family call.
"""
import numpy as np, pandas as pd, glob, os, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi")
sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]
names = [os.path.basename(f).replace("_eval_scores.parquet", "") for f in sorted(glob.glob(f"{O}/m*_eval_scores.parquet"))]
names = [n for n in names if os.path.exists(f"{O}/{n}_train_oof.parquet")]
E = None
for n in names:
    t = pd.read_parquet(f"{O}/{n}_eval_scores.parquet")
    t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values
    t = t[["slot", "score"]].rename(columns={"score": n}).set_index("slot")
    E = t if E is None else E.join(t, how="inner")
print("models", len(names), "eval rows", len(E))
Z = np.nanmean(np.stack([((E[n] - E[n].mean()) / E[n].std()).values for n in names]), 0)
s = pd.DataFrame({"slot": E.index.values, "fused": Z}).merge(sm, on="slot")
s["rank"] = s.fused.rank(ascending=False, method="first").astype(int)
cand = pd.read_csv(f"{O}/r2_candidates/r13_ndwrank_cinew_f4.csv", dtype=str, keep_default_na=False)
mem = set(cand.pair_id[cand.predicted_behavior == "other_coordination"])
mm = s[s.pair_id.isin(mem)].sort_values("rank")
print(f"member pairs {len(mm)} of {len(mem)}")
print("  their natural fused ranks: min %d  p25 %d  median %d  p75 %d  max %d" %
      tuple(int(mm["rank"].quantile(q)) for q in (0, .25, .5, .75, 1)))
for k in (330, 500, 1000, 2000, 5000, 20000):
    print(f"    within top {k}: {int((mm['rank'] <= k).sum())}/{len(mm)}")
nonmem = s[~s.pair_id.isin(mem)]
print(f"  fused score: members median {mm.fused.median():.3f} vs non-members at rank 250 {nonmem.fused.nlargest(250).min():.3f}, "
      f"at rank 1000 {nonmem.fused.nlargest(1000).min():.3f}, overall median {nonmem.fused.median():.3f}")
json.dump(dict(n=len(mm), median_rank=int(mm["rank"].median()), in_top330=int((mm["rank"] <= 330).sum())),
          open(f"{O}/r3/t95_member_block.json", "w"), indent=1)
