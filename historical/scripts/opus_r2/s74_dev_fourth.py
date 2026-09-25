"""Does the fourth family also exist (unlabelled) in the dev phase?  p2 over dev pairs by label status and dev OOF score;
and eval: how many pairs does the P model place high, to estimate the fourth family's share of eval positives."""
import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
d = pd.read_parquet(f"{OUT}/s23_infoshare_dev.parquet"); d["p2"] = np.minimum(d.za0 - d.zf0, d.za1 - d.zf1)
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
print(dv.columns.tolist()[:20])
sc = "oof" if "oof" in dv.columns else [c for c in dv.columns if "oof" in c or "score" in c][0]
x = d.merge(dv[["slot", "label", sc]].rename(columns={sc: "oof"}), on="slot", how="left")
x["lab"] = np.where(x.label == 1, "pos", np.where(x.label == 0, "neg", "U"))
x["rk"] = x.oof.rank(ascending=False)
for thr in [3, 4, 5]:
    print(f"dev p2>{thr}: by label", x[x.p2 > thr].lab.value_counts().to_dict(), "| among dev OOF top-600:", x[(x.p2 > thr) & (x.rk <= 600)].lab.value_counts().to_dict(),
          "| top-1000:", int(((x.p2 > thr) & (x.rk <= 1000)).sum()))
e = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet"); e["p2"] = np.minimum(e.za0 - e.zf0, e.za1 - e.zf1)
e = e.merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"), on="slot"); e["rk"] = e.risk_score.rank(ascending=False)
for thr in [3, 4, 5]:
    print(f"eval p2>{thr}: top-600 {int(((e.p2 > thr) & (e.rk <= 600)).sum())}, top-1000 {int(((e.p2 > thr) & (e.rk <= 1000)).sum())}, all {int((e.p2 > thr).sum())}")
# null rate of p2 > thr among low-score pairs (dev U, rank>5000) to compute excess
for nm, z in [("dev", x), ("eval", e)]:
    lo = z[z.rk > 5000]
    print(nm, "null rate p2>4 among rank>5000:", round((lo.p2 > 4).mean(), 5), " n", len(lo))
