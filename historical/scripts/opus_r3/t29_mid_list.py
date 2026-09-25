"""List mid-rank (>600) non-member pairs with BF_hier > 4, with placebo D (t27) and tilt BF; mark B6."""
import pandas as pd, numpy as np
R3 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r3"
B = pd.read_parquet(f"{R3}/t17_bf_sub_eval.parquet"); P = pd.read_parquet(f"{R3}/t27_placebo_ext.parquet"); P["Dn"] = P.D / np.sqrt(P.n)
P6 = pd.read_parquet(f"{R3}/t26_placebo.parquet"); P6 = P6[P6.set == "eval_B6"]; P6["Dn"] = P6.D / np.sqrt(P6.n)
PP = pd.concat([P[["slot", "D", "Dn", "bfO"]], P6[["slot", "D", "Dn", "bfO"]]])
b6 = open(f"{R3}/b6_promoted.txt").read().split(",")
m = B[(B.member != True) & (B.rk_r2j2m > 600) & (B.bf_hier > 4)].merge(PP, on="slot", how="left"); m["B6"] = m.pair_id.isin(b6)
pd.set_option("display.width", 220)
print(m.sort_values("bf_hier", ascending=False)[["pair_id", "rk_r2j2m", "predicted_behavior", "bf", "bf_sub", "bf_hier", "D", "Dn", "bfO", "nd", "B6"]].round(2).to_string(index=False))
