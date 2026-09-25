"""R3-P6b: are dev-labelled positive pairs ever eval pairs? (answer: 0 of 364/365)."""
import pandas as pd, numpy as np
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet")
lo = o.key // 12000; hi = o.key % 12000; o["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
ev = set(e85.slot)
for s in ("devsub11", "devsub12"):
    g = o[o.src == s]; pos = g[g.y == 1]
    print(f"{s}: labelled positives {len(pos)}, of which are eval pairs: {int(pos.slot.isin(ev).sum())}; unlabelled {int((g.y == 0).sum())} of which eval pairs {int(g[g.y == 0].slot.isin(ev).sum())}")
M = pd.read_parquet(f"{OUT}/r3/t41_mid_clean.parquet"); c = M[(M.set == "eval") & (M.rk > 600) & (M.llr > 5)]
lab = o[o.src == "devsub11"][["slot", "y", "fam"]].drop_duplicates("slot")
j = c.merge(lab, on="slot", how="left"); print("eval mid-rank LLR>5 candidates:", len(c), "| present in devsub11:", int(j.y.notna().sum()), "| dev-labelled positive:", int((j.y == 1).sum()))
print(j[j.y == 1][["pair_id", "rk", "predicted_behavior", "llr", "fam_y" if "fam_y" in j else "fam"]].to_string(index=False) if (j.y == 1).any() else "none")
