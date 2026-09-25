"""R3-P6: relate the eval ranking to the dev-phase P-model OOF score of the same player pair (dev and eval share pools; phase-specific collusion test)."""
import pandas as pd, numpy as np
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; C = f"{OUT}/r2_candidates"
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id"]]
cand = pd.read_csv(f"{C}/r9_subh.csv", dtype=str); cand["rk"] = cand.risk_score.astype(float).rank(ascending=False, method="first"); cand = cand.merge(e85, on="pair_id")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet"); o = o[o.src == "devsub11"].copy()
lo = o.key // 12000; hi = o.key % 12000; o["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
o["dev_rk"] = o.oof.rank(ascending=False, method="first"); o["dev_pct"] = o.oof.rank(pct=True)
j = cand.merge(o[["slot", "oof", "dev_rk", "dev_pct", "y", "hid"]], on="slot", how="left")
print("eval pairs with a devsub11 twin:", int(j.oof.notna().sum()), "of", len(j))
bins = [0, 155, 344, 450, 600, 1000, 2000, 5000, 20000, 200000]
for a, b in zip(bins[:-1], bins[1:]):
    g = j[(j.rk > a) & (j.rk <= b)]; gg = g[g.oof.notna()]
    print(f"eval rank ({a},{b}]: n {len(g)}, with twin {len(gg)}, dev pct median {gg.dev_pct.median():.4f}, share dev_rk<=1000 {(gg.dev_rk <= 1000).mean():.3f}, share dev_rk<=450 {(gg.dev_rk <= 450).mean():.3f}, dev hid flag mean {gg.hid.mean():.3f}")
for fam in ("other_coordination", "coordinated_isolation", "directed_transfer", "soft_play"):
    g = j[(j.rk <= 450) & (j.predicted_behavior == fam) & j.oof.notna()]
    print(f"top450 {fam}: n {len(g)}, dev_rk<=450 {(g.dev_rk <= 450).mean():.3f}, median dev_rk {g.dev_rk.median():.0f}")
print("devsub11 unlabelled pairs in dev top-450:", int(((o.dev_rk <= 450) & (o.y == 0)).sum()), "| of them eval pairs:", int(o[(o.dev_rk <= 450) & (o.y == 0)].slot.isin(set(e85.slot)).sum()))
