"""How many KNOWN-family positives would our fourth-family membership rule (p2 > 2.5 within the P top-600) select?  Apply it on
exposure-matched devsub11/12 (dev has no fourth family) -> expected known-family contamination among the 77 eval members."""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
oof = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet")
for src in ["devsub11", "devsub12"]:
    d = oof[oof.src == src].copy(); lo = d.key // 12000; hi = d.key % 12000
    d["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
    s = pd.read_parquet(f"{OUT}/s23_infoshare_{src}.parquet"); s["p2"] = np.minimum(s.za0 - s.zf0, s.za1 - s.zf1)
    d = d.merge(s[["slot", "p2"]], on="slot", how="left")
    d["rk"] = d.oof.rank(ascending=False, method="first")
    for thr in (2.0, 2.5, 3.0, 4.0):
        sel = d[(d.rk <= 600) & (d.p2 > thr)]
        print(f"{src} p2>{thr} & top600: n {len(sel)}  labelled pos {int(sel.y.sum())} {sel[sel.y == 1].fam.value_counts().to_dict()}  hidden {int(sel.hid.sum())}  U-nonhidden {int(((sel.label < 0) & ~sel.hid).sum())}  confirmed neg {int((sel.label == 0).sum())}")
    pos = d[d.y == 1]
    print(f"   {src}: labelled positives in top600 {int((pos.rk <= 600).sum())}; CI positives p2 quantiles {pos[pos.fam == 'coordinated_isolation'].p2.quantile([.5, .9, .95, .99]).round(2).to_dict()}")
e = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")
print("eval members:", int(e.member.sum()), "| member predicted_behavior in r2j2m:", e[e.member].predicted_behavior.value_counts().to_dict())
print("eval non-member top600 predicted CI with bf>10:", int(((~e.member) & (e.rk_r2j2m <= 600) & (e.bf > 10) & (e.predicted_behavior == 'coordinated_isolation')).sum()))
