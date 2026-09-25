"""R4-Y4: transfer sanity of an evidence patch on the eval pairs that are almost surely positive (risk rank <= TOPK and routed to the family):
relative position and cell composition of the picked hands vs the dev truth, and vs the deployed R15 picks."""
import numpy as np, pandas as pd, sys
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
fam, patch, f_role, f_cand, base_csv = sys.argv[1:6]; TOPK = int(sys.argv[6]) if len(sys.argv) > 6 else 450
base = pd.read_csv(base_csv, usecols=["pair_id", "risk_score", "predicted_behavior"]); base["rk"] = base.risk_score.rank(ascending=False, method="first")
top = set(base[(base.rk <= TOPK) & (base.predicted_behavior == fam)].pair_id); print("likely-positive routed pairs:", len(top))
hmap = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi
role = pd.read_parquet(f_role); cand = pd.read_parquet(f_cand); pid = cand[["pair_id", "slot"]].drop_duplicates()
pt = pd.read_csv(patch); new = pt.melt("pair_id", value_name="hand_id").assign(h=lambda d: d.hand_id.map(hmap)).merge(pid, on="pair_id")
old = cand[cand.r <= 5][["pair_id", "slot", "h"]]
cols = ["x_rel", "x_big", "x_cell", "x_s_fold_to_r", "x_dir", "x_hsS_last", "x_k"]
def prof(df, nm):
    d = df[df.pair_id.isin(top)].merge(role[["slot", "h"] + cols], on=["slot", "h"]); print(f"{nm:22s} n={len(d):5d} " + " ".join(f"{c[2:]} {d[c].mean():.3f}" for c in cols) + f" | rel q25/50/75 {d.x_rel.quantile(.25):.2f}/{d.x_rel.quantile(.5):.2f}/{d.x_rel.quantile(.75):.2f}")
prof(new, "eval NEW picks"); prof(old, "eval R15 picks")
dv = pd.read_parquet(f"{O}/r4/x2_role_dev.parquet").merge(pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"})[["slot", "h", "ev", "fam"]], on=["slot", "h"]); dv = dv[(dv.fam == fam) & dv.ev]
print(f"{'dev TRUE evidence':22s} n={len(dv):5d} " + " ".join(f"{c[2:]} {dv[c].mean():.3f}" for c in cols) + f" | rel q25/50/75 {dv.x_rel.quantile(.25):.2f}/{dv.x_rel.quantile(.5):.2f}/{dv.x_rel.quantile(.75):.2f}")
