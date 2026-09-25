"""R4-Z1: do fourth-family member pairs show an excess of any OUTCOME-level hand pattern (a sparse script like the known families' ~6 planted hands per phase),
beyond the dense partner-card substitution? Per-hand role features for (a) the 77 F4 members, (b) likely-positive known-family eval pairs, (c) random eval pairs of the same pools.
Reports per-pair-phase counts of cell hands (mean per pair) for each group."""
import numpy as np, pandas as pd, sys, subprocess, os
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"; PY = sys.executable
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi)); loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi")
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); a = ev.player_1.map(pmap).values; b = ev.player_2.map(pmap).values; ev["pa"] = np.minimum(a, b); ev["pb"] = np.maximum(a, b)
ev["slot"] = loc.pool.loc[ev.pa].values * 900 + loc.local.loc[ev.pa].values * 30 + loc.local.loc[ev.pb].values
ndw = pd.read_csv(f"{O}/r2_candidates/r2n_NDw_on_r2j2m.csv", usecols=["pair_id", "risk_score", "predicted_behavior"]); ndw["rk"] = ndw.risk_score.rank(ascending=False, method="first")
g = ev.merge(ndw, on="pair_id"); g["grp"] = "rest"
g.loc[g.predicted_behavior == "other_coordination", "grp"] = "F4"
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"): g.loc[(g.rk <= 250) & (g.predicted_behavior == fam), "grp"] = fam[:2].upper() + "_top"
pools = set(g[g.grp == "F4"].slot // 900); ctrl = g[(g.grp == "rest") & (g.rk > 20000) & (g.slot // 900).isin(pools)].sample(600, random_state=1); ctrl["grp"] = "control"
sel = pd.concat([g[g.grp != "rest"], ctrl]); sel[["slot", "pa", "pb"]].to_parquet(f"{O}/r4/z1_pairs.parquet"); print(sel.grp.value_counts().to_dict(), flush=True)
if not os.path.exists(f"{O}/r4/z1_role.parquet"):
    subprocess.run([PY, "x2_role_feats.py", "eval", f"{O}/r4/z1_pairs.parquet", f"{O}/r4/z1_role.parquet"], check=True)
x = pd.read_parquet(f"{O}/r4/z1_role.parquet").merge(sel[["slot", "grp", "shared_hands"]], on="slot")
x["small_cell"] = x.x_cell * (1 - x.x_big); x["S_fold_to_R"] = x.x_s_fold_to_r; x["R_fold_to_S"] = x.x_r_fold_to_s; x["both_sd"] = x.x_sdS * x.x_sdR; x["bigR"] = x.x_bigR; x["big_any"] = x.x_big
x["bigS"] = x.x_big * (x.x_dir == -1); x["vol"] = x.x_vol; x["hu_post"] = ((x.x_stmax >= 1) & (x.x_o_post == 0) & (x.x_vol == 1)).astype(float)
x["S_strong_fold"] = x.x_s_fold_to_r * (x.x_hsS_last >= 0.6); x["R_strong_fold"] = x.x_r_fold_to_s * (x.x_hsR_last >= 0.6); x["S_weak_call_sd"] = ((x.x_sdS == 1) & (x.x_hsS_last <= 0.35) & (x.x_dir == 1)).astype(float)
x["pair_both_aggr_pre"] = x.x_s_aggr_pre * x.x_r_aggr_pre; x["reraise_war"] = ((x.x_s_aggr_n + x.x_r_aggr_n) >= 3).astype(float)
cols = ["vol", "small_cell", "bigR", "bigS", "big_any", "S_fold_to_R", "R_fold_to_S", "S_strong_fold", "R_strong_fold", "both_sd", "hu_post", "S_weak_call_sd", "pair_both_aggr_pre", "reraise_war"]
per = x.groupby(["grp", "slot"])[cols].sum().join(x.groupby(["grp", "slot"]).size().rename("n"))
pd.set_option("display.width", 250); print("mean count of cell hands per pair (eval phase):"); print(per.groupby("grp").mean().round(2).T.to_string())
rate = per[cols].div(per.n, axis=0); print("rate per co-seated hand (x100):"); print((rate.groupby("grp").mean() * 100).round(2).T.to_string())
