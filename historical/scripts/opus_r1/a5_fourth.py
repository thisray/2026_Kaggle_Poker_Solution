"""Search for the hidden fourth family: strong partner-specific surprisal anomalies that the known-family pair model does not explain."""
import numpy as np, pandas as pd, sys
from pairfeat2 import build2
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
FEATSET = sys.argv[1] if len(sys.argv) > 1 else "both"
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
evalp["key"] = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)) * 12000 + np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap))
posp = set(labels.loc[labels.label == 1, "player_1"].map(pmap)) | set(labels.loc[labels.label == 1, "player_2"].map(pmap))
lab = labels.set_index("key")
anom_cols = ["n_hi5_act_z_max", "n_hi3_act_z_max", "sur_sum_act_c_max", "fold_to_strong_hs_z_max", "check_hu_strong_eq_z_max", "call_to_dead_eq_z_max", "weak_aggr_act_z_max", "xs_fold_to_c_max", "xs_aggr_act_c_max", "xs_pass_hu_c_max"]
for phase_name, tab in [("eval", "ptab_eval"), ("dev", "ptab_devsub11")]:
    t = pd.read_parquet(f"{OUT}/{tab}.parquet"); t["key"] = t.p_lo * 12000 + t.p_hi
    t["label"] = t.key.map(lab.label).fillna(-1).astype(int)
    t["touch"] = t.p_lo.isin(posp) | t.p_hi.isin(posp)
    F = build2(t)
    if phase_name == "eval":
        m = t.key.isin(set(evalp.key)).values
        sc = pd.read_parquet(f"{OUT}/m5_{FEATSET}_eval_scores.parquet").set_index("key").score
        known = t.key.map(sc).values
    else:
        m = ((t.n >= 38) & (~t.touch) & (t.label == -1)).values
        oo = pd.read_parquet(f"{OUT}/m5_{FEATSET}_train_oof.parquet"); oo = oo[oo.src == "devsub11"].set_index("key").oof
        known = t.key.map(oo).values
    G = F[m].copy(); G["known"] = known[m]; G["key"] = t.key.values[m]; G["n"] = t.n.values[m]; G["pool"] = t.pool.values[m]
    # robust z of anomaly columns across the population
    Z = pd.DataFrame({c: (G[c] - G[c].median()) / (G[c].quantile(0.75) - G[c].quantile(0.25) + 1e-6) for c in anom_cols})
    G["anom"] = Z.clip(lower=0).max(axis=1); G["anom_arg"] = Z.idxmax(axis=1)
    print(f"===== {phase_name}: population {len(G)}")
    print("known-score quantiles:", np.round(np.quantile(G.known.dropna(), [0.5, 0.99, 0.995, 0.999]), 4))
    hi_anom = G[(G.anom > G.anom.quantile(0.998))]
    print("top-0.2% anomaly pairs:", len(hi_anom), " of which known>0.2:", int((hi_anom.known > 0.2).sum()), " known<0.02:", int((hi_anom.known < 0.02).sum()))
    print("anomaly driver among unexplained (known<0.02):", hi_anom[hi_anom.known < 0.02].anom_arg.value_counts().to_dict())
    print("anomaly driver among explained (known>0.2):", hi_anom[hi_anom.known > 0.2].anom_arg.value_counts().to_dict())
    cand = G[(G.known < 0.02)].sort_values("anom", ascending=False).head(40)
    print(cand[["key", "pool", "n", "known", "anom", "anom_arg"] + anom_cols[:6]].round(2).to_string(index=False))
    cand.to_parquet(f"{OUT}/a5_fourth_candidates_{phase_name}.parquet")
