"""Rule-level audit: behavior-specific action candidates vs evidence; chronological selection test."""
import numpy as np, pandas as pd
import handfeat2 as HF2, pairindex as PI
OUT = HF2.OUT; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, loc.local.loc[dv.p_lo].values, loc.local.loc[dv.p_hi].values)
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evid = pd.read_csv(f"{RAW}/development_evidence.csv")
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
key2slot = dict(zip(dv.key, dv.slot)); pid2slot = dict(zip(labels.pair_id, labels.key.map(key2slot)))
evid["slot"] = evid.pair_id.map(pid2slot); evid["h"] = evid.hand_id.map(hmap)
ev_rank = dict(zip(zip(evid.slot, evid.h), evid.evidence_rank))
Hd, Sd, Td, SLd = PI.all_pair_hands(0)
lab = dv.set_index("slot").label.reindex(SLd).values; fam = dv.set_index("slot").fam.reindex(SLd).values
idx = np.where(lab >= 0)[0]
X = HF2.features(Hd[idx], Sd[idx], Td[idx])
D = pd.DataFrame({"sl": SLd[idx], "h": Hd[idx], "lab": lab[idx], "fam": fam[idx]})
D["rk"] = [ev_rank.get((a, b), 0) for a, b in zip(D.sl, D.h)]; D["ev"] = D.rk > 0
D["ts"] = np.load(f"{OUT}/np/h_ts.npy")[D.h.values]
rules = {
 "fold_strong_hs": X["S_fold_to_strong_hs_mx"] >= 1,
 "fold_strong_eq": X["S_fold_to_strong_eq_mx"] >= 1,
 "big_fold_strong": X["S_big_fold_to_strong_mx"] >= 1,
 "check_hu_strong_hs": X["S_check_hu_strong_hs_mx"] >= 1,
 "check_hu_strong_eq": X["S_check_hu_strong_eq_mx"] >= 1,
 "call_dead": X["S_call_to_dead_eq_mx"] >= 1,
 "sur5_any": X["S_n_hi5_act_mx"] >= 1,
 "sur_max_fold_gt4": X["S_sur_max_fold_to_mx"] > 4,
 "sur_max_pass_hu_gt4": X["S_sur_max_pass_hu_mx"] > 4,
 "sur_max_aggr_gt4": X["S_sur_max_aggr_act_mx"] > 4,
}
grp = np.where(D.lab == 0, "neg", np.where(D.ev, D.fam + "|EV", D.fam + "|non"))
R = pd.DataFrame({k: v.values for k, v in rules.items()}); R["grp"] = grp
print(R.groupby("grp").mean().T.round(4).to_string())
print(R.groupby("grp").size())
# surprisal max distribution
for c in ["S_sur_max_act_mx", "S_sur_max_fold_to_mx", "S_sur_max_pass_hu_mx", "S_sur_max_aggr_act_mx"]:
    print(c, pd.Series(X[c].values).groupby(grp).quantile(0.5).round(2).to_dict(), " p90:", pd.Series(X[c].values).groupby(grp).quantile(0.9).round(2).to_dict())
# chronological selection test with a combined rule
D["flag"] = (rules["fold_strong_hs"] | rules["check_hu_strong_eq"] | rules["sur_max_fold_gt4"] | rules["sur_max_pass_hu_gt4"]).values
P = D[D.lab == 1].sort_values(["sl", "ts"]).copy()
P["flag_order"] = P[P.flag].groupby("sl").cumcount() + 1
x = P[P.flag]
print("P(evidence | flagged, order k) per family:")
print(x.assign(k=x.flag_order.clip(upper=12)).groupby(["fam", "k"]).ev.agg(["mean", "size"]).round(3).unstack(0).to_string())
print("flagged per pos pair:", P.groupby("sl").flag.sum().describe().round(2).to_dict(), " flagged per neg pair:", D[D.lab == 0].groupby("sl").flag.sum().describe().round(2).to_dict())
print("evidence flagged frac by family:", P[P.ev].groupby("fam").flag.mean().round(3).to_dict())
