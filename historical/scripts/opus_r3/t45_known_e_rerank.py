"""R3-E12: known-family evidence re-ranker that corrects for policy memorisation.
Base: R15 dev OOF blend b = 0.6 rank(TabICL) + 0.4 rank(r11 ranker) within pair (t3). Extra per-hand features from the two
members' decisions: share of decisions policy_v1 / policy_v2 trained on, and CLEAN surprisal (v2 where v2 did not train,
else v1 where v1 did not train, else missing) - max / mean / max over aggressive actions, plus the gap to the original
(v1) surprisal. LightGBM binary on ev with the existing pool folds; AP@5 per pair vs the base blend, per family."""
import numpy as np, pandas as pd, lightgbm as lgb
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"; D = f"{OUT}/np"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "rs_blend"]]
tab = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"})
d = d.merge(tab, on=["slot", "hand_id"]); g_ = d.groupby("slot")
d["r_tab"] = g_.tab.rank(pct=True); d["r_rs"] = g_.rs_blend.rank(pct=True); d["b"] = 0.6 * d.r_tab + 0.4 * d.r_rs
hidx = pd.read_parquet(f"{D}/hand_index.parquet").set_index("hand_id").hi
pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
lab = pd.read_csv(f"{RAW}/development_labels.csv"); lab = lab[lab.label == 1].copy()
a = lab.player_1.map(pmap).values; b_ = lab.player_2.map(pmap).values; lo = np.minimum(a, b_); hi = np.maximum(a, b_)
lab["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values; lab["pa"] = a; lab["pb"] = b_
d = d.merge(lab[["slot", "pa", "pb", "behavior_family"]], on="slot")
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
a_act = np.load(f"{D}/a_act.npy", mmap_mode="r"); a_amt = np.load(f"{D}/a_amount.npy", mmap_mode="r"); a_tc = np.load(f"{D}/a_to_call.npy", mmap_mode="r")
Y = np.load(f"{OUT}/dec_Y.npy", mmap_mode="r"); P1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
N = int(off[-1]); in1 = np.zeros(N, bool); in1[np.random.RandomState(0).choice(N, 4_000_000, replace=False)] = True
in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
F = np.full((len(d), 9), np.nan)
for i, (h, pa_, pb_) in enumerate(zip(hidx.loc[d.hand_id].values, d.pa.values, d.pb.values)):
    ks = np.arange(off[h], off[h + 1]); pl = np.asarray(sp[h])[np.asarray(a_seat[ks])]; ks = ks[(pl == pa_) | (pl == pb_)]
    if not len(ks): continue
    y = np.asarray(Y[ks]); p1 = np.asarray(P1[ks])[np.arange(len(ks)), y]; p2 = np.asarray(P2[ks])[np.arange(len(ks)), y]
    s1 = -np.log(np.clip(p1, 1e-6, 1)); qc = np.where(~in2[ks], p2, np.where(~in1[ks], p1, np.nan)); sc = -np.log(np.clip(qc, 1e-6, 1))
    act = np.asarray(a_act[ks]); ag = (act == 3) | (act == 4) | ((act == 5) & (np.asarray(a_amt[ks]) > np.asarray(a_tc[ks])))
    F[i] = [in1[ks].mean(), in2[ks].mean(), np.nanmax(sc) if np.isfinite(sc).any() else np.nan, np.nanmean(sc) if np.isfinite(sc).any() else np.nan,
            np.nanmax(sc[ag]) if (ag & np.isfinite(sc)).any() else 0.0, s1.max(), s1[ag].max() if ag.any() else 0.0, len(ks), ag.sum()]
cols = ["sh1", "sh2", "sur_c_max", "sur_c_mean", "sur_c_aggr_max", "sur_1_max", "sur_1_aggr_max", "nd", "nag"]
for j, c in enumerate(cols): d[c] = F[:, j]
d["gap_aggr"] = d.sur_c_aggr_max - d.sur_1_aggr_max; d["gap_max"] = d.sur_c_max - d.sur_1_max
d["fam"] = d.behavior_family.astype("category").cat.codes
X_BASE = ["r_tab", "r_rs", "b"]; X_ALL = X_BASE + cols + ["gap_aggr", "gap_max", "fam"]
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").head(5).ev.values; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.ev.sum()))
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=3)
for name, X in (("base_feats_only", X_BASE), ("with_memo_feats", X_ALL)):
    for seed in (3, 7, 11):
        p = np.zeros(len(d)); params["seed"] = seed
        for f in sorted(d.fold.unique()):
            tr, va = d.fold != f, d.fold == f
            m = lgb.train(params, lgb.Dataset(d.loc[tr, X], d.loc[tr, "ev"]), num_boost_round=400)
            p[va.values] = m.predict(d.loc[va, X])
        d[f"p_{name}_{seed}"] = p
    d[f"p_{name}"] = d[[f"p_{name}_{s}" for s in (3, 7, 11)]].mean(1)
res = d.groupby("slot").apply(lambda g: pd.Series({"fam": g.behavior_family.iloc[0], "base": ap5(g, "b"), "lgb_base": ap5(g, "p_base_feats_only"), "lgb_memo": ap5(g, "p_with_memo_feats")}))
print("E (AP@5) dev OOF, mean over pairs:"); print(res.groupby("fam")[["base", "lgb_base", "lgb_memo"]].mean().round(4).to_string()); print("all", res[["base", "lgb_base", "lgb_memo"]].mean().round(4).to_dict())
print("pairs improved/worse (memo vs lgb_base):", int((res.lgb_memo > res.lgb_base + 1e-9).sum()), int((res.lgb_memo < res.lgb_base - 1e-9).sum()))
m = lgb.train(params, lgb.Dataset(d[X_ALL], d.ev), num_boost_round=400)
print(pd.Series(m.feature_importance("gain"), index=X_ALL).sort_values(ascending=False).round(0).to_string())
d.to_parquet(f"{OUT}/r3/t45_known_e_rerank.parquet"); res.to_parquet(f"{OUT}/r3/t45_known_e_rerank_pairs.parquet")
