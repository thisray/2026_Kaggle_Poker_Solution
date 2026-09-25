"""Round-6: ordering-only model on the fixed top-5, with relative features; seeds 4/5/6."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
n = pd.read_parquet(f"{DST}/r6_narrow_candidates.parquet")
n = n.sort_values(["slot", "rank_u_r5b"]).reset_index(drop=True)
res = {}


def ap5_flags(flags, n_g):
    hits, s = 0, 0.0
    for r, z in enumerate(flags[:5], start=1):
        if z:
            hits += 1
            s += hits / r
    return s / min(5, max(int(n_g), 1))


# relative features within pair (computed over ALL candidates in the narrow table)
g = n.groupby("slot")
for c in ["u0", "u_rr", "u_r5b", "t1_score", "s1_stage1", "sc_r5"]:
    n[f"{c}_max"] = g[c].transform("max")
    n[f"{c}_mean20"] = g[c].transform("mean")
    n[f"{c}_rk"] = g[c].rank(ascending=False, method="first")
n["u_r5b_minus_max"] = n.u_r5b - n.u_r5b_max
n["u_r5b_gap6"] = n.u_r5b - g.u_r5b.transform(lambda x: x.sort_values(ascending=False).iloc[5] if len(x) > 5 else x.min())
n["nn_contrib_rk"] = g.nn_contrib.rank(ascending=False, method="first")
n["lin_contrib_rk"] = g.lin_contrib.rank(ascending=False, method="first")
n["m_p"] = g.m_p.transform("max")
n["rel_in_top5"] = (n.rank_u_r5b <= 5).astype(int)
feat_cols = [c for c in n.columns if c not in ("slot", "pool", "pair_player_lo", "pair_player_hi", "hand_id", "fold", "ev", "top5_pick", "h_p")]
feat_cols = [c for c in feat_cols if n[c].dtype.kind in "fiu"]

top5 = n[n.rank_u_r5b <= 5].copy()


def eval_orders(df, order_col):
    aps = []
    for s_, gg in df.groupby("slot"):
        n_g = int(gg.m_p.iloc[0])
        o = gg.sort_values(order_col, ascending=False)
        aps.append(ap5_flags(o.ev.values.astype(int), n_g))
    return round(float(np.mean(aps)), 4)


res["current_u_r5b"] = eval_orders(top5, "u_r5b")
res["order_by_u0"] = eval_orders(top5, "u0")
res["order_by_nn"] = eval_orders(top5, "nn_contrib")
for seed in [4, 5, 6]:
    oof = np.zeros(len(top5))
    params = dict(objective="lambdarank", metric="map", ndcg_eval_at=[5], learning_rate=0.05,
                  num_leaves=4, min_data_in_leaf=20, feature_fraction=0.6, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=seed)
    for f in range(5):
        tr = top5.fold.values != f
        va = top5.fold.values == f
        gtr = top5.slot.values[tr]
        ds = lgb.Dataset(top5.loc[tr, feat_cols], label=top5.loc[tr, "ev"].values, group=np.bincount(np.unique(gtr, return_inverse=True)[1]))
        m = lgb.train(params, ds, num_boost_round=300)
        oof[va] = m.predict(top5.loc[va, feat_cols])
    top5["ord_score"] = oof
    res[f"lambdarank_seed{seed}"] = eval_orders(top5, "ord_score")
    # blend with u_r5b
    top5["rb"] = top5.u_r5b.rank(pct=True) + 0.1 * top5.ord_score.rank(pct=True)
    res[f"lambdarank_seed{seed}_blend"] = eval_orders(top5, "rb")
print(json.dumps(res, indent=2))
json.dump(res, open(f"{DST}/r6_order_model.json", "w"), indent=2)
