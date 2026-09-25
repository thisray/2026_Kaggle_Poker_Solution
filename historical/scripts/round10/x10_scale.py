"""Round-10 proxy experiment: does evidence-MAP@5 scale with the number of
training pairs? Train the same within-pair ranker on 25/50/75/100% of the
training folds' pairs (validation folds always untouched) and measure OOF E
with the r5-style x3 decoder. Also test censoring-aware sample weights:
non-evidence candidates that sit after the pair's last evidence AND score in
the model's top ranks are likely censored positives -> downweight.

Read-only w.r.t. raw data; writes one JSON receipt.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

ART = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts")
PACK = ART / "round8_raw_20260917" / "dev_pack"
OUT = ART / "round10_research_20260918"
OUT.mkdir(parents=True, exist_ok=True)

meta = pl.read_csv(PACK / "meta.csv")
tab = np.load(PACK / "tab.npy", mmap_mode="r")
x_mom = np.nan_to_num(np.c_[tab.mean(1), tab.max(1)], nan=0.0, posinf=30.0, neginf=-30.0).clip(-30, 30).astype(np.float32)
SCORES = ["sc_r5", "u0", "u_rr", "u_r5b", "lin_contrib", "nn_contrib", "t1_score", "s1_stage1", "gen_logit", "gen_rank_pct", "rank_u_r5b"]
s = meta.select(SCORES).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
X = np.hstack([x_mom, s])
ev = meta["ev"].cast(pl.Int32).to_numpy()
slot = meta["slot"].cast(pl.Int64).to_numpy()
fold = meta["fold"].cast(pl.Int64).to_numpy()
m_p = meta["m_p"].cast(pl.Float32).to_numpy()
u0 = meta["u_r5b"].cast(pl.Float64).to_numpy()
print("rows", X.shape, "pairs", len(np.unique(slot)), "folds", np.unique(fold))

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, seed=71, num_threads=8, num_boost_round=200)


def map5(score, evv, slotv):
    df = pl.DataFrame({"slot": slotv, "ev": evv, "sc": score}).sort(["slot", "sc"], descending=[False, True])
    vals = []
    for _pid, g in df.group_by("slot", maintain_order=True):
        rel = g["ev"].to_numpy()
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        top = rel[:5]
        hits = np.cumsum(top)
        vals.append(float(np.sum((hits / np.arange(1, len(top) + 1)) * top) / min(n_rel, 5)))
    return float(np.mean(vals)), len(vals)


ts_df = pl.read_csv(
    ART / "round3_research_20260917" / "r6_narrow_candidates_v2.csv",
    columns=["slot", "hand_id", "ts_pct_in_pair"],
)
meta = meta.join(ts_df, on=["slot", "hand_id"], how="left")
assert meta["ts_pct_in_pair"].null_count() == 0, "ts join failed"
ts = meta["ts_pct_in_pair"].cast(pl.Float64).to_numpy()
pair_stats = meta.group_by("slot").agg(pl.col("ev").sum().alias("n_ev"))
print("pairs with ev>0:", pair_stats.filter(pl.col("n_ev") > 0).height)

results = {}
rng = np.random.RandomState(71)
pair_ids = np.unique(slot)

pair_fold = {p: int(fold[np.where(slot == p)[0][0]]) for p in pair_ids}
for frac in (0.25, 0.5, 0.75, 1.0):
    oof = np.zeros(len(X))
    for f in range(5):
        tr_pair = [p for p in pair_ids if pair_fold[p] != f]
        rng.shuffle(tr_pair)
        keep = set(tr_pair[: max(1, int(len(tr_pair) * frac))])
        tr = np.isin(slot, list(keep))
        te = np.isin(slot, [p for p in pair_ids if pair_fold[p] == f])
        ds = lgb.Dataset(X[tr], ev[tr], weight=(1.0 / np.maximum(m_p[tr], 1.0)))
        m = lgb.train({**PARAMS}, ds)
        oof[te] = m.predict(X[te])
    # rank within pair as score for MAP@5
    rank = pl.DataFrame({"slot": slot, "sc": oof}).with_columns(
        pl.col("sc").rank(descending=True, method="ordinal").over("slot").alias("r")
    )["r"].to_numpy()
    e_rank, npairs = map5(-rank.astype(float), ev, slot)
    results[f"frac_{frac}"] = {"E_rank": e_rank, "pairs": npairs, "n_train_pairs_mean": float(frac * 297)}
    print(frac, "E_rank", round(e_rank, 4), flush=True)

# censoring-aware weights at frac=1.0
oof = np.zeros(len(X))
for f in range(5):
    tr = np.isin(slot, [p for p in pair_ids if fold[np.where(slot == p)[0][0]] != f])
    te = np.isin(slot, [p for p in pair_ids if fold[np.where(slot == p)[0][0]] == f])
    # censoring-aware weights: late non-evidence candidates with high frozen OOF score (u0)
    w = 1.0 / np.maximum(m_p, 1.0)
    last_ev = (
        pl.DataFrame({"slot": slot, "ev": ev, "ts": ts})
        .filter(pl.col("ev") == 1)
        .group_by("slot")
        .agg(pl.col("ts").max().alias("last_ev"))
    )
    tmp = pl.DataFrame({"slot": slot, "ts": ts, "ev": ev, "u0": u0}).join(last_ev, on="slot", how="left")
    censored = tmp.select(
        ((pl.col("ev") == 0) & pl.col("last_ev").is_not_null() & (pl.col("ts") > pl.col("last_ev")) & (pl.col("u0") > 0.5))
    ).to_series().to_numpy()
    w[tr] = np.where(censored[tr], w[tr] * 0.2, w[tr])
    ds = lgb.Dataset(X[tr], ev[tr], weight=w[tr])
    m = lgb.train({**PARAMS}, ds)
    oof[te] = m.predict(X[te])
rank = pl.DataFrame({"slot": slot, "sc": oof}).with_columns(
    pl.col("sc").rank(descending=True, method="ordinal").over("slot").alias("r")
)["r"].to_numpy()
e_cens, _ = map5(-rank.astype(float), ev, slot)
results["censoring_aware"] = {"E_rank": e_cens}
print("censoring-aware E_rank", round(e_cens, 4))

(OUT / "x10_scaling.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
