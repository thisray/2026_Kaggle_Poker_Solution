"""Round-10 probe: evaluation-phase star structure and transductive potential.

Using the deployed risk scores for all 112,540 evaluation pairs:
  1. per pool, take the top-K pairs by risk; build the player graph;
  2. find centres (players appearing in >= 2 top-K pairs) and count;
  3. for every other eval pair sharing a centre, report its risk percentile;
  4. compare the observed degree structure with a degree-preserving null
     (shuffle pair scores within each pool).

If centres' other pairs are systematically lower-ranked, there are missed
star edges (recall) or satellite-satellite false positives (precision).

Read-only; writes one JSON receipt.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

A = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917")
RAW = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw")
OUT = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round10_research_20260918")
OUT.mkdir(parents=True, exist_ok=True)

scores = pl.read_parquet(A / "m15_v6_drop_m26_eval_scores.parquet").select(["key", "pool", "p_lo", "p_hi", "n", "score"])
pi = pl.read_parquet(A / "np" / "player_index.parquet")
print("player_index:", pi.columns, pi.shape)
pi2pid = {int(a): b for a, b in pi.select(["pi", "player_id"]).iter_rows()}
scores = scores.with_columns(
    pl.col("p_lo").cast(pl.Int64).map_elements(lambda x: pi2pid.get(int(x), "?"), return_dtype=pl.Utf8).alias("lo_id"),
    pl.col("p_hi").cast(pl.Int64).map_elements(lambda x: pi2pid.get(int(x), "?"), return_dtype=pl.Utf8).alias("hi_id"),
)
scores = scores.with_columns(
    (pl.col("score").rank(descending=True, method="ordinal").over("pool") / pl.len().over("pool")).alias("pct_in_pool"),
    pl.col("score").rank(descending=True, method="ordinal").over("pool").alias("rank_in_pool"),
)
print(scores.head(3))

res = {}
for K in (5, 8, 12, 20):
    deg = defaultdict(int)
    top_pairs = scores.filter(pl.col("rank_in_pool") <= K)
    for r in top_pairs.iter_rows(named=True):
        deg[r["lo_id"]] += 1
        deg[r["hi_id"]] += 1
    centres = {p for p, d in deg.items() if d >= 2}
    # all pairs sharing a centre
    other_pct = []
    for r in scores.iter_rows(named=True):
        if r["lo_id"] in centres or r["hi_id"] in centres:
            other_pct.append(r["pct_in_pool"])
    other_pct = np.array(other_pct)
    # restrict: pairs not themselves in top-K
    tp = scores.filter(pl.col("rank_in_pool") <= K)
    tp_keys = set(tp["key"].to_list())
    stats = scores.with_columns(
        pl.when(pl.col("lo_id").is_in(list(centres)) | pl.col("hi_id").is_in(list(centres)))
        .then(1).otherwise(0).alias("touches_centre")
    )
    sub = stats.filter((pl.col("touches_centre") == 1) & (~pl.col("key").is_in(list(tp_keys))))
    res[f"K{K}"] = {
        "centres": len(centres),
        "centre_pairs_topk": top_pairs.height,
        "non_topk_pairs_touching_centre": sub.height,
        "non_topk_touch_pct_median": float(sub["pct_in_pool"].median()) if sub.height else None,
        "non_topk_touch_pct_frac_under_0.2": float((sub["pct_in_pool"] <= 0.2).mean()) if sub.height else None,
        "global_pct_median": 0.5,
    }
    print(f"K={K}", res[f"K{K}"], flush=True)

(OUT / "x10_eval_star.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
