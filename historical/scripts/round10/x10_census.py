"""Round-10 forensic probe: evidence census and star-group timing structure (v2).

Read-only analysis of the labelled development cohort. Answers:
  A. evidence count per pair, position of evidence hands in the pair's own
     shared-hand timeline (percentile by hand_idx), inter-evidence gaps;
  B. star groups: players in >= 2 positive pairs; satellite-satellite labels;
     evidence timing gap between two pairs sharing a centre.

Output: JSON receipt only.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

RAW = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw")
OUT = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round10_research_20260918")
OUT.mkdir(parents=True, exist_ok=True)

labels = pl.read_csv(RAW / "development_labels.csv")
evidence = pl.read_csv(RAW / "development_evidence.csv")
hands = pl.read_parquet(RAW / "hands.parquet", columns=["hand_id", "table_id", "started_at", "phase"])
seats = pl.read_parquet(RAW / "seats.parquet", columns=["hand_id", "player_id"])

dev_hands = (
    hands.filter(pl.col("phase") == "development")
    .sort(["table_id", "started_at", "hand_id"])
    .with_columns(pl.int_range(0, pl.len()).over("table_id").cast(pl.Int32).alias("hand_idx"))
    .select(["hand_id", "table_id", "hand_idx"])
)
n_dev_per_table = int(dev_hands.group_by("table_id").len()["len"].max())

pos = labels.filter(pl.col("label") == 1)
pos_players = set(pos["player_1"].to_list()) | set(pos["player_2"].to_list())

# all shared dev hands of positive pairs
seats_dev = (
    seats.lazy().join(dev_hands.lazy(), on="hand_id", how="inner")
    .filter(pl.col("player_id").is_in(list(pos_players)))
    .collect()
)
pair_hand_rows = []
for r in labels.iter_rows(named=True):
    if r["label"] != 1:
        continue
    a, b, pid = r["player_1"], r["player_2"], r["pair_id"]
    sub = seats_dev.filter(pl.col("player_id").is_in([a, b]))
    by_hand = sub.group_by("hand_id").agg(
        pl.col("player_id").n_unique().alias("np"), pl.col("hand_idx").first().alias("hand_idx")
    )
    shared = by_hand.filter(pl.col("np") == 2)
    for h, hi in shared.select(["hand_id", "hand_idx"]).iter_rows():
        pair_hand_rows.append((pid, h, hi))
ph = pl.DataFrame(pair_hand_rows, schema=["pair_id", "hand_id", "hand_idx"], orient="row")
print("pair-hand rows:", ph.height)

es = evidence.select(["pair_id", "hand_id"]).with_columns(pl.lit(True).alias("_is_ev"))
ph = ph.join(es, on=["pair_id", "hand_id"], how="left")
ph = ph.sort(["pair_id", "hand_idx"]).with_columns(
    pl.int_range(0, pl.len()).over("pair_id").alias("j"),
    pl.len().over("pair_id").alias("n_shared"),
)
ph = ph.with_columns((pl.col("j") / (pl.col("n_shared") - 1).clip(1)).alias("pct"))
ev_rows = ph.filter(pl.col("_is_ev").fill_null(False))
print("matched evidence rows:", ev_rows.height)

cnt = (
    evidence.group_by("pair_id").len().rename({"len": "n_ev"})
)
cnt_dist = {int(r["n_ev"]): int(r["count"]) for r in cnt["n_ev"].value_counts().sort("n_ev").iter_rows(named=True)}

stats = ev_rows.group_by("pair_id").agg(
    pl.col("pct").min().alias("first_ev_pct"),
    pl.col("pct").max().alias("last_ev_pct"),
    (pl.col("pct").max() - pl.col("pct").min()).alias("span_pct"),
    pl.len().alias("n_ev"),
    pl.col("n_shared").first().alias("n_shared"),
)
first_pct = stats["first_ev_pct"].to_numpy()
last_pct = stats["last_ev_pct"].to_numpy()
span_pct = stats["span_pct"].to_numpy()
n_shared = stats["n_shared"].to_numpy()

# star structure
pair_by_players = {}
player_pairs = defaultdict(list)
for r in labels.iter_rows(named=True):
    key = (r["player_1"], r["player_2"])
    pair_by_players[key] = (r["pair_id"], r["label"])
    player_pairs[r["player_1"]].append((r["pair_id"], r["player_2"], r["label"]))
    player_pairs[r["player_2"]].append((r["pair_id"], r["player_1"], r["label"]))
centers = {p: v for p, v in player_pairs.items() if sum(1 for x in v if x[2] == 1) >= 2}
star_rows = []
for center, links in centers.items():
    partners = [p for _, p, lab in links if lab == 1]
    sat_sat_labels = []
    for i in range(len(partners)):
        for j in range(i + 1, len(partners)):
            key = tuple(sorted([partners[i], partners[j]]))
            sat_sat_labels.append(pair_by_players.get(key, ("NA", -1))[1])
    star_rows.append({"center": center, "degree": len(partners), "sat_sat_labels": sat_sat_labels})
sat_label_counts = defaultdict(int)
for r in star_rows:
    for v in r["sat_sat_labels"]:
        sat_label_counts[int(v)] += 1

# evidence gap between pairs sharing a centre
ev_idx = defaultdict(list)
for r in ev_rows.select(["pair_id", "hand_idx"]).iter_rows():
    ev_idx[r[0]].append(r[1])
overlaps = []
for center, links in centers.items():
    pids = [pid for pid, _p, lab in links if lab == 1 and pid in ev_idx]
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = ev_idx[pids[i]], ev_idx[pids[j]]
            d = float(np.min(np.abs(np.subtract.outer(np.array(a), np.array(b)))))
            overlaps.append(d / n_dev_per_table)
print("star groups:", len(star_rows), "sat-sat label counts:", dict(sat_label_counts), "overlap pairs:", len(overlaps))

result = {
    "n_positives": int(pos.height),
    "n_evidence_rows": evidence.height,
    "evidence_count_distribution": cnt_dist,
    "shared_dev_hands_median": float(np.median(n_shared)),
    "shared_dev_hands_p10_p90": [float(np.quantile(n_shared, 0.10)), float(np.quantile(n_shared, 0.90))],
    "ev_first_pct_quantiles": {str(q): float(np.quantile(first_pct, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
    "ev_last_pct_quantiles": {str(q): float(np.quantile(last_pct, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
    "ev_span_pct_quantiles": {str(q): float(np.quantile(span_pct, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
    "ev_first_pct_mean": float(first_pct.mean()),
    "ev_first_pct_frac_below_0.1": float(np.mean(first_pct < 0.1)),
    "n_pairs_first_ev_below_0.05": int(np.sum(first_pct < 0.05)),
    "star_centers": len(centers),
    "star_degree_distribution": {str(k): sum(1 for r in star_rows if r["degree"] == k) for k in sorted({r["degree"] for r in star_rows})},
    "sat_sat_pairs_total": sum(len(r["sat_sat_labels"]) for r in star_rows),
    "sat_sat_label_counts": {str(k): int(v) for k, v in sat_label_counts.items()},
    "center_pair_evidence_gap_phase_frac": {
        "n": len(overlaps),
        "median": float(np.median(overlaps)) if overlaps else None,
        "p25": float(np.quantile(overlaps, 0.25)) if overlaps else None,
        "p75": float(np.quantile(overlaps, 0.75)) if overlaps else None,
        "frac_below_0.05": float(np.mean(np.array(overlaps) < 0.05)) if overlaps else None,
    },
    "n_dev_per_table": n_dev_per_table,
}
(OUT / "x10_census.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
