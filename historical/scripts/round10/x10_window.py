"""Round-10 probe: cross-pair evidence timing (group-episode hypothesis).

For every shared hand of a labelled positive pair, mark whether another
positive pair that shares a player has an evidence hand within +/-K hands.
Test the lift over base rate, within the model's top-K candidate ranks, and
against a within-pair permutation null.

Read-only; writes one JSON receipt.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

RAW = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw")
NARROW = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917/r6_narrow_candidates_v2.csv")
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
pos = labels.filter(pl.col("label") == 1)
pos_players = set(pos["player_1"].to_list()) | set(pos["player_2"].to_list())

# pair-id -> (player_1, player_2), and player -> list of (pair_id, other_player)
pair_players = {}
player_pairs = defaultdict(list)
for r in labels.iter_rows(named=True):
    if r["label"] != 1:
        continue
    a, b, pid = r["player_1"], r["player_2"], r["pair_id"]
    pair_players[pid] = (a, b)
    player_pairs[a].append(pid)
    player_pairs[b].append(pid)

# shared dev hands of positive pairs
seats_dev = (
    seats.lazy().join(dev_hands.lazy(), on="hand_id", how="inner")
    .filter(pl.col("player_id").is_in(list(pos_players)))
    .collect()
)
rows = []
for r in labels.iter_rows(named=True):
    if r["label"] != 1:
        continue
    a, b, pid = r["player_1"], r["player_2"], r["pair_id"]
    sub = seats_dev.filter(pl.col("player_id").is_in([a, b]))
    by_hand = sub.group_by("hand_id").agg(
        pl.col("player_id").n_unique().alias("np"), pl.col("hand_idx").first().alias("hand_idx")
    )
    for h, hi in by_hand.filter(pl.col("np") == 2).select(["hand_id", "hand_idx"]).iter_rows():
        rows.append((pid, h, hi))
ph = pl.DataFrame(rows, schema=["pair_id", "hand_id", "hand_idx"], orient="row")

ev = evidence.join(dev_hands.select(["hand_id", "hand_idx"]), on="hand_id")
ev_map = {}
player_ev = defaultdict(list)  # player -> list of (pair_id, hand_idx)
for r in ev.iter_rows(named=True):
    pid, hid, hi = r["pair_id"], r["hand_id"], r["hand_idx"]
    ev_map[(pid, hid)] = hi
    a, b = pair_players[pid]
    player_ev[a].append((pid, hi))
    player_ev[b].append((pid, hi))

ph = ph.with_columns(pl.struct(["pair_id", "hand_id"]).map_elements(
    lambda s: (s["pair_id"], s["hand_id"]) in ev_map, return_dtype=pl.Boolean
).alias("is_ev"))

hi_arr = ph["hand_idx"].to_numpy()
pid_arr = ph["pair_id"].to_list()
is_ev_arr = ph["is_ev"].to_numpy()
n = ph.height

# cross-pair window flag: another positive pair sharing a player has evidence within K
player_ev_arrays = {p: np.array(sorted({hi for _, hi in v}), dtype=np.int64) for p, v in player_ev.items()}
pair_ev_idx = defaultdict(list)
for r in ev.iter_rows(named=True):
    pair_ev_idx[r["pair_id"]].append(r["hand_idx"])

K = 50
in_window = np.zeros(n, dtype=bool)
min_gap = np.full(n, np.inf)
for i in range(n):
    pid = pid_arr[i]
    hi = hi_arr[i]
    a, b = pair_players[pid]
    best = np.inf
    for q in player_pairs[a] + player_pairs[b]:
        if q == pid:
            continue
        arr = pair_ev_idx.get(q)
        if arr is None:
            continue
        d = float(np.min(np.abs(np.array(arr) - hi)))
        if d < best:
            best = d
    min_gap[i] = best
    in_window[i] = best <= K

ph = ph.with_columns(pl.Series("in_window", in_window))
base = is_ev_arr.mean()
lift = float(is_ev_arr[in_window].mean() / base) if in_window.any() else None
frac_window = float(in_window.mean())

# null: permute evidence labels within each pair's own shared hands
rng = np.random.RandomState(17)
perm_hi = np.array(hi_arr, dtype=np.int64)
for pid in np.unique(np.array(pid_arr)):
    m = np.array(pid_arr) == pid
    perm_hi[m] = rng.permutation(perm_hi[m])
null_min_gap = np.full(n, np.inf)
pair_ev_set = {pid: set(v) for pid, v in pair_ev_idx.items()}
for i in range(n):
    pid = pid_arr[i]
    a, b = pair_players[pid]
    hi = perm_hi[i]
    best = np.inf
    for q in player_pairs[a] + player_pairs[b]:
        if q == pid:
            continue
        arr = pair_ev_idx.get(q)
        if arr is None:
            continue
        d = float(np.min(np.abs(np.array(arr) - hi)))
        if d < best:
            best = d
    null_min_gap[i] = best
null_in_window = null_min_gap <= K

# exploitability: does the window flag add signal inside the model's top candidates?
key2pair = {}
for r in labels.iter_rows(named=True):
    key2pair[tuple(sorted([r["player_1"], r["player_2"]]))] = r["pair_id"]
narrow = pl.read_csv(NARROW, columns=["slot", "pair_player_lo", "pair_player_hi", "hand_id", "rank_u_r5b", "ev"])
narrow = narrow.with_columns(
    pl.struct(["pair_player_lo", "pair_player_hi"]).map_elements(
        lambda s: key2pair.get(tuple(sorted([s["pair_player_lo"], s["pair_player_hi"]])), None),
        return_dtype=pl.Utf8,
    ).alias("pair_id")
)
print("narrow rows with pair_id:", narrow.filter(pl.col("pair_id").is_not_null()).height, "/", narrow.height)

nm = narrow.join(ph.select(["pair_id", "hand_id", "in_window"]), on=["pair_id", "hand_id"], how="left")
res = {"K": K, "n_rows": n}
res["base_rate"] = float(base)
res["frac_in_window"] = frac_window
res["P_ev_given_window"] = float(is_ev_arr[in_window].mean()) if in_window.any() else None
res["lift_window"] = lift
res["null_frac_in_window"] = float(null_in_window.mean())
res["null_P_ev_given_window"] = float(is_ev_arr[null_in_window].mean()) if null_in_window.any() else None
res["null_lift_window"] = float(is_ev_arr[null_in_window].mean() / base) if null_in_window.any() else None
res["median_min_gap_obs"] = float(np.median(min_gap[np.isfinite(min_gap)]))
res["median_min_gap_null"] = float(np.median(null_min_gap[np.isfinite(null_min_gap)]))

# within narrow top-12
for rk in (5, 12, 20):
    sel = nm.filter(pl.col("rank_u_r5b") <= rk)
    if sel.height == 0:
        continue
    sel = sel.fill_null(False)
    y = sel["ev"].to_numpy().astype(bool)
    w = sel["in_window"].to_numpy().astype(bool)
    res[f"top{rk}_base"] = float(y.mean())
    res[f"top{rk}_P_ev_given_window"] = float(y[w].mean()) if w.any() else None
    res[f"top{rk}_lift"] = float(y[w].mean() / y.mean()) if w.any() and y.mean() > 0 else None
    res[f"top{rk}_frac_window"] = float(w.mean())

(OUT / "x10_window.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
