"""Round-10 probe: exact-condition anatomy of directed_transfer candidates.

For every DT candidate hand in the narrow table (top-20 by r5b), locate the
"folder folds facing partner" event, reconstruct the decision moment (street,
board prefix, pot, to_call), and compute exact equities with all 12 hole
cards known:
  * folder vs partner equity (full enumeration of remaining runouts);
  * folder rank among all six players at that street;
  * equity - pot_odds "excess";
  * folder/partner Chen scores.
Then measure separation between evidence (ev=1) and non-evidence (ev=0) and
compare against the existing MC-equity columns.

Read-only; writes one JSON receipt + one per-candidate parquet.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

import sys

sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/round10-research-20260918/code")
from pk_eval import eval_cards  # noqa: E402

RAW = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw")
NARROW = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917/r6_narrow_candidates_v2.csv")
OUT = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round10_research_20260918")
OUT.mkdir(parents=True, exist_ok=True)

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
CARD_MAP = {f"{r}{s}": i * 4 + j for i, r in enumerate(RANK_CHARS) for j, s in enumerate(SUIT_CHARS)}


def chen(c1, c2):
    r1, s1, r2, s2 = c1 >> 2, c1 & 3, c2 >> 2, c2 & 3
    hi, lo = max(r1, r2), min(r1, r2)
    base = {12: 10, 11: 8, 10: 7, 9: 6}.get(hi, (hi + 2) / 2)
    if r1 == r2:
        return max(5.0, base * 2)
    sc = base + (2 if s1 == s2 else 0)
    gap = hi - lo - 1
    sc -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 10:
        sc += 1
    return float(sc)


@njit(cache=True)
def _eq_hu(f1, f2, p1, p2, others, board, nb):
    """Exact 2-player equity for folder (f1,f2) vs partner (p1,p2) given board prefix nb.
    Enumerates all remaining board cards from the stub with all known hole cards removed."""
    stub = np.empty(52, np.int64)
    k = 0
    used = np.zeros(52, np.int8)
    for c in (f1, f2, p1, p2):
        used[c] = 1
    for c in others:
        used[c] = 1
    for i in range(nb):
        used[board[i]] = 1
    for c in range(52):
        if used[c] == 0:
            stub[k] = c
            k += 1
    need = 5 - nb
    wins = 0.0
    n = 0
    rc1 = np.zeros(13, np.int64); sc1 = np.zeros(4, np.int64); sm1 = np.zeros(4, np.int64)
    rc2 = np.zeros(13, np.int64); sc2 = np.zeros(4, np.int64); sm2 = np.zeros(4, np.int64)
    hc = np.empty(7, np.int64)
    if need == 0:
        cards = np.empty(7, np.int64)
        cards[0] = f1; cards[1] = f2
        for i in range(nb):
            cards[2 + i] = board[i]
        vf = eval_cards(cards, 7, rc1, sc1, sm1)
        cards[0] = p1; cards[1] = p2
        vp = eval_cards(cards, 7, rc2, sc2, sm2)
        return 1.0 if vf > vp else (0.5 if vf == vp else 0.0)
    if need == 1:
        for a in range(k):
            cards = np.empty(7, np.int64)
            cards[0] = f1; cards[1] = f2
            for i in range(nb):
                cards[2 + i] = board[i]
            cards[6] = stub[a]
            vf = eval_cards(cards, 7, rc1, sc1, sm1)
            cards[0] = p1; cards[1] = p2
            vp = eval_cards(cards, 7, rc2, sc2, sm2)
            wins += 1.0 if vf > vp else (0.5 if vf == vp else 0.0)
            n += 1
        return wins / n
    # need == 2
    for a in range(k):
        for b in range(a + 1, k):
            cards = np.empty(7, np.int64)
            cards[0] = f1; cards[1] = f2
            for i in range(nb):
                cards[2 + i] = board[i]
            cards[5] = stub[a]; cards[6] = stub[b]
            vf = eval_cards(cards, 7, rc1, sc1, sm1)
            cards[0] = p1; cards[1] = p2
            vp = eval_cards(cards, 7, rc2, sc2, sm2)
            wins += 1.0 if vf > vp else (0.5 if vf == vp else 0.0)
            n += 1
    return wins / n


@njit(cache=True)
def _street_value(h1, h2, board, nb):
    cards = np.empty(7, np.int64)
    cards[0] = h1; cards[1] = h2
    for i in range(nb):
        cards[2 + i] = board[i]
    rc = np.zeros(13, np.int64); sc = np.zeros(4, np.int64); sm = np.zeros(4, np.int64)
    return eval_cards(cards, 2 + nb, rc, sc, sm)


narrow = pl.read_csv(NARROW)
labels = pl.read_csv(RAW / "development_labels.csv")
key2pair = {}
for r in labels.iter_rows(named=True):
    key2pair[tuple(sorted([r["player_1"], r["player_2"]]))] = r["pair_id"]
narrow = narrow.with_columns(
    pl.struct(["pair_player_lo", "pair_player_hi"]).map_elements(
        lambda s: key2pair.get(tuple(sorted([s["pair_player_lo"], s["pair_player_hi"]])), None),
        return_dtype=pl.Utf8,
    ).alias("pair_id")
)
dt = narrow.filter(pl.col("family") == "directed_transfer")
hand_ids = dt["hand_id"].unique().to_list()
print("DT candidates:", dt.height, "unique hands:", len(hand_ids))

hands = (
    pl.read_parquet(RAW / "hands.parquet", columns=["hand_id", "board_cards"])
    .filter(pl.col("hand_id").is_in(hand_ids))
)
seats = (
    pl.read_parquet(RAW / "seats.parquet", columns=["hand_id", "player_id", "hole_card_1", "hole_card_2", "seat_no"])
    .filter(pl.col("hand_id").is_in(hand_ids))
)
actions = (
    pl.scan_parquet(RAW / "actions.parquet")
    .filter(pl.col("hand_id").is_in(hand_ids))
    .sort(["hand_id", "action_no"])
    .collect()
)
print("actions rows for DT hands:", actions.height)

board_map = {}
for r in hands.iter_rows(named=True):
    bc = r["board_cards"]
    cards = [] if not bc else [CARD_MAP[c] for c in bc.split()]
    board_map[r["hand_id"]] = cards
seat_map = {}
for r in seats.iter_rows(named=True):
    seat_map.setdefault(r["hand_id"], {})[r["player_id"]] = (
        CARD_MAP[r["hole_card_1"]],
        CARD_MAP[r["hole_card_2"]],
        r["seat_no"],
    )

BY_HAND = {}
for r in actions.iter_rows(named=True):
    BY_HAND.setdefault(r["hand_id"], []).append(r)

rows = []
for r in dt.iter_rows(named=True):
    hid, pid = r["hand_id"], r["pair_id"]
    a_id, b_id = r["pair_player_lo"], r["pair_player_hi"]
    sm = seat_map.get(hid, {})
    if a_id not in sm or b_id not in sm:
        continue
    board = board_map.get(hid, [])
    acts = BY_HAND.get(hid, [])
    # walk actions, track last aggressor + pot
    last_aggr = None
    pot = 0.0
    per_street_aggr = None
    best_excess = -99.0
    best = None
    for act in acts:
        street = act["street"]
        nb = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]
        if act["action"] in ("bet", "raise") or (act["action"] == "all_in" and (act["amount"] or 0) > (act["to_call"] or 0)):
            last_aggr = act["player_id"]
        if act["action"] == "fold" and act["player_id"] in (a_id, b_id):
            partner = b_id if act["player_id"] == a_id else a_id
            to_call = float(act["to_call"] or 0)
            pot_before = float(act["pot_before"] or 0)
            if to_call > 0 and last_aggr == partner:
                f1, f2, _ = sm[act["player_id"]]
                p1, p2, _ = sm[partner]
                others = np.array(
                    [c for pid2, (h1, h2, _) in sm.items() if pid2 not in (act["player_id"], partner) for c in (h1, h2)],
                    dtype=np.int64,
                )
                if nb >= 3:
                    eq = _eq_hu(f1, f2, p1, p2, others, np.array(board[:nb], dtype=np.int64), nb)
                else:
                    eq = float("nan")
                pot_odds = to_call / (pot_before + to_call) if (pot_before + to_call) > 0 else float("nan")
                excess = eq - pot_odds if not np.isnan(eq) else float("nan")
                # folder rank among all six at this street
                vals = np.full(6, -1, dtype=np.int64)
                for i, (pl_id, (h1, h2, _)) in enumerate(sm.items()):
                    vals[i] = _street_value(h1, h2, np.array(board[:nb], dtype=np.int64), nb) if nb >= 3 else 0
                my_val = vals[list(sm.keys()).index(act["player_id"])]
                rk = int((vals > my_val).sum()) + 1
                rec = {
                    "pair_id": pid,
                    "hand_id": hid,
                    "family": r["family"],
                    "ev": int(r["ev"]),
                    "rank_u_r5b": int(r["rank_u_r5b"]),
                    "folder": act["player_id"],
                    "street": street,
                    "to_call_bb": to_call,
                    "pot_before_bb": pot_before,
                    "pot_odds": pot_odds,
                    "eq_exact": eq,
                    "excess": excess,
                    "folder_rank6": rk,
                    "chen_folder": chen(f1, f2),
                    "chen_partner": chen(p1, p2),
                    "nb": nb,
                    "n_active": act["players_active"],
                }
                if excess > best_excess or (np.isnan(best_excess) and best is None):
                    best_excess = excess
                    best = rec
    if best is not None:
        rows.append(best)

res = pl.DataFrame(rows)
res.write_parquet(OUT / "x10_dt_events.parquet")
print("rows with fold-to-partner event:", res.height, "/", dt.height)

import sklearn.metrics as M  # noqa: E402

rep = {"n_dt_candidates": dt.height, "n_with_fold_to_partner": res.height}
sub = res.filter(pl.col("eq_exact").is_not_nan())
y = sub["ev"].to_numpy().astype(int)
rep["n_postflop_events"] = int(sub.height)
rep["ev_rate"] = float(y.mean())
for col in ["eq_exact", "pot_odds", "excess", "folder_rank6", "chen_folder", "chen_partner", "nb", "n_active"]:
    x = sub[col].to_numpy().astype(float)
    m = np.isfinite(x)
    if m.sum() > 20 and len(np.unique(y[m])) > 1:
        rep[f"auc_{col}"] = float(M.roc_auc_score(y[m], x[m]))
# threshold scan on excess
xs = sub["excess"].to_numpy().astype(float)
best = {"thr": None, "f1": -1}
for thr in np.arange(-0.4, 0.75, 0.05):
    pred = xs >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp + fp == 0:
        continue
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    if f1 > best["f1"]:
        best = {"thr": float(thr), "precision": prec, "recall": rec, "f1": f1}
rep["excess_rule_best"] = best
# existing MC feature comparison (from narrow table)
mc = dt.select(["pair_id", "hand_id", "ev", "eq_fold_to_mx", "DS_ftp_eq_max", "z_eq_fold_to_mx"])
mc = mc.join(res.select(["pair_id", "hand_id", "excess"]), on=["pair_id", "hand_id"], how="left")
for col in ["eq_fold_to_mx", "DS_ftp_eq_max", "z_eq_fold_to_mx"]:
    x = mc[col].to_numpy().astype(float); yy = mc["ev"].to_numpy().astype(int)
    m = np.isfinite(x)
    rep[f"mc_auc_{col}"] = float(M.roc_auc_score(yy[m], x[m])) if m.sum() > 20 else None
print(json.dumps(rep, indent=1))
(OUT / "x10_dt_rule.json").write_text(json.dumps(rep, indent=1))
