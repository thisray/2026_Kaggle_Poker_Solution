"""Late selected-solution evidence stages.

The recovered production spine produces the shared pair ensemble, family
router, and full-hand evidence cache.  This module adds the narrow R18
right-censored coordinated-isolation patch without depending on saved
prediction CSVs.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit

FAMILY = "coordinated_isolation"
FEATURES = [
    "actor_net", "partner_net", "actor_win", "partner_win",
    "actor_fold", "partner_fold", "pair_net", "sd_any", "iso",
    "pair_win", "xfer", "own", "par", "y1", "y2", "o_fold_after",
    "o_in_at_trig", "o_callraise_after", "end_pre", "o_alive_endpre",
    "o_vol_before", "pa_at_trig", "out_won", "out_sd",
    "all_out_folded", "both_flop", "card_gap", "all_in_at_trigger",
    "o_fold_fraction",
]
MODEL_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.035,
    "num_leaves": 15,
    "min_child_samples": 40,
    "reg_lambda": 20.0,
    "colsample_bytree": 0.85,
    "verbosity": -1,
    "random_state": 27,
}
EVIDENCE_COLUMNS = [f"evidence_hand_{rank}" for rank in range(1, 6)]


@njit(cache=True)
def _sequence_features(hands, seat_a, seat_b, offsets, action_seat, street, action, players_active, dealt):
    output = np.zeros((len(hands), 11), np.int64)
    for row in range(len(hands)):
        hand = hands[row]
        first = -1
        for index in range(offsets[hand], offsets[hand + 1]):
            if street[index] != 0:
                break
            if action_seat[index] == seat_a[row] or action_seat[index] == seat_b[row]:
                first = index
                break
        if first < 0:
            output[row, 0] = -1
            continue
        actor = action_seat[first]
        partner = seat_b[row] if actor == seat_a[row] else seat_a[row]
        output[row, 0] = 0 if actor == seat_a[row] else 1
        output[row, 1] = action[first]
        output[row, 10] = players_active[first]
        alive = dealt[row].copy()
        outsider_voluntary = 0
        for index in range(offsets[hand], first):
            if action[index] == 0:
                alive[action_seat[index]] = 0
            elif action_seat[index] != seat_a[row] and action_seat[index] != seat_b[row] and action[index] >= 2:
                outsider_voluntary += 1
        output[row, 9] = outsider_voluntary
        for seat in range(6):
            if alive[seat] == 1 and seat != seat_a[row] and seat != seat_b[row]:
                output[row, 4] += 1
        partner_first = -1
        outsider_folds = 0
        outsider_continues = 0
        last_street = 0
        for index in range(first + 1, offsets[hand + 1]):
            last_street = street[index]
            if street[index] != 0:
                continue
            seat = action_seat[index]
            if seat == partner and partner_first < 0:
                partner_first = action[index]
            if seat != seat_a[row] and seat != seat_b[row]:
                if action[index] == 0:
                    outsider_folds += 1
                    alive[seat] = 0
                elif action[index] >= 2:
                    outsider_continues += 1
        output[row, 2] = partner_first
        output[row, 3] = outsider_folds
        output[row, 5] = outsider_continues
        output[row, 6] = 1 if last_street == 0 else 0
        for seat in range(6):
            if alive[seat] == 1 and seat != seat_a[row] and seat != seat_b[row]:
                output[row, 7] += 1
    return output


def _slot_table(work_dir: Path, data_dir: Path, phase: str) -> pd.DataFrame:
    import pairindex as pair_index

    source = "development_labels.csv" if phase == "development" else "evaluation_pairs.csv"
    frame = pd.read_csv(data_dir / source, dtype={"pair_id": str})
    player_index = pd.read_parquet(work_dir / "np/player_index.parquet")
    player_map = dict(zip(player_index.player_id, player_index.pi))
    local = pd.read_parquet(work_dir / "player_local_v1.parquet").set_index("player_gi")
    low = np.minimum(frame.player_1.map(player_map), frame.player_2.map(player_map)).astype(int)
    high = np.maximum(frame.player_1.map(player_map), frame.player_2.map(player_map)).astype(int)
    pool = local.pool.loc[low].to_numpy(int)
    frame["slot"] = pair_index.pair_slot(
        pool,
        local.local.loc[low].to_numpy(int),
        local.local.loc[high].to_numpy(int),
    )
    return frame


def _shared_hands(work_dir: Path, pair_slots: pd.DataFrame, phase: int) -> pd.DataFrame:
    import pairindex as pair_index

    hands, _, _, slots = pair_index.all_pair_hands(phase)
    keep = np.isin(slots, pair_slots.slot.to_numpy(int))
    hand_index = pd.read_parquet(work_dir / "np/hand_index.parquet").set_index("hi").hand_id
    result = pd.DataFrame({"slot": slots[keep], "h": hands[keep]})
    result["hand_id"] = result.h.map(hand_index)
    result = result.merge(pair_slots[["slot", "pair_id"]], on="slot", validate="many_to_one")
    return result


def _extract_features(work_dir: Path, keyed_hands: pd.DataFrame) -> pd.DataFrame:
    array_dir = work_dir / "np"
    load = lambda name: np.load(array_dir / f"{name}.npy", mmap_mode="r")
    seats = load("s_player")
    net = load("s_net")
    won = load("s_won")
    showdown = load("s_sd")
    folded = load("s_folded")
    big_blind = load("h_bb")
    player_features = np.load(work_dir / "P_v1.npy", mmap_mode="r")
    relation_features = np.load(work_dir / "R_v1.npy", mmap_mode="r")
    feature_lines = (work_dir / "feature_names_v1.txt").read_text().splitlines()
    relation_names = feature_lines[0][2:].split(",")
    player_names = feature_lines[1][2:].split(",")

    local = pd.read_parquet(work_dir / "player_local_v1.parquet")
    members = np.full((int(local.pool.max()) + 1, 30), -1, np.int64)
    members[local.pool.to_numpy(int), local.local.to_numpy(int)] = local.player_gi.to_numpy(int)

    frame = keyed_hands.copy().reset_index(drop=True)
    hand = frame.h.to_numpy(np.int64)
    slot = frame.slot.to_numpy(np.int64)
    player_a = members[slot // 900, (slot % 900) // 30]
    player_b = members[slot // 900, slot % 30]
    hand_seats = np.asarray(seats[hand])
    seat_a = np.argmax(hand_seats == player_a[:, None], axis=1)
    seat_b = np.argmax(hand_seats == player_b[:, None], axis=1)
    row = np.arange(len(hand))
    if not ((hand_seats[row, seat_a] == player_a) & (hand_seats[row, seat_b] == player_b) & (seat_a != seat_b)).all():
        raise ValueError("Invalid player-slot/shared-hand mapping")

    blinds = np.asarray(big_blind[hand], float)
    net_values = np.asarray(net[hand], float)
    won_values = np.asarray(won[hand])
    showdown_values = np.asarray(showdown[hand])
    folded_values = np.asarray(folded[hand])
    frame["na"] = net_values[row, seat_a] / blinds
    frame["nb"] = net_values[row, seat_b] / blinds
    frame["pair_net"] = frame.na + frame.nb
    frame["wa"] = won_values[row, seat_a] > 0
    frame["wb"] = won_values[row, seat_b] > 0
    frame["pair_win"] = frame.wa | frame.wb
    frame["sd_any"] = showdown_values.sum(axis=1) > 0
    frame["fa"] = folded_values[row, seat_a] > 0
    frame["fb"] = folded_values[row, seat_b] > 0

    def relation(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
        return np.asarray(relation_features[hand, left, right, relation_names.index(name)], float)

    frame["iso"] = relation(seat_a, seat_b, "iso_ofold") + relation(seat_b, seat_a, "iso_ofold")
    frame["xfer"] = (relation(seat_a, seat_b, "flow") != 0) | (relation(seat_b, seat_a, "flow") != 0)
    action_path = work_dir / "dec_Y.npy"
    if action_path.exists():
        action = np.load(action_path, mmap_mode="r")
    else:
        raw_action = np.load(work_dir / "np/a_act.npy", mmap_mode="r")
        amount = np.load(work_dir / "np/a_amount.npy", mmap_mode="r")
        to_call = np.load(work_dir / "np/a_to_call.npy", mmap_mode="r")
        aggressive = np.isin(raw_action, (3, 4)) | ((raw_action == 5) & (amount > to_call))
        action = np.where(raw_action == 0, 0, np.where(raw_action == 1, 1, np.where(aggressive, 3, 2))).astype(np.int8)
    sequence = _sequence_features(
        hand,
        seat_a.astype(np.int64),
        seat_b.astype(np.int64),
        load("a_off"),
        load("a_seat"),
        load("a_st"),
        action,
        load("a_players_active"),
        (hand_seats >= 0).astype(np.int64),
    )
    sequence_columns = [
        "who", "y1", "y2", "o_fold_after", "o_in_at_trig",
        "o_callraise_after", "end_pre", "o_alive_endpre", "_unused",
        "o_vol_before", "pa_at_trig",
    ]
    for index, column in enumerate(sequence_columns):
        frame[column] = sequence[:, index]

    equity = np.asarray(player_features[hand, :, player_names.index("pf_eq_rand")], np.float32)
    last_street = np.asarray(player_features[hand, :, player_names.index("last_street")], np.float32)
    frame["own"] = np.where(frame.who == 0, equity[row, seat_a], equity[row, seat_b])
    frame["par"] = np.where(frame.who == 0, equity[row, seat_b], equity[row, seat_a])
    outsider_won = np.zeros(len(hand), bool)
    outsider_showdown = np.zeros(len(hand), bool)
    for seat in range(6):
        outsider = (seat != seat_a) & (seat != seat_b) & (hand_seats[:, seat] >= 0)
        outsider_won |= outsider & (won_values[:, seat] > 0)
        outsider_showdown |= outsider & (showdown_values[:, seat] > 0)
    frame["out_won"] = outsider_won
    frame["out_sd"] = outsider_showdown
    frame["all_out_folded"] = (~outsider_won) & (~outsider_showdown)
    frame["both_flop"] = np.minimum(last_street[row, seat_a], last_street[row, seat_b]) >= 1
    frame["ts"] = np.asarray(load("h_ts")[hand])
    return frame.drop(columns="_unused")


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["pool"] = result.slot // 900
    for stem, left, right in (
        ("net", "na", "nb"),
        ("win", "wa", "wb"),
        ("fold", "fa", "fb"),
    ):
        result[f"actor_{stem}"] = np.where(result.who == 0, result[left], result[right])
        result[f"partner_{stem}"] = np.where(result.who == 0, result[right], result[left])
    result["card_gap"] = result.par - result.own
    result["all_in_at_trigger"] = result.pa_at_trig == 6
    result["o_fold_fraction"] = result.o_fold_after / (1 + result.o_in_at_trig)
    missing = set(FEATURES + ["ts"]) - set(result)
    if missing:
        raise ValueError(f"Missing R18 gameplay columns: {sorted(missing)}")
    return result


def _uncensored_rows(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("slot").ev.sum()
    if (counts > 5).any() or (counts < 1).any():
        raise ValueError("R18 expects positive pairs with one to five listed hands")
    last = frame[frame.ev.astype(bool)].groupby("slot").ts.max()
    return ((frame.ts <= frame.slot.map(last)) | (frame.slot.map(counts) < 5)).to_numpy()


def _first_five_marginal(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    output = np.zeros(len(frame), float)
    ordered = frame.sort_values(["slot", "ts", "h"], kind="stable")
    for _, group in ordered.groupby("slot", sort=False):
        distribution = np.zeros(5, float)
        distribution[0] = 1.0
        for index in group.index:
            value = float(np.clip(probability[index], 0.0, 1.0))
            output[index] = value * distribution.sum()
            distribution = np.r_[
                distribution[0] * (1 - value),
                distribution[1:] * (1 - value) + distribution[:-1] * value,
            ]
    return output


def prepare_ci_patch(
    work_dir: Path,
    data_dir: Path,
    evidence_cache: Path,
    output_path: Path,
    threads: int,
) -> dict[str, object]:
    """Fit R18 from development truth and write an evaluation-only patch."""
    development = _slot_table(work_dir, data_dir, "development")
    development = development[
        development.label.eq(1) & development.behavior_family.eq(FAMILY)
    ][["slot", "pair_id"]]
    dev_hands = _shared_hands(work_dir, development, phase=0)
    truth = pd.read_csv(data_dir / "development_evidence.csv", dtype=str)
    truth_set = set(zip(truth.pair_id, truth.hand_id))
    dev_hands["ev"] = [
        int((pair_id, hand_id) in truth_set)
        for pair_id, hand_id in zip(dev_hands.pair_id, dev_hands.hand_id)
    ]
    dev_hands["fam"] = FAMILY

    evaluation = _slot_table(work_dir, data_dir, "evaluation")
    player_index = pd.read_parquet(work_dir / "np/player_index.parquet")
    player_map = dict(zip(player_index.player_id, player_index.pi))
    low = np.minimum(evaluation.player_1.map(player_map), evaluation.player_2.map(player_map))
    high = np.maximum(evaluation.player_1.map(player_map), evaluation.player_2.map(player_map))
    evaluation["key"] = low * 12000 + high
    family = pd.read_parquet(work_dir / "m7_family_eval.parquet")[["key", "family"]]
    evaluation = evaluation.merge(family, on="key", validate="one_to_one")
    # Extract every CI-routed pair while the large R/P gameplay arrays are
    # still available.  The exact top-4,000 risk gate is applied later, after
    # the final three-model rank fusion has been trained.
    ci_pairs = evaluation[evaluation.family.eq(FAMILY)][["slot", "pair_id"]]
    eval_hands = _shared_hands(work_dir, ci_pairs, phase=1)
    eval_hands["fam"] = FAMILY

    development_features = _prepare(_extract_features(work_dir, dev_hands))
    evaluation_features = _prepare(_extract_features(work_dir, eval_hands))
    include = _uncensored_rows(development_features)
    model = lgb.LGBMClassifier(**MODEL_PARAMS, n_jobs=threads)
    model.fit(
        development_features.loc[include, FEATURES].astype(float),
        development_features.loc[include, "ev"].astype(int),
    )
    probability = model.predict_proba(evaluation_features[FEATURES].astype(float))[:, 1]
    evaluation_features["q"] = _first_five_marginal(evaluation_features, probability)

    cache = pd.read_parquet(evidence_cache)
    cache = cache[cache.slot.isin(set(ci_pairs.slot))].copy()
    cache["base_rank"] = cache.groupby("slot").s1.rank(ascending=False, method="first")
    candidates = cache[cache.base_rank.le(20)][["slot", "h", "s1", "base_rank"]]
    candidates = candidates.merge(
        evaluation_features[["slot", "h", "pair_id", "hand_id", "ts", "q", "y1", "pa_at_trig"]],
        on=["slot", "h"],
        validate="one_to_one",
    )
    candidates["base_pct"] = (-candidates.base_rank).groupby(candidates.slot).rank(pct=True)
    candidates["q_pct"] = candidates.groupby("slot").q.rank(pct=True)
    candidates["score"] = 0.5 * candidates.base_pct + 0.5 * candidates.q_pct
    candidates.loc[~((candidates.pa_at_trig == 6) & candidates.y1.isin([2, 3])), "score"] -= 100.0
    top = candidates.sort_values(
        ["slot", "score", "ts", "h"],
        ascending=[True, False, True, True],
        kind="stable",
    ).groupby("slot", sort=False).head(5)
    top["rank"] = top.groupby("pair_id", sort=False).cumcount() + 1
    patch = top.pivot(index="pair_id", columns="rank", values="hand_id")
    patch.columns = [f"evidence_hand_{rank}" for rank in patch.columns]
    for column in EVIDENCE_COLUMNS:
        if column not in patch:
            patch[column] = "NO_EVIDENCE"
    patch = patch[EVIDENCE_COLUMNS].reset_index()
    patch.to_csv(output_path, index=False)
    model.booster_.save_model(str(output_path.with_suffix(".lgb.txt")))
    receipt = {
        "development_ci_pairs": int(development.slot.nunique()),
        "development_hands": len(development_features),
        "uncensored_training_hands": int(include.sum()),
        "evaluation_ci_pairs_extracted": int(ci_pairs.slot.nunique()),
        "evaluation_hands": len(evaluation_features),
        "candidate_rows": len(candidates),
        "patch_rows": len(patch),
    }
    output_path.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def apply_evidence_patch(
    submission: pd.DataFrame,
    patch_path: Path,
    gate_top: int = 4000,
) -> tuple[pd.DataFrame, int]:
    patch = pd.read_csv(patch_path, dtype=str).set_index("pair_id")
    output = submission.set_index("pair_id").copy()
    routed = output.predicted_behavior.eq(FAMILY)
    gated = set(output.nlargest(gate_top, "risk_score").index)
    pair_ids = patch.index.intersection(output.index[routed & output.index.isin(gated)])
    before = output.loc[pair_ids, EVIDENCE_COLUMNS].copy()
    output.loc[pair_ids, EVIDENCE_COLUMNS] = patch.loc[pair_ids, EVIDENCE_COLUMNS]
    changed = int((before.to_numpy() != output.loc[pair_ids, EVIDENCE_COLUMNS].to_numpy()).any(axis=1).sum())
    return output.reset_index(), changed
