"""Placebo, position, street/stake, and card-removal audits for the fourth-family signal."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (  # noqa: E402
    AUDIT_ART,
    OUT,
    ensure_audit_artifact_dir,
    first_preflop_indices,
    load_core_arrays,
    pair_slot_from_players,
    player_maps,
    read_candidate,
    read_evaluation_pairs,
)


TARGET_FILE = "r2d_p2comb_other.csv"


@njit(cache=True)
def first_action_by_street(off, a_seat, a_st):
    """Return first action index by hand, seat, and street."""
    n_hands = len(off) - 1
    out = -np.ones((n_hands, 6, 4), dtype=np.int64)
    for h in range(n_hands):
        for k in range(off[h], off[h + 1]):
            seat = a_seat[k]
            street = a_st[k]
            if out[h, seat, street] < 0:
                out[h, seat, street] = k
    return out


def score_aggregate(keys, rf, vf, ra, va, n_groups):
    """Aggregate residual scores by group and direction."""
    n_keys = n_groups * 2
    sum_rf = np.bincount(keys, weights=rf, minlength=n_keys)
    sum_vf = np.bincount(keys, weights=vf, minlength=n_keys)
    sum_ra = np.bincount(keys, weights=ra, minlength=n_keys)
    sum_va = np.bincount(keys, weights=va, minlength=n_keys)
    n = np.bincount(keys, minlength=n_keys)
    zf = sum_rf / np.sqrt(sum_vf + 1e-9)
    za = sum_ra / np.sqrt(sum_va + 1e-9)
    delta = za - zf
    result = pd.DataFrame(
        {
            "group": np.repeat(np.arange(n_groups), 2),
            "direction": np.tile([0, 1], n_groups),
            "n": n,
            "zf": zf,
            "za": za,
            "delta": delta,
        }
    )
    wide = result.pivot(index="group", columns="direction", values="delta")
    result["p2"] = np.nan
    for group in range(n_groups):
        if group in wide.index and 0 in wide.columns and 1 in wide.columns:
            value = wide.loc[group, [0, 1]].to_numpy(dtype=float)
            if np.isfinite(value).all() and (result.loc[result.group == group, "n"] > 0).all():
                result.loc[result.group == group, "p2"] = float(value.min())
    return result


def event_value(data, first, h, actor_seat, partner_seat, mu, required_street=None):
    """Build one Rao-score contribution when actor acts before partner."""
    if required_street is None:
        actor_k = int(first[h, actor_seat])
        partner_k = int(first[h, partner_seat])
    else:
        actor_k = int(required_street[h, actor_seat])
        partner_k = int(required_street[h, partner_seat])
    if actor_k < 0 or (partner_k >= 0 and partner_k < actor_k):
        return None
    pf = float(data["probs"][actor_k, 0])
    pa = float(data["probs"][actor_k, 3])
    y = int(data["y"][actor_k])
    rf = (1.0 if y == 0 else 0.0) - pf
    ra = (1.0 if y == 3 else 0.0) - pa
    return {
        "h": h,
        "actor_seat": int(actor_seat),
        "partner_seat": int(partner_seat),
        "actor_k": actor_k,
        "pf": pf,
        "pa": pa,
        "rf": rf,
        "ra": ra,
        "vf": pf * (1.0 - pf),
        "va": pa * (1.0 - pa),
    }


def target_pairs(data: dict):
    candidate = read_candidate(TARGET_FILE)
    target = candidate[candidate.predicted_behavior == "other_coordination"].copy()
    eval_pairs = read_evaluation_pairs().set_index("pair_id")
    target = target.join(eval_pairs[["player_1", "player_2", "shared_hands"]], on="pair_id", how="inner")
    pi_by_id, id_by_pi, pool_by_pi, local_by_pi = player_maps(data)
    target["p1_pi"] = target.player_1.map(pi_by_id).astype(int)
    target["p2_pi"] = target.player_2.map(pi_by_id).astype(int)
    target["slot"] = [
        pair_slot_from_players(pool_by_pi, local_by_pi, int(a), int(b))
        for a, b in zip(target.p1_pi, target.p2_pi)
    ]
    target = target.reset_index(drop=True)
    return target, pi_by_id, id_by_pi, pool_by_pi, local_by_pi


def shared_eval_hands(data: dict, p1_pi: int, p2_pi: int):
    phase_hands = np.flatnonzero(np.asarray(data["h_phase"]) == 1)
    seated = np.asarray(data["s_player"])[phase_hands]
    mask = (seated == p1_pi).any(axis=1) & (seated == p2_pi).any(axis=1)
    return phase_hands[mask]


def enrich_member_event(data, row, first, h, actor_pi, partner_pi, direction, mu, source, street):
    seats = np.asarray(data["s_player"][h])
    actor_seat = int(np.flatnonzero(seats == actor_pi)[0])
    partner_seat = int(np.flatnonzero(seats == partner_pi)[0])
    action = event_value(data, first, h, actor_seat, partner_seat, mu)
    if action is None:
        return None
    action.update(
        {
            "pair_id": row.pair_id,
            "pair_index": int(row.pair_index),
            "direction": int(direction),
            "actor_pi": int(actor_pi),
            "partner_pi": int(partner_pi),
            "source": source,
            "street": int(street),
            "bb": int(data["h_bb"][h]),
            "sb": int(data["h_sb"][h]),
            "button": int(data["h_btn"][h]),
        }
    )
    return action


def build_member_events(data, target, first, first_street, mu):
    rows = []
    for row in target.itertuples():
        hands = shared_eval_hands(data, int(row.p1_pi), int(row.p2_pi))
        for h in hands:
            h = int(h)
            for direction, actor, partner in [
                (0, int(row.p1_pi), int(row.p2_pi)),
                (1, int(row.p2_pi), int(row.p1_pi)),
            ]:
                event = enrich_member_event(data, row, first, h, actor, partner, direction, mu, "member", 0)
                if event is not None:
                    rows.append(event)
                for street in [1, 2, 3]:
                    event = event_value(
                        data,
                        first,
                        h,
                        int(np.flatnonzero(data["s_player"][h] == actor)[0]),
                        int(np.flatnonzero(data["s_player"][h] == partner)[0]),
                        mu,
                        required_street=first_street[:, :, street],
                    )
                    if event is None:
                        continue
                    event.update(
                        {
                            "pair_id": row.pair_id,
                            "pair_index": int(row.pair_index),
                            "direction": int(direction),
                            "actor_pi": actor,
                            "partner_pi": partner,
                            "source": "member",
                            "street": street,
                            "bb": int(data["h_bb"][h]),
                            "sb": int(data["h_sb"][h]),
                            "button": int(data["h_btn"][h]),
                        }
                    )
                    rows.append(event)
    return pd.DataFrame(rows)


def build_pseudo_events(data, target, first, mu):
    """Use non-partner players seated in the exact member-pair hands as pseudo partners."""
    rows = []
    group_keys = []
    group_map = {}
    for row in target.itertuples():
        p1 = int(row.p1_pi)
        p2 = int(row.p2_pi)
        hands = shared_eval_hands(data, p1, p2)
        for anchor_role, anchor in [("p1", p1), ("p2", p2)]:
            for h in hands:
                h = int(h)
                seats = np.asarray(data["s_player"][h])
                for pseudo in seats:
                    pseudo = int(pseudo)
                    if pseudo in (p1, p2):
                        continue
                    key = (row.pair_id, anchor_role, pseudo)
                    if key not in group_map:
                        group_map[key] = len(group_keys)
                        group_keys.append(key)
                    group = group_map[key]
                    anchor_seat = int(np.flatnonzero(seats == anchor)[0])
                    pseudo_seat = int(np.flatnonzero(seats == pseudo)[0])
                    for direction, actor_seat, partner_seat in [
                        (0, anchor_seat, pseudo_seat),
                        (1, pseudo_seat, anchor_seat),
                    ]:
                        action = event_value(data, first, h, actor_seat, partner_seat, mu)
                        if action is None:
                            continue
                        action.update(
                            {
                                "group": group,
                                "pair_id": row.pair_id,
                                "anchor_role": anchor_role,
                                "anchor_pi": anchor,
                                "pseudo_pi": pseudo,
                                "direction": direction,
                                "h": h,
                            }
                        )
                        rows.append(action)
    return pd.DataFrame(rows), group_keys


def add_draw_options(data, events):
    values = []
    for row in events.itertuples():
        seats = np.asarray(data["s_player"][int(row.h)])
        other = [seat for seat in range(6) if seat not in (int(row.actor_seat), int(row.partner_seat))]
        values.append([float(data["pfeq"][int(row.h), seat]) for seat in other])
    draw = np.asarray(values, dtype=np.float32)
    return draw


def p2_from_event_frame(events: pd.DataFrame, n_groups: int):
    if events.empty:
        return score_aggregate(
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            n_groups,
        )
    keys = (events["pair_index"].to_numpy(dtype=np.int64) * 2 + events["direction"].to_numpy(dtype=np.int64))
    return score_aggregate(
        keys,
        events.rf.to_numpy(float) * events.e.to_numpy(float),
        events.vf.to_numpy(float) * events.e.to_numpy(float) ** 2,
        events.ra.to_numpy(float) * events.e.to_numpy(float),
        events.va.to_numpy(float) * events.e.to_numpy(float) ** 2,
        n_groups,
    )


def add_partner_covariate(events: pd.DataFrame, data: dict, mu: float):
    result = events.copy()
    result["e"] = [float(data["pfeq"][int(h), int(s)]) - mu for h, s in zip(result.h, result.partner_seat)]
    return result


def strata_summary(events: pd.DataFrame, strata, label: str):
    if events.empty:
        return pd.DataFrame()
    work = events.copy()
    work["stratum"] = work[strata].astype(str).agg("/".join, axis=1)
    work["key"] = work.pair_index * 2 + work.direction
    grouped = (
        work.groupby(["stratum", "pair_index", "direction"], observed=True)
        .agg(
            n=("rf", "size"),
            sf=("rf", lambda x: 0.0),
        )
        .drop(columns="sf")
    )
    sums = (
        work.assign(
            rf_e=work.rf * work.e,
            vf_e2=work.vf * work.e * work.e,
            ra_e=work.ra * work.e,
            va_e2=work.va * work.e * work.e,
        )
        .groupby(["stratum", "pair_index", "direction"], observed=True)[["rf_e", "vf_e2", "ra_e", "va_e2"]]
        .sum()
    )
    grouped = grouped.join(sums)
    grouped["zf"] = grouped.rf_e / np.sqrt(grouped.vf_e2 + 1e-9)
    grouped["za"] = grouped.ra_e / np.sqrt(grouped.va_e2 + 1e-9)
    grouped["delta"] = grouped.za - grouped.zf
    pooled = (
        work.assign(
            rf_e=work.rf * work.e,
            vf_e2=work.vf * work.e * work.e,
            ra_e=work.ra * work.e,
            va_e2=work.va * work.e * work.e,
        )
        .groupby("stratum", observed=True)
        .agg(
            n_obs=("rf", "size"),
            n_pairs=("pair_index", "nunique"),
            rf_e=("rf_e", "sum"),
            vf_e2=("vf_e2", "sum"),
            ra_e=("ra_e", "sum"),
            va_e2=("va_e2", "sum"),
        )
    )
    pooled["pooled_zf"] = pooled.rf_e / np.sqrt(pooled.vf_e2 + 1e-9)
    pooled["pooled_za"] = pooled.ra_e / np.sqrt(pooled.va_e2 + 1e-9)
    pooled["pooled_delta"] = pooled.pooled_za - pooled.pooled_zf
    direction_stats = grouped.groupby("stratum", observed=True).agg(
        direction_groups=("delta", "size"),
        mean_direction_delta=("delta", "mean"),
        median_direction_delta=("delta", "median"),
        p95_direction_delta=("delta", lambda x: x.quantile(0.95)),
    )
    result = pooled.join(direction_stats).reset_index()
    result.insert(0, "label", label)
    return result


def main():
    ensure_audit_artifact_dir()
    data = load_core_arrays()
    mu = float(data["pfeq"].mean())
    target, _, _, _, _ = target_pairs(data)
    target["pair_index"] = np.arange(len(target), dtype=np.int64)
    first = first_preflop_indices(data["off"], data["a_seat"], data["a_st"])
    first_street = first_action_by_street(data["off"], data["a_seat"], data["a_st"])
    member_events = build_member_events(data, target, first, first_street, mu)
    member_events = add_partner_covariate(data=data, events=member_events, mu=mu)
    member_events.to_parquet(AUDIT_ART / "member_position_street_events.parquet", index=False)

    pre = member_events[member_events.street == 0].copy()
    original_scores = p2_from_event_frame(pre, len(target))
    original_scores = original_scores[original_scores.direction.isin([0, 1])]
    original_p2 = original_scores.pivot(index="group", columns="direction", values="delta")
    original_p2["p2"] = original_p2.min(axis=1)
    independent = pd.read_parquet(AUDIT_ART / "independent_p2_eval_with_risk.parquet")
    slots = target[["pair_index", "slot"]].merge(independent[["slot", "p2"]], on="slot", how="left")
    slot_p2 = slots.set_index("pair_index").p2
    p2_diff = original_p2.p2 - slot_p2

    pseudo_events, pseudo_groups = build_pseudo_events(data, target, first, mu)
    pseudo_events = add_partner_covariate(data=data, events=pseudo_events.rename(columns={"group": "pair_index"}), mu=mu)
    pseudo_events["group"] = pseudo_events["pair_index"]
    pseudo_scores = p2_from_event_frame(pseudo_events.rename(columns={"pair_index": "pair_index"}), len(pseudo_groups))
    pseudo_scores.to_csv(AUDIT_ART / "pseudo_partner_scores.csv", index=False)
    pseudo_detail = pd.DataFrame(pseudo_groups, columns=["member_pair_id", "anchor_role", "pseudo_pi"])
    pseudo_detail["group"] = np.arange(len(pseudo_detail))
    pseudo_detail = pseudo_detail.merge(
        pseudo_scores[pseudo_scores.direction == 0][["group", "n", "delta"]].rename(columns={"n": "n_dir0", "delta": "delta0"}),
        on="group",
        how="left",
    ).merge(
        pseudo_scores[pseudo_scores.direction == 1][["group", "n", "delta"]].rename(columns={"n": "n_dir1", "delta": "delta1"}),
        on="group",
        how="left",
    )
    has_both = (pseudo_detail.n_dir0 > 0) & (pseudo_detail.n_dir1 > 0)
    pseudo_detail["p2"] = np.where(
        has_both,
        pseudo_detail[["delta0", "delta1"]].min(axis=1),
        np.nan,
    )
    pseudo_detail.to_csv(AUDIT_ART / "pseudo_partner_detail.csv", index=False)

    strata = pd.concat(
        [
            strata_summary(pre, ["direction"], "direction_before_partner"),
            strata_summary(pre, ["bb"], "big_blind"),
            strata_summary(pre, ["actor_seat"], "actor_seat"),
            strata_summary(pre.assign(blind_role=np.select(
                [pre.actor_seat == pre.sb, pre.actor_seat == pre.bb], ["small_blind", "big_blind"], default="other"
            )), ["blind_role"], "blind_role"),
            strata_summary(pre, ["street"], "street"),
            strata_summary(member_events, ["street"], "street_all_first_actions"),
        ],
        ignore_index=True,
    )
    strata.to_csv(AUDIT_ART / "member_position_street_stake_summary.csv", index=False)

    # Each original preflop event gets four valid alternative cards: the other seats' actual cards.
    # This is a conditional card-removal draw because each alternative is disjoint from actor and board.
    draw_options = add_draw_options(data, pre)
    rng = np.random.default_rng(20260918)
    n_reps = 200
    n_events = len(pre)
    group_ids = pre.pair_index.to_numpy(np.int64)
    directions = pre.direction.to_numpy(np.int64)
    keys = group_ids * 2 + directions
    rf = pre.rf.to_numpy(float)
    vf = pre.vf.to_numpy(float)
    ra = pre.ra.to_numpy(float)
    va = pre.va.to_numpy(float)
    null_p2 = np.full((n_reps, len(target)), np.nan, dtype=float)
    null_delta = np.full((n_reps, len(target), 2), np.nan, dtype=float)
    for rep in range(n_reps):
        choice = rng.integers(0, draw_options.shape[1], size=n_events)
        e = draw_options[np.arange(n_events), choice].astype(float) - mu
        scores = score_aggregate(
            keys,
            rf * e,
            vf * e * e,
            ra * e,
            va * e * e,
            len(target),
        )
        wide = scores.pivot(index="group", columns="direction", values="delta")
        null_delta[rep, :, :] = wide.reindex(range(len(target))).reindex(columns=[0, 1]).to_numpy()
        null_p2[rep] = np.nanmin(null_delta[rep], axis=1)
    observed_p2 = original_p2.reindex(range(len(target))).p2.to_numpy()
    finite_null = np.isfinite(null_p2)
    card_summary = {
        "n_reps": n_reps,
        "n_member_pairs": int(len(target)),
        "n_preflop_events": int(n_events),
        "null_max_p2": float(np.nanmax(null_p2)),
        "null_p2_quantiles_all_pair_reps": {
            str(q): float(np.nanquantile(null_p2, q)) for q in [0.5, 0.95, 0.99, 0.999]
        },
        "null_p2_gt_4_pair_reps": int(np.nansum(null_p2 > 4.0)),
        "observed_p2_min": float(np.nanmin(observed_p2)),
        "observed_p2_median": float(np.nanmedian(observed_p2)),
        "observed_p2_max": float(np.nanmax(observed_p2)),
        "pairwise_null_ge_observed_fraction": float(
            np.nanmean(null_p2 >= observed_p2[None, :])
        ),
        "member_pairs_with_any_null_ge_observed": int(
            np.sum(np.nanmax(null_p2, axis=0) >= observed_p2)
        ),
        "finite_pair_replicates": int(finite_null.sum()),
    }
    (AUDIT_ART / "card_removal_summary.json").write_text(json.dumps(card_summary, indent=2, default=float))
    np.save(AUDIT_ART / "card_removal_null_p2.npy", null_p2)

    pseudo_finite = pseudo_detail[np.isfinite(pseudo_detail.p2)]
    pseudo_summary = {
        "n_target_members": int(len(target)),
        "n_pseudo_groups": int(len(pseudo_detail)),
        "n_pseudo_groups_both_directions": int(
            ((pseudo_detail.n_dir0 > 0) & (pseudo_detail.n_dir1 > 0)).sum()
        ),
        "p2_quantiles": {
            str(q): float(pseudo_finite.p2.quantile(q)) for q in [0.5, 0.9, 0.95, 0.99]
        },
        "p2_mean": float(pseudo_finite.p2.mean()),
        "p2_gt_2_5": int((pseudo_finite.p2 > 2.5).sum()),
        "p2_gt_4": int((pseudo_finite.p2 > 4.0).sum()),
        "p2_max": float(pseudo_finite.p2.max()),
        "member_p2_quantiles": {
            str(q): float(np.nanquantile(observed_p2, q)) for q in [0.5, 0.9, 0.95, 0.99]
        },
    }
    (AUDIT_ART / "pseudo_partner_summary.json").write_text(json.dumps(pseudo_summary, indent=2, default=float))

    summary = {
        "target_file": TARGET_FILE,
        "target_members": int(len(target)),
        "member_events": int(len(member_events)),
        "member_preflop_events": int(len(pre)),
        "member_vs_independent_p2_max_abs_diff": float(np.nanmax(np.abs(p2_diff.to_numpy()))),
        "member_vs_independent_p2_allclose_1e-8": bool(np.nanmax(np.abs(p2_diff.to_numpy())) <= 1e-8),
        "pseudo": pseudo_summary,
        "card_removal": card_summary,
        "position_street_stake_rows": int(len(strata)),
    }
    (AUDIT_ART / "placebo_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
