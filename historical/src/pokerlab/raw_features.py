"""Raw competition audit and feature materialization for the poker task.

The implementation keeps the large joins inside DuckDB and writes compact
Parquet artifacts. It never treats identifiers as predictive features. All
features are derived from gameplay, verified metadata, or exposure counts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Iterable

import duckdb


RAW_FILES = {
    "players": "players.parquet",
    "hands": "hands.parquet",
    "seats": "seats.parquet",
    "actions": "actions.parquet",
    "development_labels": "development_labels.csv",
    "development_evidence": "development_evidence.csv",
    "evaluation_pairs": "evaluation_pairs.csv",
    "sample_submission": "sample_submission.csv",
}

KNOWN_FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")
PRESSURE_ACTIONS = ("bet", "raise", "all_in")
RESPONSE_ACTIONS = ("call", "raise", "all_in", "fold", "check")

HAND_FEATURE_COLUMNS = (
    "a_net_bb", "b_net_bb", "net_gap_bb", "transfer_ab_bb", "transfer_ba_bb",
    "transfer_max_bb", "transfer_sum_bb", "a_contrib_bb", "b_contrib_bb",
    "total_contrib_bb", "contrib_gap_bb", "pot_bb", "final_pot_bb",
    "players_dealt", "players_at_showdown", "showdown_both", "a_folded",
    "b_folded", "a_showdown", "b_showdown", "board_count", "board_rank_sum",
    "a_hole_rank_sum", "b_hole_rank_sum", "a_hole_rank_gap", "a_hole_pair",
    "b_hole_pair", "a_hole_suited", "b_hole_suited", "hole_rank_gap",
    "a_action_count", "b_action_count", "a_pressure_count", "b_pressure_count",
    "a_fold_count", "b_fold_count", "a_call_count", "b_call_count",
    "a_check_count", "b_check_count", "a_pressure_rate", "b_pressure_rate",
    "a_fold_rate", "b_fold_rate", "a_call_rate", "b_call_rate",
    "a_check_rate", "b_check_rate", "a_pressure_amount_bb",
    "b_pressure_amount_bb", "a_amount_to_bb", "b_amount_to_bb",
    "a_mean_players_active", "b_mean_players_active", "a_pressure_delta",
    "b_pressure_delta", "a_fold_delta", "b_fold_delta", "a_call_delta",
    "b_call_delta", "b_response_after_a", "a_response_after_b",
    "outsider_fold_after_a", "outsider_fold_after_b", "a_fold_after_b_pressure",
    "b_fold_after_a_pressure", "mutual_pressure_sequence", "a_first_pressure_pos",
    "b_first_pressure_pos", "hand_risk_proxy",
    "b_immediate_fold_after_a", "b_immediate_call_after_a",
    "b_immediate_raise_after_a", "a_immediate_fold_after_b",
    "a_immediate_call_after_b", "a_immediate_raise_after_b",
    "outsider_immediate_fold_after_a", "outsider_immediate_fold_after_b",
    "outsider_immediate_pressure_after_a", "outsider_immediate_pressure_after_b",
    "a_facing_call_opportunity", "b_facing_call_opportunity",
    "a_fold_facing_call", "b_fold_facing_call", "a_call_facing_call",
    "b_call_facing_call", "a_raise_facing_call", "b_raise_facing_call",
    "a_all_in_call", "b_all_in_call", "a_all_in_aggressive",
    "b_all_in_aggressive", "a_preflop_pressure", "b_preflop_pressure",
    "a_flop_pressure", "b_flop_pressure", "a_turn_pressure",
    "b_turn_pressure", "a_river_pressure", "b_river_pressure",
    "hand_risk_rank_pct", "transfer_max_rank_pct", "response_rank_pct",
    "outsider_rank_pct", "joint_showdown_rank_pct", "hand_risk_prev5_mean",
    "hand_risk_next5_mean", "hand_risk_window20_mean", "hand_risk_window50_mean",
    "hand_risk_window20_peak",
)


@dataclass(frozen=True)
class FeatureBuildConfig:
    data_dir: str
    output_dir: str
    threads: int = 16
    memory_limit: str = "80GB"


def _path(root: Path, name: str) -> Path:
    path = root / RAW_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sql_literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _rank_expr(column: str) -> str:
    return (
        f"CASE upper(substr({column}, 1, 1)) "
        "WHEN '2' THEN 2 WHEN '3' THEN 3 WHEN '4' THEN 4 "
        "WHEN '5' THEN 5 WHEN '6' THEN 6 WHEN '7' THEN 7 "
        "WHEN '8' THEN 8 WHEN '9' THEN 9 WHEN 'T' THEN 10 "
        "WHEN 'J' THEN 11 WHEN 'Q' THEN 12 WHEN 'K' THEN 13 "
        "WHEN 'A' THEN 14 ELSE NULL END"
    )


def _board_rank_sum(clean_column: str) -> str:
    terms = [_rank_expr(f"substr({clean_column}, {pos}, 1)") for pos in (1, 3, 5, 7, 9)]
    return "coalesce(" + ", 0) + coalesce(".join(terms) + ", 0)"


def _create_views(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    readers = {
        "players": f"read_parquet({_sql_literal(_path(root, 'players'))})",
        "hands": f"read_parquet({_sql_literal(_path(root, 'hands'))})",
        "seats": f"read_parquet({_sql_literal(_path(root, 'seats'))})",
        "actions": f"read_parquet({_sql_literal(_path(root, 'actions'))})",
        "development_labels": f"read_csv_auto({_sql_literal(_path(root, 'development_labels'))}, header=true)",
        "development_evidence": f"read_csv_auto({_sql_literal(_path(root, 'development_evidence'))}, header=true)",
        "evaluation_pairs": f"read_csv_auto({_sql_literal(_path(root, 'evaluation_pairs'))}, header=true)",
        "sample_submission": f"read_csv_auto({_sql_literal(_path(root, 'sample_submission'))}, header=true)",
    }
    for name, reader in readers.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {reader}")


def _create_shared_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE player_pool AS
        SELECT player_id, min(table_id) AS pool_id,
               count(DISTINCT table_id) AS pool_count
        FROM seats s JOIN hands h USING (hand_id)
        GROUP BY player_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE target_pairs AS
        SELECT l.pair_id, l.player_1, l.player_2, 'development' AS phase,
               p1.pool_id, CAST(NULL AS BIGINT) AS reported_shared_hands
        FROM development_labels l
        JOIN player_pool p1 ON p1.player_id = l.player_1
        JOIN player_pool p2 ON p2.player_id = l.player_2
        WHERE p1.pool_id = p2.pool_id
        UNION ALL
        SELECT e.pair_id, e.player_1, e.player_2, 'evaluation' AS phase,
               p1.pool_id, e.shared_hands AS reported_shared_hands
        FROM evaluation_pairs e
        JOIN player_pool p1 ON p1.player_id = e.player_1
        JOIN player_pool p2 ON p2.player_id = e.player_2
        WHERE p1.pool_id = p2.pool_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE action_player AS
        SELECT hand_id, player_id,
               count(*)::BIGINT AS action_count,
               count(*) FILTER (action = 'fold')::BIGINT AS fold_count,
               count(*) FILTER (action = 'call')::BIGINT AS call_count,
               count(*) FILTER (action = 'check')::BIGINT AS check_count,
               count(*) FILTER (action IN ('bet', 'raise', 'all_in'))::BIGINT AS pressure_count,
               sum(amount) FILTER (action IN ('bet', 'raise', 'all_in'))::DOUBLE AS pressure_amount,
               sum(amount)::DOUBLE AS amount_sum,
               sum(amount_to)::DOUBLE AS amount_to_sum,
               avg(pot_before)::DOUBLE AS mean_pot_before,
               avg(players_active)::DOUBLE AS mean_players_active,
               min(action_no)::BIGINT AS first_action_no,
               max(action_no)::BIGINT AS last_action_no
        FROM actions
        GROUP BY hand_id, player_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE player_phase_stats AS
        SELECT h.phase, a.player_id,
               count(*)::BIGINT AS action_hands,
               sum(a.action_count)::DOUBLE AS action_count,
               sum(a.fold_count)::DOUBLE AS fold_count,
               sum(a.call_count)::DOUBLE AS call_count,
               sum(a.pressure_count)::DOUBLE AS pressure_count
        FROM action_player a JOIN hands h USING (hand_id)
        GROUP BY h.phase, a.player_id
        """
    )


def _pair_hands_sql(phase: str) -> str:
    phase_name = phase.lower()
    return f"""
        CREATE OR REPLACE TABLE pair_hands_{phase_name} AS
        SELECT p.pair_id, p.player_1, p.player_2, p.phase, p.pool_id,
               p.reported_shared_hands, h.hand_id, h.table_id, h.started_at,
               h.big_blind, h.small_blind, h.board_cards, h.final_pot,
               h.players_dealt, h.players_at_showdown,
               s1.seat_no AS a_seat, s2.seat_no AS b_seat,
               s1.starting_stack AS a_starting_stack, s2.starting_stack AS b_starting_stack,
               s1.hole_card_1 AS a_hole_card_1, s1.hole_card_2 AS a_hole_card_2,
               s2.hole_card_1 AS b_hole_card_1, s2.hole_card_2 AS b_hole_card_2,
               s1.total_contribution AS a_total_contribution,
               s2.total_contribution AS b_total_contribution,
               s1.net_chips AS a_net_chips, s2.net_chips AS b_net_chips,
               s1.folded AS a_folded, s2.folded AS b_folded,
               s1.went_to_showdown AS a_showdown, s2.went_to_showdown AS b_showdown,
               s1.won_share AS a_won_share, s2.won_share AS b_won_share,
               coalesce(a1.action_count, 0)::DOUBLE AS a_action_count,
               coalesce(a2.action_count, 0)::DOUBLE AS b_action_count,
               coalesce(a1.fold_count, 0)::DOUBLE AS a_fold_count,
               coalesce(a2.fold_count, 0)::DOUBLE AS b_fold_count,
               coalesce(a1.call_count, 0)::DOUBLE AS a_call_count,
               coalesce(a2.call_count, 0)::DOUBLE AS b_call_count,
               coalesce(a1.check_count, 0)::DOUBLE AS a_check_count,
               coalesce(a2.check_count, 0)::DOUBLE AS b_check_count,
               coalesce(a1.pressure_count, 0)::DOUBLE AS a_pressure_count,
               coalesce(a2.pressure_count, 0)::DOUBLE AS b_pressure_count,
               coalesce(a1.pressure_amount, 0)::DOUBLE AS a_pressure_amount,
               coalesce(a2.pressure_amount, 0)::DOUBLE AS b_pressure_amount,
               coalesce(a1.amount_to_sum, 0)::DOUBLE AS a_amount_to_sum,
               coalesce(a2.amount_to_sum, 0)::DOUBLE AS b_amount_to_sum,
               coalesce(a1.mean_pot_before, 0)::DOUBLE AS a_mean_pot_before,
               coalesce(a2.mean_pot_before, 0)::DOUBLE AS b_mean_pot_before,
               coalesce(a1.mean_players_active, 0)::DOUBLE AS a_mean_players_active,
               coalesce(a2.mean_players_active, 0)::DOUBLE AS b_mean_players_active,
               coalesce(a1.first_action_no, 0)::DOUBLE AS a_first_action_no,
               coalesce(a2.first_action_no, 0)::DOUBLE AS b_first_action_no,
               coalesce(a1.last_action_no, 0)::DOUBLE AS a_last_action_no,
               coalesce(a2.last_action_no, 0)::DOUBLE AS b_last_action_no,
               coalesce(ps1.pressure_count / nullif(ps1.action_count, 0), 0)::DOUBLE AS a_background_pressure_rate,
               coalesce(ps2.pressure_count / nullif(ps2.action_count, 0), 0)::DOUBLE AS b_background_pressure_rate,
               coalesce(ps1.fold_count / nullif(ps1.action_count, 0), 0)::DOUBLE AS a_background_fold_rate,
               coalesce(ps2.fold_count / nullif(ps2.action_count, 0), 0)::DOUBLE AS b_background_fold_rate,
               coalesce(ps1.call_count / nullif(ps1.action_count, 0), 0)::DOUBLE AS a_background_call_rate,
               coalesce(ps2.call_count / nullif(ps2.action_count, 0), 0)::DOUBLE AS b_background_call_rate
        FROM target_pairs p
        JOIN seats s1 ON s1.player_id = p.player_1
        JOIN seats s2 ON s2.player_id = p.player_2 AND s2.hand_id = s1.hand_id
        JOIN hands h ON h.hand_id = s1.hand_id AND h.phase = p.phase AND h.table_id = p.pool_id
        LEFT JOIN action_player a1 ON a1.hand_id = h.hand_id AND a1.player_id = p.player_1
        LEFT JOIN action_player a2 ON a2.hand_id = h.hand_id AND a2.player_id = p.player_2
        LEFT JOIN player_phase_stats ps1 ON ps1.phase = p.phase AND ps1.player_id = p.player_1
        LEFT JOIN player_phase_stats ps2 ON ps2.phase = p.phase AND ps2.player_id = p.player_2
        WHERE p.phase = '{phase_name}'
        """


def _relationship_sql(phase: str) -> tuple[str, str]:
    phase_name = phase.lower()
    pressure = "('bet', 'raise', 'all_in')"
    response = "('call', 'raise', 'all_in', 'fold', 'check')"
    first_sql = f"""
        CREATE OR REPLACE TABLE pair_hand_first_{phase_name} AS
        SELECT ph.pair_id, ph.hand_id,
               min(a.action_no) FILTER (WHERE a.player_id = ph.player_1 AND a.action IN {pressure}) AS first_a_pressure,
               min(a.action_no) FILTER (WHERE a.player_id = ph.player_2 AND a.action IN {pressure}) AS first_b_pressure,
               min(a.action_no) FILTER (WHERE a.player_id = ph.player_1 AND a.action = 'fold') AS first_a_fold,
               min(a.action_no) FILTER (WHERE a.player_id = ph.player_2 AND a.action = 'fold') AS first_b_fold,
               min(a.action_no) FILTER (WHERE a.player_id = ph.player_1 AND a.action = 'call') AS first_a_call,
               min(a.action_no) FILTER (WHERE a.player_id = ph.player_2 AND a.action = 'call') AS first_b_call
        FROM pair_hands_{phase_name} ph
        LEFT JOIN actions a ON a.hand_id = ph.hand_id
        GROUP BY ph.pair_id, ph.hand_id
        """
    relation_sql = f"""
        CREATE OR REPLACE TABLE pair_hand_relation_{phase_name} AS
        WITH ordered AS (
            SELECT a.hand_id, a.action_no, a.street, a.player_id, a.action,
                   a.amount, a.to_call,
                   lead(a.player_id) OVER (
                       PARTITION BY a.hand_id ORDER BY a.action_no
                   ) AS next_player,
                   lead(a.action) OVER (
                       PARTITION BY a.hand_id ORDER BY a.action_no
                   ) AS next_action,
                   lead(a.street) OVER (
                       PARTITION BY a.hand_id ORDER BY a.action_no
                   ) AS next_street
            FROM actions a
            JOIN (SELECT DISTINCT hand_id FROM pair_hands_{phase_name}) relevant
              USING (hand_id)
        ), summary AS (
            SELECT f.pair_id, f.hand_id,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2
                         AND f.first_a_pressure IS NOT NULL
                         AND a.action_no > f.first_a_pressure
                         AND a.action IN {response}
                   )::DOUBLE AS b_response_after_a,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1
                         AND f.first_b_pressure IS NOT NULL
                         AND a.action_no > f.first_b_pressure
                         AND a.action IN {response}
                   )::DOUBLE AS a_response_after_b,
                   count(*) FILTER (
                       WHERE a.player_id NOT IN (ph.player_1, ph.player_2)
                         AND f.first_a_pressure IS NOT NULL
                         AND a.action_no > f.first_a_pressure
                         AND a.action = 'fold'
                   )::DOUBLE AS outsider_fold_after_a,
                   count(*) FILTER (
                       WHERE a.player_id NOT IN (ph.player_1, ph.player_2)
                         AND f.first_b_pressure IS NOT NULL
                         AND a.action_no > f.first_b_pressure
                         AND a.action = 'fold'
                   )::DOUBLE AS outsider_fold_after_b,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1
                         AND a.action IN {pressure}
                         AND a.next_player = ph.player_2
                         AND a.next_street = a.street
                         AND a.next_action = 'fold'
                   )::DOUBLE AS b_immediate_fold_after_a,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1
                         AND a.action IN {pressure}
                         AND a.next_player = ph.player_2
                         AND a.next_street = a.street
                         AND a.next_action = 'call'
                   )::DOUBLE AS b_immediate_call_after_a,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1
                         AND a.action IN {pressure}
                         AND a.next_player = ph.player_2
                         AND a.next_street = a.street
                         AND a.next_action IN ('raise', 'all_in')
                   )::DOUBLE AS b_immediate_raise_after_a,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2
                         AND a.action IN {pressure}
                         AND a.next_player = ph.player_1
                         AND a.next_street = a.street
                         AND a.next_action = 'fold'
                   )::DOUBLE AS a_immediate_fold_after_b,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2
                         AND a.action IN {pressure}
                         AND a.next_player = ph.player_1
                         AND a.next_street = a.street
                         AND a.next_action = 'call'
                   )::DOUBLE AS a_immediate_call_after_b,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2
                         AND a.action IN {pressure}
                         AND a.next_player = ph.player_1
                         AND a.next_street = a.street
                         AND a.next_action IN ('raise', 'all_in')
                   )::DOUBLE AS a_immediate_raise_after_b,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1
                         AND a.action IN {pressure}
                         AND a.next_player NOT IN (ph.player_1, ph.player_2)
                         AND a.next_street = a.street
                         AND a.next_action = 'fold'
                   )::DOUBLE AS outsider_immediate_fold_after_a,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2
                         AND a.action IN {pressure}
                         AND a.next_player NOT IN (ph.player_1, ph.player_2)
                         AND a.next_street = a.street
                         AND a.next_action = 'fold'
                   )::DOUBLE AS outsider_immediate_fold_after_b,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1
                         AND a.action IN {pressure}
                         AND a.next_player NOT IN (ph.player_1, ph.player_2)
                         AND a.next_street = a.street
                         AND a.next_action IN {pressure}
                   )::DOUBLE AS outsider_immediate_pressure_after_a,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2
                         AND a.action IN {pressure}
                         AND a.next_player NOT IN (ph.player_1, ph.player_2)
                         AND a.next_street = a.street
                         AND a.next_action IN {pressure}
                   )::DOUBLE AS outsider_immediate_pressure_after_b,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.to_call > 0
                   )::DOUBLE AS a_facing_call_opportunity,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.to_call > 0
                   )::DOUBLE AS b_facing_call_opportunity,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.to_call > 0 AND a.action = 'fold'
                   )::DOUBLE AS a_fold_facing_call,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.to_call > 0 AND a.action = 'fold'
                   )::DOUBLE AS b_fold_facing_call,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.to_call > 0 AND a.action = 'call'
                   )::DOUBLE AS a_call_facing_call,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.to_call > 0 AND a.action = 'call'
                   )::DOUBLE AS b_call_facing_call,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.to_call > 0 AND a.action IN ('raise', 'all_in')
                   )::DOUBLE AS a_raise_facing_call,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.to_call > 0 AND a.action IN ('raise', 'all_in')
                   )::DOUBLE AS b_raise_facing_call,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.action = 'all_in'
                         AND a.to_call > 0 AND a.amount <= a.to_call
                   )::DOUBLE AS a_all_in_call,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.action = 'all_in'
                         AND a.to_call > 0 AND a.amount <= a.to_call
                   )::DOUBLE AS b_all_in_call,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.action = 'all_in'
                         AND (a.to_call = 0 OR a.amount > a.to_call)
                   )::DOUBLE AS a_all_in_aggressive,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.action = 'all_in'
                         AND (a.to_call = 0 OR a.amount > a.to_call)
                   )::DOUBLE AS b_all_in_aggressive,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.action IN {pressure} AND a.street = 'preflop'
                   )::DOUBLE AS a_preflop_pressure,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.action IN {pressure} AND a.street = 'preflop'
                   )::DOUBLE AS b_preflop_pressure,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.action IN {pressure} AND a.street = 'flop'
                   )::DOUBLE AS a_flop_pressure,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.action IN {pressure} AND a.street = 'flop'
                   )::DOUBLE AS b_flop_pressure,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.action IN {pressure} AND a.street = 'turn'
                   )::DOUBLE AS a_turn_pressure,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.action IN {pressure} AND a.street = 'turn'
                   )::DOUBLE AS b_turn_pressure,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_1 AND a.action IN {pressure} AND a.street = 'river'
                   )::DOUBLE AS a_river_pressure,
                   count(*) FILTER (
                       WHERE a.player_id = ph.player_2 AND a.action IN {pressure} AND a.street = 'river'
                   )::DOUBLE AS b_river_pressure
            FROM pair_hand_first_{phase_name} f
            JOIN pair_hands_{phase_name} ph USING (pair_id, hand_id)
            LEFT JOIN ordered a ON a.hand_id = f.hand_id
            GROUP BY f.pair_id, f.hand_id
        )
        FROM summary
        """
    return first_sql, relation_sql


def _feature_sql(phase: str) -> str:
    phase_name = phase.lower()
    board_clean = "regexp_replace(ph.board_cards, '[^A-Za-z0-9]', '', 'g')"
    a_r1 = _rank_expr("ph.a_hole_card_1")
    a_r2 = _rank_expr("ph.a_hole_card_2")
    b_r1 = _rank_expr("ph.b_hole_card_1")
    b_r2 = _rank_expr("ph.b_hole_card_2")
    board_sum = _board_rank_sum("board_clean")
    return f"""
        CREATE OR REPLACE TABLE pair_hand_features_{phase_name} AS
        WITH base AS (
            SELECT ph.*, f.first_a_pressure, f.first_b_pressure,
                   f.first_a_fold, f.first_b_fold, r.b_response_after_a,
                   r.a_response_after_b, r.outsider_fold_after_a,
                   r.outsider_fold_after_b,
                   r.b_immediate_fold_after_a, r.b_immediate_call_after_a,
                   r.b_immediate_raise_after_a, r.a_immediate_fold_after_b,
                   r.a_immediate_call_after_b, r.a_immediate_raise_after_b,
                   r.outsider_immediate_fold_after_a,
                   r.outsider_immediate_fold_after_b,
                   r.outsider_immediate_pressure_after_a,
                   r.outsider_immediate_pressure_after_b,
                   r.a_facing_call_opportunity, r.b_facing_call_opportunity,
                   r.a_fold_facing_call, r.b_fold_facing_call,
                   r.a_call_facing_call, r.b_call_facing_call,
                   r.a_raise_facing_call, r.b_raise_facing_call,
                   r.a_all_in_call, r.b_all_in_call,
                   r.a_all_in_aggressive, r.b_all_in_aggressive,
                   r.a_preflop_pressure, r.b_preflop_pressure,
                   r.a_flop_pressure, r.b_flop_pressure,
                   r.a_turn_pressure, r.b_turn_pressure,
                   r.a_river_pressure, r.b_river_pressure,
                   {board_clean} AS board_clean,
                   {a_r1}::DOUBLE AS a_rank_1, {a_r2}::DOUBLE AS a_rank_2,
                   {b_r1}::DOUBLE AS b_rank_1, {b_r2}::DOUBLE AS b_rank_2
            FROM pair_hands_{phase_name} ph
            JOIN pair_hand_first_{phase_name} f USING (pair_id, hand_id)
            JOIN pair_hand_relation_{phase_name} r USING (pair_id, hand_id)
        ), derived AS (
            SELECT *,
                   greatest(a_rank_1, a_rank_2) + least(a_rank_1, a_rank_2) AS a_hole_rank_sum,
                   greatest(b_rank_1, b_rank_2) + least(b_rank_1, b_rank_2) AS b_hole_rank_sum,
                   abs(a_rank_1 - a_rank_2) AS a_hole_rank_gap,
                   abs(b_rank_1 - b_rank_2) AS b_hole_rank_gap,
                   abs((a_rank_1 + a_rank_2) - (b_rank_1 + b_rank_2)) AS hole_rank_gap,
                   (a_rank_1 = a_rank_2)::INTEGER AS a_hole_pair,
                   (b_rank_1 = b_rank_2)::INTEGER AS b_hole_pair,
                   (lower(substr(a_hole_card_1, 2, 1)) = lower(substr(a_hole_card_2, 2, 1)))::INTEGER AS a_hole_suited,
                   (lower(substr(b_hole_card_1, 2, 1)) = lower(substr(b_hole_card_2, 2, 1)))::INTEGER AS b_hole_suited,
                   length(board_clean) / 2.0 AS board_count,
                   {board_sum}::DOUBLE AS board_rank_sum,
                   CASE WHEN a_action_count > 0 THEN a_pressure_count / a_action_count ELSE 0 END AS a_pressure_rate,
                   CASE WHEN b_action_count > 0 THEN b_pressure_count / b_action_count ELSE 0 END AS b_pressure_rate,
                   CASE WHEN a_action_count > 0 THEN a_fold_count / a_action_count ELSE 0 END AS a_fold_rate,
                   CASE WHEN b_action_count > 0 THEN b_fold_count / b_action_count ELSE 0 END AS b_fold_rate,
                   CASE WHEN a_action_count > 0 THEN a_call_count / a_action_count ELSE 0 END AS a_call_rate,
                   CASE WHEN b_action_count > 0 THEN b_call_count / b_action_count ELSE 0 END AS b_call_rate,
                   CASE WHEN a_action_count > 0 THEN a_check_count / a_action_count ELSE 0 END AS a_check_rate,
                   CASE WHEN b_action_count > 0 THEN b_check_count / b_action_count ELSE 0 END AS b_check_rate,
                   (CASE WHEN a_action_count > 0 THEN a_pressure_count / a_action_count ELSE 0 END) - a_background_pressure_rate AS a_pressure_delta,
                   (CASE WHEN b_action_count > 0 THEN b_pressure_count / b_action_count ELSE 0 END) - b_background_pressure_rate AS b_pressure_delta,
                   (CASE WHEN a_action_count > 0 THEN a_fold_count / a_action_count ELSE 0 END) - a_background_fold_rate AS a_fold_delta,
                   (CASE WHEN b_action_count > 0 THEN b_fold_count / b_action_count ELSE 0 END) - b_background_fold_rate AS b_fold_delta,
                   (CASE WHEN a_action_count > 0 THEN a_call_count / a_action_count ELSE 0 END) - a_background_call_rate AS a_call_delta,
                   (CASE WHEN b_action_count > 0 THEN b_call_count / b_action_count ELSE 0 END) - b_background_call_rate AS b_call_delta,
                   least(greatest(-a_net_chips, 0), greatest(b_net_chips, 0)) / nullif(big_blind, 0) AS transfer_ab_bb,
                   least(greatest(-b_net_chips, 0), greatest(a_net_chips, 0)) / nullif(big_blind, 0) AS transfer_ba_bb
            FROM base
        )
        , raw_features AS (
        SELECT pair_id, player_1, player_2, phase, pool_id, reported_shared_hands,
               hand_id, table_id, started_at,
               a_net_chips / nullif(big_blind, 0) AS a_net_bb,
               b_net_chips / nullif(big_blind, 0) AS b_net_bb,
               abs(a_net_chips - b_net_chips) / nullif(big_blind, 0) AS net_gap_bb,
               transfer_ab_bb, transfer_ba_bb,
               greatest(transfer_ab_bb, transfer_ba_bb) AS transfer_max_bb,
               transfer_ab_bb + transfer_ba_bb AS transfer_sum_bb,
               a_total_contribution / nullif(big_blind, 0) AS a_contrib_bb,
               b_total_contribution / nullif(big_blind, 0) AS b_contrib_bb,
               (a_total_contribution + b_total_contribution) / nullif(big_blind, 0) AS total_contrib_bb,
               abs(a_total_contribution - b_total_contribution) / nullif(big_blind, 0) AS contrib_gap_bb,
               final_pot / nullif(big_blind, 0) AS final_pot_bb,
               ((a_mean_pot_before + b_mean_pot_before) / 2.0) / nullif(big_blind, 0) AS pot_bb,
               (coalesce(a_mean_players_active, 0) + coalesce(b_mean_players_active, 0)) / 2.0 AS players_active_mean,
               players_dealt, players_at_showdown,
               (a_showdown AND b_showdown)::INTEGER AS showdown_both,
               a_folded::INTEGER AS a_folded, b_folded::INTEGER AS b_folded,
               a_showdown::INTEGER AS a_showdown, b_showdown::INTEGER AS b_showdown,
               board_count, board_rank_sum, a_hole_rank_sum, b_hole_rank_sum,
               a_hole_rank_gap, b_hole_rank_gap, hole_rank_gap, a_hole_pair,
               b_hole_pair, a_hole_suited, b_hole_suited,
               a_action_count, b_action_count, a_pressure_count, b_pressure_count,
               a_fold_count, b_fold_count, a_call_count, b_call_count,
               a_check_count, b_check_count, a_pressure_rate, b_pressure_rate,
               a_fold_rate, b_fold_rate, a_call_rate, b_call_rate,
               a_check_rate, b_check_rate,
               a_pressure_amount / nullif(big_blind, 0) AS a_pressure_amount_bb,
               b_pressure_amount / nullif(big_blind, 0) AS b_pressure_amount_bb,
               a_amount_to_sum / nullif(big_blind, 0) AS a_amount_to_bb,
               b_amount_to_sum / nullif(big_blind, 0) AS b_amount_to_bb,
               a_mean_players_active, b_mean_players_active,
               a_pressure_delta, b_pressure_delta, a_fold_delta, b_fold_delta,
               a_call_delta, b_call_delta,
               coalesce(b_response_after_a, 0) AS b_response_after_a,
               coalesce(a_response_after_b, 0) AS a_response_after_b,
               coalesce(outsider_fold_after_a, 0) AS outsider_fold_after_a,
               coalesce(outsider_fold_after_b, 0) AS outsider_fold_after_b,
               (first_b_pressure IS NOT NULL AND first_a_fold IS NOT NULL AND first_b_pressure < first_a_fold)::INTEGER AS a_fold_after_b_pressure,
               (first_a_pressure IS NOT NULL AND first_b_fold IS NOT NULL AND first_a_pressure < first_b_fold)::INTEGER AS b_fold_after_a_pressure,
               (coalesce(b_response_after_a, 0) > 0 AND coalesce(a_response_after_b, 0) > 0)::INTEGER AS mutual_pressure_sequence,
               coalesce(first_a_pressure, 0)::DOUBLE AS a_first_pressure_pos,
               coalesce(first_b_pressure, 0)::DOUBLE AS b_first_pressure_pos,
               (
                   greatest(transfer_ab_bb, transfer_ba_bb)
                   + 0.5 * abs(a_pressure_delta - b_pressure_delta)
                   + 0.25 * (outsider_fold_after_a + outsider_fold_after_b)
                   + 0.25 * (a_fold_after_b_pressure + b_fold_after_a_pressure)
               )::DOUBLE AS hand_risk_proxy
               ,coalesce(b_immediate_fold_after_a, 0)::DOUBLE AS b_immediate_fold_after_a
               ,coalesce(b_immediate_call_after_a, 0)::DOUBLE AS b_immediate_call_after_a
               ,coalesce(b_immediate_raise_after_a, 0)::DOUBLE AS b_immediate_raise_after_a
               ,coalesce(a_immediate_fold_after_b, 0)::DOUBLE AS a_immediate_fold_after_b
               ,coalesce(a_immediate_call_after_b, 0)::DOUBLE AS a_immediate_call_after_b
               ,coalesce(a_immediate_raise_after_b, 0)::DOUBLE AS a_immediate_raise_after_b
               ,coalesce(outsider_immediate_fold_after_a, 0)::DOUBLE AS outsider_immediate_fold_after_a
               ,coalesce(outsider_immediate_fold_after_b, 0)::DOUBLE AS outsider_immediate_fold_after_b
               ,coalesce(outsider_immediate_pressure_after_a, 0)::DOUBLE AS outsider_immediate_pressure_after_a
               ,coalesce(outsider_immediate_pressure_after_b, 0)::DOUBLE AS outsider_immediate_pressure_after_b
               ,coalesce(a_facing_call_opportunity, 0)::DOUBLE AS a_facing_call_opportunity
               ,coalesce(b_facing_call_opportunity, 0)::DOUBLE AS b_facing_call_opportunity
               ,coalesce(a_fold_facing_call, 0)::DOUBLE AS a_fold_facing_call
               ,coalesce(b_fold_facing_call, 0)::DOUBLE AS b_fold_facing_call
               ,coalesce(a_call_facing_call, 0)::DOUBLE AS a_call_facing_call
               ,coalesce(b_call_facing_call, 0)::DOUBLE AS b_call_facing_call
               ,coalesce(a_raise_facing_call, 0)::DOUBLE AS a_raise_facing_call
               ,coalesce(b_raise_facing_call, 0)::DOUBLE AS b_raise_facing_call
               ,coalesce(a_all_in_call, 0)::DOUBLE AS a_all_in_call
               ,coalesce(b_all_in_call, 0)::DOUBLE AS b_all_in_call
               ,coalesce(a_all_in_aggressive, 0)::DOUBLE AS a_all_in_aggressive
               ,coalesce(b_all_in_aggressive, 0)::DOUBLE AS b_all_in_aggressive
               ,coalesce(a_preflop_pressure, 0)::DOUBLE AS a_preflop_pressure
               ,coalesce(b_preflop_pressure, 0)::DOUBLE AS b_preflop_pressure
               ,coalesce(a_flop_pressure, 0)::DOUBLE AS a_flop_pressure
               ,coalesce(b_flop_pressure, 0)::DOUBLE AS b_flop_pressure
               ,coalesce(a_turn_pressure, 0)::DOUBLE AS a_turn_pressure
               ,coalesce(b_turn_pressure, 0)::DOUBLE AS b_turn_pressure
               ,coalesce(a_river_pressure, 0)::DOUBLE AS a_river_pressure
               ,coalesce(b_river_pressure, 0)::DOUBLE AS b_river_pressure
        FROM derived
        )
        SELECT raw_features.*,
               percent_rank() OVER (
                   PARTITION BY pair_id ORDER BY hand_risk_proxy, hand_id
               )::DOUBLE AS hand_risk_rank_pct,
               percent_rank() OVER (
                   PARTITION BY pair_id ORDER BY transfer_max_bb, hand_id
               )::DOUBLE AS transfer_max_rank_pct,
               percent_rank() OVER (
                   PARTITION BY pair_id
                   ORDER BY (b_response_after_a + a_response_after_b), hand_id
               )::DOUBLE AS response_rank_pct,
               percent_rank() OVER (
                   PARTITION BY pair_id
                   ORDER BY (outsider_fold_after_a + outsider_fold_after_b), hand_id
               )::DOUBLE AS outsider_rank_pct,
               percent_rank() OVER (
                   PARTITION BY pair_id ORDER BY showdown_both, hand_id
               )::DOUBLE AS joint_showdown_rank_pct,
               coalesce(avg(hand_risk_proxy) OVER (
                   PARTITION BY pair_id ORDER BY started_at, hand_id
                   ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
               ), hand_risk_proxy)::DOUBLE AS hand_risk_prev5_mean,
               coalesce(avg(hand_risk_proxy) OVER (
                   PARTITION BY pair_id ORDER BY started_at, hand_id
                   ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING
               ), hand_risk_proxy)::DOUBLE AS hand_risk_next5_mean,
               avg(hand_risk_proxy) OVER (
                   PARTITION BY pair_id ORDER BY started_at, hand_id
                   ROWS BETWEEN 10 PRECEDING AND 10 FOLLOWING
               )::DOUBLE AS hand_risk_window20_mean,
               avg(hand_risk_proxy) OVER (
                   PARTITION BY pair_id ORDER BY started_at, hand_id
                   ROWS BETWEEN 25 PRECEDING AND 25 FOLLOWING
               )::DOUBLE AS hand_risk_window50_mean,
               max(hand_risk_proxy) OVER (
                   PARTITION BY pair_id ORDER BY started_at, hand_id
                   ROWS BETWEEN 10 PRECEDING AND 10 FOLLOWING
               )::DOUBLE AS hand_risk_window20_peak
        FROM raw_features
        """


def _pair_aggregate_sql() -> str:
    aggregations: list[str] = []
    for column in HAND_FEATURE_COLUMNS:
        aggregations.extend(
            [
                f"avg({column})::DOUBLE AS f_{column}_mean",
                f"coalesce(stddev_samp({column}), 0)::DOUBLE AS f_{column}_std",
                f"max({column})::DOUBLE AS f_{column}_max",
                f"quantile_cont({column}, 0.9)::DOUBLE AS f_{column}_q90",
            ]
        )
    return """
        CREATE OR REPLACE TABLE pair_features AS
        WITH aggregate AS (
            SELECT pair_id, any_value(player_1) AS player_1,
                   any_value(player_2) AS player_2, any_value(phase) AS phase,
                   any_value(pool_id) AS table_id,
                   any_value(reported_shared_hands) AS reported_shared_hands,
                   count(*)::DOUBLE AS f_shared_hands,
                   min(started_at) AS first_started_at,
                   max(started_at) AS last_started_at,
                   """ + ",\n                   ".join(aggregations) + """
            FROM pair_hand_features_all
            GROUP BY pair_id
        ), meta AS (
            SELECT tp.pair_id,
                   abs(p1.account_age_days - p2.account_age_days)::DOUBLE AS f_account_age_abs_gap,
                   (p1.experience_hands_bucket = p2.experience_hands_bucket)::INTEGER AS f_experience_match,
                   (p1.preferred_stake = p2.preferred_stake)::INTEGER AS f_preferred_stake_match,
                   (p1.region_bucket = p2.region_bucket)::INTEGER AS f_region_match,
                   (p1.client_family = p2.client_family)::INTEGER AS f_client_match
            FROM target_pairs tp
            JOIN players p1 ON p1.player_id = tp.player_1
            JOIN players p2 ON p2.player_id = tp.player_2
        )
        SELECT a.*, m.f_account_age_abs_gap, m.f_experience_match,
               m.f_preferred_stake_match, m.f_region_match, m.f_client_match,
               coalesce(a.reported_shared_hands, a.f_shared_hands)::DOUBLE AS f_shared_hands_reported,
               log(1 + a.f_shared_hands)::DOUBLE AS f_log_shared_hands,
               a.f_showdown_both_mean / greatest(a.f_shared_hands, 1)::DOUBLE AS f_joint_showdown_rate,
               (a.f_hand_risk_proxy_max + a.f_hand_risk_proxy_q90) / 2.0 AS f_generic_tail_score,
               a.f_transfer_ab_bb_mean - a.f_transfer_ba_bb_mean AS f_directional_transfer_imbalance,
               abs(a.f_transfer_ab_bb_mean - a.f_transfer_ba_bb_mean) AS f_directional_transfer_abs_imbalance,
               (a.f_a_pressure_delta_mean - a.f_b_pressure_delta_mean) AS f_pressure_direction_gap,
               abs(a.f_a_pressure_delta_mean - a.f_b_pressure_delta_mean) AS f_pressure_abs_gap,
               (a.f_outsider_fold_after_a_mean + a.f_outsider_fold_after_b_mean) AS f_outsider_pressure_total,
               (a.f_a_fold_after_b_pressure_mean + a.f_b_fold_after_a_pressure_mean) AS f_partner_fold_sequence,
               (a.f_b_response_after_a_mean + a.f_a_response_after_b_mean) AS f_mutual_response_total
        FROM aggregate a JOIN meta m USING (pair_id)
        """


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_manifest(root: Path) -> dict[str, dict[str, object]]:
    return {
        name: {"path": str(_path(root, name)), "bytes": _path(root, name).stat().st_size,
               "sha256": _sha256(_path(root, name))}
        for name in RAW_FILES
    }


def _write_pool_folds(con: duckdb.DuckDBPyConnection, output_dir: Path, n_splits: int = 5) -> None:
    con.execute(
        f"""
        COPY (
            SELECT pool_id AS table_id,
                   mod(abs(hash(pool_id)), {n_splits})::INTEGER AS fold
            FROM (SELECT DISTINCT pool_id FROM target_pairs)
            ORDER BY pool_id
        ) TO {_sql_literal(output_dir / 'pool_folds.csv')} (HEADER, DELIMITER ',')
        """
    )


def build_features(config: FeatureBuildConfig) -> dict[str, object]:
    """Build audited pair and pair-hand feature tables on the execution host."""
    root = Path(config.data_dir).resolve()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        _path(root, name)
    db_path = output_dir / "feature_cache.duckdb"
    started = time.time()
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"PRAGMA threads={max(1, int(config.threads))}")
        con.execute(f"SET memory_limit='{config.memory_limit}'")
        con.execute("SET preserve_insertion_order=false")
        _create_views(con, root)
        _create_shared_tables(con)
        phase_counts: dict[str, int] = {}
        for phase in ("development", "evaluation"):
            con.execute(_pair_hands_sql(phase))
            first_sql, relation_sql = _relationship_sql(phase)
            con.execute(first_sql)
            con.execute(relation_sql)
            con.execute(_feature_sql(phase))
            relation = f"pair_hand_features_{phase}"
            count = con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0]
            phase_counts[phase] = int(count)
            con.execute(
                f"COPY {relation} TO {_sql_literal(output_dir / (relation + '.parquet'))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        con.execute(
            "CREATE OR REPLACE TABLE pair_hand_features_all AS "
            "SELECT * FROM pair_hand_features_development UNION ALL "
            "SELECT * FROM pair_hand_features_evaluation"
        )
        con.execute(_pair_aggregate_sql())
        con.execute(
            f"COPY pair_features TO {_sql_literal(output_dir / 'pair_features.parquet')} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        _write_pool_folds(con, output_dir)
        pair_count = con.execute("SELECT count(*) FROM pair_features").fetchone()[0]
        report = {
            "status": "completed",
            "config": asdict(config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "raw": _raw_manifest(root),
            "phase_pair_hand_rows": phase_counts,
            "pair_features": int(pair_count),
            "elapsed_seconds": round(time.time() - started, 3),
            "artifacts": {
                "database": str(db_path),
                "pair_features": str(output_dir / "pair_features.parquet"),
                "development_pair_hands": str(output_dir / "pair_hand_features_development.parquet"),
                "evaluation_pair_hands": str(output_dir / "pair_hand_features_evaluation.parquet"),
                "pool_folds": str(output_dir / "pool_folds.csv"),
            },
        }
        (output_dir / "feature_build_manifest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return report
    finally:
        con.close()


def audit_feature_membership(data_dir: str, feature_dir: str) -> dict[str, object]:
    """Run deterministic membership and count checks after feature materialization."""
    root = Path(data_dir).resolve()
    output_dir = Path(feature_dir).resolve()
    con = duckdb.connect()
    try:
        con.execute(
            f"CREATE VIEW eval AS SELECT * FROM read_csv_auto({_sql_literal(_path(root, 'evaluation_pairs'))}, header=true)"
        )
        con.execute(
            f"CREATE VIEW labels AS SELECT * FROM read_csv_auto({_sql_literal(_path(root, 'development_labels'))}, header=true)"
        )
        con.execute(
            f"CREATE VIEW hands AS SELECT * FROM read_parquet({_sql_literal(_path(root, 'hands'))})"
        )
        con.execute(
            f"CREATE VIEW seats AS SELECT * FROM read_parquet({_sql_literal(_path(root, 'seats'))})"
        )
        con.execute(
            f"CREATE VIEW pf AS SELECT * FROM read_parquet({_sql_literal(output_dir / 'pair_features.parquet')})"
        )
        con.execute(
            f"CREATE VIEW ph AS SELECT * FROM read_parquet({_sql_literal(output_dir / 'pair_hand_features_evaluation.parquet')})"
        )
        checks = {
            "evaluation_pair_count": con.execute("SELECT count(*) FROM eval").fetchone()[0],
            "evaluation_feature_pair_count": con.execute("SELECT count(*) FROM pf WHERE phase='evaluation'").fetchone()[0],
            "evaluation_hand_count_mismatch": con.execute(
                """
                SELECT count(*) FROM (
                    SELECT e.pair_id, e.shared_hands, count(ph.hand_id) AS actual
                    FROM eval e LEFT JOIN ph USING (pair_id)
                    GROUP BY e.pair_id, e.shared_hands
                    HAVING e.shared_hands <> actual
                )
                """
            ).fetchone()[0],
            "duplicate_evaluation_pair_features": con.execute(
                "SELECT count(*) - count(DISTINCT pair_id) FROM pf WHERE phase='evaluation'"
            ).fetchone()[0],
            "duplicate_evaluation_pair_hands": con.execute(
                "SELECT count(*) - count(DISTINCT concat(pair_id, '::', hand_id)) FROM ph"
            ).fetchone()[0],
            "wrong_phase_evaluation_hands": con.execute(
                """
                SELECT count(*) FROM ph p JOIN hands h USING (hand_id)
                WHERE h.phase <> 'evaluation'
                """
            ).fetchone()[0],
            "nonfinite_pair_features": con.execute(
                """
                SELECT count(*) FROM pf
                WHERE isinf(f_shared_hands) OR isnan(f_shared_hands)
                """
            ).fetchone()[0],
        }
        result = {"status": "pass" if all(v == 0 for k, v in checks.items() if k not in {"evaluation_pair_count", "evaluation_feature_pair_count"}) else "fail", "checks": checks}
        (output_dir / "feature_membership_audit.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    finally:
        con.close()


__all__ = [
    "FeatureBuildConfig",
    "HAND_FEATURE_COLUMNS",
    "RAW_FILES",
    "audit_feature_membership",
    "build_features",
]
