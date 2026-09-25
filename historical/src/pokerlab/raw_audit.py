"""DuckDB audit of the eight official competition files.

No credentials are embedded. The audit separates hard contract violations from
diagnostic distributions whose interpretation depends on the organizer's game
and payout semantics.
"""
from __future__ import annotations
from pathlib import Path
import json


def _quote(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def audit(data_dir: str, output: str) -> dict:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Install raw extras first: pip install -e '.[raw]'") from exc
    root = Path(data_dir)
    expected = {"players": "parquet", "hands": "parquet", "seats": "parquet", "actions": "parquet",
                "development_labels": "csv", "development_evidence": "csv",
                "evaluation_pairs": "csv", "sample_submission": "csv"}
    missing = [f"{name}.{ext}" for name, ext in expected.items() if not (root/f"{name}.{ext}").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing data: {missing}")
    con = duckdb.connect()
    report = {"status": "executed_full_contract_audit", "tables": {}, "checks": {}}
    try:
        for name, ext in expected.items():
            reader = f"read_parquet({_quote(root/f'{name}.{ext}')})" if ext == "parquet" else (
                f"read_csv_auto({_quote(root/f'{name}.{ext}')}, header=true)")
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM {reader}")
            report["tables"][name] = {"rows": con.execute(f"SELECT count(*) FROM {name}").fetchone()[0],
                "schema": con.execute(f"DESCRIBE {name}").fetchall()}
        queries = {
            "duplicate_player_ids": "SELECT count(*)-count(DISTINCT player_id) FROM players",
            "duplicate_hand_ids": "SELECT count(*)-count(DISTINCT hand_id) FROM hands",
            "duplicate_seat_keys": "SELECT count(*) FROM (SELECT hand_id,player_id FROM seats GROUP BY 1,2 HAVING count(*)<>1)",
            "duplicate_action_keys": "SELECT count(*) FROM (SELECT hand_id,action_no FROM actions GROUP BY 1,2 HAVING count(*)<>1)",
            "non_six_player_hands": "SELECT count(*) FROM (SELECT hand_id FROM seats GROUP BY 1 HAVING count(*)<>6)",
            "nonpositive_blinds": "SELECT count(*) FROM hands WHERE big_blind IS NULL OR big_blind<=0",
            "null_phase": "SELECT count(*) FROM hands WHERE phase IS NULL",
            "nonzero_net_hand_count": "SELECT count(*) FROM (SELECT hand_id,sum(net_chips) z FROM seats GROUP BY 1 HAVING abs(z)>0.000001)",
            "unknown_action_hand": "SELECT count(*) FROM actions a ANTI JOIN hands h USING(hand_id)",
            "unknown_seat_player": "SELECT count(*) FROM seats s ANTI JOIN players p USING(player_id)",
            "unknown_label_player_1": "SELECT count(*) FROM development_labels l ANTI JOIN players p ON p.player_id=l.player_1",
            "unknown_label_player_2": "SELECT count(*) FROM development_labels l ANTI JOIN players p ON p.player_id=l.player_2",
            "unknown_evaluation_player_1": "SELECT count(*) FROM evaluation_pairs e ANTI JOIN players p ON p.player_id=e.player_1",
            "unknown_evaluation_player_2": "SELECT count(*) FROM evaluation_pairs e ANTI JOIN players p ON p.player_id=e.player_2",
            "same_player_labeled_pair": "SELECT count(*) FROM development_labels WHERE player_1=player_2",
            "same_player_evaluation_pair": "SELECT count(*) FROM evaluation_pairs WHERE player_1=player_2",
            "evidence_wrong_phase": "SELECT count(*) FROM development_evidence e JOIN hands h USING(hand_id) WHERE h.phase<>'development'",
            "evidence_unknown_hand": "SELECT count(*) FROM development_evidence e ANTI JOIN hands h USING(hand_id)",
            "evidence_unknown_pair": "SELECT count(*) FROM development_evidence e ANTI JOIN development_labels l USING(pair_id)",
            "evidence_duplicate_rank": "SELECT count(*) FROM (SELECT pair_id,evidence_rank FROM development_evidence GROUP BY 1,2 HAVING count(*)<>1)",
            "evaluation_overlaps_labeled_pair": "SELECT count(*) FROM evaluation_pairs e JOIN development_labels l USING(pair_id)",
            "negative_amount": "SELECT count(*) FROM actions WHERE amount<0 OR amount_to<0",
            "amount_to_below_increment": "SELECT count(*) FROM actions WHERE amount_to + 0.000001 < amount",
            "negative_pot": "SELECT count(*) FROM actions WHERE pot_before<0",
            "negative_stack_before": "SELECT count(*) FROM actions WHERE stack_before<0",
            "invalid_players_active": "SELECT count(*) FROM actions WHERE players_active<1 OR players_active>6",
            "noncontiguous_action_numbers": "SELECT count(*) FROM (SELECT hand_id FROM actions GROUP BY 1 HAVING min(action_no)<>0 OR max(action_no)+1<>count(*))",
            "action_unknown_street": "SELECT count(*) FROM actions WHERE street NOT IN ('preflop','flop','turn','river')",
            "action_unknown_type": "SELECT count(*) FROM actions WHERE action NOT IN ('fold','check','call','bet','raise','all_in')",
            "evaluation_shared_hands_mismatch": """
                SELECT count(*) FROM (
                    SELECT e.pair_id, e.shared_hands, count(*) AS actual
                    FROM evaluation_pairs e
                    JOIN seats s1 ON s1.player_id=e.player_1
                    JOIN seats s2 ON s2.player_id=e.player_2 AND s2.hand_id=s1.hand_id
                    JOIN hands h ON h.hand_id=s1.hand_id AND h.phase='evaluation'
                    GROUP BY e.pair_id,e.shared_hands
                    HAVING e.shared_hands<>actual
                )
            """,
        }
        for name, sql in queries.items():
            report["checks"][name] = con.execute(sql).fetchone()[0]
        report["distributions"] = {
            "phase": con.execute("SELECT phase,count(*) FROM hands GROUP BY 1").fetchall(),
            "label_family": con.execute("SELECT label,behavior_family,count(*) FROM development_labels GROUP BY 1,2").fetchall(),
            "actions": con.execute("SELECT street,action,count(*) FROM actions GROUP BY 1,2").fetchall(),
            "all_in_inferred": con.execute(
                """
                SELECT
                    count(*) FILTER (WHERE action='all_in' AND to_call>0 AND amount<=to_call) AS inferred_call,
                    count(*) FILTER (WHERE action='all_in' AND (to_call=0 OR amount>to_call)) AS inferred_aggressive
                FROM actions
                """
            ).fetchone(),
            "contribution_vs_final_pot": con.execute(
                """
                SELECT quantile_cont(total_contribution, 0.01),
                       quantile_cont(total_contribution, 0.50),
                       quantile_cont(total_contribution, 0.99),
                       quantile_cont(final_pot, 0.01),
                       quantile_cont(final_pot, 0.50),
                       quantile_cont(final_pot, 0.99)
                FROM (
                    SELECT h.hand_id, h.final_pot, sum(s.total_contribution) AS total_contribution
                    FROM hands h JOIN seats s USING(hand_id)
                    GROUP BY h.hand_id,h.final_pot
                )
                """
            ).fetchone(),
        }
        hard_checks = {
            name: value for name, value in report["checks"].items()
            if name not in {"nonzero_net_hand_count"}
        }
        report["status"] = "pass" if all(value == 0 for value in hard_checks.values()) else "fail"
        report["interpretation"] = (
            "Nonzero net is retained as a diagnostic because rake, rounding, refunds, "
            "or synthetic payout rules may explain it. amount_to>=amount and inferred "
            "all-in categories are recorded as semantic evidence; they are not a claim "
            "that every stack field is a replayable ledger."
        )
    finally:
        con.close()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report
