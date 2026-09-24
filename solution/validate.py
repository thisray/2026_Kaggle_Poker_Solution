"""Strict input and submission validation for the competition release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


EXPECTED_ROWS = 112_540
OFFICIAL_FILES = (
    "players.parquet",
    "hands.parquet",
    "seats.parquet",
    "actions.parquet",
    "development_labels.csv",
    "development_evidence.csv",
    "evaluation_pairs.csv",
    "sample_submission.csv",
)
EVIDENCE_COLUMNS = tuple(f"evidence_hand_{index}" for index in range(1, 6))
SUBMISSION_COLUMNS = (
    "pair_id",
    "risk_score",
    "predicted_behavior",
    *EVIDENCE_COLUMNS,
)
BEHAVIORS = frozenset(
    {
        "none",
        "directed_transfer",
        "soft_play",
        "coordinated_isolation",
        "other_coordination",
    }
)
NO_EVIDENCE = "NO_EVIDENCE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_files(data_dir: str | Path) -> dict[str, int]:
    root = Path(data_dir)
    missing = [name for name in OFFICIAL_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official competition files: {missing}")

    evaluation_pairs = pd.read_csv(
        root / "evaluation_pairs.csv",
        usecols=["pair_id", "player_1", "player_2"],
        dtype=str,
    )
    sample = pd.read_csv(root / "sample_submission.csv", dtype=str)
    if len(evaluation_pairs) != EXPECTED_ROWS or len(sample) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} evaluation rows; found "
            f"{len(evaluation_pairs)} pairs and {len(sample)} sample rows"
        )
    if evaluation_pairs.pair_id.isna().any() or evaluation_pairs.pair_id.duplicated().any():
        raise ValueError("evaluation_pairs.csv must contain unique, non-null pair_id values")
    if set(evaluation_pairs.pair_id) != set(sample.pair_id.astype(str)):
        raise ValueError("sample_submission.csv pair IDs do not match evaluation_pairs.csv")
    return {"evaluation_pairs": len(evaluation_pairs), "sample_rows": len(sample)}


def _validate_schema(submission: pd.DataFrame, expected_pairs: pd.DataFrame) -> pd.DataFrame:
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}")
    if len(submission) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows; found {len(submission)}")
    if submission.isna().any().any():
        raise ValueError("Submission contains null cells")

    text = submission.astype(str)
    if text.apply(lambda column: column.str.strip().eq("").any()).any():
        raise ValueError("Submission contains blank cells")
    if text.pair_id.duplicated().any():
        raise ValueError("Submission contains duplicate pair_id values")
    if set(text.pair_id) != set(expected_pairs.pair_id.astype(str)):
        raise ValueError("Submission pair IDs do not match evaluation_pairs.csv")

    risk = pd.to_numeric(submission.risk_score, errors="raise").to_numpy(float)
    if not np.isfinite(risk).all() or ((risk < 0.0) | (risk > 1.0)).any():
        raise ValueError("risk_score values must be finite and in [0, 1]")
    if not text.predicted_behavior.isin(BEHAVIORS).all():
        invalid = sorted(set(text.predicted_behavior) - BEHAVIORS)
        raise ValueError(f"Unknown predicted_behavior values: {invalid}")

    for row in text[list(EVIDENCE_COLUMNS)].itertuples(index=False, name=None):
        hands = [value for value in row if value != NO_EVIDENCE]
        if len(hands) != len(set(hands)):
            raise ValueError("Submission repeats an evidence hand within a pair")
    return text


def _validate_evidence_membership(
    text: pd.DataFrame,
    expected_pairs: pd.DataFrame,
    data_dir: Path,
) -> int:
    evidence = text.melt(
        id_vars="pair_id",
        value_vars=EVIDENCE_COLUMNS,
        value_name="hand_id",
    )[["pair_id", "hand_id"]]
    evidence = evidence.loc[evidence.hand_id.ne(NO_EVIDENCE)].drop_duplicates()

    connection = duckdb.connect()
    try:
        connection.register("submitted_evidence", evidence)
        connection.register("evaluation_pairs", expected_pairs)
        hands_path = str((data_dir / "hands.parquet").resolve())
        seats_path = str((data_dir / "seats.parquet").resolve())

        invalid_hands = connection.execute(
            """
            SELECT count(*)
            FROM submitted_evidence e
            LEFT JOIN read_parquet(?) h USING (hand_id)
            WHERE h.hand_id IS NULL OR h.phase <> 'evaluation'
            """,
            [hands_path],
        ).fetchone()[0]
        if invalid_hands:
            raise ValueError(
                f"Found {invalid_hands} unknown or non-evaluation evidence hand references"
            )

        invalid_membership = connection.execute(
            """
            SELECT count(*)
            FROM submitted_evidence e
            JOIN evaluation_pairs p USING (pair_id)
            LEFT JOIN read_parquet(?) s1
              ON s1.hand_id = e.hand_id AND s1.player_id = p.player_1
            LEFT JOIN read_parquet(?) s2
              ON s2.hand_id = e.hand_id AND s2.player_id = p.player_2
            WHERE s1.player_id IS NULL OR s2.player_id IS NULL
            """,
            [seats_path, seats_path],
        ).fetchone()[0]
        if invalid_membership:
            raise ValueError(
                f"Found {invalid_membership} evidence references without both pair players seated"
            )
    finally:
        connection.close()
    return len(evidence)


def validate_submission(
    submission_path: str | Path,
    data_dir: str | Path,
) -> dict[str, object]:
    submission_file = Path(submission_path)
    root = Path(data_dir)
    validate_input_files(root)
    expected_pairs = pd.read_csv(
        root / "evaluation_pairs.csv",
        usecols=["pair_id", "player_1", "player_2"],
        dtype=str,
    )
    submission = pd.read_csv(submission_file, dtype=str, keep_default_na=False)
    text = _validate_schema(submission, expected_pairs)
    evidence_references = _validate_evidence_membership(text, expected_pairs, root)
    return {
        "status": "PASS",
        "submission": str(submission_file),
        "rows": len(submission),
        "unique_pairs": int(text.pair_id.nunique()),
        "evidence_references": evidence_references,
        "sha256": sha256_file(submission_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()
    report = validate_submission(args.submission, args.data_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
