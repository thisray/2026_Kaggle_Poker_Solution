"""Strict schema and optional shared-hand/phase validation."""
from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd
from .metrics import BEHAVIORS, EVIDENCE_COLUMNS, SUBMISSION_COLUMNS, NO_EVIDENCE


def validate_submission(submission: pd.DataFrame, expected_pair_ids: Iterable[str],
                        legal_pair_hands: pd.DataFrame | None = None,
                        phase: str = "evaluation") -> dict[str, int]:
    """Raise ValueError on invalid output; return a small validation summary.

    legal_pair_hands must be built by joining BOTH players to the hand; this
    function cannot independently verify a falsely constructed membership table.
    Required columns: pair_id, hand_id, phase. Pass phase='development' for OOF.
    """
    if list(submission.columns) != list(SUBMISSION_COLUMNS):
        raise ValueError(f"Exact column order required: {SUBMISSION_COLUMNS}")
    if submission.isna().any().any():
        raise ValueError("Submission contains null cells")
    text = submission.astype(str)
    if text.apply(lambda col: col.str.strip().eq("").any()).any():
        raise ValueError("Submission contains blank cells; use NO_EVIDENCE")
    expected_list = [str(x) for x in expected_pair_ids]
    if len(expected_list) != len(set(expected_list)):
        raise ValueError("Expected pair IDs are duplicated")
    if text.pair_id.duplicated().any():
        raise ValueError("Duplicate submitted pair_id")
    if set(text.pair_id) != set(expected_list):
        raise ValueError("Missing or unexpected pair_id")
    try:
        risk = pd.to_numeric(submission.risk_score, errors="raise").to_numpy(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("risk_score is not numeric") from exc
    if not np.isfinite(risk).all() or ((risk < 0) | (risk > 1)).any():
        raise ValueError("risk_score must be finite and in [0, 1]")
    if not text.predicted_behavior.isin(BEHAVIORS).all():
        raise ValueError("Unknown predicted_behavior")
    used = 0
    for values in text[list(EVIDENCE_COLUMNS)].itertuples(index=False, name=None):
        hands = [h for h in values if h != NO_EVIDENCE]
        if len(hands) != len(set(hands)):
            raise ValueError("Repeated evidence hand within a pair")
        used += len(hands)
    if legal_pair_hands is not None:
        if not {"pair_id", "hand_id", "phase"}.issubset(legal_pair_hands.columns):
            raise ValueError("Legal membership needs pair_id, hand_id, phase")
        legal = legal_pair_hands.loc[legal_pair_hands.phase.eq(phase), ["pair_id", "hand_id"]].astype(str)
        legal = legal.drop_duplicates()
        actual = text.melt(id_vars="pair_id", value_vars=EVIDENCE_COLUMNS,
                           value_name="hand_id")
        actual = actual.loc[actual.hand_id.ne(NO_EVIDENCE), ["pair_id", "hand_id"]]
        merged = actual.merge(legal, on=["pair_id", "hand_id"], how="left", indicator=True)
        if not merged._merge.eq("both").all():
            bad = merged.loc[merged._merge.ne("both"), ["pair_id", "hand_id"]].head(5)
            raise ValueError(f"Unknown/non-shared/wrong-phase evidence: {bad.to_dict('records')}")
    return {"pairs": len(submission), "evidence_slots_used": used,
            "membership_checked": int(legal_pair_hands is not None)}
