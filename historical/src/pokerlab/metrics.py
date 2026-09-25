"""Independent implementation of the published metric's valid-input semantics.

Reference: Kaggle / florianderoofr / slash-poker-competition-metric, v1.
Scores with ties use stable ordering after sorting pair_id. Do not substitute
sklearn.average_precision_score: its treatment of ties is different.
Run official-notebook parity on the real dataset before relying on this module.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence
import numpy as np
import pandas as pd

KNOWN = ("directed_transfer", "soft_play", "coordinated_isolation")
BEHAVIORS = frozenset((*KNOWN, "none", "other_coordination"))
EVIDENCE_COLUMNS = tuple(f"evidence_hand_{i}" for i in range(1, 6))
SUBMISSION_COLUMNS = ("pair_id", "risk_score", "predicted_behavior", *EVIDENCE_COLUMNS)
NO_EVIDENCE = "NO_EVIDENCE"


def stable_ap(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Binary AP; ties retain INPUT order, which the caller must canonicalize."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError("Expected equal-length one-dimensional labels and scores")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Labels must be binary and scores finite")
    if y.sum() == 0:
        return 0.0
    ranked = y[np.argsort(-s, kind="mergesort")].astype(float)
    return float((ranked * np.cumsum(ranked) / np.arange(1, len(y) + 1)).sum() / y.sum())


def clean_evidence(values: Iterable[object]) -> list[str]:
    """Remove empty/sentinel values. Production validator rejects empty cells."""
    result = []
    for x in values:
        if x is None or (not isinstance(x, (list, tuple, dict)) and pd.isna(x)):
            continue
        s = str(x).strip()
        if not s or s.upper() == NO_EVIDENCE:
            continue
        result.append(s)
    return result


def evidence_ap5(predicted: Iterable[object], relevant: Iterable[object]) -> float:
    """AP@5 with a min(5, number of relevant hands) denominator.

    Duplicate predictions consume a rank but cannot score a second hit.
    A production submission containing duplicates is invalid, regardless of this
    helper's behavior. Sentinels are removed BEFORE rank positions are assigned.
    """
    truth = set(clean_evidence(relevant))
    if not truth:
        return 0.0
    hits, total, seen = 0, 0.0, set()
    for rank, hand in enumerate(clean_evidence(predicted)[:5], 1):
        if hand in truth and hand not in seen:
            hits += 1
            total += hits / rank
        seen.add(hand)
    return total / min(5, len(truth))


@dataclass(frozen=True)
class Score:
    total: float
    pair_ap: float
    evidence_map5: float
    behavior_map: float
    family_ap: Mapping[str, float]
    n_pairs: int
    n_positives: int

    def as_dict(self) -> dict:
        return asdict(self)


def score_submission(truth: pd.DataFrame, submission: pd.DataFrame,
                     evidence_truth: Mapping[str, Iterable[str]]) -> Score:
    """Score labeled pairs. Truth needs pair_id, label, behavior_family.

    This metric does NOT establish that a hand is shared or in the correct phase;
    use validate_submission with a legal pair-hand membership table as well.
    """
    from .validation import validate_submission
    required = {"pair_id", "label", "behavior_family"}
    if not required.issubset(truth.columns) or truth.empty:
        raise ValueError("Nonempty truth must have pair_id, label, behavior_family")
    if truth.pair_id.isna().any() or truth.pair_id.duplicated().any():
        raise ValueError("Truth pair_id must be unique and non-null")
    if not np.isin(truth.label, [0, 1]).all():
        raise ValueError("Truth label must be binary")
    if not truth.loc[truth.label.eq(1), "behavior_family"].isin(BEHAVIORS - {"none"}).all():
        raise ValueError("Positive truth must have a target behavior")
    validate_submission(submission, truth.pair_id.tolist())
    t = truth.assign(pair_id=truth.pair_id.astype(str)).sort_values("pair_id", kind="mergesort")
    pred = submission.assign(pair_id=submission.pair_id.astype(str)).set_index("pair_id").loc[t.pair_id]
    y = t.label.to_numpy(dtype=int)
    risk = pred.risk_score.to_numpy(dtype=float)
    pair_ap = stable_ap(y, risk)
    families = {
        f: stable_ap((t.behavior_family.to_numpy() == f).astype(int),
                     risk * (pred.predicted_behavior.to_numpy() == f))
        for f in KNOWN
    }
    vals = []
    for pid in t.loc[t.label.eq(1), "pair_id"]:
        vals.append(evidence_ap5(pred.loc[pid, list(EVIDENCE_COLUMNS)].tolist(),
                                evidence_truth.get(pid, [])))
    ev = float(np.mean(vals)) if vals else 0.0
    beh = float(np.mean(list(families.values())))
    return Score(0.7 * pair_ap + 0.2 * ev + 0.1 * beh, pair_ap, ev, beh,
                 families, len(t), int(y.sum()))
