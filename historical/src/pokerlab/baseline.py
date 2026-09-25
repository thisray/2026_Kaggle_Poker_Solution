"""Group-CV reference pipeline on EXTRACTED features, not raw Parquet.

Input features must have independently verified, fold-safe provenance. The code
cannot infer whether a supplied feature was generated using validation labels.
Advanced state reconstruction, PNU, value models and MIL remain research tasks.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .metrics import KNOWN, NO_EVIDENCE, SUBMISSION_COLUMNS, EVIDENCE_COLUMNS, score_submission
from .validation import validate_submission


@dataclass(frozen=True)
class Config:
    pair_features: tuple[str, ...]
    hand_features: tuple[str, ...]
    n_splits: int = 5
    seed: int = 2026
    max_iter: int = 100
    max_leaf_nodes: int = 7
    min_samples_leaf: int = 10
    risk_threshold: float = 0.5
    family_threshold: float = 0.5


def _check_inputs(pairs: pd.DataFrame, hands: pd.DataFrame, config: Config) -> None:
    if not {"pair_id", "table_id", "phase", *config.pair_features}.issubset(pairs.columns):
        raise ValueError("Pair features require pair_id/table_id/phase and registered feature columns")
    if not {"pair_id", "hand_id", "phase", *config.hand_features}.issubset(hands.columns):
        raise ValueError("Hand features require pair_id/hand_id/phase and registered feature columns")
    if pairs.pair_id.duplicated().any() or pairs.pair_id.isna().any():
        raise ValueError("Unique non-null pairs required")
    if hands[["pair_id", "hand_id"]].duplicated().any():
        raise ValueError("Duplicate pair-hand rows")
    if not set(hands.pair_id).issubset(set(pairs.pair_id)):
        raise ValueError("Hand rows refer to pairs outside feature table")
    if not config.pair_features or not config.hand_features:
        raise ValueError("Explicit nonempty feature allowlists required")
    for names, frame in ((config.pair_features, pairs), (config.hand_features, hands)):
        if any(not x.startswith("f_") for x in names):
            raise ValueError("Only explicitly registered f_* numeric feature columns are accepted")
        if not all(pd.api.types.is_numeric_dtype(frame[x]) for x in names):
            raise ValueError("All model features must be numeric")
        if np.isinf(frame[list(names)].to_numpy(float)).any():
            raise ValueError("Infinite feature values are not allowed")


class ConstantClassifier:
    """Defined behavior when a training split contains a single class."""
    def __init__(self, label: Any):
        self.classes_ = np.array([label])
    def predict_proba(self, X: Any) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


def _fit(X: pd.DataFrame, y: np.ndarray, config: Config, weights: np.ndarray | None = None):
    classes = np.unique(y)
    if len(classes) == 0:
        raise ValueError("Training set is empty")
    if len(classes) == 1:
        return ConstantClassifier(classes[0])
    model = HistGradientBoostingClassifier(max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes, min_samples_leaf=config.min_samples_leaf,
        learning_rate=0.06, l2_regularization=1.0, early_stopping=False,
        random_state=config.seed)
    with threadpool_limits(limits=1):
        model.fit(X, y, sample_weight=weights)
    return model


def _prob(model: Any, X: pd.DataFrame, labels: tuple) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0, len(labels)))
    with threadpool_limits(limits=1):
        raw = model.predict_proba(X)
    lookup = {c: i for i, c in enumerate(model.classes_)}
    return np.column_stack([raw[:, lookup[k]] if k in lookup else np.zeros(len(X)) for k in labels])


def fit_models(pairs: pd.DataFrame, labels: pd.DataFrame, hands: pd.DataFrame,
               evidence: dict[str, set[str]], config: Config) -> dict:
    _check_inputs(pairs, hands, config)
    joined = pairs.merge(labels[["pair_id", "label", "behavior_family"]],
                         on="pair_id", how="inner", validate="one_to_one")
    if len(joined) != len(pairs) or not joined.label.isin([0, 1]).all():
        raise ValueError("Reference baseline requires exactly one trusted label for every train pair")
    risk_model = _fit(joined[list(config.pair_features)], joined.label.to_numpy(int), config)
    pos = joined[joined.label.eq(1) & joined.behavior_family.isin(KNOWN)]
    if pos.empty:
        raise ValueError("At least one known-family positive is required")
    family_model = _fit(pos[list(config.pair_features)], pos.behavior_family.to_numpy(), config)
    positive_ids = set(joined.loc[joined.label.eq(1), "pair_id"])
    eh = hands[hands.pair_id.isin(positive_ids)].copy()
    if eh.empty:
        raise ValueError("Need candidate hands for positive training pairs")
    e_y = np.array([int(str(h) in evidence.get(str(p), set()))
                    for p, h in zip(eh.pair_id, eh.hand_id)])
    # Equal total weight per pair, not per action/hand. Unlisted hands are
    # 'not in the published selection', NOT guaranteed benign gameplay.
    weights = 1.0 / eh.groupby("pair_id").pair_id.transform("count").to_numpy(float)
    weights *= len(weights) / weights.sum()
    evidence_model = _fit(eh[list(config.hand_features)], e_y, config, weights)
    return {"risk": risk_model, "family": family_model, "evidence": evidence_model,
            "config": config}


def predict(models: dict, pairs: pd.DataFrame, hands: pd.DataFrame) -> pd.DataFrame:
    config = models["config"]
    _check_inputs(pairs, hands, config)
    phase = pairs.phase.unique()
    if len(phase) != 1 or not hands.phase.eq(phase[0]).all():
        raise ValueError("Prediction pairs/hands must use one matching phase")
    X = pairs[list(config.pair_features)]
    risk = _prob(models["risk"], X, (1,))[:, 0]
    family = _prob(models["family"], X, KNOWN)
    best_idx = family.argmax(axis=1)
    best_conf = family.max(axis=1)
    behavior = np.array(KNOWN, dtype=object)[best_idx]
    behavior[best_conf < config.family_threshold] = "other_coordination"
    behavior[risk < config.risk_threshold] = "none"
    out = pd.DataFrame({"pair_id": pairs.pair_id.to_numpy(), "risk_score": risk,
                        "predicted_behavior": behavior})
    for c in EVIDENCE_COLUMNS:
        out[c] = NO_EVIDENCE
    candidates = hands[["pair_id", "hand_id"]].copy()
    candidates["score"] = _prob(models["evidence"], hands[list(config.hand_features)], (1,))[:, 0]
    # ALL pairs receive evidence. Hand IDs break ties only; never a model feature.
    candidates = candidates.sort_values(["pair_id", "score", "hand_id"],
                                         ascending=[True, False, True], kind="mergesort")
    top = candidates.groupby("pair_id", sort=False).head(5).groupby("pair_id").hand_id.agg(list)
    out = out.set_index("pair_id")
    for pid, chosen in top.items():
        out.loc[pid, list(EVIDENCE_COLUMNS[:len(chosen)])] = list(map(str, chosen))
    out = out.reset_index()[list(SUBMISSION_COLUMNS)]
    validate_submission(out, pairs.pair_id, hands, phase=str(phase[0]))
    return out


def run_group_cv(pairs: pd.DataFrame, labels: pd.DataFrame, hands: pd.DataFrame,
                 evidence: dict[str, set[str]], config: Config) -> tuple[pd.DataFrame, dict]:
    """Fixed-config outer CV only. No hyperparameter tuning on validation."""
    _check_inputs(pairs, hands, config)
    if not pairs.phase.eq("development").all() or not hands.phase.eq("development").all():
        raise ValueError("Public OOF evaluation must use development features/evidence")
    if config.n_splits < 2 or pairs.table_id.nunique() < config.n_splits:
        raise ValueError("Not enough distinct groups")
    outputs, fold_reports, manifests = [], [], []
    splitter = GroupKFold(n_splits=config.n_splits)
    for fold, (tr, va) in enumerate(splitter.split(pairs, groups=pairs.table_id)):
        train, valid = pairs.iloc[tr], pairs.iloc[va]
        train_ids, valid_ids = set(train.pair_id), set(valid.pair_id)
        train_hands = hands[hands.pair_id.isin(train_ids)]
        valid_hands = hands[hands.pair_id.isin(valid_ids)]
        assert not set(train.table_id) & set(valid.table_id)
        models = fit_models(train, labels[labels.pair_id.isin(train_ids)], train_hands,
                            {k: v for k, v in evidence.items() if k in train_ids}, config)
        pred = predict(models, valid, valid_hands)
        report = score_submission(labels[labels.pair_id.isin(valid_ids)], pred, evidence).as_dict()
        fold_reports.append({"fold": fold, **report})
        manifests.extend({"pair_id": p, "fold": fold} for p in valid.pair_id)
        outputs.append(pred)
    oof = pd.concat(outputs, ignore_index=True)
    total = score_submission(labels, oof, evidence).as_dict()
    return oof, {"pooled_oof": total, "folds": fold_reports,
                 "fold_manifest": manifests, "config": asdict(config),
                 "scope": "fixed-configuration group CV on supplied features; not nested model selection"}
