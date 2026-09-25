"""Pool-disjoint training and submission generation on real poker features.

The module deliberately keeps model inputs explicit. Pair/player/table keys are
used for joins, grouping, or deterministic tie handling only. Evidence training
uses set membership from the published development evidence; unlisted hands are
not interpreted as confirmed benign behavior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .metrics import (
    EVIDENCE_COLUMNS,
    KNOWN,
    NO_EVIDENCE,
    SUBMISSION_COLUMNS,
    score_submission,
)
from .raw_features import HAND_FEATURE_COLUMNS
from .validation import validate_submission


FAMILY_EVIDENCE_FEATURES = {
    "directed_transfer": (
        "a_net_bb", "b_net_bb", "net_gap_bb", "transfer_ab_bb", "transfer_ba_bb",
        "transfer_max_bb", "transfer_sum_bb", "a_contrib_bb", "b_contrib_bb",
        "total_contrib_bb", "contrib_gap_bb", "pot_bb", "final_pot_bb",
        "players_at_showdown", "showdown_both", "a_showdown", "b_showdown",
        "a_pressure_count", "b_pressure_count", "a_pressure_rate", "b_pressure_rate",
        "a_pressure_amount_bb", "b_pressure_amount_bb", "a_amount_to_bb", "b_amount_to_bb",
        "a_mean_players_active", "b_mean_players_active", "a_pressure_delta",
        "b_pressure_delta", "a_call_delta", "b_call_delta", "hand_risk_proxy",
    ),
    "soft_play": (
        "a_net_bb", "b_net_bb", "transfer_max_bb", "transfer_sum_bb", "pot_bb",
        "showdown_both", "a_folded", "b_folded", "a_showdown", "b_showdown",
        "a_action_count", "b_action_count", "a_pressure_rate", "b_pressure_rate",
        "a_fold_rate", "b_fold_rate", "a_call_rate", "b_call_rate",
        "a_check_rate", "b_check_rate", "a_pressure_delta", "b_pressure_delta",
        "a_fold_delta", "b_fold_delta", "a_call_delta", "b_call_delta",
        "b_response_after_a", "a_response_after_b", "a_fold_after_b_pressure",
        "b_fold_after_a_pressure", "mutual_pressure_sequence", "hand_risk_proxy",
        "a_facing_call_opportunity", "b_facing_call_opportunity", "a_fold_facing_call",
        "b_fold_facing_call", "a_call_facing_call", "b_call_facing_call",
    ),
    "coordinated_isolation": (
        "a_net_bb", "b_net_bb", "transfer_max_bb", "pot_bb", "final_pot_bb",
        "showdown_both", "a_folded", "b_folded", "a_pressure_count", "b_pressure_count",
        "a_pressure_rate", "b_pressure_rate", "a_pressure_delta", "b_pressure_delta",
        "b_response_after_a", "a_response_after_b", "outsider_fold_after_a",
        "outsider_fold_after_b", "a_fold_after_b_pressure", "b_fold_after_a_pressure",
        "mutual_pressure_sequence", "a_first_pressure_pos", "b_first_pressure_pos",
        "outsider_immediate_fold_after_a", "outsider_immediate_fold_after_b",
        "outsider_immediate_pressure_after_a", "outsider_immediate_pressure_after_b",
        "a_preflop_pressure", "b_preflop_pressure", "a_flop_pressure", "b_flop_pressure",
        "a_turn_pressure", "b_turn_pressure", "a_river_pressure", "b_river_pressure",
        "hand_risk_proxy",
    ),
}

OPTIONAL_HAND_FEATURE_COLUMNS = frozenset({
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
    "hand_risk_next5_mean", "hand_risk_window20_mean",
    "hand_risk_window50_mean", "hand_risk_window20_peak",
})


@dataclass(frozen=True)
class TrainConfig:
    features_dir: str
    data_dir: str
    output_dir: str
    n_splits: int = 5
    seed: int = 20260917
    max_iter: int = 280
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 10
    extra_trees: int = 320
    n_jobs: int = 8


class ConstantClassifier:
    """A deterministic classifier for a fold missing a class."""

    def __init__(self, label: object):
        self.classes_ = np.asarray([label], dtype=object)
        self.label = label

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


def _read_csv(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    usecols = list(columns) if columns is not None else None
    return pd.read_csv(path, usecols=usecols, dtype={"pair_id": str, "hand_id": str})


def _finite_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = frame[columns].copy()
    values = values.replace([np.inf, -np.inf], np.nan)
    for column in columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
        median = values[column].median()
        values[column] = values[column].fillna(0.0 if pd.isna(median) else median)
    return values.astype(np.float32)


def _balance_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    total = counts.sum()
    weights = np.ones(len(y), dtype=float)
    for value in (0, 1):
        if counts[value] > 0:
            weights[y == value] = total / (2.0 * counts[value])
    return weights


def _fit_binary_hgb(X: pd.DataFrame, y: np.ndarray, config: TrainConfig):
    classes = np.unique(y)
    if len(classes) == 1:
        return ConstantClassifier(int(classes[0]))
    model = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        learning_rate=0.045,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=config.seed,
    )
    with threadpool_limits(limits=1):
        model.fit(X, y, sample_weight=_balance_weights(y))
    return model


def _fit_extra(X: pd.DataFrame, y: np.ndarray, config: TrainConfig):
    classes = np.unique(y)
    if len(classes) == 1:
        return ConstantClassifier(int(classes[0]))
    model = ExtraTreesClassifier(
        n_estimators=config.extra_trees,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.65,
        class_weight="balanced_subsample",
        random_state=config.seed,
        n_jobs=config.n_jobs,
    )
    model.fit(X, y)
    return model


def _fit_logistic(X: pd.DataFrame, y: np.ndarray, config: TrainConfig):
    classes = np.unique(y)
    if len(classes) == 1:
        return ConstantClassifier(int(classes[0]))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.35,
            max_iter=500,
            class_weight="balanced",
            random_state=config.seed,
        ),
    )
    with threadpool_limits(limits=1):
        model.fit(X, y)
    return model


def _fit_family(X: pd.DataFrame, y: np.ndarray, config: TrainConfig):
    classes = np.unique(y)
    if len(classes) == 1:
        return ConstantClassifier(classes[0])
    model = ExtraTreesClassifier(
        n_estimators=max(180, config.extra_trees // 2),
        max_depth=10,
        min_samples_leaf=4,
        max_features=0.7,
        class_weight="balanced",
        random_state=config.seed + 17,
        n_jobs=config.n_jobs,
    )
    model.fit(X, y)
    return model


def _proba(model: Any, X: pd.DataFrame, labels: tuple[object, ...]) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    return np.column_stack([
        raw[:, classes.index(label)] if label in classes else np.zeros(len(X), dtype=float)
        for label in labels
    ])


def _evidence_truth(path: Path) -> dict[str, set[str]]:
    frame = _read_csv(path, ["pair_id", "hand_id"])
    return frame.groupby("pair_id").hand_id.agg(set).to_dict()


def _rank_evidence(pair_ids: pd.Series, hands: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    pair_values = hands.pair_id.astype(str).to_numpy()
    hand_values = hands.hand_id.astype(str).to_numpy()
    score_values = np.asarray(scores, dtype=float)
    if len(pair_values) != len(score_values):
        raise ValueError("Evidence scores must align one-to-one with pair-hand rows")
    pair_codes, unique_pairs = pd.factorize(pair_values, sort=False)
    # Numeric lexsort avoids a full string DataFrame sort and keeps the stable
    # input order as the deterministic tie breaker.
    order = np.lexsort((np.arange(len(hands), dtype=np.int64), -score_values, pair_codes))
    counts = np.bincount(pair_codes, minlength=len(unique_pairs))
    top_counts = np.minimum(counts, len(EVIDENCE_COLUMNS))
    starts = np.cumsum(np.r_[0, counts[:-1]])
    group_ids = np.repeat(np.arange(len(unique_pairs)), top_counts)
    selected_positions = np.concatenate([
        order[start:start + count]
        for start, count in zip(starts, top_counts)
        if count
    ]) if group_ids.size else np.empty(0, dtype=np.int64)
    selected_ranks = np.concatenate([
        np.arange(count, dtype=np.int64)
        for count in top_counts
        if count
    ]) if group_ids.size else np.empty(0, dtype=np.int64)
    top_values = np.full(
        (len(unique_pairs), len(EVIDENCE_COLUMNS)), NO_EVIDENCE, dtype=object
    )
    top_values[group_ids, selected_ranks] = hand_values[selected_positions]
    output_pairs = pair_ids.astype(str).to_numpy()
    pair_lookup = pd.Series(np.arange(len(unique_pairs), dtype=np.int64), index=unique_pairs)
    output_codes = pair_lookup.reindex(output_pairs).to_numpy()
    if pd.isna(output_codes).any():
        raise ValueError("Evidence candidates do not cover every requested pair")
    rows = pd.DataFrame({"pair_id": output_pairs})
    rows[list(EVIDENCE_COLUMNS)] = top_values[output_codes.astype(np.int64)]
    return rows


def _make_submission(
    pairs: pd.DataFrame,
    risk: np.ndarray,
    family_scores: np.ndarray,
    evidence: pd.DataFrame,
    family_threshold: float,
    risk_threshold: float,
) -> pd.DataFrame:
    best = family_scores.argmax(axis=1)
    confidence = family_scores.max(axis=1)
    behavior = np.asarray(KNOWN, dtype=object)[best]
    uncertain = confidence < family_threshold
    behavior[uncertain] = "other_coordination"
    behavior[np.asarray(risk) < risk_threshold] = "none"
    base = pd.DataFrame({
        "pair_id": pairs.pair_id.astype(str).to_numpy(),
        "risk_score": np.clip(np.asarray(risk, dtype=float), 0.0, 1.0),
        "predicted_behavior": behavior,
    })
    result = base.merge(evidence, on="pair_id", how="left", validate="one_to_one")
    for column in EVIDENCE_COLUMNS:
        result[column] = result[column].fillna(NO_EVIDENCE)
    return result[list(SUBMISSION_COLUMNS)]


def _score_grid(
    truth: pd.DataFrame,
    risk: np.ndarray,
    family_scores: np.ndarray,
    evidence: pd.DataFrame,
    evidence_truth: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates: list[tuple[float, float, float, dict]] = []
    for family_threshold in np.linspace(0.20, 0.80, 13):
        for risk_threshold in np.linspace(0.0, 0.80, 17):
            submission = _make_submission(
                truth, risk, family_scores, evidence, float(family_threshold), float(risk_threshold)
            )
            score = score_submission(truth, submission, evidence_truth)
            candidates.append((score.total, family_threshold, risk_threshold, score.as_dict()))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    best = candidates[0]
    chosen = _make_submission(
        truth,
        risk,
        family_scores,
        evidence,
        float(best[1]),
        float(best[2]),
    )
    return chosen, {
        "best_family_threshold": float(best[1]),
        "best_risk_threshold": float(best[2]),
        "best_score": best[3],
        "grid_top10": [
            {"total": float(row[0]), "family_threshold": float(row[1]),
             "risk_threshold": float(row[2]), "score": row[3]}
            for row in candidates[:10]
        ],
    }


def _feature_columns(pair_features: pd.DataFrame) -> list[str]:
    excluded = {"f_shared_hands_reported"}
    columns = [
        column for column in pair_features.columns
        if column.startswith("f_") and column not in excluded
        and pd.api.types.is_numeric_dtype(pair_features[column])
    ]
    if not columns:
        raise ValueError("No numeric f_* pair features found")
    return columns


def _load_tables(config: TrainConfig) -> dict[str, pd.DataFrame | list[str] | dict[str, set[str]]]:
    features_dir = Path(config.features_dir)
    data_dir = Path(config.data_dir)
    pair_features = pd.read_parquet(features_dir / "pair_features.parquet")
    pair_features["pair_id"] = pair_features.pair_id.astype(str)
    development = pair_features.loc[pair_features.phase.eq("development")].reset_index(drop=True)
    evaluation = pair_features.loc[pair_features.phase.eq("evaluation")].reset_index(drop=True)
    labels = _read_csv(data_dir / "development_labels.csv")
    labels["pair_id"] = labels.pair_id.astype(str)
    evidence_truth = _evidence_truth(data_dir / "development_evidence.csv")
    pool_folds = _read_csv(features_dir / "pool_folds.csv")
    pool_folds["table_id"] = pool_folds["table_id"].astype(str)
    pair_hands_development = pd.read_parquet(features_dir / "pair_hand_features_development.parquet")
    pair_hands_evaluation = pd.read_parquet(features_dir / "pair_hand_features_evaluation.parquet")
    for frame in (pair_hands_development, pair_hands_evaluation):
        frame["pair_id"] = frame.pair_id.astype(str)
        frame["hand_id"] = frame.hand_id.astype(str)
    pair_columns = _feature_columns(development)
    missing = [column for column in HAND_FEATURE_COLUMNS if column not in pair_hands_development.columns]
    missing_evaluation = [column for column in HAND_FEATURE_COLUMNS if column not in pair_hands_evaluation.columns]
    if missing != missing_evaluation:
        raise ValueError("Development and evaluation hand feature schemas do not match")
    required_missing = [column for column in missing if column not in OPTIONAL_HAND_FEATURE_COLUMNS]
    if required_missing:
        raise ValueError(f"Missing required hand features: {required_missing}")
    hand_feature_columns = [column for column in HAND_FEATURE_COLUMNS if column not in missing]
    if not hand_feature_columns:
        raise ValueError("No compatible hand features found")
    family_feature_columns = {
        family: [column for column in columns if column in hand_feature_columns]
        for family, columns in FAMILY_EVIDENCE_FEATURES.items()
    }
    if any(not columns for columns in family_feature_columns.values()):
        raise ValueError("At least one family evidence feature set is empty")
    return {
        "development": development,
        "evaluation": evaluation,
        "labels": labels,
        "evidence_truth": evidence_truth,
        "pool_folds": pool_folds,
        "pair_hands_development": pair_hands_development,
        "pair_hands_evaluation": pair_hands_evaluation,
        "pair_columns": pair_columns,
        "hand_feature_columns": hand_feature_columns,
        "family_feature_columns": family_feature_columns,
        "missing_optional_hand_features": missing,
    }


def _fold_assignment(development: pd.DataFrame, pool_folds: pd.DataFrame) -> np.ndarray:
    mapping = dict(zip(pool_folds.table_id.astype(str), pool_folds.fold.astype(int)))
    missing = sorted(set(development.table_id.astype(str)) - set(mapping))
    if missing:
        raise ValueError(f"Pool fold manifest missing tables: {missing[:5]}")
    return development.table_id.astype(str).map(mapping).to_numpy(dtype=int)


def _fit_evidence_model(
    train_hands: pd.DataFrame,
    evidence_truth: dict[str, set[str]],
    config: TrainConfig,
    seed_offset: int = 0,
    feature_columns: Iterable[str] | None = None,
) -> Any:
    feature_columns = list(feature_columns or HAND_FEATURE_COLUMNS)
    train_hands = train_hands.loc[train_hands.pair_id.isin(set(evidence_truth))].copy()
    if train_hands.empty:
        return ConstantClassifier(0)
    y = np.asarray([
        int(str(hand) in evidence_truth.get(str(pair), set()))
        for pair, hand in zip(train_hands.pair_id, train_hands.hand_id)
    ], dtype=int)
    X_train = _finite_frame(train_hands, feature_columns)
    # A pair-balanced sample weight avoids a prolific pair dominating the ranker.
    pair_counts = train_hands.groupby("pair_id").pair_id.transform("count").to_numpy(float)
    weights = 1.0 / np.maximum(pair_counts, 1.0)
    weights *= len(weights) / weights.sum()
    classes = np.unique(y)
    if len(classes) == 1:
        return ConstantClassifier(int(classes[0]))
    model = HistGradientBoostingClassifier(
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        learning_rate=0.05,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=config.seed + 101 + seed_offset,
    )
    with threadpool_limits(limits=1):
        model.fit(X_train, y, sample_weight=weights * _balance_weights(y))
    return model


def _evidence_model_scores(
    train_hands: pd.DataFrame,
    valid_hands: pd.DataFrame,
    evidence_truth: dict[str, set[str]],
    config: TrainConfig,
    feature_columns: Iterable[str] | None = None,
) -> np.ndarray:
    if valid_hands.empty:
        return np.zeros(0, dtype=float)
    columns = list(feature_columns or HAND_FEATURE_COLUMNS)
    model = _fit_evidence_model(train_hands, evidence_truth, config, feature_columns=columns)
    return _proba(model, _finite_frame(valid_hands, columns), (1,))[:, 0]


def _routed_evidence_scores(
    train_hands: pd.DataFrame,
    valid_hands: pd.DataFrame,
    evidence_truth_by_family: dict[str, dict[str, set[str]]],
    pair_route: dict[str, str],
    config: TrainConfig,
    feature_columns: Iterable[str] | None = None,
    feature_columns_by_family: dict[str, Iterable[str]] | None = None,
) -> np.ndarray:
    """Score each pair-hand with its OOF/full-data predicted family expert."""
    if valid_hands.empty:
        return np.zeros(0, dtype=float)
    base_columns = list(feature_columns or HAND_FEATURE_COLUMNS)
    models = {
        family: _fit_evidence_model(
            train_hands,
            evidence_truth_by_family.get(family, {}),
            config,
            seed_offset=family_index,
            feature_columns=(feature_columns_by_family or {}).get(family, base_columns),
        )
        for family_index, family in enumerate(KNOWN)
    }
    route = valid_hands.pair_id.map(pair_route)
    scores = np.zeros(len(valid_hands), dtype=float)
    for family in KNOWN:
        mask = route.eq(family)
        if mask.any():
            columns = list((feature_columns_by_family or {}).get(family, base_columns))
            scores[mask.to_numpy()] = _proba(
                models[family],
                _finite_frame(valid_hands.loc[mask], columns),
                (1,),
            )[:, 0]
    return scores


def _risk_oof(
    development: pd.DataFrame,
    labels: pd.DataFrame,
    pair_hands: pd.DataFrame,
    fold_id: np.ndarray,
    pair_columns: list[str],
    evidence_truth: dict[str, set[str]],
    config: TrainConfig,
    hand_feature_columns: list[str],
    family_feature_columns: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    joined = development.merge(
        labels[["pair_id", "label", "behavior_family"]],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(development):
        raise ValueError("Every development pair must have one trusted label")
    X_all = _finite_frame(joined, pair_columns)
    y = joined.label.to_numpy(dtype=int)
    models = {"hgb": np.zeros(len(joined)), "extra": np.zeros(len(joined)), "logistic": np.zeros(len(joined))}
    family = np.zeros((len(joined), len(KNOWN)), dtype=float)
    evidence_scores = {
        "universal": np.zeros(len(pair_hands), dtype=float),
        "family_route_a25": np.zeros(len(pair_hands), dtype=float),
        "family_route_a50": np.zeros(len(pair_hands), dtype=float),
        "family_route_a75": np.zeros(len(pair_hands), dtype=float),
        "family_semantic_a50": np.zeros(len(pair_hands), dtype=float),
        "family_semantic_a75": np.zeros(len(pair_hands), dtype=float),
    }
    fold_reports = []
    fold_by_pair = dict(zip(development.pair_id.astype(str), fold_id))
    joined_fold_id = joined.pair_id.astype(str).map(fold_by_pair).to_numpy(dtype=int)
    family_by_pair = dict(zip(joined.pair_id.astype(str), joined.behavior_family.astype(str)))
    for fold in range(config.n_splits):
        train_mask = joined_fold_id != fold
        valid_mask = joined_fold_id == fold
        train_x, valid_x = X_all.loc[train_mask], X_all.loc[valid_mask]
        train_y, valid_y = y[train_mask], y[valid_mask]
        hgb = _fit_binary_hgb(train_x, train_y, config)
        extra = _fit_extra(train_x, train_y, config)
        logistic = _fit_logistic(train_x, train_y, config)
        models["hgb"][valid_mask] = _proba(hgb, valid_x, (1,))[:, 0]
        models["extra"][valid_mask] = _proba(extra, valid_x, (1,))[:, 0]
        models["logistic"][valid_mask] = _proba(logistic, valid_x, (1,))[:, 0]
        positive_train = joined.loc[train_mask & joined.label.eq(1)]
        family_model = _fit_family(
            X_all.loc[train_mask & joined.label.eq(1)],
            positive_train.behavior_family.to_numpy(),
            config,
        )
        family[valid_mask] = _proba(family_model, valid_x, KNOWN)
        train_ids = set(joined.loc[train_mask, "pair_id"])
        valid_ids = set(joined.loc[valid_mask, "pair_id"])
        valid_hand_mask = pair_hands.pair_id.isin(valid_ids)
        valid_hands = pair_hands.loc[valid_hand_mask]
        universal = _evidence_model_scores(
            pair_hands.loc[pair_hands.pair_id.isin(train_ids)],
            valid_hands,
            evidence_truth,
            config,
            feature_columns=hand_feature_columns,
        )
        valid_pair_ids = joined.loc[valid_mask, "pair_id"].astype(str).to_numpy()
        valid_family_names = np.asarray(KNOWN, dtype=object)[family[valid_mask].argmax(axis=1)]
        pair_route = dict(zip(valid_pair_ids, valid_family_names))
        train_truth_by_family = {
            family_name: {
                pid: evidence_truth[pid]
                for pid in train_ids
                if pid in evidence_truth and family_by_pair.get(str(pid)) == family_name
            }
            for family_name in KNOWN
        }
        routed = _routed_evidence_scores(
            pair_hands.loc[pair_hands.pair_id.isin(train_ids)],
            valid_hands,
            train_truth_by_family,
            pair_route,
            config,
            feature_columns=hand_feature_columns,
        )
        routed_semantic = _routed_evidence_scores(
            pair_hands.loc[pair_hands.pair_id.isin(train_ids)],
            valid_hands,
            train_truth_by_family,
            pair_route,
            config,
            feature_columns_by_family=family_feature_columns,
        )
        valid_hand_indices = valid_hand_mask.to_numpy()
        evidence_scores["universal"][valid_hand_indices] = universal
        for specialist_weight in (0.25, 0.50, 0.75):
            name = f"family_route_a{int(specialist_weight * 100):02d}"
            evidence_scores[name][valid_hand_indices] = (
                (1.0 - specialist_weight) * universal + specialist_weight * routed
            )
        for specialist_weight in (0.50, 0.75):
            name = f"family_semantic_a{int(specialist_weight * 100):02d}"
            evidence_scores[name][valid_hand_indices] = (
                (1.0 - specialist_weight) * universal + specialist_weight * routed_semantic
            )
        fold_reports.append({
            "fold": fold,
            "train_pairs": int(train_mask.sum()),
            "valid_pairs": int(valid_mask.sum()),
            "train_positives": int(train_y.sum()),
            "valid_positives": int(valid_y.sum()),
        })
    evidence_variants = {
        name: _rank_evidence(development.pair_id, pair_hands, scores)
        for name, scores in evidence_scores.items()
    }
    return models, family, {
        "folds": fold_reports,
        "evidence_oof_rows": {name: len(frame) for name, frame in evidence_variants.items()},
        "evidence_variants": evidence_variants,
    }


def _full_models(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    labels: pd.DataFrame,
    pair_hands_development: pd.DataFrame,
    pair_hands_evaluation: pd.DataFrame,
    evidence_truth: dict[str, set[str]],
    pair_columns: list[str],
    config: TrainConfig,
    hand_feature_columns: list[str],
    family_feature_columns: dict[str, list[str]],
    evidence_names: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, pd.DataFrame]]:
    joined = development.merge(labels[["pair_id", "label", "behavior_family"]], on="pair_id", validate="one_to_one")
    X_train = _finite_frame(joined, pair_columns)
    X_eval = _finite_frame(evaluation, pair_columns)
    y = joined.label.to_numpy(dtype=int)
    risk_models = {
        "hgb": _fit_binary_hgb(X_train, y, config),
        "extra": _fit_extra(X_train, y, config),
        "logistic": _fit_logistic(X_train, y, config),
    }
    risk = {name: _proba(model, X_eval, (1,))[:, 0] for name, model in risk_models.items()}
    family_model = _fit_family(X_train.loc[joined.label.eq(1)], joined.loc[joined.label.eq(1), "behavior_family"].to_numpy(), config)
    family = _proba(family_model, X_eval, KNOWN)
    evidence_scores = _evidence_model_scores(
        pair_hands_development,
        pair_hands_evaluation,
        evidence_truth,
        config,
        feature_columns=hand_feature_columns,
    )
    family_by_pair = dict(zip(joined.pair_id.astype(str), joined.behavior_family.astype(str)))
    train_truth_by_family = {
        family_name: {
            pid: evidence_truth[pid]
            for pid in evidence_truth
            if family_by_pair.get(str(pid)) == family_name
        }
        for family_name in KNOWN
    }
    eval_pair_route = dict(zip(
        evaluation.pair_id.astype(str),
        np.asarray(KNOWN, dtype=object)[family.argmax(axis=1)],
    ))
    routed_scores = _routed_evidence_scores(
        pair_hands_development,
        pair_hands_evaluation,
        train_truth_by_family,
        eval_pair_route,
        config,
        feature_columns=hand_feature_columns,
    )
    routed_semantic_scores = _routed_evidence_scores(
        pair_hands_development,
        pair_hands_evaluation,
        train_truth_by_family,
        eval_pair_route,
        config,
        feature_columns_by_family=family_feature_columns,
    )
    available_names = (
        "universal", "family_route_a25", "family_route_a50", "family_route_a75",
        "family_semantic_a50", "family_semantic_a75",
    )
    requested_names = tuple(evidence_names) if evidence_names is not None else available_names
    unknown_names = sorted(set(requested_names) - set(available_names))
    if unknown_names:
        raise ValueError(f"Unknown evidence variants: {unknown_names}")
    evidence_variants = {}
    if "universal" in requested_names:
        evidence_variants["universal"] = _rank_evidence(
            evaluation.pair_id, pair_hands_evaluation, evidence_scores
        )
    for specialist_weight in (0.25, 0.50, 0.75):
        name = f"family_route_a{int(specialist_weight * 100):02d}"
        if name in requested_names:
            mixed = (1.0 - specialist_weight) * evidence_scores + specialist_weight * routed_scores
            evidence_variants[name] = _rank_evidence(
                evaluation.pair_id, pair_hands_evaluation, mixed
            )
    for specialist_weight in (0.50, 0.75):
        name = f"family_semantic_a{int(specialist_weight * 100):02d}"
        if name in requested_names:
            mixed = (1.0 - specialist_weight) * evidence_scores + specialist_weight * routed_semantic_scores
            evidence_variants[name] = _rank_evidence(
                evaluation.pair_id, pair_hands_evaluation, mixed
            )
    return risk, family, evidence_variants


def _safe_json(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_submit(config: TrainConfig) -> dict[str, object]:
    """Run fixed pool OOF, train full-data candidates, and write submissions."""
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tables = _load_tables(config)
    development = tables["development"]
    evaluation = tables["evaluation"]
    labels = tables["labels"]
    evidence_truth = tables["evidence_truth"]
    pool_folds = tables["pool_folds"]
    pair_hands_development = tables["pair_hands_development"]
    pair_hands_evaluation = tables["pair_hands_evaluation"]
    pair_columns = tables["pair_columns"]
    hand_feature_columns = tables["hand_feature_columns"]
    family_feature_columns = tables["family_feature_columns"]
    fold_id = _fold_assignment(development, pool_folds)
    models_oof, family_oof, oof_details = _risk_oof(
        development,
        labels,
        pair_hands_development,
        fold_id,
        pair_columns,
        evidence_truth,
        config,
        hand_feature_columns,
        family_feature_columns,
    )
    truth = development.merge(labels[["pair_id", "label", "behavior_family"]], on="pair_id", validate="one_to_one")
    evidence_oof_variants = oof_details["evidence_variants"]
    oof_results: dict[str, object] = {}
    blend_risk = 0.45 * models_oof["hgb"] + 0.35 * models_oof["extra"] + 0.20 * models_oof["logistic"]
    risk_oof_candidates = {**models_oof, "blend": blend_risk}
    for evidence_name, evidence in evidence_oof_variants.items():
        for model_name, risk in risk_oof_candidates.items():
            candidate_name = f"{model_name}_{evidence_name}"
            chosen, grid = _score_grid(truth, risk, family_oof, evidence, evidence_truth)
            validate_submission(
                chosen,
                truth.pair_id.tolist(),
                pair_hands_development[["pair_id", "hand_id", "phase"]],
                phase="development",
            )
            oof_results[candidate_name] = grid
            chosen.to_csv(output / f"oof_{candidate_name}.csv", index=False)
    ranked = sorted(
        ((float(oof_results[name]["best_score"]["total"]), name) for name in oof_results),
        reverse=True,
    )
    selected_name = ranked[0][1]
    selected_evidence_name = selected_name.split("_", 1)[1]
    best_blend_evidence_name = max(
        evidence_oof_variants,
        key=lambda name: float(oof_results[f"blend_{name}"]["best_score"]["total"]),
    )
    full_evidence_names = tuple(dict.fromkeys(
        ("universal", selected_evidence_name, best_blend_evidence_name)
    ))
    risk_full, family_full, evidence_full_variants = _full_models(
        development,
        evaluation,
        labels,
        pair_hands_development,
        pair_hands_evaluation,
        evidence_truth,
        pair_columns,
        config,
        hand_feature_columns,
        family_feature_columns,
        evidence_names=full_evidence_names,
    )
    candidates: dict[str, pd.DataFrame] = {}
    risk_full_candidates = {
        **risk_full,
        "blend": 0.45 * risk_full["hgb"] + 0.35 * risk_full["extra"] + 0.20 * risk_full["logistic"],
    }
    for candidate_name, grid in oof_results.items():
        model_name, evidence_name = candidate_name.split("_", 1)
        if evidence_name not in evidence_full_variants:
            continue
        candidates[candidate_name] = _make_submission(
            evaluation,
            risk_full_candidates[model_name],
            family_full,
            evidence_full_variants[evidence_name],
            float(grid["best_family_threshold"]),
            float(grid["best_risk_threshold"]),
        )
    for name, submission in candidates.items():
        validate_submission(
            submission,
            evaluation.pair_id.tolist(),
            pair_hands_evaluation[["pair_id", "hand_id", "phase"]],
            phase="evaluation",
        )
        submission.to_csv(output / f"submission_{name}.csv", index=False)
    selected_submission = candidates[selected_name]
    selected_submission.to_csv(output / "submission_selected.csv", index=False)
    report = {
        "status": "completed",
        "config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "features": {
            "pair_columns": pair_columns,
            "pair_count_development": len(development),
            "pair_count_evaluation": len(evaluation),
            "pair_hand_count_development": len(pair_hands_development),
            "pair_hand_count_evaluation": len(pair_hands_evaluation),
            "hand_feature_columns": hand_feature_columns,
            "missing_optional_hand_features": tables["missing_optional_hand_features"],
        },
        "folds": {
            "n_splits": config.n_splits,
            "pool_counts": pd.Series(fold_id).value_counts().sort_index().to_dict(),
            "reports": oof_details["folds"],
        },
        "oof": oof_results,
        "evidence_variants": list(evidence_oof_variants),
        "full_evidence_variants": list(evidence_full_variants),
        "evidence_oof_rows": oof_details["evidence_oof_rows"],
        "selected_candidate": selected_name,
        "candidate_oof_totals": {name: float(oof_results[name]["best_score"]["total"]) for name in oof_results},
        "outputs": {name: str(output / f"submission_{name}.csv") for name in candidates},
        "selected_output": str(output / "submission_selected.csv"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_safe_json), encoding="utf-8"
    )
    pd.DataFrame({"pair_id": development.pair_id, "fold": fold_id}).to_csv(output / "pair_fold_manifest.csv", index=False)
    return report


__all__ = ["TrainConfig", "train_and_submit"]
