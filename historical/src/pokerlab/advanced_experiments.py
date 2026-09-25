"""Fold-safe experiments for the research-pack modules not used in the first release.

The implementations are deliberately modest and auditable. They measure PNU,
LOFO generic detection, an outcome-value proxy, gated instance aggregation, and
pool-relative graph features on the same fixed development pool folds. None of
these diagnostics is allowed to alter the released submission automatically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .metrics import KNOWN, evidence_ap5


@dataclass(frozen=True)
class AdvancedConfig:
    features_dir: str
    data_dir: str
    output_dir: str
    n_splits: int = 5
    seed: int = 20260917
    max_iter: int = 160
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 10
    n_jobs: int = 8


class _ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = np.full(len(frame), self.value, dtype=float)
        return np.column_stack([1.0 - p, p])


def _read_csv(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=list(columns) if columns is not None else None,
        dtype={"pair_id": str, "hand_id": str},
    )


def _finite(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = frame[columns].copy()
    values = values.replace([np.inf, -np.inf], np.nan)
    for column in columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
        median = values[column].median()
        values[column] = values[column].fillna(0.0 if pd.isna(median) else median)
    return values.astype(np.float32)


def _balance(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = np.ones(len(y), dtype=float)
    for label in (0, 1):
        if counts[label] > 0:
            weights[y == label] = len(y) / (2.0 * counts[label])
    return weights


def _fit_classifier(X: pd.DataFrame, y: np.ndarray, config: AdvancedConfig, seed_offset: int = 0):
    labels = np.unique(y)
    if len(labels) == 1:
        return _ConstantModel(float(labels[0]))
    model = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        learning_rate=0.045,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=config.seed + seed_offset,
    )
    with threadpool_limits(limits=1):
        model.fit(X, y, sample_weight=_balance(y))
    return model


def _predict_positive(model, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    if raw.shape[1] == 1:
        return np.zeros(len(X), dtype=float)
    classes = list(getattr(model, "classes_", (0, 1)))
    return raw[:, classes.index(1)] if 1 in classes else np.zeros(len(X), dtype=float)


def _fit_regressor(X: pd.DataFrame, y: np.ndarray, config: AdvancedConfig, seed_offset: int = 0):
    model = HistGradientBoostingRegressor(
        max_iter=max(80, config.max_iter // 2),
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        learning_rate=0.05,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=config.seed + seed_offset,
    )
    with threadpool_limits(limits=1):
        model.fit(X, y)
    return model


def _average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    if int(np.asarray(y).sum()) == 0:
        return 0.0
    return float(average_precision_score(y, scores))


def _fold_ids(pairs: pd.DataFrame, features_dir: Path) -> np.ndarray:
    manifest = _read_csv(features_dir / "pool_folds.csv")
    mapping = dict(zip(manifest.table_id.astype(str), manifest.fold.astype(int)))
    values = pairs.table_id.astype(str).map(mapping)
    if values.isna().any():
        raise ValueError("Pool fold manifest does not cover every development pair")
    return values.to_numpy(dtype=int)


def _evidence_truth(path: Path) -> dict[str, set[str]]:
    frame = _read_csv(path, ["pair_id", "hand_id"])
    return frame.groupby("pair_id").hand_id.agg(set).to_dict()


def _load_tables(config: AdvancedConfig) -> dict[str, object]:
    features_dir = Path(config.features_dir)
    data_dir = Path(config.data_dir)
    pairs = pd.read_parquet(features_dir / "pair_features.parquet")
    pairs["pair_id"] = pairs.pair_id.astype(str)
    pairs = pairs.loc[pairs.phase.eq("development")].reset_index(drop=True)
    labels = _read_csv(data_dir / "development_labels.csv")
    labels["pair_id"] = labels.pair_id.astype(str)
    joined = pairs.merge(
        labels[["pair_id", "label", "behavior_family"]],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(pairs):
        raise ValueError("Every development pair must have one label")
    hands = pd.read_parquet(features_dir / "pair_hand_features_development.parquet")
    hands["pair_id"] = hands.pair_id.astype(str)
    hands["hand_id"] = hands.hand_id.astype(str)
    pair_columns = [
        column for column in pairs.columns
        if column.startswith("f_")
        and column != "f_shared_hands_reported"
        and pd.api.types.is_numeric_dtype(pairs[column])
    ]
    if not pair_columns:
        raise ValueError("No numeric pair features found")
    hand_columns = [
        column for column in hands.columns
        if column in {
            "pot_bb", "final_pot_bb", "players_dealt", "players_at_showdown",
            "board_count", "board_rank_sum", "a_action_count", "b_action_count",
            "a_pressure_count", "b_pressure_count", "a_fold_rate", "b_fold_rate",
            "a_call_rate", "b_call_rate", "a_net_bb", "b_net_bb", "net_gap_bb",
            "transfer_max_bb", "transfer_sum_bb", "showdown_both", "hand_risk_proxy",
            "b_response_after_a", "a_response_after_b", "outsider_fold_after_a",
            "outsider_fold_after_b", "mutual_pressure_sequence",
        }
    ]
    required_value = {"transfer_max_bb", "pot_bb"}
    if not required_value.issubset(hand_columns):
        raise ValueError(f"Value proxy requires hand columns: {sorted(required_value)}")
    return {
        "pairs": joined,
        "hands": hands,
        "pair_columns": pair_columns,
        "hand_columns": hand_columns,
        "fold_id": _fold_ids(joined, features_dir),
        "evidence_truth": _evidence_truth(data_dir / "development_evidence.csv"),
    }


class _PNULogistic:
    """Small non-negative PU-risk optimizer with an explicit class-prior input."""

    def __init__(self, prior: float, steps: int = 360, learning_rate: float = 0.025, l2: float = 0.01):
        self.prior = float(prior)
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

    def fit(self, X: pd.DataFrame, positive_mask: np.ndarray) -> "_PNULogistic":
        array = np.asarray(X, dtype=np.float64)
        self.mean_ = array.mean(axis=0)
        self.scale_ = array.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        normalized = (array - self.mean_) / self.scale_
        positive = np.asarray(positive_mask, dtype=bool)
        unlabeled = ~positive
        if not positive.any() or not unlabeled.any():
            self.weights_ = np.zeros(array.shape[1], dtype=float)
            self.bias_ = 0.0
            return self
        weights = np.zeros(array.shape[1], dtype=float)
        bias = 0.0
        first = np.zeros_like(weights)
        second = np.zeros_like(weights)
        first_bias = 0.0
        second_bias = 0.0
        for step in range(1, self.steps + 1):
            logits = normalized @ weights + bias
            probabilities = self._sigmoid(logits)
            grad_logits = np.zeros(len(array), dtype=float)
            positive_count = max(int(positive.sum()), 1)
            unlabeled_count = max(int(unlabeled.sum()), 1)
            grad_logits[positive] += self.prior * (probabilities[positive] - 1.0) / positive_count
            correction = (
                np.logaddexp(0.0, logits[unlabeled]).mean()
                - self.prior * np.logaddexp(0.0, -logits[positive]).mean()
            )
            if correction > 0.0:
                grad_logits[unlabeled] += probabilities[unlabeled] / unlabeled_count
                grad_logits[positive] -= self.prior * probabilities[positive] / positive_count
            gradient = normalized.T @ grad_logits + self.l2 * weights
            gradient_bias = float(grad_logits.sum())
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            first_bias = 0.9 * first_bias + 0.1 * gradient_bias
            second_bias = 0.999 * second_bias + 0.001 * gradient_bias * gradient_bias
            bias -= self.learning_rate * (first_bias / (1.0 - 0.9**step)) / (np.sqrt(second_bias / (1.0 - 0.999**step)) + 1e-8)
            weights -= self.learning_rate * (first / (1.0 - 0.9**step)) / (np.sqrt(second / (1.0 - 0.999**step)) + 1e-8)
        self.weights_ = weights
        self.bias_ = bias
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        normalized = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return normalized @ self.weights_ + self.bias_


def _run_pnu(pairs: pd.DataFrame, pair_columns: list[str], fold_id: np.ndarray, config: AdvancedConfig) -> tuple[dict, pd.DataFrame]:
    X = _finite(pairs, pair_columns)
    y = pairs.label.to_numpy(dtype=int)
    rows = []
    oof_frame = pd.DataFrame({"pair_id": pairs.pair_id, "label": y})
    for prior in (0.001, 0.005, 0.01, 0.02, 0.05):
        scores = np.zeros(len(pairs), dtype=float)
        fold_ap = []
        for fold in range(config.n_splits):
            train_mask = fold_id != fold
            valid_mask = fold_id == fold
            model = _PNULogistic(prior).fit(X.loc[train_mask], y[train_mask] == 1)
            scores[valid_mask] = model.decision_function(X.loc[valid_mask])
            fold_ap.append(_average_precision(y[valid_mask], scores[valid_mask]))
        name = f"pnu_prior_{prior:g}"
        oof_frame[name] = scores
        rows.append({
            "prior": prior,
            "pair_ap": _average_precision(y, scores),
            "fold_pair_ap": fold_ap,
            "positive_count": int(y.sum()),
            "unlabeled_count": int((y == 0).sum()),
        })
    best = max(rows, key=lambda row: row["pair_ap"])
    return {
        "method": "formal_non_negative_pu_logistic_surrogate",
        "priors": rows,
        "selected_by_oof_for_diagnostic_only": best["prior"],
        "selection_warning": "The prior grid is a sensitivity analysis, not an estimate of evaluation prevalence.",
    }, oof_frame


def _run_lofo(pairs: pd.DataFrame, pair_columns: list[str], fold_id: np.ndarray, config: AdvancedConfig) -> tuple[dict, pd.DataFrame]:
    X = _finite(pairs, pair_columns)
    y = pairs.label.to_numpy(dtype=int)
    families = pairs.behavior_family.astype(str).to_numpy()
    oof_frame = pd.DataFrame({"pair_id": pairs.pair_id, "label": y})
    rows = []
    for hidden in KNOWN:
        scores = np.zeros(len(pairs), dtype=float)
        hidden_truth = ((y == 1) & (families == hidden)).astype(int)
        fold_ap = []
        for fold in range(config.n_splits):
            train_mask = fold_id != fold
            valid_mask = fold_id == fold
            train_mask &= ~((y == 1) & (families == hidden))
            model = _fit_classifier(X.loc[train_mask], y[train_mask], config, seed_offset=fold + 31)
            scores[valid_mask] = _predict_positive(model, X.loc[valid_mask])
            fold_ap.append(_average_precision(hidden_truth[valid_mask], scores[valid_mask]))
        name = f"lofo_hidden_{hidden}"
        oof_frame[name] = scores
        rows.append({
            "hidden_family": hidden,
            "held_out_family_pair_ap": _average_precision(hidden_truth, scores),
            "fold_pair_ap": fold_ap,
            "hidden_positive_count": int(hidden_truth.sum()),
            "hidden_family_used_in_training": False,
        })
    return {
        "method": "pool_disjoint_leave_one_known_family_out_generic_detector",
        "families": rows,
        "interpretation": "Held-out-family AP is a controlled transfer diagnostic, not an official leaderboard score.",
    }, oof_frame


def _group_values(pair_ids: pd.Series, values: np.ndarray, reducer: str) -> pd.DataFrame:
    codes, unique = pd.factorize(pair_ids.astype(str), sort=False)
    values = np.asarray(values, dtype=float)
    result = pd.DataFrame({"pair_id": unique.astype(str)})
    if reducer == "max":
        aggregate = np.full(len(unique), -np.inf, dtype=float)
        np.maximum.at(aggregate, codes, values)
    elif reducer == "mean":
        aggregate = np.bincount(codes, weights=values, minlength=len(unique))
        counts = np.bincount(codes, minlength=len(unique))
        aggregate = aggregate / np.maximum(counts, 1)
    else:
        raise ValueError(f"Unknown group reducer: {reducer}")
    result["value"] = aggregate
    return result


def _run_value_proxy(hands: pd.DataFrame, pairs: pd.DataFrame, hand_columns: list[str], fold_id: np.ndarray, config: AdvancedConfig) -> tuple[dict, pd.DataFrame]:
    pair_fold = dict(zip(pairs.pair_id.astype(str), fold_id))
    hands = hands.copy()
    hands["fold"] = hands.pair_id.map(pair_fold)
    feature_columns = [
        column for column in (
            "pot_bb", "final_pot_bb", "players_dealt", "players_at_showdown",
            "board_count", "board_rank_sum", "a_action_count", "b_action_count",
            "a_pressure_count", "b_pressure_count", "a_fold_rate", "b_fold_rate",
            "a_call_rate", "b_call_rate", "hand_risk_proxy",
        )
        if column in hand_columns
    ]
    observed = pd.to_numeric(hands["transfer_max_bb"], errors="coerce").fillna(0.0).clip(-1000.0, 1000.0).to_numpy(float)
    scores = np.zeros(len(pairs), dtype=float)
    raw_scores = np.zeros(len(pairs), dtype=float)
    pair_index = dict(zip(pairs.pair_id.astype(str), np.arange(len(pairs))))
    for fold in range(config.n_splits):
        train_hands = hands.loc[hands.fold != fold]
        valid_hands = hands.loc[hands.fold == fold]
        model = _fit_regressor(
            _finite(train_hands, feature_columns),
            observed[train_hands.index.to_numpy()],
            config,
            seed_offset=fold + 101,
        )
        predicted = model.predict(_finite(valid_hands, feature_columns))
        residual = observed[valid_hands.index.to_numpy()] - predicted
        valid_ids = valid_hands.pair_id.astype(str)
        residual_max = _group_values(valid_ids, residual, "max").set_index("pair_id")["value"]
        residual_positive = _group_values(valid_ids, np.maximum(residual, 0.0), "mean").set_index("pair_id")["value"]
        raw_max = _group_values(valid_ids, observed[valid_hands.index.to_numpy()], "max").set_index("pair_id")["value"]
        for pid in residual_max.index:
            position = pair_index[pid]
            scores[position] = float(residual_max.loc[pid] + 0.5 * residual_positive.loc[pid])
            raw_scores[position] = float(raw_max.loc[pid])
    y = pairs.label.to_numpy(dtype=int)
    frame = pd.DataFrame({"pair_id": pairs.pair_id, "label": y, "value_proxy_oof": scores, "raw_transfer_oof": raw_scores})
    return {
        "method": "cross_fitted_observed_transfer_value_residual_proxy",
        "pair_ap": _average_precision(y, scores),
        "raw_transfer_max_pair_ap": _average_precision(y, raw_scores),
        "hand_regression_target": "transfer_max_bb",
        "features": feature_columns,
        "counterfactual_status": "Not a legal-action rollout or causal effect; use only as a directional outcome-value diagnostic.",
    }, frame


def _attention_bag(pair_ids: pd.Series, values: pd.DataFrame, gate_scores: np.ndarray) -> pd.DataFrame:
    codes, unique = pd.factorize(pair_ids.astype(str), sort=False)
    gate_scores = np.asarray(gate_scores, dtype=float)
    group_max = np.full(len(unique), -np.inf, dtype=float)
    np.maximum.at(group_max, codes, gate_scores)
    exponent = np.exp(np.clip(gate_scores - group_max[codes], -40.0, 40.0))
    denominators = np.bincount(codes, weights=exponent, minlength=len(unique))
    attention = exponent / np.maximum(denominators[codes], 1e-12)
    output = pd.DataFrame({"pair_id": unique.astype(str)})
    entropy = np.bincount(
        codes,
        weights=-attention * np.log(np.maximum(attention, 1e-12)),
        minlength=len(unique),
    )
    counts = np.bincount(codes, minlength=len(unique))
    output["mil_gate_max"] = group_max
    output["mil_gate_entropy"] = entropy / np.maximum(np.log(np.maximum(counts, 2)), 1.0)
    for column in values.columns:
        numeric = pd.to_numeric(values[column], errors="coerce").fillna(0.0).to_numpy(float)
        weighted = np.bincount(codes, weights=attention * numeric, minlength=len(unique))
        maximum = np.full(len(unique), -np.inf, dtype=float)
        np.maximum.at(maximum, codes, numeric)
        output[f"mil_mean_{column}"] = weighted
        output[f"mil_max_{column}"] = maximum
    return output


def _evidence_map_for_bag(hands: pd.DataFrame, gate_scores: np.ndarray, truth: dict[str, set[str]], positive_ids: set[str]) -> float:
    frame = hands[["pair_id", "hand_id"]].copy()
    frame["score"] = np.asarray(gate_scores, dtype=float)
    values = []
    for pair_id, group in frame.groupby("pair_id", sort=False):
        if str(pair_id) not in positive_ids:
            continue
        ordered = group.sort_values(["score", "hand_id"], ascending=[False, True], kind="mergesort")
        values.append(evidence_ap5(ordered.hand_id.tolist()[:5], truth.get(str(pair_id), set())))
    return float(np.mean(values)) if values else 0.0


def _run_mil(hands: pd.DataFrame, pairs: pd.DataFrame, hand_columns: list[str], fold_id: np.ndarray, evidence_truth: dict[str, set[str]], config: AdvancedConfig) -> tuple[dict, pd.DataFrame]:
    gate_columns = [
        column for column in (
            "transfer_max_bb", "pot_bb", "final_pot_bb", "players_at_showdown",
            "a_pressure_rate", "b_pressure_rate", "a_fold_rate", "b_fold_rate",
            "a_call_rate", "b_call_rate", "hand_risk_proxy",
        )
        if column in hand_columns
    ]
    bag_columns = [column for column in ("transfer_max_bb", "pot_bb", "showdown_both", "hand_risk_proxy", "mutual_pressure_sequence") if column in hand_columns]
    pair_index = dict(zip(pairs.pair_id.astype(str), np.arange(len(pairs))))
    y = pairs.label.to_numpy(dtype=int)
    scores = np.zeros(len(pairs), dtype=float)
    evidence_scores = []
    for fold in range(config.n_splits):
        train_ids = set(pairs.loc[fold_id != fold, "pair_id"].astype(str))
        valid_ids = set(pairs.loc[fold_id == fold, "pair_id"].astype(str))
        train_hands = hands.loc[hands.pair_id.isin(train_ids)]
        valid_hands = hands.loc[hands.pair_id.isin(valid_ids)]
        train_targets = np.asarray([
            int(str(hand) in evidence_truth.get(str(pair), set()))
            for pair, hand in zip(train_hands.pair_id, train_hands.hand_id)
        ], dtype=int)
        gate = _fit_classifier(_finite(train_hands, gate_columns), train_targets, config, seed_offset=fold + 151)
        train_gate = _predict_positive(gate, _finite(train_hands, gate_columns))
        valid_gate = _predict_positive(gate, _finite(valid_hands, gate_columns))
        train_bag = _attention_bag(train_hands.pair_id, train_hands[bag_columns], train_gate)
        valid_bag = _attention_bag(valid_hands.pair_id, valid_hands[bag_columns], valid_gate)
        bag_features = [column for column in train_bag.columns if column != "pair_id"]
        train_labeled = train_bag.merge(pairs[["pair_id", "label"]], on="pair_id", validate="one_to_one")
        model = _fit_classifier(_finite(train_labeled, bag_features), train_labeled.label.to_numpy(dtype=int), config, seed_offset=fold + 181)
        valid_scores = _predict_positive(model, _finite(valid_bag, bag_features))
        for pair_id, score in zip(valid_bag.pair_id.astype(str), valid_scores):
            scores[pair_index[pair_id]] = float(score)
        positive_valid_ids = set(pairs.loc[(fold_id == fold) & pairs.label.eq(1), "pair_id"].astype(str))
        evidence_scores.append(_evidence_map_for_bag(valid_hands, valid_gate, evidence_truth, positive_valid_ids))
    oof = pd.DataFrame({"pair_id": pairs.pair_id, "label": y, "mil_pair_oof": scores})
    return {
        "method": "evidence_gated_instance_aggregation_with_separate_pair_head",
        "pair_ap": _average_precision(y, scores),
        "evidence_map5": float(np.mean(evidence_scores)) if evidence_scores else 0.0,
        "gate_features": gate_columns,
        "bag_features": bag_columns,
        "fold_evidence_map5": evidence_scores,
        "attention_is_not_submission_evidence": True,
    }, oof


def _graph_frame(pairs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    candidates = [
        "f_transfer_max_bb_mean", "f_hand_risk_proxy_q90", "f_outsider_pressure_total",
        "f_showdown_both_rate", "f_net_gap_bb_mean", "f_mutual_pressure_rate",
    ]
    available = [column for column in candidates if column in pairs.columns]
    if not available:
        raise ValueError("No graph residual features found")
    output = pd.DataFrame({"pair_id": pairs.pair_id.astype(str)})
    for column in available:
        values = pd.to_numeric(pairs[column], errors="coerce")
        median = values.groupby(pairs.table_id.astype(str)).transform("median")
        scale = median.abs() + 1.0
        output[f"graph_residual_{column}"] = (values - median).fillna(0.0)
        output[f"graph_relative_{column}"] = ((values - median) / scale).fillna(0.0)
        output[f"graph_rank_{column}"] = values.groupby(pairs.table_id.astype(str)).rank(pct=True).fillna(0.5)
    output["graph_pool_pair_count"] = pairs.groupby(pairs.table_id.astype(str))["pair_id"].transform("count").to_numpy(dtype=float)
    return output, [column for column in output.columns if column != "pair_id"]


def _run_graph(pairs: pd.DataFrame, fold_id: np.ndarray, config: AdvancedConfig) -> tuple[dict, pd.DataFrame]:
    graph, graph_columns = _graph_frame(pairs)
    y = pairs.label.to_numpy(dtype=int)
    scores = np.zeros(len(pairs), dtype=float)
    for fold in range(config.n_splits):
        train_mask = fold_id != fold
        valid_mask = fold_id == fold
        model = _fit_classifier(_finite(graph, graph_columns).loc[train_mask], y[train_mask], config, seed_offset=fold + 211)
        scores[valid_mask] = _predict_positive(model, _finite(graph, graph_columns).loc[valid_mask])
    output = pd.DataFrame({"pair_id": pairs.pair_id, "label": y, "graph_residual_oof": scores})
    return {
        "method": "pool_relative_multi_relation_residual_features",
        "pair_ap": _average_precision(y, scores),
        "features": graph_columns,
        "pool_identity_used_as_model_feature": False,
        "pool_aggregates_use_unlabeled_gameplay_only": True,
    }, output


def run_advanced(config: AdvancedConfig) -> dict[str, object]:
    tables = _load_tables(config)
    pairs = tables["pairs"]
    hands = tables["hands"]
    pair_columns = tables["pair_columns"]
    hand_columns = tables["hand_columns"]
    fold_id = tables["fold_id"]
    evidence_truth = tables["evidence_truth"]
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    pnu, pnu_oof = _run_pnu(pairs, pair_columns, fold_id, config)
    pnu_oof.to_csv(output / "pnu_oof.csv", index=False)
    lofo, lofo_oof = _run_lofo(pairs, pair_columns, fold_id, config)
    lofo_oof.to_csv(output / "lofo_oof.csv", index=False)
    value, value_oof = _run_value_proxy(hands, pairs, hand_columns, fold_id, config)
    value_oof.to_csv(output / "value_proxy_oof.csv", index=False)
    mil, mil_oof = _run_mil(hands, pairs, hand_columns, fold_id, evidence_truth, config)
    mil_oof.to_csv(output / "mil_oof.csv", index=False)
    graph, graph_oof = _run_graph(pairs, fold_id, config)
    graph_oof.to_csv(output / "graph_oof.csv", index=False)
    report = {
        "status": "completed",
        "config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data": {
            "pair_count": len(pairs),
            "pair_hand_count": len(hands),
            "positive_count": int(pairs.label.sum()),
            "pair_features": pair_columns,
            "hand_features": hand_columns,
            "fold_counts": pd.Series(fold_id).value_counts().sort_index().to_dict(),
        },
        "pnu": pnu,
        "lofo": lofo,
        "value_impact": value,
        "gated_mil": mil,
        "graph": graph,
        "outputs": {
            "pnu_oof": str(output / "pnu_oof.csv"),
            "lofo_oof": str(output / "lofo_oof.csv"),
            "value_proxy_oof": str(output / "value_proxy_oof.csv"),
            "mil_oof": str(output / "mil_oof.csv"),
            "graph_oof": str(output / "graph_oof.csv"),
        },
        "release_decision": "Diagnostics only; promote a branch only after independent paired pool evidence and legal submission validation.",
    }
    (output / "advanced_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return report


__all__ = ["AdvancedConfig", "run_advanced"]
