"""Probe population shift, exposure confounding, and pool-relative timing.

This is a read-only diagnostic for the real-data artifacts. It deliberately does
not fit a competition model or create a submission. The probe is designed to
answer whether the current labeled-only development cohort is a usable proxy
for the evaluation candidate population before expensive model work begins.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import duckdb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/features_v2"
DEFAULT_DATA = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
DEFAULT_OUTPUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/frontier_probe_20260917"


@dataclass(frozen=True)
class ProbeConfig:
    features_dir: str
    data_dir: str
    output_dir: str
    seed: int = 20260917
    max_eval_domain_rows: int = 4000
    domain_folds: int = 5


def _finite_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def _json_number(value: Any) -> float | int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _safe_auc(y_true: np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    values = np.asarray(scores, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0 or np.unique(y_true[mask]).size < 2:
        return None
    if np.unique(values[mask]).size < 2:
        return 0.5
    return float(roc_auc_score(y_true[mask], values[mask]))


def _safe_ap(y_true: np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    values = np.asarray(scores, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0 or y_true[mask].sum() == 0:
        return None
    return float(average_precision_score(y_true[mask], values[mask]))


def _pool_column(frame: pd.DataFrame) -> str:
    for candidate in ("pool_id", "table_id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("No pool grouping column found")


def _load_inputs(config: ProbeConfig) -> dict[str, pd.DataFrame]:
    feature_dir = Path(config.features_dir)
    data_dir = Path(config.data_dir)
    pair = pd.read_parquet(feature_dir / "pair_features.parquet")
    pair["pair_id"] = pair["pair_id"].astype(str)
    labels = pd.read_csv(data_dir / "development_labels.csv", dtype={"pair_id": str})
    labels["pair_id"] = labels["pair_id"].astype(str)
    evaluation_pairs = pd.read_csv(data_dir / "evaluation_pairs.csv", dtype={"pair_id": str})
    evaluation_pairs["pair_id"] = evaluation_pairs["pair_id"].astype(str)
    pair_hands = pd.read_parquet(
        feature_dir / "pair_hand_features_development.parquet"
    )
    pair_hands["pair_id"] = pair_hands["pair_id"].astype(str)
    return {
        "pair": pair,
        "labels": labels,
        "evaluation_pairs": evaluation_pairs,
        "pair_hands": pair_hands,
    }


def _population_summary(
    pair: pd.DataFrame,
    labels: pd.DataFrame,
    evaluation_pairs: pd.DataFrame,
    data_dir: Path,
) -> dict[str, Any]:
    def quote(path: Path) -> str:
        return "'" + str(path.resolve()).replace("'", "''") + "'"

    con = duckdb.connect()
    try:
        player_pool = con.execute(
            f"""
            SELECT player_id, min(table_id) AS pool_id
            FROM read_parquet({quote(data_dir / 'seats.parquet')}) s
            JOIN read_parquet({quote(data_dir / 'hands.parquet')}) h USING (hand_id)
            GROUP BY player_id
            """
        ).fetchdf()
    finally:
        con.close()
    pool_sizes = player_pool.groupby("pool_id", sort=False)["player_id"].nunique()
    possible_pairs = int((pool_sizes * (pool_sizes - 1) // 2).sum())
    dev_pair = pair.loc[pair["phase"].eq("development")].copy()
    eval_pair = pair.loc[pair["phase"].eq("evaluation")].copy()
    dev_labeled = dev_pair.merge(labels[["pair_id", "label"]], on="pair_id", how="inner")
    pool_col = _pool_column(pair)
    if pool_col in evaluation_pairs.columns:
        evaluation_pool_frame = evaluation_pairs
    else:
        evaluation_pool_frame = evaluation_pairs.merge(
            eval_pair[["pair_id", pool_col]], on="pair_id", how="left", validate="one_to_one"
        )
    dev_pool_count = int(dev_labeled[pool_col].nunique())
    eval_pool_count = int(eval_pair[pool_col].nunique())
    family_counts = (
        labels.loc[labels["label"].eq(1), "behavior_family"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    return {
        "players": {
            "pool_count": int(pool_sizes.size),
            "pool_size_min": int(pool_sizes.min()),
            "pool_size_median": float(pool_sizes.median()),
            "pool_size_max": int(pool_sizes.max()),
            "possible_unordered_pairs": possible_pairs,
        },
        "candidate_cohorts": {
            "development_labeled_pairs": int(len(dev_labeled)),
            "development_labeled_pools": dev_pool_count,
            "evaluation_pairs": int(len(evaluation_pairs)),
            "evaluation_feature_pairs": int(len(eval_pair)),
            "evaluation_feature_pools": eval_pool_count,
            "labeled_fraction_of_possible_pairs": float(len(dev_labeled) / possible_pairs),
            "evaluation_fraction_of_possible_pairs": float(len(evaluation_pairs) / possible_pairs),
            "evaluation_pairs_per_pool_mean": float(evaluation_pool_frame.groupby(
                pool_col, dropna=False
            ).size().mean()),
        },
        "labels": {
            "positive_count": int(labels["label"].sum()),
            "negative_count": int((labels["label"].eq(0)).sum()),
            "positive_rate_in_labeled_cohort": float(labels["label"].mean()),
            "positive_family_counts": family_counts,
        },
    }


def _domain_shift_probe(
    pair: pd.DataFrame,
    config: ProbeConfig,
) -> dict[str, Any]:
    development = pair.loc[pair["phase"].eq("development")].copy()
    evaluation = pair.loc[pair["phase"].eq("evaluation")].copy()
    numeric = [
        column for column in pair.columns
        if column.startswith("f_") and pd.api.types.is_numeric_dtype(pair[column])
    ]
    if not numeric:
        raise ValueError("No numeric f_* pair features found")
    exposure = [column for column in numeric if "shared_hands" in column]
    core = [column for column in numeric if column not in exposure]
    rng = np.random.default_rng(config.seed)
    eval_size = min(len(evaluation), max(len(development), config.max_eval_domain_rows))
    if eval_size > config.max_eval_domain_rows:
        eval_size = config.max_eval_domain_rows
    eval_indices = rng.choice(len(evaluation), size=eval_size, replace=False)
    sampled_eval = evaluation.iloc[eval_indices]
    sampled_dev = development
    domain = pd.concat(
        [sampled_dev.assign(_domain=0), sampled_eval.assign(_domain=1)],
        ignore_index=True,
    )
    groups = domain[_pool_column(domain)].astype(str).to_numpy()
    y = domain["_domain"].to_numpy(dtype=int)

    def evaluate_columns(columns: list[str], include_univariate: bool = True) -> dict[str, Any]:
        x = domain[columns].apply(_finite_series)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=300,
                solver="liblinear",
                random_state=config.seed,
            ),
        )
        fold_scores: list[float] = []
        splitter = GroupKFold(n_splits=config.domain_folds)
        for train_idx, valid_idx in splitter.split(x, y, groups):
            model.fit(x.iloc[train_idx], y[train_idx])
            prediction = model.predict_proba(x.iloc[valid_idx])[:, 1]
            fold_scores.append(float(roc_auc_score(y[valid_idx], prediction)))
        univariate: list[dict[str, Any]] = []
        if include_univariate:
            for column in columns:
                values = x[column]
                auc = _safe_auc(y, values.to_numpy(dtype=float))
                if auc is None:
                    continue
                univariate.append({
                    "feature": column,
                    "auc": auc,
                    "distance_from_random": abs(auc - 0.5),
                    "development_median": _json_number(values.iloc[:len(sampled_dev)].median()),
                    "evaluation_median": _json_number(values.iloc[len(sampled_dev):].median()),
                })
        univariate.sort(key=lambda row: row["distance_from_random"], reverse=True)
        return {
            "feature_count": len(columns),
            "group_cv_auc": fold_scores,
            "group_cv_auc_mean": float(np.mean(fold_scores)),
            "group_cv_auc_min": float(np.min(fold_scores)),
            "top_univariate_features": univariate[:20],
        }

    no_rank = [column for column in core if "rank_pct" not in column]
    no_rank_extreme = [
        column for column in no_rank
        if not any(column.endswith(suffix) for suffix in ("_max", "_q90", "_std"))
    ]
    mean_meta = [
        column for column in no_rank
        if column.endswith("_mean") or column in {
            "f_account_age_abs_gap",
            "f_experience_match",
            "f_preferred_stake_match",
            "f_region_match",
            "f_client_match",
            "f_generic_tail_score",
            "f_directional_transfer_imbalance",
            "f_directional_transfer_abs_imbalance",
            "f_pressure_direction_gap",
            "f_pressure_abs_gap",
            "f_outsider_pressure_total",
            "f_partner_fold_sequence",
            "f_mutual_response_total",
        }
    ]
    feature_ablation = {
        "no_rank_pct": evaluate_columns(no_rank, include_univariate=False),
        "no_rank_or_extreme": evaluate_columns(no_rank_extreme, include_univariate=False),
        "mean_and_meta": evaluate_columns(mean_meta, include_univariate=False),
    }

    return {
        "domain_definition": "development labeled pair versus evaluation feature pair",
        "development_rows": int(len(sampled_dev)),
        "evaluation_rows_sampled": int(len(sampled_eval)),
        "pool_group_column": _pool_column(domain),
        "all_numeric_features": evaluate_columns(numeric),
        "non_exposure_numeric_features": evaluate_columns(core),
        "feature_family_ablation": feature_ablation,
        "exposure_features_excluded": exposure,
    }


def _pool_time_summary(data_dir: Path) -> dict[str, Any]:
    hands_path = data_dir / "hands.parquet"
    columns = ["table_id", "phase", "started_at"]
    hands = pd.read_parquet(hands_path, columns=columns)
    hands["started_at"] = pd.to_datetime(hands["started_at"], utc=True, errors="coerce")
    global_phase = (
        hands.dropna(subset=["started_at"])
        .groupby("phase")["started_at"]
        .agg(["count", "min", "max"])
        .reset_index()
    )
    phase_pivot = hands.pivot_table(
        index="table_id",
        columns="phase",
        values="started_at",
        aggfunc=["min", "max", "count"],
    )
    result: dict[str, Any] = {
        "global_phase_ranges": [
            {
                "phase": str(row["phase"]),
                "count": int(row["count"]),
                "min": row["min"].isoformat(),
                "max": row["max"].isoformat(),
            }
            for _, row in global_phase.iterrows()
        ],
        "table_group_count": int(hands["table_id"].nunique()),
    }
    if isinstance(phase_pivot.columns, pd.MultiIndex) and "development" in phase_pivot.columns.get_level_values(1):
        try:
            dev_max = phase_pivot[("max", "development")]
            eval_min = phase_pivot[("min", "evaluation")]
            both = dev_max.notna() & eval_min.notna()
            before = both & (dev_max < eval_min)
            after = both & (dev_max > eval_min)
            result["same_table_phase_order"] = {
                "tables_with_both_phases": int(both.sum()),
                "development_entirely_before_evaluation": int(before.sum()),
                "phases_interleaved_or_equal": int((both & ~before & ~after).sum()),
                "development_extends_after_evaluation": int(after.sum()),
            }
        except KeyError:
            pass
    return result


def _temporal_signal_probe(
    pair_hands: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    pool_col = _pool_column(pair_hands)
    required = {"pair_id", pool_col, "hand_id", "started_at"}
    missing = required - set(pair_hands.columns)
    if missing:
        raise ValueError(f"Missing temporal columns: {sorted(missing)}")
    frame = pair_hands.copy()
    frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["started_at"])
    hand_time = frame[[pool_col, "hand_id", "started_at"]].drop_duplicates()
    hand_time = hand_time.sort_values([pool_col, "started_at", "hand_id"])
    hand_time["pool_time_pct"] = hand_time.groupby(pool_col, sort=False).cumcount()
    denominators = hand_time.groupby(pool_col, sort=False)["hand_id"].transform("size").sub(1).clip(lower=1)
    hand_time["pool_time_pct"] = hand_time["pool_time_pct"] / denominators
    frame = frame.drop(columns=["pool_time_pct"], errors="ignore").merge(
        hand_time[[pool_col, "hand_id", "pool_time_pct"]],
        on=[pool_col, "hand_id"],
        how="left",
        validate="many_to_one",
    )
    frame["outsider_pressure"] = frame.get("outsider_fold_after_a", 0) + frame.get(
        "outsider_fold_after_b", 0
    )
    frame["mutual_response"] = frame.get("b_response_after_a", 0) + frame.get(
        "a_response_after_b", 0
    )
    metrics = [
        name for name in (
            "hand_risk_proxy",
            "transfer_max_bb",
            "outsider_pressure",
            "mutual_response",
            "mutual_pressure_sequence",
        ) if name in frame.columns
    ]
    labels_small = labels[["pair_id", "label"]].drop_duplicates("pair_id")
    windows = {
        "early": (0.0, 1.0 / 3.0),
        "middle": (1.0 / 3.0, 2.0 / 3.0),
        "late": (2.0 / 3.0, 1.000001),
        "full": (0.0, 1.000001),
    }
    rows: list[dict[str, Any]] = []
    for window_name, (lower, upper) in windows.items():
        selected = frame.loc[frame["pool_time_pct"].between(lower, upper, inclusive="left")]
        if window_name == "late":
            selected = frame.loc[frame["pool_time_pct"].ge(lower)]
        if selected.empty:
            continue
        grouped = selected.groupby("pair_id", sort=False)
        for metric in metrics:
            values = grouped[metric].agg(
                mean="mean",
                maximum="max",
                q90=lambda series: series.quantile(0.9),
            ).reset_index()
            joined = labels_small.merge(values, on="pair_id", how="inner", validate="one_to_one")
            y = joined["label"].to_numpy(dtype=int)
            for aggregation in ("mean", "maximum", "q90"):
                score = _finite_series(joined[aggregation]).to_numpy(dtype=float)
                rows.append({
                    "window": window_name,
                    "metric": metric,
                    "aggregation": aggregation,
                    "pair_count": int(len(joined)),
                    "positive_count": int(y.sum()),
                    "ap": _safe_ap(y, score),
                    "auc": _safe_auc(y, score),
                    "positive_median": _json_number(joined.loc[joined["label"].eq(1), aggregation].median()),
                    "negative_median": _json_number(joined.loc[joined["label"].eq(0), aggregation].median()),
                })
    valid_rows = [row for row in rows if row["ap"] is not None]
    top_by_ap = sorted(valid_rows, key=lambda row: row["ap"], reverse=True)[:20]
    return {
        "pool_group_column": pool_col,
        "pair_hand_rows": int(len(frame)),
        "unique_pairs": int(frame["pair_id"].nunique()),
        "time_reference_unique_hands": int(len(hand_time)),
        "metrics": metrics,
        "window_results": rows,
        "top_window_signals_by_ap": top_by_ap,
    }


def run_probe(config: ProbeConfig) -> dict[str, Any]:
    inputs = _load_inputs(config)
    data_dir = Path(config.data_dir)
    report: dict[str, Any] = {
        "probe": "frontier_shift_probe",
        "config": asdict(config),
        "population": _population_summary(
            inputs["pair"], inputs["labels"], inputs["evaluation_pairs"], data_dir
        ),
        "domain_shift": _domain_shift_probe(inputs["pair"], config),
        "pool_time": _pool_time_summary(data_dir),
        "temporal_signals": _temporal_signal_probe(inputs["pair_hands"], inputs["labels"]),
        "limitations": [
            "The development temporal probe only covers labeled pair-hand rows, not the full unlabelled development pair universe.",
            "A domain classifier establishes cohort separability, not its causal source or a guaranteed leaderboard gain.",
            "The temporal AP/AUC rows are descriptive diagnostics and are not an official evaluation score.",
        ],
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frontier_shift_probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=DEFAULT_FEATURES)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--max-eval-domain-rows", type=int, default=4000)
    parser.add_argument("--domain-folds", type=int, default=5)
    args = parser.parse_args()
    return ProbeConfig(
        features_dir=args.features_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_eval_domain_rows=args.max_eval_domain_rows,
        domain_folds=args.domain_folds,
    )


if __name__ == "__main__":
    result = run_probe(_parse_args())
    print(json.dumps({
        "population": result["population"],
        "domain_shift": {
            "all_auc": result["domain_shift"]["all_numeric_features"]["group_cv_auc_mean"],
            "core_auc": result["domain_shift"]["non_exposure_numeric_features"]["group_cv_auc_mean"],
            "ablation_auc": {
                key: value["group_cv_auc_mean"]
                for key, value in result["domain_shift"]["feature_family_ablation"].items()
            },
        },
        "pool_time": result["pool_time"],
        "top_temporal_signals": result["temporal_signals"]["top_window_signals_by_ap"][:10],
    }, ensure_ascii=False, indent=2, default=str))
