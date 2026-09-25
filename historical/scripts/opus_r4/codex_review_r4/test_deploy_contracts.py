from pathlib import Path
import sys

import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path(
    "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
)
DEPENDENCY_ROOT = Path(
    "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"
)
sys.path.insert(0, str(DEPENDENCY_ROOT))
import ci_censored_event as censored


def assemble_like_deploy(frame, new_features, role_features):
    new_columns = [
        column
        for column in new_features.columns
        if column not in ("slot", "h", "pa", "pb", "pair_id")
    ]
    role_columns = [
        column
        for column in role_features.columns
        if column.startswith("x_") and column not in ("x_k", "x_n")
    ]
    original_keys = set(map(tuple, frame[["slot", "h"]].to_numpy()))
    result = frame.merge(
        new_features[["slot", "h"] + new_columns],
        on=["slot", "h"],
        how="left",
        validate="one_to_one",
    )
    missing_new = result[new_columns].isna().any(axis=1)
    result[new_columns] = result[new_columns].fillna(0.0)
    result = result.merge(
        role_features[["slot", "h"] + role_columns],
        on=["slot", "h"],
        how="left",
        validate="one_to_one",
    )
    result = result.sort_values(["slot", "ts", "h"], kind="stable").reset_index(
        drop=True
    )
    cell_columns = []
    cells = {
        "sp": result.both_flop.astype(float),
        "ci": ((result.pa_at_trig == 6) & result.y1.isin([2, 3])).astype(float),
        "fold": result.x_s_fold_to_r.astype(float),
        "sfold": (
            result.x_s_fold_to_r * (result.x_hsS_last >= 0.55)
        ).astype(float),
        "hu": (result.both_flop & result.all_out_folded).astype(float),
    }
    for name, values in cells.items():
        result[f"c_{name}"] = values
        result[f"c_k_{name}"] = values.groupby(result.slot).cumsum() - values
        result[f"c_n_{name}"] = values.groupby(result.slot).transform("sum")
        result[f"c_rel_{name}"] = (
            result[f"c_k_{name}"] + 0.5
        ) / result[f"c_n_{name}"].clip(lower=1)
        cell_columns.extend(
            [f"c_{name}", f"c_k_{name}", f"c_n_{name}", f"c_rel_{name}"]
        )
    features = censored.FEATURES + new_columns + role_columns + cell_columns
    return result, features, original_keys, missing_new


def test_dev_eval_assembly_feature_and_alignment_contracts():
    dev_base = censored.prepare(pd.read_parquet(ARTIFACT_ROOT / "t5_dev_seq.parquet"))
    dev_base = dev_base[dev_base.fam == "directed_transfer"].reset_index(drop=True)
    eval_base = censored.prepare(
        pd.read_parquet(ARTIFACT_ROOT / "r4/y1_dt_eval_full.parquet")
    )
    dev, dev_features, dev_keys, dev_missing_new = assemble_like_deploy(
        dev_base,
        pd.read_parquet(ARTIFACT_ROOT / "r3/t58_seq_feats.parquet"),
        pd.read_parquet(ARTIFACT_ROOT / "r4/x2_role_dev.parquet"),
    )
    evaluation, eval_features, eval_keys, eval_missing_new = assemble_like_deploy(
        eval_base,
        pd.read_parquet(ARTIFACT_ROOT / "r4/y2_dt_eval_newfeats.parquet"),
        pd.read_parquet(ARTIFACT_ROOT / "r4/x2_role_eval_dt.parquet"),
    )

    assert dev_features == eval_features
    assert len(dev_features) == len(set(dev_features))
    assert set(map(tuple, dev[["slot", "h"]].to_numpy())) == dev_keys
    assert set(map(tuple, evaluation[["slot", "h"]].to_numpy())) == eval_keys
    assert not dev_missing_new.any()
    assert not eval_missing_new.any()
    assert dev[dev_features].notna().all().all()
    assert evaluation[eval_features].notna().all().all()
    assert np.isfinite(dev[dev_features].astype(float).to_numpy()).all()
    assert np.isfinite(evaluation[eval_features].astype(float).to_numpy()).all()
    assert dev.index.equals(pd.RangeIndex(len(dev)))
    assert evaluation.index.equals(pd.RangeIndex(len(evaluation)))

    key_probability = {
        (row.slot, row.h): (index + 1) / (len(evaluation) + 1)
        for index, row in enumerate(evaluation.itertuples(index=False))
    }
    probabilities = np.asarray(
        [key_probability[(row.slot, row.h)] for row in evaluation.itertuples(index=False)]
    )
    marginal = censored.first_k_marginal(evaluation, probabilities)
    attached = evaluation[["slot", "h"]].assign(probability=probabilities, q=marginal)
    for row in attached.itertuples(index=False):
        assert row.probability == key_probability[(row.slot, row.h)]


def test_uncensored_training_rows_are_family_scoped_and_valid():
    dev = censored.prepare(pd.read_parquet(ARTIFACT_ROOT / "t5_dev_seq.parquet"))
    dev = dev[dev.fam == "directed_transfer"].reset_index(drop=True)
    included = censored.uncensored_training_rows(dev)
    assert included.dtype == bool
    assert len(included) == len(dev)
    counts = dev.groupby("slot").ev.sum()
    last_event = dev[dev.ev.astype(bool)].groupby("slot").ts.max()
    for index, row in enumerate(dev.itertuples(index=False)):
        expected = row.ts <= last_event[row.slot] or counts[row.slot] < 5
        assert included[index] == expected
