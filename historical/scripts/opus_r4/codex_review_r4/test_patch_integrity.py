from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path(
    "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
)
EVIDENCE = [f"evidence_hand_{index}" for index in range(1, 6)]


def candidate_sets(path):
    frame = pd.read_parquet(path)
    assert not frame.duplicated(["pair_id", "hand_id"]).any()
    assert frame.groupby("pair_id").size().eq(20).all()
    return frame.groupby("pair_id").hand_id.apply(set).to_dict()


def test_r17_patch_integrity_against_r13_base():
    result_path = ARTIFACT_ROOT / "r4/cand/r17_r13_dt_cistack.csv"
    base_path = ARTIFACT_ROOT / "r2_candidates/r13_ndwrank_cinew_f4.csv"
    result = pd.read_csv(result_path, dtype=str, keep_default_na=False)
    base = pd.read_csv(base_path, dtype=str, keep_default_na=False)

    assert result.columns.tolist() == base.columns.tolist()
    assert result.pair_id.tolist() == base.pair_id.tolist()
    assert result.risk_score.tolist() == base.risk_score.tolist()
    assert result.predicted_behavior.tolist() == base.predicted_behavior.tolist()

    changed = (result[EVIDENCE].to_numpy() != base[EVIDENCE].to_numpy()).any(axis=1)
    patched = result.loc[changed]
    allowed_families = {"directed_transfer", "coordinated_isolation"}
    assert set(patched.predicted_behavior).issubset(allowed_families)
    assert patched[EVIDENCE].apply(lambda row: row.nunique() == 5, axis=1).all()

    dt_sets = candidate_sets(ARTIFACT_ROOT / "r4/y1_dt_eval_candidates.parquet")
    ci_sets = candidate_sets(ARTIFACT_ROOT / "r18/main/ci_eval_candidates.parquet")
    for row in patched.itertuples(index=False):
        evidence = {getattr(row, column) for column in EVIDENCE}
        candidates = (
            dt_sets[row.pair_id]
            if row.predicted_behavior == "directed_transfer"
            else ci_sets[row.pair_id]
        )
        assert evidence.issubset(candidates)

    assert not np.any(result[EVIDENCE].eq("").to_numpy())


def test_candidate_frames_cover_all_routed_base_pairs():
    base = pd.read_csv(
        ARTIFACT_ROOT / "r2_candidates/r13_ndwrank_cinew_f4.csv",
        dtype=str,
        keep_default_na=False,
    )
    expected = {
        "directed_transfer": set(
            base.loc[base.predicted_behavior == "directed_transfer", "pair_id"]
        ),
        "coordinated_isolation": set(
            base.loc[base.predicted_behavior == "coordinated_isolation", "pair_id"]
        ),
    }
    actual = {
        "directed_transfer": set(
            pd.read_parquet(
                ARTIFACT_ROOT / "r4/y1_dt_eval_candidates.parquet",
                columns=["pair_id"],
            ).pair_id
        ),
        "coordinated_isolation": set(
            pd.read_parquet(
                ARTIFACT_ROOT / "r18/main/ci_eval_candidates.parquet",
                columns=["pair_id"],
            ).pair_id
        ),
    }
    for family in expected:
        missing = expected[family] - actual[family]
        extra = actual[family] - expected[family]
        assert not missing and not extra, (
            f"{family} candidate coverage differs from routed base: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
