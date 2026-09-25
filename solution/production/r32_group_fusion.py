"""Build a reproducible grouped pair-risk fusion from trained model scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OLD = ("v6ens_base", "v6ens_cat", "v6ens_cat11")
NEW = ("r32_new_lgb_s3", "r32_new_lgb_s5", "r32_new_lgb_s11",
       "r32_new_cat_s17", "r32_new_cat_s23")
NEW_WEIGHT = 0.7


def build(work_dir: Path, data_dir: Path, base_path: Path, output_path: Path) -> None:
    player_index = pd.read_parquet(work_dir / "np/player_index.parquet")
    player_map = dict(zip(player_index.player_id, player_index.pi))
    pairs = pd.read_csv(data_dir / "evaluation_pairs.csv", dtype={"pair_id": str})
    low = np.minimum(pairs.player_1.map(player_map), pairs.player_2.map(player_map))
    high = np.maximum(pairs.player_1.map(player_map), pairs.player_2.map(player_map))
    pairs["key"] = low * 12000 + high
    scores = pairs[["pair_id", "key"]].copy()
    for tag in (*OLD, *NEW):
        source = pd.read_parquet(work_dir / f"m15_{tag}_eval_scores.parquet")[["key", "score"]]
        scores = scores.merge(source.rename(columns={"score": tag}), on="key", validate="one_to_one")
        standard = scores[tag].std()
        if not np.isfinite(standard) or standard <= 0:
            raise ValueError(f"Invalid score distribution for {tag}")
        scores[tag] = (scores[tag] - scores[tag].mean()) / standard
    scores["fused"] = (1 - NEW_WEIGHT) * scores[list(OLD)].mean(axis=1) + NEW_WEIGHT * scores[list(NEW)].mean(axis=1)
    base = pd.read_csv(base_path, dtype={"pair_id": str})
    if set(scores.pair_id) != set(base.pair_id):
        raise ValueError("Risk scores do not cover the submission pairs")
    members = set(base.loc[base.predicted_behavior.eq("other_coordination"), "pair_id"])
    scores["member"] = scores.pair_id.isin(members)
    nonmembers = scores.loc[~scores.member].sort_values(["fused", "pair_id"], ascending=[False, True])
    member_rows = scores.loc[scores.member].sort_values(["fused", "pair_id"], ascending=[False, True])
    ordered = pd.concat([nonmembers.iloc[:250], member_rows, nonmembers.iloc[250:]]).pair_id.to_numpy()
    count = len(ordered)
    ranks = pd.Series((count - np.arange(count)) / count, index=ordered)
    base["risk_score"] = base.pair_id.map(ranks)
    if base.risk_score.isna().any():
        raise ValueError("Missing fused risk")
    base.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work-dir", "data-dir", "base", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.work_dir, args.data_dir, args.base, args.out)
