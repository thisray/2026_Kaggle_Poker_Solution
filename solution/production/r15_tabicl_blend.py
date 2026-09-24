"""Blend historical TabICLv2 and Round11 hand ranks within each pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


TABICL_WEIGHT = 0.6


def blend(ranker_path: Path, tabicl_path: Path, output_path: Path) -> None:
    tabicl = pd.read_csv(tabicl_path)[["slot", "hand_id", "score"]].rename(
        columns={"score": "tab"}
    )
    ranker = pd.read_csv(ranker_path)[["slot", "pair_id", "hand_id", "score"]].rename(
        columns={"score": "rs"}
    )
    frame = ranker.merge(tabicl, on=["slot", "hand_id"], how="left", validate="one_to_one")
    if frame.tab.isna().any():
        raise ValueError(f"Missing TabICL scores for {int(frame.tab.isna().sum())} rows")

    def within_pair_rank(values: np.ndarray) -> np.ndarray:
        output = np.empty(len(values))
        for indices in frame.groupby("slot", sort=False).indices.values():
            output[indices] = rankdata(-values[indices], method="average")
        return output

    tab_rank = within_pair_rank(frame.tab.to_numpy())
    ranker_rank = within_pair_rank(frame.rs.to_numpy())
    frame["score"] = -(TABICL_WEIGHT * tab_rank + (1 - TABICL_WEIGHT) * ranker_rank)
    frame["ranker"] = -ranker_rank
    frame["cat"] = np.nan
    frame[["slot", "pair_id", "hand_id", "score", "ranker", "cat"]].to_csv(
        output_path, index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker-scores", type=Path, required=True)
    parser.add_argument("--tabicl-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    blend(args.ranker_scores, args.tabicl_scores, args.out)
