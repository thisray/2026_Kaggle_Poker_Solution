"""Export fold-safe development candidate scores and gameplay extras."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from r15_tabicl_input import EXTRAS, SCORES


def build(candidates_path: Path, neural_path: Path, scores_path: Path, extras_path: Path) -> None:
    candidates = pd.read_parquet(candidates_path)
    neural = pd.read_parquet(neural_path)[["sl", "h", "nn_cal"]]
    if candidates.duplicated(["slot", "h"]).any() or neural.duplicated(["sl", "h"]).any():
        raise ValueError("Duplicate development candidate or neural keys")
    frame = candidates.merge(
        neural.rename(columns={"sl": "slot"}), on=["slot", "h"],
        how="left", validate="one_to_one",
    )
    if frame.nn_cal.isna().any():
        raise ValueError("Neural OOF scores do not cover development candidates")
    clipped = np.clip(frame.nn_cal.to_numpy(float), 1e-6, 1 - 1e-6)
    neural_logit = np.log(clipped / (1 - clipped))
    frame["sc_r5"] = frame.sc
    frame["u_rr"] = frame.u0 + 0.5 * frame.lin
    frame["lin_contrib"] = 0.5 * frame.lin
    frame["nn_contrib"] = 0.1 * neural_logit
    frame["u_r5b"] = frame.u_rr + frame.nn_contrib
    frame["t1_score"] = frame.t1
    frame["s1_stage1"] = frame.s1
    clipped_stage1 = np.clip(frame.s1.to_numpy(float), 1e-6, 1 - 1e-6)
    frame["gen_logit"] = np.log(clipped_stage1 / (1 - clipped_stage1))
    frame["gen_rank_pct"] = frame.gen_rank_all
    frame["rank_u_r5b"] = frame.groupby("slot").u_r5b.rank(
        ascending=False, method="first"
    )
    if not set(SCORES + EXTRAS).issubset(frame):
        raise ValueError("Development score or gameplay feature is missing")
    meta = [
        "slot", "pair_id", "pool", "fold", "pair_player_lo", "pair_player_hi",
        "hand_id", "ev", "m_p",
    ]
    frame[meta + SCORES].to_csv(scores_path, index=False)
    frame[["slot", "hand_id", *EXTRAS]].to_csv(extras_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--extras", type=Path, required=True)
    args = parser.parse_args()
    build(args.candidates, args.neural, args.scores, args.extras)
