"""Rebuild the historical typed-zoo development frame from official labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from late_stage import _extract_features, _shared_hands, _slot_table


def build(work_dir: Path, data_dir: Path) -> None:
    positive = _slot_table(work_dir, data_dir, "development")
    positive = positive[positive.label.eq(1)].copy()
    if positive.slot.duplicated().any() or positive.pair_id.duplicated().any():
        raise ValueError("Duplicate labelled positive pair")
    player_index = pd.read_parquet(work_dir / "np/player_index.parquet")
    player_map = dict(zip(player_index.player_id, player_index.pi))
    first = positive.player_1.map(player_map).to_numpy(int)
    second = positive.player_2.map(player_map).to_numpy(int)
    positive["pa"] = np.minimum(first, second)
    positive["pb"] = np.maximum(first, second)
    role_pairs = work_dir / "r4/dev_pairs.parquet"
    role_pairs.parent.mkdir(parents=True, exist_ok=True)
    positive[["slot", "pa", "pb"]].to_parquet(role_pairs, index=False)
    keyed = _shared_hands(work_dir, positive, phase=0)
    keyed = keyed.merge(
        positive[["slot", "behavior_family", "pa", "pb"]],
        on="slot", validate="many_to_one"
    ).rename(columns={"behavior_family": "fam"})
    evidence = pd.read_csv(
        data_dir / "development_evidence.csv",
        dtype={"pair_id": str, "hand_id": str},
    )
    if evidence.duplicated(["pair_id", "hand_id"]).any():
        raise ValueError("Duplicate development evidence")
    membership = pd.MultiIndex.from_frame(evidence[["pair_id", "hand_id"]])
    keyed["ev"] = pd.MultiIndex.from_frame(
        keyed[["pair_id", "hand_id"]]
    ).isin(membership)
    frame = _extract_features(work_dir, keyed)
    output = work_dir / "t5_dev_seq.parquet"
    frame.to_parquet(output, index=False)

    hand_index = pd.read_parquet(work_dir / "np/hand_index.parquet").set_index("hand_id")
    evidence["h"] = hand_index.hi.reindex(evidence.hand_id).to_numpy()
    if evidence.h.isna().any():
        raise ValueError("Development evidence has unknown hand IDs")
    timestamps = np.load(work_dir / "np/h_ts.npy", mmap_mode="r")
    evidence["ts"] = timestamps[evidence.h.to_numpy(int)]
    evidence["chron"] = evidence.groupby("pair_id").ts.rank()
    rank_path = work_dir / "r4/g1_evidence_rank.parquet"
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_parquet(rank_path, index=False)
    print({"dev_rows": len(frame), "evidence_rows": len(evidence)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.work_dir, args.data_dir)
