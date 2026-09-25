"""Build the historical typed-zoo evaluation frames for one routed family."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from late_stage import _extract_features, _shared_hands, _slot_table


TAGS = {"directed_transfer": "dt", "soft_play": "sp"}


def build(work_dir: Path, data_dir: Path, base_path: Path,
          scored_path: Path, family: str) -> None:
    base = pd.read_csv(base_path, dtype={"pair_id": str})
    scored = pd.read_csv(scored_path, dtype={"pair_id": str, "hand_id": str})
    routed = set(base.loc[base.predicted_behavior.eq(family), "pair_id"])
    scored = scored[scored.pair_id.isin(routed)].copy()
    if scored.empty or scored.duplicated(["slot", "hand_id"]).any():
        raise ValueError(f"Missing or duplicate {family} scored candidates")
    hand_index = pd.read_parquet(work_dir / "np/hand_index.parquet").set_index("hand_id")
    scored["h"] = hand_index.hi.reindex(scored.hand_id).to_numpy()
    if scored.h.isna().any():
        raise ValueError("Unknown scored candidate hand")
    scored["h"] = scored.h.astype(int)
    scored = scored.sort_values(
        ["slot", "score", "hand_id"], ascending=[True, False, True], kind="stable"
    ).groupby("slot", sort=False).head(20).copy()
    scored["r"] = scored.groupby("slot", sort=False).cumcount() + 1
    slot_pairs = _slot_table(work_dir, data_dir, "evaluation")
    slot_pairs = slot_pairs[slot_pairs.pair_id.isin(set(scored.pair_id))]
    if slot_pairs.empty or slot_pairs.slot.nunique() != scored.slot.nunique():
        raise ValueError("Scored family pairs are absent from evaluation pairs")
    player_index = pd.read_parquet(work_dir / "np/player_index.parquet")
    player_map = dict(zip(player_index.player_id, player_index.pi))
    first = slot_pairs.player_1.map(player_map).to_numpy(int)
    second = slot_pairs.player_2.map(player_map).to_numpy(int)
    slot_pairs["pa"] = np.minimum(first, second)
    slot_pairs["pb"] = np.maximum(first, second)
    keyed = _shared_hands(work_dir, slot_pairs, phase=1)
    keyed = keyed.merge(slot_pairs[["slot", "pa", "pb"]], on="slot", validate="many_to_one")
    keyed["fam"] = family
    full = _extract_features(work_dir, keyed)
    tag = TAGS[family]
    target = work_dir / "r4"
    target.mkdir(parents=True, exist_ok=True)
    full.to_parquet(target / f"y1_{tag}_eval_full.parquet", index=False)
    scored[["pair_id", "slot", "h", "hand_id", "r", "score"]].to_parquet(
        target / f"y1_{tag}_eval_candidates.parquet", index=False
    )
    print({"family": family, "pairs": scored.slot.nunique(), "shared_hands": len(full)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(TAGS), required=True)
    args = parser.parse_args()
    build(args.work_dir, args.data_dir, args.base, args.scored, args.family)
