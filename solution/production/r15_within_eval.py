"""Infer the historical family-aware within-pair s2 score in complete pair chunks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import handdesc
import handfeat2
import orient
import withinfeat


FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")


def infer(cache_path: Path, models_dir: Path, output_path: Path, pair_chunk: int, pair_limit: int) -> None:
    work_dir = Path(os.environ["POKER_WORK_DIR"])
    if output_path.exists():
        raise FileExistsError(output_path)
    cache = pd.read_parquet(cache_path, columns=["slot", "h", "s1"])
    cache = cache.sort_values(["slot", "h"], kind="stable").reset_index(drop=True)
    slots = cache.slot.drop_duplicates().to_numpy(np.int64)
    if pair_limit:
        slots = slots[:pair_limit]
        cache = cache[cache.slot.isin(slots)].reset_index(drop=True)
    if pair_chunk < 1 or not len(cache):
        raise ValueError("Use a positive pair chunk and a nonempty hand cache")

    local = pd.read_parquet(work_dir / "player_local_v1.parquet")
    members = np.full((int(local.pool.max()) + 1, 30), -1, np.int64)
    members[local.pool.to_numpy(int), local.local.to_numpy(int)] = local.player_gi.to_numpy(int)
    seated = np.load(work_dir / "np/s_player.npy", mmap_mode="r")
    family_map = pd.read_parquet(work_dir / "m7_family_eval.parquet").set_index("key").family
    models = [
        lgb.Booster(model_file=str(models_dir / f"m25_handfeat2_fam_m19w10_f{fold}.txt"))
        for fold in range(5)
    ]
    all_slots = cache.slot.to_numpy(np.int64)
    writer = None
    try:
        for start in range(0, len(slots), pair_chunk):
            group_slots = slots[start:start + pair_chunk]
            first = int(np.searchsorted(all_slots, group_slots[0], side="left"))
            last = int(np.searchsorted(all_slots, group_slots[-1], side="right"))
            rows = cache.iloc[first:last].copy().reset_index(drop=True)
            pair_slot = rows.slot.to_numpy(np.int64)
            hands = rows.h.to_numpy(np.int64)
            player_a = members[pair_slot // 900, (pair_slot % 900) // 30]
            player_b = members[pair_slot // 900, pair_slot % 30]
            hand_seats = np.asarray(seated[hands])
            seat_a = np.argmax(hand_seats == player_a[:, None], axis=1)
            seat_b = np.argmax(hand_seats == player_b[:, None], axis=1)
            index = np.arange(len(rows))
            if not ((hand_seats[index, seat_a] == player_a) &
                    (hand_seats[index, seat_b] == player_b) &
                    (seat_a != seat_b)).all():
                raise ValueError("A pair member is absent from a shared hand")

            first_stage = rows.s1.to_numpy(float)
            features = pd.concat(
                [handfeat2.features(hands, seat_a, seat_b),
                 handdesc.descriptors(hands, seat_a, seat_b, "dec_probs_v1.npy")],
                axis=1,
            )
            roles = orient.features(pair_slot * 2 + 1, hands, seat_a, seat_b, first_stage)
            zscores = withinfeat.pair_z(
                pd.concat([features, roles], axis=1), pair_slot,
                list(features.columns) + list(roles.columns),
            )
            matrix = pd.concat([features, roles, zscores], axis=1)
            matrix["gen_logit"] = np.log(np.clip(first_stage, 1e-6, 1 - 1e-6) /
                                          (1 - np.clip(first_stage, 1e-6, 1 - 1e-6)))
            matrix["gen_rank_pct"] = rows.groupby("slot").s1.rank(pct=True).to_numpy()
            keys = np.minimum(player_a, player_b) * 12000 + np.maximum(player_a, player_b)
            families = family_map.reindex(keys).to_numpy()
            if pd.isna(families).any():
                raise ValueError("Missing family prediction for an evaluation pair")
            for family_index, family in enumerate(FAMILIES):
                matrix[f"fam_{family_index}"] = (families == family).astype(np.float32)
            prediction = np.mean(
                [model.predict(matrix[model.feature_name()], num_threads=4) for model in models],
                axis=0,
            ).astype(np.float32)
            rows["s2"] = prediction
            table = pa.Table.from_pandas(rows[["slot", "h", "s1", "s2"]], preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            print(f"within-pair s2: {min(start + pair_chunk, len(slots))}/{len(slots)} pairs", flush=True)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-chunk", type=int, default=1000)
    parser.add_argument("--pair-limit", type=int, default=0)
    args = parser.parse_args()
    infer(args.cache, args.models_dir, args.out, args.pair_chunk, args.pair_limit)
