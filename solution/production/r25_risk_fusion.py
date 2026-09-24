"""Rebuild the historical r25 64+5 model risk fusion from model scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build(
    artifact_root: Path,
    player_local_path: Path,
    slot_pairs_path: Path,
    base_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text())
    old_models = manifest["old_models"]
    new_models = manifest["new_models"]
    new_weight = float(manifest["wnew"])
    local = pd.read_parquet(player_local_path).set_index("player_gi")
    slot_pairs = pd.read_csv(slot_pairs_path)[["slot", "pair_id"]]

    def standardized_score(path: Path, name: str) -> pd.Series:
        table = pd.read_parquet(path)
        table["slot"] = (
            table.pool * 900
            + local.local.loc[table.p_lo].to_numpy() * 30
            + local.local.loc[table.p_hi].to_numpy()
        )
        score = table.set_index("slot").score
        return ((score - score.mean()) / score.std()).rename(name)

    old = pd.concat(
        [
            standardized_score(artifact_root / f"{name}_eval_scores.parquet", name)
            for name in old_models
        ],
        axis=1,
        join="inner",
    )
    new = pd.concat(
        [
            standardized_score(artifact_root / "r4" / f"m15_{name}_eval_scores.parquet", name)
            for name in new_models
        ],
        axis=1,
        join="inner",
    )
    fused = pd.DataFrame(
        {"fused": (1 - new_weight) * old.mean(axis=1) + new_weight * new.mean(axis=1).reindex(old.index)}
    ).dropna().reset_index()
    scores = slot_pairs.merge(fused, on="slot", validate="one_to_one")
    if len(scores) != len(slot_pairs):
        raise ValueError("Model scores do not cover every evaluation pair")

    base = pd.read_csv(base_path, dtype=str, keep_default_na=False)
    members = set(base.loc[base.predicted_behavior.eq("other_coordination"), "pair_id"])
    scores["member"] = scores.pair_id.isin(members)
    nonmembers = scores[~scores.member].sort_values(
        ["fused", "pair_id"], ascending=[False, True]
    )
    member_rows = scores[scores.member].sort_values(
        ["fused", "pair_id"], ascending=[False, True]
    )
    order = pd.concat([nonmembers.iloc[:250], member_rows, nonmembers.iloc[250:]]).pair_id.to_numpy()
    count = len(order)
    risk = pd.Series((count - np.arange(count)) / count, index=order)
    output = base.copy()
    output["risk_score"] = output.pair_id.map(risk).map(lambda value: repr(float(value)))
    if output.risk_score.isna().any():
        raise ValueError("Missing fused risk for a submission pair")
    output.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--player-local", type=Path, required=True)
    parser.add_argument("--slot-pairs", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("r25_fusion_manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.artifact_root, args.player_local, args.slot_pairs, args.base, args.manifest, args.out)
