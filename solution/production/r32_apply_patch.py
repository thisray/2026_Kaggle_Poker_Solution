"""Apply one historical family-routed evidence patch without changing risk or behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EVIDENCE = [f"evidence_hand_{rank}" for rank in range(1, 6)]


def apply(base_path: Path, patch_path: Path, family: str, output_path: Path) -> None:
    base = pd.read_csv(base_path, dtype=str, keep_default_na=False)
    patch = pd.read_csv(patch_path, dtype=str, keep_default_na=False)
    if base.pair_id.duplicated().any() or patch.pair_id.duplicated().any():
        raise ValueError("Duplicate pair IDs")
    if set(patch.pair_id) - set(base.pair_id):
        raise ValueError("Patch contains unknown pair IDs")

    output = base.set_index("pair_id").copy()
    routed = output.index[output.predicted_behavior.eq(family)]
    selected = patch[patch.pair_id.isin(routed)]
    before = output.loc[selected.pair_id, EVIDENCE].to_numpy().copy()
    output.loc[selected.pair_id, EVIDENCE] = selected.set_index("pair_id")[EVIDENCE].to_numpy()
    changed = int((before != output.loc[selected.pair_id, EVIDENCE].to_numpy()).any(axis=1).sum())
    result = output.reset_index()[base.columns]
    if not result[["pair_id", "risk_score", "predicted_behavior"]].equals(
        base[["pair_id", "risk_score", "predicted_behavior"]]
    ):
        raise AssertionError("A family evidence patch changed risk or behavior")
    result.to_csv(output_path, index=False)
    receipt = {
        "base": base_path.name,
        "patch": patch_path.name,
        "family": family,
        "rows": len(patch),
        "applied": len(selected),
        "evidence_rows_changed": changed,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    output_path.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--family", choices=("directed_transfer", "soft_play", "coordinated_isolation", "other_coordination"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    apply(args.base, args.patch, args.family, args.out)
