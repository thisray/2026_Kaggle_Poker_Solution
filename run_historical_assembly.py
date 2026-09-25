#!/usr/bin/env python3
"""Assemble the two selected submissions from their competition-time intermediates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EVIDENCE = [f"evidence_hand_{rank}" for rank in range(1, 6)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(base_path: Path, patch_path: Path, family: str, output_path: Path) -> dict[str, object]:
    base = pd.read_csv(base_path, dtype=str, keep_default_na=False)
    patch = pd.read_csv(patch_path, dtype=str, keep_default_na=False)
    if base.pair_id.duplicated().any() or patch.pair_id.duplicated().any():
        raise ValueError("Duplicate pair IDs")
    if set(patch.pair_id) - set(base.pair_id):
        raise ValueError("Patch has pair IDs outside the base submission")

    output = base.set_index("pair_id").copy()
    eligible = output.index[output.predicted_behavior.eq(family)]
    applied = patch[patch.pair_id.isin(eligible)]
    before = output.loc[applied.pair_id, EVIDENCE].to_numpy().copy()
    output.loc[applied.pair_id, EVIDENCE] = applied.set_index("pair_id")[EVIDENCE].to_numpy()
    result = output.reset_index()[base.columns]
    if not result[["pair_id", "risk_score", "predicted_behavior"]].equals(
        base[["pair_id", "risk_score", "predicted_behavior"]]
    ):
        raise AssertionError("An evidence patch changed pair risk or behavior")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return {
        "base": str(base_path),
        "base_sha256": sha256(base_path),
        "patch": str(patch_path),
        "patch_sha256": sha256(patch_path),
        "family": family,
        "rows": len(result),
        "patch_rows": len(patch),
        "applied_pairs": len(applied),
        "changed_evidence_rows": int((before != output.loc[applied.pair_id, EVIDENCE].to_numpy()).any(axis=1).sum()),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == root or root in output_dir.parents:
        raise ValueError("Use a separate output directory; do not overwrite historical artifacts")

    specs = (
        ("r10_ci", "r2_candidates/r2n_NDw_on_r2j2m.csv", "r18/main/patch_r18hard.csv", "coordinated_isolation"),
        ("r32_r30_dtgb15", "r5/cand/r30_r29_spzoo.csv", "r5/patch_r5_dt_zoo_gb15.csv", "directed_transfer"),
    )
    report = {}
    for name, base_name, patch_name, family in specs:
        base_path = root / base_name
        patch_path = root / patch_name
        for path in (base_path, patch_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        report[name] = assemble(base_path, patch_path, family, output_dir / f"{name}.csv")
    (output_dir / "historical_assembly_receipt.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
