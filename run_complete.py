#!/usr/bin/env python3
"""Run the fully connected solution-method pipeline from the eight official files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from solution.production_runner import run_production_spine
from solution.validate import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tabicl-checkpoint", type=Path)
    parser.add_argument("--download-public-checkpoint", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume-after-policy", action="store_true")
    parser.add_argument("--resume-after-evidence", action="store_true")
    args = parser.parse_args()
    start = time.time()
    output_dir = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    run_production_spine(
        data_dir, output_dir, selected=True, threads=args.threads,
        resume_after_policy=args.resume_after_policy,
        resume_after_evidence=args.resume_after_evidence,
        complete=True, tabicl_checkpoint=args.tabicl_checkpoint,
        download_public_checkpoint=args.download_public_checkpoint,
    )
    results = {}
    for name in ("r10_ci", "r32_r30_dtgb15"):
        output = output_dir / f"{name}.csv"
        results[name] = {
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "validation": validate_submission(output, data_dir),
        }
    report = {
        "method_path": "fully_connected_non_identical_reproduction",
        "runtime_seconds": time.time() - start,
        "outputs": results,
    }
    (output_dir / "run_complete_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
