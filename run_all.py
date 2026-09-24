#!/usr/bin/env python3
"""Run the complete competition solution from the eight official data files."""

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
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=("all", "r10_ci", "r32_r30_dtgb15"),
        default="all",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--resume-after-policy",
        action="store_true",
        help="Reuse complete policy intermediates after an interrupted local run.",
    )
    parser.add_argument(
        "--resume-after-evidence",
        action="store_true",
        help="Reuse complete policy and evidence intermediates after an interrupted local run.",
    )
    args = parser.parse_args()
    started = time.time()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_production_spine(
        data_dir,
        output_dir,
        selected=True,
        threads=args.threads,
        resume_after_policy=args.resume_after_policy,
        resume_after_evidence=args.resume_after_evidence,
    )
    names = [args.variant] if args.variant != "all" else ["r10_ci", "r32_r30_dtgb15"]
    outputs = {}
    for name in names:
        path = output_dir / f"{name}.csv"
        outputs[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "validation": validate_submission(path, data_dir),
        }
    report = {
        "runtime_seconds": time.time() - started,
        "outputs": outputs,
        "selected_assembly": json.loads(
            (output_dir / "selected_assembly_receipt.json").read_text()
        ),
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
