"""Read-only candidate lineage and rank/behavior interaction audit; run on GB10."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    names = {"ndw": "r2n_NDw_on_r2j2m.csv", "r9": "r9_subh.csv",
             "r10": "r10_ci.csv", "r11": "r11_ci_rank.csv",
             "r12": "r12_ndwrank_ci_f4.csv"}
    frames, lineage = {}, {}
    for key, name in names.items():
        path = args.root / name
        frame = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("pair_id").sort_index()
        assert frame.index.is_unique
        frames[key] = frame
        lineage[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(frame)}
    ref = frames["ndw"]
    for frame in frames.values():
        assert frame.index.equals(ref.index)
    ev = [f"evidence_hand_{i}" for i in range(1, 6)]
    families = ["directed_transfer", "soft_play", "coordinated_isolation"]

    def order(frame, family=None):
        risk = frame.risk_score.astype(float).to_numpy()
        if family is not None:
            risk = risk * frame.predicted_behavior.eq(family).to_numpy()
        return np.argsort(-risk, kind="stable")

    def ranks(frame):
        result = np.empty(len(frame), dtype=int)
        result[order(frame)] = np.arange(1, len(frame) + 1)
        return result

    diffs = {}
    for left, right in [("ndw", "r10"), ("r10", "r11"), ("r11", "r9"), ("r10", "r12"), ("r9", "r12")]:
        a, b = frames[left], frames[right]
        diffs[f"{left}->{right}"] = {
            "risk_strings_changed": int(a.risk_score.ne(b.risk_score).sum()),
            "global_rank_changed": int((ranks(a) != ranks(b)).sum()),
            "behavior_changed": int(a.predicted_behavior.ne(b.predicted_behavior).sum()),
            "evidence_rows_changed": int((a[ev] != b[ev]).any(axis=1).sum()),
            "class_order_positions_changed": {f: int((order(a, f) != order(b, f)).sum()) for f in families},
        }
    a, b = frames["r10"], frames["r11"]
    ra, rb = ranks(a), ranks(b)
    moved = rb < ra - 20
    details = []
    for idx in np.flatnonzero(moved):
        old_family = a.predicted_behavior.iloc[idx]
        new_family = frames["r12"].predicted_behavior.iloc[idx]
        crossed = (ra < ra[idx]) & (rb > rb[idx])
        details.append({"rank_before": int(ra[idx]), "rank_after": int(rb[idx]),
                        "old_family": old_family, "new_family": new_family,
                        "crossed_pairs": int(crossed.sum()),
                        "crossed_same_old_family": int((crossed & a.predicted_behavior.eq(old_family).to_numpy()).sum())})
    structural = {
        "r12_risk_equals_ndw": bool(frames["r12"].risk_score.equals(ref.risk_score)),
        "r12_labels_and_evidence_equal_r9": bool(frames["r12"][["predicted_behavior", *ev]].equals(frames["r9"][["predicted_behavior", *ev]])),
        "r11_risk_equals_r9": bool(frames["r11"].risk_score.equals(frames["r9"].risk_score)),
        "r11_labels_and_evidence_equal_r10": bool(frames["r11"][["predicted_behavior", *ev]].equals(frames["r10"][["predicted_behavior", *ev]])),
    }
    print(json.dumps({"observed_at": datetime.now(timezone.utc).isoformat(), "source_base_sha": args.source_sha,
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if __file__ != "<stdin>" else "stdin; hash recorded by caller",
                      "host": platform.node(), "python": platform.python_version(), "pandas": pd.__version__,
                      "numpy": np.__version__, "inputs": lineage, "diffs": diffs,
                      "promoted_rank_summary": details, "structural_checks": structural,
                      "hidden_labels_used": False, "candidate_writes": False}, indent=2))


if __name__ == "__main__":
    main()
