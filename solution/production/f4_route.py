"""Restore the historical fourth-family routing and monotone rank insertion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main(baseline_path: Path, work_dir: Path) -> None:
    baseline = pd.read_csv(baseline_path, dtype={"pair_id": str})
    signal = pd.read_parquet(work_dir / "s38_combined_eval.parquet")[["pair_id", "p2", "zc"]]
    frame = baseline.merge(signal, on="pair_id", how="left", validate="one_to_one")
    frame["p2"] = frame.p2.fillna(0.0)
    frame["zc"] = frame.zc.fillna(-9.0)
    frame = frame.sort_values(["risk_score", "pair_id"], ascending=[False, True], kind="stable")
    frame = frame.reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    selected = ((frame["rank"] <= 600) & (frame.p2 > 2.5)) | (
        (frame["rank"] <= 1000) & (frame.zc > 3.02)
    )
    frame.loc[selected, "predicted_behavior"] = "other_coordination"

    # Historical c7 rule: keep members already above the 250th nonmember;
    # insert the remaining members as a block immediately after it.
    base_order = frame.sort_values(["risk_score", "pair_id"], ascending=[False, True])
    base_order = base_order.reset_index(drop=True)
    base_order["base_rank"] = np.arange(len(base_order))
    members = base_order[base_order.predicted_behavior.eq("other_coordination")].copy()
    nonmembers = base_order[~base_order.predicted_behavior.eq("other_coordination")].copy()
    threshold = int(nonmembers.base_rank.iloc[249])
    members["order_key"] = np.where(
        members.base_rank < threshold,
        members.base_rank.astype(float),
        threshold + 0.5 + 1e-4 * np.arange(len(members)),
    )
    nonmembers["order_key"] = nonmembers.base_rank.astype(float)
    order = pd.concat([nonmembers, members]).sort_values("order_key").pair_id.to_numpy()
    count = len(order)
    risk = pd.Series((count - np.arange(count)) / count, index=order)
    output = frame.set_index("pair_id").loc[baseline.pair_id].reset_index()
    output["risk_score"] = output.pair_id.map(risk)
    output = output[baseline.columns]
    target = work_dir / "f4_routed_baseline.csv"
    output.to_csv(target, index=False)
    (work_dir / "f4_pair_ids.txt").write_text("\n".join(members.pair_id) + "\n")
    receipt = {
        "routed_pairs": int(selected.sum()),
        "kept_above_insertion_point": int((members.base_rank < threshold).sum()),
        "risk_changed_rows": int((output.risk_score.to_numpy() != baseline.risk_score.to_numpy()).sum()),
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (work_dir / "f4_route.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(os.environ["POKER_WORK_DIR"]))
