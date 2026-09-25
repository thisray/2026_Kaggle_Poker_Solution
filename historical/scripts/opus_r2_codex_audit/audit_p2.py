"""Independent reproduction of the partner-card p2 statistic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (  # noqa: E402
    AUDIT_ART,
    OUT,
    compute_first_preflop_scores,
    ensure_audit_artifact_dir,
    load_core_arrays,
    pair_slot_from_players,
    player_maps,
    read_evaluation_pairs,
    rank_frame,
)


def reference_agreement(ours: pd.DataFrame, phase_name: str):
    reference = pd.read_parquet(OUT / f"s23_infoshare_{phase_name}.parquet")
    merged = reference.merge(ours, on="slot", suffixes=("_ref", "_ours"), how="outer", indicator=True)
    metrics = {"reference_rows": int(len(reference)), "ours_rows": int(len(ours))}
    metrics["slot_set_equal"] = bool((merged._merge == "both").all())
    columns = ["n_is", "zf0", "zf1", "za0", "za1", "chi", "zmax"]
    max_abs = {}
    exact = {}
    for column in columns:
        left = merged[f"{column}_ref"].to_numpy(dtype=float)
        right = merged[f"{column}_ours"].to_numpy(dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        max_abs[column] = float(np.max(np.abs(left[valid] - right[valid]))) if valid.any() else None
        exact[column] = bool(np.array_equal(left[valid], right[valid])) if valid.any() else False
    metrics["exact_equal"] = exact
    metrics["max_abs_difference"] = max_abs
    metrics["allclose_1e-10"] = bool(
        all(value is not None and value <= 1e-10 for value in max_abs.values())
    )
    return metrics


def add_dev_slots(dev: pd.DataFrame, data: dict) -> pd.DataFrame:
    _, _, pool_by_pi, local_by_pi = player_maps(data)
    values_are_gi = np.issubdtype(dev.p_lo.dtype, np.integer)
    slots = [
        pair_slot_from_players(
            pool_by_pi,
            local_by_pi,
            int(a) if values_are_gi else int(a),
            int(b) if values_are_gi else int(b),
        )
        for a, b in zip(dev.p_lo, dev.p_hi)
    ]
    result = dev.copy()
    result["slot"] = np.asarray(slots, dtype=np.int64)
    return result


def main():
    ensure_audit_artifact_dir()
    data = load_core_arrays()
    mu = float(data["pfeq"].mean())
    all_scores = {}
    agreements = {}
    for phase, name in [(0, "dev"), (1, "eval")]:
        scores, first, pair_arrays, _ = compute_first_preflop_scores(data, phase)
        scores.to_parquet(AUDIT_ART / f"independent_p2_{name}.parquet", index=False)
        all_scores[name] = scores
        agreements[name] = reference_agreement(scores, name)

    dev = add_dev_slots(pd.read_parquet(OUT / "m1_dev_oof.parquet"), data)
    dev_scores = all_scores["dev"].merge(dev, on="slot", how="inner")
    eval_risk = pd.read_csv(
        "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped/eval_risk_with_slot.csv"
    )
    eval_scores = all_scores["eval"].merge(eval_risk, on="slot", how="inner")
    eval_scores = rank_frame(eval_scores)
    eval_scores.to_parquet(AUDIT_ART / "independent_p2_eval_with_risk.parquet", index=False)

    null = dev_scores[(dev_scores.label == -1) & (dev_scores.oof < 0.02) & (dev_scores.n >= 38)]
    labelled_neg = dev_scores[dev_scores.label == 0]
    positive_counts = {}
    for family in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        group = dev_scores[(dev_scores.label == 1) & (dev_scores.fam == family)]
        positive_counts[family] = {
            "n": int(len(group)),
            "p2_gt_4": int((group.p2 > 4.0).sum()),
            "p2_gt_3": int((group.p2 > 3.0).sum()),
        }
    top450 = eval_scores[eval_scores["rank"] <= 450]
    summary = {
        "mu_pf_eq_mean": mu,
        "dev_null_low_oof": {
            "n": int(len(null)),
            "mean_chi": float(null.chi.mean()),
            "mean_p2": float(null.p2.mean()),
            "p95_p2": float(null.p2.quantile(0.95)),
            "p999_p2": float(null.p2.quantile(0.999)),
            "mean_n_is": float(null.n_is.mean()),
            "sd_per_direction_z": float(
                pd.concat([null.zf0, null.zf1, null.za0, null.za1], ignore_index=True).std()
            ),
        },
        "dev_labelled_negative": {
            "n": int(len(labelled_neg)),
            "mean_chi": float(labelled_neg.chi.mean()),
            "p2_gt_4": int((labelled_neg.p2 > 4.0).sum()),
        },
        "dev_positive_p2_counts": positive_counts,
        "eval_all_slots": {
            "n": int(len(all_scores["eval"])),
            "p2_gt_4": int((all_scores["eval"].p2 > 4.0).sum()),
        },
        "eval_risk_join": {
            "n": int(len(eval_scores)),
            "top450_n": int(len(top450)),
            "top450_p2_gt_4": int((top450.p2 > 4.0).sum()),
            "top450_p2_gt_3": int((top450.p2 > 3.0).sum()),
            "ranks_451_2000_p2_gt_4": int(
                ((eval_scores["rank"] > 450) & (eval_scores["rank"] <= 2000) & (eval_scores.p2 > 4.0)).sum()
            ),
            "deep_p2_gt_4": int(((eval_scores["rank"] > 2000) & (eval_scores.p2 > 4.0)).sum()),
        },
        "reference_agreement": agreements,
    }
    (AUDIT_ART / "independent_p2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
