"""Audit scheduled and best candidate submissions without contacting Kaggle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import AUDIT_ART, DATA, NP, OUT, ensure_audit_artifact_dir  # noqa: E402


EV_COLS = [f"evidence_hand_{i}" for i in range(1, 6)]
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior"] + EV_COLS
EXPECTED_SHA = {
    "r2d_p2comb_other_ev_on_r15.csv": "0d62b027f30696f6604fdebd8b082bc0164bbd7b024ec9bc246f3e0e9270ff22",
    "r2e_p2comb_other_evall_on_r15.csv": "49553c883ff88b8bdbaa8c4af35384b3467af71a8783ac4cdce60c750fc91314",
    "r2c_p2top600_other_ev.csv": "1f911504ff0367e6d663db06c448f14032e0c257cde397fd18df08124966fb72",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["risk_num"] = pd.to_numeric(result.risk_score, errors="coerce")
    result = result.sort_values(["risk_num", "pair_id"], ascending=[False, True]).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def row_diff(base: pd.DataFrame, candidate: pd.DataFrame):
    left = base.set_index("pair_id").loc[candidate.pair_id]
    right = candidate.set_index("pair_id").loc[candidate.pair_id]
    semantic = pd.DataFrame(index=candidate.pair_id)
    semantic["risk_score"] = ~np.isclose(
        pd.to_numeric(left.risk_score, errors="coerce"),
        pd.to_numeric(right.risk_score, errors="coerce"),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    for col in ["predicted_behavior"] + EV_COLS:
        semantic[col] = left[col].to_numpy(dtype=object) != right[col].to_numpy(dtype=object)
    changed = semantic.any(axis=1)
    exact_text = (left[SUB_COLS[1:]].to_numpy(dtype=object) != right[SUB_COLS[1:]].to_numpy(dtype=object)).any(axis=1)
    return {
        "semantic_changed_rows": int(changed.sum()),
        "exact_text_changed_rows": int(exact_text.sum()),
        "changed_columns": {col: int(semantic[col].sum()) for col in semantic.columns},
        "changed_pair_ids": candidate.pair_id[changed.to_numpy()].tolist(),
        "left": left,
        "right": right,
        "semantic": semantic,
    }


def evidence_legality(candidate: pd.DataFrame, eval_pairs: pd.DataFrame, data):
    hand_index = pd.read_parquet(NP / "hand_index.parquet")
    hand_to_hi = dict(zip(hand_index.hand_id, hand_index.hi.astype(int)))
    player_index = pd.read_parquet(NP / "player_index.parquet")
    pi_by_id = dict(zip(player_index.player_id, player_index.pi.astype(int)))
    s_player = np.load(NP / "s_player.npy", mmap_mode="r")
    phase = np.load(NP / "h_phase.npy", mmap_mode="r")
    pair_lookup = eval_pairs.set_index("pair_id")
    nan_cells = 0
    blank_cells = 0
    invalid_hand_id = 0
    wrong_phase = 0
    missing_player = 0
    duplicate_rows = 0
    bad_rows = []
    for row in candidate.itertuples(index=False):
        cells = [getattr(row, col) for col in EV_COLS]
        if any(pd.isna(value) for value in cells):
            nan_cells += 1
        if any(value == "" for value in cells):
            blank_cells += 1
        actual = [value for value in cells if value != "NO_EVIDENCE" and not pd.isna(value)]
        bad = False
        if len(actual) != len(set(actual)):
            duplicate_rows += 1
            bad = True
        try:
            pair = pair_lookup.loc[row.pair_id]
            p1 = int(pi_by_id[pair.player_1])
            p2 = int(pi_by_id[pair.player_2])
        except (KeyError, TypeError):
            missing_player += 1
            bad = True
            p1 = p2 = -1
        for hand_id in actual:
            if hand_id not in hand_to_hi:
                invalid_hand_id += 1
                bad = True
                continue
            hi = hand_to_hi[hand_id]
            if phase[hi] != 1:
                wrong_phase += 1
                bad = True
            seated = np.asarray(s_player[hi])
            if p1 not in seated or p2 not in seated:
                missing_player += 1
                bad = True
        if bad:
            bad_rows.append(row.pair_id)
    return {
        "nan_cells_or_rows": int(nan_cells),
        "blank_cells_or_rows": int(blank_cells),
        "invalid_hand_id_cells": int(invalid_hand_id),
        "wrong_phase_cells": int(wrong_phase),
        "missing_player_cells_or_pairs": int(missing_player),
        "duplicate_evidence_rows": int(duplicate_rows),
        "bad_pair_ids": bad_rows,
        "legality_pass": not bad_rows and nan_cells == 0 and blank_cells == 0,
    }


def candidate_report(name: str, base_path: Path, eval_pairs: pd.DataFrame, data):
    path = OUT / "r2_candidates" / name
    base = pd.read_csv(base_path, dtype=str)
    candidate = pd.read_csv(path, dtype=str)
    diff = row_diff(base, candidate)
    old_rank = rank_frame(base[["pair_id", "risk_score"]].assign(predicted_behavior=base.predicted_behavior))
    new_rank = rank_frame(candidate[["pair_id", "risk_score"]].assign(predicted_behavior=candidate.predicted_behavior))
    old_rank = old_rank[["pair_id", "rank", "risk_num"]].rename(columns={"rank": "old_rank", "risk_num": "old_risk"})
    new_rank = new_rank[["pair_id", "rank", "risk_num"]].rename(columns={"rank": "new_rank", "risk_num": "new_risk"})
    rank_change = old_rank.merge(new_rank, on="pair_id")
    target = candidate[candidate.predicted_behavior == "other_coordination"].merge(
        rank_change, on="pair_id", how="left"
    )
    sample_ids = set(pd.read_csv(DATA / "sample_submission.csv", usecols=["pair_id"]).pair_id)
    candidate_ids = set(candidate.pair_id)
    behavior_counts = candidate.predicted_behavior.value_counts(dropna=False).to_dict()
    report = {
        "candidate": name,
        "path": str(path),
        "base": str(base_path),
        "sha256": sha256(path),
        "expected_sha256_from_queue": EXPECTED_SHA[name],
        "sha_matches_queue": sha256(path) == EXPECTED_SHA[name],
        "rows": int(len(candidate)),
        "unique_pair_id": int(candidate.pair_id.nunique()),
        "sample_pair_set_equal": candidate_ids == sample_ids,
        "base_pair_set_equal": candidate_ids == set(base.pair_id),
        "risk_no_nan": bool(pd.to_numeric(candidate.risk_score, errors="coerce").notna().all()),
        "risk_in_0_1": bool(pd.to_numeric(candidate.risk_score, errors="coerce").between(0, 1).all()),
        "behavior_counts": behavior_counts,
        "row_diffs": {
            "semantic_changed_rows": diff["semantic_changed_rows"],
            "exact_text_changed_rows": diff["exact_text_changed_rows"],
            "changed_columns": diff["changed_columns"],
        },
        "target_other_count": int(len(target)),
        "target_old_rank_range": [int(target.old_rank.min()), int(target.old_rank.max())] if len(target) else [],
        "target_new_rank_range": [int(target.new_rank.min()), int(target.new_rank.max())] if len(target) else [],
        "target_old_rank_quantiles": {
            str(q): float(target.old_rank.quantile(q)) for q in [0.0, 0.5, 0.9, 1.0]
        }
        if len(target)
        else {},
        "target_new_rank_quantiles": {
            str(q): float(target.new_rank.quantile(q)) for q in [0.0, 0.5, 0.9, 1.0]
        }
        if len(target)
        else {},
        "target_rank_rows": target[["pair_id", "old_rank", "new_rank", "old_risk", "new_risk"]]
        .sort_values("new_rank")
        .to_dict(orient="records"),
        "evidence_legality": evidence_legality(candidate, eval_pairs, data),
    }
    return report, candidate, diff


def main():
    ensure_audit_artifact_dir()
    eval_pairs = pd.read_csv(DATA / "evaluation_pairs.csv")
    data = {"placeholder": True}
    specs = {
        "r2d_p2comb_other_ev_on_r15.csv": OUT.parent / "round15_campaign" / "r15_tabicl_blend.csv",
        "r2e_p2comb_other_evall_on_r15.csv": OUT.parent / "round15_campaign" / "r15_tabicl_blend.csv",
        "r2c_p2top600_other_ev.csv": OUT.parent / "round11_scoped" / "r11_scoped.csv",
    }
    reports = {}
    # The legality helper reads the arrays directly; no large arrays are loaded here.
    for name, base_path in specs.items():
        report, _, _ = candidate_report(name, base_path, eval_pairs, data)
        reports[name] = report
        target_rows = pd.DataFrame(report["target_rank_rows"])
        target_rows.to_csv(AUDIT_ART / f"{name.replace('.csv', '')}_target_ranks.csv", index=False)
    (AUDIT_ART / "candidate_audit.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=float))
    print(json.dumps(reports, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
