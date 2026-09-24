"""Run the recovered production spine from official files.

This is intentionally a fixed execution path, not an experiment harness. It
rebuilds the deployed Opus R1 pair ensemble, family router, and m19 evidence
detector before assembling a legal baseline submission.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata


PRODUCTION_DIR = Path(__file__).with_name("production")


def _run(script: str, work_dir: Path, data_dir: Path, *args: str, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env.update(
        POKER_WORK_DIR=str(work_dir),
        POKER_DATA_DIR=str(data_dir),
        PYTHONPATH=str(PRODUCTION_DIR),
    )
    if extra_env:
        env.update(extra_env)
    command = [sys.executable, str(PRODUCTION_DIR / script), *args]
    print("[production]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PRODUCTION_DIR, env=env, check=True)


def _remove_intermediates(work_dir: Path, *names: str) -> None:
    for name in names:
        path = work_dir / name
        if path.exists():
            size_gib = path.stat().st_size / 1024**3
            path.unlink()
            print(f"[production] removed {name} ({size_gib:.2f} GiB)", flush=True)


def _log_disk(path: Path) -> None:
    usage = shutil.disk_usage(path)
    print(
        f"[production] disk used={usage.used / 1024**3:.2f} GiB "
        f"free={usage.free / 1024**3:.2f} GiB",
        flush=True,
    )


def _build_evidence_cache(work_dir: Path, data_dir: Path, output_dir: Path) -> Path:
    cache_path = work_dir / "production_m19_eval_hands.parquet"
    preview_path = output_dir / "production_m19_preview.csv"
    config = {
        "risk_file": str(work_dir / "m1_eval_scores.parquet"),
        "family_file": str(work_dir / "m7_family_eval.parquet"),
        "hand_cache": str(cache_path),
        "s1_featmod": "handfeat2",
        "s1_models": [str(work_dir / f"m19w10_hand_f{fold}.txt") for fold in range(5)],
        "s1_desc": True,
        "s1_probs": "dec_probs_v1.npy",
        "variants": [{"out": str(preview_path), "score_col": "s1", "mode": "plt5", "scale": 1.0}],
    }
    _run("build_submission2.py", work_dir, data_dir, json.dumps(config))
    preview_path.unlink()
    return cache_path


def _assemble_rank_fusion(
    work_dir: Path,
    data_dir: Path,
    baseline_path: Path,
    output_path: Path,
    weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
) -> None:
    player_index = pd.read_parquet(work_dir / "np/player_index.parquet")
    player_map = dict(zip(player_index.player_id, player_index.pi))
    evaluation_pairs = pd.read_csv(data_dir / "evaluation_pairs.csv")
    low = np.minimum(evaluation_pairs.player_1.map(player_map), evaluation_pairs.player_2.map(player_map))
    high = np.maximum(evaluation_pairs.player_1.map(player_map), evaluation_pairs.player_2.map(player_map))
    evaluation_pairs["key"] = low * 12000 + high

    scores = evaluation_pairs[["pair_id", "key"]].copy()
    tags = ("v6ens_base", "v6ens_cat", "v6ens_cat11")
    for tag in tags:
        model_scores = pd.read_parquet(work_dir / f"m15_{tag}_eval_scores.parquet")[["key", "score"]]
        scores = scores.merge(model_scores.rename(columns={"score": tag}), on="key", validate="one_to_one")

    count = len(scores)
    scores["mix"] = sum(
        weight * rankdata(scores[tag]) / count
        for tag, weight in zip(tags, weights)
    )
    order = scores.sort_values(["mix", "pair_id"], ascending=[False, True]).pair_id.to_numpy()
    risk = pd.Series((count - np.arange(count)) / count, index=order)

    output = pd.read_csv(baseline_path, dtype={"pair_id": str})
    output["risk_score"] = output.pair_id.map(risk)
    output.to_csv(output_path, index=False)


def run_production_spine(
    data_dir: Path,
    output_dir: Path,
    selected: bool = False,
    threads: int | None = None,
    build_r15_cache: bool = False,
    resume_after_policy: bool = False,
    resume_after_evidence: bool = False,
) -> Path:
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    worker_threads = str(min(16, threads or (os.cpu_count() or 4)))
    work_dir = output_dir / "production_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    initial_stages: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = [
        ("e1_export.py", (), None),
        ("e2_kernel.py", (), {"NUMBA_NUM_THREADS": worker_threads}),
        ("e3_pairs.py", (), None),
        ("m1_pair.py", (), None),
        ("e4_infoset.py", (), {"NUMBA_NUM_THREADS": worker_threads}),
        ("m4_policy.py", (), {"NUMBA_NUM_THREADS": worker_threads}),
        ("m4b_policy_v2.py", (), {"NUMBA_NUM_THREADS": worker_threads}),
    ]
    if not (resume_after_policy or resume_after_evidence):
        for script, args, extra_env in initial_stages:
            _run(script, work_dir, data_dir, *args, extra_env=extra_env)
        if selected:
            _run("f4_infoshare.py", work_dir, data_dir, extra_env={"NUMBA_NUM_THREADS": worker_threads})
            _run("f4_combined.py", work_dir, data_dir, extra_env={"NUMBA_NUM_THREADS": worker_threads})
        _remove_intermediates(work_dir, "dec_X.npy", "policy_v1.txt", "policy_v2.txt")
        _log_disk(work_dir)

    evidence_stages: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = [
        ("e5_sur_kernel.py", (), {"NUMBA_NUM_THREADS": worker_threads}),
        ("m5_build_tables.py", (), None),
        ("m5_pair.py", ("both",), None),
        ("m7_family.py", (), None),
        ("m19_hand_hnw.py", (), {"HNW": "10", "MTAG": "m19w10", "DESC": "1"}),
    ]
    if not resume_after_evidence:
        for script, args, extra_env in evidence_stages:
            _run(script, work_dir, data_dir, *args, extra_env=extra_env)
        evidence_cache = _build_evidence_cache(work_dir, data_dir, output_dir)
    else:
        evidence_cache = work_dir / "production_m19_eval_hands.parquet"
        if not evidence_cache.is_file():
            raise FileNotFoundError(f"Missing evidence cache for resume: {evidence_cache}")
    r15_cache_path = work_dir / "r15_within_eval_hands.parquet"
    if build_r15_cache:
        _run("r15_within_train.py", work_dir, data_dir, extra_env={"RUNS": "fam", "GEN": "m19w10"})
        _run(
            "r15_within_eval.py", work_dir, data_dir,
            "--cache", str(evidence_cache),
            "--models-dir", str(work_dir),
            "--out", str(r15_cache_path),
            extra_env={"NUMBA_NUM_THREADS": worker_threads},
        )
    ci_patch_path = work_dir / "r18_ci_patch.csv"
    if selected:
        os.environ["POKER_WORK_DIR"] = str(work_dir)
        os.environ["POKER_DATA_DIR"] = str(data_dir)
        if str(PRODUCTION_DIR) not in sys.path:
            sys.path.insert(0, str(PRODUCTION_DIR))
        from late_stage import prepare_ci_patch

        receipt = prepare_ci_patch(
            work_dir,
            data_dir,
            evidence_cache,
            ci_patch_path,
            threads=int(worker_threads),
        )
        print(f"[production] R18 patch prepared: {receipt}", flush=True)
    _run("m26_handagg_m19.py", work_dir, data_dir)
    _remove_intermediates(
        work_dir,
        "R_v1.npy", "R2_v1.npy", "P2_v1.npy", "dec_probs_v1.npy",
        "m26_handscore_phase0.npy", "m26_slot_phase0.npy", "m26_h_phase0.npy",
        "m26_handscore_phase1.npy", "m26_slot_phase1.npy", "m26_h_phase1.npy",
    )
    _log_disk(work_dir)

    final_stages: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = [
        ("e8_kernel.py", ("dec_probs_v2.npy", "v2"), {"NUMBA_NUM_THREADS": worker_threads}),
        ("m14_tables3.py", (), None),
    ]
    for script, args, extra_env in final_stages:
        _run(script, work_dir, data_dir, *args, extra_env=extra_env)
    _remove_intermediates(work_dir, "R4_v2.npy", "P4_v2.npy")
    _log_disk(work_dir)

    model_stages: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = [
        ("m36_pair_ens.py", ("drop", "m26", "v6ens_base"), {"LEARNER": "lgb", "SEEDS": "7"}),
        ("m36_pair_ens.py", ("drop", "m26", "v6ens_cat"), {"LEARNER": "cat", "SEEDS": "7"}),
        ("m36_pair_ens.py", ("drop", "m26", "v6ens_cat11"), {"LEARNER": "cat", "SEEDS": "11"}),
    ]
    for script, args, extra_env in model_stages:
        _run(script, work_dir, data_dir, *args, extra_env=extra_env)

    baseline_path = output_dir / "production_m19_baseline.csv"
    r32_baseline_path = output_dir / "production_m19_r32_baseline.csv"
    variants = [{"out": str(baseline_path), "score_col": "s1", "mode": "plt5", "scale": 1.0}]
    if selected:
        variants.append({"out": str(r32_baseline_path), "score_col": "s1", "mode": "plt5", "scale": 1.5})
    config = {
        "risk_file": str(work_dir / "m15_v6ens_base_eval_scores.parquet"),
        "family_file": str(work_dir / "m7_family_eval.parquet"),
        "hand_cache": str(evidence_cache),
        "s1_featmod": "handfeat2",
        "s1_models": [str(work_dir / f"m19w10_hand_f{fold}.txt") for fold in range(5)],
        "s1_desc": True,
        "s1_probs": "dec_probs_v1.npy",
        "variants": variants,
    }
    _run("build_submission2.py", work_dir, data_dir, json.dumps(config))

    if selected:
        from late_stage import apply_evidence_patch

        r10_prepatch = output_dir / "r10_ci_prepatch.csv"
        _assemble_rank_fusion(work_dir, data_dir, baseline_path, r10_prepatch)
        _run("f4_route.py", work_dir, data_dir, str(r10_prepatch))
        _run("f4_hand_tables.py", work_dir, data_dir, str(work_dir / "f4_pair_ids.txt"), "ext")
        _run("f4_ndw.py", work_dir, data_dir)
        r10, r10_changed = apply_evidence_patch(
            pd.read_csv(work_dir / "f4_ndw_baseline.csv", dtype={"pair_id": str}), ci_patch_path
        )
        r10_path = output_dir / "r10_ci.csv"
        r10.to_csv(r10_path, index=False)

        r32_prepatch = output_dir / "r32_r30_dtgb15_prepatch.csv"
        _assemble_rank_fusion(
            work_dir,
            data_dir,
            r32_baseline_path,
            r32_prepatch,
            weights=(0.30, 0.45, 0.25),
        )
        r32, r32_changed = apply_evidence_patch(
            pd.read_csv(r32_prepatch, dtype={"pair_id": str}), ci_patch_path
        )
        r32_path = output_dir / "r32_r30_dtgb15.csv"
        r32.to_csv(r32_path, index=False)
        receipt = {
            "r10_ci_patch_changed_rows": r10_changed,
            "r32_ci_patch_changed_rows": r32_changed,
            "r10_path": str(r10_path),
            "r32_path": str(r32_path),
        }
        if build_r15_cache:
            receipt["r15_within_cache"] = str(r15_cache_path)
        (output_dir / "selected_assembly_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        _remove_intermediates(work_dir, "P_v1.npy", "dec_Y.npy", "dec_probs_v2.npy")
        return r10_path

    output_path = output_dir / "production_spine.csv"
    _assemble_rank_fusion(work_dir, data_dir, baseline_path, output_path)
    return output_path
