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
PUBLIC_DIR = PRODUCTION_DIR.parent.parent


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


def _write_slot_pairs(work_dir: Path, data_dir: Path, output_path: Path) -> None:
    player_index = pd.read_parquet(work_dir / "np/player_index.parquet")
    player_map = dict(zip(player_index.player_id, player_index.pi))
    local = pd.read_parquet(work_dir / "player_local_v1.parquet").set_index("player_gi")
    pairs = pd.read_csv(data_dir / "evaluation_pairs.csv", dtype={"pair_id": str})
    low = np.minimum(pairs.player_1.map(player_map), pairs.player_2.map(player_map))
    high = np.maximum(pairs.player_1.map(player_map), pairs.player_2.map(player_map))
    if low.isna().any() or high.isna().any():
        raise ValueError("Evaluation pair has an unknown player")
    if not (local.pool.loc[low].to_numpy() == local.pool.loc[high].to_numpy()).all():
        raise ValueError("Evaluation pair crosses pools")
    pairs["slot"] = (
        local.pool.loc[low].to_numpy() * 900
        + local.local.loc[low].to_numpy() * 30
        + local.local.loc[high].to_numpy()
    )
    if pairs.slot.duplicated().any() or pairs.pair_id.duplicated().any():
        raise ValueError("Duplicate evaluation slot or pair ID")
    pairs[["slot", "pair_id"]].to_csv(output_path, index=False)


def _r25_model_recipes() -> tuple[dict, dict]:
    manifest = json.loads((PRODUCTION_DIR / "r25_fusion_manifest.json").read_text())
    recipes = json.loads((PRODUCTION_DIR / "r25_training_map.json").read_text())
    expected = manifest["old_models"] + manifest["new_models"]
    missing = [tag for tag in expected if tag not in recipes]
    if missing:
        raise RuntimeError(
            "The historical 64+5 risk training map is incomplete; "
            f"{len(missing)} models lack recovered training commands: {missing}"
        )
    absent_scripts = sorted({
        recipe["script"] for recipe in recipes.values()
        if not recipe.get("reuse") and not (PRODUCTION_DIR / recipe["script"]).is_file()
    })
    if absent_scripts:
        raise RuntimeError(f"Risk training scripts are absent: {absent_scripts}")
    return manifest, recipes


def _run_r25_models(work_dir: Path, data_dir: Path, manifest: dict, recipes: dict) -> None:
    for tag in manifest["old_models"] + manifest["new_models"]:
        recipe = recipes[tag]
        if not recipe.get("reuse"):
            _run(recipe["script"], work_dir, data_dir, *recipe["args"], extra_env=recipe["env"])
        prefix = work_dir / "r4" if tag in manifest["new_models"] else work_dir
        score_path = prefix / (
            f"m15_{tag}_eval_scores.parquet" if tag in manifest["new_models"]
            else f"{tag}_eval_scores.parquet"
        )
        if not score_path.is_file():
            raise FileNotFoundError(f"Risk model did not export scores: {tag}: {score_path}")


def run_production_spine(
    data_dir: Path,
    output_dir: Path,
    selected: bool = False,
    threads: int | None = None,
    build_r15_cache: bool = False,
    build_r5_models: bool = False,
    resume_after_policy: bool = False,
    resume_after_evidence: bool = False,
    complete: bool = False,
    tabicl_checkpoint: Path | None = None,
    download_public_checkpoint: bool = False,
    tabicl_device: str = "cpu",
) -> Path:
    if complete:
        r25_manifest, r25_recipes = _r25_model_recipes()
        if not selected:
            raise ValueError("Complete mode requires selected submission assembly")
        build_r15_cache = True
        build_r5_models = True
        if tabicl_checkpoint is None and not download_public_checkpoint:
            raise ValueError("Complete mode requires a TabICL checkpoint or explicit public-checkpoint download")
    if build_r5_models and not build_r15_cache:
        raise ValueError("--build-r5-models requires --build-r15-cache")
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
    if build_r5_models:
        _run("r5_train.py", work_dir, data_dir, extra_env={
            "R5_TEMPLATE": "m25_handfeat2_m19w10_oof.parquet",
            "POKER_BUILD_T1": "1" if complete else "0",
            "NUMBA_NUM_THREADS": worker_threads,
        })
    complete_dir = work_dir / "complete_evidence"
    if complete:
        _run("seq_prep.py", work_dir, data_dir, extra_env={
            "POKER_SEQ_SOURCE": "m19w10_handscores.parquet",
            "POKER_SEQ_POSITIVE_ONLY": "1", "NUMBA_NUM_THREADS": worker_threads,
        })
        _run("seq_within.py", work_dir, data_dir, extra_env={
            "POKER_SEQ_GB_OOF": "m25_handfeat2_m19w10_oof.parquet", "THREADS": worker_threads,
        })
        _run("r5_candidates.py", work_dir, data_dir,
             "--hand-cache", str(r15_cache_path),
             "--stats", str(work_dir / "r5_template_stats.npz"),
             "--rerank", str(work_dir / "r5_rerank_beta.npz"),
             "--models-dir", str(work_dir),
             "--out", str(work_dir / "r5_candidates.parquet"))
        _run("r5_nn_build_seq.py", work_dir, data_dir,
             "--candidates", str(work_dir / "r5_candidates.parquet"),
             "--out-dir", str(complete_dir))
        _run("r5_nn_infer.py", work_dir, data_dir,
             "--candidate-dir", str(complete_dir),
             "--models-dir", str(work_dir / "seq"),
             "--out", str(complete_dir / "r5_neural.parquet"),
             extra_env={"THREADS": worker_threads})
        complete_dir.mkdir(parents=True, exist_ok=True)
        _run("r15_dev_scores.py", work_dir, data_dir,
             "--candidates", str(work_dir / "r5_dev_candidates.parquet"),
             "--neural", str(work_dir / "seqwithin_oof.parquet"),
             "--scores", str(complete_dir / "dev_candidate_scores.csv"),
             "--extras", str(complete_dir / "dev_extras.csv"))
        _run("r15_tabicl_input.py", work_dir, data_dir,
             "--candidates", str(complete_dir / "dev_candidate_scores.csv"),
             "--extras", str(complete_dir / "dev_extras.csv"),
             "--out", str(complete_dir / "dev_tabicl.csv"))
        from .production.r15_tabicl_input import EXTRAS, SCORES

        feature_path = complete_dir / "tabicl_features.json"
        feature_path.write_text(json.dumps(SCORES + EXTRAS, indent=2) + "\n")
        _run("r11_prepare_moments.py", work_dir, data_dir,
             "--candidates", str(complete_dir / "dev_candidate_scores.csv"),
             "--np-dir", str(work_dir / "np"),
             "--legacy-code", str(PUBLIC_DIR / "historical/scripts/round11/legacy_r8"),
             "--out", str(complete_dir / "dev_ranker_pack"))
        _run("r11_ranker.py", work_dir, data_dir,
             "fit", "--pack", str(complete_dir / "dev_ranker_pack"),
             "--out", str(complete_dir / "ranker_fit"),
             "--threads", worker_threads)
        cv_args = ["cv", "--input", str(complete_dir / "dev_tabicl.csv"),
                   "--features", str(feature_path),
                   "--out", str(complete_dir / "tabicl_cv"),
                   "--device", tabicl_device,
                   "--threads", worker_threads]
        if tabicl_checkpoint is not None:
            cv_args.extend(["--checkpoint", str(tabicl_checkpoint.resolve())])
        if download_public_checkpoint:
            cv_args.append("--download-public-checkpoint")
        _run("r15_tabicl_model.py", work_dir, data_dir, *cv_args)
        _run("r15_tabicl_blend.py", work_dir, data_dir,
             "--ranker-scores", str(complete_dir / "ranker_fit/oof_scores.csv.gz"),
             "--tabicl-scores", str(complete_dir / "tabicl_cv/predictions.csv.gz"),
             "--out", str(complete_dir / "dev_scored_blend.csv"))
        dev_scored = pd.read_csv(complete_dir / "dev_scored_blend.csv")
        dev_hands = pd.read_parquet(work_dir / "r5_dev_candidates.parquet")[["slot", "hand_id", "h"]]
        dev_scored = dev_scored.merge(dev_hands, on=["slot", "hand_id"], validate="one_to_one")
        dev_scored = dev_scored.sort_values(
            ["slot", "score", "hand_id"], ascending=[True, False, True], kind="stable"
        )
        dev_scored["r"] = dev_scored.groupby("slot").cumcount() + 1
        dev_scored[["slot", "h", "r"]].to_parquet(
            complete_dir / "dev_role_candidates.parquet", index=False
        )
        _run("r32_dev_frames.py", work_dir, data_dir,
             "--work-dir", str(work_dir), "--data-dir", str(data_dir))
        _run("r32_seq_feats.py", work_dir, data_dir)
        _run("r32_role_feats.py", work_dir, data_dir,
             "dev", str(work_dir / "r4/dev_pairs.parquet"),
             str(work_dir / "r4/x2c_role_dev.parquet"),
             extra_env={"ORIENT": "cand", "CANDFILE": str(complete_dir / "dev_role_candidates.parquet")})
        _run("r32_kernel_feats.py", work_dir, data_dir,
             str(work_dir / "r4/x2c_role_dev.parquet"),
             str(work_dir / "r4/x11_kernel_dev.parquet"))
        fit_args = ["fit", "--input", str(complete_dir / "dev_tabicl.csv"),
                    "--features", str(feature_path),
                    "--out", str(complete_dir / "tabicl_fit"),
                    "--device", tabicl_device,
                    "--threads", worker_threads]
        if tabicl_checkpoint is not None:
            fit_args.extend(["--checkpoint", str(tabicl_checkpoint.resolve())])
        if download_public_checkpoint:
            fit_args.append("--download-public-checkpoint")
        _run("r15_tabicl_model.py", work_dir, data_dir, *fit_args)
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
    if complete:
        _run("r25_mil_lr.py", work_dir, data_dir)
        _run("r25_infoshare.py", work_dir, data_dir)
        _run("r25_fam_llr.py", work_dir, data_dir)
    if not complete:
        _remove_intermediates(
            work_dir, "R_v1.npy", "R2_v1.npy", "P2_v1.npy", "dec_probs_v1.npy",
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
    if complete:
        _run("r25_more_devsub.py", work_dir, data_dir, "13,14,15")
    _remove_intermediates(
        work_dir, "R4_v2.npy", "P4_v2.npy",
        *(() if not complete else (
            "R2_v1.npy", "P2_v1.npy",
            "m26_handscore_phase0.npy", "m26_slot_phase0.npy", "m26_h_phase0.npy",
            "m26_handscore_phase1.npy", "m26_slot_phase1.npy", "m26_h_phase1.npy",
        )),
    )
    _log_disk(work_dir)

    model_stages: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = [
        ("m36_pair_ens.py", ("drop", "m26", "v6ens_base"), {"LEARNER": "lgb", "SEEDS": "7"}),
        ("m36_pair_ens.py", ("drop", "m26", "v6ens_cat"), {"LEARNER": "cat", "SEEDS": "7"}),
        ("m36_pair_ens.py", ("drop", "m26", "v6ens_cat11"), {"LEARNER": "cat", "SEEDS": "11"}),
    ]
    for script, args, extra_env in model_stages:
        _run(script, work_dir, data_dir, *args, extra_env=extra_env)
    if complete:
        _run_r25_models(work_dir, data_dir, r25_manifest, r25_recipes)

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
        r32_prepatch = output_dir / "r32_r30_dtgb15_prepatch.csv"
        if complete:
            slot_pairs_path = work_dir / "evaluation_slot_pairs.csv"
            _write_slot_pairs(work_dir, data_dir, slot_pairs_path)
            _run("r25_risk_fusion.py", work_dir, data_dir,
                 "--artifact-root", str(work_dir),
                 "--player-local", str(work_dir / "player_local_v1.parquet"),
                 "--slot-pairs", str(slot_pairs_path),
                 "--base", str(work_dir / "f4_ndw_baseline.csv"),
                 "--out", str(r32_prepatch))
            _run("r15_complete_evidence.py", work_dir, data_dir,
                 "prepare", "--candidates", str(work_dir / "r5_candidates.parquet"),
                 "--r10", str(work_dir / "f4_ndw_baseline.csv"),
                 "--r32", str(r32_prepatch), "--out-dir", str(complete_dir))
            _run("r15_candidate_scores.py", work_dir, data_dir,
                 "--input", str(complete_dir / "gated_candidates.parquet"),
                 "--np-dir", str(work_dir / "np"),
                 "--local-index", str(work_dir / "player_local_v1.parquet"),
                 "--nn", str(complete_dir / "r5_neural.parquet"),
                 "--slot-pairs", str(slot_pairs_path),
                 "--out", str(complete_dir / "eval_candidate_scores.csv"))
            _run("r15_eval_extras.py", work_dir, data_dir,
                 "--candidates", str(complete_dir / "eval_candidate_scores.csv"),
                 "--hand-cache", str(r15_cache_path),
                 "--stats", str(work_dir / "r5_template_stats.npz"),
                 "--out", str(complete_dir / "eval_extras.csv"))
            _run("r15_tabicl_input.py", work_dir, data_dir,
                 "--candidates", str(complete_dir / "eval_candidate_scores.csv"),
                 "--extras", str(complete_dir / "eval_extras.csv"),
                 "--out", str(complete_dir / "eval_tabicl.csv"))
            _run("r11_prepare_moments.py", work_dir, data_dir,
                 "--candidates", str(complete_dir / "eval_candidate_scores.csv"),
                 "--np-dir", str(work_dir / "np"),
                 "--legacy-code", str(PUBLIC_DIR / "historical/scripts/round11/legacy_r8"),
                 "--out", str(complete_dir / "eval_ranker_pack"))
            _run("r11_ranker.py", work_dir, data_dir,
                 "predict", "--pack", str(complete_dir / "eval_ranker_pack"),
                 "--model", str(complete_dir / "ranker_fit/full"),
                 "--out", str(complete_dir / "ranker_scores.csv"))
            _run("r15_tabicl_model.py", work_dir, data_dir,
                 "predict", "--input", str(complete_dir / "eval_tabicl.csv"),
                 "--features", str(feature_path),
                 "--device", tabicl_device,
                 "--fitted", str(complete_dir / "tabicl_fit/classifier.pkl"),
                 "--out", str(complete_dir / "tabicl_eval"))
            _run("r15_tabicl_blend.py", work_dir, data_dir,
                 "--ranker-scores", str(complete_dir / "ranker_scores.csv"),
                 "--tabicl-scores", str(complete_dir / "tabicl_eval/predictions.csv.gz"),
                 "--out", str(complete_dir / "scored_blend.csv"))
            r10_evidence = complete_dir / "r10_evidence.csv"
            _run("r15_complete_evidence.py", work_dir, data_dir,
                 "patch", "--base", str(work_dir / "f4_ndw_baseline.csv"),
                 "--scored", str(complete_dir / "scored_blend.csv"),
                 "--out", str(r10_evidence))
            for family, tag, mode, par, patch_name in (
                ("soft_play", "sp", "rank", "0.35", "patch_r5_sp_zoo.csv"),
                ("directed_transfer", "dt", "stack", "1.0", "patch_r5_dt_zoo_gb15.csv"),
            ):
                full_path = work_dir / f"r4/y1_{tag}_eval_full.parquet"
                candidate_path = work_dir / f"r4/y1_{tag}_eval_candidates.parquet"
                role_path = work_dir / f"r4/x2c_role_eval_{tag}.parquet"
                kernel_path = work_dir / f"r4/x11_kernel_eval_{tag}.parquet"
                new_path = work_dir / f"r4/y2_{tag}_eval_newfeats.parquet"
                _run("r32_eval_frames.py", work_dir, data_dir,
                     "--work-dir", str(work_dir), "--data-dir", str(data_dir),
                     "--base", str(r32_prepatch),
                     "--scored", str(complete_dir / "scored_blend.csv"),
                     "--family", family)
                _run("r32_role_feats.py", work_dir, data_dir,
                     "eval", str(full_path), str(role_path),
                     extra_env={"ORIENT": "cand", "CANDFILE": str(candidate_path)})
                _run("r32_kernel_feats.py", work_dir, data_dir,
                     str(role_path), str(kernel_path))
                _run("r32_eval_seq_feats.py", work_dir, data_dir,
                     str(full_path), str(new_path))
                _run("r32_zoo_deploy.py", work_dir, data_dir,
                     family, mode, par, str(full_path), str(new_path),
                     str(role_path), str(kernel_path), str(candidate_path),
                     str(complete_dir / patch_name),
                     extra_env={
                         "NJ": worker_threads,
                         "GB": "1.5" if family == "directed_transfer" else "1.0",
                         "POKER_TABICL_EVAL": str(complete_dir / "tabicl_eval/predictions.csv.gz"),
                     })
            r10_base = r10_evidence
            r32_base = r32_prepatch
        else:
            _assemble_rank_fusion(
                work_dir, data_dir, r32_baseline_path, r32_prepatch,
                weights=(0.30, 0.45, 0.25),
            )
            r10_base = work_dir / "f4_ndw_baseline.csv"
            r32_base = r32_prepatch
        r10, r10_changed = apply_evidence_patch(
            pd.read_csv(r10_base, dtype={"pair_id": str}), ci_patch_path
        )
        r10_path = output_dir / "r10_ci.csv"
        r10.to_csv(r10_path, index=False)
        r32_path = output_dir / "r32_r30_dtgb15.csv"
        if complete:
            from r32_apply_patch import apply as apply_family_patch

            sp_patch_path = complete_dir / "patch_r5_sp_zoo.csv"
            dt_patch_path = complete_dir / "patch_r5_dt_zoo_gb15.csv"
            if not sp_patch_path.is_file() or not dt_patch_path.is_file():
                raise FileNotFoundError(
                    "The historical soft-play and directed-transfer zoo patches "
                    f"are required for r32: {sp_patch_path}, {dt_patch_path}. "
                    "A CI patch must not be reused for r32."
                )
            r30_path = output_dir / "r30_r29_spzoo.csv"
            apply_family_patch(r32_base, sp_patch_path, "soft_play", r30_path)
            apply_family_patch(r30_path, dt_patch_path, "directed_transfer", r32_path)
            r32_patch_family = "directed_transfer"
        else:
            pd.read_csv(r32_base, dtype={"pair_id": str}).to_csv(r32_path, index=False)
            r32_patch_family = None
        receipt = {
            "r10_ci_patch_changed_rows": r10_changed,
            "r32_patch_family": r32_patch_family,
            "r10_path": str(r10_path),
            "r32_path": str(r32_path),
        }
        if build_r15_cache:
            receipt["r15_within_cache"] = str(r15_cache_path)
        if build_r5_models:
            receipt["r5_models_dir"] = str(work_dir)
        receipt["complete_method_path"] = complete
        (output_dir / "selected_assembly_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        _remove_intermediates(
            work_dir, "P_v1.npy", "dec_Y.npy", "dec_probs_v2.npy",
            *(() if not complete else ("dec_probs_v1.npy", "R_v1.npy")),
        )
        return r10_path

    output_path = output_dir / "production_spine.csv"
    _assemble_rank_fusion(work_dir, data_dir, baseline_path, output_path)
    return output_path
