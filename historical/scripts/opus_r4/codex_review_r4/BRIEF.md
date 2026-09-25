# Codex task: independent code review + tests of the R4 evidence-patch pipeline (testing only)

Work ONLY inside this directory (`codex_review_r4/`). Do NOT modify any file outside it. Do NOT submit anything to Kaggle, do NOT use the network.
Python: `/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python` (conda env; never system python). Prefix heavy commands with `taskset -c 16,17 nice -n 15` and set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2`. The machine is shared: keep jobs light (no model training over more than ~2 minutes).

## Files under review (read-only, parent directory `..` = /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920)
- `../x2_role_feats.py` — label-free role-oriented per-hand features for every co-seated hand of a pair list (dev phase 0 or eval phase 1).
- `../y1_eval_frames.py` — eval frame (all co-seated eval hands of pairs routed to a family) + frozen R15 top-20 candidates.
- `../y2_eval_newfeats.py` — copy of the other session's t61 (t58 feature block) with paths as arguments.
- `../y3_deploy_event.py` — train the censored-event model on dev, predict eval, first-five DP, rank blend with R15, write patch.
- `../y3b_deploy_stack.py` — same, but final score = logit(TabICL eval probability) + BETA * logit(first-five marginal q).
- `../y5_build_candidates.py` — apply patches to a base candidate.
- dependency (read-only): `/home/thisray/projects/260916_Kaggle_Poker_workers/r18/ci_censored_event.py` (`prepare`, `uncensored_training_rows`, `first_k_marginal`, `rank_candidates`, `pair_ap`).
Artifacts: `/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/` (`x2_role_dev.parquet`, `x2_role_eval_dt.parquet`, `x2_role_eval_ci.parquet`, `y1_dt_eval_full.parquet`, `y1_dt_eval_candidates.parquet`, `y2_dt_eval_newfeats.parquet`, `patch_r4_dt_w035.csv`, `patch_r4_ci_stack_b3.csv`, `cand/r17_r13_dt_cistack.csv`, `cand/r18_r14_dt_cistack.csv`), dev sequence table `.../opus_r1_20260917/t5_dev_seq.parquet`, raw arrays `.../opus_r1_20260917/np/*.npy`.

## Intended behaviour (check the code implements exactly this)
1. Orientation: for each pair, receiver R = the member who received more directed chip flow from the partner summed over ALL co-seated hands of the phase (flow in a hand = min(loss of one member, gain of the other), in big blinds). No label is used anywhere in eval features.
2. Every per-pair cumulative feature (`x_k`, `x_k_bigR`, `x_k_cell`, `c_k_*`) counts strictly EARLIER hands in (ts, h) order; `x_rel = (k + 0.5) / n`; `c_rel_* = (c_k + 0.5) / max(c_n, 1)`.
3. Dev and eval features come from the same code path; the feature list used for training equals the list used for prediction (same order).
4. Training rows: dev rows of the family that survive right-censoring (`uncensored_training_rows`). Prediction: all co-seated eval hands of the routed pairs. DP: `first_k_marginal` over the full time-ordered sequence of each pair; candidates = frozen R15 top-20; DT final score = 0.65 * pct-rank(R15) + 0.35 * pct-rank(q); CI final score = logit(tab) + 3 * logit(q). Top-5 by score, ties by (ts, h).
5. Patches touch only evidence columns of pairs routed to the patch family; risk_score / predicted_behavior / row order identical to the base.

## Tasks
1. Code review against the intended behaviour: row alignment after merges/sorts (the DP uses positional indexing on the frame — check that `p`, the frame and `first_k_marginal` stay aligned after `assemble()` sorts and resets the index), merge cardinalities, dtype/NaN handling, anything that would make dev and eval features differ in definition, any use of labels on the eval side. Report each finding with file:line, severity (bug / risk / nit) and a concrete fix.
2. Tests (write `test_*.py` here and run them):
   a. Recompute `x_k`, `x_rel`, `x_k_cell`, `x_k_bigR`, `x_n_cell` for 20 random eval DT pairs directly from `x2_role_eval_dt.parquet` rows with explicit Python loops; compare with the stored columns.
   b. For 10 random eval DT pairs recompute netR/netS/conR/conS/x_dir/x_s_fold_to_r for every hand directly from the raw `np/*.npy` arrays with an independent implementation (explicit loops over actions) and compare with the stored features; also recompute the pair orientation (receiver seat) independently.
   c. `first_k_marginal` vs brute-force enumeration over all 2^n event patterns for n <= 10 random probabilities.
   d. Patch integrity: for `cand/r17_r13_dt_cistack.csv` vs its base `.../r2_candidates/r13_ndwrank_cinew_f4.csv`: risk/behaviour/row order identical; every changed row is routed to DT or CI; every evidence hand of a patched row is one of that pair's 20 R15 candidates (`y1_dt_eval_candidates.parquet` for DT, `.../r18/main/ci_eval_candidates.parquet` for CI); no duplicates within a row.
   e. Dev/eval definitional parity: run `x2_role_feats.py`-equivalent logic (import its row loop by copying it verbatim) on 5 dev pairs and check equality with `x2_role_dev.parquet`.
3. Write `REVIEW.md` here: summary verdict (are the R4 patches trustworthy?), findings table, test results with the exact commands you ran.
