#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920
ORIENT=cand CANDFILE=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r18/main/ci_eval_candidates.parquet taskset -c 19 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x2_role_feats_v2.py eval /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r3/t61_ci_eval_feats.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2c_role_eval_ci.parquet
taskset -c 19 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x9_newfeats_fixed.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2c_role_eval_ci.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x9_newfix_eval_ci.parquet
echo CI_DONE
