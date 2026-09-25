#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920
ORIENT=cand CANDFILE=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/y1_dt_eval_candidates.parquet taskset -c 18 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x2_role_feats_v2.py eval /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/y1_dt_eval_full.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2c_role_eval_dt.parquet
taskset -c 18 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x9_newfeats_fixed.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2c_role_eval_dt.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x9_newfix_eval_dt.parquet
echo DT_DONE
