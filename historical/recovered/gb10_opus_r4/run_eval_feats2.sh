#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920
taskset -c 16 nice -n 10 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x11_kernel_feats.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2c_role_eval_dt.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x11_kernel_eval_dt.parquet
taskset -c 16 nice -n 10 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x11_kernel_feats.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2c_role_eval_ci.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x11_kernel_eval_ci.parquet
echo KER_DONE
