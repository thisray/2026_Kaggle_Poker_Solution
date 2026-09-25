#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920
taskset -c 17 nice -n 10 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python y1_eval_frames.py soft_play /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates/r13_ndwrank_cinew_f4.csv
taskset -c 17 nice -n 10 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python y2_eval_newfeats.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/y1_sp_eval_full.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/y2_sp_eval_newfeats.parquet
ORIENT=cand CANDFILE=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/y1_sp_eval_candidates.parquet taskset -c 17 nice -n 10 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x2_role_feats_v2.py eval /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/y1_sp_eval_full.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2c_role_eval_sp.parquet
taskset -c 17 nice -n 10 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x11_kernel_feats.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2c_role_eval_sp.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x11_kernel_eval_sp.parquet
echo SP_DONE
