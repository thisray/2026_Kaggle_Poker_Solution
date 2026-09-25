#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920
ORIENT=flow FLIP=1 taskset -c 16 nice -n 10 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x2_role_feats.py dev /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_pairlist.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_roleflip.parquet
taskset -c 16 nice -n 10 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x11_kernel_feats.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_roleflip.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_kernelflip.parquet
echo P2_FEATS_DONE
