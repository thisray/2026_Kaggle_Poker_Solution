#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920
/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python - <<PYEOF
import pandas as pd
p = pd.read_parquet("/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_pairs.parquet"); p[p.grp != "pos"][["slot", "pa", "pb"]].to_parquet("/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_pairlist.parquet")
PYEOF
ORIENT=flow taskset -c 18 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x2_role_feats_v2.py dev /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_pairlist.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_role.parquet
taskset -c 18 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x11_kernel_feats.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_role.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_band_kernel.parquet
taskset -c 18 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python x11_kernel_feats.py /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/x2_role_dev.parquet /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r4/p1_pos_kernel_flow.parquet
echo P1_FEATS_DONE
