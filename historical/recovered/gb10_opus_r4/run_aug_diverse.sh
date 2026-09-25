#!/usr/bin/env bash
# R3-P26: extend R4's exposure-matched data augmentation (5 dev subsamples) to learner families R4 did not cover.
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
mkdir -p logs_q3
run() { TAG=$1; shift; env PYTHONPATH=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917 "$@" EXTRA_SUBS=13,14,15 THREADS=4 taskset -c $CORES nice -n 12 $PY q2_pair_more_subs.py $PU $MIL "$TAG" >> logs_q3/$TAG.log 2>&1; }
sA() { CORES=4,5,6,7
  PU=pos  MIL=m26 run q3_dart_a  SEED=1301 LEAVES=31 FF=0.5  LR=0.03  BOOST=dart
  PU=pos  MIL=m26 run q3_extra_a SEED=1307 LEAVES=23 FF=0.55 LR=0.032 EXTRA=1
  PU=pos  MIL=m26 run q3_view_a  SEED=1311 LEAVES=47 FF=0.45 LR=0.028 FEATVIEW=101:0.75
  echo DONE_A >> logs_q3/all.log; }
sB() { CORES=8,9,10,11
  PU=none MIL=m26 run q3_none_a  SEED=1319 LEAVES=31 FF=0.5  LR=0.03
  PU=pos  MIL=m26 run q3_deep_a  SEED=1327 LEAVES=79 FF=0.35 LR=0.022 MINLEAF=35
  PU=pos  MIL=m9  run q3_m9_a    SEED=1333 LEAVES=63 FF=0.4  LR=0.026
  echo DONE_B >> logs_q3/all.log; }
sA & sB & wait
echo DONE_ALL >> logs_q3/all.log
