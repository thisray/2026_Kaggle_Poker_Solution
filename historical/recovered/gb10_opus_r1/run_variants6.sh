#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
run() { TAG=$1; shift; env "$@" THREADS=4 taskset -c $CORES nice -n 12 $PY t69_pair_variant.py $PU $MIL "$TAG" >> logs_r3/t69_$TAG.log 2>&1; }
sA() { CORES=10,11,12,13
  PU=pos MIL=m26 run o_p2_a SEED=901 LEAVES=47 FF=0.45 LR=0.028 FEATVIEW=71:0.7
  PU=pos MIL=m9  run o_p2_b SEED=907 LEAVES=31 FF=0.5  LR=0.03  BOOST=goss
  PU=pos MIL=m26 run o_p2_c SEED=911 LEAVES=79 FF=0.35 LR=0.022 MINLEAF=35
  PU=pos MIL=m26 run o_p2_d SEED=919 LEAVES=23 FF=0.55 LR=0.032 EXTRA=1; }
sB() { CORES=14,15,2,3
  PU=pos MIL=m9  run o_p2_e SEED=929 LEAVES=63 FF=0.4  LR=0.026 FEATVIEW=83:0.6
  PU=pos MIL=m26 run o_p2_f SEED=937 LEAVES=31 FF=0.6  LR=0.03  TOUCH=1
  PU=pos MIL=m26 run o_p2_g SEED=941 LEAVES=127 FF=0.3 LR=0.018 ROUNDS=1600 MINLEAF=25
  PU=pos MIL=m9  run o_p2_h SEED=947 LEAVES=31 FF=0.5  LR=0.03  EXTRA=1 FEATVIEW=97:0.65; }
sA & sB & wait; echo DONE6
