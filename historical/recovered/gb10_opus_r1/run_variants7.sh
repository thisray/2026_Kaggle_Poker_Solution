#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
run() { TAG=$1; shift; env "$@" THREADS=4 taskset -c $CORES nice -n 12 $PY t69_pair_variant.py $PU $MIL "$TAG" >> logs_r3/t69_$TAG.log 2>&1; }
sA() { CORES=10,11,12,13
  PU=pos MIL=m26 run o_p3_a SEED=1201 LEAVES=47 FF=0.45 LR=0.028 FEATVIEW=101:0.75
  PU=pos MIL=m9  run o_p3_b SEED=1207 LEAVES=31 FF=0.5  LR=0.03  BOOST=goss
  PU=pos MIL=m26 run o_p3_c SEED=1211 LEAVES=79 FF=0.35 LR=0.022 MINLEAF=35
  PU=pos MIL=m26 run o_p3_d SEED=1219 LEAVES=23 FF=0.55 LR=0.032 EXTRA=1; }
sB() { CORES=14,15,2,3
  PU=pos MIL=m9  run o_p3_e SEED=1229 LEAVES=63 FF=0.4  LR=0.026 FEATVIEW=103:0.5
  PU=pos MIL=m26 run o_p3_f SEED=1237 LEAVES=31 FF=0.6  LR=0.03  TOUCH=1
  PU=pos MIL=m26 run o_p3_g SEED=1241 LEAVES=127 FF=0.3 LR=0.018 ROUNDS=1600 MINLEAF=25
  PU=pos MIL=m9  run o_p3_h SEED=1247 LEAVES=31 FF=0.5  LR=0.03  EXTRA=1 FEATVIEW=107:0.55; }
sA & sB & wait; echo DONE7
