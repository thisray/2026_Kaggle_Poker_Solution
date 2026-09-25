#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
run() { TAG=$1; shift; env "$@" THREADS=4 taskset -c $CORES nice -n 12 $PY t69_pair_variant.py $PU $MIL "$TAG" >> logs_r3/t69_$TAG.log 2>&1; }
sA() { CORES=10,11,12,13
  PU=pos  MIL=m26 run o_view_a FEATVIEW=11:0.6 SEED=611 LEAVES=31
  PU=pos  MIL=m26 run o_view_b FEATVIEW=23:0.5 SEED=623 LEAVES=63 LR=0.025
  PU=drop MIL=m26 run o_view_c FEATVIEW=37:0.65 SEED=637 LEAVES=31; }
sB() { CORES=14,15,2,3
  PU=pos  MIL=m9  run o_view_d FEATVIEW=41:0.55 SEED=641 LEAVES=31
  PU=pos  MIL=m26 run o_view_e FEATVIEW=53:0.45 BOOST=goss SEED=653 LEAVES=63 LR=0.035
  PU=drop MIL=m26 run o_view_f FEATVIEW=67:0.7 EXTRA=1 SEED=667 LEAVES=47; }
sA & sB & wait; echo ALLDONE5
