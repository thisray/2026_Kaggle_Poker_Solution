#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
run() { TAG=$1; shift; env "$@" THREADS=4 taskset -c $CORES nice -n 12 $PY t69_pair_variant.py $PU $MIL "$TAG" >> logs_r3/t69_$TAG.log 2>&1; }
sA() { CORES=10,11,12,13
  PU=pos  MIL=m26 run o_goss_b  BOOST=goss SEED=151 LEAVES=63 LR=0.04
  PU=pos  MIL=m26 run o_extra_c EXTRA=1 SEED=163 LEAVES=63 FF=0.4
  PU=drop MIL=m26 run o_extra_d EXTRA=1 SEED=211 LEAVES=31 FF=0.6
}
sB() { CORES=14,15,2,3
  PU=drop MIL=m26 run o_goss_e  BOOST=goss SEED=307 LEAVES=31 LR=0.03
  PU=pos  MIL=m26 run o_dart_a  BOOST=dart SEED=401 LEAVES=31 ROUNDS=400
  PU=drop MIL=m9  run o_extra_f EXTRA=1 SEED=419 LEAVES=47 FF=0.5
}
sA & sB & wait; echo ALLDONE3
