#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
run() { TAG=$1; shift; env "$@" THREADS=4 taskset -c $CORES nice -n 12 $PY t69_pair_variant.py $PU $MIL "$TAG" >> logs_r3/t69_$TAG.log 2>&1; }
sA() { CORES=10,11,12,13
  PU=pos MIL=m26 run o_pos_i SEED=71  LEAVES=31 FF=0.45 LR=0.025 ROUNDS=1000
  PU=pos MIL=m9  run o_pos_j SEED=83  LEAVES=31 FF=0.5  LR=0.03
  PU=pos MIL=m26 run o_pos_k SEED=97  LEAVES=63 FF=0.6  LR=0.03  MINLEAF=25
}
sB() { CORES=14,15,2,3
  PU=pos MIL=m26 run o_pos_l SEED=113 LEAVES=15 FF=0.4  LR=0.035 L2=5
  PU=pos MIL=m26 run o_pos_m SEED=131 LEAVES=95 FF=0.35 LR=0.02  ROUNDS=1400 MINLEAF=30
  PU=drop MIL=m9 run o_drop_n SEED=149 LEAVES=31 FF=0.55 LR=0.03
}
sA & sB & wait; echo ALLDONE2
