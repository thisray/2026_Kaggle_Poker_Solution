#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
run() { TAG=$1; shift; env "$@" THREADS=4 taskset -c $CORES nice -n 12 $PY t69_pair_variant.py $PU $MIL "$TAG" >> logs_r3/t69_$TAG.log 2>&1; }
streamA() { CORES=10,11,12,13
  PU=pos  MIL=m26  run o_pos_a  SEED=11  LEAVES=31 FF=0.5  LR=0.03
  PU=pos  MIL=m26  run o_pos_b  SEED=23  LEAVES=63 FF=0.35 LR=0.02 ROUNDS=1200
  PU=drop MIL=m26  run o_drop_c SEED=101 LEAVES=63 FF=0.35 LR=0.02 ROUNDS=1200
  PU=drop MIL=m26  run o_drop_d SEED=202 LEAVES=15 FF=0.7  LR=0.04 MINLEAF=80
}
streamB() { CORES=14,15,2,3
  PU=none MIL=m26  run o_none_e SEED=303 LEAVES=31 FF=0.5  LR=0.03
  PU=drop MIL=m9   run o_drop_f SEED=404 LEAVES=31 FF=0.5  LR=0.03
  PU=drop MIL=m26  run o_drop_g SEED=505 LEAVES=127 MINLEAF=20 FF=0.3 L2=10 LR=0.02 ROUNDS=1500
  PU=pos  MIL=m26  run o_pos_h  SEED=606 LEAVES=15 FF=0.6  MINLEAF=60 LR=0.03
}
mkdir -p logs_r3; streamA & streamB & wait; echo ALLDONE
