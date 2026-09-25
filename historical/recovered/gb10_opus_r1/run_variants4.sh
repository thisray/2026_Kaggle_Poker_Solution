#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
sA() { SEEDS=37 LEARNER=cat taskset -c 4,5,6,7 nice -n 12 $PY m36_pair_ens.py pos m26 o_cat_pos2 > logs_r3/t69_o_cat_pos2.log 2>&1
       SEEDS=41 LEARNER=cat taskset -c 4,5,6,7 nice -n 12 $PY m36_pair_ens.py pos m9  o_cat_pos3 > logs_r3/t69_o_cat_pos3.log 2>&1; }
sB() { run() { TAG=$1; shift; env "$@" THREADS=4 taskset -c 10,11,12,13 nice -n 12 $PY t69_pair_variant.py $PU $MIL "$TAG" >> logs_r3/t69_$TAG.log 2>&1; }
       PU=pos  MIL=m26 run o_goss_g BOOST=goss SEED=523 LEAVES=31 LR=0.03
       PU=pos  MIL=m9  run o_goss_h BOOST=goss SEED=541 LEAVES=63 LR=0.035
       PU=pos  MIL=m26 run o_extra_i EXTRA=1 SEED=557 LEAVES=95 FF=0.35 LR=0.025
       PU=drop MIL=m26 run o_goss_j BOOST=goss SEED=571 LEAVES=63 LR=0.04; }
sA & sB & wait; echo ALLDONE4
