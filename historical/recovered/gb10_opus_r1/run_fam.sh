#!/usr/bin/env bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
run() { TAG=$1; shift; env "$@" THREADS=4 taskset -c $CORES nice -n 12 $PY t69_pair_variant.py $PU $MIL "$TAG" >> logs_r3/t69_$TAG.log 2>&1; }
sA() { CORES=10,11,12,13
  PU=pos MIL=m26 run o_fam_dt  FAMTARGET=directed_transfer SEED=801 LEAVES=31
  PU=pos MIL=m26 run o_fam_sp  FAMTARGET=soft_play SEED=811 LEAVES=31
  PU=pos MIL=m26 run o_fam_ci  FAMTARGET=coordinated_isolation SEED=821 LEAVES=31; }
sB() { CORES=14,15,2,3
  PU=drop MIL=m26 run o_fam_dt2 FAMTARGET=directed_transfer SEED=833 LEAVES=63 LR=0.025
  PU=drop MIL=m26 run o_fam_sp2 FAMTARGET=soft_play SEED=841 LEAVES=63 LR=0.025
  PU=drop MIL=m9  run o_fam_dt3 FAMTARGET=directed_transfer SEED=853 LEAVES=31 FF=0.6; }
sA & sB & wait; echo FAMDONE
