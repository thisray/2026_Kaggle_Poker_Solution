#!/bin/bash
# Wait for policy v2 probabilities, then build v2 surprisal and suppressed-action tensors.
W=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917; A=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
cd $W
until grep -q -E "mean surprisal v2|Traceback|Killed" m4b.log; do sleep 20; done
grep -q "mean surprisal v2" m4b.log || { echo POLICY_V2_FAILED; exit 1; }
sleep 30
TAG=v2 PROBS=dec_probs_v2.npy NUMBA_NUM_THREADS=16 taskset -c 0-15,18,19 nice -n 5 $PY e5_sur_kernel.py > e5v2.log 2>&1
NUMBA_NUM_THREADS=16 taskset -c 0-15,18,19 nice -n 5 $PY e8_kernel.py dec_probs_v2.npy v2 > e8v2.log 2>&1
ls -la $A/R2_v2.npy $A/R4_v2.npy && echo CHAIN_V2_DONE
