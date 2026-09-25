#!/bin/bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r5-20260920
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
run_fam () {
  fam="$1"; cores="$2"
  for cfg in x_base x_deep x_shal x_goss x_extra x_col2 x_a1 x_bag5 x_mcs5 xgb cat2; do
    FAM=$fam CFG=$cfg VIEW=flip NJ=4 nice -n 15 taskset -c $cores $PY e1b_zoo2.py >> logs/zoo2_$fam.log 2>&1
  done
  for cfg in x_base x_deep x_goss x_extra x_col2 x_bag5; do
    FAM=$fam CFG=$cfg XFAM=1 NJ=4 nice -n 15 taskset -c $cores $PY e1b_zoo2.py >> logs/zoo2_$fam.log 2>&1
  done
  echo "DONE $fam" >> logs/zoo2_$fam.log
}
run_fam directed_transfer 12-15 &
run_fam soft_play 16-19 &
run_fam coordinated_isolation 8-11 &
wait
echo ALLDONE
