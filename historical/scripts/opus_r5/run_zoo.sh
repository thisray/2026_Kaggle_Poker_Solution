#!/bin/bash
cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r5-20260920
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
CFGS="lgb_base lgb_sym lgb_deep lgb_shal lgb_goss lgb_dart lgb_extra lgb_col3 lgb_kern lgb_gplay lgb_role lgb_l2 lgb_bag cat"
run_fam () {
  fam=$1; cores=$2
  for cfg in $CFGS; do
    FAM=$fam CFG=$cfg NJ=4 nice -n 15 taskset -c $cores $PY e1_zoo.py >> logs/zoo_${fam}.log 2>&1
  done
  echo "DONE $fam" >> logs/zoo_${fam}.log
}
run_fam directed_transfer 12-15 &
run_fam soft_play 16-19 &
run_fam coordinated_isolation 8-11 &
wait
echo ALLDONE
