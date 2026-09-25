#!/usr/bin/env bash
set -euo pipefail
PY=/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python
OPUS=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917
DST=/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917
R8=/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917
W=/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917
export OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
cd $W
for i in $(seq 0 9); do
  OFF=$((i*11254)); LIM=11254
  if [ $i -eq 9 ]; then LIM=0; fi
  if [ ! -f $R8/eval_scores_$i.csv ]; then
    nice -n 10 $PY code/prepare_pack.py --candidates $R8/eval_candidates.csv --np-dir $OPUS/np --out $R8/eval_pack_$i --pair-offset $OFF --pair-limit $LIM --exact
    nice -n 10 $PY code/train_moments_plus_deploy.py predict --pack $R8/eval_pack_$i --out $R8/eval_scores_$i.csv --model-dir $R8/moments_plus_full
    rm -rf $R8/eval_pack_$i
  fi
done
head -1 $R8/eval_scores_0.csv > $R8/eval_scores_all.csv
for i in $(seq 0 9); do tail -n +2 $R8/eval_scores_$i.csv >> $R8/eval_scores_all.csv; done
wc -l $R8/eval_scores_all.csv
nice -n 10 $PY code/assemble_evidence.py --base $DST/r6_submission.csv --eval-pairs /home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw/evaluation_pairs.csv --candidates $R8/eval_candidates.csv --scores $R8/eval_scores_all.csv --np-dir $OPUS/np --out $R8/r7_submission.csv
echo EVAL_COMBO_DONE
