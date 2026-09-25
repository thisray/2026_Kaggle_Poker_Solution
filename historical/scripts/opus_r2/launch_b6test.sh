#!/usr/bin/env bash
# Scheduled B6 paired test for the last 2026-09-19 slot (skips if already submitted or 5 submissions that day).
W=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917; C=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates
cd $W
setsid ./sched_submit2.sh "2026-09-19 23:30:00" $C/r2f_NDw_all_on_r2j2mB6.csv "opus-r2 B6 paired test: NDw fourth-family evidence applied to all 85 other-labelled pairs + 8 mid-rank pairs with pair Bayes factor>6 promoted after the member block (vs NDw 0.92303)" < /dev/null > /dev/null 2>&1 &
disown -a; exit 0
