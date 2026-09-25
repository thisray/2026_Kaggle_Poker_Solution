#!/usr/bin/env bash
# One-shot scheduled submission: a single bounded sleep until the UTC quota reset, then submit + fetch the score.
set -uo pipefail
TARGET="$1"; F="$2"; MSG="$3"
W=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
LOG=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates/sched_submit.log
now=$(date -u +%s); tgt=$(date -u -d "$TARGET" +%s); wait=$((tgt - now))
echo "$(date -u +%FT%TZ) scheduled $F at $TARGET (sleep ${wait}s) pid $$" >> $LOG
if [ $wait -gt 0 ]; then sleep $wait; fi
$W/submit_r2.sh "$F" "$MSG" >> $LOG 2>&1
$W/poll_scores.sh >> $LOG 2>&1
echo "$(date -u +%FT%TZ) done" >> $LOG
