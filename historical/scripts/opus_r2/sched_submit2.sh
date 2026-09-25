#!/usr/bin/env bash
# Fallback one-shot scheduled submission: bounded sleep until TARGET (UTC), then submit only if (a) this file was not already
# submitted (sha check inside submit_r2.sh) and (b) fewer than 5 submissions were made on the TARGET's UTC day.
set -uo pipefail
TARGET="$1"; F="$2"; MSG="$3"
W=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
LOG=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates/sched_submit.log
K=/home/thisray/miniforge3/envs/kaggle_tartanimu_260827/bin/kaggle; C=detect-suspicious-value-transfers-in-poker
now=$(date -u +%s); tgt=$(date -u -d "$TARGET" +%s); wait=$((tgt - now))
echo "$(date -u +%FT%TZ) fallback scheduled $F at $TARGET (sleep ${wait}s) pid $$" >> $LOG
if [ $wait -gt 0 ]; then sleep $wait; fi
DAY=$(date -u -d "$TARGET" +%F)
T=$($K auth print-access-token 2>/dev/null | tail -1); export KAGGLE_API_TOKEN="$T"; unset T
N=$($K competitions submissions -c $C 2>/dev/null | grep -c " $DAY ")
if [ "$N" -ge 5 ]; then echo "$(date -u +%FT%TZ) skip $F: already $N submissions on $DAY" >> $LOG; exit 0; fi
$W/submit_r2.sh "$F" "$MSG" >> $LOG 2>&1
echo "$(date -u +%FT%TZ) fallback done ($N earlier submissions on $DAY)" >> $LOG
