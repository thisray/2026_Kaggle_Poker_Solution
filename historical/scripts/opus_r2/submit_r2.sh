#!/usr/bin/env bash
# Usage: submit_r2.sh <csv path> <message>
# Submits one candidate to Kaggle with a duplicate check on the sha256 prefix (16 chars) embedded in the description.
# Auth: the Kaggle CLI's own cached OAuth login (token captured into an env var at runtime, never printed).
set -euo pipefail
F="$1"; MSG="$2"
K=/home/thisray/miniforge3/envs/kaggle_tartanimu_260827/bin/kaggle
C=detect-suspicious-value-transfers-in-poker
SHA=$(sha256sum "$F" | cut -c1-16)
LOG=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates/submit_r2.log
T=$($K auth print-access-token 2>/dev/null | tail -1); export KAGGLE_API_TOKEN="$T"; unset T
if $K competitions submissions -c $C 2>/dev/null | grep -q "$SHA"; then
  echo "$(date -u +%FT%TZ) $F already submitted (sha $SHA); skip" | tee -a $LOG; exit 0
fi
echo "$(date -u +%FT%TZ) submitting $F sha=$SHA" | tee -a $LOG
$K competitions submit -c $C -f "$F" -m "$MSG; sha256=$SHA" 2>&1 | tail -1 | tee -a $LOG
