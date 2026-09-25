#!/usr/bin/env bash
# Poll until the two newest submissions are scored (or 20 minutes pass); print the top rows.
K=/home/thisray/miniforge3/envs/kaggle_tartanimu_260827/bin/kaggle
T=$($K auth print-access-token 2>/dev/null | tail -1); export KAGGLE_API_TOKEN="$T"; unset T
for i in $(seq 1 40); do
  out=$($K competitions submissions -c detect-suspicious-value-transfers-in-poker 2>/dev/null | sed -n 3,4p)
  if ! echo "$out" | grep -q -E "PENDING|RUNNING"; then echo "$out" | awk '{print $1, $2, $(NF-1), $NF}'; exit 0; fi
  sleep 30
done
echo "timeout"; echo "$out" | awk '{print $1, $2, $(NF-1), $NF}'
