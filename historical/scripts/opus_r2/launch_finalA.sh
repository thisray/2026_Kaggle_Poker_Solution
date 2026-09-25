#!/usr/bin/env bash
# Safety net for the final day: submit final candidate A (POST2 evidence + B6) at 2026-09-20 12:00 UTC unless already submitted.
W=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917; C=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates
cd $W
setsid ./sched_submit2.sh "2026-09-20 12:00:00" $C/r2x_post2_on_r2j2mB6.csv "opus-r2 final A: r2j2m base + B6 (8 mid-rank BF>6 pairs promoted) + fourth-family evidence = posterior-weighted mixture over labeller hypotheses (POST2; NDw-family)" < /dev/null > /dev/null 2>&1 &
disown -a; exit 0
