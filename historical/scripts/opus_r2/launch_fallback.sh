#!/usr/bin/env bash
# Launch the remaining fallback submissions detached (setsid, no inherited fds).
W=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917; C=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates
cd $W
setsid ./sched_submit2.sh "2026-09-19 01:01:00" $C/r2n_NDw_on_r2j2m.csv "opus-r2 NDw (fallback): r2j2m base + fourth-family evidence = first 5 hands with an active partner-card decision AND the pair won the pot" < /dev/null > /dev/null 2>&1 &
setsid ./sched_submit2.sh "2026-09-19 01:02:00" $C/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv "opus-r2 r2j2m base (fallback): LGB+Cat P ensemble + 77 fourth-family pairs monotone-inserted + r15 evidence + c-first fourth-family evidence" < /dev/null > /dev/null 2>&1 &
disown -a
exit 0
