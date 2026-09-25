#!/usr/bin/env bash
# Replace the 01:00 fallbacks with batch-1 submissions right after the 2026-09-19 00:00 UTC quota reset.
W=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917; C=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates
LOG=$C/sched_submit.log
for p in 948436 950827 950828; do kill $p 2>/dev/null && echo "$(date -u +%FT%TZ) killed old fallback pid $p (rescheduled to 00:01 UTC)" >> $LOG; done
pkill -f "sleep 28576" 2>/dev/null; pkill -f "sleep 28547" 2>/dev/null; pkill -f "sleep 28607" 2>/dev/null
cd $W
setsid ./sched_submit2.sh "2026-09-19 00:01:00" $C/r2n_ND_on_r2j2m.csv "opus-r2 batch1 ND: r2j2m base (LGB+Cat P ensemble, 77 fourth-family pairs monotone-inserted, r15 evidence) + fourth-family evidence = first 5 hands with an active partner-card decision (per-decision tilt model)" < /dev/null > /dev/null 2>&1 &
setsid ./sched_submit2.sh "2026-09-19 00:01:40" $C/r2n_NDw_on_r2j2m.csv "opus-r2 batch1 NDw: r2j2m base + fourth-family evidence = first 5 hands with an active partner-card decision AND the pair won the pot" < /dev/null > /dev/null 2>&1 &
setsid ./sched_submit2.sh "2026-09-19 00:02:20" $C/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv "opus-r2 batch1 base r2j2m: LGB+Cat P ensemble + 77 fourth-family pairs monotone-inserted + r15 evidence + c-first fourth-family evidence" < /dev/null > /dev/null 2>&1 &
disown -a; exit 0
