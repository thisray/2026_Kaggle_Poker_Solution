#!/usr/bin/env bash
# Final research package for BOTH 2026-09-20 lanes (R3 structural/P work + R4 evidence/augmentation work).
# NOT in git. Usage: build_package_final.sh <name>      e.g. Poker_Final_20260920
set -euo pipefail
NAME="$1"
SCR=/private/tmp/claude-501/-Users-vegapunk-Vega-Code-260916-Kaggle-Poker/665d9cb6-a737-4556-97e9-1f2d86c8c2b9/scratchpad/pkg
WT=/Volumes/SKC3000D2048G/VegaExternal/Projects/260916_Kaggle_Poker/worktrees/opus-r2-20260918
PREV=/Users/vegapunk/Downloads/Poker_OpusR3_PFusion_20260920.zip
ART=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917
W1=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917
W4=/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920
STAGE="$SCR/$NAME"; rm -rf "$STAGE"; mkdir -p "$STAGE"/{docs,scripts,artifacts/candidates,artifacts/tables,artifacts/r3,artifacts/r4,artifacts/logs}
# --- documents: everything, plus two entry points ---
cp "$WT"/docs/*.md "$STAGE/docs/"
cp "$WT/docs/35_external_brief_r3_20260919.md" "$STAGE/README_FIRST_external_brief.md"
cp "$WT/docs/41_opus_r4_handoff_20260920.md" "$STAGE/README_SECOND_handoff.md"
cp "$WT/docs/34_opus_r3_top1_push_20260919.md" "$STAGE/RESEARCH_LOG_R3.md"
cp "$WT/docs/40_opus_r4_research_20260920.md" "$STAGE/RESEARCH_LOG_R4.md"
cp "$WT/docs/kaggle_submit_queue.md" "$STAGE/SUBMISSION_QUEUE.md"
cp "$WT/docs/experiment_ledger.md" "$STAGE/EXPERIMENT_LEDGER.md"
cp "$WT/docs/PACKAGE_INDEX.md" "$STAGE/PACKAGE_INDEX.md"
# --- scripts from the repo, plus the two worker dirs' live copies ---
for d in opus_r2 opus_r3 opus_r4 r18_chatgpt r19_chatgpt; do [ -d "$WT/scripts/$d" ] && cp -R "$WT/scripts/$d" "$STAGE/scripts/"; done
mkdir -p "$STAGE/scripts/worker_opus_r1_live" "$STAGE/scripts/worker_opus_r4_live"
scp -q "gb10:$W1/*.py" "$STAGE/scripts/worker_opus_r1_live/" 2>/dev/null || true
scp -q "gb10:$W4/*.py" "$STAGE/scripts/worker_opus_r4_live/" 2>/dev/null || true
# --- carry forward the reference script sets from the previous package ---
if [ -f "$PREV" ]; then
  TMPX="$SCR/_oldx"; rm -rf "$TMPX"; mkdir -p "$TMPX"; unzip -q "$PREV" -d "$TMPX"
  for d in opus_r1_reference round11_reference round15_reference; do
    src=$(find "$TMPX" -maxdepth 3 -type d -name "$d" | head -1)
    [ -n "$src" ] && cp -R "$src" "$STAGE/scripts/" || true
  done
  rm -rf "$TMPX"
fi
# --- candidates: every file that is a real submission option, with its receipt ---
for f in r2n_NDw_on_r2j2m.csv r3_probe_NDw_noE.csv r9_subh.csv r9_subp.csv r10_ci.csv r11_ci_rank.csv \
         r12_ndwrank_ci_f4.csv r13_ndwrank_cinew_f4.csv r19_fuseALL_cinew.csv r20_fuseALL_oldci.csv; do
  scp -q "gb10:$ART/r2_candidates/$f" "$STAGE/artifacts/candidates/" || echo "missing $f"
done
for f in r25_aug5w07_old64_dtens_cistack.csv r24_aug5w07_dtens_cistack.csv r21_grpfuse_dtens_cistack.csv r19_r13_dtens_cistack.csv; do
  scp -q "gb10:$ART/r4/cand/$f" "$STAGE/artifacts/candidates/" || echo "missing $f"
  scp -q "gb10:$ART/r4/cand/${f%.csv}.receipt.json" "$STAGE/artifacts/candidates/" || true
done
scp -q "gb10:$ART/r4/patch_r4_dt_ens3_w035.csv" "gb10:$ART/r4/patch_r4_ci_stack_b3.csv" "gb10:$ART/r18/main/patch_r18new_hard.csv" "$STAGE/artifacts/candidates/" || true
# --- measurement outputs: this round's json/parquet from both lanes ---
scp -q "gb10:$ART/r3/t[3-9]*.json" "gb10:$ART/r3/t[3-9]*.parquet" "$STAGE/artifacts/r3/" 2>/dev/null || true
scp -q "gb10:$ART/r4/*.json" "$STAGE/artifacts/r4/" 2>/dev/null || true
scp -q "gb10:$ART/r4/g1_evidence_rank.parquet" "gb10:$ART/r4/g2_sublists.parquet" "gb10:$ART/r4/g3_patterns.parquet" "$STAGE/artifacts/r4/" 2>/dev/null || true
for f in t5_dev_seq.parquet t4_wrong_vs_hit.parquet player_local_v1.parquet; do scp -q "gb10:$ART/$f" "$STAGE/artifacts/tables/" || echo "missing $f"; done
scp -q "gb10:$W1/logs_r3/t[7-9]*.log" "$STAGE/artifacts/logs/" 2>/dev/null || true
scp -q "gb10:$W4/logs_q3/*.log" "gb10:$W4/logs/q2_*.log" "$STAGE/artifacts/logs/" 2>/dev/null || true
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
OUTZ="/Users/vegapunk/Downloads/$NAME.zip"; rm -f "$OUTZ"
( cd "$SCR" && zip -q -r -X "$OUTZ" "$NAME" )
unzip -tq "$OUTZ" > /dev/null && echo "unzip test PASS"
echo "path: $OUTZ"
echo "files: $(unzip -l "$OUTZ" | tail -1 | awk '{print $2}')  size: $(du -h "$OUTZ" | cut -f1)  sha256: $(shasum -a 256 "$OUTZ" | cut -d' ' -f1)"
rm -rf "$STAGE"
