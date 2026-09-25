#!/usr/bin/env bash
# Build the Opus R4 research package (NOT in git).  Usage: build_package_r4.sh <name>   e.g. Poker_OpusR4_Handoff_20260920
set -euo pipefail
NAME="$1"
SCR="${SCRATCH:-/private/tmp/claude-501/-Users-vegapunk-Vega-Code-260916-Kaggle-Poker/49e9fd1b-a4c3-4342-98b5-5ef806a4e08d/scratchpad}/pkg"
WT=/Volumes/SKC3000D2048G/VegaExternal/Projects/260916_Kaggle_Poker/worktrees/opus-r4-20260920
ART=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917
STAGE="$SCR/$NAME"; rm -rf "$STAGE"; mkdir -p "$STAGE"/{docs,scripts,receipts,artifacts/r4/cand,artifacts/r4/results,artifacts/r2_candidates}
cp "$WT"/docs/*.md "$STAGE/docs/"; cp "$WT/docs/41_opus_r4_handoff_20260920.md" "$STAGE/README_STATE.md"; cp "$WT/README.md" "$STAGE/README_REPO.md"; cp "$WT/AGENTS.md" "$STAGE/" 2>/dev/null || true
for d in opus_r4 opus_r3 opus_r2 opus_r1 r18_chatgpt r19_chatgpt round20; do [ -d "$WT/scripts/$d" ] && cp -R "$WT/scripts/$d" "$STAGE/scripts/"; done
cp -R "$WT/receipts/." "$STAGE/receipts/" 2>/dev/null || true
# R4 results: every json / receipt / patch, plus the two recommended candidates and the LB-verified reference
ssh gb10 "cd $ART/r4 && ls *.json patch_r4_*.csv patch_r4_*.receipt.json cand/*.receipt.json 2>/dev/null" | while read f; do scp -q "gb10:$ART/r4/$f" "$STAGE/artifacts/r4/results/$(basename $f)" </dev/null; done
for f in ${CANDS:-r29_aug5_typedDT_cistack_f4hybrid03 r26_aug5_typedDT_cistack r25_aug5w07_old64_dtens_cistack}; do scp -q "gb10:$ART/r4/cand/$f.csv" "gb10:$ART/r4/cand/$f.receipt.json" "$STAGE/artifacts/r4/cand/"; done
for f in r10_ci.csv r10_ci.receipt.json r19_fuseALL_cinew.receipt.json r13_ndwrank_cinew_f4.csv.receipt.json r14_fused_rank.receipt.json r15_grpfuse_rank.receipt.json r17_fuse48_cinew.receipt.json r12_ndwrank_ci_f4.receipt.json; do scp -q "gb10:$ART/r2_candidates/$f" "$STAGE/artifacts/r2_candidates/" 2>/dev/null || echo "missing $f"; done
ssh gb10 "cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920/logs && tar czf - *.log" | tar xzf - -C "$STAGE/artifacts/r4/results/" 2>/dev/null || true
cat > "$STAGE/README.md" <<'MD'
# Poker Opus R4 handoff package (2026-09-20)

Start with `README_STATE.md` (= docs/41). Then `docs/40_opus_r4_research_20260920.md` (this round, full numbers), `docs/34_opus_r3_top1_push_20260919.md` (concurrent session, sections 33+), `docs/35_external_brief_r3_20260919.md` (problem and mechanisms), `docs/kaggle_submit_queue.md` (candidates), `docs/experiment_ledger.md`.
- `scripts/opus_r4/` this round (u*: structure probes, v*: transductive, w*: exclusivity, x*: event model, y*: eval deployment, z*: fourth family, p*/q*: pair model, `codex_review_r4/`: independent review and tests).
- `artifacts/r4/results/`: result JSONs, patch CSVs with receipts, run logs. `artifacts/r4/cand/`: recommended candidates (NOT submitted). `artifacts/r2_candidates/`: LB-verified `r10_ci.csv` and receipts of the bases.
- Not included: raw competition data, numpy feature arrays, model weights, credentials.
- Submission rule: no Kaggle submission without the user's explicit per-file approval.
MD
( cd "$STAGE" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS.txt )
OUTZ="/Users/vegapunk/Downloads/$NAME.zip"; rm -f "$OUTZ"; ( cd "$SCR" && zip -q -r -X "$OUTZ" "$NAME" )
unzip -tq "$OUTZ" > /dev/null && echo "unzip test PASS"
echo "files: $(unzip -l "$OUTZ" | tail -1 | awk '{print $2}')  size: $(du -h "$OUTZ" | cut -f1)  sha256: $(shasum -a 256 "$OUTZ" | cut -d' ' -f1)"
rm -rf "$STAGE"
