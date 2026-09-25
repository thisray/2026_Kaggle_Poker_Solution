#!/usr/bin/env bash
# Build the Opus R5 research package (NOT in git).  Usage: build_package_r5.sh <name>
set -euo pipefail
NAME="$1"
SCR="${SCRATCH:-/private/tmp/claude-501/-Users-vegapunk-Vega-Code-260916-Kaggle-Poker/582ac8be-e9eb-408a-aaee-91f7e097f178/scratchpad}/pkg"
WT="${WT:-/Users/vegapunk/Vega/Code/260916_Kaggle_Poker}"
ART=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917
STAGE="$SCR/$NAME"; rm -rf "$STAGE"; mkdir -p "$STAGE"/{docs,scripts,receipts,artifacts/r5/cand,artifacts/r5/results,artifacts/r4/cand,artifacts/r4/results,artifacts/r2_candidates}
cp "$WT"/docs/*.md "$STAGE/docs/"
cp "$WT/docs/43_opus_r5_handoff_20260920.md" "$STAGE/README_STATE.md"
cp "$WT/README.md" "$STAGE/README_REPO.md" 2>/dev/null || true
cp "$WT/AGENTS.md" "$STAGE/" 2>/dev/null || true
for d in opus_r5 opus_r4 opus_r3 opus_r2 opus_r1 r18_chatgpt r19_chatgpt round20; do [ -d "$WT/scripts/$d" ] && cp -R "$WT/scripts/$d" "$STAGE/scripts/"; done
cp -R "$WT/receipts/." "$STAGE/receipts/" 2>/dev/null || true
cp -R "$WT/src" "$STAGE/" 2>/dev/null || true
cp -R "$WT/tests" "$STAGE/" 2>/dev/null || true
# R5 results
ssh gb10 "cd $ART/r5 && ls *.json *.csv 2>/dev/null" | while read f; do scp -q "gb10:$ART/r5/$f" "$STAGE/artifacts/r5/results/$(basename "$f")" </dev/null; done
for f in ${R5CANDS:-r30_r29_spzoo r31_r26_spzoo}; do scp -q "gb10:$ART/r5/cand/$f.csv" "gb10:$ART/r5/cand/$f.receipt.json" "$STAGE/artifacts/r5/cand/" </dev/null; done
# R4 results and the two R4 candidates that R5 builds on
ssh gb10 "cd $ART/r4 && ls *.json patch_r4_*.csv 2>/dev/null" | while read f; do scp -q "gb10:$ART/r4/$f" "$STAGE/artifacts/r4/results/$(basename "$f")" </dev/null; done
for f in ${R4CANDS:-r29_aug5_typedDT_cistack_f4hybrid03 r26_aug5_typedDT_cistack}; do scp -q "gb10:$ART/r4/cand/$f.csv" "gb10:$ART/r4/cand/$f.receipt.json" "$STAGE/artifacts/r4/cand/" </dev/null; done
# LB-verified reference and the empty-evidence probe that fixes the score algebra
for f in r10_ci.csv r10_ci.receipt.json r2n_NDw_on_r2j2m.csv r3_probe_NDw_noE.csv r13_ndwrank_cinew_f4.csv; do scp -q "gb10:$ART/r2_candidates/$f" "$STAGE/artifacts/r2_candidates/" </dev/null 2>/dev/null || echo "missing $f"; done
# run logs
ssh gb10 "cd /home/thisray/projects/260916_Kaggle_Poker_workers/opus-r5-20260920/logs && tar czf - *.log" | tar xzf - -C "$STAGE/artifacts/r5/results/" 2>/dev/null || true
cat > "$STAGE/README.md" <<'MD'
# Poker — Opus R5 research package (2026-09-20, deadline day)

**Read `README_STATE.md` first** (= `docs/43_opus_r5_handoff_20260920.md`). It is standalone:
competition and metric, the verified score algebra, where the remaining points are, what has
been closed, the candidates and their checks, and the next experiments.

Then, in order of usefulness:
- `docs/42_opus_r5_research_20260920.md` — this round, every number with its script.
- `docs/41_opus_r4_handoff_20260920.md` — R4 (the two-type evidence-listing discovery).
- `docs/34_opus_r3_top1_push_20260919.md` §33+ and `docs/35_external_brief_r3_20260919.md` — the concurrent R3 session and the mechanism overview.
- `docs/kaggle_submit_queue.md` — candidate ledger. `docs/experiment_ledger.md` — experiment table.

Code: `scripts/opus_r5/` (d*: budget/ceiling/calibration/exposure/behaviour/routing, e*: event-model
zoo and fusion, f*: fourth-family tau and transfer, g6z: eval deployment, y5r5: candidate builder),
`scripts/opus_r4/` (g*: typed decoder, z*: fourth family, q*: pair augmentation), `scripts/opus_r1..r3/`,
`src/pokerlab/metrics.py` (independent implementation of the official metric).

Artifacts: `artifacts/r5/cand/` the two R5 candidates (**not submitted**), `artifacts/r5/results/`
result JSONs, the soft_play patch and run logs, `artifacts/r4/` the R4 bases and patches,
`artifacts/r2_candidates/` the LB-verified `r10_ci.csv` (public 0.92501), the NDw base and the
empty-evidence probe `r3_probe_NDw_noE.csv` that pins `0.7P + 0.1B = 0.78700`.

Not included: raw competition data, numpy feature arrays, model weights, credentials.

**Submission rule: no Kaggle submission without the user's explicit per-file approval in chat.**
MD
( cd "$STAGE" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS.txt )
OUTZ="/Users/vegapunk/Downloads/$NAME.zip"; rm -f "$OUTZ"; ( cd "$SCR" && zip -q -r -X "$OUTZ" "$NAME" )
unzip -tq "$OUTZ" > /dev/null && echo "unzip test PASS"
echo "files: $(unzip -l "$OUTZ" | tail -1 | awk '{print $2}')  size: $(du -h "$OUTZ" | cut -f1)  sha256: $(shasum -a 256 "$OUTZ" | cut -d' ' -f1)"
rm -rf "$STAGE"
