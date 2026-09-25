#!/usr/bin/env bash
# Build the Opus R2 research package (NOT in git).  Usage: build_package.sh <name>   e.g. Poker_OpusR2_Research_20260919
set -euo pipefail
NAME="$1"
SCR=/private/tmp/claude-501/-Users-vegapunk-Vega-Code-260916-Kaggle-Poker/665d9cb6-a737-4556-97e9-1f2d86c8c2b9/scratchpad/pkg
WT=/Volumes/SKC3000D2048G/VegaExternal/Projects/260916_Kaggle_Poker/worktrees/opus-r2-20260918
OLD=/Users/vegapunk/Downloads/Poker_OpusR2_Research_20260918.zip
ART=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917
STAGE="$SCR/$NAME"; rm -rf "$STAGE"; mkdir -p "$STAGE"/{docs,scripts,artifacts/r2_candidates,artifacts/tables}
# docs + handoff
cp "$WT"/docs/*.md "$STAGE/docs/"
cp "$WT/docs/31_opus_r2_handoff_20260918.md" "$STAGE/README_STATE.md"
[ -f "$SCR/EXTERNAL_WORKER_BRIEF.md" ] && cp "$SCR/EXTERNAL_WORKER_BRIEF.md" "$STAGE/"
# scripts: current opus_r2 + reference folders from the previous package
cp -R "$WT/scripts/opus_r2" "$STAGE/scripts/"
TMPX="$SCR/_oldx"; rm -rf "$TMPX"; mkdir -p "$TMPX"; unzip -q "$OLD" -d "$TMPX"
for d in opus_r1_reference round11_reference round15_reference; do cp -R "$TMPX"/Poker_OpusR2_Research_20260918/scripts/$d "$STAGE/scripts/" 2>/dev/null || true; done
rm -rf "$TMPX"
# candidates (LB-verified + current probe set) and receipts
CANDS="r2c_p2top600_other.csv r2c_p2top600_other_ev.csv r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv r2n_NDw_on_r2j2m.csv r2n_ND_on_r2j2m.csv r2n_NDdevpaw_on_r2j2m.csv r2x_post2_on_r2j2m.csv r2x_post2_on_r2j2mB6.csv r2f_NDw_all_on_r2j2mB6.csv r2j2mB6_lgbcat2_p2comb_other_ev_on_r15.csv r2n_NDdevAw_on_r2j2m.csv r2n_NDpaw_on_r2j2m.csv r2n_NDacw_on_r2j2m.csv r2n_NDdevFw_on_r2j2m.csv r2m_M3_both_on_r2j2m.csv r2n_NDdev_on_r2j2m.csv r2n_NDf_on_r2j2m.csv"
for f in $CANDS; do scp -q gb10:$ART/r2_candidates/$f "$STAGE/artifacts/r2_candidates/"; done
for f in $(ssh gb10 "ls $ART/r2_candidates/ | grep -E 'receipt|\.log$'"); do scp -q gb10:$ART/r2_candidates/$f "$STAGE/artifacts/r2_candidates/"; done
# key tables
TABS="s94_unified_table_v6.csv s95_posterior.json c21_hand_tables.parquet s23_infoshare_eval.parquet s38_combined_eval.parquet s76_responder_eval.parquet s77_eval_comb3.parquet s84_pair_bf_eval.parquet s84_pair_bf_dev.parquet s85_eval_bf.parquet s73_f4_mech2.parquet s83_hand_tilt_posterior_p.parquet c11_hand_q.parquet c12_hand_qdev.parquet c14_hand_tables_ext.parquet s66_dev_outcomes.parquet s71_f4_hands.parquet s75_f4_hypothesis_table.csv"
for f in $TABS; do scp -q gb10:$ART/$f "$STAGE/artifacts/tables/" || echo "missing $f"; done
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
OUTZ="/Users/vegapunk/Downloads/$NAME.zip"; rm -f "$OUTZ"
( cd "$SCR" && zip -q -r -X "$OUTZ" "$NAME" )
unzip -tq "$OUTZ" > /dev/null && echo "unzip test PASS"
echo "files: $(unzip -l "$OUTZ" | tail -1 | awk '{print $2}')  size: $(du -h "$OUTZ" | cut -f1)  sha256: $(shasum -a 256 "$OUTZ" | cut -d' ' -f1)"
rm -rf "$STAGE"
