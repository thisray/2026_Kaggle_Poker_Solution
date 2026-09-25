#!/usr/bin/env bash
# Build the Opus R3 research package (NOT in git).  Usage: build_package_r3.sh <name>   e.g. Poker_OpusR3_Research_20260919
set -euo pipefail
NAME="$1"
SCR=/private/tmp/claude-501/-Users-vegapunk-Vega-Code-260916-Kaggle-Poker/665d9cb6-a737-4556-97e9-1f2d86c8c2b9/scratchpad/pkg
WT=/Volumes/SKC3000D2048G/VegaExternal/Projects/260916_Kaggle_Poker/worktrees/opus-r2-20260918
PREV=/Users/vegapunk/Downloads/Poker_OpusR2_Research_20260919.zip
ART=/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917
STAGE="$SCR/$NAME"; rm -rf "$STAGE"; mkdir -p "$STAGE"/{docs,scripts,artifacts/r2_candidates,artifacts/tables,artifacts/r3}
cp "$WT"/docs/*.md "$STAGE/docs/"
cp "$WT/docs/35_external_brief_r3_20260919.md" "$STAGE/README_FIRST_EXTERNAL_BRIEF.md"
cp "$WT/docs/34_opus_r3_top1_push_20260919.md" "$STAGE/R3_RESEARCH_LOG.md"
cp -R "$WT/scripts/opus_r2" "$WT/scripts/opus_r3" "$WT/scripts/r18_chatgpt" "$WT/scripts/r19_chatgpt" "$STAGE/scripts/"
TMPX="$SCR/_oldx"; rm -rf "$TMPX"; mkdir -p "$TMPX"; unzip -q "$PREV" -d "$TMPX"
for d in opus_r1_reference round11_reference round15_reference; do cp -R "$TMPX"/Poker_OpusR2_Research_20260919/scripts/$d "$STAGE/scripts/" 2>/dev/null || true; done
rm -rf "$TMPX"
CANDS="r2n_NDw_on_r2j2m.csv r3_probe_NDw_noE.csv r9_subh.csv r9_subp.csv r10_ci.csv r11_ci_rank.csv r12_ndwrank_ci_f4.csv r13_ndwrank_cinew_f4.csv r17_fuse48_cinew.csv r18_fuse48_oldci.csv r19_fuseALL_cinew.csv r20_fuseALL_oldci.csv"
for f in $CANDS; do scp -q gb10:$ART/r2_candidates/$f "$STAGE/artifacts/r2_candidates/"; done
for f in $(ssh -n gb10 "ls $ART/r2_candidates/ | grep -E '^r3_.*receipt|receipt.*B6|^r[5-9].*receipt'"); do scp -q gb10:$ART/r2_candidates/$f "$STAGE/artifacts/r2_candidates/"; done
R3F="t26_placebo.parquet t52_dev_feats.parquet t53_cond_auc.json t54_minimal_rerank.json t56_single_feature.json t57b_p_rerank.json t59_family_event_new.json t60_ci_new_dev.json t62_ci_new_patch.json t63_ci_new_sanity.json t64_ci_bag.json t65_slate_exchange.json t67_pair_fusion.json t68_fusion_vs_deployed.json t70_fusion_select.json t73_stack_calib.json t75_missed_profile.json t27_placebo_ext.parquet t28_placebo_null.parquet t33_mid_posterior.parquet t17a_bf_eval.parquet t17_bf_sub_eval.parquet t17_bf_sub_devsub11.parquet t17_hands_eval.parquet t34_re_pairs.parquet t34_re_fit.json t35_gap_gate.json t36_member_table.parquet t37_re2d.json t37_re2d_pairs.parquet t37_pairllr_r9.parquet t37_pairllr_d91rv1.parquet t40_v1_ratio.parquet t41_mid_clean.parquet t42_placebo_clean.parquet t43_member_devphase.parquet t44_known_e_memo.parquet t45_known_e_rerank_pairs.parquet t20_ci_patch_main.parquet t20_ci_patch_plus.parquet b6_promoted.txt r3_new8_promote.txt r3_b7_promote.txt"
for f in $R3F; do scp -q gb10:$ART/r3/$f "$STAGE/artifacts/r3/" || echo "missing $f"; done
TABS="s85_eval_bf.parquet s66_dev_outcomes.parquet t5_dev_seq.parquet t4_wrong_vs_hit.parquet c14_hand_tables_ext.parquet s38_combined_eval.parquet"
for f in $TABS; do scp -q gb10:$ART/$f "$STAGE/artifacts/tables/" || echo "missing $f"; done
mkdir -p "$STAGE/artifacts/logs_r3"; scp -q "gb10:/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917/logs_r3/t3[4-9]*.log" "gb10:/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917/logs_r3/t4*.log" "$STAGE/artifacts/logs_r3/" || echo "missing logs"
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )
OUTZ="/Users/vegapunk/Downloads/$NAME.zip"; rm -f "$OUTZ"
( cd "$SCR" && zip -q -r -X "$OUTZ" "$NAME" )
unzip -tq "$OUTZ" > /dev/null && echo "unzip test PASS"
echo "files: $(unzip -l "$OUTZ" | tail -1 | awk '{print $2}')  size: $(du -h "$OUTZ" | cut -f1)  sha256: $(shasum -a 256 "$OUTZ" | cut -d' ' -f1)"
rm -rf "$STAGE"
