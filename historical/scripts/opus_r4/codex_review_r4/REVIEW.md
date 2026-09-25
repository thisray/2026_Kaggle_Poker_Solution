# R4 evidence-patch pipeline 獨立審查

## 結論

**Needs revision；R4 patches 目前不能視為完全符合 brief、可直接信任的產物。**

patch 的機械完整性良好：`r17_r13_dt_cistack.csv` 保留 base 的 `pair_id` 順序、`risk_score`、`predicted_behavior`，1,830 個變更列全部屬於 DT／CI，五個 evidence 無重複且都在該 pair 的 frozen top-20 candidates 中。DP、排序／index 對齊、right-censoring、`x2` 累積欄位與 raw-array 重算也都通過。

但有兩個會改變模型輸入或適用母體的已證實 bug：

1. `y1` 實際只處理 frozen R15 shortlist 與 routed pairs 的交集，而非所有 routed pairs。相對於最終 base，缺少 27,145 個 DT 與 12,594 個 CI pairs。
2. `y2` 的 receiver/sender 以整期總淨收益決定，不是指定的 BB-normalized directed flow；DT artifact 有 835 / 1,529 pairs（54.6%）方向與 `x2` 不同。

因此，即使最終 CSV 沒有破壞 base 欄位，evidence ranking 仍不是 brief 所定義的完整 R4 pipeline 結果。本次是 testing-only 審查，未修改父目錄受審程式或 artifacts。

## Findings

| 位置 | 嚴重度 | Finding 與證據 | 具體修正 |
|---|---|---|---|
| `../y1_eval_frames.py:11-14`, `:21-27` | **bug** | `sc = sc[sc.pair_id.isin(ids)]` 靜默取 routed set 與 4,000-pair R15 shortlist 的交集，後續 frame 也只由 `sc` 建立。最終 base 有 28,674 DT／13,186 CI routed pairs，但 candidates 僅 1,529 DT／592 CI；分別缺 27,145／12,594，且沒有 extra。未涵蓋者 risk score 最高仍達 DT 0.981、CI 0.979。這違反「prediction = all co-seated eval hands of routed pairs」。 | 為所有 routed pairs 產生 frozen top-20 candidates，並在寫檔前 assert `set(sc.pair_id) == ids`、每 pair 恰有 20 筆、full frame 覆蓋同一 pair set。若 shortlist 才是產品規格，則必須修改 brief／輸出命名並把 cohort 明確固定，不能宣稱 all routed pairs。 |
| `../y2_eval_newfeats.py:29-30`, `:41` | **bug** | `recv_is_T = pair_net.nT > pair_net.nS` 用未除 BB 的總淨收益定向；brief 要求的是所有共桌 hands 的雙向 chip flow（BB 單位）。全量 DT 比對有 835 / 1,529 pairs 方向不同，造成 `net_r/net_s`、`s_fold_to_r` 等 NEW features 與同一列的 `x_*` receiver 語意互相矛盾。 | 抽出一個 dev/eval 共用 orientation helper，按 `min(loss, gain) / bb` 累計 directed flow；`y2` 直接使用該 orientation。同步重建 dev `t58` 與 eval new-features，重新訓練／產生 patches，並保留目前 failing regression test。 |
| `../y2_eval_newfeats.py:14` | **risk** | `inner` merge 沒有 `validate`，會靜默丟失 frame 中找不到 evaluation-pair metadata 的列，也可能因重複 `pair_id` 放大資料。現有 DT artifact 實測無 row loss、duplicate 或 null。 | 改為 `left` merge + `validate="many_to_one"`（若 `seq.pair_id` 可重複），merge 後 assert row count、`(slot,h)` key set 與 `pa/pb` 完整，再開始特徵計算。 |
| `../y3_deploy_event.py:20-22`; `../y3b_deploy_stack.py:22-24` | **risk** | NEW feature join 缺列時直接 `fillna(0.0)`，會把 join failure 當成真實零值；只有 ROLE join 有 non-null assert，也沒有對完整 feature matrix 做 finite／duplicate-name 檢查。現有 dev/eval artifacts 的 key、row count、feature order、null、finite 均通過。 | 每次 merge 前後 assert 唯一 key、row count 與 key set；區分「來源列不存在」和「欄位本身按規格補零」。另 assert `len(FS) == len(set(FS))` 及 train/eval matrix 全 finite。 |
| `../y5_build_candidates.py:10-16` | **risk** | builder 僅過濾 routed family 並保護 risk／behavior／row order；未檢查 patch `pair_id` 唯一、五個 evidence 不重複、或 evidence 是否屬於該 pair 的 frozen top-20。指定的 `r17` artifact 實測全部通過，但任意錯誤 patch 可被接受。 | 在套用前驗證 exact schema、`pair_id` uniqueness、每列 5 個唯一 hand IDs、family 全匹配，以及 candidate membership；任何不符直接失敗，不要靜默 skip。 |
| `../x2_role_feats.py:16` | **risk** | directed-flow 完全相等時以 `>=` 固定選 T-side 為 receiver；brief 的「received more」未定義 tie。eval DT 中有 2 / 1,529 個 tie pairs（dev sample universe為 0）。 | 在規格中明訂 deterministic tie-break，並在程式與測試共用；若 tie 不應建模，則明確排除或使用無方向特徵。 |
| `../y1_eval_frames.py:16-18` | **risk** | base top-5 與 R15 top-5 的一致性只計算並列印比例，不會因不一致而停止。這會允許 blend partner 與實際 deployed ranking 不同。現有受測 candidates 每 pair 都恰有 20 筆，最終 patched evidence membership 通過。 | 對完整預期 pair set assert top-5 set（以及規格若要求時的順序）一致；比例僅留作 receipt，不應取代 gate。 |
| `../y3b_deploy_stack.py:3-6` | **nit** | docstring 仍描述 rank blend、參數 `W` 與 `y3_deploy_event.py` 用法，和實際 BETA logit stack 不一致，容易造成錯誤執行／receipt 解讀。 | 更新 usage、參數名稱與說明為 `BETA`／`y3b_deploy_stack.py`，移除 inherited rank-blend 描述。 |

## 已驗證行為

- `x2_role_feats.py` 的主要 orientation 本身使用全 phase 共桌 hands、BB-normalized directed flow，eval features 未讀取 pair target label。
- 20 個固定 seed 隨機 DT pairs 的 `x_k`、`x_rel`、`x_k_cell`、`x_k_bigR`、`x_n_cell` 逐列 loop 重算一致。
- 10 個固定 seed 隨機 DT pairs 的 orientation、`netR/netS/conR/conS/x_dir/x_s_fold_to_r` 由 raw arrays 與 action loop 獨立重算一致。
- 5 個 dev pairs 使用從 `x2_role_feats.py` verbatim 複製的 row loop 重建全部輸出欄位，與 `x2_role_dev.parquet` 一致。
- `first_k_marginal` 對 `n=1..10` 的隨機 probabilities，與完整 `2^n` event-pattern 枚舉一致。
- `assemble()` 後 frame 依 `(slot, ts, h)` 排序並重設 RangeIndex；模型 `p`、frame 與 `first_k_marginal` 的 positional indexing 在現有 artifacts 上保持對齊。
- dev/eval feature list 的內容與順序相同；現有 merge 沒有 row loss、duplicate、NaN 或 infinity。`uncensored_training_rows` 只作用於 family-filtered dev rows，逐列條件重算一致。
- DT scoring 使用 `rank_candidates` 的 `(1-W) * pct-rank(R15) + W * pct-rank(q)`；`W=0.35` 即 0.65／0.35。stack 實作為 `logit(tab) + BETA * logit(q)`；候選最後以 score 降冪、`ts,h` 升冪 tie-break。
- 未發現 eval pair target-label leakage。`evaluation_pairs.csv` 在 `y2` 只提供 pair/member mapping；`Y` 是逐 action 的觀測決策類別，不是 pair target label。

## 測試結果

最終結果：**7 passed, 2 failed，55.65 秒**。兩個 failure 都是上述已證實規格 bug，而非測試環境錯誤。

| 測試 | 結果 |
|---|---|
| a. 20 random DT pairs 累積欄位 loop 重算 | PASS |
| b. 10 random DT pairs raw arrays／actions／orientation 重算 | PASS |
| c. `first_k_marginal` 對完整 `2^n` 枚舉，`n=1..10` | PASS |
| d. `r17` vs `r13` patch integrity、candidate membership、row order | PASS |
| e. 5 dev pairs verbatim `x2` row-loop parity | PASS |
| assemble feature order／merge keys／finite／DP index alignment | PASS |
| right-censoring family scope與逐列條件 | PASS |
| all-routed candidate/frame coverage | **FAIL**：DT missing 27,145；CI missing 12,594（runner 先回報 DT） |
| `y2` receiver 語意遵循 directed-flow orientation | **FAIL**：全量診斷 835 / 1,529 DT pairs 不一致 |

### 實際命令

先確認指定 conda environment 存在：

```bash
/home/thisray/miniforge3/bin/conda env list
```

原先嘗試 pytest；指定環境未安裝 pytest，因此未執行任何 test case，且依 no-network 限制沒有安裝套件：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 taskset -c 16,17 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python -m pytest -q
```

改用本目錄內、僅依賴 Python 標準函式庫的 `run_tests.py` 執行相同斷言；以下是產生上述最終結果的命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 NUMBA_CACHE_DIR="$PWD/.numba_cache" taskset -c 16,17 nice -n 15 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python run_tests.py
```

## 產出檔案

- `test_role_features.py`：a、b、e，以及 `y2` orientation regression。
- `test_first_k_marginal.py`：c。
- `test_patch_integrity.py`：d 與 all-routed coverage regression。
- `test_deploy_contracts.py`：assemble／feature parity／finite／censoring／index alignment。
- `run_tests.py`：無 pytest 依賴的測試 runner。

