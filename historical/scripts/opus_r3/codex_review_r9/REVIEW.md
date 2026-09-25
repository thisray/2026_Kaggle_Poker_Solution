# R9 第四家族 evidence decoder 審查

## 結論

**有條件通過。** R9 evidence picks 使用的 likelihood 與 decoder 數學符合預期模型，獨立合成測試也沒有發現排名計算錯誤。實作正確使用 Beta 與常態 quadrature、由 decision ratio 建立 hand likelihood、在 `(u, rho)` 節點上形成 pair posterior，並在每個 posterior 節點分別執行 first-5 DP，最後才平均 DP 分數。C-order reshape 也一致採用 u-major、rho-minor 次序。

因此，目前的 `r9_*.csv` 可視為可信的模型輸出。不過，它們還不是可完全自我稽核的 artifacts：`t37_re2d.json` 與 `t37_re2d_pairs.parquet` 使用未含 prefix 的共用檔名，後續執行可能覆寫先前 R9 candidates 對應的資料。現有 artifacts 確實出現此情況：R9 CSV 時間約為 00:31，共用 JSON 則為 00:33，而且 JSON 內含四筆 CV 結果，與題目所述 R9 執行使用 `SKIPCV=1` 不一致。每個 CSV receipt 的 SHA-256 仍與目前 CSV 完全相符，但 receipt 未記錄足以證明該 JSON 屬於同一次執行的設定與 source lineage。

本次沒有發現 bug 等級問題。若要把未來 candidates 視為 bit-for-bit 可重現的正式 artifacts，仍應先修正以下風險。

## Findings

| 位置 | 嚴重度 | 發現 | 具體修正方式 |
|---|---|---|---|
| `t37_re2d.py:179-185` | risk | Diagnostics 使用固定檔名，不受 `PREFIX` 區隔；receipt 只包含少量執行設定。後續 invocation 能覆寫已產生 R9 candidates 所對應的 fitted parameters 與 simulation，現有 artifact 的時間與內容已證實這個問題。 | 改寫為 `t37_re2d_{PREFIX}.json` 與 `t37_re2d_pairs_{PREFIX}.parquet`。每份 candidate receipt 應包含 `BASE`、`PREFIX`、`NR`、`NU`、`NS`、`OOT`、`RV1`、`WIT`、`SKIPCV`、source SHA-256、fitted parameters，以及 diagnostics hashes；並採 atomic write。 |
| `t37_re2d.py:142` | risk | 個別 posterior probability 不高於 `1e-10` 的節點會被移除，其餘權重再正規化；這不完全等於規格中的「every node」。R9 M3 共 384 個節點，所以每個 pair 的丟棄質量上限為 `3.84e-8`，實質影響應極小，但極接近的第五／第六名仍可能因此交換。 | Candidate generation 不要剪枝，或將所有節點分塊執行 DP。若保留剪枝，應記錄 discarded mass 與第五／第六名 score margin，並 assert 截斷誤差不可能改變排序。 |
| `t37_re2d.py:88-90` | risk | 即使 L-BFGS-B 失敗、objective 非有限值或停在有問題的邊界，`fit` 仍直接使用 `r.x`。失敗的 fit 可能無聲地流入 decoder。 | 強制檢查 `r.success`、`r.x`／`r.fun` 有限，以及 projected gradient 足夠小；否則 fail closed。把 optimizer status 寫入 receipt，並考慮 deterministic multi-start。 |
| `t37_re2d.py:38-45` | risk | `(k, slot)` 與 `(slot, h)` merges 都沒有 cardinality 驗證。重複 helper key 會倍增 decision rows 並改變 hand likelihood；inner merge 遺漏的 `(slot, h)` 則會被靜默刪除。目前 artifacts 是乾淨的，但程式沒有強制保證。 | 使用 `validate="many_to_one"`，在每次 merge 前後 assert row count，assert helper keys 唯一，並明確回報／拒絕 unmatched rows。 |
| `t37_re2d.py:68-80` | nit | `reduceat` 的正確性依賴 decisions 按 `hid` 連續排列，以及所選 hands 按 pair 連續排列。目前 `G` 建立流程與第 45 行排序（包含 CV masks）確實滿足條件，但函式本身沒有驗證。 | 在函式邊界 assert `hid` 單調、每個 `hid` 只屬於一個 slot、slots 連續；或在 reduction 前局部 lexsort。 |
| `t40_v1_ratio.py:14-16` | risk | Helper 的 pair scope 固定由 `r5_subh.csv` 決定，但題目指定 R9 使用 `r5r_subh.csv`。現有 artifacts 中沒有造成 R9 差異：87 個 `r5r` 目標 pairs 是 91 個 `r5` pairs 的子集，14,938 個目標 decision rows 全部有覆蓋。然而未來 base 改變時，可能讓 `t37` 第 39 行 abort，或讓 helper 與 candidate scope 語意不一致。 | 將 `BASE` 參數化並與 `t37` 使用相同值；或為所有相關 `sub_meta_eval` rows 產生 ratios，使 helper coverage 不依賴 candidate CSV。將 scope 寫入 metadata。 |
| `t40_v1_ratio.py:21-26` | risk | Own-card re-prediction comparison 只印出結果而未強制驗證；寫檔前也未 assert output key 唯一與數值有限。Feature order 或模型不匹配時，可能產生錯誤 `r_v1`，卻仍通過 `t37` 僅檢查 not-null 的條件。 | Assert 最大 re-prediction error 在明定 tolerance 內；assert `(k, slot)` keys 唯一且 `r_v1` 為有限正值；將 validation metrics 與 parquet 一起保存。 |

## Alignment 與模型核對

- `G` 依 `(slot, ts, h)` 排序後指派遞增 `hid`；每個 pair 使用 `sort=False` grouping，因此完整保留 hand 時間順序。
- Decision rows 依 `(slot, h)` merge，再按 `hid` 排序，所以同一 hand 的 decisions 對 `reduceat` 而言是連續的；hand 內的 decision 次序不影響加總。
- `pair_ll` 的 pair grouping 也是連續的，因為遞增 `hid` 遵循 slot-major 的 `G` 次序。以 pool 建立的 CV masks 會選取完整 slots／pairs，且不改變這個次序。
- M3 tensors shape 為 `hands x U x R`。C-order flatten 讓 rho 成為 fast axis；`w.reshape(-1)`、`qp.reshape(hands, -1)`、`np.repeat(u, len(rho))` 與 `np.tile(rho, len(u))` 全部使用相同的 u-major 次序。
- Posterior weight 正比於完整 pair likelihood 乘上兩組 quadrature weights。`first_k_at_nodes` 先執行，再乘 posterior weights，因此實作的是 `E[DP(q)]`，不是 `DP(E[q])`。
- ACT 使用 `q_plant * (1 - exp(l0 - lf))`；WITNESS 使用 `q_plant * (1 - exp(LW))`，兩者均符合指定定義。
- OOT/RV1 以 `(k, slot)` 合併：v2-training／v1-holdout rows 使用 `r_v1`，兩個 policies 都訓練過的 rows 使用 `1`，其餘 rows 保留 v2 ratio。
- 實際 `t40_v1_ratio.parquet` 有 15,795 rows 與 15,795 個唯一 `(k, slot)` keys，ratios 全為有限正值；14,938 個 R9 target rows 的 join 缺漏為零。
- 現有四份 R9 CSV 的 SHA-256 均與各自 receipt 完全相符。

## 合成測試

所有測試只使用指定的 conda Python。`t37_re2d.py` 函式由 AST 載入，因此不會執行該檔案頂層的大型資料載入與 artifact 寫入。

測試涵蓋：

- `test_first_k_and_quadrature.py`：對 `n = 1, 4, 7, 10`、四個 node columns 與 `k = 1, 3, 5`，以全部 `2^n` patterns 的窮舉結果比對 `first_k_at_nodes`；並對三組指定 `(mu, kappa)` 檢查 Beta normalization、mean 與 second moment。
- `test_t37_pair_ll.py`：使用實際 `nodes`、`hand_terms` 與 `pair_ll` definitions，對 3 pairs、每 pair 5 hands、每 hand 1–4 decisions 的資料，以獨立的 node／hand／decision 三層 loops 比對。
- `test_t37_decoder.py`：針對一個 9-hand pair，以獨立的 node／hand／decision loops，比對 vectorized M3 PLANT 與 ACT integrated first-5 scores，包含 pair-win filtering、node pruning／renormalization 與 `1e-6` tie-break。

成功執行的 exact command：

```sh
OPENBLAS_NUM_THREADS=1 taskset -c 12,14 nice -n 12 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python -m unittest discover -v -p 'test_*.py'
```

結果：

```text
Ran 4 tests in 1.328s

OK
```

四個 test methods 共含 17 個獨立檢查情境：12 個 exhaustive first-K cases、3 個 Beta parameter cases、1 個 pair-likelihood case、1 個 decoder case。

Syntax check：

```sh
/home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python -m py_compile test_first_k_and_quadrature.py test_t37_pair_ll.py test_t37_decoder.py
```

執行成功，沒有輸出。

最初曾嘗試以下命令：

```sh
OPENBLAS_NUM_THREADS=1 taskset -c 12,14 nice -n 12 /home/thisray/miniforge3/envs/kaggle_poker_opus_260917/bin/python -m pytest -q test_first_k_and_quadrature.py test_t37_pair_ll.py test_t37_decoder.py
```

指定環境沒有 `pytest` module，因此沒有執行任何測試。由於禁止使用網路，未安裝任何套件；測試已改為標準庫 `unittest`。
