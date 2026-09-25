# Round 20：scorer／t47–t48 交互作用稽核

日期：2026-09-20（Asia/Taipei）

範圍：只做靜態 code／文件稽核；沒有使用 hidden labels、沒有跑 M4 data probe、沒有訓練，也沒有提交。GB10 的 read-only audit 結果採用主線回報；目前正式 submission quota 保持 1。

## 結論

1. **「pure pair-AP effect」這個說法不正確。** t48 只把 `risk_score` 換掉，並以 assertion 固定 `predicted_behavior` 與 evidence；它是 risk-only intervention，不是只改 PairAP。官方 scorer 對 PairAP 用 `risk`，同時對每個 known family 用 `risk * 1[predicted_behavior == family]` 算 BehaviorMAP；因此 risk 變動會同時改 P 與 B。[官方 metric notebook](https://www.kaggle.com/code/florianderoofr/slash-poker-competition-metric)（本地 archived source SHA-256 `d0164221aefd839626d99fedb799d50c2aa47c5fcb90f05e90520fc225e96901`）；[repo scorer](../../src/pokerlab/metrics.py#L101-L116)

2. **t47 的定位則成立。** t47 固定 risk／behavior，只改 CI evidence；在 scorer 語意下 P、B 不變，故 `r10_ci − NDw = 0.00198` 是 `0.20 × ΔE` 的 evidence-only 效果。[t47](../opus_r3/t47_ndw_ci.py#L7-L16)；[queue 的 LB 對照](../../docs/kaggle_submit_queue.md#L372-L390)

3. **`r12 ≈ 0.92675` 目前只是 interaction=0 的加法假設，不是已證明的預測。** queue 的算術是 `0.92501 + (0.92115 − 0.91941) = 0.92675`；它把 `r11→r9` 的 label/evidence bundle 增量搬到另一個 risk context，忽略了 risk mask 與 class ranking 的交互作用。[queue](../../docs/kaggle_submit_queue.md#L376-L390)

## 四角結構與精確恒等式

GB10 read-only audit 回報的 lineage 是完整 2×2 factorial（`q` 是 `predicted_behavior + evidence` bundle）：

| | `q0`：r10 的 label/evidence | `q1`：r9 的 label/evidence |
|---|---|---|
| `r0`：NDw risk | **r10** | **r12** |
| `r1`：r9 risk | **r11** | **r9** |

這與 audit 的 structural checks 一致：`r12.risk == NDw.risk`、`r12[q] == r9[q]`、`r11.risk == r9.risk`、`r11[q] == r10[q]`。[candidate audit checks](candidate_audit.py#L47-L80)；r12 應重新核對的完整 SHA-256 為 `4139bb5ea92ca4676c031f4650c07f3ab442fa04ed4990318dc9348eff9a51b4`。

令 `S(r,q) = 0.70P(r) + 0.20E(q) + 0.10B(r,q)`，則四角的精確 interaction identity 是

```text
I = S(r9) - S(r11) - S(r12) + S(r10)
  = 0.10 * [B(r9) - B(r11) - B(r12) + B(r10)]
```

第二行成立是因為 factorial 結構使 P 在同一 risk row 抵消、E 在同一 bundle column 抵消；剩下的只有 risk 與 predicted behavior mask 的 joint effect。等價地，

```text
S_add(r12) = S(r10) + S(r9) - S(r11)
S_add(r12) - S(r12) = I
S(r12) = 0.92675 - I       # 0.92675 使用 queue 顯示的四捨五入 LB 值
```

所以只有在 `I = 0` 被四角 scorer 實測支持時，才可把 `0.92675` 當作合理外推。`r10→r11` 的 GB10 audit 已看到 global order 改變 2,299 列，且 DT/SP/CI class-order positions 分別改變 284/1,014/239；`r9→r12` 也改變 DT/SP/CI 283/310/0。另 `r11→r9` 改變 20 個 behavior labels、91 個 evidence rows。這是實質的 factorial interaction 結構；order counts 本身不提供 `I` 的符號或大小，不能把它假設為零。

同理，`r10→r11` 的 LB `−0.00560` 只能解讀為

```text
0.70 * ΔP + 0.10 * ΔB
```

不是純 `0.70 * ΔP`，更不能由總和推論 16 個 promoted pairs 逐一皆負；AP 的跨列 rank crossings 與三個 class masks 都會混入此 aggregate delta。[t48](../opus_r3/t48_rank_only.py#L6-L17)

## 最小 GB10 驗證提案（read-only）

1. 保留現有 `candidate_audit.py` 的四檔 lineage／factorial／class-order 檢查，並核對 r12 完整 SHA；不修改 queue、不寫 candidate、不提交。
2. 四個 actual eval CSV 沒有對應的 local truth，不能直接呼叫 scorer，也不能拿 OOF truth 冒充 eval truth。改在 GB10 的 frozen dev OOF 上，依相同 feature-defined selection 建立四角 interventions，配 known dev labels；或使用 fully synthetic binary/family/evidence labels 測試官方 scorer／`pokerlab.metrics` 的代數。輸出必須標成 `I_dev`／`I_synthetic`，不是 `I_eval`。
3. 在 surrogate 四角報告 `I = S9−S11−S12+S10`，並確認 `I ≈ 0.1 × (B9−B11−B12+B10)` 及 `P11=P9、P10=P12、E11=E10、E9=E12`（容許浮點誤差）。這只能驗證 scorer algebra 與 local interaction，不估計 actual eval/private `I`；因此 `0.92675` 仍是未驗證的 submission-free extrapolation。
