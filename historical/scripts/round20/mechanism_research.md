# Round20：政策 nuisance 記憶與 collusion 偵測的兩個正交機制

日期：2026-09-20
研究基線：`origin/main` / `e8750f8`
工作區：`/Volumes/SKC3000D2048G/VegaExternal/Projects/260916_Kaggle_Poker/worktrees/breakthrough-round20-20260920`
狀態：research-only；不提交 Kaggle、不推論隱藏標籤、不在 M4 執行 raw-data/full-data/heavy training。

本輪只新增本檔案；不修改 shared queue、canonical main 或既有候選。以下「新」是指「相對於本 repo 已審計的實作與文件，尚未被完整實作的機制」，不是宣稱學術文獻中的原創定理。

## 一、結論先行

Round20 保留兩個互相正交、但用途不同的機制：

1. **Composite Gameplay Null + cross-fitted conditional residual（CGN-CR）**：把 F4 的 partner-card policy 不是只和 normal `q0` 比，而是和一個真正 normalized、分組校準的多模式 gameplay null 比：normal、card-independent known-family（CI/DT/SP）以及只使用公開 partner action 的 policy。這是本輪唯一建議進入 GB10 最小 probe 的候選機制。
2. **Paired phase-null negative-control bridge（PNCB）**：把同 pair 的 dev/phase-null score 當成負控制與 nuisance calibration，不當成 eval-minus-dev 的正向 evidence，也不當成 temporal persistence feature。它的用途是檢查 policy memorization、phase/generator bias 與 self-inclusion 是否仍能製造假訊號；因此本輪只允許作 gate/calibration，不能改動安全候選。

兩者的分工如下：

| 機制 | 要回答的問題 | 可否直接排序候選 | Round20 verdict |
|---|---|---:|---|
| CGN-CR | partner-card substitution 是否仍超過合理的已知 gameplay null？ | 可作最小 probe；通過 guard 後才可進候選 | **PROMOTE TO MINIMUM SUFFICIENT PROBE** |
| PNCB | score 是否只是在重現 phase、policy memory、公開回應或 generator/估計偏差？ | 不可；只作負控制/校準 | **HOLD AS FALSIFICATION/CALIBRATION ONLY** |

這兩個機制都不提供 hidden label 的 exact inference，也不允許從 LB 反推 hidden label。沒有實測前，不對 uplift、effect size 或 F4 recall 作任何宣稱。

## 二、為何 normal-vs-substitution BF 不足

現有 `q_partner/q0` 或其 mixture 版本主要回答：「看到 partner 私有牌的 policy likelihood，是否比 normal policy 高？」但 F4 面對的不是單純的 normal 世界。它可能在 action distribution 上模仿已知 CI/DT/SP family，也可能只是對 partner 已公開的 action 作正常 response。若 null 只有 `q0`，這兩種合法或已知的 gameplay mode 都會被誤標成 partner-private-card evidence。

同樣地，將 partner-card likelihood 與平均 outsider likelihood 相減，並沒有把 known-family 與 public-response nuisance 放進 reference distribution。它還把多個 source 的行為差異壓成一個均值，無法知道「partner 勝過所有 outsider」還是「所有 source 都比 self 更有解釋力」。

本輪特別保留以下數學警告。t42 的 mixture score 若寫成

```text
D = log((1-a) q0_y + a qP_y) - log((1-a) q0_y + a qO_y)
  = log((1-a) + a qP_y/q0_y)
  - log((1-a) + a qO_y/q0_y)
```

則 `q0` 只被因式分解到兩個 nonlinear ratio 裡，**在 `a < 1` 時不會 exact cancel**。因此不能把 `D` 直接解讀成 `log qP - log qO`，也不能假設 common `q0` 會消掉 q0 的 self-inclusion、phase 或 style/meta 污染。

## 三、證據邊界：Round18/19、docs34 §27–31 與實際 code

### 3.1 已有文件提出過什麼，但尚未等於本輪新機制

- `docs/37_breakthrough_research_20260919.md` 已提出 pre-peer/post-peer/post-fold 的 source gate，以及保留 partner 對各 outsider source 的 profile。這是正確的資料切分方向，但不是 composite gameplay likelihood null。
- `docs/38_breakthrough_research_20260919.md` 已提出 matched conditional randomization、temporal self-baseline、observable-witness 與 conditional size policy scaffold。這使得本輪不能把「cross-fitting」或「matched null」本身宣稱為新；新處必須是多模式 policy distribution 與 phase-null bridge 的組合。
- `docs/34_opus_r3_top1_push_20260919.md` §27–§30 已把 policy-v1/v2 train-share memory、known-family memory rerank 負結果與 upstream refit 風險說清楚。Round20 不把 memory feature 的 rerank 負結果誤讀成「已證明沒有 policy nuisance」。
- docs34 §31 的 t46/t46b 結果顯示 phase-local：dev positives 幾乎沒有 eval pair overlap，且同 pair 的 cross-phase rank 不穩。這否定的是「同 pair 的 dev-to-eval score contrast 可當額外 evidence」；它不自動否定「dev score 可作負控制來檢查 false positive」。兩者的 estimand 不同，見第六節。

### 3.2 實際 model/code 讀取後的硬限制

| 位置 | 已確認事實 | 對 Round20 的限制 |
|---|---|---|
| `scripts/opus_r1/m4b_policy_v2.py` | `ctx_counts` 是 player × phase × context × action 的 full counts，而且使用目前 row 的 `Y`；`h_phase` 用來索引 phase-specific counts，不應描述成 FN1/FN2 的直接 predictor。 | OOT/RV1 只排除了 row membership，沒有自動排除 target self-included style/meta。任何新 ratio 先要做 self-row/causal-prefix audit。 |
| `scripts/opus_r1/m4b_policy_v2.py` | q0/qP/qO 共用一組可能含 phase 與目前 Y 的 policy feature。 | 不能以「common q0」推論污染抵消；t42 的 q0 在 mixture ratio 內仍存在。 |
| `scripts/opus_r2/s83_hand_tilt.py` | `collect` 是 partner active 時的所有 decision，沒有 partner first public action gate。 | `partner active` 不等於 `pre_peer`；新 probe 必須顯式使用 action-order event time。 |
| `scripts/opus_r3/t17a_eval_only.py` | `pre` 實作為 `m.st == 0`，即 preflop，不是 partner first public action 前。 | 不可把舊的 `pre` 名稱直接當作 information-set-safe pre-peer evidence。 |
| `scripts/opus_r2/s47_mech_lr.py`、`s48_mech_lrmin.py` | normal `pi0` 與單一固定 F4/partner mechanism 的 mixture/LR。 | 已有 normal-vs-substitution 先例，但沒有 known-family + public-response 的 joint null。 |
| `scripts/opus_r2/s49_fam_mech_e.py`、`s53_fam_llr_p.py` | CI/DT/SP 與 F4 分別產生 family LLR/features。 | 是 separate family tests，不是一個 normalized composite null；family score 不能直接當作 mixture component 而不重校準。 |
| `scripts/opus_r2/s76_responder_stat.py`、`s77_combined3.py` | public responder 以 standardized z，再與其他 p/z 分數 Stouffer combine。 | 已有公開回應訊號，但不是 `q_public` 進入 action likelihood null；本輪不重複做 z-score ensemble。 |
| `scripts/opus_r3/t39_policy_overfit.py`、`t40_v1_ratio.py`、`t41_mid_clean.py` | policy v2/v1 的 random train-share 與 OOT/RV1 修補揭露記憶問題。 | OOT/RV1 是必要但不充分的 hygiene；不能當作 full fold-safe feature generation。 |
| `scripts/opus_r3/t42_placebo_clean.py` | 用 mixture 後的 partner-vs-平均-outsider `D`，且 eval excess 偏向 card-independent。 | 平均 outsider 不是本輪的 composite null；並且 q0 不 exact cancel。 |
| `scripts/opus_r3/t43_member_devphase.py`、`t46_phase_twins.py`、`t46b_label_overlap.py` | dev/eval phase overlap、同 pair rank persistence 與 phase-local evidence 的檢查。 | phase contrast 不升級為 evidence；phase-null 只能作 negative-control/calibration，且必須測 generator/估計偏差。 |
| `scripts/round15/generic_conditional.py`、`scripts/r19_chatgpt/conditional_size_policy.py` | 已有 GroupKFold/OOF 的 conditional ratio scaffold。 | 不能宣稱「cross-fitted conditional ratio」本身新；本輪新的是其 reference object 與 phase-NC bridge。 |

### 3.3 Main 的 candidate audit / CI oracle：足夠但不可過度外推的新證據

本輪不再另開 broad search；以下只把 main 已產生的 receipt 納入邊界：

- `receipts/round20_candidate_audit_20260920.json`（AITOPATOM，base `e8750f8`）顯示候選間有些變化不是行為本身：`ndw -> r10` 的 risk string、global rank、behavior 都不變，但 evidence rows 有 564 個變化；`r10 -> r11` 有 2,299 個 risk/global-rank 變化而 behavior/evidence rows 不變；`r11 -> r9` 與 `r10 -> r12` 各有 20 個 behavior、91 個 evidence-row 變化，且 class-order 位置可大幅移動。這支持「候選 slate 對 evidence/rank policy 敏感」的審慎結論，不支持由 rank movement 推斷 exact hidden label 或 F4 mechanism。
- `receipts/round20_ci_oracle_20260920.json` 是 frozen-policy、exploratory OOF 的 92-pair / 81-pool CI oracle：r18 的 AP5 `0.7091` 高於 r15 的 `0.6542`，但 receipt 同時明確標示 oracle 不是 deployable gain、沒有 hidden labels、沒有 retraining；error cells 仍包含 false top-5 與 missed truth。這可用來說明 selection headroom 與 known-CI 的 evaluation ceiling，不能當作新 policy likelihood 已驗證，也不能拿來校準 F4。

所以本輪機制選擇只針對一個已證實的缺口：現有候選與 CI oracle 都顯示「action/evidence/rank 的 policy nuisance 仍可能改變解讀」，但沒有足夠證據支持更強的 hidden-label 故事。CGN-CR 用 composite null 解釋這個缺口；PNCB 只負責把 phase/generator nuisance 作為負控制檢查，兩者都不把 receipt 的 oracle 或 rank movement 轉成保證。

## 四、機制一：Composite Gameplay Null + source-dependent residual

### 4.1 核心構造

對每一個有明確 information set 的 action row `d`，先固定不含 hidden label 的 observable context：

```text
C_d = own/public state,
      street、facing price、active set、seat/action-order、可見 history、
      actor style 的 fold-safe 摘要、pool/table/time strata、exposure/opportunity。
```

再定義三種 policy mode：

```text
qN  (a | C) = q0(a | C)
qKI (a | C) = sum_f kappa_f(C) q_f^KI(a | C),  f in {CI, DT, SP}
qPA (a | C) = q_public(a | C, public partner-action history)
```

其中：

- `qN` 是 normal policy；不使用 partner 私有牌。
- `qKI` 是 card-independent known-family policy。`q_f^KI` 可以由已知 CI/DT/SP 的 family policy 估計，但在 action prediction 上只能使用 own/public state，不得把 partner 私有牌或 future action 偷渡進去。`kappa` 只在 training/reference folds 學習。
- `qPA` 只使用當下已經公開的 partner action history、其 event time 與一般 public state。它不是 partner-card policy 的代名詞；若在 strict pre-peer row 尚沒有 partner public action，該 mode 必須退化為可見的 prior 或權重受限為零，不能讀取未來 action。

把三者組成一個每個 action 都和為一的 null：

```text
q_null(a | C) = piN(C)  qN(a | C)
              + piKI(C) qKI(a | C)
              + piPA(C) qPA(a | C)

piN(C), piKI(C), piPA(C) >= 0
piN(C) + piKI(C) + piPA(C) = 1
```

最小版本先讓 `pi` 按 information channel、street、facing/opportunity 分組，而不是立即學一個高維 gating network。每個 `q` 與 `pi` 都在 pool-grouped training folds 內估計，並在未見 pool 的 fold 上輸出。這是 gameplay mode mixture，不是把多個分數平均；normal、known-family、public-response 的 action probabilities 先合成，之後才計算 likelihood。

partner alternative 使用既有的 card-swap / partner-card policy 作為起點，但它的判定改成相對於 composite null 的 source residual：

```text
ellP_d = log clip(q_partner(Y_d | C_d) / q_null(Y_d | C_d))
S_CGN_pre(pair)  = sum over pre-peer d of ellP_d
S_CGN_post(pair) = sum over post-peer d of ellP_d
```

`pre-peer` 是較尖銳的 falsification channel：partner 尚未公開當前 action，較能排除「只是正常回應 partner public action」的解釋；但它不是所有 detection 的硬門檻。`post-peer` 在條件化於已觀察到的 public partner action、event time 與可見 state 後，可以提供有效的 source-dependent residual，應獨立報告、不可與 pre-peer 混成不可解釋的總分，也不可讀取 future action/private card。若 post-peer residual 只在 qPA mode 出現，結論應是 public-response residual，而不是自動稱為 F4。

### 4.2 source-dependent nested residual 版本

若直接擬合 `q_partner` 仍受 policy memory 影響，改用 nested conditional model：

```text
q_alt(a | C, S) = q_null(a | C)
                  * exp(delta_S(C, a)) / Z(C, S)
```

`S` 是 source/information channel（normal reference、known-family、public response、partner-card candidate），而 `delta_partner` 是在 composite null 上的 residual，而不是另一個可任意吸收 style 的 absolute model。`Z` 只負責 action normalization。候選 evidence 是 held-out、group-calibrated 的 `delta_partner(Y,C)` 是否超過 zero，而非 model training fit。

這個 nested 形式對本 repo 的實際痛點是有針對性的：若 q0、qP、qO 共同含有 style/meta leakage，absolute likelihood 可能都變好看；但在 null 上要求 source residual 在 known-family/public-response controls 上接近 zero，可以把「政策記憶」與「partner source-specific residual」分開審計。

### 4.3 為何相對於已嘗試內容是新的

本 repo 已有以下局部元件，但沒有本機制的完整 object：

- `s47/s48`：normal 對一個 F4/partner mixture 的 LR；沒有 CI/DT/SP 與 public-action mode 的共同 normalized null。
- `s49/s53`：CI/DT/SP、F4 各自計算 family LLR；separate feature/score 不等於 `q_null = mixture(...)`。
- `t26/t27/t42`：partner 對 average outsider 的 D；outsider mean 不是 known-family/public-response gameplay null，而且 `q0` 在 mixture D 中不會消失。
- `s76/s77`：public responder z-score 及 Stouffer combine；沒有把 public-response action distribution 放進 partner-card LR 的 denominator。
- `round15/generic_conditional.py` 與 `r19_chatgpt/conditional_size_policy.py`：已證明 repo 有 conditional OOF ratio 的 scaffold，但 reference 是 own/partner size 或 general conditional model，並非這個三模式 gameplay null。

因此本機制的 novelty boundary 是：**以一個 normalized、group-calibrated 的 gameplay-mode mixture 作 denominator，再在其上估 source-dependent partner residual**。這不是 generic graph/HMM/PU，也不是把已存在的 family score 再做一次平均。

### 4.4 最小可行 probe（只交給 GB10，Round20 不執行）

1. 從既有 arrays 重建 action-order，明確產生 `pre_peer`、`post_peer`、`post_fold`；`m.st == 0` 不能當 substitute。
2. 先做 feature manifest：`ctx_counts` 對 validation target 必須排除 target row；若 feature 語意是因果時間，進一步只用 current action 之前的 prefix。檢查不能只看 OOT/RV1 row membership。
3. 以 pool/table 為 group 做 OOF；在 train/reference folds 估 `qN`、`qKI`、`qPA` 與 `pi`，在 held-out pool 輸出 `q_null`。已知 CI/DT/SP 只用其 training labels；eval/U 不硬標成 negative。
4. 同時報四個 held-out object：normal BF、既有 outsider `D`、`q_partner/q_null`、nested residual。先比較 log loss、calibration、grouped tail stability、effective sample size，再看候選 ranking。
5. `qPA` 做 source ablation：若 composite score 的所有 excess 都在 post-peer 或 qPA 開啟後消失，結論是「原 BF 讀到 public response」，不是 F4。

### 4.5 必過的 falsification / safety gate

- partner-fold、seat permutation、同 table outsider permutation 應接近 null；若不接近，停止，不升級 score。
- 將 `h_phase` 所索引的 phase-specific counts 做 phase-index neutralization、去除 current-Y self-count、改用 leave-one-row/leave-one-hand/prefix counts 後，ratio 的 calibration 不應出現不合理跳變；若跳變，先標成 generator/估計偏差疑點，不宣稱是規則違反。
- `qPA` 不得讀 future partner action、private card、hand outcome 或 hidden label；`qKI` 不得用 eval/U 的假 negative。
- known CI/DT/SP 的 held-out family checks 應落在其預期 null/reference；若 `q_partner/q_null` 只對 known family 有 signal，不能稱為 F4 evidence。
- `pi` 與 residual 的 group tail 必須在未見 pool 穩定；不得只用 LB 或任何外部 hidden label feedback 調整。

## 五、機制二：Paired phase-null negative-control bridge

### 5.1 目的不是 phase contrast

這個機制刻意不使用：

```text
eval score - dev score
同 pair 的 dev rank percentile
phase ID 當作 positive feature
```

它改問一個更窄的問題：**同一個 extractor 是否在一個沒有 target exposure 的 phase-null 上也會亮起？若會，亮起的部分能否被當作 nuisance calibration，而不是 evidence？**

設：

```text
Y_p = eval phase、source-appropriate 的 CGN residual，按 pair/hand 聚合
W_p = 同 pair、同 information channel 的 dev phase-null score
Z_p = eval phase 的 partner-fold、outsider-source 或 seat-scrambled score
C_p = exact opportunity/state/exposure/seat/time/pool strata
```

解讀是：`Y` 是 primary detector output；`W` 是可能共享 player/style/phase-generator nuisance、但理論上不應受 eval-only F4 exposure 影響的 negative-control outcome；`Z` 是共享 nuisance 但不應讓 target partner-card mechanism 產生 effect 的 negative-control source。這是 observational negative-control 的「類比移植」，不是宣稱 Poker 實驗具有 paper 中的 causal identification。

### 5.2 Cross-fitted bridge / conditional density-ratio 實作

兩個低複雜度版本可先擇一：

**A. Score-density-ratio calibration**

在 group-fold training data 上，以 `C` 條件化，估計 phase-null/control score 與 matched reference score 的 ratio。可以用直接 conditional density-ratio 方法，或用小型、可審計的二元 classifier 將其 reparameterize；所有 ratio 只用 OOF 輸出：

```text
w_NC(y | C) = p(Y=y | C, phase-null/control)
              / p(Y=y | C, matched reference)
```

`w_NC` 的用途是把 primary residual 的 tail 校準到 null，而不是把 `W` 直接加進候選 rank。group calibration 應至少按 pool/table、phase/opportunity、street/facing 分層，並回報 overlap/ESS。

**B. Paired bridge residual**

在 training folds 內以低維 ridge/spline/monotone calibration fit `b`，檢查：

```text
E[ Y_p - b(W_p, C_p) | Z_p, C_p ] ~= 0
S_bridge_p = Y_p - b_hat(W_p, C_p)
```

`S_bridge` 只在通過 negative-control moment、phase-only AUC 與 permutation checks 後，才可用來判斷「原始 score 的哪一部分不是 shared nuisance」。它不是 causal effect estimator；`b` 的作用是 diagnostic bridge，不是為了從 phase difference 創造新 evidence。

這種做法把兩個要求分開：conditional density ratio 處理 distributional nuisance，cross-fitting 防止 bridge 直接記住 held-out pool；negative control 則用來檢查 target exposure 以外的共同偏差。若 `W` 與 `Y` 完全沒有穩定的 conditional relation，也不能硬套 bridge；它應被標成 weak/invalid negative control。

### 5.3 為何 docs34 §31 的 phase rejection 不會邏輯上排除它

docs34 §31 的結論是：dev/eval collusion evidence phase-local，同 pair cross-phase rank 不穩，故不能把 dev score 當作同 pair 的正向 persistence evidence 或 `eval - dev` temporal baseline。這個結論與 PNCB 相容，因為 PNCB 不需要 rank persistence：

| 用法 | 所需假設 | docs34 §31 的影響 |
|---|---|---|
| temporal causal baseline：`eval - dev` 是同 pair 的 counterfactual change | response mechanism 跨 phase 可 transport、pair signal 具穩定 persistence、phase drift 可調整 | phase-local / rank 不穩使其不安全；本輪拒絕 |
| negative-control outcome：dev score 不受 eval-only target exposure 影響，但可共享 nuisance | no target exposure effect on `W`；有足夠 shared nuisance/overlap；generator/估計偏差可被檢測 | rank 不穩不是自動失效，反而提醒不能把 `W` 當 positive evidence；仍須做 overlap 與 bias checks |

因此「phase-null 能否作 negative control」不能由 t46 單獨證明；t46 只說 direct phase contrast 不可靠。真正的 validity test 是：

1. `W` 在 matched phase-null/partner-fold controls 上不應反映 target exposure。
2. `W` 與 `Y` 若共享 nuisance，應在 `C` 條件化後提供可重現的 calibration information；不必有 pair rank persistence。
3. `Z`、phase permutation、seat/source permutation 上的 bridge moment 應接近零。
4. 輸出對 phase-indexed counts 與 self-included current-Y counts 的敏感度；若比對 gameplay exposure 更敏感，應判為 generator/估計偏差疑點，而非 negative control 的 evidence。

### 5.4 temporal baseline 與未證實 generator/估計偏差的邊界

**合法但不宜當 evidence 的 temporal baseline**：若 dev/eval 的行為生成機制確實有可運輸性，dev 可作預期 policy baseline；但本 repo 的 phase-local 結果已使此假設不安全。本輪不把它拿來加分、排序或推論 persistence。

**可用來抓 generator/估計偏差的 phase-null**：把 dev score 當「不應因 eval-only F4 exposure 改變的 control output」。它的成功條件是檢查 false positive，不是把 dev 的高低轉成 eval evidence。

**未證實的 generator/估計偏差風險**：`m4b_policy_v2.py` 的 `ctx_counts` full player-phase counts 直接含目前 `Y`，而 `h_phase` 用來索引這些 phase-specific counts；FN1/FN2 不應描述成直接吃 phase ID。即使 row membership 用 OOT/RV1 處理，target row 仍可能被 self-included style/meta 看見，phase 的 action composition 也可能被模型記憶。這可能造成 target leakage 或估計偏差，但本 memo 沒有足夠證據把它分類為規則違反或「非法」生成。是否允許，必須依 competition 的明確 conditional rules 判定；在規則未核實前只作風險標記。必做的對照是：

- fold/OOF counts 排除 validation target row；對有因果時間意義的 counts 改用 action-prefix 或 leave-one-hand/leave-one-row 版本。
- neutralize 或 permutation `h_phase` 所索引的 phase-specific counts 後重算 diagnostic score，而不是假設模型直接使用 phase ID。
- phase label permutation、hand-order permutation、partner-fold 與 outsider-source swap。
- 比較 full-current-Y、leave-one-row、prefix-only 三個版本的 null calibration；不報「差異很大」之類未測量的 claim。

### 5.5 最小 probe 與 stop rule

這個機制本輪只列為 downstream GB10 gate：

1. 依 t43/t46/t46b 的 lineage 重建 `Y` 與 `W`；不把 hidden labels、LB 或 evidence truth 帶入 bridge。
2. 以 pool-grouped OOF fit `w_NC` 或 `b`，分別報 matched overlap、ESS、negative-control moment、phase-only AUC、source/seat permutation。
3. 在 current-Y self-count 修復前，不用 raw score 的 phase difference 做任何 candidate promotion。
4. 若 `W` 沒有穩定 shared-nuisance relation、bridge 不穩、`Z` 對 `Y` 有非零 effect、或修復 self-inclusion 後 result 消失，PNCB 立即停止，保留為失敗的 diagnostic attempt，不把它變成新特徵。

## 六、兩個機制的差異與實驗對照

| 對照 | normal-vs-sub BF | outsider `D` | CGN-CR | PNCB |
|---|---|---|---|---|
| denominator/reference | `q0` | 平均 outsider mixture | normal + known-family + public-action gameplay modes | matched phase-null/control distribution |
| target | partner-card substitution vs normal | partner source vs outsider average | partner source residual beyond plausible modes | score 是否含 shared nuisance/generator bias |
| 是否需要 pair phase persistence | 否 | 否 | 否；但要有 action-order-safe source split | 否；只要求 control validity/overlap |
| 是否可直接排序 | 理論上可，但易混 nuisance | 可作 diagnostic，不能當 composite null | 通過 safety gate 後才可 probe candidate | 不可，僅 calibration/falsification |
| 主要 failure | known family/public response 被當 F4 | source heterogeneity 被均值壓扁；q0 未消失 | qKI/qPA misfit、self-count、future-action leak | phase drift、弱 negative control、generator/估計偏差 |

## 七、Cross-fitting 與資料 hygiene 的不可妥協項

cross-fitting 在本輪不是裝飾。它的最低定義是：所有 policy components、family mixture weights、conditional ratio、bridge/calibration function 都在 group-held-out training folds 估計，再對未見 pool/table 輸出。使用同一 pool 的 row split 不能視為足夠，尤其 player/style/context counts 會跨 row 傳遞記憶。

本 repo 特別需要區分三種「沒有 row membership」：

1. **model row membership 被移除**：OOT/RV1 可做到的部分。
2. **target self-included aggregate 被移除**：`ctx_counts` 必須明確排除目前 `Y`，而不是依賴 q0/qP/qO 共用 feature。
3. **未來資訊被移除**：若 action policy 要作 causal/temporal interpretation，必須只用當前 action 前的 public prefix；phase 或 full-hand aggregate 不能冒充當前 information set。

第 2、3 項沒有通過前，任何 mixture BF、conditional ratio 或 PNCB 都只能是 code audit，不得作效果結論。尤其不能因為 qP、qO 共用 q0 就宣稱 leakage 會抵銷；第 2 節的 nonlinear formula 已說明原因。

## 八、Literature：方法依據與不可過度外推之處

以下四篇是本輪保留的 primary/official paper pages；它們支持方法的統計動機，不是 Poker hidden-label validity 的證明。

1. [Sugiyama et al., 2010, Conditional Density Estimation via Least-Squares Density Ratio Estimation](https://proceedings.mlr.press/v9/sugiyama10a.html)：直接估 conditional density ratio、避免先獨立估兩個 density，支持在 matched context 下直接校準 source/null ratio。本文移植到 action/score context 時仍須自行處理 group leakage 與 support overlap。
2. [Chernozhukov et al., 2018, Double/debiased machine learning for treatment and structural parameters](https://academic.oup.com/ectj/article/21/1/C1/5056401)：Neyman-orthogonal score 與 cross-fitting 可降低 flexible nuisance fitting 帶來的 overfit/regularization bias。這只支持 OOF nuisance hygiene，不代表本任務取得 causal identification。
3. [Lipsitch, Tchetgen Tchetgen, and Cohen, 2010, Negative controls](https://pubmed.ncbi.nlm.nih.gov/20335814/)：negative-control exposure/outcome 不應與 primary causal path 有關，但可共享 bias/confounding，用來偵測 spurious inference。PNCB 是這個思想對 phase/policy generator 的保守類比。
4. [Miao et al., 2018, A confounding bridge approach for double negative control inference](https://arxiv.org/abs/1808.04945)：以 bridge moment 使用 negative-control variables，並明確指出需要額外 assumptions/completeness。Round20 只取其「用 control 做 residual/bridge diagnostic」的結構，不宣稱 Poker 滿足其條件。
## 九、推薦順序與本輪停止線

### 推薦順序（下一個有 GB10 資源的 round）

1. 先產生 fold-safe feature manifest，明記 `ctx_counts` 的 target self-row、hand、prefix 排除規則；不把 raw data/feature bank 搬到 M4。
2. 只做 CGN-CR 最小 probe：`qN/qKI/qPA`、grouped OOF、pre-peer/post-peer 分開的 action-order channels、normal BF / outsider D / composite residual 三者並列。
3. 以 partner-fold、outsider、seat/phase permutation 與 known-family held-out calibration 做 safety gate。
4. 若 CGN-CR 的 signal 通過，才做 PNCB；PNCB 仍只作 false-positive/generator-bias gate，不回灌 phase contrast。
5. 只有在 self-inclusion、future-action leak、group calibration、negative-control moments 都通過後，才考慮下游 evidence slate；本輪不提交。

### 明確停止線

- 不用 LB 反推 exact hidden labels。
- 不把 docs34 §31 改寫成「phase score 完全無用」或「phase score 證明 collusion」；精確結論是 direct phase contrast 不安全，phase-null negative-control 仍待 validity test。
- 不把 `q0` 的 common factor 當成 exact cancellation。
- 不把 OOT/RV1 當成已完成的 feature-level OOF；`ctx_counts` 的 current-Y self-inclusion 必須另行修復/驗證。
- 不因一個新 ratio 在 eval candidate 上變尖，就稱為 F4；先看 known-family、public-response、partner-fold、phase/permutation controls。
- 若只剩 heavy retraining 才能回答，將問題移交 GB10/AITOPATOM，不在 M4 執行。

## 十、Round20 verdict

**機制一：CGN-CR — 通過 novelty boundary，允許最小 probe。** 其具體新意不是「又做一個 BF」，而是用 normalized composite gameplay null 把 normal、已知 card-independent families、public partner-response 放到同一個 denominator，並在其上估 source-dependent partner residual。它直接對準 F4 與 known CI/DT/SP 混淆的缺口；但在 `ctx_counts` self-inclusion 與 action-order audit 完成前，不得宣稱有效。

**機制二：PNCB — 保留為負控制/校準，禁止升級候選。** phase-local evidence 與缺乏 rank persistence 足以否定 temporal phase contrast，卻不等於否定 negative-control 用法。PNCB 只在 held-out null、overlap、bridge moment、phase/source permutation 與 self-inclusion sensitivity 都合理時，才可用來辨識 generator/估計偏差；它不增加 evidence，也不從 phase 反推 hidden label。

本輪沒有提交、沒有改 main/shared queue、沒有在 M4 使用 competition raw data 或執行 heavy compute。
