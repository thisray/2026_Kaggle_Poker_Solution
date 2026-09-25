# Round 20 Decoder Audit：從 first-K 邊際到 expected AP@5

查核日期：2026-09-20（Asia/Taipei）
查核基線：指定 worktree，HEAD / origin/main = e8750f8
範圍：只讀取 t37、R18/R19 decoder 與既有 review；未讀 raw competition data，未執行資料 run、訓練、submission 或 large probe。

## 結論先行

1. R18 的 first-K DP、R19 的 random-effects node posterior，以及 t37 的
   E[DP(q)] 積分，在「計算每手屬於 latent first-five event set 的邊際機率」
   這個目標上是相符的。歷史 R9 review 的 synthetic tests 也只驗證了這一層。
2. 這個邊際機率排序一般不會最大化 E[AP@5]。一旦 posterior mixture 使手牌
   相關，AP 的 prefix-precision interaction 會需要 pair moment；random
   cardinality 與空真集合還會改變 denominator。下面給出一個含空集合、cap=5
   的代數反例。
3. R19 已有正確的可重用 building block：capped_ap.py 的 moments + subset DP。
   正確的 F4 版本應在每個 posterior node 先算 AP moments，再對 node／rule
   mixture 加權，最後以現有 optimize 解 slate；不能先平均 q 再做 DP，也不能
   用邊際 q 排序代替 slate utility。
4. 目前 worktree 沒有 actual F4 exact-AP receipt。t37 的數字是 likelihood
   validation 或 self-consistent simulation，不是已觀測 F4 AP；.70909、
   .75435、.94783 是 NEW Round20 ci_oracle_audit.py 的 observed OOF 與
   ORACLE outputs，不是 screen_ci_ap.py 結果。它們仍是 CI-only
   shortlist/control 結果，不能直接移植成 F4 posterior-mixture AP。因此
   本輪的實證 verdict 是「演算法可行，F4 實資料 feasibility 尚未驗證」。

本輪沒有引用外部 primary literature：所需結論可由 AP 的 adjacent-swap
代數直接推出；文獻不會替代本題的 annotation、censoring 與 posterior
semantics audit。

## 1. 目標與 AP 代數

令 G 是 latent true evidence set，L=(l1,...,l5) 是輸出的五手，Ii=1{i∈G}，
H_r=Σ_{s≤r} I_{l_s}，D=min(5,|G|)。官方語意是：

    AP@5(L,G) = 0,                                      if |G| = 0
                 (1/D) Σ_{r=1..5} I_{l_r} H_r / r,       otherwise

若要比較一個固定 shortlist 內的順序，AP 不是單純的 Σ P(i∈G)。
對相鄰的 a,b（位置 r、r+1），令 H 為其前綴命中數；交換兩者後，所有
suffix contribution 會抵消，得到：

    AP(...,a,b,...) - AP(...,b,a,...)
      = (Ia - Ib)(H + 1) / (D r (r+1))

空集合按官方定義貢獻 0。若 Ia、Ib 在一個單一 posterior 下彼此獨立，
且與其餘 relevance indicators 獨立，交換的期望差號誌由
P(a∈G)-P(b∈G) 決定；這是 marginal ranking 可以適用的窄條件。
它不是 correlated posterior mixture 的一般定理，也沒有解決候選
shortlist 漏掉 true hand 的問題。

對任意 posterior，固定 order 的期望 AP 可寫成兩階矩：

    A_i = E[I_i / D]
    B_ij = E[I_i I_j / D]
    E[AP(L)] = Σ_{r=1..5} ( A_{l_r} + Σ_{s<r} B_{l_s,l_r} ) / r

這正是為什麼需要 denominator-weighted pair moments，而不是普通
inclusion probability 或 E[I_i]E[I_j]。

## 2. 具體反例：random cardinality + correlated mixture

取 cap K=5，候選全集為 C、A、B、D1,...,D20。posterior mixture 為：

| posterior component | probability |
|---|---:|
| G=empty | 0.20 |
| G={C,B} | 0.30 |
| G={A,D_j}，每個 j=1,...,20 | 0.31/20 |
| G={C,D_j}，每個 j=1,...,20 | 0.19/20 |

因此 |G| 是 random（0 或 2），且所有非空集合都低於 cap；cap 不必啟用
就已足以反駁「對所有 capped random-cardinality posterior 都可排序
marginal」的說法。每個 component 可以視為 conditional-independent
Bernoulli model 的 deterministic limit；相關性來自共同的 latent component。

邊際 inclusion probability 為：

| hand | P(i∈G) |
|---|---:|
| C | 0.49 |
| A | 0.31 |
| B | 0.30 |
| each D_j | 0.025 |

所以 marginal ranking 會把 C 放在 A 前、再把 A 放在 B 前。比較兩個都
接同一個後綴的輸出：

    L1 = (C, A, B, D1, D2)
    L2 = (C, B, A, D1, D2)

這裡 P(A)=0.31、P(B)=0.30、P(C,A)=0、P(C,B)=0.30。由 adjacent-swap
公式：

    E[AP(L1)-AP(L2)]
      = [P(A)-P(B)+P(C,A)-P(C,B)] / 12
      = [0.31-0.30+0-0.30] / 12
      = -0.0241666667

也就是 B 的 marginal 較低，卻應排在 A 前；L2 的 expected AP 高出
約 0.02417。空集合已正確貢獻 0，而 D_j 被分散到 tail，故這不是
「單純把高 marginal 的 D 放掉」造成的例外。

這個反例只證明 correlated posterior mixture 下不存在一般性的 marginal
optimality guarantee；它不證明特定 M3 的 rho/u posterior 實際上已經
產生這個依賴型態。t37 仍需用實際 node-wise A、B moments probe 量測
current marginal slate 與 exact-mixture slate 的 gap。E[first-five
inclusion] 在一般情況下不足以決定 E[AP@5]。

## 3. 實作逐項稽核

### 3.1 R18：first-K marginal 是 membership decoder，不是 AP decoder

ci_censored_event.py:61-77 的 first_k_marginal 在 full chronological
hand coverage 上做 conditional-independent first-five DP；其 docstring
也明確要求 full shared hands。這部分的 recurrence 正確。

但 ci_censored_event.py:79-92 只把 q 與既有 rank r 做 percentile blend，
ci_censored_event.py:163-169 再取 top five。它沒有計算 A_i、B_ij，也沒有
max expected AP。這是 heuristic reranking，不應稱為 AP-optimal decoder。

更重要的是 r3_cand_event.py 對非-candidate shared hands 將 p 設為 0，再
進 first-five DP。候選外的手不只是「不能輸出」；它們仍會成為 latent
first-five event，影響 cap、後續 membership 與 D。正確做法是：

- DP 的 p 保留 full hand sequence；
- shortlist 只限制可輸出的 candidate positions；
- candidate 外事件仍留在 moments 的 pre/suffix distribution。

### 3.2 R19 random-effects helper：積分順序正確，但目標仍是 marginal

f4_random_effects.py:49-56 的 first_k_at_nodes 在每個 node 做 DP。
f4_random_effects.py:63-73 先用 pair posterior weight 算
E[DP(q)]，並另存 qpm/qam 與 naive DP(E[q])。這個「先 DP、後
integrate」是保留 shared rho correlation 的正確做法；歷史
test_random_effects.py 的 enumeration 也只驗證這個 first-K probability。

它仍然輸出每手 first5_plant_re / first5_act_re，而不是 expected AP
moments。把其中的 q 排序，最多是 marginal-membership policy。

### 3.3 t37_re2d.py：node posterior 正確，slate objective 不完整

- t37_re2d.py:74-80 的 pair likelihood 對每個 (u,rho) node 累積 hand
  likelihood，再乘 quadrature weights；模型內的 node posterior 形成方式
  與 R19 specification 一致。
- t37_re2d.py:142-149 在每個 pair 對 node posterior 加權後計算
  first_k_at_nodes(qa2*pw) 或 first_k_at_nodes(qp2*pw)。這是
  E[first-five membership]，不是 E[AP@5]。
- t37_re2d.py:150 的 m3_mix 直接以 sa+sp_ 排序。它沒有 model posterior
  weight，也沒有 ACT/PLANT 各自的 A、B moments；若 ACT/PLANT 是 alternative
  hypotheses，應使用 π_ACT、π_PLANT 加權的 moments；若是 union event，
  則需要明確的 joint union likelihood。直接相加不是可解釋的 posterior
  mixture utility。
- t37_re2d.py:168-175 的 simulation 遇到 empty truth 直接 continue，故
  reported simulation mean 是 conditional on non-empty truth。若要估計
  unconditional expected AP，empty draw 應以 zero contribution 計入；
  這會把同一 pair 的非空 AP 按 non-empty mass 加權，影響 pair-level
  mean 與跨 pair/model comparison，但不代表 pair 內 slate order 必然改變。
- t37_re2d.py:146 剪除 node weight ≤1e-10 後重新 normalize。這對 marginal
  score 通常很小，但近 tie 的第五／第六名或 exact AP slate 可能改變；
  exact decoder 應保留全部 nodes 或記錄 discarded mass 並做 ranking-stability
  bound。
- t37_re2d.py:89-94 在 optimizer 非 success 時只印 WARNING，仍回傳參數
  進 decoder。這是 fit gate 的 risk，不是 AP 代數 bug；正式 exact decoder
  應 fail closed。

另有可重現性問題：t37_re2d.py:16 與 t34_re_f4ev.py:17 以 local
scripts/opus_r3 path import f4_random_effects，但目前 clean worktree
只有 scripts/r19_chatgpt/f4_random_effects.py，沒有
scripts/opus_r3/f4_random_effects.py。codex_review_r9 的 BRIEF 與 tests
假設 parent directory 有該 helper；歷史 REVIEW 的「4 tests OK」不能
直接代表目前 checkout 可重現。這輪不修檔，僅記錄為 source-lineage gate。

### 3.4 R19 capped_ap.py：已存在的正確 building block

capped_ap.py:17-42 先對 full chronological event vector 建 pre/suffix
Poisson-binomial states，再計算：

- a_i = E[I_i / min(5,|G|)]；
- b_ij = E[I_i I_j / min(5,|G|)]。

因為若 G 是 empty，I_i 與 I_i I_j 都為 0，不會出現 zero denominator；
expected AP 自然是 0。capped_ap.py:52-78 的 subset DP 在候選數
M≤18 時同時搜尋五手的 subset 與 order。capped_ap.py:80-81 的
expected_ap 也正好是上面的二階矩公式。

所以不需要重造 decoder。需要修正的是輸入 semantics：

- single independent node：直接用 moments(p_z, candidates,
  constant_denominator=False)；
- posterior mixture：每個 node 都算 moments，先加權平均 a_z、b_z，
  再呼叫同一個 optimize；
- 不可使用 moments(E[p_z],...)，那會把 correlated mixture 偽裝成
  independent pseudo-posterior；
- constant_denominator=True 只有在 |G|=5 幾乎確定，或刻意研究固定
  denominator surrogate 時才適用。

### 3.5 screen_ci_ap.py：已有 scaffold，但不是 F4 的結論

screen_ci_ap.py:24-39 已經比較 R18hard、event_marginal、
event_exact、mixture_marginal、mixture_exact；固定 15-candidate union
也讓 exact subset DP 成本可控，這是可沿用的實驗形狀。

但有三個限制：

1. screen_ci_ap.py:30 使用 constant_denominator=True；這不是 random
   cardinality 下的 official AP。
2. screen_ci_ap.py:31 以 .5*a+.5*teach/5 與
   .5*b+.5*outer(teach,teach)/5 建立 mixture surrogate。它不是由
   ACT/PLANT 或 t37 (u,rho) posterior 生成的 true mixture moments；
   outer(teach,teach) 也不能補回 shared latent correlation。
3. screen_ci_ap.py:40-42 對 empty truth 沒有 zero guard，會以 len(truth)
   作 denominator；它只適合作為既有 CI positive-pair screen，不能外推
   到 empty-capable deployment decoder。

目前 tracked worktree 沒有 ci_exact_ap_screen.json 或 F4 exact-AP receipt，
本輪不重建該結果；Round20 ci_oracle_audit.py 已產生 observed OOF
.70909、ORACLE fixedset .75435、ORACLE top10 .94783。這組新 audit
數字只支持「candidate shortlist／候選宇宙是主要杠杆」的 CI control
結論，不支持「對原本五手做一次 marginal reorder 就足夠」，也不支持
F4 gain。

### 3.6 cardinality_prior.py：不是 official AP 的無條件 decoder

cardinality_prior.py:14-24 明確計算 P(hand is first K | total events >= K)。
這只有在 route 已有可信先驗保證 P(|G|≥5)=1，或問題明確改成該條件
分布時才合理。若 |G| 可以是 0、1、...、4，它會重加權 posterior，
卻沒有同步處理 official AP 的 min(5,|G|) denominator；不可直接替換
R18/R19 evidence decoder。

## 4. F4 對 CI negative/control scope 的 delta

目前能安全說的分界如下：

| scope | worktree 證據 | 可支持的結論 |
|---|---|---|
| R18 CI | docs/36 與 ledger 記載 CI-specific right-censored pool-OOF E=0.7090942；generic DT/SP extension 低於既有 baseline | censoring 對 CI 有 family-specific signal；不能視為 F4 通用證據 |
| R19 CI AP screen | screen_ci_ap.py 有 exact-vs-marginal scaffold；本 checkout 沒有其 result receipt | 可重用 protocol；不是 Round20 CI oracle 結果 |
| Round20 CI oracle audit | ci_oracle_audit.py：observed OOF .70909、ORACLE fixedset .75435、ORACLE top10 .94783 | 支持 shortlist／candidate-universe 是主要杠杆；不能外推成 F4 mixture gain |
| F4 t34/t37 | ledger 的 R3-F13/R3-F15 是 held-out likelihood 與 self-consistent simulation；例如 M3 truth 的 ACT/PLANT simulation 改善，以及 RE-vs-fixed simulation 改善 | 支持 random-effects 機制有 model-based feasibility；不是 actual F4 AP |
| F4 exact posterior-mixture AP | capped_ap building block 已存在；t37 node/rule mixture integration 尚未執行 | 尚無 actual feasibility verdict |

建議在 GB10 的最小實資料 probe 定義：

    Delta_F4 = mean( U_mix(exact_slate) - U_mix(current_t37_slate) )

其中兩個 slate 必須在同一個 t37 posterior、同一個 candidate shortlist、
同一個 full-hand event coverage、同一個 ACT/PLANT/WITNESS semantics 與
official constant_denominator=False 下評分。這個 delta 不需要 F4 truth；
它是同一 posterior 下 exact slate 相對 current slate 的 expected-utility
gain。若未來取得 F4 truth，另行報告兩個 slate 的 observed AP，不能把
observed AP 與 U_mix delta 混成同一指標。CI control 用完全相同的
decoder protocol 計算 Delta_CI；CI 只作 negative/control slice，不把
它的 family-specific gain 借給 F4。報告至少包括 paired mean、
wins/losses、empty-rate、candidate Recall@M、每個 pool/family strata
與 node discarded mass。

若沒有 actual F4 labels，則只能做 prior-predictive 或 posterior
self-consistent simulation，並把結果標成 simulation；不能稱為 actual
F4 AP 或 LB feasibility。t37 現有 simulation 也必須先移除 empty
continue 才能與 official expected AP 對齊。

## 5. 建議的 exact / small-beam decoder

### 5.1 M≤18：直接沿用 capped_ap.py

對一個 pair、rule hypothesis h 與 posterior node z：

1. 用完整按 timestamp 排序的 hands 建立 p_h,z。ACT 可用
   q_act*pairwin，PLANT 可用 q_plant*pairwin；WITNESS 必須用其獨立
   witness event probability。pairwin 或 witness gate 的 semantics 必須
   與真實 annotation rule 一致。
2. 只把 shortlist candidate 的 full-sequence positions 傳給 moments；
   candidate 外 hands 仍留在 p_h,z，讓 pre/suffix 與 denominator 正確。
3. 計算 a_h,z、b_h,z，並以 pair posterior w_z 與 rule posterior π_h
   做：

       A = Σ_h Σ_z π_h w_z a_h,z
       B = Σ_h Σ_z π_h w_z b_h,z

4. 呼叫 optimize(A,B,k=5)，取得同時考慮 subset 與 order 的 slate。
5. 用 expected_ap(order,A,B) 對 best value 做 equality check；以
   all-node / no-prune toy case 作 numerical regression。

這個方法只需要二階 moments，因為 AP 的 numerator 對 relevance
indicators 是二次式；denominator 已在 a、b 內正確 marginalize。

### 5.2 M>18：small beam，仍以 expected utility 擴展

對 prefix S=(l1,...,lr) 使用：

    F(S) = Σ_{t=1..r} ( A_{l_t}
                       + Σ_{s<t} B_{l_s,l_t} ) / t

每一層從所有未使用 candidate 擴展，保留 beam B 個 F 最大 prefix；最後
在長度 5 取 F 最大者。這是「對 expected AP 做 beam」，不是對 q 做
beam。為避免 mixture mode 被單一路徑吃掉，可保留：

- F top prefixes；
- 每個 rule/node family 的 marginal top prefix；
- 至少一個 current-t37 slate prefix；
- pairwise synergy 高的 prefix。

M≤18 的 exact result 應作 beam calibration set；若 beam 在 exact set 上
常漏掉最優 slate，就不能把大 M 結果稱 exact。第一個低成本版本可沿用
screen_ci_ap.py 的 M=10–15 union，先把 denominator、true mixture moments
與 empty handling 修正，再移植到 F4。這不是對 t37 原本五手做 reorder：
候選 union 內的成員也必須由 exact subset DP／beam 重新選取。

### 5.3 shortlist 與 ordering 必須分開報

Round20 ci_oracle_audit.py 的 .75435 ORACLE fixedset 與 .94783 ORACLE
top10 應被解讀為 shortlist／candidate-universe signal，而非原五手
reorder gain。候選 union
先報 Recall@M / oracle AP@5；decoder 再報 shortlist 內的 expected AP。
否則 exact DP 即使完美，也只是在錯的 slate universe 中最優。對 t37，
若從全 hand 改成 top-M：

- 不可把 shortlist 外 hand 的 p 清成 0；
- 不可把 shortlist 內的 expected AP gain 寫成 full-hand gain；
- 必須把 current top5、ACT、PLANT、WITNESS 的 union 規則固定在
  outer validation 之前。

## 6. Round 20 verdict

- Math：t37/R19 first-K marginal implementation 通過「membership DP」
  審查；不通過「correlated-mixture expected AP optimality」主張。
- Existing code：capped_ap.py 已提供可用 exact moments/subset-DP；
  screen_ci_ap.py 是可重用 scaffold，但 fixed denominator 與 surrogate
  mixture 必須修正。
- F4：random-effects model 有 likelihood/simulation feasibility signal，
  但 actual F4 expected AP delta 尚未存在；不得把 simulation 或 CI
  result 當作 F4 evidence。
- Safety：本輪未跑資料、未訓練、未 submission、未修改 shared queue/main。
- Output：本 task 由我新增的檔案只有本 audit；其他 concurrent main-writeup
  變更均未觸碰。未建立 commit 或 round handoff zip，以遵守本 task 的
  exclusive-output 限制。
