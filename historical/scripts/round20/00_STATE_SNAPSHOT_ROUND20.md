# Round 20｜外部審查狀態快照

時間：2026-09-20 00:41–00:49 UTC（台北08:41–08:49）。
起始 source：e8750f80ca0cbb1b0651b3825df9713fcc494a2b。
主文件：docs/39_breakthrough_research_20260920.md。

## 已確認數字

- 最佳 public 0.92501，thisray 暫列第8；Top-1 0.93876，差0.01375。
- 今日4/5已用，截止前僅剩1次。截止09-20 22:00 UTC；下一次重置09-21 00:00 UTC，來不及。
- 本輪沒有提交、模型訓練或全資料推論。GB10只跑read-only候選比對、92個CI query的oracle。
- R12=0.92675−I；I是risk與behavior的交互作用，尚不可由公開分數識別。
- CI當前OOF .7090942；固定五手重排oracle .7543478；top10選五oracle .9478261；top20 .9826087。這些是上限，不是可部署增益。
- 真證據有12手不滿足六人active gate；另有8手不在top20。
- 完全相同R15 CI排名：完整真值分母 .6542391，候選內分母 .6648007；後者是診斷偏差。

## 候選血統與完整 SHA-256

所有CSV仍僅在GB10：
/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates/

| 檔案 | 狀態 | SHA-256 |
|---|---|---|
| r2n_NDw_on_r2j2m.csv | 歷史對照 | 0afd59cc842611bd88ecf533b4b969dfc8e6a92fd30137b808735b112a7f37e4 |
| r9_subh.csv | 歷史對照 | b930b8c6ecfe79f286b005599d1b11f00d62c8afd93b3e7eb15607a2dac48403 |
| r10_ci.csv | public最佳0.92501 | 437da364b0975a1b6ed6c54cb48c0d3db73c82b9fb630fe8b703dbdfcf75506c |
| r11_ci_rank.csv | 歷史對照 | c9835c02ad6c4e6706e011aa785d653c49394839716d163e811bc482daff1f4b |
| r12_ndwrank_ci_f4.csv | 未提交；本輪不建立新候選 | 4139bb5ea92ca4676c031f4650c07f3ab442fa04ed4990318dc9348eff9a51b4 |

## 最重要的外部審查問題

1. 最後一個slot的R12是否值得，如何在不使用hidden labels下評估joint P/B interaction風險？
2. CI是否該用局部swap chooser，而非繼續擴召回或只重排原五手？哪些completion差分特徵可區分同樣raise＋六人active的真／假？
3. 五人active真證據是否反映不同合法trigger？如何軟化gate而不增加更多假手？
4. F4 M3的posterior-marginal取前五，改為posterior expectedAP選擇能否在模型錯設下保留收益？
5. policy OOT/RV1之後仍有style/count self-inclusion，該如何cross-fit composite gameplay null？
6. 哪些既有rerank負結果需要僅重算完整分母／pool folds就重新判讀？

## 執行／規則邊界

只從可觀察牌局與公開dev labels建方法；不利用ID、row/file ordering、生成器內部、不手工為eval records指定答案。大型實驗／提交交實驗worker。ZIP不包含競賽資料、submission CSV、權重或secret；receipts包含路徑與checksum。解壓後可執行 shasum -a 256 -c SHA256SUMS.txt 驗證每一檔。
