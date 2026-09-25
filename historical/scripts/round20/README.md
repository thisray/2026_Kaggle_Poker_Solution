# Poker Round 20 審查包

先讀 00_STATE_SNAPSHOT_ROUND20.md，再讀 docs/39_breakthrough_research_20260920.md；最後的 Experiment Handoff 是實驗worker的接手順序。

- docs/：累積研究，日期較早者是歷史，不代表現在最佳。
- scripts/round20/：兩個GB10 read-only probes、三份平行研究memo、打包腳本。
- scripts/opus_r3/、scripts/r18_chatgpt/、scripts/r19_chatgpt/：本輪引用的重要source；不是完整訓練runtime。
- src/pokerlab/metrics.py：repo metric實作；與官方來源連結一起檢查。
- receipts/：已執行探針的aggregate輸出、官方狀態與hash。
- SHA256SUMS.txt：解壓後執行 shasum -a 256 -c SHA256SUMS.txt 可核對全部內容。

不含raw data、features、weights、credentials或submission CSV。兩個probe只能在GB10 conda環境對既有artifact路徑重跑；不應在M4下載資料。它們不寫入候選、不提交。

ZIP自身hash保存在repository的 receipts/round20_package_integrity_20260920.json；它在ZIP產生後建立，不置入自身ZIP，避免自我參照。ZIP中的其他receipts均為打包當下原樣，不改寫數值。

重現指令（從repo root執行；只有scalar/aggregate JSON回到本機）：

~~~
ssh -o BatchMode=yes thisray@aitopatom-9a11.local '/home/thisray/miniforge3/bin/conda run --no-capture-output -n kaggle_poker_opus_260917 python - --root /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r2_candidates --source-sha e8750f80ca0cbb1b0651b3825df9713fcc494a2b' < scripts/round20/candidate_audit.py
ssh -o BatchMode=yes thisray@aitopatom-9a11.local '/home/thisray/miniforge3/bin/conda run --no-capture-output -n kaggle_poker_opus_260917 python - --root /home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r18' < scripts/round20/ci_oracle_audit.py
~~~

CI oracle使用既有seed260919、81 pools／92pairs的探索性OOF；輸入receipt驗證每對五手完整真值。並非新的independent holdout或提交結果。
