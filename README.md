# Detect Suspicious Value Transfers in Poker — 解法與重建狀態

本專案記錄 Kaggle 競賽的實際提交方法，並提供目前可執行的部分重建程式。它尚未證明能完整重現兩份 final-evaluated submissions；請先閱讀下方的差異說明。

## 競賽結果

| 項目 | 已確認內容 |
| --- | --- |
| 隊伍 | `thisray` |
| Private Leaderboard | 暫列第 6／370，`0.92888` |
| Public Leaderboard | `0.92501` |
| 產生隊伍榜單分數的檔案 | `r10_ci.csv`，submission ref `56374362` |
| 原始 r10 SHA-256 | `437da364b0975a1b6ed6c54cb48c0d3db73c82b9fb630fe8b703dbdfcf75506c` |

Kaggle 顯示本隊沒有手動選定 final submissions，因此自動採用 Public 分數最高的兩份：r10 與 `r32_r30_dtgb15.csv`（ref `56406417`；Public `0.92456`，Private `0.92738`）。r10 是最後顯示的隊伍成績來源；較早的 NDw base 曾有 Private `0.92893`，但不在自動選入的 Public 前二名。

## 競賽時的方法

評分由配對排序、行為家族與手牌證據組成，權重分別為 70%、10% 與 20%。特徵來自牌局行為：action、下注額、位置、stack、牌面、底牌、結果、時間與對手反應。識別碼只用於 join、grouping、fold 與穩定排序。

原 r10 的 pair risk 基底是 LightGBM 與兩個 CatBoost 的 rank fusion（`0.50/0.25/0.25`）。另有三個已知行為家族的分類器；第四類 `other_coordination` 由評估期牌局中的 partner-card-dependence 訊號尋找，最後有 77 個 pair。歷史 evidence 鏈使用各家族 ranker、decision-context 修正與 TabICLv2 融合，第四類另採 NDw evidence。r10 最後的 R18 變更只替換 591 個 routed CI pairs 的候選 evidence，其中 564 列的五手順序實際改變；原檔的 pair risk 與 behavior 不變。

## 目前程式的範圍

目前 runner 可從八份官方資料建立特徵、訓練三模型 pair ensemble 與三類分類器、產生 m19 evidence、套用 CI adapter，並執行 submission 合法性檢查。它不讀取歷史 prediction CSV。

但原 r10 的完整第四類 discovery／rank insertion／NDw 路徑，以及 family-ranker／TabICLv2 evidence 鏈尚未接入。現在的 r32-named 輸出只調整較小 ensemble 的權重及 baseline evidence 變換，沒有重建原 r32 的大型 risk／DT／SP evidence ensembles。因此兩個輸出檔名是「重建目標」，不能視為與原 submitted artifacts 等價；缺失的模型階段不能用浮點 nondeterminism 解釋。

## 資料與環境

從 Kaggle 取得八份官方檔案並放在同一目錄：`players.parquet`、`hands.parquet`、`seats.parquet`、`actions.parquet`、`development_labels.csv`、`development_evidence.csv`、`evaluation_pairs.csv`、`sample_submission.csv`。本 repository 不包含競賽資料。

已記錄的 GB10 環境是 Ubuntu 24.04.4（aarch64）與 Python 3.11.16。在隔離的 conda 環境安裝目前 runner 的固定版本依賴：

```bash
conda create -n poker-solution python=3.11.16 pip -y
conda activate poker-solution
python -m pip install -r requirements.txt
```

這份 `requirements.txt` 只涵蓋目前的部分重建；原 TabICLv2 路徑還需要對應的套件、checkpoint、來源與授權資訊。

## 執行與已量測結果

在本目錄執行：

```bash
python run_all.py --data-dir /path/to/competition-data --output-dir outputs
```

預設 `--variant all` 產生兩個重建目標 CSV 與 `outputs/run_report.json`。validator 會檢查 112,540 個唯一配對、score／behavior 範圍、空格、重複 evidence、evaluation phase 及每手兩人入座；失敗時非零退出。這是提交格式與手牌合法性檢查，並非 Private score 驗證。

GB10 紀錄使用同一份官方資料 workspace，兩次 adapter 修正後從保存點續跑；合計 `3:51:51`，peak RSS 19.25 GiB。兩份輸出均通過合法性。與原 submitted files 比較：

| 重建目標 | Risk Spearman | Behavior 不同列數 | 平均共同 evidence IDs | 五手集合全同 |
| --- | ---: | ---: | ---: | ---: |
| r10 | 0.974487 | 77／112,540 | 3.4350／5 | 16.9504% |
| r32 | 0.968706 | 87／112,540 | 3.4337／5 | 16.8660% |

Kaggle private CPU Notebook v5 也完成執行與合法性檢查；它的輸出仍沒有新的 competition score。全體一致率與集合交集不能換算成 Private AP 或 EvidenceMAP@5。原方法完整重建及相近分數尚未確認。

## 五個案例與授權

[五個 submitted-evidence 案例](docs/CASE_REVIEWS.md)只使用原 r10 真正提交的 hand IDs，列出可觀察行動與合理的良性解釋；它們不代表目前部分重建選到的手牌，也不證明玩家意圖。

本 repository 原創程式使用 [MIT License](LICENSE)。競賽資料不隨程式散布，仍受 Kaggle 競賽規則約束。
