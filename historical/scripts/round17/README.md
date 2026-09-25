# Round 17 handoff builder

`build_handoff.sh` 會打包本輪 quota audit 報告、累積的 Markdown research docs、Round 17 scripts 與 JSON receipts，供外部 ChatGPT review。

builder 明確排除 raw competition data、model weights、credentials 與 submission artifacts；ZIP 寫入 `/Users/vegapunk/Downloads/Poker_Round17_Handoff_20260919.zip`，並在 ZIP 內建立 `SHA256SUMS.txt`。
