# Round 16 handoff builder

`build_handoff.sh` packages the research report, accumulated Markdown documents, Round 16 scripts, and JSON receipts for external review.

The builder deliberately excludes raw competition data, model weights, credentials, and submission artifacts. It uses an explicit local workspace path and writes the ZIP to `/Users/vegapunk/Downloads/Poker_Round16_Handoff_20260919.zip`.
