# Round 18 handoff builder

`build_handoff.sh` packages the Round 18 breakthrough report, accumulated Markdown research documents, Round 18 scripts, and JSON receipts for external ChatGPT review.

The builder explicitly excludes raw competition data, model weights, credentials, and submission artifacts. It writes `/Users/vegapunk/Downloads/Poker_Round18_Handoff_20260919.zip` and creates an inner `SHA256SUMS.txt` manifest.
