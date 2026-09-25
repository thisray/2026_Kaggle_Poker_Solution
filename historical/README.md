# Historical Competition Source

This directory preserves the competition-time source code for review. It is a source archive, not a second one-click runner. The current portable entry point is [`../run_all.py`](../run_all.py); its reproduction limits are stated in the repository README.

The files are copied from the private research repository at source commit `f871072`. `scripts/` contains the original competition scripts, `src/` the supporting package, `tests/` the small source tests, and `recovered/` the GB10 worker scripts that were recovered after the competition. No competition data, trained weights, checkpoints, submission CSVs, credentials, or large feature tables are included. Original scripts may retain GB10-specific absolute paths and historical experiment settings; review them as the executed method record, not as portable commands to run unchanged.

## Selected-submission source map

| Component | Historical source |
| --- | --- |
| Shared raw export, policy, pair and hand features | `scripts/opus_r1/`, `src/pokerlab/` |
| Round11 evidence ranker and scoped candidate patch | `scripts/round8/`, `scripts/round11/` |
| TabICLv2 fitting, evaluation features and rank blend | `scripts/round15/` |
| r10 fourth-family routing, monotone insertion and NDw evidence | `scripts/opus_r2/c1_build_r2a.py`, `c7_build_monotone.py`, `c14_extend_tables.py`, `c15_build_ndw.py` |
| r10 censored-event CI evidence and final assembly | `scripts/r18_chatgpt/`, `scripts/opus_r3/t47_ndw_ci.py` |
| r32 risk fusion, typed evidence and final zoo patches | `scripts/opus_r3/`, `scripts/opus_r4/`, `scripts/opus_r5/` |
| GB10-only model and execution source recovered later | `recovered/gb10_opus_r1/`, `recovered/gb10_opus_r4/`, `recovered/gb10_opus_r5/` |

The selected r10 and r32 files were assembled over multiple competition-time stages. The source is now present for inspection, including original training and patch code; not every historical command or checkpoint has been consolidated into the portable `run_all.py` path. Do not interpret the portable runner's current output as a verified reproduction of the leaderboard scores.
