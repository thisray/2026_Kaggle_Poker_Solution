# Detect Suspicious Value Transfers in Poker — Solution

The revised `run_complete.py` connects the recovered Round11 ranker, 24-feature TabICLv2 evidence view, 64+5 risk manifest, and family-specific r32 zoo patches. All 69 risk members have source mappings. Several original shell configurations were not preserved and are identified as reconstructed in the mapping. A fresh eight-file run was started and stopped during policy inference to prioritize the publication deliverables; the revised path has **not completed** end to end.

## Competition result

| Item | Verified competition-time record |
| --- | --- |
| Team | `thisray` |
| Provisional private rank | 6th of 370, score `0.92888` |
| Public score | `0.92501` |
| Displayed result | `r10_ci.csv`, submission `56374362` |
| Second automatically selected entry | `r32_r30_dtgb15.csv`, submission `56406417`; public `0.92456`, private `0.92738` |

The original selected CSV hashes are `437da364b0975a1b6ed6c54cb48c0d3db73c82b9fb630fe8b703dbdfcf75506c` (r10) and `97478ea2553168a6f3fe0209d898996a2bba1e70b9f871e3eb36eca7cd4a4e5a` (r32). Fresh retraining does not need to match these bytes.

## Primary reproduction target

Place the eight official files in one directory: `players.parquet`, `hands.parquet`, `seats.parquet`, `actions.parquet`, `development_labels.csv`, `development_evidence.csv`, `evaluation_pairs.csv`, and `sample_submission.csv`. On a capable Linux compute host with sufficient CPU memory and disk, the target command is:

```bash
conda create -n poker-solution python=3.11.16 pip -y
conda activate poker-solution
python -m pip install -r requirements.txt -r requirements_tabicl.txt
python run_complete.py \
  --data-dir /path/to/competition-data \
  --output-dir /path/to/output \
  --tabicl-checkpoint /path/to/tabicl-classifier-v2-20260212.ckpt
```

The command provides the full preprocessing, training, and inference path. Its end-to-end output remains unverified because our fresh run was stopped before completion. The public TabICLv2 checkpoint can be supplied locally or downloaded with `--download-public-checkpoint`. The pinned dependency is `tabicl==2.2.0`; the checkpoint `tabicl-classifier-v2-20260212.ckpt` is available from [jingang/TabICL](https://huggingface.co/jingang/TabICL/blob/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt) under BSD-3-Clause. The tested checkpoint SHA-256 is `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`. The model receipt records package version and checkpoint hash. Competition data, weights, and generated submissions are not redistributed.

TabICLv2 defaults to CPU inference and training, matching the recorded GB10 setup; `--tabicl-device cuda` is optional when a compatible CUDA Torch installation is available.

The intended method is raw preprocessing; pair-risk and behavior training; R5 family and sequence candidate models; the Round11 learned ranker; eleven candidate scores plus thirteen gameplay extras for TabICLv2; a 0.6 TabICL rank blend; the fourth-family NDw route; the r10 coordinated-isolation patch; and the r32 64+5 risk fusion with soft-play then directed-transfer zoo patches. Pair, player, and table IDs are joins and split keys, never model features.

The original selected files passed the competition validator and exact final-assembly audit. For the revised raw-data path, compilation, CLI checks, a 24-feature CPU TabICL fit, and a 20-hand Round11 moments pack passed. We have **not** measured fresh end-to-end output similarity or a fresh leaderboard score. Reviewers can run the documented command in a clean environment and request clarification if needed.

## Secondary audit and legacy diagnostic

`run_historical_assembly.py` is an optional exact final-assembly audit from four saved competition-time intermediates. On GB10 it reproduced both selected CSV hashes. This proves the last assembly steps, not upstream retraining. `run_all.py` is an earlier partial raw-data reconstruction, not the selected-submission method. See [RUNNING.md](RUNNING.md).

The [historical source archive](historical/README.md), [solution writeup](WRITEUP.md), and [five evidence case reviews](docs/CASE_REVIEWS.md) are included. Original source is under the [MIT License](LICENSE); competition data remain subject to Kaggle's rules.
