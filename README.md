# Detect Suspicious Value Transfers in Poker — Solution and Reproduction Status

This repository documents our competition-time solution and provides an executable **partial reconstruction**. The current code has not been verified as a complete reproduction of either final-evaluated submission. Please read the distinction below before attributing the historical leaderboard scores to this code.

## Competition result

| Item | Verified record |
| --- | --- |
| Team | `thisray` |
| Provisional private rank | 6th of 370, score `0.92888` |
| Public score | `0.92501` |
| Leaderboard-producing submission | `r10_ci.csv`, submission ref `56374362` |
| Original r10 SHA-256 | `437da364b0975a1b6ed6c54cb48c0d3db73c82b9fb630fe8b703dbdfcf75506c` |

We made no manual final selections, so Kaggle automatically evaluated our two highest-public-score submissions: r10 and `r32_r30_dtgb15.csv` (ref `56406417`; public `0.92456`, private `0.92738`). r10 produced the displayed team result. An earlier NDw base scored slightly higher on the private split (`0.92893`), but was not among the two automatically selected submissions.

## Competition-time method

The metric combined pair ranking (70%), evidence retrieval (20%), and behavior classification (10%). Features described poker activity—actions, bet sizes, positions, stacks, cards, boards, outcomes, timing, and responses to opponents. Identifiers were used for joins, grouping, folds, and stable ordering, not as predictive features.

The r10 pair-risk base blended the ranks of one LightGBM and two CatBoost models with weights `0.50/0.25/0.25`. A separate classifier predicted the three disclosed behavior families. A partner-card-dependence signal nominated 77 evaluation pairs for the residual `other_coordination` route. Historical evidence retrieval used family-specific rankers, decision-context adjustments, a TabICLv2 blend, and a separate NDw rule for that residual route. The final r10 CI patch was applied to 591 routed pairs and changed 564 ordered five-hand lists; its pair risks and predicted behaviors were unchanged from NDw.

The [Solution Writeup](https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/writeups/ranking-suspicious-poker-pairs-and-finding-reviewa) is currently a **private draft**, not a published article.

## What the code currently reproduces—and what it does not

`run_all.py` reads the eight official competition files, builds features, trains the three-model pair ensemble and three-family classifier, restores the fourth-family signal, monotone rank insertion, and NDw evidence rule, generates m19 baseline evidence, applies a CI adapter, and validates output format and hand membership. It does not read a historical prediction CSV.

This is **not yet the complete selected-submission method**. The r10 fourth-family branch has been restored, but a fresh run routed 74 of the original 77 pairs. More importantly, the historical family-ranker/TabICLv2 evidence chain is still missing. The r32-named output changes weights and baseline evidence within a smaller ensemble; it does not reconstruct the historical r32 risk, directed-transfer, soft-play, or residual-family ensembles. The output filenames name reconstruction targets, not verified equivalents of the submitted files. Missing method branches cannot be explained by floating-point nondeterminism.

The original Round11 ranker training code, TabICLv2 fit/predict code, and exact R15 rank-blend function are included as standalone modules, but `run_all.py` does not use them yet: their upstream candidate-score and feature-table builders have not been restored in this repository.

Standalone modules also document the historical r25 64+5 risk fusion and r32 family-routed evidence patch assembly. They require model-score and patch inputs that `run_all.py` does not regenerate; they are not a shortcut around the missing upstream training and inference stages.

## Data and setup

Download the competition data from Kaggle and place these eight files in one directory: `players.parquet`, `hands.parquet`, `seats.parquet`, `actions.parquet`, `development_labels.csv`, `development_evidence.csv`, `evaluation_pairs.csv`, and `sample_submission.csv`. Competition data are not redistributed here.

The recorded GB10 environment was Ubuntu 24.04.4 (aarch64), Python 3.11.16. For the current partial runner, use an isolated conda environment:

```bash
conda create -n poker-solution python=3.11.16 pip -y
conda activate poker-solution
python -m pip install -r requirements.txt
python run_all.py --data-dir /path/to/competition-data --output-dir outputs
```

The default `--variant all` writes two reconstruction-target CSVs and `outputs/run_report.json`. The validator checks 112,540 unique pairs, score and behavior values, nonempty cells, duplicate evidence IDs, evaluation-phase hands, and both pair members' seats. Passing these checks does **not** establish a similar leaderboard score. `requirements.txt` covers only the current partial runner; restoring the historical TabICLv2 path also requires its package, checkpoint, provenance, and license information.
The standalone TabICL module has additional versions in `requirements_tabicl.txt`; those dependencies are not needed for the current `run_all.py` path.

## Measured reconstruction status

Private Kaggle CPU Notebook v6 reran this code from the eight official files in `33,918.728` seconds. Both outputs passed submission-legality checks. Against the original submitted files:

| Reconstruction target | Risk Spearman | Behavior rows differing | Mean shared evidence IDs | Exact five-hand sets |
| --- | ---: | ---: | ---: | ---: |
| r10 | 0.969198 | 3 / 112,540 | 3.4380 / 5 | 17.0197% |
| r32 | 0.966357 | 87 / 112,540 | 3.4342 / 5 | 16.8687% |

The r10 original top-300 pair set overlaps the replay by 285 pairs. For the original 77 fourth-family pairs, mean evidence overlap is 4.805/5, with three behavior mismatches. The r32 original top-300 overlap is 216 pairs, and all 87 original residual-family behaviors still differ. Neither replay has a new competition score. Similarity figures cannot be converted into private Pair AP or Evidence MAP@5. Complete historical-method reproduction and a similar-score check remain open.

## Evidence reviews and license

The [five case reviews](docs/CASE_REVIEWS.md) use hand IDs actually submitted in the original r10 file. Each separates observable play from a plausible benign explanation. They are not claims about player intent, and the current partial reconstruction may select different hands.

Original code in this repository is under the [MIT License](LICENSE). Competition data remain subject to Kaggle's rules and are not distributed here.
