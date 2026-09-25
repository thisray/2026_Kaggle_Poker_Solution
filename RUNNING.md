# Running the solution code

There are two supported entry points. Choose the one that matches the inputs you have. Neither command publishes a Kaggle writeup, submits to Kaggle, or changes this repository's visibility.

| Entry point | Inputs | Output | Scope |
| --- | --- | --- | --- |
| `run_historical_assembly.py` | Four saved competition-time intermediate CSVs | The two **selected historical submission CSVs** | Exact final assembly; does not retrain upstream models |
| `run_all.py` | Eight official competition files | Two **legal partial-reconstruction CSVs** | Rebuilds part of the method; does not reproduce the selected submissions |

## A. Assemble the selected submissions

Use an isolated Python 3.11 conda environment with `pandas` installed. On our GB10, the four inputs live under `/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917`. A reviewer with those saved intermediate files can use any equivalent root directory. Competition-derived files are not redistributed in GitHub.

| Path relative to `--artifact-root` | Role |
| --- | --- |
| `r2_candidates/r2n_NDw_on_r2j2m.csv` | r10 pair-risk, behavior, and pre-CI evidence base |
| `r18/main/patch_r18hard.csv` | r10 CI evidence patch |
| `r5/cand/r30_r29_spzoo.csv` | r32 risk, behavior, and pre-DT-zoo evidence base |
| `r5/patch_r5_dt_zoo_gb15.csv` | r32 directed-transfer evidence patch |

```bash
conda create -n poker-solution python=3.11.16 pip -y
conda activate poker-solution
python -m pip install -r requirements.txt
python run_historical_assembly.py \
  --artifact-root /path/to/saved-competition-intermediates \
  --output-dir /path/to/separate-output-directory
```

The output directory receives `r10_ci.csv`, `r32_r30_dtgb15.csv`, and `historical_assembly_receipt.json`. The receipt records input and output hashes and applied patch counts. Do not place the output directory inside the artifact root. On GB10, this command yielded 112,540 rows in each CSV; r10 applied the CI patch to 591 pairs and had SHA-256 `437da364b0975a1b6ed6c54cb48c0d3db73c82b9fb630fe8b703dbdfcf75506c`; r32 applied the DT patch to 1,529 pairs and had SHA-256 `97478ea2553168a6f3fe0209d898996a2bba1e70b9f871e3eb36eca7cd4a4e5a`. These match the historical selected files. This verifies the **last assembly step**, not the full upstream training chain.

## B. Run the portable partial reconstruction from official data

Download these eight official files into one directory: `players.parquet`, `hands.parquet`, `seats.parquet`, `actions.parquet`, `development_labels.csv`, `development_evidence.csv`, `evaluation_pairs.csv`, and `sample_submission.csv`. After the same environment setup:

```bash
python run_all.py --data-dir /path/to/competition-data --output-dir outputs
```

This writes `outputs/r10_ci.csv`, `outputs/r32_r30_dtgb15.csv`, and `outputs/run_report.json`. The names denote targets only: these are **not** the selected historical files, and their leaderboard scores are unknown. The optional `--build-r15-cache --build-r5-models` flags train more original components but do not turn this path into a complete selected-submission reproduction. See [README.md](README.md) for measured differences.

## Where the remaining method code lives

The [historical source map](historical/README.md) points to the original raw features, risk ensemble, family evidence, TabICLv2, and r32 zoo-patch code. Those modules used competition-time paths, saved checkpoints, and intermediate feature tables. They are available to inspect or adapt, but we have **not** consolidated them into a maintained, single-command raw-data-to-selected-submission pipeline. Do not report path A as independent end-to-end reproduction from the eight official files.
