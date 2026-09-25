# Running the solution code

## A. Primary prize-verification reproduction — `run_complete.py`

The target is one command from the eight official competition files through preprocessing, training, inference, and validation. Use an isolated Python 3.11 conda environment on a host with sufficient CPU memory and scratch disk. TabICLv2 defaults to CPU, as in the recorded GB10 environment:

```bash
conda create -n poker-solution python=3.11.16 pip -y
conda activate poker-solution
python -m pip install -r requirements.txt -r requirements_tabicl.txt
python run_complete.py \
  --data-dir /path/to/competition-data \
  --output-dir /path/to/output \
  --tabicl-checkpoint /path/to/tabicl-classifier-v2-20260212.ckpt
```

**Verification status:** the r32 manifest names 64 old and five new risk models. `solution/production/r25_training_map.json` maps all 69 to source code; several original shell configurations are explicitly marked as reconstructed. A fresh run from the eight files was stopped during policy inference to prioritize publication. The complete raw-data path has not yet produced two fresh CSVs, so no similarity or new score is claimed.

The target command needs only the eight official files and the public TabICLv2 checkpoint. `--download-public-checkpoint` explicitly permits the package to fetch it. The command does not submit to Kaggle or change repository visibility. Its intended outputs are `r10_ci.csv`, `r32_r30_dtgb15.csv`, and `run_complete_report.json`; the validator checks pair count, evidence uniqueness, evaluation phase, and shared pair membership. Plan for many hours on a large-memory Linux CPU host.

## B. Exact historical final-assembly audit — `run_historical_assembly.py`

This optional audit needs saved competition-time intermediate CSVs. It is separate from fresh training reproduction:

```bash
python run_historical_assembly.py \
  --artifact-root /path/to/saved-competition-intermediates \
  --output-dir /path/to/separate-output-directory
```

| Path relative to `--artifact-root` | Assembly role |
| --- | --- |
| `r2_candidates/r2n_NDw_on_r2j2m.csv` | r10 pre-CI base |
| `r18/main/patch_r18hard.csv` | r10 coordinated-isolation patch |
| `r5/cand/r30_r29_spzoo.csv` | r32 pre-DT base |
| `r5/patch_r5_dt_zoo_gb15.csv` | r32 directed-transfer patch |

The GB10 audit matched both original hashes. Its receipt records input and output hashes and applied patch counts. These saved files are not redistributed and are not inputs to the primary command.

## C. Legacy partial reconstruction — `run_all.py`

```bash
python run_all.py --data-dir /path/to/competition-data --output-dir outputs
```

This older path was run from the eight files and produced legal but partial reconstructions. It omits parts of the selected r10/r32 evidence and risk methods. The output names are reconstruction targets, not verified copies of the leaderboard submissions.
