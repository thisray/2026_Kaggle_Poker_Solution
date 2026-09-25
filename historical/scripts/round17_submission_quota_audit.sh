#!/usr/bin/env bash
set -euo pipefail

# Read-only quota audit. The Kaggle token must already be present in the environment.
competition_slug="detect-suspicious-value-transfers-in-poker"
conda_environment="nv_kaggle_260320"
quota_script="/Users/vegapunk/.codex/plugins/cache/nvidia-kaggle/nvidia-kaggle/0.1.0/skills/nvidia-kaggle-skill/scripts/submission_quota.py"

exec conda run -n "${conda_environment}" python "${quota_script}" "${competition_slug}" \
  --by-user --by-day --overall --as-json
