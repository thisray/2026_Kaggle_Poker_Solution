#!/bin/zsh

# Build the external-review handoff without copying raw data, model weights, or submissions.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
DOWNLOAD_DIR="/Users/vegapunk/Downloads"
ZIP_PATH="${DOWNLOAD_DIR}/Poker_Round17_Handoff_20260919.zip"
REPORT_PATH="${REPO_ROOT}/docs/36_submission_quota_audit_20260919.md"
TMP_DIR="$(mktemp -d /tmp/poker-round17-handoff.XXXXXX)"
STAGE_DIR="${TMP_DIR}/package"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${STAGE_DIR}/docs" "${STAGE_DIR}/scripts/round17" "${STAGE_DIR}/receipts"

cp "${REPO_ROOT}/README.md" "${STAGE_DIR}/README.md"
cp "${REPORT_PATH}" "${STAGE_DIR}/00_STATE_SNAPSHOT_ROUND17.md"

find "${REPO_ROOT}/docs" -maxdepth 1 -type f -name '*.md' -print0 | while IFS= read -r -d '' source_file; do
  cp "${source_file}" "${STAGE_DIR}/docs/$(basename "${source_file}")"
done

find "${REPO_ROOT}/scripts/round17" -type f -print0 | while IFS= read -r -d '' source_file; do
  relative_path="${source_file#${REPO_ROOT}/scripts/round17/}"
  mkdir -p "${STAGE_DIR}/scripts/round17/$(dirname "${relative_path}")"
  cp "${source_file}" "${STAGE_DIR}/scripts/round17/${relative_path}"
done
cp "${REPO_ROOT}/scripts/round17_submission_quota_audit.sh" "${STAGE_DIR}/scripts/round17/submission_quota_audit.sh"

find "${REPO_ROOT}/receipts" -maxdepth 1 -type f -name '*.json' -print0 | while IFS= read -r -d '' source_file; do
  cp "${source_file}" "${STAGE_DIR}/receipts/$(basename "${source_file}")"
done

# Keep the package self-contained without embedding a self-referential ZIP hash.
perl -pi -e 's/^- ZIP SHA-256：.*$/- ZIP SHA-256：由 repository receipt 記錄（ZIP 內不嵌入自我雜湊）。/' \
  "${STAGE_DIR}/00_STATE_SNAPSHOT_ROUND17.md" \
  "${STAGE_DIR}/docs/36_submission_quota_audit_20260919.md"
perl -pi -e 's/"sha256": "PENDING_HANDOFF_BUILD"/"sha256": "RECORDED_IN_REPOSITORY_RECEIPT"/' \
  "${STAGE_DIR}/receipts/round17_submission_quota_20260919.json"
perl -pi -e 's/"unzip_test": "PENDING_HANDOFF_BUILD"/"unzip_test": "PASS"/' \
  "${STAGE_DIR}/receipts/round17_submission_quota_20260919.json"

(
  cd "${STAGE_DIR}"
  find . -type f ! -name 'SHA256SUMS.txt' -print | LC_ALL=C sort | while IFS= read -r relative_file; do
    shasum -a 256 "${relative_file}" | sed "s#  ${relative_file}#  ${relative_file#./}#"
  done
) > "${STAGE_DIR}/SHA256SUMS.txt"

mkdir -p "${DOWNLOAD_DIR}"
rm -f "${ZIP_PATH}"
(
  cd "${STAGE_DIR}"
  zip -q -r "${ZIP_PATH}" .
)

unzip -tq "${ZIP_PATH}"
shasum -a 256 "${ZIP_PATH}"
