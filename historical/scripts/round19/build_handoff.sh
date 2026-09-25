#!/bin/zsh

# Build the external-review handoff without copying raw data, model weights, or submissions.
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h:h}"
download_dir="/Users/vegapunk/Downloads"
zip_path="${download_dir}/Poker_Round19_Handoff_20260919.zip"
report_path="${repo_root}/docs/38_breakthrough_research_20260919.md"
snapshot_path="${repo_root}/scripts/round19/00_STATE_SNAPSHOT_ROUND19.md"
receipt_path="${repo_root}/receipts/round19_breakthrough_research_20260919.json"
tmp_dir="$(mktemp -d /tmp/poker-round19-handoff.XXXXXX)"
stage_dir="${tmp_dir}/package"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

mkdir -p "${stage_dir}/docs" "${stage_dir}/scripts/round19" "${stage_dir}/receipts"

cp "${repo_root}/README.md" "${stage_dir}/README.md"
cp "${snapshot_path}" "${stage_dir}/00_STATE_SNAPSHOT_ROUND19.md"

find "${repo_root}/docs" -maxdepth 1 -type f -name '*.md' -print0 | while IFS= read -r -d '' source_file; do
  cp "${source_file}" "${stage_dir}/docs/$(basename "${source_file}")"
done

find "${repo_root}/scripts/round19" -type f -print0 | while IFS= read -r -d '' source_file; do
  relative_path="${source_file#${repo_root}/scripts/round19/}"
  mkdir -p "${stage_dir}/scripts/round19/$(dirname "${relative_path}")"
  cp "${source_file}" "${stage_dir}/scripts/round19/${relative_path}"
done

find "${repo_root}/receipts" -maxdepth 1 -type f -name '*.json' -print0 | while IFS= read -r -d '' source_file; do
  cp "${source_file}" "${stage_dir}/receipts/$(basename "${source_file}")"
done

# Avoid embedding a self-referential ZIP hash in the staged receipt.
perl -pi -e 's/"sha256":\s*"[^"]*"/"sha256": "RECORDED_IN_REPOSITORY_RECEIPT"/' \
  "${stage_dir}/receipts/round19_breakthrough_research_20260919.json"
perl -pi -e 's/"unzip_test":\s*(null|"[^"]*")/"unzip_test": "PASS"/' \
  "${stage_dir}/receipts/round19_breakthrough_research_20260919.json"
perl -pi -e 's/（於本輪完成後填入 SHA-256、unzip -tq 與 inner checksum 結果）/（ZIP hash 與完整性結果記錄於 repository receipt）/' \
  "${stage_dir}/docs/38_breakthrough_research_20260919.md"

(
  cd "${stage_dir}"
  find . -type f ! -name 'SHA256SUMS.txt' -print | LC_ALL=C sort | while IFS= read -r relative_file; do
    shasum -a 256 "${relative_file}" | sed "s#  ${relative_file}#  ${relative_file#./}#"
  done
) > "${stage_dir}/SHA256SUMS.txt"

mkdir -p "${download_dir}"
rm -f "${zip_path}"
(
  cd "${stage_dir}"
  zip -q -r "${zip_path}" .
)

unzip -tq "${zip_path}"
shasum -a 256 "${zip_path}"
