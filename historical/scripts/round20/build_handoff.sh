#!/bin/zsh
# Build and verify a documentation-only handoff. Never overwrite an existing ZIP.
set -euo pipefail
script_dir="${0:A:h}"
repo_root="${script_dir:h:h}"
zip_path="${1:-/Users/vegapunk/Downloads/Poker_Round20_Handoff_20260920.zip}"
[[ ! -e "$zip_path" ]] || { print -u2 "Refusing to overwrite: $zip_path"; exit 1; }
scratch_dir="$(mktemp -d /tmp/poker-round20-package.XXXXXX)"
[[ "$scratch_dir" == /tmp/poker-round20-package.* ]] || exit 1
cleanup() {
  [[ -d "$scratch_dir" && "$scratch_dir" == /tmp/poker-round20-package.* ]] && rm -rf -- "$scratch_dir"
}
trap cleanup EXIT
stage_dir="$scratch_dir/package"
mkdir -p "$stage_dir/docs" "$stage_dir/scripts" "$stage_dir/receipts" "$stage_dir/src/pokerlab"
cp "$script_dir/README.md" "$stage_dir/README.md"
cp "$script_dir/00_STATE_SNAPSHOT_ROUND20.md" "$stage_dir/00_STATE_SNAPSHOT_ROUND20.md"
cp "$repo_root"/docs/*.md "$stage_dir/docs/"
cp -R "$script_dir" "$stage_dir/scripts/round20"
# Copy only source files: no data, candidate CSVs, model files, or caches.
for source_dir in opus_r1 opus_r2 opus_r3 r18_chatgpt r19_chatgpt; do
  mkdir -p "$stage_dir/scripts/$source_dir"
  find "$repo_root/scripts/$source_dir" -maxdepth 1 -type f -name '*.py' -exec cp {} "$stage_dir/scripts/$source_dir/" \;
done
cp "$repo_root/src/pokerlab/metrics.py" "$stage_dir/src/pokerlab/"
for receipt in "$repo_root"/receipts/*.json; do
  [[ "${receipt:t}" == round20_package_integrity_20260920.json ]] && continue
  cp "$receipt" "$stage_dir/receipts/"
done
(
  cd "$stage_dir"
  find . -type f ! -name SHA256SUMS.txt | LC_ALL=C sort | while IFS= read -r file; do
    shasum -a 256 "${file#./}"
  done
) > "$stage_dir/SHA256SUMS.txt"
mkdir -p "${zip_path:h}"
(cd "$stage_dir" && zip -q -r "$zip_path" .)
unzip -tq "$zip_path"
verify_dir="$scratch_dir/verify"
mkdir "$verify_dir"
unzip -q "$zip_path" -d "$verify_dir"
(cd "$verify_dir" && shasum -a 256 -c SHA256SUMS.txt)
print "Manifest entries: $(wc -l < "$verify_dir/SHA256SUMS.txt" | tr -d ' ')"
print "ZIP SHA-256:"
shasum -a 256 "$zip_path"
