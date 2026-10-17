#!/bin/bash
# Back up run roots, custody (lock secrets) and models to an external volume, then verify.
# Usage: scripts/backup.sh /Volumes/<drive>
set -euo pipefail
dest="${1:?usage: backup.sh /Volumes/<drive>}"
[ -d "$dest" ] || { echo "not mounted: $dest" >&2; exit 1; }
out="$dest/hsi-lab-backup"
mkdir -p "$out"
for d in hsi-runs hsi-custody hsi-models; do
  rsync -a --delete "$HOME/$d/" "$out/$d/"
  diff <(cd "$HOME/$d" && find . -type f -exec shasum -a 256 {} + | sort -k2) \
       <(cd "$out/$d" && find . -type f -exec shasum -a 256 {} + | sort -k2) >/dev/null \
    && echo "OK $d" || { echo "MISMATCH $d" >&2; exit 1; }
done
chmod -R go-rwx "$out/hsi-custody"
