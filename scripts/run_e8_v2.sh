#!/bin/sh
# Pilot -> lock -> run -> eval -> verify for E8 v2, local MLX, ~6-8 h. Safe to re-run: stages resume.
set -e
cd "$(dirname "$0")/.."
export HSI_MLX_MEM_GB=${HSI_MLX_MEM_GB:-24} HSI_MLX_CACHE_GB=${HSI_MLX_CACHE_GB:-4}
P=packets/e8-v2.frozen.json
[ -z "$(git status --porcelain)" ] || { echo "repo dirty: commit or stash first"; exit 1; }
RD=$(uv run python -c "from pathlib import Path; from hsi.contracts.io import load_frozen; from hsi.pipeline import rundir; print(rundir(load_frozen(Path('$P'))[1]))")
[ -f "$RD/pilot.json" ] || uv run hsi pilot $P
[ -f "$RD/lock.public.json" ] || uv run hsi lock $P
[ -f "$RD/run/manifest.json" ] || uv run hsi run $P
[ -f "$RD/eval/receipt.json" ] || uv run hsi eval "$RD"
uv run hsi verify-receipt "$RD/eval/receipt.json"
echo "DONE: $RD/eval/receipt.json"
