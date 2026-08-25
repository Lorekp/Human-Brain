#!/usr/bin/env bash
# Windowed-decoder retro-probe of the U-STATE archives (A4 diagnostic tier).
#
# The week-1 logged_states.npz binaries were local-only and are no longer on
# disk; the committed manifests carry per-array sha256s, so this script
# regenerates both archives DETERMINISTICALLY (seed 7, same code path) and the
# probe itself refuses to score anything until every array hash matches the
# committed manifest and the rebuilt replay segmentation reproduces the
# committed curriculum digest. Verified regeneration == the archived states.
#
# CPU-only; measured ~0.7 GFLOPS on the throttled host, hence the reduced
# (disclosed) logistic budget on the primary `union` family while exact kernel
# ridge carries the full depth curve everywhere at full window dimensionality.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRATCH="${LOCALAPPDATA:-$HOME/AppData/Local}/Temp/hb_win_probe"
FREEZE="${LOCALAPPDATA:-$HOME/AppData/Local}/Temp/hb_bible_freeze"
mkdir -p "$SCRATCH/byte" "$SCRATCH/concept" "$REPO/experiments/windowed_probes"

cp "$REPO/experiments/ustate_probes/capacity_precondition.json" "$SCRATCH/byte/"

cd "$REPO"
export PYTHONPATH=src OPENBLAS_NUM_THREADS=2

# 1) regenerate archives (deterministic)
python experiments/ustate_probes/log_states_union.py --encoding byte    --out-dir "$SCRATCH/byte"
python experiments/ustate_probes/log_states_union.py --encoding concept --out-dir "$SCRATCH/concept"

# 2) windowed sweep k in {1,4,16,64} x families {union,count_union}
COMMON=(--freeze-dir "$FREEZE"
        --depths 1,4,16,64 --families union,count_union
        --logistic-families union --logistic-iters 250 --logistic-decay-every 125
        --max-logistic-d 1024)

OPENBLAS_NUM_THREADS=4 python -m humanbrain.windowed_probe \
  --npz "$SCRATCH/byte/logged_states.npz" \
  --manifest experiments/ustate_probes/logged_states_manifest.json \
  --encoding byte "${COMMON[@]}" \
  --out experiments/windowed_probes/windowed_probes_byte.json &

python -m humanbrain.windowed_probe \
  --npz "$SCRATCH/concept/logged_states.npz" \
  --manifest experiments/ustate_probes_concept/logged_states_manifest.json \
  --encoding concept "${COMMON[@]}" &
wait

echo "done: experiments/windowed_probes/windowed_probes_{byte,concept}.json"
