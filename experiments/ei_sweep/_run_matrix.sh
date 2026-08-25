#!/usr/bin/env bash
# Launch the ei_sweep matrix: 12 runs (fraction x seeds at N=1024) + in-process
# determinism re-check on every arm. Protocol pinned in PREREGISTRATION.md
# (written before execution): ridge / massed / data-scale 42 / all_transition /
# codebook explicit — identical settings to codebook_compare confirmatory cells.
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root of the worktree this lives in

PY="C:/Users/laure/Desktop/DeepSim/Human Brain/.venv/Scripts/python.exe"
OUT_ROOT="experiments"
LOG="experiments/ei_sweep/_run_log.txt"
echo "ei_sweep run log — start $(date -Is)" > "$LOG"

FAIL=0
for FRAC in 0.000 0.100 0.200 0.350; do
  for SEED in 7 8 9; do
    ARM="f${FRAC}_N1024_s${SEED}"
    if [ -f "$OUT_ROOT/ei_sweep/$ARM.json" ]; then
      echo "skip $ARM (exists)" | tee -a "$LOG"
      continue
    fi
    echo "=== $ARM ===" | tee -a "$LOG"
    HUMANBRAIN_DEVICE=cpu "$PY" benchmarks/ladder_experiments.py \
      --experiment ei_sweep \
      --arm "$ARM" \
      --N 1024 \
      --seed "$SEED" \
      --readout-rule ridge \
      --schedule massed \
      --data-scale 42 \
      --supervision all_transition \
      --codebook explicit \
      --inhibitory-fraction "$FRAC" \
      --check-determinism >> "$LOG" 2>&1
    RC=$?
    echo "$ARM exit=$RC" | tee -a "$LOG"
    if [ $RC -ne 0 ]; then FAIL=1; fi
  done
done

echo "matrix complete $(date -Is), FAIL=$FAIL" | tee -a "$LOG"
exit $FAIL
