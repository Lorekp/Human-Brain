#!/usr/bin/env bash
# Launch the full codebook_compare matrix: 27 runs + in-process determinism
# re-check on every arm (doubles compute; total still minutes).
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root of the worktree this lives in

OUT_ROOT="experiments"
LOG="experiments/codebook_compare/_run_log.txt"
echo "codebook_compare run log — start $(date -Is)" > "$LOG"

FAIL=0
for CB in explicit random similarity; do
  for N in 128 512 1024; do
    for SEED in 7 8 9; do
      ARM="${CB}_N${N}_s${SEED}"
      if [ -f "$OUT_ROOT/codebook_compare/$ARM.json" ]; then
        echo "skip $ARM (exists)" | tee -a "$LOG"
        continue
      fi
      echo "=== $ARM ===" | tee -a "$LOG"
      HUMANBRAIN_DEVICE=cpu python benchmarks/ladder_experiments.py \
        --experiment codebook_compare \
        --arm "$ARM" \
        --N "$N" \
        --seed "$SEED" \
        --readout-rule ridge \
        --schedule massed \
        --data-scale 42 \
        --supervision all_transition \
        --codebook "$CB" \
        --check-determinism >> "$LOG" 2>&1
      RC=$?
      echo "$ARM exit=$RC" | tee -a "$LOG"
      if [ $RC -ne 0 ]; then FAIL=1; fi
    done
  done
done

echo "matrix complete $(date -Is), FAIL=$FAIL" | tee -a "$LOG"
exit $FAIL
