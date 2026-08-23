#!/usr/bin/env bash
# Launch the codebook_compare SCALE matrix on the CUDA full-touch path:
# 3 codebooks x N in SCALE_NS x seeds {7,8,9}, strictly serialized, one GPU.
# Mirrors _run_matrix.sh conventions (skip-existing, per-run exit logging).
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root of the worktree this lives in

OUT_ROOT="experiments"
LOG="experiments/codebook_compare/_run_scale_log.txt"
SCALE_NS="${SCALE_NS:-16384 32768 4096}"
echo "codebook_compare SCALE run log — start $(date -Is) (SCALE_NS: $SCALE_NS)" > "$LOG"

FAIL=0
for N in $SCALE_NS; do
  for CB in explicit random similarity; do
    for SEED in 7 8 9; do
      ARM="${CB}_N${N}_s${SEED}"
      if [ -f "$OUT_ROOT/codebook_compare/$ARM.json" ]; then
        echo "skip $ARM (exists)" | tee -a "$LOG"
        continue
      fi
      echo "=== $ARM ===" | tee -a "$LOG"
      python benchmarks/ladder_experiments.py \
        --experiment codebook_compare \
        --arm "$ARM" \
        --N "$N" \
        --seed "$SEED" \
        --readout-rule ridge \
        --schedule massed \
        --data-scale 42 \
        --supervision all_transition \
        --codebook "$CB" \
        --device cuda \
        --ridge-backend auto \
        --with-state-separation >> "$LOG" 2>&1
      RC=$?
      echo "$ARM exit=$RC" | tee -a "$LOG"
      if [ $RC -ne 0 ]; then FAIL=1; fi
    done
  done
done

echo "scale matrix complete $(date -Is), FAIL=$FAIL" | tee -a "$LOG"
exit $FAIL
