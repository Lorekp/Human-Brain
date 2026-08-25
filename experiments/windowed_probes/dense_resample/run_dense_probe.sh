#!/usr/bin/env bash
# Dense-resample probe: does archive sparsity limit the U-STATE emptiness
# conclusion? (follow-up to experiments/windowed_probes/FINDINGS.md)
#
# Step 1  re-log ONE concept arm (lam097_concept) at stride 1 -- every token
#         gets an observation-sink snapshot instead of every 16th -- over the
#         SAME freeze corpus, seed and decision rule (unchanged week-1 harness).
# Step 2  score the dense artifact with the UNCHANGED windowed scorer at
#         k in {1,4,16,64}; k=1 windows are the archived boundary states by
#         construction, so k=1 must reproduce the week-1 lam097_concept row
#         bit-for-bit (same interpreter/numpy) -- that anchors the comparison.
#
# Binaries stay local-only (week-1 convention); the manifest is committed as
# the hash anchor. Budget: replay ~seconds, scoring ~15 min on this CPU.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE"/../../.. && pwd)"
FREEZE_DIR="${LOCALAPPDATA:-$HOME/AppData/Local}/Temp/hb_bible_freeze"
ART_DIR="${LOCALAPPDATA:-$HOME/AppData/Local}/Temp/hb_dense_probe/concept"
ARM="lam097_concept"

mkdir -p "$ART_DIR"

# git-bash note: native python.exe cannot open /c/-style MSYS paths.
HERE_WIN="$(cygpath -w "$HERE")"
ART_WIN="$(cygpath -w "$ART_DIR")"
FREEZE_WIN="$(cygpath -w "$FREEZE_DIR")"

echo "== step 1: dense re-log ($ARM, stride 1) =="
python "$HERE_WIN\\log_dense_arm.py" \
  --arm "$ARM" --stride 1 \
  --freeze-dir "$FREEZE_WIN" \
  --out-dir "$ART_WIN"

echo "== integrity: dense artifact vs its manifest =="
PYTHONPATH="$REPO/src" python - "$ART_DIR" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
art = Path(sys.argv[1])
man = json.loads((art / "logged_states_manifest.json").read_text(encoding="utf-8"))
z = np.load(art / "logged_states.npz")
bad = []
for name, meta in sorted(man["npz_arrays"].items()):
    import hashlib
    h = hashlib.sha256(np.ascontiguousarray(z[name]).tobytes()).hexdigest()
    if h != meta["sha256"]:
        bad.append(name)
assert not bad, bad
arm = man["arms"][0]
print(f"verified {len(man['npz_arrays'])} arrays; stride={arm['families']['union']['stride']}")
print("union stride_rows:", arm["families"]["union"]["stride_rows_shape"])
print("curriculum_digest:", man["curriculum_digest"])
expected = "f0bc99f69ef3caca1a3efe54b421a565e12f957e081d8ec29ab8f3c4c29998be"
assert man["curriculum_digest"] == expected, "curriculum digest drifted off the committed corpus"
print("curriculum digest matches committed concept corpus")
PY

cp "$ART_DIR/logged_states_manifest.json" "$HERE/dense_manifest_lam097_concept_s1.json"

echo "== step 2: windowed scoring on the dense artifact =="
cd "$REPO"
PYTHONPATH=src python -m humanbrain.windowed_probe \
  --npz "$ART_WIN\\logged_states.npz" \
  --manifest "$HERE_WIN\\dense_manifest_lam097_concept_s1.json" \
  --freeze-dir "$FREEZE_WIN" \
  --encoding concept \
  --arms "$ARM" \
  --depths 1,4,16,64 \
  --logistic-iters 250 \
  --logistic-decay-every 125 \
  --max-logistic-d 1024 \
  --out "$HERE_WIN\\windowed_probes_dense_${ARM}.json"

echo "== done =="
