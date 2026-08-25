#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# remote_bootstrap.sh — vast.ai Ubuntu+CUDA instance bootstrap for the
# Human Brain full-scale R2 campaign.
#
# Turns a clean vast.ai Ubuntu+CUDA box (run as root, as vast.ai images do)
# into a verified, ready-to-launch environment:
#
#   1. system deps (git, curl, build tools, Python 3.11+)      [apt]
#   2. Rust/Cargo stable toolchain                             [rustup]
#   3. clone of github.com/Lorekp/Human-Brain pinned at
#      codex/a013-cognition-design @ 3df61667af1841129b71e1efc699d8ca8a9d7f60
#   4. python venv inside the clone + pip install -e . scipy pytest
#   5. cargo build --release --locked --offline of the vendored native kernel
#   6. deterministic test battery, CPU-only health check:
#      procedural (runtime/topology/address-space-benchmark) + qa_corpus suites
#   7. deterministic regeneration of the Bible freeze-candidate corpus from
#      pinned PG10 (dual-fetch, SHA-256 enforced against the pinned digest),
#      plus an independent hash re-check in this script
#   8. prints a BOOTSTRAP-READY marker line summarizing pins
#
# EVERY step is idempotent: rerunning skips completed work (apt packages,
# existing toolchain/venv/clone at the right SHA, built kernel, existing
# corpus candidate) and re-verifies rather than redoing.
#
# Env overrides (all optional):
#   WORK_ROOT            working dir (default: $HOME/humanbrain-remote)
#   PINNED_REPO_URL      default https://github.com/Lorekp/Human-Brain.git
#   PINNED_SHA           default 3df61667af1841129b71e1efc699d8ca8a9d7f60
#   FORCE_CORPUS_REGEN   1 -> rebuild even if the candidate already exists
#   SKIP_APT             1 -> skip the apt step (images with everything baked in)
#
# Expected cold-start wall time: ~5-10 min (dominated by apt + pip + cargo).
# ---------------------------------------------------------------------------

set -euo pipefail

# --- pinned inputs (do not change lightly; these define reproducibility) ----
PINNED_REPO_URL="${PINNED_REPO_URL:-https://github.com/Lorekp/Human-Brain.git}"
PINNED_BRANCH="${PINNED_BRANCH:-codex/a013-cognition-design}"
PINNED_SHA="${PINNED_SHA:-3df61667af1841129b71e1efc699d8ca8a9d7f60}"
# SHA-256 of the raw pg10.txt payload, pinned in tools/build_bible_qa_corpus.py
PINNED_PG10_SHA256="0204adaed1f25700aa854218cae63c7172228c41088f335e99167a071eed83c0"

WORK_ROOT="${WORK_ROOT:-$HOME/humanbrain-remote}"
CLONE_DIR="$WORK_ROOT/Human-Brain"
VENV_DIR="$CLONE_DIR/.venv"
NATIVE_CACHE="$WORK_ROOT/native-cache"
# The corpus builder writes under "$LOCALAPPDATA/Temp/hb_bible_freeze"; we aim
# that variable at our own root so nothing lands outside WORK_ROOT.
FREEZE_ROOT="$WORK_ROOT/hb_bible_freeze_root"
CORPUS_DIR="$FREEZE_ROOT/Temp/hb_bible_freeze"
CORPUS_CANDIDATE="$CORPUS_DIR/FREEZE-CANDIDATE-v3-BIBLE.json"

log() { printf '[bootstrap] %s\n' "$*" >&2; }
die() { printf '[bootstrap] FATAL: %s\n' "$*" >&2; exit 1; }

mkdir -p "$WORK_ROOT"

# --- privilege helper (vast.ai images are root; be tolerant anyway) --------
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
fi

# --- 1. system dependencies -------------------------------------------------
system_ready() {
  command -v git >/dev/null 2>&1 && command -v curl >/dev/null 2>&1 \
    && command -v cc >/dev/null 2>&1 \
    && find_py >/dev/null 2>&1
}

find_py() {
  # Prefer 3.11 explicitly; accept any >= 3.11 interpreter present.
  local cand
  for cand in python3.11 python3.12 python3.13 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
        printf '%s' "$cand"
        return 0
      fi
    fi
  done
  return 1
}

if [ "${SKIP_APT:-0}" = "1" ]; then
  log "step 1/8: SKIP_APT=1 -> skipping apt step"
elif system_ready >/dev/null 2>&1; then
  log "step 1/8: system deps already present (git/curl/cc/python>=3.11) -> skipping apt"
else
  log "step 1/8: installing apt dependencies..."
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y
  # Base tooling always; python3.11 best-effort (jammy has it in universe,
  # noble falls back to its stock python3 which is >= 3.11).
  $SUDO apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential pkg-config python3-venv
  $SUDO apt-get install -y --no-install-recommends python3.11 python3.11-venv || true
fi

PY="$(find_py)" || die "no Python >= 3.11 interpreter found after apt step"
log "step 1/8: using interpreter: $(command -v "$PY") ($("$PY" --version))"

# Ensure the venv module is usable (Debian splits it out).
if ! "$PY" -m venv --help >/dev/null 2>&1; then
  log "installing venv support for $PY..."
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends python3-venv \
    || $SUDO apt-get install -y --no-install-recommends "python$(basename "$(command -v "$PY")" | sed 's/python//')-venv"
fi

# --- 2. Rust toolchain -------------------------------------------------------
export PATH="$HOME/.cargo/bin:$PATH"
if command -v cargo >/dev/null 2>&1; then
  log "step 2/8: cargo already present ($(cargo --version)) -> skipping rustup"
else
  log "step 2/8: installing Rust stable via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
  export PATH="$HOME/.cargo/bin:$PATH"
  command -v cargo >/dev/null 2>&1 || die "rustup finished but cargo is not on PATH"
  # Make cargo reachable in future login shells (guarded -> idempotent).
  grep -qs '\$HOME/.cargo/bin' "$HOME/.bashrc" || \
    printf '\nexport PATH="$HOME/.cargo/bin:$PATH"\n' >> "$HOME/.bashrc"
  log "step 2/8: installed $(cargo --version)"
fi

# --- 3. repository clone pinned at the campaign SHA -------------------------
log "step 3/8: ensuring repo clone at pinned SHA $PINNED_SHA..."
if [ ! -d "$CLONE_DIR/.git" ]; then
  git clone --branch "$PINNED_BRANCH" "$PINNED_REPO_URL" "$CLONE_DIR"
else
  git -C "$CLONE_DIR" fetch origin "$PINNED_BRANCH"
fi
cd "$CLONE_DIR"
if [ "$(git rev-parse HEAD)" != "$PINNED_SHA" ]; then
  # Detach onto the exact pinned commit; fetch it explicitly if a fresh clone
  # of an advanced branch tip does not already contain the object.
  git checkout --detach "$PINNED_SHA" 2>/dev/null \
    || { git fetch --depth=1 origin "$PINNED_SHA" && git checkout --detach "$PINNED_SHA"; } \
    || die "could not materialize pinned SHA $PINNED_SHA"
fi
[ "$(git rev-parse HEAD)" = "$PINNED_SHA" ] || die "HEAD $(git rev-parse HEAD) != pinned $PINNED_SHA"
log "step 3/8: HEAD verified at pinned SHA $(git rev-parse --short HEAD)"

# --- 4. virtualenv + editable install ---------------------------------------
log "step 4/8: ensuring venv + pip install -e . scipy pytest..."
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PY" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install --quiet -e . scipy pytest
"$VENV_DIR/bin/python" -c 'import humanbrain, numpy, scipy, pytest' \
  || die "editable install import check failed"
log "step 4/8: venv ready at $VENV_DIR"

# --- 5. native Rust kernel (vendored deps -> offline) ------------------------
log "step 5/8: cargo build --release of the native kernel..."
(cd src/humanbrain/_native_src && cargo build --release --locked --offline)
# NOTE: the runtime additionally builds its own content-addressed copy of the
# kernel on first use into HUMANBRAIN_NATIVE_CACHE; this manual build proves
# the toolchain + vendored sources compile cleanly before any test runs.

# --- 6. deterministic test battery (CPU-only health check) -------------------
export HUMANBRAIN_NATIVE_CACHE="$NATIVE_CACHE"
log "step 6/8: running deterministic test battery (CPU-only)..."
"$VENV_DIR/bin/python" -m pytest \
  tests/test_procedural_runtime.py \
  tests/test_procedural_topology.py \
  tests/test_procedural_address_space_benchmark.py \
  tests/test_qa_corpus.py \
  -q
log "step 6/8: test battery PASSED"

# Non-fatal GPU inventory (campaign runs will use the CUDA path).
if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else
  log "WARN: nvidia-smi not available; CUDA path unverified on this box"
fi

# --- 7. Bible corpus regeneration from pinned PG10 ---------------------------
if [ -f "$CORPUS_CANDIDATE" ] && [ "${FORCE_CORPUS_REGEN:-0}" != "1" ]; then
  log "step 7/8: corpus candidate already present -> skipping regen (FORCE_CORPUS_REGEN=1 forces)"
else
  log "step 7/8: regenerating Bible corpus deterministically from pinned PG10..."
  mkdir -p "$FREEZE_ROOT"
  LOCALAPPDATA="$FREEZE_ROOT" "$VENV_DIR/bin/python" tools/build_bible_qa_corpus.py
fi
[ -f "$CORPUS_CANDIDATE" ] || die "corpus candidate missing after build step"
RAW_PG10="$CORPUS_DIR/books/pg10.fetch1.txt"
[ -f "$RAW_PG10" ] || die "raw PG10 fetch artifact missing: $RAW_PG10"
ACTUAL_PG10_SHA="$(sha256sum "$RAW_PG10" | cut -d' ' -f1)"
[ "$ACTUAL_PG10_SHA" = "$PINNED_PG10_SHA256" ] \
  || die "PG10 sha mismatch: got $ACTUAL_PG10_SHA want $PINNED_PG10_SHA256"
CORPUS_SHA="$(sha256sum "$CORPUS_CANDIDATE" | cut -d' ' -f1)"
log "step 7/8: PG10 sha verified; candidate sha256=$CORPUS_SHA"

# --- 8. READY marker ----------------------------------------------------------
log "step 8/8: bootstrap complete"
printf 'BOOTSTRAP-READY commit=%s pg10_sha=%s corpus_candidate_sha=%s venv=%s native_cache=%s\n' \
  "$PINNED_SHA" "$PINNED_PG10_SHA256" "$CORPUS_SHA" "$VENV_DIR" "$NATIVE_CACHE"
