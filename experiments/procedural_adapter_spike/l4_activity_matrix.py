"""L4 activity-matrix rerun harness (post-seam validation).

Re-runs the L3 activity matrix (FINDINGS.md §3.3(b)) extended to 20%
activity -- N_cfg = 1e6 x activity {1%, 2%, 5%, 20%} x stepping_mode
{full_touch, event_proportional} -- so the CPU-EP upper-band question gets
its answer the day Seam B (indexed staged membership,
docs/research/cpu-seams-spec.md) merges. The harness is seam-aware:

- It PROBES the running native library for both seams before measuring:
  * Seam A ``read_cells``: ``hasattr(lib, 'humanbrain_procedural_runtime_
    read_cells')`` + the Python binding. Present -> the per-token mirror-read
    diagnostic uses the optimized targeted read; absent -> falls back to one
    snapshot()-based extraction per cell for context (per-token mirroring is
    O(R) and therefore not measurable per token; see FINDINGS §2).
  * Seam B indexed staged membership: internal Rust field, deliberately no
    ABI change, so presence is detected BEHAVIORALLY -- one advance() call
    whose events repeatedly re-target M distinct fresh gids (paying the
    staged-membership lookup R-1 extra times per gid while |staged| ~ M)
    vs a control call with the same event count over fully distinct gids.
    Linear-scan builds pay ~O(M) comparisons per repeat-hit (ratio >> 2.5);
    indexed builds pay O(log M) (ratio ~ 1). A read-only grep for
    ``staged_index`` in _native_src corroborates (the built library is
    authoritative; the source tree may be ahead of it).
- Every (mode x activity) cell runs in a FRESH runtime with its own chunked
  fixture build (spec protocol: 2,048-wide windows, settle, TIMING STARTS
  POST-BUILD), so cells cannot contaminate each other's timings.

Capacity planning against the fail-stop contracts (overflow poisons the
runtime mid-measurement): capacities scale off the largest single advance()
call with documented headroom (L3 constants QUEUE_HEADROOM=8,
SPIKE_HEADROOM=4). The 20% row adds a staging-pressure term the L3 matrix
never hit: only injected cells spike (downstream weight is subthreshold), so
one token generates fanout*A edges that stage up to
``cfg*(1-exp(-f*A/cfg))`` fresh recipients inside the NEXT call, on top of
the fixture residents. The fixture is therefore the largest power of two
that fits ``cfg - staged_peak`` (capped at the L3 protocol's 2^18): rows at
1-5% keep the exact L3 fixture (2^18); the 20% row runs 2^16 so
residents+staged stays safely under cfg=1e6. Within-row EP-vs-FT ratios
remain apples-to-apples (identical schedules); cross-row absolute walls
carry the fixture-size caveat recorded per row.

Usage:
    python l4_activity_matrix.py                       # full protocol, 12 tokens/cell
    python l4_activity_matrix.py --tokens 4 --label seam-a-baseline
    python l4_activity_matrix.py --probe-only          # seam detection only

Artifact: <outdir>/l4_activity_matrix_<label>.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Self-contained bootstrap: run from any cwd, the worktree's src/ is put on
# sys.path before the humanbrain imports below.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np

from humanbrain.native import _F64_POINTER, _GlifConfig, _load_library
from humanbrain.procedural_runtime import NativeProceduralRuntime, _NativeResident

MASK64 = 0xFFFFFFFFFFFFFFFF

# --- protocol constants (L3-pinned unless noted) ---------------------------
CONFIGURED_CELLS = 1_000_000
ACTIVITY_LEVELS = ((1, 10_000), (2, 20_000), (5, 50_000), (20, 200_000))
MODES = ("full_touch", "event_proportional")
TOKENS_PER_CELL = 12          # L3 protocol
TOKEN_WINDOW_STEPS = 100
BUILD_RESIDENTS_DEFAULT = 1 << 18   # L3 §3.3(b) residency ratchet
BUILD_PER_WINDOW = 2048            # chunked fixture construction per spec
SETTLE_STEPS = 400
SPIKE_HEADROOM = 4                 # status-9 guard
QUEUE_HEADROOM = 8                 # status-3 guard: peak <= batch + fanout*batch
STAGED_TAIL_FACTOR = 1.10          # coupon-concentration safety on expected staging
MIN_FIXTURE = 1 << 14              # below this a row is capacity-infeasible
MAX_TOKEN_SECONDS = 900.0          # circuit breaker default (merge-day runs)
MAX_TOKEN_SECONDS_EFFECTIVE = MAX_TOKEN_SECONDS  # overridden via --max-token-seconds
MIRROR_PROBE_K = 16                # hot-cell read width (acceptance parity)
MIRROR_PROBE_TOKENS = 4
MIRROR_SKIP_ABOVE_S = 60.0         # skip mirror when any stream token exceeds this

# --quick mode: cap every cell's fixture at 2^16 so an end-to-end smoke run
# finishes in minutes (protocol default remains 2^18 where capacity allows).
QUICK_FIXTURE_CAP: int | None = None

# --- Seam-B behavioral probe constants -------------------------------------
# Single decisive scale. Scan arm: M fresh gids x R repeats inside ONE call,
# so repeat deliveries pay the staged-membership lookup while |staged| grows
# to M; quadratic .position() cost ~ R*M^2 comparisons (~0.5-3 s here),
# indexed O(M log M) cost is marshal-dominated (~30-70 ms). Verdict uses the
# ABSOLUTE scan-arm median against a floor plus a paired-reference sanity
# ratio. Earlier designs failed on this desktop: a two-scale growth ratio
# amplified ms-scale jitter into verdict flips (1513x vs 21x on the SAME
# linear library), because subtracting near-noise reference terms is not
# load-stable. Absolute level at a scale where the quadratic term dominates
# by orders of magnitude is.
PROBE_M_TARGETS = 24_576           # staged-entry count within one call
PROBE_REPEATS = 16                 # deliveries per target (15 repeat-hits)
PROBE_REPS = 3                     # timed reps per arm
# Empirically anchored on both ground-truth libraries (2026-08-24):
#   fa9070e linear: scan medians 8.5-10.4 s (quadratic term dominates)
#   WIP indexed:    scan medians 49-145 ms (marshal-dominated)
# 400 ms floor keeps >=4x margin below the slowest observed indexed wall
# and >20x margin above the fastest observed linear wall.
PROBE_LINEAR_FLOOR_MS = 400.0
# Subthreshold under the stress GLIF (threshold floor = 1.0): worst-case
# accumulation is prime (x1) + measurement (xR) = 17 x current = 1.7 ...
# exceeds 1.0! Probe current therefore 0.05 pA (17 x 0.05 = 0.85 < 1.0).
# The original 100 pA value SPIKED (163,842 stray spikes in the first smoke
# run) and invalidated that attempt's timings.
PROBE_CURRENT_PA = 0.05


class NativeEvent(ctypes.Structure):
    _fields_ = [
        ("due_step", ctypes.c_uint64),
        ("target_gid", ctypes.c_uint64),
        ("source_gid", ctypes.c_uint64),
        ("edge_slot", ctypes.c_uint64),
        ("current_pa", ctypes.c_double),
    ]


class NativeConfig(ctypes.Structure):
    _fields_ = [
        ("configured_cells", ctypes.c_uint64),
        ("fanout", ctypes.c_uint64),
        ("seed", ctypes.c_uint64),
        ("resident_capacity", ctypes.c_size_t),
        ("event_capacity", ctypes.c_size_t),
        ("spike_capacity", ctypes.c_size_t),
        ("interval_distinct_capacity", ctypes.c_size_t),
        ("recurrent_weight_pa", ctypes.c_double),
        ("recurrent_delay_steps", ctypes.c_uint64),
        ("recurrent_decay", ctypes.c_double),
        ("eligibility_decay", ctypes.c_double),
        ("stepping_mode", ctypes.c_uint32),
    ]


class NativeCounters(ctypes.Structure):
    _fields_ = [
        ("current_step", ctypes.c_uint64),
        ("resident_peak", ctypes.c_size_t),
        ("resident_distinct", ctypes.c_size_t),
        ("updated_distinct", ctypes.c_size_t),
        ("spiking_distinct", ctypes.c_size_t),
        ("cell_step_evaluations", ctypes.c_uint64),
        ("spike_events", ctypes.c_uint64),
        ("generated_edges", ctypes.c_uint64),
        ("delivered_events", ctypes.c_uint64),
        ("configured_ids_scanned", ctypes.c_uint64),
        ("queue_peak", ctypes.c_size_t),
        ("native_payload_bytes", ctypes.c_size_t),
        ("retained_nonzero_cells", ctypes.c_size_t),
        ("overflow_code", ctypes.c_uint32),
    ]


LIB = _load_library()

LIB.humanbrain_procedural_runtime_create.argtypes = [
    ctypes.POINTER(_GlifConfig),
    ctypes.POINTER(NativeConfig),
    _F64_POINTER,
    _F64_POINTER,
    _F64_POINTER,
]
LIB.humanbrain_procedural_runtime_create.restype = ctypes.c_void_p
LIB.humanbrain_procedural_runtime_destroy.argtypes = [ctypes.c_void_p]
LIB.humanbrain_procedural_runtime_advance.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.POINTER(NativeEvent),
    ctypes.c_size_t,
]
LIB.humanbrain_procedural_runtime_advance.restype = ctypes.c_uint32
LIB.humanbrain_procedural_runtime_counts.argtypes = [ctypes.c_void_p] + [
    ctypes.POINTER(ctypes.c_size_t)
] * 3
LIB.humanbrain_procedural_runtime_counts.restype = ctypes.c_uint32
LIB.humanbrain_procedural_runtime_dormant_snapshot.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
LIB.humanbrain_procedural_runtime_dormant_snapshot.restype = ctypes.c_uint32
LIB.humanbrain_procedural_runtime_snapshot.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(NativeCounters),
]
LIB.humanbrain_procedural_runtime_snapshot.restype = ctypes.c_uint32

# Seam A FFI probe + optional binding (argtypes declared only if exported --
# touching a missing attribute on a CDLL raises AttributeError).
READ_CELLS_EXPORT = "humanbrain_procedural_runtime_read_cells"
HAS_READ_CELLS_FFI = hasattr(LIB, READ_CELLS_EXPORT)
HAS_READ_CELLS_PY = hasattr(NativeProceduralRuntime, "read_cells")
if HAS_READ_CELLS_FFI:
    getattr(LIB, READ_CELLS_EXPORT).argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(_NativeResident),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    getattr(LIB, READ_CELLS_EXPORT).restype = ctypes.c_uint32


STRESS_GLIF = _GlifConfig(
    1,      # cell_count (unused by the sparse runtime)
    2,      # after_channel_count
    3,      # spike_cut_steps -- short cut keeps integration lifetimes tight
    1.0,    # leak_ns -> v' = I exactly; fast state flushes with syn current
    1.0,    # voltage_step_scale
    0.0,    # threshold_beta_scale
    0.9,    # threshold_exp_decay
    0.0,    # threshold_voltage_factor
    0.0,    # threshold_beta_factor
    0.9,    # spike_threshold_decay
    1.0,    # threshold_inf_relative_mv -- spike floor
    0.0,    # voltage_reset_scale
    0.0,    # voltage_reset_offset_mv
    0.5,    # spike_cut_threshold_decay
    0.0,    # threshold_spike_increment_mv
)
STRESS_DECAYS = np.array([0.5, 0.5], dtype=np.float64)
STRESS_AMPLITUDES = np.array([0.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# seam detection
# ---------------------------------------------------------------------------
def _source_symbol_present(symbol: str) -> dict:
    """Read-only corroboration: does the native source contain Seam-B's
    distinctive signatures (``staged_index: BTreeMap<u64, u32>`` field,
    ``staged_index.get(/insert(`` call sites)? The bare name
    ``staged_index`` is NOT used: pre-existing local merge-loop counters
    carry it even at fa9070e and would false-positive.

    Never authoritative (built library may lag the source tree); recorded
    alongside the behavioral probe with that caveat.
    """
    patterns = [
        "BTreeMap<u64, u32>",
        "staged_index.get(",
        "staged_index.insert(",
    ]
    candidates = [
        _REPO_ROOT / "src" / "humanbrain" / "_native_src" / "src" / "procedural.rs",
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        return {
            "file": str(path),
            "symbol_present": any(p in text for p in patterns)
            if symbol == "staged_index"
            else symbol in text,
        }
    return {"file": None, "symbol_present": None}


def _provenance() -> dict:
    """Pin exactly what native binary this process is running against.

    The loader is source-addressed (it builds/attests a digest of
    _native_src), so uncommitted edits change the loaded binary. Recorded
    here so artifacts can never be misread as coming from HEAD.
    """
    info: dict = {
        "head_commit": git_commit(),
        "procedural_rs_dirty": None,
        "source_digest": None,
        "library_path": None,
    }
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src/humanbrain/_native_src/src/procedural.rs",
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            check=False,
        ).stdout.strip()
        info["procedural_rs_dirty"] = bool(status)
        if status:
            info["procedural_rs_status"] = status.splitlines()[0]
    except OSError:
        pass
    try:
        from humanbrain.native import loaded_library_identity

        identity = loaded_library_identity()
        info["source_digest"] = getattr(identity, "source_digest", None)
        library_path = getattr(identity, "library_path", None)
        info["library_path"] = str(library_path) if library_path else None
    except Exception as error:  # noqa: BLE001 - provenance is best-effort
        info["library_identity_error"] = repr(error)
    return info


def detect_read_cells_seam() -> dict:
    ffi = HAS_READ_CELLS_FFI
    py = HAS_READ_CELLS_PY
    return {
        "ffi_export_present": ffi,
        "python_binding_present": py,
        "available": bool(ffi and py),
        "method": "hasattr on loaded CDLL + runtime class",
    }


def _probe_one_scale() -> dict:
    """Scan-heavy vs resident-reference walls at the decisive scale."""
    batch = PROBE_M_TARGETS * PROBE_REPEATS
    blocks = PROBE_REPS + 1
    cfg_cells = max(
        262_144,
        1 << math.ceil(math.log2(blocks * PROBE_M_TARGETS * 2)),
    )
    # No spikes -> queue peak == injected batch; capacity carries buffer
    # headroom only. Interval ledger stays at its contract minimum.
    event_capacity = 4 * batch + 1024
    spike_capacity = 8_192
    rt_scan = _make_runtime(
        stepping_mode="full_touch",
        configured_cells=cfg_cells,
        batch=batch,
        fanout=4,
        event_capacity=event_capacity,
        spike_capacity=spike_capacity,
        weight_pa=0.25,
        interval_distinct_capacity=cfg_cells,
    )
    rt_resident = _make_runtime(
        stepping_mode="full_touch",
        configured_cells=cfg_cells,
        batch=batch,
        fanout=4,
        event_capacity=event_capacity,
        spike_capacity=spike_capacity,
        weight_pa=0.25,
        interval_distinct_capacity=cfg_cells,
    )
    spikes_seen = 0
    scan_ms: list[float] = []
    reference_ms: list[float] = []

    def _repeat_targets(base: int):
        # Native validation rejects edge_slot >= fanout, so slots stay 0;
        # canonical-key uniqueness across repeats comes from the varying
        # source gid ((due_step, target, source, edge_slot)).
        return [
            (base + i, (i * PROBE_REPEATS + j) % cfg_cells, 0)
            for i in range(PROBE_M_TARGETS)
            for j in range(PROBE_REPEATS)
        ]

    def _prime(base: int) -> None:
        nonlocal spikes_seen
        _, spikes = _timed_probe_call(
            rt_resident,
            [(base + i, (base + i) % cfg_cells, 0)
             for i in range(PROBE_M_TARGETS)],
        )
        spikes_seen += spikes

    try:
        for rep in range(PROBE_REPS):
            base = rep * PROBE_M_TARGETS
            # Scan arm: FRESH block xR repeats in one call -- first delivery
            # stages each gid, repeats pay the staged-membership lookup as
            # |staged| grows to M inside the call.
            wall_s, spikes_s = _timed_probe_call(rt_scan, _repeat_targets(base))
            spikes_seen += spikes_s
            # Prime then reference on the twin: absorb makes the block
            # resident, so identical volume takes the binary-search fast
            # path with zero staged-membership work.
            _prime(base)
            wall_r, spikes_r = _timed_probe_call(
                rt_resident, _repeat_targets(base)
            )
            spikes_seen += spikes_r
            scan_ms.append(wall_s)
            reference_ms.append(wall_r)
    finally:
        rt_scan.close()
        rt_resident.close()

    if spikes_seen:
        raise RuntimeError(
            f"seam probe produced {spikes_seen} stray spikes; "
            "probe current must stay subthreshold"
        )
    return {
        "configured_cells": cfg_cells,
        "events_per_call": batch,
        "scan_heavy_median_ms": round(statistics.median(scan_ms), 3),
        "resident_reference_median_ms": round(
            statistics.median(reference_ms), 3
        ),
        "scan_heavy_walls_ms": [round(w, 3) for w in scan_ms],
        "reference_walls_ms": [round(w, 3) for w in reference_ms],
    }


def detect_indexed_staged_seam() -> dict:
    """Behavioral FFI probe for Seam B (indexed staged membership).

    One decisive scale: M=24,576 staged entries x R=16 repeat deliveries
    inside a single advance() call ("scan-heavy"), vs identical volume into
    just-absorbed RESIDENT gids ("resident-reference", reported as context).
    A quadratic .position() scan costs ~R*M^2/4 comparisons (~0.5-10 s here);
    an index costs O(M log M), marshal-dominated (~tens of ms). Verdict:
    ABSOLUTE scan-arm median against an empirically anchored floor -- chosen
    over a two-scale growth classifier whose subtraction amplified desktop
    jitter into verdict flips on an unchanged library.
    """
    probe = _probe_one_scale()
    scan_med = probe["scan_heavy_median_ms"]
    ref_med = probe["resident_reference_median_ms"]
    scan_over_ref = round(scan_med / max(ref_med, 1e-9), 3)
    # Verdict on the ABSOLUTE scan-arm median only. The resident-reference
    # arm is reported as context but NOT used in classification: on the
    # linear build it pays unrelated resident-count-dependent costs (the
    # documented quadratic-class public-API terms), which made any ratio
    # test unstable while the absolute scan wall separated cleanly.
    verdict = "linear_scan" if scan_med >= PROBE_LINEAR_FLOOR_MS else "indexed"
    source = _source_symbol_present("staged_index")
    return {
        "method": (
            f"behavioral single-scale probe: M={PROBE_M_TARGETS} staged "
            f"entries x R={PROBE_REPEATS} repeat deliveries vs identical "
            "volume on the resident fast path"
        ),
        **probe,
        "linear_floor_ms": PROBE_LINEAR_FLOOR_MS,
        "scan_over_reference": scan_over_ref,
        "verdict": verdict,
        "indexed_staged_available": verdict == "indexed",
        "source_corroboration": source,
        "caveat": (
            "behavioral verdict reflects the LOADED library; the source-tree "
            "grep may disagree if the build is stale"
        ),
    }


def _timed_probe_call(rt: "Runtime", targets) -> tuple[float, int]:
    """One advance() with pre-built (target, source, edge_slot) tuples."""
    count = len(targets)
    buffer = rt.buffer
    due = rt.step + TOKEN_WINDOW_STEPS - 1
    for index, (gid, source, slot) in enumerate(targets):
        entry = buffer[index]
        entry.due_step = due
        entry.target_gid = gid
        entry.source_gid = source
        entry.edge_slot = slot
        entry.current_pa = PROBE_CURRENT_PA
    start = time.perf_counter()
    status = LIB.humanbrain_procedural_runtime_advance(
        rt.handle, TOKEN_WINDOW_STEPS, buffer, count
    )
    elapsed = time.perf_counter() - start
    if status != 0:
        raise RuntimeError(f"probe advance failed with native status {status}")
    rt.step += TOKEN_WINDOW_STEPS
    spikes = ctypes.c_size_t()
    pending = ctypes.c_size_t()
    active = ctypes.c_size_t()
    assert LIB.humanbrain_procedural_runtime_counts(
        rt.handle, ctypes.byref(active), ctypes.byref(pending), ctypes.byref(spikes)
    ) == 0
    return elapsed * 1e3, spikes.value


# ---------------------------------------------------------------------------
# capacity planning
# ---------------------------------------------------------------------------
def plan_cell(activity_events: int, *, fanout: int = 4) -> dict | None:
    """Per-cell capacity plan; returns None if the row cannot fit cfg.

    Only injected cells spike (downstream weight 0.25 pA is subthreshold),
    so one token generates fanout*A edges delivering inside the next call,
    staging up to cfg*(1-exp(-f*A/cfg)) fresh recipients on top of the
    fixture. staged_peak carries a 10% concentration tail plus the injected
    non-resident upper bound; the fixture is the largest power of two that
    fits cfg - staged_peak (protocol cap 2^18; --quick caps it at 2^16 for
    smoke validation).
    """
    configured = CONFIGURED_CELLS
    batch = activity_events
    event_capacity = QUEUE_HEADROOM * batch
    spike_capacity = SPIKE_HEADROOM * batch
    expected_fresh = configured * (
        1.0 - math.exp(-fanout * batch / configured)
    )
    staged_peak = math.ceil(expected_fresh * STAGED_TAIL_FACTOR) + batch
    budget = configured - staged_peak
    fixture_cap = min(BUILD_RESIDENTS_DEFAULT, budget)
    if QUICK_FIXTURE_CAP is not None:
        fixture_cap = min(fixture_cap, QUICK_FIXTURE_CAP)
    if fixture_cap < MIN_FIXTURE:
        return None
    fixture = 1 << int(math.floor(math.log2(fixture_cap)))
    interval_distinct_capacity = max(
        configured, event_capacity, fixture + staged_peak + batch
    )
    return {
        "fixture_residents": fixture,
        "expected_fresh_staging_per_token": int(expected_fresh),
        "staged_peak_model": int(staged_peak),
        "resident_capacity": configured,
        "event_capacity": event_capacity,
        "spike_capacity": spike_capacity,
        "interval_distinct_capacity": interval_distinct_capacity,
        "fanout": fanout,
    }


# ---------------------------------------------------------------------------
# runtime driver (raw C-ABI hot path, L3-identical protocol)
# ---------------------------------------------------------------------------
class Runtime:
    """Raw C-ABI runtime with internal step bookkeeping (L3 protocol)."""

    def __init__(self, *, stepping_mode: str, plan: dict, weight_pa: float) -> None:
        mode_id = {"full_touch": 0, "event_proportional": 1}[stepping_mode]
        self.stepping_mode = stepping_mode
        self.plan = plan
        self.fanout = plan["fanout"]
        config = NativeConfig(
            CONFIGURED_CELLS,
            self.fanout,
            71,
            plan["resident_capacity"],
            plan["event_capacity"],
            plan["spike_capacity"],
            plan["interval_distinct_capacity"],
            weight_pa,
            1,
            # Aggressive synaptic decay (L3): the 150 k pA drive must flush
            # below threshold within a few steps or driven cells re-spike
            # every refractory cycle and blow the spike budget.
            1e-8,
            math.exp(-0.05 / 80.0),
            mode_id,
        )
        self.handle = LIB.humanbrain_procedural_runtime_create(
            ctypes.byref(STRESS_GLIF),
            ctypes.byref(config),
            STRESS_DECAYS.ctypes.data_as(_F64_POINTER),
            STRESS_DECAYS.ctypes.data_as(_F64_POINTER),
            STRESS_AMPLITUDES.ctypes.data_as(_F64_POINTER),
        )
        if not self.handle:
            raise RuntimeError(f"native create failed ({stepping_mode})")
        self.buffer = (NativeEvent * (plan["event_capacity"] // 4))()
        self.counters = NativeCounters()
        self.step = 0

    def close(self) -> None:
        if self.handle:
            LIB.humanbrain_procedural_runtime_destroy(self.handle)
            self.handle = None

    def advance(self, steps: int, targets) -> float:
        count = len(targets)
        if count > len(self.buffer):
            raise RuntimeError("event batch exceeds buffer")
        buffer = self.buffer
        due = self.step + steps - 1  # deliver on the window's final step
        for index, gid in enumerate(targets):
            entry = buffer[index]
            entry.due_step = due
            entry.target_gid = gid
            entry.source_gid = gid
            entry.edge_slot = 0
            entry.current_pa = 150_000.0
        start = time.perf_counter()
        status = LIB.humanbrain_procedural_runtime_advance(
            self.handle, steps, buffer, count
        )
        elapsed = time.perf_counter() - start
        if status != 0:
            raise RuntimeError(
                f"advance failed with native status {status} at step {self.step}"
            )
        self.step += steps
        return elapsed

    def settle(self, steps: int) -> None:
        self.advance(steps, [])

    def live_state(self) -> dict:
        active = ctypes.c_size_t()
        pending = ctypes.c_size_t()
        spikes = ctypes.c_size_t()
        dormant = ctypes.c_size_t()
        assert LIB.humanbrain_procedural_runtime_counts(
            self.handle, ctypes.byref(active), ctypes.byref(pending),
            ctypes.byref(spikes),
        ) == 0
        assert LIB.humanbrain_procedural_runtime_dormant_snapshot(
            self.handle, None, 0, ctypes.byref(dormant)
        ) == 0
        return {
            "active_residents": active.value - dormant.value,
            "dormant": dormant.value,
            "pending": pending.value,
        }

    def total_work(self) -> int:
        assert LIB.humanbrain_procedural_runtime_snapshot(
            self.handle, None, 0, None, 0, None, 0, ctypes.byref(self.counters)
        ) == 0
        return self.counters.cell_step_evaluations

    def read_cells_raw(self, gids):
        """Optimized targeted read (Seam A); returns (eligibilities, ns).

        eligibilities[i] is None when the cell is not actively resident
        (dormant/absent read as absent, matching the seam contract).
        """
        if not HAS_READ_CELLS_FFI:
            raise RuntimeError("read_cells FFI export not present")
        count = len(gids)
        gid_buffer = (ctypes.c_uint64 * count)(*gids)
        out = (_NativeResident * count)()
        found = (ctypes.c_uint8 * count)()
        start = time.perf_counter_ns()
        status = int(
            getattr(LIB, READ_CELLS_EXPORT)(
                self.handle, gid_buffer, count, out, found
            )
        )
        elapsed_ns = time.perf_counter_ns() - start
        if status != 0:
            raise RuntimeError(f"read_cells failed with native status {status}")
        eligibilities = [
            out[i].eligibility if found[i] else None for i in range(count)
        ]
        return eligibilities, elapsed_ns


def _make_runtime(
    *,
    stepping_mode: str,
    configured_cells: int,
    batch: int,
    fanout: int,
    event_capacity: int,
    spike_capacity: int,
    weight_pa: float,
    interval_distinct_capacity: int | None = None,
) -> Runtime:
    """Direct-construction helper used by the seam probe (plan bypass)."""
    rt = Runtime.__new__(Runtime)
    mode_id = {"full_touch": 0, "event_proportional": 1}[stepping_mode]
    rt.stepping_mode = stepping_mode
    if interval_distinct_capacity is None:
        interval_distinct_capacity = max(configured_cells, event_capacity)
    rt.plan = {
        "resident_capacity": configured_cells,
        "event_capacity": event_capacity,
        "spike_capacity": spike_capacity,
        "interval_distinct_capacity": interval_distinct_capacity,
        "fanout": fanout,
    }
    rt.fanout = fanout
    config = NativeConfig(
        configured_cells,
        fanout,
        71,
        configured_cells,
        event_capacity,
        spike_capacity,
        interval_distinct_capacity,
        weight_pa,
        1,
        1e-8,
        math.exp(-0.05 / 80.0),
        mode_id,
    )
    rt.handle = LIB.humanbrain_procedural_runtime_create(
        ctypes.byref(STRESS_GLIF),
        ctypes.byref(config),
        STRESS_DECAYS.ctypes.data_as(_F64_POINTER),
        STRESS_DECAYS.ctypes.data_as(_F64_POINTER),
        STRESS_AMPLITUDES.ctypes.data_as(_F64_POINTER),
    )
    if not rt.handle:
        raise RuntimeError(f"native create failed ({stepping_mode})")
    rt.buffer = (NativeEvent * (event_capacity // 4))()
    rt.counters = NativeCounters()
    rt.step = 0
    return rt


def build_residency(runtime: Runtime, total: int, per_window: int) -> float:
    """Chunked fixture construction (spec protocol): fresh gids in fixed
    windows so every call fits the per-call sizing. NOT part of timing."""
    started = time.perf_counter()
    built = 0
    reported_at = 0.25
    while built < total:
        chunk = min(per_window, total - built)
        runtime.advance(TOKEN_WINDOW_STEPS, range(built, built + chunk))
        built += chunk
        if total >= 65_536:
            frac = built / total
            if frac >= reported_at:
                print(
                    f"    [build] {built:,}/{total:,} "
                    f"({time.perf_counter() - started:5.1f}s)",
                    flush=True,
                )
                reported_at += 0.25
    return time.perf_counter() - started


def token_targets(count: int, token_index: int, pool: int) -> list[int]:
    """L3 pool-locality schedule: alternating halves of a 2x vocabulary."""
    offset = ((token_index % 2) * count) % pool
    return [(offset + i) % pool for i in range(count)]


def run_cell(
    mode: str,
    pct: int,
    events_per_token: int,
    *,
    tokens: int,
    skip_mirror: bool,
) -> dict | None:
    plan = plan_cell(events_per_token)
    if plan is None:
        print(f"  {mode} {pct}%: CAPACITY-INFEASIBLE, skipped", flush=True)
        return None
    fixture = plan["fixture_residents"]
    print(
        f"  {mode} {pct}%: fixture={fixture:,} "
        f"(staged_peak_model={plan['staged_peak_model']:,})",
        flush=True,
    )
    rt = Runtime(stepping_mode=mode, plan=plan, weight_pa=0.25)
    row: dict = {
        "mode": mode,
        "activity_pct_of_configured": pct,
        "events_per_token": events_per_token,
        "vocabulary_pool": 2 * events_per_token,
        "capacity_plan": plan,
    }
    try:
        build_wall = build_residency(rt, fixture, BUILD_PER_WINDOW)
        rt.settle(SETTLE_STEPS)
        row["build"] = {
            "fixture_residents_built": fixture,
            "per_window": BUILD_PER_WINDOW,
            "settle_steps": SETTLE_STEPS,
            "build_wall_s": round(build_wall, 3),
            "state_after_settle": rt.live_state(),
        }

        # ---- timed section (starts strictly post-build) -------------------
        walls_ms = []
        work_sum = 0
        truncated = False
        for token in range(tokens):
            targets = token_targets(events_per_token, token, 2 * events_per_token)
            wall_ms = rt.advance(TOKEN_WINDOW_STEPS, targets) * 1e3
            walls_ms.append(wall_ms)
            work_sum += rt.total_work()
            if wall_ms > MAX_TOKEN_SECONDS_EFFECTIVE * 1e3 and token < tokens - 1:
                truncated = True
                print(
                    f"    token {token} took {wall_ms/1e3:.1f}s > "
                    f"{MAX_TOKEN_SECONDS_EFFECTIVE:.0f}s breaker; truncating stream",
                    flush=True,
                )
                break
        ordered = sorted(walls_ms)
        median_ms = statistics.median(walls_ms)
        row["stream"] = {
            "tokens_measured": len(walls_ms),
            "truncated_by_breaker": truncated,
            "median_token_ms": round(median_ms, 3),
            "p90_token_ms": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 3),
            "tokens_per_second_end_to_end": round(1e3 / median_ms, 3)
            if median_ms
            else None,
            "cell_steps_per_token_median_call": int(work_sum / len(walls_ms)),
            "walls_ms": [round(w, 1) for w in walls_ms],
        }
        row["final_state"] = rt.live_state()

        # ---- mirror-read diagnostic (optimized path vs fallback) ----------
        # Skipped entirely when the stream itself hit the breaker: re-driving
        # a schedule whose single tokens run minutes-to-hours adds nothing to
        # the mirror-cost question and is unbounded wall time.
        slow_stream = max(walls_ms, default=0.0) > MIRROR_SKIP_ABOVE_S * 1e3
        if truncated or slow_stream:
            row["mirror_read"] = {
                "ran": False,
                "path": "skipped_slow_stream",
                "reason": (
                    f"stream truncated={truncated} or max token "
                    f"{max(walls_ms, default=0.0)/1e3:.1f}s > "
                    f"{MIRROR_SKIP_ABOVE_S}s; per-token read overhead is "
                    "meaningless at this token cost"
                ),
            }
        else:
            row["mirror_read"] = mirror_diagnostic(
                rt, events_per_token, start_token=tokens, skip=skip_mirror
            )
    finally:
        rt.close()
    return row


def mirror_diagnostic(
    rt: Runtime, events_per_token: int, *, start_token: int, skip: bool
) -> dict:
    """Per-token host-mirror read cost at K=16 (Phase-B shape).

    Optimized path (Seam A present): re-drive a few schedule tokens; after
    each advance, read the previous token's first-K targets via read_cells
    (mirror pattern: read what you just touched). Reports median read wall
    and overhead vs the advance wall.
    Fallback (no Seam A): one snapshot()-based extraction for context;
    per-token mirroring is O(all residents) and marked unavailable.
    """
    if skip:
        return {"ran": False, "reason": "skipped by flag"}
    if not (HAS_READ_CELLS_FFI and HAS_READ_CELLS_PY):
        # Fallback path: the only read seam is O(all residents); per-token
        # mirroring is impossible at scale, so no per-token overhead number
        # exists. The incumbent's measured cost is recorded by reference.
        return {
            "ran": True,
            "path": "snapshot_fallback",
            "available": False,
            "note": (
                "read_cells absent: targeted per-token mirroring impossible. "
                "Incumbent snapshot()-extract measured in "
                "l2_eligibility_read_cost.json (161 ms/token @R=100k; ~1.6 s "
                "projected @1e6) and l4_seams_acceptance.json"
            ),
            "median_read_us": None,
            "overhead_pct_of_token": None,
        }
    advances_ms = []
    reads_ns = []
    pool = 2 * events_per_token
    prev_targets = token_targets(
        events_per_token, start_token - 1, pool
    )[:MIRROR_PROBE_K]
    for token in range(start_token, start_token + MIRROR_PROBE_TOKENS):
        targets = token_targets(events_per_token, token, pool)
        wall_ms = rt.advance(TOKEN_WINDOW_STEPS, targets) * 1e3
        advances_ms.append(wall_ms)
        _, ns = rt.read_cells_raw(prev_targets)
        reads_ns.append(ns)
        prev_targets = targets[:MIRROR_PROBE_K]
    median_adv = statistics.median(advances_ms)
    median_read_ns = statistics.median(reads_ns)
    return {
        "ran": True,
        "path": "read_cells_optimized",
        "probe_k": MIRROR_PROBE_K,
        "probe_tokens": MIRROR_PROBE_TOKENS,
        "median_read_us": round(median_read_ns / 1e3, 3),
        "median_advance_ms_context": round(median_adv, 3),
        "overhead_pct_of_token": round(
            (median_read_ns / 1e3) / (median_adv * 1e3) * 100.0, 4
        )
        if median_adv
        else None,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _raise_process_priority() -> None:
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(handle, 0x00000080)  # HIGH_PRIORITY_CLASS
    except (AttributeError, OSError):
        pass


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    ).stdout.strip()
    return out or "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=TOKENS_PER_CELL)
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--modes", type=str, default="ft,ep",
                        help="comma set from {ft,ep}")
    parser.add_argument("--activities", type=str, default="1,2,5,20",
                        help="comma list of activity percents")
    parser.add_argument("--outdir", type=str,
                        default=str(_REPO_ROOT / "experiments" / "l4_activity_matrix"))
    parser.add_argument("--skip-mirror", action="store_true")
    parser.add_argument("--probe-only", action="store_true",
                        help="run seam detection only, no matrix cells")
    parser.add_argument("--quick", action="store_true",
                        help="cap fixtures at 2^16 for fast smoke validation")
    parser.add_argument("--max-token-seconds", type=float, default=None,
                        help="override the per-token circuit breaker")
    args = parser.parse_args(argv)

    global MAX_TOKEN_SECONDS_EFFECTIVE
    if args.max_token_seconds is not None:
        MAX_TOKEN_SECONDS_EFFECTIVE = args.max_token_seconds

    if args.quick:
        global QUICK_FIXTURE_CAP
        QUICK_FIXTURE_CAP = 1 << 16

    _raise_process_priority()
    mode_map = {"ft": "full_touch", "ep": "event_proportional"}
    modes = tuple(mode_map[m.strip()] for m in args.modes.split(",") if m.strip())
    pct_to_count = {pct: count for pct, count in ACTIVITY_LEVELS}
    activities = tuple(
        (int(p), pct_to_count[int(p)]) for p in args.activities.split(",") if p.strip()
    )

    print(f"L4 activity-matrix harness @ commit {git_commit()}", flush=True)
    print("detecting seams ...", flush=True)
    read_cells_info = detect_read_cells_seam()
    indexed_info = detect_indexed_staged_seam()
    seam_level = []
    if read_cells_info["available"]:
        seam_level.append("seam_a")
    if indexed_info["indexed_staged_available"]:
        seam_level.append("seam_b")
    seam_detection = {
        "read_cells": read_cells_info,
        "indexed_staged": indexed_info,
        "seam_level": "+".join(seam_level) if seam_level else "none",
        "provenance": _provenance(),
    }
    print(
        f"  read_cells: available={read_cells_info['available']} "
        f"(ffi={read_cells_info['ffi_export_present']}, "
        f"py={read_cells_info['python_binding_present']})"
    )
    print(
        f"  indexed staged: verdict={indexed_info['verdict']} "
        f"(scan={indexed_info['scan_heavy_median_ms']}ms vs "
        f"ref={indexed_info['resident_reference_median_ms']}ms, "
        f"floor={indexed_info['linear_floor_ms']}ms)"
    )
    print(f"  seam level: {seam_detection['seam_level']}")

    results: dict = {
        "spec": (
            "L3 activity matrix rerun (FINDINGS.md 3.3(b)) extended to 20%; "
            "post-seam validation per docs/research/cpu-seams-spec.md"
        ),
        "label": args.label,
        "commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seam_detection": seam_detection,
        "protocol": {
            "configured_cells": CONFIGURED_CELLS,
            "activity_levels_pct": [p for p, _ in activities],
            "events_per_token": {str(p): c for p, c in activities},
            "modes": list(modes),
            "tokens_per_cell": args.tokens,
            "token_window_steps": TOKEN_WINDOW_STEPS,
            "build_recipe": (
                f"fresh runtime per cell; {BUILD_RESIDENTS_DEFAULT:,}-resident "
                f"default fixture in {BUILD_PER_WINDOW}-wide windows "
                "(chunked per spec; capacity-shrunk for high-activity rows); "
                f"settle {SETTLE_STEPS}; timing starts post-build"
            ),
            "pool_rule": (
                "2x events_per_token, alternating halves (L3 locality rule)"
            ),
            "fixture_sizing_rule": (
                "fixture = largest 2^k <= min(2^18, cfg - staged_peak); "
                "staged_peak = ceil(1.10 * cfg*(1-exp(-f*A/cfg))) + A "
                "(only injected cells spike; downstream is subthreshold)"
            ),
            "max_token_seconds_breaker": MAX_TOKEN_SECONDS_EFFECTIVE,
            "quick_mode": bool(args.quick),
            "stress_glif": "unit-leak stress configuration (L3-identical)",
        },
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"l4_activity_matrix_{args.label}.json"

    def _save() -> None:
        # Incremental artifact: every completed cell is durable, so an
        # interrupted run still yields its partial rows.
        out_path.write_text(json.dumps(results, indent=2))

    exit_code = 0
    if not args.probe_only:
        rows = []
        results["rows"] = rows
        for mode in modes:
            for pct, count in activities:
                row = run_cell(
                    mode, pct, count, tokens=args.tokens,
                    skip_mirror=args.skip_mirror,
                )
                if row is not None:
                    rows.append(row)
                print(
                    f"    -> {mode} {pct}%: "
                    + (
                        f"{row['stream']['median_token_ms']}ms/tok, "
                        f"{row['stream']['tokens_per_second_end_to_end']} tok/s"
                        if row else "skipped"
                    ),
                    flush=True,
                )
                _save()
        ratios = {}
        for pct, _ in activities:
            ep = next(
                (r for r in rows if r and r["mode"] == "event_proportional"
                 and r["activity_pct_of_configured"] == pct), None)
            ft = next(
                (r for r in rows if r and r["mode"] == "full_touch"
                 and r["activity_pct_of_configured"] == pct), None)
            if ep and ft:
                ratios[f"{pct}pct"] = round(
                    ep["stream"]["tokens_per_second_end_to_end"]
                    / ft["stream"]["tokens_per_second_end_to_end"], 3,
                ) if ft["stream"]["tokens_per_second_end_to_end"] else None
        results["ep_over_ft_ratio_by_activity"] = ratios

    _save()
    print(f"\nwrote {out_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
