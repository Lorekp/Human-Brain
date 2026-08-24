"""Seam B bit-parity + timing driver (docs/research/cpu-seams-spec.md B.8b).

Runs ONE fixed stress-GLIF schedule through a native procedural runtime and
writes spike traces plus delivered/generated counts to JSON so two builds can
be compared bit-for-bit (gate B-G3) and the post-change build's per-token wall
times can be gated (B-G2: N_cfg=1e6, 5% uniform activity = 50k events/token,
fanout 4, EP and FT, 15 tokens).

The schedule is pure arithmetic (no RNG): residency ratchet of 2^18 cells in
2,048-event windows, settle, then alternating halves of a 100k vocabulary
pool -- byte-identical across any two runs of this script regardless of which
source tree backs the native library.

Self-contained on purpose: ``--src-root`` selects WHICH tree's
``humanbrain`` package (and therefore which natively built kernel) drives the
run -- the acceptance harness points it at the landing tree and at the
pre-Seam-B parity-baseline worktree. Timing starts post-build: only the
per-token ``advance()`` calls are timed, never the fixture construction
(public-API builds pay quadratic-class terms; see cpu-seams-spec.md Seam A/B
fixture warnings).

Usage:
  python l4_seam_b_parity_driver.py --src-root <tree>/src --out result.json
  [--library <explicit kernel path>]   # bypasses build_native()/attestation
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--library", type=Path, default=None)
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    # Imported AFTER the path insert: resolves humanbrain (and its native
    # build plan / cache) against the selected tree.
    from humanbrain.native import _GlifConfig, _load_library

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

    class NativeSpike(ctypes.Structure):
        _fields_ = [
            ("step", ctypes.c_uint64),
            ("gid", ctypes.c_uint64),
        ]

    # --- fixed fixture (identical across every tree) ------------------------
    CONFIGURED_CELLS = 1_000_000
    BUILD_TOTAL = 1 << 18
    BUILD_WINDOW = 2048
    WINDOW_STEPS = 100
    SETTLE_STEPS = 400
    TOKENS = 15
    EVENTS_PER_TOKEN = 50_000          # 5% uniform activity
    VOCAB_POOL = 2 * EVENTS_PER_TOKEN
    DRIVE_CURRENT_PA = 150_000.0

    STRESS_GLIF = _GlifConfig(
        1,      # cell_count (unused by the sparse runtime)
        2,      # after_channel_count
        3,      # spike_cut_steps
        1.0,    # leak_ns -> v' = I exactly
        1.0,    # voltage_step_scale
        0.0,    # threshold_beta_scale
        0.9,    # threshold_exp_decay
        0.0,    # threshold_voltage_factor
        0.0,    # threshold_beta_factor
        0.9,    # spike_threshold_decay
        1.0,    # threshold_inf_relative_mv
        0.0,    # voltage_reset_scale
        0.0,    # voltage_reset_offset_mv
        0.5,    # spike_cut_threshold_decay
        0.0,    # threshold_spike_increment_mv
    )
    DECAYS = (ctypes.c_double * 2)(0.5, 0.5)
    AMPLITUDES = (ctypes.c_double * 2)(0.0, 0.0)

    if args.library is not None:
        lib = ctypes.CDLL(str(args.library))
        if lib.humanbrain_kernel_abi_version() != 3:
            raise RuntimeError(f"ABI mismatch in {args.library}")
    else:
        lib = _load_library()

    mode_ids = {"full_touch": 0, "event_proportional": 1}
    max_batch = EVENTS_PER_TOKEN
    QUEUE_HEADROOM = 8
    SPIKE_HEADROOM = 4
    event_capacity = QUEUE_HEADROOM * max_batch
    spike_capacity = SPIKE_HEADROOM * max_batch

    lib.humanbrain_procedural_runtime_create.argtypes = [
        ctypes.POINTER(_GlifConfig),
        ctypes.POINTER(NativeConfig),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.humanbrain_procedural_runtime_create.restype = ctypes.c_void_p
    lib.humanbrain_procedural_runtime_destroy.argtypes = [ctypes.c_void_p]
    lib.humanbrain_procedural_runtime_advance.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(NativeEvent),
        ctypes.c_size_t,
    ]
    lib.humanbrain_procedural_runtime_advance.restype = ctypes.c_uint32
    lib.humanbrain_procedural_runtime_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(NativeSpike),
        ctypes.c_size_t,
        ctypes.POINTER(NativeCounters),
    ]
    lib.humanbrain_procedural_runtime_snapshot.restype = ctypes.c_uint32

    def commit_of(tree_hint: Path) -> str:
        probe = tree_hint
        for _ in range(6):
            if (probe / ".git").exists():
                completed = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=probe,
                    check=False,
                )
                if completed.returncode == 0:
                    return completed.stdout.strip()
            probe = probe.parent
        return "unknown"

    def run_mode(mode: str) -> dict:
        config = NativeConfig(
            CONFIGURED_CELLS,
            4,                       # fanout (shipped)
            71,                      # seed
            CONFIGURED_CELLS,        # resident capacity
            event_capacity,
            spike_capacity,
            CONFIGURED_CELLS,        # interval distinct >= resident
            0.25,                    # subthreshold recurrent weight
            1,                       # recurrent delay steps
            1e-8,                    # aggressive synaptic decay (stress cfg)
            math.exp(-0.05 / 80.0),  # realistic eligibility tau (stress cfg)
            mode_ids[mode],
        )
        handle = lib.humanbrain_procedural_runtime_create(
            ctypes.byref(STRESS_GLIF),
            ctypes.byref(config),
            DECAYS,
            DECAYS,
            AMPLITUDES,
        )
        if not handle:
            raise RuntimeError(f"native create failed ({mode})")
        buffer = (NativeEvent * max_batch)()
        trace_buffer = (NativeSpike * spike_capacity)()
        counters = NativeCounters()

        def advance(steps: int, targets) -> float:
            due = step_holder[0] + steps - 1
            for index, gid in enumerate(targets):
                entry = buffer[index]
                entry.due_step = due
                entry.target_gid = gid
                entry.source_gid = gid
                entry.edge_slot = 0
                entry.current_pa = DRIVE_CURRENT_PA
            start = time.perf_counter()
            status = lib.humanbrain_procedural_runtime_advance(
                handle, steps, buffer, len(targets)
            )
            elapsed = time.perf_counter() - start
            if status != 0:
                raise RuntimeError(
                    f"advance failed with status {status} at step {step_holder[0]} ({mode})"
                )
            step_holder[0] += steps
            return elapsed

        def collect_spikes() -> list:
            status = lib.humanbrain_procedural_runtime_snapshot(
                handle,
                None,
                0,
                None,
                0,
                trace_buffer,
                spike_capacity,
                ctypes.byref(counters),
            )
            assert status == 0, f"snapshot failed with {status}"
            return [
                [int(trace_buffer[offset].step), int(trace_buffer[offset].gid)]
                for offset in range(counters.spike_events)
            ]

        step_holder = [0]
        try:
            build_started = time.perf_counter()
            built = 0
            while built < BUILD_TOTAL:
                chunk = min(BUILD_WINDOW, BUILD_TOTAL - built)
                advance(WINDOW_STEPS, range(built, built + chunk))
                built += chunk
            advance(SETTLE_STEPS, [])

            walls_ms: list = []
            spikes: list = []
            for token in range(TOKENS):
                offset = ((token % 2) * EVENTS_PER_TOKEN) % VOCAB_POOL
                targets = [(offset + i) % VOCAB_POOL for i in range(EVENTS_PER_TOKEN)]
                walls_ms.append(advance(WINDOW_STEPS, targets) * 1000.0)
                spikes.extend(collect_spikes())

            median_ms = statistics.median(walls_ms)
            return {
                "token_walls_ms": [round(value, 3) for value in walls_ms],
                "median_token_ms": round(median_ms, 3),
                "tokens_per_second_end_to_end": round(1000.0 / median_ms, 1),
                "spikes": spikes,
                "delivered_events": int(counters.delivered_events),
                "generated_edges": int(counters.generated_edges),
                "cell_step_evaluations": int(counters.cell_step_evaluations),
                "resident_peak": int(counters.resident_peak),
                "spike_events": int(counters.spike_events),
                "final_step": int(counters.current_step),
                "build_wall_s": round(time.perf_counter() - build_started, 1),
            }
        finally:
            lib.humanbrain_procedural_runtime_destroy(handle)

    results = {
        "driver": "l4_seam_b_parity_driver.py",
        "src_root": str(src_root),
        "source_commit": commit_of(src_root),
        "schedule": {
            "configured_cells": CONFIGURED_CELLS,
            "residency_built": BUILD_TOTAL,
            "build_window": BUILD_WINDOW,
            "window_steps": WINDOW_STEPS,
            "settle_steps": SETTLE_STEPS,
            "tokens": TOKENS,
            "events_per_token": EVENTS_PER_TOKEN,
            "vocabulary_pool": VOCAB_POOL,
            "drive_current_pa": DRIVE_CURRENT_PA,
            "fanout": 4,
            "weight_pa": 0.25,
        },
        "modes": {
            mode: run_mode(mode) for mode in ("event_proportional", "full_touch")
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    for mode, record in results["modes"].items():
        print(
            f"{mode}: median {record['median_token_ms']} ms/token "
            f"({record['tokens_per_second_end_to_end']} tok/s), "
            f"spikes {record['spike_events']}, "
            f"delivered {record['delivered_events']}, "
            f"generated {record['generated_edges']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
