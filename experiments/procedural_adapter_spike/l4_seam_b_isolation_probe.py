"""Seam B isolation probe: EP sparse-regime row against a chosen source tree.

Drives the exact FINDINGS 3.3(a)/B-G4 fixture (16,384-resident build,
400-step settle, 15 tokens of 512 fresh + 128 reactivation drives, stress
GLIF, event-proportional) through the native kernel built from --src-root,
and prints one JSON line {"ep_tok_s": ..., "ft_tok_s": ...}. Used to A/B the
indexed staged-membership build against an index-disabled variant under
identical, interleaved conditions.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True, type=Path)
    parser.add_argument("--label", default="probe")
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

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

    MASK64 = 0xFFFFFFFFFFFFFFFF
    CONFIGURED_CELLS = 1_000_000
    BUILD_TOTAL = 16_384
    BUILD_WINDOW = 2048
    WINDOW_STEPS = 100
    SETTLE_STEPS = 400
    TOKENS = 15
    QUEUE_HEADROOM = 8
    SPIKE_HEADROOM = 4
    MAX_BATCH = max(640, BUILD_WINDOW)

    STRESS_GLIF = _GlifConfig(
        1, 2, 3, 1.0, 1.0, 0.0, 0.9, 0.0, 0.0, 0.9,
        1.0, 0.0, 0.0, 0.5, 0.0,
    )
    DECAYS = (ctypes.c_double * 2)(0.5, 0.5)
    AMPLITUDES = (ctypes.c_double * 2)(0.0, 0.0)

    lib = _load_library()
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

    def fresh_targets(count: int, token_index: int, low: int, high: int):
        span = high - low
        state = (0x9E3779B97F4A7C15 * (token_index + 1)) & MASK64
        out = []
        seen = set()
        guard = 0
        while len(out) < count and guard < count * 20:
            guard += 1
            state ^= (state << 13) & MASK64
            state ^= state >> 7
            state ^= (state << 17) & MASK64
            gid = low + state % span
            if gid not in seen:
                seen.add(gid)
                out.append(gid)
        if len(out) < count:
            raise RuntimeError("target generation exhausted")
        return out

    def run_mode(mode_id: int) -> float:
        event_capacity = QUEUE_HEADROOM * MAX_BATCH
        spike_capacity = SPIKE_HEADROOM * MAX_BATCH
        config = NativeConfig(
            CONFIGURED_CELLS,
            4,
            71,
            65_536,
            event_capacity,
            spike_capacity,
            max(65_536, event_capacity),
            0.25,
            1,
            1e-8,
            math.exp(-0.05 / 80.0),
            mode_id,
        )
        handle = lib.humanbrain_procedural_runtime_create(
            ctypes.byref(STRESS_GLIF),
            ctypes.byref(config),
            DECAYS,
            DECAYS,
            AMPLITUDES,
        )
        if not handle:
            raise RuntimeError("native create failed")
        buffer = (NativeEvent * MAX_BATCH)()

        def advance(steps: int, targets) -> float:
            due = step_holder[0] + steps - 1
            for index, gid in enumerate(targets):
                entry = buffer[index]
                entry.due_step = due
                entry.target_gid = gid
                entry.source_gid = gid
                entry.edge_slot = 0
                entry.current_pa = 150_000.0
            start = time.perf_counter()
            status = lib.humanbrain_procedural_runtime_advance(
                handle, steps, buffer, len(targets)
            )
            elapsed = time.perf_counter() - start
            if status != 0:
                raise RuntimeError(f"advance status {status}")
            step_holder[0] += steps
            return elapsed

        step_holder = [0]
        try:
            built = 0
            while built < BUILD_TOTAL:
                chunk = min(BUILD_WINDOW, BUILD_TOTAL - built)
                advance(WINDOW_STEPS, range(built, built + chunk))
                built += chunk
            advance(SETTLE_STEPS, [])
            walls = []
            for token in range(TOKENS):
                targets = fresh_targets(512, token, 16_384, CONFIGURED_CELLS)
                targets += list(range(128))
                walls.append(advance(WINDOW_STEPS, targets))
            # one token per median window; walls are seconds per advance call
            return 1.0 / statistics.median(walls)
        finally:
            lib.humanbrain_procedural_runtime_destroy(handle)

    ep_rate = run_mode(1)
    print(
        json.dumps(
            {
                "label": args.label,
                "src_root": str(src_root),
                "ep_tok_s": round(ep_rate, 1),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
