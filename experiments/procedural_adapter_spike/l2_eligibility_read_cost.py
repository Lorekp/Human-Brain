"""L2 spike: what does reading eligibility for K target cells cost today?

Question (roadmap R5 prerequisite, Layer 2): the host-side SparseCellReadout
mirror needs the eligibility trace of a handful of cells per teach() call.
The only read seam today is ``snapshot()``, which has no GID filter: the
native side merges and copies every resident record and Python converts every
record into a frozen dataclass (procedural_runtime.py:_read_snapshot,
procedural.rs:humanbrain_procedural_runtime_snapshot).

Measured here, at 2^20 configured cells (the text-grow target):
  - counts()/snapshot() wall time vs resident count R and requested K;
  - the split between raw FFI copy and Python object conversion;
  - the cost of actually extracting K eligibilities once you have paid O(R);
  - host mirror maintenance alternatives at N_cfg = 1e6 (fresh np.zeros vs
    persistent-buffer scatter);
  - L1 event-marshal cost at the text-like injection rate (~2,600
    events/token) flagged as an open caveat in logs/text_scale_cpu_findings.md.

CPU-only, runs in ~15 s.
"""

from __future__ import annotations

import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

from humanbrain.procedural_runtime import (
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralRuntimeConfig,
    _NativeCounters,
    _NativeEvent,
    _NativeResident,
)
from humanbrain.procedural_topology import TopologyManifest

CONFIGURED_CELLS = 1_048_576
FANOUT = 8
FILL_CURRENT_PA = 100.0     # subthreshold drive: perturbs state, no spikes
SPIKE_CURRENT_PA = 150_000.0
TOKEN_STEPS = 100


def make_runtime(fill_count: int) -> NativeProceduralRuntime:
    # headroom so downstream expansion from hot-cell spikes cannot overflow
    capacity = fill_count + 8_192
    return NativeProceduralRuntime(
        ProceduralRuntimeConfig(
            manifest=TopologyManifest(
                cell_count=CONFIGURED_CELLS,
                fanout=FANOUT,
                topology="uniform_no_self",
                seed=71,
            ),
            resident_capacity=capacity,
            event_capacity=capacity + 16_384,
            spike_capacity=4_096,
            interval_distinct_capacity=capacity,
            recurrent_weight_pa=325.0,
            recurrent_delay_steps=1,
        )
    )


def timed(fn, repeats: int) -> float:
    """Median wall seconds of fn() over `repeats` calls."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def fill_residents(runtime: NativeProceduralRuntime, count: int) -> None:
    """Make exactly `count` cells resident with subthreshold current."""
    step = runtime._current_step
    events = [
        ProceduralInputEvent(step, gid, gid, 0, FILL_CURRENT_PA)
        for gid in range(count)
    ]
    result = runtime.advance(TOKEN_STEPS, events)
    assert result.spikes == (), (
        f"fill current produced {len(result.spikes)} spikes; "
        "lower FILL_CURRENT_PA"
    )


def main() -> int:
    rows: list[dict[str, object]] = []
    print(f"N_cfg={CONFIGURED_CELLS:,}  fanout={FANOUT}")
    print(
        f"{'residents':>10} {'counts_us':>10} {'ffi_only_us':>12} "
        f"{'snapshot_us':>12} {'ns/resident':>12} {'K_extract_us':>13}"
    )

    for residents in (1_000, 10_000, 50_000, 100_000):
        with make_runtime(residents) as runtime:
            fill_residents(runtime, residents)

            # give K real, spiky targets: strong-drive 64 cells so their own
            # eligibility traces are hot and their fanout edges deliver
            step = runtime._current_step
            hot_gids = list(range(residents, residents + 64))
            hot_events = [
                ProceduralInputEvent(step, gid, gid, 0, SPIKE_CURRENT_PA)
                for gid in hot_gids
            ]
            result = runtime.advance(TOKEN_STEPS, hot_events)
            hot_spiked = {spike.gid for spike in result.spikes}
            assert len(hot_spiked) == 64

            # counts-only roundtrip
            def counts_only() -> None:
                a = ctypes.c_size_t()
                b = ctypes.c_size_t()
                c = ctypes.c_size_t()
                runtime._library.humanbrain_procedural_runtime_counts(
                    runtime._handle, ctypes.byref(a), ctypes.byref(b), ctypes.byref(c)
                )

            counts_us = timed(counts_only, repeats=50) * 1e6

            # raw FFI: allocate R records, call snapshot, touch nothing
            def ffi_only() -> None:
                rc = ctypes.c_size_t(); pc = ctypes.c_size_t(); sc = ctypes.c_size_t()
                runtime._library.humanbrain_procedural_runtime_counts(
                    runtime._handle, ctypes.byref(rc), ctypes.byref(pc), ctypes.byref(sc)
                )
                buffer = (_NativeResident * rc.value)() if rc.value else None
                counters = _NativeCounters()
                runtime._library.humanbrain_procedural_runtime_snapshot(
                    runtime._handle,
                    buffer, 0 if buffer is None else len(buffer),
                    None, 0,
                    None, 0,
                    ctypes.byref(counters),
                )

            ffi_repeats = 20 if residents <= 10_000 else 5
            ffi_us = timed(ffi_only, repeats=ffi_repeats) * 1e6

            # full public snapshot(): FFI + Python dataclass conversion
            snap_repeats = 10 if residents <= 10_000 else 3
            snapshot_us = timed(lambda: runtime.snapshot(), repeats=snap_repeats) * 1e6

            # the actual read: K cells' eligibility out of the converted blob
            snapshot = runtime.snapshot()
            states = {state.gid: state for state in snapshot.residents}
            k_targets = hot_gids[:16]

            def extract_k() -> None:
                for gid in k_targets:
                    states[gid].eligibility

            extract_us = timed(extract_k, repeats=200) * 1e6
            assert all(states[gid].eligibility > 0.9 for gid in k_targets)

            ns_per_resident = snapshot_us * 1e3 / max(len(snapshot.residents), 1)
            rows.append(
                {
                    "configured_cells": CONFIGURED_CELLS,
                    "requested_read_cells": len(k_targets),
                    "residents": len(snapshot.residents),
                    "hot_cells_spiked": len(hot_spiked),
                    "counts_us": round(counts_us, 2),
                    "ffi_only_us": round(ffi_us, 2),
                    "snapshot_full_us": round(snapshot_us, 2),
                    "ns_per_resident": round(ns_per_resident, 1),
                    "k_extract_us": round(extract_us, 3),
                }
            )
            print(
                f"{len(snapshot.residents):>10,} {counts_us:>10.1f} "
                f"{ffi_us:>12.1f} {snapshot_us:>12.1f} {ns_per_resident:>12.0f} "
                f"{extract_us:>13.2f}"
            )

    # --- host mirror maintenance at N_cfg = 1e6 ---
    import numpy as np

    n_cfg = CONFIGURED_CELLS
    gids = np.arange(100_000, dtype=np.int64)
    values = np.random.default_rng(0).uniform(size=100_000)

    fresh_us = timed(lambda: np.zeros(n_cfg, dtype=np.float64), repeats=50) * 1e6
    buffer = np.zeros(n_cfg, dtype=np.float64)

    def scatter() -> None:
        buffer[gids] = values
        buffer[gids] = 0.0  # restore for repeat

    scatter_us = timed(scatter, repeats=50) * 1e6
    print(f"\nmirror reset (fresh np.zeros(1e6)):      {fresh_us:8.1f} us")
    print(f"persistent buffer scatter/reset (R=100k): {scatter_us:8.1f} us")

    # --- L1 event-marshal cost at text-like rate ---
    with make_runtime(1_024) as runtime:
        # text-like batch: distinct target gids -> canonical keys unique;
        # rebuilt per call so due_step tracks the moving runtime step
        def marshal_only() -> list[ProceduralInputEvent]:
            s = runtime._current_step
            return [
                ProceduralInputEvent(s, gid, gid, 0, FILL_CURRENT_PA)
                for gid in range(2_600)
            ]

        batch = marshal_only()
        marshal_us = timed(marshal_only, repeats=20) * 1e6

        def marshal_and_advance() -> None:
            runtime.advance(1, marshal_only())

        advance_us = timed(marshal_and_advance, repeats=20) * 1e6
    print(f"\nevent marshal ({len(batch)} events):       {marshal_us:8.1f} us")
    print(f"advance(1, {len(batch)} events) incl delivery:  {advance_us:8.1f} us")

    projection = {
        row["residents"]: row["snapshot_full_us"] / 1e3 for row in rows
    }
    summary = {
        "rows": rows,
        "mirror_fresh_zeros_1e6_us": round(fresh_us, 1),
        "mirror_scatter_reset_R100k_us": round(scatter_us, 1),
        "event_marshal_2600_us": round(marshal_us, 1),
        "advance_with_2600_events_us": round(advance_us, 1),
        "linear_projection_snapshot_s_at_1e6_residents":
            round(rows[-1]["ns_per_resident"] * 1_000_000 / 1e9, 2),
    }
    out_path = Path(__file__).with_name("l2_eligibility_read_cost.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path.name}")
    print(
        "linear projection: full snapshot at 1e6 residents ~"
        f" {summary['linear_projection_snapshot_s_at_1e6_residents']} s/token"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
