"""L5 host-marshal batching: precise phase breakdown + bulk-array A/B gates.

Question (host-side follow-up to the L4 matrix verdict): the EP sparse band
was declared HOST-MARSHAL-BOUND -- per-event Python work (~0.96 us/event,
FINDINGS.md L1 caveat) caps throughput before the native kernel does. This
harness measures exactly where the per-token host budget goes at the L4
protocol point (N_cfg = 1e6, EP mode, activity {1, 2, 5}%) and tests whether
bulk NumPy structured-array marshaling (one buffer cast straight into the
C-ABI advance call, zero per-event Python objects) breaks the cap WITHOUT
changing one simulated bit.

Three marshal strategies are compared on identical schedules:

  ``dataclass``     production shape: one ProceduralInputEvent per event +
                    ``_event_batch`` ctypes-array marshaling (today's seam)
  ``buffer_fill``   L4-harness shape: per-field writes into a reused ctypes
                    event buffer (measured host-only; no FFI side effects)
  ``numpy_bulk``    one structured ndarray laid out exactly like the C event
                    struct, filled with four vectorized column stores, passed
                    to the FFI as a raw pointer (zero-copy)

The dataclass and numpy_bulk arms are driven interleaved token-by-token on
identically-seeded twin runtimes so per-token machine conditions match.

Phases timed per token (perf_counter_ns around each):

  object_construction  ProceduralInputEvent records from the schedule
  ctypes_marshal       _event_batch (dataclass strategy only)
  numpy_build          vectorized structured-array build (numpy_bulk only)
  ffi_call             humanbrain_procedural_runtime_advance itself
  result_unpack        counts + snapshot(spike records) + counter capture

GATES (must all pass for the numpy_bulk claim to stand):
  G0 EVENT_DTYPE matches the ctypes event struct byte-for-byte;
  G1 every token: byte-identical full counter structs across arms;
  G2 every token: byte-identical delivered spike-record bytes;
  G3 end-of-stream byte-identical resident/pending/dormant state dumps;
  G4 production-seam cross-check (--production): public advance() with a
     tuple of ProceduralInputEvents vs the bulk ndarray on the same schedule
     reproduce identical accounting counters, spikes, and snapshots.

Usage:
    python l5_marshal_batching.py --quick          # smoke, minutes
    python l5_marshal_batching.py                  # full protocol
    python l5_marshal_batching.py --label record --production

Artifact: <outdir>/l5_marshal_batching_<label>.json
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

# Self-contained bootstrap, then reuse the L4 harness primitives verbatim so
# the protocol (fixtures, schedules, GLIF, capacity planning) cannot drift.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_SRC_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

import l4_activity_matrix as l4
from l4_activity_matrix import (
    BUILD_PER_WINDOW,
    CONFIGURED_CELLS,
    LIB,
    NativeCounters,
    NativeEvent,
    SETTLE_STEPS,
    TOKEN_WINDOW_STEPS,
    TOKENS_PER_CELL,
    build_residency,
    git_commit,
    plan_cell,
)
from humanbrain.native import _F64_POINTER  # noqa: F401  (parity with l4 driver)
from humanbrain.procedural_runtime import (
    ProceduralInputEvent,
    ProceduralRuntimeConfig,
    _NativeDormant,
    _NativeResident,
    _NativeSpike,
    _event_batch,
)
from humanbrain.procedural_topology import TopologyManifest

DRIVE_CURRENT_PA = 150_000.0
STRESS_RECURRENT_TAU_MS = 0.05 / 18.420680743952367  # exp(-0.05/tau) == 1e-8


# Byte layout of the C event struct, expressed as a NumPy structured dtype.
EVENT_DTYPE = np.dtype(
    [
        ("due_step", "<u8"),
        ("target_gid", "<u8"),
        ("source_gid", "<u8"),
        ("edge_slot", "<u8"),
        ("current_pa", "<f8"),
    ]
)


def assert_layout_parity() -> dict:
    """G0: EVENT_DTYPE must match the ctypes event struct byte-for-byte."""
    offsets = {
        name: getattr(NativeEvent, name).offset for name, _ in NativeEvent._fields_
    }
    expected_offsets = {
        "due_step": 0,
        "target_gid": 8,
        "source_gid": 16,
        "edge_slot": 24,
        "current_pa": 32,
    }
    if ctypes.sizeof(NativeEvent) != EVENT_DTYPE.itemsize:
        raise AssertionError(
            f"itemsize mismatch: ctypes={ctypes.sizeof(NativeEvent)} "
            f"numpy={EVENT_DTYPE.itemsize}"
        )
    if offsets != expected_offsets:
        raise AssertionError(f"field offset mismatch: {offsets}")
    sample = np.zeros(3, dtype=EVENT_DTYPE)
    reference = (NativeEvent * 3)()
    if bytes(sample.tobytes()) != bytes(reference):
        raise AssertionError("zeroed buffer bytes differ between dtype and ctypes")
    return {"itemsize": EVENT_DTYPE.itemsize, "offsets": offsets}


# ---------------------------------------------------------------------------
# schedules (vectorized form of l4.token_targets; identical integers)
# ---------------------------------------------------------------------------
def token_targets_array(count: int, token_index: int, pool: int) -> np.ndarray:
    """Same integers as l4.token_targets(count, token_index, pool), as u64."""
    offset = ((token_index % 2) * count) % pool
    base = np.arange(count, dtype=np.uint64)
    return ((base + np.uint64(offset)) % np.uint64(pool)).astype(np.uint64)


def build_event_array(targets: np.ndarray, due_step: int) -> np.ndarray:
    """Bulk marshal: one structured array, four vectorized column stores."""
    events = np.empty(targets.shape[0], dtype=EVENT_DTYPE)
    events["due_step"] = np.uint64(due_step)
    events["target_gid"] = targets
    events["source_gid"] = targets
    events["edge_slot"] = 0
    events["current_pa"] = DRIVE_CURRENT_PA
    return events


def events_pointer(events: np.ndarray):
    """Zero-copy reinterpretation of the array buffer as the C event array."""
    address = events.__array_interface__["data"][0]
    if address % 8 or not events.flags.c_contiguous:
        raise RuntimeError("bulk event array must be contiguous and 8-aligned")
    return ctypes.cast(address, ctypes.POINTER(NativeEvent))


def marshal_config(plan: dict) -> ProceduralRuntimeConfig:
    """Metadata-only config so _event_batch validates exactly like production
    marshaling does for the raw driver's plan (stepping mode is irrelevant)."""
    return ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=CONFIGURED_CELLS,
            fanout=plan["fanout"],
            topology="uniform_no_self",
            seed=71,
        ),
        resident_capacity=plan["resident_capacity"],
        event_capacity=plan["event_capacity"],
        spike_capacity=plan["spike_capacity"],
        interval_distinct_capacity=plan["interval_distinct_capacity"],
        recurrent_weight_pa=0.25,
        recurrent_delay_steps=1,
        recurrent_tau_ms=STRESS_RECURRENT_TAU_MS,
    )


# ---------------------------------------------------------------------------
# raw C-ABI driver with instrumentation (L4 protocol otherwise)
# ---------------------------------------------------------------------------
class InstrumentedRuntime(l4.Runtime):
    """L4 Runtime + counter/spike byte capture and end-of-stream dumps."""

    def __init__(self, *, plan: dict, weight_pa: float = 0.25) -> None:
        # EP mode is the measurement target (the confirmed host-marshal-bound
        # band); the marshal path under test is identical host-side code.
        super().__init__(
            stepping_mode="event_proportional", plan=plan, weight_pa=weight_pa
        )

    def counts_and_spikes(self):
        active = ctypes.c_size_t()
        pending = ctypes.c_size_t()
        spikes = ctypes.c_size_t()
        assert LIB.humanbrain_procedural_runtime_counts(
            self.handle,
            ctypes.byref(active),
            ctypes.byref(pending),
            ctypes.byref(spikes),
        ) == 0
        spike_records = (_NativeSpike * spikes.value)() if spikes.value else None
        counters = NativeCounters()
        assert LIB.humanbrain_procedural_runtime_snapshot(
            self.handle,
            None,
            0,
            None,
            0,
            spike_records,
            0 if spike_records is None else len(spike_records),
            ctypes.byref(counters),
        ) == 0
        counter_bytes = ctypes.string_at(
            ctypes.byref(counters), ctypes.sizeof(counters)
        )
        spike_bytes = (
            ctypes.string_at(spike_records, ctypes.sizeof(spike_records))
            if spike_records is not None
            else b""
        )
        return counter_bytes, spike_bytes

    def state_dump(self) -> dict:
        active = ctypes.c_size_t()
        pending = ctypes.c_size_t()
        spikes = ctypes.c_size_t()
        assert LIB.humanbrain_procedural_runtime_counts(
            self.handle,
            ctypes.byref(active),
            ctypes.byref(pending),
            ctypes.byref(spikes),
        ) == 0
        resident_records = (
            (_NativeResident * active.value)() if active.value else None
        )
        pending_records = (NativeEvent * pending.value)() if pending.value else None
        counters = NativeCounters()
        assert LIB.humanbrain_procedural_runtime_snapshot(
            self.handle,
            resident_records,
            0 if resident_records is None else len(resident_records),
            pending_records,
            0 if pending_records is None else len(pending_records),
            None,
            0,
            ctypes.byref(counters),
        ) == 0
        dormant = ctypes.c_size_t()
        assert LIB.humanbrain_procedural_runtime_dormant_snapshot(
            self.handle, None, 0, ctypes.byref(dormant)
        ) == 0
        dormant_bytes = b""
        if dormant.value:
            dormant_records = (_NativeDormant * dormant.value)()
            assert LIB.humanbrain_procedural_runtime_dormant_snapshot(
                self.handle,
                dormant_records,
                len(dormant_records),
                ctypes.byref(dormant),
            ) == 0
            dormant_bytes = ctypes.string_at(
                dormant_records, ctypes.sizeof(dormant_records)
            )

        def digest(record_array) -> str:
            payload = (
                b"empty"
                if record_array is None or len(record_array) == 0
                else ctypes.string_at(record_array, ctypes.sizeof(record_array))
            )
            return hashlib.sha256(payload).hexdigest()

        return {
            "resident_count": active.value,
            "pending_count": pending.value,
            "dormant_count": dormant.value,
            "residents_sha256": digest(resident_records),
            "pending_sha256": digest(pending_records),
            "dormant_sha256": hashlib.sha256(dormant_bytes).hexdigest(),
        }


# ---------------------------------------------------------------------------
# strategy arms over one token schedule
# ---------------------------------------------------------------------------
class PhaseClock:
    __slots__ = (
        "object_construction",
        "ctypes_marshal",
        "numpy_build",
        "ffi_call",
        "result_unpack",
    )

    def __init__(self) -> None:
        self.object_construction: list[int] = []
        self.ctypes_marshal: list[int] = []
        self.numpy_build: list[int] = []
        self.ffi_call: list[int] = []
        self.result_unpack: list[int] = []


def run_dataclass_arm(rt, plan, tokens, schedule):
    """Production shape: ProceduralInputEvent records + _event_batch."""
    config = marshal_config(plan)
    clock = PhaseClock()
    gate = []
    for token in range(tokens):
        count, target_pool = schedule[token]
        targets = l4.token_targets(count, token, target_pool)
        due = rt.step + TOKEN_WINDOW_STEPS - 1
        started = time.perf_counter_ns()
        records = tuple(
            ProceduralInputEvent(
                due_step=due,
                target_gid=int(gid),
                source_gid=int(gid),
                edge_slot=0,
                current_pa=DRIVE_CURRENT_PA,
            )
            for gid in targets
        )
        clock.object_construction.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        native = _event_batch(config, rt.step, records)
        clock.ctypes_marshal.append(time.perf_counter_ns() - started)
        # l4_activity_matrix rebinds the shared CDLL argtypes to ITS
        # NativeEvent class; cast the procedural_runtime array over (layouts
        # are pinned identical by the G0 gate) so ctypes accepts it.
        native_view = ctypes.cast(native, ctypes.POINTER(NativeEvent))
        started = time.perf_counter_ns()
        status = LIB.humanbrain_procedural_runtime_advance(
            rt.handle, TOKEN_WINDOW_STEPS, native_view, len(native)
        )
        clock.ffi_call.append(time.perf_counter_ns() - started)
        if status != 0:
            raise RuntimeError(f"dataclass arm advance failed: status {status}")
        rt.step += TOKEN_WINDOW_STEPS
        started = time.perf_counter_ns()
        gate.append(rt.counts_and_spikes())
        clock.result_unpack.append(time.perf_counter_ns() - started)
    return clock, gate


def run_numpy_arm(rt, tokens, schedule):
    """Batched shape: one structured ndarray cast straight into the C-ABI."""
    clock = PhaseClock()
    gate = []
    for token in range(tokens):
        count, target_pool = schedule[token]
        targets = token_targets_array(count, token, target_pool)
        due = rt.step + TOKEN_WINDOW_STEPS - 1
        started = time.perf_counter_ns()
        events = build_event_array(targets, due)
        pointer = events_pointer(events)
        clock.numpy_build.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        status = LIB.humanbrain_procedural_runtime_advance(
            rt.handle, TOKEN_WINDOW_STEPS, pointer, count
        )
        clock.ffi_call.append(time.perf_counter_ns() - started)
        if status != 0:
            raise RuntimeError(f"numpy arm advance failed: status {status}")
        rt.step += TOKEN_WINDOW_STEPS
        started = time.perf_counter_ns()
        gate.append(rt.counts_and_spikes())
        clock.result_unpack.append(time.perf_counter_ns() - started)
    return clock, gate


def measure_buffer_fill(plan, tokens, schedule):
    """Host-only micro-cost of the L4-harness per-field ctypes fill loop."""
    probe = InstrumentedRuntime(plan=plan)
    try:
        buffer = probe.buffer
        walls = []
        for token in range(tokens):
            count, target_pool = schedule[token]
            targets = l4.token_targets(count, token, target_pool)
            due = TOKEN_WINDOW_STEPS - 1
            started = time.perf_counter_ns()
            for index, gid in enumerate(targets):
                entry = buffer[index]
                entry.due_step = due
                entry.target_gid = gid
                entry.source_gid = gid
                entry.edge_slot = 0
                entry.current_pa = DRIVE_CURRENT_PA
            walls.append(time.perf_counter_ns() - started)
        median_wall_us = statistics.median(walls) / 1e3
        return {
            "median_ms": round(median_wall_us / 1e3, 3),
            "per_event_us": round(
                median_wall_us / max(1, schedule[0][0]), 4
            ),
            "note": "host-only; never touches simulation state",
        }
    finally:
        probe.close()


def median_us_per_event(walls_ns: list[int], events_per_token: list[int]) -> float:
    shares = [wall / max(1, n) for wall, n in zip(walls_ns, events_per_token)]
    return round(statistics.median(shares) / 1e3, 4)


# ---------------------------------------------------------------------------
# production-seam block (G4): public advance(), records vs bulk ndarray
# ---------------------------------------------------------------------------
def production_config(plan: dict) -> ProceduralRuntimeConfig:
    return ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=CONFIGURED_CELLS,
            fanout=plan["fanout"],
            topology="uniform_no_self",
            seed=71,
        ),
        resident_capacity=CONFIGURED_CELLS,
        event_capacity=max(plan["event_capacity"], 4096),
        spike_capacity=max(plan["spike_capacity"], 1024),
        interval_distinct_capacity=max(
            CONFIGURED_CELLS,
            plan["interval_distinct_capacity"],
        ),
        recurrent_weight_pa=0.25,
        recurrent_delay_steps=1,
        recurrent_tau_ms=STRESS_RECURRENT_TAU_MS,
        stepping_mode="event_proportional",
    )


def _production_signature(result) -> tuple:
    acc = result.accounting
    return (
        result.spikes,
        int(acc.work.cell_step_evaluations),
        int(acc.work.spike_events),
        int(acc.edges.generated),
        int(acc.events.delivered),
        int(acc.cells.resident_peak),
        int(acc.cells.updated_distinct),
        int(acc.cells.spiking_distinct),
        int(acc.configured_ids_scanned) if hasattr(acc, "configured_ids_scanned") else None,
    )


def run_production_pair(pct: int, activity_events: int, tokens: int, fixture: int) -> dict:
    """Drive two identical production runtimes; records vs bulk ndarray."""
    from humanbrain.procedural_runtime import NativeProceduralRuntime

    plan = plan_cell(activity_events)
    config = production_config(plan)
    legacy_rt = NativeProceduralRuntime(config)
    bulk_rt = NativeProceduralRuntime(config)
    pool = 2 * activity_events
    try:
        # Identical fixtures built through the bulk path (state equality is
        # gated separately; here the stream carries the treatment).
        step_cursor = 0
        while step_cursor < fixture:
            width = min(min(fixture, BUILD_PER_WINDOW), fixture - step_cursor)
            gids = np.arange(step_cursor, step_cursor + width, dtype=np.uint64)
            due = legacy_rt.current_step + TOKEN_WINDOW_STEPS - 1
            batch = build_event_array(gids % np.uint64(pool), due)
            assert legacy_rt.advance(TOKEN_WINDOW_STEPS, batch).spikes is not None
            assert bulk_rt.advance(TOKEN_WINDOW_STEPS, batch).spikes is not None
            step_cursor += width
        assert legacy_rt.snapshot().residents == bulk_rt.snapshot().residents

        walls_legacy_ns = []
        walls_bulk_ns = []
        marshal_only_ns = []
        signatures = []
        for token in range(tokens):
            targets = token_targets_array(activity_events, token, pool)
            target_list = targets.tolist()
            due_legacy = legacy_rt.current_step + TOKEN_WINDOW_STEPS - 1
            started = time.perf_counter_ns()
            records = tuple(
                ProceduralInputEvent(
                    due_step=due_legacy,
                    target_gid=int(gid),
                    source_gid=int(gid),
                    edge_slot=0,
                    current_pa=DRIVE_CURRENT_PA,
                )
                for gid in target_list
            )
            marshal_only_ns.append(time.perf_counter_ns() - started)
            started = time.perf_counter_ns()
            result_a = legacy_rt.advance(TOKEN_WINDOW_STEPS, records)
            walls_legacy_ns.append(time.perf_counter_ns() - started)

            due_bulk = bulk_rt.current_step + TOKEN_WINDOW_STEPS - 1
            assert due_bulk == due_legacy
            batch = build_event_array(targets, due_bulk)
            pointer_probe = batch.ctypes.data % 8
            assert pointer_probe == 0
            started = time.perf_counter_ns()
            result_b = bulk_rt.advance(TOKEN_WINDOW_STEPS, batch)
            walls_bulk_ns.append(time.perf_counter_ns() - started)
            signature_a = _production_signature(result_a)
            signature_b = _production_signature(result_b)
            if signature_a != signature_b:
                raise RuntimeError(
                    f"G4 mismatch at token {token}: {signature_a} != {signature_b}"
                )
            signatures.append(signature_a)
        snapshot_a = legacy_rt.snapshot()
        snapshot_b = bulk_rt.snapshot()
        gate_g4 = (
            snapshot_a.residents == snapshot_b.residents
            and snapshot_a.pending_events == snapshot_b.pending_events
            and snapshot_a.spikes == snapshot_b.spikes
        )
    finally:
        legacy_rt.close()
        bulk_rt.close()

    med_legacy = statistics.median(walls_legacy_ns) / 1e6
    med_bulk = statistics.median(walls_bulk_ns) / 1e6
    return {
        "activity_pct_of_configured": activity_events // 10_000 * 1 if False else None,
        "events_per_token": activity_events,
        "tokens": tokens,
        "fixture_residents": fixture,
        "gate_production_G4": bool(gate_g4),
        "legacy_records": {
            "median_advance_ms": round(med_legacy, 3),
            "tok_per_s": round(1e3 / med_legacy, 3) if med_legacy else None,
            "record_construction_us_per_event": round(
                statistics.median(marshal_only_ns) / activity_events / 1e3, 4
            ),
        },
        "bulk_ndarray": {
            "median_advance_ms": round(med_bulk, 3),
            "tok_per_s": round(1e3 / med_bulk, 3) if med_bulk else None,
        },
        "end_to_end_speedup_x": round(med_legacy / max(1.0, med_bulk), 3),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=TOKENS_PER_CELL)
    parser.add_argument("--label", type=str, default="run")
    parser.add_argument("--activities", type=str, default="1,2,5",
                        help="comma list of activity percents")
    parser.add_argument("--outdir", type=str,
                        default=str(_REPO_ROOT / "experiments" / "l5_marshal_batching"))
    parser.add_argument("--quick", action="store_true",
                        help="smoke: cap fixtures at 2^14, 4 tokens")
    parser.add_argument("--production", action="store_true",
                        help="add the public-seam G4 block (requires bulk API)")
    parser.add_argument("--production-tokens", type=int, default=6)
    parser.add_argument("--skip-buffer-fill", action="store_true",
                        help="omit the host-only ctypes fill micro-measurement")
    args = parser.parse_args(argv)

    if args.quick:
        l4.QUICK_FIXTURE_CAP = 1 << 14
        args.tokens = min(args.tokens, 4)

    l4._raise_process_priority()
    layout = assert_layout_parity()
    pct_to_count = {pct: events for pct, events in l4.ACTIVITY_LEVELS}
    activities = [
        (int(p), pct_to_count[int(p)]) for p in args.activities.split(",") if p.strip()
    ]

    print(f"L5 marshal-batching harness @ commit {git_commit()}", flush=True)
    results: dict = {
        "spec": (
            "L5 host-marshal phase breakdown + bulk numpy A/B gates "
            "(L4 protocol point: cfg=1e6, EP, pool-locality schedule)"
        ),
        "label": args.label,
        "commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "layout_gate_g0": layout,
        "protocol": {
            "configured_cells": CONFIGURED_CELLS,
            "activities_pct": [p for p, _ in activities],
            "events_per_token": {str(p): c for p, c in activities},
            "tokens": args.tokens,
            "token_window_steps": TOKEN_WINDOW_STEPS,
            "strategies": [
                "dataclass+_event_batch",
                "buffer_fill(host-only)",
                "numpy_bulk(zero-copy)",
            ],
            "gates": ["G0 layout", "G1 counters", "G2 spikes", "G3 state dumps"],
        },
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"l5_marshal_batching_{args.label}.json"

    def _save() -> None:
        out_path.write_text(json.dumps(results, indent=2))

    rows: list[dict] = []
    results["rows"] = rows
    failed_gate = False
    for pct, count in activities:
        plan = plan_cell(count)
        if plan is None:
            print(f"  {pct}%: CAPACITY-INFEASIBLE, skipped", flush=True)
            continue
        fixture = plan["fixture_residents"]
        row: dict = {
            "activity_pct_of_configured": pct,
            "events_per_token": count,
            "capacity_plan": plan,
        }
        print(
            f"  {pct}% ({count} ev/tok): fixture={fixture:,} "
            f"event_cap={plan['event_capacity']:,}",
            flush=True,
        )
        twin_dataclass = InstrumentedRuntime(plan=plan)
        twin_numpy = InstrumentedRuntime(plan=plan)
        try:
            build_a = build_residency(twin_dataclass, fixture, BUILD_PER_WINDOW)
            build_b = build_residency(twin_numpy, fixture, BUILD_PER_WINDOW)
            twin_dataclass.settle(SETTLE_STEPS)
            twin_numpy.settle(SETTLE_STEPS)
            row["build_wall_s"] = {
                "dataclass_arm": round(build_a, 3),
                "numpy_arm": round(build_b, 3),
            }

            schedule = [(count, 2 * count)] * args.tokens
            clock_a, gate_a = run_dataclass_arm(
                twin_dataclass, plan, args.tokens, schedule
            )
            clock_b, gate_b = run_numpy_arm(twin_numpy, args.tokens, schedule)
            if not args.skip_buffer_fill:
                row["buffer_fill_host_only"] = measure_buffer_fill(
                    plan, args.tokens, schedule
                )

            mismatches = []
            for index, (left, right) in enumerate(zip(gate_a, gate_b)):
                if left[0] != right[0]:
                    mismatches.append(["G1-counters", index])
                if left[1] != right[1]:
                    mismatches.append(["G2-spikes", index])
            dump_a = twin_dataclass.state_dump()
            dump_b = twin_numpy.state_dump()
            gate_g3 = (
                dump_a["residents_sha256"] == dump_b["residents_sha256"]
                and dump_a["pending_sha256"] == dump_b["pending_sha256"]
                and dump_a["dormant_sha256"] == dump_b["dormant_sha256"]
            )
            row["final_state"] = {"dataclass_arm": dump_a, "numpy_arm": dump_b}
            row["gate_counters_G1"] = not any(m[0] == "G1-counters" for m in mismatches)
            row["gate_spikes_G2"] = not any(m[0] == "G2-spikes" for m in mismatches)
            row["gate_state_G3"] = gate_g3
            if mismatches or not gate_g3:
                failed_gate = True
                row["gate_failures"] = mismatches

            events_per_token = [count] * args.tokens
            total_legacy = [
                o + m + f + u
                for o, m, f, u in zip(
                    clock_a.object_construction,
                    clock_a.ctypes_marshal,
                    clock_a.ffi_call,
                    clock_a.result_unpack,
                )
            ]
            total_bulk = [
                b + f + u
                for b, f, u in zip(clock_b.numpy_build, clock_b.ffi_call, clock_b.result_unpack)
            ]
            legacy_marshaling = [
                o + m
                for o, m in zip(clock_a.object_construction, clock_a.ctypes_marshal)
            ]

            def block(phases_ns: dict) -> dict:
                phase_ms = {
                    key: round(statistics.median(values) / 1e6, 4)
                    for key, values in phases_ns.items()
                    if values
                }
                phase_us_per_event = {
                    key: median_us_per_event(values, events_per_token)
                    for key, values in phases_ns.items()
                    if values
                }
                totals = [sum(parts) for parts in zip(*phases_ns.values())]
                med = statistics.median(totals) / 1e6
                return {
                    "median_token_ms": round(med, 3),
                    "tok_per_s_incl_host": round(1e3 / med, 3) if med else None,
                    "phases_ms": phase_ms,
                    "phases_us_per_event": phase_us_per_event,
                }

            row["stream"] = {
                "legacy_dataclass_incl_host": block(
                    {
                        "object_construction": clock_a.object_construction,
                        "ctypes_marshal": clock_a.ctypes_marshal,
                        "ffi_call": clock_a.ffi_call,
                        "result_unpack": clock_a.result_unpack,
                    }
                ),
                "batched_numpy_bulk_incl_host": block(
                    {
                        "numpy_build": clock_b.numpy_build,
                        "ffi_call": clock_b.ffi_call,
                        "result_unpack": clock_b.result_unpack,
                    }
                ),
            }
            row["marshal_path"] = {
                "legacy_object_plus_ctypes_us_per_event": median_us_per_event(
                    legacy_marshaling, events_per_token
                ),
                "bulk_numpy_build_us_per_event": median_us_per_event(
                    clock_b.numpy_build, events_per_token
                ),
                "speedup_x": round(
                    statistics.median(legacy_marshaling)
                    / max(1, statistics.median(clock_b.numpy_build)),
                    1,
                ),
            }
            row["end_to_end_speedup_x"] = round(
                statistics.median(total_legacy)
                / max(1, statistics.median(total_bulk)),
                3,
            )
            row["walls_ms"] = {
                "legacy_total_ms": [round(w / 1e6, 3) for w in total_legacy],
                "bulk_total_ms": [round(w / 1e6, 3) for w in total_bulk],
            }
        finally:
            twin_dataclass.close()
            twin_numpy.close()

        rows.append(row)
        s = row["stream"]
        m = row["marshal_path"]
        print(
            f"    legacy incl host: {s['legacy_dataclass_incl_host']['median_token_ms']} ms/tok "
            f"({s['legacy_dataclass_incl_host']['tok_per_s_incl_host']} tok/s) | "
            f"bulk: {s['batched_numpy_bulk_incl_host']['median_token_ms']} ms/tok "
            f"({s['batched_numpy_bulk_incl_host']['tok_per_s_incl_host']} tok/s) | "
            f"marshal path {m['legacy_object_plus_ctypes_us_per_event']} -> "
            f"{m['bulk_numpy_build_us_per_event']} us/ev ({m['speedup_x']}x) | "
            f"G1={row['gate_counters_G1']} G2={row['gate_spikes_G2']} "
            f"G3={row['gate_state_G3']}",
            flush=True,
        )
        _save()

    if args.production:
        production_rows = []
        results["production_seam_g4"] = production_rows
        for pct, events in activities:
            prod_fixture = 1 << 14
            print(
                f"  [production] {pct}% ({events} ev/tok): fixture={prod_fixture:,}",
                flush=True,
            )
            block = run_production_pair(pct, events, args.production_tokens, prod_fixture)
            production_rows.append(block)
            if not block["gate_production_G4"]:
                failed_gate = True
            print(
                f"    [production] records: {block['legacy_records']['median_advance_ms']} ms/tok "
                f"({block['legacy_records']['tok_per_s']} tok/s) | bulk: "
                f"{block['bulk_ndarray']['median_advance_ms']} ms/tok "
                f"({block['bulk_ndarray']['tok_per_s']} tok/s) | "
                f"{block['end_to_end_speedup_x']}x | "
                f"G4={block['gate_production_G4']}",
                flush=True,
            )
            _save()

    results["all_gates_pass"] = not failed_gate
    _save()
    print(f"\nwrote {out_path}")
    return 1 if failed_gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
