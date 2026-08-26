"""Targeted read seam tests: ``NativeProceduralRuntime.read_cells`` (Seam A).

Contract under test (docs/research/cpu-seams-spec.md, Seam A): request-order
targeted reads of *active* residency with explicit absence, byte-identical to
what ``snapshot()`` reports for the same cells at the same instant, refusing
to serve a poisoned runtime.

Dormancy note: under the shipped production GLIF constants a spiked cell keeps
sub-normal voltage/threshold tails for ~10^6 steps, so natural dormancy
horizons dominate any CI-scale horizon (see
tests/test_procedural_runtime_ep.py and
experiments/procedural_adapter_spike/l3_event_proportional_throughput.py,
which isolates the mechanism with a stress configuration). The dormant-
semantics tests below follow that precedent: they swap the fitted-cell
constants for the exact-arithmetic stress fixture used by the kernel-level
parity tests (post-spike cells go exactly quiet), while exercising the real
public binding, locks, and FFI marshalling.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import humanbrain.procedural_runtime as procedural_runtime_module
from humanbrain.native import _GlifConfig
from humanbrain.procedural_runtime import (
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralRuntimeConfig,
    ProceduralRuntimeError,
)
from humanbrain.procedural_topology import TopologyManifest


def test_native_resident_layout_declares_the_complete_ffi_image() -> None:
    resident = procedural_runtime_module._NativeResident

    assert ctypes.sizeof(resident) == 72
    assert resident.refractory_steps.offset == 48
    assert resident.padding_zero.offset == 52
    assert resident.synaptic_current_pa.offset == 56
    assert resident.eligibility.offset == 64
    assert bytes(resident())[52:56] == b"\x00" * 4


def _config(
    cell_count: int,
    stepping_mode: str,
    *,
    resident_capacity: int | None = None,
    recurrent_weight_pa: float = 325.0,
    recurrent_tau_ms: float = 8.0,
) -> ProceduralRuntimeConfig:
    return ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=cell_count,
            fanout=4,
            topology="uniform_no_self",
            seed=71,
        ),
        resident_capacity=resident_capacity or cell_count,
        event_capacity=max(8 * cell_count, 512),
        spike_capacity=max(8 * cell_count, 512),
        interval_distinct_capacity=max(8 * cell_count, 512),
        recurrent_weight_pa=recurrent_weight_pa,
        recurrent_delay_steps=1,
        recurrent_tau_ms=recurrent_tau_ms,
        stepping_mode=stepping_mode,
    )


def _stress_native_parameters() -> tuple:
    """Kernel-fixture-equivalent GLIF: post-spike cells go exactly quiet, so
    fast-rest retirement (dormancy) is reachable inside a test horizon."""

    glif = _GlifConfig(
        1,  # cell_count placeholder (the runtime overrides it)
        2,  # after_channel_count
        3,  # spike_cut_steps
        1.0,  # leak_ns
        1.0,  # voltage_step_scale
        0.0,  # threshold_beta_scale
        0.9,  # threshold_exp_decay
        0.0,  # threshold_voltage_factor
        0.0,  # threshold_beta_factor
        0.9,  # spike_threshold_decay
        1.0,  # threshold_inf_relative_mv
        0.0,  # voltage_reset_scale
        0.0,  # voltage_reset_offset_mv
        0.5,  # spike_cut_threshold_decay
        0.0,  # threshold_spike_increment_mv
    )
    return (
        glif,
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
    )


@pytest.fixture()
def stress_native_parameters(monkeypatch):
    monkeypatch.setattr(
        procedural_runtime_module,
        "_native_parameters",
        _stress_native_parameters,
    )


def _drive(target: int, due_step: int, current_pa: float = 150_000.0):
    return ProceduralInputEvent(
        due_step=due_step,
        target_gid=target,
        source_gid=target,
        edge_slot=0,
        current_pa=current_pa,
    )


def test_native_resident_raw_records_are_repeatable_and_zero_padded(
    stress_native_parameters,
) -> None:
    def read_image() -> tuple[bytes, bytes]:
        runtime = NativeProceduralRuntime(_config(32, "full_touch"))
        try:
            runtime.advance(1, [_drive(3, 0, 0.5)])
            requested = (ctypes.c_uint64 * 3)(3, 17, 3)
            resident = procedural_runtime_module._NativeResident
            records = (resident * 3)()
            found = (ctypes.c_uint8 * 3)()
            ctypes.memset(ctypes.addressof(records), 0xA5, ctypes.sizeof(records))
            ctypes.memset(ctypes.addressof(found), 0xA5, ctypes.sizeof(found))

            status = runtime._library.humanbrain_procedural_runtime_read_cells(
                runtime._handle,
                requested,
                len(requested),
                records,
                found,
            )

            assert status == 0
            assert tuple(found) == (1, 0, 1)
            assert tuple(record.padding_zero for record in records) == (0, 0, 0)
            image = bytes(records)
            for record_index in range(len(records)):
                padding_start = record_index * 72 + 52
                assert image[padding_start : padding_start + 4] == b"\x00" * 4
            return image, bytes(found)
        finally:
            runtime.close()

    assert read_image() == read_image()


def _drive_to_dormancy(
    runtime: NativeProceduralRuntime, target: int
) -> tuple[int, float, int]:
    """Advance until `target` retires to the dormant store; return its entry."""

    runtime.advance(1, [_drive(target, runtime.current_step)])
    for _ in range(20):
        runtime.advance(10)
        dormant = runtime.dormant_cells()
        if dormant:
            entries = dict((entry[0], entry) for entry in dormant)
            assert target in entries, f"wrong cell dormanized: {dormant}"
            return entries[target]
    raise AssertionError("stress fixture must reach dormancy inside 200 steps")


def _raw_advance_one_step(
    runtime: NativeProceduralRuntime, events: list[ProceduralInputEvent]
) -> None:
    """One step through the raw native advance seam.

    Used once a dormant population exists because -- pre-existing behavior,
    outside Seam A's performance-only scope -- ``advance()``'s
    ScaleAccounting validation rejects such runtimes: native
    ``resident_peak`` counts dormant capacity slots while ledger
    ``resident_distinct`` counts only actively stepped cells (and a dormant
    reactivation additionally never marks ``SEEN_RESIDENT``), so the host-side
    ``resident_peak <= resident_distinct`` / ``updated <= resident``
    relations trip. The L3 spike script drives the same raw seam for the
    same reason.
    """
    from humanbrain.procedural_runtime import _NativeEvent

    native = (_NativeEvent * len(events))(
        *[
            _NativeEvent(
                event.due_step,
                event.target_gid,
                event.source_gid,
                event.edge_slot,
                event.current_pa,
            )
            for event in events
        ]
    )
    status = int(
        runtime._library.humanbrain_procedural_runtime_advance(
            runtime._handle,
            1,
            native,
            len(native),
        )
    )
    assert status == 0
    runtime._current_step += 1


def test_read_cells_empty_sequence_skips_the_ffi_call(stress_native_parameters):
    runtime = NativeProceduralRuntime(_config(32, "event_proportional"))
    try:
        assert runtime.read_cells(()) == ()
        assert runtime.read_cells([]) == ()
    finally:
        runtime.close()


def test_read_cells_order_duplicates_partial_found_and_bit_parity(
    stress_native_parameters,
):
    cells = 64
    runtime = NativeProceduralRuntime(
        _config(
            cells,
            "event_proportional",
            recurrent_weight_pa=0.01,  # fanout edges stay far sub-threshold
            recurrent_tau_ms=0.005,  # synaptic tails flush to exact zero fast
        )
    )
    try:
        # gid 1: actively resident (subthreshold perturbation, no spike).
        # gid 3: spiked, then retired to the dormant store.
        # gid 17: configured but never touched -> absent.
        runtime.advance(1, [_drive(1, 0, 0.5)])
        _drive_to_dormancy(runtime, 3)

        # Out-of-range requests are a caller-side courtesy: rejection happens
        # before any FFI call (natively an out-of-range gid is simply absent).
        with pytest.raises(ValueError):
            runtime.read_cells([1, 17, 1, 3, cells])

        # Keep gid 1 actively resident at read time: its subthreshold tail
        # would otherwise flush to canonical zero and evict during the
        # dormancy march above. Raw seam: public advance() rejects runtimes
        # holding a dormant population (see _raw_advance_one_step).
        _raw_advance_one_step(runtime, [_drive(1, runtime.current_step, 0.5)])

        results = runtime.read_cells([1, 17, 1, 3])
        assert isinstance(results, tuple)
        assert len(results) == 4
        present_first, absent, present_duplicate, dormant = results
        assert present_first is not None
        assert present_duplicate is not None
        assert absent is None
        assert dormant is None
        assert present_first.gid == 1
        assert present_duplicate.gid == 1

        # Duplicates get independent, bit-identical answers.
        assert present_first == present_duplicate

        # Bit parity against snapshot() on every overlapping field.
        snapshot_states = {
            state.gid: state for state in runtime.snapshot().residents
        }
        assert 1 in snapshot_states
        expected = snapshot_states[1]
        assert present_first.voltage_relative_mv.hex() == (
            expected.voltage_relative_mv.hex()
        )
        for left, right in zip(
            present_first.after_spike_currents_pa,
            expected.after_spike_currents_pa,
        ):
            assert left.hex() == right.hex()
        assert present_first.threshold_spike_mv.hex() == (
            expected.threshold_spike_mv.hex()
        )
        assert present_first.threshold_voltage_mv.hex() == (
            expected.threshold_voltage_mv.hex()
        )
        assert present_first.refractory_steps == expected.refractory_steps
        assert present_first.synaptic_current_pa.hex() == (
            expected.synaptic_current_pa.hex()
        )
        assert present_first.eligibility.hex() == expected.eligibility.hex()

        # Canonical positive zeros on absence: -0.0 would hex differently.
        assert runtime.read_cells([17]) == (None,)
    finally:
        runtime.close()


def test_read_cells_reports_dormant_cells_as_absent_until_reactivation(
    stress_native_parameters,
):
    runtime = NativeProceduralRuntime(
        _config(
            64,
            "event_proportional",
            recurrent_weight_pa=0.01,
            recurrent_tau_ms=0.005,
        )
    )
    try:
        _drive_to_dormancy(runtime, 3)
        assert runtime.dormant_cells() != ()
        assert runtime.read_cells([3]) == (None,)

        # Reactivate the dormant cell through the raw advance seam (see
        # _raw_advance_one_step for why the public advance() cannot observe
        # this transition). The native semantics under test are unaffected.
        _raw_advance_one_step(runtime, [_drive(3, runtime.current_step)])

        assert runtime.dormant_cells() == ()
        reactivated = runtime.read_cells([3])
        assert reactivated is not None and len(reactivated) == 1
        state = reactivated[0]
        assert state is not None
        assert state.gid == 3
        assert state.eligibility > 0.0
        assert state.synaptic_current_pa != 0.0

        # The reactivated state must agree with snapshot() bit for bit.
        snapshot_state = {
            resident.gid: resident
            for resident in runtime.snapshot().residents
        }[3]
        assert state.eligibility.hex() == snapshot_state.eligibility.hex()
        assert (
            state.synaptic_current_pa.hex()
            == snapshot_state.synaptic_current_pa.hex()
        )
    finally:
        runtime.close()


def test_read_cells_is_deterministic_across_identically_seeded_runtimes():
    def run() -> tuple:
        runtime = NativeProceduralRuntime(_config(64, "full_touch"))
        try:
            trace = []
            for step in range(60):
                batch = (
                    [_drive(gid, step, 0.75) for gid in (3, 17, 41)]
                    if step % 11 == 3
                    else []
                )
                result = runtime.advance(1, batch)
                trace.extend((spike.step, spike.gid) for spike in result.spikes)
            requests = [3, 55, 41, 41, 17, 0, 63]
            return trace, runtime.read_cells(requests)
        finally:
            runtime.close()

    first_trace, first_reads = run()
    second_trace, second_reads = run()
    assert first_trace == second_trace
    assert len(first_reads) == 7
    assert first_reads == second_reads
    # The mixed request must contain both hits and absences.
    assert any(state is not None for state in first_reads)
    assert any(state is None for state in first_reads)


def test_read_cells_fails_closed_with_poisoned_after_capacity_fail_stop():
    cells = 16
    runtime = NativeProceduralRuntime(
        _config(cells, "full_touch", resident_capacity=4)
    )
    try:
        with pytest.raises(ProceduralRuntimeError) as overflow:
            runtime.advance(1, [_drive(gid, 0) for gid in range(6)])
        assert overflow.value.status == 2  # PROCEDURAL_RESIDENT_CAPACITY

        with pytest.raises(ProceduralRuntimeError) as poisoned:
            runtime.read_cells([0, 1, 2])
        assert poisoned.value.status == 5  # PROCEDURAL_POISONED
    finally:
        runtime.close()


def test_read_cells_on_closed_runtime_raises_the_standard_error(
    stress_native_parameters,
):
    runtime = NativeProceduralRuntime(_config(32, "full_touch"))
    runtime.close()
    with pytest.raises(RuntimeError, match="procedural runtime is closed"):
        runtime.read_cells([1])
