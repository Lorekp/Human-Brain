from __future__ import annotations

import ctypes
import hashlib
import json
import math
import struct
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

import humanbrain.native as native_module
import humanbrain.procedural_runtime as procedural_runtime_module
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.procedural_runtime import (
    NativeDenseProceduralReference,
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralResidentState,
    ProceduralRuntimeConfig,
    ProceduralRuntimeError,
    ProceduralSnapshot,
)
from humanbrain.procedural_topology import ProceduralTopology, TopologyManifest


class _NativeBoundaryProxy:
    def __init__(
        self,
        real: object,
        replacements: dict[str, Callable[..., object]],
    ) -> None:
        self._real = real
        self._replacements = replacements

    def __getattr__(self, name: str) -> object:
        replacement = self._replacements.get(name)
        return replacement if replacement is not None else getattr(self._real, name)


def _thread_call(function: Callable[[], object]) -> tuple[threading.Thread, SimpleNamespace]:
    outcome = SimpleNamespace(value=None, error=None)

    def invoke() -> None:
        try:
            outcome.value = function()
        except BaseException as error:  # surfaced deterministically in the caller
            outcome.error = error

    thread = threading.Thread(target=invoke)
    thread.start()
    return thread, outcome


def _runtime_config(cell_count: int) -> ProceduralRuntimeConfig:
    return ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=cell_count,
            fanout=2,
            topology="uniform_no_self",
            seed=71,
        ),
        resident_capacity=8,
        event_capacity=32,
        spike_capacity=32,
        interval_distinct_capacity=40,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
    )


def test_empty_native_advance_keeps_100_billion_cells_implicit() -> None:
    with NativeProceduralRuntime(_runtime_config(100_000_000_000)) as runtime:
        result = runtime.advance(4)
        snapshot = runtime.snapshot()

    assert result.spikes == ()
    assert result.retained_nonzero_cells == 0
    assert result.capacity_overflowed is False
    assert snapshot.current_step == 4
    assert snapshot.residents == ()
    assert snapshot.pending_events == ()
    assert snapshot.poisoned is False

    accounting = result.accounting.to_dict()
    assert accounting["cells"] == {
        "configured": 100_000_000_000,
        "addressable": 100_000_000_000,
        "resident_peak": 0,
        "resident_distinct": 0,
        "updated_distinct": 0,
        "spiking_distinct": 0,
    }
    assert accounting["work"] == {
        "cell_step_evaluations": 0,
        "spike_events": 0,
        "analytic_catchups": 0,
        "analytic_step_equivalents": 0,
    }
    assert accounting["edges"] == {
        "configured": 200_000_000_000,
        "resident": 0,
        "mutable": 0,
        "generated": 0,
    }
    assert accounting["events"]["delivered"] == 0
    assert accounting["events"]["queue_peak"] == 0
    assert accounting["memory"]["host_bytes"] < 64 * 1024


def test_advance_and_snapshot_are_serialized_per_runtime_instance() -> None:
    runtime = NativeProceduralRuntime(_runtime_config(17))
    real_library = runtime._library
    advance_entered = threading.Event()
    release_advance = threading.Event()
    counts_entered = threading.Event()

    def blocking_advance(*args: object) -> object:
        advance_entered.set()
        assert release_advance.wait(2.0)
        return real_library.humanbrain_procedural_runtime_advance(*args)

    def observed_counts(*args: object) -> object:
        counts_entered.set()
        return real_library.humanbrain_procedural_runtime_counts(*args)

    runtime._library = _NativeBoundaryProxy(
        real_library,
        {
            "humanbrain_procedural_runtime_advance": blocking_advance,
            "humanbrain_procedural_runtime_counts": observed_counts,
        },
    )
    advance_thread, advance_outcome = _thread_call(lambda: runtime.advance(1))
    assert advance_entered.wait(2.0)
    snapshot_thread, snapshot_outcome = _thread_call(runtime.snapshot)
    try:
        assert not counts_entered.wait(0.1)
    finally:
        release_advance.set()
        advance_thread.join(2.0)
        snapshot_thread.join(2.0)
        runtime.close()

    assert not advance_thread.is_alive()
    assert not snapshot_thread.is_alive()
    assert advance_outcome.error is None
    assert snapshot_outcome.error is None
    assert snapshot_outcome.value.current_step == 1


@pytest.mark.parametrize("inflight_operation", ("advance", "snapshot"))
def test_close_waits_for_each_inflight_runtime_operation(
    inflight_operation: str,
) -> None:
    runtime = NativeProceduralRuntime(_runtime_config(17))
    real_library = runtime._library
    operation_entered = threading.Event()
    release_operation = threading.Event()
    destroy_entered = threading.Event()
    deferred_destroy: list[object] = []

    def blocking_call(function: Callable[..., object]) -> Callable[..., object]:
        def invoke(*args: object) -> object:
            operation_entered.set()
            assert release_operation.wait(2.0)
            return function(*args)

        return invoke

    def observed_destroy(handle: object) -> None:
        destroy_entered.set()
        if release_operation.is_set():
            real_library.humanbrain_procedural_runtime_destroy(handle)
        else:
            deferred_destroy.append(handle)

    replacements: dict[str, Callable[..., object]] = {
        "humanbrain_procedural_runtime_destroy": observed_destroy,
    }
    operation: Callable[[], object]
    if inflight_operation == "advance":
        replacements["humanbrain_procedural_runtime_advance"] = blocking_call(
            real_library.humanbrain_procedural_runtime_advance
        )
        operation = lambda: runtime.advance(1)
    else:
        replacements["humanbrain_procedural_runtime_counts"] = blocking_call(
            real_library.humanbrain_procedural_runtime_counts
        )
        operation = runtime.snapshot
    runtime._library = _NativeBoundaryProxy(real_library, replacements)

    operation_thread, operation_outcome = _thread_call(operation)
    assert operation_entered.wait(2.0)
    close_thread, close_outcome = _thread_call(runtime.close)
    try:
        assert not destroy_entered.wait(0.1)
    finally:
        release_operation.set()
        operation_thread.join(2.0)
        close_thread.join(2.0)
        for handle in deferred_destroy:
            real_library.humanbrain_procedural_runtime_destroy(handle)

    assert not operation_thread.is_alive()
    assert not close_thread.is_alive()
    assert operation_outcome.error is None
    assert close_outcome.error is None
    runtime.close()  # idempotent


def test_dense_advance_and_snapshot_are_serialized_per_instance() -> None:
    runtime = NativeDenseProceduralReference.materialized(_runtime_config(17))
    real_library = runtime._library
    advance_entered = threading.Event()
    release_advance = threading.Event()
    counts_entered = threading.Event()

    def blocking_advance(*args: object) -> object:
        advance_entered.set()
        assert release_advance.wait(2.0)
        return real_library.humanbrain_dense_reference_advance(*args)

    def observed_counts(*args: object) -> object:
        counts_entered.set()
        return real_library.humanbrain_dense_reference_counts(*args)

    runtime._library = _NativeBoundaryProxy(
        real_library,
        {
            "humanbrain_dense_reference_advance": blocking_advance,
            "humanbrain_dense_reference_counts": observed_counts,
        },
    )
    advance_thread, advance_outcome = _thread_call(lambda: runtime.advance(1))
    assert advance_entered.wait(2.0)
    snapshot_thread, snapshot_outcome = _thread_call(runtime.snapshot)
    try:
        assert not counts_entered.wait(0.1)
    finally:
        release_advance.set()
        advance_thread.join(2.0)
        snapshot_thread.join(2.0)
        runtime.close()

    assert not advance_thread.is_alive()
    assert not snapshot_thread.is_alive()
    assert advance_outcome.error is None
    assert snapshot_outcome.error is None
    assert snapshot_outcome.value.current_step == 1


@pytest.mark.parametrize("inflight_operation", ("advance", "snapshot"))
def test_dense_close_waits_for_each_inflight_operation_and_is_idempotent(
    inflight_operation: str,
) -> None:
    runtime = NativeDenseProceduralReference.materialized(_runtime_config(17))
    real_library = runtime._library
    operation_entered = threading.Event()
    release_operation = threading.Event()
    destroy_entered = threading.Event()
    deferred_destroy: list[object] = []

    def blocking_call(function: Callable[..., object]) -> Callable[..., object]:
        def invoke(*args: object) -> object:
            operation_entered.set()
            assert release_operation.wait(2.0)
            return function(*args)

        return invoke

    def observed_destroy(handle: object) -> None:
        destroy_entered.set()
        if release_operation.is_set():
            real_library.humanbrain_dense_reference_destroy(handle)
        else:
            deferred_destroy.append(handle)

    replacements: dict[str, Callable[..., object]] = {
        "humanbrain_dense_reference_destroy": observed_destroy,
    }
    operation: Callable[[], object]
    if inflight_operation == "advance":
        replacements["humanbrain_dense_reference_advance"] = blocking_call(
            real_library.humanbrain_dense_reference_advance
        )
        operation = lambda: runtime.advance(1)
    else:
        replacements["humanbrain_dense_reference_counts"] = blocking_call(
            real_library.humanbrain_dense_reference_counts
        )
        operation = runtime.snapshot
    runtime._library = _NativeBoundaryProxy(real_library, replacements)

    operation_thread, operation_outcome = _thread_call(operation)
    assert operation_entered.wait(2.0)
    close_thread, close_outcome = _thread_call(runtime.close)
    try:
        assert not destroy_entered.wait(0.1)
    finally:
        release_operation.set()
        operation_thread.join(2.0)
        close_thread.join(2.0)
        for handle in deferred_destroy:
            real_library.humanbrain_dense_reference_destroy(handle)

    assert not operation_thread.is_alive()
    assert not close_thread.is_alive()
    assert operation_outcome.error is None
    assert close_outcome.error is None
    runtime.close()


@pytest.mark.parametrize("undersized", ("residents", "pending", "spikes"))
def test_native_snapshot_rejects_every_undersized_destination_before_writes(
    undersized: str,
) -> None:
    runtime = NativeProceduralRuntime(_runtime_config(17))
    try:
        runtime.advance(1, (ProceduralInputEvent(0, 5, 0, 0, 150_000.0),))
        resident_count = ctypes.c_size_t()
        pending_count = ctypes.c_size_t()
        spike_count = ctypes.c_size_t()
        status = runtime._library.humanbrain_procedural_runtime_counts(
            runtime._handle,
            ctypes.byref(resident_count),
            ctypes.byref(pending_count),
            ctypes.byref(spike_count),
        )
        assert status == 0
        assert resident_count.value > 0
        assert pending_count.value > 0
        assert spike_count.value > 0

        residents = (
            procedural_runtime_module._NativeResident * resident_count.value
        )()
        pending = (procedural_runtime_module._NativeEvent * pending_count.value)()
        spikes = (procedural_runtime_module._NativeSpike * spike_count.value)()
        counters = procedural_runtime_module._NativeCounters()
        counters.current_step = (1 << 64) - 1
        before = (bytes(residents), bytes(pending), bytes(spikes), bytes(counters))
        capacities = {
            "residents": resident_count.value,
            "pending": pending_count.value,
            "spikes": spike_count.value,
        }
        capacities[undersized] -= 1

        status = runtime._library.humanbrain_procedural_runtime_snapshot(
            runtime._handle,
            residents,
            capacities["residents"],
            pending,
            capacities["pending"],
            spikes,
            capacities["spikes"],
            ctypes.byref(counters),
        )

        assert status == 7
        assert (bytes(residents), bytes(pending), bytes(spikes), bytes(counters)) == before
        assert runtime.snapshot().poisoned is False
    finally:
        runtime.close()


@pytest.mark.parametrize("undersized", ("cells", "pending", "spikes"))
def test_dense_snapshot_rejects_every_undersized_destination_before_writes(
    undersized: str,
) -> None:
    runtime = NativeDenseProceduralReference.materialized(_runtime_config(17))
    try:
        runtime.advance(1, (ProceduralInputEvent(0, 5, 0, 0, 150_000.0),))
        cell_count = ctypes.c_size_t()
        pending_count = ctypes.c_size_t()
        spike_count = ctypes.c_size_t()
        status = runtime._library.humanbrain_dense_reference_counts(
            runtime._handle,
            ctypes.byref(cell_count),
            ctypes.byref(pending_count),
            ctypes.byref(spike_count),
        )
        assert status == 0
        cells = (procedural_runtime_module._NativeResident * cell_count.value)()
        pending = (procedural_runtime_module._NativeEvent * pending_count.value)()
        spikes = (procedural_runtime_module._NativeSpike * spike_count.value)()
        current_step = ctypes.c_uint64((1 << 64) - 1)
        poison = ctypes.c_uint32(0xFFFF_FFFF)
        before = (
            bytes(cells),
            bytes(pending),
            bytes(spikes),
            bytes(current_step),
            bytes(poison),
        )
        capacities = {
            "cells": cell_count.value,
            "pending": pending_count.value,
            "spikes": spike_count.value,
        }
        capacities[undersized] -= 1

        status = runtime._library.humanbrain_dense_reference_snapshot(
            runtime._handle,
            cells,
            capacities["cells"],
            pending,
            capacities["pending"],
            spikes,
            capacities["spikes"],
            ctypes.byref(current_step),
            ctypes.byref(poison),
        )

        assert status == 7
        assert (
            bytes(cells),
            bytes(pending),
            bytes(spikes),
            bytes(current_step),
            bytes(poison),
        ) == before
        assert runtime.snapshot().poisoned is False
    finally:
        runtime.close()


def _state_bits(state: ProceduralResidentState) -> bytes:
    return struct.pack(
        "<Qdddddidd",
        state.gid,
        state.voltage_relative_mv,
        *state.after_spike_currents_pa,
        state.threshold_spike_mv,
        state.threshold_voltage_mv,
        state.refractory_steps,
        state.synaptic_current_pa,
        state.eligibility,
    )


def _event_bits(event: ProceduralInputEvent) -> bytes:
    return struct.pack(
        "<QQQQd",
        event.due_step,
        event.target_gid,
        event.source_gid,
        event.edge_slot,
        event.current_pa,
    )


def _snapshot_bits(snapshot: ProceduralSnapshot) -> tuple[object, ...]:
    return (
        snapshot.current_step,
        tuple(map(_state_bits, snapshot.residents)),
        tuple(map(_event_bits, snapshot.pending_events)),
        snapshot.spikes,
        snapshot.poisoned,
        snapshot.overflow_reason,
        snapshot.native_payload_bytes,
    )


def _native_counter_snapshot(
    runtime: NativeProceduralRuntime,
) -> procedural_runtime_module._NativeCounters:
    counters = procedural_runtime_module._NativeCounters()
    status = runtime._library.humanbrain_procedural_runtime_snapshot(
        runtime._handle,
        None,
        0,
        None,
        0,
        None,
        0,
        ctypes.byref(counters),
    )
    assert status == 0
    return counters


def _committed_counter_values(
    counters: procedural_runtime_module._NativeCounters,
) -> tuple[int, ...]:
    return tuple(
        int(getattr(counters, field))
        for field in (
            "current_step",
            "resident_peak",
            "resident_distinct",
            "updated_distinct",
            "spiking_distinct",
            "cell_step_evaluations",
            "spike_events",
            "generated_edges",
            "delivered_events",
            "configured_ids_scanned",
            "queue_peak",
        )
    )


def _is_canonical_rest(state: ProceduralResidentState) -> bool:
    values = (
        state.voltage_relative_mv,
        *state.after_spike_currents_pa,
        state.threshold_spike_mv,
        state.threshold_voltage_mv,
        state.synaptic_current_pa,
        state.eligibility,
    )
    return state.refractory_steps == 0 and all(
        value == 0.0 and math.copysign(1.0, value) == 1.0 for value in values
    )


def _assert_sparse_matches_dense(
    sparse: NativeProceduralRuntime,
    dense: NativeDenseProceduralReference,
) -> None:
    sparse_snapshot = sparse.snapshot()
    dense_snapshot = dense.snapshot()
    dense_nonzero = tuple(
        state for state in dense_snapshot.residents if not _is_canonical_rest(state)
    )

    assert sparse_snapshot.current_step == dense_snapshot.current_step
    assert tuple(map(_state_bits, sparse_snapshot.residents)) == tuple(
        map(_state_bits, dense_nonzero)
    )
    assert tuple(map(_event_bits, sparse_snapshot.pending_events)) == tuple(
        map(_event_bits, dense_snapshot.pending_events)
    )
    assert sparse_snapshot.spikes == dense_snapshot.spikes


def _glif_bits(state: ProceduralResidentState) -> bytes:
    return struct.pack(
        "<dddddi",
        state.voltage_relative_mv,
        *state.after_spike_currents_pa,
        state.threshold_spike_mv,
        state.threshold_voltage_mv,
        state.refractory_steps,
    )


@pytest.mark.parametrize("current_pa", (4_000.0, 150_000.0))
def test_dense_oracle_preserves_the_existing_native_allen_glif5_transition(
    current_pa: float,
) -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(17, 2, "uniform_no_self", 71),
        resident_capacity=17,
        event_capacity=64,
        spike_capacity=64,
        interval_distinct_capacity=81,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
    )
    population = AdaptiveGlifPopulation(17)
    current = np.zeros(17, dtype=np.float64)
    current[5] = current_pa

    with NativeDenseProceduralReference.materialized(config) as dense:
        dense.advance(1, (ProceduralInputEvent(0, 5, 0, 0, current_pa),))
        expected_spikes = population.step(current)
        actual = dense.snapshot().residents[5]

        expected = ProceduralResidentState(
            gid=5,
            voltage_relative_mv=float(population.state.voltage_relative_mv[5]),
            after_spike_currents_pa=tuple(
                float(value)
                for value in population.state.after_spike_currents_pa[5]
            ),
            threshold_spike_mv=float(population.state.threshold_spike_mv[5]),
            threshold_voltage_mv=float(population.state.threshold_voltage_mv[5]),
            refractory_steps=int(population.state.refractory_steps[5]),
            synaptic_current_pa=0.0,
            eligibility=0.0,
        )

    assert _glif_bits(actual) == _glif_bits(expected)
    assert bool(expected_spikes[5]) is (current_pa == 150_000.0)


def test_fitted_glif_constants_are_shared_by_population_and_procedural_ffi() -> None:
    population = AdaptiveGlifPopulation(1)
    constants = population.constants
    glif, after_decay, after_cut_decay, after_amplitudes = (
        procedural_runtime_module._native_parameters()
    )

    assert bytes(glif) == bytes(
        native_module._glif_config(population)
    )
    assert after_decay.tobytes() == constants.after_spike_decay.tobytes()
    assert after_cut_decay.tobytes() == constants.after_spike_cut_decay.tobytes()
    assert after_amplitudes.tobytes() == constants.after_spike_amplitudes_pa.tobytes()


@pytest.mark.parametrize("recurrent_delay_steps", (1, 2, 3))
def test_sparse_runtime_is_bit_exact_to_independent_dense_native_reference(
    recurrent_delay_steps: int,
) -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(17, 4, "uniform_no_self", 71),
        resident_capacity=17,
        event_capacity=4_096,
        spike_capacity=4_096,
        interval_distinct_capacity=4_113,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=recurrent_delay_steps,
        recurrent_tau_ms=0.05 / math.log(2.0),
        eligibility_tau_ms=40.0,
    )
    events = (
        ProceduralInputEvent(2, 9, 4, 1, -0.0),
        ProceduralInputEvent(1, 3, 2, 2, 1.0),
        ProceduralInputEvent(0, 12, 3, 3, 5e-324),
        ProceduralInputEvent(0, 4, 3, 0, 150_000.0),
        ProceduralInputEvent(1, 3, 0, 0, 1e16),
        ProceduralInputEvent(0, 5, 0, 0, 150_000.0),
        ProceduralInputEvent(2, 9, 3, 0, 0.0),
        ProceduralInputEvent(1, 3, 1, 1, -1e16),
    )
    assert struct.pack("<d", events[0].current_pa) == struct.pack("<d", 0.0)

    with (
        NativeProceduralRuntime(config) as sparse,
        NativeDenseProceduralReference.materialized(config) as dense,
    ):
        first = sparse.advance(1, tuple(reversed(events)))
        dense_first = dense.advance(1, events)

        assert first.spikes == dense_first
        assert tuple(spike.gid for spike in first.spikes) == (4, 5)
        assert first.accounting.edges.generated == 2 * config.manifest.fanout
        expected_targets = {
            edge.edge_slot: edge.target_gid
            for edge in ProceduralTopology(config.manifest).base_edges(5)
        }
        generated = {
            event.edge_slot: event.target_gid
            for event in sparse.snapshot().pending_events
            if event.source_gid == 5
            and event.due_step == recurrent_delay_steps
        }
        assert generated == expected_targets
        expected_inhibitory_targets = {
            edge.edge_slot: edge.target_gid
            for edge in ProceduralTopology(config.manifest).base_edges(4)
        }
        inhibitory = {
            event.edge_slot: event.current_pa
            for event in sparse.snapshot().pending_events
            if event.source_gid == 4
            and event.due_step == recurrent_delay_steps
            and expected_inhibitory_targets.get(event.edge_slot) == event.target_gid
        }
        assert inhibitory == {
            slot: -config.recurrent_weight_pa
            for slot in range(config.manifest.fanout)
        }
        _assert_sparse_matches_dense(sparse, dense)

        for steps in (1, 2, 7, 19):
            sparse_result = sparse.advance(steps)
            dense_spikes = dense.advance(steps)
            assert sparse_result.spikes == dense_spikes
            _assert_sparse_matches_dense(sparse, dense)
            if sparse.snapshot().current_step == 2:
                cancelled = next(
                    state for state in sparse.snapshot().residents if state.gid == 3
                )
                assert struct.pack("<d", cancelled.synaptic_current_pa) == (
                    struct.pack("<d", 0.5)
                )

        snapshot = sparse.snapshot()
        assert all(state.gid != 9 for state in snapshot.residents)
        assert all(state.gid != 12 for state in snapshot.residents)
        assert any(state.gid == 3 for state in snapshot.residents)


def test_high_uint64_spike_expands_the_counter64_v2_source_row() -> None:
    source = (1 << 32) + 1
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            100_000_000_000,
            6,
            "uniform_no_self",
            0xFFFF_FFFF_FFFF_FFFE,
        ),
        resident_capacity=8,
        event_capacity=64,
        spike_capacity=64,
        interval_distinct_capacity=72,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=3,
    )
    event = ProceduralInputEvent(0, source, 1 << 32, 5, 150_000.0)

    with NativeProceduralRuntime(config) as runtime:
        result = runtime.advance(1, (event,))
        snapshot = runtime.snapshot()

    assert len(result.spikes) == 1
    assert result.spikes[0].gid == source
    assert result.configured_ids_scanned == 0
    expected = {
        edge.edge_slot: edge.target_gid
        for edge in ProceduralTopology(config.manifest).base_edges(source)
    }
    actual = {
        pending.edge_slot: pending.target_gid
        for pending in snapshot.pending_events
        if pending.source_gid == source
    }
    assert actual == expected
    assert any(target > 0xFFFF_FFFF for target in actual.values())


def test_collision_heavy_native_row_matches_counter64_oracle_in_slot_order() -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(129, 128, "uniform_no_self", 1),
        resident_capacity=1,
        event_capacity=128,
        spike_capacity=1,
        interval_distinct_capacity=129,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
    )
    with NativeProceduralRuntime(config) as runtime:
        runtime.advance(1, (ProceduralInputEvent(0, 0, 1, 0, 150_000.0),))
        pending = runtime.snapshot().pending_events

    actual = tuple(
        event.target_gid for event in sorted(pending, key=lambda event: event.edge_slot)
    )
    expected = tuple(
        edge.target_gid for edge in ProceduralTopology(config.manifest).base_edges(0)
    )
    assert actual == expected
    assert len(set(actual)) == config.manifest.fanout


def test_native_work_and_allocation_do_not_scale_with_configured_address_count() -> None:
    observations: list[tuple[int, int, int, int]] = []
    for cell_count in (1_000_000, 100_000_000_000):
        config = ProceduralRuntimeConfig(
            manifest=TopologyManifest(cell_count, 2, "uniform_no_self", 71),
            resident_capacity=8,
            event_capacity=32,
            spike_capacity=32,
            interval_distinct_capacity=40,
            recurrent_weight_pa=325.0,
            recurrent_delay_steps=1,
        )
        event = ProceduralInputEvent(0, 5, 4, 1, 1.0)
        with NativeProceduralRuntime(config) as runtime:
            result = runtime.advance(4, (event,))
            snapshot = runtime.snapshot()
        accounting = result.accounting
        observations.append(
            (
                accounting.work.cell_step_evaluations,
                accounting.edges.generated,
                result.configured_ids_scanned,
                snapshot.native_payload_bytes,
            )
        )

    assert observations[0] == observations[1]
    assert observations[0][0] == 4
    assert observations[0][1] == 0
    assert observations[0][2] == 0
    assert observations[0][3] < 64 * 1024


def test_native_scratch_payload_is_stable_across_repeated_bounded_workloads() -> None:
    config = _runtime_config(17)
    with NativeProceduralRuntime(config) as runtime:
        runtime.advance(1, (ProceduralInputEvent(0, 5, 0, 0, 1.0),))
        first_payload = runtime.snapshot().native_payload_bytes
        runtime.advance(1, (ProceduralInputEvent(1, 6, 0, 0, 1.0),))
        second_payload = runtime.snapshot().native_payload_bytes

    assert first_payload == second_payload
    assert first_payload > 0


def test_resident_capacity_overflow_is_deterministic_and_fail_stop() -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(17, 2, "uniform_no_self", 71),
        resident_capacity=1,
        event_capacity=8,
        spike_capacity=8,
        interval_distinct_capacity=9,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
    )
    events = (
        ProceduralInputEvent(0, 2, 0, 0, 1.0),
        ProceduralInputEvent(0, 3, 1, 1, 1.0),
    )

    with NativeProceduralRuntime(config) as runtime:
        with pytest.raises(
            ProceduralRuntimeError, match="resident capacity exhausted"
        ) as first:
            runtime.advance(1, tuple(reversed(events)))
        snapshot = runtime.snapshot()
        with pytest.raises(
            ProceduralRuntimeError, match="runtime is poisoned"
        ) as second:
            runtime.advance(1)

    assert first.value.status == 2
    assert second.value.status == 5
    assert snapshot.current_step == 0
    assert snapshot.poisoned is True
    assert snapshot.overflow_reason == "resident capacity exhausted"
    assert len(snapshot.residents) == 1
    assert snapshot.residents[0].gid == 2


def test_generated_event_capacity_overflow_is_deterministic_and_fail_stop() -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(17, 2, "uniform_no_self", 71),
        resident_capacity=2,
        event_capacity=1,
        spike_capacity=1,
        interval_distinct_capacity=3,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
    )
    events = (ProceduralInputEvent(0, 2, 0, 0, 150_000.0),)

    with NativeProceduralRuntime(config) as runtime:
        with pytest.raises(
            ProceduralRuntimeError, match="event capacity exhausted"
        ) as error:
            runtime.advance(1, events)
        snapshot = runtime.snapshot()

    assert error.value.status == 3
    assert snapshot.current_step == 0
    assert len(snapshot.residents) == 1
    assert snapshot.residents[0].gid == 2
    assert snapshot.pending_events == ()
    assert snapshot.poisoned is True


def test_spike_capacity_exhaustion_is_exact_deterministic_fail_stop() -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(17, 2, "uniform_no_self", 71),
        resident_capacity=2,
        event_capacity=8,
        spike_capacity=1,
        interval_distinct_capacity=4,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
    )
    events = (
        ProceduralInputEvent(0, 2, 0, 0, 150_000.0),
        ProceduralInputEvent(0, 3, 1, 1, 150_000.0),
    )

    def fail_once() -> tuple[int, int, ProceduralSnapshot, bytes, tuple[int, ...], int]:
        with NativeProceduralRuntime(config) as runtime:
            with pytest.raises(
                ProceduralRuntimeError, match="interval spike capacity exhausted"
            ) as first:
                runtime.advance(1, tuple(reversed(events)))
            snapshot = runtime.snapshot()
            counters = _native_counter_snapshot(runtime)
            with pytest.raises(
                ProceduralRuntimeError, match="runtime is poisoned"
            ) as second:
                runtime.advance(1)
        return (
            first.value.status,
            second.value.status,
            snapshot,
            bytes(counters),
            _committed_counter_values(counters),
            int(counters.overflow_code),
        )

    first = fail_once()
    second = fail_once()

    assert first[0:2] == (9, 5)
    assert first[4] == (0,) * 11
    assert first[5] == 9
    assert first[2].current_step == 0
    assert first[2].poisoned is True
    assert first[2].overflow_reason == "interval spike capacity exhausted"
    assert _snapshot_bits(first[2]) == _snapshot_bits(second[2])
    assert first[3:] == second[3:]


def test_generated_recurrent_key_collision_is_exact_deterministic_fail_stop() -> None:
    config = _runtime_config(17)
    source_gid = 2
    edge = ProceduralTopology(config.manifest).base_edges(source_gid)[0]
    collision_key = (1, edge.target_gid, source_gid, edge.edge_slot)
    events = (
        ProceduralInputEvent(*collision_key, 1.0),
        ProceduralInputEvent(0, source_gid, 0, 1, 150_000.0),
    )

    def fail_once() -> tuple[int, int, ProceduralSnapshot, bytes, tuple[int, ...], int]:
        with NativeProceduralRuntime(config) as runtime:
            with pytest.raises(
                ProceduralRuntimeError, match="duplicate canonical event key"
            ) as first:
                runtime.advance(1, tuple(reversed(events)))
            snapshot = runtime.snapshot()
            counters = _native_counter_snapshot(runtime)
            with pytest.raises(
                ProceduralRuntimeError, match="runtime is poisoned"
            ) as second:
                runtime.advance(1)
        return (
            first.value.status,
            second.value.status,
            snapshot,
            bytes(counters),
            _committed_counter_values(counters),
            int(counters.overflow_code),
        )

    first = fail_once()
    second = fail_once()

    matching_keys = [
        (
            event.due_step,
            event.target_gid,
            event.source_gid,
            event.edge_slot,
        )
        for event in first[2].pending_events
        if (
            event.due_step,
            event.target_gid,
            event.source_gid,
            event.edge_slot,
        )
        == collision_key
    ]
    assert first[0:2] == (4, 5)
    assert first[4] == (0,) * 11
    assert first[5] == 4
    assert first[2].current_step == 0
    assert first[2].poisoned is True
    assert first[2].overflow_reason == "duplicate canonical event key"
    assert len(matching_keys) == 2
    assert _snapshot_bits(first[2]) == _snapshot_bits(second[2])
    assert first[3:] == second[3:]


def test_interval_distinct_capacity_is_deterministic_and_fail_stop() -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(17, 2, "uniform_no_self", 71),
        resident_capacity=1,
        event_capacity=2,
        spike_capacity=1,
        interval_distinct_capacity=1,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
        recurrent_tau_ms=0.05 / math.log(2.0),
    )
    events = (
        ProceduralInputEvent(0, 2, 0, 0, 5e-324),
        ProceduralInputEvent(1, 3, 0, 1, 5e-324),
    )

    with NativeProceduralRuntime(config) as runtime:
        with pytest.raises(
            ProceduralRuntimeError, match="distinct-cell capacity exhausted"
        ) as error:
            runtime.advance(2, events)
        snapshot = runtime.snapshot()

    assert error.value.status == 10
    assert snapshot.current_step == 1
    assert snapshot.poisoned is True
    assert snapshot.residents == ()


@pytest.mark.parametrize("adapter", ("sparse", "dense"))
def test_cross_call_duplicate_event_is_rejected_transactionally_without_poison(
    adapter: str,
) -> None:
    config = _runtime_config(17)
    runtime = (
        NativeProceduralRuntime(config)
        if adapter == "sparse"
        else NativeDenseProceduralReference.materialized(config)
    )
    event = ProceduralInputEvent(5, 7, 3, 1, 11.0)
    try:
        runtime.advance(1, (event,))
        before = runtime.snapshot()

        with pytest.raises(
            ProceduralRuntimeError, match="duplicate canonical event key"
        ) as error:
            runtime.advance(1, (event,))

        after = runtime.snapshot()
        assert error.value.status == 4
        assert _snapshot_bits(after) == _snapshot_bits(before)
        assert after.poisoned is False
        runtime.advance(1)
        assert runtime.snapshot().current_step == 2
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("resident_capacity", True, TypeError),
        ("resident_capacity", 0, ValueError),
        ("resident_capacity", 18, ValueError),
        ("event_capacity", False, TypeError),
        ("event_capacity", 0, ValueError),
        ("spike_capacity", True, TypeError),
        ("spike_capacity", 0, ValueError),
        ("interval_distinct_capacity", False, TypeError),
        ("interval_distinct_capacity", 7, ValueError),
        ("recurrent_delay_steps", 0, ValueError),
        ("recurrent_delay_steps", 1 << 64, ValueError),
        ("recurrent_weight_pa", float("inf"), ValueError),
        ("recurrent_tau_ms", float("nan"), ValueError),
        ("recurrent_tau_ms", 5e-324, ValueError),
        ("eligibility_tau_ms", 1e308, ValueError),
    ),
)
def test_runtime_config_rejects_every_invalid_native_bound(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    values: dict[str, object] = {
        "manifest": TopologyManifest(17, 2, "uniform_no_self", 71),
        "resident_capacity": 8,
        "event_capacity": 32,
        "spike_capacity": 32,
        "interval_distinct_capacity": 40,
        "recurrent_weight_pa": 325.0,
        "recurrent_delay_steps": 1,
    }
    values[field] = value

    with pytest.raises(error):
        ProceduralRuntimeConfig(**values)  # type: ignore[arg-type]


def test_runtime_rejects_non_uint64_manifest_and_non_uniform_slice() -> None:
    with pytest.raises(ValueError, match="fit uint64"):
        ProceduralRuntimeConfig(
            manifest=TopologyManifest(1 << 64, 1, "uniform_no_self", 71),
            resident_capacity=1,
            event_capacity=1,
            spike_capacity=1,
            interval_distinct_capacity=2,
            recurrent_weight_pa=325.0,
            recurrent_delay_steps=1,
        )
    with pytest.raises(ValueError, match="uniform_no_self"):
        ProceduralRuntimeConfig(
            manifest=TopologyManifest(17, 2, "ring_1d", 71),
            resident_capacity=1,
            event_capacity=1,
            spike_capacity=1,
            interval_distinct_capacity=2,
            recurrent_weight_pa=325.0,
            recurrent_delay_steps=1,
        )


def test_event_and_advance_validation_complete_before_ffi() -> None:
    config = _runtime_config(17)
    with pytest.raises(ValueError, match="finite"):
        ProceduralInputEvent(0, 1, 0, 0, float("nan"))

    with NativeProceduralRuntime(config) as runtime:
        invalid = ProceduralInputEvent(0, 17, 0, 0, 1.0)
        with pytest.raises(ValueError, match="target_gid"):
            runtime.advance(1, (invalid,))
        with pytest.raises(TypeError, match="step_count"):
            runtime.advance(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least one"):
            runtime.advance(0)

        over_capacity = tuple(
            ProceduralInputEvent(
                index,
                1 + index % 16,
                index % 17,
                index % 2,
                1.0,
            )
            for index in range(config.event_capacity + 1)
        )
        with pytest.raises(ValueError, match="event_capacity"):
            runtime.advance(1, over_capacity)

        assert runtime.snapshot().poisoned is False


def test_gate3_slice_protocol_keeps_material_timing_frozen() -> None:
    protocol_path = Path("benchmarks/procedural_runtime_gate3_slice_protocol_v1.json")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)

    assert protocol["name"] == "procedural-runtime-gate3-slice-v1"
    assert protocol["baseline_commit"] == "8291fe1a0039db7141817c438854a1817a0a8300"
    assert protocol["topology_protocol"] == "random123-philox2x64-10-v2"
    assert protocol["supported_topologies"] == ["uniform_no_self"]
    assert protocol["public_seam"]["raw_ffi_lifetime"] == (
        "Raw C handles require external lifetime synchronization. A NULL/0 "
        "destination pair explicitly omits that output; every supplied destination "
        "capacity is validated before any destination write."
    )
    assert protocol["determinism"]["duplicate_event_keys"] == (
        "Duplicate canonical keys in a submitted input-event batch or the already "
        "queued frontier are rejected before native simulation-state mutation; "
        "runtime remains unpoisoned."
    )
    assert protocol["resource_gate"]["sparse_memory_bound"] == (
        "O(R_capacity + Q_capacity + S_capacity + F + D_capacity), "
        "independent of N_configured"
    )
    assert {
        "returned-spike capacity exhaustion before step commit",
        "spike-generated event-key collision after mutation is fail-stop",
    } <= set(protocol["exactness_gate"]["cases"])
    assert protocol["residency"]["fail_stop_statuses"] == {
        "submitted_duplicate_input": {"status": 4, "poisons": False},
        "returned_spike_capacity": {"status": 9, "poisons": True},
        "spike_generated_event_key_collision": {"status": 4, "poisons": True},
        "poisoned_followup": {"status": 5},
    }
    assert protocol["residency"]["failed_step_commit"] == (
        "The biological step and completed-interval counters remain at the last "
        "completed boundary. Bounded in-progress state is deterministic and "
        "snapshot-visible; rollback is not claimed."
    )
    assert protocol["source_binding"]["framing"] == (
        "humanbrain-procedural-runtime-source-v1: ordered UTF-8 path length/path "
        "plus uint64 byte length/raw bytes"
    )
    assert protocol["source_binding"]["files"] == list(
        native_module.procedural_runtime_source_manifest()
    )
    bound_paths = {item["path"] for item in protocol["source_binding"]["files"]}
    assert ".gitattributes" in bound_paths
    assert {
        f"src/humanbrain/_native_src/{relative_path}"
        for relative_path, _path in native_module._native_build_input_paths()
    } <= bound_paths
    assert protocol["source_binding"]["bundle_sha256"] == (
        native_module.procedural_runtime_source_bundle_sha256()
    )
    assert protocol["source_binding"]["limits"] == {
        "maximum_file_bytes": native_module._MAX_NATIVE_SOURCE_FILE_BYTES,
        "maximum_aggregate_bytes": native_module._MAX_NATIVE_SOURCE_BUNDLE_BYTES,
        "actual_aggregate_bytes": sum(
            int(item["size"]) for item in protocol["source_binding"]["files"]
        ),
    }
    assert hashlib.sha256(protocol_bytes).hexdigest() == (
        "8f1afa73ba47b619459fd23539a599e47c28fbe0d2b9d637f054179107e9722e"
    )
    assert protocol["timing"]["status"] == "frozen-pending-explicit-approval"
    assert protocol["timing"]["executed"] is False
    assert protocol["claims"]["speed"] is None
    assert protocol["claims"]["capability"] is None
    assert protocol["forbidden_runs"] == [
        "material timing",
        "Gate-3 timing sweep",
        "GPU test",
        "DEV/SWE/canonical/private evaluation",
        "Fly Brain",
    ]


def test_native_source_digest_frames_paths_lengths_and_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "src").mkdir(parents=True)
        (root / "vendor").mkdir()
        (root / ".cargo").mkdir()
        (root / ".cargo" / "config.toml").write_bytes(
            native_module._VENDORED_CARGO_CONFIG
        )
    (first / "Cargo.toml").write_bytes(b"a")
    (first / "Cargo.lock").write_bytes(b"bc")
    (first / "src" / "x.rs").write_bytes(b"")
    (second / "Cargo.toml").write_bytes(b"ab")
    (second / "Cargo.lock").write_bytes(b"c")
    (second / "src" / "y.rs").write_bytes(b"")

    first_digest = native_module._framed_sha256(
        "native-source-boundary-test",
        native_module._native_build_input_manifest(first),
    )
    second_digest = native_module._framed_sha256(
        "native-source-boundary-test",
        native_module._native_build_input_manifest(second),
    )

    assert b"".join(
        path.read_bytes()
        for path in (
            first / ".cargo/config.toml",
            first / "Cargo.toml",
            first / "Cargo.lock",
            first / "src/x.rs",
        )
    ) == b"".join(
        path.read_bytes()
        for path in (
            second / ".cargo/config.toml",
            second / "Cargo.toml",
            second / "Cargo.lock",
            second / "src/y.rs",
        )
    )
    assert first_digest != second_digest
