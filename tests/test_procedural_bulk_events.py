"""Bulk NumPy event-array marshaling: layout gate, bit-exact parity, guards.

Contract under test: :func:`procedural_input_events` plus passing a
:data:`PROCEDURAL_EVENT_DTYPE` array to ``advance()`` is behaviorally
indistinguishable from building one :class:`ProceduralInputEvent` per event --
same validation outcomes (rule-for-rule), same delivered bytes into the FFI,
and therefore byte-identical counters, spikes, and resident state for the same
schedule. The zero-copy dtype is pinned against the ctypes struct at import;
these tests additionally pin observable semantics.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
import pytest

from humanbrain.procedural_runtime import (
    PROCEDURAL_EVENT_DTYPE,
    NativeDenseProceduralReference,
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralRuntimeConfig,
    _NativeEvent,
    _event_batch,
    _event_batch_array,
    procedural_input_events,
)
from humanbrain.procedural_topology import TopologyManifest

_SPIKE_DRIVE_PA = 150_000.0


def _config(cell_count: int, stepping_mode: str = "event_proportional"):
    return ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=cell_count,
            fanout=4,
            topology="uniform_no_self",
            seed=71,
        ),
        resident_capacity=cell_count,
        event_capacity=max(64, 8 * cell_count),
        spike_capacity=max(64, 8 * cell_count),
        interval_distinct_capacity=max(64, 8 * cell_count),
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
        stepping_mode=stepping_mode,
    )


def _records(due_step: int, gids, edge_slot: int = 0, current_pa: float = _SPIKE_DRIVE_PA):
    return tuple(
        ProceduralInputEvent(
            due_step=due_step,
            target_gid=int(gid),
            source_gid=int(gid),
            edge_slot=edge_slot,
            current_pa=current_pa,
        )
        for gid in gids
    )


def _array(due_step: int, gids, edge_slot: int = 0, current_pa: float = _SPIKE_DRIVE_PA):
    targets = np.asarray(list(gids), dtype=np.uint64)
    return procedural_input_events(
        due_steps=due_step,
        target_gids=targets,
        source_gids=targets,
        edge_slots=edge_slot,
        current_pa=current_pa,
    )


def test_dtype_matches_native_event_layout():
    assert PROCEDURAL_EVENT_DTYPE.itemsize == ctypes.sizeof(_NativeEvent)
    for name, _field in _NativeEvent._fields_:
        assert getattr(_NativeEvent, name).offset == PROCEDURAL_EVENT_DTYPE.fields[name][1]
    zeros = np.zeros(4, dtype=PROCEDURAL_EVENT_DTYPE)
    reference = (_NativeEvent * 4)()
    assert zeros.tobytes() == bytes(reference)


def test_builder_bytes_match_record_marshal_bytes():
    config = _config(1024)
    gids = [7, 19, 500]
    records = _records(10, gids, edge_slot=2, current_pa=123.5)
    legacy = _event_batch(config, 0, records)
    bulk = _array(10, gids, edge_slot=2, current_pa=123.5)
    owner, pointer, count = _event_batch_array(config, 0, bulk)
    assert count == len(records)
    assert owner is not None
    native_from_array = (_NativeEvent * count).from_address(bulk.ctypes.data)
    assert bytes(native_from_array) == bytes(legacy)


def test_builder_rejects_non_exact_integer_columns():
    with pytest.raises(TypeError):
        procedural_input_events(1, [1.5], [1], 0, 1.0)
    with pytest.raises(TypeError):
        procedural_input_events(True, [1], [1], 0, 1.0)
    with pytest.raises(ValueError):
        procedural_input_events(1, [-2], [1], 0, 1.0)
    with pytest.raises(ValueError):
        procedural_input_events(1, [(1 << 64)], [1], 0, 1.0)


def test_builder_canonicalizes_negative_zero_like_records():
    config = _config(1024)
    records = _records(5, [11], current_pa=-0.0)
    legacy = _event_batch(config, 0, records)
    assert legacy[0].current_pa == 0.0
    assert struct.pack("<d", legacy[0].current_pa) == struct.pack("<d", 0.0)
    bulk = procedural_input_events(5, [11], [11], 0, -0.0)
    assert bulk["current_pa"].tobytes() == struct.pack("<d", 0.0)
    owner, pointer, count = _event_batch_array(config, 0, bulk)
    assert count == 1
    delivered = (_NativeEvent * count).from_address(bulk.ctypes.data)
    assert struct.pack("<d", delivered[0].current_pa) == struct.pack("<d", 0.0)


def test_advance_parity_bit_identical_counters_spikes_state():
    cells = 512
    schedule_gids = [[(g * 13 + step) % cells for g in range(9)] for step in range(0, 40, 8)]
    legacy = NativeProceduralRuntime(_config(cells))
    bulk = NativeProceduralRuntime(_config(cells))
    try:
        step = 0
        for gids in schedule_gids:
            result_a = legacy.advance(4, _records(step, gids))
            result_b = bulk.advance(4, _array(step, gids))
            step += 4
            assert result_a.spikes == result_b.spikes
            assert result_a.retained_nonzero_cells == result_b.retained_nonzero_cells
            assert result_a.configured_ids_scanned == result_b.configured_ids_scanned
            a = result_a.accounting
            b = result_b.accounting
            assert a.cells.resident_peak == b.cells.resident_peak
            assert a.cells.updated_distinct == b.cells.updated_distinct
            assert a.cells.spiking_distinct == b.cells.spiking_distinct
            assert a.work.cell_step_evaluations == b.work.cell_step_evaluations
            assert a.work.spike_events == b.work.spike_events
            assert a.edges.generated == b.edges.generated
            assert a.events.delivered == b.events.delivered
            assert a.events.queue_peak == b.events.queue_peak
        snapshot_a = legacy.snapshot()
        snapshot_b = bulk.snapshot()
        assert snapshot_a.current_step == snapshot_b.current_step
        assert snapshot_a.residents == snapshot_b.residents
        assert snapshot_a.pending_events == snapshot_b.pending_events
        assert snapshot_a.spikes == snapshot_b.spikes
        assert legacy.dormant_cells() == bulk.dormant_cells()
    finally:
        legacy.close()
        bulk.close()


def test_empty_array_matches_no_event_call():
    runtime = NativeProceduralRuntime(_config(128))
    try:
        baseline = NativeProceduralRuntime(_config(128))
        try:
            first = runtime.advance(3, np.zeros(0, dtype=PROCEDURAL_EVENT_DTYPE))
            second = baseline.advance(3)
            assert first.spikes == () and second.spikes == ()
            assert (
                first.accounting.cells.resident_distinct
                == second.accounting.cells.resident_distinct
                == 0
            )
        finally:
            baseline.close()
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "make_events,match",
    [
        (lambda step: _records(step - 1, [1]), "due_step cannot precede"),
        (lambda step: _array(step - 1, [1]), "due_step cannot precede"),
        (lambda step: _records(step, [999_999]), "target_gid is outside"),
        (lambda step: _array(step, [999_999]), "target_gid is outside"),
        (lambda step: _records(step, [1], edge_slot=4), "edge_slot is outside"),
        (lambda step: _array(step, [1], edge_slot=4), "edge_slot is outside"),
        (
            lambda step: _records(step, [3, 3]),
            "duplicate canonical ordering key",
        ),
        (
            lambda step: _array(step, [3, 3]),
            "duplicate canonical ordering key",
        ),
    ],
)
def test_validation_rules_match_between_record_and_array_paths(make_events, match):
    runtime = NativeProceduralRuntime(_config(256))
    try:
        runtime.advance(2)
        step = runtime._current_step
        with pytest.raises(ValueError, match=match):
            runtime.advance(2, make_events(step))
    finally:
        runtime.close()


def test_capacity_overflow_message_is_shared():
    small = ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=16, fanout=4, topology="uniform_no_self", seed=71
        ),
        resident_capacity=16,
        event_capacity=4,
        spike_capacity=8,
        interval_distinct_capacity=16,
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
    )

    def expect_capacity(make_events):
        runtime = NativeProceduralRuntime(small)
        try:
            with pytest.raises(ValueError, match="exceeds event_capacity"):
                runtime.advance(1, make_events())
        finally:
            runtime.close()

    expect_capacity(lambda: _records(0, range(5)))
    expect_capacity(lambda: _array(0, range(5)))


def test_wrong_bulk_dtype_is_rejected_before_ffi():
    runtime = NativeProceduralRuntime(_config(128))
    try:
        with pytest.raises(TypeError, match="PROCEDURAL_EVENT_DTYPE"):
            runtime.advance(1, np.zeros(3, dtype=np.float64))
    finally:
        runtime.close()


def test_non_finite_bulk_currents_are_rejected():
    runtime = NativeProceduralRuntime(_config(128))
    try:
        with pytest.raises(ValueError, match="finite"):
            procedural_input_events(0, [1], [1], 0, float("inf"))
        hand_crafted = np.zeros(2, dtype=PROCEDURAL_EVENT_DTYPE)
        hand_crafted["target_gid"] = (1, 2)
        hand_crafted["source_gid"] = (1, 2)
        hand_crafted["current_pa"] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            runtime.advance(1, hand_crafted)
    finally:
        runtime.close()


def test_dense_reference_parity_with_bulk_arrays():
    config = _config(64, stepping_mode="full_touch")
    dense = NativeDenseProceduralReference.materialized(config)
    dense_bulk = NativeDenseProceduralReference.materialized(config)
    try:
        for due_step in range(1, 6):
            spikes_a = dense.advance(1, _records(due_step, [due_step * 3 % 64]))
            spikes_b = dense_bulk.advance(1, _array(due_step, [due_step * 3 % 64]))
            assert spikes_a == spikes_b
        assert dense.snapshot().residents == dense_bulk.snapshot().residents
    finally:
        dense.close()
        dense_bulk.close()
