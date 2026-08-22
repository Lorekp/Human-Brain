from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.procedural_runtime import (
    NativeDenseProceduralReference,
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralResidentState,
    ProceduralRuntimeConfig,
    ProceduralRuntimeError,
)
from humanbrain.procedural_topology import ProceduralTopology, TopologyManifest


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


@pytest.mark.parametrize("recurrent_delay_steps", (1, 2, 3))
def test_sparse_runtime_is_bit_exact_to_independent_dense_native_reference(
    recurrent_delay_steps: int,
) -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(17, 4, "uniform_no_self", 71),
        resident_capacity=17,
        event_capacity=4_096,
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


def test_native_work_and_allocation_do_not_scale_with_configured_address_count() -> None:
    observations: list[tuple[int, int, int, int]] = []
    for cell_count in (1_000_000, 100_000_000_000):
        config = ProceduralRuntimeConfig(
            manifest=TopologyManifest(cell_count, 2, "uniform_no_self", 71),
            resident_capacity=8,
            event_capacity=32,
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


def test_resident_capacity_overflow_is_deterministic_and_fail_stop() -> None:
    config = ProceduralRuntimeConfig(
        manifest=TopologyManifest(17, 2, "uniform_no_self", 71),
        resident_capacity=1,
        event_capacity=8,
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


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("resident_capacity", True, TypeError),
        ("resident_capacity", 0, ValueError),
        ("resident_capacity", 18, ValueError),
        ("event_capacity", False, TypeError),
        ("event_capacity", 0, ValueError),
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
            recurrent_weight_pa=325.0,
            recurrent_delay_steps=1,
        )
    with pytest.raises(ValueError, match="uniform_no_self"):
        ProceduralRuntimeConfig(
            manifest=TopologyManifest(17, 2, "ring_1d", 71),
            resident_capacity=1,
            event_capacity=1,
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
    protocol = json.loads(
        Path("benchmarks/procedural_runtime_gate3_slice_protocol_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert protocol["name"] == "procedural-runtime-gate3-slice-v1"
    assert protocol["baseline_commit"] == "8291fe1a0039db7141817c438854a1817a0a8300"
    assert protocol["topology_protocol"] == "random123-philox2x64-10-v2"
    assert protocol["supported_topologies"] == ["uniform_no_self"]
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
