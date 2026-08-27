"""Actual-native R3-F0 mechanism fixtures through the token-only Python seam."""

from __future__ import annotations

import humanbrain.native_temporal_runtime as runtime_adapter
import pytest
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
    TemporalStepMode,
)

SOURCE_TOKEN = 100
TARGET_TOKEN = 101
ACTIVE_COLUMNS = 11
COLUMN_COUNT = 22
BASE_CELLS_PER_COLUMN = 2
MAX_PYRAMIDAL_CELLS = COLUMN_COUNT * BASE_CELLS_PER_COLUMN


@pytest.fixture
def native_fixture_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypass only today's intentionally non-runnable protocol authority."""

    authority = runtime_adapter._FreshAuthority(b"\x41" * 32, b"\x92" * 32)
    monkeypatch.setattr(runtime_adapter, "_fresh_authority", lambda: authority)


def _definition(
    plasticity_policy: TemporalPlasticityPolicy,
) -> NativeTemporalCircuitDefinition:
    config = NativeTemporalCircuitConfig(
        column_count=COLUMN_COUNT,
        max_pyramidal_cells=MAX_PYRAMIDAL_CELLS,
        segment_capacity=ACTIVE_COLUMNS,
        construction_seed=17,
        connected_permanence_ticks=250,
        initial_permanence_ticks=245,
        permanence_increment_ticks=17,
        negative_decrement_ticks=10,
        base_cells_per_column=BASE_CELLS_PER_COLUMN,
        novel_race_cap=BASE_CELLS_PER_COLUMN,
        prediction_quorum=ACTIVE_COLUMNS,
        max_segments_per_cell=4,
        theta_match=18,
        theta_active=20,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=plasticity_policy,
    )
    token_codes = {
        SOURCE_TOKEN: tuple(range(ACTIVE_COLUMNS)),
        TARGET_TOKEN: tuple(range(ACTIVE_COLUMNS, COLUMN_COUNT)),
    }
    physical_cell_count = 2 * MAX_PYRAMIDAL_CELLS + 3 * COLUMN_COUNT
    return NativeTemporalCircuitDefinition(
        config,
        token_codes,
        (0.0,) * physical_cell_count,
    )


def _learn_transition(circuit: NativeTemporalCircuit):
    return circuit.advance_block(
        [SOURCE_TOKEN, TARGET_TOKEN],
        mode=TemporalStepMode.LEARN,
        reset_before=True,
    )[1]


def _read_transition(circuit: NativeTemporalCircuit):
    return circuit.advance_block(
        [SOURCE_TOKEN, TARGET_TOKEN],
        mode=TemporalStepMode.READ_ONLY,
        reset_before=True,
    )[1]


def test_below_connected_segments_cross_then_emit_physical_predicted_winners(
    native_fixture_authority: None,
) -> None:
    definition = _definition(TemporalPlasticityPolicy.PLASTIC)
    assert (
        definition.config.initial_permanence_ticks
        < definition.config.connected_permanence_ticks
        <= definition.config.initial_permanence_ticks + definition.config.permanence_increment_ticks
    )

    with NativeTemporalCircuit.fresh(definition) as circuit:
        allocated = _learn_transition(circuit)
        matching_reinforcement = _learn_transition(circuit)
        active = _read_transition(circuit)
        identity = circuit.checkpoint_identity()

    allocated_winners = tuple(spike.cell_id for spike in allocated.winner_spikes)
    matching_winners = tuple(spike.cell_id for spike in matching_reinforcement.winner_spikes)
    active_winners = tuple(spike.cell_id for spike in active.winner_spikes)

    assert allocated.allocated_segments == ACTIVE_COLUMNS
    assert allocated.predicted_columns == 0
    assert matching_reinforcement.allocated_segments == 0
    assert matching_reinforcement.predicted_columns == 0
    assert active.allocated_segments == 0
    assert active.predicted is True
    assert active.predicted_columns == ACTIVE_COLUMNS
    assert matching_winners == allocated_winners
    assert active_winners == allocated_winners
    assert all(
        active_spike.tick < matching_spike.tick < allocated_spike.tick
        for allocated_spike, matching_spike, active_spike in zip(
            allocated.winner_spikes,
            matching_reinforcement.winner_spikes,
            active.winner_spikes,
            strict=True,
        )
    )
    assert all(
        basket.tick == winner.tick + 1
        for winner, basket in zip(
            active.winner_spikes,
            active.basket_spikes,
            strict=True,
        )
    )
    assert active.inhibition_onset_ticks == tuple(
        basket.tick + 1 for basket in active.basket_spikes
    )
    assert identity.segment_count == ACTIVE_COLUMNS
    assert identity.poison_code == 0


def test_growth_off_learn_keeps_native_segment_store_structurally_empty(
    native_fixture_authority: None,
) -> None:
    with NativeTemporalCircuit.fresh(_definition(TemporalPlasticityPolicy.GROWTH_OFF)) as circuit:
        before = circuit.checkpoint_identity()
        steps = tuple(_learn_transition(circuit) for _ in range(3))
        after = circuit.checkpoint_identity()

    assert all(step.allocated_segments == 0 for step in steps)
    assert all(step.punished_segments == 0 for step in steps)
    assert all(step.predicted_columns == 0 for step in steps)
    assert before.segment_count == after.segment_count == 0
    assert before.cohort_count == after.cohort_count == 0
    assert before.next_reserve_cell == after.next_reserve_cell
    assert before.semantic_index_sha256 == after.semantic_index_sha256
    assert before.circuit_identity_sha256 == after.circuit_identity_sha256
