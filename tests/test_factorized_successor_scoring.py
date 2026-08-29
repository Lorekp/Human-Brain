from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import run_cn2_dev_4096 as cn2
from experiments.r3_pilot.factorized_successor_scoring import (
    FactorizedSuccessorScorer,
    FactorizedSuccessorStop,
)
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    TemporalStepMode,
)


def _step(
    token: int,
    winners: tuple[int, ...],
    *,
    event: int,
    allocated: int,
    matching: int = 0,
    active: int = 0,
    predicted_columns: int = 0,
) -> object:
    return SimpleNamespace(
        token_id=token,
        event_count=event,
        winner_spikes=tuple(
            SimpleNamespace(cell_id=winner) for winner in winners
        ),
        allocated_segments=allocated,
        punished_segments=0,
        predicted_columns=predicted_columns,
        active_segments=active,
        matching_segments=matching,
        candidate_segments=matching,
        postings_visited=matching,
    )


def test_one_shared_column_allocates_ten_and_replays_complete_tuple() -> None:
    source_columns = tuple(range(11))
    a_columns = tuple(range(11, 22))
    b_columns = tuple(range(21, 32))
    source = tuple(range(100, 111))
    a = tuple(range(200, 211))
    b = (a[-1], *range(300, 310))
    scorer = FactorizedSuccessorScorer()

    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=source_columns,
        reset_before=True,
    )
    scorer.observe(
        _step(1, a, event=2, allocated=11, matching=16, active=8),
        columns=a_columns,
        reset_before=False,
    )
    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=source_columns,
        reset_before=True,
    )
    scorer.observe(
        _step(
            2,
            b,
            event=2,
            allocated=10,
            matching=16,
            active=0,
            predicted_columns=1,
        ),
        columns=b_columns,
        reset_before=False,
    )
    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=source_columns,
        reset_before=True,
    )
    scorer.observe(
        _step(
            2,
            b,
            event=2,
            allocated=0,
            matching=11,
            active=11,
            predicted_columns=11,
        ),
        columns=b_columns,
        reset_before=False,
    )

    summary = scorer.finalize(terminal_segment_count=21)
    assert b != a
    assert set(a) & set(b) == {a[-1]}
    assert summary["novel_complete_keys"] == 2
    assert summary["repeated_complete_keys"] == 1
    assert summary["actual_allocated_segments"] == 21
    assert summary["partial_novel_allocation_events"] == 1
    assert summary["column_binding_count"] == 21
    telemetry = summary["matching_active_telemetry"]
    assert telemetry["matching_segments_total"] == 43
    assert telemetry["active_segments_total"] == 19
    assert telemetry["matching_minus_active_segments_total"] == 24
    assert telemetry["matching_exceeds_active_steps"] == 2
    assert telemetry["matching_with_zero_active_steps"] == 1
    assert telemetry["pair_histogram"] == {
        "0:0": 3,
        "11:11": 1,
        "16:0": 1,
        "16:8": 1,
    }


def test_native_one_shared_column_fixture_flows_through_scorer() -> None:
    source_columns = tuple(range(11))
    a_columns = tuple(range(11, 22))
    b_columns = tuple(range(21, 32))
    _protocol, production = cn2._definition()
    definition = NativeTemporalCircuitDefinition(
        replace(production.config, segment_capacity=21),
        ((0, source_columns), (1, a_columns), (2, b_columns)),
        production.threshold_offsets_mv,
    )
    scorer = FactorizedSuccessorScorer()

    with cn2._development_authority(cn2._source_binding()):
        with NativeTemporalCircuit.fresh(definition) as circuit:
            for token, columns, reset_before in (
                (0, source_columns, True),
                (1, a_columns, False),
                (0, source_columns, True),
                (2, b_columns, False),
                (0, source_columns, True),
                (2, b_columns, False),
            ):
                step = circuit.advance_block(
                    [token],
                    mode=TemporalStepMode.LEARN,
                    reset_before=reset_before,
                )[0]
                scorer.observe(step, columns=columns, reset_before=reset_before)
            terminal_segments = circuit.checkpoint_identity().segment_count

    summary = scorer.finalize(terminal_segment_count=terminal_segments)
    assert summary["actual_allocated_segments"] == 21
    assert summary["allocation_histogram"] == {"0": 4, "10": 1, "11": 1}
    assert summary["column_binding_count"] == 21
    assert summary["repeated_complete_keys"] == 1
    assert sum(summary["matching_active_telemetry"]["pair_histogram"].values()) == 6


def test_novel_zero_allocation_is_valid_when_complete_tuple_is_new() -> None:
    columns = tuple(range(11))
    source = tuple(range(100, 111))
    owners = tuple(range(200, 211))
    scorer = FactorizedSuccessorScorer()

    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=columns,
        reset_before=True,
    )
    scorer.observe(
        _step(1, owners, event=2, allocated=0, matching=11, active=11),
        columns=columns,
        reset_before=False,
    )

    summary = scorer.finalize(terminal_segment_count=0)
    assert summary["novel_complete_keys"] == 1
    assert summary["partial_novel_allocation_events"] == 1
    assert summary["allocation_histogram"] == {"0": 2}


def test_shared_column_owner_drift_stops_without_committing_failed_step() -> None:
    source_columns = tuple(range(11))
    a_columns = tuple(range(11, 22))
    b_columns = tuple(range(21, 32))
    source = tuple(range(100, 111))
    a = tuple(range(200, 211))
    drifted_b = tuple(range(300, 311))
    scorer = FactorizedSuccessorScorer()

    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=source_columns,
        reset_before=True,
    )
    scorer.observe(
        _step(1, a, event=2, allocated=11),
        columns=a_columns,
        reset_before=False,
    )
    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=source_columns,
        reset_before=True,
    )
    before = scorer.summary()

    with pytest.raises(FactorizedSuccessorStop) as caught:
        scorer.observe(
            _step(2, drifted_b, event=2, allocated=10),
            columns=b_columns,
            reset_before=False,
        )

    assert caught.value.kind == "EXACT_SOURCE_COLUMN_OWNER_DRIFT"
    assert caught.value.details["column"] == 21
    assert scorer.summary() == before


def test_physical_owner_cannot_alias_across_output_columns() -> None:
    source_columns = tuple(range(11))
    a_columns = tuple(range(11, 22))
    disjoint_b_columns = tuple(range(22, 33))
    source = tuple(range(100, 111))
    a = tuple(range(200, 211))
    aliased_b = (a[-1], *range(300, 310))
    scorer = FactorizedSuccessorScorer()

    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=source_columns,
        reset_before=True,
    )
    scorer.observe(
        _step(1, a, event=2, allocated=11),
        columns=a_columns,
        reset_before=False,
    )
    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=source_columns,
        reset_before=True,
    )
    before = scorer.summary()

    with pytest.raises(FactorizedSuccessorStop) as caught:
        scorer.observe(
            _step(2, aliased_b, event=2, allocated=11),
            columns=disjoint_b_columns,
            reset_before=False,
        )

    assert caught.value.kind == "OWNER_COLUMN_ALIAS"
    assert caught.value.details == {
        "owner": a[-1],
        "expected_column": a_columns[-1],
        "observed_column": disjoint_b_columns[0],
    }
    assert scorer.summary() == before


def test_repeated_tuple_and_terminal_allocation_drift_stop() -> None:
    columns = tuple(range(11))
    source = tuple(range(100, 111))
    owners = tuple(range(200, 211))
    scorer = FactorizedSuccessorScorer()
    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=columns,
        reset_before=True,
    )
    scorer.observe(
        _step(1, owners, event=2, allocated=11),
        columns=columns,
        reset_before=False,
    )
    scorer.observe(
        _step(0, source, event=1, allocated=0),
        columns=columns,
        reset_before=True,
    )

    with pytest.raises(FactorizedSuccessorStop) as repeated:
        scorer.observe(
            _step(1, tuple(range(300, 311)), event=2, allocated=0),
            columns=columns,
            reset_before=False,
        )
    assert repeated.value.kind == "REPEATED_COMPLETE_KEY_DRIFT"

    with pytest.raises(FactorizedSuccessorStop) as terminal:
        scorer.finalize(terminal_segment_count=10)
    assert terminal.value.kind == "TERMINAL_SEGMENT_ACCOUNTING_DRIFT"
    assert terminal.value.details == {
        "expected_sum_of_actual_allocations": 11,
        "observed_terminal_segment_count": 10,
    }
