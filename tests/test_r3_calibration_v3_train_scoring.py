from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace

from experiments.r3_pilot.calibration_v3_train_scoring import (
    R3F0TrainExpectations,
    R3F0TrainScorer,
    score_r3_f0_train_structure,
)


@dataclass(frozen=True)
class _Spike:
    cell_id: int


@dataclass(frozen=True)
class _Step:
    token_id: int
    event_count: int
    candidate_segments: int
    matching_segments: int
    active_segments: int
    allocated_segments: int
    predicted_columns: int
    predicted: bool
    punished_segments: int
    winner_spikes: tuple[_Spike, ...]
    winner_relay_spikes: tuple[_Spike, ...]
    winner_context: tuple[int, ...]
    context: tuple[int, ...]


@dataclass(frozen=True)
class _Config:
    column_count: int
    max_pyramidal_cells: int
    segment_capacity: int
    construction_seed: int = 17
    connected_permanence_ticks: int = 250
    initial_permanence_ticks: int = 255
    permanence_increment_ticks: int = 50
    negative_decrement_ticks: int = 2
    base_cells_per_column: int = 8
    novel_race_cap: int = 8
    prediction_quorum: int = 11
    max_segments_per_cell: int = 2
    theta_match: int = 10
    theta_active: int = 13
    context_policy: str = "BOUND_WINNERS"
    plasticity_policy: str = "PLASTIC"


@dataclass(frozen=True)
class _Identity:
    segment_count: int
    cohort_count: int
    next_reserve_cell: int
    event_count: int
    learning_step: int
    prior_context_length: int = 22
    winner_context_length: int = 22
    poison_code: int = 0
    body_sha256: str = "1" * 64
    circuit_identity_sha256: str = "2" * 64
    logical_state_sha256: str = "3" * 64
    protocol_identity_sha256: str = "4" * 64
    semantic_index_sha256: str = "5" * 64
    source_bundle_identity_sha256: str = "6" * 64


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _stream_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<i", token))
    return digest.hexdigest()


def _step(
    token_id: int,
    event_count: int,
    winners: tuple[int, ...],
    *,
    max_pyramidal_cells: int,
    recognition: int,
    allocated: int,
    predicted_columns: int,
) -> _Step:
    relays = tuple(max_pyramidal_cells + winner for winner in winners)
    context = winners + relays
    return _Step(
        token_id=token_id,
        event_count=event_count,
        candidate_segments=recognition,
        matching_segments=recognition,
        active_segments=recognition,
        allocated_segments=allocated,
        predicted_columns=predicted_columns,
        predicted=predicted_columns == 11,
        punished_segments=0,
        winner_spikes=tuple(_Spike(cell) for cell in winners),
        winner_relay_spikes=tuple(_Spike(cell) for cell in relays),
        winner_context=context,
        context=context,
    )


def _passing_case() -> tuple[
    list[tuple[bool, _Step]],
    _Identity,
    _Config,
    dict[int, tuple[int, ...]],
    R3F0TrainExpectations,
]:
    column_count = 22
    max_pyramidal_cells = 512
    base_cells = 8
    token_columns = {0: tuple(range(11, 22)), 1: tuple(range(11))}
    reset_winners = tuple(column * base_cells for column in token_columns[0])
    records: list[tuple[bool, _Step]] = [
        (
            True,
            _step(
                0,
                1,
                reset_winners,
                max_pyramidal_cells=max_pyramidal_cells,
                recognition=0,
                allocated=0,
                predicted_columns=0,
            ),
        )
    ]
    novel_rows: list[tuple[int, ...]] = []
    for row_index in range(16):
        slope = 1 if row_index < 8 else 2
        intercept = row_index % 8
        winners = tuple(
            column * base_cells + ((slope * position + intercept) % 8)
            for position, column in enumerate(token_columns[1])
        )
        novel_rows.append(winners)
        records.append(
            (
                False,
                _step(
                    1,
                    row_index + 2,
                    winners,
                    max_pyramidal_cells=max_pyramidal_cells,
                    recognition=0,
                    allocated=11,
                    predicted_columns=0,
                ),
            )
        )
    base_population = column_count * base_cells
    reserve_winners = tuple(base_population + position * 8 for position in range(11))
    novel_rows.append(reserve_winners)
    records.append(
        (
            False,
            _step(
                1,
                18,
                reserve_winners,
                max_pyramidal_cells=max_pyramidal_cells,
                recognition=0,
                allocated=11,
                predicted_columns=0,
            ),
        )
    )
    records.extend(
        (
            (
                True,
                _step(
                    0,
                    1,
                    reset_winners,
                    max_pyramidal_cells=max_pyramidal_cells,
                    recognition=0,
                    allocated=0,
                    predicted_columns=0,
                ),
            ),
            (
                False,
                _step(
                    1,
                    2,
                    novel_rows[0],
                    max_pyramidal_cells=max_pyramidal_cells,
                    recognition=11,
                    allocated=0,
                    predicted_columns=11,
                ),
            ),
        )
    )
    tokens = [step.token_id for _, step in records]
    codebook_preimage = [[token, list(token_columns[token])] for token in sorted(token_columns)]
    expectations = R3F0TrainExpectations(
        column_count=column_count,
        max_pyramidal_cells=max_pyramidal_cells,
        base_cells_per_column=base_cells,
        cohort_width=8,
        max_segments_per_cell=2,
        expected_book_resets=2,
        expected_tokens=20,
        expected_transitions=18,
        expected_distinct_keys=17,
        expected_segments=187,
        expected_recruited_columns=11,
        expected_next_reserve_cell=base_population + 88,
        expected_max_depth=16,
        expected_final_event_count=2,
        train_stream_sha256=_stream_sha256(tokens),
        token_codebook_sha256=_canonical_sha256(codebook_preimage),
    )
    config = _Config(
        column_count=column_count,
        max_pyramidal_cells=max_pyramidal_cells,
        segment_capacity=expectations.expected_segments,
    )
    identity = _Identity(
        segment_count=expectations.expected_segments,
        cohort_count=11,
        next_reserve_cell=expectations.expected_next_reserve_cell,
        event_count=2,
        learning_step=20,
    )
    return records, identity, config, token_columns, expectations


def test_pure_train_scorer_accepts_exact_replay_and_full_cohort_growth() -> None:
    records, identity, config, token_columns, expectations = _passing_case()

    result = score_r3_f0_train_structure(
        records,
        identity=identity,
        config=config,
        token_columns=token_columns,
        expectations=expectations,
    )

    assert result["status"] == "PASS_R3_F0_TRAIN_STRUCTURE"
    assert result["distinct_temporal_keys"] == 17
    assert result["repeat_transitions"] == 1
    assert result["overlap_novel_rows"] == 17
    assert result["total_cohorts"] == 11
    assert result["recruited_columns"] == 11
    assert result["committed_pyramidal_cells"] == 264
    assert result["maximum_column_depth"] == 16
    assert result["maximum_owner_load"] == 2


def test_pure_train_scorer_stops_at_first_fanout_count_drift() -> None:
    records, identity, config, token_columns, expectations = _passing_case()
    reset, final_repeat = records[-1]
    records[-1] = (reset, replace(final_repeat, candidate_segments=0))

    scorer = R3F0TrainScorer(
        config=config,
        token_columns=token_columns,
        expectations=expectations,
    )
    result = None
    for record in records:
        result = scorer.observe_record(record)
        if result is not None:
            break
    assert result is not None
    assert scorer.stopped is True
    assert scorer.observe_record(records[-1]) is result
    assert scorer.finalize(identity) is result

    assert result == {
        "status": "STOP",
        "stop_kind": "SOURCE_FANOUT_COUNTS",
        "call_index": 19,
        "book_ordinal": 1,
        "token_index": 1,
        "expected": [11, 11, 11],
        "observed": [0, 11, 11],
        "source_fanout": 1,
    }


def test_public_streaming_scorer_accepts_flat_identity_snapshots() -> None:
    records, identity, config, token_columns, expectations = _passing_case()
    expected = score_r3_f0_train_structure(
        records,
        identity=identity,
        config=config,
        token_columns=token_columns,
        expectations=expectations,
    )
    mapping = {
        label: getattr(identity, label)
        for label in identity.__dataclass_fields__
    }

    for snapshot in (mapping, tuple(mapping.items())):
        scorer = R3F0TrainScorer(
            config=config,
            token_columns=token_columns,
            expectations=expectations,
        )
        for reset_before, step in records:
            assert scorer.observe(reset_before, step) is None
        observed = scorer.finalize(snapshot)
        assert observed == expected
        assert scorer.finalize(snapshot) is observed


def test_overlap_kernel_receives_novel_key_owner_rows_only() -> None:
    column_count = 22
    max_pyramidal_cells = 256
    token_columns = {0: tuple(range(11, 22)), 1: tuple(range(11))}
    reset_winners = tuple(column * 8 for column in token_columns[0])
    first = tuple(column * 8 for column in token_columns[1])
    second = first[:5] + tuple(column * 8 + 1 for column in token_columns[1][5:])
    records = [
        (
            True,
            _step(
                0,
                1,
                reset_winners,
                max_pyramidal_cells=max_pyramidal_cells,
                recognition=0,
                allocated=0,
                predicted_columns=0,
            ),
        ),
        (
            False,
            _step(
                1,
                2,
                first,
                max_pyramidal_cells=max_pyramidal_cells,
                recognition=0,
                allocated=11,
                predicted_columns=0,
            ),
        ),
        (
            False,
            _step(
                1,
                3,
                second,
                max_pyramidal_cells=max_pyramidal_cells,
                recognition=0,
                allocated=11,
                predicted_columns=0,
            ),
        ),
    ]
    tokens = [0, 1, 1]
    preimage = [[token, list(token_columns[token])] for token in sorted(token_columns)]
    expectations = R3F0TrainExpectations(
        column_count=column_count,
        max_pyramidal_cells=max_pyramidal_cells,
        max_segments_per_cell=255,
        expected_book_resets=1,
        expected_tokens=3,
        expected_transitions=2,
        expected_distinct_keys=2,
        expected_segments=22,
        expected_recruited_columns=0,
        expected_next_reserve_cell=176,
        expected_max_depth=8,
        expected_final_event_count=3,
        train_stream_sha256=_stream_sha256(tokens),
        token_codebook_sha256=_canonical_sha256(preimage),
    )
    config = replace(
        _Config(column_count, max_pyramidal_cells, 22),
        max_segments_per_cell=255,
    )
    identity = _Identity(
        segment_count=22,
        cohort_count=0,
        next_reserve_cell=176,
        event_count=3,
        learning_step=3,
    )

    result = score_r3_f0_train_structure(
        records,
        identity=identity,
        config=config,
        token_columns=token_columns,
        expectations=expectations,
    )

    assert result["status"] == "STOP"
    assert result["stop_kind"] == "SAME_TOKEN_WINNER_OVERLAP"
    assert result["left_index"] == 0
    assert result["right_index"] == 1
    assert result["observed"] == 5
