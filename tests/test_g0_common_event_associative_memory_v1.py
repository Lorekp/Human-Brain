from __future__ import annotations

from experiments.g0.common_event_associative_memory_v1 import (
    CHECKPOINT_ARRAY_BYTES,
    CommonAssociationLesson,
    fit_common_event_associations,
)
from experiments.g0.frozen_regulatory_adapter_v1 import (
    ACTION_CLAMP,
    ActionEvent,
    ObservationEvent,
    serialize_event,
)
from experiments.g0.spatial_candidate_adapter_v1 import SpatialStep, encode_transcript


def _regulatory_observations() -> tuple[bytes, ...]:
    states = tuple((0xB100 + index, index % 2) for index in range(8))
    return (
        serialize_event(ObservationEvent(states, time=4)),
        serialize_event(ActionEvent(ACTION_CLAMP, states[3][0], 1)),
    )


def _spatial_observations(offset: int = 0) -> tuple[bytes, ...]:
    rows: list[bytes] = []
    for step_index in range(2):
        image = bytes(
            (index * 19 + step_index * 31 + offset) % 256
            for index in range(7 * 7 * 3)
        )
        rows.append(
            encode_transcript(
                (
                    SpatialStep(
                        image=image,
                        direction=(step_index + offset) % 4,
                        action_id=(step_index + 2) % 7,
                        reward=0.0,
                        terminated=False,
                        truncated=False,
                    ),
                )
            )
        )
    return tuple(rows)


def test_common_associative_memory_learns_both_shapes_and_freezes() -> None:
    regulatory = _regulatory_observations()
    spatial = _spatial_observations()
    memory = fit_common_event_associations(
        (
            CommonAssociationLesson(regulatory, b"\x35"),
            CommonAssociationLesson(spatial, b"\xa5"),
        )
    )
    checkpoint = memory.checkpoint_sha256

    regulatory_recall = memory.recall(regulatory)
    spatial_recall = memory.recall(spatial)
    assert regulatory_recall.status == "RECALLED"
    assert regulatory_recall.output.token == 0x35
    assert spatial_recall.status == "RECALLED"
    assert spatial_recall.output.token == 0xA5
    assert len(regulatory_recall.delayed_trace_cells) == 12
    assert len(spatial_recall.delayed_trace_cells) == 12
    assert regulatory_recall.delayed_trace_cells != spatial_recall.delayed_trace_cells
    assert memory.checkpoint_sha256 == checkpoint

    summary = memory.training_summary
    assert summary.lesson_count == 2
    assert summary.work.teacher_cell_spikes == 8
    assert summary.work.excitatory_update_events == 8
    assert summary.work.inhibitory_update_events == 24
    assert summary.work.committed_excitatory_ticks == 192
    assert summary.work.committed_inhibitory_ticks == 288
    assert summary.checkpoint_array_bytes == CHECKPOINT_ARRAY_BYTES == 1_572_864

    reversed_spatial = memory.recall(tuple(reversed(spatial)))
    assert reversed_spatial.status == "STOP_UNKNOWN"
    assert reversed_spatial.delayed_trace_cells != spatial_recall.delayed_trace_cells
    assert memory.recall(_spatial_observations(offset=7)).status == "STOP_UNKNOWN"


def test_teacher_cell_and_ambiguity_controls_stop_physically() -> None:
    observations = _regulatory_observations()
    teacher_off = fit_common_event_associations(
        (CommonAssociationLesson(observations, b"\x61", teacher_gate=False),)
    )
    assert teacher_off.training_summary.work.teacher_cell_spikes == 0
    assert teacher_off.training_summary.work.committed_excitatory_ticks == 0
    assert teacher_off.recall(observations).status == "STOP_UNKNOWN"

    cell_off = fit_common_event_associations(
        (CommonAssociationLesson(observations, b"\x61", cell_gate=False),)
    )
    assert cell_off.training_summary.work.teacher_cell_spikes == 4
    assert cell_off.training_summary.work.committed_excitatory_ticks == 0
    assert cell_off.recall(observations).status == "STOP_UNKNOWN"

    ambiguous = fit_common_event_associations(
        (
            CommonAssociationLesson(observations, b"\x00"),
            CommonAssociationLesson(observations, b"\xff"),
        )
    )
    recall = ambiguous.recall(observations)
    assert recall.status == "STOP_AMBIGUOUS_COLLISION"
    assert recall.output.decision == "STOP"
    assert recall.output.stop_reason == "COLLISION"
