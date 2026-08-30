from __future__ import annotations

from collections import Counter

from experiments.g0.common_event_cells_v1 import (
    ASSEMBLY_ACTIVE_CELLS,
    GEOMETRY_RECORD,
    INPUT_CELL_COUNT,
    OUTPUT_CELL_COUNT,
    OUTPUT_CELL_OFFSET,
    STOP_CELL_ID,
    CommonEventCells,
)
from experiments.g0.frozen_regulatory_adapter_v1 import (
    ObservationEvent,
    serialize_event,
)
from experiments.g0.spatial_candidate_adapter_v1 import SpatialStep, encode_transcript


def _regulatory_fixture() -> bytes:
    states = tuple((0xA100 + index, index % 2) for index in range(8))
    return serialize_event(ObservationEvent(states, time=3))


def _spatial_fixture() -> bytes:
    image = bytes((index * 17 + 3) % 256 for index in range(7 * 7 * 3))
    return encode_transcript(
        (
            SpatialStep(
                image=image,
                direction=1,
                action_id=2,
                reward=0.0,
                terminated=False,
                truncated=False,
            ),
        )
    )


def test_common_projection_accepts_both_opaque_shapes_and_preserves_order_causally() -> None:
    cells = CommonEventCells()
    regulatory = cells.project(_regulatory_fixture())
    spatial_payload = _spatial_fixture()
    spatial = cells.project(spatial_payload)

    assert regulatory.byte_count == len(_regulatory_fixture())
    assert spatial.byte_count == len(spatial_payload)
    assert len(regulatory.byte_emissions) == regulatory.byte_count
    assert len(spatial.byte_emissions) == spatial.byte_count
    assert GEOMETRY_RECORD["domain_decoder_count"] == 0
    assert INPUT_CELL_COUNT == 65_536

    for frame in (regulatory, spatial):
        assert len(frame.start_cells) == ASSEMBLY_ACTIVE_CELLS
        assert len(frame.end_cells) == ASSEMBLY_ACTIVE_CELLS
        assert len(frame.trace_cells) == ASSEMBLY_ACTIVE_CELLS
        for emission in frame.byte_emissions:
            assert len(emission.active_cells) == 3 * ASSEMBLY_ACTIVE_CELLS
            assert len(set(emission.active_cells)) == len(emission.active_cells)
            assert all(0 <= cell < INPUT_CELL_COUNT for cell in emission.active_cells)

    swapped = bytearray(spatial_payload)
    left, right = next(
        (index, index + 1)
        for index in range(1, len(swapped) - 1)
        if swapped[index] != swapped[index + 1]
    )
    swapped[left], swapped[right] = swapped[right], swapped[left]
    deranged = cells.project(bytes(swapped))

    assert Counter(row.value_cells for row in spatial.byte_emissions) == Counter(
        row.value_cells for row in deranged.byte_emissions
    )
    assert tuple(row.position_cells for row in spatial.byte_emissions) == tuple(
        row.position_cells for row in deranged.byte_emissions
    )
    assert tuple(row.binding_cells for row in spatial.byte_emissions) != tuple(
        row.binding_cells for row in deranged.byte_emissions
    )
    assert spatial.trace_cells != deranged.trace_cells


def test_shared_factorized_output_emits_four_cells_or_physical_stop() -> None:
    cells = CommonEventCells()
    stop = cells.race_output(((0, 0, 0, 0),) * 4)
    assert stop.decision == "STOP"
    assert stop.active_cells == (STOP_CELL_ID,)
    assert stop.stop_reason == "TONIC_STOP"

    digits = (2, 2, 1, 1)  # base-4 encoding of opaque byte token 165
    scores = tuple(
        tuple(13 if candidate == digit else 0 for candidate in range(4))
        for digit in digits
    )
    selected = cells.race_output(scores)
    assert selected.decision == "TOKEN"
    assert selected.token == 165
    assert len(selected.active_cells) == 4
    assert len(set(selected.active_cells)) == 4
    assert all(
        OUTPUT_CELL_OFFSET <= cell < OUTPUT_CELL_OFFSET + OUTPUT_CELL_COUNT
        for cell in selected.active_cells
    )
    assert OUTPUT_CELL_COUNT == 17 < 256
