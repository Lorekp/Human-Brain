from __future__ import annotations

import dataclasses
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.g0 import common_event_cells_v1 as v1
from experiments.g0 import common_event_cells_v2 as v2


def test_v2_projects_shared_physical_fillers_without_whole_byte_aliases() -> None:
    cells = v2.CommonOneByteFillerCells()
    frames = tuple(cells.project_one_byte(bytes((value,))) for value in range(256))

    assert v2.FILLER_CELL_COUNT == 16
    assert v2.INPUT_CELL_COUNT == v1.INPUT_CELL_COUNT + 16
    assert v2.OUTPUT_CELL_COUNT == v1.OUTPUT_CELL_COUNT == 17
    assert v2.TOTAL_CELL_COUNT == v1.TOTAL_CELL_COUNT + 16 == 65_569
    assert (
        sha256(Path(v1.__file__).read_bytes()).hexdigest()
        == v2.EXPECTED_V1_SOURCE_SHA256
    )
    assert v2.EXPECTED_V1_GEOMETRY_SHA256 == v1.GEOMETRY_SHA256
    assert v2.GEOMETRY_RECORD["v1_source_sha256"] == v2.EXPECTED_V1_SOURCE_SHA256
    assert v2.GEOMETRY_RECORD["v1_geometry_sha256"] == v1.GEOMETRY_SHA256
    assert v2.CONFIG_RECORD["geometry_sha256"] == v2.GEOMETRY_SHA256
    assert v2.CONFIG_RECORD["geometry_audit_sha256"] == v2.GEOMETRY_AUDIT.sha256
    assert sha256(Path(v2.__file__).read_bytes()).hexdigest() == v2.SOURCE_SHA256
    assert len(v2.SUBSTRATE_ROOT_SHA256) == 64
    assert all(len(frame.filler_cells) == 4 for frame in frames)
    assert all(
        all(
            v2.FILLER_CELL_OFFSET + lane * v2.FILLER_RADIX
            <= cell_id
            < v2.FILLER_CELL_OFFSET + (lane + 1) * v2.FILLER_RADIX
            for lane, cell_id in enumerate(frame.filler_cells)
        )
        for frame in frames
    )
    activations = Counter(
        cell_id for frame in frames for cell_id in frame.filler_cells
    )
    assert len(activations) == 16
    assert set(activations.values()) == {64}
    assert len({frame.filler_cells for frame in frames}) == 256
    assert len({frame.opaque_frame.sha256 for frame in frames}) == 256

    cover = tuple(frames[value] for value in v2.COVER_VALUES)
    heldout = frames[v2.HELDOUT_COMBINATION_VALUE]
    cover_fillers = {cell_id for frame in cover for cell_id in frame.filler_cells}
    assert set(heldout.filler_cells) <= cover_fillers
    assert heldout.filler_cells not in {frame.filler_cells for frame in cover}
    assert heldout.opaque_frame.sha256 not in {
        frame.opaque_frame.sha256 for frame in cover
    }
    assert cells.recurrent_cells(heldout) == heldout.recurrent_input_cells
    assert heldout.record()["raw_byte_retained"] is False
    assert heldout.record()["host_digit_tuple_retained"] is False

    permuted = frames[v2.LANE_PERMUTATION_CONTROL_VALUE]
    assert set(permuted.filler_cells) != set(heldout.filler_cells)
    forged_permutation = dataclasses.replace(
        heldout,
        filler_cells=permuted.filler_cells,
    )
    forged_alias = dataclasses.replace(
        heldout,
        opaque_frame=cover[0].opaque_frame,
    )
    with pytest.raises(ValueError, match="authentication"):
        cells.recurrent_cells(forged_permutation)
    with pytest.raises(ValueError, match="authentication"):
        cells.recurrent_cells(forged_alias)

    scores = (
        (24, 0, 0, 0),
        (0, 24, 0, 0),
        (0, 0, 24, 0),
        (0, 0, 0, 24),
    )
    assert cells.race_output(scores) == v1.CommonEventCells().race_output(scores)
    assert v2.GEOMETRY_AUDIT.unique_filler_tuple_count == 256
    assert v2.GEOMETRY_AUDIT.whole_byte_filler_alias_pair_count == 0
    assert v2.GEOMETRY_AUDIT.heldout_new_filler_cell_count == 0
    assert v2.LEARNER_EXECUTION_COUNT == 0
    assert (
        v2.CORPUS_INPUT_COUNT,
        v2.PATH2_INPUT_COUNT,
        v2.DEV_INPUT_COUNT,
        v2.M0_INPUT_COUNT,
        v2.PROTECTED_INPUT_COUNT,
        v2.G0_INPUT_COUNT,
        v2.CODING_INPUT_COUNT,
    ) == (0, 0, 0, 0, 0, 0, 0)
