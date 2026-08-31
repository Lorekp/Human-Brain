from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import (
    common_cross_context_relational_schema_consolidation_v1 as constant_schema,
)
from experiments.r3_pilot import common_one_byte_copy_relation_v1 as copy_relation


def test_copy_relation_stops_before_learning_without_input_digit_fillers() -> None:
    constant_source_before = constant_schema.SOURCE_SHA256
    stop = copy_relation.preflight_common_one_byte_copy_relation()

    assert stop is copy_relation.COPY_RELATION_PREALLOCATION_STOP
    assert stop.status == "STOP_DIGIT_FILLER_PREALLOCATION_REQUIRED"
    assert (
        stop.stop_code
        == "STOP_COMMON_EVENT_INPUT_DIGIT_FILLER_CELLS_NOT_PREALLOCATED"
    )
    assert stop.audit.audited_byte_value_count == 256
    assert stop.audit.unique_value_assembly_count == 256
    assert stop.audit.unique_binding_assembly_count == 256
    assert stop.audit.unique_trace_identity_count == 256
    assert stop.audit.candidate_counts_by_lane_and_digit == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    assert stop.audit.observed_input_digit_filler_cell_count == 0
    assert stop.audit.required_input_digit_filler_cell_count == 16
    assert stop.audit.host_digit_grouping_used_as_neural_evidence is False
    assert stop.audit.raw_byte_or_name_used_as_neural_evidence is False
    assert stop.added_cell_count == 0
    assert all(count == 0 for _, count in stop.learning_execution_counts)
    assert copy_relation.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA == 0
    assert (
        copy_relation.CORPUS_INPUT_COUNT,
        copy_relation.PATH2_INPUT_COUNT,
        copy_relation.DEV_INPUT_COUNT,
        copy_relation.M0_INPUT_COUNT,
        copy_relation.PROTECTED_INPUT_COUNT,
        copy_relation.G0_INPUT_COUNT,
        copy_relation.CODING_INPUT_COUNT,
    ) == (0, 0, 0, 0, 0, 0, 0)

    with pytest.raises(
        copy_relation.CopyRelationPreallocationRequired
    ) as stopped:
        copy_relation.fit_common_one_byte_copy_relation(
            lessons=(b"must-not-be-inspected",)
        )
    assert stopped.value.stop is stop
    assert all(count == 0 for _, count in stop.learning_execution_counts)
    assert constant_source_before == constant_schema.SOURCE_SHA256
    constant_path = Path(constant_schema.__file__).resolve()
    assert sha256(constant_path.read_bytes()).hexdigest() == constant_source_before
    assert stop.stop_root_sha256 == copy_relation.STOP_ROOT_SHA256
