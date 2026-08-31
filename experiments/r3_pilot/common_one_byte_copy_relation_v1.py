"""Fail-closed preallocation audit for a learned one-byte COPY relation.

The banked common projection has physical base-four lane/value cells on its
output surface, but its input surface represents a byte with whole-value hash
assemblies and whole-value recurrent trace cells.  A local learner therefore
cannot identify a digit filler that is shared across novel byte combinations.

This module freezes that exact representation blocker before any lesson,
episodic write, synaptic update, output race, or recall.  Host-side base-four
grouping is used only by the representation audit that refuses training; it is
never admitted as neural evidence.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

from experiments.g0 import common_event_cells_v1 as projection
from experiments.r3_pilot import (
    common_cross_context_relational_schema_consolidation_v1 as constant_schema,
)

SCOPE = "COMMON_ONE_BYTE_COPY_RELATION_V1"
STATUS = "STOP_DIGIT_FILLER_PREALLOCATION_REQUIRED"
STOP_CODE = "STOP_COMMON_EVENT_INPUT_DIGIT_FILLER_CELLS_NOT_PREALLOCATED"
CLAIM_BOUNDARY = (
    "self-posed-common-representation-preflight-only-no-copy-learning-"
    "or-variable-binding-evidence"
)

EXPECTED_COMMON_EVENT_SOURCE_SHA256 = (
    "f425c07c44942ff7cd9672a506f616d9e6f9640c52340b481a81ad7b1603eaeb"
)
EXPECTED_COMMON_EVENT_GEOMETRY_SHA256 = (
    "373eeffc3bc165efedbb1d60f251dfa8d7cbe4466dd31659bf847c5b88ad8c05"
)
EXPECTED_CONSTANT_SCHEMA_SOURCE_SHA256 = (
    "93d91ac5d1b0b4874e625534061535b80dc332ad64efb47aa72e15a6500c0044"
)
EXPECTED_CONSTANT_SCHEMA_CONFIG_SHA256 = (
    "1d01e21a747319ebae647141baf6e0feeb73e6559f3f832b6293432538658106"
)
EXPECTED_CONSTANT_SCHEMA_CHECKPOINT_SHA256 = (
    "befc989c208e8f7e5d8412a5e9aad7a21f26a42475848ffabe61bf07f8e43b4d"
)

AUDITED_BYTE_VALUE_COUNT = projection.OUTPUT_TOKEN_COUNT
REQUIRED_INPUT_DIGIT_FILLER_CELL_COUNT = (
    projection.OUTPUT_DIGIT_LANES * projection.OUTPUT_DIGIT_RADIX
)
OBSERVED_INPUT_DIGIT_FILLER_CELL_COUNT = 0
ADDED_CELL_COUNT = 0
REUSED_COMMON_EVENT_CELL_COUNT = projection.TOTAL_CELL_COUNT
PRESERVED_CONSTANT_SCHEMA_DECLARED_CELL_COUNT = (
    constant_schema.TOTAL_DECLARED_CELL_COUNT
)

SELF_POSED_REPRESENTATION_AUDIT_INPUT_COUNT = AUDITED_BYTE_VALUE_COUNT
LEARNING_EXECUTION_COUNTS = (
    ("fit_calls_completed", 0),
    ("training_lessons_consumed", 0),
    ("teacher_on_lessons", 0),
    ("teacher_off_lessons", 0),
    ("cell_off_controls", 0),
    ("episodic_commits", 0),
    ("synaptic_update_events", 0),
    ("output_races", 0),
    ("recalls", 0),
    ("constant_schema_mutations", 0),
    ("physical_learning_cell_steps", 0),
)
CORPUS_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
CODING_INPUT_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_COMMON_EVENT_SOURCE_PATH = _ROOT / "experiments/g0/common_event_cells_v1.py"
_CONSTANT_SCHEMA_SOURCE_PATH = (
    _ROOT
    / "experiments/r3_pilot/common_cross_context_relational_schema_consolidation_v1.py"
)
_CONFIG_NAMESPACE = b"common-one-byte-copy-relation-v1-config\0"
_AUDIT_NAMESPACE = b"common-one-byte-copy-relation-v1-audit\0"
_STOP_NAMESPACE = b"common-one-byte-copy-relation-v1-stop\0"


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _authenticate_dependencies() -> None:
    if (
        sha256(_COMMON_EVENT_SOURCE_PATH.read_bytes()).hexdigest()
        != EXPECTED_COMMON_EVENT_SOURCE_SHA256
        or projection.GEOMETRY_SHA256 != EXPECTED_COMMON_EVENT_GEOMETRY_SHA256
        or sha256(_CONSTANT_SCHEMA_SOURCE_PATH.read_bytes()).hexdigest()
        != EXPECTED_CONSTANT_SCHEMA_SOURCE_SHA256
        or constant_schema.SOURCE_SHA256
        != EXPECTED_CONSTANT_SCHEMA_SOURCE_SHA256
        or constant_schema.CONFIG_SHA256
        != EXPECTED_CONSTANT_SCHEMA_CONFIG_SHA256
        or constant_schema.DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA != 0
    ):
        raise RuntimeError("banked common representation dependency identity changed")


def _output_digit(value: int, lane: int) -> int:
    """Host diagnostic used only to reject the unfit input representation."""

    exponent = projection.OUTPUT_DIGIT_LANES - lane - 1
    return (value // (projection.OUTPUT_DIGIT_RADIX**exponent)) % (
        projection.OUTPUT_DIGIT_RADIX
    )


@dataclass(frozen=True, slots=True)
class DigitFillerPreallocationAudit:
    audited_byte_value_count: int
    unique_value_assembly_count: int
    unique_binding_assembly_count: int
    unique_trace_identity_count: int
    candidate_cells_by_lane_and_digit: tuple[tuple[tuple[int, ...], ...], ...]
    candidate_counts_by_lane_and_digit: tuple[tuple[int, ...], ...]
    observed_input_digit_filler_cell_count: int
    required_input_digit_filler_cell_count: int
    audit_projection_work_total: int
    host_digit_grouping_used_as_neural_evidence: bool
    raw_byte_or_name_used_as_neural_evidence: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


def _audit_digit_filler_preallocation() -> DigitFillerPreallocationAudit:
    cells = projection.CommonEventCells()
    frames = tuple(cells.project(bytes((value,))) for value in range(256))
    sensitive_cells = tuple(
        frozenset(
            (
                *frame.byte_emissions[0].value_cells,
                *frame.byte_emissions[0].binding_cells,
                *frame.trace_cells,
            )
        )
        for frame in frames
    )

    lane_candidates: list[tuple[tuple[int, ...], ...]] = []
    for lane in range(projection.OUTPUT_DIGIT_LANES):
        digit_candidates: list[tuple[int, ...]] = []
        for digit in range(projection.OUTPUT_DIGIT_RADIX):
            matching = tuple(
                sensitive_cells[value]
                for value in range(256)
                if _output_digit(value, lane) == digit
            )
            nonmatching = tuple(
                sensitive_cells[value]
                for value in range(256)
                if _output_digit(value, lane) != digit
            )
            shared = set.intersection(*(set(row) for row in matching))
            other = set.union(*(set(row) for row in nonmatching))
            digit_candidates.append(tuple(sorted(shared - other)))
        lane_candidates.append(tuple(digit_candidates))

    candidates = tuple(lane_candidates)
    counts = tuple(tuple(len(row) for row in lane) for lane in candidates)
    observed_count = sum(sum(lane) for lane in counts)
    record = {
        "audited_byte_value_count": len(frames),
        "unique_value_assembly_count": len(
            {frame.byte_emissions[0].value_cells for frame in frames}
        ),
        "unique_binding_assembly_count": len(
            {frame.byte_emissions[0].binding_cells for frame in frames}
        ),
        "unique_trace_identity_count": len({frame.trace_cells for frame in frames}),
        "candidate_cells_by_lane_and_digit": candidates,
        "candidate_counts_by_lane_and_digit": counts,
        "observed_input_digit_filler_cell_count": observed_count,
        "required_input_digit_filler_cell_count": (
            REQUIRED_INPUT_DIGIT_FILLER_CELL_COUNT
        ),
        "audit_projection_work_total": sum(frame.work.total for frame in frames),
        "host_digit_grouping_used_as_neural_evidence": False,
        "raw_byte_or_name_used_as_neural_evidence": False,
    }
    return DigitFillerPreallocationAudit(
        **record,
        sha256=_digest(_AUDIT_NAMESPACE, record),
    )


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "stop_code": STOP_CODE,
    "claim_boundary": CLAIM_BOUNDARY,
    "common_event_source_sha256": EXPECTED_COMMON_EVENT_SOURCE_SHA256,
    "common_event_geometry_sha256": EXPECTED_COMMON_EVENT_GEOMETRY_SHA256,
    "constant_schema_source_sha256": EXPECTED_CONSTANT_SCHEMA_SOURCE_SHA256,
    "constant_schema_config_sha256": EXPECTED_CONSTANT_SCHEMA_CONFIG_SHA256,
    "constant_schema_checkpoint_sha256": (
        EXPECTED_CONSTANT_SCHEMA_CHECKPOINT_SHA256
    ),
    "input_representation": (
        "whole-byte-value-assemblies-position-value-bindings-and-whole-byte-traces"
    ),
    "output_representation": "four-base-four-lane-value-cells-plus-tonic-stop",
    "required_input_digit_filler_cell_count": (
        REQUIRED_INPUT_DIGIT_FILLER_CELL_COUNT
    ),
    "observed_input_digit_filler_cell_count": (
        OBSERVED_INPUT_DIGIT_FILLER_CELL_COUNT
    ),
    "minimum_independent_contexts_if_unblocked": 2,
    "host_digit_decoder_in_learner": False,
    "whole_byte_static_result_lookup": False,
    "added_cell_count": ADDED_CELL_COUNT,
    "domain_specific_computational_primitive_delta": 0,
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)


@dataclass(frozen=True, slots=True)
class FrozenCopyRelationPreallocationStop:
    status: str
    stop_code: str
    reason: str
    audit: DigitFillerPreallocationAudit
    source_sha256: str
    config_sha256: str
    common_event_source_sha256: str
    common_event_geometry_sha256: str
    constant_schema_source_sha256: str
    constant_schema_config_sha256: str
    constant_schema_checkpoint_sha256: str
    learning_execution_counts: tuple[tuple[str, int], ...]
    self_posed_representation_audit_input_count: int
    added_cell_count: int
    reused_common_event_cell_count: int
    preserved_constant_schema_declared_cell_count: int
    domain_specific_computational_primitive_delta: int
    stop_root_sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "stop_root_sha256"
        }
        record["audit"] = self.audit.record()
        if include_sha256:
            record["stop_root_sha256"] = self.stop_root_sha256
        return record


class CopyRelationPreallocationRequired(RuntimeError):
    """Raised before training because the current input lacks digit fillers."""

    def __init__(self, stop: FrozenCopyRelationPreallocationStop) -> None:
        super().__init__(f"{stop.stop_code}: {stop.reason}")
        self.stop = stop


SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()
_authenticate_dependencies()
_AUDIT = _audit_digit_filler_preallocation()
if (
    _AUDIT.audited_byte_value_count != AUDITED_BYTE_VALUE_COUNT
    or _AUDIT.unique_value_assembly_count != AUDITED_BYTE_VALUE_COUNT
    or _AUDIT.unique_binding_assembly_count != AUDITED_BYTE_VALUE_COUNT
    or _AUDIT.unique_trace_identity_count != AUDITED_BYTE_VALUE_COUNT
    or _AUDIT.observed_input_digit_filler_cell_count
    != OBSERVED_INPUT_DIGIT_FILLER_CELL_COUNT
):
    raise RuntimeError("common input digit-filler preallocation audit changed")

_STOP_PAYLOAD = {
    "status": STATUS,
    "stop_code": STOP_CODE,
    "reason": (
        "the current common input exposes only whole-byte identities; a held-out "
        "cross-lane byte combination has no shared physical lane/value filler to "
        "route, so proceeding would require host digit decoding or memorization"
    ),
    "audit": _AUDIT,
    "source_sha256": SOURCE_SHA256,
    "config_sha256": CONFIG_SHA256,
    "common_event_source_sha256": EXPECTED_COMMON_EVENT_SOURCE_SHA256,
    "common_event_geometry_sha256": EXPECTED_COMMON_EVENT_GEOMETRY_SHA256,
    "constant_schema_source_sha256": EXPECTED_CONSTANT_SCHEMA_SOURCE_SHA256,
    "constant_schema_config_sha256": EXPECTED_CONSTANT_SCHEMA_CONFIG_SHA256,
    "constant_schema_checkpoint_sha256": (
        EXPECTED_CONSTANT_SCHEMA_CHECKPOINT_SHA256
    ),
    "learning_execution_counts": LEARNING_EXECUTION_COUNTS,
    "self_posed_representation_audit_input_count": (
        SELF_POSED_REPRESENTATION_AUDIT_INPUT_COUNT
    ),
    "added_cell_count": ADDED_CELL_COUNT,
    "reused_common_event_cell_count": REUSED_COMMON_EVENT_CELL_COUNT,
    "preserved_constant_schema_declared_cell_count": (
        PRESERVED_CONSTANT_SCHEMA_DECLARED_CELL_COUNT
    ),
    "domain_specific_computational_primitive_delta": 0,
}
COPY_RELATION_PREALLOCATION_STOP = FrozenCopyRelationPreallocationStop(
    **_STOP_PAYLOAD,
    stop_root_sha256=_digest(_STOP_NAMESPACE, _STOP_PAYLOAD),
)
STOP_ROOT_SHA256 = COPY_RELATION_PREALLOCATION_STOP.stop_root_sha256


def preflight_common_one_byte_copy_relation() -> FrozenCopyRelationPreallocationStop:
    """Return the authenticated STOP without consuming a lesson."""

    _authenticate_dependencies()
    if sha256(_SOURCE_PATH.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("copy-relation preallocation STOP source identity changed")
    return COPY_RELATION_PREALLOCATION_STOP


def fit_common_one_byte_copy_relation(lessons: object) -> NoReturn:
    """Fail before inspecting lessons; the current representation is unfit."""

    del lessons
    raise CopyRelationPreallocationRequired(preflight_common_one_byte_copy_relation())


__all__ = [
    "ADDED_CELL_COUNT",
    "CLAIM_BOUNDARY",
    "CODING_INPUT_COUNT",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "COPY_RELATION_PREALLOCATION_STOP",
    "CORPUS_INPUT_COUNT",
    "DEV_INPUT_COUNT",
    "DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA",
    "EXPECTED_COMMON_EVENT_GEOMETRY_SHA256",
    "EXPECTED_COMMON_EVENT_SOURCE_SHA256",
    "EXPECTED_CONSTANT_SCHEMA_CHECKPOINT_SHA256",
    "EXPECTED_CONSTANT_SCHEMA_CONFIG_SHA256",
    "EXPECTED_CONSTANT_SCHEMA_SOURCE_SHA256",
    "G0_INPUT_COUNT",
    "LEARNING_EXECUTION_COUNTS",
    "M0_INPUT_COUNT",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "REQUIRED_INPUT_DIGIT_FILLER_CELL_COUNT",
    "SCOPE",
    "SOURCE_SHA256",
    "STATUS",
    "STOP_CODE",
    "STOP_ROOT_SHA256",
    "CopyRelationPreallocationRequired",
    "DigitFillerPreallocationAudit",
    "FrozenCopyRelationPreallocationStop",
    "fit_common_one_byte_copy_relation",
    "preflight_common_one_byte_copy_relation",
]
