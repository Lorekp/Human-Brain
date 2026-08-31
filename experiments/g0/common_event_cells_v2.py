"""Common one-byte projection with compositional physical filler cells.

V1 supplies opaque distributed event identity and the shared fitted-GLIF output
race.  This successor appends sixteen generic input neurons: four base-four
lanes by four filler values.  Projecting one byte activates exactly one neuron
per lane.  The returned frame retains only neuron addresses, so downstream
learning can reuse fillers without receiving the byte or a host digit tuple.

This is a representation substrate only.  It does not learn or execute COPY.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from experiments.g0 import common_event_cells_v1 as v1

SCOPE = "G0_COMMON_EVENT_CELLS_V2"
STATUS = "COMMON_ONE_BYTE_COMPOSITIONAL_FILLER_PROJECTION_ONLY"
CLAIM_BOUNDARY = (
    "self-posed-reusable-one-byte-input-filler-substrate-only-"
    "no-copy-learning-or-transfer-evidence"
)

EXPECTED_V1_SOURCE_SHA256 = (
    "f425c07c44942ff7cd9672a506f616d9e6f9640c52340b481a81ad7b1603eaeb"
)
EXPECTED_V1_GEOMETRY_SHA256 = (
    "373eeffc3bc165efedbb1d60f251dfa8d7cbe4466dd31659bf847c5b88ad8c05"
)

FILLER_LANES = v1.OUTPUT_DIGIT_LANES
FILLER_RADIX = v1.OUTPUT_DIGIT_RADIX
FILLER_CELL_OFFSET = v1.TOTAL_CELL_COUNT
FILLER_CELL_COUNT = FILLER_LANES * FILLER_RADIX
INPUT_CELL_COUNT = v1.INPUT_CELL_COUNT + FILLER_CELL_COUNT
OUTPUT_CELL_COUNT = v1.OUTPUT_CELL_COUNT
TOTAL_CELL_COUNT = v1.TOTAL_CELL_COUNT + FILLER_CELL_COUNT
OPAQUE_RECURRENT_IDENTITY_CELL_COUNT = v1.TRACE_LANES
ACTIVE_FILLER_CELLS_PER_BYTE = FILLER_LANES
ACTIVE_RECURRENT_INPUT_CELLS = (
    OPAQUE_RECURRENT_IDENTITY_CELL_COUNT + ACTIVE_FILLER_CELLS_PER_BYTE
)
ACTIVE_BYTE_EMISSION_CELLS = 3 * v1.ASSEMBLY_ACTIVE_CELLS + FILLER_LANES

AUDITED_BYTE_VALUE_COUNT = v1.OUTPUT_TOKEN_COUNT
COVER_VALUES = (0x00, 0x55, 0xAA, 0xFF)
HELDOUT_COMBINATION_VALUE = 0x1B
LANE_PERMUTATION_CONTROL_VALUE = 0x4B

GENERATOR_EXECUTION_COUNT = 0
CORPUS_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
CODING_INPUT_COUNT = 0
LEARNER_EXECUTION_COUNT = 0
DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA = 0

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = Path(__file__).resolve()
_V1_SOURCE_PATH = _ROOT / "experiments/g0/common_event_cells_v1.py"
_FRAME_NAMESPACE = b"humanbrain.g0.common-event-cells-v2.frame\0"
_GEOMETRY_NAMESPACE = b"humanbrain.g0.common-event-cells-v2.geometry\0"
_AUDIT_NAMESPACE = b"humanbrain.g0.common-event-cells-v2.audit\0"
_CONFIG_NAMESPACE = b"humanbrain.g0.common-event-cells-v2.config\0"
_SUBSTRATE_NAMESPACE = b"humanbrain.g0.common-event-cells-v2.substrate\0"


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


def _authenticate_v1() -> None:
    if (
        sha256(_V1_SOURCE_PATH.read_bytes()).hexdigest()
        != EXPECTED_V1_SOURCE_SHA256
        or v1.GEOMETRY_SHA256 != EXPECTED_V1_GEOMETRY_SHA256
        or v1.TOTAL_CELL_COUNT != 65_553
        or v1.OUTPUT_CELL_COUNT != 17
    ):
        raise RuntimeError("banked CommonEventCells V1 identity changed")


def _filler_cell_id(lane: int, filler: int) -> int:
    return FILLER_CELL_OFFSET + lane * FILLER_RADIX + filler


def _project_filler_cells(value: int) -> tuple[int, ...]:
    cells: list[int] = []
    for lane in range(FILLER_LANES):
        exponent = FILLER_LANES - lane - 1
        filler = (value // (FILLER_RADIX**exponent)) % FILLER_RADIX
        cells.append(_filler_cell_id(lane, filler))
    return tuple(cells)


@dataclass(frozen=True, slots=True)
class CommonOneByteFillerFrame:
    """Authenticated physical input; no byte or digit tuple is retained."""

    opaque_frame: v1.CommonEventFrame
    filler_cells: tuple[int, ...]
    projection_work_total: int
    sha256: str

    @property
    def opaque_trace_cells(self) -> tuple[int, ...]:
        return self.opaque_frame.trace_cells

    @property
    def recurrent_input_cells(self) -> tuple[int, ...]:
        return (*self.opaque_trace_cells, *self.filler_cells)

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "opaque_frame_sha256": self.opaque_frame.sha256,
            "opaque_trace_cells": self.opaque_trace_cells,
            "filler_cells": self.filler_cells,
            "recurrent_input_cells": self.recurrent_input_cells,
            "projection_work_total": self.projection_work_total,
            "raw_byte_retained": False,
            "host_digit_tuple_retained": False,
            "raw_name_or_hash_retained_as_answer_evidence": False,
            "filler_identity_surface": "physical-neuron-addresses-only",
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


class CommonOneByteFillerCells:
    """Project one byte to opaque identity plus four shared filler neurons."""

    __slots__ = ("_v1",)

    def __init__(self) -> None:
        _authenticate_v1()
        self._v1 = v1.CommonEventCells()

    def project_one_byte(self, event_bytes: bytes) -> CommonOneByteFillerFrame:
        if type(event_bytes) is not bytes or len(event_bytes) != 1:
            raise ValueError("V2 compositional filler projection requires exactly one byte")
        opaque_frame = self._v1.project(event_bytes)
        filler_cells = _project_filler_cells(event_bytes[0])
        record = {
            "opaque_frame_sha256": opaque_frame.sha256,
            "opaque_trace_cells": opaque_frame.trace_cells,
            "filler_cells": filler_cells,
            "recurrent_input_cells": (*opaque_frame.trace_cells, *filler_cells),
            "projection_work_total": opaque_frame.work.total + FILLER_LANES,
            "raw_byte_retained": False,
            "host_digit_tuple_retained": False,
            "raw_name_or_hash_retained_as_answer_evidence": False,
            "filler_identity_surface": "physical-neuron-addresses-only",
        }
        return CommonOneByteFillerFrame(
            opaque_frame=opaque_frame,
            filler_cells=filler_cells,
            projection_work_total=opaque_frame.work.total + FILLER_LANES,
            sha256=_digest(_FRAME_NAMESPACE, record),
        )

    def authenticate_frame(self, frame: CommonOneByteFillerFrame) -> None:
        if type(frame) is not CommonOneByteFillerFrame:
            raise TypeError("V2 frame must have exact physical frame type")
        expected = _digest(_FRAME_NAMESPACE, frame.record(include_sha256=False))
        if expected != frame.sha256:
            raise ValueError("V2 physical filler frame authentication failed")
        if len(frame.filler_cells) != FILLER_LANES:
            raise ValueError("V2 physical filler frame lane count changed")
        for lane, cell_id in enumerate(frame.filler_cells):
            lane_start = FILLER_CELL_OFFSET + lane * FILLER_RADIX
            if not lane_start <= cell_id < lane_start + FILLER_RADIX:
                raise ValueError("V2 physical filler frame lane address changed")

    def recurrent_cells(
        self, frame: CommonOneByteFillerFrame
    ) -> tuple[int, ...]:
        """Return authenticated physical cells without reopening host labels."""

        self.authenticate_frame(frame)
        return frame.recurrent_input_cells

    def race_output(
        self, raw_scores: tuple[tuple[int, ...], ...]
    ) -> v1.CommonOutputDecision:
        """Use the exact unchanged V1 physical output framing/race."""

        return self._v1.race_output(raw_scores)


GEOMETRY_RECORD = {
    "scope": SCOPE,
    "v1_source_sha256": EXPECTED_V1_SOURCE_SHA256,
    "v1_geometry_sha256": EXPECTED_V1_GEOMETRY_SHA256,
    "projection_input": "exactly-one-opaque-byte",
    "opaque_recurrent_identity_cells": OPAQUE_RECURRENT_IDENTITY_CELL_COUNT,
    "filler": {
        "encoding": "four-base-four-physical-input-lanes",
        "lanes": FILLER_LANES,
        "radix": FILLER_RADIX,
        "cell_offset": FILLER_CELL_OFFSET,
        "cell_count": FILLER_CELL_COUNT,
        "active_cells_per_byte": ACTIVE_FILLER_CELLS_PER_BYTE,
        "raw_digit_tuple_retained_after_projection": False,
    },
    "active_recurrent_input_cells": ACTIVE_RECURRENT_INPUT_CELLS,
    "active_byte_emission_cells": ACTIVE_BYTE_EMISSION_CELLS,
    "input_cell_count": INPUT_CELL_COUNT,
    "output_cell_count": OUTPUT_CELL_COUNT,
    "total_cell_count": TOTAL_CELL_COUNT,
    "output_surface": "exact-v1-four-lane-fitted-glif-plus-tonic-stop",
    "learner_cell_count": 0,
    "domain_specific_computational_primitive_delta": 0,
}
GEOMETRY_SHA256 = _digest(_GEOMETRY_NAMESPACE, GEOMETRY_RECORD)


@dataclass(frozen=True, slots=True)
class FillerGeometryAudit:
    audited_byte_value_count: int
    unique_filler_tuple_count: int
    whole_byte_filler_alias_pair_count: int
    reachable_filler_cell_count: int
    activation_count_by_filler_cell: tuple[tuple[int, int], ...]
    exactly_one_filler_per_lane_for_every_byte: bool
    heldout_combination_value: int
    heldout_new_filler_cell_count: int
    heldout_opaque_identity_seen_in_cover: bool
    lane_permutation_control_value: int
    lane_permutation_changed_filler_cell_count: int
    raw_label_available_after_projection: bool
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


def _build_geometry_audit() -> FillerGeometryAudit:
    cells = CommonOneByteFillerCells()
    frames = tuple(cells.project_one_byte(bytes((value,))) for value in range(256))
    filler_tuples = tuple(frame.filler_cells for frame in frames)
    counts = Counter(cell for row in filler_tuples for cell in row)
    tuple_counts = Counter(filler_tuples)
    alias_pairs = sum(count * (count - 1) // 2 for count in tuple_counts.values())
    one_per_lane = all(
        all(
            FILLER_CELL_OFFSET + lane * FILLER_RADIX
            <= cell_id
            < FILLER_CELL_OFFSET + (lane + 1) * FILLER_RADIX
            for lane, cell_id in enumerate(frame.filler_cells)
        )
        for frame in frames
    )
    cover_frames = tuple(frames[value] for value in COVER_VALUES)
    heldout = frames[HELDOUT_COMBINATION_VALUE]
    cover_fillers = {cell for frame in cover_frames for cell in frame.filler_cells}
    cover_opaque = {frame.opaque_frame.sha256 for frame in cover_frames}
    permuted = frames[LANE_PERMUTATION_CONTROL_VALUE]
    changed = len(set(heldout.filler_cells) ^ set(permuted.filler_cells))
    record = {
        "audited_byte_value_count": len(frames),
        "unique_filler_tuple_count": len(tuple_counts),
        "whole_byte_filler_alias_pair_count": alias_pairs,
        "reachable_filler_cell_count": len(counts),
        "activation_count_by_filler_cell": tuple(sorted(counts.items())),
        "exactly_one_filler_per_lane_for_every_byte": one_per_lane,
        "heldout_combination_value": HELDOUT_COMBINATION_VALUE,
        "heldout_new_filler_cell_count": len(set(heldout.filler_cells) - cover_fillers),
        "heldout_opaque_identity_seen_in_cover": (
            heldout.opaque_frame.sha256 in cover_opaque
        ),
        "lane_permutation_control_value": LANE_PERMUTATION_CONTROL_VALUE,
        "lane_permutation_changed_filler_cell_count": changed,
        "raw_label_available_after_projection": False,
    }
    return FillerGeometryAudit(
        **record,
        sha256=_digest(_AUDIT_NAMESPACE, record),
    )


_authenticate_v1()
GEOMETRY_AUDIT = _build_geometry_audit()
if (
    GEOMETRY_AUDIT.audited_byte_value_count != AUDITED_BYTE_VALUE_COUNT
    or GEOMETRY_AUDIT.unique_filler_tuple_count != AUDITED_BYTE_VALUE_COUNT
    or GEOMETRY_AUDIT.whole_byte_filler_alias_pair_count != 0
    or GEOMETRY_AUDIT.reachable_filler_cell_count != FILLER_CELL_COUNT
    or not GEOMETRY_AUDIT.exactly_one_filler_per_lane_for_every_byte
    or GEOMETRY_AUDIT.heldout_new_filler_cell_count != 0
    or GEOMETRY_AUDIT.heldout_opaque_identity_seen_in_cover
    or GEOMETRY_AUDIT.lane_permutation_changed_filler_cell_count == 0
    or any(
        count != AUDITED_BYTE_VALUE_COUNT // FILLER_RADIX
        for _, count in GEOMETRY_AUDIT.activation_count_by_filler_cell
    )
):
    raise RuntimeError("V2 common filler geometry audit failed")

CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "geometry_sha256": GEOMETRY_SHA256,
    "geometry_audit_sha256": GEOMETRY_AUDIT.sha256,
    "cover_values": COVER_VALUES,
    "heldout_combination_value": HELDOUT_COMBINATION_VALUE,
    "lane_permutation_control_value": LANE_PERMUTATION_CONTROL_VALUE,
    "host_value_used_only_during_projection": True,
    "raw_byte_name_hash_or_digit_tuple_used_after_projection": False,
    "copy_learner_present": False,
    "learner_execution_count": 0,
    "generator_corpus_path2_dev_m0_protected_g0_coding_inputs": (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ),
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)
SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()
SUBSTRATE_ROOT_SHA256 = _digest(
    _SUBSTRATE_NAMESPACE,
    {
        "source_sha256": SOURCE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "geometry_sha256": GEOMETRY_SHA256,
        "geometry_audit_sha256": GEOMETRY_AUDIT.sha256,
    },
)


__all__ = [
    "ACTIVE_FILLER_CELLS_PER_BYTE",
    "ACTIVE_RECURRENT_INPUT_CELLS",
    "CLAIM_BOUNDARY",
    "CODING_INPUT_COUNT",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "CORPUS_INPUT_COUNT",
    "COVER_VALUES",
    "DEV_INPUT_COUNT",
    "DOMAIN_SPECIFIC_COMPUTATIONAL_PRIMITIVE_DELTA",
    "FILLER_CELL_COUNT",
    "FILLER_CELL_OFFSET",
    "FILLER_LANES",
    "FILLER_RADIX",
    "G0_INPUT_COUNT",
    "GENERATOR_EXECUTION_COUNT",
    "GEOMETRY_AUDIT",
    "GEOMETRY_RECORD",
    "GEOMETRY_SHA256",
    "HELDOUT_COMBINATION_VALUE",
    "INPUT_CELL_COUNT",
    "LANE_PERMUTATION_CONTROL_VALUE",
    "LEARNER_EXECUTION_COUNT",
    "M0_INPUT_COUNT",
    "OUTPUT_CELL_COUNT",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "SCOPE",
    "SOURCE_SHA256",
    "STATUS",
    "SUBSTRATE_ROOT_SHA256",
    "TOTAL_CELL_COUNT",
    "CommonOneByteFillerCells",
    "CommonOneByteFillerFrame",
    "FillerGeometryAudit",
]
