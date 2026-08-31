"""Target-free static inference for the banked occurrence-factor learner.

Normal import authenticates one immutable payload and reconstructs only its
direct physical query vocabulary, output registries, and frozen fitted-GLIF
state.  It imports neither the banked executable learner nor the V2 annotation
capsule, performs no fitting, and exposes no FIT source bundle fixture.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import numpy as np
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCHEMA = "PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_STATIC_CHECKPOINT_CAPSULE_V1"
READY = "PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_READY"
STOP_ZERO = "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_ZERO"
STOP_AMBIGUOUS = "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_AMBIGUOUS"
STOP_UNSUPPORTED = "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_UNSUPPORTED"

BANKED_COMMIT = "3ac16dc22caae8ce2ac14b82f8579494c0b30dc0"
BANKED_SOURCE_SHA256 = (
    "e656dfdf4799800c6d6c633e5a7474d9414a22a1d69dc6ca08b847cf883e929e"
)
BANKED_TEST_SHA256 = (
    "0d63e23bff61e40860d16227186d6497a0f2ba40c49855dc682e43c79c8956e8"
)
BANKED_PROTOCOL_SHA256 = (
    "792b517f364f075c46fef51671ea26b8117eb0f78d37d3b1d0cdee6867c26e61"
)
BANKED_CHECKPOINT_SHA256 = (
    "aee5d7da81f11a2d71bd1a6cb37383f8364a5e5427fe5055ef15c11790dba290"
)
BANKED_SUMMARY_SHA256 = (
    "dce5727421810f9759d3d04a7d6501123eeccd8fe97bb3bc4db83cc2977ff4b5"
)
BANKED_PHYSICAL_WORK_SHA256 = (
    "a4bda9c6593ba7dd86b2e5b5bb8301c2d1e42c1e4011f6cf51ff11d5bb20f9c2"
)

ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_fit_occurrence_factor_static_checkpoint_capsule_v1.npz"
)
ARTIFACT_SHA256 = "42ca5251b7a045ca98da4e2b008a82d8c89d42f229f6b3eb670ade7a30919711"
ARTIFACT_CONTENT_ROOT = (
    "9dcd2022ad8550bf597ee3614f3052699216e2fe63c3a57e67e3bafb0b1eeba8"
)
ARTIFACT_BYTES = 19_970

NORMAL_RUNTIME_TEACHER_TARGET_READ_COUNT = 0
NORMAL_RUNTIME_FIT_COUNT = 0
NORMAL_RUNTIME_LEARNER_IMPORT_COUNT = 0
NORMAL_RUNTIME_ANNOTATION_IMPORT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
PANEL_SPEND_COUNT = 0

QUERY_SOURCE = 0
QUERY_EDGE = 1
QUERY_FACTOR = 2
QUERY_RELATION = 3
QUERY_DOMAIN = 4
QUERY_COPY = 5
QUERY_SIGNATURE = 6

EVENT_SOURCE = 1
EVENT_EDGE = 2
EVENT_FACTOR = 3
EVENT_RELATION = 4
EVENT_DOMAIN = 5
EVENT_COPY = 6
EVENT_SIGNATURE = 7
EVENT_KIND_COUNT = 8

FACTOR_KINDS = ("literal", "role")
RELATION_OPERATORS = ("eq", "ne", "lt", "le", "gt", "ge")
RHS_SLOT = 0
RHS_VALUE = 1
SIGNATURE_PARAMETER = 0
SIGNATURE_RETURN = 1

MAX_LOCAL_SLOT = 4
MAX_CHILD_POSITION = 2
SIGNATURE_POSITIONS = (0,)
NODE_CARRIER_CELL_COUNT = 256
NODE_CARRIER_ACTIVE_CELL_COUNT = 12
COPY_GROUP_CELL_COUNT = 64
COPY_GROUP_ACTIVE_CELL_COUNT = 8
TYPE_CELL_COUNT = 48
TYPE_ACTIVE_CELL_COUNT = 8
VALUE_EVENT_CELL_COUNT = 96
RAW_SPAN_ACTIVE_CELL_COUNT = 12

CARRIER_OFFSET = 0
EVENT_KIND_OFFSET = CARRIER_OFFSET + NODE_CARRIER_CELL_COUNT
SOURCE_KIND_OFFSET = EVENT_KIND_OFFSET + EVENT_KIND_COUNT
CHILD_POSITION_OFFSET = SOURCE_KIND_OFFSET + 2
OCCUPANCY_OFFSET = CHILD_POSITION_OFFSET + MAX_CHILD_POSITION
FACTOR_KIND_OFFSET = OCCUPANCY_OFFSET + 7
TYPE_OFFSET = FACTOR_KIND_OFFSET + len(FACTOR_KINDS)
RELATION_OPERATOR_OFFSET = TYPE_OFFSET + TYPE_CELL_COUNT
RHS_KIND_OFFSET = RELATION_OPERATOR_OFFSET + len(RELATION_OPERATORS)
RIGHT_SLOT_OFFSET = RHS_KIND_OFFSET + 2
VALUE_EVENT_OFFSET = RIGHT_SLOT_OFFSET + MAX_LOCAL_SLOT
COPY_GROUP_OFFSET = VALUE_EVENT_OFFSET + VALUE_EVENT_CELL_COUNT
SIGNATURE_ROLE_OFFSET = COPY_GROUP_OFFSET + COPY_GROUP_CELL_COUNT
OUTPUT_CELL_COUNT = SIGNATURE_ROLE_OFFSET + 2
INPUT_CELL_COUNT = 372

TICK_CAP = 4
SCORE_THRESHOLD_NUMERATOR = 3
SCORE_THRESHOLD_DENOMINATOR = 4
OUTPUT_CURRENT_PA = 5_000.0
OUTPUT_STEPS = 30
TICK_DTYPE = np.dtype(np.uint8)

_FORMAT = "public-train-fit-occurrence-factor-learner-v1-static-capsule-v1"
_ARTIFACT_KEYS = {"metadata", "excitatory", "inhibitory"}
_CONTENT_NAMESPACE = b"public-train-fit-occurrence-factor-static-v1-content\0"
_NAMESPACE = b"public-train-fit-occurrence-factor-learner-v1\0"
_SPIKE_KERNEL = CompiledRestingGlifRace.compile(step_count=OUTPUT_STEPS)


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("ascii"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


def _array_frame(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    return b"".join(
        (
            name.encode("ascii"),
            b"\0",
            array.dtype.str.encode("ascii"),
            b"\0",
            _canonical_json(tuple(int(item) for item in array.shape)),
            b"\0",
            array.tobytes(order="C"),
        )
    )


def _checkpoint_payload(
    metadata: Mapping[str, object],
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> bytes:
    return b"".join(
        (
            _canonical_json(dict(metadata)),
            b"\0",
            _array_frame("excitatory", excitatory),
            _array_frame("inhibitory", inhibitory),
        )
    )


def _deep_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_deep_tuple(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _deep_tuple(item) for key, item in value.items()}
    return value


def _validate_cells(
    cells: tuple[int, ...],
    *,
    width: int,
    count: int | None = None,
) -> None:
    if (
        type(cells) is not tuple
        or tuple(sorted(set(cells))) != cells
        or any(type(cell) is not int or not 0 <= cell < width for cell in cells)
        or (count is not None and len(cells) != count)
    ):
        raise ValueError("physical sparse assembly changed")


@dataclass(frozen=True, slots=True)
class RawPhysicalSourceEvent:
    assembly_cells: tuple[int, ...]
    direction: str
    cue_slot: int
    selected_slot: int

    def __post_init__(self) -> None:
        _validate_cells(
            self.assembly_cells,
            width=512,
            count=RAW_SPAN_ACTIVE_CELL_COUNT,
        )
        expected_selected = (
            (self.cue_slot + 1) % MAX_LOCAL_SLOT
            if self.direction == "NEXT"
            else (self.cue_slot - 1) % MAX_LOCAL_SLOT
        )
        if (
            self.direction not in ("NEXT", "PREV")
            or type(self.cue_slot) is not int
            or type(self.selected_slot) is not int
            or not 0 <= self.cue_slot < MAX_LOCAL_SLOT
            or not 0 <= self.selected_slot < MAX_LOCAL_SLOT
            or self.selected_slot != expected_selected
        ):
            raise ValueError("raw physical source geometry changed")

    def neural_record(self) -> dict[str, object]:
        return {
            "source_kind_cell": 0,
            "assembly_cells": self.assembly_cells,
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
        }


@dataclass(frozen=True, slots=True)
class LiteralPhysicalSourceEvent:
    typed_value_event_cells: tuple[int, ...]
    typed_value_type_cells: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_cells(self.typed_value_event_cells, width=VALUE_EVENT_CELL_COUNT)
        _validate_cells(
            self.typed_value_type_cells,
            width=TYPE_CELL_COUNT,
            count=TYPE_ACTIVE_CELL_COUNT,
        )
        if not self.typed_value_event_cells:
            raise ValueError("typed literal event must activate physical cells")

    def neural_record(self) -> dict[str, object]:
        return {
            "source_kind_cell": 1,
            "typed_value_event_cells": self.typed_value_event_cells,
            "typed_value_type_cells": self.typed_value_type_cells,
        }


PhysicalSourceEvent = RawPhysicalSourceEvent | LiteralPhysicalSourceEvent
PhysicalFeature = tuple[object, ...]


def _source_event_feature(event: PhysicalSourceEvent) -> PhysicalFeature:
    if isinstance(event, RawPhysicalSourceEvent):
        return (
            "raw",
            event.direction,
            event.cue_slot,
            event.selected_slot,
            event.assembly_cells,
        )
    return (
        "literal",
        event.typed_value_event_cells,
        event.typed_value_type_cells,
    )


@dataclass(frozen=True, slots=True)
class _Query:
    query_kind_cell: int
    source_event: PhysicalSourceEvent | None = None
    source_context: tuple[PhysicalSourceEvent, ...] = ()
    primary_carrier_cells: tuple[int, ...] = ()
    secondary_carrier_cells: tuple[int, ...] = ()
    local_slot: int | None = None
    signature_role_cell: int | None = None
    signature_position_cell: int | None = None

    def neural_record(self) -> dict[str, object]:
        record: dict[str, object] = {"query_kind_cell": self.query_kind_cell}
        if self.source_event is not None:
            record["physical_source_event"] = self.source_event.neural_record()
            record["unordered_physical_source_context"] = tuple(
                item.neural_record() for item in self.source_context
            )
        if self.primary_carrier_cells:
            record["primary_carrier_cells"] = self.primary_carrier_cells
        if self.secondary_carrier_cells:
            record["secondary_carrier_cells"] = self.secondary_carrier_cells
        if self.local_slot is not None:
            record["local_slot"] = self.local_slot
        if self.signature_role_cell is not None:
            record["signature_role_cell"] = self.signature_role_cell
        if self.signature_position_cell is not None:
            record["signature_position_cell"] = self.signature_position_cell
        return record


def _direct_features(query: _Query) -> tuple[PhysicalFeature, ...]:
    if query.query_kind_cell == QUERY_SOURCE:
        if query.source_event is None or not query.source_context:
            return ()
        features = (
            (
                QUERY_SOURCE,
                _source_event_feature(query.source_event),
                tuple(
                    _source_event_feature(context_event)
                    for context_event in query.source_context
                ),
            ),
        )
    elif query.query_kind_cell == QUERY_EDGE:
        features = (
            (
                QUERY_EDGE,
                query.primary_carrier_cells,
                query.secondary_carrier_cells,
            ),
        )
    elif query.query_kind_cell in (
        QUERY_FACTOR,
        QUERY_RELATION,
        QUERY_DOMAIN,
        QUERY_COPY,
    ):
        features = (
            (
                query.query_kind_cell,
                query.primary_carrier_cells,
                query.local_slot,
            ),
        )
    elif query.query_kind_cell == QUERY_SIGNATURE:
        features = (
            (
                QUERY_SIGNATURE,
                query.primary_carrier_cells,
                query.signature_role_cell,
                query.signature_position_cell,
            ),
        )
    else:
        return ()
    return tuple(sorted(set(features), key=_canonical_json))


@dataclass(frozen=True, slots=True)
class _StaticProtocol:
    carrier_registry: tuple[tuple[int, ...], ...]
    copy_group_registry: tuple[tuple[int, ...], ...]
    type_registry: tuple[tuple[int, ...], ...]
    value_event_registry: tuple[tuple[int, ...], ...]
    feature_registry: tuple[PhysicalFeature, ...]
    occupancy_registry: tuple[tuple[int, ...], ...]
    protocol_sha256: str


class _Decoded(NamedTuple):
    status: str
    query: _Query
    carrier_cells: tuple[int, ...]
    event_kind_cell: int | None
    source_kind_cell: int | None
    child_position_cell: int | None
    occupancy_cells: tuple[int, ...]
    factor_kind_cell: int | None
    physical_type_cells: tuple[int, ...]
    operator_cell: int | None
    rhs_kind_cell: int | None
    right_slot_cell: int | None
    value_event_cells: tuple[int, ...]
    copy_group_cells: tuple[int, ...]
    signature_role_cell: int | None
    spike_cells: tuple[int, ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class StaticCheckpointCapsule:
    artifact_sha256: str
    content_root: str
    checkpoint_sha256: str
    protocol_sha256: str
    _protocol: _StaticProtocol
    _checkpoint_metadata: Mapping[str, object]
    _excitatory: np.ndarray
    _inhibitory: np.ndarray
    _banked_view_sha256s: tuple[str, ...]
    _banked_certificate_sha256s: tuple[str, ...]

    def authenticate(self) -> None:
        if (
            self.artifact_sha256 != ARTIFACT_SHA256
            or self.content_root != ARTIFACT_CONTENT_ROOT
            or self.checkpoint_sha256 != BANKED_CHECKPOINT_SHA256
            or self.protocol_sha256 != BANKED_PROTOCOL_SHA256
            or self._protocol.protocol_sha256 != BANKED_PROTOCOL_SHA256
            or self._excitatory.shape != (OUTPUT_CELL_COUNT, INPUT_CELL_COUNT)
            or self._inhibitory.shape != self._excitatory.shape
            or self._excitatory.dtype != TICK_DTYPE
            or self._inhibitory.dtype != TICK_DTYPE
            or self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or np.any(self._excitatory > TICK_CAP)
            or np.any(self._inhibitory > TICK_CAP)
            or sha256(
                _checkpoint_payload(
                    self._checkpoint_metadata,
                    self._excitatory,
                    self._inhibitory,
                )
            ).hexdigest()
            != BANKED_CHECKPOINT_SHA256
        ):
            raise RuntimeError("static occurrence-factor checkpoint changed")


def _immutable_array(array: np.ndarray) -> np.ndarray:
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=TICK_DTYPE).reshape(
        (OUTPUT_CELL_COUNT, INPUT_CELL_COUNT)
    )
    immutable.setflags(write=False)
    return immutable


@lru_cache(maxsize=1)
def load_checkpoint_capsule(path: Path = ARTIFACT_PATH) -> StaticCheckpointCapsule:
    path = Path(path).resolve()
    raw = path.read_bytes()
    if len(raw) != ARTIFACT_BYTES or sha256(raw).hexdigest() != ARTIFACT_SHA256:
        raise RuntimeError("static occurrence-factor artifact bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _ARTIFACT_KEYS:
            raise RuntimeError("static occurrence-factor artifact members changed")
        metadata_array = archive["metadata"]
        excitatory = np.array(archive["excitatory"], copy=True, order="C")
        inhibitory = np.array(archive["inhibitory"], copy=True, order="C")
    if metadata_array.shape != () or metadata_array.dtype.kind != "U":
        raise RuntimeError("static occurrence-factor metadata framing changed")
    metadata_text = str(metadata_array.item())
    metadata = json.loads(metadata_text)
    if _canonical_json(metadata).decode("ascii") != metadata_text:
        raise RuntimeError("static occurrence-factor metadata is not canonical")
    required = {
        "array_dtype",
        "array_shape",
        "artifact_content_root",
        "banked_aggregate_view_counts",
        "banked_certificate_sha256s",
        "banked_checkpoint_sha256",
        "banked_commit",
        "banked_physical_work_sha256",
        "banked_protocol_sha256",
        "banked_source_sha256",
        "banked_summary_sha256",
        "banked_test_sha256",
        "banked_view_sha256s",
        "carrier_registry",
        "checkpoint_metadata",
        "copy_group_registry",
        "excitatory_sha256",
        "feature_registry",
        "fit_source_bundles_embedded",
        "format",
        "inhibitory_sha256",
        "input_cell_count",
        "normal_runtime_fit_count",
        "normal_runtime_imports_annotation_v2",
        "normal_runtime_imports_banked_learner",
        "normal_runtime_teacher_target_read_count",
        "occupancy_registry",
        "output_cell_count",
        "output_current_pa_hex",
        "output_steps",
        "purpose",
        "score_threshold_denominator",
        "score_threshold_numerator",
        "tick_cap",
        "type_registry",
        "value_event_registry",
    }
    if set(metadata) != required:
        raise RuntimeError("static occurrence-factor metadata keys changed")
    content_record = dict(metadata)
    content_root = str(content_record.pop("artifact_content_root"))
    if (
        content_root != ARTIFACT_CONTENT_ROOT
        or sha256(_CONTENT_NAMESPACE + _canonical_json(content_record)).hexdigest()
        != content_root
    ):
        raise RuntimeError("static occurrence-factor content root changed")
    expected_authority = {
        "format": _FORMAT,
        "purpose": "artifact-recovery-only-not-new-scientific-evidence",
        "banked_commit": BANKED_COMMIT,
        "banked_source_sha256": BANKED_SOURCE_SHA256,
        "banked_test_sha256": BANKED_TEST_SHA256,
        "banked_protocol_sha256": BANKED_PROTOCOL_SHA256,
        "banked_checkpoint_sha256": BANKED_CHECKPOINT_SHA256,
        "banked_summary_sha256": BANKED_SUMMARY_SHA256,
        "banked_physical_work_sha256": BANKED_PHYSICAL_WORK_SHA256,
        "array_shape": [OUTPUT_CELL_COUNT, INPUT_CELL_COUNT],
        "array_dtype": "|u1",
        "tick_cap": TICK_CAP,
        "output_cell_count": OUTPUT_CELL_COUNT,
        "input_cell_count": INPUT_CELL_COUNT,
        "score_threshold_numerator": SCORE_THRESHOLD_NUMERATOR,
        "score_threshold_denominator": SCORE_THRESHOLD_DENOMINATOR,
        "output_current_pa_hex": OUTPUT_CURRENT_PA.hex(),
        "output_steps": OUTPUT_STEPS,
        "banked_aggregate_view_counts": [77, 68, 164, 9, 18, 18, 18],
        "normal_runtime_imports_banked_learner": False,
        "normal_runtime_imports_annotation_v2": False,
        "normal_runtime_teacher_target_read_count": 0,
        "normal_runtime_fit_count": 0,
        "fit_source_bundles_embedded": False,
    }
    if any(metadata.get(key) != value for key, value in expected_authority.items()):
        raise RuntimeError("static occurrence-factor banked authority changed")
    if (
        excitatory.shape != (OUTPUT_CELL_COUNT, INPUT_CELL_COUNT)
        or inhibitory.shape != excitatory.shape
        or excitatory.dtype != TICK_DTYPE
        or inhibitory.dtype != TICK_DTYPE
        or not excitatory.flags.c_contiguous
        or not inhibitory.flags.c_contiguous
        or sha256(excitatory.tobytes(order="C")).hexdigest()
        != metadata["excitatory_sha256"]
        or sha256(inhibitory.tobytes(order="C")).hexdigest()
        != metadata["inhibitory_sha256"]
    ):
        raise RuntimeError("static occurrence-factor arrays changed")

    carriers = tuple(tuple(int(cell) for cell in row) for row in metadata["carrier_registry"])
    copies = tuple(tuple(int(cell) for cell in row) for row in metadata["copy_group_registry"])
    types = tuple(tuple(int(cell) for cell in row) for row in metadata["type_registry"])
    values = tuple(tuple(int(cell) for cell in row) for row in metadata["value_event_registry"])
    features = tuple(_deep_tuple(row) for row in metadata["feature_registry"])
    occupancy = tuple(tuple(int(cell) for cell in row) for row in metadata["occupancy_registry"])
    if (
        len(carriers) != 77
        or len(copies) != 18
        or len(types) != 2
        or len(values) != 13
        or len(features) != INPUT_CELL_COUNT
        or len(set(features)) != INPUT_CELL_COUNT
        or occupancy
        != ((0,), (1,), (3,), (0, 2), (1, 2), (0, 1, 2), (0, 1, 2, 3))
        or len(metadata["banked_view_sha256s"]) != 9
        or len(metadata["banked_certificate_sha256s"]) != 9
    ):
        raise RuntimeError("static occurrence-factor registries changed")
    protocol = _StaticProtocol(
        carriers,
        copies,
        types,
        values,
        features,
        occupancy,
        BANKED_PROTOCOL_SHA256,
    )
    capsule = StaticCheckpointCapsule(
        ARTIFACT_SHA256,
        content_root,
        BANKED_CHECKPOINT_SHA256,
        BANKED_PROTOCOL_SHA256,
        protocol,
        metadata["checkpoint_metadata"],
        _immutable_array(excitatory),
        _immutable_array(inhibitory),
        tuple(str(value) for value in metadata["banked_view_sha256s"]),
        tuple(str(value) for value in metadata["banked_certificate_sha256s"]),
    )
    capsule.authenticate()
    return capsule


STATIC_CHECKPOINT = load_checkpoint_capsule()
_FEATURE_INDEX = {
    feature: index
    for index, feature in enumerate(STATIC_CHECKPOINT._protocol.feature_registry)
}


def _canonical_source_bundle(
    events: Iterable[PhysicalSourceEvent],
) -> tuple[PhysicalSourceEvent, ...]:
    return tuple(
        sorted(events, key=lambda item: _canonical_json(item.neural_record()))
    )


def _source_query(
    context: tuple[PhysicalSourceEvent, ...],
    event: PhysicalSourceEvent,
) -> _Query:
    return _Query(QUERY_SOURCE, source_event=event, source_context=context)


def _active_columns(query: _Query) -> np.ndarray:
    return np.asarray(
        tuple(
            _FEATURE_INDEX[feature]
            for feature in _direct_features(query)
            if feature in _FEATURE_INDEX
        ),
        dtype=np.intp,
    )


def _one(spikes: tuple[int, ...]) -> int | None:
    return spikes[0] if len(spikes) == 1 else None


def _registered_assembly(
    spikes: tuple[int, ...],
    registry: tuple[tuple[int, ...], ...],
) -> tuple[int, ...] | None:
    if not spikes:
        return ()
    matches = tuple(item for item in registry if item == spikes)
    return matches[0] if len(matches) == 1 else None


def _decode(query: _Query, *, cell_gain: float = 1.0) -> _Decoded:
    protocol = STATIC_CHECKPOINT._protocol
    physical_features = _direct_features(query)
    active = _active_columns(query)
    raw = np.sum(
        STATIC_CHECKPOINT._excitatory[:, active].astype(np.int16)
        - STATIC_CHECKPOINT._inhibitory[:, active].astype(np.int16),
        axis=1,
        dtype=np.int64,
    )
    threshold = max(
        1,
        (
            len(physical_features) * SCORE_THRESHOLD_NUMERATOR
            + SCORE_THRESHOLD_DENOMINATOR
            - 1
        )
        // SCORE_THRESHOLD_DENOMINATOR,
    )
    currents = np.greater_equal(raw, threshold).astype(np.float64)
    currents *= OUTPUT_CURRENT_PA * cell_gain
    spike_mask = np.asarray(_SPIKE_KERNEL.accumulated_spike_mask(currents), dtype=np.bool_)
    spike_cells = tuple(int(cell) for cell in np.flatnonzero(spike_mask))

    def head(offset: int, width: int) -> tuple[int, ...]:
        return tuple(
            cell - offset for cell in spike_cells if offset <= cell < offset + width
        )

    carrier = _registered_assembly(
        head(CARRIER_OFFSET, NODE_CARRIER_CELL_COUNT),
        protocol.carrier_registry,
    )
    kind = _one(head(EVENT_KIND_OFFSET, EVENT_KIND_COUNT))
    source_kind = _one(head(SOURCE_KIND_OFFSET, 2))
    child_position = _one(head(CHILD_POSITION_OFFSET, MAX_CHILD_POSITION))
    occupancy_index = _one(head(OCCUPANCY_OFFSET, len(protocol.occupancy_registry)))
    occupancy = (
        protocol.occupancy_registry[occupancy_index]
        if occupancy_index is not None
        else ()
    )
    factor_kind = _one(head(FACTOR_KIND_OFFSET, len(FACTOR_KINDS)))
    type_cells = _registered_assembly(
        head(TYPE_OFFSET, TYPE_CELL_COUNT), protocol.type_registry
    )
    operator = _one(head(RELATION_OPERATOR_OFFSET, len(RELATION_OPERATORS)))
    rhs_kind = _one(head(RHS_KIND_OFFSET, 2))
    right_slot = _one(head(RIGHT_SLOT_OFFSET, MAX_LOCAL_SLOT))
    value_cells = _registered_assembly(
        head(VALUE_EVENT_OFFSET, VALUE_EVENT_CELL_COUNT),
        protocol.value_event_registry,
    )
    copy_cells = _registered_assembly(
        head(COPY_GROUP_OFFSET, COPY_GROUP_CELL_COUNT),
        protocol.copy_group_registry,
    )
    signature_role = _one(head(SIGNATURE_ROLE_OFFSET, 2))

    valid = (
        carrier is not None
        and type_cells is not None
        and value_cells is not None
        and copy_cells is not None
        and kind is not None
    )
    expected_source_kind = (
        0 if isinstance(query.source_event, RawPhysicalSourceEvent) else 1
    )
    if valid and query.query_kind_cell == QUERY_SOURCE:
        valid = (
            kind == EVENT_SOURCE
            and bool(carrier)
            and source_kind == expected_source_kind
            and child_position is None
            and not occupancy
            and factor_kind is None
            and not type_cells
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and not value_cells
            and not copy_cells
            and signature_role is None
        )
    elif valid and query.query_kind_cell == QUERY_EDGE:
        valid = (
            kind == EVENT_EDGE
            and not carrier
            and source_kind is None
            and child_position is not None
            and bool(occupancy)
            and factor_kind is None
            and not type_cells
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and not value_cells
            and not copy_cells
            and signature_role is None
        )
    elif valid and query.query_kind_cell == QUERY_FACTOR:
        valid = (
            kind == EVENT_FACTOR
            and not carrier
            and source_kind is None
            and child_position is None
            and not occupancy
            and factor_kind is not None
            and bool(type_cells)
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and not value_cells
            and not copy_cells
            and signature_role is None
        )
    elif valid and query.query_kind_cell == QUERY_RELATION:
        rhs_valid = (
            rhs_kind == RHS_SLOT
            and right_slot is not None
            and not value_cells
            and not type_cells
        ) or (
            rhs_kind == RHS_VALUE
            and right_slot is None
            and bool(value_cells)
            and bool(type_cells)
        )
        valid = (
            kind == EVENT_RELATION
            and not carrier
            and source_kind is None
            and child_position is None
            and not occupancy
            and factor_kind is None
            and operator is not None
            and rhs_valid
            and not copy_cells
            and signature_role is None
        )
    elif valid and query.query_kind_cell == QUERY_DOMAIN:
        valid = (
            kind == EVENT_DOMAIN
            and not carrier
            and source_kind is None
            and child_position is None
            and not occupancy
            and factor_kind is None
            and bool(type_cells)
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and bool(value_cells)
            and not copy_cells
            and signature_role is None
        )
    elif valid and query.query_kind_cell == QUERY_COPY:
        valid = (
            kind == EVENT_COPY
            and not carrier
            and source_kind is None
            and child_position is None
            and not occupancy
            and factor_kind is None
            and not type_cells
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and not value_cells
            and bool(copy_cells)
            and signature_role is None
        )
    elif valid and query.query_kind_cell == QUERY_SIGNATURE:
        valid = (
            kind == EVENT_SIGNATURE
            and not carrier
            and source_kind is None
            and child_position is None
            and not occupancy
            and factor_kind is None
            and bool(type_cells)
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and not value_cells
            and not copy_cells
            and signature_role == query.signature_role_cell
        )
    else:
        valid = False

    status = READY if valid else STOP_ZERO if not spike_cells else STOP_AMBIGUOUS
    record = {
        "status": status,
        "query": query.neural_record(),
        "carrier_cells": carrier or (),
        "event_kind_cell": kind,
        "source_kind_cell": source_kind,
        "child_position_cell": child_position,
        "occupancy_cells": occupancy,
        "factor_kind_cell": factor_kind,
        "physical_type_cells": type_cells or (),
        "operator_cell": operator,
        "rhs_kind_cell": rhs_kind,
        "right_slot_cell": right_slot,
        "value_event_cells": value_cells or (),
        "copy_group_cells": copy_cells or (),
        "signature_role_cell": signature_role,
        "spike_cells": spike_cells,
    }
    return _Decoded(
        status,
        query,
        carrier or (),
        kind,
        source_kind,
        child_position,
        occupancy,
        factor_kind,
        type_cells or (),
        operator,
        rhs_kind,
        right_slot,
        value_cells or (),
        copy_cells or (),
        signature_role,
        spike_cells,
        _digest("decoded-query", record),
    )


class SourceRelay(NamedTuple):
    carrier_cells: tuple[int, ...]
    source_event: PhysicalSourceEvent
    decision_receipt_sha256: str
    sha256: str


class EdgeRelay(NamedTuple):
    parent_carrier_cells: tuple[int, ...]
    child_carrier_cells: tuple[int, ...]
    child_position: int
    parent_factor_slot_occupancy: tuple[int, ...]
    decision_receipt_sha256: str
    sha256: str


class FactorSlotRelay(NamedTuple):
    carrier_cells: tuple[int, ...]
    local_slot: int
    factor_kind: str
    factor_kind_cell: int
    physical_type_cells: tuple[int, ...]
    decision_receipt_sha256: str
    sha256: str


class RelationRelay(NamedTuple):
    carrier_cells: tuple[int, ...]
    left_slot: int
    operator: str
    operator_cell: int
    rhs_kind_cell: int
    right_slot: int | None
    right_value_event_cells: tuple[int, ...]
    right_value_type_cells: tuple[int, ...]
    decision_receipt_sha256: str
    sha256: str


class DomainRelay(NamedTuple):
    carrier_cells: tuple[int, ...]
    local_slot: int
    physical_type_cells: tuple[int, ...]
    complete_allowed_value_event_cells: tuple[tuple[int, ...], ...]
    decision_receipt_sha256: str
    sha256: str


class LiteralCopyGroupRelay(NamedTuple):
    copy_group_cells: tuple[int, ...]
    members: tuple[tuple[tuple[int, ...], int], ...]
    decision_receipt_sha256s: tuple[str, ...]
    sha256: str


class SignatureRelay(NamedTuple):
    root_carrier_cells: tuple[int, ...]
    signature_role: str
    signature_role_cell: int
    signature_position: int
    physical_type_cells: tuple[int, ...]
    decision_receipt_sha256: str
    sha256: str


class OccurrenceFactorRelayView(NamedTuple):
    root_carrier_cells: tuple[int, ...]
    occurrence_carriers: tuple[tuple[int, ...], ...]
    source_relays: tuple[SourceRelay, ...]
    edges: tuple[EdgeRelay, ...]
    factor_slots: tuple[FactorSlotRelay, ...]
    relations: tuple[RelationRelay, ...]
    domains: tuple[DomainRelay, ...]
    literal_copy_groups: tuple[LiteralCopyGroupRelay, ...]
    signature_relays: tuple[SignatureRelay, ...]
    decision_receipt_sha256s: tuple[str, ...]
    checkpoint_sha256: str
    protocol_sha256: str
    physical_work_receipt_sha256: str
    sha256: str


def _generic_queries(
    carriers: tuple[tuple[int, ...], ...],
) -> tuple[_Query, ...]:
    queries: list[_Query] = []
    queries.extend(
        _Query(
            QUERY_EDGE,
            primary_carrier_cells=parent,
            secondary_carrier_cells=child,
        )
        for parent in carriers
        for child in carriers
        if parent != child
    )
    for query_kind in (QUERY_FACTOR, QUERY_RELATION, QUERY_DOMAIN, QUERY_COPY):
        queries.extend(
            _Query(
                query_kind,
                primary_carrier_cells=carrier,
                local_slot=local_slot,
            )
            for carrier in carriers
            for local_slot in range(MAX_LOCAL_SLOT)
        )
    queries.extend(
        _Query(
            QUERY_SIGNATURE,
            primary_carrier_cells=carrier,
            signature_role_cell=role,
            signature_position_cell=position,
        )
        for carrier in carriers
        for role in (SIGNATURE_PARAMETER, SIGNATURE_RETURN)
        for position in SIGNATURE_POSITIONS
    )
    return tuple(sorted(queries, key=lambda item: _canonical_json(item.neural_record())))


def _assemble_view(
    protocol: _StaticProtocol,
    *,
    request_sha256: str,
    checkpoint_sha256: str,
    decisions: tuple[_Decoded, ...],
    physical_query_count: int,
) -> tuple[str, str | None, OccurrenceFactorRelayView | None]:
    ready = tuple(row for row in decisions if row.status == READY)
    by_kind: dict[int, list[_Decoded]] = defaultdict(list)
    for row in ready:
        by_kind[row.query.query_kind_cell].append(row)

    source_by_carrier: dict[tuple[int, ...], SourceRelay] = {}
    for row in by_kind[QUERY_SOURCE]:
        assert row.query.source_event is not None
        record = (
            row.carrier_cells,
            row.query.source_event.neural_record(),
            row.receipt_sha256,
        )
        relay = SourceRelay(
            row.carrier_cells,
            row.query.source_event,
            row.receipt_sha256,
            _digest("source-relay", record),
        )
        if not row.carrier_cells or row.carrier_cells in source_by_carrier:
            return STOP_AMBIGUOUS, "one source acquired a reused carrier", None
        source_by_carrier[row.carrier_cells] = relay
    carriers = set(source_by_carrier)
    if len(carriers) not in (8, 9):
        return STOP_UNSUPPORTED, "source bundle is not a retained 8/9-node FIT shape", None

    edge_relays: list[EdgeRelay] = []
    inbound: dict[tuple[int, ...], int] = defaultdict(int)
    children: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = defaultdict(
        list
    )
    edge_keys: set[tuple[tuple[int, ...], int]] = set()
    for row in by_kind[QUERY_EDGE]:
        parent = row.query.primary_carrier_cells
        child = row.query.secondary_carrier_cells
        assert row.child_position_cell is not None
        key = (parent, row.child_position_cell)
        if (
            parent not in carriers
            or child not in carriers
            or parent == child
            or key in edge_keys
        ):
            return STOP_AMBIGUOUS, "learned edge carrier/position is not unique", None
        edge_keys.add(key)
        inbound[child] += 1
        children[parent].append((row.child_position_cell, child))
        record = (
            parent,
            child,
            row.child_position_cell,
            row.occupancy_cells,
            row.receipt_sha256,
        )
        edge_relays.append(
            EdgeRelay(
                parent,
                child,
                row.child_position_cell,
                row.occupancy_cells,
                row.receipt_sha256,
                _digest("edge-relay", record),
            )
        )
    roots = tuple(carrier for carrier in carriers if inbound[carrier] == 0)
    if (
        len(edge_relays) != len(carriers) - 1
        or len(roots) != 1
        or any(inbound[carrier] != 1 for carrier in carriers if carrier not in roots)
    ):
        return STOP_AMBIGUOUS, "learned edges do not form one rooted tree", None
    root = roots[0]
    for entries in children.values():
        if tuple(position for position, _child in sorted(entries)) != tuple(
            range(len(entries))
        ):
            return STOP_AMBIGUOUS, "learned child positions are not contiguous", None

    ordered_carriers: list[tuple[int, ...]] = []

    def visit(carrier: tuple[int, ...]) -> None:
        if carrier in ordered_carriers:
            raise ValueError("cycle")
        ordered_carriers.append(carrier)
        for _position, child in sorted(children.get(carrier, ())):
            visit(child)

    try:
        visit(root)
    except ValueError:
        return STOP_AMBIGUOUS, "learned carrier tree contains a cycle", None
    if set(ordered_carriers) != carriers:
        return STOP_AMBIGUOUS, "learned carrier tree is disconnected", None

    factor_relays: list[FactorSlotRelay] = []
    factors_by_carrier: dict[tuple[int, ...], list[FactorSlotRelay]] = defaultdict(
        list
    )
    for row in by_kind[QUERY_FACTOR]:
        carrier = row.query.primary_carrier_cells
        local_slot = row.query.local_slot
        assert local_slot is not None
        assert row.factor_kind_cell is not None
        if carrier not in carriers:
            return STOP_AMBIGUOUS, "factor slot lacks a source-bound carrier", None
        record = (
            carrier,
            local_slot,
            row.factor_kind_cell,
            row.physical_type_cells,
            row.receipt_sha256,
        )
        relay = FactorSlotRelay(
            carrier,
            local_slot,
            FACTOR_KINDS[row.factor_kind_cell],
            row.factor_kind_cell,
            row.physical_type_cells,
            row.receipt_sha256,
            _digest("factor-slot-relay", record),
        )
        factors_by_carrier[carrier].append(relay)
        factor_relays.append(relay)
    for carrier in ordered_carriers:
        rows = sorted(
            factors_by_carrier.get(carrier, ()), key=lambda item: item.local_slot
        )
        if tuple(row.local_slot for row in rows) != tuple(range(len(rows))) or not rows:
            return STOP_AMBIGUOUS, "factor slots are not one complete ordered tuple", None
    for edge in edge_relays:
        parent_factors = factors_by_carrier[edge.parent_carrier_cells]
        child_factors = factors_by_carrier[edge.child_carrier_cells]
        if (
            len(edge.parent_factor_slot_occupancy) != len(child_factors)
            or len(set(edge.parent_factor_slot_occupancy))
            != len(edge.parent_factor_slot_occupancy)
            or any(
                not 0 <= slot < len(parent_factors)
                for slot in edge.parent_factor_slot_occupancy
            )
        ):
            return STOP_AMBIGUOUS, "edge occupancy does not bind child factors", None

    relation_relays: list[RelationRelay] = []
    relation_keys: set[tuple[tuple[int, ...], int]] = set()
    for row in by_kind[QUERY_RELATION]:
        carrier = row.query.primary_carrier_cells
        left_slot = row.query.local_slot
        assert left_slot is not None
        assert row.operator_cell is not None
        assert row.rhs_kind_cell is not None
        key = (carrier, left_slot)
        local_factors = factors_by_carrier.get(carrier, ())
        rhs_valid = (
            row.rhs_kind_cell == RHS_SLOT
            and row.right_slot_cell is not None
            and 0 <= row.right_slot_cell < len(local_factors)
            and not row.value_event_cells
            and not row.physical_type_cells
        ) or (
            row.rhs_kind_cell == RHS_VALUE
            and row.right_slot_cell is None
            and bool(row.value_event_cells)
            and bool(row.physical_type_cells)
        )
        if (
            key in relation_keys
            or not 0 <= left_slot < len(local_factors)
            or not rhs_valid
        ):
            return STOP_AMBIGUOUS, "relation is not bound to valid factor slots", None
        relation_keys.add(key)
        record = (
            carrier,
            left_slot,
            row.operator_cell,
            row.rhs_kind_cell,
            row.right_slot_cell,
            row.value_event_cells,
            row.physical_type_cells,
            row.receipt_sha256,
        )
        relation_relays.append(
            RelationRelay(
                carrier,
                left_slot,
                RELATION_OPERATORS[row.operator_cell],
                row.operator_cell,
                row.rhs_kind_cell,
                row.right_slot_cell,
                row.value_event_cells,
                row.physical_type_cells,
                row.receipt_sha256,
                _digest("relation-relay", record),
            )
        )

    domain_relays: list[DomainRelay] = []
    domain_keys: set[tuple[tuple[int, ...], int]] = set()
    for row in by_kind[QUERY_DOMAIN]:
        carrier = row.query.primary_carrier_cells
        local_slot = row.query.local_slot
        assert local_slot is not None
        key = (carrier, local_slot)
        local_factors = factors_by_carrier.get(carrier, ())
        if (
            key in domain_keys
            or not 0 <= local_slot < len(local_factors)
            or local_factors[local_slot].factor_kind_cell != 0
            or not row.physical_type_cells
            or not row.value_event_cells
        ):
            return STOP_AMBIGUOUS, "literal domain is not bound to a literal slot", None
        domain_keys.add(key)
        record = (
            carrier,
            local_slot,
            row.physical_type_cells,
            row.value_event_cells,
            row.receipt_sha256,
        )
        domain_relays.append(
            DomainRelay(
                carrier,
                local_slot,
                row.physical_type_cells,
                (row.value_event_cells,),
                row.receipt_sha256,
                _digest("domain-relay", record),
            )
        )

    copy_members: dict[
        tuple[int, ...], list[tuple[tuple[int, ...], int, str]]
    ] = defaultdict(list)
    for row in by_kind[QUERY_COPY]:
        assert row.query.local_slot is not None
        copy_members[row.copy_group_cells].append(
            (
                row.query.primary_carrier_cells,
                row.query.local_slot,
                row.receipt_sha256,
            )
        )
    terminal_carriers = {carrier for carrier in carriers if not children.get(carrier)}
    terminal_literal_refs = {
        (carrier, factor.local_slot)
        for carrier in terminal_carriers
        for factor in factors_by_carrier[carrier]
        if factor.factor_kind_cell == 0
    }
    seen_copy_refs: set[tuple[tuple[int, ...], int]] = set()
    copy_relays: list[LiteralCopyGroupRelay] = []
    for group, rows in sorted(copy_members.items()):
        members = tuple(sorted((carrier, slot) for carrier, slot, _receipt in rows))
        receipts = tuple(sorted(receipt for _carrier, _slot, receipt in rows))
        if not group or seen_copy_refs & set(members):
            return STOP_AMBIGUOUS, "literal-copy member was assigned twice", None
        seen_copy_refs.update(members)
        record = (group, members, receipts)
        copy_relays.append(
            LiteralCopyGroupRelay(
                group,
                members,
                receipts,
                _digest("literal-copy-group-relay", record),
            )
        )
    if seen_copy_refs != terminal_literal_refs:
        return STOP_UNSUPPORTED, "literal-copy groups do not cover terminal literals", None

    signature_relays: list[SignatureRelay] = []
    signature_keys: set[tuple[int, int]] = set()
    for row in by_kind[QUERY_SIGNATURE]:
        role = row.query.signature_role_cell
        position = row.query.signature_position_cell
        assert role is not None and position is not None
        key = (role, position)
        if (
            row.query.primary_carrier_cells != root
            or key in signature_keys
            or not row.physical_type_cells
        ):
            return STOP_AMBIGUOUS, "signature relay is not uniquely root-keyed", None
        signature_keys.add(key)
        role_name = "parameter" if role == SIGNATURE_PARAMETER else "return"
        record = (
            root,
            role,
            position,
            row.physical_type_cells,
            row.receipt_sha256,
        )
        signature_relays.append(
            SignatureRelay(
                root,
                role_name,
                role,
                position,
                row.physical_type_cells,
                row.receipt_sha256,
                _digest("signature-relay", record),
            )
        )
    expected_signature = {
        (SIGNATURE_PARAMETER, 0, protocol.type_registry[1]),
        (SIGNATURE_RETURN, 0, protocol.type_registry[0]),
    }
    if {
        (row.signature_role_cell, row.signature_position, row.physical_type_cells)
        for row in signature_relays
    } != expected_signature:
        return STOP_UNSUPPORTED, "minimal List[Int]-to-Int signature is incomplete", None

    expected_factor_count = 16 if len(carriers) == 8 else 20
    if (
        len(factor_relays) != expected_factor_count
        or len(relation_relays) != 1
        or len(domain_relays) != 2
        or len(copy_relays) != 2
        or len(signature_relays) != 2
    ):
        return STOP_UNSUPPORTED, "retained FIT occurrence-factor shape is incomplete", None

    ordered_sources = tuple(source_by_carrier[carrier] for carrier in ordered_carriers)
    work_receipt = _digest(
        "physical-work",
        {
            "request_sha256": request_sha256,
            "physical_query_count": physical_query_count,
            "output_cell_count": OUTPUT_CELL_COUNT,
            "fitted_glif_step_count": physical_query_count * OUTPUT_STEPS,
        },
    )
    view_record = {
        "root_carrier_cells": root,
        "occurrence_carriers": tuple(ordered_carriers),
        "source_relays": tuple(row.sha256 for row in ordered_sources),
        "edge_relays": tuple(row.sha256 for row in sorted(edge_relays)),
        "factor_relays": tuple(row.sha256 for row in sorted(factor_relays)),
        "relation_relays": tuple(row.sha256 for row in sorted(relation_relays)),
        "domain_relays": tuple(row.sha256 for row in sorted(domain_relays)),
        "copy_relays": tuple(row.sha256 for row in sorted(copy_relays)),
        "signature_relays": tuple(row.sha256 for row in sorted(signature_relays)),
        "decision_receipt_sha256s": tuple(
            sorted(row.receipt_sha256 for row in ready)
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol.protocol_sha256,
        "physical_work_receipt_sha256": work_receipt,
    }
    view = OccurrenceFactorRelayView(
        root,
        tuple(ordered_carriers),
        ordered_sources,
        tuple(sorted(edge_relays)),
        tuple(sorted(factor_relays)),
        tuple(sorted(relation_relays)),
        tuple(sorted(domain_relays)),
        tuple(sorted(copy_relays)),
        tuple(sorted(signature_relays)),
        tuple(sorted(row.receipt_sha256 for row in ready)),
        checkpoint_sha256,
        protocol.protocol_sha256,
        work_receipt,
        _digest("occurrence-factor-view", view_record),
    )
    return READY, None, view


def _infer_static(
    events: Iterable[PhysicalSourceEvent],
) -> tuple[str, str | None, OccurrenceFactorRelayView | None]:
    supplied = tuple(events)
    if any(
        type(item) not in (RawPhysicalSourceEvent, LiteralPhysicalSourceEvent)
        for item in supplied
    ):
        raise TypeError("static inference accepts only neutral physical source events")
    records = tuple(_canonical_json(item.neural_record()) for item in supplied)
    ordered = _canonical_source_bundle(supplied)
    request_sha = _digest(
        "inference-request",
        tuple(item.neural_record() for item in ordered),
    )
    if not ordered:
        return STOP_ZERO, "no physical source event was supplied", None
    if len(records) != len(set(records)):
        return STOP_AMBIGUOUS, "a physical source event was duplicated", None

    source_decisions = tuple(
        _decode(_source_query(ordered, event)) for event in ordered
    )
    if any(row.status == STOP_AMBIGUOUS for row in source_decisions):
        return (
            STOP_AMBIGUOUS,
            "one physical source activated conflicting carrier fields",
            None,
        )
    if any(row.status != READY for row in source_decisions):
        return (
            STOP_ZERO,
            "one physical source had no learned contextual carrier",
            None,
        )
    carriers = tuple(sorted(row.carrier_cells for row in source_decisions))
    if len(carriers) != len(set(carriers)):
        return (
            STOP_AMBIGUOUS,
            "multiple physical sources activated one carrier",
            None,
        )

    generic_queries = _generic_queries(carriers)
    generic_decisions = tuple(_decode(query) for query in generic_queries)
    physical_query_count = len(source_decisions) + len(generic_decisions)
    if any(row.status == STOP_AMBIGUOUS for row in generic_decisions):
        return (
            STOP_AMBIGUOUS,
            "one recurrent carrier query activated conflicting relay fields",
            None,
        )
    return _assemble_view(
        STATIC_CHECKPOINT._protocol,
        request_sha256=request_sha,
        checkpoint_sha256=STATIC_CHECKPOINT.checkpoint_sha256,
        decisions=(*source_decisions, *generic_decisions),
        physical_query_count=physical_query_count,
    )


class _CertificateReceipt(NamedTuple):
    authority: object
    consumer: StaticCheckpointCapsule
    status: str
    stop_reason: str | None
    request_sha256: str
    view: OccurrenceFactorRelayView | None
    sha256: str


_CERTIFICATE_AUTHORITY = object()
_ISSUED_CERTIFICATES: dict[int, OccurrenceFactorStaticCertificate] = {}


class OccurrenceFactorStaticCertificate:
    """Opaque exact-object authority over one static physical inference."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _CertificateReceipt) -> None:
        if receipt.authority is not _CERTIFICATE_AUTHORITY:
            raise TypeError("occurrence-factor certificate requires private authority")
        self._receipt = receipt

    @property
    def status(self) -> str:
        return self._receipt.status

    @property
    def stop_reason(self) -> str | None:
        return self._receipt.stop_reason

    @property
    def checkpoint_sha256(self) -> str:
        return self._receipt.consumer.checkpoint_sha256

    @property
    def sha256(self) -> str:
        return self._receipt.sha256


def _issue_certificate(
    events: tuple[PhysicalSourceEvent, ...],
    *,
    status: str,
    stop_reason: str | None,
    view: OccurrenceFactorRelayView | None,
) -> OccurrenceFactorStaticCertificate:
    ordered = _canonical_source_bundle(events)
    request_sha = _digest(
        "inference-request",
        tuple(item.neural_record() for item in ordered),
    )
    payload = {
        "status": status,
        "stop_reason": stop_reason,
        "request_sha256": request_sha,
        "view_sha256": view.sha256 if view is not None else None,
        "checkpoint_sha256": STATIC_CHECKPOINT.checkpoint_sha256,
        "protocol_sha256": STATIC_CHECKPOINT.protocol_sha256,
    }
    receipt = _CertificateReceipt(
        _CERTIFICATE_AUTHORITY,
        STATIC_CHECKPOINT,
        status,
        stop_reason,
        request_sha,
        view,
        _digest("opaque-certificate", payload),
    )
    certificate = OccurrenceFactorStaticCertificate(receipt)
    _ISSUED_CERTIFICATES[id(certificate)] = certificate
    return certificate


def _authenticate_certificate(
    certificate: OccurrenceFactorStaticCertificate,
) -> _CertificateReceipt:
    if (
        type(certificate) is not OccurrenceFactorStaticCertificate
        or _ISSUED_CERTIFICATES.get(id(certificate)) is not certificate
        or certificate._receipt.authority is not _CERTIFICATE_AUTHORITY
    ):
        raise TypeError("occurrence-factor certificate is not an issued exact object")
    receipt = certificate._receipt
    payload = {
        "status": receipt.status,
        "stop_reason": receipt.stop_reason,
        "request_sha256": receipt.request_sha256,
        "view_sha256": receipt.view.sha256 if receipt.view is not None else None,
        "checkpoint_sha256": receipt.consumer.checkpoint_sha256,
        "protocol_sha256": receipt.consumer.protocol_sha256,
    }
    if receipt.sha256 != _digest("opaque-certificate", payload):
        raise RuntimeError("occurrence-factor certificate receipt changed")
    return receipt


def infer_occurrence_factors(
    events: Iterable[PhysicalSourceEvent],
) -> OccurrenceFactorStaticCertificate:
    """Infer same-FIT physical relays from one unordered neutral event bundle."""

    STATIC_CHECKPOINT.authenticate()
    supplied = tuple(events)
    status, reason, view = _infer_static(supplied)
    STATIC_CHECKPOINT.authenticate()
    return _issue_certificate(
        supplied,
        status=status,
        stop_reason=reason,
        view=view,
    )


def require_occurrence_factors(
    certificate: OccurrenceFactorStaticCertificate,
) -> OccurrenceFactorRelayView:
    """Authenticate an issued READY certificate and expose physical relays."""

    receipt = _authenticate_certificate(certificate)
    receipt.consumer.authenticate()
    if receipt.status != READY or receipt.view is None:
        raise RuntimeError(receipt.stop_reason or receipt.status)
    return receipt.view


CLAIM_BOUNDARY = (
    "target-free artifact recovery and public-TRAIN same-FIT unordered-bundle "
    "contextual source replay plus recurrent physical occurrence/factor "
    "composition only; local-cue generalization, unseen prompt transfer, "
    "canonical joint selection, DEV, protected, M0, C1, Path-2, G0, coding "
    "benchmarks, and broad capability remain unproven"
)


__all__ = [
    "ARTIFACT_BYTES",
    "ARTIFACT_CONTENT_ROOT",
    "ARTIFACT_PATH",
    "ARTIFACT_SHA256",
    "BANKED_CHECKPOINT_SHA256",
    "BANKED_COMMIT",
    "BANKED_PHYSICAL_WORK_SHA256",
    "BANKED_PROTOCOL_SHA256",
    "BANKED_SOURCE_SHA256",
    "BANKED_SUMMARY_SHA256",
    "BANKED_TEST_SHA256",
    "CLAIM_BOUNDARY",
    "READY",
    "STATIC_CHECKPOINT",
    "STOP_AMBIGUOUS",
    "STOP_UNSUPPORTED",
    "STOP_ZERO",
    "LiteralPhysicalSourceEvent",
    "OccurrenceFactorRelayView",
    "OccurrenceFactorStaticCertificate",
    "RawPhysicalSourceEvent",
    "infer_occurrence_factors",
    "load_checkpoint_capsule",
    "require_occurrence_factors",
]
