"""Fresh public-TRAIN FIT learner for occurrence-local structural relays.

The independent V2 compiler capsule supplies offline teacher targets.  Runtime
inference receives only unordered physical raw-span or typed-literal source
events.  A first fitted-GLIF lookup learns a stable distributed occurrence
carrier for each source event.  Those learned carriers are then recurrently
reused in generic carrier-pair and carrier-slot queries for edges, factor
slots, relations, domains, literal-copy groups, and signature types.

This is deliberately a narrow same-cohort FIT mechanism result.  It opens no
heldout, DEV, protected, M0, C1, Path-2, or G0 surface and makes no coding or
transfer claim.  Compiler labels and every teacher/audit hash remain outside
the physical query records.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_annotation_capsule_v2 as annotation_v2,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCHEMA = "PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1"
READY = "PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_READY"
STOP_ZERO = "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_ZERO"
STOP_AMBIGUOUS = "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_AMBIGUOUS"
STOP_UNSUPPORTED = "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_UNSUPPORTED"
STOP_DEPENDENCY = "STOP_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_LEARNER_V1_DEPENDENCY"

V2_BANK_COMMIT = "56b9c73cc8515bd9b5377b26effe758d75857750"
V2_SOURCE_PATH = (
    "experiments/r3_pilot/"
    "public_train_fit_occurrence_factor_annotation_capsule_v2.py"
)
V2_SOURCE_SHA256 = (
    "f41f5b904edbe6aa4ecf29f45117eb86aa7dfd135a11c721199ecb0a6542f92c"
)
V2_CAPSULE_SHA256 = (
    "75d9fcd9abcfeba051fe1c0caade3fabdcbbb140b956350d3e2dd5d82ec0fc9f"
)
V2_TARGET_SET_SHA256 = (
    "be3c88adc730d58ece81cddab97f90484f346194f2621ec8dbe18c3c0475d02b"
)
V2_DEPENDENCY_SHA256 = (
    "b39497d751558a8459982a15cc87cd4ed1c4719a3bdd39adcff58ee0d1e76940"
)
V2_TYPE_VOCABULARY_SHA256 = (
    "99afc8040d2430afb70c68ec2213918e3f524b40302f8a17c2feae8798314553"
)
V2_SURFACE_BOUNDARY_SHA256 = (
    "886b16e151442c7753f0bd423b1ba454b3d035cbc56fa6d3b46d1d8e64cae502"
)

FIT_IDENTITY_COUNT = 9
FIT_APPEARANCE_COUNT = 12
FIT_LESSON_COUNT = 372
FIT_PASS_COUNT = 1

HELDOUT_INPUT_COUNT = 0
HELDOUT_TARGET_READ_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_INPUT_COUNT = 0
PROGRAM_INPUT_COUNT = 0
TASK_INPUT_COUNT = 0
FAMILY_INPUT_COUNT = 0
NAME_INPUT_COUNT = 0
SOURCE_ORDER_INPUT_COUNT = 0
FIXED_INDEX_INPUT_COUNT = 0
AUDIT_HASH_INPUT_COUNT = 0
V1_PARTIAL_TARGET_READ_COUNT = 0
V4_CHECKPOINT_READ_COUNT = 0
V5_CHECKPOINT_READ_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0

QUERY_SOURCE = 0
QUERY_EDGE = 1
QUERY_FACTOR = 2
QUERY_RELATION = 3
QUERY_DOMAIN = 4
QUERY_COPY = 5
QUERY_SIGNATURE = 6
QUERY_KIND_COUNT = 7

EVENT_SOURCE = annotation_v2.EVENT_SOURCE
EVENT_EDGE = annotation_v2.EVENT_EDGE
EVENT_FACTOR = annotation_v2.EVENT_FACTOR
EVENT_RELATION = annotation_v2.EVENT_RELATION
EVENT_DOMAIN = annotation_v2.EVENT_DOMAIN
EVENT_COPY = annotation_v2.EVENT_COPY
EVENT_SIGNATURE = annotation_v2.EVENT_SIGNATURE
EVENT_KIND_COUNT = 8

FACTOR_KINDS = annotation_v2.FACTOR_KINDS
RELATION_OPERATORS = annotation_v2.RELATION_OPERATORS
RHS_SLOT = annotation_v2.RHS_SLOT
RHS_VALUE = annotation_v2.RHS_VALUE
SIGNATURE_PARAMETER = 0
SIGNATURE_RETURN = 1

MAX_LOCAL_SLOT = 4
MAX_CHILD_POSITION = 2
SIGNATURE_POSITIONS = (0,)
OCCUPANCY_REGISTRY = (
    (0,),
    (1,),
    (3,),
    (0, 2),
    (1, 2),
    (0, 1, 2),
    (0, 1, 2, 3),
)

NODE_CARRIER_CELL_COUNT = annotation_v2.NODE_CARRIER_CELL_COUNT
COPY_GROUP_CELL_COUNT = annotation_v2.COPY_GROUP_CELL_COUNT
TYPE_CELL_COUNT = annotation_v2.TYPE_CELL_COUNT
VALUE_EVENT_CELL_COUNT = 96

CARRIER_OFFSET = 0
EVENT_KIND_OFFSET = CARRIER_OFFSET + NODE_CARRIER_CELL_COUNT
SOURCE_KIND_OFFSET = EVENT_KIND_OFFSET + EVENT_KIND_COUNT
CHILD_POSITION_OFFSET = SOURCE_KIND_OFFSET + 2
OCCUPANCY_OFFSET = CHILD_POSITION_OFFSET + MAX_CHILD_POSITION
FACTOR_KIND_OFFSET = OCCUPANCY_OFFSET + len(OCCUPANCY_REGISTRY)
TYPE_OFFSET = FACTOR_KIND_OFFSET + len(FACTOR_KINDS)
RELATION_OPERATOR_OFFSET = TYPE_OFFSET + TYPE_CELL_COUNT
RHS_KIND_OFFSET = RELATION_OPERATOR_OFFSET + len(RELATION_OPERATORS)
RIGHT_SLOT_OFFSET = RHS_KIND_OFFSET + 2
VALUE_EVENT_OFFSET = RIGHT_SLOT_OFFSET + MAX_LOCAL_SLOT
COPY_GROUP_OFFSET = VALUE_EVENT_OFFSET + VALUE_EVENT_CELL_COUNT
SIGNATURE_ROLE_OFFSET = COPY_GROUP_OFFSET + COPY_GROUP_CELL_COUNT
OUTPUT_CELL_COUNT = SIGNATURE_ROLE_OFFSET + 2

TICK_CAP = 4
SCORE_THRESHOLD_NUMERATOR = 3
SCORE_THRESHOLD_DENOMINATOR = 4
OUTPUT_CURRENT_PA = 5_000.0
OUTPUT_STEPS = 30
TICK_DTYPE = np.dtype(np.uint8)

_ROOT = Path(__file__).resolve().parents[2]
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


def _file_sha256(relative_path: str) -> str:
    return sha256((_ROOT / relative_path).read_bytes()).hexdigest()


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
            count=annotation_v2.RAW_SPAN_ACTIVE_CELL_COUNT,
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
            count=annotation_v2.TYPE_ACTIVE_CELL_COUNT,
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
    """Return direct physical conjunction cells, never a hash projection."""

    if query.query_kind_cell == QUERY_SOURCE:
        if query.source_event is None or not query.source_context:
            return ()
        selected = _source_event_feature(query.source_event)
        features = (
            (
                QUERY_SOURCE,
                selected,
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
class _Target:
    event_kind_cell: int
    carrier_cells: tuple[int, ...] = ()
    source_kind_cell: int | None = None
    child_position_cell: int | None = None
    occupancy_cells: tuple[int, ...] = ()
    factor_kind_cell: int | None = None
    physical_type_cells: tuple[int, ...] = ()
    operator_cell: int | None = None
    rhs_kind_cell: int | None = None
    right_slot_cell: int | None = None
    value_event_cells: tuple[int, ...] = ()
    copy_group_cells: tuple[int, ...] = ()
    signature_role_cell: int | None = None

    def neural_record(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value not in (None, ())
        }


@dataclass(frozen=True, slots=True)
class _Lesson:
    query: _Query
    target: _Target

    def root_record(self) -> dict[str, object]:
        return {
            "query": self.query.neural_record(),
            "target": self.target.neural_record(),
        }


@dataclass(frozen=True, slots=True)
class _FrozenProtocol:
    dependency_sha256: str
    carrier_registry: tuple[tuple[int, ...], ...]
    copy_group_registry: tuple[tuple[int, ...], ...]
    type_registry: tuple[tuple[int, ...], ...]
    value_event_registry: tuple[tuple[int, ...], ...]
    feature_registry: tuple[PhysicalFeature, ...]
    source_bundles: tuple[tuple[PhysicalSourceEvent, ...], ...]
    lessons: tuple[_Lesson, ...]
    source_lesson_count: int
    edge_lesson_count: int
    factor_lesson_count: int
    relation_lesson_count: int
    domain_lesson_count: int
    copy_lesson_count: int
    signature_lesson_count: int
    query_target_conflict_count: int
    trace_collision_count: int
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


def _physical_source_event(source: annotation_v2.SourceTarget) -> PhysicalSourceEvent:
    if isinstance(source, annotation_v2.RawSourceTarget):
        observation = source.observation
        return RawPhysicalSourceEvent(
            observation.assembly_cells,
            observation.direction,
            observation.cue_slot,
            observation.selected_slot,
        )
    if isinstance(source, annotation_v2.LiteralSourceTarget):
        return LiteralPhysicalSourceEvent(
            source.typed_value_event_cells,
            source.typed_value_type_cells,
        )
    raise TypeError("V2 source target type changed")


def _canonical_source_bundle(
    events: Iterable[PhysicalSourceEvent],
) -> tuple[PhysicalSourceEvent, ...]:
    supplied = tuple(events)
    return tuple(sorted(supplied, key=lambda item: _canonical_json(item.neural_record())))


def _source_query(
    context: tuple[PhysicalSourceEvent, ...],
    event: PhysicalSourceEvent,
) -> _Query:
    return _Query(
        QUERY_SOURCE,
        source_event=event,
        source_context=context,
    )


def _verify_dependency() -> tuple[
    str,
    tuple[annotation_v2.FitOccurrenceFactorTarget, ...],
]:
    capsule = annotation_v2.CAPSULE
    checks = (
        _file_sha256(V2_SOURCE_PATH) == V2_SOURCE_SHA256,
        capsule.status == annotation_v2.READY,
        capsule.capsule_sha256 == V2_CAPSULE_SHA256,
        capsule.target_set_sha256 == V2_TARGET_SET_SHA256,
        capsule.dependency_sha256 == V2_DEPENDENCY_SHA256,
        capsule.type_vocabulary_sha256 == V2_TYPE_VOCABULARY_SHA256,
        capsule.surface_boundary_sha256 == V2_SURFACE_BOUNDARY_SHA256,
        capsule.counts.fit_identity_count == FIT_IDENTITY_COUNT,
        capsule.counts.fit_appearance_count == FIT_APPEARANCE_COUNT,
        capsule.counts.emitted_target_record_count == FIT_LESSON_COUNT,
        annotation_v2.HELDOUT_TARGET_READ_COUNT == 0,
        annotation_v2.HELDOUT_INPUT_LOOKUP_COUNT == 0,
        annotation_v2.LEARNER_ARRAY_COUNT == 0,
        annotation_v2.LEARNING_RUN_COUNT == 0,
    )
    if not all(checks):
        raise RuntimeError(STOP_DEPENDENCY)
    targets = annotation_v2.require_ready_annotation_capsule_v2(capsule)
    dependency_record = {
        "v2_bank_commit": V2_BANK_COMMIT,
        "v2_source_sha256": V2_SOURCE_SHA256,
        "v2_capsule_sha256": V2_CAPSULE_SHA256,
        "v2_target_set_sha256": V2_TARGET_SET_SHA256,
        "v2_dependency_sha256": V2_DEPENDENCY_SHA256,
        "v2_type_vocabulary_sha256": V2_TYPE_VOCABULARY_SHA256,
        "v2_surface_boundary_sha256": V2_SURFACE_BOUNDARY_SHA256,
    }
    return _digest("dependency", dependency_record), targets


def _build_protocol() -> _FrozenProtocol:
    dependency_sha, targets = _verify_dependency()
    lessons: list[_Lesson] = []
    source_bundles: list[tuple[PhysicalSourceEvent, ...]] = []
    carrier_registry: set[tuple[int, ...]] = set()
    copy_registry: set[tuple[int, ...]] = set()
    value_registry: set[tuple[int, ...]] = set()

    for identity_target in targets:
        source_pairs = tuple(
            (_physical_source_event(source), source)
            for source in identity_target.sources
        )
        context = _canonical_source_bundle(event for event, _source in source_pairs)
        if len(context) != len(set(context)):
            raise RuntimeError(
                "one V2 identity contains duplicate stripped physical source events"
            )
        source_bundles.append(context)
        for event, source in source_pairs:
            carrier_registry.add(source.node_occurrence_cells)
            if isinstance(event, LiteralPhysicalSourceEvent):
                value_registry.add(event.typed_value_event_cells)
                source_kind = 1
            else:
                source_kind = 0
            lessons.append(
                _Lesson(
                    _source_query(context, event),
                    _Target(
                        EVENT_SOURCE,
                        carrier_cells=source.node_occurrence_cells,
                        source_kind_cell=source_kind,
                    ),
                )
            )

        for edge in identity_target.edges:
            lessons.append(
                _Lesson(
                    _Query(
                        QUERY_EDGE,
                        primary_carrier_cells=edge.parent_occurrence_cells,
                        secondary_carrier_cells=edge.child_occurrence_cells,
                    ),
                    _Target(
                        EVENT_EDGE,
                        child_position_cell=edge.child_position_cell,
                        occupancy_cells=edge.parent_factor_slot_occupancy_cells,
                    ),
                )
            )
        for factor in identity_target.factor_slots:
            lessons.append(
                _Lesson(
                    _Query(
                        QUERY_FACTOR,
                        primary_carrier_cells=factor.node_occurrence_cells,
                        local_slot=factor.factor_slot_cell,
                    ),
                    _Target(
                        EVENT_FACTOR,
                        factor_kind_cell=factor.factor_kind_cell,
                        physical_type_cells=factor.physical_type_cells,
                    ),
                )
            )
        for relation in identity_target.relations:
            if relation.right_typed_value_event_cells:
                value_registry.add(relation.right_typed_value_event_cells)
            lessons.append(
                _Lesson(
                    _Query(
                        QUERY_RELATION,
                        primary_carrier_cells=relation.node_occurrence_cells,
                        local_slot=relation.left_factor_slot_cell,
                    ),
                    _Target(
                        EVENT_RELATION,
                        physical_type_cells=relation.right_typed_value_type_cells,
                        operator_cell=relation.operator_cell,
                        rhs_kind_cell=relation.rhs_kind_cell,
                        right_slot_cell=relation.right_factor_slot_cell,
                        value_event_cells=relation.right_typed_value_event_cells,
                    ),
                )
            )
        for domain in identity_target.domains:
            if len(domain.complete_allowed_value_event_cells) != 1:
                raise RuntimeError(
                    "minimal retained FIT topology requires singleton literal domains"
                )
            value_event = domain.complete_allowed_value_event_cells[0]
            value_registry.add(value_event)
            lessons.append(
                _Lesson(
                    _Query(
                        QUERY_DOMAIN,
                        primary_carrier_cells=domain.node_occurrence_cells,
                        local_slot=domain.factor_slot_cell,
                    ),
                    _Target(
                        EVENT_DOMAIN,
                        physical_type_cells=domain.physical_type_cells,
                        value_event_cells=value_event,
                    ),
                )
            )
        for copy_group in identity_target.literal_copy_groups:
            copy_registry.add(copy_group.copy_group_cells)
            for carrier, local_slot in copy_group.terminal_occurrence_slot_cells:
                lessons.append(
                    _Lesson(
                        _Query(
                            QUERY_COPY,
                            primary_carrier_cells=carrier,
                            local_slot=local_slot,
                        ),
                        _Target(
                            EVENT_COPY,
                            copy_group_cells=copy_group.copy_group_cells,
                        ),
                    )
                )
        for signature in identity_target.signatures:
            lessons.append(
                _Lesson(
                    _Query(
                        QUERY_SIGNATURE,
                        primary_carrier_cells=signature.root_occurrence_cells,
                        signature_role_cell=signature.signature_role_cell,
                        signature_position_cell=signature.signature_position_cell,
                    ),
                    _Target(
                        EVENT_SIGNATURE,
                        physical_type_cells=signature.physical_type_cells,
                        signature_role_cell=signature.signature_role_cell,
                    ),
                )
            )

    ordered_lessons = tuple(
        sorted(lessons, key=lambda item: _canonical_json(item.root_record()))
    )
    query_targets: dict[bytes, set[bytes]] = defaultdict(set)
    for lesson in ordered_lessons:
        query_targets[_canonical_json(lesson.query.neural_record())].add(
            _canonical_json(lesson.target.neural_record())
        )
    conflict_count = sum(len(items) > 1 for items in query_targets.values())
    duplicate_query_count = len(ordered_lessons) - len(query_targets)
    feature_sets = tuple(_direct_features(lesson.query) for lesson in ordered_lessons)
    trace_collision_count = len(feature_sets) - len(set(feature_sets))
    feature_registry = tuple(
        sorted(
            {feature for features in feature_sets for feature in features},
            key=_canonical_json,
        )
    )
    category_counts = {
        query_kind: sum(
            lesson.query.query_kind_cell == query_kind for lesson in ordered_lessons
        )
        for query_kind in range(QUERY_KIND_COUNT)
    }
    expected_counts = {
        QUERY_SOURCE: 77,
        QUERY_EDGE: 68,
        QUERY_FACTOR: 164,
        QUERY_RELATION: 9,
        QUERY_DOMAIN: 18,
        QUERY_COPY: 18,
        QUERY_SIGNATURE: 18,
    }
    neural_bytes = b"".join(
        _canonical_json(lesson.query.neural_record()) for lesson in ordered_lessons
    )
    forbidden = (
        b"sha256",
        b"teacher",
        b"compiler",
        b"task",
        b"family",
        b"desired",
        b"program",
        b"entrypoint",
        b"property",
        b"source_order",
        b"fixed_index",
    )
    if (
        len(targets) != FIT_IDENTITY_COUNT
        or len(ordered_lessons) != FIT_LESSON_COUNT
        or category_counts != expected_counts
        or len(carrier_registry) != 77
        or len(copy_registry) != 18
        or len(source_bundles) != FIT_IDENTITY_COUNT
        or len(set(source_bundles)) != FIT_IDENTITY_COUNT
        or conflict_count
        or duplicate_query_count
        or trace_collision_count
        or any(token in neural_bytes for token in forbidden)
    ):
        raise RuntimeError("public-TRAIN FIT learner protocol is not identifiable")

    carriers = tuple(sorted(carrier_registry))
    copies = tuple(sorted(copy_registry))
    types = tuple(sorted(tuple(item) for item in annotation_v2.TYPE_ASSEMBLIES))
    values = tuple(sorted(value_registry))
    bundles = tuple(
        sorted(
            source_bundles,
            key=lambda bundle: _canonical_json(
                tuple(item.neural_record() for item in bundle)
            ),
        )
    )
    root_record = {
        "schema": SCHEMA,
        "dependency_sha256": dependency_sha,
        "v2_bank_commit": V2_BANK_COMMIT,
        "fit_identity_count": FIT_IDENTITY_COUNT,
        "fit_appearance_count": FIT_APPEARANCE_COUNT,
        "carrier_registry": carriers,
        "copy_group_registry": copies,
        "type_registry": types,
        "value_event_registry": values,
        "direct_physical_feature_registry": feature_registry,
        "source_bundles": tuple(
            tuple(item.neural_record() for item in bundle) for bundle in bundles
        ),
        "lessons": tuple(lesson.root_record() for lesson in ordered_lessons),
        "category_counts": category_counts,
        "query_target_conflict_count": conflict_count,
        "trace_collision_count": trace_collision_count,
    }
    return _FrozenProtocol(
        dependency_sha,
        carriers,
        copies,
        types,
        values,
        feature_registry,
        bundles,
        ordered_lessons,
        category_counts[QUERY_SOURCE],
        category_counts[QUERY_EDGE],
        category_counts[QUERY_FACTOR],
        category_counts[QUERY_RELATION],
        category_counts[QUERY_DOMAIN],
        category_counts[QUERY_COPY],
        category_counts[QUERY_SIGNATURE],
        conflict_count,
        trace_collision_count,
        _digest("protocol", root_record),
    )


PROTOCOL = _build_protocol()
FIT_SOURCE_BUNDLES = PROTOCOL.source_bundles
INPUT_CELL_COUNT = len(PROTOCOL.feature_registry)
_FEATURE_INDEX = {
    feature: index for index, feature in enumerate(PROTOCOL.feature_registry)
}


def _active_columns(query: _Query) -> np.ndarray:
    columns = tuple(
        _FEATURE_INDEX[feature]
        for feature in _direct_features(query)
        if feature in _FEATURE_INDEX
    )
    return np.asarray(columns, dtype=np.intp)


def _head_rows() -> tuple[tuple[int, ...], ...]:
    return (
        tuple(CARRIER_OFFSET + cell for cell in range(NODE_CARRIER_CELL_COUNT)),
        tuple(EVENT_KIND_OFFSET + cell for cell in range(EVENT_KIND_COUNT)),
        tuple(SOURCE_KIND_OFFSET + cell for cell in range(2)),
        tuple(CHILD_POSITION_OFFSET + cell for cell in range(MAX_CHILD_POSITION)),
        tuple(OCCUPANCY_OFFSET + cell for cell in range(len(OCCUPANCY_REGISTRY))),
        tuple(FACTOR_KIND_OFFSET + cell for cell in range(len(FACTOR_KINDS))),
        tuple(TYPE_OFFSET + cell for cell in range(TYPE_CELL_COUNT)),
        tuple(
            RELATION_OPERATOR_OFFSET + cell
            for cell in range(len(RELATION_OPERATORS))
        ),
        tuple(RHS_KIND_OFFSET + cell for cell in range(2)),
        tuple(RIGHT_SLOT_OFFSET + cell for cell in range(MAX_LOCAL_SLOT)),
        tuple(VALUE_EVENT_OFFSET + cell for cell in range(VALUE_EVENT_CELL_COUNT)),
        tuple(COPY_GROUP_OFFSET + cell for cell in range(COPY_GROUP_CELL_COUNT)),
        tuple(SIGNATURE_ROLE_OFFSET + cell for cell in range(2)),
    )


def _target_rows(target: _Target) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(CARRIER_OFFSET + cell for cell in target.carrier_cells),
        (EVENT_KIND_OFFSET + target.event_kind_cell,),
        (
            (SOURCE_KIND_OFFSET + target.source_kind_cell,)
            if target.source_kind_cell is not None
            else ()
        ),
        (
            (CHILD_POSITION_OFFSET + target.child_position_cell,)
            if target.child_position_cell is not None
            else ()
        ),
        (
            (OCCUPANCY_OFFSET + OCCUPANCY_REGISTRY.index(target.occupancy_cells),)
            if target.occupancy_cells
            else ()
        ),
        (
            (FACTOR_KIND_OFFSET + target.factor_kind_cell,)
            if target.factor_kind_cell is not None
            else ()
        ),
        tuple(TYPE_OFFSET + cell for cell in target.physical_type_cells),
        (
            (RELATION_OPERATOR_OFFSET + target.operator_cell,)
            if target.operator_cell is not None
            else ()
        ),
        (
            (RHS_KIND_OFFSET + target.rhs_kind_cell,)
            if target.rhs_kind_cell is not None
            else ()
        ),
        (
            (RIGHT_SLOT_OFFSET + target.right_slot_cell,)
            if target.right_slot_cell is not None
            else ()
        ),
        tuple(VALUE_EVENT_OFFSET + cell for cell in target.value_event_cells),
        tuple(COPY_GROUP_OFFSET + cell for cell in target.copy_group_cells),
        (
            (SIGNATURE_ROLE_OFFSET + target.signature_role_cell,)
            if target.signature_role_cell is not None
            else ()
        ),
    )


def _train(
    protocol: _FrozenProtocol,
    *,
    teacher_gain: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    excitatory = np.zeros((OUTPUT_CELL_COUNT, INPUT_CELL_COUNT), dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    if teacher_gain not in (0, 1):
        raise ValueError("teacher gain must be physically off or on")
    if not teacher_gain:
        return excitatory, inhibitory, 0, 0
    heads = _head_rows()
    committed_e = 0
    committed_h = 0
    for lesson in protocol.lessons:
        active = _active_columns(lesson.query)
        for registered, target in zip(heads, _target_rows(lesson.target), strict=True):
            target_set = set(target)
            if target:
                rows = np.asarray(target, dtype=np.intp)
                before = excitatory[np.ix_(rows, active)].astype(np.uint16)
                after = np.minimum(before + 1, TICK_CAP).astype(TICK_DTYPE)
                committed_e += int(np.sum(after.astype(np.uint16) - before))
                excitatory[np.ix_(rows, active)] = after
            competitors = tuple(row for row in registered if row not in target_set)
            if competitors:
                rows = np.asarray(competitors, dtype=np.intp)
                before = inhibitory[np.ix_(rows, active)].astype(np.uint16)
                after = np.minimum(before + 1, TICK_CAP).astype(TICK_DTYPE)
                committed_h += int(np.sum(after.astype(np.uint16) - before))
                inhibitory[np.ix_(rows, active)] = after
    return excitatory, inhibitory, committed_e, committed_h


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


def _decode(
    protocol: _FrozenProtocol,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    query: _Query,
    *,
    cell_gain: float = 1.0,
) -> _Decoded:
    physical_features = _direct_features(query)
    active = _active_columns(query)
    raw = np.sum(
        excitatory[:, active].astype(np.int16)
        - inhibitory[:, active].astype(np.int16),
        axis=1,
        dtype=np.int64,
    )
    score_threshold = max(
        1,
        (
            len(physical_features) * SCORE_THRESHOLD_NUMERATOR
            + SCORE_THRESHOLD_DENOMINATOR
            - 1
        )
        // SCORE_THRESHOLD_DENOMINATOR,
    )
    currents = np.greater_equal(raw, score_threshold).astype(np.float64)
    currents *= OUTPUT_CURRENT_PA * cell_gain
    spike_mask = np.asarray(_SPIKE_KERNEL.accumulated_spike_mask(currents), dtype=np.bool_)
    spike_cells = tuple(int(cell) for cell in np.flatnonzero(spike_mask))

    def head(offset: int, width: int) -> tuple[int, ...]:
        return tuple(
            cell - offset for cell in spike_cells if offset <= cell < offset + width
        )

    carrier_spikes = head(CARRIER_OFFSET, NODE_CARRIER_CELL_COUNT)
    carrier = _registered_assembly(carrier_spikes, protocol.carrier_registry)
    kind = _one(head(EVENT_KIND_OFFSET, EVENT_KIND_COUNT))
    source_kind = _one(head(SOURCE_KIND_OFFSET, 2))
    child_position = _one(head(CHILD_POSITION_OFFSET, MAX_CHILD_POSITION))
    occupancy_index = _one(head(OCCUPANCY_OFFSET, len(OCCUPANCY_REGISTRY)))
    occupancy = (
        OCCUPANCY_REGISTRY[occupancy_index]
        if occupancy_index is not None
        else ()
    )
    factor_kind = _one(head(FACTOR_KIND_OFFSET, len(FACTOR_KINDS)))
    type_spikes = head(TYPE_OFFSET, TYPE_CELL_COUNT)
    type_cells = _registered_assembly(type_spikes, protocol.type_registry)
    operator = _one(
        head(RELATION_OPERATOR_OFFSET, len(RELATION_OPERATORS))
    )
    rhs_kind = _one(head(RHS_KIND_OFFSET, 2))
    right_slot = _one(head(RIGHT_SLOT_OFFSET, MAX_LOCAL_SLOT))
    value_spikes = head(VALUE_EVENT_OFFSET, VALUE_EVENT_CELL_COUNT)
    value_cells = _registered_assembly(value_spikes, protocol.value_event_registry)
    copy_spikes = head(COPY_GROUP_OFFSET, COPY_GROUP_CELL_COUNT)
    copy_cells = _registered_assembly(copy_spikes, protocol.copy_group_registry)
    signature_role = _one(head(SIGNATURE_ROLE_OFFSET, 2))

    assemblies_valid = carrier is not None and type_cells is not None
    assemblies_valid = assemblies_valid and value_cells is not None and copy_cells is not None
    valid = assemblies_valid and kind is not None
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

    if valid:
        status = READY
    elif not spike_cells:
        status = STOP_ZERO
    else:
        status = STOP_AMBIGUOUS
    decoded_record = {
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
        _digest("decoded-query", decoded_record),
    )


def _lesson_exact(decoded: _Decoded, lesson: _Lesson) -> bool:
    target = lesson.target
    return (
        decoded.status == READY
        and decoded.carrier_cells == target.carrier_cells
        and decoded.event_kind_cell == target.event_kind_cell
        and decoded.source_kind_cell == target.source_kind_cell
        and decoded.child_position_cell == target.child_position_cell
        and decoded.occupancy_cells == target.occupancy_cells
        and decoded.factor_kind_cell == target.factor_kind_cell
        and decoded.physical_type_cells == target.physical_type_cells
        and decoded.operator_cell == target.operator_cell
        and decoded.rhs_kind_cell == target.rhs_kind_cell
        and decoded.right_slot_cell == target.right_slot_cell
        and decoded.value_event_cells == target.value_event_cells
        and decoded.copy_group_cells == target.copy_group_cells
        and decoded.signature_role_cell == target.signature_role_cell
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


class LearnerBuildSummary(NamedTuple):
    status: str
    dependency_sha256: str
    fit_identity_count: int
    fit_appearance_count: int
    fit_lesson_count: int
    fit_exact_count: int
    source_lesson_count: int
    edge_lesson_count: int
    factor_lesson_count: int
    relation_lesson_count: int
    domain_lesson_count: int
    copy_lesson_count: int
    signature_lesson_count: int
    local_source_cue_count: int
    contextual_source_query_count: int
    query_target_conflict_count: int
    trace_collision_count: int
    teacher_off_ready_count: int
    cell_off_ready_count: int
    geometry_derangement_ready_count: int
    nonlesson_query_count: int
    nonlesson_ready_count: int
    nonlesson_ambiguous_count: int
    duplicate_source_status: str
    input_order_invariant: bool
    fresh_zero_state_nonzero_count: int
    learned_state_bytes: int
    committed_excitatory_ticks: int
    committed_inhibitory_ticks: int
    physical_work_count: int
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
    protocol: _FrozenProtocol,
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
    children: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
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
    for _parent, entries in children.items():
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
    factors_by_carrier: dict[tuple[int, ...], list[FactorSlotRelay]] = defaultdict(list)
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
        rows = sorted(factors_by_carrier.get(carrier, ()), key=lambda item: item.local_slot)
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
        "decision_receipt_sha256s": tuple(sorted(row.receipt_sha256 for row in ready)),
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


def _infer_arrays(
    protocol: _FrozenProtocol,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    checkpoint_sha256: str,
    events: Iterable[PhysicalSourceEvent],
) -> tuple[
    str,
    str | None,
    OccurrenceFactorRelayView | None,
    int,
]:
    supplied = tuple(events)
    if any(
        type(item) not in (RawPhysicalSourceEvent, LiteralPhysicalSourceEvent)
        for item in supplied
    ):
        raise TypeError("learner inference accepts only physical source events")
    records = tuple(_canonical_json(item.neural_record()) for item in supplied)
    ordered = _canonical_source_bundle(supplied)
    request_sha = _digest(
        "inference-request",
        tuple(item.neural_record() for item in ordered),
    )
    if not ordered:
        return STOP_ZERO, "no physical source event was supplied", None, 0
    if len(records) != len(set(records)):
        return STOP_AMBIGUOUS, "a physical source event was duplicated", None, 0

    source_decisions = tuple(
        _decode(
            protocol,
            excitatory,
            inhibitory,
            _source_query(ordered, event),
        )
        for event in ordered
    )
    if any(row.status == STOP_AMBIGUOUS for row in source_decisions):
        return (
            STOP_AMBIGUOUS,
            "one physical source activated conflicting carrier fields",
            None,
            len(source_decisions),
        )
    if any(row.status != READY for row in source_decisions):
        return (
            STOP_ZERO,
            "one physical source had no learned contextual carrier",
            None,
            len(source_decisions),
        )
    carriers = tuple(sorted(row.carrier_cells for row in source_decisions))
    if len(carriers) != len(set(carriers)):
        return (
            STOP_AMBIGUOUS,
            "multiple physical sources activated one carrier",
            None,
            len(source_decisions),
        )

    generic_queries = _generic_queries(carriers)
    generic_decisions = tuple(
        _decode(protocol, excitatory, inhibitory, query)
        for query in generic_queries
    )
    physical_query_count = len(source_decisions) + len(generic_decisions)
    if any(row.status == STOP_AMBIGUOUS for row in generic_decisions):
        return (
            STOP_AMBIGUOUS,
            "one recurrent carrier query activated conflicting relay fields",
            None,
            physical_query_count,
        )
    status, reason, view = _assemble_view(
        protocol,
        request_sha256=request_sha,
        checkpoint_sha256=checkpoint_sha256,
        decisions=(*source_decisions, *generic_decisions),
        physical_query_count=physical_query_count,
    )
    return status, reason, view, physical_query_count


def _geometry_derangement_queries(
    protocol: _FrozenProtocol,
) -> tuple[_Query, ...]:
    queries: list[_Query] = []
    for context in protocol.source_bundles:
        for event in context:
            if not isinstance(event, RawPhysicalSourceEvent):
                continue
            used = {
                (item.direction, item.cue_slot, item.selected_slot)
                for item in context
                if isinstance(item, RawPhysicalSourceEvent)
                and item.assembly_cells == event.assembly_cells
            }
            candidates = tuple(
                RawPhysicalSourceEvent(
                    event.assembly_cells,
                    direction,
                    cue_slot,
                    (
                        (cue_slot + 1) % MAX_LOCAL_SLOT
                        if direction == "NEXT"
                        else (cue_slot - 1) % MAX_LOCAL_SLOT
                    ),
                )
                for direction in ("NEXT", "PREV")
                for cue_slot in range(MAX_LOCAL_SLOT)
                if (
                    direction,
                    cue_slot,
                    (
                        (cue_slot + 1) % MAX_LOCAL_SLOT
                        if direction == "NEXT"
                        else (cue_slot - 1) % MAX_LOCAL_SLOT
                    ),
                )
                not in used
            )
            if not candidates:
                raise RuntimeError("no valid absent geometry remains for a raw control")
            replacement = min(
                candidates,
                key=lambda item: _canonical_json(item.neural_record()),
            )
            queries.append(_source_query(context, replacement))
    if len(queries) != 59:
        raise RuntimeError("raw geometry control coverage changed")
    return tuple(queries)


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


class _CertificateReceipt(NamedTuple):
    authority: object
    consumer: FrozenPublicTrainFitOccurrenceFactorLearnerV1
    status: str
    stop_reason: str | None
    request_sha256: str
    view: OccurrenceFactorRelayView | None
    sha256: str


_CERTIFICATE_AUTHORITY = object()
_ISSUED_CERTIFICATES: dict[int, OccurrenceFactorLearnerCertificate] = {}


class OccurrenceFactorLearnerCertificate:
    """Opaque exact-object authority over one physical learner inference."""

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
    consumer: FrozenPublicTrainFitOccurrenceFactorLearnerV1,
    events: tuple[PhysicalSourceEvent, ...],
    *,
    status: str,
    stop_reason: str | None,
    view: OccurrenceFactorRelayView | None,
) -> OccurrenceFactorLearnerCertificate:
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
        "checkpoint_sha256": consumer.checkpoint_sha256,
        "protocol_sha256": consumer.protocol_sha256,
    }
    receipt = _CertificateReceipt(
        _CERTIFICATE_AUTHORITY,
        consumer,
        status,
        stop_reason,
        request_sha,
        view,
        _digest("opaque-certificate", payload),
    )
    certificate = OccurrenceFactorLearnerCertificate(receipt)
    _ISSUED_CERTIFICATES[id(certificate)] = certificate
    return certificate


def _authenticate_certificate(
    certificate: OccurrenceFactorLearnerCertificate,
) -> _CertificateReceipt:
    if (
        type(certificate) is not OccurrenceFactorLearnerCertificate
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


class FrozenPublicTrainFitOccurrenceFactorLearnerV1:
    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_metadata",
        "_protocol",
        "build_summary",
    )

    def __init__(
        self,
        protocol: _FrozenProtocol,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        metadata: Mapping[str, object],
        checkpoint_sha256: str,
        build_summary: LearnerBuildSummary,
    ) -> None:
        self._protocol = protocol
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._metadata = dict(metadata)
        self._checkpoint_sha256 = checkpoint_sha256
        self.build_summary = build_summary
        self._authenticate_checkpoint()

    @property
    def checkpoint_sha256(self) -> str:
        self._authenticate_checkpoint()
        return self._checkpoint_sha256

    @property
    def protocol_sha256(self) -> str:
        return self._protocol.protocol_sha256

    @property
    def learned_state_bytes(self) -> int:
        return self._excitatory.nbytes + self._inhibitory.nbytes

    def _authenticate_checkpoint(self) -> None:
        if (
            self._protocol is not PROTOCOL
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
                    self._metadata,
                    self._excitatory,
                    self._inhibitory,
                )
            ).hexdigest()
            != self._checkpoint_sha256
        ):
            raise RuntimeError("frozen public-TRAIN FIT learner checkpoint changed")

    def infer(
        self,
        events: Iterable[PhysicalSourceEvent],
    ) -> OccurrenceFactorLearnerCertificate:
        self._authenticate_checkpoint()
        supplied = tuple(events)
        status, reason, view, _work_count = _infer_arrays(
            self._protocol,
            self._excitatory,
            self._inhibitory,
            self._checkpoint_sha256,
            supplied,
        )
        self._authenticate_checkpoint()
        return _issue_certificate(
            self,
            supplied,
            status=status,
            stop_reason=reason,
            view=view,
        )


def build_frozen_public_train_fit_occurrence_factor_learner_v1(
    protocol: _FrozenProtocol = PROTOCOL,
) -> FrozenPublicTrainFitOccurrenceFactorLearnerV1:
    if protocol is not PROTOCOL:
        raise RuntimeError(STOP_DEPENDENCY)
    fresh_e = np.zeros((OUTPUT_CELL_COUNT, INPUT_CELL_COUNT), dtype=TICK_DTYPE)
    fresh_h = np.zeros_like(fresh_e)
    off_e, off_h, _off_e_ticks, _off_h_ticks = _train(protocol, teacher_gain=0)
    fitted_e, fitted_h, committed_e, committed_h = _train(protocol, teacher_gain=1)
    metadata = {
        "schema": SCHEMA,
        "protocol_sha256": protocol.protocol_sha256,
        "dependency_sha256": protocol.dependency_sha256,
        "v2_bank_commit": V2_BANK_COMMIT,
        "array_shape": fitted_e.shape,
        "array_dtype": fitted_e.dtype.str,
        "tick_cap": TICK_CAP,
        "score_threshold_fraction": (
            SCORE_THRESHOLD_NUMERATOR,
            SCORE_THRESHOLD_DENOMINATOR,
        ),
        "output_current_pa_hex": OUTPUT_CURRENT_PA.hex(),
        "output_steps": OUTPUT_STEPS,
        "fit_pass_count": FIT_PASS_COUNT,
        "fresh_weights_only": True,
        "direct_physical_conjunction_encoder": True,
        "whole_record_hash_input_count": 0,
        "heldout_input_count": HELDOUT_INPUT_COUNT,
        "heldout_target_read_count": HELDOUT_TARGET_READ_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "c1_input_count": C1_INPUT_COUNT,
        "path2_input_count": PATH2_INPUT_COUNT,
        "g0_input_count": G0_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "v1_partial_target_read_count": V1_PARTIAL_TARGET_READ_COUNT,
        "v4_checkpoint_read_count": V4_CHECKPOINT_READ_COUNT,
        "v5_checkpoint_read_count": V5_CHECKPOINT_READ_COUNT,
        "canonical_panel_spend_count": CANONICAL_PANEL_SPEND_COUNT,
    }
    checkpoint = sha256(_checkpoint_payload(metadata, fitted_e, fitted_h)).hexdigest()

    fit_decisions = tuple(
        _decode(protocol, fitted_e, fitted_h, lesson.query)
        for lesson in protocol.lessons
    )
    fit_exact = sum(
        _lesson_exact(decision, lesson)
        for decision, lesson in zip(fit_decisions, protocol.lessons, strict=True)
    )
    teacher_off_ready = sum(
        _decode(protocol, off_e, off_h, lesson.query).status == READY
        for lesson in protocol.lessons
    )
    cell_off_ready = sum(
        _decode(protocol, fitted_e, fitted_h, lesson.query, cell_gain=0.0).status
        == READY
        for lesson in protocol.lessons
    )
    geometry_queries = _geometry_derangement_queries(protocol)
    geometry_ready = sum(
        _decode(protocol, fitted_e, fitted_h, query).status == READY
        for query in geometry_queries
    )

    lesson_query_records = {
        _canonical_json(lesson.query.neural_record()) for lesson in protocol.lessons
    }
    nonlesson_queries: list[_Query] = []
    views: list[OccurrenceFactorRelayView] = []
    aggregate_inference_work = 0
    for bundle in protocol.source_bundles:
        source_decisions = tuple(
            _decode(
                protocol,
                fitted_e,
                fitted_h,
                _source_query(bundle, event),
            )
            for event in bundle
        )
        if any(row.status != READY for row in source_decisions):
            raise RuntimeError("fitted source head failed before recurrent controls")
        carriers = tuple(sorted(row.carrier_cells for row in source_decisions))
        nonlesson_queries.extend(
            query
            for query in _generic_queries(carriers)
            if _canonical_json(query.neural_record()) not in lesson_query_records
        )
        status, reason, view, work_count = _infer_arrays(
            protocol,
            fitted_e,
            fitted_h,
            checkpoint,
            bundle,
        )
        if status != READY or view is None:
            raise RuntimeError(reason or "same-FIT physical view did not assemble")
        views.append(view)
        aggregate_inference_work += work_count

    unique_nonlessons = tuple(
        {
            _canonical_json(query.neural_record()): query
            for query in nonlesson_queries
        }.values()
    )
    nonlesson_decisions = tuple(
        _decode(protocol, fitted_e, fitted_h, query)
        for query in unique_nonlessons
    )
    nonlesson_ready = sum(row.status == READY for row in nonlesson_decisions)
    nonlesson_ambiguous = sum(
        row.status == STOP_AMBIGUOUS for row in nonlesson_decisions
    )

    first_bundle = protocol.source_bundles[0]
    reverse_status, reverse_reason, reverse_view, reverse_work = _infer_arrays(
        protocol,
        fitted_e,
        fitted_h,
        checkpoint,
        reversed(first_bundle),
    )
    order_invariant = (
        reverse_status == READY
        and reverse_reason is None
        and reverse_view is not None
        and reverse_view.sha256 == views[0].sha256
    )
    duplicate_status, _duplicate_reason, _duplicate_view, _duplicate_work = (
        _infer_arrays(
            protocol,
            fitted_e,
            fitted_h,
            checkpoint,
            (*first_bundle, first_bundle[0]),
        )
    )

    local_source_cues = {
        _canonical_json(event.neural_record())
        for bundle in protocol.source_bundles
        for event in bundle
    }
    view_counts = (
        sum(len(view.source_relays) for view in views),
        sum(len(view.edges) for view in views),
        sum(len(view.factor_slots) for view in views),
        sum(len(view.relations) for view in views),
        sum(len(view.domains) for view in views),
        sum(len(view.literal_copy_groups) for view in views),
        sum(len(view.signature_relays) for view in views),
    )
    expected_view_counts = (77, 68, 164, 9, 18, 18, 18)
    physical_work_count = (
        len(fit_decisions)
        + len(protocol.lessons)
        + len(protocol.lessons)
        + len(geometry_queries)
        + len(nonlesson_decisions)
        + aggregate_inference_work
        + reverse_work
    )
    physical_work_receipt = _digest(
        "build-physical-work",
        {
            "physical_work_count": physical_work_count,
            "fit_decision_receipts": tuple(row.receipt_sha256 for row in fit_decisions),
            "view_roots": tuple(view.sha256 for view in views),
            "nonlesson_decision_receipts": tuple(
                row.receipt_sha256 for row in nonlesson_decisions
            ),
        },
    )
    ready = (
        fit_exact == FIT_LESSON_COUNT
        and teacher_off_ready == 0
        and cell_off_ready == 0
        and geometry_ready == 0
        and len(unique_nonlessons) == 1_675
        and nonlesson_ready == 0
        and nonlesson_ambiguous == 0
        and duplicate_status == STOP_AMBIGUOUS
        and order_invariant
        and view_counts == expected_view_counts
    )
    summary_record = {
        "status": READY if ready else STOP_ZERO,
        "dependency_sha256": protocol.dependency_sha256,
        "fit_identity_count": FIT_IDENTITY_COUNT,
        "fit_appearance_count": FIT_APPEARANCE_COUNT,
        "fit_lesson_count": FIT_LESSON_COUNT,
        "fit_exact_count": fit_exact,
        "source_lesson_count": protocol.source_lesson_count,
        "edge_lesson_count": protocol.edge_lesson_count,
        "factor_lesson_count": protocol.factor_lesson_count,
        "relation_lesson_count": protocol.relation_lesson_count,
        "domain_lesson_count": protocol.domain_lesson_count,
        "copy_lesson_count": protocol.copy_lesson_count,
        "signature_lesson_count": protocol.signature_lesson_count,
        "local_source_cue_count": len(local_source_cues),
        "contextual_source_query_count": protocol.source_lesson_count,
        "query_target_conflict_count": protocol.query_target_conflict_count,
        "trace_collision_count": protocol.trace_collision_count,
        "teacher_off_ready_count": teacher_off_ready,
        "cell_off_ready_count": cell_off_ready,
        "geometry_derangement_ready_count": geometry_ready,
        "nonlesson_query_count": len(unique_nonlessons),
        "nonlesson_ready_count": nonlesson_ready,
        "nonlesson_ambiguous_count": nonlesson_ambiguous,
        "duplicate_source_status": duplicate_status,
        "input_order_invariant": order_invariant,
        "fresh_zero_state_nonzero_count": int(
            np.count_nonzero(fresh_e) + np.count_nonzero(fresh_h)
        ),
        "learned_state_bytes": fitted_e.nbytes + fitted_h.nbytes,
        "committed_excitatory_ticks": committed_e,
        "committed_inhibitory_ticks": committed_h,
        "physical_work_count": physical_work_count,
        "checkpoint_sha256": checkpoint,
        "protocol_sha256": protocol.protocol_sha256,
        "physical_work_receipt_sha256": physical_work_receipt,
    }
    summary = LearnerBuildSummary(
        **summary_record,
        sha256=_digest("build-summary", summary_record),
    )
    if summary.status != READY:
        raise RuntimeError(
            "public-TRAIN FIT learner stopped: "
            f"{fit_exact}/{FIT_LESSON_COUNT} exact, "
            f"controls={teacher_off_ready}/{cell_off_ready}/{geometry_ready}/"
            f"{nonlesson_ready}/{nonlesson_ambiguous}, "
            f"nonlessons={len(unique_nonlessons)}, views={view_counts}"
        )
    fitted_e.setflags(write=False)
    fitted_h.setflags(write=False)
    return FrozenPublicTrainFitOccurrenceFactorLearnerV1(
        protocol,
        fitted_e,
        fitted_h,
        metadata,
        checkpoint,
        summary,
    )


STATIC_CONSUMER = build_frozen_public_train_fit_occurrence_factor_learner_v1()
BUILD_SUMMARY = STATIC_CONSUMER.build_summary


def infer_public_train_fit_occurrence_factors_v1(
    events: Iterable[PhysicalSourceEvent],
) -> OccurrenceFactorLearnerCertificate:
    """Public deep seam: unordered stripped physical bundle to opaque authority."""

    return STATIC_CONSUMER.infer(events)


def require_public_train_fit_occurrence_factors_v1(
    certificate: OccurrenceFactorLearnerCertificate,
) -> OccurrenceFactorRelayView:
    """Authenticate an issued READY certificate and expose physical relays."""

    receipt = _authenticate_certificate(certificate)
    receipt.consumer._authenticate_checkpoint()
    if receipt.status != READY or receipt.view is None:
        raise RuntimeError(receipt.stop_reason or receipt.status)
    return receipt.view


CLAIM_BOUNDARY = (
    "public-TRAIN same-FIT unordered-bundle contextual source replay and recurrent "
    "physical occurrence/factor composition only; local-cue generalization, unseen "
    "prompt transfer, canonical joint selection, DEV, protected, M0, C1, Path-2, "
    "G0, coding benchmarks, and broad capability remain unproven"
)


__all__ = [
    "BUILD_SUMMARY",
    "CLAIM_BOUNDARY",
    "FIT_SOURCE_BUNDLES",
    "PROTOCOL",
    "READY",
    "STATIC_CONSUMER",
    "STOP_AMBIGUOUS",
    "STOP_UNSUPPORTED",
    "STOP_ZERO",
    "V2_BANK_COMMIT",
    "LiteralPhysicalSourceEvent",
    "OccurrenceFactorLearnerCertificate",
    "OccurrenceFactorRelayView",
    "RawPhysicalSourceEvent",
    "build_frozen_public_train_fit_occurrence_factor_learner_v1",
    "infer_public_train_fit_occurrence_factors_v1",
    "require_public_train_fit_occurrence_factors_v1",
]
