"""Fresh physical structural relays for one complete eight-occurrence fixture.

V5 learned stable occurrence carriers, edges, child positions, and parent-slot
occupancy for a four-node tree.  This successor keeps that carrier-keyed seam
but learns the structural fields needed by a retained packed plan slice:
ordered factor slots, physical type/signature relays, generic relations,
finite literal domains, and literal-copy groups.  The self-posed treatment is
the smallest complete retained shape: eight occurrences and seven edges.

All learning targets are private self-posed protocol data.  Runtime inference
accepts only unordered physical event assemblies and returns an opaque
exact-object certificate with READY/ZERO/AMBIGUOUS/UNSUPPORTED semantics.  It
does not import a packed planner or construct a caller structural certificate.
The learned checkpoint and every treatment are fresh in this module; no V5
weights, cue hashes, or target records are replayed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot.public_train_factor_workspace_v4_static_checkpoint_capsule_v1 import (
    RawSpanObservation,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCHEMA = "PROMPT_OCCURRENCE_FACTOR_RELAYS_V6"
READY = "PROMPT_OCCURRENCE_FACTOR_RELAYS_V6_READY"
STOP_ZERO = "STOP_PROMPT_OCCURRENCE_FACTOR_RELAYS_V6_ZERO"
STOP_AMBIGUOUS = "STOP_PROMPT_OCCURRENCE_FACTOR_RELAYS_V6_AMBIGUOUS"
STOP_UNSUPPORTED = "STOP_PROMPT_OCCURRENCE_FACTOR_RELAYS_V6_UNSUPPORTED"

PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_INPUT_COUNT = 0
DESIRED_INPUT_COUNT = 0
PROGRAM_INPUT_COUNT = 0
PROPERTY_INPUT_COUNT = 0
TASK_INPUT_COUNT = 0
FAMILY_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

EVENT_ROOT = 0
EVENT_SOURCE = 1
EVENT_EDGE = 2
EVENT_FACTOR = 3
EVENT_RELATION = 4
EVENT_DOMAIN = 5
EVENT_COPY = 6
EVENT_SIGNATURE = 7
EVENT_KIND_COUNT = 8

FACTOR_LITERAL = 0
FACTOR_ROLE = 1
FACTOR_KINDS = ("literal", "role")

RELATION_OPERATORS = ("eq", "ne", "lt", "le", "gt", "ge")
RHS_SLOT = 0
RHS_VALUE = 1

DOMAIN_LITERAL = 0
DOMAIN_CLASS = 1
CLASS_INT = 0
CLASS_EXACT_VALUE = 1

SIGNATURE_PARAMETER = 0
SIGNATURE_RETURN = 1

EVENT_CELL_COUNT = 1_024
EVENT_KIND_INPUT_CELL_COUNT = 16
NODE_INPUT_OFFSET = EVENT_KIND_INPUT_CELL_COUNT
NODE_INPUT_CELL_COUNT = 256
PAYLOAD_INPUT_OFFSET = NODE_INPUT_OFFSET + NODE_INPUT_CELL_COUNT
PAYLOAD_INPUT_CELL_COUNT = EVENT_CELL_COUNT - PAYLOAD_INPUT_OFFSET
EVENT_ACTIVE_CELL_COUNT = 18
NODE_ACTIVE_CELL_COUNT = 6
PAYLOAD_ACTIVE_CELL_COUNT = 11

INPUT_CELL_COUNT = 4_096
TRACE_ACTIVE_CELL_COUNT = 24

CARRIER_CELL_COUNT = 256
CARRIER_ACTIVE_CELL_COUNT = 12
COPY_GROUP_CELL_COUNT = 64
COPY_GROUP_ACTIVE_CELL_COUNT = 8
TYPE_CELL_COUNT = 48
TYPE_ACTIVE_CELL_COUNT = 8
VALUE_RELAY_CELL_COUNT = 96
VALUE_RELAY_ACTIVE_CELL_COUNT = 8
CLASS_CELL_COUNT = 32
CLASS_ACTIVE_CELL_COUNT = 4
MAX_LOCAL_SLOT = 4
MAX_CHILD_POSITION = 2

OCCUPANCY_REGISTRY = (
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 1),
    (1, 2),
    (0, 1, 2),
    (0, 1, 2, 3),
)

PRIMARY_OFFSET = 0
SECONDARY_OFFSET = PRIMARY_OFFSET + CARRIER_CELL_COUNT
COPY_GROUP_OFFSET = SECONDARY_OFFSET + CARRIER_CELL_COUNT
EVENT_KIND_OFFSET = COPY_GROUP_OFFSET + COPY_GROUP_CELL_COUNT
LOCAL_SLOT_OFFSET = EVENT_KIND_OFFSET + EVENT_KIND_COUNT
FACTOR_KIND_OFFSET = LOCAL_SLOT_OFFSET + MAX_LOCAL_SLOT
TYPE_OFFSET = FACTOR_KIND_OFFSET + len(FACTOR_KINDS)
CHILD_POSITION_OFFSET = TYPE_OFFSET + TYPE_CELL_COUNT
OCCUPANCY_OFFSET = CHILD_POSITION_OFFSET + MAX_CHILD_POSITION
RELATION_OPERATOR_OFFSET = OCCUPANCY_OFFSET + len(OCCUPANCY_REGISTRY)
RHS_KIND_OFFSET = RELATION_OPERATOR_OFFSET + len(RELATION_OPERATORS)
RIGHT_SLOT_OFFSET = RHS_KIND_OFFSET + 2
VALUE_RELAY_OFFSET = RIGHT_SLOT_OFFSET + MAX_LOCAL_SLOT
DOMAIN_KIND_OFFSET = VALUE_RELAY_OFFSET + VALUE_RELAY_CELL_COUNT
CLASS_OFFSET = DOMAIN_KIND_OFFSET + 2
SIGNATURE_ROLE_OFFSET = CLASS_OFFSET + CLASS_CELL_COUNT
OUTPUT_CELL_COUNT = SIGNATURE_ROLE_OFFSET + 2

TICK_CAP = 4
SCORE_THRESHOLD = 7
OUTPUT_CURRENT_PA = 5_000.0
OUTPUT_STEPS = 30
TICK_DTYPE = np.dtype(np.uint8)

_NAMESPACE = b"prompt-occurrence-factor-relays-v6\0"
_SPIKE_KERNEL = CompiledRestingGlifRace.compile(step_count=OUTPUT_STEPS)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("utf-8"))
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


def _sparse_cells(
    label: str,
    key: object,
    *,
    offset: int,
    width: int,
    count: int,
) -> tuple[int, ...]:
    selected: set[int] = set()
    counter = 0
    while len(selected) < count:
        digest = sha256()
        digest.update(_NAMESPACE)
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(key))
        digest.update(counter.to_bytes(4, "little"))
        block = digest.digest()
        for start in range(0, len(block), 4):
            selected.add(offset + int.from_bytes(block[start : start + 4], "little") % width)
            if len(selected) == count:
                break
        counter += 1
    return tuple(sorted(selected))


def _allocate_disjoint(
    labels: tuple[str, ...],
    *,
    width: int,
    count: int,
    namespace: str,
) -> dict[str, tuple[int, ...]]:
    used: set[int] = set()
    result: dict[str, tuple[int, ...]] = {}
    for label in labels:
        selected: set[int] = set()
        counter = 0
        while len(selected) < count:
            for cell in _sparse_cells(
                namespace,
                (label, counter),
                offset=0,
                width=width,
                count=count,
            ):
                if cell not in used:
                    selected.add(cell)
                    if len(selected) == count:
                        break
            counter += 1
        frozen = tuple(sorted(selected))
        used.update(frozen)
        result[label] = frozen
    return result


@dataclass(frozen=True, slots=True)
class PhysicalTypeAssembly:
    value_type: str
    active_cells: tuple[int, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        cells = tuple(self.active_cells)
        if (
            self.value_type not in {"Int", "List[Int]"}
            or len(cells) != TYPE_ACTIVE_CELL_COUNT
            or tuple(sorted(cells)) != cells
            or len(set(cells)) != len(cells)
            or any(type(cell) is not int or not 0 <= cell < TYPE_CELL_COUNT for cell in cells)
        ):
            raise ValueError("physical type assembly changed")
        object.__setattr__(self, "active_cells", cells)
        object.__setattr__(
            self,
            "sha256",
            _digest("physical-type-assembly", (self.value_type, cells)),
        )

    def record(self) -> tuple[object, ...]:
        return (self.value_type, self.active_cells, self.sha256)


_TYPE_CELL_MAP = _allocate_disjoint(
    ("Int", "List[Int]"),
    width=TYPE_CELL_COUNT,
    count=TYPE_ACTIVE_CELL_COUNT,
    namespace="type-assembly",
)
TYPE_ASSEMBLIES = tuple(
    PhysicalTypeAssembly(name, _TYPE_CELL_MAP[name]) for name in ("Int", "List[Int]")
)
TYPE_BY_NAME = {item.value_type: item for item in TYPE_ASSEMBLIES}

_CLASS_CELL_MAP = _allocate_disjoint(
    ("int", "exact-value"),
    width=CLASS_CELL_COUNT,
    count=CLASS_ACTIVE_CELL_COUNT,
    namespace="domain-class",
)
EXACT_INT_CLASS_CELLS = tuple(sorted((*_CLASS_CELL_MAP["int"], *_CLASS_CELL_MAP["exact-value"])))


@dataclass(frozen=True, slots=True)
class TypedValueEvent:
    """Exact compact Int cells; the integer itself is never a field."""

    canonical_token_index: int
    typed_output_active_cells: tuple[int, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        cells = tuple(self.typed_output_active_cells)
        int_start = output_cells.INT_OFFSET
        int_stop = int_start + output_cells.INT_KNOWN_CELL_COUNT
        if (
            type(self.canonical_token_index) is not int
            or self.canonical_token_index < 0
            or len(cells) != 2
            or cells[0] != output_cells.TYPE_OFFSET
            or not int_start <= cells[1] < int_stop
            or tuple(sorted(cells)) != cells
        ):
            raise ValueError("typed value event must be one known compact Int pair")
        object.__setattr__(self, "typed_output_active_cells", cells)
        object.__setattr__(
            self,
            "sha256",
            _digest(
                "typed-value-event",
                (self.canonical_token_index, self.typed_output_active_cells),
            ),
        )

    def record(self) -> tuple[object, ...]:
        return (
            self.canonical_token_index,
            self.typed_output_active_cells,
            self.sha256,
        )


def _typed_value(token_index: int, value: int) -> TypedValueEvent:
    encoded = output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=output_cells.OutputType.INT,
    )
    if not encoded.supported:
        raise RuntimeError("self-posed value left the compact Int bank")
    return TypedValueEvent(token_index, encoded.active_global_cell_indices())


def _source_cells(source_sha256: str) -> tuple[int, ...]:
    return _sparse_cells(
        "source-identity",
        source_sha256,
        offset=PAYLOAD_INPUT_OFFSET,
        width=PAYLOAD_INPUT_CELL_COUNT,
        count=6,
    )


@dataclass(frozen=True, slots=True)
class PhysicalRelayEvent:
    feature_cells: tuple[int, ...]
    raw_observation: RawSpanObservation | None = None
    typed_value_event: TypedValueEvent | None = None
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        cells = tuple(self.feature_cells)
        source = self.raw_observation or self.typed_value_event
        if (
            len(cells) != EVENT_ACTIVE_CELL_COUNT
            or tuple(sorted(cells)) != cells
            or len(set(cells)) != len(cells)
            or any(type(cell) is not int or not 0 <= cell < EVENT_CELL_COUNT for cell in cells)
            or sum(cell < NODE_INPUT_OFFSET for cell in cells) != 1
            or sum(NODE_INPUT_OFFSET <= cell < PAYLOAD_INPUT_OFFSET for cell in cells)
            != NODE_ACTIVE_CELL_COUNT
            or sum(cell >= PAYLOAD_INPUT_OFFSET for cell in cells) != PAYLOAD_ACTIVE_CELL_COUNT
            or (self.raw_observation is not None and self.typed_value_event is not None)
            or (
                source is not None
                and not set(_source_cells(source.sha256))
                <= {cell for cell in cells if cell >= PAYLOAD_INPUT_OFFSET}
            )
        ):
            raise ValueError("physical V6 relay event geometry changed")
        object.__setattr__(self, "feature_cells", cells)
        object.__setattr__(
            self,
            "sha256",
            _digest(
                "physical-relay-event",
                {
                    "feature_cells": cells,
                    "raw_observation_sha256": (
                        self.raw_observation.sha256 if self.raw_observation is not None else None
                    ),
                    "typed_value_sha256": (
                        self.typed_value_event.sha256
                        if self.typed_value_event is not None
                        else None
                    ),
                },
            ),
        )

    def record(self) -> dict[str, object]:
        return {
            "feature_cells": self.feature_cells,
            "raw_observation_sha256": (
                self.raw_observation.sha256 if self.raw_observation is not None else None
            ),
            "typed_value_sha256": (
                self.typed_value_event.sha256 if self.typed_value_event is not None else None
            ),
            "sha256": self.sha256,
        }


def _event_trace_cells(event: PhysicalRelayEvent) -> tuple[int, ...]:
    return _sparse_cells(
        "event-conjunction-trace",
        event.record(),
        offset=0,
        width=INPUT_CELL_COUNT,
        count=TRACE_ACTIVE_CELL_COUNT,
    )


@dataclass(frozen=True, slots=True)
class _Lesson:
    event: PhysicalRelayEvent
    primary_carrier_cells: tuple[int, ...]
    secondary_carrier_cells: tuple[int, ...] = ()
    copy_group_cells: tuple[int, ...] = ()
    event_kind_cell: int = EVENT_SOURCE
    local_slot: int | None = None
    factor_kind_cell: int | None = None
    type_cells: tuple[int, ...] = ()
    child_position: int | None = None
    parent_slot_occupancy: tuple[int, ...] = ()
    relation_operator_cell: int | None = None
    rhs_kind_cell: int | None = None
    right_slot: int | None = None
    value_relay_cells: tuple[int, ...] = ()
    domain_kind_cell: int | None = None
    class_cells: tuple[int, ...] = ()
    signature_role_cell: int | None = None
    sha256: str = ""

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "event": self.event.record(),
            "primary_carrier_cells": self.primary_carrier_cells,
            "secondary_carrier_cells": self.secondary_carrier_cells,
            "copy_group_cells": self.copy_group_cells,
            "event_kind_cell": self.event_kind_cell,
            "local_slot": self.local_slot,
            "factor_kind_cell": self.factor_kind_cell,
            "type_cells": self.type_cells,
            "child_position": self.child_position,
            "parent_slot_occupancy": self.parent_slot_occupancy,
            "relation_operator_cell": self.relation_operator_cell,
            "rhs_kind_cell": self.rhs_kind_cell,
            "right_slot": self.right_slot,
            "value_relay_cells": self.value_relay_cells,
            "domain_kind_cell": self.domain_kind_cell,
            "class_cells": self.class_cells,
            "signature_role_cell": self.signature_role_cell,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result


@dataclass(frozen=True, slots=True)
class FrozenOccurrenceFactorProtocolV6:
    lessons: tuple[_Lesson, ...]
    treatment_events: tuple[PhysicalRelayEvent, ...]
    counterfactual_events: tuple[PhysicalRelayEvent, ...]
    auxiliary_relation_events: tuple[PhysicalRelayEvent, ...]
    carrier_registry: tuple[tuple[int, ...], ...]
    copy_group_registry: tuple[tuple[int, ...], ...]
    typed_value_registry: tuple[tuple[TypedValueEvent, tuple[int, ...]], ...]
    protocol_sha256: str

    def root_record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "lesson_sha256s": tuple(item.sha256 for item in self.lessons),
            "treatment_event_sha256s": tuple(item.sha256 for item in self.treatment_events),
            "counterfactual_event_sha256s": tuple(
                item.sha256 for item in self.counterfactual_events
            ),
            "auxiliary_relation_event_sha256s": tuple(
                item.sha256 for item in self.auxiliary_relation_events
            ),
            "carrier_registry": self.carrier_registry,
            "copy_group_registry": self.copy_group_registry,
            "type_assemblies": tuple(item.record() for item in TYPE_ASSEMBLIES),
            "typed_value_registry": tuple(
                (event.record(), cells) for event, cells in self.typed_value_registry
            ),
            "relation_operator_registry": RELATION_OPERATORS,
            "generic_factor_kinds": FACTOR_KINDS,
            "fresh_weights_only": True,
            "v5_checkpoint_reads": 0,
            "caller_structural_certificate_count": 0,
            "public_train_input_count": PUBLIC_TRAIN_INPUT_COUNT,
            "canonical_input_count": CANONICAL_INPUT_COUNT,
            "desired_input_count": DESIRED_INPUT_COUNT,
            "program_input_count": PROGRAM_INPUT_COUNT,
            "property_input_count": PROPERTY_INPUT_COUNT,
            "task_input_count": TASK_INPUT_COUNT,
            "family_input_count": FAMILY_INPUT_COUNT,
            "dev_input_count": DEV_INPUT_COUNT,
            "heldout_input_count": HELDOUT_INPUT_COUNT,
            "protected_input_count": PROTECTED_INPUT_COUNT,
        }


_FACTOR_PATTERN = (
    (
        (FACTOR_LITERAL, "Int"),
        (FACTOR_LITERAL, "Int"),
        (FACTOR_ROLE, "Int"),
        (FACTOR_ROLE, "List[Int]"),
    ),
    ((FACTOR_LITERAL, "Int"), (FACTOR_LITERAL, "Int"), (FACTOR_ROLE, "Int")),
    ((FACTOR_LITERAL, "Int"), (FACTOR_LITERAL, "Int"), (FACTOR_ROLE, "Int")),
    ((FACTOR_LITERAL, "Int"),),
    ((FACTOR_LITERAL, "Int"), (FACTOR_ROLE, "Int")),
    ((FACTOR_ROLE, "Int"),),
    ((FACTOR_LITERAL, "Int"),),
    ((FACTOR_ROLE, "List[Int]"),),
)
_EDGE_PATTERN = (
    (0, 1, 0, (0, 1, 2)),
    (1, 2, 0, (0, 1, 2)),
    (2, 3, 0, (0,)),
    (2, 4, 1, (1, 2)),
    (4, 5, 0, (1,)),
    (4, 6, 1, (0,)),
    (0, 7, 1, (3,)),
)


def _raw_observation(key: str, ordinal: int) -> RawSpanObservation:
    assembly = _sparse_cells(
        "raw-source-assembly",
        key,
        offset=0,
        width=512,
        count=12,
    )
    direction = "NEXT" if ordinal % 2 == 0 else "PREV"
    cue = ordinal % 4
    selected = (cue + 1) % 4 if direction == "NEXT" else (cue - 1) % 4
    return RawSpanObservation(direction, cue, selected, assembly)


def _make_protocol() -> FrozenOccurrenceFactorProtocolV6:
    graph_names = ("alpha", "beta")
    node_labels = tuple(f"{graph}:n{node}" for graph in graph_names for node in range(8))
    group_labels = tuple(f"{graph}:g{group}" for graph in graph_names for group in range(2))
    node_cues = _allocate_disjoint(
        node_labels,
        width=NODE_INPUT_CELL_COUNT,
        count=NODE_ACTIVE_CELL_COUNT,
        namespace="node-cue",
    )
    carriers = _allocate_disjoint(
        node_labels,
        width=CARRIER_CELL_COUNT,
        count=CARRIER_ACTIVE_CELL_COUNT,
        namespace="occurrence-carrier",
    )
    groups = _allocate_disjoint(
        group_labels,
        width=COPY_GROUP_CELL_COUNT,
        count=COPY_GROUP_ACTIVE_CELL_COUNT,
        namespace="copy-group",
    )

    graph_values = {
        "alpha": (_typed_value(103, -14), _typed_value(106, -15), _typed_value(140, 0)),
        "beta": (_typed_value(203, -2), _typed_value(206, 2), _typed_value(240, 5)),
    }
    auxiliary_values = (_typed_value(340, 7), _typed_value(341, -3))
    all_values = (
        tuple(
            {event.sha256: event for values in graph_values.values() for event in values}.values()
        )
        + auxiliary_values
    )
    value_cells = _allocate_disjoint(
        tuple(event.sha256 for event in all_values),
        width=VALUE_RELAY_CELL_COUNT,
        count=VALUE_RELAY_ACTIVE_CELL_COUNT,
        namespace="typed-value-relay",
    )
    value_registry = tuple((event, value_cells[event.sha256]) for event in all_values)

    def make_event(
        graph: str,
        node: int,
        event_kind: int,
        payload: object,
        *,
        raw: RawSpanObservation | None = None,
        value: TypedValueEvent | None = None,
    ) -> PhysicalRelayEvent:
        source = raw or value
        payload_cells = set(
            _source_cells(source.sha256)
            if source is not None
            else _sparse_cells(
                "payload",
                (graph, node, event_kind, payload),
                offset=PAYLOAD_INPUT_OFFSET,
                width=PAYLOAD_INPUT_CELL_COUNT,
                count=PAYLOAD_ACTIVE_CELL_COUNT,
            )
        )
        counter = 0
        while len(payload_cells) < PAYLOAD_ACTIVE_CELL_COUNT:
            for cell in _sparse_cells(
                "payload-fill",
                (graph, node, event_kind, payload, counter),
                offset=PAYLOAD_INPUT_OFFSET,
                width=PAYLOAD_INPUT_CELL_COUNT,
                count=PAYLOAD_ACTIVE_CELL_COUNT,
            ):
                payload_cells.add(cell)
                if len(payload_cells) == PAYLOAD_ACTIVE_CELL_COUNT:
                    break
            counter += 1
        cells = (
            event_kind,
            *(NODE_INPUT_OFFSET + cell for cell in node_cues[f"{graph}:n{node}"]),
            *sorted(payload_cells),
        )
        return PhysicalRelayEvent(tuple(sorted(cells)), raw, value)

    lessons: list[_Lesson] = []
    graph_event_sets: dict[str, tuple[PhysicalRelayEvent, ...]] = {}

    def append_lesson(**fields: object) -> _Lesson:
        partial = _Lesson(**fields)
        lesson = _Lesson(
            **{
                name: getattr(partial, name)
                for name in partial.__dataclass_fields__
                if name != "sha256"
            },
            sha256=_digest("protocol-lesson", partial.record(include_sha256=False)),
        )
        lessons.append(lesson)
        return lesson

    for graph_index, graph in enumerate(graph_names):
        graph_events: list[PhysicalRelayEvent] = []
        literal_a, literal_b, relation_value = graph_values[graph]
        source_values = {3: literal_a, 6: literal_b}
        for node in range(8):
            if node in source_values:
                event = make_event(
                    graph,
                    node,
                    EVENT_ROOT if node == 0 else EVENT_SOURCE,
                    ("source", node),
                    value=source_values[node],
                )
            else:
                raw = _raw_observation(f"{graph}:source:{node}", graph_index * 8 + node)
                event = make_event(
                    graph,
                    node,
                    EVENT_ROOT if node == 0 else EVENT_SOURCE,
                    ("source", node),
                    raw=raw,
                )
            append_lesson(
                event=event,
                primary_carrier_cells=carriers[f"{graph}:n{node}"],
                event_kind_cell=EVENT_ROOT if node == 0 else EVENT_SOURCE,
            )
            graph_events.append(event)

        for parent, child, position, occupancy in _EDGE_PATTERN:
            event = make_event(
                graph,
                parent,
                EVENT_EDGE,
                ("edge", child, position, occupancy),
            )
            append_lesson(
                event=event,
                primary_carrier_cells=carriers[f"{graph}:n{parent}"],
                secondary_carrier_cells=carriers[f"{graph}:n{child}"],
                event_kind_cell=EVENT_EDGE,
                child_position=position,
                parent_slot_occupancy=occupancy,
            )
            graph_events.append(event)

        for node, factors in enumerate(_FACTOR_PATTERN):
            for slot, (factor_kind, value_type) in enumerate(factors):
                event = make_event(
                    graph,
                    node,
                    EVENT_FACTOR,
                    ("factor", slot, factor_kind, value_type),
                )
                append_lesson(
                    event=event,
                    primary_carrier_cells=carriers[f"{graph}:n{node}"],
                    event_kind_cell=EVENT_FACTOR,
                    local_slot=slot,
                    factor_kind_cell=factor_kind,
                    type_cells=TYPE_BY_NAME[value_type].active_cells,
                )
                graph_events.append(event)

        if graph == "alpha":
            relation_operator = RELATION_OPERATORS.index("ne")
            rhs_kind = RHS_VALUE
            right_slot = None
            relation_source = relation_value
        else:
            relation_operator = RELATION_OPERATORS.index("le")
            rhs_kind = RHS_SLOT
            right_slot = 1
            relation_source = None
        relation_event = make_event(
            graph,
            4,
            EVENT_RELATION,
            ("relation", relation_operator, rhs_kind, right_slot),
            value=relation_source,
        )
        append_lesson(
            event=relation_event,
            primary_carrier_cells=carriers[f"{graph}:n4"],
            event_kind_cell=EVENT_RELATION,
            local_slot=0,
            relation_operator_cell=relation_operator,
            rhs_kind_cell=rhs_kind,
            right_slot=right_slot,
            value_relay_cells=(
                value_cells[relation_source.sha256] if relation_source is not None else ()
            ),
        )
        graph_events.append(relation_event)

        for domain_index, (node, domain_value) in enumerate(((3, literal_a), (6, literal_b))):
            domain_event = make_event(
                graph,
                node,
                EVENT_DOMAIN,
                ("domain", domain_index),
                value=domain_value,
            )
            append_lesson(
                event=domain_event,
                primary_carrier_cells=carriers[f"{graph}:n{node}"],
                event_kind_cell=EVENT_DOMAIN,
                local_slot=0,
                type_cells=TYPE_BY_NAME["Int"].active_cells,
                value_relay_cells=value_cells[domain_value.sha256],
                domain_kind_cell=DOMAIN_LITERAL,
                class_cells=EXACT_INT_CLASS_CELLS,
            )
            graph_events.append(domain_event)

        for group_index, node in enumerate((3, 6)):
            copy_event = make_event(
                graph,
                node,
                EVENT_COPY,
                ("copy", group_index),
            )
            append_lesson(
                event=copy_event,
                primary_carrier_cells=carriers[f"{graph}:n{node}"],
                copy_group_cells=groups[f"{graph}:g{group_index}"],
                event_kind_cell=EVENT_COPY,
                local_slot=0,
            )
            graph_events.append(copy_event)

        for signature_role, value_type in (
            (SIGNATURE_PARAMETER, "List[Int]"),
            (SIGNATURE_RETURN, "Int"),
        ):
            signature_event = make_event(
                graph,
                0,
                EVENT_SIGNATURE,
                ("signature", signature_role, value_type),
            )
            append_lesson(
                event=signature_event,
                primary_carrier_cells=carriers[f"{graph}:n0"],
                event_kind_cell=EVENT_SIGNATURE,
                local_slot=0,
                type_cells=TYPE_BY_NAME[value_type].active_cells,
                signature_role_cell=signature_role,
            )
            graph_events.append(signature_event)
        graph_event_sets[graph] = tuple(graph_events)

    auxiliary_events: list[PhysicalRelayEvent] = []
    auxiliary_specs = (
        ("eq", RHS_SLOT, 1, None),
        ("lt", RHS_SLOT, 1, None),
        ("gt", RHS_VALUE, None, auxiliary_values[0]),
        ("ge", RHS_VALUE, None, auxiliary_values[1]),
    )
    for auxiliary_index, (operator, rhs_kind, right_slot, source) in enumerate(auxiliary_specs):
        operator_cell = RELATION_OPERATORS.index(operator)
        event = make_event(
            "alpha",
            4,
            EVENT_RELATION,
            ("auxiliary-relation", auxiliary_index, operator_cell, rhs_kind, right_slot),
            value=source,
        )
        append_lesson(
            event=event,
            primary_carrier_cells=carriers["alpha:n4"],
            event_kind_cell=EVENT_RELATION,
            local_slot=0,
            relation_operator_cell=operator_cell,
            rhs_kind_cell=rhs_kind,
            right_slot=right_slot,
            value_relay_cells=value_cells[source.sha256] if source is not None else (),
        )
        auxiliary_events.append(event)

    partial = FrozenOccurrenceFactorProtocolV6(
        tuple(lessons),
        graph_event_sets["alpha"],
        graph_event_sets["beta"],
        tuple(auxiliary_events),
        tuple(sorted(carriers.values())),
        tuple(sorted(groups.values())),
        value_registry,
        "",
    )
    protocol = FrozenOccurrenceFactorProtocolV6(
        **{
            name: getattr(partial, name)
            for name in partial.__dataclass_fields__
            if name != "protocol_sha256"
        },
        protocol_sha256=_digest("protocol", partial.root_record()),
    )
    _audit_protocol(protocol)
    return protocol


def _audit_protocol(protocol: FrozenOccurrenceFactorProtocolV6) -> None:
    operator_cells = {
        lesson.relation_operator_cell
        for lesson in protocol.lessons
        if lesson.event_kind_cell == EVENT_RELATION
    }
    rhs_cells = {
        lesson.rhs_kind_cell
        for lesson in protocol.lessons
        if lesson.event_kind_cell == EVENT_RELATION
    }
    if (
        len(protocol.treatment_events) != 38
        or len(protocol.counterfactual_events) != 38
        or len(protocol.auxiliary_relation_events) != 4
        or len(protocol.lessons) != 80
        or len({lesson.event.sha256 for lesson in protocol.lessons}) != 80
        or operator_cells != set(range(len(RELATION_OPERATORS)))
        or rhs_cells != {RHS_SLOT, RHS_VALUE}
        or protocol.protocol_sha256 != _digest("protocol", protocol.root_record())
    ):
        raise RuntimeError("V6 self-posed protocol audit changed")


PROTOCOL = _make_protocol()
SELF_POSED_TREATMENT_EVENTS = PROTOCOL.treatment_events
SELF_POSED_COUNTERFACTUAL_EVENTS = PROTOCOL.counterfactual_events


def _head_rows(
    protocol: FrozenOccurrenceFactorProtocolV6,
) -> tuple[tuple[int, ...], ...]:
    carriers = tuple(sorted({cell for item in protocol.carrier_registry for cell in item}))
    groups = tuple(sorted({cell for item in protocol.copy_group_registry for cell in item}))
    values = tuple(
        sorted({cell for _event, item in protocol.typed_value_registry for cell in item})
    )
    types = tuple(sorted({cell for item in TYPE_ASSEMBLIES for cell in item.active_cells}))
    return (
        tuple(PRIMARY_OFFSET + cell for cell in carriers),
        tuple(SECONDARY_OFFSET + cell for cell in carriers),
        tuple(COPY_GROUP_OFFSET + cell for cell in groups),
        tuple(EVENT_KIND_OFFSET + cell for cell in range(EVENT_KIND_COUNT)),
        tuple(LOCAL_SLOT_OFFSET + cell for cell in range(MAX_LOCAL_SLOT)),
        tuple(FACTOR_KIND_OFFSET + cell for cell in range(len(FACTOR_KINDS))),
        tuple(TYPE_OFFSET + cell for cell in types),
        tuple(CHILD_POSITION_OFFSET + cell for cell in range(MAX_CHILD_POSITION)),
        tuple(OCCUPANCY_OFFSET + cell for cell in range(len(OCCUPANCY_REGISTRY))),
        tuple(RELATION_OPERATOR_OFFSET + cell for cell in range(len(RELATION_OPERATORS))),
        tuple(RHS_KIND_OFFSET + cell for cell in range(2)),
        tuple(RIGHT_SLOT_OFFSET + cell for cell in range(MAX_LOCAL_SLOT)),
        tuple(VALUE_RELAY_OFFSET + cell for cell in values),
        tuple(DOMAIN_KIND_OFFSET + cell for cell in range(2)),
        tuple(CLASS_OFFSET + cell for cell in EXACT_INT_CLASS_CELLS),
        tuple(SIGNATURE_ROLE_OFFSET + cell for cell in range(2)),
    )


def _lesson_target_rows(lesson: _Lesson) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(PRIMARY_OFFSET + cell for cell in lesson.primary_carrier_cells),
        tuple(SECONDARY_OFFSET + cell for cell in lesson.secondary_carrier_cells),
        tuple(COPY_GROUP_OFFSET + cell for cell in lesson.copy_group_cells),
        (EVENT_KIND_OFFSET + lesson.event_kind_cell,),
        ((LOCAL_SLOT_OFFSET + lesson.local_slot,) if lesson.local_slot is not None else ()),
        (
            (FACTOR_KIND_OFFSET + lesson.factor_kind_cell,)
            if lesson.factor_kind_cell is not None
            else ()
        ),
        tuple(TYPE_OFFSET + cell for cell in lesson.type_cells),
        (
            (CHILD_POSITION_OFFSET + lesson.child_position,)
            if lesson.child_position is not None
            else ()
        ),
        (
            (OCCUPANCY_OFFSET + OCCUPANCY_REGISTRY.index(lesson.parent_slot_occupancy),)
            if lesson.parent_slot_occupancy
            else ()
        ),
        (
            (RELATION_OPERATOR_OFFSET + lesson.relation_operator_cell,)
            if lesson.relation_operator_cell is not None
            else ()
        ),
        ((RHS_KIND_OFFSET + lesson.rhs_kind_cell,) if lesson.rhs_kind_cell is not None else ()),
        ((RIGHT_SLOT_OFFSET + lesson.right_slot,) if lesson.right_slot is not None else ()),
        tuple(VALUE_RELAY_OFFSET + cell for cell in lesson.value_relay_cells),
        (
            (DOMAIN_KIND_OFFSET + lesson.domain_kind_cell,)
            if lesson.domain_kind_cell is not None
            else ()
        ),
        tuple(CLASS_OFFSET + cell for cell in lesson.class_cells),
        (
            (SIGNATURE_ROLE_OFFSET + lesson.signature_role_cell,)
            if lesson.signature_role_cell is not None
            else ()
        ),
    )


def _train(
    protocol: FrozenOccurrenceFactorProtocolV6,
    *,
    teacher_gain: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    excitatory = np.zeros((OUTPUT_CELL_COUNT, INPUT_CELL_COUNT), dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    if teacher_gain not in (0, 1):
        raise ValueError("teacher gain must be physically off or on")
    if not teacher_gain:
        return excitatory, inhibitory, 0, 0
    heads = _head_rows(protocol)
    committed_e = committed_h = 0
    for lesson in protocol.lessons:
        active = np.asarray(_event_trace_cells(lesson.event), dtype=np.intp)
        for registered, target in zip(heads, _lesson_target_rows(lesson), strict=True):
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


class _DecodedEvent(NamedTuple):
    status: str
    event: PhysicalRelayEvent
    primary_carrier_cells: tuple[int, ...]
    secondary_carrier_cells: tuple[int, ...]
    copy_group_cells: tuple[int, ...]
    event_kind_cell: int | None
    local_slot: int | None
    factor_kind_cell: int | None
    type_assembly: PhysicalTypeAssembly | None
    child_position: int | None
    parent_slot_occupancy: tuple[int, ...]
    relation_operator_cell: int | None
    rhs_kind_cell: int | None
    right_slot: int | None
    typed_value_event: TypedValueEvent | None
    value_relay_cells: tuple[int, ...]
    domain_kind_cell: int | None
    class_cells: tuple[int, ...]
    signature_role_cell: int | None
    raw_scores: tuple[int, ...]
    spike_cells: tuple[int, ...]
    sha256: str


def _assembly_from_spikes(
    spikes: tuple[int, ...],
    registry: tuple[tuple[int, ...], ...],
) -> tuple[str, tuple[int, ...]]:
    if not spikes:
        return STOP_ZERO, ()
    matches = tuple(item for item in registry if item == spikes)
    if len(matches) == 1:
        return READY, matches[0]
    return STOP_AMBIGUOUS, ()


def _one(spikes: tuple[int, ...]) -> int | None:
    return spikes[0] if len(spikes) == 1 else None


def _decode_event(
    protocol: FrozenOccurrenceFactorProtocolV6,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    event: PhysicalRelayEvent,
    *,
    cell_gain: float = 1.0,
) -> _DecodedEvent:
    active = np.asarray(_event_trace_cells(event), dtype=np.intp)
    raw = np.sum(
        excitatory[:, active].astype(np.int16) - inhibitory[:, active].astype(np.int16),
        axis=1,
        dtype=np.int64,
    )
    currents = np.greater_equal(raw, SCORE_THRESHOLD).astype(np.float64)
    currents *= OUTPUT_CURRENT_PA * cell_gain
    spike_mask = np.asarray(_SPIKE_KERNEL.accumulated_spike_mask(currents), dtype=np.bool_)
    spike_cells = tuple(int(cell) for cell in np.flatnonzero(spike_mask))

    def head(offset: int, width: int) -> tuple[int, ...]:
        return tuple(cell - offset for cell in spike_cells if offset <= cell < offset + width)

    primary_spikes = head(PRIMARY_OFFSET, CARRIER_CELL_COUNT)
    secondary_spikes = head(SECONDARY_OFFSET, CARRIER_CELL_COUNT)
    group_spikes = head(COPY_GROUP_OFFSET, COPY_GROUP_CELL_COUNT)
    primary_status, primary = _assembly_from_spikes(primary_spikes, protocol.carrier_registry)
    secondary_status, secondary = _assembly_from_spikes(secondary_spikes, protocol.carrier_registry)
    group_status, group = _assembly_from_spikes(group_spikes, protocol.copy_group_registry)
    kind = _one(head(EVENT_KIND_OFFSET, EVENT_KIND_COUNT))
    local_slot = _one(head(LOCAL_SLOT_OFFSET, MAX_LOCAL_SLOT))
    factor_kind = _one(head(FACTOR_KIND_OFFSET, len(FACTOR_KINDS)))
    type_spikes = head(TYPE_OFFSET, TYPE_CELL_COUNT)
    type_matches = tuple(item for item in TYPE_ASSEMBLIES if item.active_cells == type_spikes)
    type_assembly = type_matches[0] if len(type_matches) == 1 else None
    position = _one(head(CHILD_POSITION_OFFSET, MAX_CHILD_POSITION))
    occupancy_spike = _one(head(OCCUPANCY_OFFSET, len(OCCUPANCY_REGISTRY)))
    occupancy = OCCUPANCY_REGISTRY[occupancy_spike] if occupancy_spike is not None else ()
    operator = _one(head(RELATION_OPERATOR_OFFSET, len(RELATION_OPERATORS)))
    rhs_kind = _one(head(RHS_KIND_OFFSET, 2))
    right_slot = _one(head(RIGHT_SLOT_OFFSET, MAX_LOCAL_SLOT))
    value_spikes = head(VALUE_RELAY_OFFSET, VALUE_RELAY_CELL_COUNT)
    value_matches = tuple(
        (registered_event, cells)
        for registered_event, cells in protocol.typed_value_registry
        if cells == value_spikes
    )
    typed_value = value_matches[0][0] if len(value_matches) == 1 else None
    domain_kind = _one(head(DOMAIN_KIND_OFFSET, 2))
    class_cells = head(CLASS_OFFSET, CLASS_CELL_COUNT)
    signature_role = _one(head(SIGNATURE_ROLE_OFFSET, 2))

    valid = primary_status == READY and kind is not None
    if valid and kind in (EVENT_ROOT, EVENT_SOURCE):
        valid = (
            (event.raw_observation is None) != (event.typed_value_event is None)
            and secondary_status == STOP_ZERO
            and group_status == STOP_ZERO
            and local_slot is None
            and factor_kind is None
            and type_assembly is None
            and position is None
            and not occupancy
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and typed_value is None
            and domain_kind is None
            and not class_cells
            and signature_role is None
        )
    elif valid and kind == EVENT_EDGE:
        valid = (
            secondary_status == READY
            and group_status == STOP_ZERO
            and position is not None
            and bool(occupancy)
            and event.raw_observation is None
            and event.typed_value_event is None
            and local_slot is None
            and factor_kind is None
            and type_assembly is None
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and typed_value is None
            and domain_kind is None
            and not class_cells
            and signature_role is None
        )
    elif valid and kind == EVENT_FACTOR:
        valid = (
            secondary_status == STOP_ZERO
            and group_status == STOP_ZERO
            and local_slot is not None
            and factor_kind is not None
            and type_assembly is not None
            and position is None
            and not occupancy
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and typed_value is None
            and domain_kind is None
            and not class_cells
            and signature_role is None
        )
    elif valid and kind == EVENT_RELATION:
        rhs_valid = (
            rhs_kind == RHS_SLOT
            and right_slot is not None
            and typed_value is None
            and event.typed_value_event is None
        ) or (
            rhs_kind == RHS_VALUE
            and right_slot is None
            and typed_value is event.typed_value_event
            and typed_value is not None
        )
        valid = (
            secondary_status == STOP_ZERO
            and group_status == STOP_ZERO
            and local_slot is not None
            and operator is not None
            and rhs_valid
            and factor_kind is None
            and type_assembly is None
            and position is None
            and not occupancy
            and domain_kind is None
            and not class_cells
            and signature_role is None
        )
    elif valid and kind == EVENT_DOMAIN:
        valid = (
            secondary_status == STOP_ZERO
            and group_status == STOP_ZERO
            and local_slot is not None
            and factor_kind is None
            and type_assembly is TYPE_BY_NAME["Int"]
            and position is None
            and not occupancy
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and typed_value is event.typed_value_event
            and typed_value is not None
            and domain_kind == DOMAIN_LITERAL
            and class_cells == EXACT_INT_CLASS_CELLS
            and signature_role is None
        )
    elif valid and kind == EVENT_COPY:
        valid = (
            secondary_status == STOP_ZERO
            and group_status == READY
            and local_slot is not None
            and factor_kind is None
            and type_assembly is None
            and position is None
            and not occupancy
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and typed_value is None
            and domain_kind is None
            and not class_cells
            and signature_role is None
        )
    elif valid and kind == EVENT_SIGNATURE:
        valid = (
            secondary_status == STOP_ZERO
            and group_status == STOP_ZERO
            and local_slot is not None
            and factor_kind is None
            and type_assembly is not None
            and position is None
            and not occupancy
            and operator is None
            and rhs_kind is None
            and right_slot is None
            and typed_value is None
            and domain_kind is None
            and not class_cells
            and signature_role is not None
        )
    else:
        valid = False

    if valid:
        status = READY
    elif primary_status == STOP_ZERO and not spike_cells:
        status = STOP_ZERO
    else:
        status = STOP_AMBIGUOUS
    record = {
        "status": status,
        "event_sha256": event.sha256,
        "primary_carrier_cells": primary,
        "secondary_carrier_cells": secondary,
        "copy_group_cells": group,
        "event_kind_cell": kind,
        "local_slot": local_slot,
        "factor_kind_cell": factor_kind,
        "type_sha256": type_assembly.sha256 if type_assembly is not None else None,
        "child_position": position,
        "parent_slot_occupancy": occupancy,
        "relation_operator_cell": operator,
        "rhs_kind_cell": rhs_kind,
        "right_slot": right_slot,
        "typed_value_sha256": typed_value.sha256 if typed_value is not None else None,
        "value_relay_cells": value_spikes,
        "domain_kind_cell": domain_kind,
        "class_cells": class_cells,
        "signature_role_cell": signature_role,
        "spike_cells": spike_cells,
    }
    return _DecodedEvent(
        status,
        event,
        primary,
        secondary,
        group,
        kind,
        local_slot,
        factor_kind,
        type_assembly,
        position,
        occupancy,
        operator,
        rhs_kind,
        right_slot,
        typed_value,
        value_spikes,
        domain_kind,
        class_cells,
        signature_role,
        tuple(int(value) for value in raw),
        spike_cells,
        _digest("event-relay-receipt", record),
    )


def _lesson_exact(decoded: _DecodedEvent, lesson: _Lesson) -> bool:
    return (
        decoded.status == READY
        and decoded.primary_carrier_cells == lesson.primary_carrier_cells
        and decoded.secondary_carrier_cells == lesson.secondary_carrier_cells
        and decoded.copy_group_cells == lesson.copy_group_cells
        and decoded.event_kind_cell == lesson.event_kind_cell
        and decoded.local_slot == lesson.local_slot
        and decoded.factor_kind_cell == lesson.factor_kind_cell
        and (decoded.type_assembly.active_cells if decoded.type_assembly is not None else ())
        == lesson.type_cells
        and decoded.child_position == lesson.child_position
        and decoded.parent_slot_occupancy == lesson.parent_slot_occupancy
        and decoded.relation_operator_cell == lesson.relation_operator_cell
        and decoded.rhs_kind_cell == lesson.rhs_kind_cell
        and decoded.right_slot == lesson.right_slot
        and decoded.value_relay_cells == lesson.value_relay_cells
        and decoded.domain_kind_cell == lesson.domain_kind_cell
        and decoded.class_cells == lesson.class_cells
        and decoded.signature_role_cell == lesson.signature_role_cell
    )


class FactorSlotRelay(NamedTuple):
    carrier_cells: tuple[int, ...]
    local_slot: int
    factor_kind: str
    factor_kind_cell: int
    type_assembly: PhysicalTypeAssembly
    event_receipt_sha256: str
    sha256: str


class CarrierEdgeRelay(NamedTuple):
    parent_carrier_cells: tuple[int, ...]
    child_carrier_cells: tuple[int, ...]
    child_position: int
    parent_slot_occupancy: tuple[int, ...]
    event_receipt_sha256: str
    sha256: str


class RelationConstraintRelay(NamedTuple):
    carrier_cells: tuple[int, ...]
    left_slot: int
    operator: str
    operator_cell: int
    right_slot: int | None
    right_value_event: TypedValueEvent | None
    value_relay_cells: tuple[int, ...]
    event_receipt_sha256: str
    sha256: str


class DomainRelay(NamedTuple):
    carrier_cells: tuple[int, ...]
    local_slot: int
    domain_kind_cell: int
    type_assembly: PhysicalTypeAssembly
    typed_value_event: TypedValueEvent
    value_relay_cells: tuple[int, ...]
    class_cells: tuple[int, ...]
    event_receipt_sha256: str
    sha256: str


class LiteralCopyGroupRelay(NamedTuple):
    group_cells: tuple[int, ...]
    members: tuple[tuple[tuple[int, ...], int], ...]
    event_receipt_sha256s: tuple[str, ...]
    sha256: str


class SignatureTypeRelay(NamedTuple):
    root_carrier_cells: tuple[int, ...]
    signature_role: str
    signature_role_cell: int
    position: int
    type_assembly: PhysicalTypeAssembly
    event_receipt_sha256: str
    sha256: str


class CarrierSourceRelay(NamedTuple):
    carrier_cells: tuple[int, ...]
    raw_observation: RawSpanObservation | None
    typed_value_event: TypedValueEvent | None
    event_receipt_sha256: str
    sha256: str


class OccurrenceFactorRelayView(NamedTuple):
    root_carrier_cells: tuple[int, ...]
    occurrence_carriers: tuple[tuple[int, ...], ...]
    source_relays: tuple[CarrierSourceRelay, ...]
    edges: tuple[CarrierEdgeRelay, ...]
    factor_slots: tuple[FactorSlotRelay, ...]
    relations: tuple[RelationConstraintRelay, ...]
    domains: tuple[DomainRelay, ...]
    literal_copy_groups: tuple[LiteralCopyGroupRelay, ...]
    signature_relays: tuple[SignatureTypeRelay, ...]
    event_receipt_sha256s: tuple[str, ...]
    checkpoint_sha256: str
    protocol_sha256: str
    physical_work_receipt_sha256: str
    sha256: str


class CarrierBuildSummary(NamedTuple):
    status: str
    fresh_zero_state_nonzero_count: int
    fit_lesson_count: int
    fit_exact_count: int
    teacher_off_ready_count: int
    relation_operator_count: int
    rhs_kind_count: int
    treatment_event_count: int
    learned_state_bytes: int
    committed_excitatory_ticks: int
    committed_inhibitory_ticks: int
    checkpoint_sha256: str
    protocol_sha256: str
    sha256: str


def _checkpoint_payload(
    metadata: Mapping[str, object], excitatory: np.ndarray, inhibitory: np.ndarray
) -> bytes:
    return b"".join(
        (
            _canonical_json(dict(metadata)),
            b"\0",
            _array_frame("excitatory", excitatory),
            _array_frame("inhibitory", inhibitory),
        )
    )


def _assemble_view(
    consumer: FrozenOccurrenceFactorRelayV6,
    request_sha256: str,
    decisions: tuple[_DecodedEvent, ...],
) -> tuple[str, str | None, OccurrenceFactorRelayView | None]:
    if len(decisions) != 38:
        return STOP_UNSUPPORTED, "the frozen complete slice requires 38 physical events", None
    by_kind: dict[int, list[_DecodedEvent]] = defaultdict(list)
    for row in decisions:
        assert row.event_kind_cell is not None
        by_kind[row.event_kind_cell].append(row)
    expected_counts = {
        EVENT_ROOT: 1,
        EVENT_SOURCE: 7,
        EVENT_EDGE: 7,
        EVENT_FACTOR: 16,
        EVENT_RELATION: 1,
        EVENT_DOMAIN: 2,
        EVENT_COPY: 2,
        EVENT_SIGNATURE: 2,
    }
    if {key: len(by_kind.get(key, ())) for key in expected_counts} != expected_counts:
        return STOP_UNSUPPORTED, "the complete factor/edge/relation/domain shape changed", None

    root = by_kind[EVENT_ROOT][0].primary_carrier_cells
    source_rows = tuple((*by_kind[EVENT_ROOT], *by_kind[EVENT_SOURCE]))
    source_by_carrier: dict[tuple[int, ...], _DecodedEvent] = {}
    for row in source_rows:
        if row.primary_carrier_cells in source_by_carrier:
            return STOP_AMBIGUOUS, "one carrier acquired multiple source identities", None
        source_by_carrier[row.primary_carrier_cells] = row

    edge_relays: list[CarrierEdgeRelay] = []
    children: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    inbound: dict[tuple[int, ...], int] = defaultdict(int)
    edge_keys: set[tuple[tuple[int, ...], int]] = set()
    for row in by_kind[EVENT_EDGE]:
        assert row.child_position is not None
        key = (row.primary_carrier_cells, row.child_position)
        if key in edge_keys or row.primary_carrier_cells == row.secondary_carrier_cells:
            return STOP_AMBIGUOUS, "an edge/position carrier key was reused", None
        edge_keys.add(key)
        children[row.primary_carrier_cells].append(
            (row.child_position, row.secondary_carrier_cells)
        )
        inbound[row.secondary_carrier_cells] += 1
        record = (
            row.primary_carrier_cells,
            row.secondary_carrier_cells,
            row.child_position,
            row.parent_slot_occupancy,
            row.sha256,
        )
        edge_relays.append(CarrierEdgeRelay(*record, _digest("carrier-edge-relay", record)))
    carriers = set(source_by_carrier)
    if (
        len(carriers) != 8
        or inbound[root] != 0
        or any(inbound[carrier] != 1 for carrier in carriers if carrier != root)
    ):
        return STOP_AMBIGUOUS, "physical edge relays do not form one eight-node tree", None
    for parent, entries in children.items():
        positions = tuple(position for position, _child in sorted(entries))
        if positions != tuple(range(len(entries))):
            return STOP_AMBIGUOUS, "physical child positions are not contiguous", None
        if parent not in carriers or any(child not in carriers for _position, child in entries):
            return STOP_AMBIGUOUS, "edge endpoint lacks an exact source-bound carrier", None

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
        return STOP_AMBIGUOUS, "physical occurrence carrier tree contains a cycle", None
    if set(ordered_carriers) != carriers:
        return STOP_AMBIGUOUS, "physical occurrence carrier tree is disconnected", None
    occurrence_index = {carrier: index for index, carrier in enumerate(ordered_carriers)}
    actual_edges = tuple(
        sorted(
            (
                occurrence_index[row.parent_carrier_cells],
                occurrence_index[row.child_carrier_cells],
                row.child_position,
                row.parent_slot_occupancy,
            )
            for row in edge_relays
        )
    )
    if actual_edges != tuple(sorted(_EDGE_PATTERN)):
        return (
            STOP_UNSUPPORTED,
            "physical carrier tree is not the retained eight-occurrence shape",
            None,
        )

    factor_relays: list[FactorSlotRelay] = []
    factors_by_carrier: dict[tuple[int, ...], list[FactorSlotRelay]] = defaultdict(list)
    for row in by_kind[EVENT_FACTOR]:
        assert row.local_slot is not None
        assert row.factor_kind_cell is not None
        assert row.type_assembly is not None
        record = (
            row.primary_carrier_cells,
            row.local_slot,
            FACTOR_KINDS[row.factor_kind_cell],
            row.factor_kind_cell,
            row.type_assembly.sha256,
            row.sha256,
        )
        relay = FactorSlotRelay(
            row.primary_carrier_cells,
            row.local_slot,
            FACTOR_KINDS[row.factor_kind_cell],
            row.factor_kind_cell,
            row.type_assembly,
            row.sha256,
            _digest("factor-slot-relay", record),
        )
        factors_by_carrier[row.primary_carrier_cells].append(relay)
        factor_relays.append(relay)
    actual_factors: list[tuple[tuple[int, str], ...]] = []
    for carrier in ordered_carriers:
        relays = sorted(factors_by_carrier.get(carrier, ()), key=lambda item: item.local_slot)
        if tuple(item.local_slot for item in relays) != tuple(range(len(relays))):
            return STOP_AMBIGUOUS, "factor-slot relays are not an exact ordered tuple", None
        actual_factors.append(
            tuple((item.factor_kind_cell, item.type_assembly.value_type) for item in relays)
        )
    if tuple(actual_factors) != _FACTOR_PATTERN:
        return STOP_UNSUPPORTED, "factor-slot tuples do not match the retained generic shape", None
    for edge in edge_relays:
        parent_factors = factors_by_carrier[edge.parent_carrier_cells]
        child_factors = factors_by_carrier[edge.child_carrier_cells]
        if (
            len(edge.parent_slot_occupancy) != len(child_factors)
            or len(set(edge.parent_slot_occupancy)) != len(edge.parent_slot_occupancy)
            or any(not 0 <= slot < len(parent_factors) for slot in edge.parent_slot_occupancy)
        ):
            return (
                STOP_AMBIGUOUS,
                "edge occupancy does not bind the complete child factor tuple",
                None,
            )

    relation_relays: list[RelationConstraintRelay] = []
    for row in by_kind[EVENT_RELATION]:
        assert row.local_slot is not None
        assert row.relation_operator_cell is not None
        local_factors = factors_by_carrier[row.primary_carrier_cells]
        if not 0 <= row.local_slot < len(local_factors):
            return STOP_AMBIGUOUS, "relation left slot is outside its carrier tuple", None
        if row.rhs_kind_cell == RHS_SLOT:
            if row.right_slot is None or not 0 <= row.right_slot < len(local_factors):
                return STOP_AMBIGUOUS, "relation right slot is outside its carrier tuple", None
        elif row.rhs_kind_cell != RHS_VALUE or row.typed_value_event is None:
            return STOP_AMBIGUOUS, "relation value event is incomplete", None
        record = (
            row.primary_carrier_cells,
            row.local_slot,
            RELATION_OPERATORS[row.relation_operator_cell],
            row.relation_operator_cell,
            row.right_slot,
            row.typed_value_event.sha256 if row.typed_value_event is not None else None,
            row.value_relay_cells,
            row.sha256,
        )
        relation_relays.append(
            RelationConstraintRelay(
                row.primary_carrier_cells,
                row.local_slot,
                RELATION_OPERATORS[row.relation_operator_cell],
                row.relation_operator_cell,
                row.right_slot,
                row.typed_value_event,
                row.value_relay_cells,
                row.sha256,
                _digest("relation-constraint-relay", record),
            )
        )

    domain_relays: list[DomainRelay] = []
    domain_keys: set[tuple[tuple[int, ...], int]] = set()
    for row in by_kind[EVENT_DOMAIN]:
        assert row.local_slot is not None
        assert row.type_assembly is not None
        assert row.typed_value_event is not None
        key = (row.primary_carrier_cells, row.local_slot)
        local_factors = factors_by_carrier[row.primary_carrier_cells]
        if (
            key in domain_keys
            or not 0 <= row.local_slot < len(local_factors)
            or local_factors[row.local_slot].factor_kind_cell != FACTOR_LITERAL
        ):
            return STOP_AMBIGUOUS, "domain relay is not bound to one literal factor slot", None
        domain_keys.add(key)
        record = (
            row.primary_carrier_cells,
            row.local_slot,
            row.domain_kind_cell,
            row.type_assembly.sha256,
            row.typed_value_event.sha256,
            row.value_relay_cells,
            row.class_cells,
            row.sha256,
        )
        domain_relays.append(
            DomainRelay(
                row.primary_carrier_cells,
                row.local_slot,
                int(row.domain_kind_cell),
                row.type_assembly,
                row.typed_value_event,
                row.value_relay_cells,
                row.class_cells,
                row.sha256,
                _digest("domain-relay", record),
            )
        )

    copy_members: dict[tuple[int, ...], list[tuple[tuple[int, ...], int, str]]] = defaultdict(list)
    for row in by_kind[EVENT_COPY]:
        assert row.local_slot is not None
        copy_members[row.copy_group_cells].append(
            (row.primary_carrier_cells, row.local_slot, row.sha256)
        )
    terminal_carriers = {carrier for carrier in carriers if not children.get(carrier)}
    terminal_literal_refs = {
        (carrier, factor.local_slot)
        for carrier in terminal_carriers
        for factor in factors_by_carrier[carrier]
        if factor.factor_kind_cell == FACTOR_LITERAL
    }
    seen_refs: set[tuple[tuple[int, ...], int]] = set()
    copy_relays: list[LiteralCopyGroupRelay] = []
    for group, rows in sorted(copy_members.items()):
        members = tuple(sorted((carrier, slot) for carrier, slot, _receipt in rows))
        receipts = tuple(sorted(receipt for _carrier, _slot, receipt in rows))
        if seen_refs & set(members):
            return STOP_AMBIGUOUS, "literal copy member was assigned twice", None
        seen_refs.update(members)
        record = (group, members, receipts)
        copy_relays.append(
            LiteralCopyGroupRelay(
                group,
                members,
                receipts,
                _digest("literal-copy-group-relay", record),
            )
        )
    if seen_refs != terminal_literal_refs:
        return STOP_UNSUPPORTED, "literal-copy relays do not cover every terminal literal", None

    signature_relays: list[SignatureTypeRelay] = []
    signature_keys: set[tuple[int, int]] = set()
    for row in by_kind[EVENT_SIGNATURE]:
        assert row.signature_role_cell is not None
        assert row.local_slot is not None
        assert row.type_assembly is not None
        key = (row.signature_role_cell, row.local_slot)
        if row.primary_carrier_cells != root or key in signature_keys:
            return STOP_AMBIGUOUS, "signature type relay is not uniquely root-keyed", None
        signature_keys.add(key)
        role = "parameter" if row.signature_role_cell == SIGNATURE_PARAMETER else "return"
        record = (
            root,
            role,
            row.signature_role_cell,
            row.local_slot,
            row.type_assembly.sha256,
            row.sha256,
        )
        signature_relays.append(
            SignatureTypeRelay(
                root,
                role,
                row.signature_role_cell,
                row.local_slot,
                row.type_assembly,
                row.sha256,
                _digest("signature-type-relay", record),
            )
        )
    expected_signature = {
        (SIGNATURE_PARAMETER, 0, "List[Int]"),
        (SIGNATURE_RETURN, 0, "Int"),
    }
    if {
        (row.signature_role_cell, row.position, row.type_assembly.value_type)
        for row in signature_relays
    } != expected_signature:
        return STOP_UNSUPPORTED, "typed signature relays are incomplete", None

    source_relays: list[CarrierSourceRelay] = []
    for carrier in ordered_carriers:
        row = source_by_carrier[carrier]
        record = (
            carrier,
            row.event.raw_observation.sha256 if row.event.raw_observation is not None else None,
            row.event.typed_value_event.sha256 if row.event.typed_value_event is not None else None,
            row.sha256,
        )
        source_relays.append(
            CarrierSourceRelay(
                carrier,
                row.event.raw_observation,
                row.event.typed_value_event,
                row.sha256,
                _digest("carrier-source-relay", record),
            )
        )

    work_receipt = _digest(
        "physical-relay-work",
        {
            "request_sha256": request_sha256,
            "event_receipt_sha256s": tuple(row.sha256 for row in decisions),
            "score_count": len(decisions) * OUTPUT_CELL_COUNT,
            "fitted_glif_steps": len(decisions) * OUTPUT_STEPS,
        },
    )
    view_record = {
        "root_carrier_cells": root,
        "occurrence_carriers": tuple(ordered_carriers),
        "source_sha256s": tuple(item.sha256 for item in source_relays),
        "edge_sha256s": tuple(item.sha256 for item in edge_relays),
        "factor_sha256s": tuple(item.sha256 for item in factor_relays),
        "relation_sha256s": tuple(item.sha256 for item in relation_relays),
        "domain_sha256s": tuple(item.sha256 for item in domain_relays),
        "copy_sha256s": tuple(item.sha256 for item in copy_relays),
        "signature_sha256s": tuple(item.sha256 for item in signature_relays),
        "event_receipt_sha256s": tuple(sorted(row.sha256 for row in decisions)),
        "checkpoint_sha256": consumer.checkpoint_sha256,
        "protocol_sha256": consumer.protocol_sha256,
        "physical_work_receipt_sha256": work_receipt,
    }
    view = OccurrenceFactorRelayView(
        root,
        tuple(ordered_carriers),
        tuple(source_relays),
        tuple(sorted(edge_relays)),
        tuple(sorted(factor_relays)),
        tuple(sorted(relation_relays)),
        tuple(sorted(domain_relays)),
        tuple(sorted(copy_relays)),
        tuple(sorted(signature_relays)),
        tuple(sorted(row.sha256 for row in decisions)),
        consumer.checkpoint_sha256,
        consumer.protocol_sha256,
        work_receipt,
        _digest("occurrence-factor-relay-view", view_record),
    )
    return READY, None, view


class _CertificateReceipt(NamedTuple):
    authority: object
    consumer: FrozenOccurrenceFactorRelayV6
    status: str
    stop_reason: str | None
    request_sha256: str
    event_receipt_sha256s: tuple[str, ...]
    view: OccurrenceFactorRelayView | None
    sha256: str


_CERTIFICATE_AUTHORITY = object()
_ISSUED_CERTIFICATES: dict[int, OccurrenceFactorRelayCertificate] = {}


class OccurrenceFactorRelayCertificate:
    """Opaque exact-object authority over one V6 physical inference."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _CertificateReceipt) -> None:
        if receipt.authority is not _CERTIFICATE_AUTHORITY:
            raise TypeError("V6 relay certificate requires private authority")
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
    consumer: FrozenOccurrenceFactorRelayV6,
    *,
    status: str,
    stop_reason: str | None,
    request_sha256: str,
    decisions: tuple[_DecodedEvent, ...],
    view: OccurrenceFactorRelayView | None,
) -> OccurrenceFactorRelayCertificate:
    payload = {
        "status": status,
        "stop_reason": stop_reason,
        "request_sha256": request_sha256,
        "event_receipt_sha256s": tuple(row.sha256 for row in decisions),
        "view_sha256": view.sha256 if view is not None else None,
        "checkpoint_sha256": consumer.checkpoint_sha256,
        "protocol_sha256": consumer.protocol_sha256,
    }
    receipt = _CertificateReceipt(
        _CERTIFICATE_AUTHORITY,
        consumer,
        status,
        stop_reason,
        request_sha256,
        tuple(row.sha256 for row in decisions),
        view,
        _digest("opaque-certificate", payload),
    )
    certificate = OccurrenceFactorRelayCertificate(receipt)
    _ISSUED_CERTIFICATES[id(certificate)] = certificate
    return certificate


def _authenticate_certificate(
    certificate: OccurrenceFactorRelayCertificate,
) -> _CertificateReceipt:
    if (
        type(certificate) is not OccurrenceFactorRelayCertificate
        or _ISSUED_CERTIFICATES.get(id(certificate)) is not certificate
        or certificate._receipt.authority is not _CERTIFICATE_AUTHORITY
    ):
        raise TypeError("V6 relay certificate is not an issued exact object")
    receipt = certificate._receipt
    payload = {
        "status": receipt.status,
        "stop_reason": receipt.stop_reason,
        "request_sha256": receipt.request_sha256,
        "event_receipt_sha256s": receipt.event_receipt_sha256s,
        "view_sha256": receipt.view.sha256 if receipt.view is not None else None,
        "checkpoint_sha256": receipt.consumer.checkpoint_sha256,
        "protocol_sha256": receipt.consumer.protocol_sha256,
    }
    if receipt.sha256 != _digest("opaque-certificate", payload):
        raise RuntimeError("V6 relay certificate receipt changed")
    return receipt


class FrozenOccurrenceFactorRelayV6:
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
        protocol: FrozenOccurrenceFactorProtocolV6,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        metadata: Mapping[str, object],
        checkpoint_sha256: str,
        build_summary: CarrierBuildSummary,
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
            or self._protocol.protocol_sha256 != _digest("protocol", self._protocol.root_record())
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
            raise RuntimeError("frozen V6 occurrence-factor checkpoint changed")

    def infer(
        self,
        events: Iterable[PhysicalRelayEvent],
    ) -> OccurrenceFactorRelayCertificate:
        self._authenticate_checkpoint()
        supplied = tuple(events)
        if any(type(item) is not PhysicalRelayEvent for item in supplied):
            raise TypeError("V6 inference accepts only physical relay events")
        ordered = tuple(sorted(supplied, key=lambda item: item.sha256))
        request_sha = _digest("inference-request", tuple(item.sha256 for item in ordered))
        if not ordered:
            return _issue_certificate(
                self,
                status=STOP_ZERO,
                stop_reason="no physical relay event was supplied",
                request_sha256=request_sha,
                decisions=(),
                view=None,
            )
        if len({item.sha256 for item in ordered}) != len(ordered):
            return _issue_certificate(
                self,
                status=STOP_AMBIGUOUS,
                stop_reason="a physical relay event was duplicated",
                request_sha256=request_sha,
                decisions=(),
                view=None,
            )
        decisions = tuple(
            _decode_event(self._protocol, self._excitatory, self._inhibitory, event)
            for event in ordered
        )
        if any(row.status == STOP_AMBIGUOUS for row in decisions):
            return _issue_certificate(
                self,
                status=STOP_AMBIGUOUS,
                stop_reason="one physical event activated conflicting relay fields",
                request_sha256=request_sha,
                decisions=decisions,
                view=None,
            )
        if any(row.status != READY for row in decisions):
            return _issue_certificate(
                self,
                status=STOP_ZERO,
                stop_reason="one physical event had no learned V6 relay join",
                request_sha256=request_sha,
                decisions=decisions,
                view=None,
            )
        status, reason, view = _assemble_view(self, request_sha, decisions)
        self._authenticate_checkpoint()
        return _issue_certificate(
            self,
            status=status,
            stop_reason=reason,
            request_sha256=request_sha,
            decisions=decisions,
            view=view,
        )


def build_frozen_occurrence_factor_relays_v6(
    protocol: FrozenOccurrenceFactorProtocolV6 = PROTOCOL,
) -> FrozenOccurrenceFactorRelayV6:
    if protocol is not PROTOCOL or protocol.protocol_sha256 != _digest(
        "protocol", protocol.root_record()
    ):
        raise RuntimeError("V6 occurrence-factor protocol authentication failed")
    fresh_e = np.zeros((OUTPUT_CELL_COUNT, INPUT_CELL_COUNT), dtype=TICK_DTYPE)
    fresh_h = np.zeros_like(fresh_e)
    off_e, off_h, _off_committed_e, _off_committed_h = _train(protocol, teacher_gain=0)
    treatment_e, treatment_h, committed_e, committed_h = _train(protocol, teacher_gain=1)
    metadata = {
        "schema": SCHEMA,
        "protocol_sha256": protocol.protocol_sha256,
        "array_shape": treatment_e.shape,
        "array_dtype": treatment_e.dtype.str,
        "tick_cap": TICK_CAP,
        "score_threshold": SCORE_THRESHOLD,
        "output_current_pa_hex": OUTPUT_CURRENT_PA.hex(),
        "output_steps": OUTPUT_STEPS,
        "fresh_weights_only": True,
        "fit_pass_count": 1,
        "v5_checkpoint_reads": 0,
        "public_train_input_count": PUBLIC_TRAIN_INPUT_COUNT,
        "canonical_input_count": CANONICAL_INPUT_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
    }
    checkpoint = sha256(_checkpoint_payload(metadata, treatment_e, treatment_h)).hexdigest()
    fit_decisions = tuple(
        _decode_event(protocol, treatment_e, treatment_h, lesson.event)
        for lesson in protocol.lessons
    )
    fit_exact = sum(
        _lesson_exact(decision, lesson)
        for decision, lesson in zip(fit_decisions, protocol.lessons, strict=True)
    )
    off_ready = sum(
        _decode_event(protocol, off_e, off_h, lesson.event).status == READY
        for lesson in protocol.lessons
    )
    summary_record = {
        "status": READY if fit_exact == len(protocol.lessons) and off_ready == 0 else STOP_ZERO,
        "fresh_zero_state_nonzero_count": int(
            np.count_nonzero(fresh_e) + np.count_nonzero(fresh_h)
        ),
        "fit_lesson_count": len(protocol.lessons),
        "fit_exact_count": fit_exact,
        "teacher_off_ready_count": off_ready,
        "relation_operator_count": len(RELATION_OPERATORS),
        "rhs_kind_count": 2,
        "treatment_event_count": len(protocol.treatment_events),
        "learned_state_bytes": treatment_e.nbytes + treatment_h.nbytes,
        "committed_excitatory_ticks": committed_e,
        "committed_inhibitory_ticks": committed_h,
        "checkpoint_sha256": checkpoint,
        "protocol_sha256": protocol.protocol_sha256,
    }
    summary = CarrierBuildSummary(
        **summary_record,
        sha256=_digest("build-summary", summary_record),
    )
    if summary.status != READY:
        raise RuntimeError(
            "V6 occurrence-factor self-posed fit stopped: "
            f"{fit_exact}/{len(protocol.lessons)} exact, {off_ready} teacher-off READY"
        )
    treatment_e.setflags(write=False)
    treatment_h.setflags(write=False)
    return FrozenOccurrenceFactorRelayV6(
        protocol,
        treatment_e,
        treatment_h,
        metadata,
        checkpoint,
        summary,
    )


STATIC_CONSUMER = build_frozen_occurrence_factor_relays_v6()


def infer_occurrence_factor_relays_v6(
    events: Iterable[PhysicalRelayEvent],
) -> OccurrenceFactorRelayCertificate:
    """Public deep seam: unordered physical events to opaque V6 authority."""

    return STATIC_CONSUMER.infer(events)


def require_occurrence_factor_relays_v6(
    certificate: OccurrenceFactorRelayCertificate,
) -> OccurrenceFactorRelayView:
    """Authenticate a READY exact object and expose its physical relay view."""

    receipt = _authenticate_certificate(certificate)
    receipt.consumer._authenticate_checkpoint()
    if receipt.status != READY or receipt.view is None:
        raise RuntimeError(receipt.stop_reason or receipt.status)
    return receipt.view


__all__ = [
    "CANONICAL_INPUT_COUNT",
    "DEV_INPUT_COUNT",
    "DOMAIN_CLASS",
    "DOMAIN_LITERAL",
    "FACTOR_KINDS",
    "HELDOUT_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PROTOCOL",
    "READY",
    "RELATION_OPERATORS",
    "SCHEMA",
    "SELF_POSED_COUNTERFACTUAL_EVENTS",
    "SELF_POSED_TREATMENT_EVENTS",
    "STATIC_CONSUMER",
    "STOP_AMBIGUOUS",
    "STOP_UNSUPPORTED",
    "STOP_ZERO",
    "OccurrenceFactorRelayCertificate",
    "OccurrenceFactorRelayView",
    "PhysicalRelayEvent",
    "TypedValueEvent",
    "build_frozen_occurrence_factor_relays_v6",
    "infer_occurrence_factor_relays_v6",
    "require_occurrence_factor_relays_v6",
]
