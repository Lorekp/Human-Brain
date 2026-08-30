"""Fresh self-posed occurrence-slot carrier learner and frozen consumer.

The module closes one deliberately narrow seam left by prompt selection V4.  An
unordered bundle of physical event assemblies is decoded by fresh
teacher-gated weights into stable distributed node-occurrence carriers.  Edge
relays publish two carriers, a child-position relay, and an exact generic
parent-factor-slot occupancy relay.  Non-edge relays bind their learned carrier
to either an exact raw span observation or an exact typed-literal event.

Inference never accepts a structural certificate, semantic production string,
path, depth, task/family identity, program, desired value, or property row.
Graph assembly uses equality of learned carrier assemblies only; input order is
canonicalized away.  The first frozen slice intentionally supports exactly one
root, one nested node, and two leaves.  Missing joins stop and conflicting joins
remain explicit ambiguity.
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

SCHEMA = "PROMPT_OCCURRENCE_SLOT_CARRIER_V5"
READY = "PROMPT_OCCURRENCE_SLOT_CARRIER_V5_READY"
STOP_ZERO = "STOP_PROMPT_OCCURRENCE_SLOT_CARRIER_V5_ZERO_JOIN"
STOP_UNSUPPORTED = "STOP_PROMPT_OCCURRENCE_SLOT_CARRIER_V5_UNSUPPORTED_SHAPE"
STOP_AMBIGUOUS = "STOP_PROMPT_OCCURRENCE_SLOT_CARRIER_V5_AMBIGUOUS_JOIN"

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

EVENT_CELL_COUNT = 512
INPUT_CELL_COUNT = 1_024
FEATURE_ACTIVE_CELL_COUNT = 12
TRACE_ACTIVE_CELL_COUNT = 24
ROLE_INPUT_CELL_COUNT = 32
NODE_INPUT_OFFSET = ROLE_INPUT_CELL_COUNT
NODE_INPUT_CELL_COUNT = 160
SOURCE_INPUT_OFFSET = NODE_INPUT_OFFSET + NODE_INPUT_CELL_COUNT
SOURCE_INPUT_CELL_COUNT = EVENT_CELL_COUNT - SOURCE_INPUT_OFFSET

CARRIER_CELL_COUNT = 256
CARRIER_ACTIVE_CELL_COUNT = 12
ROLE_ROOT = 0
ROLE_EDGE = 1
ROLE_ORIENTATION = 2
ROLE_PARAMETER = 3
ROLE_LITERAL_SLOT = 4
ROLE_LITERAL_COPY = 5
ROLE_CELL_COUNT = 6
MAX_CHILD_POSITION = 3
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
ROLE_OUTPUT_OFFSET = SECONDARY_OFFSET + CARRIER_CELL_COUNT
POSITION_OUTPUT_OFFSET = ROLE_OUTPUT_OFFSET + ROLE_CELL_COUNT
OCCUPANCY_OUTPUT_OFFSET = POSITION_OUTPUT_OFFSET + MAX_CHILD_POSITION
OUTPUT_CELL_COUNT = OCCUPANCY_OUTPUT_OFFSET + len(OCCUPANCY_REGISTRY)

TICK_CAP = 4
SCORE_THRESHOLD = 7
OUTPUT_CURRENT_PA = 5_000.0
OUTPUT_STEPS = 30
TICK_DTYPE = np.dtype(np.uint8)

_NAMESPACE = b"prompt-occurrence-slot-carrier-v5\0"
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


def _source_feature_cells(source_sha256: str) -> tuple[int, ...]:
    return _sparse_cells(
        "source-feature",
        source_sha256,
        offset=SOURCE_INPUT_OFFSET,
        width=SOURCE_INPUT_CELL_COUNT,
        count=6,
    )


def _event_trace_cells(event: PhysicalOccurrenceEvent) -> tuple[int, ...]:
    """Generic conjunction projection over physical cells and source identity."""

    return _sparse_cells(
        "event-conjunction-trace",
        event.record(),
        offset=0,
        width=INPUT_CELL_COUNT,
        count=TRACE_ACTIVE_CELL_COUNT,
    )


@dataclass(frozen=True, slots=True)
class TypedLiteralEvent:
    """Exact physical Int-copy event; the integer value is not a field."""

    canonical_token_index: int
    typed_output_active_cells: tuple[int, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        active = tuple(self.typed_output_active_cells)
        int_start = output_cells.INT_OFFSET
        int_stop = int_start + output_cells.INT_KNOWN_CELL_COUNT
        if (
            type(self.canonical_token_index) is not int
            or self.canonical_token_index < 0
            or len(active) != 2
            or active[0] != output_cells.TYPE_OFFSET
            or not int_start <= active[1] < int_stop
            or tuple(sorted(active)) != active
        ):
            raise ValueError("typed literal event must be one exact known Int cell pair")
        object.__setattr__(self, "typed_output_active_cells", active)
        object.__setattr__(
            self,
            "sha256",
            _digest(
                "typed-literal-event",
                {
                    "canonical_token_index": self.canonical_token_index,
                    "typed_output_active_cells": active,
                },
            ),
        )

    def record(self) -> dict[str, object]:
        return {
            "canonical_token_index": self.canonical_token_index,
            "typed_output_active_cells": self.typed_output_active_cells,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class PhysicalOccurrenceEvent:
    """One label-free physical event assembly and its exact optional source."""

    feature_cells: tuple[int, ...]
    raw_observation: RawSpanObservation | None = None
    typed_literal_event: TypedLiteralEvent | None = None
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        cells = tuple(self.feature_cells)
        if (
            len(cells) != FEATURE_ACTIVE_CELL_COUNT
            or tuple(sorted(cells)) != cells
            or len(set(cells)) != len(cells)
            or any(type(cell) is not int or not 0 <= cell < EVENT_CELL_COUNT for cell in cells)
            or sum(cell < NODE_INPUT_OFFSET for cell in cells) != 1
            or sum(NODE_INPUT_OFFSET <= cell < SOURCE_INPUT_OFFSET for cell in cells) != 5
            or sum(cell >= SOURCE_INPUT_OFFSET for cell in cells) != 6
            or (self.raw_observation is not None and self.typed_literal_event is not None)
        ):
            raise ValueError("physical occurrence event geometry changed")
        source = self.raw_observation or self.typed_literal_event
        if source is not None and tuple(cell for cell in cells if cell >= SOURCE_INPUT_OFFSET) != (
            _source_feature_cells(source.sha256)
        ):
            raise ValueError("physical event is not bound to its exact source identity")
        source_kind = (
            0
            if self.raw_observation is not None
            else 1
            if self.typed_literal_event is not None
            else None
        )
        source_sha = source.sha256 if source is not None else None
        object.__setattr__(self, "feature_cells", cells)
        object.__setattr__(
            self,
            "sha256",
            _digest(
                "physical-occurrence-event",
                {
                    "feature_cells": cells,
                    "source_kind_cell": source_kind,
                    "source_sha256": source_sha,
                },
            ),
        )

    def record(self) -> dict[str, object]:
        return {
            "feature_cells": self.feature_cells,
            "raw_observation": (
                self.raw_observation.record(include_sha256=True)
                if self.raw_observation is not None
                else None
            ),
            "typed_literal_event": (
                self.typed_literal_event.record()
                if self.typed_literal_event is not None
                else None
            ),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _Lesson:
    event: PhysicalOccurrenceEvent
    primary_carrier_cells: tuple[int, ...]
    secondary_carrier_cells: tuple[int, ...]
    role_cell: int
    child_position: int | None
    parent_factor_slot_occupancy: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "event": self.event.record(),
            "primary_carrier_cells": self.primary_carrier_cells,
            "secondary_carrier_cells": self.secondary_carrier_cells,
            "role_cell": self.role_cell,
            "child_position": self.child_position,
            "parent_factor_slot_occupancy": self.parent_factor_slot_occupancy,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FrozenOccurrenceSlotProtocolV5:
    lessons: tuple[_Lesson, ...]
    treatment_events: tuple[PhysicalOccurrenceEvent, ...]
    counterfactual_events: tuple[PhysicalOccurrenceEvent, ...]
    counterfactual_pairs: tuple[tuple[str, str], ...]
    carrier_registry: tuple[tuple[int, ...], ...]
    protocol_sha256: str

    def root_record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "lesson_sha256s": tuple(item.sha256 for item in self.lessons),
            "treatment_event_sha256s": tuple(item.sha256 for item in self.treatment_events),
            "counterfactual_event_sha256s": tuple(
                item.sha256 for item in self.counterfactual_events
            ),
            "counterfactual_pairs": self.counterfactual_pairs,
            "carrier_registry": self.carrier_registry,
            "generic_parent_factor_slot_occupancy": True,
            "event_order_neural_evidence": False,
            "path_or_depth_neural_evidence": False,
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


def _literal_event(token_index: int, value: int) -> TypedLiteralEvent:
    encoded = output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=output_cells.OutputType.INT,
    )
    if not encoded.supported:
        raise RuntimeError("self-posed literal left the compact Int bank")
    return TypedLiteralEvent(token_index, encoded.active_global_cell_indices())


def _raw_observation(key: str, ordinal: int) -> RawSpanObservation:
    assembly = _sparse_cells(
        "raw-span-assembly",
        key,
        offset=0,
        width=512,
        count=12,
    )
    direction = "NEXT" if ordinal % 2 == 0 else "PREV"
    cue = ordinal % 4
    selected = (cue + 1) % 4 if direction == "NEXT" else (cue - 1) % 4
    return RawSpanObservation(direction, cue, selected, assembly)


def _allocate_disjoint(
    labels: tuple[str, ...],
    *,
    width: int,
    count: int,
    namespace: str,
) -> dict[str, tuple[int, ...]]:
    used: set[int] = set()
    allocated: dict[str, tuple[int, ...]] = {}
    for label in labels:
        counter = 0
        cells: set[int] = set()
        while len(cells) < count:
            for cell in _sparse_cells(
                namespace,
                (label, counter),
                offset=0,
                width=width,
                count=count,
            ):
                if cell not in used:
                    cells.add(cell)
                    if len(cells) == count:
                        break
            counter += 1
        frozen = tuple(sorted(cells))
        used.update(frozen)
        allocated[label] = frozen
    return allocated


def _make_protocol() -> FrozenOccurrenceSlotProtocolV5:
    graph_names = ("alpha", "beta")
    node_labels = tuple(
        f"{graph}:{node}"
        for graph in graph_names
        for node in ("root", "nested", "parameter", "literal")
    )
    node_cues = _allocate_disjoint(
        node_labels,
        width=NODE_INPUT_CELL_COUNT,
        count=5,
        namespace="node-occurrence-cue",
    )
    carriers = _allocate_disjoint(
        node_labels,
        width=CARRIER_CELL_COUNT,
        count=CARRIER_ACTIVE_CELL_COUNT,
        namespace="teacher-carrier",
    )

    def support_event(
        role: int,
        label: str,
        source: RawSpanObservation | TypedLiteralEvent,
    ) -> PhysicalOccurrenceEvent:
        cells = (
            role,
            *(NODE_INPUT_OFFSET + cell for cell in node_cues[label]),
            *_source_feature_cells(source.sha256),
        )
        return PhysicalOccurrenceEvent(
            tuple(sorted(cells)),
            raw_observation=source if isinstance(source, RawSpanObservation) else None,
            typed_literal_event=source if isinstance(source, TypedLiteralEvent) else None,
        )

    def edge_event(parent: str, child: str) -> PhysicalOccurrenceEvent:
        peer = _sparse_cells(
            "edge-child-occurrence-cue",
            child,
            offset=SOURCE_INPUT_OFFSET,
            width=SOURCE_INPUT_CELL_COUNT,
            count=6,
        )
        cells = (
            ROLE_EDGE,
            *(NODE_INPUT_OFFSET + cell for cell in node_cues[parent]),
            *peer,
        )
        return PhysicalOccurrenceEvent(tuple(sorted(cells)))

    lesson_rows: list[_Lesson] = []
    graph_events: dict[str, tuple[PhysicalOccurrenceEvent, ...]] = {}
    for graph_index, graph in enumerate(graph_names):
        root = f"{graph}:root"
        nested = f"{graph}:nested"
        parameter = f"{graph}:parameter"
        literal = f"{graph}:literal"
        root_source = _raw_observation(f"{graph}:root-source", graph_index * 3)
        nested_source = _raw_observation(f"{graph}:nested-source", graph_index * 3 + 1)
        parameter_source = _raw_observation(f"{graph}:parameter-source", graph_index * 3 + 2)
        literal_source = _literal_event(17 + graph_index, -2 if graph_index == 0 else 3)
        specifications = (
            (support_event(ROLE_ROOT, root, root_source), root, None, ROLE_ROOT, None, ()),
            (edge_event(root, nested), root, nested, ROLE_EDGE, 0, (0, 1, 2)),
            (edge_event(nested, parameter), nested, parameter, ROLE_EDGE, 0, (0,)),
            (edge_event(nested, literal), nested, literal, ROLE_EDGE, 1, (1,)),
            (
                support_event(ROLE_ORIENTATION, root, root_source),
                root,
                None,
                ROLE_ORIENTATION,
                None,
                (),
            ),
            (
                support_event(ROLE_ORIENTATION, nested, nested_source),
                nested,
                None,
                ROLE_ORIENTATION,
                None,
                (),
            ),
            (
                support_event(ROLE_PARAMETER, parameter, parameter_source),
                parameter,
                None,
                ROLE_PARAMETER,
                None,
                (),
            ),
            (
                support_event(ROLE_LITERAL_SLOT, literal, literal_source),
                literal,
                None,
                ROLE_LITERAL_SLOT,
                None,
                (),
            ),
            (
                support_event(ROLE_LITERAL_COPY, literal, literal_source),
                literal,
                None,
                ROLE_LITERAL_COPY,
                None,
                (),
            ),
        )
        events: list[PhysicalOccurrenceEvent] = []
        for event, primary, secondary, role, position, occupancy in specifications:
            record = {
                "event": event.record(),
                "primary_carrier_cells": carriers[primary],
                "secondary_carrier_cells": carriers[secondary] if secondary else (),
                "role_cell": role,
                "child_position": position,
                "parent_factor_slot_occupancy": occupancy,
            }
            lesson_rows.append(
                _Lesson(
                    event,
                    carriers[primary],
                    carriers[secondary] if secondary else (),
                    role,
                    position,
                    occupancy,
                    _digest("protocol-lesson", record),
                )
            )
            events.append(event)
        graph_events[graph] = tuple(events)

    lessons = tuple(lesson_rows)
    pairs = tuple(
        (lessons[index].sha256, lessons[index + 9].sha256) for index in range(9)
    )
    carrier_registry = tuple(sorted(carriers.values()))
    partial = FrozenOccurrenceSlotProtocolV5(
        lessons,
        graph_events["alpha"],
        graph_events["beta"],
        pairs,
        carrier_registry,
        "",
    )
    protocol = FrozenOccurrenceSlotProtocolV5(
        **{
            field_name: getattr(partial, field_name)
            for field_name in partial.__dataclass_fields__
            if field_name != "protocol_sha256"
        },
        protocol_sha256=_digest("protocol", partial.root_record()),
    )
    _audit_protocol(protocol)
    return protocol


def _audit_protocol(protocol: FrozenOccurrenceSlotProtocolV5) -> None:
    if (
        len(protocol.lessons) != 18
        or len(protocol.treatment_events) != 9
        or len(protocol.counterfactual_events) != 9
        or len(protocol.counterfactual_pairs) != 9
        or len({row.event.sha256 for row in protocol.lessons}) != 18
        or any(
            protocol.lessons[index].role_cell
            != protocol.lessons[index + 9].role_cell
            for index in range(9)
        )
        or any(
            protocol.lessons[index].primary_carrier_cells
            == protocol.lessons[index + 9].primary_carrier_cells
            for index in range(9)
        )
        or protocol.protocol_sha256 != _digest("protocol", protocol.root_record())
    ):
        raise RuntimeError("occurrence-slot V5 protocol audit changed")
    endpoint_carriers = {
        carrier
        for lesson in protocol.lessons
        if lesson.role_cell == ROLE_EDGE
        for carrier in (lesson.primary_carrier_cells, lesson.secondary_carrier_cells)
    }
    support_carriers = {
        lesson.primary_carrier_cells
        for lesson in protocol.lessons
        if lesson.role_cell != ROLE_EDGE
    }
    if endpoint_carriers != support_carriers or endpoint_carriers != set(
        protocol.carrier_registry
    ):
        raise RuntimeError("stable carrier reuse is not total across event roles")


PROTOCOL = _make_protocol()
SELF_POSED_TREATMENT_EVENTS = PROTOCOL.treatment_events
SELF_POSED_COUNTERFACTUAL_EVENTS = PROTOCOL.counterfactual_events


class CarrierSourceBinding(NamedTuple):
    carrier_cells: tuple[int, ...]
    source_kind_cell: int
    raw_observation: RawSpanObservation | None
    typed_literal_event: TypedLiteralEvent | None
    event_receipt_sha256s: tuple[str, ...]
    sha256: str


class CarrierEdge(NamedTuple):
    parent_carrier_cells: tuple[int, ...]
    child_carrier_cells: tuple[int, ...]
    child_position: int
    parent_factor_slot_occupancy: tuple[int, ...]
    relay_receipt_sha256: str
    sha256: str


class OccurrenceSlotCarrierView(NamedTuple):
    root_carrier_cells: tuple[int, ...]
    source_bindings: tuple[CarrierSourceBinding, ...]
    edges: tuple[CarrierEdge, ...]
    event_relay_receipt_sha256s: tuple[str, ...]
    checkpoint_sha256: str
    protocol_sha256: str
    physical_work_receipt_sha256: str
    sha256: str


class CarrierBuildSummary(NamedTuple):
    status: str
    fresh_zero_state_nonzero_count: int
    fit_lesson_count: int
    fit_exact_count: int
    teacher_off_nonzero_count: int
    teacher_off_ready_count: int
    committed_excitatory_ticks: int
    committed_inhibitory_ticks: int
    learned_state_bytes: int
    checkpoint_sha256: str
    protocol_sha256: str
    sha256: str


class _DecodedEvent(NamedTuple):
    status: str
    event: PhysicalOccurrenceEvent
    primary_carrier_cells: tuple[int, ...]
    secondary_carrier_cells: tuple[int, ...]
    role_cell: int | None
    child_position: int | None
    parent_factor_slot_occupancy: tuple[int, ...]
    raw_scores: tuple[int, ...]
    spike_cells: tuple[int, ...]
    sha256: str


def _head_rows(protocol: FrozenOccurrenceSlotProtocolV5) -> tuple[tuple[int, ...], ...]:
    carrier_cells = tuple(sorted({cell for row in protocol.carrier_registry for cell in row}))
    return (
        tuple(PRIMARY_OFFSET + cell for cell in carrier_cells),
        tuple(SECONDARY_OFFSET + cell for cell in carrier_cells),
        tuple(ROLE_OUTPUT_OFFSET + cell for cell in range(ROLE_CELL_COUNT)),
        tuple(POSITION_OUTPUT_OFFSET + cell for cell in range(MAX_CHILD_POSITION)),
        tuple(OCCUPANCY_OUTPUT_OFFSET + cell for cell in range(len(OCCUPANCY_REGISTRY))),
    )


def _lesson_target_rows(lesson: _Lesson) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(PRIMARY_OFFSET + cell for cell in lesson.primary_carrier_cells),
        tuple(SECONDARY_OFFSET + cell for cell in lesson.secondary_carrier_cells),
        (ROLE_OUTPUT_OFFSET + lesson.role_cell,),
        (
            (POSITION_OUTPUT_OFFSET + lesson.child_position,)
            if lesson.child_position is not None
            else ()
        ),
        (
            (
                OCCUPANCY_OUTPUT_OFFSET
                + OCCUPANCY_REGISTRY.index(lesson.parent_factor_slot_occupancy),
            )
            if lesson.parent_factor_slot_occupancy
            else ()
        ),
    )


def _train(
    protocol: FrozenOccurrenceSlotProtocolV5,
    *,
    teacher_gain: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    excitatory = np.zeros((OUTPUT_CELL_COUNT, INPUT_CELL_COUNT), dtype=TICK_DTYPE)
    inhibitory = np.zeros_like(excitatory)
    if teacher_gain not in (0, 1):
        raise ValueError("teacher gain must be physically off or on")
    if not teacher_gain:
        return excitatory, inhibitory, 0, 0
    head_rows = _head_rows(protocol)
    committed_e = committed_h = 0
    for lesson in protocol.lessons:
        active = np.asarray(_event_trace_cells(lesson.event), dtype=np.intp)
        for registered, target in zip(head_rows, _lesson_target_rows(lesson), strict=True):
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


def _carrier_from_spikes(
    spikes: tuple[int, ...],
    registry: tuple[tuple[int, ...], ...],
) -> tuple[str, tuple[int, ...]]:
    if not spikes:
        return STOP_ZERO, ()
    matches = tuple(candidate for candidate in registry if candidate == spikes)
    if len(matches) == 1:
        return READY, matches[0]
    return STOP_AMBIGUOUS, ()


def _decode_event(
    protocol: FrozenOccurrenceSlotProtocolV5,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    event: PhysicalOccurrenceEvent,
    *,
    cell_gain: float = 1.0,
) -> _DecodedEvent:
    active = np.asarray(_event_trace_cells(event), dtype=np.intp)
    raw_array = np.sum(
        excitatory[:, active].astype(np.int16) - inhibitory[:, active].astype(np.int16),
        axis=1,
        dtype=np.int64,
    )
    admitted = np.greater_equal(raw_array, SCORE_THRESHOLD)
    currents = admitted.astype(np.float64) * OUTPUT_CURRENT_PA * cell_gain
    spike_mask = np.asarray(_SPIKE_KERNEL.accumulated_spike_mask(currents), dtype=np.bool_)
    spike_cells = tuple(int(cell) for cell in np.flatnonzero(spike_mask))
    primary_spikes = tuple(
        cell for cell in spike_cells if PRIMARY_OFFSET <= cell < SECONDARY_OFFSET
    )
    secondary_spikes = tuple(
        cell - SECONDARY_OFFSET
        for cell in spike_cells
        if SECONDARY_OFFSET <= cell < ROLE_OUTPUT_OFFSET
    )
    primary_status, primary = _carrier_from_spikes(primary_spikes, protocol.carrier_registry)
    secondary_status, secondary = _carrier_from_spikes(
        secondary_spikes,
        protocol.carrier_registry,
    )
    role_spikes = tuple(
        cell - ROLE_OUTPUT_OFFSET
        for cell in spike_cells
        if ROLE_OUTPUT_OFFSET <= cell < POSITION_OUTPUT_OFFSET
    )
    position_spikes = tuple(
        cell - POSITION_OUTPUT_OFFSET
        for cell in spike_cells
        if POSITION_OUTPUT_OFFSET <= cell < OCCUPANCY_OUTPUT_OFFSET
    )
    occupancy_spikes = tuple(
        cell - OCCUPANCY_OUTPUT_OFFSET
        for cell in spike_cells
        if cell >= OCCUPANCY_OUTPUT_OFFSET
    )
    if primary_status != READY or len(role_spikes) != 1:
        status = (
            STOP_ZERO
            if primary_status == STOP_ZERO and not role_spikes and not spike_cells
            else STOP_AMBIGUOUS
        )
        role = position = None
        occupancy: tuple[int, ...] = ()
        secondary = ()
    else:
        role = role_spikes[0]
        if role == ROLE_EDGE:
            if (
                secondary_status != READY
                or len(position_spikes) != 1
                or len(occupancy_spikes) != 1
                or event.raw_observation is not None
                or event.typed_literal_event is not None
            ):
                status = STOP_AMBIGUOUS
                position = None
                occupancy = ()
            else:
                status = READY
                position = position_spikes[0]
                occupancy = OCCUPANCY_REGISTRY[occupancy_spikes[0]]
        elif (
            secondary_spikes
            or position_spikes
            or occupancy_spikes
            or secondary_status == READY
            or (event.raw_observation is None and event.typed_literal_event is None)
        ):
            status = STOP_AMBIGUOUS
            position = None
            occupancy = ()
            secondary = ()
        else:
            status = READY
            position = None
            occupancy = ()
            secondary = ()
    record = {
        "status": status,
        "event_sha256": event.sha256,
        "primary_carrier_cells": primary,
        "secondary_carrier_cells": secondary,
        "role_cell": role,
        "child_position": position,
        "parent_factor_slot_occupancy": occupancy,
        "raw_scores": tuple(int(value) for value in raw_array),
        "spike_cells": spike_cells,
    }
    return _DecodedEvent(
        status,
        event,
        primary,
        secondary,
        role,
        position,
        occupancy,
        tuple(int(value) for value in raw_array),
        spike_cells,
        _digest("event-relay-receipt", record),
    )


def _lesson_exact(decoded: _DecodedEvent, lesson: _Lesson) -> bool:
    return (
        decoded.status == READY
        and decoded.primary_carrier_cells == lesson.primary_carrier_cells
        and decoded.secondary_carrier_cells == lesson.secondary_carrier_cells
        and decoded.role_cell == lesson.role_cell
        and decoded.child_position == lesson.child_position
        and decoded.parent_factor_slot_occupancy == lesson.parent_factor_slot_occupancy
    )


def _checkpoint_payload(metadata: Mapping[str, object], e: np.ndarray, h: np.ndarray) -> bytes:
    return b"".join(
        (
            _canonical_json(dict(metadata)),
            b"\0",
            _array_frame("excitatory", e),
            _array_frame("inhibitory", h),
        )
    )


class _CertificateReceipt(NamedTuple):
    authority: object
    consumer: FrozenOccurrenceSlotCarrierV5
    status: str
    stop_reason: str | None
    request_sha256: str
    event_receipt_sha256s: tuple[str, ...]
    view: OccurrenceSlotCarrierView | None
    sha256: str


_CERTIFICATE_AUTHORITY = object()
_ISSUED_CERTIFICATES: dict[int, OccurrenceSlotCarrierCertificate] = {}


class OccurrenceSlotCarrierCertificate:
    """Opaque exact-object authority for one frozen carrier inference."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _CertificateReceipt) -> None:
        if receipt.authority is not _CERTIFICATE_AUTHORITY:
            raise TypeError("occurrence-slot certificate requires private authority")
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
    consumer: FrozenOccurrenceSlotCarrierV5,
    *,
    status: str,
    stop_reason: str | None,
    request_sha256: str,
    decisions: tuple[_DecodedEvent, ...],
    view: OccurrenceSlotCarrierView | None,
) -> OccurrenceSlotCarrierCertificate:
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
    certificate = OccurrenceSlotCarrierCertificate(receipt)
    _ISSUED_CERTIFICATES[id(certificate)] = certificate
    return certificate


def _authenticate_certificate(
    certificate: OccurrenceSlotCarrierCertificate,
) -> _CertificateReceipt:
    if (
        type(certificate) is not OccurrenceSlotCarrierCertificate
        or _ISSUED_CERTIFICATES.get(id(certificate)) is not certificate
        or certificate._receipt.authority is not _CERTIFICATE_AUTHORITY
    ):
        raise TypeError("occurrence-slot certificate is not an issued exact object")
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
        raise RuntimeError("occurrence-slot certificate receipt changed")
    return receipt


class FrozenOccurrenceSlotCarrierV5:
    """Immutable static consumer over one fresh self-posed checkpoint."""

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
        protocol: FrozenOccurrenceSlotProtocolV5,
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
            or self._protocol.protocol_sha256 != _digest(
                "protocol", self._protocol.root_record()
            )
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
            raise RuntimeError("frozen occurrence-slot checkpoint changed")

    def infer(
        self,
        events: Iterable[PhysicalOccurrenceEvent],
    ) -> OccurrenceSlotCarrierCertificate:
        self._authenticate_checkpoint()
        supplied = tuple(events)
        if any(type(item) is not PhysicalOccurrenceEvent for item in supplied):
            raise TypeError("carrier inference accepts only physical occurrence events")
        ordered = tuple(sorted(supplied, key=lambda item: item.sha256))
        request_sha = _digest(
            "inference-request",
            tuple(item.sha256 for item in ordered),
        )
        if not ordered:
            return _issue_certificate(
                self,
                status=STOP_ZERO,
                stop_reason="no physical occurrence event was supplied",
                request_sha256=request_sha,
                decisions=(),
                view=None,
            )
        if len({item.sha256 for item in ordered}) != len(ordered):
            return _issue_certificate(
                self,
                status=STOP_AMBIGUOUS,
                stop_reason="a physical occurrence event was duplicated",
                request_sha256=request_sha,
                decisions=(),
                view=None,
            )
        decisions = tuple(
            _decode_event(
                self._protocol,
                self._excitatory,
                self._inhibitory,
                event,
            )
            for event in ordered
        )
        if any(row.status == STOP_AMBIGUOUS for row in decisions):
            return _issue_certificate(
                self,
                status=STOP_AMBIGUOUS,
                stop_reason="one physical event activated multiple carrier joins",
                request_sha256=request_sha,
                decisions=decisions,
                view=None,
            )
        if any(row.status != READY for row in decisions):
            return _issue_certificate(
                self,
                status=STOP_ZERO,
                stop_reason="one physical event had no learned carrier join",
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


def _assemble_view(
    consumer: FrozenOccurrenceSlotCarrierV5,
    request_sha: str,
    decisions: tuple[_DecodedEvent, ...],
) -> tuple[str, str | None, OccurrenceSlotCarrierView | None]:
    if len(decisions) != 9:
        return STOP_UNSUPPORTED, "only the frozen nine-event two-level shape is supported", None
    roots = tuple(row for row in decisions if row.role_cell == ROLE_ROOT)
    edges = tuple(row for row in decisions if row.role_cell == ROLE_EDGE)
    if len(roots) != 1:
        return (
            STOP_AMBIGUOUS if len(roots) > 1 else STOP_UNSUPPORTED,
            "the carrier graph requires exactly one learned root relay",
            None,
        )
    if len(edges) != 3:
        return STOP_UNSUPPORTED, "the first carrier slice requires exactly three edges", None

    roles_by_carrier: dict[tuple[int, ...], set[int]] = defaultdict(set)
    source_events: dict[tuple[int, ...], list[_DecodedEvent]] = defaultdict(list)
    for row in decisions:
        if row.role_cell != ROLE_EDGE and row.role_cell is not None:
            roles_by_carrier[row.primary_carrier_cells].add(row.role_cell)
            source_events[row.primary_carrier_cells].append(row)

    edge_keys: set[tuple[tuple[int, ...], int]] = set()
    parents: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    inbound: dict[tuple[int, ...], int] = defaultdict(int)
    carrier_edges: list[CarrierEdge] = []
    all_carriers: set[tuple[int, ...]] = set(roles_by_carrier)
    for row in edges:
        assert row.child_position is not None
        key = (row.primary_carrier_cells, row.child_position)
        if (
            key in edge_keys
            or row.primary_carrier_cells == row.secondary_carrier_cells
            or not row.parent_factor_slot_occupancy
        ):
            return STOP_AMBIGUOUS, "edge carrier or parent slot joined more than once", None
        edge_keys.add(key)
        parents[row.primary_carrier_cells].append(
            (row.child_position, row.secondary_carrier_cells)
        )
        inbound[row.secondary_carrier_cells] += 1
        all_carriers.update((row.primary_carrier_cells, row.secondary_carrier_cells))
        edge_record = {
            "parent_carrier_cells": row.primary_carrier_cells,
            "child_carrier_cells": row.secondary_carrier_cells,
            "child_position": row.child_position,
            "parent_factor_slot_occupancy": row.parent_factor_slot_occupancy,
            "relay_receipt_sha256": row.sha256,
        }
        carrier_edges.append(
            CarrierEdge(
                row.primary_carrier_cells,
                row.secondary_carrier_cells,
                row.child_position,
                row.parent_factor_slot_occupancy,
                row.sha256,
                _digest("carrier-edge", edge_record),
            )
        )
    if len(all_carriers) != 4:
        return STOP_UNSUPPORTED, "the first carrier slice requires four joined nodes", None

    root = roots[0].primary_carrier_cells
    if inbound[root] != 0 or any(
        inbound[carrier] != 1 for carrier in all_carriers if carrier != root
    ):
        return STOP_AMBIGUOUS, "carrier endpoints do not form one rooted tree", None
    for _parent, children in parents.items():
        positions = tuple(position for position, _child in sorted(children))
        if positions != tuple(range(len(positions))):
            return STOP_AMBIGUOUS, "learned child-position relays are not contiguous", None

    depths = {root: 0}
    frontier = [root]
    while frontier:
        parent = frontier.pop()
        for _position, child in parents.get(parent, ()):
            if child in depths:
                return STOP_AMBIGUOUS, "carrier joins contain a cycle", None
            depths[child] = depths[parent] + 1
            frontier.append(child)
    if set(depths) != all_carriers or sorted(depths.values()) != [0, 1, 2, 2]:
        return STOP_UNSUPPORTED, "carrier graph lies outside the frozen two-level shape", None

    source_bindings: list[CarrierSourceBinding] = []
    for carrier in sorted(all_carriers):
        rows = source_events.get(carrier, [])
        source_keys = {
            (
                0 if row.event.raw_observation is not None else 1,
                (
                    row.event.raw_observation.sha256
                    if row.event.raw_observation is not None
                    else row.event.typed_literal_event.sha256
                    if row.event.typed_literal_event is not None
                    else None
                ),
            )
            for row in rows
        }
        if len(source_keys) != 1 or None in {key[1] for key in source_keys}:
            return STOP_AMBIGUOUS, "carrier source events do not bind one exact identity", None
        source_kind, _source_sha = next(iter(source_keys))
        raw = rows[0].event.raw_observation if source_kind == 0 else None
        literal = rows[0].event.typed_literal_event if source_kind == 1 else None
        roles = roles_by_carrier[carrier]
        outgoing = bool(parents.get(carrier))
        if carrier == root:
            required = {ROLE_ROOT, ROLE_ORIENTATION}
        elif outgoing:
            required = {ROLE_ORIENTATION}
        elif source_kind == 0:
            required = {ROLE_PARAMETER}
        else:
            required = {ROLE_LITERAL_SLOT, ROLE_LITERAL_COPY}
        if roles != required:
            return STOP_UNSUPPORTED, "carrier role coverage is incomplete", None
        source_record = {
            "carrier_cells": carrier,
            "source_kind_cell": source_kind,
            "raw_observation_sha256": raw.sha256 if raw is not None else None,
            "typed_literal_event_sha256": literal.sha256 if literal is not None else None,
            "event_receipt_sha256s": tuple(sorted(row.sha256 for row in rows)),
        }
        source_bindings.append(
            CarrierSourceBinding(
                carrier,
                source_kind,
                raw,
                literal,
                tuple(sorted(row.sha256 for row in rows)),
                _digest("carrier-source-binding", source_record),
            )
        )

    work_receipt = _digest(
        "physical-carrier-work",
        {
            "request_sha256": request_sha,
            "event_receipt_sha256s": tuple(row.sha256 for row in decisions),
            "score_count": len(decisions) * OUTPUT_CELL_COUNT,
            "fitted_glif_steps": len(decisions) * OUTPUT_STEPS,
        },
    )
    view_record = {
        "root_carrier_cells": root,
        "source_binding_sha256s": tuple(row.sha256 for row in source_bindings),
        "edge_sha256s": tuple(row.sha256 for row in sorted(carrier_edges)),
        "event_relay_receipt_sha256s": tuple(sorted(row.sha256 for row in decisions)),
        "checkpoint_sha256": consumer.checkpoint_sha256,
        "protocol_sha256": consumer.protocol_sha256,
        "physical_work_receipt_sha256": work_receipt,
    }
    view = OccurrenceSlotCarrierView(
        root,
        tuple(source_bindings),
        tuple(sorted(carrier_edges)),
        tuple(sorted(row.sha256 for row in decisions)),
        consumer.checkpoint_sha256,
        consumer.protocol_sha256,
        work_receipt,
        _digest("carrier-view", view_record),
    )
    return READY, None, view


def build_frozen_occurrence_slot_carrier_v5(
    protocol: FrozenOccurrenceSlotProtocolV5 = PROTOCOL,
) -> FrozenOccurrenceSlotCarrierV5:
    """Train fresh once, freeze, replay the self-posed gate, and return the consumer."""

    if protocol is not PROTOCOL or protocol.protocol_sha256 != _digest(
        "protocol", protocol.root_record()
    ):
        raise RuntimeError("occurrence-slot V5 protocol authentication failed")
    fresh_e = np.zeros((OUTPUT_CELL_COUNT, INPUT_CELL_COUNT), dtype=TICK_DTYPE)
    fresh_h = np.zeros_like(fresh_e)
    fresh_nonzero = int(np.count_nonzero(fresh_e) + np.count_nonzero(fresh_h))
    teacher_off_e, teacher_off_h, _off_e, _off_h = _train(protocol, teacher_gain=0)
    treatment_e, treatment_h, committed_e, committed_h = _train(
        protocol,
        teacher_gain=1,
    )
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
        "public_train_input_count": PUBLIC_TRAIN_INPUT_COUNT,
        "canonical_input_count": CANONICAL_INPUT_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
    }
    checkpoint = sha256(
        _checkpoint_payload(metadata, treatment_e, treatment_h)
    ).hexdigest()
    fit_decisions = tuple(
        _decode_event(protocol, treatment_e, treatment_h, lesson.event)
        for lesson in protocol.lessons
    )
    fit_exact = sum(
        _lesson_exact(decision, lesson)
        for decision, lesson in zip(fit_decisions, protocol.lessons, strict=True)
    )
    off_decisions = tuple(
        _decode_event(
            protocol,
            teacher_off_e,
            teacher_off_h,
            lesson.event,
            cell_gain=1.0,
        )
        for lesson in protocol.lessons
    )
    off_ready = sum(row.status == READY for row in off_decisions)
    summary_record = {
        "status": READY if fit_exact == len(protocol.lessons) and off_ready == 0 else STOP_ZERO,
        "fresh_zero_state_nonzero_count": fresh_nonzero,
        "fit_lesson_count": len(protocol.lessons),
        "fit_exact_count": fit_exact,
        "teacher_off_nonzero_count": int(
            np.count_nonzero(teacher_off_e) + np.count_nonzero(teacher_off_h)
        ),
        "teacher_off_ready_count": off_ready,
        "committed_excitatory_ticks": committed_e,
        "committed_inhibitory_ticks": committed_h,
        "learned_state_bytes": treatment_e.nbytes + treatment_h.nbytes,
        "checkpoint_sha256": checkpoint,
        "protocol_sha256": protocol.protocol_sha256,
    }
    summary = CarrierBuildSummary(
        **summary_record,
        sha256=_digest("build-summary", summary_record),
    )
    if summary.status != READY:
        raise RuntimeError(
            "occurrence-slot V5 self-posed fit gate stopped: "
            f"{summary.fit_exact_count}/{summary.fit_lesson_count} exact, "
            f"{summary.teacher_off_ready_count} teacher-off READY"
        )
    treatment_e.setflags(write=False)
    treatment_h.setflags(write=False)
    return FrozenOccurrenceSlotCarrierV5(
        protocol,
        treatment_e,
        treatment_h,
        metadata,
        checkpoint,
        summary,
    )


STATIC_CONSUMER = build_frozen_occurrence_slot_carrier_v5()


def infer_occurrence_slot_carriers(
    events: Iterable[PhysicalOccurrenceEvent],
) -> OccurrenceSlotCarrierCertificate:
    """Public deep seam: physical events in, opaque authenticated certificate out."""

    return STATIC_CONSUMER.infer(events)


def require_occurrence_slot_carrier(
    certificate: OccurrenceSlotCarrierCertificate,
) -> OccurrenceSlotCarrierView:
    """Authenticate one READY certificate and expose only its safe physical view."""

    receipt = _authenticate_certificate(certificate)
    receipt.consumer._authenticate_checkpoint()
    if receipt.status != READY or receipt.view is None:
        raise RuntimeError(receipt.stop_reason or receipt.status)
    return receipt.view


__all__ = [
    "CANONICAL_INPUT_COUNT",
    "DEV_INPUT_COUNT",
    "HELDOUT_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PROTOCOL",
    "READY",
    "SCHEMA",
    "SELF_POSED_COUNTERFACTUAL_EVENTS",
    "SELF_POSED_TREATMENT_EVENTS",
    "STATIC_CONSUMER",
    "STOP_AMBIGUOUS",
    "STOP_UNSUPPORTED",
    "STOP_ZERO",
    "CarrierBuildSummary",
    "CarrierEdge",
    "CarrierSourceBinding",
    "FrozenOccurrenceSlotCarrierV5",
    "FrozenOccurrenceSlotProtocolV5",
    "OccurrenceSlotCarrierCertificate",
    "OccurrenceSlotCarrierView",
    "PhysicalOccurrenceEvent",
    "TypedLiteralEvent",
    "build_frozen_occurrence_slot_carrier_v5",
    "infer_occurrence_slot_carriers",
    "require_occurrence_slot_carrier",
]
