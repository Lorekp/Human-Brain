"""Authenticated bounded source observations over opaque fixed-size payloads.

This module is a neutral immutable graph/view, not learned bundle recognition:
it has no OPEN/EXTEND/EOS vocabulary, learned recognizer, or graph-action
certificate.  Callers supply opaque event and carrier payloads, source-root and
observable boundary carrier indices, plus primitive directed incidences.  Every
payload is projected through ``CommonEventCells``; only authenticated roots,
traces, counts, primitive incidences, and measured cell work survive.  Raw
payloads and transitive closure never survive.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import NamedTuple

from experiments.g0 import common_event_cells_v1 as common_events

SCHEMA = "COMMON_ROOTED_STRUCTURED_EVENT_GRAPH_V1"

PAYLOAD_BYTES = 16
TRACE_LANES = 12
MAX_EVENTS = 8
MAX_CARRIERS = 16
MAX_ROOT_CARRIERS = 8
MAX_BOUNDARY_CARRIERS = 8
MAX_INCIDENCES = 24

_EVENT_NAMESPACE = b"common-rooted-structured-event-v1"
_CARRIER_NAMESPACE = b"common-rooted-structured-carrier-v1"
_EVENT_ROOT_NAMESPACE = b"common-rooted-structured-event-root-v1"
_CARRIER_ROOT_NAMESPACE = b"common-rooted-structured-carrier-root-v1"
_INCIDENCE_NAMESPACE = b"common-rooted-structured-incidence-v1"
_GRAPH_NAMESPACE = b"common-rooted-structured-event-graph-v1"
_CELLS = common_events.CommonEventCells()


class StructuredEventGraphError(ValueError):
    """Base error for an invalid bounded structured-event graph."""


class StructuredEventGraphCapacityError(StructuredEventGraphError):
    """The complete graph exceeds a frozen v1 capacity."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _authenticate_projection(
    semantic_sha256: str,
    frame_sha256: str,
    trace_cells: tuple[int, ...],
    projection_cell_steps: int,
    sha256_value: str,
    namespace: bytes,
) -> None:
    if type(trace_cells) is not tuple:
        raise TypeError("structured projection trace must be one exact tuple")
    payload = {
        "semantic": semantic_sha256,
        "frame": frame_sha256,
        "trace": trace_cells,
        "projection_cell_steps": projection_cell_steps,
    }
    if (
        not _valid_sha256(semantic_sha256)
        or not _valid_sha256(frame_sha256)
        or len(trace_cells) != TRACE_LANES
        or any(type(cell) is not int for cell in trace_cells)
        or type(projection_cell_steps) is not int
        or projection_cell_steps <= 0
        or sha256_value != _digest(namespace, payload)
    ):
        raise StructuredEventGraphError(
            "structured projection root authentication failed"
        )


class StructuredEventRootV1(NamedTuple):
    """Payload-free common-cell projection of one effect-event occurrence."""

    semantic_sha256: str
    frame_sha256: str
    trace_cells: tuple[int, ...]
    projection_cell_steps: int
    sha256: str

    def authenticate(self) -> None:
        _authenticate_projection(
            self.semantic_sha256,
            self.frame_sha256,
            self.trace_cells,
            self.projection_cell_steps,
            self.sha256,
            _EVENT_ROOT_NAMESPACE,
        )


class StructuredCarrierRootV1(NamedTuple):
    """Payload-free common-cell projection of one anonymous graph endpoint."""

    semantic_sha256: str
    frame_sha256: str
    trace_cells: tuple[int, ...]
    projection_cell_steps: int
    sha256: str

    def authenticate(self) -> None:
        _authenticate_projection(
            self.semantic_sha256,
            self.frame_sha256,
            self.trace_cells,
            self.projection_cell_steps,
            self.sha256,
            _CARRIER_ROOT_NAMESPACE,
        )


class StructuredEventIncidenceV1(NamedTuple):
    """Primitive edge over the flattened carrier-then-event vertex order."""

    source_index: int
    target_index: int
    ordinal: int
    sha256: str

    def authenticate(
        self,
        carrier_roots: tuple[StructuredCarrierRootV1, ...],
        event_roots: tuple[StructuredEventRootV1, ...],
    ) -> None:
        if type(carrier_roots) is not tuple or type(event_roots) is not tuple:
            raise TypeError("structured roots must be exact tuples")
        vertex_roots = tuple(root.sha256 for root in carrier_roots) + tuple(
            root.sha256 for root in event_roots
        )
        vertex_count = len(vertex_roots)
        if (
            type(self.source_index) is not int
            or type(self.target_index) is not int
            or type(self.ordinal) is not int
            or not 0 <= self.source_index < vertex_count
            or not 0 <= self.target_index < vertex_count
            or self.source_index == self.target_index
            or not 0 <= self.ordinal <= 0xFF
        ):
            raise StructuredEventGraphError("invalid structured-event incidence")
        payload = {
            "source_index": self.source_index,
            "source": vertex_roots[self.source_index],
            "target_index": self.target_index,
            "target": vertex_roots[self.target_index],
            "ordinal": self.ordinal,
        }
        if self.sha256 != _digest(_INCIDENCE_NAMESPACE, payload):
            raise StructuredEventGraphError(
                "structured-event incidence authentication failed"
            )


class RootedStructuredEventGraphV1(NamedTuple):
    """Immutable payload-free view of bounded events, carriers, and edges."""

    event_roots: tuple[StructuredEventRootV1, ...]
    carrier_roots: tuple[StructuredCarrierRootV1, ...]
    root_carrier_indices: tuple[int, ...]
    boundary_carrier_indices: tuple[int, ...]
    incidences: tuple[StructuredEventIncidenceV1, ...]
    event_count: int
    carrier_count: int
    root_count: int
    boundary_count: int
    incidence_count: int
    projection_cell_steps: int
    sha256: str

    def authenticate(self) -> None:
        if type(self.event_roots) is not tuple:
            raise TypeError("structured-event roots must be one exact tuple")
        if type(self.carrier_roots) is not tuple:
            raise TypeError("structured carrier roots must be one exact tuple")
        if type(self.root_carrier_indices) is not tuple:
            raise TypeError("root carrier indices must be one exact tuple")
        if type(self.boundary_carrier_indices) is not tuple:
            raise TypeError("boundary carrier indices must be one exact tuple")
        if type(self.incidences) is not tuple:
            raise TypeError("structured-event incidences must be one exact tuple")
        for root in self.event_roots:
            if type(root) is not StructuredEventRootV1:
                raise TypeError("exact structured-event roots required")
            root.authenticate()
        for root in self.carrier_roots:
            if type(root) is not StructuredCarrierRootV1:
                raise TypeError("exact structured carrier roots required")
            root.authenticate()
        for incidence in self.incidences:
            if type(incidence) is not StructuredEventIncidenceV1:
                raise TypeError("exact structured-event incidences required")
            incidence.authenticate(self.carrier_roots, self.event_roots)
        incidence_keys = tuple(
            (edge.source_index, edge.target_index, edge.ordinal)
            for edge in self.incidences
        )
        payload = {
            "events": tuple(root.sha256 for root in self.event_roots),
            "carriers": tuple(root.sha256 for root in self.carrier_roots),
            "root_carrier_indices": self.root_carrier_indices,
            "boundary_carrier_indices": self.boundary_carrier_indices,
            "incidences": tuple(edge.sha256 for edge in self.incidences),
            "event_count": self.event_count,
            "carrier_count": self.carrier_count,
            "root_count": self.root_count,
            "boundary_count": self.boundary_count,
            "incidence_count": self.incidence_count,
            "projection_cell_steps": self.projection_cell_steps,
        }
        if (
            type(self.event_count) is not int
            or self.event_count != len(self.event_roots)
            or not 0 <= self.event_count <= MAX_EVENTS
            or type(self.carrier_count) is not int
            or self.carrier_count != len(self.carrier_roots)
            or not 0 <= self.carrier_count <= MAX_CARRIERS
            or type(self.root_count) is not int
            or self.root_count != len(self.root_carrier_indices)
            or not 0 <= self.root_count <= min(MAX_ROOT_CARRIERS, self.carrier_count)
            or self.root_carrier_indices
            != tuple(sorted(set(self.root_carrier_indices)))
            or any(
                not 0 <= index < self.carrier_count
                for index in self.root_carrier_indices
            )
            or type(self.boundary_count) is not int
            or self.boundary_count != len(self.boundary_carrier_indices)
            or not (
                0
                <= self.boundary_count
                <= min(MAX_BOUNDARY_CARRIERS, self.carrier_count)
            )
            or self.boundary_carrier_indices
            != tuple(sorted(set(self.boundary_carrier_indices)))
            or any(
                not 0 <= index < self.carrier_count
                for index in self.boundary_carrier_indices
            )
            or set(self.root_carrier_indices) & set(self.boundary_carrier_indices)
            or type(self.incidence_count) is not int
            or self.incidence_count != len(self.incidences)
            or not 0 <= self.incidence_count <= MAX_INCIDENCES
            or incidence_keys != tuple(sorted(set(incidence_keys)))
            or type(self.projection_cell_steps) is not int
            or self.projection_cell_steps
            != sum(root.projection_cell_steps for root in self.event_roots)
            + sum(root.projection_cell_steps for root in self.carrier_roots)
            or self.sha256 != _digest(_GRAPH_NAMESPACE, payload)
        ):
            raise StructuredEventGraphError(
                "rooted structured-event graph authentication failed"
            )


def _project_event(payload: bytes) -> StructuredEventRootV1:
    frame = _CELLS.project(payload)
    semantic_root = sha256(_EVENT_NAMESPACE + b"\0" + payload).hexdigest()
    root_payload = {
        "semantic": semantic_root,
        "frame": frame.sha256,
        "trace": frame.trace_cells,
        "projection_cell_steps": frame.work.total,
    }
    root = StructuredEventRootV1(
        semantic_sha256=semantic_root,
        frame_sha256=frame.sha256,
        trace_cells=frame.trace_cells,
        projection_cell_steps=frame.work.total,
        sha256=_digest(_EVENT_ROOT_NAMESPACE, root_payload),
    )
    root.authenticate()
    return root


def _project_carrier(payload: bytes) -> StructuredCarrierRootV1:
    frame = _CELLS.project(payload)
    semantic_root = sha256(_CARRIER_NAMESPACE + b"\0" + payload).hexdigest()
    root_payload = {
        "semantic": semantic_root,
        "frame": frame.sha256,
        "trace": frame.trace_cells,
        "projection_cell_steps": frame.work.total,
    }
    root = StructuredCarrierRootV1(
        semantic_sha256=semantic_root,
        frame_sha256=frame.sha256,
        trace_cells=frame.trace_cells,
        projection_cell_steps=frame.work.total,
        sha256=_digest(_CARRIER_ROOT_NAMESPACE, root_payload),
    )
    root.authenticate()
    return root


def project_rooted_structured_event_graph_v1(
    event_payloads: tuple[bytes, ...],
    *,
    carrier_payloads: tuple[bytes, ...],
    root_carrier_indices: tuple[int, ...],
    boundary_carrier_indices: tuple[int, ...],
    incidences: tuple[tuple[int, int, int], ...],
) -> RootedStructuredEventGraphV1:
    """Project one ordered primitive graph without retaining opaque payloads."""

    if type(event_payloads) is not tuple:
        raise TypeError("event payloads must be one exact tuple")
    if type(carrier_payloads) is not tuple:
        raise TypeError("carrier payloads must be one exact tuple")
    if type(root_carrier_indices) is not tuple:
        raise TypeError("root carrier indices must be one exact tuple")
    if type(boundary_carrier_indices) is not tuple:
        raise TypeError("boundary carrier indices must be one exact tuple")
    if type(incidences) is not tuple:
        raise TypeError("incidences must be one exact tuple")
    if len(event_payloads) > MAX_EVENTS:
        raise StructuredEventGraphCapacityError(
            f"structured-event graph exceeds {MAX_EVENTS} events"
        )
    if len(carrier_payloads) > MAX_CARRIERS:
        raise StructuredEventGraphCapacityError(
            f"structured-event graph exceeds {MAX_CARRIERS} carriers"
        )
    if len(root_carrier_indices) > MAX_ROOT_CARRIERS:
        raise StructuredEventGraphCapacityError(
            f"structured-event graph exceeds {MAX_ROOT_CARRIERS} roots"
        )
    if len(boundary_carrier_indices) > MAX_BOUNDARY_CARRIERS:
        raise StructuredEventGraphCapacityError(
            "structured-event graph exceeds "
            f"{MAX_BOUNDARY_CARRIERS} observable boundaries"
        )
    if len(incidences) > MAX_INCIDENCES:
        raise StructuredEventGraphCapacityError(
            f"structured-event graph exceeds {MAX_INCIDENCES} incidences"
        )
    for payload in event_payloads + carrier_payloads:
        if type(payload) is not bytes:
            raise TypeError("each structured payload must be exact bytes")
        if len(payload) != PAYLOAD_BYTES:
            raise StructuredEventGraphError(
                f"each structured payload must be exactly {PAYLOAD_BYTES} bytes"
            )
    if any(type(index) is not int for index in root_carrier_indices):
        raise TypeError("root carrier indices must be exact integers")
    if root_carrier_indices != tuple(sorted(set(root_carrier_indices))):
        raise StructuredEventGraphError(
            "root carrier indices must be canonical and unique"
        )
    if any(type(index) is not int for index in boundary_carrier_indices):
        raise TypeError("boundary carrier indices must be exact integers")
    if boundary_carrier_indices != tuple(sorted(set(boundary_carrier_indices))):
        raise StructuredEventGraphError(
            "boundary carrier indices must be canonical and unique"
        )
    if any(
        not 0 <= index < len(carrier_payloads) for index in root_carrier_indices
    ):
        raise StructuredEventGraphError("root is outside the carrier graph")
    if any(
        not 0 <= index < len(carrier_payloads)
        for index in boundary_carrier_indices
    ):
        raise StructuredEventGraphError("boundary is outside the carrier graph")
    if set(root_carrier_indices) & set(boundary_carrier_indices):
        raise StructuredEventGraphError("roots and boundaries must be disjoint")
    if any(
        type(edge) is not tuple
        or len(edge) != 3
        or any(type(value) is not int for value in edge)
        for edge in incidences
    ):
        raise TypeError("incidences must be exact integer triples")
    if incidences != tuple(sorted(set(incidences))):
        raise StructuredEventGraphError("incidences must be canonical and unique")

    event_roots = tuple(_project_event(payload) for payload in event_payloads)
    carrier_roots = tuple(_project_carrier(payload) for payload in carrier_payloads)
    vertex_roots = tuple(root.sha256 for root in carrier_roots) + tuple(
        root.sha256 for root in event_roots
    )
    projected_incidences: list[StructuredEventIncidenceV1] = []
    for source_index, target_index, ordinal in incidences:
        if (
            not 0 <= source_index < len(vertex_roots)
            or not 0 <= target_index < len(vertex_roots)
            or source_index == target_index
            or not 0 <= ordinal <= 0xFF
        ):
            raise StructuredEventGraphError("invalid structured-event incidence")
        edge_payload = {
            "source_index": source_index,
            "source": vertex_roots[source_index],
            "target_index": target_index,
            "target": vertex_roots[target_index],
            "ordinal": ordinal,
        }
        projected_incidences.append(
            StructuredEventIncidenceV1(
                source_index=source_index,
                target_index=target_index,
                ordinal=ordinal,
                sha256=_digest(_INCIDENCE_NAMESPACE, edge_payload),
            )
        )
    incidence_tuple = tuple(projected_incidences)
    projection_steps = sum(
        root.projection_cell_steps for root in event_roots
    ) + sum(root.projection_cell_steps for root in carrier_roots)
    graph_payload = {
        "events": tuple(root.sha256 for root in event_roots),
        "carriers": tuple(root.sha256 for root in carrier_roots),
        "root_carrier_indices": root_carrier_indices,
        "boundary_carrier_indices": boundary_carrier_indices,
        "incidences": tuple(edge.sha256 for edge in incidence_tuple),
        "event_count": len(event_roots),
        "carrier_count": len(carrier_roots),
        "root_count": len(root_carrier_indices),
        "boundary_count": len(boundary_carrier_indices),
        "incidence_count": len(incidence_tuple),
        "projection_cell_steps": projection_steps,
    }
    graph = RootedStructuredEventGraphV1(
        event_roots=event_roots,
        carrier_roots=carrier_roots,
        root_carrier_indices=root_carrier_indices,
        boundary_carrier_indices=boundary_carrier_indices,
        incidences=incidence_tuple,
        event_count=len(event_roots),
        carrier_count=len(carrier_roots),
        root_count=len(root_carrier_indices),
        boundary_count=len(boundary_carrier_indices),
        incidence_count=len(incidence_tuple),
        projection_cell_steps=projection_steps,
        sha256=_digest(_GRAPH_NAMESPACE, graph_payload),
    )
    graph.authenticate()
    return graph
