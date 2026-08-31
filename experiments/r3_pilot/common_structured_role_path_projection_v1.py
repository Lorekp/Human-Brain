"""Bounded cell-relay role paths over a neutral structured-event graph.

The module consumes only :class:`RootedStructuredEventGraphV1`.  Primitive
incidences wire a reverse, depth-bounded GLIF relay circuit seeded at observable
boundary carriers.  The host materializes that circuit and collects its spike
traces; it does not compute graph reachability or attach a relevance label.

Each event is retained.  Its event trace is joined with a cell projection of
the anonymous structural path-channel spikes.  The resulting observations are
binary-deduplicated by ``(event semantic root, role-path semantic root)``.  No
name, opcode, action, absolute vertex index, or incidence ordinal enters a
factor cue.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import NamedTuple

import numpy as np

from experiments.g0 import common_event_cells_v1 as common_events
from experiments.r3_pilot.common_rooted_structured_event_graph_v1 import (
    MAX_CARRIERS,
    MAX_EVENTS,
    MAX_INCIDENCES,
    TRACE_LANES as EVENT_TRACE_LANES,
    RootedStructuredEventGraphV1,
    StructuredEventRootV1,
)
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCHEMA = "COMMON_STRUCTURED_ROLE_PATH_PROJECTION_V1"

MAX_ROLE_DEPTH = 3
ROLE_PATH_BITS = sum(1 << depth for depth in range(1, MAX_ROLE_DEPTH + 1))
ROLE_PATH_PAYLOAD_BYTES = 16
ROLE_TRACE_LANES = 12
FACTOR_CUE_CELLS = EVENT_TRACE_LANES + ROLE_TRACE_LANES
MAX_VERTICES = MAX_CARRIERS + MAX_EVENTS
MAX_FACTORS = MAX_EVENTS

RELAY_STEPS = 64
_CARRIER_KIND = 0
_EVENT_KIND = 1
_MAX_PREVIOUS_CHANNELS = 1 << (MAX_ROLE_DEPTH - 1)
# Conservative upper bound for live NumPy data buffers at the widest relay
# step, including the compiled response's vector algebra and a possible
# contiguous boolean result copy.  Python object/allocator headers are not
# claimed as array state.
TRANSIENT_NUMPY_ARRAY_BOUND_BYTES_PER_VERTEX = _MAX_PREVIOUS_CHANNELS * (
    4 * np.dtype(np.float64).itemsize + 4 * np.dtype(np.bool_).itemsize
)

_ROLE_SEMANTIC_NAMESPACE = b"common-structured-role-path-semantic-v1"
_ROLE_ROOT_NAMESPACE = b"common-structured-role-path-root-v1"
_FACTOR_NAMESPACE = b"common-structured-role-factor-v1"
_VIEW_NAMESPACE = b"common-structured-role-projection-v1"
_RELAY_CONFIGURATION_NAMESPACE = b"common-structured-role-relay-config-v1"

_ROLE_CELLS = common_events.CommonEventCells()


class StructuredRoleProjectionError(ValueError):
    """The graph or its projected structural-role view is invalid."""


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


def _build_relay() -> tuple[float, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(RELAY_STEPS)
    crossing = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    spike_step = RELAY_STEPS // 2
    lower = float(crossing[spike_step])
    upper = float(crossing[spike_step - 1])
    if not lower < upper:
        raise RuntimeError("structured-role GLIF relay interval changed")
    active_current = (lower + upper) / 2.0
    relay = CompiledRestingGlifRace.compile(step_count=RELAY_STEPS)
    control = relay.accumulated_spike_mask(
        np.asarray((0.0, active_current), dtype=np.float64)
    )
    if tuple(bool(value) for value in control) != (False, True):
        raise RuntimeError("structured-role GLIF relay calibration changed")
    return active_current, relay


_RELAY_ACTIVE_CURRENT_PA, _RELAY = _build_relay()
RELAY_CONFIGURATION_SHA256 = _digest(
    _RELAY_CONFIGURATION_NAMESPACE,
    {
        "schema": SCHEMA,
        "max_role_depth": MAX_ROLE_DEPTH,
        "relay_steps": RELAY_STEPS,
        "active_current_pa": _RELAY_ACTIVE_CURRENT_PA.hex(),
    },
)


def _role_payload(spike_bits: int) -> bytes:
    if type(spike_bits) is not int or not 0 <= spike_bits < (1 << ROLE_PATH_BITS):
        raise StructuredRoleProjectionError("invalid structural-role spike trace")
    return spike_bits.to_bytes(2, "little") + bytes(ROLE_PATH_PAYLOAD_BYTES - 2)


class StructuralRolePathV1(NamedTuple):
    """Authenticated cell projection of anonymous relay path-channel spikes."""

    relay_spike_bits: int
    semantic_sha256: str
    frame_sha256: str
    trace_cells: tuple[int, ...]
    projection_cell_steps: int
    sha256: str

    def authenticate(self) -> None:
        payload_bytes = _role_payload(self.relay_spike_bits)
        payload = {
            "relay_spike_bits": self.relay_spike_bits,
            "semantic": self.semantic_sha256,
            "frame": self.frame_sha256,
            "trace": self.trace_cells,
            "projection_cell_steps": self.projection_cell_steps,
        }
        if type(self.trace_cells) is not tuple:
            raise TypeError("role-path trace cells must be one exact tuple")
        if (
            self.semantic_sha256
            != sha256(
                _ROLE_SEMANTIC_NAMESPACE + b"\0" + payload_bytes
            ).hexdigest()
            or not _valid_sha256(self.frame_sha256)
            or len(self.trace_cells) != ROLE_TRACE_LANES
            or any(type(cell) is not int for cell in self.trace_cells)
            or type(self.projection_cell_steps) is not int
            or self.projection_cell_steps <= 0
            or self.sha256 != _digest(_ROLE_ROOT_NAMESPACE, payload)
        ):
            raise StructuredRoleProjectionError(
                "structural role-path authentication failed"
            )


class StructuredEventRoleFactorV1(NamedTuple):
    """One binary event-semantic and anonymous-role observation."""

    event_root: StructuredEventRootV1
    role_path: StructuralRolePathV1
    cue_cells: tuple[int, ...]
    presence_cell: int
    sha256: str

    def authenticate(self) -> None:
        if type(self.event_root) is not StructuredEventRootV1:
            raise TypeError("role factor requires an exact structured-event root")
        if type(self.role_path) is not StructuralRolePathV1:
            raise TypeError("role factor requires an exact structural role path")
        self.event_root.authenticate()
        self.role_path.authenticate()
        if type(self.cue_cells) is not tuple:
            raise TypeError("role-factor cue cells must be one exact tuple")
        payload = {
            "event_root": self.event_root.sha256,
            "event_semantic": self.event_root.semantic_sha256,
            "role_path": self.role_path.sha256,
            "role_semantic": self.role_path.semantic_sha256,
            "cue_cells": self.cue_cells,
            "presence_cell": self.presence_cell,
        }
        if (
            self.cue_cells
            != self.event_root.trace_cells + self.role_path.trace_cells
            or len(self.cue_cells) != FACTOR_CUE_CELLS
            or any(type(cell) is not int for cell in self.cue_cells)
            or type(self.presence_cell) is not int
            or self.presence_cell != 1
            or self.sha256 != _digest(_FACTOR_NAMESPACE, payload)
        ):
            raise StructuredRoleProjectionError(
                "structured event-role factor authentication failed"
            )


class StructuredRoleProjectionWorkV1(NamedTuple):
    boundary_seed_cell_steps: int
    reverse_synaptic_reads: int
    relay_cell_steps: int
    role_projection_cell_steps: int
    cue_cell_reads: int

    @property
    def total(self) -> int:
        return (
            self.boundary_seed_cell_steps
            + self.reverse_synaptic_reads
            + self.relay_cell_steps
            + self.role_projection_cell_steps
            + self.cue_cell_reads
        )

    def authenticate(self) -> None:
        if any(type(value) is not int or value < 0 for value in self):
            raise StructuredRoleProjectionError(
                "structured-role work counters must be nonnegative integers"
            )

    def record(self) -> dict[str, int]:
        return {
            "boundary_seed_cell_steps": self.boundary_seed_cell_steps,
            "reverse_synaptic_reads": self.reverse_synaptic_reads,
            "relay_cell_steps": self.relay_cell_steps,
            "role_projection_cell_steps": self.role_projection_cell_steps,
            "cue_cell_reads": self.cue_cell_reads,
            "total": self.total,
        }


class StructuredRoleProjectionCapacityV1(NamedTuple):
    depth_cap: int
    event_count: int
    vertex_count: int
    incidence_count: int
    boundary_seed_cell_count: int
    relay_cell_count: int
    relay_synapse_count: int
    role_path_bits: int
    role_path_payload_bytes: int
    factor_count: int
    factor_cap: int
    factor_cue_cell_count: int
    transient_numpy_array_bound_bytes: int
    checkpoint_array_bytes: int

    def authenticate(self) -> None:
        if any(type(value) is not int or value < 0 for value in self):
            raise StructuredRoleProjectionError(
                "structured-role capacities must be nonnegative integers"
            )
        if (
            self.depth_cap != MAX_ROLE_DEPTH
            or not 0 <= self.event_count <= MAX_EVENTS
            or not 0 <= self.vertex_count <= MAX_VERTICES
            or not 0 <= self.incidence_count <= MAX_INCIDENCES
            or self.boundary_seed_cell_count != self.vertex_count
            or self.relay_cell_count
            != self.vertex_count * ((1 << MAX_ROLE_DEPTH) - 1)
            or self.relay_synapse_count
            != self.incidence_count * ((1 << MAX_ROLE_DEPTH) - 1)
            or self.role_path_bits != ROLE_PATH_BITS
            or self.role_path_payload_bytes != ROLE_PATH_PAYLOAD_BYTES
            or not 0 <= self.factor_count <= self.event_count
            or self.factor_cap != MAX_FACTORS
            or self.factor_cue_cell_count != FACTOR_CUE_CELLS
            or self.transient_numpy_array_bound_bytes
            != TRANSIENT_NUMPY_ARRAY_BOUND_BYTES_PER_VERTEX * self.vertex_count
            or self.checkpoint_array_bytes != 0
        ):
            raise StructuredRoleProjectionError(
                "structured-role capacity authentication failed"
            )

    def record(self) -> dict[str, int]:
        return dict(zip(self._fields, self, strict=True))


class RootedStructuredRoleProjectionV1(NamedTuple):
    """Immutable binary role-factor view of one authenticated event graph."""

    graph_sha256: str
    factors: tuple[StructuredEventRoleFactorV1, ...]
    source_event_count: int
    factor_count: int
    collapsed_duplicate_count: int
    role_path_count: int
    relay_configuration_sha256: str
    capacity: StructuredRoleProjectionCapacityV1
    work: StructuredRoleProjectionWorkV1
    sha256: str

    def authenticate(self) -> None:
        if type(self.factors) is not tuple:
            raise TypeError("structured-role factors must be one exact tuple")
        if type(self.capacity) is not StructuredRoleProjectionCapacityV1:
            raise TypeError("structured-role capacity record must be exact")
        if type(self.work) is not StructuredRoleProjectionWorkV1:
            raise TypeError("structured-role work record must be exact")
        for factor in self.factors:
            if type(factor) is not StructuredEventRoleFactorV1:
                raise TypeError("structured-role factors must be exact")
            factor.authenticate()
        self.capacity.authenticate()
        self.work.authenticate()
        keys = tuple(
            (
                factor.event_root.semantic_sha256,
                factor.role_path.semantic_sha256,
            )
            for factor in self.factors
        )
        unique_roles = {
            factor.role_path.sha256: factor.role_path for factor in self.factors
        }
        payload = {
            "graph": self.graph_sha256,
            "factors": tuple(factor.sha256 for factor in self.factors),
            "source_event_count": self.source_event_count,
            "factor_count": self.factor_count,
            "collapsed_duplicate_count": self.collapsed_duplicate_count,
            "role_path_count": self.role_path_count,
            "relay_configuration": self.relay_configuration_sha256,
            "capacity": self.capacity.record(),
            "work": self.work.record(),
        }
        if (
            not _valid_sha256(self.graph_sha256)
            or type(self.source_event_count) is not int
            or self.source_event_count != self.capacity.event_count
            or type(self.factor_count) is not int
            or self.factor_count != len(self.factors)
            or self.factor_count != self.capacity.factor_count
            or type(self.collapsed_duplicate_count) is not int
            or self.collapsed_duplicate_count
            != self.source_event_count - self.factor_count
            or type(self.role_path_count) is not int
            or self.role_path_count != len(unique_roles)
            or self.relay_configuration_sha256 != RELAY_CONFIGURATION_SHA256
            or keys != tuple(sorted(set(keys)))
            or self.work.boundary_seed_cell_steps
            != self.capacity.boundary_seed_cell_count * RELAY_STEPS
            or self.work.reverse_synaptic_reads
            != self.capacity.relay_synapse_count
            or self.work.relay_cell_steps
            != self.capacity.relay_cell_count * RELAY_STEPS
            or self.work.role_projection_cell_steps
            != sum(
                path.projection_cell_steps for path in unique_roles.values()
            )
            or self.work.cue_cell_reads != self.factor_count * FACTOR_CUE_CELLS
            or self.sha256 != _digest(_VIEW_NAMESPACE, payload)
        ):
            raise StructuredRoleProjectionError(
                "rooted structured-role projection authentication failed"
            )


def _project_role_path(relay_spike_bits: int) -> StructuralRolePathV1:
    payload_bytes = _role_payload(relay_spike_bits)
    frame = _ROLE_CELLS.project(payload_bytes)
    semantic_root = sha256(
        _ROLE_SEMANTIC_NAMESPACE + b"\0" + payload_bytes
    ).hexdigest()
    payload = {
        "relay_spike_bits": relay_spike_bits,
        "semantic": semantic_root,
        "frame": frame.sha256,
        "trace": frame.trace_cells,
        "projection_cell_steps": frame.work.total,
    }
    path = StructuralRolePathV1(
        relay_spike_bits=relay_spike_bits,
        semantic_sha256=semantic_root,
        frame_sha256=frame.sha256,
        trace_cells=frame.trace_cells,
        projection_cell_steps=frame.work.total,
        sha256=_digest(_ROLE_ROOT_NAMESPACE, payload),
    )
    path.authenticate()
    return path


def _relay_path_spike_bits(
    graph: RootedStructuredEventGraphV1,
) -> tuple[tuple[int, ...], int, int]:
    """Run the bounded reverse relay circuit and return event channel traces."""

    vertex_count = graph.carrier_count + graph.event_count
    if vertex_count == 0:
        return (), 0, 0

    seed_currents = np.zeros((vertex_count,), dtype=np.float64)
    for boundary_index in graph.boundary_carrier_indices:
        seed_currents[boundary_index] = _RELAY_ACTIVE_CURRENT_PA
    frontier = _RELAY.accumulated_spike_mask(seed_currents).reshape(
        vertex_count, 1
    )
    del seed_currents
    event_spike_bits = [0] * graph.event_count
    reverse_synaptic_reads = 0
    relay_cell_count = 0

    for depth in range(1, MAX_ROLE_DEPTH + 1):
        previous_channel_count = frontier.shape[1]
        arrivals = np.zeros(
            (vertex_count, previous_channel_count), dtype=np.uint8
        )
        for incidence in graph.incidences:
            arrivals[incidence.source_index] += frontier[incidence.target_index]
        reverse_synaptic_reads += graph.incidence_count * previous_channel_count

        currents = np.ascontiguousarray(arrivals, dtype=np.float64)
        currents *= _RELAY_ACTIVE_CURRENT_PA
        del arrivals
        spikes = _RELAY.accumulated_spike_mask(currents.reshape(-1)).reshape(
            vertex_count, previous_channel_count
        )
        del currents
        relay_cell_count += vertex_count * previous_channel_count

        next_frontier = np.zeros(
            (vertex_count, previous_channel_count << 1), dtype=np.bool_
        )
        for previous_code in range(previous_channel_count):
            next_frontier[
                : graph.carrier_count,
                (previous_code << 1) | _CARRIER_KIND,
            ] = spikes[: graph.carrier_count, previous_code]
            next_frontier[
                graph.carrier_count :, (previous_code << 1) | _EVENT_KIND
            ] = spikes[graph.carrier_count :, previous_code]

        bit_offset = (1 << depth) - 2
        for event_index, row in enumerate(
            next_frontier[graph.carrier_count :]
        ):
            for code, active in enumerate(row):
                if bool(active):
                    event_spike_bits[event_index] |= 1 << (bit_offset + code)
        del spikes, frontier
        frontier = next_frontier

    return (
        tuple(event_spike_bits),
        reverse_synaptic_reads,
        relay_cell_count,
    )


def project_structured_role_paths_v1(
    graph: RootedStructuredEventGraphV1,
) -> RootedStructuredRoleProjectionV1:
    """Project binary anonymous event-role factors through a depth-three relay."""

    if type(graph) is not RootedStructuredEventGraphV1:
        raise TypeError("an exact RootedStructuredEventGraphV1 is required")
    graph.authenticate()

    spike_bits, reverse_synaptic_reads, relay_cell_count = (
        _relay_path_spike_bits(graph)
    )
    if len(spike_bits) != graph.event_count:
        raise StructuredRoleProjectionError(
            "relay trace did not retain every structured event"
        )

    paths_by_bits: dict[int, StructuralRolePathV1] = {}
    factors_by_key: dict[tuple[str, str], StructuredEventRoleFactorV1] = {}
    for event_root, event_bits in zip(
        graph.event_roots, spike_bits, strict=True
    ):
        role_path = paths_by_bits.get(event_bits)
        if role_path is None:
            role_path = _project_role_path(event_bits)
            paths_by_bits[event_bits] = role_path
        key = (event_root.semantic_sha256, role_path.semantic_sha256)
        existing = factors_by_key.get(key)
        if existing is not None:
            if (
                existing.event_root != event_root
                or existing.role_path != role_path
            ):
                raise StructuredRoleProjectionError(
                    "event-role semantic projection collision"
                )
            continue
        cue_cells = event_root.trace_cells + role_path.trace_cells
        payload = {
            "event_root": event_root.sha256,
            "event_semantic": event_root.semantic_sha256,
            "role_path": role_path.sha256,
            "role_semantic": role_path.semantic_sha256,
            "cue_cells": cue_cells,
            "presence_cell": 1,
        }
        factor = StructuredEventRoleFactorV1(
            event_root=event_root,
            role_path=role_path,
            cue_cells=cue_cells,
            presence_cell=1,
            sha256=_digest(_FACTOR_NAMESPACE, payload),
        )
        factor.authenticate()
        factors_by_key[key] = factor

    factors = tuple(factors_by_key[key] for key in sorted(factors_by_key))
    vertex_count = graph.carrier_count + graph.event_count
    capacity = StructuredRoleProjectionCapacityV1(
        depth_cap=MAX_ROLE_DEPTH,
        event_count=graph.event_count,
        vertex_count=vertex_count,
        incidence_count=graph.incidence_count,
        boundary_seed_cell_count=vertex_count,
        relay_cell_count=relay_cell_count,
        relay_synapse_count=reverse_synaptic_reads,
        role_path_bits=ROLE_PATH_BITS,
        role_path_payload_bytes=ROLE_PATH_PAYLOAD_BYTES,
        factor_count=len(factors),
        factor_cap=MAX_FACTORS,
        factor_cue_cell_count=FACTOR_CUE_CELLS,
        transient_numpy_array_bound_bytes=(
            TRANSIENT_NUMPY_ARRAY_BOUND_BYTES_PER_VERTEX * vertex_count
        ),
        checkpoint_array_bytes=0,
    )
    work = StructuredRoleProjectionWorkV1(
        boundary_seed_cell_steps=vertex_count * RELAY_STEPS,
        reverse_synaptic_reads=reverse_synaptic_reads,
        relay_cell_steps=relay_cell_count * RELAY_STEPS,
        role_projection_cell_steps=sum(
            path.projection_cell_steps for path in paths_by_bits.values()
        ),
        cue_cell_reads=len(factors) * FACTOR_CUE_CELLS,
    )
    capacity.authenticate()
    work.authenticate()
    record = {
        "graph": graph.sha256,
        "factors": tuple(factor.sha256 for factor in factors),
        "source_event_count": graph.event_count,
        "factor_count": len(factors),
        "collapsed_duplicate_count": graph.event_count - len(factors),
        "role_path_count": len(paths_by_bits),
        "relay_configuration": RELAY_CONFIGURATION_SHA256,
        "capacity": capacity.record(),
        "work": work.record(),
    }
    projected = RootedStructuredRoleProjectionV1(
        graph_sha256=graph.sha256,
        factors=factors,
        source_event_count=graph.event_count,
        factor_count=len(factors),
        collapsed_duplicate_count=graph.event_count - len(factors),
        role_path_count=len(paths_by_bits),
        relay_configuration_sha256=RELAY_CONFIGURATION_SHA256,
        capacity=capacity,
        work=work,
        sha256=_digest(_VIEW_NAMESPACE, record),
    )
    projected.authenticate()
    return projected


__all__ = [
    "FACTOR_CUE_CELLS",
    "MAX_FACTORS",
    "MAX_ROLE_DEPTH",
    "RELAY_CONFIGURATION_SHA256",
    "RELAY_STEPS",
    "ROLE_PATH_BITS",
    "ROLE_PATH_PAYLOAD_BYTES",
    "ROLE_TRACE_LANES",
    "RootedStructuredRoleProjectionV1",
    "StructuralRolePathV1",
    "StructuredEventRoleFactorV1",
    "StructuredRoleProjectionCapacityV1",
    "StructuredRoleProjectionError",
    "StructuredRoleProjectionWorkV1",
    "TRANSIENT_NUMPY_ARRAY_BOUND_BYTES_PER_VERTEX",
    "project_structured_role_paths_v1",
]
