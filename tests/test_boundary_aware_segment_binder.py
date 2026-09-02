from __future__ import annotations

from dataclasses import fields
from inspect import signature

from humanbrain.boundary_aware_segment_binder import (
    BoundaryAwareSegmentBinder,
    BoundaryBinderAblation,
    BoundarySegmentDemonstration,
)

_OPEN = b"\xe1"
_BRIDGE = b"\xe2"
_TAIL = b"\xe3"
_CLOSE = b"\xe4"


def _rows(prefix: bytes) -> tuple[tuple[bytes, bytes, bytes], ...]:
    return tuple(
        (
            prefix + b"-role-" + str(ordinal).encode("ascii"),
            prefix + b"-seed-" + str(ordinal).encode("ascii"),
            prefix + b"-payload-" + str(ordinal).encode("ascii"),
        )
        for ordinal in range(3)
    )


def _source(rows: tuple[tuple[bytes, bytes, bytes], ...]) -> tuple[bytes, ...]:
    return tuple(
        event
        for role, seed, payload in rows
        for event in (_OPEN, role, _BRIDGE, seed, payload, _TAIL, _CLOSE)
    )


def _demonstration(prefix: bytes) -> BoundarySegmentDemonstration:
    rows = _rows(prefix)
    role, seed, _ = rows[1]
    return BoundarySegmentDemonstration(
        query_events=(prefix + b"-query", seed, prefix + b"-request"),
        source_events=_source(rows),
        demonstrated_role=role,
    )


def _binder() -> BoundaryAwareSegmentBinder:
    return BoundaryAwareSegmentBinder.grow(
        (
            _demonstration(b"amber"),
            _demonstration(b"cobalt"),
            _demonstration(b"denim"),
        )
    )


def _fresh(position: int) -> tuple[tuple[bytes, ...], tuple[bytes, ...], bytes]:
    rows = _rows(b"fresh-" + str(position).encode("ascii"))
    role, seed, _ = rows[position]
    query = (b"fresh-query", seed, b"fresh-request")
    return query, _source(rows), role


def test_fresh_names_bind_at_first_middle_and_last_segment_positions() -> None:
    binder = _binder()

    decisions = tuple(binder.respond(*_fresh(position)[:2]) for position in range(3))

    assert tuple(decision.decision for decision in decisions) == ("BIND", "BIND", "BIND")
    assert tuple(decision.role_identity for decision in decisions) == tuple(
        _fresh(position)[2] for position in range(3)
    )
    assert all(decision.boundary_match_count == 2 for decision in decisions)
    assert binder.training_summary.demonstration_count == 3
    assert binder.training_summary.role_offset == -2
    assert binder.training_summary.propagation_hop_count == 2
    assert binder.training_summary.boundary_synapse_count == 2
    assert binder.training_summary.left_boundary_hop_bound == 1
    assert binder.training_summary.right_boundary_hop_bound == 1


def test_genuine_cross_segment_ambiguity_physically_stops() -> None:
    binder = _binder()
    rows = _rows(b"ambiguous")
    source = _source(rows)

    decision = binder.respond((rows[0][1], rows[2][1]), source)

    assert decision.decision == "STOP"
    assert decision.stop_stage == "equality"
    assert decision.stop_reason == "equality:collision"
    assert decision.races[-1].stopped
    assert decision.races[-1].simultaneous_spike_cells == 2


def test_missing_nearest_boundary_physically_stops_without_crossing_segments() -> None:
    binder = _binder()
    query, source, role = _fresh(1)
    changed = list(source)
    role_column = changed.index(role)
    changed[role_column - 1] = b"unseen-local-delimiter"

    decision = binder.respond(query, tuple(changed))

    assert decision.decision == "STOP"
    assert decision.stop_stage == "boundary"
    assert decision.stop_reason == "boundary:left_missing_boundary"
    assert decision.boundary_match_count == 0
    assert decision.races[-1].population == "left_boundary_missing"
    assert decision.races[-1].stopped


def test_equality_propagation_and_boundary_ablations_each_physically_stop() -> None:
    binder = _binder()
    query, source, _ = _fresh(0)
    checkpoint = binder.checkpoint_sha256

    equality = binder.respond(
        query,
        source,
        ablation=BoundaryBinderAblation.EQUALITY_OFF,
    )
    propagation = binder.respond(
        query,
        source,
        ablation=BoundaryBinderAblation.PROPAGATION_OFF,
    )
    boundary = binder.respond(
        query,
        source,
        ablation=BoundaryBinderAblation.BOUNDARY_OFF,
    )

    assert equality.decision == "STOP" and equality.stop_stage == "equality"
    assert propagation.decision == "STOP" and propagation.stop_stage == "propagation"
    assert boundary.decision == "STOP" and boundary.stop_stage == "boundary"
    assert equality.races[-1].stopped
    assert propagation.races[-1].stopped
    assert boundary.races[-1].stopped
    assert binder.checkpoint_sha256 == checkpoint


def test_public_interface_has_no_delimiter_or_location_routing_input() -> None:
    assert tuple(field.name for field in fields(BoundarySegmentDemonstration)) == (
        "query_events",
        "source_events",
        "demonstrated_role",
    )
    assert tuple(signature(BoundaryAwareSegmentBinder.respond).parameters) == (
        "self",
        "query_events",
        "source_events",
        "ablation",
    )
