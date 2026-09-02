from __future__ import annotations

from dataclasses import fields
from inspect import signature

import pytest
from humanbrain.variability_aware_segment_binder import (
    VariabilityAwareSegmentBinder,
    VariabilityBinderAblation,
    VariabilitySegmentDemonstration,
)

_LEFT = b"\xd1"
_STABLE = b"\xd2"
_RIGHT = b"\xd3"

_Row = tuple[tuple[bytes, ...], bytes, bytes]


def _rows(prefix: bytes) -> tuple[_Row, ...]:
    rows: list[_Row] = []
    for ordinal in range(3):
        suffix = str(ordinal).encode("ascii")
        seed = prefix + b"-seed-" + suffix
        role = prefix + b"-role-" + suffix
        segment = (
            _LEFT,
            seed,
            prefix + b"-variable-a-" + suffix,
            _STABLE,
            prefix + b"-variable-b-" + suffix,
            role,
            _RIGHT,
        )
        rows.append((segment, seed, role))
    return tuple(rows)


def _source(rows: tuple[_Row, ...]) -> tuple[bytes, ...]:
    return tuple(event for segment, _, _ in rows for event in segment)


def _demonstration(prefix: bytes, target_index: int) -> VariabilitySegmentDemonstration:
    rows = _rows(prefix)
    _, seed, role = rows[target_index]
    return VariabilitySegmentDemonstration(
        query_events=(prefix + b"-query", seed, prefix + b"-request"),
        source_events=_source(rows),
        demonstrated_role=role,
    )


def _binder(prefix: bytes = b"training") -> VariabilityAwareSegmentBinder:
    return VariabilityAwareSegmentBinder.grow(
        tuple(
            _demonstration(prefix + b"-" + str(position).encode("ascii"), position)
            for position in range(3)
        )
    )


def _fresh(position: int) -> tuple[tuple[bytes, ...], tuple[bytes, ...], bytes]:
    prefix = b"fresh-" + str(position).encode("ascii")
    rows = _rows(prefix)
    _, seed, role = rows[position]
    return (b"fresh-query", seed, b"fresh-request"), _source(rows), role


def test_variable_hops_transfer_fresh_names_at_first_middle_and_last_segments() -> None:
    binder = _binder()

    decisions = tuple(binder.respond(*_fresh(position)[:2]) for position in range(3))

    assert tuple(decision.decision for decision in decisions) == ("BIND", "BIND", "BIND")
    assert tuple(decision.role_identity for decision in decisions) == tuple(
        _fresh(position)[2] for position in range(3)
    )
    assert all(decision.boundary_match_count == 2 for decision in decisions)
    assert all(
        decision.physical_cell_steps == sum(race.physical_cell_steps for race in decision.races)
        for decision in decisions
    )
    assert binder.training_summary.demonstration_count == 3
    assert binder.training_summary.role_offset == 4
    assert binder.training_summary.propagation_hop_count == 4
    assert binder.training_summary.stable_intermediate_synapse_count == 1
    assert binder.training_summary.wildcard_intermediate_hop_count == 2
    assert binder.training_summary.wildcard_support_ticks == 6
    assert binder.training_summary.boundary_synapse_count == 2
    assert binder.training_summary.left_boundary_hop == 5
    assert binder.training_summary.right_boundary_hop == 1
    assert binder.checkpoint_sha256 == _binder(b"independent-renaming").checkpoint_sha256


def test_stable_hop_corruption_and_unsupported_depth_physically_stop() -> None:
    binder = _binder()
    query, source, role = _fresh(1)
    role_column = source.index(role)

    corrupted = list(source)
    corrupted[role_column - 2] = b"fresh-corrupted-stable-event"
    stable_decision = binder.respond(query, tuple(corrupted))

    shifted = list(source)
    shifted.insert(role_column, b"fresh-extra-variable-depth")
    depth_decision = binder.respond(query, tuple(shifted))

    assert stable_decision.decision == "STOP"
    assert stable_decision.stop_stage == "propagation"
    assert stable_decision.stop_reason == "propagation:tonic_stop"
    assert stable_decision.races[-1].population == "recurrent_stable_hop_2"
    assert stable_decision.races[-1].stopped
    assert depth_decision.decision == "STOP"
    assert depth_decision.stop_stage == "boundary"
    assert depth_decision.stop_reason == "boundary:right_missing_boundary"
    assert depth_decision.races[-1].stopped


def test_ambiguity_and_missing_boundary_physically_stop() -> None:
    binder = _binder()
    rows = _rows(b"ambiguous")
    source = _source(rows)

    ambiguity = binder.respond((rows[0][1], rows[2][1]), source)

    query, fresh_source, role = _fresh(1)
    missing = list(fresh_source)
    missing[fresh_source.index(role) + 1] = b"fresh-missing-boundary"
    boundary = binder.respond(query, tuple(missing))

    assert ambiguity.decision == "STOP"
    assert ambiguity.stop_stage == "equality"
    assert ambiguity.stop_reason == "equality:collision"
    assert ambiguity.races[-1].stopped
    assert ambiguity.races[-1].simultaneous_spike_cells == 2
    assert boundary.decision == "STOP"
    assert boundary.stop_stage == "boundary"
    assert boundary.stop_reason == "boundary:right_missing_boundary"
    assert boundary.races[-1].stopped


def test_final_role_identity_collision_physically_stops() -> None:
    binder = _binder()
    query, source, expected_role = _fresh(1)
    rows = _rows(b"fresh-1")
    duplicated = list(source)
    duplicated[duplicated.index(rows[0][2])] = expected_role

    decision = binder.respond(query, tuple(duplicated))

    assert decision.decision == "STOP"
    assert decision.stop_stage == "identity"
    assert decision.stop_reason == "identity:collision"
    assert decision.boundary_match_count == 2
    assert decision.races[-1].population == "role_identity_uniqueness"
    assert decision.races[-1].simultaneous_spike_cells == 2
    assert decision.races[-1].stopped


def test_equality_propagation_and_boundary_ablations_each_physically_stop() -> None:
    binder = _binder()
    query, source, _ = _fresh(0)
    checkpoint = binder.checkpoint_sha256

    equality = binder.respond(
        query,
        source,
        ablation=VariabilityBinderAblation.EQUALITY_OFF,
    )
    propagation = binder.respond(
        query,
        source,
        ablation=VariabilityBinderAblation.PROPAGATION_OFF,
    )
    boundary = binder.respond(
        query,
        source,
        ablation=VariabilityBinderAblation.BOUNDARY_OFF,
    )

    assert equality.decision == "STOP" and equality.stop_stage == "equality"
    assert propagation.decision == "STOP" and propagation.stop_stage == "propagation"
    assert boundary.decision == "STOP" and boundary.stop_stage == "boundary"
    assert equality.races[-1].stopped
    assert propagation.races[-1].stopped
    assert boundary.races[-1].stopped
    assert binder.checkpoint_sha256 == checkpoint


def test_growth_requires_independent_evidence_and_api_has_no_routing_hints() -> None:
    demonstration = _demonstration(b"duplicate", 1)

    with pytest.raises(ValueError, match="at least two independent"):
        VariabilityAwareSegmentBinder.grow((demonstration,))
    with pytest.raises(ValueError, match="independent, not duplicate"):
        VariabilityAwareSegmentBinder.grow((demonstration, demonstration))

    assert tuple(field.name for field in fields(VariabilitySegmentDemonstration)) == (
        "query_events",
        "source_events",
        "demonstrated_role",
    )
    assert tuple(signature(VariabilityAwareSegmentBinder.respond).parameters) == (
        "self",
        "query_events",
        "source_events",
        "ablation",
    )
