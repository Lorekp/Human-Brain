from __future__ import annotations

from dataclasses import fields
from inspect import signature

from humanbrain.cross_stream_segment_binder import (
    CrossStreamSegmentBinder,
    SegmentBinderAblation,
    SegmentRoleDemonstration,
)

_OPEN = b"\xf0"
_INNER_ONE = b"\xf1"
_INNER_TWO = b"\xf2"
_TAIL = b"\xf3"
_CLOSE = b"\xf4"


def _source(rows: tuple[tuple[bytes, bytes], ...]) -> tuple[bytes, ...]:
    return tuple(
        event
        for role, discriminator in rows
        for event in (
            _OPEN,
            role,
            _INNER_ONE,
            _INNER_TWO,
            discriminator,
            _TAIL,
            _CLOSE,
        )
    )


def _demonstration(
    prefix: bytes,
    target_index: int,
) -> SegmentRoleDemonstration:
    rows = tuple(
        (
            prefix + b"-role-" + str(index).encode("ascii"),
            prefix + b"-disc-" + str(index).encode("ascii"),
        )
        for index in range(3)
    )
    target_role, target_discriminator = rows[target_index]
    return SegmentRoleDemonstration(
        query_events=(prefix + b"-query", target_discriminator, prefix + b"-tail"),
        source_events=_source(rows),
        demonstrated_role=target_role,
    )


def _binder() -> CrossStreamSegmentBinder:
    return CrossStreamSegmentBinder.grow(
        (
            _demonstration(b"alpha", 0),
            _demonstration(b"beta", 1),
            _demonstration(b"gamma", 2),
        )
    )


def _fresh_stream() -> tuple[tuple[bytes, ...], tuple[bytes, ...], bytes]:
    rows = (
        (b"fresh-role-amber", b"fresh-disc-amber"),
        (b"fresh-role-cobalt", b"fresh-disc-cobalt"),
        (b"fresh-role-denim", b"fresh-disc-denim"),
    )
    query = (b"fresh-query", rows[1][1], b"fresh-request-tail")
    return query, _source(rows), rows[1][0]


def test_fresh_equality_propagates_to_role_in_one_of_three_matched_segments() -> None:
    binder = _binder()
    query, source, expected_role = _fresh_stream()

    decision = binder.respond(query, source)

    assert decision.decision == "BIND"
    assert decision.role_identity == expected_role
    assert decision.stop_stage is None
    assert decision.stop_reason is None
    assert decision.containment_match_count == 5
    assert tuple(race.population for race in decision.races) == (
        "query_source_equality",
        "recurrent_propagation_hop_1",
        "recurrent_propagation_hop_2",
        "recurrent_propagation_hop_3",
        "segment_containment_role",
        "role_identity_uniqueness",
    )
    assert decision.physical_cell_steps == sum(race.physical_cell_steps for race in decision.races)
    assert binder.training_summary.demonstration_count == 3
    assert binder.training_summary.role_offset == -3
    assert binder.training_summary.propagation_hop_count == 3
    assert binder.training_summary.intermediate_event_synapse_count == 2
    assert binder.training_summary.containment_synapse_count == 5


def test_missing_duplicate_or_broken_containment_discriminator_physically_stops() -> None:
    binder = _binder()
    query, source, _ = _fresh_stream()

    omitted = binder.respond((b"not-present", b"also-absent"), source)
    assert omitted.decision == "STOP"
    assert omitted.stop_stage == "equality"
    assert omitted.stop_reason == "equality:tonic_stop"
    assert omitted.races[-1].stopped

    duplicated_source = list(source)
    duplicated_source[4] = query[1]
    duplicated = binder.respond(query, tuple(duplicated_source))
    assert duplicated.decision == "STOP"
    assert duplicated.stop_stage == "equality"
    assert duplicated.stop_reason == "equality:collision"
    assert duplicated.races[-1].simultaneous_spike_candidates == 2

    broken_envelope = list(source)
    broken_envelope[13] = b"fresh-unseen-close"
    broken = binder.respond(query, tuple(broken_envelope))
    assert broken.decision == "STOP"
    assert broken.stop_stage == "containment"
    assert broken.stop_reason == "containment:tonic_stop"
    assert broken.containment_match_count == 4


def test_equality_and_recurrent_propagation_ablations_causally_stop() -> None:
    binder = _binder()
    query, source, _ = _fresh_stream()
    checkpoint = binder.checkpoint_sha256

    equality_off = binder.respond(
        query,
        source,
        ablation=SegmentBinderAblation.EQUALITY_OFF,
    )
    propagation_off = binder.respond(
        query,
        source,
        ablation=SegmentBinderAblation.PROPAGATION_OFF,
    )

    assert equality_off.decision == "STOP"
    assert equality_off.stop_stage == "equality"
    assert equality_off.races[-1].stopped
    assert propagation_off.decision == "STOP"
    assert propagation_off.stop_stage == "propagation"
    assert propagation_off.races[-1].stopped
    assert binder.checkpoint_sha256 == checkpoint


def test_public_surfaces_contain_no_boundary_span_or_candidate_routing() -> None:
    assert tuple(field.name for field in fields(SegmentRoleDemonstration)) == (
        "query_events",
        "source_events",
        "demonstrated_role",
    )
    assert tuple(signature(CrossStreamSegmentBinder.respond).parameters) == (
        "self",
        "query_events",
        "source_events",
        "ablation",
    )
