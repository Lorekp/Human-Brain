from __future__ import annotations

import pytest
from humanbrain.role_relative_compositor import (
    CompositionAblation,
    PendingConsequenceError,
    RoleRelativeSequenceCompositor,
    RoleSequenceDemonstration,
    Valence,
)


def _demonstration(
    context: bytes,
    roles: tuple[bytes, bytes],
    emissions: tuple[bytes, ...],
) -> RoleSequenceDemonstration:
    return RoleSequenceDemonstration(
        context_event=context,
        role_inventory=roles,
        emissions=emissions,
    )


def test_opaque_role_composition_and_chain_local_revision_are_causal() -> None:
    first = b"\x10"
    second = b"\x20"
    ambiguous = b"\x7f"
    revision = b"\x70"
    demonstrations = (
        _demonstration(first, (b"\x01", b"\x02"), (b"\xf0", b"\x01", b"\xf1")),
        _demonstration(first, (b"\x03", b"\x04"), (b"\xf0", b"\x03", b"\xf1")),
        _demonstration(second, (b"\x05", b"\x06"), (b"\xe0", b"\x06", b"\xe1")),
        _demonstration(second, (b"\x07", b"\x08"), (b"\xe0", b"\x08", b"\xe1")),
        _demonstration(ambiguous, (b"\x31", b"\x32"), (b"\xa0", b"\x31")),
        _demonstration(ambiguous, (b"\x33", b"\x34"), (b"\xa0", b"\x33")),
        _demonstration(ambiguous, (b"\x35", b"\x36"), (b"\xb0", b"\x36")),
        _demonstration(ambiguous, (b"\x37", b"\x38"), (b"\xb0", b"\x38")),
        _demonstration(revision, (b"\x41", b"\x42"), (b"\xc0", b"\x41")),
        _demonstration(revision, (b"\x43", b"\x44"), (b"\xc0", b"\x43")),
        _demonstration(revision, (b"\x45", b"\x46"), (b"\xc0", b"\x45")),
        _demonstration(revision, (b"\x51", b"\x52"), (b"\xd0", b"\x52")),
        _demonstration(revision, (b"\x53", b"\x54"), (b"\xd0", b"\x54")),
    )
    model = RoleRelativeSequenceCompositor.grow(demonstrations)
    checkpoint = model.checkpoint_sha256
    roles = (b"\x91", b"\x92")

    result = model.respond((b"\x99", first, b"\x98", second, b"\x97"), roles)

    assert result.decision == "EMIT"
    assert result.emissions == (
        b"\xf0",
        b"\x91",
        b"\xf1",
        b"\xe0",
        b"\x92",
        b"\xe1",
    )
    assert result.emissions not in {row.emissions for row in demonstrations}
    assert result.eligibility is not None
    assert result.eligibility.checkpoint_sha256 == checkpoint
    assert result.eligibility.query_sha256 == result.query_sha256
    assert result.eligibility.physical_cell_steps == result.physical_cell_steps
    assert len(result.eligibility.transitions) == len(result.races)
    assert {edge.bank for edge in result.eligibility.transitions} == {
        "chunk_root",
        "context_chunk",
        "role_binding",
        "state_boundary",
        "state_emission",
        "state_next",
    }
    assert all(not race.stopped for race in result.races)
    assert all(race.simultaneous_spike_candidates == 1 for race in result.races)
    assert model.training_summary.recurring_chunks == 6

    neutral = model.resolve(Valence.NEUTRAL)
    assert neutral.changed_transitions == ()
    assert neutral.prior_training_sha256 == checkpoint
    assert neutral.training_sha256 == checkpoint
    assert model.checkpoint_sha256 == checkpoint

    compositor_off = model.respond(
        (first, b"\x98", second),
        roles,
        ablation=CompositionAblation.COMPOSITOR_OFF,
    )
    recurrence_off = model.respond(
        (first, second),
        roles,
        ablation=CompositionAblation.RECURRENCE_OFF,
    )
    collision = model.respond((ambiguous,), roles)

    assert compositor_off.decision == "STOP"
    assert compositor_off.emissions == ()
    assert compositor_off.eligibility is None
    assert compositor_off.stop_reason == "context_chunk:tonic_stop"
    assert recurrence_off.decision == "STOP"
    assert recurrence_off.emissions == ()
    assert recurrence_off.eligibility is None
    assert recurrence_off.stop_reason == "state_next:tonic_stop"
    assert collision.decision == "STOP"
    assert collision.emissions == ()
    assert collision.eligibility is None
    assert collision.stop_reason == "context_chunk:collision"
    assert collision.races[-1].simultaneous_spike_candidates == 2
    assert collision.races[-1].stop_relay_spike_step is not None

    first_attempt = model.respond((b"\x99", revision, b"\x98"), roles)
    assert first_attempt.emissions == (b"\xc0", b"\x91")
    with pytest.raises(PendingConsequenceError, match="resolve the pending"):
        model.respond((revision,), roles)

    negative = model.resolve(Valence.NEGATIVE)
    assert negative.prior_training_sha256 == checkpoint
    assert negative.training_sha256 != checkpoint
    assert len(negative.changed_transitions) == 1
    assert negative.changed_transitions[0].bank == "context_chunk"
    with pytest.raises(PendingConsequenceError, match="no pending"):
        model.resolve(Valence.NEGATIVE)

    revised = model.respond((revision,), roles)
    assert revised.emissions == (b"\xd0", b"\x92")
    model.resolve(Valence.NEUTRAL)

    control = RoleRelativeSequenceCompositor.grow(demonstrations)
    before_control = control.checkpoint_sha256
    control_first = control.respond((revision,), roles, ablation=CompositionAblation.NOGO_OFF)
    no_update = control.resolve(Valence.NEGATIVE)
    control_repeat = control.respond((revision,), roles, ablation=CompositionAblation.NOGO_OFF)
    control.resolve(Valence.NEUTRAL)

    assert control_first.emissions == (b"\xc0", b"\x91")
    assert control_repeat.emissions == control_first.emissions
    assert control_repeat.races[0].winner_cell == control_first.races[0].winner_cell
    assert no_update.changed_transitions == ()
    assert no_update.prior_training_sha256 == before_control
    assert no_update.training_sha256 == before_control
    assert control.checkpoint_sha256 == before_control
