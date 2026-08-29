from __future__ import annotations

import inspect

import pytest

from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from humanbrain.resting_glif_tier_beam import RETAINED, STOP_LEADING_TIER


def _encode(value: object) -> output_cells.TypedOutputEncoding:
    return output_cells.TypedOutputCellBank.encode(value)


def test_unique_all_case_semantic_vector_wins_physical_tier() -> None:
    desired = (_encode(2), _encode(-3))
    keys = ("wrong-right", "exact", "wrong-left")
    candidates = (
        (_encode(2), _encode(-2)),
        (_encode(2), _encode(-3)),
        (_encode(1), _encode(-3)),
    )

    audit = semantic_race.select_exact_semantic_vector(keys, candidates, desired)
    repeated = semantic_race.select_exact_semantic_vector(keys, candidates, desired)

    assert audit.status == semantic_race.SELECTED
    assert audit.winner_key == "exact"
    assert audit.exact_winner_keys == ("exact",)
    assert audit.exact_match_spikes == ((1, 0), (1, 1), (0, 1))
    assert audit.currents_pa == (600.0, 3_400.0, 600.0, 2_000.0)
    assert audit.stop_cell == 3
    assert audit.tier.status == RETAINED
    assert audit.tier.rounds[0].tier_cells == (1,)
    assert audit.tier.retained_candidate_cells == (1,)
    assert (
        audit.tier.inhibition_projection_before_sha256
        == audit.tier.inhibition_projection_after_sha256
    )
    assert audit.detector_sha256s == tuple(
        output_cells.FixedTypedCoincidenceDetector.for_target(target).sha256
        for target in desired
    )
    assert audit.sha256 == repeated.sha256
    assert audit.currents_sha256 == repeated.currents_sha256
    assert audit.tier_sha256 == repeated.tier_sha256


def test_stop_cell_wins_when_no_vector_is_exact_on_every_case() -> None:
    desired = (_encode(4), _encode(-1))
    audit = semantic_race.select_exact_semantic_vector(
        ("left-only", "right-only"),
        (
            (_encode(4), _encode(0)),
            (_encode(3), _encode(-1)),
        ),
        desired,
    )

    assert audit.status == semantic_race.STOP_NO_EXACT
    assert audit.winner_key is None
    assert audit.exact_winner_keys == ()
    assert audit.currents_pa == (600.0, 600.0, 2_000.0)
    assert audit.tier.status == STOP_LEADING_TIER
    assert audit.tier.rounds[0].tier_cells == (audit.stop_cell,)
    assert audit.tier.retained_candidate_cells == ()


def test_multiple_exact_vectors_stop_without_candidate_index_tie_break() -> None:
    desired = (_encode(7), _encode(0))
    exact = (_encode(7), _encode(0))
    audit = semantic_race.select_exact_semantic_vector(
        ("first", "second", "wrong"),
        (exact, exact, (_encode(7), _encode(1))),
        desired,
    )

    assert audit.status == semantic_race.STOP_MULTIPLE_EXACT
    assert audit.winner_key is None
    assert audit.exact_winner_keys == ("first", "second")
    assert audit.tier.rounds[0].tier_cells == (0, 1)
    assert audit.tier.rounds[0].race.simultaneous_spike_candidates == 2


def test_interface_rejects_misalignment_and_contains_no_host_ranking() -> None:
    desired = (_encode(1),)

    with pytest.raises(ValueError, match="unique"):
        semantic_race.select_exact_semantic_vector(
            ("same", "same"),
            ((_encode(1),), (_encode(0),)),
            desired,
        )
    with pytest.raises(ValueError, match="align"):
        semantic_race.select_exact_semantic_vector(
            ("one",),
            ((_encode(1), _encode(2)),),
            desired,
        )
    with pytest.raises(ValueError, match="supported"):
        semantic_race.select_exact_semantic_vector(
            ("one",),
            ((_encode(1),),),
            (_encode(output_cells.INT_MAX + 1),),
        )

    signature = inspect.signature(semantic_race.select_exact_semantic_vector)
    assert tuple(signature.parameters) == (
        "candidate_keys",
        "candidate_outputs",
        "desired_outputs",
    )
    source = inspect.getsource(semantic_race.select_exact_semantic_vector)
    for forbidden in ("argmax", "argsort", "argpartition", "sorted(", ".sort("):
        assert forbidden not in source
