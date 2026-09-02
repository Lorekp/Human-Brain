from __future__ import annotations

import json
from itertools import product

from experiments.r4_integrated import (
    causal_sequence_composer_fixture_v1 as fixture,
)

EXPECTED_PAYLOAD_SHA256 = (
    "28a0801a7e9b50e6235c6a793c9481c186e056ba0e8b48a3c332e31539c18e56"
)
EXPECTED_CUSTODY_SHA256 = (
    "c98abb519bbd6a4cc81b291cb1c45d39f21b991752353cdd2f980f4a5af40439"
)


def _payload_pair(emissions: fixture.Emission) -> tuple[bytes, bytes]:
    return emissions[0][0], emissions[1][0]


def test_fixture_roots_and_public_data_are_frozen() -> None:
    fixture.authenticate_fixture()

    assert fixture.CAUSAL_SEQUENCE_PAYLOAD_SHA256 == EXPECTED_PAYLOAD_SHA256
    assert fixture.CAUSAL_SEQUENCE_CUSTODY_SHA256 == EXPECTED_CUSTODY_SHA256

    payload = fixture.public_payload()
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in (
        "python",
        "ast",
        "task",
        "label",
        "answer",
        "target",
        "candidate",
        "solution",
        "correct",
    ):
        assert forbidden not in serialized

    assert tuple(payload) == ("schema", "lessons", "queries")
    assert all(
        tuple(query.public_observation())
        == ("schema", "cue_events", "role_inventory", "observation_sha256")
        for query in fixture.QUERIES
    )


def test_three_grid_cells_identify_a_fresh_role_bound_composition() -> None:
    combinations = tuple(
        fixture._cue_combinations(lesson.cue_events)[0]
        for lesson in fixture.TRAINING_LESSONS
    )
    held_out_combination = fixture._cue_combinations(
        fixture.HELD_OUT_CASE.cue_events
    )

    assert len(combinations) == len(set(combinations)) == 3
    assert set(combinations) | set(held_out_combination) == set(
        product(range(2), repeat=2)
    )
    assert set(combinations).isdisjoint(held_out_combination)

    all_roles = tuple(
        role
        for item in (*fixture.TRAINING_LESSONS, *fixture.QUERIES)
        for role in item.role_inventory
    )
    assert len(all_roles) == len(set(all_roles))
    assert all(len(role) == 1 for role in all_roles)
    assert all(
        tuple(role for _chunk, role in lesson.demonstrated_emissions)
        == lesson.role_inventory
        for lesson in fixture.TRAINING_LESSONS
    )

    held_out = fixture.scoring_emissions(fixture.HELD_OUT_CASE)
    assert held_out is not None
    assert tuple(role for _chunk, role in held_out) == fixture.HELD_OUT_CASE.role_inventory

    training_pairs = {
        _payload_pair(lesson.demonstrated_emissions)
        for lesson in fixture.TRAINING_LESSONS
    }
    held_out_pair = _payload_pair(held_out)
    assert held_out_pair not in training_pairs
    assert set(held_out_pair) <= {chunk for pair in training_pairs for chunk in pair}


def test_unresolved_cues_and_branch_local_revision_evidence() -> None:
    assert tuple(
        len(fixture._cue_combinations(probe.cue_events)) for probe in fixture.PROBES
    ) == (2, 0)
    assert all(fixture.scoring_emissions(probe) is None for probe in fixture.PROBES)

    training_pairs = {
        _payload_pair(lesson.demonstrated_emissions)
        for lesson in fixture.TRAINING_LESSONS
    }
    held_out = fixture.scoring_emissions(fixture.HELD_OUT_CASE)
    assert held_out is not None
    held_out_left, alternate_right = _payload_pair(held_out)

    selected_rights = {
        right for left, right in training_pairs if left == held_out_left
    }
    assert len(selected_rights) == 1
    selected_right = next(iter(selected_rights))
    assert selected_right != alternate_right

    donor_lefts = {
        left for left, right in training_pairs if right == alternate_right
    }
    assert donor_lefts
    assert any((donor, selected_right) in training_pairs for donor in donor_lefts)
    assert (held_out_left, selected_right) in training_pairs
    assert (held_out_left, alternate_right) not in training_pairs
