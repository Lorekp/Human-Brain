from __future__ import annotations

from collections.abc import Hashable, Sequence, Set
from random import Random
from typing import TypeVar

import pytest
from humanbrain.occurrence_neighborhood_scorer import OccurrenceNeighborhoodScorer

EventT = TypeVar("EventT", bound=Hashable)


def _scalar_scores(
    source_events: Sequence[EventT],
    radius: int,
    learned_features: Set[tuple[int, EventT]],
    excluded_center_events: Set[EventT],
) -> tuple[int, ...]:
    scores: list[int] = []
    for center, center_event in enumerate(source_events):
        neighborhood = frozenset(
            (offset, source_events[neighbor])
            for offset in range(-radius, radius + 1)
            if offset != 0
            if 0 <= (neighbor := center + offset) < len(source_events)
        )
        scores.append(
            0 if center_event in excluded_center_events else len(learned_features & neighborhood)
        )
    return tuple(scores)


def test_inverted_index_has_exact_scalar_parity_across_roles_and_boundaries() -> None:
    generator = Random(910_247)
    vocabulary = tuple(f"event-{index}".encode() for index in range(19))

    for source_length in (1, 2, 7, 31, 129):
        source = tuple(generator.choice(vocabulary) for _ in range(source_length))
        for radius in (1, 2, 4):
            scorer = OccurrenceNeighborhoodScorer(source, radius)
            assert scorer.candidate_count == source_length
            for _role in range(12):
                learned_features = frozenset(
                    (
                        generator.randint(-radius - 2, radius + 2),
                        generator.choice(vocabulary),
                    )
                    for _ in range(2 * radius + 5)
                )
                excluded = frozenset(generator.sample(vocabulary, k=3))

                assert scorer.score(
                    learned_features,
                    excluded_center_events=excluded,
                ) == _scalar_scores(source, radius, learned_features, excluded)


def test_full_source_population_and_zero_scores_remain_in_source_order() -> None:
    source = (b"left", b"marker", b"repeat", b"marker", b"right", b"repeat")
    scorer = OccurrenceNeighborhoodScorer(source, neighborhood_radius=2)
    features = frozenset(
        {
            (-2, b"left"),
            (-1, b"marker"),
            (0, b"repeat"),
            (3, b"right"),
            (1, b"absent"),
        }
    )

    assert scorer.score(features) == _scalar_scores(source, 2, features, frozenset())
    assert scorer.score(
        features,
        excluded_center_events=frozenset({b"repeat"}),
    ) == _scalar_scores(source, 2, features, frozenset({b"repeat"}))
    assert len(scorer.score(frozenset())) == len(source)
    assert scorer.score(frozenset()) == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("radius", [0, -1, True, 1.5])
def test_radius_requires_a_positive_integer(radius: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        OccurrenceNeighborhoodScorer((b"event",), radius)  # type: ignore[arg-type]
