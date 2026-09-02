"""Sparse exact scoring for relative-event occurrence neighborhoods.

The scorer is a computational index only.  It preserves one integer score for
every source occurrence and does not select a candidate, resolve a tie, or
implement STOP.  Callers keep those decisions in their neuronal race.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence, Set
from typing import Generic, TypeVar

__all__ = ["OccurrenceNeighborhoodScorer"]


EventT = TypeVar("EventT", bound=Hashable)


class OccurrenceNeighborhoodScorer(Generic[EventT]):
    """Reuse an event-to-position index across role-neighborhood score vectors."""

    __slots__ = ("_positions_by_event", "_radius", "_source_events")

    def __init__(
        self,
        source_events: Sequence[EventT],
        neighborhood_radius: int,
    ) -> None:
        if type(neighborhood_radius) is not int or neighborhood_radius < 1:
            raise ValueError("neighborhood_radius must be a positive integer")

        events = tuple(source_events)
        mutable_positions: dict[EventT, list[int]] = {}
        for position, event in enumerate(events):
            mutable_positions.setdefault(event, []).append(position)

        self._source_events = events
        self._radius = neighborhood_radius
        self._positions_by_event = {
            event: tuple(positions) for event, positions in mutable_positions.items()
        }

    @property
    def candidate_count(self) -> int:
        """Number of source occurrences represented in every returned score vector."""

        return len(self._source_events)

    def score(
        self,
        learned_features: Set[tuple[int, EventT]],
        *,
        excluded_center_events: Set[EventT] = frozenset(),
    ) -> tuple[int, ...]:
        """Count distinct learned relative-event matches at every source center.

        Features outside the configured neighborhood, including the center at
        offset zero, cannot contribute.  Exclusions suppress a center after
        scoring, matching a caller that keeps already-bound identities in the
        physical candidate population with zero current.
        """

        scores = [0] * len(self._source_events)
        radius = self._radius
        positions_by_event = self._positions_by_event
        candidate_count = len(scores)

        for relative_offset, neighbor_event in learned_features:
            if relative_offset == 0 or not -radius <= relative_offset <= radius:
                continue
            for neighbor_position in positions_by_event.get(neighbor_event, ()):
                center = neighbor_position - relative_offset
                if 0 <= center < candidate_count:
                    scores[center] += 1

        for excluded_event in excluded_center_events:
            for center in positions_by_event.get(excluded_event, ()):
                scores[center] = 0

        return tuple(scores)
