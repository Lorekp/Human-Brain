"""Physical resting-GLIF tier enumeration bounded by a STOP-cell frontier.

The complete population races from rest in every round.  Each earliest
simultaneous-spike tier is retained and self-inhibited through the committed
O(N) fixed projection.  Enumeration ends immediately after the first complete
tier containing STOP, so cells physically ranked below STOP are never awaited.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from humanbrain.cells import AllenGlif5Parameters
from humanbrain.resting_glif_tier_beam import CompiledRestingGlifTierBeam
from humanbrain.resting_glif_race import RestingGlifRaceResult


STOP_FRONTIER_REACHED = "STOP_FRONTIER_REACHED"


@dataclass(frozen=True, slots=True)
class StopBoundedTierRound:
    """One complete physical tier and whether it contains the STOP cell."""

    tier_index: int
    race: RestingGlifRaceResult
    tier_cells: tuple[int, ...]
    stop_frontier: bool
    currents_before_pa: tuple[float, ...]
    currents_after_pa: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StopBoundedTierResult:
    """All physical tiers through STOP and their admissibility ledger."""

    status: str
    stop_cell: int
    admissible_candidate_cells: tuple[int, ...]
    excluded_candidate_cells: tuple[int, ...]
    rounds: tuple[StopBoundedTierRound, ...]
    final_currents_pa: tuple[float, ...]
    inhibition_projection_before_sha256: str
    inhibition_projection_after_sha256: str


class CompiledStopBoundedRestingGlifTiers:
    """Adapter that ends committed tie-tier dynamics at physical STOP."""

    def __init__(self, tier_selector: CompiledRestingGlifTierBeam) -> None:
        self._tier_selector = tier_selector
        self.candidate_count = tier_selector.candidate_count
        self.min_current_pa = tier_selector.min_current_pa
        self.max_current_pa = tier_selector.max_current_pa

    @classmethod
    def compile(
        cls,
        *,
        candidate_count: int,
        step_count: int,
        min_current_pa: Real,
        max_current_pa: Real,
        inhibition_current_pa: Real,
        parameters: AllenGlif5Parameters | None = None,
    ) -> CompiledStopBoundedRestingGlifTiers:
        """Compile using the committed complete-tier physical primitive."""

        return cls(
            CompiledRestingGlifTierBeam.compile(
                candidate_count=candidate_count,
                step_count=step_count,
                min_current_pa=min_current_pa,
                max_current_pa=max_current_pa,
                inhibition_current_pa=inhibition_current_pa,
                beam_width=1,
                parameters=parameters,
            )
        )

    @property
    def inhibition_projection_sha256(self) -> str:
        return self._tier_selector.inhibition_projection_sha256

    @property
    def live_inhibition_projection_sha256(self) -> str:
        return self._tier_selector.live_inhibition_projection_sha256

    @property
    def projection_storage_shape(self) -> tuple[int, ...]:
        return self._tier_selector._projection.weights_pa.shape

    def enumerate(
        self,
        currents_pa: Sequence[float] | np.ndarray,
        *,
        stop_cell: int,
    ) -> StopBoundedTierResult:
        """Enumerate complete tiers through the first physical STOP spike."""

        currents = np.asarray(currents_pa, dtype=np.float64)
        if currents.shape != (self.candidate_count,):
            raise ValueError("currents_pa must match the compiled candidate count")
        if not np.all(np.isfinite(currents)):
            raise ValueError("currents_pa must contain only finite values")
        if np.any(currents < self.min_current_pa) or np.any(
            currents > self.max_current_pa
        ):
            raise ValueError("currents_pa left the compiled current envelope")
        if type(stop_cell) is not int or not 0 <= stop_cell < self.candidate_count:
            raise ValueError("stop_cell must identify one compiled cell")

        working = np.ascontiguousarray(currents.copy())
        projection_before = self.live_inhibition_projection_sha256
        if projection_before != self.inhibition_projection_sha256:
            raise RuntimeError("tier self-inhibition changed before STOP enumeration")

        admissible: list[int] = []
        rounds: list[StopBoundedTierRound] = []
        inhibited = np.zeros(self.candidate_count, dtype=np.bool_)
        while True:
            before = tuple(float(current) for current in working)
            race, tier_cells = self._tier_selector._first_tier(working)
            tier_ids = np.asarray(tier_cells, dtype=np.intp)
            if np.any(inhibited[tier_ids]):
                raise RuntimeError("a physically inhibited tier cell fired twice")
            stop_frontier = stop_cell in tier_cells
            trigger = np.zeros(self.candidate_count, dtype=np.float64)
            trigger[tier_ids] = 1.0
            working += self._tier_selector._projection.project(trigger)
            inhibited[tier_ids] = True
            admissible.extend(cell for cell in tier_cells if cell != stop_cell)
            rounds.append(
                StopBoundedTierRound(
                    tier_index=len(rounds),
                    race=race,
                    tier_cells=tier_cells,
                    stop_frontier=stop_frontier,
                    currents_before_pa=before,
                    currents_after_pa=tuple(float(current) for current in working),
                )
            )
            if stop_frontier:
                break

        projection_after = self.live_inhibition_projection_sha256
        if projection_after != projection_before:
            raise RuntimeError("tier self-inhibition changed during STOP enumeration")
        admissible_seen = np.zeros(self.candidate_count, dtype=np.bool_)
        if admissible:
            admissible_seen[np.asarray(admissible, dtype=np.intp)] = True
        excluded = tuple(
            cell
            for cell in range(self.candidate_count)
            if cell != stop_cell and not admissible_seen[cell]
        )
        return StopBoundedTierResult(
            status=STOP_FRONTIER_REACHED,
            stop_cell=stop_cell,
            admissible_candidate_cells=tuple(admissible),
            excluded_candidate_cells=excluded,
            rounds=tuple(rounds),
            final_currents_pa=tuple(float(current) for current in working),
            inhibition_projection_before_sha256=projection_before,
            inhibition_projection_after_sha256=projection_after,
        )


__all__ = [
    "CompiledStopBoundedRestingGlifTiers",
    "STOP_FRONTIER_REACHED",
    "StopBoundedTierResult",
    "StopBoundedTierRound",
]
