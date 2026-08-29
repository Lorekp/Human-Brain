"""Physical B-best selection through repeated resting GLIF5 races.

Every round races the complete candidate population.  A materialized fixed
winner-triggered projection subtracts current only from the physical winner;
candidates are never removed or masked from later races.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real

import numpy as np

from humanbrain.cells import AllenGlif5Parameters
from humanbrain.resting_glif_race import (
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
    RestingGlifRaceResult,
)


def _finite_current(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real current")
    current = float(value)
    if not np.isfinite(current):
        raise ValueError(f"{name} must be finite")
    return current


def _projection_sha256(weights_pa: np.ndarray, inhibition_current_pa: float) -> str:
    digest = sha256()
    digest.update(b"resting-glif-winner-self-inhibition-v1")
    digest.update(np.asarray(weights_pa.shape, dtype="<i8").tobytes())
    digest.update(np.asarray((inhibition_current_pa,), dtype="<f8").tobytes())
    digest.update(np.asarray(weights_pa, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FixedWinnerSelfInhibition:
    """Materialized winner-spike to same-cell inhibitory current projection."""

    candidate_count: int
    inhibition_current_pa: float
    weights_pa: np.ndarray
    sha256: str

    @classmethod
    def materialize(
        cls,
        *,
        candidate_count: int,
        inhibition_current_pa: Real,
    ) -> FixedWinnerSelfInhibition:
        if type(candidate_count) is not int or candidate_count < 1:
            raise ValueError("candidate_count must be a positive integer")
        inhibition = _finite_current(
            inhibition_current_pa,
            "inhibition_current_pa",
        )
        if inhibition <= 0.0:
            raise ValueError("inhibition_current_pa must be positive")
        weights = np.full(candidate_count, -inhibition, dtype=np.float64)
        frozen = np.ascontiguousarray(weights)
        digest = _projection_sha256(frozen, inhibition)
        frozen.setflags(write=False)
        return cls(
            candidate_count=candidate_count,
            inhibition_current_pa=inhibition,
            weights_pa=frozen,
            sha256=digest,
        )

    @property
    def live_sha256(self) -> str:
        return _projection_sha256(self.weights_pa, self.inhibition_current_pa)

    def project(self, winner_spikes: Sequence[float] | np.ndarray) -> np.ndarray:
        spikes = np.asarray(winner_spikes, dtype=np.float64)
        if spikes.shape != (self.candidate_count,):
            raise ValueError("winner trigger must have one spike slot per candidate")
        if not np.all(np.isfinite(spikes)):
            raise ValueError("winner trigger must contain only finite values")
        if np.count_nonzero(spikes) != 1 or not np.all(
            np.logical_or(spikes == 0.0, spikes == 1.0)
        ):
            raise ValueError("winner trigger must be exactly one-hot")
        return np.ascontiguousarray(self.weights_pa * spikes)


@dataclass(frozen=True, slots=True)
class RestingGlifBeamRound:
    """One complete-population physical race and its winner inhibition."""

    rank: int
    race: RestingGlifRaceResult
    currents_before_pa: tuple[float, ...]
    currents_after_pa: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RestingGlifBeamResult:
    """B winners in physical race order plus the complete round ledger."""

    winner_cells: tuple[int, ...]
    rounds: tuple[RestingGlifBeamRound, ...]
    final_currents_pa: tuple[float, ...]
    inhibition_projection_before_sha256: str
    inhibition_projection_after_sha256: str


class CompiledRestingGlifBeam:
    """Reusable bounded-current selector with a fixed physical beam width."""

    def __init__(
        self,
        *,
        race: CompiledRestingGlifRace,
        projection: FixedWinnerSelfInhibition,
        beam_width: int,
        min_current_pa: float,
        max_current_pa: float,
    ) -> None:
        self._race = race
        self._projection = projection
        self.beam_width = beam_width
        self.candidate_count = projection.candidate_count
        self.min_current_pa = min_current_pa
        self.max_current_pa = max_current_pa

    @classmethod
    def compile(
        cls,
        *,
        candidate_count: int,
        step_count: int,
        min_current_pa: Real,
        max_current_pa: Real,
        inhibition_current_pa: Real,
        beam_width: int = 4,
        parameters: AllenGlif5Parameters | None = None,
    ) -> CompiledRestingGlifBeam:
        """Compile one selector and validate its declared current envelope."""

        if type(candidate_count) is not int or candidate_count < 1:
            raise ValueError("candidate_count must be a positive integer")
        if type(beam_width) is not int or not 1 <= beam_width <= candidate_count:
            raise ValueError("beam_width must be in [1, candidate_count]")
        lower = _finite_current(min_current_pa, "min_current_pa")
        upper = _finite_current(max_current_pa, "max_current_pa")
        if lower > upper:
            raise ValueError("min_current_pa must not exceed max_current_pa")
        inhibition = _finite_current(
            inhibition_current_pa,
            "inhibition_current_pa",
        )
        if inhibition <= upper - lower:
            raise ValueError(
                "inhibition_current_pa must exceed the declared current span"
            )
        projection = FixedWinnerSelfInhibition.materialize(
            candidate_count=candidate_count,
            inhibition_current_pa=inhibition,
        )
        return cls(
            race=CompiledRestingGlifRace.compile(
                step_count=step_count,
                parameters=parameters,
            ),
            projection=projection,
            beam_width=beam_width,
            min_current_pa=lower,
            max_current_pa=upper,
        )

    @property
    def inhibition_projection_sha256(self) -> str:
        return self._projection.sha256

    @property
    def live_inhibition_projection_sha256(self) -> str:
        return self._projection.live_sha256

    def select(
        self,
        currents_pa: Sequence[float] | np.ndarray,
    ) -> RestingGlifBeamResult:
        """Select B cells without sorting, masking, or deleting candidates."""

        currents = np.asarray(currents_pa, dtype=np.float64)
        if currents.shape != (self.candidate_count,):
            raise ValueError("currents_pa must match the compiled candidate count")
        if not np.all(np.isfinite(currents)):
            raise ValueError("currents_pa must contain only finite values")
        if np.any(currents < self.min_current_pa) or np.any(
            currents > self.max_current_pa
        ):
            raise ValueError("currents_pa left the compiled current envelope")
        working = np.ascontiguousarray(currents.copy())
        projection_before = self.live_inhibition_projection_sha256
        if projection_before != self.inhibition_projection_sha256:
            raise RuntimeError("winner self-inhibition projection changed before selection")

        rounds: list[RestingGlifBeamRound] = []
        winner_seen = np.zeros(self.candidate_count, dtype=np.bool_)
        for rank in range(self.beam_width):
            before = tuple(float(current) for current in working)
            race = self._race.race_many(working[None, :], require_unique=True)[0]
            if winner_seen[race.winner_cell]:
                raise RuntimeError("a physically inhibited candidate won twice")
            winner_seen[race.winner_cell] = True
            trigger = np.zeros(self.candidate_count, dtype=np.float64)
            trigger[race.winner_cell] = 1.0
            working += self._projection.project(trigger)
            rounds.append(
                RestingGlifBeamRound(
                    rank=rank,
                    race=race,
                    currents_before_pa=before,
                    currents_after_pa=tuple(float(current) for current in working),
                )
            )

        projection_after = self.live_inhibition_projection_sha256
        if projection_after != projection_before:
            raise RuntimeError("winner self-inhibition projection changed during selection")
        return RestingGlifBeamResult(
            winner_cells=tuple(round_.race.winner_cell for round_ in rounds),
            rounds=tuple(rounds),
            final_currents_pa=tuple(float(current) for current in working),
            inhibition_projection_before_sha256=projection_before,
            inhibition_projection_after_sha256=projection_after,
        )


__all__ = [
    "CompiledRestingGlifBeam",
    "FixedWinnerSelfInhibition",
    "RestingGlifBeamResult",
    "RestingGlifBeamRound",
    "RestingGlifRaceFailure",
]
