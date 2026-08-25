"""R3 segment-growth pilot mechanics (path-2 mainline), prereg-frozen.

Implements the frozen §2 mechanism spec of
``docs/research/r3-segment-growth-preregistration.md`` verbatim: persistent
winner-cell election plus distal segment growth during passage replay, driven
against winner sets ``W_{t-1}`` supplied by the caller per step.

Frozen literals (do not tune): ``k = 32`` candidate cells per token pool,
``theta_a = 13`` connected-active synapses for a segment to depolarize its
cell, ``theta_m = 10`` potential-active synapses for a segment to be
"matching" (eligible to learn / win election), ``connectedPermanence = 0.50``,
``initialPermanence = 0.21``, at most ``maxSegmentsPerCell = 255`` segments
per cell and ``s = 20`` contacts per segment, +/- ``0.10`` confirm increments,
and negative-evidence decay ``delta_neg = 0.04 x increment = 0.004`` (ON;
there is no decay-off arm).

Growth sampling uses a seeded rng keyed ``(construction_seed, step_index,
cell_id)`` so identical call sequences are bit-identical (prereg §1 tripwire).

Interpretation note (the only judgement call): the prereg confirm bullet says
"active connected synapses ... += 0.10"; taken hyper-literally nothing could
ever first reach ``connectedPermanence`` from ``initialPermanence = 0.21`` and
no bigram could ever become predicted (G-S0 (i) would be unsatisfiable). We
therefore reinforce every *active* (presynaptically winning) synapse by the
increment and decay inactive ones, which is the htm.core reading the prereg
adopts its constants from. Negative evidence is applied strictly to the
*then*-active-connected synapse indices captured at depolarization time, as
written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StepResult:
    """Outcome of one replay step for one arriving token pool."""

    pool_id: Any
    step_index: int
    burst: bool
    elected_winner: int | None
    fired_cells: Tuple[int, ...]
    punished_segments: Tuple[Tuple[int, int], ...]
    allocated_segment: bool
    removed_contacts: int


class _Segment:
    """Distal segment: contacts (presynaptic cell ids) + float permanences."""

    __slots__ = ("presyn", "perm")

    def __init__(self) -> None:
        self.presyn: List[int] = []
        self.perm: List[float] = []


class SegmentGrowthTracker:
    """Winner-cell election + distal segment growth over token-cell pools.

    The caller supplies, per replay step, the arriving token's pool id and the
    previous step's winner set ``W_{t-1}`` (elected winners of bursting pools,
    fired cells of predicted-path pools). All growth/punishment rules are
    local and deterministic; numpy is the only dependency.
    """

    CELLS_PER_POOL = 32  # k
    THETA_ACTIVE = 13  # theta_a: connected-active synapses to depolarize
    THETA_MATCH = 10  # theta_m: potential-active synapses to be "matching"
    CONNECTED_PERMANENCE = 0.50
    INITIAL_PERMANENCE = 0.21
    MAX_SEGMENTS_PER_CELL = 255
    MAX_SYNAPSES_PER_SEGMENT = 20  # s: contact budget cap
    PERMANENCE_INCREMENT = 0.10
    NEGATIVE_EVIDENCE_RATIO = 0.04
    DELTA_NEGATIVE = 0.004  # delta_neg = 0.04 x increment, ON by default

    def __init__(
        self,
        pools: Mapping[Any, Sequence[int]],
        *,
        growth_on: bool = True,
        construction_seed: int = 17,
        cells_per_pool: int = CELLS_PER_POOL,
        theta_active: int = THETA_ACTIVE,
        theta_match: int = THETA_MATCH,
        connected_permanence: float = CONNECTED_PERMANENCE,
        initial_permanence: float = INITIAL_PERMANENCE,
        max_segments_per_cell: int = MAX_SEGMENTS_PER_CELL,
        max_synapses_per_segment: int = MAX_SYNAPSES_PER_SEGMENT,
        permanence_increment: float = PERMANENCE_INCREMENT,
    ) -> None:
        if not pools:
            raise ValueError("pools mapping must not be empty")
        seen: set[int] = set()
        self._pool_cells: Dict[Any, Tuple[int, ...]] = {}
        for pool_id, cells in pools.items():
            arr = np.unique(np.asarray([int(c) for c in cells], dtype=np.int64))
            if arr.size != cells_per_pool:
                raise ValueError(
                    f"pool {pool_id!r} must own exactly {cells_per_pool} cells"
                )
            overlap = seen.intersection(arr.tolist())
            if overlap:
                raise ValueError(f"cell pools must be disjoint; overlap {sorted(overlap)}")
            seen.update(arr.tolist())
            self._pool_cells[pool_id] = tuple(int(c) for c in arr)

        self._growth_on = bool(growth_on)
        self._construction_seed = int(construction_seed)
        self._cells_per_pool = int(cells_per_pool)
        self._theta_active = int(theta_active)
        self._theta_match = int(theta_match)
        self._connected_permanence = float(connected_permanence)
        self._initial_permanence = float(initial_permanence)
        self._max_segments_per_cell = int(max_segments_per_cell)
        self._max_synapses_per_segment = int(max_synapses_per_segment)
        self._permanence_increment = float(permanence_increment)

        self._segments: Dict[int, List[_Segment]] = {}
        self._wins: Dict[int, int] = {}
        self._step_index = 0

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return self._step_index

    @property
    def growth_on(self) -> bool:
        return self._growth_on

    def total_segments(self) -> int:
        return sum(len(segs) for segs in self._segments.values())

    def segment_count(self, cell: int) -> int:
        return len(self._segments.get(int(cell), ()))

    def prior_wins(self, cell: int) -> int:
        return self._wins.get(int(cell), 0)

    def total_contacts(self) -> int:
        return sum(
            len(seg.presyn) for segs in self._segments.values() for seg in segs
        )

    def segment_contacts(self, cell: int, segment_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Copies of (presynaptic ids, permanences) for inspection/tests."""
        segs = self._segments.get(int(cell))
        if segs is None or not 0 <= segment_index < len(segs):
            raise IndexError(f"no segment {segment_index} on cell {cell}")
        seg = segs[segment_index]
        return (
            np.asarray(seg.presyn, dtype=np.int64).copy(),
            np.asarray(seg.perm, dtype=np.float64).copy(),
        )

    def all_permanences(self) -> np.ndarray:
        parts = [
            np.asarray(seg.perm, dtype=np.float64)
            for segs in self._segments.values()
            for seg in segs
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    @staticmethod
    def election_sort_key(best_count: int, wins: int, index: int) -> Tuple[int, int, int]:
        """Best-match / least-used ordering: highest best-count, then fewest
        prior election wins, then lowest cell index."""
        return (-int(best_count), int(wins), int(index))

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def step(self, pool_id: Any, winner_set: Sequence[int]) -> StepResult:
        """Advance one replay step for the arriving token's pool."""
        if pool_id not in self._pool_cells:
            raise ValueError(f"unknown pool {pool_id!r}")
        pool_cells = self._pool_cells[pool_id]
        pool_set = frozenset(pool_cells)
        w = np.unique(np.asarray([int(x) for x in winner_set], dtype=np.int64))

        # Global recognition pass: any segment with >= theta_a connected-active
        # synapses against W depolarizes its cell. Cells inside the arriving
        # pool are confirmed (predicted path); cells outside it contributed
        # depolarization toward a population without the actual next input and
        # take negative-evidence decay now.
        hits: List[Tuple[int, int, np.ndarray]] = []
        misses: List[Tuple[int, int, np.ndarray]] = []
        for cell in sorted(self._segments):
            for si, seg in enumerate(self._segments[cell]):
                presyn = np.asarray(seg.presyn, dtype=np.int64)
                if presyn.size == 0:
                    continue
                perm = np.asarray(seg.perm, dtype=np.float64)
                active = np.isin(presyn, w)
                then_connected_active = active & (perm >= self._connected_permanence)
                if int(then_connected_active.sum()) >= self._theta_active:
                    idx = np.flatnonzero(then_connected_active)
                    (hits if cell in pool_set else misses).append((cell, si, idx))

        removed_total = 0
        punished: List[Tuple[int, int]] = []
        for cell, si, idx in misses:  # unconfirmed converged predictors
            seg = self._segments[cell][si]
            perm = np.asarray(seg.perm, dtype=np.float64)
            perm[idx] -= self.DELTA_NEGATIVE
            presyn_arr = np.asarray(seg.presyn, dtype=np.int64)
            seg.presyn, seg.perm, removed = self._purge(presyn_arr, perm)
            removed_total += removed
            punished.append((cell, si))
        for cell, si, _idx in hits:  # predicted-path confirm
            seg = self._segments[cell][si]
            presyn_arr = np.asarray(seg.presyn, dtype=np.int64)
            perm = np.asarray(seg.perm, dtype=np.float64)
            active = np.isin(presyn_arr, w)
            perm[active] += self._permanence_increment
            perm[~active] -= self._permanence_increment
            np.clip(perm, 0.0, 1.0, out=perm)
            seg.presyn, seg.perm, removed = self._purge(presyn_arr, perm)
            removed_total += removed

        burst = not hits
        elected: int | None = None
        allocated = False
        if burst:
            elected, allocated = self._elect(pool_cells, w)
        self._step_index += 1

        return StepResult(
            pool_id=pool_id,
            step_index=self._step_index - 1,
            burst=burst,
            elected_winner=elected,
            fired_cells=tuple(cell for cell, _, _ in hits),
            punished_segments=tuple(punished),
            allocated_segment=allocated,
            removed_contacts=removed_total,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _purge(
        presyn: np.ndarray, perm: np.ndarray
    ) -> Tuple[List[int], List[float], int]:
        """Drop contacts whose permanence reached p <= 0; they free budget."""
        keep = perm > 0.0
        removed = int((~keep).sum())
        return (
            presyn[keep].tolist(),
            perm[keep].tolist(),
            removed,
        )

    def _potential_active(self, seg: _Segment, w: np.ndarray) -> int:
        """Potential-active count: existing contacts whose presynaptic cell is
        in W (contacts at p <= 0 do not exist)."""
        if not seg.presyn or w.size == 0:
            return 0
        return int(np.isin(np.asarray(seg.presyn, dtype=np.int64), w).sum())

    def _elect(self, pool_cells: Tuple[int, ...], w: np.ndarray) -> Tuple[int, bool]:
        """Winner election on a burst (prereg §2). With growth ON the elected
        cell also learns (reinforce best segment, or allocate exactly ONE new
        segment); with growth OFF machinery runs for diagnostics only.
        Returns (winner, allocated_segment)."""
        candidates: List[Tuple[int, int, int]] = []  # (best_count, wins, cell)
        best_segment: Dict[int, int] = {}
        for cell in sorted(pool_cells):
            segs = self._segments.get(cell, [])
            best_count = 0
            best_si = -1
            for si, seg in enumerate(segs):
                count = self._potential_active(seg, w)
                if count > best_count:
                    best_count, best_si = count, si
            if best_count >= self._theta_match:
                candidates.append((best_count, self.prior_wins(cell), cell))
                best_segment[cell] = best_si

        if candidates:
            winner = min(
                candidates, key=lambda c: self.election_sort_key(c[0], c[1], c[2])
            )[2]
            self._wins[winner] = self.prior_wins(winner) + 1
            if self._growth_on:
                # Repeated (context, token) conjunction reinforces the existing
                # best-matching segment; never allocates a duplicate here.
                self._learn_on_segment(winner, best_segment[winner], w)
            return winner, False

        # No candidates: least-used cell (fewest owned segments, then fewest
        # prior wins, then lowest index) gets ONE brand-new segment grown
        # toward W_{t-1}.
        winner = min(
            sorted(pool_cells),
            key=lambda c: (
                self.segment_count(c),
                self.prior_wins(c),
                c,
            ),
        )
        self._wins[winner] = self.prior_wins(winner) + 1
        if self._growth_on:
            self._allocate_segment(winner, w)
            return winner, True
        return winner, False

    def _allocate_segment(self, cell: int, w: np.ndarray) -> None:
        segs = self._segments.setdefault(cell, [])
        if len(segs) >= self._max_segments_per_cell:
            raise RuntimeError(
                f"STOP: maxSegmentsPerCell={self._max_segments_per_cell} exceeded "
                f"on cell {cell}; prereg §1 forbids silently continuing"
            )
        seg = _Segment()
        segs.append(seg)
        self._fill_toward(cell, seg, w)

    def _learn_on_segment(self, cell: int, segment_index: int, w: np.ndarray) -> None:
        """Reinforce the chosen segment toward W (+inc active, -inc inactive,
        clipped to [0, 1]) and grow into freed/remaining contact budget."""
        seg = self._segments[cell][segment_index]
        presyn = np.asarray(seg.presyn, dtype=np.int64)
        perm = np.asarray(seg.perm, dtype=np.float64)
        if presyn.size:
            active = np.isin(presyn, w)
            perm[active] += self._permanence_increment
            perm[~active] -= self._permanence_increment
            np.clip(perm, 0.0, 1.0, out=perm)
            seg.presyn, seg.perm, _removed = self._purge(presyn, perm)
        self._fill_toward(cell, seg, w)

    def _fill_toward(self, cell: int, seg: _Segment, w: np.ndarray) -> None:
        """Grow new contacts sampled uniformly WITHOUT replacement from W
        (seeded rng keyed (construction_seed, step_index, cell_id)), up to the
        s-contact cap. Later events only adjust permanences unless removals
        freed budget."""
        budget = self._max_synapses_per_segment - len(seg.presyn)
        if budget <= 0 or w.size == 0:
            return
        existing = set(seg.presyn)
        available = np.asarray([int(x) for x in w if int(x) not in existing], dtype=np.int64)
        if available.size == 0:
            return
        n_new = min(budget, int(available.size))
        rng = np.random.default_rng(
            [self._construction_seed, self._step_index, int(cell)]
        )
        chosen = np.sort(rng.choice(available, size=n_new, replace=False))
        seg.presyn.extend(int(c) for c in chosen)
        seg.perm.extend([self._initial_permanence] * n_new)
