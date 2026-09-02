"""Vectorized physical first-spike races for known sequential race rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .cortex import CellRace, DendriticCortex


class DendriticFirstSpikeBatch:
    """Compile immutable cortex dynamics and race independent rows together.

    The caller still supplies every candidate cell and current in temporal
    order.  This class only amortizes the numerical first-crossing work and
    returns one ordinary :class:`CellRace` per row, in that same order.  A
    caller can therefore retain its existing physical trace and dependency
    construction without introducing a host-side action selector.

    Rows must be knowable before the batch starts.  A caller must not batch a
    causal branch whose later candidate set or current depends on an earlier
    row's winner.
    """

    def __init__(self, cortex: DendriticCortex) -> None:
        if not isinstance(cortex, DendriticCortex):
            raise TypeError("cortex must be a DendriticCortex")

        cell = cortex.parameters.cell
        self._cell_count = cortex.cell_count
        self._rest_mv = float(cell.rest_mv)
        self._threshold_relative_mv = float(cell.threshold_relative_mv)
        self._dt_ms = float(cell.dt_ms)
        self._threshold_offset = np.array(
            cortex._threshold_offset,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        self._rising_response = np.array(
            cortex._unit_nmda_response[: cortex._unit_nmda_peak_step + 1],
            dtype=np.float64,
            copy=True,
            order="C",
        )
        self._threshold_offset.setflags(write=False)
        self._rising_response.setflags(write=False)

    def race_many(
        self,
        cell_ids: Sequence[Sequence[int]] | np.ndarray,
        current_pa: Sequence[Sequence[float]] | np.ndarray,
    ) -> tuple[CellRace, ...]:
        """Race a dense rows-by-candidates matrix from fitted GLIF3 rest."""

        raw_ids = np.asarray(cell_ids)
        if raw_ids.ndim != 2 or raw_ids.shape[0] < 1 or raw_ids.shape[1] < 1:
            raise ValueError("cell_ids must be a nonempty rows-by-candidates matrix")
        if raw_ids.dtype.kind not in {"i", "u"}:
            raise ValueError("cell_ids must contain integers")
        if np.any(raw_ids < 0) or np.any(raw_ids >= self._cell_count):
            raise ValueError("cell id outside cortical sheet")
        active = raw_ids.astype(np.intp, copy=False)
        ordered_ids = np.sort(active, axis=1)
        if np.any(ordered_ids[:, 1:] == ordered_ids[:, :-1]):
            raise ValueError("cell_ids must be unique within each row")

        currents = np.asarray(current_pa, dtype=np.float64)
        if currents.shape != active.shape:
            raise ValueError(f"current_pa must have shape {active.shape}")
        if not np.all(np.isfinite(currents)):
            raise ValueError("current_pa must contain only finite values")
        if np.any(currents < 0.0):
            raise ValueError("current_pa must be nonnegative")
        currents = np.ascontiguousarray(currents)

        thresholds = self._threshold_relative_mv + self._threshold_offset[active]
        response_needed = np.full(active.shape, np.inf, dtype=np.float64)
        np.divide(
            thresholds,
            currents,
            out=response_needed,
            where=currents > 0.0,
        )
        response_size = int(self._rising_response.size)
        first_steps = np.searchsorted(
            self._rising_response,
            response_needed.reshape(-1),
            side="left",
        ).reshape(active.shape)
        valid = first_steps < response_size
        if not np.all(np.any(valid, axis=1)):
            raise RuntimeError("active human cells did not spike under arousal")

        winning_steps = np.min(
            np.where(valid, first_steps, response_size),
            axis=1,
        )
        winning_response = self._rising_response[winning_steps]
        first_bin = first_steps == winning_steps[:, None]
        margins = winning_response[:, None] * currents - thresholds
        winner_indices = np.argmax(np.where(first_bin, margins, -np.inf), axis=1)
        row_indices = np.arange(active.shape[0], dtype=np.intp)
        winners = active[row_indices, winner_indices]
        peak_currents = np.max(currents, axis=1)

        return tuple(
            CellRace(
                winner=int(winners[row]),
                spike_time_ms=(int(winning_steps[row]) + 1) * self._dt_ms,
                peak_voltage_mv=max(
                    self._rest_mv,
                    self._rest_mv + float(winning_response[row] * peak_currents[row]),
                ),
                active_cells=active.shape[1],
                simulated_steps=int(winning_steps[row]) + 1,
            )
            for row in range(active.shape[0])
        )
