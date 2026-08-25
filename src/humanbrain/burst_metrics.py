"""DD-HTM burst-fraction tracking over observation-sink events.

Adopted from the HTM/TM deep dive (``docs/research/deep-dive-htm-tm.md``,
adopt row: "Burst fraction as a free anomaly/surprise metric in replay
logs"). In a temporal memory a *column* bursts when it fires even though no
cell in it was depolarized-predictive before the input arrived; the fraction
of bursting columns per step is a zero-cost surprise/anomaly diagnostic on
any replay.

Mapping onto this circuit (one adaptive GLIF compartment per reservoir
cell, fixed sparse recurrent topology):

- **Predictive state** = strictly positive recurrent context drive at
  arrival. Inside the kernel the token's afferent pulse is scratch-restored,
  so after every :meth:`observe` the synaptic-current bank holds only
  context current: depolarization delivered by recent spiking through
  recurrent edges, decayed by ``recurrent_decay ** token_steps`` between
  tokens. Decay is elementwise multiplication by a positive scalar, so the
  SIGN pattern sampled post-token equals the sign pattern at the next
  arrival exactly. A cell is predicted iff that pending sign is positive;
  negative (inhibitory) drive is not prediction.
- **Active set** = cells that spiked at least once during the token, i.e.
  the kernel-maintained u32 accumulator (cleared at token start, folded at
  every spike site) is nonzero. On the CPU backend this buffer and the
  current bank are live host state at sink time, so classification is exact.
  On CUDA arms the host mirrors must include both banks for exactness
  (mirror flags are family-driven); treat GPU-arm burst numbers as
  advisory unless those mirrors are enabled.
- **Burst** = active AND NOT predicted. The first token of a sequence (and
  every token after ``reset_sequence``, detected via the rewound
  observed-token counter) has an empty predictive set: genuinely novel
  input bursts, which is the DD-HTM semantics.

The tracker is a pure sink consumer: it reads two read-only views and
writes nothing, so wiring it cannot perturb dynamics. Everything below is
incremental O(active cells) per event with one boolean mask allocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from humanbrain.recurrent import RecurrentCellCircuit, RecurrentStep

BURST_METRICS_FORMAT = "humanbrain-burst-metrics-v1"


class BurstFractionTracker:
    """Incremental burst-fraction tracker consuming observation-sink events.

    Attach as an observer alongside any recorder::

        tracker = BurstFractionTracker(window_tokens=256)
        circuit.observation_sink = compose_sinks(recorder.record, tracker.record)

    or hand to :class:`~humanbrain.observatory.ActivityRecorder` via its
    ``burst_tracker`` argument, which also embeds ``tracker.snapshot()``
    into the stream footer. The callable matches the observation-sink
    contract ``sink(circuit, step)``; ``step`` carries only population
    totals, so all per-cell evidence comes from the live circuit views.
    """

    def __init__(self, *, window_tokens: int = 256) -> None:
        if window_tokens < 1:
            raise ValueError("window_tokens must be a positive integer")
        self.window_tokens = int(window_tokens)
        self._steps_observed = 0
        self._active_total = 0
        self._burst_total = 0
        self._per_cell_bursts: dict[int, int] = {}
        self._window_steps = 0
        self._window_active = 0
        self._window_bursts = 0
        self._completed_windows: list[float] = []
        # Positive-context sign pattern sampled after the most recent event:
        # the predictive set for the NEXT arrival. None until the first event
        # of a sequence (nothing can be predicted before any activity).
        self._pending_predicted: np.ndarray | None = None
        self._last_observed_tokens = -1
        self._backend: str | None = None

    def record(self, circuit: RecurrentCellCircuit, step: RecurrentStep | None) -> None:
        """Consume one observation-sink event (post-observe state)."""
        observed = int(circuit.observed_tokens)
        reset = (
            self._pending_predicted is None or observed <= self._last_observed_tokens
        )
        active = np.asarray(circuit.token_spike_counts) != 0
        if self._backend is None:
            # Exactness contract is CPU-live host state; on CUDA arms the
            # per-token mirrors may not include these banks (see module
            # docstring), so cohorts can tell which backend produced them.
            self._backend = str(getattr(circuit, "simulation_backend", "unknown"))
        if reset or self._pending_predicted is None:
            # Sequence start (or just after a reset): nothing was predicted.
            bursting = active.copy()
        else:
            bursting = active & ~self._pending_predicted

        active_count = int(np.count_nonzero(active))
        burst_count = int(np.count_nonzero(bursting))
        self._steps_observed += 1
        self._active_total += active_count
        self._burst_total += burst_count
        for gid in np.flatnonzero(bursting):
            cell = int(gid)
            self._per_cell_bursts[cell] = self._per_cell_bursts.get(cell, 0) + 1

        self._window_steps += 1
        self._window_active += active_count
        self._window_bursts += burst_count
        if self._window_steps >= self.window_tokens:
            self._completed_windows.append(
                self._window_bursts / self._window_active
                if self._window_active
                else 0.0
            )
            self._window_steps = 0
            self._window_active = 0
            self._window_bursts = 0

        # Sample the predictive state for the next arrival BEFORE anything
        # else can run: strictly positive context drive only (negative
        # entries are inhibition, never prediction).
        self._pending_predicted = np.asarray(circuit.context_current_pa) > 0.0
        self._last_observed_tokens = observed

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe cumulative metrics (embedded in replay-log footers)."""
        overall = (
            self._burst_total / self._active_total if self._active_total else None
        )
        mean_windows = (
            sum(self._completed_windows) / len(self._completed_windows)
            if self._completed_windows
            else None
        )
        return {
            "format": BURST_METRICS_FORMAT,
            "window_tokens": self.window_tokens,
            "simulation_backend": self._backend,
            "steps_observed": self._steps_observed,
            "active_cells_total": self._active_total,
            "burst_cells_total": self._burst_total,
            "burst_fraction": overall,
            "completed_window_count": len(self._completed_windows),
            "completed_window_burst_fractions": list(self._completed_windows),
            "mean_completed_window_burst_fraction": mean_windows,
            "open_window": {
                "tokens": self._window_steps,
                "active_cells": self._window_active,
                "burst_cells": self._window_bursts,
                "burst_fraction": (
                    self._window_bursts / self._window_active
                    if self._window_active
                    else None
                ),
            },
            # Sorted [gid, burst_count] pairs; sparse (only bursting cells).
            "per_cell_burst_counts": [
                [cell, count]
                for cell, count in sorted(self._per_cell_bursts.items())
            ],
        }

    @property
    def steps_observed(self) -> int:
        return self._steps_observed

    @property
    def burst_fraction(self) -> float | None:
        """Cumulative fraction of active cells that burst (None when idle)."""
        if not self._active_total:
            return None
        return self._burst_total / self._active_total


def compose_sinks(*sinks) -> Any:
    """Fan one observation-sink event out to several consumers, in order."""
    consumers = tuple(sink for sink in sinks if sink is not None)

    def _composed(circuit: RecurrentCellCircuit, step: RecurrentStep) -> None:
        for consumer in consumers:
            consumer(circuit, step)

    return _composed if consumers else None
