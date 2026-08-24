"""APICAL-AND week-1 CPU harness (preregistration §6, docs/research/apical-and-preregistration.md).

A NumPy token loop that replicates the native serial observe kernel
(``humanbrain_observe_token`` / ``glif_step_collect`` in
``src/humanbrain/_native_src/src/lib.rs``) bit-for-bit when the apical drive is
zero — the flag-off parity gate — and adds the frozen two-compartment dynamics
(apical integrator + plateau-gated multiplicative soma threshold) on top.

Wiring is copied from a real ``RecurrentCellCircuit`` instance so the RNG
consumption order (afferents, recurrent targets, inhibitory signs,
heterogeneity) is bit-faithful to the r0-c stack at the same seed.

Per-token substep order (mirrors the native serial kernel):
  1. step_current = synaptic_current; first pulse_steps substeps add
     input_current_pa at the token's afferent targets;
  2. population.step(step_current)  (soma + apical + gate);
  3. synaptic_current *= recurrent_decay; eligibility *= decays (ALL cells);
  4. per spiking presynaptic cell (ascending): eligibility += 1.0, then
     synaptic_current[targets] += weights.
Steps 3-4 touch disjoint arrays from step 2's outputs, so the batched arm axis
is bitwise-equal to independent single-arm runs (tested).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.cells import TwoCompartmentGLIF5  # noqa: E402


class ApicalTokenRunner:
    """Batched (arm_count x cells) token replay over TwoCompartmentGLIF5."""

    def __init__(
        self,
        population: TwoCompartmentGLIF5,
        *,
        wiring: dict[str, np.ndarray],
        token_steps: int,
        pulse_steps: int,
        input_current_pa: float,
        recurrent_decay: float,
        elig_decays: np.ndarray,
    ) -> None:
        self.population = population
        self.afferent_targets = wiring["afferent_targets"]
        self.recurrent_targets = wiring["recurrent_targets"]
        self.recurrent_weights_pa = wiring["recurrent_weights_pa"]
        self.token_steps = int(token_steps)
        self.pulse_steps = int(pulse_steps)
        self.input_current_pa = float(input_current_pa)
        self.recurrent_decay = float(recurrent_decay)
        self._elig_decays = np.ascontiguousarray(elig_decays, dtype=np.float64)[None, :, None]
        arms, cells = population.arm_count, population.cell_count
        self.synaptic_current_pa = np.zeros((arms, cells), dtype=np.float64)
        self.eligibility = np.zeros(
            (arms, self._elig_decays.shape[1], cells), dtype=np.float64
        )
        self.token_spike_counts = np.zeros((arms, cells), dtype=np.uint32)
        self.step_current = np.zeros((arms, cells), dtype=np.float64)
        self.last_total_spikes = np.zeros(arms, dtype=np.int64)

    @classmethod
    def from_circuit(
        cls,
        circuit,
        *,
        tau_a_ms=100.0,
        theta_p=0.5,
        g_gate=0.6,
        gate_duration_ms=100.0,
        arm_count=1,
    ) -> "ApicalTokenRunner":
        """Copy wiring/constants from a native RecurrentCellCircuit."""
        population = TwoCompartmentGLIF5(
            circuit.config.reservoir_cells,
            circuit.cell_parameters,
            tau_a_ms=tau_a_ms,
            theta_p=theta_p,
            g_gate=g_gate,
            gate_duration_ms=gate_duration_ms,
            arm_count=arm_count,
        )
        wiring = {
            "afferent_targets": np.array(circuit._afferent_targets),
            "recurrent_targets": np.array(circuit._recurrent_targets),
            "recurrent_weights_pa": np.array(circuit._recurrent_weights_pa),
        }
        return cls(
            population,
            wiring=wiring,
            token_steps=circuit._token_steps,
            pulse_steps=circuit._pulse_steps,
            input_current_pa=circuit.config.input_current_pa,
            recurrent_decay=circuit._recurrent_decay,
            elig_decays=circuit._eligibility_decays,
        )

    def reset(self) -> None:
        self.population.reset()
        self.synaptic_current_pa.fill(0.0)
        self.eligibility.fill(0.0)
        self.token_spike_counts.fill(0)
        self.last_total_spikes[:] = 0

    def observe(self, token_id: int, apical_u: np.ndarray | None = None) -> None:
        pop = self.population
        if apical_u is not None:
            pop.set_apical_drive(apical_u)
        arms = pop.arm_count
        self.token_spike_counts[:] = 0
        total = np.zeros(arms, dtype=np.int64)
        aff = self.afferent_targets[int(token_id)]
        recurrent_targets = self.recurrent_targets
        recurrent_weights = self.recurrent_weights_pa
        fanout = recurrent_targets.shape[1]
        for substep in range(self.token_steps):
            np.copyto(self.step_current, self.synaptic_current_pa)
            if substep < self.pulse_steps:
                self.step_current[:, aff] += self.input_current_pa
            fired = pop.step(self.step_current)
            # decay passes over ALL cells (native serial ordering)
            self.synaptic_current_pa *= self.recurrent_decay
            self.eligibility *= self._elig_decays
            spiking_rows, spiking_cells = np.nonzero(fired)
            if spiking_rows.size:
                self.eligibility[spiking_rows, :, spiking_cells] += 1.0
                weights = recurrent_weights[spiking_cells]
                targets = recurrent_targets[spiking_cells]
                np.add.at(
                    self.synaptic_current_pa,
                    (np.repeat(spiking_rows, fanout), targets.reshape(-1)),
                    weights.reshape(-1),
                )
                total += np.bincount(spiking_rows, minlength=arms)[:arms]
            self.token_spike_counts += fired
        self.last_total_spikes = total


class ContextSummarizer:
    """Frozen fallback context summary -> apical drive (preregistration §3).

    Causal cumulative byte histogram of the prompt-so-far, projected through a
    fixed random matrix (seed 20260824), z-scored across cells. Answer bytes
    never enter the histogram; after the decision point the drive freezes.
    """

    W_CTX_SEED = 20260824

    def __init__(self, cell_count: int, seed: int | None = None) -> None:
        rng = np.random.default_rng(self.W_CTX_SEED if seed is None else seed)
        self.projection = rng.standard_normal((cell_count, 256)) / 16.0
        self.counts = np.zeros(256, dtype=np.float64)
        self.total = 0
        self._frozen: np.ndarray | None = None

    def update_byte(self, byte_id: int) -> None:
        if self._frozen is not None:
            raise RuntimeError("context summary is frozen past the decision point")
        self.counts[int(byte_id)] += 1.0
        self.total += 1

    def drive_vector(self) -> np.ndarray:
        if self._frozen is not None:
            return self._frozen
        if self.total == 0:
            return np.zeros(self.projection.shape[0], dtype=np.float64)
        raw = self.projection @ (self.counts / self.total)
        std = float(raw.std())
        if std == 0.0:
            return np.zeros_like(raw)
        return (raw - raw.mean()) / std

    def freeze(self) -> np.ndarray:
        self._frozen = self.drive_vector()
        return self._frozen


def context_drive_for_tokens(token_ids, cell_count: int, freeze_at: int | None = None):
    """Drive matrix (tokens, cells) for a byte-token sequence.

    Row t is the summary AFTER observing token t-? -- precisely: the histogram
    includes tokens [0, t] inclusive when row t is consumed DURING token t's
    substeps (the summary leads the soma by design: the context phase of token
    t carries everything observed up to and including token t). If freeze_at is
    given, rows at or beyond freeze_at repeat the frozen decision-point drive.
    """
    summarizer = ContextSummarizer(cell_count)
    rows = np.empty((len(token_ids), cell_count), dtype=np.float64)
    for index, token in enumerate(token_ids):
        summarizer.update_byte(int(token))
        rows[index] = summarizer.drive_vector()
    if freeze_at is not None:
        frozen = summarizer.freeze() if freeze_at > 0 else np.zeros(cell_count)
        rows[freeze_at:] = frozen[None, :]
    return rows
