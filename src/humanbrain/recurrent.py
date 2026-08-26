"""Sparse recurrent token circuit made only from cells and synapses."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import re
from pathlib import Path

import numpy as np

from humanbrain.cells import (
    AdaptiveGlifPopulation,
    AllenGlif5Parameters,
)
from humanbrain.human_cell_identity import HUMAN_GLIF5_IDENTITY
from humanbrain.native import NativeThreeFactorLearner, recurrent_kernel
from humanbrain.readout import SparseCellReadout
from humanbrain.recurrent_state import RecurrentCellSynapseState
from humanbrain.topology import RECURRENT_TOPOLOGIES, build_recurrent_targets

READOUT_LEARNING_RULES = frozenset({"ridge", "three_factor", "sparse_three_factor"})


def _normalized_token_ids(token_ids: Sequence[int]) -> list[int]:
    """Convert a token sequence to exact python ints without per-item churn."""
    if isinstance(token_ids, np.ndarray):
        if token_ids.ndim == 1 and token_ids.dtype.kind in "iu":
            return token_ids.tolist()
        return [int(token_id) for token_id in token_ids]
    if isinstance(token_ids, list) and all(
        type(token_id) is int for token_id in token_ids
    ):
        return token_ids
    return [int(token_id) for token_id in token_ids]

# R0 multi-timescale eligibility-trace bank (frozen protocol §1.1):
# geometric span 80 ms -> 4 s at token_interval_ms = 5 gives readout memory
# windows of 16 -> 800 tokens. Trace 0 IS the incumbent single trace.
ELIGIBILITY_TAUS_MS = tuple(np.geomspace(80.0, 4000.0, 8))

# R0-b state-feature families (frozen protocol r0b-state-feature-protocol.md
# §1.1/§1.2). The default family "traces" reproduces incumbent behavior
# bit-for-bit; the successor families replace ONLY what feeds the readout.
FEATURE_FAMILIES = (
    "traces",
    "spike_counts",
    "voltage_quantiles",
    "assembly_overlap",
)
# Canonical family order for every ordering decision: A < B < C (§2).
_SINGLE_FAMILY_ORDER = ("spike_counts", "voltage_quantiles", "assembly_overlap")
# Hard-edged sliding windows in TOKENS: same geomspace coverage as the tau
# bank (16..800 tokens at token_interval_ms = 5), bitwise-frozen.
WINDOW_TOKENS = tuple(int(round(w)) for w in np.geomspace(16.0, 800.0, 8))
_MAX_WINDOW_TOKENS = 100_000
_COMBO_FAMILY_PATTERN = re.compile(
    r"^combo:(?P<first>spike_counts|voltage_quantiles|assembly_overlap)"
    r"\+(?P<second>spike_counts|voltage_quantiles|assembly_overlap)$"
)


def parse_feature_family(family: str) -> tuple[str, ...]:
    """Return the member families of ``family`` in canonical order.

    Single names yield a one-element tuple; ``"combo:<f1>+<f2>"`` requires two
    distinct single families in canonical order (§1.2 grammar). Anything else
    raises ValueError.
    """
    if family in FEATURE_FAMILIES:
        return ("traces",) if family == "traces" else (family,)
    match = _COMBO_FAMILY_PATTERN.match(family)
    if match is None:
        raise ValueError(
            f"unknown feature family {family!r}; expected one of "
            f"{FEATURE_FAMILIES} or 'combo:<f1>+<f2>' over "
            f"{_SINGLE_FAMILY_ORDER}"
        )
    members = (match.group("first"), match.group("second"))
    if members[0] == members[1]:
        raise ValueError(
            f"combo feature family {family!r} must name two distinct families"
        )
    if _SINGLE_FAMILY_ORDER.index(members[0]) > _SINGLE_FAMILY_ORDER.index(
        members[1]
    ):
        raise ValueError(
            f"combo feature family {family!r} must list its members in "
            f"canonical order {_SINGLE_FAMILY_ORDER}"
        )
    return members


_FAMILY_MEMBER_ROWS = {
    "spike_counts": lambda n_win: n_win,
    "voltage_quantiles": lambda n_win: 3 * n_win,
    "assembly_overlap": lambda n_win: n_win,
}


def resolved_feature_cells(config: RecurrentCircuitConfig) -> int:
    """F for ``config``: feature sources feeding the readout, bias excluded."""
    members = parse_feature_family(config.feature_family)
    if members == ("traces",):
        return len(config.eligibility_taus_ms) * config.reservoir_cells
    n_win = len(config._resolved_window_tokens())
    return (
        sum(_FAMILY_MEMBER_ROWS[member](n_win) for member in members)
        * config.reservoir_cells
    )


# R0-b incremental rollout: single families supported by the circuit so far.
# Grows with each implementation commit; "traces" is always present.
_IMPLEMENTED_FEATURE_FAMILIES = frozenset(
    {"traces", "spike_counts", "voltage_quantiles", "assembly_overlap"}
)

_BASE_STATE_ARRAYS = (
    ("_afferent_targets", "afferent_targets", np.int32),
    ("_recurrent_targets", "recurrent_targets", np.int32),
    ("_recurrent_weights_pa", "recurrent_weights_pa", np.float64),
    ("_output_heterogeneity_pa", "output_heterogeneity_pa", np.float64),
    ("_pacemaker_synapses_pa", "pacemaker_synapses_pa", np.float64),
)
_DEVELOPMENTAL_STATE_ARRAYS = (
    ("_development_cell_order", "development_cell_order", np.int32),
    ("_development_active_mask", "development_active_mask", np.bool_),
    ("_structural_scores", "structural_scores", np.float64),
    ("_structural_exposure", "structural_exposure", np.float64),
    ("_homeostatic_gain", "homeostatic_gain", np.float64),
)
_SPARSE_READOUT_STATE_ARRAYS = (
    "readout_potential_sources",
    "readout_apposition_eligibility",
    "readout_contact_sources",
    "readout_contact_weights_pa",
    "readout_contact_ages",
    "readout_output_activity_ema",
    "readout_target_presentations",
)
_CELL_SYNAPSE_METADATA_KEYS = (
    "token_cells",
    "allen_glif5_config_sha256",
    "cell_parameters",
    "circuit_config",
    "growth_examples",
    "local_updates",
    "runtime_active_reservoir_cells",
    "developmental_state",
)


@dataclass(frozen=True)
class _ReadoutStateLayout:
    direct_arrays: tuple[tuple[str, str, object], ...]
    payload_arrays: tuple[str, ...] = ()
    sparse: bool = False
    ridge_statistics: bool = False


_READOUT_STATE_LAYOUTS = {
    "ridge": _ReadoutStateLayout(
        direct_arrays=(
            ("_readout_synapses_pa", "readout_synapses_pa", np.float64),
            ("_normal_matrix", "normal_matrix", np.float64),
            ("_target_moments", "target_moments", np.float64),
        ),
        ridge_statistics=True,
    ),
    "three_factor": _ReadoutStateLayout(
        direct_arrays=(
            ("_readout_synapses_pa", "readout_synapses_pa", np.float64),
        ),
    ),
    "sparse_three_factor": _ReadoutStateLayout(
        direct_arrays=(),
        payload_arrays=_SPARSE_READOUT_STATE_ARRAYS,
        sparse=True,
    ),
}

# Ridge sufficient statistics are RUNTIME-ONLY under the v5 diet ("stats
# never migrate", r0-multitrace-protocol.md §1.5 disposition): artifact
# families that diet them away persist fitted weights + topology/config only,
# and a resumed circuit refits statistics from scratch (retraining is the
# sanctioned path). The recurrent-human-cell-checkpoint-v5 npz envelope still
# carries them by default (capture default), so only dieted callers skip them.
_RIDGE_STATISTIC_ARRAY_KEYS = frozenset({"normal_matrix", "target_moments"})
# §1.5 era-guard keys ride along ONLY in payloads that carry ridge statistics;
# a dieted capture reduces metadata to the canonical circuit schema, whose
# circuit_config already nests the tau bank at its canonical home.
_ENVELOPE_ONLY_METADATA_KEYS = frozenset(
    {
        "eligibility_taus_ms",
        "feature_cell_count",
        "feature_family",
        "feature_window_tokens",
    }
)


@dataclass(frozen=True)
class RecurrentCircuitConfig:
    reservoir_cells: int = 256
    active_reservoir_cells: int | None = None
    input_fanout: int = 24
    recurrent_fanout: int = 8
    token_interval_ms: float = 5.0
    input_pulse_ms: float = 1.5
    input_current_pa: float = 5_000.0
    recurrent_current_pa: float = 650.0
    recurrent_tau_ms: float = 8.0
    eligibility_taus_ms: tuple[float, ...] = ELIGIBILITY_TAUS_MS
    output_base_current_pa: float = 450.0
    output_race_ms: float = 30.0
    readout_ridge: float = 1.0
    output_synapse_gain_pa: float = 1_600.0
    readout_learning_rule: str = "ridge"
    readout_fan_in: int = 64
    readout_structural_interval: int = 16
    readout_structural_grace: int = 32
    readout_turnover_probability: float = 0.1
    readout_homeostatic_rate: float = 0.05
    readout_activity_decay: float = 0.99
    local_learning_rate_pa: float = 64.0
    local_competitive_margin_pa: float = 64.0
    inhibitory_fraction: float = 0.2
    random_seed: int = 17
    recurrent_topology: str = "random"
    # Deprecated single-tau keyword (v3-era metadata and legacy callers).
    # When given, it constructs the one-element tau bank ``(value,)``.
    eligibility_tau_ms: float | None = None
    # R0-b state-feature family feeding the readout (§1.2). Default
    # "traces" reproduces incumbent behavior bit-for-bit.
    feature_family: str = "traces"
    # Optional override of the frozen window bank; None uses WINDOW_TOKENS.
    feature_window_tokens: tuple[int, ...] | None = None

    def _resolved_window_tokens(self) -> tuple[int, ...]:
        """The effective hard-edged window bank in tokens."""
        windows = self.feature_window_tokens
        return tuple(int(w) for w in windows) if windows is not None else WINDOW_TOKENS

    def __post_init__(self) -> None:
        if isinstance(self.feature_window_tokens, list):
            object.__setattr__(
                self, "feature_window_tokens", tuple(self.feature_window_tokens)
            )
        parse_feature_family(self.feature_family)
        windows = self.feature_window_tokens
        if windows is not None:
            if (
                not isinstance(windows, tuple)
                or len(windows) < 1
                or any(
                    isinstance(w, bool) or not isinstance(w, int) for w in windows
                )
            ):
                raise ValueError(
                    "feature_window_tokens must be a nonempty tuple of integers"
                )
            if any(w < 1 for w in windows):
                raise ValueError("feature window lengths must be positive")
            if any(w > _MAX_WINDOW_TOKENS for w in windows):
                raise ValueError(
                    f"feature window lengths must not exceed {_MAX_WINDOW_TOKENS}"
                )
            if any(
                windows[index + 1] <= windows[index]
                for index in range(len(windows) - 1)
            ):
                raise ValueError("feature windows must be strictly increasing")
        if self.reservoir_cells < 8:
            raise ValueError("reservoir needs at least eight cells")
        active_cells = self.active_reservoir_cells or self.reservoir_cells
        if not 8 <= active_cells <= self.reservoir_cells:
            raise ValueError("active reservoir cells must be between eight and total cells")
        if not 1 <= self.input_fanout <= active_cells:
            raise ValueError("input fanout must fit the active reservoir")
        if not 1 <= self.recurrent_fanout <= active_cells:
            raise ValueError("recurrent fanout must fit the active reservoir")
        if isinstance(self.eligibility_taus_ms, list):
            object.__setattr__(
                self, "eligibility_taus_ms", tuple(self.eligibility_taus_ms)
            )
        if self.eligibility_tau_ms is not None:
            # Deprecated shim: one tau in, one trace out.
            object.__setattr__(
                self, "eligibility_taus_ms", (float(self.eligibility_tau_ms),)
            )
        taus = self.eligibility_taus_ms
        if not isinstance(taus, tuple) or len(taus) < 1:
            raise ValueError("eligibility taus must contain at least one trace")
        if not all(math.isfinite(float(tau)) for tau in taus):
            raise ValueError("eligibility taus must be finite")
        if any(float(tau) <= 0.0 for tau in taus):
            raise ValueError("eligibility taus must be positive")
        if any(
            float(taus[index + 1]) <= float(taus[index])
            for index in range(len(taus) - 1)
        ):
            raise ValueError("eligibility taus must be strictly increasing")
        positive = (
            self.token_interval_ms,
            self.input_pulse_ms,
            self.input_current_pa,
            self.recurrent_current_pa,
            self.recurrent_tau_ms,
            self.output_base_current_pa,
            self.output_race_ms,
            self.readout_ridge,
            self.output_synapse_gain_pa,
            self.local_learning_rate_pa,
            self.local_competitive_margin_pa,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("circuit time, current, and plasticity values must be positive")
        if self.input_pulse_ms > self.token_interval_ms:
            raise ValueError("input pulse cannot exceed the token interval")
        if not 0.0 <= self.inhibitory_fraction < 1.0:
            raise ValueError("inhibitory fraction must be in [0, 1)")
        if self.recurrent_topology not in RECURRENT_TOPOLOGIES:
            raise ValueError("unknown recurrent topology")
        if self.readout_learning_rule not in READOUT_LEARNING_RULES:
            raise ValueError("unknown readout learning rule")
        if self.readout_learning_rule == "sparse_three_factor" and not (
            1 <= self.readout_fan_in <= self.reservoir_cells
        ):
            raise ValueError("readout fan-in must fit the reservoir")
        if self.readout_learning_rule == "sparse_three_factor":
            if self.readout_structural_interval < 1 or self.readout_structural_grace < 1:
                raise ValueError("sparse structural clocks must be positive")
            if not 0.0 <= self.readout_turnover_probability <= 1.0:
                raise ValueError("readout turnover probability must be in [0, 1]")
            if self.readout_homeostatic_rate < 0.0:
                raise ValueError("readout homeostatic rate must be nonnegative")
            if not 0.0 < self.readout_activity_decay < 1.0:
                raise ValueError("readout activity decay must be in (0, 1)")
        if self.recurrent_topology != "random" and self.recurrent_fanout >= active_cells:
            raise ValueError("non-random topology must exclude self-connections")


@dataclass(frozen=True)
class RecurrentStep:
    token_id: int
    reservoir_spikes: int
    simulated_steps: int
    biological_time_ms: float


@dataclass(frozen=True)
class RecurrentDynamicsSample:
    """A bounded, read-only sample of live reservoir state."""

    gids: tuple[int, ...]
    voltage_relative_mv: tuple[float, ...]
    synaptic_current_pa: tuple[float, ...]
    eligibility: tuple[float, ...]


@dataclass(frozen=True)
class RecurrentRace:
    winner: int
    spike_time_ms: float
    peak_voltage_mv: float
    active_cells: int
    simulated_steps: int
    simultaneous_spike_candidates: int = 1
    discarded_simultaneous_candidates: int = 0
    excluded_token_ids: tuple[int, ...] = ()
    output_current_sha256: str = ""


class RecurrentCellCircuit:
    """A liquid-state circuit with human-fitted adaptive cells.

    Token IDs only choose fixed afferent synapses. Context persists in fitted
    membrane voltage, after-spike currents, adaptive thresholds, recurrent
    synaptic current, and an eligibility trace driven by presynaptic spikes.
    The default circuit learns synaptic strengths into output cells. The
    explicit developmental controller can additionally grow, score, prune, and
    homeostatically scale the existing sparse recurrent synapses during
    training. Output is selected by the first adaptive human-fitted cell to
    cross threshold.
    """

    _STATISTICS_BATCH_SIZE = 128

    def __init__(
        self,
        token_cells: int,
        config: RecurrentCircuitConfig | None = None,
        cell_parameters: AllenGlif5Parameters | None = None,
        observation_sink: (
            Callable[["RecurrentCellCircuit", RecurrentStep], None] | None
        ) = None,
    ) -> None:
        if token_cells < 2:
            raise ValueError("a circuit needs at least two token cells")
        self.token_cells = int(token_cells)
        self.config = config or RecurrentCircuitConfig()
        # R0-c Thread-1 logging seam: optional per-token observer callback.
        # None by default; when None, observe() is untouched bit-for-bit.
        self._observation_sink = observation_sink
        self.active_reservoir_cells = (
            self.config.active_reservoir_cells or self.config.reservoir_cells
        )
        self.cell_parameters = cell_parameters or AllenGlif5Parameters()
        self._reservoir = AdaptiveGlifPopulation(self.config.reservoir_cells, self.cell_parameters)
        self._synaptic_current_pa = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        # R0 trace bank: one contiguous f64 buffer of N_TRACE * reservoir_cells
        # elements, viewed C-order as (trace_count, reservoir_cells). Row k is
        # the eligibility trace with tau = taus[k]. The flat view IS the
        # readout feature space: source id s decodes as k, cell = divmod(s, R).
        self.trace_count = len(self.config.eligibility_taus_ms)
        # Frozen protocol §1.2: the bank IS a (K, reservoir_cells) C-order
        # view; row k is trace k. The flat view is the readout feature space.
        self._eligibility_traces = np.zeros(
            (self.trace_count, self.config.reservoir_cells), dtype=np.float64
        )
        self._feature_traces = self._eligibility_traces.reshape(-1)
        # R0-b state-feature family (frozen protocol §1.2): the trace bank
        # REMAINS MAINTAINED in every arm; the active family only decides what
        # feeds the readout feature space below.
        self.feature_members = parse_feature_family(self.config.feature_family)
        for member in self.feature_members:
            if member not in _IMPLEMENTED_FEATURE_FAMILIES:
                raise NotImplementedError(
                    f"feature family {member!r} has no implementation yet "
                    "(R0-b incremental rollout)"
                )
        self._feature_window_tokens = self.config._resolved_window_tokens()
        self._feature_ring_capacity = max(self._feature_window_tokens)
        windows = self._feature_window_tokens
        cells_count = self.config.reservoir_cells
        self._token_spike_counts = np.zeros(cells_count, dtype=np.uint32)
        if self.feature_members == ("traces",):
            feature_source_cells = self.trace_count * cells_count
        else:
            feature_source_cells = (
                sum(
                    _FAMILY_MEMBER_ROWS[member](len(windows))
                    for member in self.feature_members
                )
                * cells_count
            )
            # One persistent f64 row store; each member's feature rows are a
            # zero-copy view into it, so combos concatenate by addressing only
            # (§2.D) and every accessor returns a stable buffer.
            self._feature_rows = np.zeros(
                (feature_source_cells // cells_count, cells_count),
                dtype=np.float64,
            )
            row_cursor = 0
            for member in self.feature_members:
                span = _FAMILY_MEMBER_ROWS[member](len(windows))
                block = self._feature_rows[row_cursor : row_cursor + span]
                if member == "spike_counts":
                    self._window_counts = block
                elif member == "voltage_quantiles":
                    self._quantile_rows = block
                    self._quantile_stale = True
                else:
                    self._assembly_counts = block
                row_cursor += span
            if "spike_counts" in self.feature_members:
                self._count_ring = np.zeros(
                    (self._feature_ring_capacity, cells_count), dtype=np.uint32
                )
            if "voltage_quantiles" in self.feature_members:
                self._voltage_ring = np.zeros(
                    (self._feature_ring_capacity, cells_count), dtype=np.float64
                )
            if "assembly_overlap" in self.feature_members:
                self._occurrences = np.zeros(
                    (len(windows), self.token_cells), dtype=np.int32
                )
                self._token_ring = np.zeros(
                    (self._feature_ring_capacity,), dtype=np.int32
                )
        sparse_readout = self.config.readout_learning_rule == "sparse_three_factor"
        self._readout_synapses_pa = (
            None
            if sparse_readout
            else np.zeros(
                (self.token_cells, feature_source_cells), dtype=np.float64
            )
        )
        self._pacemaker_synapses_pa = np.zeros(self.token_cells, dtype=np.float64)
        feature_count = feature_source_cells + 1
        self._normal_matrix: np.ndarray | None = None
        self._target_moments: np.ndarray | None = None
        self._feature_batch: np.ndarray | None = None
        self._growth_feature_count = feature_count
        self._pending_growth_examples = 0
        self._growth_examples = 0
        self._local_updates = 0
        self._observed_tokens = 0
        self._learning_frozen = False
        # Race-pivot T-P8 reentrancy guard: True while an observation_sink
        # callback is executing; observe/race/reset raise a deliberate
        # RuntimeError instead of letting a driving sink recurse.
        self._sink_active = False

        rng = np.random.default_rng(self.config.random_seed)
        self._afferent_targets = np.empty(
            (self.token_cells, self.config.input_fanout), dtype=np.int32
        )
        for token in range(self.token_cells):
            self._afferent_targets[token] = rng.choice(
                self.active_reservoir_cells,
                self.config.input_fanout,
                replace=False,
            )
        self._recurrent_targets = np.zeros(
            (self.config.reservoir_cells, self.config.recurrent_fanout), dtype=np.int32
        )
        self._recurrent_targets[: self.active_reservoir_cells] = build_recurrent_targets(
            self.active_reservoir_cells,
            self.config.recurrent_fanout,
            rng,
            self.config.recurrent_topology,
        )
        signs = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        signs[: self.active_reservoir_cells] = 1.0
        inhibitory_count = round(self.config.inhibitory_fraction * self.active_reservoir_cells)
        if inhibitory_count:
            inhibitory = rng.choice(self.active_reservoir_cells, inhibitory_count, replace=False)
            signs[inhibitory] = -1.0
        self._recurrent_weights_pa = np.repeat(
            (signs * self.config.recurrent_current_pa)[:, None],
            self.config.recurrent_fanout,
            axis=1,
        )
        # Tiny fixed heterogeneity avoids artificial simultaneous spikes while
        # remaining several orders below learned and afferent currents.
        self._output_heterogeneity_pa = rng.normal(0.0, 0.01, self.token_cells)
        self._sparse_readout = (
            SparseCellReadout.create(
                output_cells=self.token_cells,
                # The sparse source space is the FULL active-family feature
                # space: potential/contact sources address s in [0, F).
                reservoir_cells=feature_source_cells,
                fan_in=self.config.readout_fan_in,
                rng=rng,
                # The flat ACTIVE-family buffer is the live source space
                # (frozen protocol §1.3); it stays a stable object whose
                # contents advance in place.
                reservoir_eligibility=self.feature_state_vector(),
                output_heterogeneity_pa=self._output_heterogeneity_pa,
                pacemaker_synapses_pa=self._pacemaker_synapses_pa,
            )
            if sparse_readout
            else None
        )
        self._token_steps = max(
            1, round(self.config.token_interval_ms / self.cell_parameters.dt_ms)
        )
        self._pulse_steps = max(1, round(self.config.input_pulse_ms / self.cell_parameters.dt_ms))
        self._recurrent_decay = np.exp(-self.cell_parameters.dt_ms / self.config.recurrent_tau_ms)
        taus = np.asarray(self.config.eligibility_taus_ms, dtype=np.float64)
        self._eligibility_decays = np.ascontiguousarray(
            np.exp(-(self.cell_parameters.dt_ms / taus)), dtype=np.float64
        )
        # Per-token decay of the sparse apposition traces, indexed by absolute
        # feature source id s (k = s // R): exp(-T / tau_k).
        self._source_trace_decays = np.ascontiguousarray(
            np.repeat(
                np.exp(-(self.config.token_interval_ms / taus)),
                self.config.reservoir_cells,
            ),
            dtype=np.float64,
        )
        self._compile_output_race()
        self._bind_native_kernel()

    def source_trace_decays(self) -> np.ndarray:
        """Per-contact apposition decay table over the F feature sources."""
        if self.feature_members != ("traces",):
            # R0-b §1.3 generalization, dt-free: block b's contacts decay per
            # token by exp(-1 / W_(block)); B repeats each window three times
            # (one block per pinned quantile). Bitwise exp(-1/W) per source.
            widths_per_block: list[int] = []
            for member in self.feature_members:
                if member == "voltage_quantiles":
                    widths_per_block.extend(
                        width for width in self._feature_window_tokens for _ in range(3)
                    )
                else:
                    widths_per_block.extend(self._feature_window_tokens)
            return np.repeat(
                np.exp(
                    -1.0 / np.asarray(widths_per_block, dtype=np.float64)
                ),
                self.config.reservoir_cells,
            )
        return self._source_trace_decays

    def feature_state_vector(self) -> np.ndarray:
        """The flat f64 readout feature space of the ACTIVE family (§1.2).

        Source id s decodes as (block, cell) = divmod(s, reservoir_cells);
        for ``traces`` this IS the incumbent flat trace bank.
        """
        if self.feature_members == ("traces",):
            return self._feature_traces
        if "voltage_quantiles" in self.feature_members:
            self._refresh_quantile_rows()
        return self._feature_rows.reshape(-1)

    # -- R0-b family boundary updates (run at the end of every observe) -----

    def _advance_feature_boundary(self, token_id: int) -> None:
        """Update the active family's transient state at the token boundary."""
        if self.feature_members == ("traces",):
            return
        t = self._observed_tokens
        if "spike_counts" in self.feature_members:
            self._advance_spike_count_boundary(t)
        if "voltage_quantiles" in self.feature_members:
            self._capture_voltage_boundary(t)
        if "assembly_overlap" in self.feature_members:
            self._advance_assembly_boundary(token_id, t)

    def _advance_spike_count_boundary(self, t: int) -> None:
        """§2.A exact integer sliding sums over the per-token count vectors.

        The kernel cleared and accumulated ``_token_spike_counts`` during this
        token's sub-steps. Each row subtracts the exiting ring entry once
        when t > W_k and adds d_t. Exits are read BEFORE the new vector is
        pushed because for the largest window (W == ring capacity) the exit
        slot aliases the incoming slot.
        """
        capacity = self._feature_ring_capacity
        incoming = self._token_spike_counts
        for k, width in enumerate(self._feature_window_tokens):
            row = self._window_counts[k]
            if t > width:
                row -= self._count_ring[(t - width - 1) % capacity]
            row += incoming
        self._count_ring[(t - 1) % capacity] = incoming

    def _capture_voltage_boundary(self, t: int) -> None:
        """§2.B boundary capture: one post-observe voltage sample per cell.

        Runs after observe() completes its full sub-step integration and
        before any strengthen/race consumption; the intra-token trajectory is
        deliberately not stored. On CUDA the sample set is identical because
        ``voltage`` is mirrored per token in family-B arms (§3 requirement 3).
        """
        capacity = self._feature_ring_capacity
        self._voltage_ring[(t - 1) % capacity] = (
            self._reservoir.state.voltage_relative_mv
        )
        self._quantile_stale = True

    def _refresh_quantile_rows(self) -> None:
        """Recompute §2.B rows: pinned np.quantile(method='linear') over each
        window's ring slice (warm-up truncated). Exact and platform-stable;
        running/approximate estimators are rejected ex ante."""
        if not self._quantile_stale:
            return
        capacity = self._feature_ring_capacity
        fill = min(self._observed_tokens, capacity)
        windows = self._feature_window_tokens
        if fill == 0:
            self._quantile_rows[...] = 0.0
            self._quantile_stale = False
            return
        samples = self._voltage_ring[:fill]
        for k, width in enumerate(windows):
            lo = max(0, fill - width)
            block = np.quantile(
                samples[lo:], (0.10, 0.50, 0.90), axis=0, method="linear"
            )
            self._quantile_rows[3 * k : 3 * k + 3] = block
        self._quantile_stale = False

    def _advance_assembly_boundary(self, token_id: int, t: int) -> None:
        """§2.C distinct-assembly coverage via occurrence counters.

        Host-only bookkeeping: no kernel or mirror changes (§3 requirement 4).
        Occurrence transitions fire only on 0->1 (arrival) and 1->0 (expiry),
        so a token repeated inside a window counts ONCE; the exit entry is
        read before the ring write because the largest window aliases slots.
        """
        capacity = self._feature_ring_capacity
        targets = self._afferent_targets[token_id]
        for k, width in enumerate(self._feature_window_tokens):
            occurrences = self._occurrences[k]
            row = self._assembly_counts[k]
            if t > width:
                leaving = self._token_ring[(t - width - 1) % capacity]
                occurrences[leaving] -= 1
                if occurrences[leaving] == 0:
                    np.add.at(row, self._afferent_targets[leaving], -1.0)
            occurrences[token_id] += 1
            if occurrences[token_id] == 1:
                np.add.at(row, targets, 1.0)
        self._token_ring[(t - 1) % capacity] = token_id

    def _bind_native_kernel(self) -> None:
        """Bind native execution to the circuit's current checkpoint arrays."""
        self._native_kernel = recurrent_kernel(
            self._reservoir,
            token_steps=self._token_steps,
            pulse_steps=self._pulse_steps,
            input_fanout=self.config.input_fanout,
            recurrent_fanout=self.config.recurrent_fanout,
            input_current_pa=self.config.input_current_pa,
            recurrent_decay=self._recurrent_decay,
            eligibility_decays=self._eligibility_decays,
            afferent_targets=self._afferent_targets,
            recurrent_targets=self._recurrent_targets,
            recurrent_weights_pa=self._recurrent_weights_pa,
            synaptic_current_pa=self._synaptic_current_pa,
            # Native kernels address the flat K·R bank (frozen protocol §1.3).
            eligibility=self._feature_traces,
            host_mirror=(
                "eligibility"
                if self.config.readout_learning_rule == "sparse_three_factor"
                else "full"
            ),
            # The u32 accumulator is live kernel state. CUDA mirrors it when
            # either the feature family or an observation sink consumes it.
            token_spike_counts=self._token_spike_counts,
            mirror_token_spike_counts=(
                "spike_counts" in self.feature_members
                or self._observation_sink is not None
            ),
            mirror_boundary_voltage="voltage_quantiles" in self.feature_members,
        )
        self._native_local_learner = (
            NativeThreeFactorLearner(
                eligibility=self.feature_state_vector(),
                output_heterogeneity_pa=self._output_heterogeneity_pa,
                readout_synapses_pa=self._readout_synapses_pa,
                pacemaker_synapses_pa=self._pacemaker_synapses_pa,
            )
            if self.config.readout_learning_rule == "three_factor"
            else None
        )

    def _compile_output_race(self) -> None:
        """Compile the exact resting GLIF5 response before its first spike."""
        p = self.cell_parameters
        steps = max(1, round(self.config.output_race_ms / p.dt_ms))
        (
            self._output_unit_voltage,
            self._output_unit_voltage_threshold,
        ) = self._reservoir.compile_resting_response(steps)
        self._output_unit_spike_gain = (
            self._output_unit_voltage - self._output_unit_voltage_threshold
        )
        if np.any(self._output_unit_spike_gain <= 0.0) or np.any(
            np.diff(self._output_unit_spike_gain) < 0.0
        ):
            raise RuntimeError("compiled resting GLIF5 spike gain must be monotonic")

    def capture_cell_synapse_state(
        self,
        *,
        include_ridge_statistics: bool = True,
    ) -> RecurrentCellSynapseState:
        """Capture persistent anatomy and learning state owned by this circuit.

        ``include_ridge_statistics=False`` is the v5 dieted capture for
        artifact families that never migrate statistics: ridge sufficient
        statistics are skipped (whether absent after a dieted restore or
        present after growth) and metadata reduces to the canonical circuit
        schema. The default keeps the statistics-carrying behavior of the
        recurrent-checkpoint npz envelope.
        """
        if not self.learning_frozen:
            raise RuntimeError("freeze recurrent readout learning before checkpointing")
        layout = _READOUT_STATE_LAYOUTS[self.config.readout_learning_rule]
        dieted = layout.ridge_statistics and not include_ridge_statistics
        if layout.ridge_statistics and not dieted:
            self._flush_growth_statistics()
            if self._normal_matrix is None or self._target_moments is None:
                raise RuntimeError("cannot checkpoint a recurrent circuit before growth")
        elif not layout.ridge_statistics:
            if self._normal_matrix is not None or self._target_moments is not None:
                raise RuntimeError("local readout must not contain ridge statistics")

        developmental_present = tuple(
            hasattr(self, attribute)
            for attribute, _, _ in _DEVELOPMENTAL_STATE_ARRAYS
        )
        if any(developmental_present) and not all(developmental_present):
            raise RuntimeError("developmental checkpoint state is incomplete")
        developmental_state = all(developmental_present)
        metadata: dict[str, object] = {
            "token_cells": self.token_cells,
            "allen_glif5_config_sha256": HUMAN_GLIF5_IDENTITY.config_sha256,
            "cell_parameters": asdict(self.cell_parameters),
            "circuit_config": asdict(self.config),
            "growth_examples": self._growth_examples,
            "local_updates": self.local_update_count,
            "runtime_active_reservoir_cells": self.active_reservoir_cells,
            "developmental_state": developmental_state,
            # Explicit v5 guards alongside the asdict'd circuit config.
            "eligibility_taus_ms": list(self.config.eligibility_taus_ms),
            "feature_cell_count": self.feature_cell_count,
        }
        if self.feature_members != ("traces",) or (
            self.config.feature_window_tokens is not None
        ):
            # R0-b v6 guards: family and window bank ride beside the count for
            # successor families (and any non-default window override).
            metadata["feature_family"] = self.config.feature_family
            metadata["feature_window_tokens"] = list(self._feature_window_tokens)
        if dieted:
            # Dieted payloads reduce to the canonical circuit metadata schema:
            # era guards belong to statistics-carrying envelopes only.
            for key in _ENVELOPE_ONLY_METADATA_KEYS:
                metadata.pop(key, None)
        excluded_arrays = (
            _RIDGE_STATISTIC_ARRAY_KEYS if dieted else frozenset()
        )
        arrays: dict[str, np.ndarray] = {}
        for attribute, key, _ in _BASE_STATE_ARRAYS + layout.direct_arrays:
            if key in excluded_arrays:
                continue
            value = getattr(self, attribute)
            if value is None:
                raise RuntimeError(f"checkpoint state {key} is missing")
            arrays[key] = value

        if layout.sparse:
            if self._sparse_readout is None:
                raise RuntimeError("sparse readout checkpoint state is missing")
            sparse_arrays = self._sparse_readout.checkpoint_arrays()
            arrays.update({key: sparse_arrays[key] for key in layout.payload_arrays})
            metadata["sparse_readout"] = self._sparse_readout.checkpoint_counters()
        elif self._sparse_readout is not None:
            raise RuntimeError("dense readout must not contain sparse checkpoint state")

        if developmental_state:
            for attribute, key, _ in _DEVELOPMENTAL_STATE_ARRAYS:
                arrays[key] = getattr(self, attribute)
        return RecurrentCellSynapseState(metadata=metadata, arrays=arrays)

    @property
    def cell_synapse_state_sha256(self) -> str:
        """Hash every persistent cell, synapse, topology, and readout array."""
        return self.capture_cell_synapse_state().sha256

    def save_cell_synapse_checkpoint(
        self,
        path: str | Path,
        *,
        envelope: Mapping[str, object],
    ) -> str:
        """Save circuit-owned persistent state under caller metadata."""
        return self.capture_cell_synapse_state().save_enveloped_checkpoint(
            Path(path),
            envelope,
        )

    @classmethod
    def load_cell_synapse_checkpoint(
        cls,
        path: str | Path,
    ) -> tuple[dict[str, object], RecurrentCellCircuit]:
        """Load one generic state envelope and restore its owning circuit."""
        envelope, state = RecurrentCellSynapseState.load_enveloped_checkpoint(
            Path(path)
        )
        return envelope, cls.restore_cell_synapse_state(state)

    @classmethod
    def decode_cell_synapse_state(
        cls,
        metadata: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
    ) -> RecurrentCellSynapseState:
        """Select and copy circuit-owned state out of a larger artifact payload."""
        _, layout, _, array_names = (
            cls._validate_cell_synapse_payload(metadata, arrays)
        )
        selected_metadata = {key: metadata[key] for key in _CELL_SYNAPSE_METADATA_KEYS}
        # Era guards ride along when the payload carries them (v5/v6); legacy
        # v3 payloads simply lack the keys and stay single-trace.
        for key in (
            "eligibility_taus_ms",
            "feature_cell_count",
            "feature_family",
            "feature_window_tokens",
        ):
            if key in metadata:
                selected_metadata[key] = metadata[key]
        if layout.sparse:
            selected_metadata["sparse_readout"] = metadata["sparse_readout"]
        return RecurrentCellSynapseState.copy_payload(
            selected_metadata,
            arrays,
            array_names,
        )

    @classmethod
    def restore_cell_synapse_state(
        cls,
        state: RecurrentCellSynapseState,
        *,
        observation_sink: (
            Callable[["RecurrentCellCircuit", RecurrentStep], None] | None
        ) = None,
    ) -> RecurrentCellCircuit:
        """Restore one validated persistent state into a newly bound circuit."""
        metadata = state.metadata
        config, layout, developmental_state, _ = cls._validate_cell_synapse_payload(
            metadata,
            state.arrays,
        )
        circuit = cls(
            token_cells=int(metadata["token_cells"]),
            config=config,
            cell_parameters=AllenGlif5Parameters(**metadata["cell_parameters"]),
            observation_sink=observation_sink,
        )
        direct_arrays = _BASE_STATE_ARRAYS + layout.direct_arrays
        for attribute, key, dtype in direct_arrays:
            existing = getattr(circuit, attribute)
            if attribute == "_normal_matrix":
                expected = (
                    circuit._growth_feature_count,
                    circuit._growth_feature_count,
                )
            elif attribute == "_target_moments":
                expected = (circuit.token_cells, circuit._growth_feature_count)
            else:
                expected = existing.shape
            value = np.array(state.arrays[key], dtype=dtype, copy=True, order="C")
            if value.shape != expected:
                raise ValueError(
                    f"checkpoint array {key} has shape {value.shape}, expected {expected}"
                )
            setattr(circuit, attribute, value)

        if layout.sparse:
            sparse_counters = metadata["sparse_readout"]
            assert isinstance(sparse_counters, dict)
            circuit._sparse_readout = SparseCellReadout.from_checkpoint(
                output_cells=circuit.token_cells,
                # Sparse source space spans the full active-family space.
                reservoir_cells=circuit.feature_cell_count,
                fan_in=circuit.config.readout_fan_in,
                reservoir_eligibility=circuit.feature_state_vector(),
                output_heterogeneity_pa=circuit._output_heterogeneity_pa,
                pacemaker_synapses_pa=circuit._pacemaker_synapses_pa,
                arrays=state.arrays,
                counters=sparse_counters,
            )
            if circuit._sparse_readout.applied_updates != int(
                metadata["local_updates"]
            ):
                raise ValueError("sparse checkpoint update counters disagree")

        if developmental_state:
            developmental_shapes = {
                "_development_cell_order": (circuit.config.reservoir_cells,),
                "_development_active_mask": (circuit.config.reservoir_cells,),
                "_structural_scores": circuit._recurrent_weights_pa.shape,
                "_structural_exposure": circuit._recurrent_weights_pa.shape,
                "_homeostatic_gain": (circuit.config.reservoir_cells,),
            }
            for attribute, key, dtype in _DEVELOPMENTAL_STATE_ARRAYS:
                value = np.array(
                    state.arrays[key],
                    dtype=dtype,
                    copy=True,
                    order="C",
                )
                expected = developmental_shapes[attribute]
                if value.shape != expected:
                    raise ValueError(
                        f"checkpoint array {key} has shape {value.shape}, "
                        f"expected {expected}"
                    )
                setattr(circuit, attribute, value)

        circuit.active_reservoir_cells = int(
            metadata["runtime_active_reservoir_cells"]
        )
        if developmental_state:
            if (
                int(np.count_nonzero(circuit._development_active_mask))
                != circuit.active_reservoir_cells
            ):
                raise ValueError("checkpoint active mask/count mismatch")
            if not np.array_equal(
                np.sort(circuit._development_cell_order),
                np.arange(circuit.config.reservoir_cells, dtype=np.int32),
            ):
                raise ValueError(
                    "checkpoint developmental cell order is not a permutation"
                )
        circuit._bind_native_kernel()
        circuit._growth_examples = int(metadata["growth_examples"])
        circuit._local_updates = int(metadata["local_updates"])
        circuit._learning_frozen = True
        return circuit

    @staticmethod
    def _validate_cell_synapse_payload(
        metadata: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
    ) -> tuple[
        RecurrentCircuitConfig,
        _ReadoutStateLayout,
        bool,
        tuple[str, ...],
    ]:
        missing_metadata = tuple(
            key for key in _CELL_SYNAPSE_METADATA_KEYS if key not in metadata
        )
        if missing_metadata:
            raise ValueError(f"checkpoint metadata {missing_metadata[0]} is missing")
        if (
            metadata["allen_glif5_config_sha256"]
            != HUMAN_GLIF5_IDENTITY.config_sha256
        ):
            raise ValueError("checkpoint uses a different adaptive Allen human cell")
        config_value = metadata["circuit_config"]
        cell_parameters = metadata["cell_parameters"]
        if not isinstance(config_value, dict) or not isinstance(cell_parameters, dict):
            raise ValueError("checkpoint circuit configuration is malformed")
        config = RecurrentCircuitConfig(**config_value)
        # Checkpoint era guards (frozen protocol §1.5): v5 payloads carry
        # explicit tau and feature-space guards and are loaded strictly;
        # legacy v3-era payloads (neither key present) restore under the
        # K=1 interpretation exactly. Mixed-era metadata raises loudly.
        trace_count = len(config.eligibility_taus_ms)
        feature_cell_count = resolved_feature_cells(config)
        metadata_taus = metadata.get("eligibility_taus_ms")
        metadata_feature_cells = metadata.get("feature_cell_count")
        if metadata_taus is None and metadata_feature_cells is None:
            if trace_count != 1:
                raise ValueError(
                    "legacy single-trace checkpoint payload cannot satisfy a "
                    f"multi-trace configuration ({trace_count} taus requested)"
                )
        else:
            if metadata_taus is None or metadata_feature_cells is None:
                raise ValueError("checkpoint multi-trace metadata is incomplete")
            if list(metadata_taus) != list(config.eligibility_taus_ms):
                raise ValueError(
                    "checkpoint metadata eligibility_taus_ms disagrees with "
                    "the circuit configuration"
                )
            if int(metadata_feature_cells) != feature_cell_count:
                raise ValueError(
                    "checkpoint metadata feature_cell_count disagrees with "
                    "the circuit configuration"
                )
        RecurrentCellCircuit._validate_feature_family_guards(metadata, config)
        layout = _READOUT_STATE_LAYOUTS[config.readout_learning_rule]
        developmental_state = bool(metadata["developmental_state"])
        array_names = tuple(
            key for _, key, _ in _BASE_STATE_ARRAYS + layout.direct_arrays
        ) + layout.payload_arrays
        if developmental_state:
            array_names += tuple(key for _, key, _ in _DEVELOPMENTAL_STATE_ARRAYS)
        missing_arrays = tuple(key for key in array_names if key not in arrays)
        if missing_arrays:
            raise ValueError(f"checkpoint array {missing_arrays[0]} is missing")
        if layout.sparse and not isinstance(metadata.get("sparse_readout"), dict):
            raise ValueError("sparse checkpoint counters are missing")
        return config, layout, developmental_state, array_names

    @staticmethod
    def _validate_feature_family_guards(
        metadata: Mapping[str, object],
        config: RecurrentCircuitConfig,
    ) -> None:
        """R0-b §1.5 v6 strictness + v5 legacy load discipline."""
        metadata_family = metadata.get("feature_family")
        metadata_windows = metadata.get("feature_window_tokens")
        if metadata_family is None and metadata_windows is None:
            # v5-era payload (or v3): its layout is the traces family by
            # construction; any other configured family must refuse loudly
            # instead of silently reshaping or retraining in place.
            if parse_feature_family(config.feature_family) != ("traces",):
                raise ValueError(
                    "checkpoint payload predates feature-family guards (v5 "
                    "era) and can only satisfy the traces family; refusing "
                    f"the requested feature family {config.feature_family!r}"
                )
            return
        if metadata_family is None or metadata_windows is None:
            raise ValueError("checkpoint feature-family metadata is incomplete")
        if metadata_family != config.feature_family:
            raise ValueError(
                f"checkpoint metadata feature family {metadata_family!r} "
                f"disagrees with the circuit configuration "
                f"{config.feature_family!r}"
            )
        if list(metadata_windows) != list(config._resolved_window_tokens()):
            raise ValueError(
                "checkpoint metadata feature_window_tokens disagrees with "
                "the circuit configuration"
            )

    @property
    def learned_synapse_count(self) -> int:
        if self.config.readout_learning_rule == "sparse_three_factor":
            assert self._sparse_readout is not None
            return self._sparse_readout.active_synapse_count
        assert self._readout_synapses_pa is not None
        return int(np.count_nonzero(self._readout_synapses_pa))

    @property
    def learning_frozen(self) -> bool:
        return self._learning_frozen

    @property
    def feature_cell_count(self) -> int:
        """F: the active family's per-bias feature-source count (§1.2)."""
        if self.feature_members == ("traces",):
            return self.trace_count * self.config.reservoir_cells
        return self._feature_rows.shape[0] * self.config.reservoir_cells

    @property
    def local_update_count(self) -> int:
        if self.config.readout_learning_rule == "sparse_three_factor":
            assert self._sparse_readout is not None
            return self._sparse_readout.applied_updates
        return self._local_updates

    @property
    def structural_recruitment_count(self) -> int:
        if self._sparse_readout is None:
            return 0
        return self._sparse_readout.structural_recruitments

    @property
    def structural_deletion_count(self) -> int:
        if self._sparse_readout is None:
            return 0
        return self._sparse_readout.structural_deletions

    @property
    def occupied_output_synapses_per_cell(self) -> float:
        return self.learned_synapse_count / self.token_cells

    @property
    def reservoir_activity_density(self) -> float:
        # Trace 0 IS the incumbent trace: the published activity-density
        # accounting stays a trace-0 statistic (frozen protocol §1.2).
        return (
            float(np.count_nonzero(self._feature_traces[: self.config.reservoir_cells]))
            / self.config.reservoir_cells
        )

    @property
    def observed_tokens(self) -> int:
        return self._observed_tokens

    @property
    def observation_sink(self):
        """Optional per-token observer callback (R0-c Thread-1 logging seam).

        When set to a callable, it is invoked as ``sink(circuit, token_id)``
        at the end of every :meth:`observe`, after the feature-family
        boundary update. The callback receives the circuit so it can read
        ``feature_state_vector()``, the raw banks, and reservoir state.
        Observers must not drive the circuit (no observe/race/reset inside).
        Instance-level only: it is not part of ``RecurrentCircuitConfig``
        and does not alter any checkpoint schema or digest.
        """
        return self._observation_sink

    @observation_sink.setter
    def observation_sink(self, sink) -> None:
        if sink is not None and not callable(sink):
            raise ValueError("observation_sink must be callable or None")
        self._observation_sink = sink

    @property
    def simulation_backend(self) -> str:
        return self._native_kernel.backend

    @property
    def simulation_device(self) -> str:
        return self._native_kernel.device_name

    @property
    def eligibility(self) -> np.ndarray:
        # Back-compat view: trace 0, the incumbent single trace.
        return self._feature_traces[: self.config.reservoir_cells].copy()

    def synchronize_structure_from_host(self) -> None:
        """Publish in-place afferent, recurrent-topology, and weight mutations."""
        self._native_kernel.synchronize_structure_from_host()

    def synchronize_state_from_device(self) -> None:
        """Make every mutable fitted-cell array available for host inspection."""
        self._native_kernel.synchronize_to_host()

    def synchronize_state_from_host(self) -> None:
        """Alias of :meth:`synchronize_state_from_device` (R0-b red-test seam)."""
        self.synchronize_state_from_device()

    def reset_sequence(self) -> None:
        self._guard_sink_reentrancy("reset")
        self._reservoir.reset()
        self._synaptic_current_pa.fill(0.0)
        self._eligibility_traces.fill(0.0)
        if self.feature_members != ("traces",):
            # R0-b transient feature state is sequence-scoped (§1.2).
            self._token_spike_counts.fill(0)
            self._feature_rows.fill(0.0)
            if "spike_counts" in self.feature_members:
                self._count_ring.fill(0)
            if "voltage_quantiles" in self.feature_members:
                self._voltage_ring.fill(0.0)
                self._quantile_stale = True
            if "assembly_overlap" in self.feature_members:
                self._occurrences.fill(0)
                self._token_ring.fill(0)
        self._native_kernel.synchronize_from_host()
        self._observed_tokens = 0

    def _guard_sink_reentrancy(self, operation: str) -> None:
        """T-P8 contract: a driving observation_sink must raise deliberately.

        Observers are read-only (observation_sink docstring); a callback that
        calls observe/race/reset would otherwise recurse unchecked (historically
        to RecursionError). Raise immediately, BEFORE any state mutation.
        """
        if self._sink_active:
            raise RuntimeError(
                "observation_sink reentrancy guard: the sink callback drove "
                f"the circuit via {operation}(); observers must not call "
                "observe/race/reset (recurrent.py observation_sink contract)"
            )

    def observe(self, token_id: int) -> RecurrentStep:
        """Apply one token as afferent current and advance the cell circuit."""
        self._guard_sink_reentrancy("observe")
        token = int(token_id)
        if not 0 <= token < self.token_cells:
            raise ValueError("token id outside output-cell population")
        spike_total = self._native_kernel.observe(token)

        self._observed_tokens += 1
        # R0-b §1.3: after the native token advance returns, the family's
        # boundary update runs, then the flat feature buffer is current for
        # strengthen/race/inspection.
        self._advance_feature_boundary(token)
        step = RecurrentStep(
            token_id=token,
            reservoir_spikes=spike_total,
            simulated_steps=self._token_steps,
            biological_time_ms=self._token_steps * self.cell_parameters.dt_ms,
        )
        if self._observation_sink is not None:
            # R0-c Thread-1: fired AFTER the family boundary update so the
            # observer reads the same post-observe state any strengthen/race
            # consumer would see. The sink must not drive the circuit (no
            # observe/race/reset calls inside the callback); T-P8 makes that
            # contract enforced rather than documented.
            self._sink_active = True
            try:
                self._observation_sink(self, step)
            finally:
                self._sink_active = False
        return step

    def capture_dynamics(self, gids: Sequence[int]) -> RecurrentDynamicsSample:
        """Copy selected live state without exposing full population arrays."""
        selected = tuple(gids)
        if any(
            type(gid) is not int or not 0 <= gid < self.active_reservoir_cells
            for gid in selected
        ):
            raise ValueError("sample gids must be exact integers in the active reservoir")
        indices = np.asarray(selected, dtype=np.intp)
        state = self._reservoir.state
        return RecurrentDynamicsSample(
            gids=selected,
            voltage_relative_mv=tuple(
                float(value) for value in state.voltage_relative_mv[indices]
            ),
            synaptic_current_pa=tuple(
                float(value) for value in self._synaptic_current_pa[indices]
            ),
            eligibility=tuple(
                float(value) for value in self._eligibility_traces[0, indices]
            ),
        )

    def strengthen(self, target_token_id: int) -> None:
        """Apply the configured readout plasticity rule after one observation."""
        target = int(target_token_id)
        if not 0 <= target < self.token_cells:
            raise ValueError("target token outside output-cell population")
        if self._observed_tokens < 1:
            raise RuntimeError("observe a token before strengthening output synapses")
        if self._learning_frozen:
            raise RuntimeError("readout learning is frozen")
        self._growth_examples += 1
        if self.config.readout_learning_rule == "sparse_three_factor":
            assert self._sparse_readout is not None
            race = self.race()
            current = self.output_current_pa()
            self._last_output_winner = race.winner
            if race.winner == target:
                self._last_teaching_current_pa = float(current[target])
                self._last_teaching_spike_time_ms = race.spike_time_ms
            else:
                teaching_current = float(
                    current[target] + self.config.output_synapse_gain_pa
                )
                p = self.cell_parameters
                teaching_voltage = self._output_unit_voltage * teaching_current
                teaching_threshold = (
                    p.threshold_inf_relative_mv
                    + self._output_unit_voltage_threshold * teaching_current
                )
                crossing = np.flatnonzero(teaching_voltage > teaching_threshold)
                if not crossing.size:
                    raise RuntimeError("teaching current did not evoke an output-cell spike")
                self._last_teaching_current_pa = teaching_current
                self._last_teaching_spike_time_ms = (int(crossing[0]) + 1) * p.dt_ms
            self._sparse_readout.teach(
                target=target,
                winner=race.winner,
                learning_rate_pa=self.config.local_learning_rate_pa,
                maximum_weight_pa=self.config.output_synapse_gain_pa,
                # Per-contact apposition decay: exp(-T / tau_(source // R)),
                # applied per slot inside the readout (frozen protocol §1.3).
                trace_decays=self._source_trace_decays,
                target_post_gate=1.0,
                winner_post_gate=1.0,
                activity_decay=self.config.readout_activity_decay,
            )
            presentation = self._sparse_readout.advance_target(target)
            if presentation % self.config.readout_structural_interval == 0:
                self._sparse_readout.turnover(
                    output=target,
                    structural_event=presentation // self.config.readout_structural_interval,
                    random_seed=self.config.random_seed,
                    minimum_age=self.config.readout_structural_grace,
                    turnover_probability=self.config.readout_turnover_probability,
                    learning_rate_pa=self.config.local_learning_rate_pa,
                    maximum_weight_pa=self.config.output_synapse_gain_pa,
                    homeostatic_rate=self.config.readout_homeostatic_rate,
                )
            return
        if self.config.readout_learning_rule == "three_factor":
            assert self._native_local_learner is not None
            winner = self._native_local_learner.update(
                target,
                self.config.local_learning_rate_pa,
                self.config.local_competitive_margin_pa,
            )
            self._local_updates += int(winner is not None)
            return

        self._ensure_growth_storage()
        assert self._feature_batch is not None
        assert self._target_moments is not None
        feature = self._feature_batch[self._pending_growth_examples]
        # R0-b §1.3: the readout consumes the ACTIVE family's flat buffer.
        feature[:-1] = self.feature_state_vector()
        feature[-1] = 1.0
        self._target_moments[target] += feature
        self._pending_growth_examples += 1
        if self._pending_growth_examples == self._STATISTICS_BATCH_SIZE:
            self._flush_growth_statistics()

    def _ensure_growth_storage(self) -> None:
        """Allocate dense ridge statistics only when the circuit learns."""
        if self._normal_matrix is None:
            self._normal_matrix = np.zeros(
                (self._growth_feature_count, self._growth_feature_count),
                dtype=np.float64,
            )
        if self._target_moments is None:
            self._target_moments = np.zeros(
                (self.token_cells, self._growth_feature_count),
                dtype=np.float64,
            )
        if self._feature_batch is None:
            self._feature_batch = np.empty(
                (self._STATISTICS_BATCH_SIZE, self._growth_feature_count),
                dtype=np.float64,
            )

    def _flush_growth_statistics(self) -> None:
        """Accumulate a rank-k batch of cell traces into readout statistics."""
        if self._pending_growth_examples < 1:
            return
        assert self._feature_batch is not None
        assert self._normal_matrix is not None
        features = self._feature_batch[: self._pending_growth_examples]
        self._normal_matrix += features.T @ features
        self._pending_growth_examples = 0

    def _consolidate_readout(self) -> None:
        """Fit output-synapse strengths from accumulated cell traces."""
        if self.config.readout_learning_rule != "ridge":
            return
        if self._growth_examples < 1:
            raise RuntimeError("cannot consolidate without growth examples")
        self._flush_growth_statistics()
        assert self._normal_matrix is not None
        assert self._target_moments is not None
        regularized = self._normal_matrix.copy()
        diagonal = np.diag_indices_from(regularized)
        regularized[diagonal] += self.config.readout_ridge
        # The pacemaker cell is a tonic biological bias and needs much less
        # regularization than the contextual reservoir synapses.
        regularized[-1, -1] -= self.config.readout_ridge * 0.99
        coefficients = np.linalg.solve(regularized, self._target_moments.T).T
        gain = self.config.output_synapse_gain_pa
        self._readout_synapses_pa[:] = coefficients[:, :-1] * gain
        self._pacemaker_synapses_pa[:] = coefficients[:, -1] * gain

    def freeze_learning(self) -> None:
        """Finalize any deferred rule state and disable further plasticity."""
        if self._learning_frozen:
            return
        if self._growth_examples < 1:
            raise RuntimeError("cannot freeze readout before growth")
        self._consolidate_readout()
        self._learning_frozen = True

    def grow_sequence(self, token_ids: Sequence[int], epochs: int = 1) -> None:
        """Replay a sequence and grow synapses from cell eligibility traces.

        On the CUDA backend the loop is double-buffered: the next token's
        kernel is enqueued before the current token's readout learning runs,
        so host-side strengthen work and the D2H eligibility staging overlap
        kernel execution. The consumed eligibility bytes are bit-identical
        to the synchronous path, so the training trajectory is unchanged.
        Token ids are validated up front (the synchronous path rejects an
        out-of-range id mid-sequence instead; valid inputs are unaffected).
        """
        if self._learning_frozen:
            raise RuntimeError("readout learning is frozen")
        ids = _normalized_token_ids(token_ids)
        if len(ids) < 2:
            raise ValueError("growth sequence needs at least two tokens")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        self._validate_token_range(ids)
        pipeline_available = callable(
            getattr(self._native_kernel, "observe_async", None)
        ) and bool(getattr(self._native_kernel, "async_pipeline_active", False))
        for _ in range(epochs):
            self.reset_sequence()
            if pipeline_available:
                self._grow_sequence_pipelined(ids)
            else:
                for index in range(len(ids) - 1):
                    self.observe(ids[index])
                    self.strengthen(ids[index + 1])
        if self.config.readout_learning_rule == "ridge":
            self._consolidate_readout()

    def _grow_sequence_pipelined(self, ids: list[int]) -> None:
        """Grow one epoch keeping the GPU saturated ahead of host learning.

        Submissions run up to ``_PIPELINE_DEPTH - 1`` tokens ahead of the
        host, so the device never drains between cooperative launches: the
        next kernels are already queued while the current one executes and
        while readout learning works on the host. Collected eligibility
        bytes are identical to the synchronous order, so learning sees the
        same trajectory.
        """
        total = len(ids)
        lookahead = self._native_kernel._PIPELINE_DEPTH - 1
        submit = self._observe_submit
        inflight: deque[tuple[int, int]] = deque([(ids[0], submit(ids[0]))])
        for index in range(total - 1):
            while (
                len(inflight) < lookahead
                and index + len(inflight) < total - 1
            ):
                token_id = ids[index + len(inflight)]
                inflight.append((token_id, submit(token_id)))
            token_id, handle = inflight.popleft()
            self._observe_collect(token_id, handle)
            self.strengthen(ids[index + 1])
        while inflight:
            self._observe_collect(*inflight.popleft())

    def _observe_submit(self, token_id: int) -> int:
        token = int(token_id)
        if not 0 <= token < self.token_cells:
            raise ValueError("token id outside output-cell population")
        return self._native_kernel.observe_async(token)

    def _observe_collect(self, token_id: int, handle: int) -> RecurrentStep:
        spike_total = self._native_kernel.collect(handle)
        self._observed_tokens += 1
        step = RecurrentStep(
            token_id=int(token_id),
            reservoir_spikes=spike_total,
            simulated_steps=self._token_steps,
            biological_time_ms=self._token_steps * self.cell_parameters.dt_ms,
        )
        if self._observation_sink is not None:
            self._observation_sink(self, step)
        return step

    def _validate_token_range(self, ids: list[int]) -> None:
        if not ids:
            return
        array = np.fromiter(ids, dtype=np.int64, count=len(ids))
        if array.min() < 0 or array.max() >= self.token_cells:
            raise ValueError("token id outside output-cell population")

    def output_current_pa(self) -> np.ndarray:
        """Read the current delivered by all learned output synapses."""
        if self.config.readout_learning_rule == "sparse_three_factor":
            assert self._sparse_readout is not None
            return self._sparse_readout.current(self.config.output_base_current_pa)
        assert self._readout_synapses_pa is not None
        # R0-b §1.3: inference consumes the ACTIVE family's flat buffer via
        # the same matrix product the trace path uses.
        return (
            self.config.output_base_current_pa
            + self._pacemaker_synapses_pa
            + self._output_heterogeneity_pa
            + self._readout_synapses_pa @ self.feature_state_vector()
        )

    def race(
        self, allowed_token_ids: Sequence[int] | None = None
    ) -> RecurrentRace:
        """Race output cells under synaptic current; the first spike is the token."""
        self._guard_sink_reentrancy("race")
        if self._observed_tokens < 1:
            raise RuntimeError("observe context before racing output cells")
        return self._race_output_candidates(
            self.output_current_pa(),
            self._normalize_race_candidates(allowed_token_ids),
        )

    def race_output_current_pa(
        self,
        current_pa: Sequence[float] | np.ndarray,
        allowed_token_ids: Sequence[int] | None = None,
    ) -> RecurrentRace:
        """Run the exact fitted-cell race for supplied output-synapse currents.

        This non-mutating seam exposes the cell-dynamics half of ``race`` for
        independent equivalence tests. Production completion obtains the same
        currents from ``output_current_pa`` through ``race``.
        """
        current = np.asarray(current_pa, dtype=np.float64)
        if current.shape != (self.token_cells,):
            raise ValueError("output current must have one value per output cell")
        if not np.all(np.isfinite(current)):
            raise ValueError("output current must be finite")

        return self._race_output_candidates(
            current,
            self._normalize_race_candidates(allowed_token_ids),
        )

    def _normalize_race_candidates(
        self,
        allowed_token_ids: Sequence[int] | None,
    ) -> np.ndarray | None:
        if allowed_token_ids is None:
            return None
        allowed_ids = tuple(int(token_id) for token_id in allowed_token_ids)
        if not allowed_ids:
            raise ValueError("candidate set needs at least one allowed output cell")
        if any(
            token_id < 0 or token_id >= self.token_cells
            for token_id in allowed_ids
        ):
            raise ValueError("allowed token outside output-cell population")
        return np.unique(np.asarray(allowed_ids, dtype=np.intp))

    def complete_broadcast(
        self,
        prompt_ids,
        max_new_tokens: int,
        *,
        broadcast_b: int = 1,
        variant: str = "P-S",
        chase_to_settle: bool | None = None,
        max_settle_rounds: int | None = None,
        record_settle_profile: bool = False,
    ):
        """Circuit-level winner-feedback broadcast free-run (race-pivot §3.1).

        Thin delegate to :func:`humanbrain.race_pivot.broadcast_free_run` — the
        host-side loop over the public ``race()``/``observe()`` primitives; no
        race semantics or cell dynamics are touched. Consumes and resets the
        sequence here (prompt ingestion follows this module's logging contract:
        one sink-visible event per output DECISION — see the race_pivot module
        docstring). Returns a ``race_pivot.BroadcastRunResult``.
        """
        from humanbrain.race_pivot import (
            MAX_SETTLE_ROUNDS_DEFAULT,
            broadcast_free_run,
        )

        return broadcast_free_run(
            self,
            prompt_ids,
            max_new_tokens,
            broadcast_b=broadcast_b,
            variant=variant,
            chase_to_settle=chase_to_settle,
            max_settle_rounds=(
                MAX_SETTLE_ROUNDS_DEFAULT
                if max_settle_rounds is None
                else max_settle_rounds
            ),
            consume_prompt=True,
            record_settle_profile=record_settle_profile,
        )

    def _race_output_candidates(
        self,
        current: np.ndarray,
        candidates: np.ndarray | None,
    ) -> RecurrentRace:
        """Race all cells or one sorted candidate subset through one GLIF core."""
        candidate_current = current if candidates is None else current[candidates]
        p = self.cell_parameters
        winning_current = candidate_current[int(np.argmax(candidate_current))]
        winner_voltage = self._output_unit_voltage * winning_current
        winner_threshold = self._output_unit_voltage_threshold * winning_current
        crossing_steps = np.flatnonzero(
            winner_voltage > (p.threshold_inf_relative_mv + winner_threshold)
        )
        if crossing_steps.size:
            step = int(crossing_steps[0])
            voltage_at_crossing = self._output_unit_voltage[step] * candidate_current
            fired = voltage_at_crossing > (
                p.threshold_inf_relative_mv
                + self._output_unit_voltage_threshold[step] * candidate_current
            )
            # Apply the fitted reset on the firing integration bin before
            # reporting its peak, exactly as stepwise simulation does.
            voltage_at_crossing[fired] = (
                p.voltage_reset_scale * voltage_at_crossing[fired] + p.voltage_reset_offset_mv
            )
            peak_relative = float(np.max(voltage_at_crossing))
            if step:
                peak_relative = max(
                    peak_relative,
                    float(winner_voltage[step - 1]),
                )
            fired_offsets = np.flatnonzero(fired)
            winner_offset = int(fired_offsets[0])
            winner = (
                winner_offset
                if candidates is None
                else int(candidates[winner_offset])
            )
            co_winners = int(np.count_nonzero(fired))
            excluded = (
                ()
                if candidates is None
                else self._excluded_before_candidate(
                    current,
                    candidates,
                    winner_step=step,
                    winner=winner,
                )
            )
            return RecurrentRace(
                winner=winner,
                spike_time_ms=(step + 1) * p.dt_ms,
                peak_voltage_mv=max(p.rest_mv, p.rest_mv + peak_relative),
                active_cells=(
                    self.token_cells if candidates is None else int(candidates.size)
                ),
                simulated_steps=step + 1,
                simultaneous_spike_candidates=co_winners,
                discarded_simultaneous_candidates=max(0, co_winners - 1),
                excluded_token_ids=excluded,
                output_current_sha256=(
                    ""
                    if candidates is None
                    else sha256(np.ascontiguousarray(current).tobytes()).hexdigest()
                ),
            )
        raise RuntimeError("output human cells did not spike during the race")

    def _excluded_before_candidate(
        self,
        current: np.ndarray,
        candidates: np.ndarray,
        *,
        winner_step: int,
        winner: int,
    ) -> tuple[int, ...]:
        """Return excluded cells encountered before the accepted deterministic tie."""
        cell_ids = np.arange(self.token_cells, dtype=np.intp)
        allowed = np.zeros(self.token_cells, dtype=np.bool_)
        allowed[candidates] = True
        first_steps = self._compiled_first_spike_steps(current)
        rejected = (~allowed) & (
            (first_steps < winner_step)
            | ((first_steps == winner_step) & (cell_ids < winner))
        )
        rejected_ids = cell_ids[rejected]
        if not rejected_ids.size:
            return ()
        order = np.lexsort((rejected_ids, first_steps[rejected_ids]))
        return tuple(int(token_id) for token_id in rejected_ids[order])

    def _compiled_first_spike_steps(self, current: np.ndarray) -> np.ndarray:
        """Vectorize exact first-spike bins without a steps-by-cells matrix."""
        step_count = int(self._output_unit_spike_gain.size)
        result = np.full(current.shape, step_count, dtype=np.intp)
        positive = current > 0.0
        if not np.any(positive):
            return result
        positive_current = current[positive]
        target_gain = self.cell_parameters.threshold_inf_relative_mv / positive_current
        guesses = np.searchsorted(
            self._output_unit_spike_gain,
            target_gain,
            side="right",
        ).astype(np.intp, copy=False)

        # Division can round at an exact threshold. Recheck the original GLIF
        # inequality and adjust only boundary guesses, preserving race semantics.
        while True:
            in_range = guesses < step_count
            needs_later = np.zeros(guesses.shape, dtype=np.bool_)
            needs_later[in_range] = ~(
                self._output_unit_spike_gain[guesses[in_range]]
                * positive_current[in_range]
                > self.cell_parameters.threshold_inf_relative_mv
            )
            if not np.any(needs_later):
                break
            guesses[needs_later] += 1
        while True:
            has_previous = guesses > 0
            previous_crosses = np.zeros(guesses.shape, dtype=np.bool_)
            previous_crosses[has_previous] = (
                self._output_unit_spike_gain[guesses[has_previous] - 1]
                * positive_current[has_previous]
                > self.cell_parameters.threshold_inf_relative_mv
            )
            if not np.any(previous_crosses):
                break
            guesses[previous_crosses] -= 1
        result[positive] = guesses
        return result

    def predict_next(self, context: Sequence[int]) -> RecurrentRace:
        """Replay a context through cell state and race output token cells."""
        if not context:
            raise ValueError("prediction context cannot be empty")
        self.reset_sequence()
        for token_id in context:
            self.observe(int(token_id))
        return self.race()
