"""Sparse recurrent token circuit made only from cells and synapses."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
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
    eligibility_tau_ms: float = 80.0
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

    def __post_init__(self) -> None:
        if self.reservoir_cells < 8:
            raise ValueError("reservoir needs at least eight cells")
        active_cells = self.active_reservoir_cells or self.reservoir_cells
        if not 8 <= active_cells <= self.reservoir_cells:
            raise ValueError("active reservoir cells must be between eight and total cells")
        if not 1 <= self.input_fanout <= active_cells:
            raise ValueError("input fanout must fit the active reservoir")
        if not 1 <= self.recurrent_fanout <= active_cells:
            raise ValueError("recurrent fanout must fit the active reservoir")
        positive = (
            self.token_interval_ms,
            self.input_pulse_ms,
            self.input_current_pa,
            self.recurrent_current_pa,
            self.recurrent_tau_ms,
            self.eligibility_tau_ms,
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
        self._observation_sink = observation_sink
        self.active_reservoir_cells = (
            self.config.active_reservoir_cells or self.config.reservoir_cells
        )
        self.cell_parameters = cell_parameters or AllenGlif5Parameters()
        self._reservoir = AdaptiveGlifPopulation(self.config.reservoir_cells, self.cell_parameters)
        self._synaptic_current_pa = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        self._eligibility = np.zeros(self.config.reservoir_cells, dtype=np.float64)
        sparse_readout = self.config.readout_learning_rule == "sparse_three_factor"
        self._readout_synapses_pa = (
            None
            if sparse_readout
            else np.zeros(
                (self.token_cells, self.config.reservoir_cells), dtype=np.float64
            )
        )
        self._pacemaker_synapses_pa = np.zeros(self.token_cells, dtype=np.float64)
        feature_count = self.config.reservoir_cells + 1
        self._normal_matrix: np.ndarray | None = None
        self._target_moments: np.ndarray | None = None
        self._feature_batch: np.ndarray | None = None
        self._growth_feature_count = feature_count
        self._pending_growth_examples = 0
        self._growth_examples = 0
        self._local_updates = 0
        self._observed_tokens = 0
        self._learning_frozen = False

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
                reservoir_cells=self.config.reservoir_cells,
                fan_in=self.config.readout_fan_in,
                rng=rng,
                reservoir_eligibility=self._eligibility,
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
        self._eligibility_decay = np.exp(
            -self.cell_parameters.dt_ms / self.config.eligibility_tau_ms
        )
        self._compile_output_race()
        self._bind_native_kernel()

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
            eligibility_decay=self._eligibility_decay,
            afferent_targets=self._afferent_targets,
            recurrent_targets=self._recurrent_targets,
            recurrent_weights_pa=self._recurrent_weights_pa,
            synaptic_current_pa=self._synaptic_current_pa,
            eligibility=self._eligibility,
            host_mirror=(
                "eligibility"
                if self.config.readout_learning_rule == "sparse_three_factor"
                else "full"
            ),
        )
        self._native_local_learner = (
            NativeThreeFactorLearner(
                eligibility=self._eligibility,
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

    def capture_cell_synapse_state(self) -> RecurrentCellSynapseState:
        """Capture persistent anatomy and learning state owned by this circuit."""
        if not self.learning_frozen:
            raise RuntimeError("freeze recurrent readout learning before checkpointing")
        layout = _READOUT_STATE_LAYOUTS[self.config.readout_learning_rule]
        if layout.ridge_statistics:
            self._flush_growth_statistics()
            if self._normal_matrix is None or self._target_moments is None:
                raise RuntimeError("cannot checkpoint a recurrent circuit before growth")
        elif self._normal_matrix is not None or self._target_moments is not None:
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
        }
        arrays: dict[str, np.ndarray] = {}
        for attribute, key, _ in _BASE_STATE_ARRAYS + layout.direct_arrays:
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
                reservoir_cells=circuit.config.reservoir_cells,
                fan_in=circuit.config.readout_fan_in,
                reservoir_eligibility=circuit._eligibility,
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
        return float(np.count_nonzero(self._eligibility)) / self.config.reservoir_cells

    @property
    def observed_tokens(self) -> int:
        return self._observed_tokens

    @property
    def simulation_backend(self) -> str:
        return self._native_kernel.backend

    @property
    def simulation_device(self) -> str:
        return self._native_kernel.device_name

    @property
    def eligibility(self) -> np.ndarray:
        return self._eligibility.copy()

    def synchronize_structure_from_host(self) -> None:
        """Publish in-place afferent, recurrent-topology, and weight mutations."""
        self._native_kernel.synchronize_structure_from_host()

    def synchronize_state_from_device(self) -> None:
        """Make every mutable fitted-cell array available for host inspection."""
        self._native_kernel.synchronize_to_host()

    def reset_sequence(self) -> None:
        self._reservoir.reset()
        self._synaptic_current_pa.fill(0.0)
        self._eligibility.fill(0.0)
        self._native_kernel.synchronize_from_host()
        self._observed_tokens = 0

    def observe(self, token_id: int) -> RecurrentStep:
        """Apply one token as afferent current and advance the cell circuit."""
        token = int(token_id)
        if not 0 <= token < self.token_cells:
            raise ValueError("token id outside output-cell population")
        spike_total = self._native_kernel.observe(token)

        self._observed_tokens += 1
        step = RecurrentStep(
            token_id=token,
            reservoir_spikes=spike_total,
            simulated_steps=self._token_steps,
            biological_time_ms=self._token_steps * self.cell_parameters.dt_ms,
        )
        if self._observation_sink is not None:
            self._observation_sink(self, step)
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
            eligibility=tuple(float(value) for value in self._eligibility[indices]),
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
                trace_decay=np.exp(
                    -self.config.token_interval_ms / self.config.eligibility_tau_ms
                ),
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
        feature[:-1] = self._eligibility
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
        return (
            self.config.output_base_current_pa
            + self._pacemaker_synapses_pa
            + self._output_heterogeneity_pa
            + self._readout_synapses_pa @ self._eligibility
        )

    def race(
        self, allowed_token_ids: Sequence[int] | None = None
    ) -> RecurrentRace:
        """Race output cells under synaptic current; the first spike is the token."""
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
