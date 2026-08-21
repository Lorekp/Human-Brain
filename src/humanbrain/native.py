"""Build and bind the dependency-free native human-cell simulation kernel."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from humanbrain.cells import AdaptiveGlifPopulation

_SOURCE_ROOT = Path(__file__).with_name("_native_src")
_BUILD_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_PATH: Path | None = None

_F64_POINTER = ctypes.POINTER(ctypes.c_double)
_I32_POINTER = ctypes.POINTER(ctypes.c_int32)
_U32_POINTER = ctypes.POINTER(ctypes.c_uint32)
_U8_POINTER = ctypes.POINTER(ctypes.c_uint8)
_SIZE_POINTER = ctypes.POINTER(ctypes.c_size_t)
_U64_POINTER = ctypes.POINTER(ctypes.c_uint64)


class NativeBuildError(RuntimeError):
    """Raised when the required native cell kernel cannot be built or loaded."""


class _GlifConfig(ctypes.Structure):
    _fields_ = [
        ("cell_count", ctypes.c_size_t),
        ("after_channel_count", ctypes.c_size_t),
        ("spike_cut_steps", ctypes.c_int32),
        ("leak_ns", ctypes.c_double),
        ("voltage_step_scale", ctypes.c_double),
        ("threshold_beta_scale", ctypes.c_double),
        ("threshold_exp_decay", ctypes.c_double),
        ("threshold_voltage_factor", ctypes.c_double),
        ("threshold_beta_factor", ctypes.c_double),
        ("spike_threshold_decay", ctypes.c_double),
        ("threshold_inf_relative_mv", ctypes.c_double),
        ("voltage_reset_scale", ctypes.c_double),
        ("voltage_reset_offset_mv", ctypes.c_double),
        ("spike_cut_threshold_decay", ctypes.c_double),
        ("threshold_spike_increment_mv", ctypes.c_double),
    ]


class _RecurrentConfig(ctypes.Structure):
    _fields_ = [
        ("token_steps", ctypes.c_size_t),
        ("pulse_steps", ctypes.c_size_t),
        ("input_fanout", ctypes.c_size_t),
        ("recurrent_fanout", ctypes.c_size_t),
        ("input_current_pa", ctypes.c_double),
        ("recurrent_decay", ctypes.c_double),
        ("eligibility_decay", ctypes.c_double),
    ]


@dataclass(frozen=True)
class NativeBuildInfo:
    source_digest: str
    library_path: Path


def _source_digest() -> str:
    digest = hashlib.sha256()
    source_paths = (
        _SOURCE_ROOT / "Cargo.toml",
        _SOURCE_ROOT / "Cargo.lock",
        _SOURCE_ROOT / "src/lib.rs",
    )
    for path in source_paths:
        digest.update(path.read_bytes())
    digest.update(platform.system().encode("ascii"))
    digest.update(platform.machine().encode("ascii"))
    return digest.hexdigest()[:16]


def _library_filename() -> str:
    if os.name == "nt":
        return "humanbrain_kernel.dll"
    if platform.system() == "Darwin":
        return "libhumanbrain_kernel.dylib"
    return "libhumanbrain_kernel.so"


def _default_cache_root() -> Path:
    override = os.environ.get("HUMANBRAIN_NATIVE_CACHE")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "humanbrain-native"


def build_native(cache_root: str | Path | None = None, *, force: bool = False) -> NativeBuildInfo:
    """Build the Rust kernel into a content-addressed cache and return its path."""
    source_digest = _source_digest()
    root = Path(cache_root) if cache_root is not None else _default_cache_root()
    target_root = root / source_digest
    library_path = target_root / "release" / _library_filename()
    if library_path.is_file() and not force:
        return NativeBuildInfo(source_digest, library_path)

    target_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target_root)
    existing_flags = environment.get("RUSTFLAGS", "").strip()
    native_flag = "-C target-cpu=native"
    environment["RUSTFLAGS"] = f"{existing_flags} {native_flag}".strip()
    command = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        str(_SOURCE_ROOT / "Cargo.toml"),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as error:
        raise NativeBuildError("Rust Cargo is required to build the human-cell kernel") from error
    if completed.returncode != 0 or not library_path.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip() or "Cargo produced no library"
        raise NativeBuildError(f"native human-cell kernel build failed: {detail}")
    return NativeBuildInfo(source_digest, library_path)


def _load_library() -> ctypes.CDLL:
    global _LIBRARY, _LIBRARY_PATH
    if _LIBRARY is not None:
        return _LIBRARY
    with _BUILD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        build = build_native()
        try:
            library = ctypes.CDLL(str(build.library_path))
        except OSError as error:
            message = f"could not load native cell kernel: {build.library_path}"
            raise NativeBuildError(message) from error

        library.humanbrain_kernel_abi_version.argtypes = []
        library.humanbrain_kernel_abi_version.restype = ctypes.c_uint32
        if library.humanbrain_kernel_abi_version() != 3:
            raise NativeBuildError("native human-cell kernel ABI mismatch")

        pointer_args = [
            ctypes.POINTER(_GlifConfig),
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _U8_POINTER,
        ]
        library.humanbrain_glif_step.argtypes = pointer_args
        library.humanbrain_glif_step.restype = ctypes.c_size_t
        recurrent_args = [
            ctypes.POINTER(_GlifConfig),
            ctypes.POINTER(_RecurrentConfig),
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _U8_POINTER,
            _I32_POINTER,
            _U8_POINTER,
            _SIZE_POINTER,
            _U64_POINTER,
            _U64_POINTER,
            _U64_POINTER,
        ]
        for name in (
            "humanbrain_observe_token",
            "humanbrain_observe_token_serial",
            "humanbrain_observe_token_parallel",
        ):
            function = getattr(library, name)
            function.argtypes = recurrent_args
            function.restype = ctypes.c_size_t
        library.humanbrain_observe_token_reference_v1.argtypes = [
            *recurrent_args[:-8],
            _F64_POINTER,
            _I32_POINTER,
        ]
        library.humanbrain_observe_token_reference_v1.restype = ctypes.c_size_t
        library.humanbrain_three_factor_update.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
        ]
        library.humanbrain_three_factor_update.restype = ctypes.c_size_t
        library.humanbrain_sparse_readout_current.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            _F64_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
        ]
        library.humanbrain_sparse_readout_current.restype = None
        library.humanbrain_sparse_three_factor_update.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _U32_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            ctypes.c_double,
        ]
        library.humanbrain_sparse_three_factor_update.restype = ctypes.c_size_t
        library.humanbrain_sparse_structural_turnover.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            _I32_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _U32_POINTER,
            _F64_POINTER,
        ]
        library.humanbrain_sparse_structural_turnover.restype = ctypes.c_uint32
        _LIBRARY = library
        _LIBRARY_PATH = build.library_path
        return library


def loaded_build_info() -> NativeBuildInfo:
    """Return the source digest and loaded library path, building if necessary."""
    _load_library()
    assert _LIBRARY_PATH is not None
    return NativeBuildInfo(_source_digest(), _LIBRARY_PATH)


def _f64(array: np.ndarray) -> _F64_POINTER:
    return array.ctypes.data_as(_F64_POINTER)


def _i32(array: np.ndarray) -> _I32_POINTER:
    return array.ctypes.data_as(_I32_POINTER)


def _u32(array: np.ndarray) -> _U32_POINTER:
    return array.ctypes.data_as(_U32_POINTER)


def _require_native_array(
    array: np.ndarray,
    *,
    dtype: np.dtype[object],
    shape: tuple[int, ...],
    name: str,
) -> None:
    if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be a contiguous {dtype} array with shape {shape}")


def _finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_scalar(name: str, value: float) -> float:
    scalar = _finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _unit_interval(name: str, value: float, *, open_interval: bool = False) -> float:
    scalar = _finite_scalar(name, value)
    valid = 0.0 < scalar < 1.0 if open_interval else 0.0 <= scalar <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return scalar


def _glif_config(population: AdaptiveGlifPopulation) -> _GlifConfig:
    parameters = population.parameters
    return _GlifConfig(
        population.cell_count,
        population.state.after_spike_currents_pa.shape[1],
        parameters.spike_cut_steps,
        parameters.leak_ns,
        population._voltage_step_scale,
        population._threshold_beta_scale,
        population._threshold_exp_decay,
        population._threshold_voltage_factor,
        population._threshold_beta_factor,
        population._spike_threshold_decay,
        parameters.threshold_inf_relative_mv,
        parameters.voltage_reset_scale,
        parameters.voltage_reset_offset_mv,
        population._spike_cut_threshold_decay,
        parameters.threshold_spike_increment_mv,
    )


class NativeGlifKernel:
    """Native step binding over one population's checkpointable NumPy state."""

    def __init__(self, population: AdaptiveGlifPopulation) -> None:
        self._library = _load_library()
        self._population = population
        self._config = _glif_config(population)
        self._spike_mask = np.empty(population.cell_count, dtype=np.bool_)

    @property
    def spike_mask(self) -> np.ndarray:
        return self._spike_mask

    def step(self, current: np.ndarray) -> np.ndarray:
        population = self._population
        state = population.state
        self._library.humanbrain_glif_step(
            ctypes.byref(self._config),
            _f64(population._after_spike_decay),
            _f64(population._after_spike_cut_decay),
            _f64(population._after_spike_amplitudes_pa),
            _f64(current),
            _f64(state.voltage_relative_mv),
            _f64(state.after_spike_currents_pa),
            _f64(state.threshold_spike_mv),
            _f64(state.threshold_voltage_mv),
            _i32(state.refractory_steps),
            self._spike_mask.ctypes.data_as(_U8_POINTER),
        )
        return self._spike_mask


class NativeRecurrentKernel:
    """One-call native token simulation over sparse recurrent connectivity."""

    backend = "cpu"

    def __init__(
        self,
        population: AdaptiveGlifPopulation,
        *,
        token_steps: int,
        pulse_steps: int,
        input_fanout: int,
        recurrent_fanout: int,
        input_current_pa: float,
        recurrent_decay: float,
        eligibility_decay: float,
        afferent_targets: np.ndarray,
        recurrent_targets: np.ndarray,
        recurrent_weights_pa: np.ndarray,
        synaptic_current_pa: np.ndarray,
        eligibility: np.ndarray,
    ) -> None:
        self._library = _load_library()
        self._population = population
        cells = population.cell_count
        expected_afferents = (afferent_targets.shape[0], input_fanout)
        _require_native_array(
            afferent_targets,
            dtype=np.dtype(np.int32),
            shape=expected_afferents,
            name="afferent targets",
        )
        _require_native_array(
            recurrent_targets,
            dtype=np.dtype(np.int32),
            shape=(cells, recurrent_fanout),
            name="recurrent targets",
        )
        _require_native_array(
            recurrent_weights_pa,
            dtype=np.dtype(np.float64),
            shape=(cells, recurrent_fanout),
            name="recurrent weights",
        )
        for array, name in (
            (synaptic_current_pa, "synaptic current"),
            (eligibility, "eligibility"),
        ):
            _require_native_array(
                array,
                dtype=np.dtype(np.float64),
                shape=(cells,),
                name=name,
            )
        if np.any(afferent_targets < 0) or np.any(afferent_targets >= cells):
            raise ValueError("afferent targets must index reservoir cells")
        if np.any(recurrent_targets < 0) or np.any(recurrent_targets >= cells):
            raise ValueError("recurrent targets must index reservoir cells")
        self._glif_config = _glif_config(population)
        self._recurrent_config = _RecurrentConfig(
            token_steps,
            pulse_steps,
            input_fanout,
            recurrent_fanout,
            input_current_pa,
            recurrent_decay,
            eligibility_decay,
        )
        self._afferent_targets = afferent_targets
        self._recurrent_targets = recurrent_targets
        self._recurrent_weights_pa = recurrent_weights_pa
        self._synaptic_current_pa = synaptic_current_pa
        self._eligibility = eligibility
        self._input_scratch = np.empty(input_fanout, dtype=np.float64)
        self._spike_mask = population._native_kernel.spike_mask
        self._active_indices = np.empty(cells, dtype=np.int32)
        self._active_mask = np.empty(cells, dtype=np.bool_)
        self._active_count = ctypes.c_size_t()
        self._scan_nanoseconds = ctypes.c_uint64()
        self._frontier_build_nanoseconds = ctypes.c_uint64()
        self._step_nanoseconds = ctypes.c_uint64()
        self._reference_step_current: np.ndarray | None = None
        self._reference_spike_indices: np.ndarray | None = None

    @property
    def device_name(self) -> str:
        return platform.processor() or platform.machine()

    def synchronize_from_host(self) -> None:
        """CPU arrays are the live native state, so no synchronization is needed."""

    def synchronize_structure_from_host(self) -> None:
        """CPU topology and weight arrays are already live native state."""

    def observe(self, token_id: int) -> int:
        return self._observe_with(self._library.humanbrain_observe_token, token_id)

    def _observe_serial(self, token_id: int) -> int:
        return self._observe_with(self._library.humanbrain_observe_token_serial, token_id)

    def _observe_parallel(self, token_id: int) -> int:
        return self._observe_with(self._library.humanbrain_observe_token_parallel, token_id)

    @property
    def last_active_cells(self) -> int:
        return int(self._active_count.value)

    @property
    def last_frontier_profile_ns(self) -> tuple[int, int, int]:
        return (
            int(self._scan_nanoseconds.value),
            int(self._frontier_build_nanoseconds.value),
            int(self._step_nanoseconds.value),
        )

    def _observe_with(self, function: object, token_id: int) -> int:
        population = self._population
        state = population.state
        afferents = self._afferent_targets[token_id]
        return int(
            function(
                ctypes.byref(self._glif_config),
                ctypes.byref(self._recurrent_config),
                _f64(population._after_spike_decay),
                _f64(population._after_spike_cut_decay),
                _f64(population._after_spike_amplitudes_pa),
                _i32(afferents),
                _i32(self._recurrent_targets),
                _f64(self._recurrent_weights_pa),
                _f64(state.voltage_relative_mv),
                _f64(state.after_spike_currents_pa),
                _f64(state.threshold_spike_mv),
                _f64(state.threshold_voltage_mv),
                _i32(state.refractory_steps),
                _f64(self._synaptic_current_pa),
                _f64(self._eligibility),
                _f64(self._input_scratch),
                self._spike_mask.ctypes.data_as(_U8_POINTER),
                _i32(self._active_indices),
                self._active_mask.ctypes.data_as(_U8_POINTER),
                ctypes.byref(self._active_count),
                ctypes.byref(self._scan_nanoseconds),
                ctypes.byref(self._frontier_build_nanoseconds),
                ctypes.byref(self._step_nanoseconds),
            )
        )

    def _observe_reference_v1(self, token_id: int) -> int:
        population = self._population
        state = population.state
        afferents = self._afferent_targets[token_id]
        cells = population.cell_count
        if self._reference_step_current is None:
            self._reference_step_current = np.empty(cells, dtype=np.float64)
            self._reference_spike_indices = np.empty(cells, dtype=np.int32)
        assert self._reference_spike_indices is not None
        return int(
            self._library.humanbrain_observe_token_reference_v1(
                ctypes.byref(self._glif_config),
                ctypes.byref(self._recurrent_config),
                _f64(population._after_spike_decay),
                _f64(population._after_spike_cut_decay),
                _f64(population._after_spike_amplitudes_pa),
                _i32(afferents),
                _i32(self._recurrent_targets),
                _f64(self._recurrent_weights_pa),
                _f64(state.voltage_relative_mv),
                _f64(state.after_spike_currents_pa),
                _f64(state.threshold_spike_mv),
                _f64(state.threshold_voltage_mv),
                _i32(state.refractory_steps),
                _f64(self._synaptic_current_pa),
                _f64(self._eligibility),
                _f64(self._reference_step_current),
                _i32(self._reference_spike_indices),
            )
        )


class NativeThreeFactorLearner:
    """Allocation-free native plasticity over checkpointable synapse arrays."""

    def __init__(
        self,
        *,
        eligibility: np.ndarray,
        output_heterogeneity_pa: np.ndarray,
        readout_synapses_pa: np.ndarray,
        pacemaker_synapses_pa: np.ndarray,
    ) -> None:
        self._library = _load_library()
        output_cells, reservoir_cells = readout_synapses_pa.shape
        _require_native_array(
            eligibility,
            dtype=np.dtype(np.float64),
            shape=(reservoir_cells,),
            name="eligibility",
        )
        _require_native_array(
            output_heterogeneity_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="output heterogeneity",
        )
        _require_native_array(
            readout_synapses_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells, reservoir_cells),
            name="readout synapses",
        )
        _require_native_array(
            pacemaker_synapses_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="pacemaker synapses",
        )
        self._eligibility = eligibility
        self._output_heterogeneity_pa = output_heterogeneity_pa
        self._readout_synapses_pa = readout_synapses_pa
        self._pacemaker_synapses_pa = pacemaker_synapses_pa
        self._output_cells = output_cells
        self._reservoir_cells = reservoir_cells

    def update(
        self,
        target_token_id: int,
        learning_rate_pa: float,
        competitive_margin_pa: float,
    ) -> int | None:
        """Update a margin-violating competitor, or return ``None`` when satisfied."""
        target = int(target_token_id)
        if not 0 <= target < self._output_cells:
            raise ValueError("target token outside output-cell population")
        if learning_rate_pa <= 0.0:
            raise ValueError("local learning rate must be positive")
        if competitive_margin_pa <= 0.0:
            raise ValueError("competitive margin must be positive")
        winner = int(
            self._library.humanbrain_three_factor_update(
                self._reservoir_cells,
                self._output_cells,
                target,
                learning_rate_pa,
                competitive_margin_pa,
                _f64(self._eligibility),
                _f64(self._output_heterogeneity_pa),
                _f64(self._readout_synapses_pa),
                _f64(self._pacemaker_synapses_pa),
            )
        )
        if winner == self._output_cells:
            return None
        if winner > self._output_cells:
            raise RuntimeError("native local learner rejected its array dimensions")
        return winner


class NativeSparseCellReadout:
    """Native current summation over explicit bounded output-cell contacts."""

    def __init__(
        self,
        *,
        eligibility: np.ndarray,
        output_heterogeneity_pa: np.ndarray,
        potential_sources: np.ndarray,
        apposition_eligibility: np.ndarray,
        contact_sources: np.ndarray,
        contact_weights_pa: np.ndarray,
        contact_ages: np.ndarray,
        pacemaker_synapses_pa: np.ndarray,
        output_activity_ema: np.ndarray,
    ) -> None:
        self._library = _load_library()
        output_cells, fan_in = contact_sources.shape
        if potential_sources.shape[0] != output_cells:
            raise ValueError("potential appositions need one row per output cell")
        candidate_count = potential_sources.shape[1]
        if apposition_eligibility.shape != (output_cells, candidate_count):
            raise ValueError("potential sources and apposition traces must have equal shape")
        if contact_weights_pa.shape != (output_cells, fan_in):
            raise ValueError("sparse contact sources and weights must have equal shape")
        reservoir_cells = eligibility.shape[0]
        _require_native_array(
            eligibility,
            dtype=np.dtype(np.float64),
            shape=(reservoir_cells,),
            name="eligibility",
        )
        _require_native_array(
            output_heterogeneity_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="output heterogeneity",
        )
        _require_native_array(
            potential_sources,
            dtype=np.dtype(np.int32),
            shape=(output_cells, candidate_count),
            name="potential apposition sources",
        )
        _require_native_array(
            apposition_eligibility,
            dtype=np.dtype(np.float64),
            shape=(output_cells, candidate_count),
            name="apposition eligibility",
        )
        _require_native_array(
            contact_sources,
            dtype=np.dtype(np.int32),
            shape=(output_cells, fan_in),
            name="sparse contact sources",
        )
        _require_native_array(
            contact_weights_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells, fan_in),
            name="sparse contact weights",
        )
        _require_native_array(
            contact_ages,
            dtype=np.dtype(np.uint32),
            shape=(output_cells, fan_in),
            name="sparse contact ages",
        )
        _require_native_array(
            output_activity_ema,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="output activity",
        )
        _require_native_array(
            pacemaker_synapses_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="pacemaker synapses",
        )
        if np.any(potential_sources < 0) or np.any(potential_sources >= reservoir_cells):
            raise ValueError("potential appositions must index reservoir cells")
        if np.any(contact_sources < 0) or np.any(contact_sources >= reservoir_cells):
            raise ValueError("sparse contacts must index reservoir cells")
        self._eligibility = eligibility
        self._output_heterogeneity_pa = output_heterogeneity_pa
        self._potential_sources = potential_sources
        self._apposition_eligibility = apposition_eligibility
        self._contact_sources = contact_sources
        self._contact_weights_pa = contact_weights_pa
        self._contact_ages = contact_ages
        self._pacemaker_synapses_pa = pacemaker_synapses_pa
        self._output_activity_ema = output_activity_ema
        self._current_pa = np.empty(output_cells, dtype=np.float64)
        self._output_cells = output_cells
        self._reservoir_cells = reservoir_cells
        self._fan_in = fan_in
        self._candidate_count = candidate_count

    def current(self, base_current_pa: float) -> np.ndarray:
        base_current_pa = _finite_scalar("base current", base_current_pa)
        self._library.humanbrain_sparse_readout_current(
            self._reservoir_cells,
            self._output_cells,
            self._fan_in,
            base_current_pa,
            _f64(self._eligibility),
            _f64(self._output_heterogeneity_pa),
            _i32(self._contact_sources),
            _f64(self._contact_weights_pa),
            _f64(self._pacemaker_synapses_pa),
            _f64(self._current_pa),
        )
        return self._current_pa.copy()

    def update(
        self,
        *,
        target: int,
        winner: int,
        learning_rate_pa: float,
        maximum_weight_pa: float,
        trace_decay: float,
        target_post_gate: float,
        winner_post_gate: float,
        activity_decay: float,
    ) -> int | None:
        target = int(target)
        winner = int(winner)
        if not 0 <= target < self._output_cells:
            raise ValueError("target token outside output-cell population")
        if not 0 <= winner < self._output_cells:
            raise ValueError("winner outside output-cell population")
        learning_rate_pa = _positive_scalar("learning rate", learning_rate_pa)
        maximum_weight_pa = _positive_scalar("maximum weight", maximum_weight_pa)
        trace_decay = _unit_interval("trace decay", trace_decay)
        target_post_gate = _unit_interval("target post gate", target_post_gate)
        winner_post_gate = _unit_interval("winner post gate", winner_post_gate)
        activity_decay = _unit_interval(
            "activity decay", activity_decay, open_interval=True
        )
        result = int(
            self._library.humanbrain_sparse_three_factor_update(
                self._reservoir_cells,
                self._output_cells,
                self._candidate_count,
                self._fan_in,
                target,
                winner,
                learning_rate_pa,
                maximum_weight_pa,
                trace_decay,
                target_post_gate,
                winner_post_gate,
                _f64(self._eligibility),
                _i32(self._potential_sources),
                _f64(self._apposition_eligibility),
                _i32(self._contact_sources),
                _f64(self._contact_weights_pa),
                _u32(self._contact_ages),
                _f64(self._pacemaker_synapses_pa),
                _f64(self._output_activity_ema),
                activity_decay,
            )
        )
        if result == self._output_cells:
            return None
        if result > self._output_cells:
            raise RuntimeError("native sparse learner rejected its dimensions")
        return result

    def turnover(
        self,
        *,
        output: int,
        structural_event: int,
        random_seed: int,
        minimum_age: int,
        turnover_probability: float,
        learning_rate_pa: float,
        maximum_weight_pa: float,
        homeostatic_rate: float,
    ) -> tuple[bool, bool]:
        output = int(output)
        structural_event = int(structural_event)
        minimum_age = int(minimum_age)
        if not 0 <= output < self._output_cells:
            raise ValueError("turnover output outside output-cell population")
        if structural_event < 0:
            raise ValueError("structural event must be nonnegative")
        if minimum_age < 0 or minimum_age > np.iinfo(np.uint32).max:
            raise ValueError("minimum age must fit uint32")
        turnover_probability = _unit_interval(
            "turnover probability", turnover_probability
        )
        learning_rate_pa = _positive_scalar("learning rate", learning_rate_pa)
        maximum_weight_pa = _positive_scalar("maximum weight", maximum_weight_pa)
        homeostatic_rate = _finite_scalar("homeostatic rate", homeostatic_rate)
        if homeostatic_rate < 0.0:
            raise ValueError("homeostatic rate must be nonnegative")
        result = int(
            self._library.humanbrain_sparse_structural_turnover(
                self._output_cells,
                self._candidate_count,
                self._fan_in,
                output,
                structural_event,
                int(random_seed) & ((1 << 64) - 1),
                minimum_age,
                turnover_probability,
                learning_rate_pa,
                maximum_weight_pa,
                homeostatic_rate,
                _i32(self._potential_sources),
                _f64(self._apposition_eligibility),
                _i32(self._contact_sources),
                _f64(self._contact_weights_pa),
                _u32(self._contact_ages),
                _f64(self._output_activity_ema),
            )
        )
        if result > 3:
            raise RuntimeError("native sparse turnover rejected its dimensions")
        return bool(result & 1), bool(result & 2)


def recurrent_kernel(*args: object, **kwargs: object) -> NativeRecurrentKernel:
    """Create the fastest exact backend suitable for this circuit size."""
    population = args[0] if args else kwargs.get("population")
    if population is None:
        raise TypeError("recurrent kernel requires a cell population")
    from humanbrain.cuda import CudaRecurrentKernel, should_use_cuda

    if should_use_cuda(population.cell_count):
        return CudaRecurrentKernel(*args, **kwargs)  # type: ignore[return-value]
    return NativeRecurrentKernel(*args, **kwargs)
