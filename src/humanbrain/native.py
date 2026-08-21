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
_U8_POINTER = ctypes.POINTER(ctypes.c_uint8)


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
        if library.humanbrain_kernel_abi_version() != 2:
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
            _I32_POINTER,
            _U8_POINTER,
        ]
        library.humanbrain_glif_step.argtypes = pointer_args
        library.humanbrain_glif_step.restype = ctypes.c_size_t
        library.humanbrain_observe_token.argtypes = [
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
            _I32_POINTER,
        ]
        library.humanbrain_observe_token.restype = ctypes.c_size_t
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


def _require_native_array(
    array: np.ndarray,
    *,
    dtype: np.dtype[object],
    shape: tuple[int, ...],
    name: str,
) -> None:
    if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be a contiguous {dtype} array with shape {shape}")


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
        self._spike_indices = np.empty(population.cell_count, dtype=np.int32)

    @property
    def spike_indices(self) -> np.ndarray:
        return self._spike_indices

    def step(self, current: np.ndarray) -> np.ndarray:
        population = self._population
        state = population.state
        spikes = np.empty(population.cell_count, dtype=np.bool_)
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
            _i32(self._spike_indices),
            spikes.ctypes.data_as(_U8_POINTER),
        )
        return spikes


class NativeRecurrentKernel:
    """One-call native token simulation over sparse recurrent connectivity."""

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
        step_current_pa: np.ndarray,
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
            (step_current_pa, "step current"),
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
        self._step_current_pa = step_current_pa
        self._spike_indices = population._native_kernel.spike_indices

    def observe(self, token_id: int) -> int:
        population = self._population
        state = population.state
        afferents = self._afferent_targets[token_id]
        return int(
            self._library.humanbrain_observe_token(
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
                _f64(self._step_current_pa),
                _i32(self._spike_indices),
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
