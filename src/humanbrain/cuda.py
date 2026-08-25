"""Persistent-device CUDA execution for large recurrent human-cell circuits."""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from humanbrain.native import NativeBuildError, _glif_config, _RecurrentConfig

if TYPE_CHECKING:
    from humanbrain.cells import AdaptiveGlifPopulation


def async_pipeline_enabled() -> bool:
    """Read the double-buffered H2D/D2H overlap switch (default: on)."""
    value = os.environ.get("HUMANBRAIN_CUDA_ASYNC", "1").strip().lower()
    return value not in {"0", "false", "off"}

_SOURCE = Path(__file__).with_name("_native_src") / "src" / "cuda.cu"
_CUDA_MINIMUM_CELLS = 100_000
_BUILD_LOCK = threading.Lock()


@dataclass(frozen=True)
class CudaBuildInfo:
    source_digest: str
    cubin_path: Path
    architecture: str


def _cuda_bin() -> Path | None:
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate = Path(cuda_path) / "bin"
        if candidate.is_dir():
            return candidate
    if os.name == "nt":
        roots = sorted(
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*/bin"),
            reverse=True,
        )
        return roots[0] if roots else None
    return None


def _load_system_library(windows_pattern: str, unix_name: str) -> ctypes.CDLL:
    if os.name == "nt":
        cuda_bin = _cuda_bin()
        candidates = sorted(cuda_bin.glob(windows_pattern), reverse=True) if cuda_bin else []
        if not candidates:
            raise NativeBuildError(f"CUDA library {windows_pattern} was not found")
        return ctypes.CDLL(str(candidates[0]))
    try:
        return ctypes.CDLL(unix_name)
    except OSError as error:
        raise NativeBuildError(f"CUDA library {unix_name} was not found") from error


class _CudaDriver:
    def __init__(self) -> None:
        self.library = (
            _load_system_library("nvcuda.dll", "libcuda.so.1")
            if os.name != "nt"
            else ctypes.WinDLL("nvcuda.dll")
        )
        library = self.library
        library.cuInit.argtypes = [ctypes.c_uint]
        library.cuInit.restype = ctypes.c_int
        library.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        library.cuDeviceGet.restype = ctypes.c_int
        library.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        library.cuDeviceGetName.restype = ctypes.c_int
        library.cuDeviceComputeCapability.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        library.cuDeviceComputeCapability.restype = ctypes.c_int
        library.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.cuDeviceGetAttribute.restype = ctypes.c_int
        library.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        library.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        library.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        library.cuCtxSetCurrent.restype = ctypes.c_int
        library.cuCtxSynchronize.argtypes = []
        library.cuCtxSynchronize.restype = ctypes.c_int
        library.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        library.cuModuleLoadData.restype = ctypes.c_int
        library.cuModuleUnload.argtypes = [ctypes.c_void_p]
        library.cuModuleUnload.restype = ctypes.c_int
        library.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        library.cuModuleGetFunction.restype = ctypes.c_int
        library.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        library.cuMemAlloc_v2.restype = ctypes.c_int
        library.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        library.cuMemFree_v2.restype = ctypes.c_int
        library.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        library.cuMemcpyHtoD_v2.restype = ctypes.c_int
        library.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        library.cuMemcpyDtoH_v2.restype = ctypes.c_int
        library.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        library.cuOccupancyMaxActiveBlocksPerMultiprocessor.restype = ctypes.c_int
        library.cuLaunchCooperativeKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.cuLaunchCooperativeKernel.restype = ctypes.c_int
        library.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        library.cuEventCreate.restype = ctypes.c_int
        library.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.cuEventRecord.restype = ctypes.c_int
        library.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        library.cuEventSynchronize.restype = ctypes.c_int
        library.cuEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.cuEventElapsedTime.restype = ctypes.c_int
        library.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
        library.cuEventDestroy_v2.restype = ctypes.c_int
        library.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        library.cuGetErrorString.restype = ctypes.c_int
        library.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        library.cuStreamCreate.restype = ctypes.c_int
        library.cuStreamDestroy.argtypes = [ctypes.c_void_p]
        library.cuStreamDestroy.restype = ctypes.c_int
        library.cuStreamWaitEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        library.cuStreamWaitEvent.restype = ctypes.c_int
        library.cuMemHostAlloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        library.cuMemHostAlloc.restype = ctypes.c_int
        library.cuMemFreeHost.argtypes = [ctypes.c_void_p]
        library.cuMemFreeHost.restype = ctypes.c_int
        library.cuMemcpyDtoHAsync_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        library.cuMemcpyDtoHAsync_v2.restype = ctypes.c_int
        library.cuMemcpyDtoDAsync_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        library.cuMemcpyDtoDAsync_v2.restype = ctypes.c_int

        self.check(library.cuInit(0), "initialize CUDA")
        device = ctypes.c_int()
        self.check(library.cuDeviceGet(ctypes.byref(device), 0), "select CUDA device")
        self.device = int(device.value)
        context = ctypes.c_void_p()
        self.check(
            library.cuDevicePrimaryCtxRetain(ctypes.byref(context), self.device),
            "retain CUDA primary context",
        )
        self.context = context
        self.make_current()

    def check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        detail = ctypes.c_char_p()
        self.library.cuGetErrorString(status, ctypes.byref(detail))
        message = (
            detail.value.decode("utf-8", errors="replace") if detail.value else f"status {status}"
        )
        raise NativeBuildError(f"{operation}: {message}")

    def make_current(self) -> None:
        self.check(self.library.cuCtxSetCurrent(self.context), "activate CUDA context")

    def attribute(self, attribute: int) -> int:
        value = ctypes.c_int()
        self.check(
            self.library.cuDeviceGetAttribute(ctypes.byref(value), attribute, self.device),
            "query CUDA device attribute",
        )
        return int(value.value)

    @property
    def architecture(self) -> str:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self.check(
            self.library.cuDeviceComputeCapability(
                ctypes.byref(major), ctypes.byref(minor), self.device
            ),
            "query CUDA compute capability",
        )
        return f"sm_{major.value}{minor.value}"

    @property
    def name(self) -> str:
        buffer = ctypes.create_string_buffer(256)
        self.check(
            self.library.cuDeviceGetName(buffer, len(buffer), self.device),
            "query CUDA device name",
        )
        return buffer.value.decode("utf-8", errors="replace")


_DRIVER: _CudaDriver | None = None


def _driver() -> _CudaDriver:
    global _DRIVER
    if _DRIVER is None:
        with _BUILD_LOCK:
            if _DRIVER is None:
                _DRIVER = _CudaDriver()
    return _DRIVER


class _Nvrtc:
    SUCCESS = 0

    def __init__(self) -> None:
        self.library = _load_system_library("nvrtc64_*.dll", "libnvrtc.so")
        library = self.library
        library.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.nvrtcCreateProgram.restype = ctypes.c_int
        library.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        library.nvrtcCompileProgram.restype = ctypes.c_int
        library.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        library.nvrtcGetProgramLogSize.restype = ctypes.c_int
        library.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        library.nvrtcGetProgramLog.restype = ctypes.c_int
        library.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        library.nvrtcGetCUBINSize.restype = ctypes.c_int
        library.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.nvrtcGetCUBIN.restype = ctypes.c_int
        library.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.nvrtcDestroyProgram.restype = ctypes.c_int

    def compile(self, source: bytes, architecture: str) -> bytes:
        program = ctypes.c_void_p()
        status = self.library.nvrtcCreateProgram(
            ctypes.byref(program), source, b"humanbrain_cuda.cu", 0, None, None
        )
        if status != self.SUCCESS:
            raise NativeBuildError(f"NVRTC program creation failed with status {status}")
        cuda_bin = _cuda_bin()
        include = cuda_bin.parent / "include" if cuda_bin is not None else None
        option_values = [
            f"--gpu-architecture={architecture}",
            "--std=c++17",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ]
        if include is not None:
            option_values.append(f"--include-path={include}")
        encoded = [value.encode("utf-8") for value in option_values]
        options = (ctypes.c_char_p * len(encoded))(*encoded)
        try:
            status = self.library.nvrtcCompileProgram(program, len(encoded), options)
            if status != self.SUCCESS:
                log_size = ctypes.c_size_t()
                self.library.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
                log = ctypes.create_string_buffer(log_size.value or 1)
                self.library.nvrtcGetProgramLog(program, log)
                raise NativeBuildError(
                    "CUDA kernel compilation failed: " + log.value.decode("utf-8", errors="replace")
                )
            cubin_size = ctypes.c_size_t()
            if self.library.nvrtcGetCUBINSize(program, ctypes.byref(cubin_size)) != self.SUCCESS:
                raise NativeBuildError("NVRTC could not size compiled CUDA device code")
            cubin = ctypes.create_string_buffer(cubin_size.value)
            if self.library.nvrtcGetCUBIN(program, cubin) != self.SUCCESS:
                raise NativeBuildError("NVRTC could not return compiled CUDA device code")
            return cubin.raw
        finally:
            self.library.nvrtcDestroyProgram(ctypes.byref(program))


def _cache_root() -> Path:
    override = os.environ.get("HUMANBRAIN_NATIVE_CACHE")
    if override:
        return Path(override) / "cuda"
    return Path(tempfile.gettempdir()) / "humanbrain-native" / "cuda"


def build_cuda(cache_root: str | Path | None = None, *, force: bool = False) -> CudaBuildInfo:
    """Compile the CUDA cell kernel with NVRTC into a device-specific cache."""
    driver = _driver()
    architecture = driver.architecture
    source = _SOURCE.read_bytes()
    digest = hashlib.sha256(source + architecture.encode("ascii")).hexdigest()[:16]
    root = Path(cache_root) if cache_root is not None else _cache_root()
    cubin_path = root / digest / f"humanbrain_{architecture}.cubin"
    if cubin_path.is_file() and not force:
        return CudaBuildInfo(digest, cubin_path, architecture)
    cubin = _Nvrtc().compile(source, architecture)
    cubin_path.parent.mkdir(parents=True, exist_ok=True)
    cubin_path.write_bytes(cubin)
    return CudaBuildInfo(digest, cubin_path, architecture)


def cuda_device_name() -> str:
    driver = _driver()
    if driver.attribute(95) == 0:  # CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
        raise NativeBuildError("CUDA device does not support cooperative launch")
    return f"{driver.name} (compute {driver.architecture.removeprefix('sm_')})"


def cuda_available() -> bool:
    try:
        cuda_device_name()
        build_cuda()
    except (NativeBuildError, OSError):
        return False
    return True


def should_use_cuda(cell_count: int) -> bool:
    """Select the large-network backend using an auto/cpu/cuda device policy."""
    device = os.environ.get("HUMANBRAIN_DEVICE", "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("HUMANBRAIN_DEVICE must be auto, cpu, or cuda")
    if device == "cpu":
        return False
    if device == "cuda":
        if not cuda_available():
            raise NativeBuildError("HUMANBRAIN_DEVICE=cuda but no CUDA backend is available")
        return True
    return cell_count >= _CUDA_MINIMUM_CELLS and cuda_available()


class CudaRecurrentKernel:
    """One cooperative GPU launch advances a complete sparse token interval."""

    backend = "cuda"
    _BLOCK_SIZE = 256

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
        host_mirror: str = "full",
    ) -> None:
        if host_mirror not in {"full", "eligibility"}:
            raise ValueError("CUDA host mirror must be 'full' or 'eligibility'")
        self._driver = _driver()
        self._driver.make_current()
        self._population = population
        self._synaptic_current_pa = synaptic_current_pa
        self._eligibility = eligibility
        self._host_mirror = host_mirror
        self._afferent_targets = afferent_targets
        self._recurrent_targets = recurrent_targets
        self._recurrent_weights_pa = recurrent_weights_pa
        self._glif = _glif_config(population)
        self._recurrent = _RecurrentConfig(
            token_steps,
            pulse_steps,
            input_fanout,
            recurrent_fanout,
            input_current_pa,
            recurrent_decay,
            eligibility_decay,
        )
        self._token_cells = int(afferent_targets.shape[0])
        self._input_fanout = input_fanout
        self._recurrent_fanout = recurrent_fanout
        self._allocations: list[int] = []
        self._arrays: dict[str, tuple[int, int]] = {}
        self._module = ctypes.c_void_p()
        self._function = ctypes.c_void_p()
        self._start_event = ctypes.c_void_p()
        self._stop_event = ctypes.c_void_p()
        self.last_kernel_milliseconds = 0.0

        cells = population.cell_count
        build = build_cuda()
        cubin = build.cubin_path.read_bytes()
        cubin_buffer = ctypes.create_string_buffer(cubin)
        library = self._driver.library
        self._driver.check(
            library.cuModuleLoadData(
                ctypes.byref(self._module), ctypes.cast(cubin_buffer, ctypes.c_void_p)
            ),
            "load CUDA cell module",
        )
        self._driver.check(
            library.cuModuleGetFunction(
                ctypes.byref(self._function), self._module, b"recurrent_token_kernel"
            ),
            "resolve CUDA token kernel",
        )
        active_blocks = ctypes.c_int()
        self._driver.check(
            library.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(active_blocks), self._function, self._BLOCK_SIZE, 0
            ),
            "compute CUDA cooperative occupancy",
        )
        multiprocessors = self._driver.attribute(16)
        self._block_count = int(active_blocks.value) * multiprocessors
        if self._block_count <= 0:
            raise NativeBuildError("CUDA cooperative kernel has zero active blocks")
        self._driver.check(
            library.cuEventCreate(ctypes.byref(self._start_event), 0), "create CUDA event"
        )
        self._driver.check(
            library.cuEventCreate(ctypes.byref(self._stop_event), 0), "create CUDA event"
        )

        state = population.state
        constants = {
            "after_decay": population.constants.after_spike_decay,
            "after_cut_decay": population.constants.after_spike_cut_decay,
            "after_amplitudes": population.constants.after_spike_amplitudes_pa,
            **self._structure_arrays_from_host(),
        }
        mutables = {
            "voltage": state.voltage_relative_mv,
            "after": state.after_spike_currents_pa,
            "threshold_spike": state.threshold_spike_mv,
            "threshold_voltage": state.threshold_voltage_mv,
            "refractory": state.refractory_steps,
            "synaptic_current": synaptic_current_pa,
            "eligibility": eligibility,
        }
        try:
            for name, array in constants.items():
                self._allocate_array(name, np.ascontiguousarray(array), upload=True)
            for name, array in mutables.items():
                self._allocate_array(name, array, upload=True)
            self._allocate_bytes("spike_mask", cells)
            self._allocate_bytes("input_mask", cells)
            self._allocate_bytes("total_spikes", ctypes.sizeof(ctypes.c_uint64))
            self._allocate_array("decay_active", self._decayed_activity_words(), upload=True)
            self._allocate_array(
                "threshold_spike_active", self._threshold_spike_active_words(), upload=True
            )
            self._allocate_array(
                "refractory_active", self._refractory_active_words(), upload=True
            )
            self._build_launch_arguments()
            if async_pipeline_enabled():
                self._build_pipeline()
        except Exception:
            self.close()
            raise

    def _build_launch_arguments(self) -> None:
        """Pre-marshal the cooperative-launch argument block once.

        Only two entries change per token (the afferent row pointer and the
        spike-count destination); cuLaunchCooperativeKernel copies parameter
        values at launch time, so patching these ctypes cells between launches
        removes per-token argument marshaling from the critical path.
        """
        values: list[object] = [
            self._glif,
            self._recurrent,
            ctypes.c_uint64(self._arrays["after_decay"][0]),
            ctypes.c_uint64(self._arrays["after_cut_decay"][0]),
            ctypes.c_uint64(self._arrays["after_amplitudes"][0]),
            ctypes.c_uint64(0),
            ctypes.c_uint64(self._arrays["incoming_offsets"][0]),
            ctypes.c_uint64(self._arrays["incoming_edges"][0]),
            ctypes.c_uint64(self._arrays["recurrent_weights"][0]),
            ctypes.c_uint64(self._arrays["voltage"][0]),
            ctypes.c_uint64(self._arrays["after"][0]),
            ctypes.c_uint64(self._arrays["threshold_spike"][0]),
            ctypes.c_uint64(self._arrays["threshold_voltage"][0]),
            ctypes.c_uint64(self._arrays["refractory"][0]),
            ctypes.c_uint64(self._arrays["synaptic_current"][0]),
            ctypes.c_uint64(self._arrays["eligibility"][0]),
            ctypes.c_uint64(self._arrays["spike_mask"][0]),
            ctypes.c_uint64(self._arrays["input_mask"][0]),
            ctypes.c_uint64(self._arrays["total_spikes"][0]),
            ctypes.c_uint64(self._arrays["decay_active"][0]),
            ctypes.c_uint64(self._arrays["threshold_spike_active"][0]),
            ctypes.c_uint64(self._arrays["refractory_active"][0]),
        ]
        self._launch_values = values
        self._arg_afferent_pointer = values[5]
        self._arg_spikes_pointer = values[18]
        self._launch_arguments = (ctypes.c_void_p * len(values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values)
        )
        self._afferent_base_pointer, _ = self._arrays["afferents"]
        self._afferent_row_bytes = (
            self._input_fanout * np.dtype(np.int32).itemsize
        )

    # ------------------------------------------------------------------
    # Double-buffered host-device overlap pipeline.
    #
    # The cooperative kernel reads and writes one primary eligibility
    # buffer in place, so a naive D2H snapshot would race with the next
    # launch. Each pipeline slot therefore owns a device-side eligibility
    # snapshot: the compute stream runs [kernel -> D2D primary->snapshot]
    # back to back, so the snapshot stays stable while any later kernel
    # overwrites the primary buffer. A non-blocking copy stream then drains
    # snapshots (and per-slot spike counters) into a pinned host staging
    # ring, overlapping the next token's kernel execution. collect() waits
    # the slot's copy event and publishes the bit-exact snapshot into the
    # host eligibility mirror, so learning sees exactly the bytes that the
    # synchronous path would have downloaded.
    # ------------------------------------------------------------------

    _PIPELINE_DEPTH = 4
    _CU_STREAM_NON_BLOCKING = 0x1
    _CU_MEMHOSTALLOC_PORTABLE = 0x1

    def _build_pipeline(self) -> None:
        library = self._driver.library
        self._compute_stream = ctypes.c_void_p()
        self._copy_stream = ctypes.c_void_p()
        self._driver.check(
            library.cuStreamCreate(
                ctypes.byref(self._compute_stream), self._CU_STREAM_NON_BLOCKING
            ),
            "create CUDA compute stream",
        )
        self._driver.check(
            library.cuStreamCreate(
                ctypes.byref(self._copy_stream), self._CU_STREAM_NON_BLOCKING
            ),
            "create CUDA copy stream",
        )
        _, eligibility_bytes = self._arrays["eligibility"]
        spike_bytes = ctypes.sizeof(ctypes.c_uint64)
        slot_stride = eligibility_bytes + spike_bytes
        self._slot_eligibility_bytes = eligibility_bytes
        self._device_slot_snapshots: list[int] = []
        self._device_slot_spikes: list[int] = []
        self._pinned_keepalive: list[object] = []
        self._pinned_addresses: list[int] = []
        self._slot_events: list[dict[str, ctypes.c_void_p]] = []
        try:
            host_block = ctypes.c_void_p()
            self._driver.check(
                library.cuMemHostAlloc(
                    ctypes.byref(host_block),
                    self._PIPELINE_DEPTH * slot_stride,
                    self._CU_MEMHOSTALLOC_PORTABLE,
                ),
                "allocate CUDA pinned staging ring",
            )
            base_address = int(host_block.value)
            self._pinned_keepalive.append(host_block)
            for slot in range(self._PIPELINE_DEPTH):
                snapshot = ctypes.c_uint64()
                self._driver.check(
                    library.cuMemAlloc_v2(
                        ctypes.byref(snapshot), eligibility_bytes
                    ),
                    f"allocate CUDA eligibility snapshot {slot}",
                )
                self._allocations.append(int(snapshot.value))
                self._device_slot_snapshots.append(int(snapshot.value))
                spikes = ctypes.c_uint64()
                self._driver.check(
                    library.cuMemAlloc_v2(ctypes.byref(spikes), spike_bytes),
                    f"allocate CUDA slot spike counter {slot}",
                )
                self._allocations.append(int(spikes.value))
                self._device_slot_spikes.append(int(spikes.value))
                address = base_address + slot * slot_stride
                self._pinned_addresses.append(address)
                self._pinned_keepalive.append(
                    (ctypes.c_char * slot_stride).from_address(address)
                )
                events = {
                    "start": ctypes.c_void_p(),
                    "stop": ctypes.c_void_p(),
                    "kernel_done": ctypes.c_void_p(),
                    "copy_done": ctypes.c_void_p(),
                }
                for name, event in events.items():
                    self._driver.check(
                        library.cuEventCreate(ctypes.byref(event), 0),
                        f"create CUDA {name} event {slot}",
                    )
                self._slot_events.append(events)
        except Exception:
            self.close()
            raise
        self._submit_sequence = 0
        self._pending_sequences: deque[int] = deque()

    @property
    def async_pipeline_active(self) -> bool:
        return hasattr(self, "_pending_sequences")

    @property
    def pending_tokens(self) -> int:
        return len(self._pending_sequences) if self.async_pipeline_active else 0

    def _launch_kernel(self, stream: ctypes.c_void_p | None) -> None:
        self._driver.check(
            self._driver.library.cuLaunchCooperativeKernel(
                self._function,
                self._block_count,
                1,
                1,
                self._BLOCK_SIZE,
                1,
                1,
                0,
                stream,
                self._launch_arguments,
            ),
            "launch CUDA token kernel",
        )

    def observe_async(self, token_id: int) -> int:
        """Enqueue one token interval without blocking the host.

        Returns a handle for :meth:`collect`. Handles must be collected in
        submission order; the eligibility host mirror is refreshed only by
        ``collect`` or the synchronous entry points, which drain first.
        """
        if not self.async_pipeline_active:
            raise RuntimeError("CUDA async pipeline is disabled")
        if not 0 <= token_id < self._token_cells:
            raise ValueError("token id outside CUDA afferent population")
        if len(self._pending_sequences) >= self._PIPELINE_DEPTH:
            raise RuntimeError("CUDA async pipeline stall: collect pending tokens first")
        self._driver.make_current()
        slot = self._submit_sequence % self._PIPELINE_DEPTH
        handle = self._submit_sequence
        self._submit_sequence += 1
        self._pending_sequences.append(handle)

        self._arg_afferent_pointer.value = (
            self._afferent_base_pointer + token_id * self._afferent_row_bytes
        )
        self._arg_spikes_pointer.value = self._device_slot_spikes[slot]

        library = self._driver.library
        events = self._slot_events[slot]
        check = self._driver.check
        check(library.cuEventRecord(events["start"], self._compute_stream),
              "record CUDA async start")
        self._launch_kernel(self._compute_stream)
        check(library.cuEventRecord(events["stop"], self._compute_stream),
              "record CUDA async stop")
        # Stable per-slot snapshot of the just-updated eligibility state.
        eligibility_pointer, eligibility_bytes = self._arrays["eligibility"]
        check(
            library.cuMemcpyDtoDAsync_v2(
                self._device_slot_snapshots[slot],
                eligibility_pointer,
                eligibility_bytes,
                self._compute_stream,
            ),
            "snapshot CUDA eligibility",
        )
        check(
            library.cuEventRecord(events["kernel_done"], self._compute_stream),
            "record CUDA kernel-done",
        )
        # Copy stream stages the snapshot into pinned memory while the next
        # kernel executes on the compute stream.
        check(
            library.cuStreamWaitEvent(self._copy_stream, events["kernel_done"], 0),
            "gate CUDA copy stream",
        )
        check(
            library.cuMemcpyDtoHAsync_v2(
                ctypes.c_void_p(self._pinned_addresses[slot]),
                self._device_slot_snapshots[slot],
                eligibility_bytes,
                self._copy_stream,
            ),
            "stage CUDA eligibility",
        )
        check(
            library.cuMemcpyDtoHAsync_v2(
                ctypes.c_void_p(
                    self._pinned_addresses[slot] + eligibility_bytes
                ),
                self._device_slot_spikes[slot],
                ctypes.sizeof(ctypes.c_uint64),
                self._copy_stream,
            ),
            "stage CUDA spike count",
        )
        check(
            library.cuEventRecord(events["copy_done"], self._copy_stream),
            "record CUDA copy-done",
        )
        return handle

    def collect(self, handle: int) -> int:
        """Wait one async token's staged results and refresh the mirror."""
        if not self.async_pipeline_active:
            raise RuntimeError("CUDA async pipeline is disabled")
        if not self._pending_sequences or handle != self._pending_sequences[0]:
            raise RuntimeError("async tokens must be collected in submission order")
        slot = handle % self._PIPELINE_DEPTH
        events = self._slot_events[slot]
        library = self._driver.library
        self._driver.check(
            library.cuEventSynchronize(events["copy_done"]),
            "wait for staged CUDA token",
        )
        elapsed = ctypes.c_float()
        self._driver.check(
            library.cuEventElapsedTime(
                ctypes.byref(elapsed), events["start"], events["stop"]
            ),
            "measure staged CUDA token kernel",
        )
        self.last_kernel_milliseconds = float(elapsed.value)
        eligibility_bytes = self._slot_eligibility_bytes
        pinned_view = np.frombuffer(
            (ctypes.c_char * eligibility_bytes).from_address(
                self._pinned_addresses[slot]
            ),
            dtype=np.float64,
        )
        np.copyto(self._eligibility, pinned_view)
        spike_view = np.frombuffer(
            (ctypes.c_char * ctypes.sizeof(ctypes.c_uint64)).from_address(
                self._pinned_addresses[slot] + eligibility_bytes
            ),
            dtype=np.uint64,
        )
        self._pending_sequences.popleft()
        return int(spike_view[0])

    def drain(self) -> None:
        """Collect every outstanding async token (no-op when idle)."""
        while self._pending_sequences:
            self.collect(self._pending_sequences[0])

    @property
    def device_name(self) -> str:
        return cuda_device_name()

    @property
    def device_bytes(self) -> int:
        return sum(size for _, size in self._arrays.values())

    @property
    def host_to_device_bytes_per_token(self) -> int:
        return 0

    @property
    def device_to_host_bytes_per_token(self) -> int:
        mutable = sum(self._arrays[name][1] for name in self._host_mirror_arrays())
        return mutable + ctypes.sizeof(ctypes.c_uint64)

    def _allocate_bytes(self, name: str, size: int) -> None:
        pointer = ctypes.c_uint64()
        self._driver.check(
            self._driver.library.cuMemAlloc_v2(ctypes.byref(pointer), size),
            f"allocate CUDA {name}",
        )
        self._allocations.append(int(pointer.value))
        self._arrays[name] = (int(pointer.value), int(size))

    def _allocate_array(self, name: str, array: np.ndarray, *, upload: bool) -> None:
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous")
        self._allocate_bytes(name, int(array.nbytes))
        if upload:
            self._copy_to_device(name, array)

    def _copy_to_device(self, name: str, array: np.ndarray) -> None:
        pointer, size = self._arrays[name]
        if int(array.nbytes) != size:
            raise ValueError(f"{name} byte size changed after CUDA binding")
        self._driver.check(
            self._driver.library.cuMemcpyHtoD_v2(pointer, ctypes.c_void_p(array.ctypes.data), size),
            f"upload CUDA {name}",
        )

    def _copy_from_device(self, name: str, array: np.ndarray) -> None:
        pointer, size = self._arrays[name]
        self._driver.check(
            self._driver.library.cuMemcpyDtoH_v2(ctypes.c_void_p(array.ctypes.data), pointer, size),
            f"download CUDA {name}",
        )

    def _structure_arrays_from_host(self) -> dict[str, np.ndarray]:
        cells = self._population.cell_count
        expected = (
            (
                self._afferent_targets,
                (self._token_cells, self._input_fanout),
                np.dtype(np.int32),
                "afferent targets",
            ),
            (
                self._recurrent_targets,
                (cells, self._recurrent_fanout),
                np.dtype(np.int32),
                "recurrent targets",
            ),
            (
                self._recurrent_weights_pa,
                (cells, self._recurrent_fanout),
                np.dtype(np.float64),
                "recurrent weights",
            ),
        )
        for array, shape, dtype, name in expected:
            if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
                raise ValueError(f"{name} must be a contiguous {dtype} array with shape {shape}")
        if np.any(self._afferent_targets < 0) or np.any(self._afferent_targets >= cells):
            raise ValueError("afferent targets must index reservoir cells")

        flat_targets = self._recurrent_targets.reshape(-1)
        if flat_targets.size > np.iinfo(np.int32).max:
            raise ValueError("CUDA backend supports at most INT32_MAX recurrent edges")
        if np.any(flat_targets < 0) or np.any(flat_targets >= cells):
            raise ValueError("recurrent targets must index reservoir cells")
        counts = np.bincount(flat_targets, minlength=cells)
        incoming_offsets = np.empty(cells + 1, dtype=np.int32)
        incoming_offsets[0] = 0
        incoming_offsets[1:] = np.cumsum(counts, dtype=np.int64)
        incoming_edges = np.argsort(flat_targets, kind="stable").astype(np.int32, copy=False)
        return {
            "afferents": self._afferent_targets,
            "incoming_offsets": incoming_offsets,
            "incoming_edges": incoming_edges,
            "recurrent_weights": self._recurrent_weights_pa,
        }

    def _mutable_arrays(self) -> dict[str, np.ndarray]:
        state = self._population.state
        return {
            "voltage": state.voltage_relative_mv,
            "after": state.after_spike_currents_pa,
            "threshold_spike": state.threshold_spike_mv,
            "threshold_voltage": state.threshold_voltage_mv,
            "refractory": state.refractory_steps,
            "synaptic_current": self._synaptic_current_pa,
            "eligibility": self._eligibility,
        }

    def _host_mirror_arrays(self) -> dict[str, np.ndarray]:
        if self._host_mirror == "eligibility":
            return {"eligibility": self._eligibility}
        return self._mutable_arrays()

    def synchronize_from_host(self) -> None:
        self._driver.make_current()
        if self.async_pipeline_active:
            self.drain()
        for name, array in self._mutable_arrays().items():
            self._copy_to_device(name, array)
        self._copy_to_device("decay_active", self._decayed_activity_words())
        self._copy_to_device(
            "threshold_spike_active", self._threshold_spike_active_words()
        )
        self._copy_to_device("refractory_active", self._refractory_active_words())

    def _pack_activity_bits(self, active: np.ndarray) -> np.ndarray:
        """Pack one bool per cell into uint64 words (8-byte padded)."""
        packed = np.packbits(active.astype(np.uint8), bitorder="little")
        remainder = packed.size % 8
        if remainder:
            packed = np.concatenate(
                [packed, np.zeros(8 - remainder, dtype=packed.dtype)]
            )
        return packed.view(np.uint64)

    def _decayed_activity_words(self) -> np.ndarray:
        """Pack the per-cell decayed-state activity bitmap (uint64 words).

        A cell is active when any after-spike current channel or its
        eligibility trace is not exactly +0.0. Negative zero is deliberately
        treated as active (signbit check): skipping a ``-0.0`` value would be
        a bit-level no-op for the after-current decay but not for the
        eligibility expression, and treating it as active keeps every path
        bit-identical to unconditional evaluation.
        """
        state = self._population.state
        after = state.after_spike_currents_pa
        eligibility = self._eligibility
        active = np.any((after != 0.0) | np.signbit(after), axis=1)
        active |= (eligibility != 0.0) | np.signbit(eligibility)
        return self._pack_activity_bits(active)

    def _threshold_spike_active_words(self) -> np.ndarray:
        """Pack the spike-threshold nonzero bitmap (uint64 words).

        A cell whose ``threshold_spike`` value is exactly +0.0 stays at those
        exact bits under decay (positive decay factor), so the kernel may skip
        its load, decay, and store. Negative zero is deliberately treated as
        active (signbit check) so every path keeps bit-identical state.
        """
        threshold = self._population.state.threshold_spike_mv
        return self._pack_activity_bits((threshold != 0.0) | np.signbit(threshold))

    def _refractory_active_words(self) -> np.ndarray:
        """Pack the refractory-countdown nonzero bitmap (uint64 words)."""
        return self._pack_activity_bits(self._population.state.refractory_steps != 0)

    def synchronize_to_host(self) -> None:
        """Download every mutable array for explicit state inspection."""
        self._driver.make_current()
        if self.async_pipeline_active:
            self.drain()
        for name, array in self._mutable_arrays().items():
            self._copy_from_device(name, array)

    def synchronize_structure_from_host(self) -> None:
        """Rebuild reverse CSR and upload all in-place structural mutations."""
        self._driver.make_current()
        if self.async_pipeline_active:
            self.drain()
        for name, array in self._structure_arrays_from_host().items():
            self._copy_to_device(name, array)

    def observe(self, token_id: int) -> int:
        if not 0 <= token_id < self._token_cells:
            raise ValueError("token id outside CUDA afferent population")
        self._driver.make_current()
        if self.async_pipeline_active:
            # Serialize behind any outstanding async tokens before touching
            # the shared primary buffers or the host mirror.
            self.drain()
        self._arg_afferent_pointer.value = (
            self._afferent_base_pointer + token_id * self._afferent_row_bytes
        )
        self._arg_spikes_pointer.value = self._arrays["total_spikes"][0]
        library = self._driver.library
        stream = (
            self._compute_stream if self.async_pipeline_active else None
        )
        check = self._driver.check
        check(library.cuEventRecord(self._start_event, stream), "record CUDA start")
        self._launch_kernel(stream)
        check(library.cuEventRecord(self._stop_event, stream), "record CUDA stop")
        check(library.cuEventSynchronize(self._stop_event), "wait for CUDA token")
        elapsed = ctypes.c_float()
        check(
            library.cuEventElapsedTime(ctypes.byref(elapsed), self._start_event, self._stop_event),
            "measure CUDA token kernel",
        )
        self.last_kernel_milliseconds = float(elapsed.value)
        for name, array in self._host_mirror_arrays().items():
            self._copy_from_device(name, array)
        spikes = np.empty(1, dtype=np.uint64)
        self._copy_from_device("total_spikes", spikes)
        return int(spikes[0])

    def close(self) -> None:
        driver = getattr(self, "_driver", None)
        if driver is None:
            return
        try:
            driver.make_current()
            if getattr(self, "async_pipeline_active", False):
                try:
                    self.drain()
                except Exception:
                    pass
            for pointer in reversed(getattr(self, "_allocations", [])):
                driver.library.cuMemFree_v2(pointer)
            self._allocations.clear()
            for event_name in (self._start_event, self._stop_event):
                if event_name is not None and event_name.value:
                    driver.library.cuEventDestroy_v2(event_name)
                    event_name.value = None
            for events in getattr(self, "_slot_events", []):
                for event in events.values():
                    if event is not None and event.value:
                        driver.library.cuEventDestroy_v2(event)
                        event.value = None
            self._slot_events = []
            library = driver.library
            for stream_name in ("_compute_stream", "_copy_stream"):
                stream = getattr(self, stream_name, None)
                if stream is not None and stream.value:
                    library.cuStreamDestroy(stream)
                    stream.value = None
            pinned_addresses = getattr(self, "_pinned_addresses", None)
            keepalive = getattr(self, "_pinned_keepalive", None)
            if pinned_addresses and keepalive is not None and keepalive:
                host_block = keepalive[0]
                if getattr(host_block, "value", None):
                    library.cuMemFreeHost(ctypes.c_void_p(host_block.value))
                    host_block.value = None
            self._pinned_keepalive = []
            self._pending_sequences = deque()
            module = getattr(self, "_module", None)
            if module is not None and module.value:
                driver.library.cuModuleUnload(module)
                module.value = None
        except (AttributeError, OSError):
            pass

    def __del__(self) -> None:
        self.close()
