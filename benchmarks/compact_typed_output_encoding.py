"""Pinned dense/compact engineering benchmark for typed physical outputs.

The public interface intentionally has only two operations: record the dense
baseline and record/compare the compact result.  All NumPy and project imports
live behind a hidden worker so affinity, priority, hash seeding, and BLAS thread
limits are fixed before NumPy enters the process.

This harness uses embedded synthetic and self-posed fixtures only.  It does not
load a Simple-SWE task bundle, corpus, partition, desired-output artifact, DEV,
held-out, M0, or C1 material.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SCHEMA = "compact-typed-output-benchmark-v1"
MIB = 1024 * 1024
EXPECTED_PROJECTION_SHA256 = (
    "f3ebca86516460b7a03ab5588cf26de2c6df07d0f5a109e8a6d5646eed015f44"
)
EXPECTED_OPERATOR_CHECKPOINT_SHA256 = (
    "92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e"
)
EXPECTED_CATALOG_OFF_CHECKPOINT_SHA256 = (
    "ed7dff6036ab522840bca1c0a1b6fd7c2e0de0768be813bb33b1920b5f291af3"
)
HOST_LOCK_PATH = (
    ROOT / "experiments" / "r3_pilot" / ".compact_typed_output_benchmark.host.lock"
)

THREAD_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PRINTABLE_ASCII = tuple(chr(code) for code in range(0x20, 0x7F))
SENTINEL_LITERALS = tuple(range(-12, 13))
RETIRED_ENCODING_ATTRIBUTES = (
    "cells",
    "active_factor_mask",
    "type_cells",
    "int_cells",
    "bool_cells",
    "list_length_cells",
    "list_item_cells",
    "char_cells",
    "string_length_cells",
    "string_item_cells",
)

SYNTHESIS_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "aggregate_sum_cost2",
        "parameter_name": "values",
        "argument_type": "List[Int]",
        "result_type": "Int",
        "literals": (0,),
        "max_ast_cost": 2,
        "arguments": (([1, -2, 3],), ([],), ([64, -64],), ([-5, 5],)),
        "desired": (2, 0, 0, 0),
        "expected_cost": 2,
        "expected_checkpoint_sha256": "c717589c63da93a09f4a18dfcb63435c9785a85aeb57d0cbaaa2e6a267e66674",
        "expected_tokens": ("sum", "variable-0"),
    },
    {
        "id": "aggregate_count_cost6",
        "parameter_name": "values",
        "argument_type": "List[Int]",
        "result_type": "Int",
        "literals": (2, 3, 5),
        "max_ast_cost": 9,
        "arguments": (([-2, 0, 2, 4],), ([3, 4],), ([2],), ([],)),
        "desired": (1, 1, 0, 0),
        "expected_cost": 6,
        "expected_checkpoint_sha256": "4f0ee145047bccba72745a65afd9c261f7dd6d7a845ecc16c90fe104ff2e0ad6",
        "expected_tokens": (
            "count",
            "bind",
            "less-than",
            "integer",
            "+",
            "3",
            "end",
            "variable-0",
            "variable-0",
        ),
    },
    {
        "id": "aggregate_clamp_map_sum_cost9",
        "parameter_name": "values",
        "argument_type": "List[Int]",
        "result_type": "Int",
        "literals": (-5, 5),
        "max_ast_cost": 9,
        "arguments": (([-9, -4, 0, 3],), ([9, -9],), ([-7],), ([7],)),
        "desired": (-6, 0, -5, 5),
        "expected_cost": 9,
        "expected_checkpoint_sha256": "09c1e11ba835fd229b2741b0816de4704ea99ae46880ebbc5247e80f862979a2",
        "expected_tokens": (
            "sum",
            "map",
            "bind",
            "minimum",
            "maximum",
            "variable-0",
            "integer",
            "-",
            "5",
            "end",
            "integer",
            "+",
            "5",
            "end",
            "variable-0",
        ),
    },
    {
        "id": "string_reverse_cost2",
        "parameter_name": "text",
        "argument_type": "String",
        "result_type": "String",
        "literals": (0,),
        "max_ast_cost": 7,
        "arguments": (("ab",), ("A1!",), ("",)),
        "desired": ("ba", "!1A", ""),
        "expected_cost": 2,
        "expected_checkpoint_sha256": "9ac87703dae72cf6efeee52b5ceaa953dffc072dc5303f3348d8a8b6f51486b6",
        "expected_tokens": ("reverse", "variable-0"),
        "expected_predictions": 63,
        "expected_races": 74,
        "expected_races_sha256": (
            "4ed8ab1c38ec4350a25314a0581c2bd5b5cdde2a1b8503a67b5154fe83a0dace"
        ),
    },
    {
        "id": "string_filter_vowels_cost6",
        "parameter_name": "text",
        "argument_type": "String",
        "result_type": "String",
        "literals": (0,),
        "max_ast_cost": 7,
        "arguments": (("aB3!",), ("xyzE",), ("IOu",), ("",)),
        "desired": ("a", "E", "IOu", ""),
        "expected_cost": 6,
        "expected_checkpoint_sha256": "9ac87703dae72cf6efeee52b5ceaa953dffc072dc5303f3348d8a8b6f51486b6",
        "expected_tokens": (
            "filter",
            "bind",
            "contains",
            "english-vowels",
            "variable-0",
            "variable-0",
        ),
        "expected_predictions": 364,
        "expected_races": 894,
    },
    {
        "id": "string_filter_digits_cost6",
        "parameter_name": "text",
        "argument_type": "String",
        "result_type": "String",
        "literals": (0,),
        "max_ast_cost": 7,
        "arguments": (("a08F9!",), ("765x",), ("G2?",), ("",)),
        "desired": ("089", "765", "2", ""),
        "expected_cost": 6,
        "expected_checkpoint_sha256": "9ac87703dae72cf6efeee52b5ceaa953dffc072dc5303f3348d8a8b6f51486b6",
        "expected_tokens": (
            "filter",
            "bind",
            "contains",
            "ascii-decimal-digits",
            "variable-0",
            "variable-0",
        ),
        "expected_predictions": 412,
        "expected_races": 996,
    },
    {
        "id": "string_reverse_consonants_cost7",
        "parameter_name": "text",
        "argument_type": "String",
        "result_type": "String",
        "literals": (0,),
        "max_ast_cost": 7,
        "arguments": (("aB3cE!",), ("xyzOU",), ("A1t",), ("",)),
        "desired": ("cB", "zyx", "t", ""),
        "expected_cost": 7,
        "expected_checkpoint_sha256": "9ac87703dae72cf6efeee52b5ceaa953dffc072dc5303f3348d8a8b6f51486b6",
        "expected_tokens": (
            "reverse",
            "filter",
            "bind",
            "contains",
            "english-consonants",
            "variable-0",
            "variable-0",
        ),
        "expected_predictions": 484,
        "expected_races": 1_274,
    },
)

PROTOCOL: dict[str, Any] = {
    "id": "compact-typed-output-encoding-v1",
    "scope": "embedded synthetic and self-posed mechanism fixtures only",
    "thread_environment": THREAD_ENVIRONMENT,
    "sampling": {
        "fast_runtime": {"warmups": 2, "measurements": 7},
        "synthesis_runtime": {"warmups": 2, "measurements": 3},
        "memory": {"warmups": 0, "fresh_process_samples": 7},
        "work_bound": {"warmups": 0, "measurements": 1},
    },
    "string_generator": {
        "length": 26,
        "alphabet": "printable-ascii-0x20-through-0x7e",
        "index_digits": "base-95 little-endian first-three-cycle",
        "position_stride": 17,
        "lane_stride": 23,
    },
    "memory_fixtures": (
        {"id": "strings_2000_max26", "count": 2_000},
        {"id": "detectors_1000_max26", "count": 1_000},
        {"id": "string_semantic_frontier_500x4", "rows": 500, "cases": 4},
    ),
    "fast_runtime_fixtures": (
        {"id": "detector_grid_500x4", "rows": 500, "cases": 4},
        {
            "id": "glif_kernel_64",
            "currents_pa": (-5_000.0, 2_500.0, 4_375.0, 5_000.0),
            "repetitions": 16,
        },
    ),
    "synthesis_fixtures": SYNTHESIS_FIXTURES,
    "work_bound_fixtures": (
        {
            "id": "sentinel_a",
            "literals": SENTINEL_LITERALS,
            "max_ast_cost": 9,
            "arguments": (([0, 0, 0, 0],), ([0, 0, 0, 0],)),
            "desired": (0, 1),
            "expected_status": "STOP_NO_EXACT_TYPED_SEMANTIC_PROGRAM",
            "expected_checkpoint_sha256": "13948fd329c79488f2e6532f3e013a10864db584db1c35b4219a1abe3fc9c477",
            "expected_declared_roots": 4_797,
            "expected_supported_roots": 4_757,
            "expected_semantic_classes": 87,
            "expected_predictions": 23_642,
            "expected_races": 63_456,
            "expected_races_sha256": (
                "2ca6e201b448b8b74070c485fe72ec66502fc303fc66e8f3f33f05c61fa82ff2"
            ),
        },
        {
            "id": "sentinel_b",
            "literals": SENTINEL_LITERALS,
            "max_ast_cost": 9,
            "arguments": (
                ([-12, -4, 5, 12],),
                ([12, 5, -4, -12],),
                ([-4, 12, -12, 5],),
                ([5, -12, 12, -4],),
            ),
            "desired": (0, 1, 2, 3),
            "expected_status": "STOP_SEMANTIC_FRONTIER_WORK_BOUND",
            "expected_checkpoint_sha256": "13948fd329c79488f2e6532f3e013a10864db584db1c35b4219a1abe3fc9c477",
            "expected_bound_name": "operator-predictions",
            "expected_bound_limit": 30_000,
            "expected_completed": 30_000,
            "expected_attempted": 30_001,
            "expected_ast_cost": 3,
            "expected_declared_roots": 1_651,
            "expected_supported_roots": 1_441,
            "expected_semantic_classes": 49,
            "expected_retained_representatives": 174,
            "expected_predictions": 30_000,
            "expected_races": 77_964,
            "expected_races_sha256": (
                "506b110fac0558063293d08527c8c0ae6ad89849e5d97b24859d6bd82123970c"
            ),
        },
    ),
}

MEMORY_FIXTURE_IDS = tuple(row["id"] for row in PROTOCOL["memory_fixtures"])
FAST_RUNTIME_FIXTURE_IDS = tuple(
    row["id"] for row in PROTOCOL["fast_runtime_fixtures"]
)
SYNTHESIS_FIXTURE_IDS = tuple(row["id"] for row in SYNTHESIS_FIXTURES)
RUNTIME_FIXTURE_IDS = FAST_RUNTIME_FIXTURE_IDS + SYNTHESIS_FIXTURE_IDS
WORK_BOUND_FIXTURE_IDS = tuple(
    row["id"] for row in PROTOCOL["work_bound_fixtures"]
)

FORBIDDEN_MODULE_PREFIXES = (
    "humanbrain.simple_" + "swe",
    "experiments.r3_pilot.public_typed_semantic_simple_" + "swe_train_bridge_v1",
)

RELEVANT_SOURCE_PATHS = (
    "benchmarks/compact_typed_output_encoding.py",
    "docs/research/compact-typed-output-encoding.md",
    "experiments/r3_pilot/public_typed_output_cells.py",
    "experiments/r3_pilot/public_typed_operator_successor.py",
    "experiments/r3_pilot/public_typed_semantic_synthesizer.py",
    "experiments/r3_pilot/public_typed_semantic_vector_race.py",
    "experiments/r3_pilot/public_train_prompt_semantic_bridge.py",
    "src/humanbrain/cells.py",
    "src/humanbrain/resting_glif_race.py",
    "src/humanbrain/resting_glif_tier_beam.py",
    "src/humanbrain/typed_program.py",
    "src/humanbrain/typed_program_types.py",
    "src/humanbrain/typed_semantic_synthesis.py",
    "tests/test_compact_typed_output_encoding_benchmark.py",
)


def _json_safe(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return [
            _json_safe(item)
            for item in sorted(value, key=lambda item: _canonical_bytes(item))
        ]
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_record(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_string(index: int, lane: int = 0) -> str:
    if type(index) is not int or index < 0:
        raise ValueError("string index must be a nonnegative integer")
    if type(lane) is not int or not 0 <= lane < 4:
        raise ValueError("string lane must be an integer in [0, 3]")
    digits = (index % 95, (index // 95) % 95, (index // (95 * 95)) % 95)
    return "".join(
        PRINTABLE_ASCII[(digits[position % 3] + 17 * position + 23 * lane) % 95]
        for position in range(26)
    )


def _parse_affinity_mask(text: str) -> int:
    try:
        value = int(text, 0)
    except (TypeError, ValueError) as error:
        raise ValueError("affinity mask must be a nonzero integer such as 0xFFF") from error
    if value <= 0:
        raise ValueError("affinity mask must be nonzero")
    if value.bit_length() > ctypes.sizeof(ctypes.c_size_t) * 8:
        raise ValueError("affinity mask exceeds this process pointer width")
    return value


def _power_scheme() -> dict[str, Any]:
    if os.name != "nt":
        return {"guid": None, "raw": None}
    completed = subprocess.run(
        ["powercfg", "/GETACTIVESCHEME"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    raw = completed.stdout.strip()
    match = re.search(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        raw,
    )
    if match is None:
        raise RuntimeError("powercfg did not return an active power-scheme GUID")
    return {"guid": match.group(0).lower(), "raw": raw}


def _windows_process_snapshot(*, set_affinity: int | None = None) -> dict[str, Any]:
    if os.name != "nt":
        available = tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else ()
        if set_affinity is not None and hasattr(os, "sched_setaffinity"):
            cpus = {index for index in range(set_affinity.bit_length()) if set_affinity & (1 << index)}
            os.sched_setaffinity(0, cpus)
            available = tuple(sorted(os.sched_getaffinity(0)))
        mask = sum(1 << index for index in available)
        return {
            "affinity_mask": f"0x{mask:X}",
            "system_affinity_mask": f"0x{mask:X}",
            "priority_class": "platform-default",
            "priority_class_value": None,
            "processor_group_count": 1,
            "active_processor_count": os.cpu_count(),
        }

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel32.GetPriorityClass.restype = wintypes.DWORD
    kernel32.GetActiveProcessorGroupCount.argtypes = []
    kernel32.GetActiveProcessorGroupCount.restype = wintypes.WORD
    kernel32.GetActiveProcessorCount.argtypes = [wintypes.WORD]
    kernel32.GetActiveProcessorCount.restype = wintypes.DWORD

    process = kernel32.GetCurrentProcess()
    group_count = int(kernel32.GetActiveProcessorGroupCount())
    if group_count != 1:
        raise RuntimeError("benchmark v1 requires exactly one Windows processor group")
    active_count = int(kernel32.GetActiveProcessorCount(0xFFFF))
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        process, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if set_affinity is not None:
        if set_affinity & ~int(system_mask.value):
            raise ValueError("requested affinity mask is outside the system affinity mask")
        if not kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(set_affinity)):
            raise ctypes.WinError(ctypes.get_last_error())
    normal_priority_class = 0x00000020
    if not kernel32.SetPriorityClass(process, normal_priority_class):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.GetProcessAffinityMask(
        process, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    priority = int(kernel32.GetPriorityClass(process))
    if priority != normal_priority_class:
        raise RuntimeError("worker process is not running at normal priority")
    return {
        "affinity_mask": f"0x{int(process_mask.value):X}",
        "system_affinity_mask": f"0x{int(system_mask.value):X}",
        "priority_class": "NORMAL_PRIORITY_CLASS",
        "priority_class_value": priority,
        "processor_group_count": group_count,
        "active_processor_count": active_count,
    }


def _system_power_status() -> dict[str, Any]:
    if os.name != "nt":
        return {
            "ac_line_status": None,
            "battery_flag": None,
            "battery_life_percent": None,
        }
    from ctypes import wintypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", wintypes.DWORD),
            ("BatteryFullLifeTime", wintypes.DWORD),
        ]

    status = SystemPowerStatus()
    get_status = ctypes.windll.kernel32.GetSystemPowerStatus
    get_status.argtypes = [ctypes.POINTER(SystemPowerStatus)]
    get_status.restype = wintypes.BOOL
    if not get_status(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "ac_line_status": int(status.ACLineStatus),
        "battery_flag": int(status.BatteryFlag),
        "battery_life_percent": int(status.BatteryLifePercent),
        "system_status_flag": int(status.SystemStatusFlag),
    }


def _cpu_model() -> str:
    if os.name == "nt":
        import winreg

        path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(value).strip()
    return platform.processor().strip()


def _process_memory_bytes() -> dict[str, int]:
    if os.name != "nt":
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        return {"working_set_bytes": peak, "private_usage_bytes": peak, "peak_working_set_bytes": peak}

    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_info.restype = wintypes.BOOL
    if not get_info(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "private_usage_bytes": int(counters.PrivateUsage),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
    }


def _runtime_conditions(np: Any, process: dict[str, Any]) -> dict[str, Any]:
    config = _json_safe(np.show_config(mode="dicts"))
    power_scheme = _power_scheme()
    power = _system_power_status()
    stable = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
        "byte_order": sys.byteorder,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "numpy_config_sha256": _sha256_record(config),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "process": process,
        "power": {
            "ac_line_status": power["ac_line_status"],
            "active_scheme_guid": power_scheme["guid"],
        },
    }
    return {
        "stable": stable,
        "stable_sha256": _sha256_record(stable),
        "numpy_config": config,
        "power_raw": {"status": power, "scheme": power_scheme},
    }


def _installed_distributions() -> list[dict[str, str]]:
    rows: set[tuple[str, str]] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or ""
        rows.add((name.lower().replace("_", "-"), distribution.version))
    return [{"name": name, "version": version} for name, version in sorted(rows)]


def _environment_worker(np: Any, process: dict[str, Any]) -> dict[str, Any]:
    conditions = _runtime_conditions(np, process)
    runtime_stream = io.StringIO()
    with contextlib.redirect_stdout(runtime_stream):
        np.show_runtime()
    distributions = _installed_distributions()
    stable = dict(conditions["stable"])
    stable["installed_distributions_sha256"] = _sha256_record(distributions)
    pyproject = ROOT / "pyproject.toml"
    stable["pyproject_sha256"] = _file_sha256(pyproject)
    conditions["stable"] = stable
    conditions["stable_sha256"] = _sha256_record(stable)
    conditions.update(
        {
            "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "installed_distributions": distributions,
            "numpy_runtime": runtime_stream.getvalue().strip(),
            "perf_counter": _json_safe(vars(time.get_clock_info("perf_counter"))),
        }
    )
    return conditions


def _scope_guard() -> dict[str, Any]:
    loaded = tuple(
        sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in FORBIDDEN_MODULE_PREFIXES
            )
        )
    )
    if loaded:
        raise RuntimeError(f"forbidden benchmark modules were loaded: {loaded!r}")
    return {
        "data_scope": "embedded-synthetic-self-posed",
        "forbidden_loaded_modules": loaded,
        "external_task_or_corpus_inputs": 0,
    }


def _output_module() -> Any:
    return importlib.import_module("experiments.r3_pilot.public_typed_output_cells")


def _typed_types_module() -> Any:
    return importlib.import_module("humanbrain.typed_program_types")


def _array_sha256(array: Any) -> str:
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _encoding_audit(np: Any, output: Any, encoded: Any) -> tuple[Any, Any]:
    materializer = getattr(output.TypedOutputCellBank, "materialize_for_audit", None)
    if materializer is not None:
        audit = materializer(encoded)
        return audit.cells, audit.active_factor_mask
    return encoded.cells, encoded.active_factor_mask


def _decode_record(output: Any, encoded: Any) -> dict[str, Any]:
    try:
        value = output.TypedOutputCellBank.decode(encoded)
    except (TypeError, ValueError) as error:
        return {"status": "error", "error_type": type(error).__name__}
    return {"status": "value", "value": _json_safe(value)}


def _encoding_cases(output: Any) -> list[tuple[str, Any, Any | None]]:
    cases: list[tuple[str, Any, Any | None]] = []
    cases.extend((f"int/{value}", value, None) for value in range(output.INT_MIN, output.INT_MAX + 1))
    cases.extend(
        (
            ("int/underflow", output.INT_MIN - 1, None),
            ("int/overflow", output.INT_MAX + 1, None),
            ("int/bool-declared", True, output.OutputType.INT),
            ("bool/false", False, None),
            ("bool/true", True, None),
            ("bool/int-declared", 0, output.OutputType.BOOL),
        )
    )
    cases.extend(
        (f"char/{ord(character):02x}", character, output.OutputType.CHAR)
        for character in output.PRINTABLE_ASCII
    )
    cases.extend(
        (
            ("char/newline", "\n", output.OutputType.CHAR),
            ("char/unicode", "é", output.OutputType.CHAR),
            ("char/multiple", "AB", output.OutputType.CHAR),
        )
    )
    list_base = [output.INT_MIN, -1, 0, output.INT_MAX]
    cases.extend((f"list/length-{length}", list_base[:length], None) for length in range(5))
    for position in range(4):
        changed = [0, 0, 0, 0]
        changed[position] = position + 1
        cases.append((f"list/position-{position}", changed, None))
        invalid = [0, 0, 0, 0]
        invalid[position] = output.INT_MAX + 1
        cases.append((f"list/invalid-position-{position}", invalid, None))
    cases.extend(
        (
            ("list/over-capacity", [0, 0, 0, 0, 0], None),
            ("list/non-list-declared", "bad", output.OutputType.LIST_INT),
        )
    )
    base_string = "".join(output.PRINTABLE_ASCII[: output.STRING_CAPACITY])
    cases.extend(
        (
            ("string/empty", "", None),
            ("string/one", "A", None),
            ("string/max", base_string, None),
        )
    )
    for position in range(output.STRING_CAPACITY):
        replacement = output.PRINTABLE_ASCII[
            (output.PRINTABLE_ASCII.index(base_string[position]) + 1) % len(output.PRINTABLE_ASCII)
        ]
        cases.append(
            (
                f"string/position-{position}",
                base_string[:position] + replacement + base_string[position + 1 :],
                None,
            )
        )
    for label, character in (("nonprintable", "\n"), ("unicode", "é")):
        for position in (0, 13, 25):
            cases.append(
                (
                    f"string/{label}-position-{position}",
                    base_string[:position] + character + base_string[position + 1 :],
                    None,
                )
            )
    cases.extend(
        (
            ("string/over-capacity", "A" * (output.STRING_CAPACITY + 1), None),
            ("string/non-string-declared", object(), output.OutputType.STRING),
            ("unknown/untyped", object(), None),
        )
    )
    return cases


def _encoding_record(np: Any, output: Any, case_id: str, value: Any, declared: Any | None) -> dict[str, Any]:
    encoded = output.TypedOutputCellBank.encode(value, declared_type=declared)
    validator = getattr(output.TypedOutputCellBank, "validate", None)
    if validator is not None and validator(encoded) is not encoded:
        raise RuntimeError("canonical validation did not preserve encoding identity")
    cells, mask = _encoding_audit(np, output, encoded)
    audit_indices = tuple(int(cell) for cell in np.flatnonzero(cells))
    active_method = getattr(encoded, "active_global_cell_indices", None)
    method_indices = audit_indices if active_method is None else tuple(active_method())
    if method_indices != audit_indices:
        raise RuntimeError("compact active addresses differ from explicit audit projection")
    output_type = getattr(encoded, "output_type")
    return {
        "id": case_id,
        "declared_type": None if declared is None else declared.value,
        "output_type": None if output_type is None else output_type.value,
        "supported": bool(encoded.supported),
        "unknown_active": bool(encoded.unknown_active),
        "active_factors": tuple(encoded.active_factors),
        "active_global_cell_indices": audit_indices,
        "cells_dtype": cells.dtype.str,
        "cells_shape": tuple(int(value) for value in cells.shape),
        "cells_sha256": _array_sha256(cells),
        "mask_dtype": mask.dtype.str,
        "mask_shape": tuple(int(value) for value in mask.shape),
        "mask_sha256": _array_sha256(mask),
        "encoding_sha256": encoded.sha256,
        "decode": _decode_record(output, encoded),
    }


def _coincidence_record(result: Any) -> dict[str, Any]:
    return {
        "spiked": bool(result.spiked),
        "spike_mask_dtype": result.spike_mask.dtype.str,
        "spike_mask_shape": tuple(int(value) for value in result.spike_mask.shape),
        "spike_mask_sha256": _array_sha256(result.spike_mask),
        "current_pa": float(result.current_pa),
        "matched_factor_cells": int(result.matched_factor_cells),
        "active_target_factors": int(result.active_target_factors),
        "detector_sha256": result.detector_sha256,
        "predicted_sha256": result.predicted_sha256,
    }


def _detector_golden(output: Any) -> tuple[list[dict[str, Any]], Any]:
    rows: list[dict[str, Any]] = []

    def capture(case_id: str, detector: Any, predictions: Sequence[Any]) -> None:
        batched = detector.detect_many(tuple(predictions))
        scalar = tuple(detector.detect(item) for item in predictions)
        batch_records = tuple(_coincidence_record(item) for item in batched)
        scalar_records = tuple(_coincidence_record(item) for item in scalar)
        if batch_records != scalar_records:
            raise RuntimeError("golden detector scalar and batch paths differ")
        rows.append(
            {
                "id": case_id,
                "detector_sha256": detector.sha256,
                "enabled": bool(detector.enabled),
                "results": batch_records,
            }
        )

    list_detector = output.FixedTypedCoincidenceDetector.for_target([1, -2])
    capture(
        "list",
        list_detector,
        tuple(
            output.TypedOutputCellBank.encode(value)
            for value in ([1, -2], [1, -1], [1, output.INT_MAX + 1], True, object())
        ),
    )
    invalid_detector = output.FixedTypedCoincidenceDetector.for_target(output.INT_MAX + 1)
    capture(
        "disabled-int",
        invalid_detector,
        tuple(output.TypedOutputCellBank.encode(value) for value in (output.INT_MAX + 1, output.INT_MAX)),
    )
    char_detector = output.FixedTypedCoincidenceDetector.for_target(
        "A", declared_type=output.OutputType.CHAR
    )
    capture(
        "char",
        char_detector,
        (
            output.TypedOutputCellBank.encode("A", declared_type=output.OutputType.CHAR),
            output.TypedOutputCellBank.encode("a", declared_type=output.OutputType.CHAR),
            output.TypedOutputCellBank.encode("A"),
        ),
    )
    string_detector = output.FixedTypedCoincidenceDetector.for_target("cat")
    capture(
        "string",
        string_detector,
        tuple(output.TypedOutputCellBank.encode(value) for value in ("cat", "car", "cats")),
    )
    return rows, list_detector


def _is_binary_digest(value: Any) -> bool:
    return type(value) is bytes and len(value) == 32


def _stored_structure(
    np: Any, output: Any, encoded: Any, detector: Any
) -> dict[str, Any]:
    encoding_fields = tuple(field.name for field in fields(type(encoded)))
    detector_fields = tuple(field.name for field in fields(type(detector)))
    encoding_arrays = tuple(
        name for name in encoding_fields if isinstance(getattr(encoded, name), np.ndarray)
    )
    detector_arrays = tuple(
        name for name in detector_fields if isinstance(getattr(detector, name), np.ndarray)
    )
    winners = getattr(encoded, "factor_winners_le", None)
    digest = getattr(encoded, "physical_digest", None)
    detector_values = tuple(getattr(detector, name) for name in detector_fields)
    detector_target_fields = sum(
        isinstance(value, type(encoded)) for value in detector_values
    )
    detector_digest_fields = sum(
        _is_binary_digest(value) for value in detector_values
    )
    return {
        "encoding_fields": encoding_fields,
        "encoding_direct_numpy_array_fields": encoding_arrays,
        "encoding_shallow_bytes": sys.getsizeof(encoded),
        "factor_winners_bytes": None if winners is None else len(winners),
        "physical_digest_bytes": None if digest is None else len(digest),
        "compact_stored_payload_bytes": None if winners is None or digest is None else len(winners) + len(digest),
        "retired_encoding_attributes_present": tuple(
            name for name in RETIRED_ENCODING_ATTRIBUTES if hasattr(type(encoded), name)
        ),
        "bank_finish_present": hasattr(output.TypedOutputCellBank, "_finish"),
        "detector_fields": detector_fields,
        "detector_stored_field_count": len(detector_fields),
        "detector_target_field_count": detector_target_fields,
        "detector_digest_field_count": detector_digest_fields,
        "detector_other_field_count": (
            len(detector_fields) - detector_target_fields - detector_digest_fields
        ),
        "detector_weights_present": hasattr(type(detector), "weights"),
        "detector_direct_numpy_array_fields": detector_arrays,
        "detector_shallow_bytes": sys.getsizeof(detector),
    }


def _prompt_golden(output: Any) -> dict[str, Any]:
    bridge = importlib.import_module("experiments.r3_pilot.public_train_prompt_semantic_bridge")
    types = _typed_types_module()
    integer = types.ProgramType.Int()
    signature = types.TypedSignature(
        entry_point="offset",
        parameters=(types.ProgramParameter("value", integer),),
        result=integer,
    )
    suffix = "Return value plus -3, then value plus +5, then 0."
    prompt = (
        bridge.FROZEN_PROMPT_PREFIX
        + "def offset(value):\n"
        + f'    """{suffix}"""'
    )
    observed = bridge.observe_public_train_prompt(prompt, signature)
    literals = tuple(
        {
            "lexeme": item.lexeme,
            "value": item.value,
            "typed_output_sha256": item.typed_output.sha256,
            "typed_output_active_cells": item.typed_output_active_cells,
            "semantic_sha256": item.semantic_sha256,
            "audit_sha256": item.audit_sha256,
        }
        for item in observed.signed_literals
    )
    return {
        "tokens": observed.tokens,
        "typed_output_projection_sha256": observed.typed_output_projection_sha256,
        "lexical_sha256": observed.lexical_sha256,
        "literal_channel_sha256": observed.literal_channel_sha256,
        "sha256": observed.sha256,
        "signed_literals": literals,
    }


def _catalog_off_checkpoint() -> str:
    semantic = importlib.import_module(
        "experiments.r3_pilot.public_typed_semantic_synthesizer"
    )
    model = semantic.build_public_typed_semantic_synthesizer(
        literals=(0,),
        max_ast_cost=7,
        _ascii_catalog_enabled=False,
    )[0]
    if model.operator.checkpoint_sha256 != EXPECTED_OPERATOR_CHECKPOINT_SHA256:
        raise RuntimeError("catalog-off operator checkpoint changed")
    if model.checkpoint_sha256 != EXPECTED_CATALOG_OFF_CHECKPOINT_SHA256:
        raise RuntimeError("catalog-off semantic checkpoint changed")
    return model.checkpoint_sha256


def _golden_worker(np: Any) -> dict[str, Any]:
    output = _output_module()
    projection_sha256 = output.TypedOutputCellBank.projection_sha256()
    if projection_sha256 != EXPECTED_PROJECTION_SHA256:
        raise RuntimeError("typed-output physical projection identity changed")
    encoding_rows = [
        _encoding_record(np, output, case_id, value, declared)
        for case_id, value, declared in _encoding_cases(output)
    ]
    detector_rows, structural_detector = _detector_golden(output)
    structural_encoding = output.TypedOutputCellBank.encode(_max_string(7))
    scientific = {
        "projection_sha256": projection_sha256,
        "encodings": encoding_rows,
        "encodings_sha256": _sha256_record(encoding_rows),
        "detectors": detector_rows,
        "detectors_sha256": _sha256_record(detector_rows),
        "prompt_literal": _prompt_golden(output),
        "catalog_off_checkpoint_sha256": _catalog_off_checkpoint(),
        "scope_guard": _scope_guard(),
    }
    scientific_sha256 = _sha256_record(scientific)
    return {
        **scientific,
        "scientific_sha256": scientific_sha256,
        # Storage must change across the cutover, so it is deliberately not
        # part of the scientific parity digest above.
        "structural_storage": _stored_structure(
            np, output, structural_encoding, structural_detector
        ),
        "sha256": scientific_sha256,
    }


def _install_production_audit_guard(output: Any) -> dict[str, int]:
    calls = {"count": 0}
    if not hasattr(output.TypedOutputCellBank, "materialize_for_audit"):
        return calls

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls["count"] += 1
        raise RuntimeError("production benchmark called audit materialization")

    setattr(output.TypedOutputCellBank, "materialize_for_audit", staticmethod(forbidden))
    return calls


def _memory_objects(output: Any, fixture_id: str) -> tuple[tuple[Any, ...], Callable[[], Any]]:
    if fixture_id == "strings_2000_max26":
        raw = tuple(_max_string(index) for index in range(2_000))
        return raw, lambda: tuple(output.TypedOutputCellBank.encode(value) for value in raw)
    if fixture_id == "detectors_1000_max26":
        raw = tuple(_max_string(index) for index in range(1_000))
        return raw, lambda: tuple(output.FixedTypedCoincidenceDetector.for_target(value) for value in raw)
    if fixture_id == "string_semantic_frontier_500x4":
        targets = tuple(_max_string(0, lane) for lane in range(4))
        raw_rows = tuple(
            tuple(
                targets[lane] if row % 127 == lane else _max_string(row + 1, lane)
                for lane in range(4)
            )
            for row in range(500)
        )
        return raw_rows, lambda: tuple(
            tuple(output.TypedOutputCellBank.encode(value) for value in row)
            for row in raw_rows
        )
    raise ValueError(f"unknown memory fixture: {fixture_id}")


def _retained_fingerprint(objects: Any) -> str:
    digest = hashlib.sha256()
    stack = [objects]
    while stack:
        value = stack.pop()
        if isinstance(value, tuple):
            stack.extend(reversed(value))
        elif hasattr(value, "sha256"):
            digest.update(str(value.sha256).encode("ascii"))
            digest.update(b"\0")
        else:
            raise RuntimeError("memory fixture retained a non-authenticated object")
    return digest.hexdigest()


def _memory_worker(np: Any, fixture_id: str) -> dict[str, Any]:
    output = _output_module()
    audit_calls = _install_production_audit_guard(output)
    raw_inputs, factory = _memory_objects(output, fixture_id)
    # Memory samples have no object warmup.  Every sample is a fresh worker,
    # and this baseline retains only its deterministic raw Python inputs.
    gc.collect()
    tracemalloc.start()
    baseline_traced, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    baseline_process = _process_memory_bytes()
    retained = factory()
    gc.collect()
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    after_process = _process_memory_bytes()
    record = {
        "fixture": fixture_id,
        "fresh_process": True,
        "object_warmups": 0,
        "retained_count": 2_000 if fixture_id != "detectors_1000_max26" else 1_000,
        "tracemalloc_baseline_bytes": baseline_traced,
        "tracemalloc_current_delta_bytes": traced_current - baseline_traced,
        "tracemalloc_peak_delta_bytes": traced_peak - baseline_traced,
        "working_set_delta_bytes": after_process["working_set_bytes"] - baseline_process["working_set_bytes"],
        "private_usage_delta_bytes": after_process["private_usage_bytes"] - baseline_process["private_usage_bytes"],
        "peak_working_set_bytes": after_process["peak_working_set_bytes"],
        "retained_sha256": _retained_fingerprint(retained),
        "production_audit_materialization_calls": audit_calls["count"],
        "scope_guard": _scope_guard(),
    }
    tracemalloc.stop()
    if audit_calls["count"]:
        raise RuntimeError("memory fixture called audit materialization")
    return record


def _coincidence_grid_fingerprint(results: Sequence[Sequence[Any]]) -> dict[str, Any]:
    records = tuple(
        tuple(_coincidence_record(result) for result in lane) for lane in results
    )
    return {
        "lane_spike_counts": tuple(sum(int(row["spiked"]) for row in lane) for lane in records),
        "results_sha256": _sha256_record(records),
    }


def _decision_record(decision: Any) -> dict[str, Any]:
    tiers = []
    for attempt in decision.tier_attempts:
        representatives = tuple(attempt.representative_candidate_ids)
        race = attempt.race
        tiers.append(
            {
                "ast_cost": int(attempt.ast_cost),
                "evaluated_candidates": int(attempt.evaluated_candidates),
                "representative_count": len(representatives),
                "representative_ids_sha256": _sha256_record(representatives),
                "race_sha256": None if race is None else race.sha256,
                "race_status": None if race is None else race.status,
                "race_winner_key": None if race is None else race.winner_key,
            }
        )
    work_bound = None if decision.work_bound is None else _json_safe(asdict(decision.work_bound))
    declared_root_candidates = (
        sum(int(attempt.evaluated_candidates) for attempt in decision.tier_attempts)
        if decision.work_bound is None
        else int(decision.work_bound.completed_declared_candidates)
    )
    record = {
        "status": decision.status,
        "stop_reason": decision.stop_reason,
        "candidate_count": int(decision.candidate_count),
        "declared_root_candidates": declared_root_candidates,
        "observed_semantic_classes": int(decision.observed_semantic_classes),
        "tier_attempts": tiers,
        "tier_attempts_sha256": _sha256_record(tiers),
        "selected_candidate_id": decision.selected_candidate_id,
        "program_tokens": None if decision.program is None else tuple(decision.program.tokens),
        "source_sha256": None if decision.source is None else hashlib.sha256(decision.source.encode("utf-8")).hexdigest(),
        "operator_predictions": int(decision.operator_predictions),
        "operator_races": int(decision.operator_races),
        "operator_races_sha256": decision.operator_races_sha256,
        "checkpoint_before_sha256": decision.checkpoint_before_sha256,
        "checkpoint_after_sha256": decision.checkpoint_after_sha256,
        "work_bound": work_bound,
    }
    record["sha256"] = _sha256_record(record)
    return record


def _model_decision_record(model: Any, decision: Any) -> dict[str, Any]:
    record = _decision_record(decision)
    record.pop("sha256")
    memo = model.operator._effect_memo
    record.update(
        {
            "operator_checkpoint_sha256": model.operator.checkpoint_sha256,
            "semantic_checkpoint_sha256": model.checkpoint_sha256,
            "effect_memo": _json_safe(asdict(model.operator.effect_memo_info())),
            "effect_memo_keys_sha256": _sha256_record(tuple(memo._entries)),
        }
    )
    record["sha256"] = _sha256_record(record)
    return record


def _time_callable(
    run: Callable[[], Any],
    fingerprint: Callable[[Any], dict[str, Any]],
    *,
    warmups: int,
    measurements: int,
    before_each: Callable[[], None] | None = None,
) -> dict[str, Any]:
    expected: dict[str, Any] | None = None
    for _ in range(warmups):
        if before_each is not None:
            before_each()
        gc.collect()
        result = run()
        observed = fingerprint(result)
        if expected is None:
            expected = observed
        elif observed != expected:
            raise RuntimeError("benchmark warmups produced different scientific results")
        del result
    samples: list[int] = []
    for _ in range(measurements):
        if before_each is not None:
            before_each()
        gc.collect()
        started = time.perf_counter_ns()
        result = run()
        elapsed = time.perf_counter_ns() - started
        observed = fingerprint(result)
        if expected is None:
            expected = observed
        elif observed != expected:
            raise RuntimeError("benchmark measurements produced different scientific results")
        samples.append(elapsed)
        del result
    if expected is None:
        raise RuntimeError("benchmark must execute at least once")
    return {
        "warmups": warmups,
        "measurements": measurements,
        "raw_nanoseconds": samples,
        "raw_seconds": tuple(value / 1_000_000_000.0 for value in samples),
        "median_nanoseconds": float(statistics.median(samples)),
        "median_seconds": statistics.median(samples) / 1_000_000_000.0,
        "result": expected,
        "result_sha256": _sha256_record(expected),
    }


def _runtime_worker(np: Any, fixture_id: str, phase: str) -> dict[str, Any]:
    output = _output_module()
    audit_calls = _install_production_audit_guard(output)
    if fixture_id == "detector_grid_500x4":
        targets = tuple(_max_string(0, lane) for lane in range(4))
        rows = tuple(
            tuple(
                output.TypedOutputCellBank.encode(
                    targets[lane] if row % 127 == lane else _max_string(row + 1, lane)
                )
                for lane in range(4)
            )
            for row in range(500)
        )
        columns = tuple(tuple(row[lane] for row in rows) for lane in range(4))
        detectors = tuple(output.FixedTypedCoincidenceDetector.for_target(value) for value in targets)

        if phase == "dense":
            if not all(isinstance(item.cells, np.ndarray) for row in rows for item in row):
                raise RuntimeError("dense detector baseline is not array-backed")
            if any(hasattr(detector, "_detect_many_trusted") for detector in detectors):
                raise RuntimeError("dense detector baseline unexpectedly exposes the compact trusted seam")
            detector_calls = tuple(detector.detect_many for detector in detectors)
            timed_boundary = "dense-public-detect-many-no-ingress-validation"
        elif phase == "compact":
            validator = getattr(output.TypedOutputCellBank, "validate", None)
            if validator is None:
                raise RuntimeError("compact detector grid requires canonical ingress validation")
            if any(hasattr(item, "cells") for row in rows for item in row):
                raise RuntimeError("compact detector grid still exposes dense encoding cells")
            trusted = tuple(
                getattr(detector, "_detect_many_trusted", None)
                for detector in detectors
            )
            if not all(callable(call) for call in trusted):
                raise RuntimeError("compact detector grid requires _detect_many_trusted")
            # Validate once at the enclosing semantic entry boundary.  Neither
            # validation nor construction is part of the timed physical loop.
            columns = tuple(
                tuple(validator(item) for item in column) for column in columns
            )
            detector_calls = trusted
            timed_boundary = "compact-trusted-detect-many-prevalidated"
        else:
            raise ValueError("runtime phase must be dense or compact")

        def run() -> Any:
            return tuple(
                detect_many(column)
                for detect_many, column in zip(detector_calls, columns, strict=True)
            )

        timed = _time_callable(
            run,
            _coincidence_grid_fingerprint,
            warmups=2,
            measurements=7,
        )
        if timed["result"]["lane_spike_counts"] != (4, 4, 4, 4):
            raise RuntimeError("detector-grid exact rows changed")
    elif fixture_id == "glif_kernel_64":
        timed_boundary = "direct-fitted-glif-spike-mask"
        currents = np.tile(np.asarray((-5_000.0, 2_500.0, 4_375.0, 5_000.0), dtype=np.float64), 16)

        def run() -> Any:
            return output._drive_spike_mask(currents)

        def fingerprint(mask: Any) -> dict[str, Any]:
            return {
                "dtype": mask.dtype.str,
                "shape": tuple(int(value) for value in mask.shape),
                "sha256": _array_sha256(mask),
                "spikes": tuple(bool(value) for value in mask),
            }

        timed = _time_callable(run, fingerprint, warmups=2, measurements=7)
        expected = (False, False, True, True) * 16
        if timed["result"]["spikes"] != expected:
            raise RuntimeError("64-row fitted-GLIF kernel parity changed")
    elif fixture_id in SYNTHESIS_FIXTURE_IDS:
        timed_boundary = "public-bottom-up-synthesize"
        semantic = importlib.import_module("experiments.r3_pilot.public_typed_semantic_synthesizer")
        types = _typed_types_module()
        spec = next(row for row in SYNTHESIS_FIXTURES if row["id"] == fixture_id)
        integer = types.ProgramType.Int()
        string = types.ProgramType.String()
        list_integer = types.ProgramType.List(integer)
        by_label = {"Int": integer, "String": string, "List[Int]": list_integer}
        signature = types.TypedSignature(
            entry_point="solve",
            parameters=(
                types.ProgramParameter(
                    spec["parameter_name"], by_label[spec["argument_type"]]
                ),
            ),
            result=by_label[spec["result_type"]],
        )
        model = semantic.build_public_typed_semantic_synthesizer(
            literals=spec["literals"], max_ast_cost=spec["max_ast_cost"]
        )[0]
        if model.operator.checkpoint_sha256 != EXPECTED_OPERATOR_CHECKPOINT_SHA256:
            raise RuntimeError("typed operator checkpoint changed")
        if model.checkpoint_sha256 != spec["expected_checkpoint_sha256"]:
            raise RuntimeError(f"semantic checkpoint changed: {fixture_id}")

        def run() -> Any:
            return model.synthesize(signature, spec["arguments"], spec["desired"])

        def before_each() -> None:
            model.operator.clear_effect_memo()

        timed = _time_callable(
            run,
            lambda decision: _model_decision_record(model, decision),
            warmups=2,
            measurements=3,
            before_each=before_each,
        )
        decision = timed["result"]
        if decision["status"] != semantic.STATUS_DONE:
            raise RuntimeError(f"synthesis fixture stopped: {fixture_id}")
        if tuple(decision["program_tokens"]) != tuple(spec["expected_tokens"]):
            raise RuntimeError(f"synthesis fixture selected a different program: {fixture_id}")
        completed_cost = decision["tier_attempts"][-1]["ast_cost"]
        if completed_cost != spec["expected_cost"]:
            raise RuntimeError(f"synthesis fixture selected a different AST cost: {fixture_id}")
        for key, decision_key in (("expected_predictions", "operator_predictions"), ("expected_races", "operator_races"), ("expected_races_sha256", "operator_races_sha256")):
            if key in spec and decision[decision_key] != spec[key]:
                raise RuntimeError(f"synthesis fixture telemetry changed: {fixture_id}/{decision_key}")
    else:
        raise ValueError(f"unknown runtime fixture: {fixture_id}")
    timed.update(
        {
            "fixture": fixture_id,
            "timed_boundary": timed_boundary,
            "production_audit_materialization_calls": audit_calls["count"],
            "scope_guard": _scope_guard(),
        }
    )
    if audit_calls["count"]:
        raise RuntimeError("runtime fixture called audit materialization")
    return timed


def _work_bound_worker(np: Any, fixture_id: str) -> dict[str, Any]:
    del np
    output = _output_module()
    audit_calls = _install_production_audit_guard(output)
    semantic = importlib.import_module("experiments.r3_pilot.public_typed_semantic_synthesizer")
    types = _typed_types_module()
    integer = types.ProgramType.Int()
    list_integer = types.ProgramType.List(integer)
    signature = types.TypedSignature(
        entry_point="solve",
        parameters=(types.ProgramParameter("values", list_integer),),
        result=integer,
    )
    spec = next(
        row for row in PROTOCOL["work_bound_fixtures"] if row["id"] == fixture_id
    )
    started = time.perf_counter_ns()
    model = semantic.build_public_typed_semantic_synthesizer(
        literals=spec["literals"], max_ast_cost=spec["max_ast_cost"]
    )[0]
    if model.operator.checkpoint_sha256 != EXPECTED_OPERATOR_CHECKPOINT_SHA256:
        raise RuntimeError("typed operator checkpoint changed")
    if model.checkpoint_sha256 != spec["expected_checkpoint_sha256"]:
        raise RuntimeError(f"semantic checkpoint changed: {fixture_id}")
    decision = model.synthesize(signature, spec["arguments"], spec["desired"])
    elapsed = time.perf_counter_ns() - started
    record = _model_decision_record(model, decision)
    if fixture_id == "sentinel_a":
        if decision.status != semantic.STATUS_STOP or decision.work_bound is not None:
            raise RuntimeError("Sentinel A no longer reaches ordinary exhaustive STOP")
        observed = (
            record["declared_root_candidates"],
            decision.candidate_count,
            decision.observed_semantic_classes,
            decision.operator_predictions,
            decision.operator_races,
            decision.operator_races_sha256,
        )
        expected = tuple(
            spec[key]
            for key in (
                "expected_declared_roots",
                "expected_supported_roots",
                "expected_semantic_classes",
                "expected_predictions",
                "expected_races",
                "expected_races_sha256",
            )
        )
    else:
        if decision.status != semantic.STATUS_WORK_BOUND_STOP or decision.work_bound is None:
            raise RuntimeError("Sentinel B no longer reaches the physical work bound")
        bound = decision.work_bound
        observed = (
            bound.bound_name,
            bound.limit,
            bound.completed,
            bound.attempted,
            bound.ast_cost,
            bound.completed_declared_candidates,
            decision.candidate_count,
            decision.observed_semantic_classes,
            bound.completed_retained_representatives,
            decision.operator_predictions,
            decision.operator_races,
            decision.operator_races_sha256,
        )
        expected = tuple(
            spec[key]
            for key in (
                "expected_bound_name",
                "expected_bound_limit",
                "expected_completed",
                "expected_attempted",
                "expected_ast_cost",
                "expected_declared_roots",
                "expected_supported_roots",
                "expected_semantic_classes",
                "expected_retained_representatives",
                "expected_predictions",
                "expected_races",
                "expected_races_sha256",
            )
        )
    if observed != expected:
        raise RuntimeError(f"{fixture_id} scientific counters changed: {observed!r}")
    if audit_calls["count"]:
        raise RuntimeError("work-bound fixture called audit materialization")
    return {
        "fixture": fixture_id,
        "wall_nanoseconds": elapsed,
        "wall_seconds": elapsed / 1_000_000_000.0,
        "decision": record,
        "decision_sha256": record["sha256"],
        "production_audit_materialization_calls": audit_calls["count"],
        "scope_guard": _scope_guard(),
    }


def _worker_payload(
    kind: str, fixture: str | None, affinity_mask: int, phase: str
) -> dict[str, Any]:
    if phase not in ("dense", "compact"):
        raise ValueError("worker phase must be dense or compact")
    process = _windows_process_snapshot(set_affinity=affinity_mask)
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"worker environment {name} must equal {expected!r}")
    np = importlib.import_module("numpy")
    if kind == "environment":
        result = _environment_worker(np, process)
        condition_sha = result["stable_sha256"]
    elif kind == "golden":
        result = _golden_worker(np)
        conditions = _runtime_conditions(np, process)
        condition_sha = conditions["stable_sha256"]
        result["worker_conditions"] = conditions
    elif kind == "memory":
        if fixture is None:
            raise ValueError("memory worker requires a fixture")
        result = _memory_worker(np, fixture)
        conditions = _runtime_conditions(np, process)
        condition_sha = conditions["stable_sha256"]
        result["worker_conditions"] = conditions
    elif kind == "runtime":
        if fixture is None:
            raise ValueError("runtime worker requires a fixture")
        result = _runtime_worker(np, fixture, phase)
        conditions = _runtime_conditions(np, process)
        condition_sha = conditions["stable_sha256"]
        result["worker_conditions"] = conditions
    elif kind == "work-bound":
        if fixture is None:
            raise ValueError("work-bound worker requires a fixture")
        result = _work_bound_worker(np, fixture)
        conditions = _runtime_conditions(np, process)
        condition_sha = conditions["stable_sha256"]
        result["worker_conditions"] = conditions
    else:
        raise ValueError(f"unknown worker kind: {kind}")
    protocol_sha256 = _sha256_record(PROTOCOL)
    result["worker"] = {
        "phase": phase,
        "kind": kind,
        "fixture": fixture,
        "timed_boundary": result.get("timed_boundary"),
        "fixture_sha256": _sha256_record(
            {
                "kind": kind,
                "fixture": fixture,
                "protocol_sha256": protocol_sha256,
            }
        ),
        "protocol_sha256": protocol_sha256,
        "harness_sha256": _file_sha256(SCRIPT_PATH),
        "source_commit": _git("rev-parse", "HEAD"),
    }
    result["worker_condition_sha256"] = condition_sha
    return result


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(THREAD_ENVIRONMENT)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT) if not existing else str(ROOT) + os.pathsep + existing
    return environment


def _run_worker(
    kind: str, fixture: str | None, affinity_mask: int, phase: str
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "__worker",
        "--kind",
        kind,
        "--phase",
        phase,
        "--affinity-mask",
        f"0x{affinity_mask:X}",
    ]
    if fixture is not None:
        command.extend(("--fixture", fixture))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError(
            "benchmark worker failed\n"
            + "command: "
            + subprocess.list2cmdline(command)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"worker returned non-JSON output: {completed.stdout!r}") from error


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _provenance() -> dict[str, Any]:
    all_unstaged = _git("diff", "--name-only")
    all_staged = _git("diff", "--cached", "--name-only")
    all_dirty = tuple(
        row for row in (*all_unstaged.splitlines(), *all_staged.splitlines()) if row
    )
    if all_dirty:
        raise RuntimeError(
            "all tracked worktree changes must be committed before capture: "
            f"{all_dirty!r}"
        )
    for relative in RELEVANT_SOURCE_PATHS:
        _git("ls-files", "--error-unmatch", "--", relative)
    unstaged = _git("diff", "--name-only", "--", *RELEVANT_SOURCE_PATHS)
    staged = _git("diff", "--cached", "--name-only", "--", *RELEVANT_SOURCE_PATHS)
    dirty = tuple(row for row in (*unstaged.splitlines(), *staged.splitlines()) if row)
    if dirty:
        raise RuntimeError(f"relevant benchmark sources must be committed: {dirty!r}")
    source_hashes = {
        relative: _file_sha256(ROOT / relative) for relative in RELEVANT_SOURCE_PATHS
    }
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "tracked_worktree_clean": True,
        "relevant_sources_clean": True,
        "source_sha256": source_hashes,
        "harness_sha256": source_hashes["benchmarks/compact_typed_output_encoding.py"],
    }


def _summarize_memory(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != 7:
        raise RuntimeError("memory fixture must contain seven fresh-process samples")
    keys = (
        "tracemalloc_current_delta_bytes",
        "tracemalloc_peak_delta_bytes",
        "working_set_delta_bytes",
        "private_usage_delta_bytes",
    )
    retained = {sample["retained_sha256"] for sample in samples}
    if len(retained) != 1:
        raise RuntimeError("memory fixture retained different scientific objects")
    return {
        "samples": list(samples),
        "medians": {
            key: float(statistics.median(sample[key] for sample in samples))
            for key in keys
        },
        "retained_sha256": next(iter(retained)),
    }


def _without_worker_conditions(record: dict[str, Any]) -> dict[str, Any]:
    clean = dict(record)
    clean.pop("worker_conditions", None)
    clean.pop("worker_condition_sha256", None)
    return clean


def _gate(actual: Any, limit: Any, passed: bool) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "limit": limit}


def _positive_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _compare_artifacts(baseline: dict[str, Any], compact: dict[str, Any], baseline_raw_sha256: str) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    gates["schema"] = _gate(compact["schema"], baseline["schema"], compact["schema"] == baseline["schema"] == SCHEMA)
    gates["protocol"] = _gate(compact["protocol_sha256"], baseline["protocol_sha256"], compact["protocol_sha256"] == baseline["protocol_sha256"])
    gates["environment"] = _gate(compact["environment"]["stable_sha256"], baseline["environment"]["stable_sha256"], compact["environment"]["stable_sha256"] == baseline["environment"]["stable_sha256"])
    gates["harness_source"] = _gate(compact["provenance"]["harness_sha256"], baseline["provenance"]["harness_sha256"], compact["provenance"]["harness_sha256"] == baseline["provenance"]["harness_sha256"])
    gates["golden_parity"] = _gate(compact["golden"]["scientific_sha256"], baseline["golden"]["scientific_sha256"], compact["golden"]["scientific_sha256"] == baseline["golden"]["scientific_sha256"])

    structure = compact["golden"]["structural_storage"]
    gates["encoding_exact_stored_fields"] = _gate(
        structure["encoding_fields"],
        ["factor_winners_le", "physical_digest"],
        structure["encoding_fields"] == ["factor_winners_le", "physical_digest"],
    )
    gates["encoding_no_dense_array"] = _gate(structure["encoding_direct_numpy_array_fields"], [], not structure["encoding_direct_numpy_array_fields"])
    gates["encoding_winner_bytes"] = _gate(structure["factor_winners_bytes"], 72, structure["factor_winners_bytes"] == 72)
    gates["encoding_payload_bytes"] = _gate(structure["compact_stored_payload_bytes"], 104, structure["compact_stored_payload_bytes"] == 104)
    gates["encoding_retired_attributes_absent"] = _gate(
        structure["retired_encoding_attributes_present"],
        [],
        not structure["retired_encoding_attributes_present"],
    )
    gates["encoding_finish_absent"] = _gate(
        structure["bank_finish_present"], False, not structure["bank_finish_present"]
    )
    gates["detector_no_dense_array"] = _gate(structure["detector_direct_numpy_array_fields"], [], not structure["detector_direct_numpy_array_fields"])
    gates["detector_weights_absent"] = _gate(
        structure["detector_weights_present"],
        False,
        not structure["detector_weights_present"],
    )
    gates["detector_target_digest_shape"] = _gate(
        {
            "stored_fields": structure["detector_stored_field_count"],
            "target_fields": structure["detector_target_field_count"],
            "digest_fields": structure["detector_digest_field_count"],
            "other_fields": structure["detector_other_field_count"],
        },
        {
            "stored_fields": 2,
            "target_fields": 1,
            "digest_fields": 1,
            "other_fields": 0,
        },
        structure["detector_stored_field_count"] == 2
        and structure["detector_target_field_count"] == 1
        and structure["detector_digest_field_count"] == 1
        and structure["detector_other_field_count"] == 0,
    )

    memory = compact["measurements"]["memory"]
    baseline_memory = baseline["measurements"]["memory"]
    string_peak = memory["strings_2000_max26"]["medians"]["tracemalloc_peak_delta_bytes"]
    detector_peak = memory["detectors_1000_max26"]["medians"]["tracemalloc_peak_delta_bytes"]
    frontier_peak = memory["string_semantic_frontier_500x4"]["medians"]["tracemalloc_peak_delta_bytes"]
    baseline_string_peak = baseline_memory["strings_2000_max26"]["medians"]["tracemalloc_peak_delta_bytes"]
    baseline_detector_peak = baseline_memory["detectors_1000_max26"]["medians"]["tracemalloc_peak_delta_bytes"]
    baseline_frontier_peak = baseline_memory["string_semantic_frontier_500x4"]["medians"]["tracemalloc_peak_delta_bytes"]
    string_recorded_ratio = _positive_ratio(50.98 * MIB, string_peak)
    string_immediate_ratio = _positive_ratio(baseline_string_peak, string_peak)
    detector_recorded_ratio = _positive_ratio(25.38 * MIB, detector_peak)
    detector_immediate_ratio = _positive_ratio(baseline_detector_peak, detector_peak)
    frontier_ratio = _positive_ratio(frontier_peak, baseline_frontier_peak)
    gates["memory_strings_absolute"] = _gate(string_peak, 2 * MIB, string_peak < 2 * MIB)
    gates["memory_strings_ratio_recorded"] = _gate(string_recorded_ratio, 20.0, string_recorded_ratio is not None and string_recorded_ratio >= 20.0)
    gates["memory_strings_ratio_immediate"] = _gate(string_immediate_ratio, 20.0, string_immediate_ratio is not None and string_immediate_ratio >= 20.0)
    gates["memory_detectors_absolute"] = _gate(detector_peak, MIB, detector_peak < MIB)
    gates["memory_detectors_ratio_recorded"] = _gate(detector_recorded_ratio, 20.0, detector_recorded_ratio is not None and detector_recorded_ratio >= 20.0)
    gates["memory_detectors_ratio_immediate"] = _gate(detector_immediate_ratio, 20.0, detector_immediate_ratio is not None and detector_immediate_ratio >= 20.0)
    gates["memory_frontier_ratio"] = _gate(frontier_ratio, 0.20, frontier_ratio is not None and frontier_ratio <= 0.20)

    runtime = compact["measurements"]["runtime"]
    baseline_runtime = baseline["measurements"]["runtime"]
    for fixture_id in RUNTIME_FIXTURE_IDS:
        gates[f"runtime_result/{fixture_id}"] = _gate(runtime[fixture_id]["result_sha256"], baseline_runtime[fixture_id]["result_sha256"], runtime[fixture_id]["result_sha256"] == baseline_runtime[fixture_id]["result_sha256"])
    grid = runtime["detector_grid_500x4"]["median_seconds"]
    grid_baseline = baseline_runtime["detector_grid_500x4"]["median_seconds"]
    grid_limit = max(0.010, 1.25 * grid_baseline)
    gates["runtime_detector_grid"] = _gate(grid, grid_limit, grid <= grid_limit)
    kernel = runtime["glif_kernel_64"]["median_seconds"]
    kernel_limit = 1.25 * baseline_runtime["glif_kernel_64"]["median_seconds"]
    gates["runtime_glif_kernel"] = _gate(kernel, kernel_limit, kernel <= kernel_limit)
    for fixture_id in SYNTHESIS_FIXTURE_IDS:
        actual = runtime[fixture_id]["median_seconds"]
        limit = 1.10 * baseline_runtime[fixture_id]["median_seconds"]
        gates[f"runtime_synthesis/{fixture_id}"] = _gate(actual, limit, actual <= limit)

    for fixture_id in WORK_BOUND_FIXTURE_IDS:
        current = compact["measurements"]["work_bound"][fixture_id]
        previous = baseline["measurements"]["work_bound"][fixture_id]
        gates[f"work_bound_result/{fixture_id}"] = _gate(current["decision_sha256"], previous["decision_sha256"], current["decision_sha256"] == previous["decision_sha256"])
        gates[f"work_bound_runtime/{fixture_id}"] = _gate(current["wall_seconds"], 60.0, current["wall_seconds"] < 60.0)

    audit_calls = compact["production_audit_materialization_calls"]
    gates["production_audit_materialization"] = _gate(audit_calls, 0, audit_calls == 0)
    passed = all(row["passed"] for row in gates.values())
    return {
        "baseline_raw_sha256": baseline_raw_sha256,
        "gates": gates,
        "passed": passed,
        "verdict": "PASS_COMPACT_TYPED_OUTPUT_ENCODING" if passed else "STOP_COMPACT_TYPED_OUTPUT_ENCODING",
    }


def _artifact_bytes(payload: dict[str, Any]) -> bytes:
    without_sha = dict(payload)
    without_sha.pop("payload_sha256", None)
    final = dict(without_sha)
    final["payload_sha256"] = _sha256_record(without_sha)
    return (json.dumps(_json_safe(final), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_artifact_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_artifact_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_artifact(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("benchmark artifact must contain one JSON object")
    expected = payload.get("payload_sha256")
    without_sha = dict(payload)
    without_sha.pop("payload_sha256", None)
    if type(expected) is not str or expected != _sha256_record(without_sha):
        raise ValueError("benchmark artifact payload hash is invalid")
    return payload, hashlib.sha256(raw).hexdigest()


def _finalize_capture(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    comparison = artifact.get("comparison")
    if artifact.get("phase") == "compact" and (
        not isinstance(comparison, dict) or not comparison.get("passed", False)
    ):
        stopped = dict(artifact)
        stopped["publication"] = {
            "status": "NOT_PUBLISHED_COMPARISON_STOP",
            "requested_output": str(path.resolve()),
        }
        stopped["stop_payload_sha256"] = _sha256_record(stopped)
        return stopped

    artifact["publication"] = {
        "status": (
            "PUBLISHED_COMPACT_PASS"
            if artifact.get("phase") == "compact"
            else "PUBLISHED_DENSE_BASELINE"
        ),
        "requested_output": str(path.resolve()),
    }
    _write_artifact_exclusive(path, artifact)
    written, raw_sha256 = _load_artifact(path)
    written["artifact_raw_sha256"] = raw_sha256
    return written


@contextlib.contextmanager
def _capture_lock(operator_attested_exclusive_host: bool) -> Any:
    if not operator_attested_exclusive_host:
        raise ValueError("operator must attest that the host is benchmark-exclusive")
    lock_path = HOST_LOCK_PATH.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    attestation = {
        "operator_attested_no_other_project_gate_or_benchmark": True,
        "precondition": "no other project gate or benchmark is running on this host",
        "mutex_path": str(lock_path.relative_to(ROOT)),
        "repository_sha256": hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest(),
        "pid": os.getpid(),
        "started_ns": time.time_ns(),
    }
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(attestation, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        yield attestation
    finally:
        lock_path.unlink(missing_ok=True)


def _capture(
    phase: str,
    output_path: Path,
    affinity_mask: int,
    baseline_path: Path | None,
    *,
    operator_attested_exclusive_host: bool,
) -> dict[str, Any]:
    if phase not in ("dense", "compact"):
        raise ValueError("phase must be dense or compact")
    baseline: dict[str, Any] | None = None
    baseline_raw_sha256: str | None = None
    if phase == "dense":
        if baseline_path is not None:
            raise ValueError("dense baseline capture cannot consume a baseline")
    else:
        if baseline_path is None:
            raise ValueError("compact capture requires the immutable dense baseline")
        if output_path.resolve() == baseline_path.resolve():
            raise ValueError("compact output must not overwrite its dense baseline")
        baseline, baseline_raw_sha256 = _load_artifact(baseline_path)
        if baseline.get("phase") != "dense" or baseline.get("schema") != SCHEMA:
            raise ValueError("baseline artifact is not the dense compact-output baseline")
        if baseline.get("protocol_sha256") != _sha256_record(PROTOCOL):
            raise ValueError("dense baseline uses a different frozen benchmark protocol")
        if baseline.get("provenance", {}).get("harness_sha256") != _file_sha256(SCRIPT_PATH):
            raise ValueError("dense baseline was recorded by a different harness source")
        recorded_mask = _parse_affinity_mask(
            baseline["environment"]["stable"]["process"]["affinity_mask"]
        )
        if affinity_mask != recorded_mask:
            raise ValueError("compact capture affinity differs from the dense baseline")

    with _capture_lock(operator_attested_exclusive_host) as host_attestation:
        if output_path.resolve().exists():
            raise FileExistsError(f"artifact already exists: {output_path.resolve()}")
        provenance = _provenance()
        environment = _run_worker("environment", None, affinity_mask, phase)
        golden_raw = _run_worker("golden", None, affinity_mask, phase)
        worker_records = [environment, golden_raw]
        worker_condition_shas = {golden_raw["worker_condition_sha256"]}
        golden = _without_worker_conditions(golden_raw)

        memory: dict[str, Any] = {}
        audit_calls = 0
        for fixture_id in MEMORY_FIXTURE_IDS:
            samples = []
            for _ in range(7):
                raw = _run_worker("memory", fixture_id, affinity_mask, phase)
                worker_records.append(raw)
                worker_condition_shas.add(raw["worker_condition_sha256"])
                audit_calls += int(raw["production_audit_materialization_calls"])
                samples.append(_without_worker_conditions(raw))
            memory[fixture_id] = _summarize_memory(samples)

        runtime: dict[str, Any] = {}
        for fixture_id in RUNTIME_FIXTURE_IDS:
            raw = _run_worker("runtime", fixture_id, affinity_mask, phase)
            worker_records.append(raw)
            worker_condition_shas.add(raw["worker_condition_sha256"])
            audit_calls += int(raw["production_audit_materialization_calls"])
            runtime[fixture_id] = _without_worker_conditions(raw)

        work_bound: dict[str, Any] = {}
        for fixture_id in WORK_BOUND_FIXTURE_IDS:
            raw = _run_worker("work-bound", fixture_id, affinity_mask, phase)
            worker_records.append(raw)
            worker_condition_shas.add(raw["worker_condition_sha256"])
            audit_calls += int(raw["production_audit_materialization_calls"])
            work_bound[fixture_id] = _without_worker_conditions(raw)

        # Full environment capture has dependency/pyproject fields that lean
        # workers intentionally omit; compare the common runtime condition.
        environment_runtime_stable = dict(environment["stable"])
        environment_runtime_stable.pop("installed_distributions_sha256", None)
        environment_runtime_stable.pop("pyproject_sha256", None)
        expected_runtime_condition = _sha256_record(environment_runtime_stable)
        if worker_condition_shas != {expected_runtime_condition}:
            raise RuntimeError(
                "worker environment/power/affinity changed during capture: "
                f"{worker_condition_shas!r} != {expected_runtime_condition!r}"
            )
        expected_worker_source = (
            phase,
            provenance["git_commit"],
            provenance["harness_sha256"],
            _sha256_record(PROTOCOL),
        )
        observed_worker_sources = {
            (
                row["worker"]["phase"],
                row["worker"]["source_commit"],
                row["worker"]["harness_sha256"],
                row["worker"]["protocol_sha256"],
            )
            for row in worker_records
        }
        if observed_worker_sources != {expected_worker_source}:
            raise RuntimeError("worker source/protocol identity changed during capture")
        if _provenance() != provenance:
            raise RuntimeError("relevant source changed during benchmark capture")

        artifact: dict[str, Any] = {
            "schema": SCHEMA,
            "phase": phase,
            "claim_boundary": "typed-output storage/runtime engineering only; no coding capability or protected evaluation evidence",
            "protocol": PROTOCOL,
            "protocol_sha256": _sha256_record(PROTOCOL),
            "command": tuple([sys.executable, *sys.argv]),
            "exclusive_host": host_attestation,
            "provenance": provenance,
            "environment": environment,
            "golden": golden,
            "measurements": {
                "memory": memory,
                "runtime": runtime,
                "work_bound": work_bound,
            },
            "production_audit_materialization_calls": audit_calls,
            "scope_guard": {
                "data_scope": "embedded-synthetic-self-posed",
                "forbidden_module_prefixes": FORBIDDEN_MODULE_PREFIXES,
                "external_task_or_corpus_inputs": 0,
            },
            "comparison": None,
        }
        if baseline is not None and baseline_raw_sha256 is not None:
            artifact["comparison"] = _compare_artifacts(
                baseline, artifact, baseline_raw_sha256
            )
        return _finalize_capture(output_path, artifact)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="record the immutable dense baseline")
    baseline.add_argument("--affinity-mask", required=True)
    baseline.add_argument("--output", required=True, type=Path)
    baseline.add_argument(
        "--attest-exclusive-host",
        required=True,
        action="store_true",
        help="attest that no other project gate or benchmark is running on this host",
    )
    compact = subparsers.add_parser("compact", help="record and compare the compact result")
    compact.add_argument("--baseline", required=True, type=Path)
    compact.add_argument("--output", required=True, type=Path)
    compact.add_argument("--affinity-mask")
    compact.add_argument(
        "--attest-exclusive-host",
        required=True,
        action="store_true",
        help="attest that no other project gate or benchmark is running on this host",
    )
    worker = subparsers.add_parser("__worker", help=argparse.SUPPRESS)
    worker.add_argument("--kind", required=True, choices=("environment", "golden", "memory", "runtime", "work-bound"))
    worker.add_argument("--phase", required=True, choices=("dense", "compact"))
    worker.add_argument("--fixture")
    worker.add_argument("--affinity-mask", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "__worker":
        payload = _worker_payload(
            args.kind,
            args.fixture,
            _parse_affinity_mask(args.affinity_mask),
            args.phase,
        )
        sys.stdout.write(json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n")
        return 0
    if args.command == "baseline":
        affinity = _parse_affinity_mask(args.affinity_mask)
        result = _capture(
            "dense",
            args.output,
            affinity,
            None,
            operator_attested_exclusive_host=args.attest_exclusive_host,
        )
    else:
        baseline, _ = _load_artifact(args.baseline)
        recorded = _parse_affinity_mask(
            baseline["environment"]["stable"]["process"]["affinity_mask"]
        )
        affinity = recorded if args.affinity_mask is None else _parse_affinity_mask(args.affinity_mask)
        result = _capture(
            "compact",
            args.output,
            affinity,
            args.baseline,
            operator_attested_exclusive_host=args.attest_exclusive_host,
        )
    sys.stdout.write(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.command == "compact" and not result["comparison"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
