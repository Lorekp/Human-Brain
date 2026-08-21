from __future__ import annotations

import ast
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from humanbrain.simple_swe.protocol import ExecutionLimits
from humanbrain.simple_swe.tasks import PrivateTask, PublicTask


Outcome = Literal[
    "format_error",
    "syntax_error",
    "policy_violation",
    "signature_error",
    "timeout",
    "memory_limit",
    "runtime_error",
    "regression_failure",
    "property_failure",
    "harness_error",
    "passed",
]


@dataclass(frozen=True)
class ExecutionResult:
    outcome: Outcome
    passed: bool
    wall_seconds: float
    detail: str
    sandboxed: bool


_DENIED_NODES = (
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.ClassDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.With,
)
_DENIED_NAMES = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}


def _preflight(source: str, task: PublicTask) -> tuple[Outcome | None, str]:
    if "```" in source:
        return "format_error", "Markdown fences are forbidden"
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return "syntax_error", "completion is not valid Python"
    for node in ast.walk(tree):
        if isinstance(node, _DENIED_NODES):
            return "policy_violation", f"{type(node).__name__} is forbidden"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return "policy_violation", "private or dunder attribute access is forbidden"
        if isinstance(node, ast.Name) and (
            node.id in _DENIED_NAMES or node.id.startswith("__")
        ):
            return "policy_violation", f"name {node.id!r} is forbidden"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return "format_error", "completion must contain exactly one function"
    function = tree.body[0]
    if function.name != task.entry_point:
        return "signature_error", "entry-point name differs from the task"
    if function.decorator_list or function.returns is not None:
        return "signature_error", "decorators and annotations are forbidden"
    arguments = function.args
    if (
        arguments.posonlyargs
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kw_defaults
        or arguments.kwarg is not None
        or arguments.defaults
        or any(argument.annotation is not None for argument in arguments.args)
    ):
        return "signature_error", "entry point must use the exact plain signature"
    if tuple(argument.arg for argument in arguments.args) != task.parameter_names:
        return "signature_error", "entry-point parameters differ from the task"
    nested_functions = [
        node for node in ast.walk(function) if isinstance(node, ast.FunctionDef)
    ]
    if len(nested_functions) != 1:
        return "policy_violation", "nested functions are forbidden"
    return None, ""


class _WindowsJob:
    _JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    _JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    _EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self, limits: ExecutionLimits) -> None:
        self._handle: int | None = None
        self._limits = limits

    def attach(self, process: subprocess.Popen[bytes]) -> None:
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        self._handle = int(handle)
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_PROCESS_TIME
            | self._JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | self._JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | self._JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        )
        information.BasicLimitInformation.PerProcessUserTimeLimit = (
            self._limits.cpu_seconds * 10_000_000
        )
        information.BasicLimitInformation.ActiveProcessLimit = self._limits.processes
        information.ProcessMemoryLimit = self._limits.memory_mib * 1024 * 1024
        if not kernel32.SetInformationJobObject(
            wintypes.HANDLE(self._handle),
            self._EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "SetInformationJobObject failed")
        process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
        if not kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(self._handle), process_handle
        ):
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "AssignProcessToJobObject failed")

    def close(self) -> None:
        if self._handle is None:
            return
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(wintypes.HANDLE(self._handle))
        self._handle = None


def _sanitized_environment(temporary_root: Path, limits: ExecutionLimits) -> dict[str, str]:
    environment = {
        "PYTHONHASHSEED": str(limits.python_hash_seed),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEMP": str(temporary_root),
        "TMP": str(temporary_root),
    }
    if os.name == "nt":
        for name in ("SYSTEMROOT", "WINDIR"):
            if value := os.environ.get(name):
                environment[name] = value
    return environment


def _payload(
    source: str,
    public: PublicTask,
    private: PrivateTask,
    limits: ExecutionLimits,
) -> bytes:
    return json.dumps(
        {
            "source": source,
            "entry_point": public.entry_point,
            "regression_cases": [asdict(case) for case in private.regression_cases],
            "property_cases": [asdict(case) for case in private.property_cases],
            "limits": {
                "cpu_seconds": limits.cpu_seconds,
                "memory_bytes": limits.memory_mib * 1024 * 1024,
                "output_bytes": limits.output_bytes,
                "processes": limits.processes,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def execute_completion(
    source: str,
    public: PublicTask,
    private: PrivateTask,
    limits: ExecutionLimits,
    *,
    python_executable: str | Path | None = None,
) -> ExecutionResult:
    """Execute one pure function without exposing hidden tests to the model."""
    started = perf_counter()
    if public.task_id != private.task_id:
        return ExecutionResult(
            "harness_error", False, perf_counter() - started, "task id mismatch", False
        )
    outcome, detail = _preflight(source, public)
    if outcome is not None:
        return ExecutionResult(
            outcome, False, perf_counter() - started, detail, False
        )
    executable = Path(python_executable or sys.executable).resolve()
    worker_source = Path(__file__).with_name("_worker.py")
    job: _WindowsJob | None = None
    with tempfile.TemporaryDirectory(prefix="humanbrain-simple-swe-") as directory:
        temporary_root = Path(directory)
        worker = temporary_root / "worker.py"
        shutil.copyfile(worker_source, worker)
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                [str(executable), "-S", "-B", str(worker)],
                cwd=temporary_root,
                env=_sanitized_environment(temporary_root, limits),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
                start_new_session=os.name != "nt",
            )
            if os.name == "nt":
                job = _WindowsJob(limits)
                job.attach(process)
        except (OSError, ValueError) as error:
            if "process" in locals() and process.poll() is None:
                process.kill()
                process.wait()
            if job is not None:
                job.close()
            return ExecutionResult(
                "harness_error",
                False,
                perf_counter() - started,
                f"sandbox unavailable: {type(error).__name__}",
                False,
            )
        try:
            stdout, stderr = process.communicate(
                _payload(source, public, private, limits),
                timeout=limits.wall_seconds,
            )
        except subprocess.TimeoutExpired:
            if job is not None:
                job.close()
            if process.poll() is None:
                process.kill()
            process.communicate()
            return ExecutionResult(
                "timeout",
                False,
                perf_counter() - started,
                "wall time exceeded",
                True,
            )
        finally:
            if job is not None:
                job.close()
        elapsed = perf_counter() - started
        if len(stdout) > limits.output_bytes or len(stderr) > limits.output_bytes:
            return ExecutionResult(
                "harness_error", False, elapsed, "sandbox output limit exceeded", True
            )
        if not stdout:
            # A Job Object CPU limit terminates the process before the worker can
            # report. Memory exhaustion normally raises MemoryError in the worker;
            # unknown abrupt termination stays fail-closed.
            if elapsed >= limits.cpu_seconds * 0.8:
                return ExecutionResult(
                    "timeout", False, elapsed, "CPU time exceeded", True
                )
            return ExecutionResult(
                "harness_error",
                False,
                elapsed,
                f"sandbox exited without a result ({process.returncode})",
                True,
            )
        try:
            response = json.loads(stdout)
            reported = response["outcome"]
            reported_detail = str(response.get("detail", ""))
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            return ExecutionResult(
                "harness_error", False, elapsed, "invalid sandbox result", True
            )
        allowed: set[str] = {
            "signature_error",
            "timeout",
            "memory_limit",
            "runtime_error",
            "regression_failure",
            "property_failure",
            "harness_error",
            "passed",
        }
        if reported not in allowed:
            return ExecutionResult(
                "harness_error", False, elapsed, "unknown sandbox outcome", True
            )
        return ExecutionResult(
            reported,
            reported == "passed",
            elapsed,
            reported_detail,
            True,
        )


__all__ = ["ExecutionResult", "Outcome", "execute_completion"]
