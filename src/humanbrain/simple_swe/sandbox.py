from __future__ import annotations

import ast
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import cast

from humanbrain.simple_swe.protocol import (
    WORKER_RESULT_OUTCOMES,
    ExecutionLimits,
    Outcome,
)
from humanbrain.simple_swe.tasks import PrivateTask, PublicTask

SANDBOX_PYTHON_FLAGS = ("-I", "-S", "-B")


class SandboxConfigurationError(RuntimeError):
    """The frozen Python runtime or required isolation setup is unavailable."""


@dataclass(frozen=True)
class ExecutionResult:
    outcome: Outcome
    passed: bool
    wall_seconds: float
    detail: str
    sandboxed: bool


@dataclass(frozen=True)
class FrozenWorkerExecution:
    """Bounded output from an externally frozen structured worker."""

    stdout: bytes
    stderr: bytes
    returncode: int | None
    timed_out: bool
    wall_seconds: float
    sandboxed: bool
    detail: str


@dataclass(frozen=True)
class _WorkerObservation:
    stdout: bytes
    stderr: bytes
    returncode: int | None
    timed_out: bool


class _SandboxLaunchError(RuntimeError):
    pass


_DENIED_NODES = (
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.ClassDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
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
_ALLOWED_ATTRIBUTES = frozenset(
    {
        "append",
        "count",
        "endswith",
        "find",
        "index",
        "isalnum",
        "isalpha",
        "isdecimal",
        "isdigit",
        "islower",
        "isspace",
        "isupper",
        "join",
        "lower",
        "lstrip",
        "pop",
        "replace",
        "reverse",
        "rfind",
        "rstrip",
        "split",
        "startswith",
        "strip",
        "upper",
    }
)


def _preflight(
    source: str, task: PublicTask, limits: ExecutionLimits
) -> tuple[Outcome | None, str]:
    if len(source.encode("utf-8")) > limits.max_completion_bytes:
        return "format_error", "completion byte limit exceeded"
    if "```" in source:
        return "format_error", "Markdown fences are forbidden"
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return "syntax_error", "completion is not valid Python"
    for nodes, node in enumerate(ast.walk(tree), start=1):
        if nodes > limits.max_ast_nodes:
            return "format_error", "completion AST-node limit exceeded"
        if isinstance(node, _DENIED_NODES):
            return "policy_violation", f"{type(node).__name__} is forbidden"
        if isinstance(node, ast.Attribute) and node.attr not in _ALLOWED_ATTRIBUTES:
            return "policy_violation", f"attribute {node.attr!r} is not allowlisted"
        if isinstance(node, ast.Name) and (
            node.id in _DENIED_NAMES or node.id.startswith("__")
        ):
            return "policy_violation", f"name {node.id!r} is forbidden"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return "format_error", "completion must contain exactly one function"
    function = tree.body[0]
    if function.name != task.signature.entry_point:
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
    if tuple(argument.arg for argument in arguments.args) != tuple(
        parameter.name for parameter in task.signature.parameters
    ):
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
            "entry_point": public.signature.entry_point,
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


_LAUNCHER_PROBE_SOURCE = (
    "import sys;"
    "print(sys.executable);"
    "print(getattr(sys, '_base_executable', '') or '')"
)


def _redirect_launcher_executable(executable: Path) -> Path:
    """Redirect a launcher executable to the real interpreter it spawns.

    Windows venv ``python.exe`` files can be launchers (uv trampolines) that
    run the actual interpreter as a child process. A Job Object with an
    ``ActiveProcessLimit`` of one forbids that child, so the launcher itself
    must never be selected for sandbox execution. Probing the executable with
    isolated flags reveals which interpreter it really runs; anything
    unprobeable resolves unchanged and later freeze-hash checks fail closed.
    """
    try:
        result = subprocess.run(
            [str(executable), "-I", "-S", "-c", _LAUNCHER_PROBE_SOURCE],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        lines = result.stdout.splitlines()
    except (OSError, ValueError, subprocess.SubprocessError):
        return executable
    for line in lines[:2]:
        candidate_text = line.strip()
        if not candidate_text:
            continue
        try:
            candidate = Path(candidate_text).resolve()
        except (OSError, ValueError):
            continue
        if candidate != executable and candidate.is_file():
            return candidate
    return executable


def resolve_python_executable(value: str | Path | None = None) -> Path:
    """Resolve a direct interpreter, avoiding a venv launcher child process."""
    if value is not None:
        resolved = Path(value).resolve()
    else:
        base_executable = getattr(sys, "_base_executable", None)
        resolved = Path(base_executable or sys.executable).resolve()
    return _redirect_launcher_executable(resolved)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_python_runtime(executable: Path) -> dict[str, object]:
    probe = (
        "import json,sys;"
        "print(json.dumps({'implementation':sys.implementation.name,"
        "'version':list(sys.version_info[:3])},sort_keys=True,separators=(',',':')))"
    )
    try:
        result = subprocess.run(
            [str(executable), "-I", "-S", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        value = json.loads(result.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        raise SandboxConfigurationError("sandbox Python runtime probe failed") from error
    if not isinstance(value, dict):
        raise SandboxConfigurationError("sandbox Python runtime probe is invalid")
    return value


def python_runtime_identity(
    executable: str | Path, language: object
) -> dict[str, str]:
    resolved = Path(executable).resolve()
    if not resolved.is_file():
        raise SandboxConfigurationError("sandbox Python executable is not a file")
    if not isinstance(language, str) or not language.startswith("python-"):
        raise SandboxConfigurationError("protocol language runtime is invalid")
    try:
        expected = tuple(
            int(part) for part in language.removeprefix("python-").split(".")
        )
    except ValueError as error:
        raise SandboxConfigurationError("protocol Python version is invalid") from error
    if len(expected) != 2:
        raise SandboxConfigurationError("protocol must freeze Python major/minor")
    probe = _probe_python_runtime(resolved)
    implementation = probe.get("implementation")
    version = probe.get("version")
    if (
        not isinstance(implementation, str)
        or not isinstance(version, list)
        or len(version) != 3
        or any(isinstance(part, bool) or not isinstance(part, int) for part in version)
    ):
        raise SandboxConfigurationError("sandbox Python runtime identity is invalid")
    if tuple(version[:2]) != expected:
        raise SandboxConfigurationError(
            f"sandbox Python must match protocol {language}, got {version[0]}.{version[1]}"
        )
    return {
        "implementation": implementation,
        "python_version": ".".join(str(part) for part in version),
        "executable_sha256": _sha256_file(resolved),
    }


def load_worker_snapshot(expected_sha256: str) -> bytes:
    worker = Path(__file__).with_name("_worker.py")
    try:
        source = worker.read_bytes()
    except OSError as error:
        raise SandboxConfigurationError("sandbox worker source is unavailable") from error
    if sha256(source).hexdigest() != expected_sha256:
        raise SandboxConfigurationError("sandbox worker differs from its freeze")
    return source


def _runtime_executable_matches(executable: Path, expected_sha256: str) -> bool:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        return False
    try:
        return executable.is_file() and _sha256_file(executable) == expected_sha256
    except OSError:
        return False


def _run_worker_process(
    executable: Path,
    worker_source: bytes,
    input_bytes: bytes,
    limits: ExecutionLimits,
) -> _WorkerObservation:
    job: _WindowsJob | None = None
    with tempfile.TemporaryDirectory(prefix="humanbrain-simple-swe-") as directory:
        temporary_root = Path(directory)
        worker = temporary_root / "worker.py"
        try:
            worker.write_bytes(worker_source)
        except OSError as error:
            raise _SandboxLaunchError(type(error).__name__) from error
        creation_flags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        )
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                [str(executable), *SANDBOX_PYTHON_FLAGS, str(worker)],
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
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if job is not None:
                job.close()
            raise _SandboxLaunchError(type(error).__name__) from error
        try:
            try:
                stdout, stderr = process.communicate(
                    input_bytes,
                    timeout=limits.wall_seconds,
                )
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                if job is not None:
                    job.close()
                if process.poll() is None:
                    process.kill()
                stdout, stderr = process.communicate()
        finally:
            if job is not None:
                job.close()
        return _WorkerObservation(
            stdout=stdout,
            stderr=stderr,
            returncode=process.returncode,
            timed_out=timed_out,
        )


def _decode_worker_result(
    observation: _WorkerObservation,
    limits: ExecutionLimits,
    elapsed: float,
) -> ExecutionResult:
    if observation.timed_out:
        return ExecutionResult("timeout", False, elapsed, "wall time exceeded", True)
    if (
        len(observation.stdout) > limits.output_bytes
        or len(observation.stderr) > limits.output_bytes
    ):
        return ExecutionResult(
            "harness_error", False, elapsed, "sandbox output limit exceeded", True
        )
    if not observation.stdout:
        # A Job Object CPU limit terminates the process before the worker can
        # report. Memory exhaustion normally raises MemoryError in the worker;
        # unknown abrupt termination stays fail-closed.
        memory_return_codes = {
            -1073741801,  # STATUS_NO_MEMORY
            -1073741523,  # STATUS_COMMITMENT_LIMIT
        }
        stderr_text = observation.stderr.decode(
            "utf-8", errors="ignore"
        ).lower()
        if (
            observation.returncode in memory_return_codes
            or "memoryerror" in stderr_text
        ):
            return ExecutionResult(
                "memory_limit", False, elapsed, "process memory limit exceeded", True
            )
        if elapsed >= limits.cpu_seconds * 0.8:
            return ExecutionResult("timeout", False, elapsed, "CPU time exceeded", True)
        return ExecutionResult(
            "harness_error",
            False,
            elapsed,
            f"sandbox exited without a result ({observation.returncode})",
            True,
        )
    try:
        response = json.loads(observation.stdout)
        reported = response["outcome"]
        reported_detail = str(response.get("detail", ""))
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return ExecutionResult(
            "harness_error", False, elapsed, "invalid sandbox result", True
        )
    if not isinstance(reported, str) or reported not in WORKER_RESULT_OUTCOMES:
        return ExecutionResult(
            "harness_error", False, elapsed, "unknown sandbox outcome", True
        )
    outcome = cast(Outcome, reported)
    return ExecutionResult(
        outcome,
        outcome == "passed",
        elapsed,
        reported_detail,
        True,
    )


def execute_frozen_worker(
    input_bytes: bytes,
    limits: ExecutionLimits,
    *,
    python_executable: str | Path | None = None,
    python_executable_sha256: str,
    worker_source: bytes,
    worker_sha256: str,
) -> FrozenWorkerExecution:
    """Run one frozen structured worker with the canonical process boundary."""

    started = perf_counter()
    if (
        type(input_bytes) is not bytes
        or type(worker_source) is not bytes
        or not isinstance(worker_sha256, str)
        or sha256(worker_source).hexdigest() != worker_sha256
    ):
        return FrozenWorkerExecution(
            b"",
            b"",
            None,
            False,
            perf_counter() - started,
            False,
            "sandbox worker snapshot differs from its freeze",
        )
    executable = resolve_python_executable(python_executable)
    if not _runtime_executable_matches(executable, python_executable_sha256):
        return FrozenWorkerExecution(
            b"",
            b"",
            None,
            False,
            perf_counter() - started,
            False,
            "sandbox Python executable differs from its freeze",
        )
    try:
        observation = _run_worker_process(executable, worker_source, input_bytes, limits)
    except _SandboxLaunchError as error:
        return FrozenWorkerExecution(
            b"",
            b"",
            None,
            False,
            perf_counter() - started,
            False,
            f"sandbox unavailable: {error}",
        )
    elapsed = perf_counter() - started
    if not _runtime_executable_matches(executable, python_executable_sha256):
        return FrozenWorkerExecution(
            b"",
            b"",
            observation.returncode,
            observation.timed_out,
            elapsed,
            False,
            "sandbox Python executable changed during execution",
        )
    if observation.timed_out:
        detail = "wall time exceeded"
    elif (
        len(observation.stdout) > limits.output_bytes
        or len(observation.stderr) > limits.output_bytes
    ):
        detail = "sandbox output limit exceeded"
    elif observation.returncode != 0:
        detail = f"sandbox worker exited abnormally ({observation.returncode})"
    else:
        detail = ""
    return FrozenWorkerExecution(
        observation.stdout,
        observation.stderr,
        observation.returncode,
        observation.timed_out,
        elapsed,
        True,
        detail,
    )


def execute_completion(
    source: str,
    public: PublicTask,
    private: PrivateTask,
    limits: ExecutionLimits,
    *,
    python_executable: str | Path | None = None,
    python_executable_sha256: str,
    worker_source: bytes,
    worker_sha256: str,
) -> ExecutionResult:
    """Execute one pure function without exposing hidden tests to the model."""
    started = perf_counter()
    if (
        public.task_id != private.task_id
        or private.public_task_sha256 != public.task_sha256
        or public.hidden_test_sha256 != private.test_sha256
    ):
        return ExecutionResult(
            "harness_error",
            False,
            perf_counter() - started,
            "public/private task commitment mismatch",
            False,
        )
    if (
        not isinstance(worker_source, bytes)
        or not isinstance(worker_sha256, str)
        or sha256(worker_source).hexdigest() != worker_sha256
    ):
        return ExecutionResult(
            "harness_error",
            False,
            perf_counter() - started,
            "sandbox worker snapshot differs from its freeze",
            False,
        )
    executable = resolve_python_executable(python_executable)
    if not _runtime_executable_matches(executable, python_executable_sha256):
        return ExecutionResult(
            "harness_error",
            False,
            perf_counter() - started,
            "sandbox Python executable differs from its freeze",
            False,
        )
    outcome, detail = _preflight(source, public, limits)
    if outcome is not None:
        return ExecutionResult(
            outcome, False, perf_counter() - started, detail, False
        )
    try:
        observation = _run_worker_process(
            executable,
            worker_source,
            _payload(source, public, private, limits),
            limits,
        )
    except _SandboxLaunchError as error:
        return ExecutionResult(
            "harness_error",
            False,
            perf_counter() - started,
            f"sandbox unavailable: {error}",
            False,
        )
    elapsed = perf_counter() - started
    if not _runtime_executable_matches(executable, python_executable_sha256):
        return ExecutionResult(
            "harness_error",
            False,
            elapsed,
            "sandbox Python executable changed during execution",
            True,
        )
    return _decode_worker_result(observation, limits, elapsed)


__all__ = [
    "SANDBOX_PYTHON_FLAGS",
    "ExecutionResult",
    "FrozenWorkerExecution",
    "Outcome",
    "SandboxConfigurationError",
    "execute_completion",
    "execute_frozen_worker",
    "load_worker_snapshot",
    "python_runtime_identity",
    "resolve_python_executable",
]
