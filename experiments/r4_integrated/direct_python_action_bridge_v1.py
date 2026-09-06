"""Qualified direct CPython launch preserving the selected Windows venv context."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.neutral_document_python_session import Case
from experiments.r4_integrated.real_input_action_bridge import (
    _files_record,
    escape_literal_content,
)

POLICY = "direct_base_python_with_selected_venv_launcher_v1"
FLAGS = ("-I", "-X", "utf8", "-B")
BOOTSTRAP = (
    "import sys, runpy; assert sys.flags.utf8_mode == 1; "
    "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
    "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
    "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
    "sys.path.insert(0, '.'); "
    "runpy.run_path('main.py', run_name='__main__')"
)


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class LaunchRecord:
    """Immutable measured-environment identity, with no callable launch policy."""

    venv_executable: str
    base_executable: str
    venv_config: str
    python_version: str
    implementation: str
    base_prefix: str
    base_exec_prefix: str
    artifacts: tuple[tuple[str, str], ...]

    def to_record(self):
        return {
            "policy": POLICY,
            "venv_executable": self.venv_executable,
            "base_executable": self.base_executable,
            "venv_config": self.venv_config,
            "python_version": self.python_version,
            "implementation": self.implementation,
            "base_prefix": self.base_prefix,
            "base_exec_prefix": self.base_exec_prefix,
            "artifacts": dict(self.artifacts),
        }

    @property
    def sha256(self):
        return sha256(json.dumps(self.to_record(), sort_keys=True,
                                 separators=(",", ":")).encode()).hexdigest()

    def verify(self):
        """Boundary authentication; call before and after a complete workload."""
        if discover_launch(Path(self.venv_executable)) != self:
            raise ValueError("qualified action environment changed")


def discover_launch(expected_venv: Path) -> LaunchRecord:
    """Inspect this selected runtime and fixed files without launching a child."""
    expected = Path(expected_venv)
    if not expected.is_absolute() or sys.platform != "win32":
        raise ValueError("direct action launch requires an absolute selected Windows venv")
    # CPython's own bypass preserves the runtime spelling. Resolving uv aliases
    # here changes base-prefix/sys.path even when both names identify one image.
    venv = Path(sys.executable)
    if (not venv.is_absolute() or venv.resolve(strict=True) != expected.resolve(strict=True)
            or sys.prefix == sys.base_prefix):
        raise ValueError("action runtime is not the selected virtual environment")
    base = Path(sys._base_executable)
    config = venv.parent.parent / "pyvenv.cfg"
    entries = dict(line.split(" = ", 1) for line in config.read_text().splitlines()
                   if " = " in line)
    if (not base.is_absolute() or base.resolve(strict=True) == venv.resolve(strict=True)
            or Path(entries["home"]).resolve(strict=True) != base.parent.resolve(strict=True)):
        raise ValueError("base executable does not match selected venv configuration")
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        raise ValueError("action launch requires the qualified CPython 3.11 runtime")
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    paths = [venv, base, config, Path(__file__).resolve(), base.parent / "python311.dll"]
    paths.extend(stdlib / name for name in (
        "runpy.py", "site.py", "subprocess.py", "encodings/__init__.py",
        "multiprocessing/popen_spawn_win32.py",
    ))
    return LaunchRecord(
        str(venv), str(base), str(config), sys.version, sys.implementation.name,
        sys.base_prefix, sys.base_exec_prefix,
        tuple(sorted((str(path.resolve(strict=True)), file_hash(path)) for path in paths)),
    )


def load_qualified_launch(path: Path, expected_sha: str, expected_venv: Path) -> LaunchRecord:
    """Admitted runs require the independently bound authored parity receipt."""
    path = Path(path)
    if not path.is_absolute() or file_hash(path) != expected_sha:
        raise ValueError("action qualification path/hash mismatch")
    qualification = json.loads(path.read_bytes())
    if (qualification["schema"] != "direct-python-action-qualification-v1"
            or qualification["status"] != "passed"
            or qualification["identity_pair"]["passed"] is not True
            or len(qualification["output_pairs"]) != 4
            or any(item["passed"] is not True for item in qualification["output_pairs"])
            or qualification["executions"] != {"identity": 2, "outputs": 8}
            or not qualification["evidence"]):
        raise ValueError("action parity qualification is incomplete")
    for evidence, expected in qualification["evidence"].items():
        if not Path(evidence).is_absolute() or file_hash(evidence) != expected:
            raise ValueError("action parity evidence changed")
    launch = discover_launch(expected_venv)
    if launch.to_record() != qualification["launch_record"]:
        raise ValueError("action parity qualification describes another environment")
    return launch


def execute_utf8(case: Case, files, before, root: Path, *, launch: LaunchRecord) -> dict:
    """Materialize trusted generated files; bypass only the venv redirect process."""
    began = time.perf_counter()
    if not isinstance(launch, LaunchRecord):
        raise TypeError("an explicit qualified launch record is required")
    if Path(sys.executable) != Path(launch.venv_executable):
        raise ValueError("action parent left the qualified virtual environment")
    root.mkdir(parents=True, exist_ok=False)
    for item in files:
        path = (root / item.path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("workspace path escapes execution directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(item.content)
    command = [launch.base_executable, *FLAGS, "-c", BOOTSTRAP]
    environment = os.environ.copy()
    environment["__PYVENV_LAUNCHER__"] = launch.venv_executable
    try:
        completed = subprocess.run(command, cwd=root, env=environment,
                                   capture_output=True, timeout=10, check=False)
        returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        returncode, stdout, stderr = None, error.stdout or b"", error.stderr or b""
    same_paths = tuple(item.path for item in files) == tuple(item.path for item in before)
    untouched = same_paths and all(
        item.content == original.content
        for item, original in zip(files, before, strict=True)
        if original.path != f"{case.module}.py"
    )
    materialized = all((root / item.path).read_bytes() == item.content for item in files)
    decoded, decode_errors = {}, {}
    for name, raw in (("stdout", stdout), ("stderr", stderr)):
        try:
            decoded[name] = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            decoded[name] = None
            decode_errors[name] = str(error)
    return {
        "returncode": returncode, "stdout": decoded["stdout"], "stderr": decoded["stderr"],
        "stdout_hex": stdout.hex(), "stderr_hex": stderr.hex(),
        "stdout_sha256": sha256(stdout).hexdigest(), "stderr_sha256": sha256(stderr).hexdigest(),
        "output_encoding": "utf-8", "decode_errors": decode_errors,
        "child_python_flags": list(FLAGS), "decoys_unchanged": untouched,
        "materialized_matches_action": materialized,
        "passed": returncode == 0 and not decode_errors and stdout.strip() == case.value.encode()
        and untouched and materialized,
        "workspace": str(root), "elapsed_seconds": time.perf_counter() - began,
        "launch_record_sha256": launch.sha256,
        "child_command": command,
        "child_environment_override": {"__PYVENV_LAUNCHER__": launch.venv_executable},
    }


def actuate(generated, ordinal, motor, root, *, launch: LaunchRecord):
    """Only generated bytes enter the existing motor; no target argument exists."""
    raw = bytes.fromhex(generated["generated_hex"])
    reason = generated["stop_reason"]
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        reason = "invalid_utf8"
    accepted = reason == "eos" and bool(raw)
    value = raw if accepted else b"pending"
    case = Case(f"generated-{ordinal}", f"qa{ordinal:03d}", value.decode(),
                f"qa_config{ordinal:03d}", "pending")
    before = case.files()
    motor_before = motor.checkpoint_sha256
    action = (motor.apply(f"Configure {case.key}".encode(), before, escape_literal_content(raw))
              if accepted else None)
    after = before if action is None else action.files
    receipt = execute_utf8(case, after, before, root, launch=launch)
    assert motor.checkpoint_sha256 == motor_before
    lossless = (receipt["returncode"] == 0
                and receipt["stdout_hex"] == (value + b"\n").hex()
                and receipt["stderr_hex"] == "" and receipt["decoys_unchanged"]
                and receipt["materialized_matches_action"])
    return {
        "generated_hex": raw.hex(), "generation_status": reason, "emitted": accepted,
        "motor_checkpoint_sha256": motor_before, "before": _files_record(before),
        "after": _files_record(after), "lossless_generated_execution": lossless,
        "execution": receipt,
    }
