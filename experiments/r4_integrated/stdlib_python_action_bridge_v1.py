"""Fresh standard-library Python actions, separate from the neuronal parent venv."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated import direct_python_action_bridge_v1 as parent_runtime
from experiments.r4_integrated import source_python_action_bridge_v1 as source_bridge

Case = source_bridge.Case
file_hash = source_bridge.file_hash
_files_record = source_bridge._files_record
escape_literal_content = source_bridge.escape_literal_content
BOOTSTRAP = source_bridge.BOOTSTRAP
BOOTSTRAP_POLICY = source_bridge.BOOTSTRAP_POLICY
BOOTSTRAP_SHA256 = source_bridge.BOOTSTRAP_SHA256
POLICY = "base_stdlib_python_no_site_v1"
FLAGS = ("-I", "-S", "-X", "utf8", "-B")
ENVIRONMENT_REMOVED = ("__PYVENV_LAUNCHER__",)


@dataclass(frozen=True)
class LaunchRecord:
    """The parent's identity is distinct from the child flags and base runtime."""

    parent_runtime: parent_runtime.LaunchRecord
    artifacts: tuple[tuple[str, str], ...]

    @property
    def base_executable(self):
        return self.parent_runtime.base_executable

    @property
    def parent_venv_executable(self):
        return self.parent_runtime.venv_executable

    def to_record(self):
        return {
            "policy": POLICY,
            "parent_runtime": self.parent_runtime.to_record(),
            "base_executable": self.base_executable,
            "flags": list(FLAGS),
            "bootstrap_policy": BOOTSTRAP_POLICY,
            "bootstrap_sha256": BOOTSTRAP_SHA256,
            "environment_removed": list(ENVIRONMENT_REMOVED),
            "artifacts": dict(self.artifacts),
        }

    @property
    def sha256(self):
        return sha256(json.dumps(self.to_record(), sort_keys=True,
                                 separators=(",", ":")).encode()).hexdigest()

    def verify(self):
        if self != discover_launch(Path(self.parent_venv_executable)):
            raise ValueError("parent or standard-library action runtime changed")


def discover_launch(expected_venv: Path) -> LaunchRecord:
    """Bind files/current parent only; child behavior requires authored qualification."""
    parent = parent_runtime.discover_launch(expected_venv)
    artifacts = dict(parent.artifacts)
    for filename in (__file__, source_bridge.__file__):
        path = Path(filename).resolve(strict=True)
        artifacts[str(path)] = file_hash(path)
    return LaunchRecord(parent, tuple(sorted(artifacts.items())))


def load_qualified_launch(path: Path, expected_sha: str, expected_venv: Path) -> LaunchRecord:
    """Require the separate identity, semantic and contamination child evidence."""
    path = Path(path)
    if not path.is_absolute() or file_hash(path) != expected_sha:
        raise ValueError("standard-library action qualification path/hash mismatch")
    value = json.loads(path.read_bytes())
    if (set(value) != {"schema", "status", "launch_record", "executions", "identity",
                       "semantic_pairs", "contamination", "evidence"}
            or value["schema"] != "stdlib-python-action-qualification-v1"
            or value["status"] != "passed"
            or value["executions"] != {"identity": 1, "semantic_pairs": 14, "contamination": 2}
            or value["identity"]["passed"] is not True
            or len(value["semantic_pairs"]) != 7
            or any(row["passed"] is not True for row in value["semantic_pairs"])
            or value["contamination"]["passed"] is not True
            or not isinstance(value["evidence"], dict) or not value["evidence"]):
        raise ValueError("standard-library action qualification is incomplete")
    for evidence, expected in value["evidence"].items():
        if not Path(evidence).is_absolute() or file_hash(evidence) != expected:
            raise ValueError("standard-library action qualification evidence changed")
    launch = discover_launch(expected_venv)
    if value["launch_record"] != launch.to_record():
        raise ValueError("standard-library qualification describes another runtime")
    return launch


def execute_utf8(case: Case, files, before, root: Path, *, launch: LaunchRecord) -> dict:
    """Materialize trusted regular Python source with the qualified module lifecycle."""
    began = time.perf_counter()
    if not isinstance(launch, LaunchRecord):
        raise TypeError("an explicit qualified launch record is required")
    if Path(sys.executable) != Path(launch.parent_venv_executable):
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
    environment.pop("__PYVENV_LAUNCHER__", None)
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
        "bootstrap_policy": BOOTSTRAP_POLICY, "bootstrap_sha256": BOOTSTRAP_SHA256,
        "child_command": command,
        "child_environment_override": {},
        "child_environment_removed": list(ENVIRONMENT_REMOVED),
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
