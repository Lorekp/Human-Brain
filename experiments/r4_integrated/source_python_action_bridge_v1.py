"""Qualified regular-source action execution using pinned CPython runpy semantics."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated import direct_python_action_bridge_v1 as runtime

LaunchRecord, Case = runtime.LaunchRecord, runtime.Case
POLICY, FLAGS = runtime.POLICY, runtime.FLAGS
discover_launch, load_qualified_launch = runtime.discover_launch, runtime.load_qualified_launch
file_hash = runtime.file_hash
_files_record, escape_literal_content = runtime._files_record, runtime.escape_literal_content
BOOTSTRAP_POLICY = "regular_source_runpy_module_code_v1"
RUNTIME_QUALIFICATION_SHA = "03a33ae6af0488727f7b165346646e89aed801f6f61be93dd0e8b60ca6534dc5"
BOOTSTRAP = (
    'import sys, runpy; '
    'assert sys.flags.utf8_mode == 1; '
    "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
    "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
    "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
    "sys.path.insert(0, '.'); "
    'import io, os\n'
    "with io.open_code(os.path.abspath('main.py')) as source:\n"
    "    code = compile(source.read(), 'main.py', 'exec')\n"
    "runpy._run_module_code(code, mod_name='__main__', pkg_name='', script_name='main.py')"
)
BOOTSTRAP_SHA256 = sha256(BOOTSTRAP.encode("utf-8")).hexdigest()


def load_qualified_bootstrap(path, expected_sha, *, runtime_qualification_path,
                             runtime_qualification_sha, expected_venv):
    """Authenticate independent bootstrap evidence and the unchanged runtime identity."""
    path = Path(path)
    if not path.is_absolute() or file_hash(path) != expected_sha:
        raise ValueError("bootstrap qualification path/hash mismatch")
    value = json.loads(path.read_bytes())
    if (set(value) != {"schema", "status", "runtime_qualification_sha256", "bridge_sha256",
                       "bootstrap_policy", "bootstrap_sha256", "paired_cases",
                       "child_executions", "evidence"}
            or value["schema"] != "source-python-bootstrap-qualification-v1"
            or value["status"] != "passed"
            or runtime_qualification_sha != RUNTIME_QUALIFICATION_SHA
            or value["runtime_qualification_sha256"] != RUNTIME_QUALIFICATION_SHA
            or value["bridge_sha256"] != file_hash(__file__)
            or value["bootstrap_policy"] != BOOTSTRAP_POLICY
            or value["bootstrap_sha256"] != BOOTSTRAP_SHA256
            or value["paired_cases"] != 7 or value["child_executions"] != 14
            or not isinstance(value["evidence"], dict) or not value["evidence"]):
        raise ValueError("bootstrap qualification is incomplete or describes another source")
    for evidence, expected in value["evidence"].items():
        if not Path(evidence).is_absolute() or file_hash(evidence) != expected:
            raise ValueError("bootstrap paired evidence changed")
    return load_qualified_launch(Path(runtime_qualification_path), runtime_qualification_sha,
                                 Path(expected_venv))


def execute_utf8(case: Case, files, before, root: Path, *, launch: LaunchRecord) -> dict:
    """Materialize trusted regular Python source with the qualified module lifecycle."""
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
        "bootstrap_policy": BOOTSTRAP_POLICY, "bootstrap_sha256": BOOTSTRAP_SHA256,
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
