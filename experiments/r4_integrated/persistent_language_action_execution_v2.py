"""Raw action application and bounded real Python verification; no task solver."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

from experiments.r4_integrated import temporary_module_action_bridge_v1 as runtime
from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
from humanbrain.sequence_assembly_observation import (
    SpliceProposal,
    SpliceWorkspace,
    ToolFeedback,
    _check_path,
    _file_digest,
    apply_splice,
)

CHILD_SECONDS = 3.0
OUTPUT_BYTES = 512
SNAPSHOT_FILE_BYTES = 4096
SNAPSHOT_FILES = 16


def qualified_launch(root: Path) -> runtime.LaunchRecord:
    """Authenticate the existing saved child/runtime qualification; launch nothing."""
    root = Path(root).resolve()
    return runtime.load_qualified_bootstrap(
        root / ".pytest-tmp/temporary-module-bootstrap-v1-qualification.json",
        "0bb809fd33aee145f169776a03a884fd62daf381ff4e67ed333a2a378c46d92b",
        runtime_qualification_path=root / ".pytest-tmp/stdlib-python-action-v1-qualification.json",
        runtime_qualification_sha=runtime.RUNTIME_QUALIFICATION_SHA,
        expected_venv=root / ".venv/Scripts/python.exe",
    )


def workspace_from_records(records: list[dict]) -> SpliceWorkspace:
    return SpliceWorkspace(
        tuple(
            OpaqueWorkspaceFile(row["path"], bytes.fromhex(row["content_hex"])) for row in records
        )
    )


def workspace_records(workspace: SpliceWorkspace) -> list[dict]:
    return [{"path": item.path, "content_hex": item.content.hex()} for item in workspace.files]


def proposal_from_record(record: dict, workspace: SpliceWorkspace) -> SpliceProposal:
    """Bind conventional reference outputs without choosing any action field."""
    return SpliceProposal(
        operation=record["operation"],
        path=record["path"],
        start=record["start"],
        end=record["end"],
        replacement=bytes.fromhex(record["replacement_hex"]),
        stop_reason=record["stop_reason"],
        workspace_sha256=record.get("workspace_sha256", workspace.sha256),
    )


def _stop(process: subprocess.Popen) -> None:
    with suppress(ProcessLookupError):
        process.kill()


def _snapshot(root: Path) -> tuple[list[dict], bool]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            return records, False
        if len(records) == SNAPSHOT_FILES or path.stat().st_size > SNAPSHOT_FILE_BYTES:
            return records, False
        with path.open("rb") as handle:
            raw = handle.read(SNAPSHOT_FILE_BYTES + 1)
        if len(raw) > SNAPSHOT_FILE_BYTES:
            return records, False
        records.append({"path": path.relative_to(root).as_posix(), "content_hex": raw.hex()})
    return records, True


def execute_proposal(
    workspace: SpliceWorkspace | list[dict],
    proposal: SpliceProposal | None,
    output: Path,
    *,
    launch: runtime.LaunchRecord,
    deadline: float,
) -> tuple[list[dict], dict]:
    """Execute once, including unchanged verification after STOP/rejected SPLICE.

    Qualification authenticates runtime/bootstrap at the caller's admission
    boundary. This host wrapper changes capture limits, not the child bootstrap.
    It receives no request, correct action, expected output or private grade.
    """
    began = time.perf_counter()
    if not isinstance(launch, runtime.LaunchRecord) or Path(sys.executable) != Path(
        launch.parent_venv_executable
    ):
        raise ValueError("action must retain the qualified parent and base runtime")
    if proposal is not None and not isinstance(proposal, SpliceProposal):
        raise TypeError("execution requires raw workspace and actual proposal")
    if deadline <= began:
        raise TimeoutError("whole-run deadline expired before action")
    if isinstance(workspace, SpliceWorkspace):
        originals = workspace.files
        workspace_sha = workspace.sha256
    elif isinstance(workspace, list) and proposal is None:
        # An actual retry can exceed the neuronal input shape. Verify those
        # complete bytes unchanged instead of truncating or inventing an edit.
        originals = tuple(
            sorted(
                (
                    OpaqueWorkspaceFile(row["path"], bytes.fromhex(row["content_hex"]))
                    for row in workspace
                ),
                key=lambda item: item.path.encode(),
            )
        )
        if len(originals) > SNAPSHOT_FILES or len(
            {item.path.casefold() for item in originals}
        ) != len(originals):
            raise ValueError("raw unchanged verification inventory is not bounded/distinct")
        for item in originals:
            _check_path(item.path)
            if len(item.content) > SNAPSHOT_FILE_BYTES:
                raise ValueError("raw unchanged verification file exceeds snapshot bound")
        workspace_sha = _file_digest(originals)
    else:
        raise TypeError("raw records are permitted only for a no-proposal unchanged verification")
    before = [{"path": item.path, "content_hex": item.content.hex()} for item in originals]
    try:
        if proposal is None:
            raise ValueError("no proposal was produced; verify the unchanged workspace")
        proposed = apply_splice(workspace, proposal)
        accepted, rejection = True, None
    except (ValueError, TypeError) as error:
        proposed = originals
        accepted, rejection = False, f"{type(error).__name__}: {error}"
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    proposed_records = []
    for item in proposed:
        path = (root / item.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("materialized workspace path escapes action root")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(item.content)
        proposed_records.append({"path": item.path, "content_hex": item.content.hex()})
    command = [launch.base_executable, *runtime.FLAGS, "-c", runtime.BOOTSTRAP]
    environment = os.environ.copy()
    environment.pop("__PYVENV_LAUNCHER__", None)
    streams = {
        name: {"bytes": bytearray(), "observed_bytes": 0, "error": None}
        for name in ("stdout", "stderr")
    }
    exceeded = threading.Event()
    timed_out = False
    process = None
    readers = []
    child_started = time.perf_counter()
    child_deadline = min(deadline, child_started + CHILD_SECONDS)
    if child_deadline <= child_started:
        raise TimeoutError("whole-run deadline expired during materialization")
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        def drain(name, pipe):
            try:
                with pipe:
                    while chunk := pipe.read(4096):
                        stream = streams[name]
                        stream["observed_bytes"] += len(chunk)
                        available = OUTPUT_BYTES - len(stream["bytes"])
                        stream["bytes"].extend(chunk[:available])
                        if len(chunk) > available:
                            exceeded.set()
                            _stop(process)
            except (OSError, ValueError) as error:
                streams[name]["error"] = f"{type(error).__name__}: {error}"

        for name in streams:
            reader = threading.Thread(
                target=drain, args=(name, getattr(process, name)), daemon=True
            )
            reader.start()
            readers.append(reader)
        try:
            process.wait(timeout=max(0.001, child_deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop(process)
            process.wait(timeout=1)
    finally:
        if process is not None:
            if process.poll() is None:
                _stop(process)
                process.wait(timeout=1)
            for reader in readers:
                reader.join(timeout=1)
    if process is None:
        raise RuntimeError("child process was not created")
    capture_complete = (
        all(not reader.is_alive() for reader in readers)
        and not exceeded.is_set()
        and all(stream["error"] is None for stream in streams.values())
    )
    after, files_complete = _snapshot(root)
    receipt = {
        "schema": "persistent-language-action-execution-v2",
        "accepted": accepted,
        "rejection": rejection,
        "proposal_sha256": proposal.sha256 if proposal is not None else None,
        "workspace_sha256": workspace_sha,
        "before_files": before,
        "proposed_files": proposed_records,
        "after_files": after,
        "files_complete": files_complete,
        "workspace": str(root),
        "returncode": None if timed_out else process.returncode,
        "timeout": timed_out,
        "output_limit_exceeded": exceeded.is_set(),
        "capture_complete": capture_complete,
        "capture_errors": {
            name: stream["error"] for name, stream in streams.items() if stream["error"] is not None
        },
        "child_pid": process.pid,
        "child_command": command,
        "child_seconds": time.perf_counter() - child_started,
        "child_timeout_seconds": CHILD_SECONDS,
        "output_limit_per_stream": OUTPUT_BYTES,
        "launch_record_sha256": launch.sha256,
        "bootstrap_policy": runtime.BOOTSTRAP_POLICY,
        "bootstrap_sha256": runtime.BOOTSTRAP_SHA256,
        "child_environment_removed": list(runtime.ENVIRONMENT_REMOVED),
        "child_environment_override": {},
        "elapsed_seconds": time.perf_counter() - began,
    }
    for name, stream in streams.items():
        raw = bytes(stream["bytes"])
        receipt[f"{name}_hex"] = raw.hex()
        receipt[f"{name}_sha256"] = hashlib.sha256(raw).hexdigest()
        receipt[f"{name}_observed_bytes"] = stream["observed_bytes"]
    return after, receipt


def feedback_from_receipt(receipt: dict) -> ToolFeedback:
    """Expose exact public result bytes, never a grade or expected response.

    A capped stream stays explicitly incomplete. The observed byte count and
    complete execution custody remain in the original receipt; no prefix is
    silently relabeled a complete diagnostic. Hex decoding is lossless.
    """
    if receipt.get("schema") != "persistent-language-action-execution-v2":
        raise ValueError("feedback requires the current actual execution schema")
    for name in ("stdout", "stderr"):
        raw = bytes.fromhex(receipt[name + "_hex"])
        if hashlib.sha256(raw).hexdigest() != receipt[name + "_sha256"]:
            raise ValueError("actual diagnostic byte hash changed")
        observed = receipt[name + "_observed_bytes"]
        if type(observed) is not int or observed < len(raw):
            raise ValueError("observed diagnostic count is invalid")
        if observed > len(raw) and not receipt["output_limit_exceeded"]:
            raise ValueError("diagnostic bytes were silently omitted")
    if receipt["output_limit_exceeded"] and receipt["capture_complete"]:
        raise ValueError("capped diagnostics cannot be complete")
    return ToolFeedback(
        returncode=receipt["returncode"],
        stdout=bytes.fromhex(receipt["stdout_hex"]),
        stderr=bytes.fromhex(receipt["stderr_hex"]),
        accepted=receipt["accepted"],
        timeout=receipt["timeout"],
        capture_complete=receipt["capture_complete"],
        files_complete=receipt["files_complete"],
        output_limit_exceeded=receipt["output_limit_exceeded"],
    )


def grade_execution(receipt: dict, grade: dict, before_records: list[dict]) -> dict:
    """Private scoring after actual execution; alternate correct edits are valid."""
    before = {row["path"]: row["content_hex"] for row in before_records}
    proposed = {row["path"]: row["content_hex"] for row in receipt["proposed_files"]}
    after = {row["path"]: row["content_hex"] for row in receipt["after_files"]}
    protected = all(
        path in before and proposed.get(path) == before[path] and after.get(path) == before[path]
        for path in grade["protected_paths"]
    )
    behavior = (
        receipt["returncode"] == grade["expected_returncode"]
        and receipt["stdout_hex"] == grade["expected_stdout_hex"]
    )
    complete = (
        receipt["files_complete"]
        and receipt["capture_complete"]
        and not receipt["timeout"]
        and not receipt["output_limit_exceeded"]
        and not receipt["capture_errors"]
    )
    return {
        "task_success": bool(behavior and protected and complete),
        "behavior_correct": behavior,
        "protected_files_unchanged": protected,
        "complete_execution": complete,
    }
