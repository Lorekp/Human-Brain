"""Parent-only export and qualified execution for the isolated splice worker."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from experiments.r4_integrated.neuronal_splice_io_v1 import (
    process_identity,
    read_row,
    write_message,
)
from experiments.r4_integrated.neuronal_splice_worker_v1 import (
    ARMS,
    IPC_SCHEMA,
    LIMITS,
    SOURCE_MODULES,
    WORKER_SOURCE,
    file_hash,
    write_json,
)


def tensor_bytes(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    if type(value) not in (str, int, float, bool, type(None)):
        raise TypeError("checkpoint contains an undeclared value type")
    return 0


def check_proposal(message, arm, index, task, model_hash, memory_hash):
    from experiments.r4_integrated.neuronal_splice_online_v1 import proposal_from_record

    if (not isinstance(message, dict)
            or set(message) != {"schema", "type", "arm", "index", "id", "attempt_id",
                                "prediction", "model_seconds"}
            or message["schema"] != IPC_SCHEMA or message["type"] != "proposal"
            or message["arm"] != arm or message["index"] != index
            or message["id"] != task["task_id"]):
        raise ValueError("worker proposal order or schema mismatch")
    prediction = message["prediction"]
    proposal = proposal_from_record(prediction["proposal"])
    if (prediction["attempt_id"] != message["attempt_id"]
            or prediction["proposal_sha256"] != proposal.sha256
            or prediction["model_sha256"] != model_hash
            or prediction["memory_sha256"] != memory_hash
            or proposal.workspace_sha256 != task["workspace_sha256"]):
        raise ValueError("worker proposal is not bound to the exported state")
    return proposal


def check_acknowledgement(message, identity, action_hash, model_hash, memory_hash):
    if (not isinstance(message, dict)
            or set(message) != {*identity, "action_sha256", "model_sha256", "memory_sha256",
                                "work", "model_seconds"}
            or any(message[key] != value for key, value in identity.items() if key != "type")
            or message["type"] != "acknowledged"
            or message["action_sha256"] != action_hash
            or message["model_sha256"] != model_hash
            or message["memory_sha256"] != memory_hash
            or message["work"].get("reader") != {}):
        raise ValueError("worker did not acknowledge this actual action unchanged")


def export_and_evaluate(root, output, models, states, roster, launch, action_journal, deadline):
    """Export once, execute each actual worker proposal, return saved custody evidence."""
    import torch

    from experiments.r4_integrated.neuronal_splice_online_v1 import (
        execute_action,
        workspace_from_records,
    )
    from humanbrain.neuronal_splice_learner import _digest

    started = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    isolated = output / "endpoint"
    isolated.mkdir(parents=True, exist_ok=False)
    package = isolated / "package"
    package.mkdir()
    sources = {}
    for relative in (*SOURCE_MODULES.values(), WORKER_SOURCE):
        source, destination = root / relative, package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        sources[relative] = file_hash(source)
        shutil.copyfile(source, destination)
        if file_hash(destination) != sources[relative]:
            raise ValueError("isolated source copy changed")
    task_ids = roster["fresh_task_ids"] + roster["retention_task_ids"]
    if len(task_ids) != 48 or set(models) != set(ARMS) or set(states) != set(ARMS):
        raise ValueError("endpoint roster/model scope mismatch")
    cases = []
    for task_id in task_ids:
        task = roster["tasks"][task_id]
        cases.append(dict(id=task_id, memory_id=task["memory_id"],
                          request_hex=task["request_hex"], files=task["files"]))
    reading = [dict(id=str(case["case_id"]), memory_id=case["memory_id"],
                    request_hex=case["query_hex"]) for case in roster["old_panel"]]
    if len(reading) != 20:
        raise ValueError("endpoint reading panel scope mismatch")
    payload, initial_hashes, custody_work = {}, {}, {}
    for arm in ARMS:
        model = models[arm]
        if model._pending is not None or len(states[arm]) != 64:
            raise ValueError("export requires idle models and all retained states")
        weights = model.state_dict()
        memories = {key: value.snapshot() for key, value in states[arm].items()}
        payload[arm] = dict(weights=weights, memories=memories, experience=model._experience)
        initial_hashes[arm] = dict(
            model_sha256=_digest(weights, custody_work),
            memories_sha256={key: _digest(value, custody_work) for key, value in memories.items()},
            experience=model._experience)
    payload = dict(schema="neuronal-splice-endpoint-state-v1", arms=payload)
    checkpoint_bytes = tensor_bytes(payload)
    if checkpoint_bytes != 170705152:
        raise ValueError("endpoint tensor payload differs from the fixed full scope")
    checkpoint = isolated / "source_free.pt"
    with checkpoint.open("xb") as stream:
        torch.save(payload, stream)
    del payload, weights, memories
    checkpoint_sha = file_hash(checkpoint)
    requests = dict(schema="neuronal-splice-endpoint-request-v1",
                    checkpoint_sha256=checkpoint_sha, source_sha256=sources,
                    receipt_limits=LIMITS, arms=list(ARMS), cases=cases, reading_cases=reading)
    request_path = isolated / "query_only.json"
    write_json(request_path, requests)
    requests_sha = file_hash(request_path)
    worker_output = isolated / "worker"
    command = [sys.executable, "-I", "-X", "utf8", str(package / WORKER_SOURCE),
               "--package-root", str(package), "--checkpoint", str(checkpoint),
               "--checkpoint-sha", checkpoint_sha, "--requests", str(request_path),
               "--requests-sha", requests_sha, "--output", str(worker_output),
               "--deadline-unix", repr(deadline)]
    spec = SimpleNamespace(config={"receipt_limits": LIMITS})
    stderr_path = isolated / "worker-stderr.bin"
    launch_record = dict(command=command, cwd=str(isolated), started_unix=time.time(),
                         parent_pid=os.getpid(), checkpoint_sha256=checkpoint_sha,
                         requests_sha256=requests_sha)
    actions, canonical, pid = {}, {}, None
    worker_started = time.perf_counter()
    with stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, cwd=isolated, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=stderr)
        try:
            pid = process.pid
            launch_record["pid"] = pid
            try:
                launch_record["process_identity"] = process_identity(pid)
            except OSError as error:
                launch_record["process_identity"] = None
                launch_record["identity_read_error"] = str(error)[:1024]
            write_json(isolated / "worker-launch.json", launch_record)
            for arm in ARMS:
                actions[arm], canonical[arm] = [], []
                for index, task_id in enumerate(task_ids):
                    if time.time() >= deadline:
                        raise TimeoutError("parent endpoint deadline")
                    task = roster["tasks"][task_id]
                    message = read_row(process.stdout, spec)
                    model_hash = initial_hashes[arm]["model_sha256"]
                    memory_hash = initial_hashes[arm]["memories_sha256"][task["memory_id"]]
                    proposal = check_proposal(message, arm, index, task, model_hash, memory_hash)
                    identity = {key: message[key] for key in
                                ("schema", "type", "arm", "index", "id", "attempt_id")}
                    del message
                    envelope, action_hash, _, grade = execute_action(
                        task=task, workspace=workspace_from_records(task["files"]),
                        proposal=proposal, attempt_id=identity["attempt_id"], phase="endpoint",
                        arm=arm, index=index, output=output, launch=launch, journal=action_journal)
                    write_message(process.stdin, {**identity, "type": "executed",
                                                  "action_sha256": action_hash,
                                                  "outcome": envelope}, spec)
                    del envelope, proposal
                    acknowledged = read_row(process.stdout, spec)
                    check_acknowledgement(
                        acknowledged, identity, action_hash, model_hash, memory_hash)
                    del acknowledged
                    actions[arm].append(bool(grade["task_success"]))
                    canonical[arm].append(bool(grade["canonical_tuple_correct"]))
            process.stdin.close()
            process.wait(timeout=max(.001, deadline - time.time()))
            if process.returncode != 0 or read_row(process.stdout, spec) is not None:
                raise ValueError("worker failed or emitted an extra protocol row")
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()
    worker_seconds = time.perf_counter() - worker_started
    if stderr_path.stat().st_size > 1024:
        raise ValueError("worker diagnostics exceed the declared bound")
    completion_path = worker_output / "completion.json"
    completion = json.loads(completion_path.read_bytes())
    if (completion["schema"] != "neuronal-splice-endpoint-completion-v1"
            or completion["status"] != "passed"
            or completion["scope"] != "integrated"
            or not (completion["pid"] == pid or completion["parent_pid"] == pid)
            or completion["process_identity"]["pid"] != completion["pid"]
            or completion["checkpoint_sha256"] != checkpoint_sha
            or completion["requests_sha256"] != requests_sha
            or completion["source_sha256"] != sources):
        raise ValueError("worker completion bindings mismatch")
    expected_origins = {name: str(package / relative) for name, relative in SOURCE_MODULES.items()}
    if completion["source_origins"] != expected_origins:
        raise ValueError("worker imported another source inventory")
    for name, filename, rows in (("endpoint_log", "endpoints.jsonl", 384),
                                  ("reading_log", "reading.jsonl", 80)):
        log = completion[name]
        path = worker_output / filename
        if (set(log) != {"path", "rows", "bytes", "sha256", "max_line_bytes",
                         "max_row_tree_bytes"}
                or log["path"] != filename or log["rows"] != rows
                or path.stat().st_size != log["bytes"] or file_hash(path) != log["sha256"]
                or not 0 < log["max_line_bytes"] <= LIMITS["max_line_bytes"]
                or not 0 < log["max_row_tree_bytes"] <= LIMITS["max_row_tree_bytes"]):
            raise ValueError("worker receipt log changed or exceeded its bound")
    for arm in ARMS:
        final = completion["final"][arm]
        expected = initial_hashes[arm]
        if (final["model_sha256"] != expected["model_sha256"]
                or final["memories_sha256"] != expected["memories_sha256"]
                or final["experience"] != expected["experience"] + 48):
            raise ValueError("worker final state changed")
        if (_digest(models[arm].state_dict(), custody_work) != expected["model_sha256"]
                or {key: _digest(state.snapshot(), custody_work)
                    for key, state in states[arm].items()} != expected["memories_sha256"]):
            raise ValueError("parent state changed during endpoint evaluation")
    for relative, expected_sha in sources.items():
        if (file_hash(root / relative) != expected_sha
                or file_hash(package / relative) != expected_sha):
            raise ValueError("source bytes changed during endpoint evaluation")
    if file_hash(checkpoint) != checkpoint_sha or file_hash(request_path) != requests_sha:
        raise ValueError("exported inputs changed during endpoint evaluation")
    return dict(schema="neuronal-splice-endpoint-parent-v1", worker=completion,
                worker_completion_sha256=file_hash(completion_path),
                worker_launch_sha256=file_hash(isolated / "worker-launch.json"),
                checkpoint_sha256=checkpoint_sha, requests_sha256=requests_sha,
                checkpoint_tensor_bytes=checkpoint_bytes, initial_hashes=initial_hashes,
                source_sha256=sources, task_success=actions, canonical_tuple_correct=canonical,
                worker_stderr_sha256=file_hash(stderr_path), custody_work=custody_work,
                worker_wall_seconds=worker_seconds,
                package_export_exchange_seconds=time.perf_counter() - started)
