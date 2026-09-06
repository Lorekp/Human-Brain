"""Teacher-free endpoint proposals with parent-owned actual tool execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ARMS = ("learned", "action_frozen", "slow_frozen", "memory_disabled")
IPC_SCHEMA = "neuronal-splice-endpoint-ipc-v1"
SOURCE_MODULES = {
    "humanbrain": "src/humanbrain/__init__.py",
    "humanbrain.neuronal_splice_learner": "src/humanbrain/neuronal_splice_learner.py",
    "humanbrain.neuronal_splice_action": "src/humanbrain/neuronal_splice_action.py",
    "humanbrain.local_memory_learner": "src/humanbrain/local_memory_learner.py",
    "humanbrain.neuronal_sequence_memory": "src/humanbrain/neuronal_sequence_memory.py",
    "humanbrain.byte_sequence_decoder": "src/humanbrain/byte_sequence_decoder.py",
    "humanbrain.plastic_memory": "src/humanbrain/plastic_memory.py",
    "humanbrain.multiscale_opaque_event_tape": "src/humanbrain/multiscale_opaque_event_tape.py",
    "experiments.r4_integrated": "experiments/r4_integrated/__init__.py",
    "experiments.r4_integrated.neuronal_splice_io_v1":
        "experiments/r4_integrated/neuronal_splice_io_v1.py",
    "experiments.r4_integrated.neuronal_splice_receipts_v1":
        "experiments/r4_integrated/neuronal_splice_receipts_v1.py",
}
WORKER_SOURCE = "experiments/r4_integrated/neuronal_splice_worker_v1.py"
LIMITS = {"max_line_bytes": 1048576, "max_row_tree_bytes": 1048576,
          "max_live_rows": 2, "max_line_copies": 2, "allocation_factor": 2}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write("\n")


def checked_hex(value):
    if not isinstance(value, str):
        raise ValueError("raw input must be canonical hex")
    result = bytes.fromhex(value)
    if result.hex() != value:
        raise ValueError("raw input must be canonical hex")
    return result


def check_sources(root, expected):
    if set(expected) != {*SOURCE_MODULES.values(), WORKER_SOURCE}:
        raise ValueError("worker requires its exact teacher-free source inventory")
    if Path(__file__).resolve() != root / WORKER_SOURCE:
        raise ValueError("worker must execute its supplied isolated source copy")
    actual_sources = {path.relative_to(root).as_posix() for path in root.rglob("*.py")}
    if actual_sources != set(expected):
        raise ValueError("isolated package contains an undeclared source file")
    for relative, expected_sha in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected_sha:
            raise ValueError("worker source identity mismatch")


def check_origins(root):
    origins = {}
    for name, relative in SOURCE_MODULES.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if path != root / relative:
            raise ValueError(f"worker module escaped isolated inventory: {name}")
        origins[name] = str(path)
    allowed = set(SOURCE_MODULES) | {"experiments"}
    for name in sys.modules:
        project_module = (name == "humanbrain" or name.startswith("humanbrain.")
                          or name == "experiments" or name.startswith("experiments."))
        if project_module and name not in allowed:
            raise ValueError(f"undeclared project module in worker: {name}")
    return origins


def check_requests(requests):
    if (set(requests) != {"schema", "checkpoint_sha256", "source_sha256", "receipt_limits",
                          "arms", "cases", "reading_cases"}
            or requests["schema"] != "neuronal-splice-endpoint-request-v1"
            or requests["arms"] != list(ARMS) or requests["receipt_limits"] != LIMITS
            or len(requests["cases"]) != 48 or len(requests["reading_cases"]) != 20):
        raise ValueError("endpoint request contract mismatch")
    for field, keys in (("cases", {"id", "memory_id", "request_hex", "files"}),
                        ("reading_cases", {"id", "memory_id", "request_hex"})):
        ids = []
        for case in requests[field]:
            if set(case) != keys:
                raise ValueError("request contains undeclared fields")
            for key in ("id", "memory_id"):
                value = case[key]
                if (not isinstance(value, str) or not value or not value.isascii()
                        or len(value) > 128):
                    raise ValueError("request identity exceeds its bound")
            ids.append(case["id"])
            if len(checked_hex(case["request_hex"])) > 31:
                raise ValueError("raw request exceeds its bound")
            if field == "cases":
                if not isinstance(case["files"], list) or not 1 <= len(case["files"]) <= 4:
                    raise ValueError("invalid file observation")
                for item in case["files"]:
                    if set(item) != {"path", "content_hex"}:
                        raise ValueError("file observation contains undeclared fields")
                    if len(checked_hex(item["content_hex"])) > 128:
                        raise ValueError("file observation exceeds its bound")
        if len(set(ids)) != len(ids):
            raise ValueError("endpoint IDs must be unique within their panel")


def check_executed(message, identity):
    if (not isinstance(message, dict)
            or set(message) != {*identity, "action_sha256", "outcome"}
            or any(message[key] != value for key, value in identity.items() if key != "type")
            or message["type"] != "executed"):
        raise ValueError("parent reply does not bind the current proposal")
    digest = message["action_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or checked_hex(digest).hex() != digest:
        raise ValueError("parent reply needs an exact action-line hash")
    outcome = message["outcome"]
    if (not isinstance(outcome, dict)
            or set(outcome) != {"attempt_id", "proposal_sha256", "workspace_sha256", "execution"}
            or not isinstance(outcome["execution"], dict)
            or set(outcome["execution"]) != {"returncode", "stdout_hex", "stderr_hex"}):
        raise ValueError("worker accepts actual execution fields only")
    for key in ("stdout_hex", "stderr_hex"):
        if len(checked_hex(outcome["execution"][key])) > 16384:
            raise ValueError("actual output exceeds the successful receipt bound")
    return outcome


def endpoint(model, state, case, arm, index, spec, incoming, outgoing, journal, deadline):
    """One pending object remains live through one actual parent execution."""
    from experiments.r4_integrated.neuronal_splice_io_v1 import read_row, write_message
    from experiments.r4_integrated.neuronal_splice_receipts_v1 import compact_prediction
    from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
    from humanbrain.neuronal_splice_action import SpliceWorkspace

    began = time.perf_counter()
    if time.time() >= deadline:
        raise TimeoutError("worker deadline before proposal")
    workspace = SpliceWorkspace(tuple(OpaqueWorkspaceFile(
        item["path"], checked_hex(item["content_hex"])) for item in case["files"]))
    prediction = model.propose(checked_hex(case["request_hex"]), workspace, state)
    model_seconds = time.perf_counter() - began
    identity = dict(schema=IPC_SCHEMA, type="proposal", arm=arm, index=index,
                    id=case["id"], attempt_id=prediction.attempt_id)
    receipt_began = time.perf_counter()
    proposal = {**identity, "prediction": compact_prediction(prediction, state),
                "model_seconds": model_seconds}
    journal.append(proposal)
    write_message(outgoing, proposal, spec)
    del proposal
    receipt_seconds = time.perf_counter() - receipt_began
    wait_began = time.perf_counter()
    message = read_row(incoming, spec)
    wait_seconds = time.perf_counter() - wait_began
    if time.time() >= deadline:
        raise TimeoutError("worker deadline after parent execution")
    began = time.perf_counter()
    outcome = check_executed(message, identity)
    work = model.finish_without_teaching(prediction, workspace, state, outcome)
    model_seconds += time.perf_counter() - began
    acknowledgement = {**identity, "type": "acknowledged",
                       "action_sha256": message["action_sha256"],
                       "model_sha256": prediction.model_sha256,
                       "memory_sha256": prediction.memory_sha256, "work": work,
                       "model_seconds": model_seconds}
    del message, outcome, prediction
    receipt_began = time.perf_counter()
    journal.append(acknowledgement)
    write_message(outgoing, acknowledgement, spec)
    receipt_seconds += time.perf_counter() - receipt_began
    return {"model_seconds": model_seconds, "receipt_seconds": receipt_seconds,
            "parent_wait_seconds": wait_seconds}


def run(args):
    root, output = Path(args.package_root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    began = time.perf_counter()
    checkpoint_path, request_path = Path(args.checkpoint), Path(args.requests)
    if (file_hash(request_path) != args.requests_sha
            or file_hash(checkpoint_path) != args.checkpoint_sha):
        raise ValueError("worker input bytes changed")
    requests = json.loads(request_path.read_bytes())
    check_requests(requests)
    if requests["checkpoint_sha256"] != args.checkpoint_sha:
        raise ValueError("requests bind another checkpoint")
    check_sources(root, requests["source_sha256"])
    sys.path[:0] = [str(root), str(root / "src")]
    import torch

    from experiments.r4_integrated.neuronal_splice_io_v1 import (
        JsonlJournal,
        process_identity,
        process_memory,
    )
    from experiments.r4_integrated.neuronal_splice_receipts_v1 import compact_reader
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest

    torch.set_num_threads(1)
    origins = check_origins(root)
    spec = SimpleNamespace(config={"receipt_limits": requests["receipt_limits"]})
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (set(payload) != {"schema", "arms"}
            or payload["schema"] != "neuronal-splice-endpoint-state-v1"
            or set(payload["arms"]) != set(ARMS)):
        raise ValueError("endpoint checkpoint schema mismatch")
    journal = JsonlJournal(output / "endpoints.jsonl", spec)
    reading_journal = JsonlJournal(output / "reading.jsonl", spec)
    final, timings, custody_work = {}, {}, {}
    try:
        for arm in ARMS:
            entry = payload["arms"][arm]
            if (set(entry) != {"weights", "memories", "experience"}
                    or len(entry["memories"]) != 64
                    or type(entry["experience"]) is not int or entry["experience"] < 0):
                raise ValueError("endpoint arm state is incomplete")
            model = NeuronalSpliceLearner(mode=arm, seed=733)
            first = next(iter(entry["memories"]))
            states = {first: model.restore_snapshot(dict(
                model=entry["weights"], memory=entry["memories"][first],
                experience=entry["experience"]))}
            for memory_id, snapshot in entry["memories"].items():
                if memory_id != first:
                    state = model.reader.memory.restore_state(snapshot)
                    model.reader._check_state(state)
                    states[memory_id] = state
            before = _digest(model.state_dict(), custody_work)
            memories_before = {key: _digest(value.snapshot(), custody_work)
                               for key, value in states.items()}
            total = dict(model_seconds=0.0, receipt_seconds=0.0, parent_wait_seconds=0.0,
                         reading_seconds=0.0)
            for index, case in enumerate(requests["cases"]):
                result = endpoint(model, states[case["memory_id"]], case, arm, index, spec,
                                  sys.stdin.buffer, sys.stdout.buffer, journal, args.deadline_unix)
                for name, value in result.items():
                    total[name] += value
            for index, case in enumerate(requests["reading_cases"]):
                read_began = time.perf_counter()
                if time.time() >= args.deadline_unix:
                    raise TimeoutError("worker reading-panel deadline")
                state = states[case["memory_id"]]
                reader_model_before = _digest(model.state_dict(), custody_work)
                reader_memory_before = _digest(state.snapshot(), custody_work)
                prediction = model.reader.predict_with_receipt(
                    checked_hex(case["request_hex"]), state, 2)
                if (reader_model_before != _digest(model.state_dict(), custody_work)
                        or reader_memory_before != _digest(state.snapshot(), custody_work)):
                    raise ValueError("reading-only prediction changed neuronal state")
                model_seconds = time.perf_counter() - read_began
                reading_journal.append(dict(
                    schema="neuronal-splice-panel-row-v1", phase="post", arm=arm,
                    index=index, memory_id=case["memory_id"], request_hex=case["request_hex"],
                    model_sha256=reader_model_before, memory_sha256=reader_memory_before,
                    prediction=compact_reader(prediction, state), model_seconds=model_seconds))
                total["reading_seconds"] += time.perf_counter() - read_began
                del prediction
            after = _digest(model.state_dict(), custody_work)
            memories_after = {key: _digest(value.snapshot(), custody_work)
                              for key, value in states.items()}
            if before != after or memories_before != memories_after or model._pending is not None:
                raise ValueError("endpoint evaluation changed neuronal state")
            final[arm] = dict(model_sha256=after, memories_sha256=memories_after,
                              experience=model._experience)
            timings[arm] = total
            del model, states, state
    finally:
        endpoint_log, reading_log = journal.close(), reading_journal.close()
    if (endpoint_log["rows"] != 384 or reading_log["rows"] != 80
            or file_hash(checkpoint_path) != args.checkpoint_sha
            or file_hash(request_path) != args.requests_sha):
        raise ValueError("endpoint completion custody mismatch")
    check_sources(root, requests["source_sha256"])
    if origins != check_origins(root):
        raise ValueError("endpoint source origins changed")
    clock = time.get_clock_info("perf_counter")
    write_json(output / "completion.json", dict(
        schema="neuronal-splice-endpoint-completion-v1", status="passed", pid=os.getpid(),
        parent_pid=os.getppid(), process_identity=process_identity(),
        checkpoint_sha256=args.checkpoint_sha, requests_sha256=args.requests_sha,
        source_sha256=requests["source_sha256"], source_origins=origins,
        endpoint_log=endpoint_log, reading_log=reading_log, final=final, timings=timings,
        custody_work=custody_work, memory=process_memory(),
        elapsed_seconds=time.perf_counter() - began,
        clock=dict(implementation=clock.implementation, monotonic=clock.monotonic,
                   adjustable=clock.adjustable, resolution=clock.resolution)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package-root", "checkpoint", "checkpoint-sha", "requests", "requests-sha",
                 "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    args.output_created = False
    try:
        run(args)
    except Exception as error:
        path = Path(args.output) / "failure.json"
        if args.output_created and not path.exists():
            write_json(path, dict(schema="neuronal-splice-endpoint-failure-v1", status="failed",
                                  pid=os.getpid(), error_type=type(error).__name__,
                                  error=str(error)[:1024]))
        raise


if __name__ == "__main__":
    main()
