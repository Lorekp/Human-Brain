"""Independent saved-evidence checks for the selected neuronal splice protocol.

This evaluator may read the parent-only roster. It never belongs in the worker
package and never imports or executes a neuronal model or action executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import time
from pathlib import Path

from experiments.r4_integrated.neuronal_splice_receipts_v1 import (
    HEAD_LIMITS,
    LINE_LIMIT,
    TREE_LIMIT,
    close,
    exact_keys,
    hash_value,
    hex_bytes,
    logsumexp,
    number,
    proposal_digest,
    require,
    tree_bytes,
    validate_head,
    validate_prediction,
    validate_proposal,
    validate_reader,
)

DRAFT = True
ROOT = Path(__file__).resolve().parents[2]
ARMS = ("learned", "action_frozen", "slow_frozen", "memory_disabled")
MODEL_BYTES = 634432
TAPE_BYTES = 656904


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def documents_adjacency(documents):
    successors = []
    for encoded in documents:
        raw = hex_bytes(encoded)
        start = len(successors)
        successors.extend(range(start + 1, start + len(raw) + 1))
        successors.append(start + len(raw))
    require(1 <= len(successors) <= 128, "document package exceeds tape geometry")
    return successors


def ordered_files(files, *, maximum=128):
    require(type(files) is list and 1 <= len(files) <= 4, "invalid workspace file count")
    result = []
    for item in files:
        exact_keys(item, ("path", "content_hex"))
        path = item["path"]
        require(type(path) is str and 0 < len(path.encode("utf-8")) <= 31,
                "invalid workspace path bound")
        raw = hex_bytes(item["content_hex"], maximum)
        result.append((path, raw))
    require(len({path.casefold() for path, _ in result}) == len(result), "duplicate file path")
    return sorted(result, key=lambda item: item[0].encode("utf-8"))


def workspace_digest(files):
    ordered = ordered_files(files)
    digest = hashlib.sha256(b"humanbrain.raw-splice-workspace.v1\0")
    digest.update(len(ordered).to_bytes(4, "big"))
    for path, raw in ordered:
        for value in (path.encode("utf-8"), raw):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def boundaries(files):
    return [(path, offset) for path, raw in ordered_files(files) for offset in range(len(raw) + 1)]


def apply_saved(files, proposal):
    """Independent literal byte splice; no teacher or semantic target is accepted."""
    validate_proposal(proposal)
    before = ordered_files(files)
    rejection = None
    if proposal["workspace_sha256"] != workspace_digest(files):
        rejection = "stale_workspace"
    elif proposal["operation"] == "SPLICE":
        found = [(path, raw) for path, raw in before if path == proposal["path"]]
        if proposal["stop_reason"] != "eos":
            rejection = "missing_eos"
        elif not found or not 0 <= proposal["start"] <= proposal["end"] <= len(found[0][1]):
            rejection = "invalid_span"
        else:
            start, end = proposal["start"], proposal["end"]
            replacement = hex_bytes(proposal["replacement_hex"], 16)
            before = [(path, raw[:start] + replacement + raw[end:] if path == proposal["path"] else raw)
                      for path, raw in before]
    return [{"path": path, "content_hex": raw.hex()} for path, raw in before], rejection


def task_grade(task, proposal, after_files, rejection, execution):
    expected_after, expected_rejection = apply_saved(task["files"], proposal)
    require(ordered_files(after_files, maximum=144) == ordered_files(expected_after, maximum=144),
            "saved after-files differ from actual selected action")
    require((rejection is None) == (expected_rejection is None), "rejection status mismatch")
    if rejection is not None:
        require(type(rejection) is str and 0 < len(rejection.encode("utf-8")) <= 1024,
                "invalid rejection diagnostic")
    stdout = hex_bytes(execution["stdout_hex"], 16384)
    stderr = hex_bytes(execution["stderr_hex"], 16384)
    require(type(execution["returncode"]) is int or execution["returncode"] is None,
            "invalid process outcome")
    for name, raw in (("stdout", stdout), ("stderr", stderr)):
        require(execution[name + "_sha256"] == hashlib.sha256(raw).hexdigest(), "output digest mismatch")
    require(type(execution["materialized_matches_action"]) is bool, "invalid materialization flag")
    success = (rejection is None and execution["returncode"] == 0 and not stderr
               and execution["materialized_matches_action"] and not execution["decode_errors"]
               and stdout == hex_bytes(task["expected_stdout_hex"])
               and ordered_files(after_files, maximum=144)
               == ordered_files(task["expected_after_files"], maximum=144)
               and proposal["operation"] == task["required_operation"])
    canonical_correct = (proposal["operation"] == "STOP" if task["required_operation"] == "STOP"
                         else proposal == task["teacher"])
    return {"task_success": success, "canonical_tuple_correct": canonical_correct}


def reader_work(query, slots, decisions, *, teaching, arm):
    require(arm in ARMS and 1 <= slots <= 128 and 1 <= decisions <= 17, "invalid reader work geometry")
    return dict(
        query_raw_bytes=len(query), query_eos_symbols=1, query_lag_shift_steps=len(query) + 1,
        query_population_normalizations=1, address_feature_slots=slots,
        address_feature_products=slots * 32768, address_feature_channel_reductions=slots * 31744,
        address_score_synapse_evaluations=slots * 1024, address_score_reductions=slots * 1023,
        address_score_evaluations=slots, address_normalizations=1,
        codec_slot_evaluations=slots, codec_output_potentials=slots * 257, codec_normalizations=slots,
        codec_response_gathers=slots * (decisions - 1),
        codec_response_values_gathered=slots * (decisions - 1) * 1028,
        codec_response_bytes_gathered=slots * (decisions - 1) * 7196,
        mixture_output_reductions=257 * decisions, prefix_belief_updates=slots * decisions,
        branch_cursor_initializations=slots, branch_cursor_advances=slots * (decisions - 1),
        teacher_offers=decisions if teaching else 0, codec_update_group_offers=int(teaching),
        address_update_group_offers=int(teaching),
        enabled_codec_update_groups=int(teaching and arm != "slow_frozen"),
        enabled_address_update_groups=int(teaching and arm != "slow_frozen"),
        address_local_synapse_contributions=slots * 1024 if teaching else 0,
        address_local_slot_reductions=(slots - 1) * 1024 if teaching else 0,
    )


def action_work(request, files, source, *, teaching, arm, stop_teacher=False):
    """Shape arithmetic independent of the neuronal feature implementation."""
    ordered = ordered_files(files)
    lengths = dict((path, len(raw)) for path, raw in ordered)
    f, b = len(ordered), sum(length + 1 for length in lengths.values())
    chosen = lengths[source["path"]]
    ends = chosen - source["start"] + 1
    span, payload = source["end"] - source["start"], len(hex_bytes(source["replacement_hex"], 16))
    raw = len(request) + sum(len(path.encode("utf-8")) for path in lengths) + sum(lengths.values())
    raw += chosen + span + payload
    eos = 3 + 2 * f + int(source["stop_reason"] == "eos")
    context_count = 1
    if not stop_teacher:
        raw += sum(lengths.values()) + chosen
        eos += f + 1
        context_count += b + ends
    work = dict(raw_bytes_encoded=raw, eos_codes_encoded=eos, code_rows_loaded=raw + eos,
                boundary_contexts=context_count, context_code_gathers=32 * context_count,
                custody_tensor_bytes_hashed=(MODEL_BYTES + TAPE_BYTES) * (1 if teaching else 2))
    shapes = {"operation": (1, 4244, 4241 * 32)}
    if not stop_teacher:
        shapes = {"start": (b, 6177, (f * 5152 + b * 1024) * 32),
                  "end": (ends, 2049, ends * 2048 * 32), **shapes}
    for head, (rows, width, products) in shapes.items():
        work[head + "_feature_products"] = products
        work[head + "_feature_channel_reductions"] = products // 32 * 31
        work[head + "_feature_normalizations"] = rows
        work[head + "_feature_components"] = rows * width
        outputs = 2 if head == "operation" else rows
        work[head + "_score_products"] = outputs * width
        work[head + "_score_reductions"] = outputs * (width - 1)
        work[head + "_score_evaluations"] = outputs
        work[head + "_softmaxes"] = 1
        if teaching:
            work[head + "_local_products"] = outputs * width
            work[head + "_local_reductions"] = 0 if head == "operation" else (rows - 1) * width
            work[head + "_teacher_offers"] = 1
            work[head + "_log_softmaxes"] = 1
            work[head + "_enabled_update_groups"] = int(arm in ("learned", "memory_disabled"))
    work["operation_bias_additions"] = 2
    return work


def check_feedback(feedback, prediction, teacher, files, successor, request, arm):
    exact_keys(feedback, ("attempt_id", "action_updates", "action_losses", "teacher_choices", "reader", "reader_updates", "work"))
    require(feedback["attempt_id"] == prediction["attempt_id"], "feedback attempt mismatch")
    validate_proposal(teacher)
    stop = teacher["operation"] == "STOP"
    heads = {"operation"} if stop else set(HEAD_LIMITS)
    for name in ("action_updates", "action_losses", "teacher_choices"):
        exact_keys(feedback[name], heads)
    boundary_list = boundaries(files)
    targets = {"operation": int(not stop)}
    if not stop:
        targets["start"] = boundary_list.index((teacher["path"], teacher["start"]))
        targets["end"] = teacher["end"] - teacher["start"]
    source = prediction["proposal"] if stop else teacher
    counts = {"start": len(boundary_list), "end": sum(path == source["path"] and offset >= source["start"]
                                                       for path, offset in boundary_list), "operation": 2}
    for head in heads:
        decision = feedback["teacher_choices"][head]
        validate_head(decision, head, count=counts[head])
        logits = decision["logits"]
        close(feedback["action_losses"][head], logsumexp(logits) - logits[targets[head]])
        require(type(feedback["action_updates"][head]) is bool, "invalid changed flag")
        if arm in ("action_frozen", "slow_frozen"):
            require(not feedback["action_updates"][head], "frozen action group changed")
    exact_keys(feedback["work"], ("action", "reader"))
    require(feedback["work"]["action"] == action_work(request, files, source, teaching=True,
                                                      arm=arm, stop_teacher=stop), "action teacher work mismatch")
    if stop:
        require(feedback["reader"] is None and feedback["reader_updates"] is None
                and feedback["work"]["reader"] == {}, "STOP offered reader teaching")
        return 0, 1
    reader = feedback["reader"]
    exact_keys(feedback["reader_updates"], ("codec", "address"))
    require(all(type(value) is bool for value in feedback["reader_updates"].values()), "invalid reader changed flags")
    require(arm != "slow_frozen" or not any(feedback["reader_updates"].values()), "frozen reader changed")
    require(reader["kind"] == "teacher", "missing reader teacher")
    result = validate_reader(reader, successor, targets=hex_bytes(teacher["replacement_hex"], 16))
    expected = reader_work(request, len(successor), result["decisions"], teaching=True, arm=arm)
    require(reader["work"] == feedback["work"]["reader"] == expected, "reader teacher work mismatch")
    return result["decisions"], 3


class Stream:
    """Keeps scalar/hash state only; the bounded pair function owns current rows."""

    def __init__(self, path, metadata):
        exact_keys(metadata, ("path", "rows", "bytes", "sha256", "max_line_bytes", "max_row_tree_bytes"))
        require(Path(path).name == metadata["path"], "journal path mismatch")
        hash_value(metadata["sha256"])
        self.handle = Path(path).open("rb")
        self.metadata, self.hasher = metadata, hashlib.sha256()
        self.rows = self.bytes = self.max_line = self.max_tree = 0

    def next(self):
        line = self.handle.readline(LINE_LIMIT + 1)
        require(line and len(line) <= LINE_LIMIT and line.endswith(b"\n") and line.isascii(), "invalid bounded line")
        row = json.loads(line)
        require(type(row) is dict, "journal row must be an object")
        size = tree_bytes(row)
        require(size <= TREE_LIMIT, "parsed receipt exceeds tree limit")
        self.hasher.update(line)
        self.rows += 1
        self.bytes += len(line)
        self.max_line, self.max_tree = max(self.max_line, len(line)), max(self.max_tree, size)
        return row, hashlib.sha256(line).hexdigest()

    def finish(self):
        require(self.handle.read(1) == b"", "journal contains trailing records")
        self.handle.close()
        for key, value in (("rows", self.rows), ("bytes", self.bytes), ("max_line_bytes", self.max_line),
                           ("sha256", self.hasher.hexdigest())):
            require(self.metadata[key] == value, "journal completion mismatch")
        require(self.metadata["max_row_tree_bytes"] <= TREE_LIMIT, "writer tree exceeded bound")


def check_action(row, *, task, files, phase, arm, index, launch, output):
    """Reconstruct the actual edit and inspect the saved executed source bytes."""
    exact_keys(row, ("schema", "phase", "arm", "index", "task_id", "attempt_id",
                     "proposal_sha256", "workspace_sha256", "proposal", "before_files",
                     "after_files", "rejection", "execution", "grade", "envelope",
                     "apply_execute_grade_seconds", "action_completed_perf"))
    require(row["schema"] == "neuronal-splice-action-row-v1"
            and (row["phase"], row["arm"], row["index"]) == (phase, arm, index),
            "action identity/order mismatch")
    require(row["task_id"] == (task["task_id"] if task else f"{phase}-{index}"),
            "action task mismatch")
    require(ordered_files(row["before_files"]) == ordered_files(files), "action observation changed")
    require(row["workspace_sha256"] == workspace_digest(files), "action workspace hash mismatch")
    require(row["proposal_sha256"] == proposal_digest(row["proposal"]), "action proposal hash mismatch")
    require(row["proposal"]["workspace_sha256"] == row["workspace_sha256"], "stale action workspace")
    after, rejection = apply_saved(files, row["proposal"])
    require(ordered_files(after, maximum=144) == ordered_files(row["after_files"], maximum=144),
            "action does not apply its selected bytes")
    require((row["rejection"] is None) == (rejection is None), "action rejection mismatch")
    execution = row["execution"]
    validate_action_launch(execution, launch)
    require("stdout" not in execution and "stderr" not in execution, "duplicate decoded process output")
    for name in ("stdout", "stderr"):
        raw = hex_bytes(execution[name + "_hex"], 16384)
        require(hashlib.sha256(raw).hexdigest() == execution[name + "_sha256"], "raw output hash mismatch")
    directory = output / "executions" / f"{phase}-{arm}-{index:06d}"
    require(Path(execution["workspace"]).resolve() == directory.resolve(), "execution directory mismatch")
    require(len(str(directory).encode("utf-8")) <= 1024, "execution path exceeds declared bound")
    for path, content in ordered_files(after, maximum=144):
        require((directory / path).read_bytes() == content, "saved executed file changed")
    require(execution["materialized_matches_action"] is True, "executed source did not match proposal")
    envelope = {"attempt_id": row["attempt_id"], "proposal_sha256": row["proposal_sha256"],
                "workspace_sha256": row["workspace_sha256"],
                "execution": {key: execution[key] for key in ("returncode", "stdout_hex", "stderr_hex")}}
    require(row["envelope"] == envelope, "action feedback differs from actual execution")
    positive(row["apply_execute_grade_seconds"])
    positive(row["action_completed_perf"])
    if task is None:
        require(row["grade"] is None, "engineering probe was graded as skill")
    else:
        require(row["grade"] == task_grade(task, row["proposal"], after, row["rejection"], execution),
                "external task grade disagrees with actual executed edit")
    return row["grade"]


def cap_task():
    files = [{"path": "main.py", "content_hex": (b"print('cap')\n" + b"#" + b"A" * 113 + b"\n").hex()}]
    files += [{"path": f"z{i}.py", "content_hex": (b"#" + b"A" * 126 + b"\n").hex()} for i in range(3)]
    return {"task_id": "cap-0", "memory_id": "cap", "files": files,
            "workspace_sha256": workspace_digest(files), "request_hex": (b"Q" * 31).hex(),
            "teacher": {"operation": "SPLICE", "path": "main.py", "start": 0, "end": 0,
                        "replacement_hex": (b"A" * 16).hex(), "stop_reason": "eos",
                        "workspace_sha256": workspace_digest(files)}}


def check_learning_pair(actions, lessons, *, task, documents, phase, arm, index,
                        launch, output, expected_before=None, expected_memory=None):
    """Own two full rows only; return scalar bindings for the later timing row."""
    action, action_hash = actions.next()
    row, _ = lessons.next()
    grade = check_action(action, task=task if phase == "regular" else None,
                         files=task["files"], phase=phase, arm=arm, index=index,
                         launch=launch, output=output)
    exact_keys(row, ("schema", "phase", "arm", "index", "task_id", "memory_id", "request_hex",
                     "workspace_sha256", "action_line_sha256", "prediction", "teacher", "feedback",
                     "model_after_sha256", "memory_after_sha256", "model_seconds", "model_instance_token",
                     "teacher_started_perf", "teacher_completed_perf"))
    require(row["schema"] == "neuronal-splice-learning-row-v1"
            and (row["phase"], row["arm"], row["index"]) == (phase, arm, index), "learning order changed")
    for key in ("task_id", "memory_id", "request_hex", "workspace_sha256", "teacher"):
        require(row[key] == task[key], "learning task fields changed")
    prediction = row["prediction"]
    require(row["action_line_sha256"] == action_hash, "correction does not bind published actual action")
    for key in ("attempt_id", "proposal_sha256", "proposal"):
        require(action[key] == prediction[key], "execution replaced the free proposal")
    require(prediction["attempt_id"].split(":")[0] == row["model_instance_token"],
            "correction did not retain the proposing instance")
    require(positive(action["action_completed_perf"]) <= positive(row["teacher_started_perf"])
            <= positive(row["teacher_completed_perf"]), "teacher preceded execution")
    successor = documents_adjacency(documents)
    validate_prediction(prediction, successor, boundaries=boundaries(task["files"]))
    query = hex_bytes(task["request_hex"], 31)
    decisions = len(prediction["payload"]["steps"])
    require(prediction["work"]["reader"] == reader_work(query, len(successor), decisions,
                                                        teaching=False, arm=arm), "free reader work mismatch")
    require(prediction["work"]["action"] == action_work(query, task["files"], prediction["proposal"],
                                                        teaching=False, arm=arm), "free action work mismatch")
    offers, labels = check_feedback(row["feedback"], prediction, row["teacher"], task["files"],
                                     successor, query, arm)
    hash_value(row["model_after_sha256"])
    require(row["memory_after_sha256"] == prediction["memory_sha256"], "local correction rewrote episode memory")
    require(expected_before is None or prediction["model_sha256"] == expected_before,
            "persistent learner weight chain broke")
    require(expected_memory is None or prediction["memory_sha256"] == expected_memory,
            "learning episode identity changed")
    if arm == "slow_frozen":
        require(row["model_after_sha256"] == prediction["model_sha256"], "frozen learner changed")
    if phase == "cap":
        proposal = prediction["proposal"]
        expected_payload = b"\0" * 16 if arm == "memory_disabled" else b"A" * 16
        require(proposal["operation"] == "STOP" and proposal["path"] == "main.py"
                and proposal["start"] == proposal["end"] == 0
                and hex_bytes(proposal["replacement_hex"]) == expected_payload
                and proposal["stop_reason"] == ("limit" if arm == "memory_disabled" else "eos")
                and decisions == offers == 17 and labels == 3, "cap geometry/output incomplete")
    return {"model_after": row["model_after_sha256"], "memory": row["memory_after_sha256"],
            "instance": row["model_instance_token"], "attempt_id": prediction["attempt_id"],
            "decisions": decisions, "offers": offers, "labels": labels,
            "model_seconds": positive(row["model_seconds"]),
            "action_partial_seconds": positive(action["apply_execute_grade_seconds"]), "grade": grade}


def check_panel(row, case, *, phase, arm, index, documents, model_hash, memory_hash):
    exact_keys(row, ("schema", "phase", "arm", "index", "memory_id", "request_hex",
                     "model_sha256", "memory_sha256", "prediction", "model_seconds"))
    require(row["schema"] == "neuronal-splice-panel-row-v1"
            and (row["phase"], row["arm"], row["index"]) == (phase, arm, index), "panel order mismatch")
    require(row["memory_id"] == case["memory_id"] and row["request_hex"] == case["query_hex"],
            "panel request changed")
    require(row["model_sha256"] == model_hash and row["memory_sha256"] == memory_hash,
            "reading panel state changed")
    successor = documents_adjacency(documents)
    prediction = row["prediction"]
    require(prediction["kind"] == "free", "panel contains teacher activity")
    result = validate_reader(prediction, successor, max_bytes=2)
    require(prediction["work"] == reader_work(hex_bytes(case["query_hex"]), len(successor),
                                             result["decisions"], teaching=False, arm=arm), "panel work mismatch")
    positive(row["model_seconds"])
    return prediction["stop_reason"] == "eos" and prediction["generated_hex"] == case["expected_hex"]


def tensor_digest(value, torch):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            raw = item.detach().contiguous().numpy().tobytes()
            digest.update(str((str(item.dtype), tuple(item.shape), len(raw))).encode())
            digest.update(raw)
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(json.dumps(key).encode())
                visit(item[key])
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def validate_tensor(value, shape, dtype, torch):
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
            and value.dtype == dtype and value.device.type == "cpu"
            and not value.requires_grad and value.grad_fn is None and value.grad is None
            and bool(torch.isfinite(value).all()), "invalid saved tensor contract")
    return value.numel() * value.element_size()


def reconstruct_tape(snapshot, documents, codes, enabled, torch):
    """Independent literal storage reconstruction, never a neuronal model call."""
    exact_keys(snapshot, ("keys", "sensory", "successor", "write_index"))
    total = 0
    for name, shape, dtype in (("keys", (128, 1024), torch.float32),
                               ("sensory", (128, 257), torch.float32),
                               ("successor", (128,), torch.int64), ("write_index", (), torch.int64)):
        total += validate_tensor(snapshot[name], shape, dtype, torch)
    expected_keys = torch.zeros(128, 1024, dtype=torch.float64)
    expected_sensory = torch.zeros(128, 257, dtype=torch.float32)
    expected_successor = torch.full((128,), -1, dtype=torch.int64)
    position = 0
    for encoded in documents:
        raw = hex_bytes(encoded)
        preceding = []
        for token in (*raw, 256):
            require(position < 128, "saved tape exceeds capacity")
            if enabled:
                expected_sensory[position, token] = 1
                for lag, previous in enumerate(reversed(preceding[-32:])):
                    expected_keys[position, lag * 32:(lag + 1) * 32] = codes[previous].double()
                norm = max(float(expected_keys[position].square().sum().sqrt()), 1e-8)
                expected_keys[position] /= norm
            expected_successor[position] = position if token == 256 else position + 1
            preceding.append(token)
            position += 1
    require(int(snapshot["write_index"].item()) == position, "saved tape allocation mismatch")
    require(torch.equal(snapshot["sensory"], expected_sensory)
            and torch.equal(snapshot["successor"], expected_successor), "saved literal/adjacency mismatch")
    torch.testing.assert_close(snapshot["keys"].double(), expected_keys, rtol=2e-6, atol=2e-7)
    require(total == TAPE_BYTES, "saved tape storage changed")
    return position


def check_saved_state(output, report, roster):
    """Authenticate slow-state lineage and every retained episode without recall."""
    import torch

    torch.set_num_threads(1)
    prior_path = ROOT / "benchmarks/local_joint_address_online_v6.decision.json"
    require(sha(prior_path) == "f9b3b02b3bacf9977bb990f115072d41a043adeb074283d390bf45b5a0bf7a97",
            "prior completed decision changed")
    prior_decision = read(prior_path)
    prior_completion_path = ROOT / ".pytest-tmp/local-joint-address-online-v6-run/completion.json"
    require(sha(prior_completion_path) == prior_decision["input_sha256"][str(prior_completion_path.relative_to(ROOT)).replace("\\", "/")],
            "prior completion changed")
    prior_completion = read(prior_completion_path)
    prior_path = prior_completion_path.parent / "source-free/source_free.pt"
    require(sha(prior_path) == prior_completion["artifacts"]["source-free/source_free.pt"]
            == report["prior"]["checkpoint_sha256"]
            == "03654f61af4cccb93688a75e75a84c089800672530cdaffdc7a30f7cc5181180", "prior checkpoint changed")
    previous = torch.load(prior_path, map_location="cpu", weights_only=True)
    prior = previous["learned"]["weights"]
    checkpoint = output / "endpoint/source_free.pt"
    require(sha(checkpoint) == report["worker"]["checkpoint_sha256"], "final checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    exact_keys(saved, ("schema", "arms"))
    require(saved["schema"] == "neuronal-splice-endpoint-state-v1", "saved state schema changed")
    exact_keys(saved["arms"], ARMS)
    geometries = {
        "reader.input_codes": ((258, 32), torch.float32),
        "reader.address_weight": ((32, 32), torch.float32),
        "reader.codec.weight": ((257, 257), torch.float64),
        "reader.codec.bias": ((257,), torch.float64),
        "actions.start_weight": ((6177,), torch.float32),
        "actions.end_weight": ((2049,), torch.float32),
        "actions.operation_weight": ((2, 4244), torch.float32),
        "actions.operation_bias": ((2,), torch.float32),
    }
    slots = retained = 0
    initial_hashes, final_hashes, memory_hashes = {}, {}, {}
    fresh_actions = {}
    for head, width, offset in (("start", 6177, 5), ("end", 2049, 6), ("operation", 4244, 7)):
        outputs = 2 if head == "operation" else 1
        generator = torch.Generator(device="cpu").manual_seed(733 + offset)
        value = torch.empty(outputs, width, dtype=torch.float32)
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (outputs + width))
        value.uniform_(-bound, bound, generator=generator)
        fresh_actions[f"actions.{head}_weight"] = value if outputs == 2 else value[0]
    fresh_actions["actions.operation_bias"] = torch.zeros(2, dtype=torch.float32)
    for arm in ARMS:
        entry = saved["arms"][arm]
        exact_keys(entry, ("weights", "memories", "experience"))
        require(isinstance(entry["weights"], dict), "saved model state must be a mapping")
        weights = dict(entry["weights"])
        exact_keys(weights, (*geometries, "_extra_state", "reader._extra_state"))
        model_bytes = sum(validate_tensor(weights[key], shape, dtype, torch)
                          for key, (shape, dtype) in geometries.items())
        require(model_bytes == MODEL_BYTES, "combined model storage changed")
        metadata = {"schema": "neuronal-byte-splice-learner-v1", "mode": arm,
                    "seed": 733, "widths": {"start": 6177, "end": 2049, "operation": 4244},
                    "feature_energy": 32, "operation_augmented_energy": 33, "learning_rate": .01,
                    "payload_cap": 16, "publication": "validated action then atomic reader then copies"}
        require(weights["_extra_state"] == metadata, "action model metadata changed")
        reader_mode = {"action_frozen": "learned", "memory_disabled": "writes_disabled"}.get(arm, arm)
        expected_reader = {**prior["_extra_state"], "mode": reader_mode}
        require(weights["reader._extra_state"] == expected_reader, "reader metadata adaptation exceeded mode")
        require(torch.equal(weights["reader.input_codes"], prior["input_codes"]), "fixed input codes changed")
        initial = {**fresh_actions, **{"reader." + key: value for key, value in prior.items()},
                   "_extra_state": metadata, "reader._extra_state": expected_reader}
        initial_hashes[arm] = tensor_digest(initial, torch)
        final_hashes[arm] = tensor_digest(weights, torch)
        require(initial_hashes[arm] == report["prior"]["initial_model_sha256"][arm], "initial model lineage mismatch")
        require(final_hashes[arm] == report["final_model_sha256"][arm]
                == report["worker"]["initial_hashes"][arm]["model_sha256"], "final model identity mismatch")
        if arm == "slow_frozen":
            require(initial_hashes[arm] == final_hashes[arm], "all-frozen slow state changed")
        if arm in ("action_frozen", "slow_frozen"):
            require(all(torch.equal(weights[key], value) for key, value in fresh_actions.items()),
                    "frozen action synapses changed")
        require(entry["experience"] == (4 if report["mode"] == "cost" else 256), "saved action count changed")
        require(set(entry["memories"]) == set(roster["memory_packages"]), "retained memory package scope changed")
        memory_hashes[arm] = {}
        for memory_id, snapshot in entry["memories"].items():
            slots += reconstruct_tape(snapshot, roster["memory_packages"][memory_id]["document_hex"],
                                      weights["reader.input_codes"], arm != "memory_disabled", torch)
            memory_hashes[arm][memory_id] = tensor_digest(snapshot, torch)
            retained += TAPE_BYTES
        retained += model_bytes
        require(memory_hashes[arm] == report["worker"]["initial_hashes"][arm]["memories_sha256"],
                "exported episode hashes mismatch")
    require(retained == 170705152 and slots == 15160, "retained-state storage/ingestion ledger mismatch")
    del prior, previous, saved, entry, weights, initial
    return {"initial_model_sha256": initial_hashes, "final_model_sha256": final_hashes,
            "memory_sha256": memory_hashes, "retained_tensor_bytes": retained,
            "reconstructed_slots": slots, "model_forwards": 0}


def check_monitor(outer, *, pid, command=None):
    require(outer["status"] == "passed" and outer["exit_code"] == 0, "monitor did not complete")
    require(outer["seconds_limit"] == 900 and outer["tree_peak_sum_mib_limit"] == 2048,
            "monitor ceilings changed")
    require(0 < number(outer["elapsed_seconds"]) <= 900
            and 0 < number(outer["observed_tree_peak_sum_mib"]) <= 2048, "observed budget exceeded")
    require(outer["deadline_unix"] - outer["started_unix"] == 900, "deadline mismatch")
    require(command is None or outer["command"] == command, "monitor command mismatch")
    for actual in {pid, outer["child_pid"]}:
        identities = [key for key in outer["observed_process_peaks_mib"] if key.split("/")[0] == str(actual)]
        require(len(identities) == 1, "actual process absent or creation identity ambiguous")
        created = (int(identities[0].split("/")[1]) - 621355968000000000) / 10_000_000
        require(outer["started_unix"] <= created <= outer["started_unix"] + outer["elapsed_seconds"],
                "process creation outside selected launch")


def expected_runtime_bootstrap():
    return ("import sys, runpy; assert sys.flags.utf8_mode == 1; "
            "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
            "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
            "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
            "sys.path.insert(0, '.'); "
            "import io, os\n"
            "with io.open_code(os.path.abspath('main.py')) as source:\n"
            "    code = compile(source.read(), 'main.py', 'exec')\n"
            "runpy._run_module_code(code, mod_name='__main__', pkg_name='', script_name='main.py')")


def expected_runtime_bootstrap_sha256():
    return hashlib.sha256(expected_runtime_bootstrap().encode()).hexdigest()





def expected_action_bootstrap():
    return (
        'import sys; '
        'assert sys.flags.utf8_mode == 1; '
        "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
        "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
        "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
        "sys.path.insert(0, '.'); "
        'import io, os\n'
        "with io.open_code(os.path.abspath('main.py')) as source:\n"
        "    code = compile(source.read(), 'main.py', 'exec')\n"
        "def _run_source(code):\n"
        "    temporary_main = type(sys)('__main__')\n"
        "    original_main = sys.modules['__main__']\n"
        "    sys.modules['__main__'] = temporary_main\n"
        "    try:\n"
        "        original_argv0 = sys.argv[0]\n"
        "        sys.argv[0] = 'main.py'\n"
        "        try:\n"
        "            namespace = temporary_main.__dict__\n"
        "            namespace.update(__name__='__main__', __file__='main.py',\n"
        "                             __cached__=None, __doc__=None, __loader__=None,\n"
        "                             __package__='', __spec__=None)\n"
        "            exec(code, namespace)\n"
        "        finally:\n"
        "            sys.argv[0] = original_argv0\n"
        "    finally:\n"
        "        sys.modules['__main__'] = original_main\n"
        "    return namespace.copy()\n"
        "_run_source(code)"
    )


def expected_action_bootstrap_sha256():
    return hashlib.sha256(expected_action_bootstrap().encode()).hexdigest()


def validate_bootstrap_qualification(execution):
    """Authenticate eight reviewed temporary-main pairs, never reexecute them."""
    policy = 'regular_source_public_temporary_module_v1'
    assert execution['qualification_sha256'] == '4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c'
    assert execution['bootstrap_policy'] == policy
    assert execution['bootstrap_sha256'] == expected_action_bootstrap_sha256()
    path = pathlib.Path(execution['bootstrap_qualification_path'])
    assert path.is_absolute() and sha(path) == execution['bootstrap_qualification_sha256']
    manifest = read(path)
    assert set(manifest) == {'schema','status','runtime_qualification_sha256','bridge_sha256',
        'bootstrap_policy','bootstrap_sha256','paired_cases','child_executions','evidence'}
    assert manifest['schema'] == 'temporary-module-bootstrap-qualification-v1' and manifest['status'] == 'passed'
    assert manifest['runtime_qualification_sha256'] == execution['qualification_sha256']
    assert manifest['bridge_sha256'] == sha(ROOT/'experiments/r4_integrated/temporary_module_action_bridge_v1.py')
    assert manifest['bootstrap_policy'] == policy
    assert manifest['bootstrap_sha256'] == expected_action_bootstrap_sha256()
    assert type(manifest['paired_cases']) is int and manifest['paired_cases'] == 8
    assert type(manifest['child_executions']) is int and manifest['child_executions'] == 16
    assert isinstance(manifest['evidence'],dict) and manifest['evidence']
    for evidence, expected in manifest['evidence'].items():
        assert pathlib.Path(evidence).is_absolute() and sha(evidence) == expected
    return {'bootstrap_qualification_sha256':execution['bootstrap_qualification_sha256'],
        'paired_cases':8,'child_executions':16,'authenticated_evidence_files':len(manifest['evidence'])}

def validate_action_launch(execution, launch):
    """Scalar identity only per action; fixed file hashing occurs at boundaries."""
    assert execution['bootstrap_policy'] == 'regular_source_public_temporary_module_v1'
    assert execution['bootstrap_sha256'] == expected_action_bootstrap_sha256()
    assert execution['launch_record_sha256'] == hashlib.sha256(canonical(launch)).hexdigest()
    assert execution['child_command'] == [launch['base_executable'], '-I', '-S', '-X', 'utf8', '-B', '-c', expected_action_bootstrap()]
    assert execution['child_environment_override'] == {}
    assert execution['child_environment_removed'] == ['__PYVENV_LAUNCHER__']
    assert execution['child_python_flags'] == ['-I','-S','-X','utf8','-B']


def validate_launch_record(launch):
    assert set(launch) == {'policy','parent_runtime','base_executable','flags',
        'bootstrap_policy','bootstrap_sha256','environment_removed','artifacts'}
    assert launch['policy'] == 'base_stdlib_python_no_site_v1'
    validate_parent_runtime(launch['parent_runtime'])
    assert launch['base_executable'] == launch['parent_runtime']['base_executable']
    assert launch['flags'] == ['-I','-S','-X','utf8','-B']
    assert launch['environment_removed'] == ['__PYVENV_LAUNCHER__']
    assert launch['bootstrap_policy'] == 'regular_source_runpy_module_code_v1'
    assert launch['bootstrap_sha256'] == expected_runtime_bootstrap_sha256()
    expected = dict(launch['parent_runtime']['artifacts'])
    for name in ('stdlib_python_action_bridge_v1.py','source_python_action_bridge_v1.py'):
        path = (ROOT/'experiments/r4_integrated'/name).resolve()
        expected[str(path)] = sha(path)
    assert launch['artifacts'] == expected
    return hashlib.sha256(canonical(launch)).hexdigest()


def validate_parent_runtime(launch):
    assert set(launch) == {'policy','venv_executable','base_executable','venv_config',
        'python_version','implementation','base_prefix','base_exec_prefix','artifacts'}
    assert launch['policy'] == 'direct_base_python_with_selected_venv_launcher_v1'
    assert launch['implementation'] == 'cpython' and launch['python_version'].startswith('3.11.')
    for key in ('venv_executable','base_executable','venv_config','base_prefix','base_exec_prefix'):
        assert pathlib.Path(launch[key]).is_absolute()
    venv, base, config = (pathlib.Path(launch[key]).resolve() for key in
                         ('venv_executable','base_executable','venv_config'))
    assert venv == (ROOT/'.venv/Scripts/python.exe').resolve() and base != venv
    assert config == (ROOT/'.venv/pyvenv.cfg').resolve()
    entries = dict(line.split(' = ',1) for line in config.read_text().splitlines() if ' = ' in line)
    assert pathlib.Path(entries['home']).resolve() == base.parent
    assert pathlib.Path(launch['base_prefix']).resolve() == base.parent
    assert pathlib.Path(launch['base_exec_prefix']).resolve() == base.parent
    expected = {venv,base,config,(ROOT/'experiments/r4_integrated/direct_python_action_bridge_v1.py').resolve(),base.parent/'python311.dll'}
    expected.update(base.parent/'Lib'/name for name in ('runpy.py','site.py','subprocess.py',
        'encodings/__init__.py','multiprocessing/popen_spawn_win32.py'))
    artifacts = launch['artifacts']
    assert {pathlib.Path(path).resolve() for path in artifacts} == expected
    assert len(artifacts) == len(expected)
    for path, expected_hash in artifacts.items():
        assert pathlib.Path(path).is_absolute() and sha(path) == expected_hash
    return hashlib.sha256(canonical(launch)).hexdigest()


def validate_qualification(path, expected_hash, launch):
    """Authenticate reviewed identity/semantic/isolation evidence, never execute it."""
    path = pathlib.Path(path)
    assert path.is_absolute() and sha(path) == expected_hash
    qualification = read(path)
    assert set(qualification) == {'schema','status','launch_record','executions',
        'identity','semantic_pairs','contamination','evidence'}
    assert qualification['schema'] == 'stdlib-python-action-qualification-v1'
    assert qualification['status'] == 'passed' and qualification['launch_record'] == launch
    assert qualification['executions'] == {'identity':1,'semantic_pairs':14,'contamination':2}
    assert all(type(value) is int for value in qualification['executions'].values())
    evidence = qualification['evidence']
    assert isinstance(evidence,dict) and evidence
    for filename, expected in evidence.items():
        assert pathlib.Path(filename).is_absolute() and sha(filename) == expected
    names = ('namespace_cleanup','utf8_bom_crlf','latin1_cookie','syntax_error',
             'runtime_error','exit_invalid_utf8','pending')
    assert len(qualification['semantic_pairs']) == 7
    for name in ('identity','contamination'):
        item = qualification[name]
        assert set(item) == {'passed','receipt_path','receipt_sha256'}
        assert item['passed'] is True
        assert pathlib.Path(item['receipt_path']).is_absolute()
        assert evidence[item['receipt_path']] == item['receipt_sha256']
    for index, (name,item) in enumerate(zip(names,qualification['semantic_pairs'],strict=True)):
        assert set(item) == {'passed','receipt_path','receipt_sha256','case','case_index'}
        assert item['passed'] is True and item['case'] == name
        assert type(item['case_index']) is int and item['case_index'] == index
        assert pathlib.Path(item['receipt_path']).is_absolute()
        assert evidence[item['receipt_path']] == item['receipt_sha256']
    return {'qualification_sha256':expected_hash,'identity_executions':1,
            'semantic_executions':14,'contamination_executions':2,
            'authenticated_evidence_files':len(evidence)}





def positive(value):
    require(number(value) > 0, "expected positive resource measurement")
    return float(value)


def project_cost(report, outer, *, audit=None, audit_outer=None):
    """Independent separate maxima, including the replicated timing append."""
    require(report["mode"] == "cost" and report["status"] == "completed", "incomplete cost")
    require((audit is None) == (audit_outer is None), "unfinished cost-audit monitor")
    require(len(report["category_probes"]) == 4, "missing action category coverage")
    category = max(positive(item["action_seconds"]) for item in report["category_probes"])
    maxima, regular_total = {}, 0.0
    for arm in ARMS:
        phases = report["timing_summary"][arm]
        require(phases["regular"]["rows"] == 4 and phases["cap"]["rows"] == 1,
                "incomplete cost learning/cap scope")
        maxima[arm] = {}
        for stage in ("model", "action", "journal"):
            samples = []
            for phase, count in (("regular", 4), ("cap", 1)):
                total = positive(phases[phase][stage + "_seconds"])
                maximum = positive(phases[phase][stage + "_max_seconds"])
                require(maximum <= total <= count * maximum + 1e-8, "invalid stage summary")
                samples.append(maximum)
                if phase == "regular":
                    regular_total += total
            if stage == "action":
                samples.append(category)
            maxima[arm][stage] = max(samples)
    residual = positive(outer["elapsed_seconds"]) - regular_total
    require(residual >= 0, "cost stage sums exceed outer duration")
    run_seconds = 512 * sum(sum(row.values()) for row in maxima.values()) + 2 * residual + 30
    parent_worker = positive(report["memory"]["peak_working_set_mib"]) + positive(
        report["worker"]["worker"]["memory"]["peak_working_set_mib"])
    run_peak = max(parent_worker, positive(outer["observed_tree_peak_sum_mib"])) + 12 + 128
    audit_seconds = audit_peak = None
    if audit is not None:
        require(audit["status"] == "passed", "independent cost audit failed")
        check_monitor(audit_outer, pid=audit["pid"])
        sampled, maximum_sum = 0.0, 0.0
        for arm in ARMS:
            phases = audit["learning_pair_timing"][arm]
            require(phases["regular"]["rows"] == 4 and phases["cap"]["rows"] == 1,
                    "audit omitted regular/cap pairs")
            for phase, count in (("regular", 4), ("cap", 1)):
                row = phases[phase]
                require(0 < number(row["max_seconds"]) <= number(row["seconds"])
                        <= count * number(row["max_seconds"]) + 1e-8, "invalid audit pair timing")
            sampled += phases["regular"]["seconds"]
            maximum_sum += max(phases[phase]["max_seconds"] for phase in ("regular", "cap"))
        remainder = positive(audit_outer["elapsed_seconds"]) - sampled
        require(remainder >= 0, "audit pair sums exceed monitor duration")
        audit_seconds = 2 * (256 * maximum_sum + remainder)
        audit_peak = positive(audit_outer["observed_tree_peak_sum_mib"]) + 12 + 32
    total = run_seconds + (audit_seconds or 0)
    peak = max(run_peak, audit_peak or 0)
    return dict(per_arm=maxima, category_action_seconds=category,
                projected_run_seconds=run_seconds, projected_audit_seconds=audit_seconds,
                projected_seconds=total, run_projected_peak_mib=run_peak,
                audit_projected_peak_mib=audit_peak, projected_tree_mib=peak,
                audit_peak_pending=audit is None,
                admissible=audit is not None and total <= 900 and peak <= 2048)


def audit(args, result):
    """Stream completed records; no model, correction or child execution occurs."""
    started = time.perf_counter()
    require((args.cost_dir is None) != (args.run_dir is None), "select exactly one saved phase")
    output = (args.cost_dir or args.run_dir).resolve()
    mode = "cost" if args.cost_dir else "run"
    require(sha(args.admission) == args.admission_sha, "admission identity changed")
    admission, report, completion, outer = (read(path) for path in (
        args.admission, output / "report.json", output / "completion.json", args.launch))
    require(admission["schema"] == "neuronal-splice-admission-v1"
            and admission["status"] == "passed" and admission["authorized_phase"] == mode
            and admission["qa_passed"] is True, "execution was not admitted")
    require(report["status"] == "completed" and report["mode"] == mode
            and report["admission_sha256"] == args.admission_sha
            and report["config"] == admission["config"]
            and report["source_sha256"] == report["source_sha256_after"] == admission["source_sha256"],
            "report/admission/source chain mismatch")
    require(completion["schema"] == "neuronal-splice-completion-v1"
            and completion["status"] == "completed" and completion["mode"] == mode
            and completion["report_sha256"] == sha(output / "report.json")
            and completion["admission_sha256"] == args.admission_sha
            and completion["logs"] == report["logs"], "completion chain mismatch")
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8", "-m",
               "experiments.r4_integrated.neuronal_splice_online_v1", mode,
               "--output", str(output), "--admission", str(args.admission),
               "--admission-sha", args.admission_sha, "--deadline-unix", repr(report["deadline_unix"])]
    check_monitor(outer, pid=report["pid"], command=command)
    require(report["deadline_unix"] == outer["deadline_unix"], "run deadline differs from watchdog")
    for relative, expected in report["source_sha256"].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == expected, "held source changed")
    roster_path = Path(admission["roster_path"])
    require(sha(roster_path) == admission["roster_sha256"] == report["roster_sha256"], "frozen roster changed")
    roster = read(roster_path)
    require(roster["schema"] == "neuronal-splice-roster-v1"
            and roster["counts"]["document_raw_bytes_per_arm"] == 3638
            and roster["counts"]["maximum_memory_slots"] == 128, "roster arithmetic amendment absent")
    runtime = report["action_launch_record"]
    validate_launch_record(runtime)
    validate_qualification((ROOT / ".pytest-tmp/stdlib-python-action-v1-qualification.json").resolve(),
                           "4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c", runtime)
    validate_bootstrap_qualification({
        "qualification_sha256": "4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c",
        "bootstrap_policy": "regular_source_public_temporary_module_v1",
        "bootstrap_sha256": expected_action_bootstrap_sha256(),
        "bootstrap_qualification_path": str((ROOT / ".pytest-tmp/temporary-module-bootstrap-v1-qualification.json").resolve()),
        "bootstrap_qualification_sha256": "0bb809fd33aee145f169776a03a884fd62daf381ff4e67ed333a2a378c46d92b"})
    lineage = check_saved_state(output, report, roster)
    worker_path = output / "endpoint/worker/completion.json"
    worker = read(worker_path)
    require(sha(worker_path) == report["worker"]["worker_completion_sha256"]
            and worker == report["worker"]["worker"] and worker["status"] == "passed", "worker completion changed")
    require(f'{worker["process_identity"]["pid"]}/{worker["process_identity"]["creation_ticks"]}'
            in outer["observed_process_peaks_mib"], "actual endpoint worker lacks owned creation identity")
    request_path = output / "endpoint/query_only.json"
    require(sha(request_path) == worker["requests_sha256"] == report["worker"]["requests_sha256"],
            "worker observations changed")
    requests = read(request_path)
    task_ids = roster["fresh_task_ids"] + roster["retention_task_ids"]
    expected_cases = [{"id": task_id, "memory_id": roster["tasks"][task_id]["memory_id"],
                       "request_hex": roster["tasks"][task_id]["request_hex"],
                       "files": roster["tasks"][task_id]["files"]} for task_id in task_ids]
    expected_reading = [{"id": str(case["case_id"]), "memory_id": case["memory_id"],
                         "request_hex": case["query_hex"]} for case in roster["old_panel"]]
    exact_keys(requests, ("schema", "checkpoint_sha256", "source_sha256", "receipt_limits", "arms", "cases", "reading_cases"))
    require(requests["cases"] == expected_cases and requests["reading_cases"] == expected_reading
            and requests["arms"] == list(ARMS) and requests["receipt_limits"] == report["config"]["receipt_limits"],
            "teacher-free worker request contains changed observations")
    package = output / "endpoint/package"
    require(requests["source_sha256"] == worker["source_sha256"] == report["worker"]["source_sha256"],
            "worker source chain mismatch")
    require({path.relative_to(package).as_posix() for path in package.rglob("*.py")}
            == set(requests["source_sha256"]), "isolated package contains undeclared source")
    for relative, expected in requests["source_sha256"].items():
        require(sha(package / relative) == sha(ROOT / relative) == expected, "isolated source changed")
    actions = Stream(output / "actions.jsonl", report["logs"]["actions"])
    lessons = Stream(output / "lessons.jsonl", report["logs"]["lessons"])
    timing = Stream(output / "timings.jsonl", report["logs"]["timings"])
    pairs = {arm: {phase: {"rows": 0, "seconds": 0., "max_seconds": 0.}
                    for phase in ("regular", "cap")} for arm in ARMS}
    measured = {arm: {phase: {"rows": 0, "model": 0., "action": 0., "journal_partial": 0.,
                              "model_max": 0., "action_max": 0., "journal_partial_max": 0.}
                       for phase in ("regular", "cap")} for arm in ARMS}
    chain = dict(lineage["initial_model_sha256"])
    instances = {}
    counts = {"learning_pairs": 0, "free_decisions": 0, "reader_offers": 0, "action_labels": 0}
    curves = {arm: [] for arm in ARMS}

    def pair(task, documents, phase, arm, index):
        began = time.perf_counter()
        values = check_learning_pair(actions, lessons, task=task, documents=documents,
                                     phase=phase, arm=arm, index=index, launch=runtime, output=output,
                                     expected_before=chain[arm] if phase == "regular" else None,
                                     expected_memory=lineage["memory_sha256"][arm][task["memory_id"]]
                                     if phase == "regular" else None)
        # The two full rows have been released before reading this small timing row.
        row, _ = timing.next()
        exact_keys(row, ("phase", "arm", "index", "model_seconds", "action_seconds", "journal_before_timing_append_seconds"))
        require((row["phase"], row["arm"], row["index"]) == (phase, arm, index), "timing order mismatch")
        close(row["model_seconds"], values["model_seconds"])
        require(positive(row["action_seconds"]) >= values["action_partial_seconds"], "action timing omits execution")
        measured[arm][phase]["rows"] += 1
        for stage, field in (("model", "model_seconds"), ("action", "action_seconds"),
                              ("journal_partial", "journal_before_timing_append_seconds")):
            value = positive(row[field])
            measured[arm][phase][stage] += value
            measured[arm][phase][stage + "_max"] = max(measured[arm][phase][stage + "_max"], value)
        for key, value in (("learning_pairs", 1), ("free_decisions", values["decisions"]),
                           ("reader_offers", values["offers"]), ("action_labels", values["labels"])):
            counts[key] += value
        if phase == "regular":
            if arm in instances:
                require(instances[arm] == values["instance"], "learning switched model instance")
            instances[arm] = values["instance"]
            require(values["attempt_id"] == f'{values["instance"]}:{index + 1}', "experience sequence changed")
            chain[arm] = values["model_after"]
            curves[arm].append(values["grade"]["task_success"])
        seconds = time.perf_counter() - began
        totals = pairs[arm][phase]
        totals["rows"] += 1
        totals["seconds"] += seconds
        totals["max_seconds"] = max(totals["max_seconds"], seconds)

    if mode == "cost":
        task = cap_task()
        documents = [(b"A" * 16).hex()] * 7 + [(b"A" * 8).hex()]
        for arm in ARMS:
            pair(task, documents, "cap", arm, 0)
        comment = (b"#" + b"A" * 14 + b"\n").hex()
        definitions = (("STOP", 0, "", "eos"), ("SPLICE", 0, comment, "eos"),
                       ("SPLICE", 0, comment, "limit"), ("SPLICE", 1, "21", "eos"))
        for index, (operation, end, payload, reason) in enumerate(definitions):
            row, digest = actions.next()
            check_action(row, task=None, files=task["files"], phase="category", arm="shared", index=index,
                         launch=runtime, output=output)
            require(row["proposal"] == {**task["teacher"], "operation": operation, "end": end,
                                         "replacement_hex": payload, "stop_reason": reason}, "category proposal changed")
            code = row["execution"]["returncode"]
            require(code == 0 if index < 3 else type(code) is int and code != 0
                    and b"SyntaxError" in hex_bytes(row["execution"]["stderr_hex"]), "category outcome mismatch")
            require(report["category_probes"][index]["action_line_sha256"] == digest, "category timing binding changed")
    selected = roster["cost_task_ids"] if mode == "cost" else roster["schedule"]
    require(report["selected_task_ids"] == selected, "acquisition schedule changed")
    for index, task_id in enumerate(selected):
        task = roster["tasks"][task_id]
        for arm in ARMS:
            pair(task, roster["memory_packages"][task["memory_id"]]["document_hex"], "regular", arm, index)
    lessons.finish()
    timing.finish()
    require(chain == lineage["final_model_sha256"] and curves == report["acquisition_first_attempts"],
            "learning trajectory/final export mismatch")
    require((counts["learning_pairs"], counts["reader_offers"], counts["action_labels"])
            == ((20, 96, 52) if mode == "cost" else (1024, 1920, 2560)), "teaching exposure changed")
    for arm in ARMS:
        for phase in ("regular", "cap"):
            summary, rows = report["timing_summary"][arm][phase], measured[arm][phase]
            require(summary["rows"] == rows["rows"], "timing summary count mismatch")
            for stage in ("model", "action"):
                close(summary[stage + "_seconds"], rows[stage])
                close(summary[stage + "_max_seconds"], rows[stage + "_max"])
            require(summary["journal_seconds"] >= rows["journal_partial"]
                    and summary["journal_max_seconds"] >= rows["journal_partial_max"],
                    "complete journal summary omits timing-row append")
    endpoint_stream = Stream(output / "endpoint/worker/endpoints.jsonl", worker["endpoint_log"])
    scores = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for index, task_id in enumerate(task_ids):
            task = roster["tasks"][task_id]
            proposal, _ = endpoint_stream.next()
            exact_keys(proposal, ("schema", "type", "arm", "index", "id", "attempt_id", "prediction", "model_seconds"))
            require((proposal["schema"], proposal["type"], proposal["arm"], proposal["index"], proposal["id"])
                    == ("neuronal-splice-endpoint-ipc-v1", "proposal", arm, index, task_id), "worker endpoint order mismatch")
            successor = documents_adjacency(roster["memory_packages"][task["memory_id"]]["document_hex"])
            prediction = proposal["prediction"]
            validate_prediction(prediction, successor, boundaries=boundaries(task["files"]))
            require(prediction["model_sha256"] == lineage["final_model_sha256"][arm]
                    and prediction["memory_sha256"] == lineage["memory_sha256"][arm][task["memory_id"]],
                    "endpoint state differs from learned export")
            action, action_hash = actions.next()
            grade = check_action(action, task=task, files=task["files"], phase="endpoint", arm=arm, index=index,
                                 launch=runtime, output=output)
            for key in ("attempt_id", "proposal", "proposal_sha256"):
                require(action[key] == prediction[key], "endpoint execution replaced a neuronal choice")
            scores[arm].append(grade["task_success"])
            identity = {key: proposal[key] for key in ("schema", "arm", "index", "id", "attempt_id")}
            model_hash, memory_hash = prediction["model_sha256"], prediction["memory_sha256"]
            del proposal, prediction, action
            acknowledgement, _ = endpoint_stream.next()
            exact_keys(acknowledgement, (*identity, "type", "action_sha256", "model_sha256", "memory_sha256", "work", "model_seconds"))
            require(all(acknowledgement[key] == value for key, value in identity.items())
                    and acknowledgement["type"] == "acknowledged" and acknowledgement["action_sha256"] == action_hash
                    and acknowledgement["model_sha256"] == model_hash and acknowledgement["memory_sha256"] == memory_hash
                    and acknowledgement["work"]["reader"] == {}, "worker acknowledgement changed state/action")
            del acknowledgement
    endpoint_stream.finish()
    actions.finish()
    require(scores == report["worker"]["task_success"], "endpoint scores disagree")
    panels = {}
    for phase, path, metadata, hashes in (
        ("pre", output / "pre_panel.jsonl", report["logs"]["pre_panel"], lineage["initial_model_sha256"]),
        ("post", output / "endpoint/worker/reading.jsonl", worker["reading_log"], lineage["final_model_sha256"])):
        panel = Stream(path, metadata)
        panels[phase] = {}
        for arm in ARMS:
            panels[phase][arm] = []
            for index, case in enumerate(roster["old_panel"]):
                row, _ = panel.next()
                panels[phase][arm].append(check_panel(
                    row, case, phase=phase, arm=arm, index=index,
                    documents=roster["memory_packages"][case["memory_id"]]["document_hex"],
                    model_hash=hashes[arm], memory_hash=lineage["memory_sha256"][arm][case["memory_id"]]))
        panel.finish()
    fresh_edit_indices = [i for i, task_id in enumerate(task_ids[:32])
                          if roster["tasks"][task_id]["required_operation"] == "SPLICE"]
    learned_edits = sum(scores["learned"][i] for i in fresh_edit_indices)
    criterion = (all(scores["learned"])
                 and all(panels["post"]["learned"][i] for i in range(20) if i not in (16, 18))
                 and all(learned_edits > sum(scores[arm][i] for i in fresh_edit_indices) for arm in ARMS[1:]))
    result.update(status="passed", mode=mode, report_sha256=sha(output / "report.json"),
                  completion_sha256=sha(output / "completion.json"), launch_sha256=sha(args.launch),
                  admission_sha256=args.admission_sha, source_sha256=report["source_sha256"],
                  learning_pair_timing=pairs, counts=counts, endpoint_scores=scores, reading_panels=panels,
                  scoped_criterion_passed=criterion if mode == "run" else False,
                  lineage=lineage, seconds=time.perf_counter() - started,
                  limits=["No neuronal forward or action replay in saved audit.",
                          "Receipt/source custody is not a filesystem sandbox or formal timing bound.",
                          "Narrow fragment-copy edit tasks do not establish general software ability."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha", required=True)
    args = parser.parse_args()
    if DRAFT:
        raise RuntimeError("draft checker cannot audit genuine artifacts")
    result = {"schema": "neuronal-splice-independent-audit-v1", "pid": os.getpid()}
    audit(args, result)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
