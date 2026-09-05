"""Source-qualified multi-answer actions and explicit same-workspace teachers.

Predictors and the learned motor never receive annotation targets. Initial
execution tests the selected bytes; authoritative evidence correctness additionally
requires a supplied, original-source annotated occurrence. Teachers are separately
labeled external corrections and never count as learner success.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.real_input_action_bridge import (
    ActionTask,
    _files_record,
    escape_literal_content,
)
from experiments.r4_integrated.utf8_action_bridge import Case, execute_utf8
from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile


def _digest(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _occurrence(inventory, index):
    candidate = inventory.candidates[index]
    document = inventory.documents[candidate.document_index]
    if document.tokens.raw[candidate.byte_start : candidate.byte_end] != candidate.emitted:
        raise ValueError("candidate bytes differ from their original source interval")
    return {
        "candidate_index": index,
        "document_index": candidate.document_index,
        "document_id_hex": document.document_id.hex(),
        "provenance_id_hex": document.provenance_id.hex(),
        "source_sha256": sha256(document.tokens.raw).hexdigest(),
        "byte_start": candidate.byte_start,
        "byte_end": candidate.byte_end,
        "emitted_hex": candidate.emitted.hex(),
    }


def _targets(task, inventory, accepted_gold_indices):
    match = re.fullmatch(r"qa(\d{3})", task.key)
    if (
        match is None
        or not 0 <= int(match[1]) < 28
        or task.module != "qa_config" + match[1]
        or task.action_instruction != ("Configure " + task.key).encode()
        or task.stale != "pending"
    ):
        raise ValueError("explicit ordinal action metadata differs from the declared roster")
    if accepted_gold_indices is None:
        if task.expected_answer is not None:
            raise ValueError("STOP annotation and canonical teacher answer disagree")
        return None
    if (
        not isinstance(accepted_gold_indices, tuple)
        or not accepted_gold_indices
        or any(
            type(i) is not int or not 0 <= i < len(inventory.candidates)
            for i in accepted_gold_indices
        )
        or len(set(accepted_gold_indices)) != len(accepted_gold_indices)
    ):
        raise ValueError("nonempty distinct authenticated gold occurrence indices required")
    records = [_occurrence(inventory, i) for i in accepted_gold_indices]
    if len({r["document_index"] for r in records}) != 1:
        raise ValueError("official gold intervals must belong to one original paragraph")
    if task.expected_answer is None or task.expected_answer.hex() not in {
        r["emitted_hex"] for r in records
    }:
        raise ValueError("canonical teacher answer is absent from annotated source occurrences")
    return records


def _state(reader, motor):
    return reader.checkpoint_sha256, motor.checkpoint_sha256


def execute_prediction(
    prediction,
    query,
    inventory,
    motor,
    task: ActionTask,
    root: Path,
    *,
    reader,
    accepted_gold_indices: tuple[int, ...] | None,
    semantic_scored: bool = True,
):
    """Execute one existing prediction; target indices are evaluator-only inputs."""
    if type(semantic_scored) is not bool or (
        not semantic_scored and accepted_gold_indices is not None
    ):
        raise ValueError("diagnostic-only actions require an original STOP annotation")
    gold = _targets(task, inventory, accepted_gold_indices)
    parent, motor_parent = _state(reader, motor)
    if (
        prediction.checkpoint_sha256 != parent
        or prediction.query_sha256 != query.sha256
        or prediction.inventory_sha256 != inventory.sha256
        or prediction.stopped != (prediction.candidate_index is None)
    ):
        raise ValueError("prediction does not belong to current reader/query/source")
    index = prediction.candidate_index
    if index is not None:
        if int(index) != index or not 0 <= int(index) < len(inventory.candidates):
            raise ValueError("selected candidate is outside the full inventory")
        index = int(index)
    occurrence = None if index is None else _occurrence(inventory, index)
    emitted = None if index is None else inventory.candidates[index].emitted
    payload = None if emitted is None else escape_literal_content(emitted)
    case = Case(task.name, task.key, task.stale, task.module, task.stale)
    before = case.files()
    action = None if payload is None else motor.apply(task.action_instruction, before, payload)
    after = before if action is None else action.files
    if _state(reader, motor) != (parent, motor_parent):
        raise RuntimeError("actuation changed learned state")
    # This expectation is selected-value transport only, independent of annotations.
    selected_value = task.stale if emitted is None else emitted.decode("utf-8", errors="strict")
    execution = execute_utf8(replace(case, value=selected_value), after, before, root)
    if _state(reader, motor) != (parent, motor_parent):
        raise RuntimeError("execution changed learned state")
    correct_no_action = gold is None and index is None and after == before
    evidence_correct = index is None if gold is None else index in accepted_gold_indices
    lossless = (
        execution["passed"]
        and bytes.fromhex(execution["stdout_hex"]) == (selected_value + "\n").encode()
        and execution["stderr_hex"] == ""
    )
    result = {
        "task": task.name,
        "reading_question_hex": query.raw.hex(),
        "reading_query_sha256": query.sha256,
        "inventory_sha256": inventory.sha256,
        "candidate_count": len(inventory.candidates),
        "action_instruction_hex": task.action_instruction.hex(),
        "reader_checkpoint_sha256": parent,
        "motor_checkpoint_sha256": motor_parent,
        "candidate_index": index,
        "stopped": bool(prediction.stopped),
        "stop_reason": prediction.stop_reason,
        "p_stop": float(prediction.p_stop),
        "span_scores_sha256": sha256(prediction.span_scores.tobytes()).hexdigest(),
        "selected_span_hex": None if emitted is None else emitted.hex(),
        "selected_occurrence": occurrence,
        "literal_payload_hex": None if payload is None else payload.hex(),
        "canonical_teacher_answer_hex": None
        if task.expected_answer is None
        else task.expected_answer.hex(),
        "accepted_gold_occurrences": gold,
        "gold_custody_sha256": _digest(gold),
        "semantic_scored": semantic_scored,
        "official_label_agreement": evidence_correct,
        "evidence_correct": evidence_correct if semantic_scored else None,
        "correct_no_action": correct_no_action if semantic_scored else None,
        "workspace_unchanged": after == before,
        "edited": action is not None and action.edited,
        "action_stop_reason": "reading_STOP" if action is None else action.stop_reason,
        "motor": None
        if action is None
        else {
            k: getattr(action, k)
            for k in (
                "request_seed_count",
                "supported_role_count",
                "physical_cell_steps",
                "evaluated_cells",
            )
        },
        "before": _files_record(before),
        "after": _files_record(after),
        "execution": execution,
        "lossless_selected_execution": lossless,
        "success": bool(lossless and evidence_correct and (gold is not None or correct_no_action))
        if semantic_scored
        else None,
        "execution_scope": (
            "passed tests selected-byte transport; success additionally requires "
            "annotated occurrence or correct unchanged STOP; diagnostic-only rows "
            "have no semantic verdict and official-label agreement is not correctness"
        ),
        "source_storage": (
            "complete cumulative paragraphs, word vectors and candidate bytes remain retained"
        ),
    }
    result["record_sha256"] = _digest(result)
    return result


def execute_teacher(
    failed_row,
    task: ActionTask,
    query,
    inventory,
    *,
    reader,
    motor,
    accepted_gold_indices: tuple[int, ...] | None,
):
    """One external correction in the same failed workspace; never trains a core."""
    if failed_row.get("semantic_scored") is False:
        raise ValueError("diagnostic-only actions cannot authorize teacher correction")
    gold = _targets(task, inventory, accepted_gold_indices)
    preserved = json.loads(json.dumps(failed_row, allow_nan=False))
    if preserved.pop("record_sha256", None) != _digest(preserved) or failed_row["success"]:
        raise ValueError("teacher requires an intact unsuccessful first-attempt receipt")
    parent, motor_parent = _state(reader, motor)
    if (
        failed_row["task"] != task.name
        or failed_row["reading_query_sha256"] != query.sha256
        or failed_row["inventory_sha256"] != inventory.sha256
        or failed_row["reader_checkpoint_sha256"] != parent
        or failed_row["motor_checkpoint_sha256"] != motor_parent
        or failed_row["action_instruction_hex"] != task.action_instruction.hex()
        or failed_row["gold_custody_sha256"] != _digest(gold)
        or failed_row["canonical_teacher_answer_hex"]
        != (None if task.expected_answer is None else task.expected_answer.hex())
    ):
        raise ValueError("teacher task/annotation/state differs from its own first attempt")
    root = Path(failed_row["execution"]["workspace"]).resolve()
    files = {}
    for item in failed_row["after"]:
        raw = bytes.fromhex(item["content_hex"])
        path = (root / item["path"]).resolve()
        if (
            not path.is_relative_to(root)
            or sha256(raw).hexdigest() != item["sha256"]
            or path.read_bytes() != raw
        ):
            raise ValueError("failed workspace changed before teacher repair")
        files[item["path"]] = raw
    if {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} != set(files):
        raise ValueError("failed workspace path roster changed")
    case = Case(task.name, task.key, task.stale, task.module, task.stale)
    original = case.files()
    if failed_row["before"] != _files_record(original):
        raise ValueError("original ordinal workspace differs")
    expected = task.stale.encode() if gold is None else task.expected_answer
    payload = escape_literal_content(expected)
    repaired = tuple(
        OpaqueWorkspaceFile(
            f.path,
            (f"{task.key} = '".encode() + payload + b"'\n")
            if f.path == task.module + ".py"
            else f.content,
        )
        for f in original
    )
    execution = _execute_existing(replace(case, value=expected.decode()), repaired, original, root)
    if _state(reader, motor) != (parent, motor_parent):
        raise RuntimeError("teacher execution changed learned state")
    result = {
        "teacher_kind": "authorized_absent_evidence_stop"
        if gold is None
        else "canonical_official_answer_repair",
        "first_attempt_sha256": failed_row["record_sha256"],
        "first_attempt": {**preserved, "record_sha256": failed_row["record_sha256"]},
        "before_repair": preserved["after"],
        "after_repair": _files_record(repaired),
        "workspace": str(root),
        "teacher_answer_hex": None if gold is None else expected.hex(),
        "accepted_gold_occurrences": gold,
        "reader_checkpoint_sha256": parent,
        "motor_checkpoint_sha256": motor_parent,
        "execution": execution,
        "passed": bool(
            execution["passed"]
            and bytes.fromhex(execution["stdout_hex"]) == expected + b"\n"
            and execution["stderr_hex"] == ""
        ),
        "counts_as_learner_success": False,
        "teacher_executions": 1,
    }
    result["record_sha256"] = _digest(result)
    return result


def _execute_existing(case: Case, files, before, root: Path) -> dict:
    """Materialize and run these trusted, generated public fixtures only."""
    began = time.perf_counter()
    if not root.is_dir():
        raise ValueError("teacher workspace must already exist")
    for item in files:
        path = (root / item.path).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("teacher cannot add or escape workspace paths")
        path.write_bytes(item.content)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                "utf8",
                "-B",
                "-c",
                "import sys, runpy; assert sys.flags.utf8_mode == 1; "
                "assert sys.stdout.encoding.lower().replace('-', '') == 'utf8'; "
                "sys.stdout.reconfigure(errors='strict', newline='\\n'); "
                "sys.stderr.reconfigure(errors='strict', newline='\\n'); "
                "sys.path.insert(0, '.'); "
                "runpy.run_path('main.py', run_name='__main__')",
            ],
            cwd=root,
            capture_output=True,
            timeout=10,
            check=False,
        )
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
    decoded = {}
    decode_errors = {}
    for name, raw in (("stdout", stdout), ("stderr", stderr)):
        try:
            decoded[name] = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            decoded[name] = None
            decode_errors[name] = str(error)
    return {
        "returncode": returncode,
        "stdout": decoded["stdout"],
        "stderr": decoded["stderr"],
        "stdout_hex": stdout.hex(),
        "stderr_hex": stderr.hex(),
        "stdout_sha256": sha256(stdout).hexdigest(),
        "stderr_sha256": sha256(stderr).hexdigest(),
        "output_encoding": "utf-8",
        "decode_errors": decode_errors,
        "child_python_flags": ["-I", "-X", "utf8", "-B"],
        "decoys_unchanged": untouched,
        "materialized_matches_action": materialized,
        "passed": returncode == 0
        and not decode_errors
        and stdout.strip() == case.value.encode()
        and untouched
        and materialized,
        "workspace": str(root),
        "elapsed_seconds": time.perf_counter() - began,
    }
