"""Explicit UTF-8 child execution for the new literal-edit action path.

Historical bridges remain unchanged. The isolated child enables UTF-8 mode
explicitly because -I ignores PYTHONUTF8/PYTHONIOENCODING. Raw output is retained;
invalid UTF-8 is a failed execution with a null decoded field, never replacement
text. Prediction, motor, workspace, and evaluator contracts match the old bridge.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.neutral_document_python_session import Case
from experiments.r4_integrated.real_input_action_bridge import (
    ActionTask,
    _files_record,
    escape_literal_content,
)


def execute_utf8(case: Case, files, before, root: Path) -> dict:
    """Materialize and run these trusted, generated public fixtures only."""
    began = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    for item in files:
        path = (root / item.path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("workspace path escapes execution directory")
        path.parent.mkdir(parents=True, exist_ok=True)
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


def execute_prediction(
    prediction, query, inventory, motor, task: ActionTask, root: Path, *, reader
):
    """Execute one existing prediction; scoring expectations enter after action.

    ``query`` is the original public reading TokenInput. ``task.action_instruction``
    is a separately declared user instruction; the motor receives it unchanged.
    The caller owns the shared motor's two executed demonstrations and all source
    ingestion. No additional prediction, gradient update, or demonstration occurs.
    """
    parent, motor_parent = reader.checkpoint_sha256, motor.checkpoint_sha256
    if (
        prediction.checkpoint_sha256 != parent
        or prediction.query_sha256 != query.sha256
        or prediction.inventory_sha256 != inventory.sha256
        or prediction.stopped != (prediction.candidate_index is None)
    ):
        raise ValueError("prediction does not belong to the current reader/query/source")
    candidate = None
    if prediction.candidate_index is not None:
        index = int(prediction.candidate_index)
        if index != prediction.candidate_index or not 0 <= index < len(inventory.candidates):
            raise ValueError("selected occurrence is outside the full candidate inventory")
        candidate = inventory.candidates[index]
    emitted = None if candidate is None else candidate.emitted
    escaped = None if emitted is None else escape_literal_content(emitted)
    case = Case(task.name, task.key, task.stale, task.module, task.stale)
    before = case.files()
    action = None if escaped is None else motor.apply(task.action_instruction, before, escaped)
    after = before if action is None else action.files
    if reader.checkpoint_sha256 != parent or motor.checkpoint_sha256 != motor_parent:
        raise RuntimeError("actuation changed learned reader or motor state")

    # The expectation is not supplied to either the predictor or motor.
    expected = task.stale if task.expected_answer is None else task.expected_answer.decode("utf-8")
    execution = execute_utf8(replace(case, value=expected), after, before, root)
    if reader.checkpoint_sha256 != parent or motor.checkpoint_sha256 != motor_parent:
        raise RuntimeError("execution changed learned reader or motor state")
    correct_no_action = task.expected_answer is None and prediction.stopped and after == before
    success = execution["passed"] and (
        correct_no_action if task.expected_answer is None else emitted == task.expected_answer
    )
    return {
        "task": task.name,
        "reading_question_hex": query.raw.hex(),
        "reading_query_sha256": query.sha256,
        "action_instruction_hex": task.action_instruction.hex(),
        "inventory_sha256": inventory.sha256,
        "candidate_count": len(inventory.candidates),
        "reader_checkpoint_sha256": parent,
        "motor_checkpoint_sha256": motor_parent,
        "candidate_index": None if candidate is None else int(prediction.candidate_index),
        "stopped": bool(prediction.stopped),
        "stop_reason": prediction.stop_reason,
        "p_stop": float(prediction.p_stop),
        "span_scores_sha256": sha256(prediction.span_scores.tobytes()).hexdigest(),
        "selected_span_hex": None if emitted is None else emitted.hex(),
        "literal_payload_hex": None if escaped is None else escaped.hex(),
        "expected_answer_hex": None if task.expected_answer is None else task.expected_answer.hex(),
        "evidence_correct": emitted == task.expected_answer,
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
        "correct_no_action": correct_no_action,
        "success": success,
        "source_storage": (
            "paragraph and normalized word vectors remain retained; workspace contains no passage"
        ),
        "actuation": (
            "unchanged learned occurrence motor; lossless UTF-8/single-quoted Python escaping"
        ),
        "execution_scope": (
            "unsupported execution PASS certifies unchanged stale state; "
            "success also requires reading STOP"
        ),
    }
