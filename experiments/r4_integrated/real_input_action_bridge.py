"""Declared instruction-to-literal-edit bridge for unchanged public QA inputs.

The reading core selects from its complete paragraph-local span inventory. A
separate user instruction addresses the workspace through the existing learned
motor. UTF-8 decoding and single-quoted Python literal escaping are fixed
actuation scaffolding, not learned comprehension or answer selection. This
module never trains the reader or motor and never supplies a teacher repair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.neutral_document_python_session import Case, execute


@dataclass(frozen=True, slots=True)
class ActionTask:
    name: str
    action_instruction: bytes
    key: str
    module: str
    stale: str
    expected_answer: bytes | None

    def __post_init__(self):
        if (
            not self.name
            or not isinstance(self.action_instruction, bytes)
            or not self.action_instruction
        ):
            raise ValueError("task identity and explicit user action instruction required")
        if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", x) is None for x in (self.key, self.module)):
            raise ValueError("workspace names must be declared Python identifiers")
        if re.fullmatch(r"[A-Za-z0-9_]+", self.stale) is None:
            raise ValueError("the existing motor prior requires one initial lexical value")
        if self.expected_answer is not None:
            if not isinstance(self.expected_answer, bytes) or not self.expected_answer:
                raise ValueError("supported expectation must be nonempty UTF-8 bytes")
            self.expected_answer.decode("utf-8", errors="strict")
            if self.expected_answer == self.stale.encode():
                raise ValueError("supported answer must differ from the initial value")


def escape_literal_content(value: bytes) -> bytes:
    """Encode the exact selected string inside existing single quotes, losslessly."""
    if not isinstance(value, bytes) or not value:
        raise ValueError("selected span must be nonempty UTF-8 bytes")
    return value.decode("utf-8", errors="strict").encode("unicode_escape").replace(b"'", b"\\'")


def _files_record(files):
    return [
        {
            "path": item.path,
            "sha256": sha256(item.content).hexdigest(),
            "content_hex": item.content.hex(),
        }
        for item in files
    ]


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
    execution = execute(replace(case, value=expected), after, before, root)
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
