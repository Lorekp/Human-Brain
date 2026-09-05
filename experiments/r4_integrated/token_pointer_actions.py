"""Source-qualified endpoint actions without materializing a span inventory.

The caller supplies the already selected pointer and its frozen prediction
custody. Annotation occurrences enter validation/scoring only. This module never
predicts, trains a reader/motor, selects a source, or chooses an answer for it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.fresh_source_actions import _digest, _execute_existing, _state
from experiments.r4_integrated.real_input_action_bridge import (
    ActionTask,
    _files_record,
    escape_literal_content,
)
from experiments.r4_integrated.utf8_action_bridge import Case, execute_utf8
from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile

Pointer = tuple[int, int, int]
Occurrence = tuple[bytes, int, int]


@dataclass(frozen=True, slots=True)
class PointerCustody:
    selected: Pointer | None
    reader_checkpoint_sha256: str
    query_sha256: str
    source_pool_sha256: str


def _source_records(documents):
    records = []
    for document in documents:
        raw, spans = document.tokens.raw, document.tokens.byte_spans
        text = raw.decode("utf-8", errors="strict")
        offsets = [0]
        for char in text:
            offsets.append(offsets[-1] + len(char.encode("utf-8")))
        expected = tuple((offsets[m.start()], offsets[m.end()]) for m in re.finditer(r"\w+", text))
        if spans != expected or not document.document_id or not document.provenance_id:
            raise ValueError("complete original UTF-8 word/source metadata required")
        records.append(
            {
                "document_id_hex": document.document_id.hex(),
                "provenance_id_hex": document.provenance_id.hex(),
                "source_sha256": sha256(raw).hexdigest(),
                "token_input_sha256": document.tokens.sha256,
                "word_spans": spans,
            }
        )
    if not records or len({r["document_id_hex"] for r in records}) != len(records):
        raise ValueError("nonempty source pool with unique document identities required")
    return records


def source_pool_sha256(documents) -> str:
    """Bind the ordered complete sources and original word boundaries, not spans."""
    return _digest(_source_records(documents))


def _occurrence(documents, selected):
    if (
        not isinstance(selected, tuple)
        or len(selected) != 3
        or any(type(i) is not int for i in selected)
    ):
        raise ValueError("pointer must be three Python integers with inclusive word end")
    d, start, end = selected
    if not 0 <= d < len(documents):
        raise ValueError("pointer document is outside the complete source pool")
    document = documents[d]
    spans = document.tokens.byte_spans
    if not 0 <= start <= end < len(spans):
        raise ValueError("pointer endpoints are outside original word boundaries")
    a, b = spans[start][0], spans[end][1]
    raw = document.tokens.raw[a:b]
    raw.decode("utf-8", errors="strict")
    return {
        "document_index": d,
        "document_id_hex": document.document_id.hex(),
        "provenance_id_hex": document.provenance_id.hex(),
        "source_sha256": sha256(document.tokens.raw).hexdigest(),
        "word_start": start,
        "word_end_inclusive": end,
        "byte_start": a,
        "byte_end": b,
        "emitted_hex": raw.hex(),
    }


def _targets(task, documents, accepted):
    match = re.fullmatch(r"qa(\d{3})", task.key)
    if (
        match is None
        or not 0 <= int(match[1]) < 28
        or task.module != "qa_config" + match[1]
        or task.action_instruction != ("Configure " + task.key).encode()
        or task.stale != "pending"
    ):
        raise ValueError("explicit ordinal task metadata differs from the declared roster")
    if accepted is None:
        if task.expected_answer is not None:
            raise ValueError("STOP annotation and teacher answer disagree")
        return None
    if not isinstance(accepted, tuple) or not accepted or len(set(accepted)) != len(accepted):
        raise ValueError("distinct nonempty original annotated occurrences required")
    by_id = {document.document_id: i for i, document in enumerate(documents)}
    records = []
    for item in accepted:
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or not isinstance(item[0], bytes)
            or any(type(i) is not int for i in item[1:])
            or item[0] not in by_id
        ):
            raise ValueError("annotation does not identify an original source interval")
        document_id, a, b = item
        d = by_id[document_id]
        spans = documents[d].tokens.byte_spans
        starts, ends = (
            {x[0]: i for i, x in enumerate(spans)},
            {x[1]: i for i, x in enumerate(spans)},
        )
        if a not in starts or b not in ends or starts[a] > ends[b]:
            raise ValueError("annotation must preserve complete original word endpoints")
        records.append(_occurrence(documents, (d, starts[a], ends[b])))
    if len({r["document_id_hex"] for r in records}) != 1 or task.expected_answer is None:
        raise ValueError("official answers must belong to one original source")
    if task.expected_answer.hex() not in {r["emitted_hex"] for r in records}:
        raise ValueError("canonical teacher answer is absent from accepted occurrences")
    return records


def execute_prediction(
    selection: Pointer | None,
    query,
    documents,
    motor,
    task: ActionTask,
    root: Path,
    *,
    reader,
    custody: PointerCustody,
    accepted_occurrences: tuple[Occurrence, ...] | None,
    semantic_scored: bool = True,
):
    """Execute a precomputed endpoint choice; annotations never enter the motor."""
    documents = tuple(documents)
    pool = source_pool_sha256(documents)
    if type(semantic_scored) is not bool or (
        not semantic_scored and accepted_occurrences is not None
    ):
        raise ValueError("diagnostic-only actions require official STOP annotation")
    gold = _targets(task, documents, accepted_occurrences)
    parent, motor_parent = _state(reader, motor)
    if (
        not isinstance(custody, PointerCustody)
        or custody.selected != selection
        or custody.reader_checkpoint_sha256 != parent
        or custody.query_sha256 != query.sha256
        or custody.source_pool_sha256 != pool
    ):
        raise ValueError("pointer differs from its reader/query/complete-source custody")
    occurrence = None if selection is None else _occurrence(documents, selection)
    emitted = None if occurrence is None else bytes.fromhex(occurrence["emitted_hex"])
    payload = None if emitted is None else escape_literal_content(emitted)
    case = Case(task.name, task.key, task.stale, task.module, task.stale)
    before = case.files()
    action = None if payload is None else motor.apply(task.action_instruction, before, payload)
    after = before if action is None else action.files
    if _state(reader, motor) != (parent, motor_parent):
        raise RuntimeError("actuation changed learned state")
    selected_value = task.stale if emitted is None else emitted.decode("utf-8", errors="strict")
    execution = execute_utf8(replace(case, value=selected_value), after, before, root)
    if _state(reader, motor) != (parent, motor_parent):
        raise RuntimeError("execution changed learned state")
    agreement = occurrence is None if gold is None else occurrence in gold
    correct_no_action = gold is None and occurrence is None and after == before
    lossless = bool(
        execution["passed"]
        and bytes.fromhex(execution["stdout_hex"]) == (selected_value + "\n").encode("utf-8")
        and execution["stderr_hex"] == ""
    )
    result = {
        "task": task.name,
        "reading_question_hex": query.raw.hex(),
        "reading_query_sha256": query.sha256,
        "source_pool_sha256": pool,
        "source_count": len(documents),
        "action_instruction_hex": task.action_instruction.hex(),
        "reader_checkpoint_sha256": parent,
        "motor_checkpoint_sha256": motor_parent,
        "selected_pointer": selection,
        "selected_occurrence": occurrence,
        "selected_span_hex": None if emitted is None else emitted.hex(),
        "literal_payload_hex": None if payload is None else payload.hex(),
        "stopped": selection is None,
        "canonical_teacher_answer_hex": None
        if task.expected_answer is None
        else task.expected_answer.hex(),
        "accepted_gold_occurrences": gold,
        "gold_custody_sha256": _digest(gold),
        "semantic_scored": semantic_scored,
        "official_label_agreement": agreement,
        "evidence_correct": agreement if semantic_scored else None,
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
        "success": bool(lossless and agreement and (gold is not None or correct_no_action))
        if semantic_scored
        else None,
        "execution_scope": "transport passed is separate from source-occurrence correctness",
        "source_storage": "original sources and word inputs remain retained; no span inventory",
    }
    result["record_sha256"] = _digest(result)
    return result


def execute_teacher(
    failed_row,
    task: ActionTask,
    query,
    documents,
    *,
    reader,
    motor,
    accepted_occurrences: tuple[Occurrence, ...] | None,
):
    """Execute one external repair in the authenticated failed workspace."""
    documents = tuple(documents)
    pool = source_pool_sha256(documents)
    gold = _targets(task, documents, accepted_occurrences)
    preserved = json.loads(json.dumps(failed_row, allow_nan=False))
    if (
        preserved.pop("record_sha256", None) != _digest(preserved)
        or failed_row["semantic_scored"] is not True
        or failed_row["success"] is not False
    ):
        raise ValueError("teacher requires an intact scored unsuccessful first attempt")
    parent, motor_parent = _state(reader, motor)
    expected_hex = None if task.expected_answer is None else task.expected_answer.hex()
    if (
        failed_row["task"] != task.name
        or failed_row["reading_query_sha256"] != query.sha256
        or failed_row["source_pool_sha256"] != pool
        or failed_row["reader_checkpoint_sha256"] != parent
        or failed_row["motor_checkpoint_sha256"] != motor_parent
        or failed_row["action_instruction_hex"] != task.action_instruction.hex()
        or failed_row["gold_custody_sha256"] != _digest(gold)
        or failed_row["canonical_teacher_answer_hex"] != expected_hex
    ):
        raise ValueError("teacher annotation/source/state differs from its own first attempt")
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
            raise ValueError("failed workspace changed before teacher correction")
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
    execution = _execute_existing(
        replace(case, value=expected.decode("utf-8")), repaired, original, root
    )
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
