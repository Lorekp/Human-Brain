"""Frozen one-shot iterator cohort for continual lexical Rust action reuse.

The four training lessons below are host-supplied synthetic successes.  They are
not live compile/run observations.  The fresh oracle is a separate artifact and
is opened exactly once, only after both selected Rust sources have compiled and
run and their V2 receipt trees have authenticated.

The supported claim is deliberately narrow: lexical association/reuse plus
authenticated custody and trusted-host execution of candidate-provided sources.
This is not source synthesis, semantic Rust understanding, debugging, transfer,
SWE performance, comparison with an external model, or intelligence.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import suppress
from ctypes import wintypes
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import common_continual_documentation_action_memory_v1 as continual
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from experiments.r3_pilot import common_promoted_rust_action_vault_v1 as rust_vault
from humanbrain import rust_selected_action_executor

SCHEMA = "humanbrain.continual-rust-iterator-cohort.v1"
READY_SCHEMA = "humanbrain.continual-rust-iterator-cohort.ready.v1"
RESULT_SCHEMA = "humanbrain.continual-rust-iterator-cohort.result.v1"
SPEND_SCHEMA = "humanbrain.continual-rust-iterator-cohort.spend.v1"
BATCH_SCHEMA = "humanbrain.continual-rust-iterator-cohort.batch.v1"
ORACLE_SCHEMA = "humanbrain.continual-rust-iterator-cohort.oracle.v1"

CLAIM_BOUNDARY = (
    "one-frozen-synthetic-two-action-iterator-cohort-with-four-host-supplied-"
    "successful-lessons-replay-free-immutable-continual-lexical-association-"
    "two-fresh-exact-factor-overlap-recalls-and-authenticated-candidate-provided-"
    "rust-source-custody-compile-run-and-post-execution-oracle-match-not-live-"
    "consequence-learning-source-synthesis-semantics-debugging-transfer-swe-"
    "external-model-comparison-or-intelligence"
)

MODULE_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = MODULE_PATH.parents[2]
BATCH_PATH = MODULE_PATH.with_suffix(".batch.json")
ORACLE_PATH = MODULE_PATH.with_suffix(".oracle.json")
READY_PATH = MODULE_PATH.with_suffix(".ready.json")
SPEND_PATH = MODULE_PATH.with_suffix(".spent.json")
RESULT_PATH = MODULE_PATH.with_suffix(".result.json")
_CANONICAL_MODULE_PATH = MODULE_PATH
_CANONICAL_REPOSITORY_ROOT = REPOSITORY_ROOT
_CANONICAL_BATCH_PATH = BATCH_PATH
_CANONICAL_ORACLE_PATH = ORACLE_PATH
_CANONICAL_READY_PATH = READY_PATH
_CANONICAL_SPEND_PATH = SPEND_PATH
_CANONICAL_RESULT_PATH = RESULT_PATH

IMPLEMENTATION_RELATIVE_PATHS = (
    "experiments/g0/common_event_cells_v1.py",
    "experiments/r3_pilot/common_continual_documentation_action_memory_v1.py",
    "experiments/r3_pilot/common_knowledge_action_memory_v1.py",
    "experiments/r3_pilot/common_promoted_rust_action_vault_v1.py",
    "src/humanbrain/cells.py",
    "src/humanbrain/human_cell_identity.py",
    "src/humanbrain/knowledge_cell_projection.py",
    "src/humanbrain/knowledge_consolidation.py",
    "src/humanbrain/native.py",
    "src/humanbrain/resting_glif_race.py",
    "src/humanbrain/rust_action_registry.py",
    "src/humanbrain/rust_selected_action_executor.py",
)

EXPECTED_BATCH_SHA256 = "d0606ad50a02d0e6895cace4494392851411b9d2e547ef16c8d9fdada7790fb0"
EXPECTED_ORACLE_SHA256 = "d1c80c7ffef644ecaca6f51e4793ae3e897e6a13b987fd948e51eacf57b2b62e"
RUST_BOOK_REVISION = "844ddcd4d6d7726143ed8165950e395ccfd1f155"
RUST_BOOK_ITERATOR_SOURCE_SHA256 = (
    "5f2a112bb6bbda36bc2ae2bf29196b433dd0fbcf649fcb0dd3f9448094592fd1"
)
RUST_BOOK_FILE_INVENTORY_SHA256 = "d6513177013ca7c4c5d2253bf0bb2e508a5ba379f400b231287f45cd32b5c444"
RUST_BOOK_ARCHIVE_SHA256 = "74fb097c9fb2b8c4cccabfe0e6eb060a78add3037f051932d60e347e68286dca"
RUST_BOOK_ITERATOR_URL = (
    f"https://github.com/rust-lang/book/blob/{RUST_BOOK_REVISION}/src/ch13-02-iterators.md"
)

MAP_CANDIDATE_ID = "iterator_map_double"
FILTER_CANDIDATE_ID = "iterator_filter_nonnegative"
MAP_CANDIDATE_PROJECTION_SOURCE = (
    "def solve(values):\n    return tools.iterator_map_double(values)\n"
)
FILTER_CANDIDATE_PROJECTION_SOURCE = (
    "def solve(values):\n    return tools.iterator_filter_nonnegative(values)\n"
)
MAP_RUST_SOURCE = """pub fn solve(values: &[i64]) -> Vec<i64> {
    values.iter().map(|value| *value * 2).collect()
}
"""
FILTER_RUST_SOURCE = """pub fn solve(values: &[i64]) -> Vec<i64> {
    values
        .iter()
        .copied()
        .filter(|value| *value >= 0)
        .collect()
}
"""

TRAINING_ROWS = (
    (
        "A1",
        MAP_CANDIDATE_ID,
        "Using iterator map, double every input value and collect the results in order.",
        "tok:map",
        ((), (3, -2, 0), (-7, 4, 4, 1)),
        ((), (6, -4, 0), (-14, 8, 8, 2)),
    ),
    (
        "B1",
        FILTER_CANDIDATE_ID,
        "Using iterator filter, keep values greater than or equal to zero and "
        "preserve their order.",
        "tok:filter",
        ((), (-3, 0, 5, -1), (-8, -2)),
        ((), (0, 5), ()),
    ),
    (
        "A2",
        MAP_CANDIDATE_ID,
        "Apply map to produce two times each item, preserving order in the output vector.",
        "tok:map",
        ((9,), (-1, -5, 6), (2, 8, -3, 0)),
        ((18,), (-2, -10, 12), (4, 16, -6, 0)),
    ),
    (
        "B2",
        FILTER_CANDIDATE_ID,
        "Apply filter to remove every negative item while leaving the surviving items ordered.",
        "tok:filter",
        ((4, -4, 1, 0), (-1, 7, -9, 7), (0,)),
        ((4, 1, 0), (7, 7), (0,)),
    ),
)

FRESH_NAMES = ("A_prime", "B_prime")
FRESH_TASKS = (
    "Build the output with map: each input item becomes twice its original value, "
    "in the same order.",
    "Build the output with filter: retain only zero-or-positive items in their original order.",
)
UNKNOWN_CONTROL_TASK = "Build the output with fold: combine the items into one accumulated total."

READY_COUNTERS = tuple(
    sorted(
        {
            "candidate_count": 2,
            "promoted_candidate_count": 2,
            "synthetic_training_lesson_count": 4,
            "training_compile_invocation_count": 0,
            "training_run_invocation_count": 0,
            "frozen_executor_preparation_count": 2,
            "evaluation_vault_recall_attempt_count": 0,
            "evaluation_vault_recall_count": 0,
            "evaluation_control_recall_count": 0,
            "evaluation_executor_preparation_attempt_count": 0,
            "evaluation_executor_preparation_count": 0,
            "evaluation_executor_call_count": 0,
            "evaluation_release_attempt_count": 0,
            "evaluation_authenticated_release_count": 0,
            "evaluation_authenticated_compile_invocation_count": 0,
            "evaluation_authenticated_run_invocation_count": 0,
            "evaluation_control_release_count": 0,
            "evaluation_unselected_release_count": 0,
            "oracle_open_attempt_count": 0,
            "oracle_read_count": 0,
        }.items()
    )
)

_READY_NAMESPACE = b"continual-rust-iterator-cohort-ready-v1"
_RESULT_NAMESPACE = b"continual-rust-iterator-cohort-result-v1"
_SPEND_NAMESPACE = b"continual-rust-iterator-cohort-spend-v1"
_PREPARATION_TEMPLATE_NAMESPACE = b"continual-rust-iterator-preparation-template-v1"
_TRAINING_TASK_NAMESPACE = b"continual-rust-iterator-cohort-training-task-v1"
_TRAINING_EXECUTION_NAMESPACE = b"continual-rust-iterator-cohort-training-execution-v1"
_TRAINING_FEEDBACK_NAMESPACE = b"continual-rust-iterator-cohort-training-feedback-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

PASS_EVENTS = (
    "READY_AUTHENTICATED",
    "CURRICULUM_REBUILT",
    "VAULT_PREPARED",
    "SPEND_MARKER_PUBLISHED",
    "ALL_RECALLS_ISSUED_BEFORE_RELEASE",
    "RECALL_PHASE_AUTHENTICATED",
    "BOTH_EXECUTORS_PREPARED_AFTER_RECALLS",
    "A_PRIME_EXECUTION_AUTHENTICATED",
    "B_PRIME_EXECUTION_AUTHENTICATED",
    "BOTH_EXECUTION_TREES_AUTHENTICATED",
    "ORACLE_OPENED_ONCE_AFTER_BOTH_EXECUTIONS",
    "ADJUDICATED_WITHOUT_MEMORY_UPDATE",
)


class ContinualRustIteratorCohortError(ValueError):
    """The frozen cohort, ordering, custody, or result failed authentication."""


def _assert_canonical_artifact_paths() -> None:
    pairs = (
        (MODULE_PATH, _CANONICAL_MODULE_PATH),
        (BATCH_PATH, _CANONICAL_BATCH_PATH),
        (ORACLE_PATH, _CANONICAL_ORACLE_PATH),
        (READY_PATH, _CANONICAL_READY_PATH),
        (SPEND_PATH, _CANONICAL_SPEND_PATH),
        (RESULT_PATH, _CANONICAL_RESULT_PATH),
    )
    if (
        not isinstance(REPOSITORY_ROOT, Path)
        or REPOSITORY_ROOT.resolve() != _CANONICAL_REPOSITORY_ROOT
        or any(not isinstance(path, Path) or path.resolve() != expected for path, expected in pairs)
    ):
        raise ContinualRustIteratorCohortError("canonical cohort artifact paths were redirected")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assert_local_implementation_modules() -> None:
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        module_path = relative.removeprefix("src/").removesuffix(".py")
        module_name = module_path.replace("/", ".")
        module = sys.modules.get(module_name)
        expected_path = (_CANONICAL_REPOSITORY_ROOT / relative).resolve()
        if (
            module is None
            or getattr(module, "__file__", None) is None
            or Path(module.__file__).resolve() != expected_path
        ):
            raise ContinualRustIteratorCohortError(
                "cohort implementation modules were imported outside the repository"
            )


def _implementation_file_sha256s() -> tuple[tuple[str, str], ...]:
    _assert_local_implementation_modules()
    return tuple(
        (relative, _file_sha256(_CANONICAL_REPOSITORY_ROOT / relative))
        for relative in IMPLEMENTATION_RELATIVE_PATHS
    )


def _preparation_template_sha256(
    preparation: rust_selected_action_executor.RustSelectedSourcePreparationV2,
) -> str:
    preparation.authenticate_structure()
    record = asdict(preparation)
    record.pop("expected_authority_sha256")
    record.pop("sha256")
    return _digest(_PREPARATION_TEMPLATE_NAMESPACE, record)


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _plain(value: object) -> object:
    try:
        return json.loads(_canonical_json(_jsonable(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustIteratorCohortError(
            "serialized record is not canonical JSON data"
        ) from exc


def _exact_record(
    value: object,
    expected_keys: set[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected_keys:
        raise ContinualRustIteratorCohortError(f"{label} fields changed")
    return dict(value)


def _dataclass_record(value: object, kind: type, *, label: str) -> dict[str, object]:
    return _exact_record(
        value,
        {field.name for field in fields(kind)},
        label=label,
    )


def _tupleish(value: object, *, label: str) -> tuple:
    if type(value) not in (list, tuple):
        raise ContinualRustIteratorCohortError(f"{label} must be one exact sequence")
    return tuple(value)


def _tuple_rows(value: object, *, label: str) -> tuple[tuple, ...]:
    return tuple(_tupleish(row, label=f"{label} row") for row in _tupleish(value, label=label))


@dataclass(frozen=True, slots=True)
class ReadyUnspentV1:
    schema: str
    status: str
    claim_boundary: str
    implementation_sha256: str
    implementation_file_sha256s: tuple[tuple[str, str], ...]
    batch_sha256: str
    oracle_filename: str
    oracle_sha256: str
    rust_book_revision: str
    rust_book_iterator_url: str
    rust_book_iterator_source_sha256: str
    rust_book_file_inventory_sha256: str
    lesson_sha256s: tuple[str, ...]
    checkpoint_sha256s: tuple[str, ...]
    final_checkpoint_array_sha256s: tuple[str, ...]
    candidate_sha256s: tuple[str, ...]
    candidate_projection_source_sha256s: tuple[str, ...]
    promoted_candidate_sha256s: tuple[str, ...]
    provenance_sha256s: tuple[str, ...]
    vault_receipt_sha256: str
    vault_binding_sha256s: tuple[str, ...]
    actual_rust_source_sha256s: tuple[str, ...]
    rustc_identity_sha256: str
    v2_preparation_template_sha256s: tuple[str, ...]
    v2_batch_sha256s: tuple[str, ...]
    v2_stdin_sha256s: tuple[str, ...]
    work_counters: tuple[tuple[str, int], ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_sha256:
            result.pop("sha256")
        return result

    def authenticate_structure(self) -> None:
        if type(self.implementation_file_sha256s) is not tuple or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or not _is_sha256(row[1])
            for row in self.implementation_file_sha256s
        ):
            raise ContinualRustIteratorCohortError("READY implementation records changed")
        roots = (
            self.implementation_sha256,
            *(root for _path, root in self.implementation_file_sha256s),
            self.batch_sha256,
            self.oracle_sha256,
            self.rust_book_iterator_source_sha256,
            self.rust_book_file_inventory_sha256,
            *self.lesson_sha256s,
            *self.checkpoint_sha256s,
            *self.final_checkpoint_array_sha256s,
            *self.candidate_sha256s,
            *self.candidate_projection_source_sha256s,
            *self.promoted_candidate_sha256s,
            *self.provenance_sha256s,
            self.vault_receipt_sha256,
            *self.vault_binding_sha256s,
            *self.actual_rust_source_sha256s,
            self.rustc_identity_sha256,
            *self.v2_preparation_template_sha256s,
            *self.v2_batch_sha256s,
            *self.v2_stdin_sha256s,
        )
        tuple_groups = (
            self.lesson_sha256s,
            self.checkpoint_sha256s,
            self.final_checkpoint_array_sha256s,
            self.candidate_sha256s,
            self.candidate_projection_source_sha256s,
            self.promoted_candidate_sha256s,
            self.provenance_sha256s,
            self.vault_binding_sha256s,
            self.actual_rust_source_sha256s,
            self.v2_preparation_template_sha256s,
            self.v2_batch_sha256s,
            self.v2_stdin_sha256s,
        )
        if (
            self.schema != READY_SCHEMA
            or self.status != "READY_UNSPENT"
            or self.claim_boundary != CLAIM_BOUNDARY
            or self.batch_sha256 != EXPECTED_BATCH_SHA256
            or self.oracle_filename != _CANONICAL_ORACLE_PATH.name
            or self.oracle_sha256 != EXPECTED_ORACLE_SHA256
            or self.rust_book_revision != RUST_BOOK_REVISION
            or self.rust_book_iterator_url != RUST_BOOK_ITERATOR_URL
            or self.rust_book_iterator_source_sha256 != RUST_BOOK_ITERATOR_SOURCE_SHA256
            or self.rust_book_file_inventory_sha256 != RUST_BOOK_FILE_INVENTORY_SHA256
            or self.work_counters != READY_COUNTERS
            or self.implementation_file_sha256s
            != tuple(
                (path, root)
                for path, root in self.implementation_file_sha256s
                if path in IMPLEMENTATION_RELATIVE_PATHS and _is_sha256(root)
            )
            or tuple(path for path, _root_value in self.implementation_file_sha256s)
            != IMPLEMENTATION_RELATIVE_PATHS
            or any(type(group) is not tuple for group in tuple_groups)
            or len(self.lesson_sha256s) != 4
            or len(self.checkpoint_sha256s) != 4
            or len(self.final_checkpoint_array_sha256s) != 3
            or len(self.candidate_sha256s) != 2
            or len(self.candidate_projection_source_sha256s) != 2
            or len(self.promoted_candidate_sha256s) != 2
            or len(self.provenance_sha256s) != 4
            or len(self.vault_binding_sha256s) != 2
            or len(self.actual_rust_source_sha256s) != 2
            or len(self.v2_preparation_template_sha256s) != 2
            or len(self.v2_batch_sha256s) != 2
            or len(self.v2_stdin_sha256s) != 2
            or self.promoted_candidate_sha256s != self.candidate_sha256s
            or any(len(set(group)) != len(group) for group in tuple_groups)
            or any(not _is_sha256(value) for value in roots)
            or _GIT_SHA_RE.fullmatch(self.rust_book_revision) is None
            or self.sha256 != _digest(_READY_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustIteratorCohortError("READY_UNSPENT receipt authentication failed")


@dataclass(frozen=True, slots=True)
class _PreparedCohortV1:
    memory: continual.FrozenContinualDocumentationActionMemoryV1
    vault: rust_vault.PreparedPromotedRustActionVaultV1
    lessons: tuple[continual.ContinualDocumentationActionLessonV1, ...]
    checkpoint_sha256s: tuple[str, ...]
    map_candidate: knowledge_action.KnowledgeActionCandidateProjectionV1
    filter_candidate: knowledge_action.KnowledgeActionCandidateProjectionV1


@dataclass(frozen=True, slots=True)
class _FreshRowV1:
    name: str
    task: str
    cases: tuple[tuple[int, ...], ...]


def _candidate(candidate_id: str, projection_source: str):
    return knowledge_action.project_knowledge_action_candidate_v1(
        candidate_id,
        projection_source,
    )


def _synthetic_lesson(
    row: tuple[
        str,
        str,
        str,
        str,
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
    ],
    *,
    candidates: dict[str, knowledge_action.KnowledgeActionCandidateProjectionV1],
) -> continual.ContinualDocumentationActionLessonV1:
    label, candidate_id, task_text, factor, inputs, outputs = row
    task = knowledge_action.project_knowledge_action_task_v1(task_text)
    evidence = continual.make_documentation_action_evidence_v1(
        factors=(factor,),
        source_sha256=RUST_BOOK_ITERATOR_SOURCE_SHA256,
        provenance_roots=(
            RUST_BOOK_ARCHIVE_SHA256,
            RUST_BOOK_FILE_INVENTORY_SHA256,
            _root(f"rust-book-revision:{RUST_BOOK_REVISION}"),
        ),
    )
    synthetic_execution = _digest(
        _TRAINING_EXECUTION_NAMESPACE,
        {"label": label, "inputs": inputs, "outputs": outputs},
    )
    synthetic_feedback = _digest(
        _TRAINING_FEEDBACK_NAMESPACE,
        {"label": label, "passed": True},
    )
    return continual.seal_continual_documentation_action_lesson_v1(
        task=task,
        task_instance_sha256=_digest(
            _TRAINING_TASK_NAMESPACE,
            {"label": label, "task": task_text, "inputs": inputs},
        ),
        task_source_sha256=_root(f"iterator-cohort:task-source:{label}"),
        evidence=evidence,
        candidate=candidates[candidate_id],
        execution_sha256=synthetic_execution,
        public_feedback_sha256=synthetic_feedback,
        observed_outputs=outputs,
        expected_outputs=outputs,
    )


def _prepare_continual_rust_iterator_cohort_v1() -> _PreparedCohortV1:
    map_candidate = _candidate(MAP_CANDIDATE_ID, MAP_CANDIDATE_PROJECTION_SOURCE)
    filter_candidate = _candidate(
        FILTER_CANDIDATE_ID,
        FILTER_CANDIDATE_PROJECTION_SOURCE,
    )
    candidates = {
        MAP_CANDIDATE_ID: map_candidate,
        FILTER_CANDIDATE_ID: filter_candidate,
    }
    lessons = tuple(_synthetic_lesson(row, candidates=candidates) for row in TRAINING_ROWS)
    memory = continual.start_continual_documentation_action_memory_v1(lessons[0])
    checkpoint_sha256s = [memory.checkpoint_sha256]
    for lesson in lessons[1:]:
        parent = memory
        parent_root = parent.checkpoint_sha256
        parent_arrays = parent.checkpoint_array_sha256s
        memory = continual.advance_continual_documentation_action_memory_v1(parent, lesson)
        parent.authenticate()
        if (
            parent.checkpoint_sha256 != parent_root
            or parent.checkpoint_array_sha256s != parent_arrays
        ):
            raise ContinualRustIteratorCohortError("continual advance mutated a prior checkpoint")
        checkpoint_sha256s.append(memory.checkpoint_sha256)
    memory.authenticate()
    if memory.promoted_candidate_sha256s != (
        map_candidate.sha256,
        filter_candidate.sha256,
    ):
        raise ContinualRustIteratorCohortError(
            "the fixed synthetic curriculum did not promote both actions"
        )
    vault = rust_vault.prepare_promoted_rust_action_vault_v1(
        memory,
        candidate_sources=(
            (map_candidate, MAP_RUST_SOURCE),
            (filter_candidate, FILTER_RUST_SOURCE),
        ),
    )
    return _PreparedCohortV1(
        memory=memory,
        vault=vault,
        lessons=lessons,
        checkpoint_sha256s=tuple(checkpoint_sha256s),
        map_candidate=map_candidate,
        filter_candidate=filter_candidate,
    )


def _load_fresh_batch(
    path: Path = _CANONICAL_BATCH_PATH,
) -> tuple[_FreshRowV1, _FreshRowV1]:
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != EXPECTED_BATCH_SHA256:
        raise ContinualRustIteratorCohortError("fresh batch bytes changed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustIteratorCohortError("fresh batch is not canonical JSON") from exc
    if type(payload) is not dict or set(payload) != {"schema", "rows"}:
        raise ContinualRustIteratorCohortError("fresh batch fields changed")
    if (
        payload["schema"] != BATCH_SCHEMA
        or type(payload["rows"]) is not list
        or len(payload["rows"]) != 2
    ):
        raise ContinualRustIteratorCohortError("fresh batch schema changed")
    rows: list[_FreshRowV1] = []
    for index, item in enumerate(payload["rows"]):
        if type(item) is not dict or set(item) != {"name", "task", "cases"}:
            raise ContinualRustIteratorCohortError("fresh batch row fields changed")
        if (
            item["name"] != FRESH_NAMES[index]
            or item["task"] != FRESH_TASKS[index]
            or type(item["cases"]) is not list
        ):
            raise ContinualRustIteratorCohortError("fresh batch row changed")
        cases: list[tuple[int, ...]] = []
        for case in item["cases"]:
            if type(case) is not list or any(type(value) is not int for value in case):
                raise ContinualRustIteratorCohortError("fresh batch case changed")
            cases.append(tuple(case))
        rows.append(_FreshRowV1(item["name"], item["task"], tuple(cases)))
    return rows[0], rows[1]


def make_ready_unspent_v1() -> ReadyUnspentV1:
    """Build the deterministic pre-execution receipt without opening the oracle."""

    if not _CANONICAL_ORACLE_PATH.is_file():
        raise ContinualRustIteratorCohortError("sealed oracle artifact is absent")
    batch = _load_fresh_batch()
    prepared = _prepare_continual_rust_iterator_cohort_v1()
    memory = prepared.memory
    vault_receipt = prepared.vault.receipt
    frozen_executors = (
        rust_selected_action_executor.prepare_rust_selected_action_executor_v2(
            prepared.vault,
            batch[0].cases,
        ),
        rust_selected_action_executor.prepare_rust_selected_action_executor_v2(
            prepared.vault,
            batch[1].cases,
        ),
    )
    rustc_identity_sha256s = tuple(executor.rustc_identity.sha256 for executor in frozen_executors)
    if len(set(rustc_identity_sha256s)) != 1:
        raise ContinualRustIteratorCohortError("frozen executors found different toolchains")
    preparations = tuple(executor.preparation for executor in frozen_executors)
    payload = {
        "schema": READY_SCHEMA,
        "status": "READY_UNSPENT",
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_sha256": _file_sha256(_CANONICAL_MODULE_PATH),
        "implementation_file_sha256s": _implementation_file_sha256s(),
        "batch_sha256": EXPECTED_BATCH_SHA256,
        "oracle_filename": _CANONICAL_ORACLE_PATH.name,
        "oracle_sha256": EXPECTED_ORACLE_SHA256,
        "rust_book_revision": RUST_BOOK_REVISION,
        "rust_book_iterator_url": RUST_BOOK_ITERATOR_URL,
        "rust_book_iterator_source_sha256": RUST_BOOK_ITERATOR_SOURCE_SHA256,
        "rust_book_file_inventory_sha256": RUST_BOOK_FILE_INVENTORY_SHA256,
        "lesson_sha256s": tuple(lesson.sha256 for lesson in prepared.lessons),
        "checkpoint_sha256s": prepared.checkpoint_sha256s,
        "final_checkpoint_array_sha256s": memory.checkpoint_array_sha256s,
        "candidate_sha256s": tuple(candidate.sha256 for candidate in memory.candidate_projections),
        "candidate_projection_source_sha256s": tuple(
            candidate.source_sha256 for candidate in memory.candidate_projections
        ),
        "promoted_candidate_sha256s": memory.promoted_candidate_sha256s,
        "provenance_sha256s": tuple(row.sha256 for row in memory.provenance),
        "vault_receipt_sha256": vault_receipt.sha256,
        "vault_binding_sha256s": vault_receipt.binding_sha256s,
        "actual_rust_source_sha256s": vault_receipt.rust_source_sha256s,
        "rustc_identity_sha256": rustc_identity_sha256s[0],
        "v2_preparation_template_sha256s": tuple(
            _preparation_template_sha256(item) for item in preparations
        ),
        "v2_batch_sha256s": tuple(item.batch_sha256 for item in preparations),
        "v2_stdin_sha256s": tuple(item.stdin_sha256 for item in preparations),
        "work_counters": READY_COUNTERS,
    }
    ready = ReadyUnspentV1(
        **payload,
        sha256=_digest(_READY_NAMESPACE, payload),
    )
    ready.authenticate_structure()
    return ready


def _ready_from_record(value: object) -> ReadyUnspentV1:
    copied = _plain(value)
    record = _dataclass_record(copied, ReadyUnspentV1, label="READY receipt")
    tuple_fields = (
        "lesson_sha256s",
        "checkpoint_sha256s",
        "final_checkpoint_array_sha256s",
        "candidate_sha256s",
        "candidate_projection_source_sha256s",
        "promoted_candidate_sha256s",
        "provenance_sha256s",
        "vault_binding_sha256s",
        "actual_rust_source_sha256s",
        "v2_preparation_template_sha256s",
        "v2_batch_sha256s",
        "v2_stdin_sha256s",
    )
    try:
        for field in tuple_fields:
            record[field] = _tupleish(record[field], label=f"READY {field}")
        record["implementation_file_sha256s"] = _tuple_rows(
            record["implementation_file_sha256s"],
            label="READY implementation records",
        )
        record["work_counters"] = _tuple_rows(
            record["work_counters"],
            label="READY work counters",
        )
        ready = ReadyUnspentV1(**record)
    except (KeyError, TypeError, IndexError, ContinualRustIteratorCohortError) as exc:
        raise ContinualRustIteratorCohortError("READY receipt fields changed") from exc
    ready.authenticate_structure()
    if _plain(ready.record()) != copied:
        raise ContinualRustIteratorCohortError("READY receipt normalization changed")
    return ready


def load_ready_unspent_v1(path: Path = _CANONICAL_READY_PATH) -> ReadyUnspentV1:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustIteratorCohortError("READY receipt could not be read") from exc
    return _ready_from_record(payload)


def _publish_pending_exclusive(pending: Path, path: Path) -> None:
    """Atomically publish one complete same-directory file without replacement."""

    if os.name == "nt":
        move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move_file.restype = wintypes.BOOL
        if not move_file(str(pending), str(path), 0x00000008):
            error = ctypes.get_last_error()
            if error in (80, 183):
                raise ContinualRustIteratorCohortError(
                    f"exclusive artifact already exists: {path.name}"
                )
            raise OSError(error, ctypes.FormatError(error), str(path))
        return
    try:
        os.link(pending, path)
    except FileExistsError as exc:
        raise ContinualRustIteratorCohortError(
            f"exclusive artifact already exists: {path.name}"
        ) from exc


def _write_exclusive_json(path: Path, value: object) -> str:
    """Publish complete canonical JSON bytes without replacing an existing path."""

    if not path.parent.is_dir():
        raise ContinualRustIteratorCohortError(f"output directory is missing: {path.parent}")
    encoded = _canonical_json(_jsonable(value)) + b"\n"
    descriptor, pending_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pending",
    )
    pending = Path(pending_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if pending.read_bytes() != encoded:
            raise ContinualRustIteratorCohortError("pending artifact bytes changed after fsync")
        _publish_pending_exclusive(pending, path)
        observed = path.read_bytes()
        if observed != encoded:
            raise ContinualRustIteratorCohortError("published artifact bytes changed")
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return sha256(observed).hexdigest()
    finally:
        with suppress(FileNotFoundError):
            pending.unlink()


def freeze_ready_unspent(path: Path = _CANONICAL_READY_PATH) -> ReadyUnspentV1:
    """Publish the deterministic all-zero READY receipt without using the oracle."""

    ready = make_ready_unspent_v1()
    _write_exclusive_json(path, ready.record())
    return ready


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=_CANONICAL_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ContinualRustIteratorCohortError(f"git {' '.join(arguments)} failed: {detail}")
    return completed


def _relative(path: Path) -> str:
    return path.relative_to(_CANONICAL_REPOSITORY_ROOT).as_posix()


def _verify_protocol_scope(protocol_commit: str, ready: ReadyUnspentV1) -> None:
    if _GIT_SHA_RE.fullmatch(protocol_commit) is None:
        raise ContinualRustIteratorCohortError("protocol commit is not one exact Git SHA")
    head = _git("rev-parse", "HEAD").stdout.strip()
    if head != protocol_commit:
        raise ContinualRustIteratorCohortError("HEAD differs from the frozen protocol commit")
    tracked = tuple(
        _relative(path)
        for path in (
            _CANONICAL_MODULE_PATH,
            _CANONICAL_BATCH_PATH,
            _CANONICAL_ORACLE_PATH,
            _CANONICAL_READY_PATH,
        )
    )
    _git("ls-files", "--error-unmatch", "--", *tracked)
    exclusions = (
        ":(exclude).codex/config.toml",
        f":(exclude){_relative(_CANONICAL_ORACLE_PATH)}",
    )
    if _git(
        "diff",
        "--quiet",
        protocol_commit,
        "--",
        ".",
        *exclusions,
        check=False,
    ).returncode:
        raise ContinualRustIteratorCohortError(
            "tracked pre-oracle repository inputs have worktree changes"
        )
    if _git(
        "diff",
        "--cached",
        "--quiet",
        protocol_commit,
        "--",
        ".",
        *exclusions,
        check=False,
    ).returncode:
        raise ContinualRustIteratorCohortError(
            "tracked pre-oracle repository inputs have staged changes"
        )
    if (
        _file_sha256(_CANONICAL_MODULE_PATH) != ready.implementation_sha256
        or _file_sha256(_CANONICAL_BATCH_PATH) != ready.batch_sha256
        or _implementation_file_sha256s() != ready.implementation_file_sha256s
    ):
        raise ContinualRustIteratorCohortError("READY-bound protocol bytes changed")
    if _CANONICAL_SPEND_PATH.exists() or _CANONICAL_RESULT_PATH.exists():
        raise ContinualRustIteratorCohortError(
            "canonical spend or terminal artifact already exists"
        )


def _make_spend_record(protocol_commit: str, ready: ReadyUnspentV1) -> dict[str, object]:
    payload = {
        "schema": SPEND_SCHEMA,
        "status": "SPENT_NO_RETRY",
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol_commit": protocol_commit,
        "ready_sha256": ready.sha256,
    }
    return {**payload, "sha256": _digest(_SPEND_NAMESPACE, payload)}


def _authenticate_spend_current(
    path: Path,
    *,
    expected: dict[str, object],
    expected_file_sha256: str,
) -> None:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustIteratorCohortError("current spend marker could not be read") from exc
    canonical = _canonical_json(_jsonable(expected)) + b"\n"
    if (
        type(value) is not dict
        or value != expected
        or raw != canonical
        or not _is_sha256(expected_file_sha256)
        or sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise ContinualRustIteratorCohortError("current spend marker authentication failed")


def _spend_publication_state(
    path: Path,
    *,
    expected: dict[str, object],
    expected_file_sha256: str,
) -> str:
    try:
        path.stat()
    except FileNotFoundError:
        return "ABSENT"
    except OSError:
        return "AMBIGUOUS"
    try:
        _authenticate_spend_current(
            path,
            expected=expected,
            expected_file_sha256=expected_file_sha256,
        )
    except ContinualRustIteratorCohortError:
        return "AMBIGUOUS"
    return "EXACT"


@dataclass(frozen=True, slots=True)
class _SpendMarkerGuard:
    handle: object
    windows: bool


@dataclass(slots=True)
class _ResultReservation:
    handle: object
    windows: bool
    path: Path
    published: bool = False
    released: bool = False


def _reserve_result_path(path: Path) -> _ResultReservation:
    """Create and hold the terminal path before any irreversible evaluation work."""

    if not path.parent.is_dir():
        raise ContinualRustIteratorCohortError(f"output directory is missing: {path.parent}")
    if os.name != "nt":
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ContinualRustIteratorCohortError(
                f"exclusive artifact already exists: {path.name}"
            ) from exc
        return _ResultReservation(descriptor, False, path)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0xC0010000,
        0,
        None,
        1,
        0x80000080,
        None,
    )
    if handle in (None, ctypes.c_void_p(-1).value):
        error = ctypes.get_last_error()
        if error in (80, 183):
            raise ContinualRustIteratorCohortError(
                f"exclusive artifact already exists: {path.name}"
            )
        raise OSError(error, ctypes.FormatError(error), str(path))
    return _ResultReservation(handle, True, path)


def _write_reserved_result_json(reservation: _ResultReservation, value: object) -> str:
    """Fill one held empty result path exactly once and verify its durable bytes."""

    if type(reservation) is not _ResultReservation or reservation.published or reservation.released:
        raise ContinualRustIteratorCohortError("terminal result reservation was already consumed")
    encoded = _canonical_json(_jsonable(value)) + b"\n"
    if not reservation.windows:
        descriptor = reservation.handle
        held = os.fstat(descriptor)
        linked = reservation.path.stat()
        if held.st_size != 0 or (held.st_dev, held.st_ino) != (linked.st_dev, linked.st_ino):
            raise ContinualRustIteratorCohortError("reserved terminal path changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("terminal result write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = os.read(descriptor, len(encoded) + 1)
        held = os.fstat(descriptor)
        linked = reservation.path.stat()
        if (held.st_dev, held.st_ino) != (linked.st_dev, linked.st_ino):
            raise ContinualRustIteratorCohortError("reserved terminal path changed")
        directory = os.open(reservation.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    else:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_size = kernel32.GetFileSizeEx
        get_size.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong))
        get_size.restype = wintypes.BOOL
        size = ctypes.c_longlong()
        if not get_size(reservation.handle, ctypes.byref(size)):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(reservation.path))
        if size.value != 0:
            raise ContinualRustIteratorCohortError("reserved terminal path changed")
        write_file = kernel32.WriteFile
        write_file.argtypes = (
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        )
        write_file.restype = wintypes.BOOL
        buffer = ctypes.create_string_buffer(encoded)
        write_count = wintypes.DWORD()
        if not write_file(
            reservation.handle,
            buffer,
            len(encoded),
            ctypes.byref(write_count),
            None,
        ):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(reservation.path))
        if write_count.value != len(encoded):
            raise ContinualRustIteratorCohortError("terminal result write was incomplete")
        flush_file = kernel32.FlushFileBuffers
        flush_file.argtypes = (wintypes.HANDLE,)
        flush_file.restype = wintypes.BOOL
        if not flush_file(reservation.handle):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(reservation.path))
        size = ctypes.c_longlong()
        if not get_size(reservation.handle, ctypes.byref(size)) or size.value != len(encoded):
            raise ContinualRustIteratorCohortError("reserved terminal size changed")
        set_pointer = kernel32.SetFilePointerEx
        set_pointer.argtypes = (
            wintypes.HANDLE,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.DWORD,
        )
        set_pointer.restype = wintypes.BOOL
        if not set_pointer(reservation.handle, 0, None, 0):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(reservation.path))
        read_file = kernel32.ReadFile
        read_file.argtypes = (
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        )
        read_file.restype = wintypes.BOOL
        read_buffer = ctypes.create_string_buffer(len(encoded))
        read_count = wintypes.DWORD()
        if not read_file(
            reservation.handle,
            read_buffer,
            len(encoded),
            ctypes.byref(read_count),
            None,
        ):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(reservation.path))
        observed = read_buffer.raw[: read_count.value]
    if observed != encoded or (
        not reservation.windows and reservation.path.read_bytes() != encoded
    ):
        raise ContinualRustIteratorCohortError("published terminal bytes changed")
    reservation.published = True
    return sha256(observed).hexdigest()


def _release_result_reservation(reservation: _ResultReservation) -> None:
    if type(reservation) is not _ResultReservation or reservation.released:
        raise ContinualRustIteratorCohortError("terminal result reservation was already released")
    if reservation.windows:
        close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        if not close_handle(reservation.handle):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error))
    else:
        os.close(reservation.handle)
    reservation.released = True


def _discard_empty_result_reservation(
    reservation: _ResultReservation,
    *,
    spend_path: Path,
) -> None:
    """Remove only an exact empty reservation when no spend marker was published."""

    if (
        type(reservation) is not _ResultReservation
        or reservation.published
        or reservation.released
        or spend_path.exists()
    ):
        raise ContinualRustIteratorCohortError("terminal result reservation cannot be discarded")
    if reservation.windows:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_size = kernel32.GetFileSizeEx
        get_size.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong))
        get_size.restype = wintypes.BOOL
        size = ctypes.c_longlong()
        if not get_size(reservation.handle, ctypes.byref(size)) or size.value != 0:
            raise ContinualRustIteratorCohortError(
                "nonempty terminal reservation was not discarded"
            )

        class _FileDispositionInfo(ctypes.Structure):
            _fields_ = (("delete_file", wintypes.BOOL),)

        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        disposition = _FileDispositionInfo(True)
        if not set_information(
            reservation.handle,
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(reservation.path))
        _release_result_reservation(reservation)
    else:
        held = os.fstat(reservation.handle)
        linked = reservation.path.stat()
        if held.st_size != 0 or (held.st_dev, held.st_ino) != (linked.st_dev, linked.st_ino):
            raise ContinualRustIteratorCohortError("changed terminal reservation was not discarded")
        reservation.path.unlink()
        directory = os.open(reservation.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _release_result_reservation(reservation)
    if reservation.path.exists():
        raise ContinualRustIteratorCohortError("empty terminal reservation was not removed")


def _acquire_spend_marker_guard(
    path: Path,
    *,
    expected: dict[str, object],
    expected_file_sha256: str,
) -> _SpendMarkerGuard:
    """Hold the marker open, denying Windows write/delete sharing."""

    _authenticate_spend_current(
        path,
        expected=expected,
        expected_file_sha256=expected_file_sha256,
    )
    expected_bytes = _canonical_json(_jsonable(expected)) + b"\n"
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            observed = os.read(descriptor, len(expected_bytes) + 1)
            if observed != expected_bytes:
                raise ContinualRustIteratorCohortError("held spend marker bytes changed")
            return _SpendMarkerGuard(descriptor, False)
        except Exception:
            os.close(descriptor)
            raise

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000,
        0x00000001,
        None,
        3,
        0x00000080,
        None,
    )
    if handle in (None, ctypes.c_void_p(-1).value):
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error), str(path))
    try:
        get_size = kernel32.GetFileSizeEx
        get_size.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong))
        get_size.restype = wintypes.BOOL
        size = ctypes.c_longlong()
        if not get_size(handle, ctypes.byref(size)):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(path))
        if size.value != len(expected_bytes):
            raise ContinualRustIteratorCohortError("held spend marker size changed")
        read_file = kernel32.ReadFile
        read_file.argtypes = (
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        )
        read_file.restype = wintypes.BOOL
        buffer = ctypes.create_string_buffer(len(expected_bytes))
        read_count = wintypes.DWORD()
        if not read_file(
            handle,
            buffer,
            len(expected_bytes),
            ctypes.byref(read_count),
            None,
        ):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(path))
        observed = buffer.raw[: read_count.value]
        if observed != expected_bytes or sha256(observed).hexdigest() != expected_file_sha256:
            raise ContinualRustIteratorCohortError("held spend marker bytes changed")
        return _SpendMarkerGuard(handle, True)
    except Exception:
        kernel32.CloseHandle(handle)
        raise


def _release_spend_marker_guard(guard: _SpendMarkerGuard) -> None:
    if guard.windows:
        close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        if not close_handle(guard.handle):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error))
        return
    os.close(guard.handle)


def _stopped_recall_is_physical(
    recall: continual.ContinualDocumentationActionRecallV1,
    expected_status: str,
) -> bool:
    recall.authenticate()
    return (
        recall.status == expected_status
        and recall.selected_action_index is None
        and recall.selected_candidate_id is None
        and recall.selected_candidate_sha256 is None
        and recall.selected_source_sha256 is None
        and recall.physical_output_record.get("decision") == "STOP"
    )


def _execution_record(
    execution: rust_selected_action_executor.RustSelectedActionExecutionV2,
) -> dict[str, object]:
    execution.authenticate_structure()
    return {
        "sha256": execution.sha256,
        "authority_sha256": execution.selected_source.authority_sha256,
        "selection_sha256": execution.selected_source.selection_sha256,
        "release_sha256": execution.selected_source.release_sha256,
        "selected_source_grant_sha256": execution.selected_source.sha256,
        "action_handle": execution.selected_source.action_handle,
        "actual_rust_source_sha256": execution.selected_source.source_sha256,
        "candidate_source_bytes_in_result": 0,
        "rustc_identity": asdict(execution.rustc_identity),
        "preparation": asdict(execution.preparation),
        "dependency_receipt": asdict(execution.dependency_receipt),
        "compile_receipt": asdict(execution.compile_receipt),
        "run_receipt": asdict(execution.run_receipt),
        "outputs": execution.outputs,
    }


def _authenticate_executor_preparations(
    executors: tuple[
        rust_selected_action_executor.PreparedRustSelectedActionExecutorV2,
        rust_selected_action_executor.PreparedRustSelectedActionExecutorV2,
    ],
    *,
    ready: ReadyUnspentV1,
    authority_sha256: str,
) -> None:
    if (
        type(executors) is not tuple
        or len(executors) != 2
        or any(
            type(item) is not rust_selected_action_executor.PreparedRustSelectedActionExecutorV2
            for item in executors
        )
        or not _is_sha256(authority_sha256)
    ):
        raise ContinualRustIteratorCohortError(
            "executor preparations must be two exact V2 executors"
        )
    identities = tuple(item.rustc_identity for item in executors)
    preparations = tuple(item.preparation for item in executors)
    try:
        for identity in identities:
            identity.authenticate_structure()
        for preparation in preparations:
            preparation.authenticate_structure()
    except Exception as exc:
        raise ContinualRustIteratorCohortError(
            "executor preparations failed structural authentication"
        ) from exc
    if (
        tuple(item.sha256 for item in identities) != (ready.rustc_identity_sha256,) * 2
        or tuple(_preparation_template_sha256(item) for item in preparations)
        != ready.v2_preparation_template_sha256s
        or tuple(item.batch_sha256 for item in preparations) != ready.v2_batch_sha256s
        or tuple(item.stdin_sha256 for item in preparations) != ready.v2_stdin_sha256s
        or tuple(item.expected_authority_sha256 for item in preparations) != (authority_sha256,) * 2
    ):
        raise ContinualRustIteratorCohortError("live executor preparations differ from READY")


def _authenticate_execution_prefix(
    executions: tuple[rust_selected_action_executor.RustSelectedActionExecutionV2, ...],
    selections: tuple[continual.ContinualDocumentationActionRecallV1, ...],
    *,
    ready: ReadyUnspentV1,
    authority_sha256: str,
) -> None:
    count = len(executions) if type(executions) is tuple else 0
    if (
        type(executions) is not tuple
        or count not in {1, 2}
        or any(
            type(item) is not rust_selected_action_executor.RustSelectedActionExecutionV2
            for item in executions
        )
        or type(selections) is not tuple
        or len(selections) != count
        or any(
            type(item) is not continual.ContinualDocumentationActionRecallV1 for item in selections
        )
        or not _is_sha256(authority_sha256)
    ):
        raise ContinualRustIteratorCohortError(
            "execution prefix requires one or two exact authenticated V2 execution trees"
        )
    try:
        ready.authenticate_structure()
        for selection in selections:
            selection.authenticate()
        for execution in executions:
            execution.authenticate_structure()
    except Exception as exc:
        raise ContinualRustIteratorCohortError(
            "execution prefix custody authentication failed"
        ) from exc

    expected_task_sha256s = tuple(
        knowledge_action.project_knowledge_action_task_v1(task).sha256 for task in FRESH_TASKS
    )[:count]
    expected_candidate_ids = (MAP_CANDIDATE_ID, FILTER_CANDIDATE_ID)[:count]
    expected_sources = (MAP_RUST_SOURCE, FILTER_RUST_SOURCE)[:count]
    selected_sources = tuple(item.selected_source for item in executions)
    preparations = tuple(item.preparation for item in executions)
    if (
        tuple(item.status for item in selections) != (continual.RECALLED,) * count
        or tuple(item.task_sha256 for item in selections) != expected_task_sha256s
        or tuple(item.selected_candidate_id for item in selections) != expected_candidate_ids
        or tuple(item.selected_candidate_sha256 for item in selections)
        != ready.candidate_sha256s[:count]
        or tuple(item.selected_source_sha256 for item in selections)
        != ready.candidate_projection_source_sha256s[:count]
        or tuple(item.checkpoint_sha256 for item in selections)
        != (ready.checkpoint_sha256s[-1],) * count
        or tuple(item.authority_sha256 for item in selected_sources) != (authority_sha256,) * count
        or tuple(item.selection_sha256 for item in selected_sources)
        != tuple(item.sha256 for item in selections)
        or tuple(item.action_handle for item in selected_sources) != expected_candidate_ids
        or tuple(item.source for item in selected_sources) != expected_sources
        or tuple(item.source_sha256 for item in selected_sources)
        != ready.actual_rust_source_sha256s[:count]
        or tuple(item.rustc_identity.sha256 for item in executions)
        != (ready.rustc_identity_sha256,) * count
        or tuple(_preparation_template_sha256(item) for item in preparations)
        != ready.v2_preparation_template_sha256s[:count]
        or tuple(item.batch_sha256 for item in preparations) != ready.v2_batch_sha256s[:count]
        or tuple(item.stdin_sha256 for item in preparations) != ready.v2_stdin_sha256s[:count]
        or tuple(item.expected_authority_sha256 for item in preparations)
        != (authority_sha256,) * count
        or any(item.compile_receipt.compile_invocations != 1 for item in executions)
        or any(item.run_receipt.run_invocations != 1 for item in executions)
        or len({item.sha256 for item in selections}) != count
        or len({item.release_sha256 for item in selected_sources}) != count
        or len({item.sha256 for item in selected_sources}) != count
        or len({item.sha256 for item in executions}) != count
    ):
        raise ContinualRustIteratorCohortError(
            "execution prefix custody differs from the frozen protocol"
        )


def _authenticate_oracle_gate(
    executions: tuple[
        rust_selected_action_executor.RustSelectedActionExecutionV2,
        rust_selected_action_executor.RustSelectedActionExecutionV2,
    ],
    selections: tuple[
        continual.ContinualDocumentationActionRecallV1,
        continual.ContinualDocumentationActionRecallV1,
    ],
    *,
    ready: ReadyUnspentV1,
    authority_sha256: str,
) -> None:
    if (
        type(executions) is not tuple
        or len(executions) != 2
        or any(
            type(item) is not rust_selected_action_executor.RustSelectedActionExecutionV2
            for item in executions
        )
        or type(selections) is not tuple
        or len(selections) != 2
        or any(
            type(item) is not continual.ContinualDocumentationActionRecallV1 for item in selections
        )
        or not _is_sha256(authority_sha256)
    ):
        raise ContinualRustIteratorCohortError(
            "oracle access requires two exact authenticated V2 execution trees"
        )
    try:
        _authenticate_execution_prefix(
            executions,
            selections,
            ready=ready,
            authority_sha256=authority_sha256,
        )
    except ContinualRustIteratorCohortError as exc:
        raise ContinualRustIteratorCohortError(
            "oracle execution custody differs from the frozen protocol"
        ) from exc


def _parse_oracle_bytes(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_names: tuple[str, str],
) -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
    if type(raw) is not bytes or type(expected_names) is not tuple:
        raise ContinualRustIteratorCohortError("sealed oracle inputs changed")
    if sha256(raw).hexdigest() != expected_sha256:
        raise ContinualRustIteratorCohortError("sealed oracle bytes changed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustIteratorCohortError("sealed oracle is not JSON") from exc
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "rows"}
        or payload["schema"] != ORACLE_SCHEMA
        or type(payload["rows"]) is not list
        or len(payload["rows"]) != 2
    ):
        raise ContinualRustIteratorCohortError("sealed oracle fields changed")
    result: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
    for index, item in enumerate(payload["rows"]):
        if (
            type(item) is not dict
            or set(item) != {"name", "expected_outputs"}
            or item["name"] != expected_names[index]
            or type(item["expected_outputs"]) is not list
        ):
            raise ContinualRustIteratorCohortError("sealed oracle row changed")
        outputs: list[tuple[int, ...]] = []
        for row in item["expected_outputs"]:
            if type(row) is not list or any(type(value) is not int for value in row):
                raise ContinualRustIteratorCohortError("sealed oracle output changed")
            outputs.append(tuple(row))
        result.append((item["name"], tuple(outputs)))
    return tuple(result)


def _training_observation(prepared: _PreparedCohortV1) -> dict[str, object]:
    return {
        "kind": "host-supplied-synthetic-successes-not-live-executions",
        "lesson_order": tuple(row[0] for row in TRAINING_ROWS),
        "lesson_sha256s": tuple(lesson.sha256 for lesson in prepared.lessons),
        "checkpoint_sha256s": prepared.checkpoint_sha256s,
        "final_checkpoint_sha256": prepared.memory.checkpoint_sha256,
        "final_checkpoint_array_sha256s": prepared.memory.checkpoint_array_sha256s,
        "training_summary": asdict(prepared.memory.training_summary),
        "provenance": tuple(row.record() for row in prepared.memory.provenance),
    }


def _vault_observation(prepared: _PreparedCohortV1) -> dict[str, object]:
    return {
        "receipt": asdict(prepared.vault.receipt),
        "authority": asdict(prepared.vault.authority),
        "candidate_projection_source_sha256s": tuple(
            candidate.source_sha256 for candidate in prepared.memory.candidate_projections
        ),
        "actual_rust_source_sha256s": prepared.vault.receipt.rust_source_sha256s,
    }


def _candidate_from_record(
    value: object,
) -> knowledge_action.KnowledgeActionCandidateProjectionV1:
    copied = _plain(value)
    record = _dataclass_record(
        copied,
        knowledge_action.KnowledgeActionCandidateProjectionV1,
        label="serialized candidate",
    )
    try:
        candidate = knowledge_action.KnowledgeActionCandidateProjectionV1(**record)
        candidate.authenticate()
    except Exception as exc:
        raise ContinualRustIteratorCohortError(
            "serialized candidate authentication failed"
        ) from exc
    if _plain(asdict(candidate)) != copied:
        raise ContinualRustIteratorCohortError("serialized candidate normalization changed")
    return candidate


def _recall_from_record(
    value: object,
) -> continual.ContinualDocumentationActionRecallV1:
    copied = _plain(value)
    record = _dataclass_record(
        copied,
        continual.ContinualDocumentationActionRecallV1,
        label="serialized recall",
    )
    try:
        record["candidates"] = tuple(
            _candidate_from_record(item)
            for item in _tupleish(record["candidates"], label="recall candidates")
        )
        record["candidate_scores"] = _tuple_rows(
            record["candidate_scores"],
            label="recall candidate scores",
        )
        record["physical_raw_scores"] = _tuple_rows(
            record["physical_raw_scores"],
            label="recall physical raw scores",
        )
        physical = _exact_record(
            record["physical_output_record"],
            {
                "decision",
                "token",
                "active_cells",
                "lane_spike_steps",
                "lane_simultaneous_spike_candidates",
                "stop_reason",
                "physical_cell_steps",
                "sha256",
            },
            label="recall physical output",
        )
        for name in (
            "active_cells",
            "lane_spike_steps",
            "lane_simultaneous_spike_candidates",
        ):
            physical[name] = _tupleish(
                physical[name],
                label=f"recall physical output {name}",
            )
        record["physical_output_record"] = physical
        work_record = _exact_record(
            record["work"],
            {
                "query_projection_cell_steps",
                "factor_equality_cell_steps",
                "synaptic_reads",
                "output_race_cell_steps",
                "total_cell_steps",
            },
            label="recall work",
        )
        total_cell_steps = work_record.pop("total_cell_steps")
        work = continual.ContinualDocumentationActionRecallWorkV1(**work_record)
        if type(total_cell_steps) is not int or total_cell_steps != work.total_cell_steps:
            raise ContinualRustIteratorCohortError("recall total work changed")
        record["work"] = work
        recall = continual.ContinualDocumentationActionRecallV1(**record)
        recall.authenticate()
    except Exception as exc:
        raise ContinualRustIteratorCohortError("serialized recall authentication failed") from exc
    if _plain(recall.record()) != copied:
        raise ContinualRustIteratorCohortError("serialized recall normalization changed")
    return recall


def _vault_receipt_from_record(
    value: object,
) -> rust_vault.PromotedRustActionVaultReceiptV1:
    copied = _plain(value)
    record = _dataclass_record(
        copied,
        rust_vault.PromotedRustActionVaultReceiptV1,
        label="serialized vault receipt",
    )
    for name in (
        "candidate_sha256s",
        "promoted_candidate_sha256s",
        "provenance_sha256s",
        "binding_sha256s",
        "rust_source_sha256s",
    ):
        record[name] = _tupleish(record[name], label=f"vault receipt {name}")
    try:
        receipt = rust_vault.PromotedRustActionVaultReceiptV1(**record)
        receipt.authenticate_structure()
    except Exception as exc:
        raise ContinualRustIteratorCohortError(
            "serialized vault receipt authentication failed"
        ) from exc
    if _plain(asdict(receipt)) != copied:
        raise ContinualRustIteratorCohortError("serialized vault receipt normalization changed")
    return receipt


def _vault_authority_from_record(
    value: object,
) -> rust_vault.PromotedRustActionVaultAuthorityV1:
    copied = _plain(value)
    record = _dataclass_record(
        copied,
        rust_vault.PromotedRustActionVaultAuthorityV1,
        label="serialized vault authority",
    )
    try:
        authority = rust_vault.PromotedRustActionVaultAuthorityV1(**record)
        authority.authenticate_structure()
    except Exception as exc:
        raise ContinualRustIteratorCohortError(
            "serialized vault authority authentication failed"
        ) from exc
    if _plain(asdict(authority)) != copied:
        raise ContinualRustIteratorCohortError("serialized vault authority normalization changed")
    return authority


def _execution_from_record(
    value: object,
    *,
    expected_source: str,
) -> rust_selected_action_executor.RustSelectedActionExecutionV2:
    copied = _plain(value)
    record = _exact_record(
        copied,
        {
            "sha256",
            "authority_sha256",
            "selection_sha256",
            "release_sha256",
            "selected_source_grant_sha256",
            "action_handle",
            "actual_rust_source_sha256",
            "candidate_source_bytes_in_result",
            "rustc_identity",
            "preparation",
            "dependency_receipt",
            "compile_receipt",
            "run_receipt",
            "outputs",
        },
        label="serialized execution",
    )
    if (
        type(record["candidate_source_bytes_in_result"]) is not int
        or record["candidate_source_bytes_in_result"] != 0
    ):
        raise ContinualRustIteratorCohortError(
            "serialized execution exposed candidate source bytes"
        )
    try:
        selected_source = rust_selected_action_executor.make_rust_selected_source_v2(
            authority_sha256=record["authority_sha256"],
            selection_sha256=record["selection_sha256"],
            release_sha256=record["release_sha256"],
            action_handle=record["action_handle"],
            source=expected_source,
        )
        if (
            selected_source.sha256 != record["selected_source_grant_sha256"]
            or selected_source.source_sha256 != record["actual_rust_source_sha256"]
        ):
            raise ContinualRustIteratorCohortError("serialized selected-source grant changed")

        identity_record = _dataclass_record(
            record["rustc_identity"],
            rust_selected_action_executor.RustcIdentityV1,
            label="serialized rustc identity",
        )
        for name in (
            "link_search_paths",
            "link_search_manifest_sha256s",
            "link_search_file_counts",
            "link_search_total_bytes",
        ):
            identity_record[name] = _tupleish(
                identity_record[name],
                label=f"rustc identity {name}",
            )
        identity = rust_selected_action_executor.RustcIdentityV1(**identity_record)

        preparation_record = _dataclass_record(
            record["preparation"],
            rust_selected_action_executor.RustSelectedSourcePreparationV2,
            label="serialized preparation",
        )
        for name in ("compile_environment", "run_environment"):
            preparation_record[name] = _tuple_rows(
                preparation_record[name],
                label=f"preparation {name}",
            )
        preparation = rust_selected_action_executor.RustSelectedSourcePreparationV2(
            **preparation_record
        )

        dependency_record = _dataclass_record(
            record["dependency_receipt"],
            rust_selected_action_executor.RustSelectedSourceDependencyReceiptV2,
            label="serialized dependency receipt",
        )
        dependency_record["dependency_paths"] = _tupleish(
            dependency_record["dependency_paths"],
            label="dependency paths",
        )
        dependency_record["dependency_content_sha256s"] = _tuple_rows(
            dependency_record["dependency_content_sha256s"],
            label="dependency content roots",
        )
        dependency = rust_selected_action_executor.RustSelectedSourceDependencyReceiptV2(
            **dependency_record
        )

        compile_record = _dataclass_record(
            record["compile_receipt"],
            rust_selected_action_executor.RustSelectedSourceCompileReceiptV2,
            label="serialized compile receipt",
        )
        compile_record["arguments"] = _tupleish(
            compile_record["arguments"],
            label="compile arguments",
        )
        compile_receipt = rust_selected_action_executor.RustSelectedSourceCompileReceiptV2(
            **compile_record
        )
        run_receipt = rust_selected_action_executor.RustSelectedSourceRunReceiptV2(
            **_dataclass_record(
                record["run_receipt"],
                rust_selected_action_executor.RustSelectedSourceRunReceiptV2,
                label="serialized run receipt",
            )
        )
        outputs = _tuple_rows(record["outputs"], label="execution outputs")
        execution = rust_selected_action_executor.RustSelectedActionExecutionV2(
            selected_source=selected_source,
            rustc_identity=identity,
            preparation=preparation,
            dependency_receipt=dependency,
            compile_receipt=compile_receipt,
            run_receipt=run_receipt,
            outputs=outputs,
            sha256=record["sha256"],
        )
        execution.authenticate_structure()
    except Exception as exc:
        raise ContinualRustIteratorCohortError(
            "serialized execution authentication failed"
        ) from exc
    if _plain(_execution_record(execution)) != copied:
        raise ContinualRustIteratorCohortError("serialized execution normalization changed")
    return execution


def _authenticate_release_root(
    *,
    execution: rust_selected_action_executor.RustSelectedActionExecutionV2,
    selection: continual.ContinualDocumentationActionRecallV1,
    prepared: _PreparedCohortV1,
    authority_sha256: str,
    action_index: int,
) -> None:
    candidate = prepared.memory.candidate_projections[action_index]
    promotion_roots = tuple(
        row.sha256
        for row in prepared.memory.provenance
        if row.candidate_sha256 == candidate.sha256 and row.passed_cell == 1
    )
    try:
        release = rust_vault._PromotedRustActionReleaseV1(
            authority_sha256=authority_sha256,
            memory_checkpoint_sha256=selection.checkpoint_sha256,
            selection_sha256=selection.sha256,
            candidate_set_sha256=selection.candidate_set_sha256,
            selected_action_index=action_index,
            candidate_id=candidate.candidate_id,
            candidate_sha256=candidate.sha256,
            candidate_projection_source_sha256=candidate.source_sha256,
            binding_sha256=prepared.vault.receipt.binding_sha256s[action_index],
            rust_source_sha256=prepared.vault.receipt.rust_source_sha256s[action_index],
            promotion_provenance_sha256s=promotion_roots,
            sha256=execution.selected_source.release_sha256,
        )
        release.authenticate()
    except Exception as exc:
        raise ContinualRustIteratorCohortError(
            "serialized vault release authentication failed"
        ) from exc


def _initial_work() -> dict[str, int]:
    return dict(READY_COUNTERS)


def _authenticate_work_shape(work: object) -> dict[str, int]:
    expected_keys = {name for name, _value in READY_COUNTERS}
    if (
        type(work) is not dict
        or set(work) != expected_keys
        or any(type(value) is not int or value < 0 for value in work.values())
    ):
        raise ContinualRustIteratorCohortError("terminal work accounting changed")
    result = dict(work)
    immutable = {
        "candidate_count",
        "promoted_candidate_count",
        "synthetic_training_lesson_count",
        "training_compile_invocation_count",
        "training_run_invocation_count",
        "frozen_executor_preparation_count",
        "evaluation_control_release_count",
        "evaluation_unselected_release_count",
    }
    initial = _initial_work()
    if (
        any(result[name] != initial[name] for name in immutable)
        or result["evaluation_vault_recall_attempt_count"] > 5
        or result["evaluation_vault_recall_count"] > result["evaluation_vault_recall_attempt_count"]
        or result["evaluation_control_recall_count"] > result["evaluation_vault_recall_count"]
        or result["evaluation_control_recall_count"] > 3
        or result["evaluation_executor_preparation_attempt_count"] > 2
        or result["evaluation_executor_preparation_count"]
        > result["evaluation_executor_preparation_attempt_count"]
        or result["evaluation_executor_call_count"] > 2
        or result["evaluation_release_attempt_count"] != result["evaluation_executor_call_count"]
        or result["evaluation_authenticated_release_count"]
        > result["evaluation_release_attempt_count"]
        or result["evaluation_authenticated_compile_invocation_count"]
        > result["evaluation_authenticated_release_count"]
        or result["evaluation_authenticated_run_invocation_count"]
        > result["evaluation_authenticated_release_count"]
        or result["oracle_open_attempt_count"] > 1
        or result["oracle_read_count"] > result["oracle_open_attempt_count"]
    ):
        raise ContinualRustIteratorCohortError("terminal work invariants changed")
    return result


def _authenticate_base_observations(
    observations: object,
    *,
    ready_sha256: str,
) -> tuple[
    ReadyUnspentV1,
    _PreparedCohortV1,
    rust_vault.PromotedRustActionVaultReceiptV1,
    rust_vault.PromotedRustActionVaultAuthorityV1,
]:
    if type(observations) is not dict or not {
        "ready",
        "training",
        "vault",
    } <= set(observations):
        raise ContinualRustIteratorCohortError("terminal base observations are incomplete")
    ready = _ready_from_record(observations["ready"])
    if ready.sha256 != ready_sha256:
        raise ContinualRustIteratorCohortError("terminal READY root changed")
    prepared = _prepare_continual_rust_iterator_cohort_v1()
    if _plain(observations["training"]) != _plain(_training_observation(prepared)):
        raise ContinualRustIteratorCohortError("terminal training observation changed")
    vault_record = _exact_record(
        observations["vault"],
        {
            "receipt",
            "authority",
            "candidate_projection_source_sha256s",
            "actual_rust_source_sha256s",
        },
        label="terminal vault observation",
    )
    receipt = _vault_receipt_from_record(vault_record["receipt"])
    authority = _vault_authority_from_record(vault_record["authority"])
    candidate_source_roots = _tupleish(
        vault_record["candidate_projection_source_sha256s"],
        label="terminal candidate source roots",
    )
    actual_source_roots = _tupleish(
        vault_record["actual_rust_source_sha256s"],
        label="terminal Rust source roots",
    )
    if (
        receipt != prepared.vault.receipt
        or receipt.sha256 != ready.vault_receipt_sha256
        or receipt.memory_checkpoint_sha256 != ready.checkpoint_sha256s[-1]
        or receipt.candidate_sha256s != ready.candidate_sha256s
        or receipt.promoted_candidate_sha256s != ready.promoted_candidate_sha256s
        or receipt.provenance_sha256s != ready.provenance_sha256s
        or receipt.binding_sha256s != ready.vault_binding_sha256s
        or receipt.rust_source_sha256s != ready.actual_rust_source_sha256s
        or authority.vault_receipt_sha256 != receipt.sha256
        or candidate_source_roots != ready.candidate_projection_source_sha256s
        or actual_source_roots != ready.actual_rust_source_sha256s
    ):
        raise ContinualRustIteratorCohortError("terminal vault custody differs from READY")
    return ready, prepared, receipt, authority


def _recalls_from_observation(
    value: object,
    *,
    prepared: _PreparedCohortV1,
) -> dict[str, continual.ContinualDocumentationActionRecallV1]:
    names = {"cell_off", "documentation_off", "unknown", "A_prime", "B_prime"}
    records = _exact_record(value, names, label="terminal recall observation")
    recalls = {name: _recall_from_record(records[name]) for name in names}
    task_a = knowledge_action.project_knowledge_action_task_v1(FRESH_TASKS[0])
    task_b = knowledge_action.project_knowledge_action_task_v1(FRESH_TASKS[1])
    unknown = knowledge_action.project_knowledge_action_task_v1(UNKNOWN_CONTROL_TASK)
    ordered = tuple(
        recalls[name]
        for name in (
            "cell_off",
            "documentation_off",
            "unknown",
            "A_prime",
            "B_prime",
        )
    )
    expected_candidates = prepared.memory.candidate_projections
    if (
        tuple(item.status for item in ordered)
        != (
            continual.STOP_CELL_OFF,
            continual.STOP_DOCUMENTATION_OFF,
            continual.STOP_NO_MATCHED_FACTOR,
            continual.RECALLED,
            continual.RECALLED,
        )
        or tuple((item.cell_gate, item.documentation_gate) for item in ordered)
        != ((False, True), (True, False), (True, True), (True, True), (True, True))
        or tuple(item.task_sha256 for item in ordered)
        != (task_a.sha256, task_b.sha256, unknown.sha256, task_a.sha256, task_b.sha256)
        or any(item.checkpoint_sha256 != prepared.memory.checkpoint_sha256 for item in ordered)
        or any(item.candidates != expected_candidates for item in ordered)
        or len({item.sha256 for item in ordered}) != 5
        or not _stopped_recall_is_physical(
            recalls["cell_off"],
            continual.STOP_CELL_OFF,
        )
        or not _stopped_recall_is_physical(
            recalls["documentation_off"],
            continual.STOP_DOCUMENTATION_OFF,
        )
        or not _stopped_recall_is_physical(
            recalls["unknown"],
            continual.STOP_NO_MATCHED_FACTOR,
        )
        or recalls["A_prime"].selected_candidate_id != MAP_CANDIDATE_ID
        or recalls["B_prime"].selected_candidate_id != FILTER_CANDIDATE_ID
    ):
        raise ContinualRustIteratorCohortError(
            "terminal recall custody differs from the frozen cohort"
        )
    return recalls


def _normalize_nested_outputs(value: object, *, label: str) -> tuple[tuple[tuple, ...], ...]:
    return tuple(_tuple_rows(item, label=f"{label} item") for item in _tupleish(value, label=label))


def _execution_prefix_from_observation(
    value: object,
    *,
    expected_count: int,
    recalls: dict[str, continual.ContinualDocumentationActionRecallV1],
    ready: ReadyUnspentV1,
    prepared: _PreparedCohortV1,
    authority: rust_vault.PromotedRustActionVaultAuthorityV1,
) -> tuple[rust_selected_action_executor.RustSelectedActionExecutionV2, ...]:
    if type(expected_count) is not int or expected_count not in {1, 2}:
        raise ContinualRustIteratorCohortError("terminal execution prefix count changed")
    execution_records = _tupleish(value, label="terminal executions")
    if len(execution_records) != expected_count:
        raise ContinualRustIteratorCohortError("terminal execution count changed")
    expected_sources = (MAP_RUST_SOURCE, FILTER_RUST_SOURCE)[:expected_count]
    executions = tuple(
        _execution_from_record(record, expected_source=expected_source)
        for record, expected_source in zip(
            execution_records,
            expected_sources,
            strict=True,
        )
    )
    selections = tuple(recalls[name] for name in ("A_prime", "B_prime")[:expected_count])
    _authenticate_execution_prefix(
        executions,
        selections,
        ready=ready,
        authority_sha256=authority.sha256,
    )
    for index, (execution, selection) in enumerate(zip(executions, selections, strict=True)):
        _authenticate_release_root(
            execution=execution,
            selection=selection,
            prepared=prepared,
            authority_sha256=authority.sha256,
            action_index=index,
        )
    return executions


def _execution_pair_from_observation(
    value: object,
    *,
    recalls: dict[str, continual.ContinualDocumentationActionRecallV1],
    ready: ReadyUnspentV1,
    prepared: _PreparedCohortV1,
    authority: rust_vault.PromotedRustActionVaultAuthorityV1,
) -> tuple[
    rust_selected_action_executor.RustSelectedActionExecutionV2,
    rust_selected_action_executor.RustSelectedActionExecutionV2,
]:
    executions = _execution_prefix_from_observation(
        value,
        expected_count=2,
        recalls=recalls,
        ready=ready,
        prepared=prepared,
        authority=authority,
    )
    return executions[0], executions[1]


def _oracle_rows_from_observation(
    value: object,
    *,
    ready: ReadyUnspentV1,
) -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
    oracle = _exact_record(
        value,
        {"filename", "sha256", "read_count", "raw_utf8", "rows"},
        label="terminal oracle observation",
    )
    if type(oracle["raw_utf8"]) is not str:
        raise ContinualRustIteratorCohortError("terminal oracle raw bytes changed")
    try:
        raw_oracle = oracle["raw_utf8"].encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ContinualRustIteratorCohortError("terminal oracle raw bytes changed") from exc
    oracle_rows = _parse_oracle_bytes(
        raw_oracle,
        expected_sha256=ready.oracle_sha256,
        expected_names=FRESH_NAMES,
    )
    if (
        oracle["filename"] != ready.oracle_filename
        or oracle["sha256"] != ready.oracle_sha256
        or type(oracle["read_count"]) is not int
        or oracle["read_count"] != 1
        or _plain(oracle["rows"]) != _plain(oracle_rows)
    ):
        raise ContinualRustIteratorCohortError("terminal oracle custody changed")
    return oracle_rows


def _authenticate_adjudication_observation(
    value: object,
    *,
    executions: tuple[
        rust_selected_action_executor.RustSelectedActionExecutionV2,
        rust_selected_action_executor.RustSelectedActionExecutionV2,
    ],
    oracle_rows: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...],
) -> bool:
    adjudication = _exact_record(
        value,
        {
            "observed_outputs",
            "expected_outputs",
            "matched",
            "memory_update_from_evaluation_count",
        },
        label="terminal adjudication",
    )
    observed_outputs = _normalize_nested_outputs(
        adjudication["observed_outputs"],
        label="adjudication observed outputs",
    )
    expected_outputs = _normalize_nested_outputs(
        adjudication["expected_outputs"],
        label="adjudication expected outputs",
    )
    matched = observed_outputs == expected_outputs
    if (
        observed_outputs != tuple(item.outputs for item in executions)
        or expected_outputs != tuple(row[1] for row in oracle_rows)
        or type(adjudication["matched"]) is not bool
        or adjudication["matched"] is not matched
        or type(adjudication["memory_update_from_evaluation_count"]) is not int
        or adjudication["memory_update_from_evaluation_count"] != 0
    ):
        raise ContinualRustIteratorCohortError("terminal adjudication authentication failed")
    return matched


def _authenticate_completed_observations(
    value: dict[str, object],
    *,
    ready: ReadyUnspentV1,
    prepared: _PreparedCohortV1,
    authority: rust_vault.PromotedRustActionVaultAuthorityV1,
    work: dict[str, int],
) -> None:
    observations = _exact_record(
        value["observations"],
        {"ready", "training", "vault", "recalls", "executions", "oracle", "adjudication"},
        label="completed terminal observations",
    )
    recalls = _recalls_from_observation(observations["recalls"], prepared=prepared)
    executions = _execution_pair_from_observation(
        observations["executions"],
        recalls=recalls,
        ready=ready,
        prepared=prepared,
        authority=authority,
    )
    oracle_rows = _oracle_rows_from_observation(observations["oracle"], ready=ready)
    matched = _authenticate_adjudication_observation(
        observations["adjudication"],
        executions=executions,
        oracle_rows=oracle_rows,
    )
    expected_status = "PASS" if matched else "STOP_ORACLE_MISMATCH"
    if (
        value["status"] != expected_status
        or value["error"] is not None
        or tuple(value["events"]) != PASS_EVENTS
    ):
        raise ContinualRustIteratorCohortError("terminal adjudication authentication failed")

    expected_work = _initial_work()
    expected_work.update(
        {
            "evaluation_vault_recall_attempt_count": len(recalls),
            "evaluation_vault_recall_count": len(recalls),
            "evaluation_control_recall_count": 3,
            "evaluation_executor_preparation_attempt_count": len(executions),
            "evaluation_executor_preparation_count": len(executions),
            "evaluation_executor_call_count": len(executions),
            "evaluation_release_attempt_count": len(executions),
            "evaluation_authenticated_release_count": len(executions),
            "evaluation_authenticated_compile_invocation_count": sum(
                item.compile_receipt.compile_invocations for item in executions
            ),
            "evaluation_authenticated_run_invocation_count": sum(
                item.run_receipt.run_invocations for item in executions
            ),
            "oracle_open_attempt_count": 1,
            "oracle_read_count": 1,
        }
    )
    if work != expected_work:
        raise ContinualRustIteratorCohortError("completed terminal work accounting changed")


def _authenticate_failure_work(events: tuple, work: dict[str, int]) -> None:
    phase = len(events)
    if not 4 <= phase <= len(PASS_EVENTS):
        raise ContinualRustIteratorCohortError("terminal failure phase changed")

    if phase < 5:
        attempts = work["evaluation_vault_recall_attempt_count"]
        successful = work["evaluation_vault_recall_count"]
        recall_ok = (
            0 <= attempts <= 5
            and successful in {attempts, max(0, attempts - 1)}
            and work["evaluation_control_recall_count"] == min(successful, 3)
        )
    elif phase == 5:
        recall_ok = (
            work["evaluation_vault_recall_attempt_count"] == 5
            and work["evaluation_vault_recall_count"] == 5
            and work["evaluation_control_recall_count"] == 3
        )
    else:
        recall_ok = (
            work["evaluation_vault_recall_attempt_count"] == 5
            and work["evaluation_vault_recall_count"] == 5
            and work["evaluation_control_recall_count"] == 3
        )

    preparation = (
        work["evaluation_executor_preparation_attempt_count"],
        work["evaluation_executor_preparation_count"],
    )
    if phase <= 5:
        preparation_ok = preparation == (0, 0)
    elif phase == 6:
        preparation_ok = preparation in ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2))
    else:
        preparation_ok = preparation == (2, 2)

    release = (
        work["evaluation_executor_call_count"],
        work["evaluation_release_attempt_count"],
        work["evaluation_authenticated_release_count"],
        work["evaluation_authenticated_compile_invocation_count"],
        work["evaluation_authenticated_run_invocation_count"],
    )
    if phase <= 6:
        release_ok = release == (0, 0, 0, 0, 0)
    elif phase == 7:
        release_ok = release in ((0, 0, 0, 0, 0), (1, 1, 0, 0, 0))
    elif phase == 8:
        release_ok = release in ((1, 1, 1, 1, 1), (2, 2, 1, 1, 1))
    else:
        release_ok = release == (2, 2, 2, 2, 2)

    oracle = (work["oracle_open_attempt_count"], work["oracle_read_count"])
    if phase <= 9:
        oracle_ok = oracle == (0, 0)
    elif phase == 10:
        oracle_ok = oracle in ((0, 0), (1, 0), (1, 1))
    else:
        oracle_ok = oracle == (1, 1)
    if not (recall_ok and preparation_ok and release_ok and oracle_ok):
        raise ContinualRustIteratorCohortError("terminal failure work differs from its event phase")


def _authenticate_failure_observations(
    terminal: dict[str, object],
    *,
    events: tuple,
    ready: ReadyUnspentV1,
    prepared: _PreparedCohortV1,
    authority: rust_vault.PromotedRustActionVaultAuthorityV1,
    work: dict[str, int],
) -> None:
    expected_keys = {"ready", "training", "vault"}
    phase = len(events)
    if phase >= 6:
        expected_keys.add("recalls")
    if phase >= 8:
        expected_keys.add("executions")
    if phase >= 11:
        expected_keys.add("oracle")
    if phase >= 12:
        expected_keys.add("adjudication")
    observations = _exact_record(
        terminal["observations"],
        expected_keys,
        label="terminal failure observations",
    )
    recalls: dict[str, continual.ContinualDocumentationActionRecallV1] | None = None
    executions: (
        tuple[
            rust_selected_action_executor.RustSelectedActionExecutionV2,
            rust_selected_action_executor.RustSelectedActionExecutionV2,
        ]
        | None
    ) = None
    oracle_rows: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] | None = None
    if "recalls" in observations:
        recalls = _recalls_from_observation(observations["recalls"], prepared=prepared)
    if "executions" in observations:
        if recalls is None:
            raise ContinualRustIteratorCohortError("terminal failure executions lack recalls")
        execution_prefix = _execution_prefix_from_observation(
            observations["executions"],
            expected_count=1 if phase == 8 else 2,
            recalls=recalls,
            ready=ready,
            prepared=prepared,
            authority=authority,
        )
        if len(execution_prefix) == 2:
            executions = execution_prefix[0], execution_prefix[1]
    if "oracle" in observations:
        oracle_rows = _oracle_rows_from_observation(
            observations["oracle"],
            ready=ready,
        )
    if "adjudication" in observations:
        if executions is None or oracle_rows is None:
            raise ContinualRustIteratorCohortError("terminal failure adjudication lacks custody")
        _authenticate_adjudication_observation(
            observations["adjudication"],
            executions=executions,
            oracle_rows=oracle_rows,
        )
    _authenticate_failure_work(events, work)


def _terminal_record(
    *,
    status: str,
    protocol_commit: str,
    ready: ReadyUnspentV1,
    spend_sha256: str,
    spend_file_sha256: str,
    events: list[str],
    work: dict[str, int],
    observations: dict[str, object],
    error: dict[str, str] | None,
) -> dict[str, object]:
    payload = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol_commit": protocol_commit,
        "ready_sha256": ready.sha256,
        "spend_sha256": spend_sha256,
        "spend_file_sha256": spend_file_sha256,
        "events": tuple(events),
        "work": dict(sorted(work.items())),
        "observations": observations,
        "error": error,
    }
    return {**payload, "sha256": _digest(_RESULT_NAMESPACE, payload)}


def _authenticate_terminal_result_structure_v1(value: object) -> dict[str, object]:
    copied = _plain(value)
    terminal = _exact_record(
        copied,
        {
            "schema",
            "status",
            "claim_boundary",
            "protocol_commit",
            "ready_sha256",
            "spend_sha256",
            "spend_file_sha256",
            "events",
            "work",
            "observations",
            "error",
            "sha256",
        },
        label="terminal result",
    )
    result = dict(terminal)
    sha = result.pop("sha256")
    expected_spend_payload = {
        "schema": SPEND_SCHEMA,
        "status": "SPENT_NO_RETRY",
        "claim_boundary": CLAIM_BOUNDARY,
        "protocol_commit": terminal["protocol_commit"],
        "ready_sha256": terminal["ready_sha256"],
    }
    expected_spend_sha256 = _digest(_SPEND_NAMESPACE, expected_spend_payload)
    expected_spend = {
        **expected_spend_payload,
        "sha256": expected_spend_sha256,
    }
    expected_spend_file_sha256 = sha256(_canonical_json(expected_spend) + b"\n").hexdigest()
    events = _tupleish(terminal["events"], label="terminal events")
    work = _authenticate_work_shape(terminal["work"])
    allowed_statuses = {
        "PASS",
        "STOP_ORACLE_MISMATCH",
        "STOP_EXECUTION_PROTOCOL_FAILURE",
    }
    if (
        terminal["schema"] != RESULT_SCHEMA
        or terminal["status"] not in allowed_statuses
        or terminal["claim_boundary"] != CLAIM_BOUNDARY
        or type(terminal["protocol_commit"]) is not str
        or _GIT_SHA_RE.fullmatch(terminal["protocol_commit"]) is None
        or not _is_sha256(terminal["ready_sha256"])
        or terminal["spend_sha256"] != expected_spend_sha256
        or terminal["spend_file_sha256"] != expected_spend_file_sha256
        or type(terminal["observations"]) is not dict
        or len(events) > len(PASS_EVENTS)
        or events != PASS_EVENTS[: len(events)]
        or not _is_sha256(sha)
        or sha != _digest(_RESULT_NAMESPACE, result)
    ):
        raise ContinualRustIteratorCohortError("terminal result authentication failed")
    known_observations = {
        "ready",
        "training",
        "vault",
        "recalls",
        "executions",
        "oracle",
        "adjudication",
    }
    if not set(terminal["observations"]) <= known_observations:
        raise ContinualRustIteratorCohortError("terminal observation fields changed")
    ready, prepared, _receipt, authority = _authenticate_base_observations(
        terminal["observations"],
        ready_sha256=terminal["ready_sha256"],
    )
    if terminal["status"] in {"PASS", "STOP_ORACLE_MISMATCH"}:
        _authenticate_completed_observations(
            terminal,
            ready=ready,
            prepared=prepared,
            authority=authority,
            work=work,
        )
    else:
        error = _exact_record(
            terminal["error"],
            {"type", "message"},
            label="terminal failure",
        )
        if len(events) < 4 or any(
            type(error[name]) is not str or not error[name] for name in error
        ):
            raise ContinualRustIteratorCohortError("terminal failure evidence changed")
        _authenticate_failure_observations(
            terminal,
            events=events,
            ready=ready,
            prepared=prepared,
            authority=authority,
            work=work,
        )
    return terminal


def load_terminal_result_v1() -> dict[str, object]:
    _assert_canonical_artifact_paths()
    try:
        raw = _CANONICAL_RESULT_PATH.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustIteratorCohortError("terminal result could not be read") from exc
    if raw != _canonical_json(payload) + b"\n":
        raise ContinualRustIteratorCohortError("terminal result is not canonical JSON")
    result = _authenticate_terminal_result_structure_v1(payload)
    ready = load_ready_unspent_v1()
    if result["ready_sha256"] != ready.sha256:
        raise ContinualRustIteratorCohortError(
            "terminal result differs from the published READY receipt"
        )
    spend = _make_spend_record(result["protocol_commit"], ready)
    _authenticate_spend_current(
        _CANONICAL_SPEND_PATH,
        expected=spend,
        expected_file_sha256=result["spend_file_sha256"],
    )
    return result


def run_one_shot(*, protocol_commit: str) -> dict[str, object]:
    """Spend the frozen cohort once, opening its oracle only after both executions."""

    _assert_canonical_artifact_paths()
    ready = load_ready_unspent_v1()
    _verify_protocol_scope(protocol_commit, ready)
    expected_ready = make_ready_unspent_v1()
    if ready != expected_ready:
        raise ContinualRustIteratorCohortError("published READY receipt changed")
    batch = _load_fresh_batch()
    prepared = _prepare_continual_rust_iterator_cohort_v1()
    if prepared.vault.receipt.sha256 != ready.vault_receipt_sha256:
        raise ContinualRustIteratorCohortError("live vault differs from READY receipt")

    events = ["READY_AUTHENTICATED", "CURRICULUM_REBUILT", "VAULT_PREPARED"]
    work = _initial_work()
    observations: dict[str, object] = {
        "ready": ready.record(),
        "training": _training_observation(prepared),
        "vault": _vault_observation(prepared),
    }

    spend = _make_spend_record(protocol_commit, ready)
    spend_file_sha256 = sha256(_canonical_json(_jsonable(spend)) + b"\n").hexdigest()
    spend_state = "UNATTEMPTED"
    spend_guard: _SpendMarkerGuard | None = None
    result_reservation: _ResultReservation | None = None
    terminal: dict[str, object] | None = None
    try:
        result_reservation = _reserve_result_path(_CANONICAL_RESULT_PATH)
        try:
            observed_spend_file_sha256 = _write_exclusive_json(_CANONICAL_SPEND_PATH, spend)
        except Exception:
            spend_state = _spend_publication_state(
                _CANONICAL_SPEND_PATH,
                expected=spend,
                expected_file_sha256=spend_file_sha256,
            )
            if spend_state == "EXACT":
                events.append("SPEND_MARKER_PUBLISHED")
            raise
        spend_state = _spend_publication_state(
            _CANONICAL_SPEND_PATH,
            expected=spend,
            expected_file_sha256=spend_file_sha256,
        )
        if spend_state != "EXACT":
            raise ContinualRustIteratorCohortError("published spend marker became ambiguous")
        events.append("SPEND_MARKER_PUBLISHED")
        if observed_spend_file_sha256 != spend_file_sha256:
            raise ContinualRustIteratorCohortError("published spend file root changed")
        spend_guard = _acquire_spend_marker_guard(
            _CANONICAL_SPEND_PATH,
            expected=spend,
            expected_file_sha256=spend_file_sha256,
        )
        checkpoint_root = prepared.memory.checkpoint_sha256
        checkpoint_arrays = prepared.memory.checkpoint_array_sha256s
        task_a = knowledge_action.project_knowledge_action_task_v1(batch[0].task)
        task_b = knowledge_action.project_knowledge_action_task_v1(batch[1].task)
        unknown_task = knowledge_action.project_knowledge_action_task_v1(UNKNOWN_CONTROL_TASK)

        recalls: dict[str, continual.ContinualDocumentationActionRecallV1] = {}
        work["evaluation_vault_recall_attempt_count"] += 1
        recalls["cell_off"] = prepared.vault.recall(task_a, cell_gate=False)
        work["evaluation_vault_recall_count"] += 1
        work["evaluation_control_recall_count"] += 1
        work["evaluation_vault_recall_attempt_count"] += 1
        recalls["documentation_off"] = prepared.vault.recall(
            task_b,
            documentation_gate=False,
        )
        work["evaluation_vault_recall_count"] += 1
        work["evaluation_control_recall_count"] += 1
        work["evaluation_vault_recall_attempt_count"] += 1
        recalls["unknown"] = prepared.vault.recall(unknown_task)
        work["evaluation_vault_recall_count"] += 1
        work["evaluation_control_recall_count"] += 1
        work["evaluation_vault_recall_attempt_count"] += 1
        recalls["A_prime"] = prepared.vault.recall(task_a)
        work["evaluation_vault_recall_count"] += 1
        work["evaluation_vault_recall_attempt_count"] += 1
        recalls["B_prime"] = prepared.vault.recall(task_b)
        work["evaluation_vault_recall_count"] += 1
        events.append("ALL_RECALLS_ISSUED_BEFORE_RELEASE")

        controls = (
            (recalls["cell_off"], continual.STOP_CELL_OFF),
            (recalls["documentation_off"], continual.STOP_DOCUMENTATION_OFF),
            (recalls["unknown"], continual.STOP_NO_MATCHED_FACTOR),
        )
        stopped: list[bool] = []
        for recall, expected_status in controls:
            recall.authenticate()
            stopped.append(_stopped_recall_is_physical(recall, expected_status))
        for recall in (recalls["A_prime"], recalls["B_prime"]):
            recall.authenticate()
        if (
            stopped != [True, True, True]
            or recalls["A_prime"].status != continual.RECALLED
            or recalls["B_prime"].status != continual.RECALLED
            or recalls["A_prime"].selected_candidate_id != MAP_CANDIDATE_ID
            or recalls["B_prime"].selected_candidate_id != FILTER_CANDIDATE_ID
            or recalls["A_prime"].selected_candidate_sha256 != prepared.map_candidate.sha256
            or recalls["B_prime"].selected_candidate_sha256 != prepared.filter_candidate.sha256
            or prepared.memory.checkpoint_sha256 != checkpoint_root
            or prepared.memory.checkpoint_array_sha256s != checkpoint_arrays
        ):
            raise ContinualRustIteratorCohortError("recall phase did not match the frozen protocol")
        observations["recalls"] = {name: recall.record() for name, recall in recalls.items()}
        events.append("RECALL_PHASE_AUTHENTICATED")

        work["evaluation_executor_preparation_attempt_count"] += 1
        executor_a = rust_selected_action_executor.prepare_rust_selected_action_executor_v2(
            prepared.vault,
            batch[0].cases,
        )
        work["evaluation_executor_preparation_count"] += 1
        work["evaluation_executor_preparation_attempt_count"] += 1
        executor_b = rust_selected_action_executor.prepare_rust_selected_action_executor_v2(
            prepared.vault,
            batch[1].cases,
        )
        work["evaluation_executor_preparation_count"] += 1
        executors = (executor_a, executor_b)
        _authenticate_executor_preparations(
            executors,
            ready=ready,
            authority_sha256=prepared.vault.authority.sha256,
        )
        events.append("BOTH_EXECUTORS_PREPARED_AFTER_RECALLS")

        _authenticate_spend_current(
            _CANONICAL_SPEND_PATH,
            expected=spend,
            expected_file_sha256=spend_file_sha256,
        )
        work["evaluation_executor_call_count"] += 1
        work["evaluation_release_attempt_count"] += 1
        execution_a = executor_a.execute_selected_once(recalls["A_prime"])
        execution_a.authenticate_structure()
        execution_record_a = _execution_record(execution_a)
        work["evaluation_authenticated_release_count"] += 1
        work["evaluation_authenticated_compile_invocation_count"] += (
            execution_a.compile_receipt.compile_invocations
        )
        work["evaluation_authenticated_run_invocation_count"] += (
            execution_a.run_receipt.run_invocations
        )
        observations["executions"] = (execution_record_a,)
        events.append("A_PRIME_EXECUTION_AUTHENTICATED")

        _authenticate_spend_current(
            _CANONICAL_SPEND_PATH,
            expected=spend,
            expected_file_sha256=spend_file_sha256,
        )
        work["evaluation_executor_call_count"] += 1
        work["evaluation_release_attempt_count"] += 1
        execution_b = executor_b.execute_selected_once(recalls["B_prime"])
        execution_b.authenticate_structure()
        execution_record_b = _execution_record(execution_b)
        work["evaluation_authenticated_release_count"] += 1
        work["evaluation_authenticated_compile_invocation_count"] += (
            execution_b.compile_receipt.compile_invocations
        )
        work["evaluation_authenticated_run_invocation_count"] += (
            execution_b.run_receipt.run_invocations
        )
        observations["executions"] = (execution_record_a, execution_record_b)
        events.append("B_PRIME_EXECUTION_AUTHENTICATED")

        executions = (execution_a, execution_b)
        selections = (recalls["A_prime"], recalls["B_prime"])
        _authenticate_oracle_gate(
            executions,
            selections,
            ready=ready,
            authority_sha256=prepared.vault.authority.sha256,
        )
        if (
            tuple(item.selected_source.source_sha256 for item in executions)
            != ready.actual_rust_source_sha256s
            or len({item.selected_source.release_sha256 for item in executions}) != 2
            or len({item.selected_source.sha256 for item in executions}) != 2
            or work["evaluation_authenticated_compile_invocation_count"] != 2
            or work["evaluation_authenticated_run_invocation_count"] != 2
            or prepared.memory.checkpoint_sha256 != checkpoint_root
            or prepared.memory.checkpoint_array_sha256s != checkpoint_arrays
        ):
            raise ContinualRustIteratorCohortError("authenticated execution phase changed")
        events.append("BOTH_EXECUTION_TREES_AUTHENTICATED")

        _authenticate_spend_current(
            _CANONICAL_SPEND_PATH,
            expected=spend,
            expected_file_sha256=spend_file_sha256,
        )
        work["oracle_open_attempt_count"] += 1
        raw_oracle = _CANONICAL_ORACLE_PATH.read_bytes()
        work["oracle_read_count"] += 1
        oracle_rows = _parse_oracle_bytes(
            raw_oracle,
            expected_sha256=ready.oracle_sha256,
            expected_names=FRESH_NAMES,
        )
        observed_outputs = (execution_a.outputs, execution_b.outputs)
        expected_outputs = tuple(row[1] for row in oracle_rows)
        passed = observed_outputs == expected_outputs
        observations["oracle"] = {
            "filename": _CANONICAL_ORACLE_PATH.name,
            "sha256": ready.oracle_sha256,
            "read_count": 1,
            "raw_utf8": raw_oracle.decode("utf-8", errors="strict"),
            "rows": oracle_rows,
        }
        events.append("ORACLE_OPENED_ONCE_AFTER_BOTH_EXECUTIONS")
        observations["adjudication"] = {
            "observed_outputs": observed_outputs,
            "expected_outputs": expected_outputs,
            "matched": passed,
            "memory_update_from_evaluation_count": 0,
        }
        events.append("ADJUDICATED_WITHOUT_MEMORY_UPDATE")
        status = "PASS" if passed else "STOP_ORACLE_MISMATCH"
        result = _terminal_record(
            status=status,
            protocol_commit=protocol_commit,
            ready=ready,
            spend_sha256=spend["sha256"],
            spend_file_sha256=spend_file_sha256,
            events=events,
            work=work,
            observations=observations,
            error=None,
        )
        _authenticate_spend_current(
            _CANONICAL_SPEND_PATH,
            expected=spend,
            expected_file_sha256=spend_file_sha256,
        )
        _authenticate_terminal_result_structure_v1(result)
        terminal = result
    except Exception as exc:
        if result_reservation is None or spend_state != "EXACT":
            raise
        if spend_guard is None:
            try:
                spend_guard = _acquire_spend_marker_guard(
                    _CANONICAL_SPEND_PATH,
                    expected=spend,
                    expected_file_sha256=spend_file_sha256,
                )
            except Exception:
                spend_state = "AMBIGUOUS"
                raise
        marker_error: Exception | None = None
        try:
            _authenticate_spend_current(
                _CANONICAL_SPEND_PATH,
                expected=spend,
                expected_file_sha256=spend_file_sha256,
            )
        except Exception as current_exc:
            marker_error = current_exc
        message = str(exc)
        if marker_error is not None:
            message = f"{message}; spend marker: {marker_error}"
        failure_observations = {name: observations[name] for name in ("ready", "training", "vault")}
        if "RECALL_PHASE_AUTHENTICATED" in events:
            failure_observations["recalls"] = observations["recalls"]
        if "A_PRIME_EXECUTION_AUTHENTICATED" in events:
            failure_observations["executions"] = observations["executions"]
        if "ORACLE_OPENED_ONCE_AFTER_BOTH_EXECUTIONS" in events:
            failure_observations["oracle"] = observations["oracle"]
        if "ADJUDICATED_WITHOUT_MEMORY_UPDATE" in events:
            failure_observations["adjudication"] = observations["adjudication"]
        failure = _terminal_record(
            status="STOP_EXECUTION_PROTOCOL_FAILURE",
            protocol_commit=protocol_commit,
            ready=ready,
            spend_sha256=spend["sha256"],
            spend_file_sha256=spend_file_sha256,
            events=events,
            work=work,
            observations=failure_observations,
            error={"type": type(exc).__name__, "message": message},
        )
        _authenticate_terminal_result_structure_v1(failure)
        terminal = failure
    finally:
        try:
            if result_reservation is not None and terminal is not None:
                _write_reserved_result_json(result_reservation, terminal)
        finally:
            try:
                if result_reservation is not None:
                    if spend_state == "ABSENT":
                        _discard_empty_result_reservation(
                            result_reservation,
                            spend_path=_CANONICAL_SPEND_PATH,
                        )
                    else:
                        _release_result_reservation(result_reservation)
            finally:
                if spend_guard is not None:
                    _release_spend_marker_guard(spend_guard)
    if terminal is None:
        raise ContinualRustIteratorCohortError("terminal result was not constructed")
    return terminal


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-ready")
    freeze.add_argument("--output", type=Path, default=_CANONICAL_READY_PATH)
    run = subparsers.add_parser("run-one-shot")
    run.add_argument("--protocol-commit", required=True)
    subparsers.add_parser("verify-result")
    arguments = parser.parse_args()
    if arguments.command == "freeze-ready":
        ready = freeze_ready_unspent(arguments.output)
        print(_canonical_json(ready.record()).decode("utf-8"))
        return 0
    if arguments.command == "run-one-shot":
        result = run_one_shot(protocol_commit=arguments.protocol_commit)
        print(_canonical_json(result).decode("utf-8"))
        return int(result["status"] != "PASS")
    result = load_terminal_result_v1()
    print(_canonical_json(result).decode("utf-8"))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CLAIM_BOUNDARY",
    "ContinualRustIteratorCohortError",
    "ReadyUnspentV1",
    "freeze_ready_unspent",
    "load_ready_unspent_v1",
    "load_terminal_result_v1",
    "make_ready_unspent_v1",
    "run_one_shot",
]
