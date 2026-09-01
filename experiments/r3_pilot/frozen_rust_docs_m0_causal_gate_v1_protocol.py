"""Frozen Rust-documentation M0 causal-gate protocol.

The protocol freezes a lexical documentation-conditioned dispatch experiment.
It does not synthesize Rust, inspect an oracle while selecting, or claim semantic
coding or transfer.  Importing this module does not build the Rust Book corpus,
run rustfmt or rustc, recall, materialize a source, execute a binary, or open the
oracle artifact.

The public freeze operation is intentionally separate from the one-shot runner:
it prepares and authenticates task-independent registries, memories, tasks, and
an executor receipt with all evaluation counters at zero.  The runner later
spends that READY_UNSPENT receipt before the first recall.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from humanbrain import consolidated_knowledge_cell_projection as consolidated_cells
from humanbrain import knowledge_consolidation, rust_action_registry, rust_selected_action_executor
from tools import build_rust_book_knowledge_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_PATH = Path(__file__).with_name("frozen_rust_docs_m0_causal_gate_v1.batch.json")
DEFAULT_ORACLE_PATH = Path(__file__).with_name("frozen_rust_docs_m0_causal_gate_v1.oracle.json")
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("frozen_rust_docs_m0_causal_gate_v1.manifest.json")
DEFAULT_SPENT_PATH = Path(__file__).with_name("frozen_rust_docs_m0_causal_gate_v1.spent.json")
DEFAULT_RESULT_PATH = Path(__file__).with_name("frozen_rust_docs_m0_causal_gate_v1.result.json")
DEFAULT_OWNER_LOCK_PATH = Path(__file__).with_name("frozen_rust_docs_m0_causal_gate_v1.owner.lock")

MANIFEST_SCHEMA = "humanbrain.rust-docs-m0-causal-gate-ready.v1"
BATCH_SCHEMA = "humanbrain.rust-docs-m0-causal-gate-batch.v1"
ORACLE_SCHEMA = "humanbrain.rust-docs-m0-causal-gate-oracle.v1"
CONTROL_SCHEMA = "humanbrain.rust-docs-m0-causal-gate-controls.v1"
EXECUTION_SCHEMA = "humanbrain.rust-docs-m0-causal-gate-execution.v1"
ADJUDICATION_SCHEMA = "humanbrain.rust-docs-m0-causal-gate-adjudication.v1"

READY_UNSPENT = "READY_UNSPENT"
TERMINAL_PASS = "TERMINAL_PASS"
TERMINAL_STOP = "TERMINAL_STOP"

CLAIM_BOUNDARY = (
    "One fresh M0 test of lexical Rust-Book-documentation-conditioned dispatch "
    "between two pre-existing authenticated Rust 2021 actions, followed by "
    "exactly one selected compile and batch run. It is not source synthesis, "
    "semantic code understanding, debugging, within-family transfer, "
    "cross-domain transfer, or broad coding-intelligence evidence."
)

FROZEN_PRIMARY_TASK = (
    "Implement solve so a worker thread sends each input integer through a "
    "channel. Return all delivered integers in ascending order, preserving "
    "duplicates."
)
FROZEN_CONTRAST_TASK = (
    "Implement solve so a worker thread updates a shared total protected by a "
    "mutex lock. Return the total as a one-element vector."
)
FROZEN_UNKNOWN_NONCE = "zzzxqvjmptk"

FROZEN_CHANNEL_SOURCE = """use std::sync::mpsc;
use std::thread;

pub fn solve(values: &[i64]) -> Vec<i64> {
    let (tx, rx) = mpsc::channel();
    let owned = values.to_vec();
    let worker = thread::spawn(move || {
        for value in owned {
            tx.send(value).unwrap();
        }
    });
    let mut output: Vec<i64> = rx.into_iter().collect();
    worker.join().unwrap();
    output.sort();
    output
}
"""
FROZEN_MUTEX_SOURCE = """use std::sync::{Arc, Mutex};
use std::thread;

pub fn solve(values: &[i64]) -> Vec<i64> {
    let shared = Arc::new(Mutex::new(0_i64));
    let update = Arc::clone(&shared);
    let owned = values.to_vec();
    let worker = thread::spawn(move || {
        let mut total = update.lock().unwrap();
        for value in owned {
            *total += value;
        }
    });
    worker.join().unwrap();
    let output = *shared.lock().unwrap();
    vec![output]
}
"""
FROZEN_CANDIDATE_SOURCES = (FROZEN_CHANNEL_SOURCE, FROZEN_MUTEX_SOURCE)

FROZEN_CASES = (
    (17, -8, 17, 4, -3, 0),
    (-21, 13),
    (6, 6, -12, 5, -1, 5),
    (144, -89, 55, -34, 21, -13, 8),
)

TASK_SPECS = (
    ("primary", FROZEN_PRIMARY_TASK),
    ("contrast", FROZEN_CONTRAST_TASK),
    ("shared_factor_thread", "thread"),
    ("unknown_factor", FROZEN_UNKNOWN_NONCE),
    ("single_factor_channel", "channel"),
    ("single_factor_mutex", "mutex"),
)
_TASK_PROJECTION_NAMES = tuple(name for name, _text in TASK_SPECS)
_CONTROL_RECALL_TASK_NAMES = (
    "primary",
    "contrast",
    "primary",
    "primary",
    "primary",
    "shared_factor_thread",
    "unknown_factor",
    "single_factor_channel",
    "single_factor_mutex",
)
CONTROL_ROSTER = (
    "primary",
    "contrast",
    "source_order_reversal",
    "cell_off",
    "plasticity_off",
    "bound_memory_splice",
    "shared_factor_thread",
    "unknown_factor",
    "single_factor_channel",
    "single_factor_mutex",
)
MAX_ORACLE_VALUES_PER_CASE = 16_384
MAX_ORACLE_TOTAL_VALUES = 65_536

_SHA256 = re.compile(r"[0-9a-f]{64}")
_BATCH_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-batch-payload-v1"
_ORACLE_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-oracle-payload-v1"
_TASK_SET_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-task-set-v1"
_CANDIDATE_SET_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-source-set-v1"
_ROSTER_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-control-roster-v1"
_UNKNOWN_CUE_NAMESPACE = b"humanbrain-rust-docs-m0-unknown-cue-absence-v1"
_READY_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-ready-receipt-v1"
_MANIFEST_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-manifest-v1"
_CONTROL_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-controls-v1"
_EXECUTION_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-execution-v1"
_ADJUDICATION_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-adjudication-v1"


class FrozenRustDocsM0GateError(RuntimeError):
    """A frozen input, root, chronology rule, or receipt changed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_document(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _plain(value: object) -> Any:
    """Return the JSON-native representation used by on-disk records."""

    return json.loads(_canonical_json(value))


def _json_exact_equal(left: object, right: object) -> bool:
    """Compare JSON-native values without Python's bool/int numeric coercions."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if len(left) != len(right):
            return False
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            for index, (right_key, right_value) in enumerate(unmatched):
                if _json_exact_equal(left_key, right_key):
                    if not _json_exact_equal(left_value, right_value):
                        return False
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_exact_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) in (str, int, float, bool, type(None)):
        return left == right
    return False


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _is_exact_splice_item(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"name", "kind", "status", "physical_recall_count"}
        and value.get("name") == "bound_memory_splice"
        and value.get("kind") == "authority_control"
        and value.get("status") == "STOP_BOUND_MEMORY_SPLICE"
        and type(value.get("physical_recall_count")) is int
        and value.get("physical_recall_count") == 0
    )


def _source_sha256(source: str) -> str:
    return sha256(source.encode("ascii", errors="strict")).hexdigest()


BATCH_PAYLOAD_SHA256 = "ba9b1ed5bc532a5e792ee4379e9eaa296f7f268a7dd2d70f595770111502f608"
ORACLE_PAYLOAD_SHA256 = "7abb4f962ff886a36bd45225070fbb2f092edbf4ae32c6ca9701e573d9d109b0"
CANDIDATE_SOURCE_SET_SHA256 = _digest(
    _CANDIDATE_SET_NAMESPACE,
    tuple(sorted(_source_sha256(source) for source in FROZEN_CANDIDATE_SOURCES)),
)
CONTROL_ROSTER_SHA256 = _digest(_ROSTER_NAMESPACE, CONTROL_ROSTER)

_BATCH_DOCUMENT = {
    "abi": "pub fn solve(values: &[i64]) -> Vec<i64>",
    "cases": [list(case) for case in FROZEN_CASES],
    "edition": "2021",
    "entry_point": "solve",
    "schema": BATCH_SCHEMA,
}
EXPECTED_BATCH_FILE_SHA256 = "7ee9bbaba3e0d4bd95447cbeb1886eff2ad67ad9dd7e81e37703b2571fa2df54"
EXPECTED_ORACLE_FILE_SHA256 = "46539d4e2e800f564b2c5207ca0aafe8f3c5d420692272b55c2686598f075be7"

IMPLEMENTATION_PATHS = (
    "data/knowledge/rust-book-nostarch-third-printing.lock.json",
    "experiments/g0/common_event_cells_v1.py",
    "experiments/r3_pilot/common_knowledge_action_memory_v1.py",
    "experiments/r3_pilot/common_rooted_structured_event_graph_v1.py",
    "experiments/r3_pilot/frozen_rust_docs_m0_causal_gate_v1.batch.json",
    "experiments/r3_pilot/frozen_rust_docs_m0_causal_gate_v1_protocol.py",
    "experiments/r3_pilot/run_frozen_rust_docs_m0_causal_gate_v1.py",
    "src/humanbrain/consolidated_knowledge_cell_projection.py",
    "src/humanbrain/knowledge_consolidation.py",
    "src/humanbrain/knowledge_corpus.py",
    "src/humanbrain/rust_action_registry.py",
    "src/humanbrain/rust_selected_action_executor.py",
    "tools/build_rust_book_knowledge_corpus.py",
)


@dataclass(frozen=True, slots=True)
class FrozenBatchV1:
    cases: tuple[tuple[int, ...], ...]
    payload_sha256: str
    file_sha256: str

    def authenticate(self) -> None:
        if (
            self.cases != FROZEN_CASES
            or self.payload_sha256 != BATCH_PAYLOAD_SHA256
            or self.file_sha256 != EXPECTED_BATCH_FILE_SHA256
        ):
            raise FrozenRustDocsM0GateError("frozen batch changed")

    def record(self) -> dict[str, object]:
        self.authenticate()
        return {
            "path": DEFAULT_BATCH_PATH.relative_to(REPO_ROOT).as_posix(),
            "file_sha256": self.file_sha256,
            "payload_sha256": self.payload_sha256,
            "case_count": len(self.cases),
            "cases": [list(case) for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class FrozenWorkCountersV1:
    recall_count: int = 0
    selected_source_materialization_count: int = 0
    selected_executor_call_count: int = 0
    authenticated_rustc_compile_receipt_count: int = 0
    authenticated_binary_run_receipt_count: int = 0
    oracle_read_count: int = 0

    def authenticate(self) -> None:
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise FrozenRustDocsM0GateError("work counters changed")

    def authenticate_ready_unspent(self) -> None:
        self.authenticate()
        if any(asdict(self).values()):
            raise FrozenRustDocsM0GateError("READY_UNSPENT must have zero evaluation work")


ZERO_EVALUATION_WORK = FrozenWorkCountersV1()


@dataclass(frozen=True, slots=True)
class ReadyUnspentReceiptV1:
    schema: str
    status: str
    claim_boundary: str
    source_summary_sha256: str
    input_records_sha256: str
    consolidation_corpus_sha256: str
    consolidation_producer_sha256: str
    knowledge_projection_sha256: str
    knowledge_provenance_sha256: str
    registry_sha256: str
    reversed_registry_sha256: str
    selector_view_sha256: str
    candidate_set_sha256: str
    candidate_source_set_sha256: str
    bound_memory_receipt_sha256: str
    bound_memory_checkpoint_sha256: str
    reversed_bound_memory_receipt_sha256: str
    no_plasticity_memory_receipt_sha256: str
    no_plasticity_checkpoint_sha256: str
    task_set_sha256: str
    task_projection_sha256s: Mapping[str, str]
    control_roster_sha256: str
    unknown_cue_absence_sha256: str
    channel_action_handle: str
    channel_structural_sha256: str
    mutex_action_handle: str
    batch_payload_sha256: str
    oracle_payload_sha256: str
    executor_preparation_sha256: str
    rustc_identity_sha256: str
    executor_harness_sha256: str
    executor_batch_protocol_sha256: str
    executor_batch_sha256: str
    executor_stdin_sha256: str
    work: FrozenWorkCountersV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = _plain(asdict(self))
        if not include_sha256:
            result.pop("sha256")
        return result

    def authenticate(self) -> None:
        task_projection_roots = (
            tuple(self.task_projection_sha256s.values())
            if type(self.task_projection_sha256s) is dict
            else ()
        )
        scalar_roots = (
            self.source_summary_sha256,
            self.input_records_sha256,
            self.consolidation_corpus_sha256,
            self.consolidation_producer_sha256,
            self.knowledge_projection_sha256,
            self.knowledge_provenance_sha256,
            self.registry_sha256,
            self.reversed_registry_sha256,
            self.selector_view_sha256,
            self.candidate_set_sha256,
            self.candidate_source_set_sha256,
            self.bound_memory_receipt_sha256,
            self.bound_memory_checkpoint_sha256,
            self.reversed_bound_memory_receipt_sha256,
            self.no_plasticity_memory_receipt_sha256,
            self.no_plasticity_checkpoint_sha256,
            self.task_set_sha256,
            *task_projection_roots,
            self.control_roster_sha256,
            self.unknown_cue_absence_sha256,
            self.channel_structural_sha256,
            self.batch_payload_sha256,
            self.oracle_payload_sha256,
            self.executor_preparation_sha256,
            self.rustc_identity_sha256,
            self.executor_harness_sha256,
            self.executor_batch_protocol_sha256,
            self.executor_batch_sha256,
            self.executor_stdin_sha256,
        )
        if (
            self.schema != MANIFEST_SCHEMA
            or self.status != READY_UNSPENT
            or self.claim_boundary != CLAIM_BOUNDARY
            or any(not _valid_sha256(value) for value in scalar_roots)
            or self.registry_sha256 != self.reversed_registry_sha256
            or type(self.task_projection_sha256s) is not dict
            or set(self.task_projection_sha256s) != set(_TASK_PROJECTION_NAMES)
            or len(task_projection_roots) != len(_TASK_PROJECTION_NAMES)
            or self.candidate_source_set_sha256 != CANDIDATE_SOURCE_SET_SHA256
            or self.control_roster_sha256 != CONTROL_ROSTER_SHA256
            or self.batch_payload_sha256 != BATCH_PAYLOAD_SHA256
            or self.oracle_payload_sha256 != ORACLE_PAYLOAD_SHA256
            or self.executor_harness_sha256 != rust_selected_action_executor.HARNESS_SHA256
            or self.executor_batch_protocol_sha256
            != rust_selected_action_executor.BATCH_PROTOCOL_SHA256
            or type(self.channel_action_handle) is not str
            or type(self.mutex_action_handle) is not str
            or self.channel_action_handle == self.mutex_action_handle
            or type(self.work) is not FrozenWorkCountersV1
        ):
            raise FrozenRustDocsM0GateError("READY_UNSPENT receipt changed")
        self.work.authenticate_ready_unspent()
        if self.sha256 != _digest(_READY_NAMESPACE, self.record(include_sha256=False)):
            raise FrozenRustDocsM0GateError("READY_UNSPENT root changed")


@dataclass(frozen=True, slots=True)
class _LivePreparedInputsV1:
    source_summary: Mapping[str, object]
    corpus: Any
    consolidation: knowledge_consolidation.KnowledgeConsolidationV1
    projection: consolidated_cells.ConsolidatedKnowledgeCellProjectionV1
    registry: rust_action_registry.PreparedRustActionRegistryV1
    reversed_registry: rust_action_registry.PreparedRustActionRegistryV1
    memory: rust_action_registry.BoundRustActionMemoryV1
    reversed_memory: rust_action_registry.BoundRustActionMemoryV1
    no_plasticity_memory: rust_action_registry.BoundRustActionMemoryV1
    tasks: Mapping[str, knowledge_action.KnowledgeActionTaskProjectionV1]
    executor: rust_selected_action_executor.PreparedRustSelectedActionExecutorV1


@dataclass(frozen=True, slots=True)
class _PreparedRustDocsM0GateV1:
    """Authenticated live READY_UNSPENT state; no recall or evaluation has run."""

    manifest: Mapping[str, object]
    batch: FrozenBatchV1
    _live: _LivePreparedInputsV1
    ready_receipt: ReadyUnspentReceiptV1

    @property
    def manifest_sha256(self) -> str:
        value = self.manifest["manifest_sha256"]
        if not _valid_sha256(value):
            raise FrozenRustDocsM0GateError("manifest root changed")
        return value

    def authenticate_ready_unspent(self) -> None:
        self.batch.authenticate()
        self.ready_receipt.authenticate()
        _authenticate_live(self._live)
        expected = _make_ready_receipt(self._live, self.batch)
        if expected != self.ready_receipt:
            raise FrozenRustDocsM0GateError("prepared READY_UNSPENT state changed")
        if self.manifest.get("readiness") != self.ready_receipt.record():
            raise FrozenRustDocsM0GateError("manifest readiness changed")


@dataclass(frozen=True, slots=True)
class FrozenControlBundleV1:
    recalls: tuple[knowledge_action.KnowledgeActionRecallV1, ...]
    items: tuple[Mapping[str, object], ...]
    ready_sha256: str
    work: FrozenWorkCountersV1
    sha256: str

    @property
    def primary_recall(self) -> knowledge_action.KnowledgeActionRecallV1:
        return self.recalls[0]

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload = {
            "schema": CONTROL_SCHEMA,
            "ready_sha256": self.ready_sha256,
            "roster_sha256": CONTROL_ROSTER_SHA256,
            "items": self.items,
            "control_count": len(self.items),
            "physical_recall_count": len(self.recalls),
            "work": asdict(self.work),
        }
        result = _plain(payload)
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self, prepared: _PreparedRustDocsM0GateV1) -> None:
        prepared.authenticate_ready_unspent()
        if type(self.work) is not FrozenWorkCountersV1:
            raise FrozenRustDocsM0GateError("control bundle work changed")
        self.work.authenticate()
        if (
            type(self.recalls) is not tuple
            or type(self.items) is not tuple
            or len(self.recalls) != 9
            or len(self.items) != len(CONTROL_ROSTER)
            or any(type(item) is not dict for item in self.items)
            or tuple(item.get("name") for item in self.items) != CONTROL_ROSTER
            or self.work != FrozenWorkCountersV1(recall_count=9)
            or self.ready_sha256 != prepared.ready_receipt.sha256
        ):
            raise FrozenRustDocsM0GateError("control bundle shape changed")
        recall_item_indexes = (0, 1, 2, 3, 4, 6, 7, 8, 9)
        for item_index, recall in zip(recall_item_indexes, self.recalls, strict=True):
            recall.authenticate()
            if _canonical_json(self.items[item_index]) != _canonical_json(
                _recall_item(CONTROL_ROSTER[item_index], recall)
            ):
                raise FrozenRustDocsM0GateError("serialized control recall changed")
        _authenticate_expected_control_outcomes(prepared, self.recalls, self.items)
        if self.sha256 != _digest(_CONTROL_NAMESPACE, self.record(include_sha256=False)):
            raise FrozenRustDocsM0GateError("control bundle root changed")


@dataclass(frozen=True, slots=True)
class AuthenticatedSelectedExecutionV1:
    execution: rust_selected_action_executor.RustSelectedActionExecutionV1
    control_bundle_sha256: str
    work: FrozenWorkCountersV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        payload = {
            "schema": EXECUTION_SCHEMA,
            "control_bundle_sha256": self.control_bundle_sha256,
            "execution": asdict(self.execution),
            "work": asdict(self.work),
        }
        result = _plain(payload)
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(
        self,
        prepared: _PreparedRustDocsM0GateV1,
        controls: FrozenControlBundleV1,
    ) -> None:
        controls.authenticate(prepared)
        self.execution.authenticate_structure()
        if type(self.work) is not FrozenWorkCountersV1:
            raise FrozenRustDocsM0GateError("selected execution work changed")
        self.work.authenticate()
        current_identity = prepared._live.executor.rustc_identity
        expected_work = FrozenWorkCountersV1(
            recall_count=9,
            selected_source_materialization_count=1,
            selected_executor_call_count=1,
            authenticated_rustc_compile_receipt_count=1,
            authenticated_binary_run_receipt_count=1,
        )
        if (
            self.work != expected_work
            or self.control_bundle_sha256 != controls.sha256
            or self.execution.preparation != prepared._live.executor.preparation
            or self.execution.rustc_identity != current_identity
            or self.execution.materialization.registry_sha256
            != prepared.ready_receipt.registry_sha256
            or self.execution.materialization.recall_sha256 != controls.primary_recall.sha256
            or self.execution.materialization.action_handle
            != prepared.ready_receipt.channel_action_handle
            or self.execution.materialization.source != FROZEN_CHANNEL_SOURCE
            or self.execution.materialization.source_sha256 != _source_sha256(FROZEN_CHANNEL_SOURCE)
            or self.execution.materialization.structural_sha256
            != prepared.ready_receipt.channel_structural_sha256
            or self.execution.compile_receipt.compile_invocations != 1
            or self.execution.run_receipt.run_invocations != 1
            or self.execution.run_receipt.case_count != len(FROZEN_CASES)
            or self.sha256 != _digest(_EXECUTION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FrozenRustDocsM0GateError("selected execution receipt authentication failed")


def _load_json_document(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenRustDocsM0GateError(f"invalid JSON artifact: {path}") from exc
    if type(value) is not dict or raw != _canonical_document(value):
        raise FrozenRustDocsM0GateError(f"noncanonical JSON artifact: {path}")
    return value, raw


def load_frozen_batch() -> FrozenBatchV1:
    """Open only the public execution batch; never touch oracle bytes."""

    value, raw = _load_json_document(DEFAULT_BATCH_PATH)
    if value != _BATCH_DOCUMENT:
        raise FrozenRustDocsM0GateError("frozen execution batch changed")
    batch = FrozenBatchV1(
        cases=FROZEN_CASES,
        payload_sha256=BATCH_PAYLOAD_SHA256,
        file_sha256=sha256(raw).hexdigest(),
    )
    batch.authenticate()
    return batch


def _project_tasks() -> dict[str, knowledge_action.KnowledgeActionTaskProjectionV1]:
    return {
        name: knowledge_action.project_knowledge_action_task_v1(text) for name, text in TASK_SPECS
    }


def _task_set_record(
    tasks: Mapping[str, knowledge_action.KnowledgeActionTaskProjectionV1],
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for name, text in TASK_SPECS:
        task = tasks[name]
        task.authenticate()
        records.append(
            {
                "name": name,
                "text": text,
                "factors": task.factors,
                "projection_sha256": task.sha256,
            }
        )
    return tuple(records)


def _task_set_sha256(
    tasks: Mapping[str, knowledge_action.KnowledgeActionTaskProjectionV1],
) -> str:
    return _digest(_TASK_SET_NAMESPACE, _task_set_record(tasks))


def _action_handle(view: rust_action_registry.RustActionSelectorViewV1, factor: str) -> str:
    handles = {
        binding.action_handle for binding in view.factor_bindings if binding.factor == factor
    }
    if len(handles) != 1:
        raise FrozenRustDocsM0GateError(f"expected one authenticated Rust action for {factor}")
    return next(iter(handles))


def _channel_structural_sha256(live: _LivePreparedInputsV1) -> str:
    source_sha256 = _source_sha256(FROZEN_CHANNEL_SOURCE)
    projections = tuple(
        entry.projection
        for entry in live.registry._entries
        if entry.source == FROZEN_CHANNEL_SOURCE and entry.projection.source_sha256 == source_sha256
    )
    if len(projections) != 1:
        raise FrozenRustDocsM0GateError("exact frozen channel projection changed")
    return projections[0].structural_sha256


def _unknown_cue_absence_record(
    live: _LivePreparedInputsV1,
) -> dict[str, object]:
    factor = f"tok:{FROZEN_UNKNOWN_NONCE}"
    selector_occurrences = sum(
        binding.factor == factor for binding in live.registry.selector_view.factor_bindings
    )
    association = live.memory._association_projection
    bound_factor_occurrences = association.factor_vocabulary.count(factor)
    if selector_occurrences != 0 or bound_factor_occurrences != 0:
        raise FrozenRustDocsM0GateError(
            "unknown control cue is present in frozen selector or bound memory"
        )
    payload = {
        "factor": factor,
        "selector_view_sha256": live.registry.selector_view.sha256,
        "association_projection_sha256": association.sha256,
        "selector_occurrences": selector_occurrences,
        "bound_factor_occurrences": bound_factor_occurrences,
    }
    return {**payload, "sha256": _digest(_UNKNOWN_CUE_NAMESPACE, payload)}


def _authenticate_unknown_cue_absence_record(value: object) -> None:
    if type(value) is not dict:
        raise FrozenRustDocsM0GateError("unknown-cue proof shape changed")
    payload = dict(value)
    observed = payload.pop("sha256", None)
    if (
        set(value)
        != {
            "factor",
            "selector_view_sha256",
            "association_projection_sha256",
            "selector_occurrences",
            "bound_factor_occurrences",
            "sha256",
        }
        or payload.get("factor") != f"tok:{FROZEN_UNKNOWN_NONCE}"
        or type(payload.get("selector_occurrences")) is not int
        or payload.get("selector_occurrences") != 0
        or type(payload.get("bound_factor_occurrences")) is not int
        or payload.get("bound_factor_occurrences") != 0
        or not _valid_sha256(payload.get("selector_view_sha256"))
        or not _valid_sha256(payload.get("association_projection_sha256"))
        or observed != _digest(_UNKNOWN_CUE_NAMESPACE, payload)
    ):
        raise FrozenRustDocsM0GateError("unknown-cue absence proof changed")


def _build_live(batch: FrozenBatchV1) -> _LivePreparedInputsV1:
    """Build and fit only pre-spend state; never recall or materialize."""

    batch.authenticate()
    corpus, source_summary = build_rust_book_knowledge_corpus.build_from_cache()
    consolidation = knowledge_consolidation.consolidate_knowledge_corpus_v1(
        corpus,
        split="acquire",
    )
    projection = consolidated_cells.project_consolidated_knowledge_to_cells_v1(
        consolidation,
        corpus=corpus,
    )
    registry = rust_action_registry.prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=FROZEN_CANDIDATE_SOURCES,
    )
    reversed_registry = rust_action_registry.prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=tuple(reversed(FROZEN_CANDIDATE_SOURCES)),
    )
    memory = registry.fit_memory(projection)
    reversed_memory = reversed_registry.fit_memory(projection)
    no_plasticity_memory = registry.fit_memory(projection, plasticity_gate=False)
    tasks = _project_tasks()
    executor = rust_selected_action_executor.prepare_rust_selected_action_executor_v1(
        registry,
        batch.cases,
    )
    live = _LivePreparedInputsV1(
        source_summary=dict(source_summary),
        corpus=corpus,
        consolidation=consolidation,
        projection=projection,
        registry=registry,
        reversed_registry=reversed_registry,
        memory=memory,
        reversed_memory=reversed_memory,
        no_plasticity_memory=no_plasticity_memory,
        tasks=tasks,
        executor=executor,
    )
    _authenticate_live(live)
    return live


def _authenticate_live(live: _LivePreparedInputsV1) -> None:
    live.consolidation.authenticate(live.corpus)
    live.projection.authenticate(live.consolidation, corpus=live.corpus)
    registry_receipt = live.registry.receipt
    reversed_receipt = live.reversed_registry.receipt
    selector = live.registry.selector_view
    reversed_selector = live.reversed_registry.selector_view
    live.memory.authenticate_structure()
    live.reversed_memory.authenticate_structure()
    live.no_plasticity_memory.authenticate_structure()
    for task in live.tasks.values():
        task.authenticate()
    preparation = live.executor.preparation
    identity = live.executor.rustc_identity
    if (
        registry_receipt != reversed_receipt
        or selector != reversed_selector
        or live.memory.receipt != live.reversed_memory.receipt
        or live.memory.training_summary.plasticity_gate is not True
        or live.no_plasticity_memory.training_summary.plasticity_gate is not False
        or preparation.expected_registry_sha256 != registry_receipt.sha256
        or preparation.rustc_identity_sha256 != identity.sha256
        or preparation.case_count != len(FROZEN_CASES)
        or preparation.harness_sha256 != rust_selected_action_executor.HARNESS_SHA256
        or preparation.batch_protocol_sha256 != rust_selected_action_executor.BATCH_PROTOCOL_SHA256
        or rust_selected_action_executor.RUST_EDITION != "2021"
    ):
        raise FrozenRustDocsM0GateError("prepared live roots changed")
    channel = _action_handle(selector, "tok:channel")
    mutex = _action_handle(selector, "tok:mutex")
    if channel == mutex:
        raise FrozenRustDocsM0GateError("channel and mutex actions are not distinct")
    _authenticate_unknown_cue_absence_record(_unknown_cue_absence_record(live))


def _make_ready_receipt(
    live: _LivePreparedInputsV1,
    batch: FrozenBatchV1,
) -> ReadyUnspentReceiptV1:
    _authenticate_live(live)
    selector = live.registry.selector_view
    preparation = live.executor.preparation
    payload = {
        "schema": MANIFEST_SCHEMA,
        "status": READY_UNSPENT,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_summary_sha256": _digest(b"rust-book-source-summary-v1", live.source_summary),
        "input_records_sha256": live.consolidation.input_records_sha256,
        "consolidation_corpus_sha256": live.consolidation.corpus_sha256,
        "consolidation_producer_sha256": live.consolidation.producer_sha256,
        "knowledge_projection_sha256": live.projection.sha256,
        "knowledge_provenance_sha256": live.projection.provenance_sha256,
        "registry_sha256": live.registry.receipt.sha256,
        "reversed_registry_sha256": live.reversed_registry.receipt.sha256,
        "selector_view_sha256": selector.sha256,
        "candidate_set_sha256": selector.candidate_set_sha256,
        "candidate_source_set_sha256": CANDIDATE_SOURCE_SET_SHA256,
        "bound_memory_receipt_sha256": live.memory.receipt.sha256,
        "bound_memory_checkpoint_sha256": live.memory.checkpoint_sha256,
        "reversed_bound_memory_receipt_sha256": live.reversed_memory.receipt.sha256,
        "no_plasticity_memory_receipt_sha256": live.no_plasticity_memory.receipt.sha256,
        "no_plasticity_checkpoint_sha256": live.no_plasticity_memory.checkpoint_sha256,
        "task_set_sha256": _task_set_sha256(live.tasks),
        "task_projection_sha256s": {
            name: live.tasks[name].sha256 for name in _TASK_PROJECTION_NAMES
        },
        "control_roster_sha256": CONTROL_ROSTER_SHA256,
        "unknown_cue_absence_sha256": _unknown_cue_absence_record(live)["sha256"],
        "channel_action_handle": _action_handle(selector, "tok:channel"),
        "channel_structural_sha256": _channel_structural_sha256(live),
        "mutex_action_handle": _action_handle(selector, "tok:mutex"),
        "batch_payload_sha256": batch.payload_sha256,
        "oracle_payload_sha256": ORACLE_PAYLOAD_SHA256,
        "executor_preparation_sha256": preparation.sha256,
        "rustc_identity_sha256": live.executor.rustc_identity.sha256,
        "executor_harness_sha256": preparation.harness_sha256,
        "executor_batch_protocol_sha256": preparation.batch_protocol_sha256,
        "executor_batch_sha256": preparation.batch_sha256,
        "executor_stdin_sha256": preparation.stdin_sha256,
        "work": ZERO_EVALUATION_WORK,
    }
    record = ReadyUnspentReceiptV1(
        **payload,
        sha256=_digest(
            _READY_NAMESPACE,
            {
                **payload,
                "work": asdict(ZERO_EVALUATION_WORK),
            },
        ),
    )
    record.authenticate()
    return record


def _git_output(arguments: tuple[str, ...]) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FrozenRustDocsM0GateError("git commitment authentication failed") from exc
    if completed.returncode != 0:
        raise FrozenRustDocsM0GateError("git commitment authentication failed")
    return completed.stdout


def _git_head() -> str:
    value = _git_output(("rev-parse", "HEAD")).decode("ascii", errors="strict").strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise FrozenRustDocsM0GateError("git HEAD is not one exact commit")
    return value


def _git_blob(commit: str, relative_path: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise FrozenRustDocsM0GateError("implementation commit is invalid")
    return _git_output(("show", f"{commit}:{relative_path}"))


def _implementation_records(commit: str) -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    for relative in IMPLEMENTATION_PATHS:
        path = REPO_ROOT / relative
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise FrozenRustDocsM0GateError(f"implementation input is missing: {relative}") from exc
        if raw != _git_blob(commit, relative):
            raise FrozenRustDocsM0GateError(f"implementation input is not committed: {relative}")
        records.append({"path": relative, "sha256": sha256(raw).hexdigest()})
    return tuple(records)


def _oracle_is_tracked() -> None:
    relative = DEFAULT_ORACLE_PATH.relative_to(REPO_ROOT).as_posix()
    _git_output(("ls-files", "--error-unmatch", relative))
    if not DEFAULT_ORACLE_PATH.is_file():
        raise FrozenRustDocsM0GateError("frozen oracle artifact is missing")


def _frozen_inputs_record(
    tasks: Mapping[str, knowledge_action.KnowledgeActionTaskProjectionV1],
) -> dict[str, object]:
    return {
        "task_records": _plain(_task_set_record(tasks)),
        "task_set_sha256": _task_set_sha256(tasks),
        "candidate_sources": [
            {
                "source_sha256": _source_sha256(source),
                "source_bytes": len(source.encode("ascii")),
            }
            for source in FROZEN_CANDIDATE_SOURCES
        ],
        "candidate_source_set_sha256": CANDIDATE_SOURCE_SET_SHA256,
        "control_roster": list(CONTROL_ROSTER),
        "control_roster_sha256": CONTROL_ROSTER_SHA256,
    }


def _manifest_payload(
    *,
    implementation_commit: str,
    implementation_files: tuple[dict[str, str], ...],
    batch: FrozenBatchV1,
    live: _LivePreparedInputsV1,
    ready: ReadyUnspentReceiptV1,
) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "status": READY_UNSPENT,
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_commit": implementation_commit,
        "implementation_files": implementation_files,
        "batch": batch.record(),
        "oracle_commitment": {
            "path": DEFAULT_ORACLE_PATH.relative_to(REPO_ROOT).as_posix(),
            "file_sha256": EXPECTED_ORACLE_FILE_SHA256,
            "payload_sha256": ORACLE_PAYLOAD_SHA256,
            "batch_payload_sha256": BATCH_PAYLOAD_SHA256,
            "runtime_read_count_at_freeze": 0,
        },
        "frozen_inputs": _frozen_inputs_record(live.tasks),
        "unknown_cue_absence": _unknown_cue_absence_record(live),
        "readiness": ready.record(),
    }


def _publish_pending_exclusive(pending: Path, path: Path) -> None:
    """Atomically publish one complete same-directory file without replacement."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move_file.restype = wintypes.BOOL
        if not move_file(str(pending), str(path), 0x00000008):  # MOVEFILE_WRITE_THROUGH
            error = ctypes.get_last_error()
            if error in (80, 183):  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
                raise FrozenRustDocsM0GateError(f"refusing to overwrite {path}")
            raise OSError(error, ctypes.FormatError(error), str(path))
        return
    try:
        os.link(pending, path)
    except FileExistsError as exc:
        raise FrozenRustDocsM0GateError(f"refusing to overwrite {path}") from exc


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    """Publish complete bytes atomically; the final path is never a write target."""

    if not path.parent.is_dir():
        raise FrozenRustDocsM0GateError(f"output directory is missing: {path.parent}")
    raw = _canonical_document(value)
    descriptor, pending_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pending",
    )
    pending = Path(pending_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if pending.read_bytes() != raw:
            raise FrozenRustDocsM0GateError("pending manifest bytes changed after fsync")
        _publish_pending_exclusive(pending, path)
        if path.read_bytes() != raw:
            raise FrozenRustDocsM0GateError("published manifest bytes changed")
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            pending.unlink()


def freeze_ready_unspent() -> dict[str, object]:
    """Create the canonical READY_UNSPENT manifest after sources are committed.

    This operation is intentionally expensive but pre-evaluative.  It reads the
    batch, builds/fits the frozen pair, and prepares a toolchain receipt.  It
    performs zero recalls, source materializations, selected-source rustc
    compiles, binary runs, or oracle reads.
    """

    for path in (DEFAULT_MANIFEST_PATH, DEFAULT_SPENT_PATH, DEFAULT_RESULT_PATH):
        if path.exists():
            raise FrozenRustDocsM0GateError(f"freeze path already exists: {path}")
    _oracle_is_tracked()  # Existence/tracking only; oracle bytes stay closed.
    implementation_commit = _git_head()
    implementation_files = _implementation_records(implementation_commit)
    batch = load_frozen_batch()
    live = _build_live(batch)
    ready = _make_ready_receipt(live, batch)
    payload = _plain(
        _manifest_payload(
            implementation_commit=implementation_commit,
            implementation_files=implementation_files,
            batch=batch,
            live=live,
            ready=ready,
        )
    )
    manifest = {
        **payload,
        "manifest_sha256": _digest(_MANIFEST_NAMESPACE, payload),
    }
    _write_json_exclusive(DEFAULT_MANIFEST_PATH, manifest)
    return manifest


def _load_manifest() -> tuple[dict[str, Any], bytes]:
    manifest, raw = _load_json_document(DEFAULT_MANIFEST_PATH)
    expected_keys = {
        "schema",
        "status",
        "claim_boundary",
        "implementation_commit",
        "implementation_files",
        "batch",
        "oracle_commitment",
        "frozen_inputs",
        "unknown_cue_absence",
        "readiness",
        "manifest_sha256",
    }
    if set(manifest) != expected_keys:
        raise FrozenRustDocsM0GateError("READY_UNSPENT manifest schema changed")
    payload = dict(manifest)
    observed = payload.pop("manifest_sha256")
    if (
        manifest["schema"] != MANIFEST_SCHEMA
        or manifest["status"] != READY_UNSPENT
        or manifest["claim_boundary"] != CLAIM_BOUNDARY
        or not _valid_sha256(observed)
        or observed != _digest(_MANIFEST_NAMESPACE, payload)
    ):
        raise FrozenRustDocsM0GateError("READY_UNSPENT manifest root changed")
    return manifest, raw


def _ready_receipt_from_record(value: object) -> ReadyUnspentReceiptV1:
    if type(value) is not dict:
        raise FrozenRustDocsM0GateError("manifest readiness is not one exact record")
    fields = dict(value)
    work_value = fields.get("work")
    if type(work_value) is not dict:
        raise FrozenRustDocsM0GateError("manifest readiness work changed")
    try:
        fields["work"] = FrozenWorkCountersV1(**work_value)
        receipt = ReadyUnspentReceiptV1(**fields)
    except TypeError as exc:
        raise FrozenRustDocsM0GateError("manifest readiness schema changed") from exc
    receipt.authenticate()
    return receipt


def _authenticate_manifest_static(manifest: Mapping[str, object], raw: bytes) -> None:
    ready = _ready_receipt_from_record(manifest.get("readiness"))
    unknown_proof = manifest.get("unknown_cue_absence")
    _authenticate_unknown_cue_absence_record(unknown_proof)
    if ready.unknown_cue_absence_sha256 != unknown_proof["sha256"]:
        raise FrozenRustDocsM0GateError("unknown-cue readiness root changed")
    commit = manifest["implementation_commit"]
    if type(commit) is not str:
        raise FrozenRustDocsM0GateError("implementation commit changed")
    expected_files = _implementation_records(commit)
    if not _json_exact_equal(manifest["implementation_files"], _plain(expected_files)):
        raise FrozenRustDocsM0GateError("implementation roots changed")
    expected_batch = FrozenBatchV1(
        FROZEN_CASES,
        BATCH_PAYLOAD_SHA256,
        EXPECTED_BATCH_FILE_SHA256,
    ).record()
    if not _json_exact_equal(manifest["batch"], expected_batch):
        raise FrozenRustDocsM0GateError("manifest batch commitment changed")
    expected_oracle = {
        "path": DEFAULT_ORACLE_PATH.relative_to(REPO_ROOT).as_posix(),
        "file_sha256": EXPECTED_ORACLE_FILE_SHA256,
        "payload_sha256": ORACLE_PAYLOAD_SHA256,
        "batch_payload_sha256": BATCH_PAYLOAD_SHA256,
        "runtime_read_count_at_freeze": 0,
    }
    if not _json_exact_equal(manifest["oracle_commitment"], expected_oracle):
        raise FrozenRustDocsM0GateError("oracle commitment changed")
    tasks = _project_tasks()
    if not _json_exact_equal(manifest["frozen_inputs"], _frozen_inputs_record(tasks)):
        raise FrozenRustDocsM0GateError("frozen task/candidate/control inputs changed")
    _oracle_is_tracked()  # No content read.
    relative_manifest = DEFAULT_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()
    if raw != _git_blob(_git_head(), relative_manifest):
        raise FrozenRustDocsM0GateError("READY_UNSPENT manifest must be committed before execution")


def _prepare_ready_unspent() -> _PreparedRustDocsM0GateV1:
    """Rebuild exact frozen roots with zero evaluation work before spending."""

    # Manifest authentication is deliberately first: with no committed manifest,
    # this function cannot build, recall, materialize, compile, run, or read oracle.
    manifest, raw = _load_manifest()
    _authenticate_manifest_static(manifest, raw)
    if DEFAULT_SPENT_PATH.exists() or DEFAULT_RESULT_PATH.exists():
        raise FrozenRustDocsM0GateError("READY_UNSPENT gate is already spent")
    batch = load_frozen_batch()
    live = _build_live(batch)
    ready = _make_ready_receipt(live, batch)
    if manifest["readiness"] != ready.record() or manifest[
        "unknown_cue_absence"
    ] != _unknown_cue_absence_record(live):
        raise FrozenRustDocsM0GateError("rebuilt READY_UNSPENT roots changed")
    prepared = _PreparedRustDocsM0GateV1(
        manifest=manifest,
        batch=batch,
        _live=live,
        ready_receipt=ready,
    )
    prepared.authenticate_ready_unspent()
    return prepared


def _recall_item(
    name: str,
    recall: knowledge_action.KnowledgeActionRecallV1,
) -> dict[str, object]:
    recall.authenticate()
    return {"name": name, "kind": "physical_recall", "recall": recall.record()}


def _score_values(recall: knowledge_action.KnowledgeActionRecallV1) -> tuple[int, ...]:
    return tuple(score for _symbol, score in recall.candidate_scores)


def _authenticate_expected_control_outcomes(
    prepared: _PreparedRustDocsM0GateV1,
    recalls: tuple[knowledge_action.KnowledgeActionRecallV1, ...],
    items: tuple[Mapping[str, object], ...],
) -> None:
    (
        primary,
        contrast,
        reversed_primary,
        cell_off,
        plasticity_off,
        shared,
        unknown,
        channel,
        mutex,
    ) = recalls
    channel_handle = prepared.ready_receipt.channel_action_handle
    mutex_handle = prepared.ready_receipt.mutex_action_handle
    expected_task_sha256s = tuple(
        prepared.ready_receipt.task_projection_sha256s[name] for name in _CONTROL_RECALL_TASK_NAMES
    )
    if (
        tuple(recall.task_sha256 for recall in recalls) != expected_task_sha256s
        or primary.status != knowledge_action.RECALLED
        or primary.selected_candidate_id != channel_handle
        or contrast.status != knowledge_action.RECALLED
        or contrast.selected_candidate_id != mutex_handle
        or reversed_primary.status != knowledge_action.RECALLED
        or reversed_primary.selected_candidate_id != channel_handle
        or reversed_primary.record() != primary.record()
        or cell_off.status != knowledge_action.STOP_CELL_OFF
        or plasticity_off.status != knowledge_action.STOP_PHYSICAL_RACE
        or any(_score_values(plasticity_off))
        or shared.status != knowledge_action.STOP_PHYSICAL_RACE
        or not _score_values(shared)
        or len(set(_score_values(shared))) != 1
        or _score_values(shared)[0] <= 0
        or unknown.status != knowledge_action.STOP_NO_MATCHED_FACTOR
        or any(_score_values(unknown))
        or channel.status != knowledge_action.RECALLED
        or channel.selected_candidate_id != channel_handle
        or mutex.status != knowledge_action.RECALLED
        or mutex.selected_candidate_id != mutex_handle
        or not _is_exact_splice_item(items[5])
    ):
        raise FrozenRustDocsM0GateError("frozen causal-control outcome changed")


def _spliced_bound_memory(
    prepared: _PreparedRustDocsM0GateV1,
) -> rust_action_registry.BoundRustActionMemoryV1:
    """Forge a receipt/object splice solely for a pre-physical authority control."""

    memory_type = type(prepared._live.memory)
    spliced = object.__new__(memory_type)
    for attribute in ("_knowledge_projection", "_association_projection", "_memory"):
        object.__setattr__(spliced, attribute, getattr(prepared._live.memory, attribute))
    object.__setattr__(
        spliced,
        "_receipt",
        prepared._live.no_plasticity_memory.receipt,
    )
    return spliced


def _run_control_roster(
    prepared: _PreparedRustDocsM0GateV1,
) -> FrozenControlBundleV1:
    """Run every fixed control before any selected source can materialize."""

    prepared.authenticate_ready_unspent()
    live = prepared._live
    primary = live.registry.recall(live.memory, live.tasks["primary"])
    contrast = live.registry.recall(live.memory, live.tasks["contrast"])
    reversed_primary = live.reversed_registry.recall(
        live.reversed_memory,
        live.tasks["primary"],
    )
    cell_off = live.registry.recall(
        live.memory,
        live.tasks["primary"],
        cell_gate=False,
    )
    plasticity_off = live.registry.recall(
        live.no_plasticity_memory,
        live.tasks["primary"],
    )
    try:
        live.registry.recall(_spliced_bound_memory(prepared), live.tasks["primary"])
    except rust_action_registry.RustActionRegistryError:
        splice_item: Mapping[str, object] = {
            "name": "bound_memory_splice",
            "kind": "authority_control",
            "status": "STOP_BOUND_MEMORY_SPLICE",
            "physical_recall_count": 0,
        }
    else:
        raise FrozenRustDocsM0GateError("bound-memory splice reached physical recall")
    shared = live.registry.recall(live.memory, live.tasks["shared_factor_thread"])
    unknown = live.registry.recall(live.memory, live.tasks["unknown_factor"])
    channel = live.registry.recall(live.memory, live.tasks["single_factor_channel"])
    mutex = live.registry.recall(live.memory, live.tasks["single_factor_mutex"])
    recalls = (
        primary,
        contrast,
        reversed_primary,
        cell_off,
        plasticity_off,
        shared,
        unknown,
        channel,
        mutex,
    )
    items = (
        _recall_item("primary", primary),
        _recall_item("contrast", contrast),
        _recall_item("source_order_reversal", reversed_primary),
        _recall_item("cell_off", cell_off),
        _recall_item("plasticity_off", plasticity_off),
        splice_item,
        _recall_item("shared_factor_thread", shared),
        _recall_item("unknown_factor", unknown),
        _recall_item("single_factor_channel", channel),
        _recall_item("single_factor_mutex", mutex),
    )
    work = FrozenWorkCountersV1(recall_count=len(recalls))
    payload = {
        "schema": CONTROL_SCHEMA,
        "ready_sha256": prepared.ready_receipt.sha256,
        "roster_sha256": CONTROL_ROSTER_SHA256,
        "items": items,
        "control_count": len(items),
        "physical_recall_count": len(recalls),
        "work": asdict(work),
    }
    bundle = FrozenControlBundleV1(
        recalls=recalls,
        items=items,
        ready_sha256=prepared.ready_receipt.sha256,
        work=work,
        sha256=_digest(_CONTROL_NAMESPACE, payload),
    )
    bundle.authenticate(prepared)
    return bundle


def _execute_selected_once(
    prepared: _PreparedRustDocsM0GateV1,
    controls: FrozenControlBundleV1,
) -> AuthenticatedSelectedExecutionV1:
    """Materialize and execute only the authenticated primary winner, once."""

    controls.authenticate(prepared)
    execution = prepared._live.executor.execute_selected_once(controls.primary_recall)
    execution.authenticate_structure()
    work = FrozenWorkCountersV1(
        recall_count=9,
        selected_source_materialization_count=1,
        selected_executor_call_count=1,
        authenticated_rustc_compile_receipt_count=1,
        authenticated_binary_run_receipt_count=1,
    )
    provisional = AuthenticatedSelectedExecutionV1(
        execution=execution,
        control_bundle_sha256=controls.sha256,
        work=work,
        sha256="0" * 64,
    )
    result = AuthenticatedSelectedExecutionV1(
        execution=execution,
        control_bundle_sha256=controls.sha256,
        work=work,
        sha256=_digest(
            _EXECUTION_NAMESPACE,
            provisional.record(include_sha256=False),
        ),
    )
    result.authenticate(prepared, controls)
    return result


def _load_oracle_after_authenticated_execution(
    prepared: _PreparedRustDocsM0GateV1,
    controls: FrozenControlBundleV1,
    selected: AuthenticatedSelectedExecutionV1,
) -> dict[str, object]:
    """Perform the first oracle-byte read only after a successful exact receipt."""

    if type(prepared) is not _PreparedRustDocsM0GateV1:
        raise TypeError("exact prepared Rust gate required before oracle")
    if type(controls) is not FrozenControlBundleV1:
        raise TypeError("exact frozen controls required before oracle")
    if type(selected) is not AuthenticatedSelectedExecutionV1:
        raise TypeError("exact authenticated selected execution required before oracle")
    selected.authenticate(prepared, controls)
    # Re-read and re-authenticate concrete toolchain bytes immediately before the
    # oracle becomes visible.
    if prepared._live.executor.rustc_identity != selected.execution.rustc_identity:
        raise FrozenRustDocsM0GateError("Rust toolchain identity changed before oracle")
    value, raw = _load_json_document(DEFAULT_ORACLE_PATH)
    if (
        sha256(raw).hexdigest() != EXPECTED_ORACLE_FILE_SHA256
        or set(value) != {"schema", "batch_payload_sha256", "expected_outputs"}
        or value.get("schema") != ORACLE_SCHEMA
        or value.get("batch_payload_sha256") != BATCH_PAYLOAD_SHA256
    ):
        raise FrozenRustDocsM0GateError("frozen oracle artifact changed")
    raw_outputs = value.get("expected_outputs")
    if type(raw_outputs) is not list or len(raw_outputs) != len(FROZEN_CASES):
        raise FrozenRustDocsM0GateError("frozen oracle output shape changed")
    expected_rows: list[tuple[int, ...]] = []
    total_values = 0
    for row in raw_outputs:
        if (
            type(row) is not list
            or len(row) > MAX_ORACLE_VALUES_PER_CASE
            or any(type(item) is not int or not -(2**63) <= item <= 2**63 - 1 for item in row)
        ):
            raise FrozenRustDocsM0GateError("frozen oracle output values changed")
        total_values += len(row)
        if total_values > MAX_ORACLE_TOTAL_VALUES:
            raise FrozenRustDocsM0GateError("frozen oracle output is too large")
        expected_rows.append(tuple(row))
    expected_outputs = tuple(expected_rows)
    if _digest(_ORACLE_NAMESPACE, expected_outputs) != ORACLE_PAYLOAD_SHA256:
        raise FrozenRustDocsM0GateError("frozen oracle payload root changed")
    matches = selected.execution.outputs == expected_outputs
    work = FrozenWorkCountersV1(
        recall_count=9,
        selected_source_materialization_count=1,
        selected_executor_call_count=1,
        authenticated_rustc_compile_receipt_count=1,
        authenticated_binary_run_receipt_count=1,
        oracle_read_count=1,
    )
    payload = {
        "schema": ADJUDICATION_SCHEMA,
        "status": TERMINAL_PASS if matches else TERMINAL_STOP,
        "oracle_file_sha256": EXPECTED_ORACLE_FILE_SHA256,
        "oracle_payload_sha256": ORACLE_PAYLOAD_SHA256,
        "batch_payload_sha256": BATCH_PAYLOAD_SHA256,
        "execution_sha256": selected.sha256,
        "outputs_match": matches,
        "work": asdict(work),
    }
    return {**payload, "sha256": _digest(_ADJUDICATION_NAMESPACE, payload)}


__all__ = [
    "ADJUDICATION_SCHEMA",
    "BATCH_PAYLOAD_SHA256",
    "BATCH_SCHEMA",
    "CANDIDATE_SOURCE_SET_SHA256",
    "CLAIM_BOUNDARY",
    "CONTROL_ROSTER",
    "CONTROL_ROSTER_SHA256",
    "CONTROL_SCHEMA",
    "DEFAULT_BATCH_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_ORACLE_PATH",
    "DEFAULT_OWNER_LOCK_PATH",
    "DEFAULT_RESULT_PATH",
    "DEFAULT_SPENT_PATH",
    "EXECUTION_SCHEMA",
    "EXPECTED_BATCH_FILE_SHA256",
    "EXPECTED_ORACLE_FILE_SHA256",
    "FROZEN_CASES",
    "FROZEN_PRIMARY_TASK",
    "MANIFEST_SCHEMA",
    "ORACLE_PAYLOAD_SHA256",
    "ORACLE_SCHEMA",
    "READY_UNSPENT",
    "TERMINAL_PASS",
    "TERMINAL_STOP",
    "FrozenRustDocsM0GateError",
    "FrozenWorkCountersV1",
    "ReadyUnspentReceiptV1",
    "freeze_ready_unspent",
    "load_frozen_batch",
]
