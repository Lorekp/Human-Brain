"""Causal raw-document knowledge to candidate-action memory.

The frozen memory retains physical factor/symbol assemblies, uint8 synaptic
co-occurrence ticks, and authenticated provenance roots.  It retains neither
corpus text nor a task answer.  At inference, task and candidate text are
projected independently; learned ticks drive a ``CommonEventCells`` race whose
authenticated symbol output is the only authority allowed to materialize a
candidate patch.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from experiments.g0 import common_event_cells_v1 as common_events
from humanbrain import knowledge_cell_projection as knowledge_projection


SCOPE = "COMMON_KNOWLEDGE_ACTION_MEMORY_V1"
STATUS = "RAW_DOCUMENT_FACTORS_CAUSALLY_SELECT_ONE_CANDIDATE_ACTION"
CLAIM_BOUNDARY = (
    "self-posed-small-vocabulary-raw-document-factor-to-authenticated-"
    "document-or-task-independent-declared-symbol-transfer-with-human-"
    "curated-declared-symbols-as-adapter-scaffolding-not-learned-abstraction-"
    "provenance-physical-cell-race-and-selected-only-materialization-no-"
    "broad-coding-transfer-repository-transfer-leetcode-swe-bench-or-"
    "benchmark-evidence"
)
SYMBOL_SOURCE_POLICY = (
    "symmetric-sorted-deduplicated-union-of-authenticated-first-line-"
    "document-symbols-and-task-independent-corpus-declared-symbols"
)

RECALLED = "RECALLED"
STOP_CELL_OFF = "STOP_KNOWLEDGE_ACTION_CELL_OFF"
STOP_NO_MATCHED_FACTOR = "STOP_KNOWLEDGE_ACTION_NO_MATCHED_FACTOR"
STOP_FACTOR_COLLISION = "STOP_KNOWLEDGE_ACTION_FACTOR_COLLISION"
STOP_CANDIDATE_SYMBOL_UNKNOWN = "STOP_KNOWLEDGE_ACTION_CANDIDATE_SYMBOL_UNKNOWN"
STOP_PHYSICAL_RACE = "STOP_KNOWLEDGE_ACTION_PHYSICAL_RACE"

MAX_QUERY_FACTORS = knowledge_projection.MAX_QUERY_FACTORS
HASH_BUCKET_COUNT = 2**32
MAX_SYMBOLS = 4
MAX_CHUNKS = knowledge_projection.MAX_ADMITTED_CHUNKS
MAX_PERSISTENT_ARRAY_BYTES = 64 * 1024
MAX_TRAINING_CELL_STEPS = 1_000_000
MAX_RECALL_CELL_STEPS = 200_000
TRACE_LANES = common_events.TRACE_LANES
OUTPUT_LANES = common_events.OUTPUT_DIGIT_LANES
OUTPUT_RADIX = common_events.OUTPUT_DIGIT_RADIX
MAX_PHYSICAL_OUTPUT_SCORE = common_events.MAX_OUTPUT_SCORE

CELL_DTYPE = np.dtype(np.int32)
WEIGHT_DTYPE = np.dtype(np.uint8)
FACTOR_BUCKET_DTYPE = np.dtype(np.uint32)
ROOT_DTYPE = np.dtype(np.uint8)

RAW_TEXT_RETENTION_COUNT = 0
TASK_INPUT_COUNT_DURING_FIT = 0
CANDIDATE_INPUT_COUNT_DURING_FIT = 0
CANDIDATE_EXECUTION_COUNT_DURING_RECALL = 0

_CELLS = common_events.CommonEventCells()
_CONSTRUCTION_TOKEN = object()
_FACTOR_EVENT_NAMESPACE = b"common-knowledge-action-factor-v1"
_FACTOR_HASH_NAMESPACE = b"common-knowledge-action-factor-hash-v1"
_SYMBOL_EVENT_NAMESPACE = b"common-knowledge-action-symbol-v1"
_TASK_NAMESPACE = b"common-knowledge-action-task-v1"
_CANDIDATE_NAMESPACE = b"common-knowledge-action-candidate-v1"
_CANDIDATE_SET_NAMESPACE = b"common-knowledge-action-candidate-set-v1"
_CHECKPOINT_NAMESPACE = b"common-knowledge-action-checkpoint-v1"
_RECALL_NAMESPACE = b"common-knowledge-action-recall-v1"
_CONFIG_NAMESPACE = b"common-knowledge-action-config-v1"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")


class KnowledgeActionMemoryError(ValueError):
    """A knowledge/action state or projection failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _array_frame(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    return (
        name.encode("ascii")
        + b"\0"
        + array.dtype.str.encode("ascii")
        + b"\0"
        + _canonical_json(array.shape)
        + b"\0"
        + memoryview(array).cast("B")
    )


def _freeze_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _event_bytes(namespace: bytes, value: str) -> bytes:
    return namespace + b"\0" + value.encode("utf-8")


def _factor_bucket(trace_cells: tuple[int, ...]) -> int:
    if type(trace_cells) is not tuple or len(trace_cells) != TRACE_LANES:
        raise KnowledgeActionMemoryError("factor trace shape changed")
    trace = np.ascontiguousarray(trace_cells, dtype=CELL_DTYPE)
    digest = sha256(
        _FACTOR_HASH_NAMESPACE + b"\0" + memoryview(trace).cast("B")
    ).digest()
    return int.from_bytes(digest[:4], "little")


def _canonical_symbol(value: str) -> str:
    result = value.casefold()
    if re.fullmatch(r"(?:[a-z_][a-z0-9_]*\.)+[a-z_][a-z0-9_]*", result) is None:
        raise KnowledgeActionMemoryError("qualified document symbol is malformed")
    return result


def _qualified_ast_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _qualified_ast_name(node.value)
        if owner is not None:
            return f"{owner}.{node.attr}"
    return None


def _candidate_ast_projection(source: str) -> tuple[str, str]:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise KnowledgeActionMemoryError("candidate source is not valid Python") from exc
    call_nodes = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    if len(call_nodes) != 1:
        raise KnowledgeActionMemoryError(
            "candidate must expose exactly one qualified AST call symbol"
        )
    call_symbol = _qualified_ast_name(call_nodes[0].func)
    if call_symbol is None or "." not in call_symbol:
        raise KnowledgeActionMemoryError(
            "candidate call must be one explicit qualified AST symbol"
        )
    symbol = _canonical_symbol(call_symbol)
    ast_sha256 = sha256(
        ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()
    return symbol, ast_sha256


def _base4_digits(value: int) -> tuple[int, ...]:
    digits = [0] * OUTPUT_LANES
    remaining = value
    for lane in range(OUTPUT_LANES - 1, -1, -1):
        digits[lane] = remaining % OUTPUT_RADIX
        remaining //= OUTPUT_RADIX
    if remaining:
        raise KnowledgeActionMemoryError("symbol address exceeds physical output")
    return tuple(digits)


CONFIG_RECORD = {
    "scope": SCOPE,
    "status": STATUS,
    "claim_boundary": CLAIM_BOUNDARY,
    "max_query_factors": MAX_QUERY_FACTORS,
    "hash_bucket_count": HASH_BUCKET_COUNT,
    "factor_hash_scheme": "CommonEventCells-trace-sha256-full-u32le",
    "max_symbols": MAX_SYMBOLS,
    "max_chunks": MAX_CHUNKS,
    "max_persistent_array_bytes": MAX_PERSISTENT_ARRAY_BYTES,
    "max_training_cell_steps": MAX_TRAINING_CELL_STEPS,
    "max_recall_cell_steps": MAX_RECALL_CELL_STEPS,
    "trace_lanes": TRACE_LANES,
    "output_lanes": OUTPUT_LANES,
    "output_radix": OUTPUT_RADIX,
    "physical_score_scheme": (
        "fixed-prefix-stop-plus-one-and-final-lane-stop-plus-binary-ticks"
    ),
    "knowledge_state": (
        "one-binary-uint8-synaptic-row-per-exact-frozen-factor-cell-fingerprint"
    ),
    "hash_role": "trace-derived-address-accelerator-and-collision-metric-only",
    "symbol_source_policy": SYMBOL_SOURCE_POLICY,
    "admission_chronology": (
        "explicit-acquire-groups-frozen-before-task-and-candidates"
    ),
    "multi_signature_group_policy": (
        "symmetric-all-to-all-binding-with-physical-race-abstention"
    ),
    "final_authority": "CommonEventCells.race_output",
    "raw_text_retention_count": RAW_TEXT_RETENTION_COUNT,
    "task_input_count_during_fit": TASK_INPUT_COUNT_DURING_FIT,
    "candidate_input_count_during_fit": CANDIDATE_INPUT_COUNT_DURING_FIT,
    "candidate_execution_count_during_recall": (
        CANDIDATE_EXECUTION_COUNT_DURING_RECALL
    ),
}
CONFIG_SHA256 = _digest(_CONFIG_NAMESPACE, CONFIG_RECORD)

PYTHON_DOCS_RECORDS_SHA256 = (
    "693f0f178a3ef4d746243a4545a0d2ac0a3b16a1dceac33e7aede48ac546bb8e"
)
PYTHON_DOCS_RECEIPT_SHA256 = (
    "9c384c545941c4092e0f54c14701d2b559fd53c5f5fda2d70d3d20ed898e7567"
)
PYTHON_STDTYPES_RAW_SHA256 = (
    "1e4f964d64e3046ae2cbf6d266c74dbbd17e0a41d3bffbe17e8f158a7d25b2d3"
)
PYTHON_DOCS_ADMITTED_GROUPS = (
    (
        "7ae88ac1f21488e6801fcd5a1a3f69386777cfd8dcb848411e70aaac9f084b1a",
        "0af206b2daff9d927152d90094585a7a9e4083b9cc617ca15dec60d227d4b172",
    ),
    (
        "fb96bd3c2cf52b037ae91a379fe15e5e661f89d9eb252371f8ff6f0c4b312a5e",
        "c0398d8974b99d048cb5e5c2f2a1fa00ceb41c0231ad4fb49cff463f94378b2f",
    ),
)


@dataclass(frozen=True, slots=True)
class KnowledgeActionTaskProjectionV1:
    factors: tuple[str, ...]
    sha256: str

    def authenticate(self) -> None:
        if (
            type(self.factors) is not tuple
            or not self.factors
            or self.factors != tuple(sorted(set(self.factors)))
            or len(self.factors) > MAX_QUERY_FACTORS
            or self.sha256
            != _digest(_TASK_NAMESPACE, {"factors": self.factors})
        ):
            raise KnowledgeActionMemoryError("task projection authentication failed")


def project_knowledge_action_task_v1(
    task_text: str,
) -> KnowledgeActionTaskProjectionV1:
    factors = knowledge_projection.lexical_factors_from_text_v1(task_text)
    if not factors:
        raise KnowledgeActionMemoryError("task projection contains no factors")
    result = KnowledgeActionTaskProjectionV1(
        factors=factors,
        sha256=_digest(_TASK_NAMESPACE, {"factors": factors}),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class KnowledgeActionCandidateProjectionV1:
    candidate_id: str
    symbol: str
    source_sha256: str
    ast_sha256: str
    sha256: str

    def authenticate(self) -> None:
        payload = {
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "source_sha256": self.source_sha256,
            "ast_sha256": self.ast_sha256,
        }
        if (
            type(self.candidate_id) is not str
            or _IDENTIFIER.fullmatch(self.candidate_id) is None
            or re.fullmatch(
                r"(?:[a-z_][a-z0-9_]*\.)+[a-z_][a-z0-9_]*", self.symbol
            )
            is None
            or not _valid_sha256(self.source_sha256)
            or not _valid_sha256(self.ast_sha256)
            or self.sha256 != _digest(_CANDIDATE_NAMESPACE, payload)
        ):
            raise KnowledgeActionMemoryError(
                "candidate projection authentication failed"
            )


def project_knowledge_action_candidate_v1(
    candidate_id: str,
    source: str,
) -> KnowledgeActionCandidateProjectionV1:
    if type(source) is not str:
        raise TypeError("candidate source must be one exact string")
    symbol, ast_sha256 = _candidate_ast_projection(source)
    source_sha256 = sha256(source.encode("utf-8")).hexdigest()
    payload = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "source_sha256": source_sha256,
        "ast_sha256": ast_sha256,
    }
    result = KnowledgeActionCandidateProjectionV1(
        candidate_id=candidate_id,
        symbol=symbol,
        source_sha256=source_sha256,
        ast_sha256=ast_sha256,
        sha256=_digest(_CANDIDATE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _candidate_set_sha256(
    candidates: tuple[KnowledgeActionCandidateProjectionV1, ...],
) -> str:
    return _digest(
        _CANDIDATE_SET_NAMESPACE,
        tuple(
            value.sha256
            for value in sorted(
                candidates, key=lambda candidate: (candidate.symbol, candidate.candidate_id)
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class KnowledgeActionTrainingSummaryV1:
    config_sha256: str
    records_sha256: str
    projection_sha256: str
    graph_sha256: str
    producer_receipt_sha256: str
    distinct_source_factor_count: int
    occupied_factor_bucket_count: int
    factor_hash_collision_count: int
    hash_bucket_count: int
    symbol_count: int
    admitted_chunk_count: int
    associated_chunk_count: int
    association_update_count: int
    projection_cell_steps: int
    synaptic_learning_cell_steps: int
    training_cell_steps: int
    trainable_state_bytes: int
    provenance_state_bytes: int
    checkpoint_array_bytes: int
    plasticity_gate: bool
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class KnowledgeActionRecallWorkV1:
    query_projection_cell_steps: int
    factor_address_cell_steps: int
    factor_trace_equality_cell_steps: int
    synaptic_reads: int
    candidate_projection_cell_steps: int
    candidate_trace_equality_cell_steps: int
    output_race_cell_steps: int
    total_cell_steps: int


@dataclass(frozen=True, slots=True)
class KnowledgeActionRecallV1:
    status: str
    selected_symbol: str | None
    selected_candidate_id: str | None
    matched_factor_count: int
    unknown_factor_count: int
    query_bucket_collision_count: int
    candidate_scores: tuple[tuple[str, int], ...]
    physical_raw_scores: tuple[tuple[int, ...], ...]
    physical_output_record: Mapping[str, object]
    work: KnowledgeActionRecallWorkV1
    checkpoint_sha256: str
    task_sha256: str
    candidate_set_sha256: str
    selected_candidate_sha256: str | None
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            "status": self.status,
            "selected_symbol": self.selected_symbol,
            "selected_candidate_id": self.selected_candidate_id,
            "matched_factor_count": self.matched_factor_count,
            "unknown_factor_count": self.unknown_factor_count,
            "query_bucket_collision_count": self.query_bucket_collision_count,
            "candidate_scores": self.candidate_scores,
            "physical_raw_scores": self.physical_raw_scores,
            "physical_output_record": dict(self.physical_output_record),
            "work": asdict(self.work),
            "checkpoint_sha256": self.checkpoint_sha256,
            "task_sha256": self.task_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "selected_candidate_sha256": self.selected_candidate_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        statuses = {
            RECALLED,
            STOP_CELL_OFF,
            STOP_NO_MATCHED_FACTOR,
            STOP_FACTOR_COLLISION,
            STOP_CANDIDATE_SYMBOL_UNKNOWN,
            STOP_PHYSICAL_RACE,
        }
        score_symbols = tuple(value[0] for value in self.candidate_scores)
        raw_score_shape_valid = (
            type(self.physical_raw_scores) is tuple
            and len(self.physical_raw_scores) == OUTPUT_LANES
            and all(
                type(row) is tuple
                and len(row) == OUTPUT_RADIX
                and all(type(value) is int and value >= 0 for value in row)
                for row in self.physical_raw_scores
            )
        )
        work_values = asdict(self.work) if type(self.work) is KnowledgeActionRecallWorkV1 else {}
        work_total = sum(
            value for name, value in work_values.items() if name != "total_cell_steps"
        )
        if (
            self.status not in statuses
            or type(self.matched_factor_count) is not int
            or self.matched_factor_count < 0
            or type(self.unknown_factor_count) is not int
            or self.unknown_factor_count < 0
            or type(self.query_bucket_collision_count) is not int
            or self.query_bucket_collision_count < 0
            or type(self.candidate_scores) is not tuple
            or any(
                type(value) is not tuple
                or len(value) != 2
                or re.fullmatch(
                    r"(?:[a-z_][a-z0-9_]*\.)+[a-z_][a-z0-9_]*", value[0]
                )
                is None
                or type(value[1]) is not int
                or value[1] < 0
                for value in self.candidate_scores
            )
            or score_symbols != tuple(sorted(set(score_symbols)))
            or not raw_score_shape_valid
            or not isinstance(self.physical_output_record, Mapping)
            or type(self.work) is not KnowledgeActionRecallWorkV1
            or any(type(value) is not int or value < 0 for value in work_values.values())
            or self.work.total_cell_steps != work_total
            or self.work.total_cell_steps > MAX_RECALL_CELL_STEPS
            or not _valid_sha256(self.checkpoint_sha256)
            or not _valid_sha256(self.task_sha256)
            or not _valid_sha256(self.candidate_set_sha256)
        ):
            raise KnowledgeActionMemoryError("recall authentication failed")
        expected_output = _CELLS.race_output(self.physical_raw_scores)
        if dict(self.physical_output_record) != dict(expected_output.record()):
            raise KnowledgeActionMemoryError("recall physical output changed")
        if self.status == RECALLED:
            if (
                self.selected_symbol is None
                or self.selected_candidate_id is None
                or not _valid_sha256(self.selected_candidate_sha256)
                or _IDENTIFIER.fullmatch(self.selected_candidate_id) is None
                or expected_output.decision != "TOKEN"
                or expected_output.token is None
                or not 0 <= expected_output.token < len(self.candidate_scores)
                or self.candidate_scores[expected_output.token][0]
                != self.selected_symbol
                or self.candidate_scores[expected_output.token][1] <= 0
                or self.matched_factor_count == 0
                or self.query_bucket_collision_count != 0
            ):
                raise KnowledgeActionMemoryError(
                    "recalled candidate is not the physical cell winner"
                )
        elif (
            self.selected_symbol is not None
            or self.selected_candidate_id is not None
            or self.selected_candidate_sha256 is not None
        ):
            raise KnowledgeActionMemoryError("abstention cannot select a candidate")
        if self.sha256 != _digest(
            _RECALL_NAMESPACE, self.record(include_sha256=False)
        ):
            raise KnowledgeActionMemoryError("recall root changed")


def _checkpoint_sha256(
    *,
    summary: KnowledgeActionTrainingSummaryV1,
    symbol_names: tuple[str, ...],
    arrays: tuple[tuple[str, np.ndarray], ...],
) -> str:
    summary_record = asdict(summary)
    summary_record.pop("checkpoint_sha256")
    framed = b"".join(_array_frame(name, value) for name, value in arrays)
    return sha256(
        _CHECKPOINT_NAMESPACE
        + b"\0"
        + _canonical_json(
            {"summary": summary_record, "symbol_names": symbol_names}
        )
        + b"\0"
        + framed
    ).hexdigest()


class FrozenCommonKnowledgeActionMemoryV1:
    """One immutable factor/symbol synaptic memory with rooted provenance."""

    __slots__ = (
        "_checkpoint_sha256",
        "_factor_buckets",
        "_factor_cells",
        "_provenance_roots",
        "_summary",
        "_symbol_cells",
        "_symbol_names",
        "_weights",
    )

    def __init__(
        self,
        *,
        factor_buckets: np.ndarray,
        factor_cells: np.ndarray,
        symbol_cells: np.ndarray,
        weights: np.ndarray,
        provenance_roots: np.ndarray,
        symbol_names: tuple[str, ...],
        summary: KnowledgeActionTrainingSummaryV1,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("knowledge-action memories are builder-owned")
        self._factor_buckets = _freeze_array(factor_buckets, FACTOR_BUCKET_DTYPE)
        self._factor_cells = _freeze_array(factor_cells, CELL_DTYPE)
        self._symbol_cells = _freeze_array(symbol_cells, CELL_DTYPE)
        self._weights = _freeze_array(weights, WEIGHT_DTYPE)
        self._provenance_roots = _freeze_array(provenance_roots, ROOT_DTYPE)
        self._symbol_names = symbol_names
        self._summary = summary
        self._checkpoint_sha256 = _checkpoint_sha256(
            summary=summary,
            symbol_names=symbol_names,
            arrays=self._checkpoint_arrays(),
        )
        if self._checkpoint_sha256 != summary.checkpoint_sha256:
            raise KnowledgeActionMemoryError("knowledge-action checkpoint changed")
        self.authenticate()

    def _checkpoint_arrays(self) -> tuple[tuple[str, np.ndarray], ...]:
        return (
            ("factor-buckets", self._factor_buckets),
            ("factor-cells", self._factor_cells),
            ("symbol-cells", self._symbol_cells),
            ("factor-symbol-weights", self._weights),
            ("provenance-roots", self._provenance_roots),
        )

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> KnowledgeActionTrainingSummaryV1:
        self.authenticate()
        return self._summary

    @property
    def symbol_vocabulary(self) -> tuple[str, ...]:
        self.authenticate()
        return self._symbol_names

    @property
    def provenance_roots(self) -> tuple[str, ...]:
        self.authenticate()
        return tuple(bytes(row).hex() for row in self._provenance_roots)

    @property
    def learned_state_shapes(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        self.authenticate()
        return tuple((name, value.shape) for name, value in self._checkpoint_arrays())

    def authenticate(self) -> None:
        arrays = self._checkpoint_arrays()
        symbol_count = self._symbol_cells.shape[0]
        distinct_source_factors = self._factor_cells.shape[0]
        occupied_buckets = len(set(int(value) for value in self._factor_buckets))
        expected_factor_buckets = np.asarray(
            tuple(
                _factor_bucket(tuple(int(value) for value in row))
                for row in self._factor_cells
            ),
            dtype=FACTOR_BUCKET_DTYPE,
        )
        if (
            any(value.flags.writeable for _, value in arrays)
            or self._factor_buckets.shape != (distinct_source_factors,)
            or self._factor_cells.shape != (distinct_source_factors, TRACE_LANES)
            or np.any(self._factor_buckets >= HASH_BUCKET_COUNT)
            or not np.array_equal(self._factor_buckets, expected_factor_buckets)
            or self._symbol_cells.shape != (symbol_count, TRACE_LANES)
            or self._weights.shape != (distinct_source_factors, symbol_count)
            or np.any(self._weights > 1)
            or self._provenance_roots.ndim != 2
            or self._provenance_roots.shape[1:] != (32,)
            or distinct_source_factors < 1
            or not 1 <= symbol_count <= MAX_SYMBOLS
            or self._symbol_names != tuple(sorted(set(self._symbol_names)))
            or len(self._symbol_names) != symbol_count
            or any(
                re.fullmatch(
                    r"(?:[a-z_][a-z0-9_]*\.)+[a-z_][a-z0-9_]*", value
                )
                is None
                for value in self._symbol_names
            )
            or symbol_count != len({tuple(row) for row in self._symbol_cells.tolist()})
        ):
            raise KnowledgeActionMemoryError(
                "knowledge-action checkpoint array authentication failed"
            )
        persistent_bytes = sum(value.nbytes for _, value in arrays)
        trainable_bytes = self._weights.nbytes
        provenance_bytes = self._provenance_roots.nbytes
        summary = self._summary
        if (
            persistent_bytes > MAX_PERSISTENT_ARRAY_BYTES
            or summary.config_sha256 != CONFIG_SHA256
            or summary.distinct_source_factor_count != distinct_source_factors
            or summary.occupied_factor_bucket_count != occupied_buckets
            or summary.factor_hash_collision_count
            != distinct_source_factors - occupied_buckets
            or summary.hash_bucket_count != HASH_BUCKET_COUNT
            or summary.symbol_count != symbol_count
            or summary.trainable_state_bytes != trainable_bytes
            or summary.provenance_state_bytes != provenance_bytes
            or summary.checkpoint_array_bytes != persistent_bytes
            or summary.training_cell_steps > MAX_TRAINING_CELL_STEPS
            or _checkpoint_sha256(
                summary=summary,
                symbol_names=self._symbol_names,
                arrays=arrays,
            )
            != self._checkpoint_sha256
        ):
            raise KnowledgeActionMemoryError(
                "knowledge-action checkpoint summary authentication failed"
            )

    def recall(
        self,
        task: KnowledgeActionTaskProjectionV1,
        candidates: tuple[KnowledgeActionCandidateProjectionV1, ...],
        *,
        cell_gate: bool = True,
    ) -> KnowledgeActionRecallV1:
        self.authenticate()
        if type(task) is not KnowledgeActionTaskProjectionV1:
            raise TypeError("exact task projection required")
        task.authenticate()
        if type(candidates) is not tuple:
            raise TypeError("candidate projections must be one exact tuple")
        if type(cell_gate) is not bool:
            raise TypeError("cell_gate must be one exact boolean")
        for candidate in candidates:
            if type(candidate) is not KnowledgeActionCandidateProjectionV1:
                raise TypeError("exact candidate projections required")
            candidate.authenticate()
        canonical_candidates = tuple(
            sorted(candidates, key=lambda value: (value.symbol, value.candidate_id))
        )
        if (
            not canonical_candidates
            or len({value.candidate_id for value in canonical_candidates})
            != len(canonical_candidates)
            or len({value.symbol for value in canonical_candidates})
            != len(canonical_candidates)
        ):
            raise KnowledgeActionMemoryError(
                "candidate projections must be non-empty and unique"
            )
        candidate_set_sha256 = _candidate_set_sha256(canonical_candidates)

        matched_factor_rows: list[int] = []
        query_buckets: list[int] = []
        query_projection_steps = 0
        factor_trace_equality_steps = 0
        source_trace_collision = False
        for factor in task.factors:
            frame = _CELLS.project(_event_bytes(_FACTOR_EVENT_NAMESPACE, factor))
            query_projection_steps += frame.work.total
            bucket = _factor_bucket(frame.trace_cells)
            query_buckets.append(bucket)
            source_rows = np.flatnonzero(self._factor_buckets == bucket)
            factor_trace_equality_steps += int(source_rows.size) * TRACE_LANES
            exact_rows = source_rows[
                np.all(
                    self._factor_cells[source_rows]
                    == np.asarray(frame.trace_cells, dtype=CELL_DTYPE),
                    axis=1,
                )
            ]
            if exact_rows.size > 1:
                source_trace_collision = True
            elif exact_rows.size == 1:
                matched_factor_rows.append(int(exact_rows[0]))
        query_bucket_collision_count = len(query_buckets) - len(set(query_buckets))
        factor_collision = source_trace_collision or query_bucket_collision_count > 0
        matched_factor_rows = sorted(set(matched_factor_rows))

        candidate_rows: dict[int, KnowledgeActionCandidateProjectionV1] = {}
        candidate_projection_steps = 0
        candidate_trace_equality_steps = 0
        unknown_candidate_symbol = len(canonical_candidates) != len(
            self._symbol_names
        )
        for candidate in canonical_candidates:
            frame = _CELLS.project(
                _event_bytes(_SYMBOL_EVENT_NAMESPACE, candidate.symbol)
            )
            candidate_projection_steps += frame.work.total
            candidate_trace_equality_steps += len(self._symbol_names) * TRACE_LANES
            equal = np.flatnonzero(
                np.all(self._symbol_cells == np.asarray(frame.trace_cells), axis=1)
            )
            if equal.size != 1 or int(equal[0]) in candidate_rows:
                unknown_candidate_symbol = True
            else:
                candidate_rows[int(equal[0])] = candidate

        ticks = np.zeros((len(self._symbol_names),), dtype=np.uint16)
        synaptic_reads = 0
        if cell_gate and not unknown_candidate_symbol and not factor_collision:
            synaptic_reads = len(matched_factor_rows) * len(candidate_rows)
            for symbol_index in candidate_rows:
                if matched_factor_rows:
                    ticks[symbol_index] = np.sum(
                        self._weights[matched_factor_rows, symbol_index],
                        dtype=np.uint16,
                    )

        raw_scores = np.zeros((OUTPUT_LANES, OUTPUT_RADIX), dtype=np.int64)
        if np.any(ticks > 0):
            raw_scores[:-1, 0] = common_events.STOP_SCORE + 1
        for symbol_index in candidate_rows:
            tick_count = int(ticks[symbol_index])
            if tick_count <= 0:
                continue
            digits = _base4_digits(symbol_index)
            if any(digit != 0 for digit in digits[:-1]):
                raise KnowledgeActionMemoryError("symbol prefix exceeds fixed race lane")
            score = common_events.STOP_SCORE + tick_count
            if score > MAX_PHYSICAL_OUTPUT_SCORE:
                raise KnowledgeActionMemoryError("physical candidate score exceeded")
            raw_scores[-1, digits[-1]] = score
        physical_raw_scores = tuple(
            tuple(int(value) for value in row) for row in raw_scores
        )
        output = _CELLS.race_output(physical_raw_scores)

        selected_symbol: str | None = None
        selected_candidate_id: str | None = None
        selected_candidate_sha256: str | None = None
        if not cell_gate:
            status = STOP_CELL_OFF
        elif unknown_candidate_symbol:
            status = STOP_CANDIDATE_SYMBOL_UNKNOWN
        elif factor_collision:
            status = STOP_FACTOR_COLLISION
        elif not matched_factor_rows:
            status = STOP_NO_MATCHED_FACTOR
        elif output.decision != "TOKEN" or output.token not in candidate_rows:
            status = STOP_PHYSICAL_RACE
        elif int(ticks[output.token]) <= 0:
            status = STOP_PHYSICAL_RACE
        else:
            status = RECALLED
            selected_symbol = self._symbol_names[output.token]
            selected_candidate_id = candidate_rows[output.token].candidate_id
            selected_candidate_sha256 = candidate_rows[output.token].sha256

        factor_address_steps = len(task.factors) * TRACE_LANES
        total_steps = (
            query_projection_steps
            + factor_address_steps
            + factor_trace_equality_steps
            + synaptic_reads
            + candidate_projection_steps
            + candidate_trace_equality_steps
            + output.physical_cell_steps
        )
        if total_steps > MAX_RECALL_CELL_STEPS:
            raise KnowledgeActionMemoryError(
                f"recall exceeds {MAX_RECALL_CELL_STEPS} cell steps"
            )
        work = KnowledgeActionRecallWorkV1(
            query_projection_cell_steps=query_projection_steps,
            factor_address_cell_steps=factor_address_steps,
            factor_trace_equality_cell_steps=factor_trace_equality_steps,
            synaptic_reads=synaptic_reads,
            candidate_projection_cell_steps=candidate_projection_steps,
            candidate_trace_equality_cell_steps=candidate_trace_equality_steps,
            output_race_cell_steps=output.physical_cell_steps,
            total_cell_steps=total_steps,
        )
        candidate_scores = tuple(
            (candidate_rows[index].symbol, int(ticks[index]))
            for index in sorted(candidate_rows)
        )
        output_record = MappingProxyType(dict(output.record()))
        record = {
            "status": status,
            "selected_symbol": selected_symbol,
            "selected_candidate_id": selected_candidate_id,
            "matched_factor_count": len(matched_factor_rows),
            "unknown_factor_count": len(task.factors) - len(matched_factor_rows),
            "query_bucket_collision_count": query_bucket_collision_count,
            "candidate_scores": candidate_scores,
            "physical_raw_scores": physical_raw_scores,
            "physical_output_record": dict(output_record),
            "work": asdict(work),
            "checkpoint_sha256": self._checkpoint_sha256,
            "task_sha256": task.sha256,
            "candidate_set_sha256": candidate_set_sha256,
            "selected_candidate_sha256": selected_candidate_sha256,
        }
        result = KnowledgeActionRecallV1(
            status=status,
            selected_symbol=selected_symbol,
            selected_candidate_id=selected_candidate_id,
            matched_factor_count=len(matched_factor_rows),
            unknown_factor_count=len(task.factors) - len(matched_factor_rows),
            query_bucket_collision_count=query_bucket_collision_count,
            candidate_scores=candidate_scores,
            physical_raw_scores=physical_raw_scores,
            physical_output_record=output_record,
            work=work,
            checkpoint_sha256=self._checkpoint_sha256,
            task_sha256=task.sha256,
            candidate_set_sha256=candidate_set_sha256,
            selected_candidate_sha256=selected_candidate_sha256,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )
        result.authenticate()
        self.authenticate()
        return result


def fit_common_knowledge_action_memory_v1(
    projection: knowledge_projection.KnowledgeCellProjectionV1,
    *,
    plasticity_gate: bool = True,
) -> FrozenCommonKnowledgeActionMemoryV1:
    """Fit immutable raw factors to authenticated document/declaration symbols."""

    if type(projection) is not knowledge_projection.KnowledgeCellProjectionV1:
        raise TypeError("exact KnowledgeCellProjectionV1 required")
    if type(plasticity_gate) is not bool:
        raise TypeError("plasticity_gate must be one exact boolean")
    projection.authenticate()
    if projection.split != "acquire":
        raise KnowledgeActionMemoryError(
            "knowledge-action fit accepts only authenticated acquire records"
        )

    factor_names = tuple(
        sorted({factor for chunk in projection.chunks for factor in chunk.content_factors})
    )
    chunk_symbols = tuple(
        tuple(
            sorted(
                {
                    _canonical_symbol(value)
                    for value in chunk.document_symbols + chunk.declared_symbols
                }
            )
        )
        for chunk in projection.chunks
    )
    symbol_names = tuple(sorted({value for values in chunk_symbols for value in values}))
    if not factor_names:
        raise KnowledgeActionMemoryError("source factor vocabulary is empty")
    if not 1 <= len(symbol_names) <= MAX_SYMBOLS:
        raise KnowledgeActionMemoryError("symbol vocabulary is empty or over capacity")

    factor_frames = tuple(
        _CELLS.project(_event_bytes(_FACTOR_EVENT_NAMESPACE, value))
        for value in factor_names
    )
    factor_cells = np.asarray(
        tuple(frame.trace_cells for frame in factor_frames), dtype=CELL_DTYPE
    )
    factor_buckets = np.asarray(
        tuple(_factor_bucket(frame.trace_cells) for frame in factor_frames),
        dtype=FACTOR_BUCKET_DTYPE,
    )
    factor_index = {factor: index for index, factor in enumerate(factor_names)}
    symbol_frames = tuple(
        _CELLS.project(_event_bytes(_SYMBOL_EVENT_NAMESPACE, value))
        for value in symbol_names
    )
    symbol_cells = np.asarray(
        tuple(frame.trace_cells for frame in symbol_frames), dtype=CELL_DTYPE
    )
    if len({tuple(row) for row in symbol_cells.tolist()}) != len(symbol_names):
        raise KnowledgeActionMemoryError("physical symbol projection collided")

    symbol_index = {value: index for index, value in enumerate(symbol_names)}
    weights = np.zeros((len(factor_names), len(symbol_names)), dtype=WEIGHT_DTYPE)
    association_updates = 0
    associated_chunks = 0
    if plasticity_gate:
        for chunk, symbols in zip(projection.chunks, chunk_symbols, strict=True):
            if not symbols:
                continue
            associated_chunks += 1
            for symbol in symbols:
                target = symbol_index[symbol]
                for factor in chunk.content_factors:
                    row = factor_index[factor]
                    if weights[row, target] == 0:
                        weights[row, target] = 1
                        association_updates += 1

    provenance_values = tuple(
        sorted(
            {
                projection.records_sha256,
                projection.sha256,
                projection.graph.sha256,
                projection.producer_receipt.sha256,
                *(chunk.sha256 for chunk in projection.chunks),
                *(
                    root
                    for chunk in projection.chunks
                    for root in chunk.provenance_roots
                ),
            }
        )
    )
    provenance_roots = np.frombuffer(
        b"".join(bytes.fromhex(value) for value in provenance_values),
        dtype=ROOT_DTYPE,
    ).reshape((-1, 32)).copy()
    projection_steps = (
        projection.graph.projection_cell_steps
        + sum(frame.work.total for frame in factor_frames)
        + sum(frame.work.total for frame in symbol_frames)
        + len(factor_names) * TRACE_LANES
    )
    training_steps = projection_steps + association_updates
    arrays = (
        ("factor-buckets", factor_buckets),
        ("factor-cells", factor_cells),
        ("symbol-cells", symbol_cells),
        ("factor-symbol-weights", weights),
        ("provenance-roots", provenance_roots),
    )
    persistent_bytes = sum(value.nbytes for _, value in arrays)
    if persistent_bytes > MAX_PERSISTENT_ARRAY_BYTES:
        raise KnowledgeActionMemoryError("persistent array capacity exceeded")
    if training_steps > MAX_TRAINING_CELL_STEPS:
        raise KnowledgeActionMemoryError("training cell-step capacity exceeded")

    summary = KnowledgeActionTrainingSummaryV1(
        config_sha256=CONFIG_SHA256,
        records_sha256=projection.records_sha256,
        projection_sha256=projection.sha256,
        graph_sha256=projection.graph.sha256,
        producer_receipt_sha256=projection.producer_receipt.sha256,
        distinct_source_factor_count=len(factor_names),
        occupied_factor_bucket_count=len(set(int(value) for value in factor_buckets)),
        factor_hash_collision_count=len(factor_names)
        - len(set(int(value) for value in factor_buckets)),
        hash_bucket_count=HASH_BUCKET_COUNT,
        symbol_count=len(symbol_names),
        admitted_chunk_count=len(projection.admitted_chunk_ids),
        associated_chunk_count=associated_chunks,
        association_update_count=association_updates,
        projection_cell_steps=projection_steps,
        synaptic_learning_cell_steps=association_updates,
        training_cell_steps=training_steps,
        trainable_state_bytes=weights.nbytes,
        provenance_state_bytes=provenance_roots.nbytes,
        checkpoint_array_bytes=persistent_bytes,
        plasticity_gate=plasticity_gate,
        checkpoint_sha256="",
    )
    checkpoint = _checkpoint_sha256(
        summary=summary, symbol_names=symbol_names, arrays=arrays
    )
    summary = KnowledgeActionTrainingSummaryV1(
        **{**asdict(summary), "checkpoint_sha256": checkpoint}
    )
    return FrozenCommonKnowledgeActionMemoryV1(
        factor_buckets=factor_buckets,
        factor_cells=factor_cells,
        symbol_cells=symbol_cells,
        weights=weights,
        provenance_roots=provenance_roots,
        symbol_names=symbol_names,
        summary=summary,
        construction_token=_CONSTRUCTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class KnowledgePatchCandidateV1:
    candidate_id: str
    source: str
    projection: KnowledgeActionCandidateProjectionV1

    @classmethod
    def from_source(cls, candidate_id: str, source: str) -> "KnowledgePatchCandidateV1":
        return cls(
            candidate_id=candidate_id,
            source=source,
            projection=project_knowledge_action_candidate_v1(candidate_id, source),
        )

    def authenticate(self) -> None:
        self.projection.authenticate()
        reparsed = project_knowledge_action_candidate_v1(
            self.candidate_id, self.source
        )
        if (
            reparsed != self.projection
        ):
            raise KnowledgeActionMemoryError("patch candidate authentication failed")


class KnowledgePatchEnvironmentV1:
    """Candidate environment that materializes and executes only the selected patch."""

    def __init__(
        self,
        memory: FrozenCommonKnowledgeActionMemoryV1,
        candidates: Sequence[KnowledgePatchCandidateV1],
    ) -> None:
        if type(memory) is not FrozenCommonKnowledgeActionMemoryV1:
            raise TypeError("exact frozen knowledge-action memory required")
        memory.authenticate()
        values = tuple(candidates)
        for value in values:
            if type(value) is not KnowledgePatchCandidateV1:
                raise TypeError("exact KnowledgePatchCandidateV1 values required")
            value.authenticate()
        if (
            not values
            or len({value.candidate_id for value in values}) != len(values)
            or len({value.projection.symbol for value in values}) != len(values)
            or tuple(sorted(value.projection.symbol for value in values))
            != memory.symbol_vocabulary
        ):
            raise KnowledgeActionMemoryError(
                "patch candidates must map exactly to acquired symbols"
            )
        self._checkpoint_sha256 = memory.checkpoint_sha256
        self._candidate_set_sha256 = _candidate_set_sha256(
            tuple(value.projection for value in values)
        )
        self._candidates = values
        self._materialized_ids: list[str] = []
        self._execution_count = 0

    @property
    def candidate_projections(self) -> tuple[KnowledgeActionCandidateProjectionV1, ...]:
        return tuple(value.projection for value in self._candidates)

    @property
    def materialized_candidate_ids(self) -> tuple[str, ...]:
        return tuple(self._materialized_ids)

    @property
    def execution_count(self) -> int:
        return self._execution_count

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    def materialize_selected(self, recall: KnowledgeActionRecallV1) -> str:
        if type(recall) is not KnowledgeActionRecallV1:
            raise TypeError("exact knowledge-action recall required")
        recall.authenticate()
        if recall.checkpoint_sha256 != self._checkpoint_sha256:
            raise KnowledgeActionMemoryError("recall checkpoint does not match environment")
        if recall.candidate_set_sha256 != self._candidate_set_sha256:
            raise KnowledgeActionMemoryError(
                "recall candidate set does not match environment"
            )
        if recall.status != RECALLED or recall.selected_candidate_id is None:
            raise KnowledgeActionMemoryError("cannot materialize an abstention")
        if self._materialized_ids or self._execution_count:
            raise KnowledgeActionMemoryError(
                "candidate environment permits one materialization only"
            )
        matches = tuple(
            value
            for value in self._candidates
            if value.candidate_id == recall.selected_candidate_id
            and value.projection.symbol == recall.selected_symbol
        )
        if len(matches) != 1:
            raise KnowledgeActionMemoryError("selected patch is not authenticated")
        if recall.selected_candidate_sha256 != matches[0].projection.sha256:
            raise KnowledgeActionMemoryError(
                "selected candidate projection does not match environment"
            )
        self._materialized_ids.append(matches[0].candidate_id)
        return matches[0].source

    def execute_selected(
        self,
        recall: KnowledgeActionRecallV1,
        *,
        function_name: str,
        cases: Iterable[tuple[object, ...]],
    ) -> tuple[object, ...]:
        source = self.materialize_selected(recall)
        namespace: dict[str, object] = {}
        exec(compile(source, "<selected-knowledge-patch>", "exec"), namespace)
        function = namespace.get(function_name)
        if not callable(function):
            raise KnowledgeActionMemoryError("selected patch did not define callable")
        self._execution_count += 1
        return tuple(function(*case) for case in cases)


@dataclass(frozen=True, slots=True)
class KnowledgeActionPilotResultV1:
    recall: KnowledgeActionRecallV1
    outputs: tuple[object, ...]
    materialized_candidate_ids: tuple[str, ...]
    execution_count: int


def run_knowledge_action_pilot_v1(
    memory: FrozenCommonKnowledgeActionMemoryV1,
    *,
    task_text: str,
    environment: KnowledgePatchEnvironmentV1,
    function_name: str,
    cases: Iterable[tuple[object, ...]],
) -> KnowledgeActionPilotResultV1:
    if environment.checkpoint_sha256 != memory.checkpoint_sha256:
        raise KnowledgeActionMemoryError("pilot memory and environment differ")
    task = project_knowledge_action_task_v1(task_text)
    recall = memory.recall(task, environment.candidate_projections)
    outputs: tuple[object, ...] = ()
    if recall.status == RECALLED:
        outputs = environment.execute_selected(
            recall, function_name=function_name, cases=cases
        )
    return KnowledgeActionPilotResultV1(
        recall=recall,
        outputs=outputs,
        materialized_candidate_ids=environment.materialized_candidate_ids,
        execution_count=environment.execution_count,
    )


def run_real_python_docs_knowledge_action_smoke_v1() -> dict[str, object]:
    """Run the four-chunk authenticated local Python-docs integration smoke."""

    from tools.build_python_docs_knowledge_corpus import build_from_cache

    corpus, source_summary = build_from_cache()
    receipt_sha256 = sha256(corpus.canonical_manifest_bytes()).hexdigest()
    if (
        corpus.receipt.records_sha256 != PYTHON_DOCS_RECORDS_SHA256
        or receipt_sha256 != PYTHON_DOCS_RECEIPT_SHA256
        or source_summary.get("records_sha256") != PYTHON_DOCS_RECORDS_SHA256
        or source_summary.get("receipt_sha256") != PYTHON_DOCS_RECEIPT_SHA256
    ):
        raise KnowledgeActionMemoryError("local Python documentation roots changed")
    selected_ids = {
        value for group in PYTHON_DOCS_ADMITTED_GROUPS for value in group
    }
    selected_records = tuple(
        chunk for chunk in corpus.chunks if chunk.chunk_id in selected_ids
    )
    if (
        len(selected_records) != len(selected_ids)
        or any(chunk.split != "acquire" for chunk in selected_records)
        or any(chunk.path != "library/stdtypes.txt" for chunk in selected_records)
        or any(
            chunk.source_file_raw_sha256 != PYTHON_STDTYPES_RAW_SHA256
            for chunk in selected_records
        )
    ):
        raise KnowledgeActionMemoryError(
            "pre-admitted Python documentation records changed"
        )
    projection = knowledge_projection.project_knowledge_corpus_to_cells_v1(
        corpus,
        admitted_chunk_groups=PYTHON_DOCS_ADMITTED_GROUPS,
        split="acquire",
    )
    memory = fit_common_knowledge_action_memory_v1(projection)
    if memory.training_summary.factor_hash_collision_count != 0:
        raise KnowledgeActionMemoryError("Python documentation source address collided")

    candidates = (
        KnowledgePatchCandidateV1.from_source(
            "strip_character_set",
            "def drop_marker(value, marker):\n"
            "    return str.lstrip(value, marker)\n",
        ),
        KnowledgePatchCandidateV1.from_source(
            "remove_whole_marker",
            "def drop_marker(value, marker):\n"
            "    return str.removeprefix(value, marker)\n",
        ),
    )
    environment = KnowledgePatchEnvironmentV1(memory, candidates)
    result = run_knowledge_action_pilot_v1(
        memory,
        task_text=(
            "Remove one complete leading sequence only when it matches; "
            "otherwise preserve the original string unchanged."
        ),
        environment=environment,
        function_name="drop_marker",
        cases=(("xyxpayload", "xy"), ("payload", "xy")),
    )
    if result.recall.query_bucket_collision_count != 0:
        raise KnowledgeActionMemoryError("Python documentation query address collided")
    return {
        "scope": SCOPE,
        "claim_boundary": CLAIM_BOUNDARY,
        "records_sha256": corpus.receipt.records_sha256,
        "receipt_sha256": receipt_sha256,
        "admitted_chunk_groups": PYTHON_DOCS_ADMITTED_GROUPS,
        "projection_sha256": projection.sha256,
        "graph_sha256": projection.graph.sha256,
        "producer_receipt_sha256": projection.producer_receipt.sha256,
        "training_summary": asdict(memory.training_summary),
        "recall": result.recall.record(),
        "outputs": result.outputs,
        "materialized_candidate_ids": result.materialized_candidate_ids,
        "execution_count": result.execution_count,
    }


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-python-docs", action="store_true")
    arguments = parser.parse_args()
    if not arguments.real_python_docs:
        parser.error("pass --real-python-docs to run the cache-dependent smoke")
    print(
        json.dumps(
            run_real_python_docs_knowledge_action_smoke_v1(),
            sort_keys=True,
            indent=2,
        )
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "CONFIG_SHA256",
    "FrozenCommonKnowledgeActionMemoryV1",
    "KnowledgeActionCandidateProjectionV1",
    "KnowledgeActionMemoryError",
    "KnowledgeActionPilotResultV1",
    "KnowledgeActionRecallV1",
    "KnowledgeActionTaskProjectionV1",
    "KnowledgePatchCandidateV1",
    "KnowledgePatchEnvironmentV1",
    "PYTHON_DOCS_ADMITTED_GROUPS",
    "PYTHON_DOCS_RECEIPT_SHA256",
    "PYTHON_DOCS_RECORDS_SHA256",
    "RECALLED",
    "STOP_CELL_OFF",
    "STOP_FACTOR_COLLISION",
    "STOP_NO_MATCHED_FACTOR",
    "STOP_PHYSICAL_RACE",
    "fit_common_knowledge_action_memory_v1",
    "project_knowledge_action_candidate_v1",
    "project_knowledge_action_task_v1",
    "run_knowledge_action_pilot_v1",
    "run_real_python_docs_knowledge_action_smoke_v1",
]


if __name__ == "__main__":
    _main()
