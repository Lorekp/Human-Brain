"""Public C2a replication and executed-consequence cohort.

Two fresh, content-root-distinct synthetic failed-to-successful Rust episodes rebuild the frozen
C1 token-edit learner without importing any C1 receipt or checkpoint.  The
resulting checkpoint is pinned before a held-out threshold-classification
query.  The query proposal is sealed before a new two-source intervention is
prepared, then the exact baseline and exact proposal bytes execute once each
before a single guarded public oracle opens.

This establishes at most frozen-mechanism replication plus one held-out
executed consequence.  It does not establish continuation of the spent C1
lineage, learned semantics, diagnosis, generic repair, consequence learning,
protected transfer, SWE performance, or intelligence.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import common_continual_token_edit_schema_v1 as continual
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from experiments.r3_pilot import common_rust_action_intervention_v1 as intervention
from humanbrain import rust_selected_action_executor as rust_executor

SCHEMA = "humanbrain.continual-rust-token-edit-schema-transfer-cohort.v1"
RESULT_DOCUMENT_SCHEMA = (
    "humanbrain.continual-rust-token-edit-schema-transfer-cohort-result-document.v1"
)
ABORT_SCHEMA = "humanbrain.continual-rust-token-edit-schema-transfer-cohort-abort.v1"
CLAIM_BOUNDARY = (
    "c2a-frozen-token-edit-mechanism-replication-from-two-new-content-root-"
    "distinct-synthetic-failed-to-successful-public-rust-episodes-plus-one-held-out-executed-"
    "consequence-under-exact-v2-source-custody-not-spent-c1-lineage-"
    "continuation-semantics-diagnosis-generic-repair-consequence-learning-"
    "protected-transfer-swe-or-intelligence"
)

STATUS_PASS = "PASS_C2_FROZEN_SCHEMA_EXECUTED_CONSEQUENCE_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C2_PROPOSAL_PUBLIC_CONSEQUENCE_V1"
STATUS_BASELINE_NONDISCRIMINATING = "FAIL_C2_NONDISCRIMINATING_BASELINE_V1"
STATUS_EXECUTION_ABORT = "FAIL_C2_EXECUTION_ABORT_V1"

TRAINING = "TRAINING"
TRANSFER = "TRANSFER"
FAILED = "FAILED"
SUCCESSFUL = "SUCCESSFUL"
BASELINE = "BASELINE"
PROPOSAL = "PROPOSAL"

FROZEN_COMMON_FILE_SHA256 = "a0380e8ec30c55887a5e2b72c835d701d597a3ad90701d3e6c07cabed67ef443"
FROZEN_LEARNER_CONFIG_SHA256 = "84b60787deed505352885fad33abe58ba8b0d6c1d16cee688a7d19da0eb912a5"
FROZEN_MECHANISM_COMMIT = "f2de425d5e06104538a0260da30ff0b287aa5f55"
FROZEN_SCHEMA_INDEX = 3
FROZEN_SCHEMA_SHA256 = "7cf44511bced68635e3c77a199542061b404b5861fe6a51c7c6ba88b01c2b014"

EXCLUDED_C1_SOURCE_SHA256S = frozenset(
    {
        "c5bc2f2af0a579325f857006e222f4f59f4f44bd58eceaa0dd8e82baba16f9c5",
        "16504bb28d57532c6621aa0a36c6db23204d54fe7ea0c56132efa8e38b234704",
        "8b225ebb888c282c0e6293215cdaecb36427efcbe3ca7317bbf322e5e1888c49",
        "126b0ce796c248396e311e31a332e28dcdfbb328e0b994d64b39a1f6e3e0877b",
        "a8102ce87ef9940510e2921ce401d3a66aa1b6ac08a6865fc84b733bc322cd71",
        "a5979ba178a792517f4e675fba66fe1a61f96c1a383e8453f32c5320f22fd04a",
    }
)

_SPEC_NAMESPACE = b"continual-rust-token-edit-schema-transfer-spec-v1"
_MATERIAL_NAMESPACE = b"continual-rust-token-edit-schema-transfer-material-v1"
_INTERVENTION_NAMESPACE = b"continual-rust-token-edit-schema-transfer-intervention-v1"
_OUTPUT_NAMESPACE = b"continual-rust-token-edit-schema-transfer-output-v1"
_EVENT_NAMESPACE = b"continual-rust-token-edit-schema-transfer-event-v1"
_ORACLE_NAMESPACE = b"continual-rust-token-edit-schema-transfer-oracle-v1"
_ORACLE_OUTPUT_NAMESPACE = b"continual-rust-token-edit-schema-transfer-oracle-output-v1"
_AUTHORITY_SNAPSHOT_NAMESPACE = b"continual-rust-token-edit-schema-transfer-authority-snapshot-v1"
_WORK_NAMESPACE = b"continual-rust-token-edit-schema-transfer-work-v1"
_RESULT_NAMESPACE = b"continual-rust-token-edit-schema-transfer-result-v1"
_ABORT_NAMESPACE = b"continual-rust-token-edit-schema-transfer-abort-v1"
_DOCUMENT_NAMESPACE = b"continual-rust-token-edit-schema-transfer-document-v1"

_SHA256_HEX = frozenset("0123456789abcdef")

A_BATCH = ((18, 4, 13, -5), (0, 21, 12))
A_EXPECTED_OUTPUTS = ((13, 4, 13, -5), (0, 13, 12))
B_BATCH = ((-11, -6, -1, 8), (0, -9, 3))
B_EXPECTED_OUTPUTS = ((-6, -6, -1, 8), (0, -6, 3))
TRANSFER_BATCH = ((-2, 4, 7, 1), (12, 0, 5))

A_FAILED_SOURCE = """\
pub fn solve(points: &[i64]) -> Vec<i64> {
    let cap = 13_i64;
    points.iter().copied().fold(Vec::new(), |mut result, point| {
        result.push(if point > point { cap } else { point });
        result
    })
}
"""

A_SUCCESSFUL_SOURCE = """\
pub fn solve(points: &[i64]) -> Vec<i64> {
    let cap = 13_i64;
    points.iter().copied().fold(Vec::new(), |mut result, point| {
        result.push(if point > cap { cap } else { point });
        result
    })
}
"""

B_FAILED_SOURCE = """\
pub fn solve(levels: &[i64]) -> Vec<i64> {
    let minimum = -6_i64;
    let mut buffer = Vec::with_capacity(levels.len());
    let mut cursor = 0_usize;
    while cursor < levels.len() {
        let level = levels[cursor];
        buffer.push(if level < level { minimum } else { level });
        cursor += 1;
    }
    buffer
}
"""

B_SUCCESSFUL_SOURCE = """\
pub fn solve(levels: &[i64]) -> Vec<i64> {
    let minimum = -6_i64;
    let mut buffer = Vec::with_capacity(levels.len());
    let mut cursor = 0_usize;
    while cursor < levels.len() {
        let level = levels[cursor];
        buffer.push(if level < minimum { minimum } else { level });
        cursor += 1;
    }
    buffer
}
"""

TRANSFER_SOURCE = """\
pub fn solve(signals: &[i64]) -> Vec<i64> {
    let threshold = 4_i64;
    let mut flags = Vec::with_capacity(signals.len());
    let mut stream = signals.iter().copied();
    loop {
        let Some(signal) = stream.next() else { break };
        flags.push(if signal >= signal && threshold >= 0 { 1 } else { 0 });
    }
    flags
}
"""

A_FAILED_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_c4", "opaque_c2.slot_c4()"
)
A_SUCCESSFUL_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_h8", "opaque_c2.slot_h8()"
)
B_FAILED_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_n6", "opaque_c2.slot_n6()"
)
B_SUCCESSFUL_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_w1", "opaque_c2.slot_w1()"
)
TRANSFER_FIRST_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_j3", "opaque_c2.slot_j3()"
)
TRANSFER_SECOND_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_r8", "opaque_c2.slot_r8()"
)

TRAINING_CANDIDATE_SOURCES = (
    (A_FAILED_CANDIDATE, A_FAILED_SOURCE),
    (A_SUCCESSFUL_CANDIDATE, A_SUCCESSFUL_SOURCE),
    (B_FAILED_CANDIDATE, B_FAILED_SOURCE),
    (B_SUCCESSFUL_CANDIDATE, B_SUCCESSFUL_SOURCE),
)


class ContinualRustTokenEditSchemaTransferCohortError(ValueError):
    """The frozen C2a protocol or retained transcript changed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _source_sha256(source: str) -> str:
    return sha256(source.encode("ascii")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _validate_outputs(
    value: object,
    *,
    label: str,
    expected_cases: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be one exact nonempty tuple")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if type(row) is not tuple:
            raise TypeError(f"{label} rows must be exact tuples")
        if any(type(item) is not int or not -(2**63) <= item < 2**63 for item in row):
            raise TypeError(f"{label} values must be exact signed 64-bit integers")
        rows.append(row)
    result = tuple(rows)
    if expected_cases is not None and len(result) != expected_cases:
        raise ContinualRustTokenEditSchemaTransferCohortError(f"{label} case count changed")
    return result


def _output_sha256(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(
        _OUTPUT_NAMESPACE,
        _validate_outputs(outputs, label="C2 output batch"),
    )


def _current_common_file_sha256() -> str:
    source_path = Path(continual.__file__)
    return sha256(source_path.read_bytes()).hexdigest()


def _content_roots_for_episode(
    *,
    failed_source: str,
    successful_source: str,
    batch: tuple[tuple[int, ...], ...],
    expected_outputs: tuple[tuple[int, ...], ...],
) -> dict[str, str]:
    failed_root = _source_sha256(failed_source)
    successful_root = _source_sha256(successful_source)
    batch_root = _digest(_MATERIAL_NAMESPACE, {"material_kind": "batch", "batch": batch})
    expected_root = _digest(
        _MATERIAL_NAMESPACE,
        {"material_kind": "expected_outputs", "outputs": expected_outputs},
    )
    repository_root = _digest(
        _MATERIAL_NAMESPACE,
        {
            "material_kind": "repository",
            "failed_source_sha256": failed_root,
            "successful_source_sha256": successful_root,
        },
    )
    task_root = _digest(
        _MATERIAL_NAMESPACE,
        {
            "material_kind": "task",
            "repository_sha256": repository_root,
            "batch_sha256": batch_root,
            "expected_output_sha256": expected_root,
        },
    )
    return {
        "repository_sha256": repository_root,
        "task_sha256": task_root,
        "task_instance_sha256": _digest(
            _MATERIAL_NAMESPACE,
            {
                "material_kind": "task_instance",
                "task_sha256": task_root,
                "batch": batch,
            },
        ),
        "task_source_sha256": _digest(
            _MATERIAL_NAMESPACE,
            {
                "material_kind": "task_source",
                "failed_source_sha256": failed_root,
                "successful_source_sha256": successful_root,
            },
        ),
        "test_manifest_sha256": _digest(
            _MATERIAL_NAMESPACE,
            {
                "material_kind": "test_manifest",
                "batch_sha256": batch_root,
                "expected_output_sha256": expected_root,
            },
        ),
    }


@dataclass(frozen=True, slots=True)
class C2EpisodeMaterialV1:
    label: str
    repository_sha256: str
    task_sha256: str
    task_instance_sha256: str
    task_source_sha256: str
    test_manifest_sha256: str
    batch: tuple[tuple[int, ...], ...]
    expected_outputs: tuple[tuple[int, ...], ...]
    failed_source_sha256: str
    successful_source_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        sources = {
            "A": (A_FAILED_SOURCE, A_SUCCESSFUL_SOURCE, A_BATCH, A_EXPECTED_OUTPUTS),
            "B": (B_FAILED_SOURCE, B_SUCCESSFUL_SOURCE, B_BATCH, B_EXPECTED_OUTPUTS),
        }
        if self.label not in sources:
            raise ContinualRustTokenEditSchemaTransferCohortError("C2 episode label changed")
        failed, successful, batch, expected = sources[self.label]
        roots = _content_roots_for_episode(
            failed_source=failed,
            successful_source=successful,
            batch=batch,
            expected_outputs=expected,
        )
        _validate_outputs(self.batch, label=f"episode {self.label} batch")
        _validate_outputs(
            self.expected_outputs,
            label=f"episode {self.label} expected outputs",
            expected_cases=len(self.batch),
        )
        material_roots = (
            self.repository_sha256,
            self.task_sha256,
            self.task_instance_sha256,
            self.task_source_sha256,
            self.test_manifest_sha256,
        )
        if (
            self.batch != batch
            or self.expected_outputs != expected
            or self.failed_source_sha256 != _source_sha256(failed)
            or self.successful_source_sha256 != _source_sha256(successful)
            or any(getattr(self, name) != value for name, value in roots.items())
            or any(not _valid_sha256(root) for root in material_roots)
            or len(set(material_roots)) != 5
            or self.failed_source_sha256 in EXCLUDED_C1_SOURCE_SHA256S
            or self.successful_source_sha256 in EXCLUDED_C1_SOURCE_SHA256S
            or self.sha256 != _digest(_MATERIAL_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 episode material authentication failed"
            )


def _make_episode_material(
    *,
    label: str,
    failed_source: str,
    successful_source: str,
    batch: tuple[tuple[int, ...], ...],
    expected_outputs: tuple[tuple[int, ...], ...],
) -> C2EpisodeMaterialV1:
    roots = _content_roots_for_episode(
        failed_source=failed_source,
        successful_source=successful_source,
        batch=batch,
        expected_outputs=expected_outputs,
    )
    record = {
        "label": label,
        **roots,
        "batch": batch,
        "expected_outputs": expected_outputs,
        "failed_source_sha256": _source_sha256(failed_source),
        "successful_source_sha256": _source_sha256(successful_source),
    }
    result = C2EpisodeMaterialV1(
        **record,
        sha256=_digest(_MATERIAL_NAMESPACE, record),
    )
    result.authenticate()
    return result


def _content_roots_for_transfer() -> dict[str, str]:
    source_root = _source_sha256(TRANSFER_SOURCE)
    batch_root = _digest(
        _MATERIAL_NAMESPACE,
        {"material_kind": "transfer_batch", "batch": TRANSFER_BATCH},
    )
    repository_root = _digest(
        _MATERIAL_NAMESPACE,
        {
            "material_kind": "transfer_repository",
            "input_source_sha256": source_root,
        },
    )
    task_root = _digest(
        _MATERIAL_NAMESPACE,
        {
            "material_kind": "transfer_task",
            "repository_sha256": repository_root,
            "batch_sha256": batch_root,
        },
    )
    contract_root = _digest(
        _MATERIAL_NAMESPACE,
        {
            "material_kind": "public_test_contract",
            "input_shape": "two-batches-of-signed-i64",
            "output_shape": "one-binary-flag-per-input",
            "classification_boundary": "inclusive-fixed-threshold",
        },
    )
    return {
        "repository_sha256": repository_root,
        "task_sha256": task_root,
        "task_instance_sha256": _digest(
            _MATERIAL_NAMESPACE,
            {
                "material_kind": "transfer_task_instance",
                "task_sha256": task_root,
                "batch": TRANSFER_BATCH,
            },
        ),
        "task_source_sha256": _digest(
            _MATERIAL_NAMESPACE,
            {
                "material_kind": "transfer_task_source",
                "input_source_sha256": source_root,
            },
        ),
        "test_manifest_sha256": _digest(
            _MATERIAL_NAMESPACE,
            {
                "material_kind": "transfer_test_manifest",
                "batch_sha256": batch_root,
                "contract_sha256": contract_root,
            },
        ),
    }


@dataclass(frozen=True, slots=True)
class C2TransferMaterialV1:
    repository_sha256: str
    task_sha256: str
    task_instance_sha256: str
    task_source_sha256: str
    test_manifest_sha256: str
    batch: tuple[tuple[int, ...], ...]
    input_source_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        roots = _content_roots_for_transfer()
        material_roots = (
            self.repository_sha256,
            self.task_sha256,
            self.task_instance_sha256,
            self.task_source_sha256,
            self.test_manifest_sha256,
        )
        if (
            self.batch != TRANSFER_BATCH
            or self.input_source_sha256 != _source_sha256(TRANSFER_SOURCE)
            or any(getattr(self, name) != value for name, value in roots.items())
            or any(not _valid_sha256(root) for root in material_roots)
            or len(set(material_roots)) != 5
            or self.input_source_sha256 in EXCLUDED_C1_SOURCE_SHA256S
            or self.sha256 != _digest(_MATERIAL_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 transfer material authentication failed"
            )


def _make_transfer_material() -> C2TransferMaterialV1:
    record = {
        **_content_roots_for_transfer(),
        "batch": TRANSFER_BATCH,
        "input_source_sha256": _source_sha256(TRANSFER_SOURCE),
    }
    result = C2TransferMaterialV1(
        **record,
        sha256=_digest(_MATERIAL_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C2TokenEditSchemaTransferSpecV1:
    schema: str
    claim_boundary: str
    frozen_common_file_sha256: str
    frozen_learner_config_sha256: str
    frozen_mechanism_commit: str
    frozen_schema_index: int
    frozen_schema_sha256: str
    episodes: tuple[C2EpisodeMaterialV1, C2EpisodeMaterialV1]
    transfer: C2TransferMaterialV1
    training_candidate_sha256s: tuple[str, str, str, str]
    transfer_candidate_sha256s: tuple[str, str]
    training_source_sha256s: tuple[str, str, str, str]
    excluded_c1_source_sha256s: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "claim_boundary": self.claim_boundary,
            "frozen_common_file_sha256": self.frozen_common_file_sha256,
            "frozen_learner_config_sha256": self.frozen_learner_config_sha256,
            "frozen_mechanism_commit": self.frozen_mechanism_commit,
            "frozen_schema_index": self.frozen_schema_index,
            "frozen_schema_sha256": self.frozen_schema_sha256,
            "episodes": tuple(episode.record() for episode in self.episodes),
            "transfer": self.transfer.record(),
            "training_candidate_sha256s": self.training_candidate_sha256s,
            "transfer_candidate_sha256s": self.transfer_candidate_sha256s,
            "training_source_sha256s": self.training_source_sha256s,
            "excluded_c1_source_sha256s": self.excluded_c1_source_sha256s,
            "expected_edited_source_committed": False,
            "expected_edited_source_sha256_committed": False,
            "transfer_oracle_outputs_committed": False,
            "transfer_oracle_output_sha256_committed": False,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        expected_episodes = (
            _make_episode_material(
                label="A",
                failed_source=A_FAILED_SOURCE,
                successful_source=A_SUCCESSFUL_SOURCE,
                batch=A_BATCH,
                expected_outputs=A_EXPECTED_OUTPUTS,
            ),
            _make_episode_material(
                label="B",
                failed_source=B_FAILED_SOURCE,
                successful_source=B_SUCCESSFUL_SOURCE,
                batch=B_BATCH,
                expected_outputs=B_EXPECTED_OUTPUTS,
            ),
        )
        expected_training_candidates = tuple(
            candidate.sha256 for candidate, _source in TRAINING_CANDIDATE_SOURCES
        )
        expected_transfer_candidates = (
            TRANSFER_FIRST_CANDIDATE.sha256,
            TRANSFER_SECOND_CANDIDATE.sha256,
        )
        expected_training_sources = tuple(
            _source_sha256(source) for _candidate, source in TRAINING_CANDIDATE_SOURCES
        )
        material_roots = tuple(
            root
            for episode in self.episodes
            for root in (
                episode.repository_sha256,
                episode.task_sha256,
                episode.task_instance_sha256,
                episode.task_source_sha256,
                episode.test_manifest_sha256,
            )
        )
        if (
            self.schema != SCHEMA
            or self.claim_boundary != CLAIM_BOUNDARY
            or self.frozen_common_file_sha256 != FROZEN_COMMON_FILE_SHA256
            or self.frozen_learner_config_sha256 != FROZEN_LEARNER_CONFIG_SHA256
            or self.frozen_learner_config_sha256 != continual.CONFIG_SHA256
            or self.frozen_mechanism_commit != FROZEN_MECHANISM_COMMIT
            or self.frozen_schema_index != FROZEN_SCHEMA_INDEX
            or self.frozen_schema_sha256 != FROZEN_SCHEMA_SHA256
            or continual.PARAMETERIZED_SCHEMAS[FROZEN_SCHEMA_INDEX].sha256 != FROZEN_SCHEMA_SHA256
            or self.episodes != expected_episodes
            or self.transfer != _make_transfer_material()
            or self.training_candidate_sha256s != expected_training_candidates
            or self.transfer_candidate_sha256s != expected_transfer_candidates
            or self.training_source_sha256s != expected_training_sources
            or self.excluded_c1_source_sha256s != tuple(sorted(EXCLUDED_C1_SOURCE_SHA256S))
            or len(set(material_roots)) != 10
            or set(self.training_source_sha256s).intersection(EXCLUDED_C1_SOURCE_SHA256S)
            or self.transfer.input_source_sha256 in EXCLUDED_C1_SOURCE_SHA256S
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 transfer specification authentication failed"
            )
        for episode in self.episodes:
            episode.authenticate()
        self.transfer.authenticate()


def make_c2_token_edit_schema_transfer_spec_v1() -> C2TokenEditSchemaTransferSpecV1:
    """Seal the content-derived public C2a protocol without an answer source."""

    episodes = (
        _make_episode_material(
            label="A",
            failed_source=A_FAILED_SOURCE,
            successful_source=A_SUCCESSFUL_SOURCE,
            batch=A_BATCH,
            expected_outputs=A_EXPECTED_OUTPUTS,
        ),
        _make_episode_material(
            label="B",
            failed_source=B_FAILED_SOURCE,
            successful_source=B_SUCCESSFUL_SOURCE,
            batch=B_BATCH,
            expected_outputs=B_EXPECTED_OUTPUTS,
        ),
    )
    transfer_material = _make_transfer_material()
    partial = C2TokenEditSchemaTransferSpecV1(
        schema=SCHEMA,
        claim_boundary=CLAIM_BOUNDARY,
        frozen_common_file_sha256=FROZEN_COMMON_FILE_SHA256,
        frozen_learner_config_sha256=FROZEN_LEARNER_CONFIG_SHA256,
        frozen_mechanism_commit=FROZEN_MECHANISM_COMMIT,
        frozen_schema_index=FROZEN_SCHEMA_INDEX,
        frozen_schema_sha256=FROZEN_SCHEMA_SHA256,
        episodes=episodes,
        transfer=transfer_material,
        training_candidate_sha256s=tuple(
            candidate.sha256 for candidate, _source in TRAINING_CANDIDATE_SOURCES
        ),
        transfer_candidate_sha256s=(
            TRANSFER_FIRST_CANDIDATE.sha256,
            TRANSFER_SECOND_CANDIDATE.sha256,
        ),
        training_source_sha256s=tuple(
            _source_sha256(source) for _candidate, source in TRAINING_CANDIDATE_SOURCES
        ),
        excluded_c1_source_sha256s=tuple(sorted(EXCLUDED_C1_SOURCE_SHA256S)),
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_SPEC_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


DEFAULT_PUBLIC_SPEC = make_c2_token_edit_schema_transfer_spec_v1()


@dataclass(frozen=True, slots=True)
class _TrainingAttemptV1:
    ordinal: int
    episode_index: int
    phase: str
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1
    source: str


_TRAINING_ATTEMPTS = (
    _TrainingAttemptV1(1, 0, FAILED, A_FAILED_CANDIDATE, A_FAILED_SOURCE),
    _TrainingAttemptV1(2, 0, SUCCESSFUL, A_SUCCESSFUL_CANDIDATE, A_SUCCESSFUL_SOURCE),
    _TrainingAttemptV1(3, 1, FAILED, B_FAILED_CANDIDATE, B_FAILED_SOURCE),
    _TrainingAttemptV1(4, 1, SUCCESSFUL, B_SUCCESSFUL_CANDIDATE, B_SUCCESSFUL_SOURCE),
)


def _executor_batch_commitments(
    batch: tuple[tuple[int, ...], ...],
) -> tuple[str, str, int]:
    cases = rust_executor._validate_batch(batch)
    stdin = rust_executor._encode_batch(cases)
    return (
        rust_executor._digest(rust_executor._V2_BATCH_NAMESPACE, cases),
        sha256(stdin).hexdigest(),
        len(cases),
    )


def _canonical_bindings(
    candidate_sources: tuple[
        tuple[knowledge_action.KnowledgeActionCandidateProjectionV1, str], ...
    ],
) -> tuple[intervention._RustActionBindingV1, ...]:
    return tuple(
        sorted(
            (
                intervention._make_binding(candidate, source)
                for candidate, source in candidate_sources
            ),
            key=lambda binding: binding.candidate.sha256,
        )
    )


def _training_intervention_sha256(
    *,
    spec: C2TokenEditSchemaTransferSpecV1,
    attempt: _TrainingAttemptV1,
    material: C2EpisodeMaterialV1,
    frame: continual.TokenSourceFrameV1,
) -> str:
    batch_sha256, stdin_sha256, case_count = _executor_batch_commitments(material.batch)
    return _digest(
        _INTERVENTION_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "domain": TRAINING,
            "ordinal": attempt.ordinal,
            "episode_material_sha256": material.sha256,
            "phase": attempt.phase,
            "candidate_sha256": attempt.candidate.sha256,
            "source_frame_sha256": frame.sha256,
            "batch_sha256": batch_sha256,
            "stdin_sha256": stdin_sha256,
            "case_count": case_count,
            "expected_output_sha256": _output_sha256(material.expected_outputs),
        },
    )


def _transfer_intervention_sha256(
    *,
    spec: C2TokenEditSchemaTransferSpecV1,
    proposal_sha256: str,
    ordinal: int,
    phase: str,
    candidate_sha256: str,
    frame: continual.TokenSourceFrameV1,
) -> str:
    batch_sha256, stdin_sha256, case_count = _executor_batch_commitments(spec.transfer.batch)
    return _digest(
        _INTERVENTION_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "domain": TRANSFER,
            "proposal_sha256": proposal_sha256,
            "ordinal": ordinal,
            "phase": phase,
            "candidate_sha256": candidate_sha256,
            "source_frame_sha256": frame.sha256,
            "transfer_material_sha256": spec.transfer.sha256,
            "batch_sha256": batch_sha256,
            "stdin_sha256": stdin_sha256,
            "case_count": case_count,
        },
    )


def _make_consequence(
    *,
    execution: rust_executor.RustSelectedActionExecutionV2,
    frame: continual.TokenSourceFrameV1,
    test_manifest_sha256: str,
    expected_outputs: tuple[tuple[int, ...], ...],
) -> continual.TokenEditConsequenceV1:
    execution.authenticate_structure()
    frame.authenticate()
    return continual.make_token_edit_consequence_v1(
        source_sha256=frame.source_sha256,
        batch_sha256=execution.preparation.batch_sha256,
        test_manifest_sha256=test_manifest_sha256,
        execution_sha256=execution.sha256,
        observed_outputs=execution.outputs,
        expected_outputs=expected_outputs,
        compile_cell=execution.compile_receipt.compile_invocations,
        run_cell=execution.run_receipt.run_invocations,
    )


@dataclass(frozen=True, slots=True)
class C2ExecutionEventV1:
    ordinal: int
    domain: str
    phase: str
    previous_event_sha256: str
    candidate_sha256: str
    authority_sha256: str
    catalogue_sha256: str
    intervention_sha256: str
    selection_sha256: str
    selected_source_grant_sha256: str
    release_sha256: str
    source_frame_sha256: str
    source_sha256: str
    batch_sha256: str
    stdin_sha256: str
    execution_sha256: str
    observed_output_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        roots = tuple(
            value
            for name, value in self.record(include_sha256=False).items()
            if name.endswith("_sha256")
        )
        allowed = {
            (1, TRAINING, FAILED),
            (2, TRAINING, SUCCESSFUL),
            (3, TRAINING, FAILED),
            (4, TRAINING, SUCCESSFUL),
            (6, TRANSFER, BASELINE),
            (7, TRANSFER, PROPOSAL),
        }
        if (
            (self.ordinal, self.domain, self.phase) not in allowed
            or any(not _valid_sha256(root) for root in roots)
            or self.sha256 != _digest(_EVENT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 execution event authentication failed"
            )


def _make_execution_event(
    *,
    ordinal: int,
    domain: str,
    phase: str,
    previous_event_sha256: str,
    candidate_sha256: str,
    authority: intervention.RustActionInterventionAuthorityV1,
    catalogue: intervention.RustActionInterventionCatalogueV1,
    selection: intervention.RustActionInterventionSelectionV1,
    execution: rust_executor.RustSelectedActionExecutionV2,
    frame: continual.TokenSourceFrameV1,
) -> C2ExecutionEventV1:
    record = {
        "ordinal": ordinal,
        "domain": domain,
        "phase": phase,
        "previous_event_sha256": previous_event_sha256,
        "candidate_sha256": candidate_sha256,
        "authority_sha256": authority.sha256,
        "catalogue_sha256": catalogue.sha256,
        "intervention_sha256": selection.intervention_sha256,
        "selection_sha256": selection.sha256,
        "selected_source_grant_sha256": execution.selected_source.sha256,
        "release_sha256": execution.selected_source.release_sha256,
        "source_frame_sha256": frame.sha256,
        "source_sha256": frame.source_sha256,
        "batch_sha256": execution.preparation.batch_sha256,
        "stdin_sha256": execution.preparation.stdin_sha256,
        "execution_sha256": execution.sha256,
        "observed_output_sha256": _output_sha256(execution.outputs),
    }
    result = C2ExecutionEventV1(
        **record,
        sha256=_digest(_EVENT_NAMESPACE, record),
    )
    result.authenticate()
    return result


def _event_chain_genesis_sha256(spec_sha256: str) -> str:
    if not _valid_sha256(spec_sha256):
        raise TypeError("C2 event-chain genesis requires one exact spec root")
    return _digest(
        _EVENT_NAMESPACE,
        {"event_chain": "C2_CAUSAL_CHAIN_GENESIS_V1", "spec_sha256": spec_sha256},
    )


@dataclass(frozen=True, slots=True)
class C2ProposalEventV1:
    ordinal: int
    previous_event_sha256: str
    pinned_checkpoint_sha256: str
    query_source_frame_sha256: str
    query_source_sha256: str
    proposal_sha256: str
    proposed_source_sha256: str
    selected_schema_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        roots = tuple(self.record(include_sha256=False).values())[1:]
        if (
            self.ordinal != 5
            or any(not _valid_sha256(root) for root in roots)
            or self.sha256 != _digest(_EVENT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 proposal event authentication failed"
            )


def _make_proposal_event(
    *,
    previous_event_sha256: str,
    pinned_checkpoint_sha256: str,
    query_source: continual.TokenSourceFrameV1,
    proposal: continual.TokenEditProposalV1,
) -> C2ProposalEventV1:
    query_source.authenticate()
    proposal.authenticate()
    if proposal.edited_source_sha256 is None or proposal.selected_schema_sha256 is None:
        raise ContinualRustTokenEditSchemaTransferCohortError(
            "C2 proposal event requires one READY proposal"
        )
    record = {
        "ordinal": 5,
        "previous_event_sha256": previous_event_sha256,
        "pinned_checkpoint_sha256": pinned_checkpoint_sha256,
        "query_source_frame_sha256": query_source.sha256,
        "query_source_sha256": query_source.source_sha256,
        "proposal_sha256": proposal.sha256,
        "proposed_source_sha256": proposal.edited_source_sha256,
        "selected_schema_sha256": proposal.selected_schema_sha256,
    }
    result = C2ProposalEventV1(
        **record,
        sha256=_digest(_EVENT_NAMESPACE, record),
    )
    result.authenticate()
    return result


_TRANSFER_ORACLE_OUTPUT_COMMITMENT_SHA256 = (
    "88bf67679dee8736e8b60e5eae806b9ad9764f4d5cc652bc6e0acb133d26301a"
)


def _make_guarded_transfer_oracle_v1(
    proposal_event_sha256: str,
    retained_transfer_events: Callable[[], tuple[C2ExecutionEventV1, ...]],
) -> Callable[[], tuple[tuple[int, ...], ...]]:
    if not _valid_sha256(proposal_event_sha256):
        raise TypeError("C2 transfer oracle requires one exact proposal-event root")
    if not callable(retained_transfer_events):
        raise TypeError("C2 transfer oracle requires one retained-event callback")
    opened = False

    def open_once() -> tuple[tuple[int, ...], ...]:
        nonlocal opened
        if opened:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "the C2 transfer oracle was already opened"
            )
        events = retained_transfer_events()
        if type(events) is not tuple or len(events) != 2:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "the C2 transfer oracle requires two retained execution events"
            )
        for event in events:
            if type(event) is not C2ExecutionEventV1:
                raise TypeError("C2 transfer oracle event receipts must be exact")
            event.authenticate()
        if (
            tuple((event.ordinal, event.domain, event.phase) for event in events)
            != ((6, TRANSFER, BASELINE), (7, TRANSFER, PROPOSAL))
            or events[0].previous_event_sha256 != proposal_event_sha256
            or events[1].previous_event_sha256 != events[0].sha256
            or len({event.sha256 for event in events}) != 2
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "the C2 transfer oracle event chain changed"
            )
        opened = True
        expected_outputs = ((0, 1, 1, 0), (1, 0, 1))
        if (
            _digest(_ORACLE_OUTPUT_NAMESPACE, expected_outputs)
            != _TRANSFER_ORACLE_OUTPUT_COMMITMENT_SHA256
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "the sealed C2 transfer oracle changed"
            )
        return expected_outputs

    return open_once


@dataclass(frozen=True, slots=True)
class C2TransferOracleOpeningV1:
    ordinal: int
    previous_event_sha256: str
    proposal_event_sha256: str
    transfer_work_snapshot_sha256: str
    transfer_execution_event_sha256s: tuple[str, str]
    invocation_count: int
    test_manifest_sha256: str
    expected_outputs: tuple[tuple[int, ...], ...]
    expected_output_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        outputs = _validate_outputs(
            self.expected_outputs,
            label="C2 transfer oracle outputs",
            expected_cases=len(TRANSFER_BATCH),
        )
        if (
            self.ordinal != 8
            or type(self.transfer_execution_event_sha256s) is not tuple
            or len(self.transfer_execution_event_sha256s) != 2
            or any(
                not _valid_sha256(root)
                for root in (
                    self.previous_event_sha256,
                    self.proposal_event_sha256,
                    self.transfer_work_snapshot_sha256,
                    *self.transfer_execution_event_sha256s,
                )
            )
            or self.previous_event_sha256 != self.transfer_work_snapshot_sha256
            or len(set(self.transfer_execution_event_sha256s)) != 2
            or self.invocation_count != 1
            or not _valid_sha256(self.test_manifest_sha256)
            or self.expected_output_sha256 != _digest(_ORACLE_OUTPUT_NAMESPACE, outputs)
            or self.expected_output_sha256 != _TRANSFER_ORACLE_OUTPUT_COMMITMENT_SHA256
            or self.sha256 != _digest(_ORACLE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 transfer oracle opening authentication failed"
            )


def _make_oracle_opening(
    *,
    previous_event_sha256: str,
    proposal_event_sha256: str,
    transfer_work_snapshot_sha256: str,
    transfer_execution_event_sha256s: tuple[str, str],
    test_manifest_sha256: str,
    expected_outputs: tuple[tuple[int, ...], ...],
) -> C2TransferOracleOpeningV1:
    record = {
        "ordinal": 8,
        "previous_event_sha256": previous_event_sha256,
        "proposal_event_sha256": proposal_event_sha256,
        "transfer_work_snapshot_sha256": transfer_work_snapshot_sha256,
        "transfer_execution_event_sha256s": transfer_execution_event_sha256s,
        "invocation_count": 1,
        "test_manifest_sha256": test_manifest_sha256,
        "expected_outputs": expected_outputs,
        "expected_output_sha256": _digest(_ORACLE_OUTPUT_NAMESPACE, expected_outputs),
    }
    result = C2TransferOracleOpeningV1(
        **record,
        sha256=_digest(_ORACLE_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C2ConsequenceEventV1:
    ordinal: int
    phase: str
    previous_event_sha256: str
    oracle_opening_sha256: str
    execution_event_sha256: str
    consequence_sha256: str
    public_test_cell: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        roots = (
            self.previous_event_sha256,
            self.oracle_opening_sha256,
            self.execution_event_sha256,
            self.consequence_sha256,
        )
        if (
            (self.ordinal, self.phase) not in {(9, BASELINE), (10, PROPOSAL)}
            or any(not _valid_sha256(root) for root in roots)
            or type(self.public_test_cell) is not int
            or self.public_test_cell not in (0, 1)
            or self.sha256 != _digest(_EVENT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 consequence event authentication failed"
            )


def _make_consequence_event(
    *,
    ordinal: int,
    phase: str,
    previous_event_sha256: str,
    oracle_opening_sha256: str,
    execution_event_sha256: str,
    consequence: continual.TokenEditConsequenceV1,
) -> C2ConsequenceEventV1:
    consequence.authenticate()
    record = {
        "ordinal": ordinal,
        "phase": phase,
        "previous_event_sha256": previous_event_sha256,
        "oracle_opening_sha256": oracle_opening_sha256,
        "execution_event_sha256": execution_event_sha256,
        "consequence_sha256": consequence.sha256,
        "public_test_cell": consequence.public_test_cell,
    }
    result = C2ConsequenceEventV1(
        **record,
        sha256=_digest(_EVENT_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C2AuthorityWorkSnapshotV1:
    """Frozen terminal view issued from one live intervention authority.

    The receipt is internally self-authenticating. Its result/document root must be
    pinned by Git or an external attestation to reject a wholly rehashed alternate
    history.
    """

    domain: str
    previous_event_sha256: str
    authority_sha256: str
    catalogue_sha256: str
    issued_selection_sha256s: tuple[str, ...]
    spent_selection_sha256s: tuple[str, ...]
    successful_execution_sha256s: tuple[str, ...]
    execution_event_sha256s: tuple[str, ...]
    issued_selection_count: int
    spent_selection_count: int
    source_release_count: int
    execution_attempt_count: int
    successful_execution_count: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        root_groups = (
            self.issued_selection_sha256s,
            self.spent_selection_sha256s,
            self.successful_execution_sha256s,
            self.execution_event_sha256s,
        )
        counts = (
            self.issued_selection_count,
            self.spent_selection_count,
            self.source_release_count,
            self.execution_attempt_count,
            self.successful_execution_count,
        )
        if (
            self.domain not in {TRAINING, TRANSFER}
            or not _valid_sha256(self.previous_event_sha256)
            or not _valid_sha256(self.authority_sha256)
            or not _valid_sha256(self.catalogue_sha256)
            or any(type(roots) is not tuple for roots in root_groups)
            or any(not _valid_sha256(root) for roots in root_groups for root in roots)
            or any(len(set(roots)) != len(roots) for roots in root_groups)
            or type(self.issued_selection_sha256s) is not tuple
            or self.spent_selection_sha256s != self.issued_selection_sha256s
            or not self.execution_event_sha256s
            or self.previous_event_sha256 != self.execution_event_sha256s[-1]
            or any(type(count) is not int or count < 0 for count in counts)
            or self.issued_selection_count != len(self.issued_selection_sha256s)
            or self.spent_selection_count != len(self.spent_selection_sha256s)
            or self.source_release_count != len(self.successful_execution_sha256s)
            or self.execution_attempt_count != len(self.successful_execution_sha256s)
            or self.successful_execution_count != len(self.successful_execution_sha256s)
            or len(self.execution_event_sha256s) != len(self.successful_execution_sha256s)
            or self.sha256
            != _digest(_AUTHORITY_SNAPSHOT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 authority work snapshot authentication failed"
            )


def _issue_authority_work_snapshot_v1(
    *,
    prepared: intervention.PreparedRustActionInterventionV1,
    domain: str,
    previous_event_sha256: str,
    selections: tuple[intervention.RustActionInterventionSelectionV1, ...],
    executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    events: tuple[C2ExecutionEventV1, ...],
) -> C2AuthorityWorkSnapshotV1:
    """Atomically freeze the authority's live issuance/spend/attempt state."""

    if type(prepared) is not intervention.PreparedRustActionInterventionV1:
        raise TypeError("C2 work snapshot requires one exact prepared authority")
    if type(selections) is not tuple or type(executions) is not tuple or type(events) is not tuple:
        raise TypeError("C2 work snapshot retained rows must be exact tuples")
    with prepared._lock:
        prepared._authenticate_current()
        authority = prepared._authority
        catalogue = prepared._catalogue
        issued_selections = tuple(prepared._issued_selections.values())
        spent_roots = frozenset(prepared._spent_selection_sha256s)
        counters = (
            len(prepared._issued_selections),
            len(prepared._spent_selection_sha256s),
            prepared._source_release_count,
            prepared._execution_attempt_count,
            prepared._successful_execution_count,
        )
    authority.authenticate_structure()
    catalogue.authenticate_structure()
    for selection in selections:
        if type(selection) is not intervention.RustActionInterventionSelectionV1:
            raise TypeError("C2 work snapshot selections must be exact")
        selection.authenticate_structure()
    for execution in executions:
        if type(execution) is not rust_executor.RustSelectedActionExecutionV2:
            raise TypeError("C2 work snapshot executions must be exact")
        execution.authenticate_structure()
    for event in events:
        if type(event) is not C2ExecutionEventV1:
            raise TypeError("C2 work snapshot events must be exact")
        event.authenticate()
    issued_roots = tuple(selection.sha256 for selection in issued_selections)
    retained_selection_roots = tuple(selection.sha256 for selection in selections)
    retained_execution_roots = tuple(execution.sha256 for execution in executions)
    retained_event_roots = tuple(event.sha256 for event in events)
    if (
        authority.catalogue_sha256 != catalogue.sha256
        or issued_selections != selections
        or tuple(root for root in issued_roots if root in spent_roots) != issued_roots
        or tuple(execution.selected_source.selection_sha256 for execution in executions)
        != retained_selection_roots
        or tuple(event.selection_sha256 for event in events) != retained_selection_roots
        or tuple(event.execution_sha256 for event in events) != retained_execution_roots
        or tuple(event.domain for event in events) != (domain,) * len(events)
    ):
        raise ContinualRustTokenEditSchemaTransferCohortError(
            "C2 live authority state differs from retained work"
        )
    record = {
        "domain": domain,
        "previous_event_sha256": previous_event_sha256,
        "authority_sha256": authority.sha256,
        "catalogue_sha256": catalogue.sha256,
        "issued_selection_sha256s": issued_roots,
        "spent_selection_sha256s": tuple(root for root in issued_roots if root in spent_roots),
        "successful_execution_sha256s": retained_execution_roots,
        "execution_event_sha256s": retained_event_roots,
        "issued_selection_count": counters[0],
        "spent_selection_count": counters[1],
        "source_release_count": counters[2],
        "execution_attempt_count": counters[3],
        "successful_execution_count": counters[4],
    }
    result = C2AuthorityWorkSnapshotV1(
        **record,
        sha256=_digest(_AUTHORITY_SNAPSHOT_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C2TransferCohortWorkV1:
    training_issued_selection_count: int
    training_spent_selection_count: int
    training_source_release_count: int
    training_execution_attempt_count: int
    training_successful_execution_count: int
    training_compile_count: int
    training_run_count: int
    schema_advance_count: int
    parent_immutability_check_count: int
    one_source_control_execution_count: int
    cell_off_control_execution_count: int
    schema_off_control_execution_count: int
    transfer_issued_selection_count: int
    transfer_spent_selection_count: int
    transfer_source_release_count: int
    transfer_execution_attempt_count: int
    transfer_successful_execution_count: int
    transfer_compile_count: int
    transfer_run_count: int
    baseline_execution_count: int
    proposal_execution_count: int
    alternative_execution_count: int
    retry_count: int
    unretained_successful_execution_count: int
    unexpected_selection_count: int
    unexpected_source_release_count: int
    oracle_open_count: int
    c1_source_reuse_count: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        values = tuple(self.record(include_sha256=False).values())
        if any(type(value) is not int or value < 0 for value in values) or self.sha256 != _digest(
            _WORK_NAMESPACE, self.record(include_sha256=False)
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 transfer audited work changed"
            )


def _derive_work(
    *,
    training_work_snapshot: C2AuthorityWorkSnapshotV1,
    transfer_work_snapshot: C2AuthorityWorkSnapshotV1,
    training_executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    training_events: tuple[C2ExecutionEventV1, ...],
    episodes: tuple[continual.FailedSuccessfulTokenEditEpisodeV1, ...],
    transfer_executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    transfer_events: tuple[C2ExecutionEventV1, ...],
    control_proposals: tuple[continual.TokenEditProposalV1, ...],
    oracle_opening: C2TransferOracleOpeningV1,
) -> C2TransferCohortWorkV1:
    training_work_snapshot.authenticate()
    transfer_work_snapshot.authenticate()
    all_source_roots = {
        execution.selected_source.source_sha256
        for execution in (*training_executions, *transfer_executions)
    }
    baseline_execution_count = sum(event.phase == BASELINE for event in transfer_events)
    proposal_execution_count = sum(event.phase == PROPOSAL for event in transfer_events)
    retained_event_count = len(training_events) + len(transfer_events)
    retained_execution_count = len(training_executions) + len(transfer_executions)
    total_attempt_count = (
        training_work_snapshot.execution_attempt_count
        + transfer_work_snapshot.execution_attempt_count
    )
    total_success_count = (
        training_work_snapshot.successful_execution_count
        + transfer_work_snapshot.successful_execution_count
    )
    total_selection_count = (
        training_work_snapshot.issued_selection_count
        + transfer_work_snapshot.issued_selection_count
    )
    total_release_count = (
        training_work_snapshot.source_release_count + transfer_work_snapshot.source_release_count
    )
    record = {
        "training_issued_selection_count": training_work_snapshot.issued_selection_count,
        "training_spent_selection_count": training_work_snapshot.spent_selection_count,
        "training_source_release_count": training_work_snapshot.source_release_count,
        "training_execution_attempt_count": training_work_snapshot.execution_attempt_count,
        "training_successful_execution_count": training_work_snapshot.successful_execution_count,
        "training_compile_count": sum(
            item.compile_receipt.compile_invocations for item in training_executions
        ),
        "training_run_count": sum(item.run_receipt.run_invocations for item in training_executions),
        "schema_advance_count": len(episodes),
        "parent_immutability_check_count": len(episodes),
        "one_source_control_execution_count": sum(
            execution.selected_source.source_sha256 == control_proposals[0].edited_source_sha256
            for execution in (*training_executions, *transfer_executions)
            if control_proposals[0].edited_source_sha256 is not None
        ),
        "cell_off_control_execution_count": sum(
            execution.selected_source.source_sha256 == control_proposals[1].edited_source_sha256
            for execution in (*training_executions, *transfer_executions)
            if control_proposals[1].edited_source_sha256 is not None
        ),
        "schema_off_control_execution_count": sum(
            execution.selected_source.source_sha256 == control_proposals[2].edited_source_sha256
            for execution in (*training_executions, *transfer_executions)
            if control_proposals[2].edited_source_sha256 is not None
        ),
        "transfer_issued_selection_count": transfer_work_snapshot.issued_selection_count,
        "transfer_spent_selection_count": transfer_work_snapshot.spent_selection_count,
        "transfer_source_release_count": transfer_work_snapshot.source_release_count,
        "transfer_execution_attempt_count": transfer_work_snapshot.execution_attempt_count,
        "transfer_successful_execution_count": transfer_work_snapshot.successful_execution_count,
        "transfer_compile_count": sum(
            item.compile_receipt.compile_invocations for item in transfer_executions
        ),
        "transfer_run_count": sum(item.run_receipt.run_invocations for item in transfer_executions),
        "baseline_execution_count": baseline_execution_count,
        "proposal_execution_count": proposal_execution_count,
        "alternative_execution_count": max(
            len(transfer_events) - baseline_execution_count - proposal_execution_count,
            0,
        ),
        "retry_count": max(total_attempt_count - retained_event_count, 0),
        "unretained_successful_execution_count": max(
            total_success_count - retained_execution_count,
            0,
        ),
        "unexpected_selection_count": max(total_selection_count - retained_event_count, 0),
        "unexpected_source_release_count": max(
            total_release_count - retained_execution_count,
            0,
        ),
        "oracle_open_count": oracle_opening.invocation_count,
        "c1_source_reuse_count": len(all_source_roots.intersection(EXCLUDED_C1_SOURCE_SHA256S)),
    }
    result = C2TransferCohortWorkV1(
        **record,
        sha256=_digest(_WORK_NAMESPACE, record),
    )
    result.authenticate()
    return result


def _authenticate_selected_execution(
    *,
    authority: intervention.RustActionInterventionAuthorityV1,
    catalogue: intervention.RustActionInterventionCatalogueV1,
    selection: intervention.RustActionInterventionSelectionV1,
    execution: rust_executor.RustSelectedActionExecutionV2,
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
    source: str,
    batch: tuple[tuple[int, ...], ...],
    intervention_sha256: str,
) -> None:
    candidate.authenticate()
    authority.authenticate_structure()
    catalogue.authenticate_structure()
    selection.authenticate_structure()
    execution.authenticate_structure()
    binding = intervention._make_binding(candidate, source)
    expected_release = intervention._make_release(selection, binding)
    batch_sha256, stdin_sha256, case_count = _executor_batch_commitments(batch)
    if (
        authority.catalogue_sha256 != catalogue.sha256
        or selection.authority_sha256 != authority.sha256
        or selection.catalogue_sha256 != catalogue.sha256
        or selection.candidate_sha256 != candidate.sha256
        or selection.intervention_sha256 != intervention_sha256
        or execution.selected_source.authority_sha256 != authority.sha256
        or execution.selected_source.selection_sha256 != selection.sha256
        or execution.selected_source.release_sha256 != expected_release.sha256
        or execution.selected_source.action_handle != candidate.candidate_id
        or execution.selected_source.source_sha256 != binding.rust_source_sha256
        or execution.selected_source.source != source
        or execution.preparation.expected_authority_sha256 != authority.sha256
        or execution.preparation.batch_sha256 != batch_sha256
        or execution.preparation.stdin_sha256 != stdin_sha256
        or execution.preparation.case_count != case_count
        or execution.run_receipt.batch_sha256 != batch_sha256
        or execution.run_receipt.stdin_sha256 != stdin_sha256
    ):
        raise ContinualRustTokenEditSchemaTransferCohortError(
            "C2 selected-source execution custody changed"
        )


def _authenticate_authority_snapshot_link_v1(
    *,
    snapshot: C2AuthorityWorkSnapshotV1,
    domain: str,
    previous_event_sha256: str,
    authority: intervention.RustActionInterventionAuthorityV1,
    catalogue: intervention.RustActionInterventionCatalogueV1,
    selections: tuple[intervention.RustActionInterventionSelectionV1, ...],
    executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    events: tuple[C2ExecutionEventV1, ...],
) -> None:
    snapshot.authenticate()
    selection_roots = tuple(selection.sha256 for selection in selections)
    execution_roots = tuple(execution.sha256 for execution in executions)
    event_roots = tuple(event.sha256 for event in events)
    expected_count = len(executions)
    if (
        snapshot.domain != domain
        or snapshot.previous_event_sha256 != previous_event_sha256
        or snapshot.authority_sha256 != authority.sha256
        or snapshot.catalogue_sha256 != catalogue.sha256
        or snapshot.issued_selection_sha256s != selection_roots
        or snapshot.spent_selection_sha256s != selection_roots
        or snapshot.successful_execution_sha256s != execution_roots
        or snapshot.execution_event_sha256s != event_roots
        or snapshot.issued_selection_count != len(selections)
        or snapshot.spent_selection_count != expected_count
        or snapshot.source_release_count != expected_count
        or snapshot.execution_attempt_count != expected_count
        or snapshot.successful_execution_count != expected_count
    ):
        raise ContinualRustTokenEditSchemaTransferCohortError(
            "C2 authority snapshot differs from the retained causal work"
        )


@dataclass(frozen=True, slots=True)
class C2TokenEditSchemaTransferResultV1:
    schema: str
    claim_boundary: str
    status: str
    spec: C2TokenEditSchemaTransferSpecV1
    common_file_sha256: str
    training_authority: intervention.RustActionInterventionAuthorityV1
    training_catalogue: intervention.RustActionInterventionCatalogueV1
    training_work_snapshot: C2AuthorityWorkSnapshotV1
    training_selections: tuple[intervention.RustActionInterventionSelectionV1, ...]
    training_executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...]
    training_consequences: tuple[continual.TokenEditConsequenceV1, ...]
    training_events: tuple[C2ExecutionEventV1, ...]
    episodes: tuple[continual.FailedSuccessfulTokenEditEpisodeV1, ...]
    checkpoint_lineage_sha256s: tuple[str, str, str]
    checkpoint_array_lineage_sha256s: tuple[str, str, str]
    pinned_checkpoint_sha256: str
    final_checkpoint: continual.FrozenContinualTokenEditSchemaV1
    query_source: continual.TokenSourceFrameV1
    one_source_proposal: continual.TokenEditProposalV1
    cell_off_proposal: continual.TokenEditProposalV1
    schema_off_proposal: continual.TokenEditProposalV1
    transfer_proposal: continual.TokenEditProposalV1
    proposal_event: C2ProposalEventV1
    transfer_authority: intervention.RustActionInterventionAuthorityV1
    transfer_catalogue: intervention.RustActionInterventionCatalogueV1
    transfer_work_snapshot: C2AuthorityWorkSnapshotV1
    transfer_selections: tuple[intervention.RustActionInterventionSelectionV1, ...]
    transfer_executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...]
    transfer_events: tuple[C2ExecutionEventV1, ...]
    oracle_opening: C2TransferOracleOpeningV1
    transfer_consequences: tuple[continual.TokenEditConsequenceV1, ...]
    consequence_events: tuple[C2ConsequenceEventV1, C2ConsequenceEventV1]
    work: C2TransferCohortWorkV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "claim_boundary": self.claim_boundary,
            "status": self.status,
            "spec_sha256": self.spec.sha256,
            "common_file_sha256": self.common_file_sha256,
            "training_authority_sha256": self.training_authority.sha256,
            "training_catalogue_sha256": self.training_catalogue.sha256,
            "training_work_snapshot_sha256": self.training_work_snapshot.sha256,
            "training_selection_sha256s": tuple(item.sha256 for item in self.training_selections),
            "training_execution_sha256s": tuple(item.sha256 for item in self.training_executions),
            "training_consequence_sha256s": tuple(
                item.sha256 for item in self.training_consequences
            ),
            "training_event_sha256s": tuple(item.sha256 for item in self.training_events),
            "episode_sha256s": tuple(item.sha256 for item in self.episodes),
            "checkpoint_lineage_sha256s": self.checkpoint_lineage_sha256s,
            "checkpoint_array_lineage_sha256s": self.checkpoint_array_lineage_sha256s,
            "pinned_checkpoint_sha256": self.pinned_checkpoint_sha256,
            "final_checkpoint_array_sha256": self.final_checkpoint.checkpoint_array_sha256,
            "query_source_frame_sha256": self.query_source.sha256,
            "one_source_proposal_sha256": self.one_source_proposal.sha256,
            "cell_off_proposal_sha256": self.cell_off_proposal.sha256,
            "schema_off_proposal_sha256": self.schema_off_proposal.sha256,
            "transfer_proposal_sha256": self.transfer_proposal.sha256,
            "proposal_event_sha256": self.proposal_event.sha256,
            "transfer_authority_sha256": self.transfer_authority.sha256,
            "transfer_catalogue_sha256": self.transfer_catalogue.sha256,
            "transfer_work_snapshot_sha256": self.transfer_work_snapshot.sha256,
            "transfer_selection_sha256s": tuple(item.sha256 for item in self.transfer_selections),
            "transfer_execution_sha256s": tuple(item.sha256 for item in self.transfer_executions),
            "transfer_event_sha256s": tuple(item.sha256 for item in self.transfer_events),
            "oracle_opening_sha256": self.oracle_opening.sha256,
            "transfer_consequence_sha256s": tuple(
                item.sha256 for item in self.transfer_consequences
            ),
            "consequence_event_sha256s": tuple(item.sha256 for item in self.consequence_events),
            "work_sha256": self.work.sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def transcript_record(self) -> dict[str, object]:
        transcript = _transcript_value(self)
        if type(transcript) is not dict:
            raise TypeError("C2 result transcript must be one exact mapping")
        return transcript

    def authenticate(self, *, expected_result_sha256: str | None = None) -> None:
        """Authenticate the transcript, optionally against an external result pin.

        Internal hashes establish self-consistency. Git custody or another external
        attestation must supply ``expected_result_sha256`` to reject a fully
        rehashed alternate transcript.
        """

        if expected_result_sha256 is not None and not _valid_sha256(expected_result_sha256):
            raise TypeError("C2 expected result root must be one exact SHA-256")
        allowed_statuses = {
            STATUS_PASS,
            STATUS_BEHAVIOR_FAIL,
            STATUS_BASELINE_NONDISCRIMINATING,
        }
        if (
            self.schema != SCHEMA
            or self.claim_boundary != CLAIM_BOUNDARY
            or self.status not in allowed_statuses
            or type(self.spec) is not C2TokenEditSchemaTransferSpecV1
            or self.common_file_sha256 != FROZEN_COMMON_FILE_SHA256
            or self.common_file_sha256 != _current_common_file_sha256()
            or type(self.training_authority) is not intervention.RustActionInterventionAuthorityV1
            or type(self.training_catalogue) is not intervention.RustActionInterventionCatalogueV1
            or type(self.training_work_snapshot) is not C2AuthorityWorkSnapshotV1
            or type(self.training_selections) is not tuple
            or len(self.training_selections) != 4
            or type(self.training_executions) is not tuple
            or len(self.training_executions) != 4
            or type(self.training_consequences) is not tuple
            or len(self.training_consequences) != 4
            or type(self.training_events) is not tuple
            or len(self.training_events) != 4
            or type(self.episodes) is not tuple
            or len(self.episodes) != 2
            or type(self.final_checkpoint) is not continual.FrozenContinualTokenEditSchemaV1
            or type(self.query_source) is not continual.TokenSourceFrameV1
            or type(self.one_source_proposal) is not continual.TokenEditProposalV1
            or type(self.cell_off_proposal) is not continual.TokenEditProposalV1
            or type(self.schema_off_proposal) is not continual.TokenEditProposalV1
            or type(self.transfer_proposal) is not continual.TokenEditProposalV1
            or type(self.proposal_event) is not C2ProposalEventV1
            or type(self.transfer_authority) is not intervention.RustActionInterventionAuthorityV1
            or type(self.transfer_catalogue) is not intervention.RustActionInterventionCatalogueV1
            or type(self.transfer_work_snapshot) is not C2AuthorityWorkSnapshotV1
            or type(self.transfer_selections) is not tuple
            or len(self.transfer_selections) != 2
            or type(self.transfer_executions) is not tuple
            or len(self.transfer_executions) != 2
            or type(self.transfer_events) is not tuple
            or len(self.transfer_events) != 2
            or type(self.oracle_opening) is not C2TransferOracleOpeningV1
            or type(self.transfer_consequences) is not tuple
            or len(self.transfer_consequences) != 2
            or type(self.consequence_events) is not tuple
            or len(self.consequence_events) != 2
            or type(self.work) is not C2TransferCohortWorkV1
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 transfer result structure changed"
            )
        self.spec.authenticate()
        self.training_work_snapshot.authenticate()
        self.transfer_work_snapshot.authenticate()
        self.final_checkpoint.authenticate(expected_checkpoint_sha256=self.pinned_checkpoint_sha256)
        self.query_source.authenticate()

        expected_training_catalogue = intervention._make_catalogue(
            _canonical_bindings(TRAINING_CANDIDATE_SOURCES)
        )
        self.training_authority.authenticate_structure()
        self.training_catalogue.authenticate_structure()
        if (
            self.training_catalogue != expected_training_catalogue
            or self.training_authority.catalogue_sha256 != self.training_catalogue.sha256
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError("C2 training authority changed")

        frames: list[continual.TokenSourceFrameV1] = []
        for index, attempt in enumerate(_TRAINING_ATTEMPTS):
            material = self.spec.episodes[attempt.episode_index]
            frame = continual.project_token_source_v1(attempt.source)
            selection = self.training_selections[index]
            execution = self.training_executions[index]
            consequence = self.training_consequences[index]
            event = self.training_events[index]
            intervention_sha256 = _training_intervention_sha256(
                spec=self.spec,
                attempt=attempt,
                material=material,
                frame=frame,
            )
            _authenticate_selected_execution(
                authority=self.training_authority,
                catalogue=self.training_catalogue,
                selection=selection,
                execution=execution,
                candidate=attempt.candidate,
                source=attempt.source,
                batch=material.batch,
                intervention_sha256=intervention_sha256,
            )
            expected_outputs = (
                material.batch if attempt.phase == FAILED else material.expected_outputs
            )
            expected_consequence = _make_consequence(
                execution=execution,
                frame=frame,
                test_manifest_sha256=material.test_manifest_sha256,
                expected_outputs=material.expected_outputs,
            )
            expected_event = _make_execution_event(
                ordinal=attempt.ordinal,
                domain=TRAINING,
                phase=attempt.phase,
                previous_event_sha256=(
                    _event_chain_genesis_sha256(self.spec.sha256)
                    if index == 0
                    else self.training_events[index - 1].sha256
                ),
                candidate_sha256=attempt.candidate.sha256,
                authority=self.training_authority,
                catalogue=self.training_catalogue,
                selection=selection,
                execution=execution,
                frame=frame,
            )
            if (
                execution.outputs != expected_outputs
                or consequence != expected_consequence
                or consequence.public_test_cell != int(attempt.phase == SUCCESSFUL)
                or event != expected_event
            ):
                raise ContinualRustTokenEditSchemaTransferCohortError(
                    "C2 training execution consequence changed"
                )
            frames.append(frame)

        _authenticate_authority_snapshot_link_v1(
            snapshot=self.training_work_snapshot,
            domain=TRAINING,
            previous_event_sha256=self.training_events[-1].sha256,
            authority=self.training_authority,
            catalogue=self.training_catalogue,
            selections=self.training_selections,
            executions=self.training_executions,
            events=self.training_events,
        )

        expected_episodes = tuple(
            continual.seal_failed_successful_token_edit_episode_v1(
                repository_sha256=material.repository_sha256,
                task_sha256=material.task_sha256,
                task_instance_sha256=material.task_instance_sha256,
                task_source_sha256=material.task_source_sha256,
                test_source_sha256=material.test_manifest_sha256,
                failed_source=frames[episode_index * 2],
                failed_consequence=self.training_consequences[episode_index * 2],
                successful_source=frames[episode_index * 2 + 1],
                successful_consequence=self.training_consequences[episode_index * 2 + 1],
            )
            for episode_index, material in enumerate(self.spec.episodes)
        )
        if self.episodes != expected_episodes:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 training episode custody changed"
            )
        empty = continual.start_continual_token_edit_schema_v1()
        after_a = continual.advance_continual_token_edit_schema_v1(empty, self.episodes[0])
        after_b = continual.advance_continual_token_edit_schema_v1(after_a, self.episodes[1])
        after_b.authenticate(expected_checkpoint_sha256=self.pinned_checkpoint_sha256)
        expected_query = continual.project_token_source_v1(TRANSFER_SOURCE)
        expected_one_source = after_a.propose(expected_query)
        expected_cell_off = after_b.propose(expected_query, cell_gate=False)
        expected_schema_off = after_b.propose(expected_query, schema_gate=False)
        expected_proposal = after_b.propose(expected_query)
        for retained_proposal in (
            self.one_source_proposal,
            self.cell_off_proposal,
            self.schema_off_proposal,
            self.transfer_proposal,
        ):
            retained_proposal.authenticate()
        expected_proposal_event = _make_proposal_event(
            previous_event_sha256=self.training_work_snapshot.sha256,
            pinned_checkpoint_sha256=self.pinned_checkpoint_sha256,
            query_source=expected_query,
            proposal=self.transfer_proposal,
        )
        promoted = tuple(
            index
            for index, support in enumerate(after_b.support_counts)
            if support >= continual.MIN_INDEPENDENT_EPISODES
        )
        training_tokens = {token for frame in frames for token in frame.tokens}
        if (
            self.checkpoint_lineage_sha256s
            != (empty.checkpoint_sha256, after_a.checkpoint_sha256, after_b.checkpoint_sha256)
            or self.checkpoint_array_lineage_sha256s
            != (
                empty.checkpoint_array_sha256,
                after_a.checkpoint_array_sha256,
                after_b.checkpoint_array_sha256,
            )
            or self.pinned_checkpoint_sha256 != after_b.checkpoint_sha256
            or self.final_checkpoint.checkpoint_sha256 != after_b.checkpoint_sha256
            or self.final_checkpoint.checkpoint_array_sha256 != after_b.checkpoint_array_sha256
            or self.final_checkpoint.support_counts != after_b.support_counts
            or self.final_checkpoint.provenance != after_b.provenance
            or self.query_source != expected_query
            or self.one_source_proposal.checkpoint.checkpoint_sha256 != after_a.checkpoint_sha256
            or self.one_source_proposal.input_source != expected_query
            or self.one_source_proposal.input_source is not self.query_source
            or self.one_source_proposal.record() != expected_one_source.record()
            or self.one_source_proposal.edited_source != expected_one_source.edited_source
            or self.one_source_proposal.status != continual.STOP_INSUFFICIENT_INDEPENDENT_SOURCES
            or self.one_source_proposal.schema_raw_scores[FROZEN_SCHEMA_INDEX] != 2
            or self.one_source_proposal.race_simultaneous_spike_candidates != 2
            or self.cell_off_proposal.record() != expected_cell_off.record()
            or self.cell_off_proposal.checkpoint is not self.final_checkpoint
            or self.cell_off_proposal.input_source is not self.query_source
            or self.cell_off_proposal.status != continual.STOP_CELL_OFF
            or self.cell_off_proposal.edited_source is not None
            or self.schema_off_proposal.record() != expected_schema_off.record()
            or self.schema_off_proposal.checkpoint is not self.final_checkpoint
            or self.schema_off_proposal.input_source is not self.query_source
            or self.schema_off_proposal.status != continual.STOP_SCHEMA_OFF
            or self.schema_off_proposal.edited_source is not None
            or self.transfer_proposal.record() != expected_proposal.record()
            or self.transfer_proposal.checkpoint is not self.final_checkpoint
            or self.transfer_proposal.input_source is not self.query_source
            or self.transfer_proposal.edited_source != expected_proposal.edited_source
            or self.transfer_proposal.status != continual.READY
            or self.transfer_proposal.selected_schema_index != FROZEN_SCHEMA_INDEX
            or self.transfer_proposal.selected_schema_sha256 != FROZEN_SCHEMA_SHA256
            or (
                self.transfer_proposal.anchor_token_index,
                self.transfer_proposal.target_token_index,
                self.transfer_proposal.donor_token_index,
            )
            != (75, 76, 78)
            or expected_query.tokens[78] != "threshold"
            or expected_query.tokens[78] in training_tokens
            or promoted != (FROZEN_SCHEMA_INDEX,)
            or after_b.support_counts[FROZEN_SCHEMA_INDEX] != 2
            or sum(after_b.support_counts) != 2
            or after_b.provenance[0].matched_schema_indices != (FROZEN_SCHEMA_INDEX,)
            or after_b.provenance[1].matched_schema_indices != (FROZEN_SCHEMA_INDEX,)
            or after_b.provenance[0].independent_cell != 1
            or after_b.provenance[1].independent_cell != 1
            or self.proposal_event != expected_proposal_event
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 checkpoint, controls, or held-out proposal changed"
            )
        if self.transfer_proposal.edited_source is None:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 READY proposal did not retain exact edited bytes"
            )

        transfer_candidate_sources = (
            (TRANSFER_FIRST_CANDIDATE, TRANSFER_SOURCE),
            (TRANSFER_SECOND_CANDIDATE, self.transfer_proposal.edited_source),
        )
        expected_transfer_catalogue = intervention._make_catalogue(
            _canonical_bindings(transfer_candidate_sources)
        )
        self.transfer_authority.authenticate_structure()
        self.transfer_catalogue.authenticate_structure()
        if (
            self.transfer_catalogue != expected_transfer_catalogue
            or self.transfer_authority.catalogue_sha256 != self.transfer_catalogue.sha256
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError("C2 transfer authority changed")
        transfer_phases = (BASELINE, PROPOSAL)
        transfer_ordinals = (6, 7)
        for index, ((candidate, source), phase, ordinal) in enumerate(
            zip(
                transfer_candidate_sources,
                transfer_phases,
                transfer_ordinals,
                strict=True,
            )
        ):
            frame = continual.project_token_source_v1(source)
            selection = self.transfer_selections[index]
            execution = self.transfer_executions[index]
            intervention_sha256 = _transfer_intervention_sha256(
                spec=self.spec,
                proposal_sha256=self.transfer_proposal.sha256,
                ordinal=ordinal,
                phase=phase,
                candidate_sha256=candidate.sha256,
                frame=frame,
            )
            _authenticate_selected_execution(
                authority=self.transfer_authority,
                catalogue=self.transfer_catalogue,
                selection=selection,
                execution=execution,
                candidate=candidate,
                source=source,
                batch=self.spec.transfer.batch,
                intervention_sha256=intervention_sha256,
            )
            expected_event = _make_execution_event(
                ordinal=ordinal,
                domain=TRANSFER,
                phase=phase,
                previous_event_sha256=(
                    self.proposal_event.sha256
                    if index == 0
                    else self.transfer_events[index - 1].sha256
                ),
                candidate_sha256=candidate.sha256,
                authority=self.transfer_authority,
                catalogue=self.transfer_catalogue,
                selection=selection,
                execution=execution,
                frame=frame,
            )
            if self.transfer_events[index] != expected_event:
                raise ContinualRustTokenEditSchemaTransferCohortError(
                    "C2 transfer execution event changed"
                )
        _authenticate_authority_snapshot_link_v1(
            snapshot=self.transfer_work_snapshot,
            domain=TRANSFER,
            previous_event_sha256=self.transfer_events[-1].sha256,
            authority=self.transfer_authority,
            catalogue=self.transfer_catalogue,
            selections=self.transfer_selections,
            executions=self.transfer_executions,
            events=self.transfer_events,
        )
        if (
            self.transfer_executions[1].selected_source.source
            != self.transfer_proposal.edited_source
            or self.transfer_executions[1].selected_source.source_sha256
            != self.transfer_proposal.edited_source_sha256
            or self.transfer_events[0].previous_event_sha256 != self.proposal_event.sha256
            or self.transfer_events[1].previous_event_sha256 != self.transfer_events[0].sha256
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 proposal bytes or event ordering changed"
            )

        self.oracle_opening.authenticate()
        expected_oracle_opening = _make_oracle_opening(
            previous_event_sha256=self.transfer_work_snapshot.sha256,
            proposal_event_sha256=self.proposal_event.sha256,
            transfer_work_snapshot_sha256=self.transfer_work_snapshot.sha256,
            transfer_execution_event_sha256s=tuple(event.sha256 for event in self.transfer_events),
            test_manifest_sha256=self.spec.transfer.test_manifest_sha256,
            expected_outputs=self.oracle_opening.expected_outputs,
        )
        if (
            self.oracle_opening != expected_oracle_opening
            or self.oracle_opening.previous_event_sha256 != self.transfer_work_snapshot.sha256
            or self.oracle_opening.proposal_event_sha256 != self.proposal_event.sha256
            or self.oracle_opening.transfer_work_snapshot_sha256
            != self.transfer_work_snapshot.sha256
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 oracle or consequence event ordering changed"
            )
        expected_transfer_consequences = tuple(
            _make_consequence(
                execution=execution,
                frame=continual.project_token_source_v1(source),
                test_manifest_sha256=self.spec.transfer.test_manifest_sha256,
                expected_outputs=self.oracle_opening.expected_outputs,
            )
            for execution, (_candidate, source) in zip(
                self.transfer_executions,
                transfer_candidate_sources,
                strict=True,
            )
        )
        if self.transfer_consequences != expected_transfer_consequences:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 transfer consequences changed"
            )
        expected_consequence_events = (
            _make_consequence_event(
                ordinal=9,
                phase=BASELINE,
                previous_event_sha256=self.oracle_opening.sha256,
                oracle_opening_sha256=self.oracle_opening.sha256,
                execution_event_sha256=self.transfer_events[0].sha256,
                consequence=self.transfer_consequences[0],
            ),
            _make_consequence_event(
                ordinal=10,
                phase=PROPOSAL,
                previous_event_sha256=self.consequence_events[0].sha256,
                oracle_opening_sha256=self.oracle_opening.sha256,
                execution_event_sha256=self.transfer_events[1].sha256,
                consequence=self.transfer_consequences[1],
            ),
        )
        if self.consequence_events != expected_consequence_events:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 consequence event chain changed"
            )
        baseline_cell, proposal_cell = (
            consequence.public_test_cell for consequence in self.transfer_consequences
        )
        expected_status = (
            STATUS_BASELINE_NONDISCRIMINATING
            if baseline_cell == 1
            else STATUS_PASS
            if proposal_cell == 1
            else STATUS_BEHAVIOR_FAIL
        )
        if self.status != expected_status:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 terminal behavior status changed"
            )

        expected_work = _derive_work(
            training_work_snapshot=self.training_work_snapshot,
            transfer_work_snapshot=self.transfer_work_snapshot,
            training_executions=self.training_executions,
            training_events=self.training_events,
            episodes=self.episodes,
            transfer_executions=self.transfer_executions,
            transfer_events=self.transfer_events,
            control_proposals=(
                self.one_source_proposal,
                self.cell_off_proposal,
                self.schema_off_proposal,
            ),
            oracle_opening=self.oracle_opening,
        )
        if self.work != expected_work:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 retained work transcript changed"
            )
        source_roots = {
            execution.selected_source.source_sha256
            for execution in (*self.training_executions, *self.transfer_executions)
        }
        if (
            source_roots.intersection(EXCLUDED_C1_SOURCE_SHA256S)
            or len(source_roots) != 6
            or len({item.sha256 for item in self.training_selections}) != 4
            or len({item.sha256 for item in self.transfer_selections}) != 2
            or len({item.sha256 for item in self.training_executions}) != 4
            or len({item.sha256 for item in self.transfer_executions}) != 2
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 source or execution replay was detected"
            )
        self.final_checkpoint.authenticate(expected_checkpoint_sha256=self.pinned_checkpoint_sha256)
        if self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False)) or (
            expected_result_sha256 is not None and self.sha256 != expected_result_sha256
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 transfer result authentication failed"
            )


def _checkpoint_transcript(
    checkpoint: continual.FrozenContinualTokenEditSchemaV1,
) -> dict[str, object]:
    checkpoint.authenticate()
    return {
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_array_sha256": checkpoint.checkpoint_array_sha256,
        "support_counts": checkpoint.support_counts,
        "training_summary": _transcript_value(checkpoint.training_summary),
        "provenance": _transcript_value(checkpoint.provenance),
    }


def _transcript_value(value: object) -> object:
    if type(value) is continual.FrozenContinualTokenEditSchemaV1:
        return _checkpoint_transcript(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _transcript_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if type(value) is dict:
        return {str(key): _transcript_value(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return tuple(_transcript_value(item) for item in value)
    if type(value) in (set, frozenset):
        return tuple(sorted(_transcript_value(item) for item in value))
    if type(value) is bytes:
        return {
            "byte_count": len(value),
            "sha256": sha256(value).hexdigest(),
        }
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise TypeError(f"unsupported C2 transcript value: {type(value).__qualname__}")


class _RunLedgerV1:
    def __init__(self) -> None:
        self.stage = "NOT_STARTED"
        self.values: dict[str, object] = {}

    def mark(self, stage: str, **values: object) -> None:
        self.stage = stage
        self.values.update(values)

    def snapshot(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "values": _transcript_value(self.values),
        }


def _seal_result(
    *,
    spec: C2TokenEditSchemaTransferSpecV1,
    training_authority: intervention.RustActionInterventionAuthorityV1,
    training_catalogue: intervention.RustActionInterventionCatalogueV1,
    training_work_snapshot: C2AuthorityWorkSnapshotV1,
    training_selections: tuple[intervention.RustActionInterventionSelectionV1, ...],
    training_executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    training_consequences: tuple[continual.TokenEditConsequenceV1, ...],
    training_events: tuple[C2ExecutionEventV1, ...],
    episodes: tuple[continual.FailedSuccessfulTokenEditEpisodeV1, ...],
    checkpoint_lineage_sha256s: tuple[str, str, str],
    checkpoint_array_lineage_sha256s: tuple[str, str, str],
    pinned_checkpoint_sha256: str,
    final_checkpoint: continual.FrozenContinualTokenEditSchemaV1,
    query_source: continual.TokenSourceFrameV1,
    one_source_proposal: continual.TokenEditProposalV1,
    cell_off_proposal: continual.TokenEditProposalV1,
    schema_off_proposal: continual.TokenEditProposalV1,
    transfer_proposal: continual.TokenEditProposalV1,
    proposal_event: C2ProposalEventV1,
    transfer_authority: intervention.RustActionInterventionAuthorityV1,
    transfer_catalogue: intervention.RustActionInterventionCatalogueV1,
    transfer_work_snapshot: C2AuthorityWorkSnapshotV1,
    transfer_selections: tuple[intervention.RustActionInterventionSelectionV1, ...],
    transfer_executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    transfer_events: tuple[C2ExecutionEventV1, ...],
    oracle_opening: C2TransferOracleOpeningV1,
    transfer_consequences: tuple[continual.TokenEditConsequenceV1, ...],
    consequence_events: tuple[C2ConsequenceEventV1, C2ConsequenceEventV1],
) -> C2TokenEditSchemaTransferResultV1:
    baseline_cell, proposal_cell = (
        consequence.public_test_cell for consequence in transfer_consequences
    )
    status = (
        STATUS_BASELINE_NONDISCRIMINATING
        if baseline_cell == 1
        else STATUS_PASS
        if proposal_cell == 1
        else STATUS_BEHAVIOR_FAIL
    )
    work = _derive_work(
        training_work_snapshot=training_work_snapshot,
        transfer_work_snapshot=transfer_work_snapshot,
        training_executions=training_executions,
        training_events=training_events,
        episodes=episodes,
        transfer_executions=transfer_executions,
        transfer_events=transfer_events,
        control_proposals=(one_source_proposal, cell_off_proposal, schema_off_proposal),
        oracle_opening=oracle_opening,
    )
    partial = C2TokenEditSchemaTransferResultV1(
        schema=SCHEMA,
        claim_boundary=CLAIM_BOUNDARY,
        status=status,
        spec=spec,
        common_file_sha256=_current_common_file_sha256(),
        training_authority=training_authority,
        training_catalogue=training_catalogue,
        training_work_snapshot=training_work_snapshot,
        training_selections=training_selections,
        training_executions=training_executions,
        training_consequences=training_consequences,
        training_events=training_events,
        episodes=episodes,
        checkpoint_lineage_sha256s=checkpoint_lineage_sha256s,
        checkpoint_array_lineage_sha256s=checkpoint_array_lineage_sha256s,
        pinned_checkpoint_sha256=pinned_checkpoint_sha256,
        final_checkpoint=final_checkpoint,
        query_source=query_source,
        one_source_proposal=one_source_proposal,
        cell_off_proposal=cell_off_proposal,
        schema_off_proposal=schema_off_proposal,
        transfer_proposal=transfer_proposal,
        proposal_event=proposal_event,
        transfer_authority=transfer_authority,
        transfer_catalogue=transfer_catalogue,
        transfer_work_snapshot=transfer_work_snapshot,
        transfer_selections=transfer_selections,
        transfer_executions=transfer_executions,
        transfer_events=transfer_events,
        oracle_opening=oracle_opening,
        transfer_consequences=transfer_consequences,
        consequence_events=consequence_events,
        work=work,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate(expected_result_sha256=result.sha256)
    return result


def _run_cohort_v1(
    spec: C2TokenEditSchemaTransferSpecV1,
    *,
    ledger: _RunLedgerV1,
) -> C2TokenEditSchemaTransferResultV1:
    if type(spec) is not C2TokenEditSchemaTransferSpecV1:
        raise TypeError("C2 run requires one exact transfer specification")
    spec.authenticate()
    if _current_common_file_sha256() != FROZEN_COMMON_FILE_SHA256:
        raise ContinualRustTokenEditSchemaTransferCohortError(
            "the frozen common learner file changed"
        )
    ledger.mark("SPEC_AUTHENTICATED", spec=spec)

    training_prepared = intervention.prepare_rust_action_intervention_v1(TRAINING_CANDIDATE_SOURCES)
    ledger.mark(
        "TRAINING_AUTHORITY_PREPARED",
        training_authority=training_prepared.authority,
        training_catalogue=training_prepared.catalogue,
    )
    training_selections: list[intervention.RustActionInterventionSelectionV1] = []
    training_executions: list[rust_executor.RustSelectedActionExecutionV2] = []
    training_consequences: list[continual.TokenEditConsequenceV1] = []
    training_events: list[C2ExecutionEventV1] = []
    episodes: list[continual.FailedSuccessfulTokenEditEpisodeV1] = []

    checkpoint = continual.start_continual_token_edit_schema_v1()
    checkpoint_roots = [checkpoint.checkpoint_sha256]
    checkpoint_array_roots = [checkpoint.checkpoint_array_sha256]
    query_source = continual.project_token_source_v1(TRANSFER_SOURCE)
    one_source_proposal: continual.TokenEditProposalV1 | None = None

    for episode_index, material in enumerate(spec.episodes):
        episode_frames: list[continual.TokenSourceFrameV1] = []
        episode_consequences: list[continual.TokenEditConsequenceV1] = []
        for attempt in _TRAINING_ATTEMPTS[episode_index * 2 : episode_index * 2 + 2]:
            frame = continual.project_token_source_v1(attempt.source)
            intervention_sha256 = _training_intervention_sha256(
                spec=spec,
                attempt=attempt,
                material=material,
                frame=frame,
            )
            selection = training_prepared.issue(
                attempt.candidate.sha256,
                intervention_sha256,
            )
            training_selections.append(selection)
            ledger.mark(
                f"TRAINING_SELECTION_{attempt.ordinal}_ISSUED",
                training_selections=tuple(training_selections),
                training_executions=tuple(training_executions),
            )
            execution = training_prepared.execute_selected(selection, material.batch)
            training_executions.append(execution)
            ledger.mark(
                f"TRAINING_EXECUTION_{attempt.ordinal}_RETURNED",
                training_executions=tuple(training_executions),
            )
            execution.authenticate_structure()
            expected_outputs = (
                material.batch if attempt.phase == FAILED else material.expected_outputs
            )
            if execution.outputs != expected_outputs:
                raise ContinualRustTokenEditSchemaTransferCohortError(
                    f"episode {material.label} {attempt.phase} output changed"
                )
            consequence = _make_consequence(
                execution=execution,
                frame=frame,
                test_manifest_sha256=material.test_manifest_sha256,
                expected_outputs=material.expected_outputs,
            )
            if consequence.public_test_cell != int(attempt.phase == SUCCESSFUL):
                raise ContinualRustTokenEditSchemaTransferCohortError(
                    f"episode {material.label} {attempt.phase} consequence changed"
                )
            event = _make_execution_event(
                ordinal=attempt.ordinal,
                domain=TRAINING,
                phase=attempt.phase,
                previous_event_sha256=(
                    _event_chain_genesis_sha256(spec.sha256)
                    if not training_events
                    else training_events[-1].sha256
                ),
                candidate_sha256=attempt.candidate.sha256,
                authority=training_prepared.authority,
                catalogue=training_prepared.catalogue,
                selection=selection,
                execution=execution,
                frame=frame,
            )
            training_consequences.append(consequence)
            training_events.append(event)
            episode_frames.append(frame)
            episode_consequences.append(consequence)
            ledger.mark(
                f"TRAINING_EXECUTION_{attempt.ordinal}_RETAINED",
                training_executions=tuple(training_executions),
                training_consequences=tuple(training_consequences),
                training_events=tuple(training_events),
            )

        episode = continual.seal_failed_successful_token_edit_episode_v1(
            repository_sha256=material.repository_sha256,
            task_sha256=material.task_sha256,
            task_instance_sha256=material.task_instance_sha256,
            task_source_sha256=material.task_source_sha256,
            test_source_sha256=material.test_manifest_sha256,
            failed_source=episode_frames[0],
            failed_consequence=episode_consequences[0],
            successful_source=episode_frames[1],
            successful_consequence=episode_consequences[1],
        )
        parent_root = checkpoint.checkpoint_sha256
        parent_array_root = checkpoint.checkpoint_array_sha256
        successor = continual.advance_continual_token_edit_schema_v1(checkpoint, episode)
        checkpoint.authenticate(expected_checkpoint_sha256=parent_root)
        if checkpoint.checkpoint_array_sha256 != parent_array_root:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 schema advance mutated its immutable parent"
            )
        checkpoint = successor
        episodes.append(episode)
        checkpoint_roots.append(checkpoint.checkpoint_sha256)
        checkpoint_array_roots.append(checkpoint.checkpoint_array_sha256)
        ledger.mark(
            f"CHECKPOINT_AFTER_{material.label}_RETAINED",
            episodes=tuple(episodes),
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            checkpoint_array_sha256=checkpoint.checkpoint_array_sha256,
        )
        if episode_index == 0:
            one_source_proposal = checkpoint.propose(query_source)
            if (
                one_source_proposal.status != continual.STOP_INSUFFICIENT_INDEPENDENT_SOURCES
                or one_source_proposal.edited_source is not None
            ):
                raise ContinualRustTokenEditSchemaTransferCohortError(
                    "one-source C2 checkpoint did not physically abstain"
                )

    training_work_snapshot = _issue_authority_work_snapshot_v1(
        prepared=training_prepared,
        domain=TRAINING,
        previous_event_sha256=training_events[-1].sha256,
        selections=tuple(training_selections),
        executions=tuple(training_executions),
        events=tuple(training_events),
    )
    ledger.mark(
        "TRAINING_AUTHORITY_WORK_SNAPSHOT_RETAINED",
        training_work_snapshot=training_work_snapshot,
    )

    if one_source_proposal is None:
        raise RuntimeError("C2 curriculum did not produce its one-source control")
    pinned_checkpoint_sha256 = checkpoint.checkpoint_sha256
    checkpoint.authenticate(expected_checkpoint_sha256=pinned_checkpoint_sha256)
    ledger.mark(
        "FINAL_CHECKPOINT_PINNED_BEFORE_QUERY_PROPOSAL",
        pinned_checkpoint_sha256=pinned_checkpoint_sha256,
        one_source_proposal=one_source_proposal,
    )

    cell_off_proposal = checkpoint.propose(query_source, cell_gate=False)
    schema_off_proposal = checkpoint.propose(query_source, schema_gate=False)
    transfer_proposal = checkpoint.propose(query_source)
    if (
        cell_off_proposal.status != continual.STOP_CELL_OFF
        or cell_off_proposal.edited_source is not None
        or schema_off_proposal.status != continual.STOP_SCHEMA_OFF
        or schema_off_proposal.edited_source is not None
        or transfer_proposal.status != continual.READY
        or transfer_proposal.selected_schema_index != FROZEN_SCHEMA_INDEX
        or transfer_proposal.selected_schema_sha256 != FROZEN_SCHEMA_SHA256
        or transfer_proposal.edited_source is None
        or (
            transfer_proposal.anchor_token_index,
            transfer_proposal.target_token_index,
            transfer_proposal.donor_token_index,
        )
        != (75, 76, 78)
    ):
        raise ContinualRustTokenEditSchemaTransferCohortError(
            "C2 controls or transfer proposal changed"
        )
    edited_frame = continual.project_token_source_v1(transfer_proposal.edited_source)
    changed_indices = tuple(
        index
        for index, (before, after) in enumerate(
            zip(query_source.tokens, edited_frame.tokens, strict=True)
        )
        if before != after
    )
    if (
        changed_indices != (transfer_proposal.target_token_index,)
        or query_source.tokens[transfer_proposal.donor_token_index] != "threshold"
        or edited_frame.tokens[transfer_proposal.target_token_index] != "threshold"
    ):
        raise ContinualRustTokenEditSchemaTransferCohortError(
            "C2 proposal was not the exact one-token donor copy"
        )
    ledger.mark(
        "TRANSFER_PROPOSAL_SEALED",
        cell_off_proposal=cell_off_proposal,
        schema_off_proposal=schema_off_proposal,
        transfer_proposal=transfer_proposal,
    )
    proposal_event = _make_proposal_event(
        previous_event_sha256=training_work_snapshot.sha256,
        pinned_checkpoint_sha256=pinned_checkpoint_sha256,
        query_source=query_source,
        proposal=transfer_proposal,
    )
    ledger.mark("PROPOSAL_EVENT_RETAINED", proposal_event=proposal_event)

    transfer_candidate_sources = (
        (TRANSFER_FIRST_CANDIDATE, TRANSFER_SOURCE),
        (TRANSFER_SECOND_CANDIDATE, transfer_proposal.edited_source),
    )
    transfer_prepared = intervention.prepare_rust_action_intervention_v1(transfer_candidate_sources)
    ledger.mark(
        "TRANSFER_AUTHORITY_PREPARED_AFTER_PROPOSAL",
        transfer_authority=transfer_prepared.authority,
        transfer_catalogue=transfer_prepared.catalogue,
    )
    transfer_selections: list[intervention.RustActionInterventionSelectionV1] = []
    transfer_executions: list[rust_executor.RustSelectedActionExecutionV2] = []
    transfer_events: list[C2ExecutionEventV1] = []
    phases = (BASELINE, PROPOSAL)
    ordinals = (6, 7)
    for (candidate, source), phase, ordinal in zip(
        transfer_candidate_sources,
        phases,
        ordinals,
        strict=True,
    ):
        frame = continual.project_token_source_v1(source)
        intervention_sha256 = _transfer_intervention_sha256(
            spec=spec,
            proposal_sha256=transfer_proposal.sha256,
            ordinal=ordinal,
            phase=phase,
            candidate_sha256=candidate.sha256,
            frame=frame,
        )
        selection = transfer_prepared.issue(candidate.sha256, intervention_sha256)
        transfer_selections.append(selection)
        ledger.mark(
            f"TRANSFER_{phase}_SELECTION_ISSUED",
            transfer_selections=tuple(transfer_selections),
            transfer_executions=tuple(transfer_executions),
        )
        execution = transfer_prepared.execute_selected(selection, spec.transfer.batch)
        transfer_executions.append(execution)
        ledger.mark(
            f"TRANSFER_{phase}_EXECUTION_RETURNED",
            transfer_executions=tuple(transfer_executions),
        )
        execution.authenticate_structure()
        if source != execution.selected_source.source:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 transfer execution changed the selected source bytes"
            )
        if phase == PROPOSAL and source != transfer_proposal.edited_source:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 proposal execution did not use the exact proposal bytes"
            )
        event = _make_execution_event(
            ordinal=ordinal,
            domain=TRANSFER,
            phase=phase,
            previous_event_sha256=(
                proposal_event.sha256 if not transfer_events else transfer_events[-1].sha256
            ),
            candidate_sha256=candidate.sha256,
            authority=transfer_prepared.authority,
            catalogue=transfer_prepared.catalogue,
            selection=selection,
            execution=execution,
            frame=frame,
        )
        transfer_events.append(event)
        ledger.mark(
            f"TRANSFER_{phase}_EXECUTION_RETAINED",
            transfer_executions=tuple(transfer_executions),
            transfer_events=tuple(transfer_events),
        )

    if len(transfer_executions) != 2:
        raise RuntimeError("C2 oracle cannot open before both transfer executions")
    transfer_work_snapshot = _issue_authority_work_snapshot_v1(
        prepared=transfer_prepared,
        domain=TRANSFER,
        previous_event_sha256=transfer_events[-1].sha256,
        selections=tuple(transfer_selections),
        executions=tuple(transfer_executions),
        events=tuple(transfer_events),
    )
    ledger.mark(
        "TRANSFER_AUTHORITY_WORK_SNAPSHOT_RETAINED",
        transfer_work_snapshot=transfer_work_snapshot,
    )
    open_oracle_once = _make_guarded_transfer_oracle_v1(
        proposal_event.sha256,
        lambda: tuple(transfer_events),
    )
    ledger.mark(
        "TRANSFER_ORACLE_OPEN_ATTEMPT",
        transfer_event_sha256s=tuple(event.sha256 for event in transfer_events),
    )
    expected_outputs = open_oracle_once()
    ledger.mark(
        "TRANSFER_ORACLE_OUTPUT_RETURNED_RAW",
        oracle_expected_outputs=expected_outputs,
    )
    expected_output_commitment = _digest(_ORACLE_OUTPUT_NAMESPACE, expected_outputs)
    ledger.mark(
        "TRANSFER_ORACLE_OUTPUT_COMMITMENT_RETAINED",
        oracle_expected_outputs=expected_outputs,
        oracle_expected_output_sha256=expected_output_commitment,
    )
    oracle_opening = _make_oracle_opening(
        previous_event_sha256=transfer_work_snapshot.sha256,
        proposal_event_sha256=proposal_event.sha256,
        transfer_work_snapshot_sha256=transfer_work_snapshot.sha256,
        transfer_execution_event_sha256s=tuple(event.sha256 for event in transfer_events),
        test_manifest_sha256=spec.transfer.test_manifest_sha256,
        expected_outputs=expected_outputs,
    )
    ledger.mark("TRANSFER_ORACLE_OPENED_ONCE", oracle_opening=oracle_opening)
    transfer_consequence_rows: list[continual.TokenEditConsequenceV1] = []
    consequence_events: list[C2ConsequenceEventV1] = []
    for index, (execution, (_candidate, source), phase, ordinal) in enumerate(
        zip(
            transfer_executions,
            transfer_candidate_sources,
            phases,
            (9, 10),
            strict=True,
        )
    ):
        consequence = _make_consequence(
            execution=execution,
            frame=continual.project_token_source_v1(source),
            test_manifest_sha256=spec.transfer.test_manifest_sha256,
            expected_outputs=expected_outputs,
        )
        event = _make_consequence_event(
            ordinal=ordinal,
            phase=phase,
            previous_event_sha256=(
                oracle_opening.sha256 if index == 0 else consequence_events[-1].sha256
            ),
            oracle_opening_sha256=oracle_opening.sha256,
            execution_event_sha256=transfer_events[index].sha256,
            consequence=consequence,
        )
        transfer_consequence_rows.append(consequence)
        consequence_events.append(event)
        ledger.mark(
            f"TRANSFER_{phase}_CONSEQUENCE_EVENT_RETAINED",
            transfer_consequences=tuple(transfer_consequence_rows),
            consequence_events=tuple(consequence_events),
        )
    transfer_consequences = tuple(transfer_consequence_rows)
    ledger.mark(
        "TRANSFER_CONSEQUENCES_RETAINED",
        transfer_consequences=transfer_consequences,
    )
    checkpoint.authenticate(expected_checkpoint_sha256=pinned_checkpoint_sha256)

    result = _seal_result(
        spec=spec,
        training_authority=training_prepared.authority,
        training_catalogue=training_prepared.catalogue,
        training_work_snapshot=training_work_snapshot,
        training_selections=tuple(training_selections),
        training_executions=tuple(training_executions),
        training_consequences=tuple(training_consequences),
        training_events=tuple(training_events),
        episodes=tuple(episodes),
        checkpoint_lineage_sha256s=tuple(checkpoint_roots),
        checkpoint_array_lineage_sha256s=tuple(checkpoint_array_roots),
        pinned_checkpoint_sha256=pinned_checkpoint_sha256,
        final_checkpoint=checkpoint,
        query_source=query_source,
        one_source_proposal=one_source_proposal,
        cell_off_proposal=cell_off_proposal,
        schema_off_proposal=schema_off_proposal,
        transfer_proposal=transfer_proposal,
        proposal_event=proposal_event,
        transfer_authority=transfer_prepared.authority,
        transfer_catalogue=transfer_prepared.catalogue,
        transfer_work_snapshot=transfer_work_snapshot,
        transfer_selections=tuple(transfer_selections),
        transfer_executions=tuple(transfer_executions),
        transfer_events=tuple(transfer_events),
        oracle_opening=oracle_opening,
        transfer_consequences=transfer_consequences,
        consequence_events=tuple(consequence_events),
    )
    ledger.mark("AUTHENTICATED_TERMINAL_RESULT", result=result)
    return result


@dataclass(frozen=True, slots=True)
class C2ExecutionAbortV1:
    schema: str
    claim_boundary: str
    status: str
    spec_sha256: str
    frozen_common_file_sha256: str
    frozen_learner_config_sha256: str
    failure_kind: str
    exception_type: str
    exception_message: str
    exception_message_sha256: str
    partial_transcript: dict[str, object]
    successful_v2_failure_receipt_retained: bool
    receipt_limit: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            self.schema != ABORT_SCHEMA
            or self.claim_boundary != CLAIM_BOUNDARY
            or self.status != STATUS_EXECUTION_ABORT
            or self.spec_sha256 != DEFAULT_PUBLIC_SPEC.sha256
            or self.frozen_common_file_sha256 != FROZEN_COMMON_FILE_SHA256
            or self.frozen_learner_config_sha256 != FROZEN_LEARNER_CONFIG_SHA256
            or self.failure_kind not in {"RUST_EXECUTION_ABORT", "PROTOCOL_ABORT"}
            or not self.exception_type
            or self.exception_message_sha256
            != sha256(self.exception_message.encode("utf-8")).hexdigest()
            or type(self.partial_transcript) is not dict
            or self.successful_v2_failure_receipt_retained
            or not self.receipt_limit
            or self.sha256 != _digest(_ABORT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 execution abort envelope authentication failed"
            )


def _make_abort(
    *,
    spec: C2TokenEditSchemaTransferSpecV1,
    ledger: _RunLedgerV1,
    error: BaseException,
) -> C2ExecutionAbortV1:
    failure_kind = (
        "RUST_EXECUTION_ABORT"
        if isinstance(error, rust_executor.RustSelectedActionExecutionError)
        else "PROTOCOL_ABORT"
    )
    message = str(error)
    record = {
        "schema": ABORT_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "status": STATUS_EXECUTION_ABORT,
        "spec_sha256": spec.sha256,
        "frozen_common_file_sha256": FROZEN_COMMON_FILE_SHA256,
        "frozen_learner_config_sha256": FROZEN_LEARNER_CONFIG_SHA256,
        "failure_kind": failure_kind,
        "exception_type": type(error).__qualname__,
        "exception_message": message,
        "exception_message_sha256": sha256(message.encode("utf-8")).hexdigest(),
        "partial_transcript": ledger.snapshot(),
        "successful_v2_failure_receipt_retained": False,
        "receipt_limit": (
            "this envelope commits the serialized retained ledger state and exception; "
            "the existing V2 executor raises before constructing a typed compiler-or-runtime "
            "failure receipt, so no unavailable low-level receipt is claimed"
        ),
    }
    partial = C2ExecutionAbortV1(**record, sha256="")
    result = replace(
        partial,
        sha256=_digest(_ABORT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _result_document(result: C2TokenEditSchemaTransferResultV1) -> dict[str, object]:
    result.authenticate(expected_result_sha256=result.sha256)
    body = {
        "schema": RESULT_DOCUMENT_SCHEMA,
        "terminal_status": result.status,
        "result_sha256": result.sha256,
        "authenticated_result_record": result.record(),
        "full_authenticated_transcript": result.transcript_record(),
    }
    return {
        **body,
        "document_sha256": _digest(_DOCUMENT_NAMESPACE, body),
    }


def _abort_document(abort: C2ExecutionAbortV1) -> dict[str, object]:
    abort.authenticate()
    body = {
        "schema": RESULT_DOCUMENT_SCHEMA,
        "terminal_status": STATUS_EXECUTION_ABORT,
        "abort_sha256": abort.sha256,
        "authenticated_abort": abort.record(),
    }
    return {
        **body,
        "document_sha256": _digest(_DOCUMENT_NAMESPACE, body),
    }


def _production_evidence_paths_v1() -> tuple[Path, Path]:
    """Derive the fixed production leaves from the resolved live module path."""

    module_path = Path(__file__).resolve(strict=True)
    result_path = module_path.parent / f"{module_path.stem}.result.json"
    reservation_path = module_path.parent / f"{result_path.name}.reservation.json"
    return result_path, reservation_path


DEFAULT_RESULT_PATH, DEFAULT_RESERVATION_PATH = _production_evidence_paths_v1()
RESERVATION_SCHEMA = "humanbrain.continual-rust-token-edit-schema-transfer-cohort-reservation.v1"
_RESERVATION_NAMESPACE = b"continual-rust-token-edit-schema-transfer-reservation-v1"


@dataclass(frozen=True, slots=True)
class C2RunReservationV1:
    schema: str
    spec_sha256: str
    destination: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            "schema": self.schema,
            "spec_sha256": self.spec_sha256,
            "destination": self.destination,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self, *, expected_destination: Path) -> None:
        destination = Path(self.destination)
        if (
            self.schema != RESERVATION_SCHEMA
            or self.spec_sha256 != DEFAULT_PUBLIC_SPEC.sha256
            or destination != expected_destination
            or self.sha256 != _digest(_RESERVATION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 run reservation authentication failed"
            )


def _make_reservation(
    spec: C2TokenEditSchemaTransferSpecV1,
    *,
    destination: Path,
) -> C2RunReservationV1:
    spec.authenticate()
    partial = C2RunReservationV1(
        schema=RESERVATION_SCHEMA,
        spec_sha256=spec.sha256,
        destination=str(destination),
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_RESERVATION_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate(expected_destination=destination)
    return result


def _authenticate_evidence_leaf_v1(
    *,
    leaf: Path,
    expected_leaf: Path,
    label: str,
) -> Path:
    leaf = Path(leaf)
    expected_leaf = Path(expected_leaf)
    if (
        not leaf.is_absolute()
        or not expected_leaf.is_absolute()
        or leaf.name != expected_leaf.name
        or leaf.parent.resolve(strict=True) != expected_leaf.parent.resolve(strict=True)
    ):
        raise ValueError(f"C2 {label} must be the exact anchored evidence leaf")
    if os.path.lexists(leaf) or leaf.is_symlink():
        raise FileExistsError(leaf)
    return leaf


def _reserve_destination(
    spec: C2TokenEditSchemaTransferSpecV1,
    *,
    destination: Path,
    reservation_path: Path,
) -> C2RunReservationV1:
    destination = _authenticate_evidence_leaf_v1(
        leaf=destination,
        expected_leaf=destination,
        label="result destination",
    )
    reservation_path = _authenticate_evidence_leaf_v1(
        leaf=reservation_path,
        expected_leaf=destination.with_name(f"{destination.name}.reservation.json"),
        label="reservation destination",
    )
    reservation = _make_reservation(spec, destination=destination)
    payload = _canonical_json(reservation.record()) + b"\n"
    with reservation_path.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("C2 reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = reservation_path.read_bytes()
    if retained != payload:
        raise OSError("C2 reservation readback changed")
    decoded = json.loads(retained.decode("ascii"))
    recovered = C2RunReservationV1(**decoded)
    recovered.authenticate(expected_destination=destination)
    if recovered != reservation:
        raise ContinualRustTokenEditSchemaTransferCohortError("C2 retained reservation changed")
    return reservation


def _verify_document_bytes(
    *,
    document: dict[str, object],
    payload: bytes,
    authenticate_terminal: Callable[[], None],
    expected_document_sha256: str | None = None,
    expected_result_sha256: str | None = None,
) -> None:
    if expected_document_sha256 is not None and not _valid_sha256(expected_document_sha256):
        raise TypeError("C2 expected document root must be one exact SHA-256")
    if expected_result_sha256 is not None and not _valid_sha256(expected_result_sha256):
        raise TypeError("C2 expected result root must be one exact SHA-256")
    if payload != _canonical_json(document) + b"\n":
        raise OSError("C2 terminal document bytes changed")
    decoded = json.loads(payload.decode("ascii"))
    if _canonical_json(decoded) + b"\n" != payload:
        raise ContinualRustTokenEditSchemaTransferCohortError(
            "C2 terminal document readback changed"
        )
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if (
        decoded.get("schema") != RESULT_DOCUMENT_SCHEMA
        or document_sha256 != _digest(_DOCUMENT_NAMESPACE, body)
        or (expected_document_sha256 is not None and document_sha256 != expected_document_sha256)
    ):
        raise ContinualRustTokenEditSchemaTransferCohortError("C2 terminal document root changed")
    terminal_status = decoded.get("terminal_status")
    if terminal_status == STATUS_EXECUTION_ABORT:
        if expected_result_sha256 is not None:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 abort document cannot satisfy a result-root pin"
            )
        abort_record = decoded.get("authenticated_abort")
        if type(abort_record) is not dict:
            raise TypeError("C2 abort document must retain one exact abort record")
        abort = C2ExecutionAbortV1(**abort_record)
        abort.authenticate()
        if decoded.get("abort_sha256") != abort.sha256:
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 abort document linkage changed"
            )
    else:
        result_sha256 = decoded.get("result_sha256")
        transcript = decoded.get("full_authenticated_transcript")
        compact_record = decoded.get("authenticated_result_record")
        if (
            terminal_status
            not in {
                STATUS_PASS,
                STATUS_BEHAVIOR_FAIL,
                STATUS_BASELINE_NONDISCRIMINATING,
            }
            or not _valid_sha256(result_sha256)
            or type(transcript) is not dict
            or type(compact_record) is not dict
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 result document linkage changed"
            )
        compact_body = dict(compact_record)
        compact_sha256 = compact_body.pop("sha256", None)
        if (
            transcript.get("sha256") != result_sha256
            or transcript.get("status") != terminal_status
            or compact_sha256 != result_sha256
            or compact_record.get("status") != terminal_status
            or _digest(_RESULT_NAMESPACE, compact_body) != result_sha256
            or (expected_result_sha256 is not None and result_sha256 != expected_result_sha256)
        ):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 persisted result record or transcript linkage changed"
            )
    authenticate_terminal()


def _publish_terminal_document(
    *,
    destination: Path,
    expected_destination: Path,
    document: dict[str, object],
    authenticate_terminal: Callable[[], None],
    expected_document_sha256: str | None = None,
    expected_result_sha256: str | None = None,
) -> None:
    destination = _authenticate_evidence_leaf_v1(
        leaf=destination,
        expected_leaf=expected_destination,
        label="terminal destination",
    )
    payload = _canonical_json(document) + b"\n"
    pending_path: Path | None = None
    published_and_verified = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".pending",
            delete=False,
        ) as pending:
            pending_path = Path(pending.name)
            if pending.write(payload) != len(payload):
                raise OSError("C2 pending terminal write was incomplete")
            pending.flush()
            os.fsync(pending.fileno())
            pending.seek(0)
            pending_readback = pending.read()
        _verify_document_bytes(
            document=document,
            payload=pending_readback,
            authenticate_terminal=authenticate_terminal,
            expected_document_sha256=expected_document_sha256,
            expected_result_sha256=expected_result_sha256,
        )
        try:
            os.link(pending_path, destination)
        except OSError:
            if os.name != "nt" or os.path.lexists(destination) or destination.is_symlink():
                raise
            os.rename(pending_path, destination)
            pending_path = None
        published = destination.read_bytes()
        if published != payload:
            raise OSError("C2 published terminal bytes changed")
        _verify_document_bytes(
            document=document,
            payload=published,
            authenticate_terminal=authenticate_terminal,
            expected_document_sha256=expected_document_sha256,
            expected_result_sha256=expected_result_sha256,
        )
        published_and_verified = True
    finally:
        if published_and_verified and pending_path is not None:
            with suppress(OSError):
                pending_path.unlink()


def _run_and_write_reserved_v1(
    *,
    destination: Path,
    reservation_path: Path,
    spec: C2TokenEditSchemaTransferSpecV1 = DEFAULT_PUBLIC_SPEC,
    expected_result_sha256: str | None = None,
    expected_document_sha256: str | None = None,
) -> C2TokenEditSchemaTransferResultV1:
    if type(spec) is not C2TokenEditSchemaTransferSpecV1:
        raise TypeError("C2 run-and-write requires one exact transfer specification")
    if expected_result_sha256 is not None and not _valid_sha256(expected_result_sha256):
        raise TypeError("C2 expected result root must be one exact SHA-256")
    if expected_document_sha256 is not None and not _valid_sha256(expected_document_sha256):
        raise TypeError("C2 expected document root must be one exact SHA-256")
    spec.authenticate()
    reservation = _reserve_destination(
        spec,
        destination=destination,
        reservation_path=reservation_path,
    )
    ledger = _RunLedgerV1()
    ledger.mark("PERSISTENT_RESERVATION_RETAINED", reservation=reservation)
    try:
        result = _run_cohort_v1(spec, ledger=ledger)
        result_pin = expected_result_sha256 or result.sha256
        result.authenticate(expected_result_sha256=result_pin)
        document = _result_document(result)
        document_sha256 = document.get("document_sha256")
        if not _valid_sha256(document_sha256):
            raise ContinualRustTokenEditSchemaTransferCohortError(
                "C2 result document did not retain one exact root"
            )
        document_pin = expected_document_sha256 or document_sha256
        _publish_terminal_document(
            destination=destination,
            expected_destination=destination,
            document=document,
            authenticate_terminal=lambda: result.authenticate(expected_result_sha256=result_pin),
            expected_document_sha256=document_pin,
            expected_result_sha256=result_pin,
        )
    except BaseException as error:
        try:
            abort = _make_abort(spec=spec, ledger=ledger, error=error)
            abort_document = _abort_document(abort)
            abort_document_sha256 = abort_document.get("document_sha256")
            if not _valid_sha256(abort_document_sha256):
                raise ContinualRustTokenEditSchemaTransferCohortError(
                    "C2 abort document did not retain one exact root"
                )
            _publish_terminal_document(
                destination=destination,
                expected_destination=destination,
                document=abort_document,
                authenticate_terminal=abort.authenticate,
                expected_document_sha256=abort_document_sha256,
            )
        except BaseException:
            pass
        raise
    return result


def _run_and_write_at_paths_for_test_v1(
    *,
    destination: Path,
    reservation_path: Path,
    spec: C2TokenEditSchemaTransferSpecV1 = DEFAULT_PUBLIC_SPEC,
    expected_result_sha256: str | None = None,
    expected_document_sha256: str | None = None,
) -> C2TokenEditSchemaTransferResultV1:
    """Private static-test seam; production callers cannot redirect evidence."""

    return _run_and_write_reserved_v1(
        destination=destination,
        reservation_path=reservation_path,
        spec=spec,
        expected_result_sha256=expected_result_sha256,
        expected_document_sha256=expected_document_sha256,
    )


def run_and_write_cohort_v1(
    spec: C2TokenEditSchemaTransferSpecV1 = DEFAULT_PUBLIC_SPEC,
    *,
    expected_result_sha256: str | None = None,
    expected_document_sha256: str | None = None,
) -> C2TokenEditSchemaTransferResultV1:
    """Spend once at the fixed path, optionally checking external root pins.

    Git custody or another external attestation must supply the optional pins;
    roots computed inside this call establish self-consistency rather than origin.
    """

    destination, reservation_path = _production_evidence_paths_v1()
    return _run_and_write_reserved_v1(
        destination=destination,
        reservation_path=reservation_path,
        spec=spec,
        expected_result_sha256=expected_result_sha256,
        expected_document_sha256=expected_document_sha256,
    )


__all__ = [
    "ABORT_SCHEMA",
    "A_BATCH",
    "A_EXPECTED_OUTPUTS",
    "A_FAILED_CANDIDATE",
    "A_FAILED_SOURCE",
    "A_SUCCESSFUL_CANDIDATE",
    "A_SUCCESSFUL_SOURCE",
    "B_BATCH",
    "B_EXPECTED_OUTPUTS",
    "B_FAILED_CANDIDATE",
    "B_FAILED_SOURCE",
    "B_SUCCESSFUL_CANDIDATE",
    "B_SUCCESSFUL_SOURCE",
    "CLAIM_BOUNDARY",
    "DEFAULT_PUBLIC_SPEC",
    "DEFAULT_RESULT_PATH",
    "EXCLUDED_C1_SOURCE_SHA256S",
    "FROZEN_COMMON_FILE_SHA256",
    "FROZEN_LEARNER_CONFIG_SHA256",
    "FROZEN_MECHANISM_COMMIT",
    "FROZEN_SCHEMA_INDEX",
    "FROZEN_SCHEMA_SHA256",
    "RESULT_DOCUMENT_SCHEMA",
    "SCHEMA",
    "STATUS_BASELINE_NONDISCRIMINATING",
    "STATUS_BEHAVIOR_FAIL",
    "STATUS_EXECUTION_ABORT",
    "STATUS_PASS",
    "TRAINING_CANDIDATE_SOURCES",
    "TRANSFER_BATCH",
    "TRANSFER_FIRST_CANDIDATE",
    "TRANSFER_SECOND_CANDIDATE",
    "TRANSFER_SOURCE",
    "C2AuthorityWorkSnapshotV1",
    "C2EpisodeMaterialV1",
    "C2ExecutionAbortV1",
    "C2ExecutionEventV1",
    "C2TokenEditSchemaTransferResultV1",
    "C2TokenEditSchemaTransferSpecV1",
    "C2TransferCohortWorkV1",
    "C2TransferMaterialV1",
    "C2TransferOracleOpeningV1",
    "ContinualRustTokenEditSchemaTransferCohortError",
    "make_c2_token_edit_schema_transfer_spec_v1",
    "run_and_write_cohort_v1",
]


if __name__ == "__main__":
    run_and_write_cohort_v1()
