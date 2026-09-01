"""Trusted/public C1 cohort for structural single-token edit proposals.

Two independent, candidate-provided Rust episodes each execute a failed source
and its successful one-token revision exactly once.  The generic continual
token-edit learner receives only authenticated source/consequence episodes.  It
selects and consolidates one relation from its fixed lexical hypothesis bank,
performs its own support update and neuronal race, and proposes (without
executing) the corresponding edit for a structurally fresh source.

This cohort establishes only a C1 structural proposal under the retained
public transcript.  It does not establish semantic understanding, transfer,
correctness of the unexecuted proposal, source synthesis, protected freshness,
hostile sandboxing, SWE performance, or general intelligence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256

from experiments.r3_pilot import common_continual_token_edit_schema_v1 as continual
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from experiments.r3_pilot import common_rust_action_intervention_v1 as intervention
from humanbrain import rust_selected_action_executor as rust_executor

SCHEMA = "humanbrain.continual-rust-token-edit-schema-cohort.v1"
CLAIM_BOUNDARY = (
    "c1-structural-single-token-copy-replacement-schema-selected-from-a-fixed-"
    "lexical-hypothesis-bank-by-two-independent-failed-to-successful-public-rust-"
    "episodes-with-exact-v2-custody-and-an-unexecuted-fresh-source-proposal-not-"
    "semantic-transfer-correctness-source-synthesis-protected-freshness-hostile-"
    "sandboxing-swe-or-intelligence"
)
STATUS_C1_STRUCTURAL_PROPOSAL = "C1_STRUCTURAL_TOKEN_EDIT_PROPOSAL_V1"

FAILED = "FAILED"
SUCCESSFUL = "SUCCESSFUL"

_SPEC_NAMESPACE = b"continual-rust-token-edit-schema-cohort-spec-v1"
_PUBLIC_NONCE_NAMESPACE = b"continual-rust-token-edit-schema-public-nonce-v1"
_MATERIAL_NAMESPACE = b"continual-rust-token-edit-schema-material-v1"
_OUTPUT_NAMESPACE = b"continual-rust-token-edit-schema-output-v1"
_INTERVENTION_NAMESPACE = b"continual-rust-token-edit-schema-intervention-v1"
_EVIDENCE_NAMESPACE = b"continual-rust-token-edit-schema-evidence-v1"
_WORK_NAMESPACE = b"continual-rust-token-edit-schema-work-v1"
_RESULT_NAMESPACE = b"continual-rust-token-edit-schema-result-v1"

_SHA256_HEX = frozenset("0123456789abcdef")

A_BATCH = ((9, -2, 7, 14), (0, 8, 3))
A_EXPECTED_OUTPUTS = ((7, -2, 7, 7), (0, 7, 3))
B_BATCH = ((-8, -3, 2, 9), (0, -4, -2))
B_EXPECTED_OUTPUTS = ((-3, -3, 2, 9), (0, -3, -2))

A_FAILED_SOURCE = (
    "pub fn solve(readings: &[i64]) -> Vec<i64> {\n"
    "    let ceiling = 7_i64;\n"
    "    readings.iter().copied().map(|reading| if reading > reading { ceiling } else { reading }).collect()\n"  # noqa: E501
    "}\n"
)

A_SUCCESSFUL_SOURCE = (
    "pub fn solve(readings: &[i64]) -> Vec<i64> {\n"
    "    let ceiling = 7_i64;\n"
    "    readings.iter().copied().map(|reading| if reading > ceiling { ceiling } else { reading }).collect()\n"  # noqa: E501
    "}\n"
)

B_FAILED_SOURCE = (
    "pub fn solve(samples: &[i64]) -> Vec<i64> {\n"
    "    let floor = -3_i64;\n"
    "    samples.iter().copied().map(|temperature| match temperature < temperature { true => floor, false => temperature }).collect()\n"  # noqa: E501
    "}\n"
)

B_SUCCESSFUL_SOURCE = (
    "pub fn solve(samples: &[i64]) -> Vec<i64> {\n"
    "    let floor = -3_i64;\n"
    "    samples.iter().copied().map(|temperature| match temperature < floor { true => floor, false => temperature }).collect()\n"  # noqa: E501
    "}\n"
)

FRESH_STRUCTURAL_SOURCE = """\
pub fn solve(measures: &[i64]) -> Vec<i64> {
    let boundary = 11_i64;
    let mut output = Vec::new();
    for measure in measures.iter().copied() {
        output.push(if measure >= measure { boundary } else { measure });
    }
    output
}
"""

EXPECTED_FRESH_EDITED_SOURCE = """\
pub fn solve(measures: &[i64]) -> Vec<i64> {
    let boundary = 11_i64;
    let mut output = Vec::new();
    for measure in measures.iter().copied() {
        output.push(if measure >= boundary { boundary } else { measure });
    }
    output
}
"""

A_FAILED_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_q7",
    "opaque_c1.slot_q7()",
)
A_SUCCESSFUL_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_z4",
    "opaque_c1.slot_z4()",
)
B_FAILED_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_m2",
    "opaque_c1.slot_m2()",
)
B_SUCCESSFUL_CANDIDATE = knowledge_action.project_knowledge_action_candidate_v1(
    "slot_v9",
    "opaque_c1.slot_v9()",
)

CANDIDATE_SOURCES = (
    (A_FAILED_CANDIDATE, A_FAILED_SOURCE),
    (A_SUCCESSFUL_CANDIDATE, A_SUCCESSFUL_SOURCE),
    (B_FAILED_CANDIDATE, B_FAILED_SOURCE),
    (B_SUCCESSFUL_CANDIDATE, B_SUCCESSFUL_SOURCE),
)


class ContinualRustTokenEditSchemaCohortError(ValueError):
    """The fixed C1 curriculum or its retained custody transcript changed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


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
        raise ContinualRustTokenEditSchemaCohortError(f"{label} case count changed")
    return result


def token_edit_cohort_output_sha256_v1(
    outputs: tuple[tuple[int, ...], ...],
) -> str:
    """Return the cohort commitment for one exact public output batch."""

    return _digest(
        _OUTPUT_NAMESPACE,
        _validate_outputs(outputs, label="token-edit cohort outputs"),
    )


@dataclass(frozen=True, slots=True)
class TokenEditEpisodeInputV1:
    label: str
    repository_sha256: str
    task_sha256: str
    task_instance_sha256: str
    task_source_sha256: str
    test_source_sha256: str
    batch: tuple[tuple[int, ...], ...]
    expected_outputs: tuple[tuple[int, ...], ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "label": self.label,
            "repository_sha256": self.repository_sha256,
            "task_sha256": self.task_sha256,
            "task_instance_sha256": self.task_instance_sha256,
            "task_source_sha256": self.task_source_sha256,
            "test_source_sha256": self.test_source_sha256,
            "batch": self.batch,
            "expected_outputs": self.expected_outputs,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        batch = _validate_outputs(self.batch, label=f"episode {self.label} batch")
        _validate_outputs(
            self.expected_outputs,
            label=f"episode {self.label} expected outputs",
            expected_cases=len(batch),
        )
        roots = (
            self.repository_sha256,
            self.task_sha256,
            self.task_instance_sha256,
            self.task_source_sha256,
            self.test_source_sha256,
        )
        if (
            self.label not in ("A", "B")
            or any(not _valid_sha256(root) for root in roots)
            or len(set(roots)) != len(roots)
            or self.sha256 != _digest(_MATERIAL_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "token-edit episode input authentication failed"
            )


def _make_episode_input(
    *,
    cohort_nonce_sha256: str,
    label: str,
    batch: tuple[tuple[int, ...], ...],
    expected_outputs: tuple[tuple[int, ...], ...],
) -> TokenEditEpisodeInputV1:
    roots = {
        name: _digest(
            _MATERIAL_NAMESPACE,
            {
                "cohort_nonce_sha256": cohort_nonce_sha256,
                "episode_label": label,
                "material_kind": name,
            },
        )
        for name in (
            "repository_sha256",
            "task_sha256",
            "task_instance_sha256",
            "task_source_sha256",
            "test_source_sha256",
        )
    }
    record = {
        "label": label,
        **roots,
        "batch": batch,
        "expected_outputs": expected_outputs,
    }
    result = TokenEditEpisodeInputV1(
        **record,
        sha256=_digest(_MATERIAL_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C1StructuralTokenEditCohortSpecV1:
    schema: str
    cohort_nonce_sha256: str
    learner_config_sha256: str
    episodes: tuple[TokenEditEpisodeInputV1, TokenEditEpisodeInputV1]
    candidate_sha256s: tuple[str, str, str, str]
    training_source_sha256s: tuple[str, str, str, str]
    fresh_source_sha256: str
    expected_fresh_edited_source_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "cohort_nonce_sha256": self.cohort_nonce_sha256,
            "learner_config_sha256": self.learner_config_sha256,
            "episodes": tuple(episode.record() for episode in self.episodes),
            "candidate_sha256s": self.candidate_sha256s,
            "training_source_sha256s": self.training_source_sha256s,
            "fresh_source_sha256": self.fresh_source_sha256,
            "expected_fresh_edited_source_sha256": (self.expected_fresh_edited_source_sha256),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            self.schema != SCHEMA
            or not _valid_sha256(self.cohort_nonce_sha256)
            or self.learner_config_sha256 != continual.CONFIG_SHA256
            or type(self.episodes) is not tuple
            or len(self.episodes) != 2
            or any(type(item) is not TokenEditEpisodeInputV1 for item in self.episodes)
            or self.fresh_source_sha256
            != sha256(FRESH_STRUCTURAL_SOURCE.encode("ascii")).hexdigest()
            or self.expected_fresh_edited_source_sha256
            != sha256(EXPECTED_FRESH_EDITED_SOURCE.encode("ascii")).hexdigest()
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 token-edit cohort specification changed"
            )
        for episode in self.episodes:
            episode.authenticate()
        expected_episodes = (
            _make_episode_input(
                cohort_nonce_sha256=self.cohort_nonce_sha256,
                label="A",
                batch=A_BATCH,
                expected_outputs=A_EXPECTED_OUTPUTS,
            ),
            _make_episode_input(
                cohort_nonce_sha256=self.cohort_nonce_sha256,
                label="B",
                batch=B_BATCH,
                expected_outputs=B_EXPECTED_OUTPUTS,
            ),
        )
        material_roots = tuple(
            root
            for episode in self.episodes
            for root in (
                episode.repository_sha256,
                episode.task_sha256,
                episode.task_instance_sha256,
                episode.task_source_sha256,
                episode.test_source_sha256,
            )
        )
        expected_candidates = tuple(candidate.sha256 for candidate, _source in CANDIDATE_SOURCES)
        expected_source_roots = tuple(
            sha256(source.encode("ascii")).hexdigest() for _candidate, source in CANDIDATE_SOURCES
        )
        if (
            self.episodes != expected_episodes
            or len(set(material_roots)) != 10
            or self.candidate_sha256s != expected_candidates
            or len(set(self.candidate_sha256s)) != 4
            or self.training_source_sha256s != expected_source_roots
            or len(set(self.training_source_sha256s)) != 4
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 token-edit cohort specification authentication failed"
            )


def make_c1_structural_token_edit_cohort_spec_v1(
    *,
    cohort_nonce_sha256: str,
) -> C1StructuralTokenEditCohortSpecV1:
    """Seal the fixed public C1 curriculum under a caller-provided nonce."""

    if not _valid_sha256(cohort_nonce_sha256):
        raise TypeError("cohort_nonce_sha256 must be one lowercase SHA-256 root")
    episodes = (
        _make_episode_input(
            cohort_nonce_sha256=cohort_nonce_sha256,
            label="A",
            batch=A_BATCH,
            expected_outputs=A_EXPECTED_OUTPUTS,
        ),
        _make_episode_input(
            cohort_nonce_sha256=cohort_nonce_sha256,
            label="B",
            batch=B_BATCH,
            expected_outputs=B_EXPECTED_OUTPUTS,
        ),
    )
    record = {
        "schema": SCHEMA,
        "cohort_nonce_sha256": cohort_nonce_sha256,
        "learner_config_sha256": continual.CONFIG_SHA256,
        "episodes": tuple(episode.record() for episode in episodes),
        "candidate_sha256s": tuple(candidate.sha256 for candidate, _source in CANDIDATE_SOURCES),
        "training_source_sha256s": tuple(
            sha256(source.encode("ascii")).hexdigest() for _candidate, source in CANDIDATE_SOURCES
        ),
        "fresh_source_sha256": sha256(FRESH_STRUCTURAL_SOURCE.encode("ascii")).hexdigest(),
        "expected_fresh_edited_source_sha256": sha256(
            EXPECTED_FRESH_EDITED_SOURCE.encode("ascii")
        ).hexdigest(),
    }
    result = C1StructuralTokenEditCohortSpecV1(
        schema=SCHEMA,
        cohort_nonce_sha256=cohort_nonce_sha256,
        learner_config_sha256=record["learner_config_sha256"],
        episodes=episodes,
        candidate_sha256s=record["candidate_sha256s"],
        training_source_sha256s=record["training_source_sha256s"],
        fresh_source_sha256=record["fresh_source_sha256"],
        expected_fresh_edited_source_sha256=record["expected_fresh_edited_source_sha256"],
        sha256=_digest(_SPEC_NAMESPACE, record),
    )
    result.authenticate()
    return result


PUBLIC_COHORT_NONCE_SHA256 = _digest(
    _PUBLIC_NONCE_NAMESPACE,
    {"curriculum": "identifier-copy-replacement-A-B", "version": 1},
)
DEFAULT_PUBLIC_SPEC = make_c1_structural_token_edit_cohort_spec_v1(
    cohort_nonce_sha256=PUBLIC_COHORT_NONCE_SHA256,
)


@dataclass(frozen=True, slots=True)
class C1TokenEditExecutionEvidenceV1:
    ordinal: int
    episode_label: str
    phase: str
    candidate_sha256: str
    intervention_sha256: str
    selection_sha256: str
    selected_source_grant_sha256: str
    release_sha256: str
    source_frame_sha256: str
    source_sha256: str
    batch_sha256: str
    stdin_sha256: str
    test_source_sha256: str
    execution_sha256: str
    consequence_sha256: str
    observed_output_sha256: str
    expected_output_sha256: str
    public_test_cell: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field: getattr(self, field) for field in self.__dataclass_fields__ if field != "sha256"
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
        if (
            type(self.ordinal) is not int
            or not 1 <= self.ordinal <= 4
            or self.episode_label not in ("A", "B")
            or self.phase not in (FAILED, SUCCESSFUL)
            or any(not _valid_sha256(root) for root in roots)
            or type(self.public_test_cell) is not int
            or self.public_test_cell not in (0, 1)
            or self.public_test_cell != int(self.phase == SUCCESSFUL)
            or self.sha256 != _digest(_EVIDENCE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 token-edit execution evidence authentication failed"
            )


def _make_execution_evidence(
    *,
    ordinal: int,
    episode_label: str,
    phase: str,
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1,
    intervention_sha256: str,
    selection: intervention.RustActionInterventionSelectionV1,
    execution: rust_executor.RustSelectedActionExecutionV2,
    frame: continual.TokenSourceFrameV1,
    material: TokenEditEpisodeInputV1,
    consequence: continual.TokenEditConsequenceV1,
) -> C1TokenEditExecutionEvidenceV1:
    record = {
        "ordinal": ordinal,
        "episode_label": episode_label,
        "phase": phase,
        "candidate_sha256": candidate.sha256,
        "intervention_sha256": intervention_sha256,
        "selection_sha256": selection.sha256,
        "selected_source_grant_sha256": execution.selected_source.sha256,
        "release_sha256": execution.selected_source.release_sha256,
        "source_frame_sha256": frame.sha256,
        "source_sha256": frame.source_sha256,
        "batch_sha256": execution.preparation.batch_sha256,
        "stdin_sha256": execution.preparation.stdin_sha256,
        "test_source_sha256": material.test_source_sha256,
        "execution_sha256": execution.sha256,
        "consequence_sha256": consequence.sha256,
        "observed_output_sha256": token_edit_cohort_output_sha256_v1(execution.outputs),
        "expected_output_sha256": token_edit_cohort_output_sha256_v1(material.expected_outputs),
        "public_test_cell": consequence.public_test_cell,
    }
    result = C1TokenEditExecutionEvidenceV1(
        **record,
        sha256=_digest(_EVIDENCE_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C1TokenEditCohortWorkV1:
    intervention_issue_count: int
    spent_selection_count: int
    source_release_count: int
    execution_attempt_count: int
    successful_execution_count: int
    compile_count: int
    run_count: int
    failed_consequence_count: int
    successful_consequence_count: int
    schema_advance_count: int
    parent_immutability_check_count: int
    proposed_source_execution_count: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field: getattr(self, field) for field in self.__dataclass_fields__ if field != "sha256"
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        values = tuple(self.record(include_sha256=False).values())
        if (
            any(type(value) is not int or value < 0 for value in values)
            or values != (4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 0)
            or self.sha256 != _digest(_WORK_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditSchemaCohortError("C1 token-edit audited work changed")


@dataclass(frozen=True, slots=True)
class _AttemptPlanV1:
    ordinal: int
    episode_index: int
    phase: str
    candidate: knowledge_action.KnowledgeActionCandidateProjectionV1
    source: str


_ATTEMPT_PLANS = (
    _AttemptPlanV1(1, 0, FAILED, A_FAILED_CANDIDATE, A_FAILED_SOURCE),
    _AttemptPlanV1(2, 0, SUCCESSFUL, A_SUCCESSFUL_CANDIDATE, A_SUCCESSFUL_SOURCE),
    _AttemptPlanV1(3, 1, FAILED, B_FAILED_CANDIDATE, B_FAILED_SOURCE),
    _AttemptPlanV1(4, 1, SUCCESSFUL, B_SUCCESSFUL_CANDIDATE, B_SUCCESSFUL_SOURCE),
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


def _intervention_sha256(
    *,
    spec: C1StructuralTokenEditCohortSpecV1,
    plan: _AttemptPlanV1,
    material: TokenEditEpisodeInputV1,
    frame: continual.TokenSourceFrameV1,
) -> str:
    batch_sha256, stdin_sha256, case_count = _executor_batch_commitments(material.batch)
    return _digest(
        _INTERVENTION_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "ordinal": plan.ordinal,
            "episode_input_sha256": material.sha256,
            "phase": plan.phase,
            "candidate_sha256": plan.candidate.sha256,
            "source_frame_sha256": frame.sha256,
            "repository_sha256": material.repository_sha256,
            "task_sha256": material.task_sha256,
            "task_instance_sha256": material.task_instance_sha256,
            "task_source_sha256": material.task_source_sha256,
            "test_source_sha256": material.test_source_sha256,
            "batch_sha256": batch_sha256,
            "stdin_sha256": stdin_sha256,
            "case_count": case_count,
            "expected_output_sha256": token_edit_cohort_output_sha256_v1(material.expected_outputs),
        },
    )


def _make_consequence(
    *,
    execution: rust_executor.RustSelectedActionExecutionV2,
    frame: continual.TokenSourceFrameV1,
    material: TokenEditEpisodeInputV1,
) -> continual.TokenEditConsequenceV1:
    execution.authenticate_structure()
    frame.authenticate()
    return continual.make_token_edit_consequence_v1(
        source_sha256=frame.source_sha256,
        batch_sha256=execution.preparation.batch_sha256,
        test_manifest_sha256=material.test_source_sha256,
        execution_sha256=execution.sha256,
        observed_outputs=execution.outputs,
        expected_outputs=material.expected_outputs,
        compile_cell=execution.compile_receipt.compile_invocations,
        run_cell=execution.run_receipt.run_invocations,
    )


def _canonical_bindings() -> tuple[intervention._RustActionBindingV1, ...]:
    return tuple(
        sorted(
            (
                intervention._make_binding(candidate, source)
                for candidate, source in CANDIDATE_SOURCES
            ),
            key=lambda binding: binding.candidate.sha256,
        )
    )


def _derive_work(
    *,
    selections: tuple[intervention.RustActionInterventionSelectionV1, ...],
    executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    consequences: tuple[continual.TokenEditConsequenceV1, ...],
    episodes: tuple[continual.FailedSuccessfulTokenEditEpisodeV1, ...],
    fresh_source: continual.TokenSourceFrameV1,
    fresh_proposal: continual.TokenEditProposalV1,
) -> C1TokenEditCohortWorkV1:
    proposed_source_roots = {
        fresh_source.source_sha256,
        fresh_proposal.edited_source_sha256,
    }
    proposed_source_roots.discard(None)
    record = {
        "intervention_issue_count": len(selections),
        "spent_selection_count": len(executions),
        "source_release_count": len(executions),
        "execution_attempt_count": len(executions),
        "successful_execution_count": len(executions),
        "compile_count": sum(item.compile_receipt.compile_invocations for item in executions),
        "run_count": sum(item.run_receipt.run_invocations for item in executions),
        "failed_consequence_count": sum(item.public_test_cell == 0 for item in consequences),
        "successful_consequence_count": sum(item.public_test_cell == 1 for item in consequences),
        "schema_advance_count": len(episodes),
        "parent_immutability_check_count": len(episodes),
        "proposed_source_execution_count": sum(
            execution.selected_source.source_sha256 in proposed_source_roots
            for execution in executions
        ),
    }
    result = C1TokenEditCohortWorkV1(
        **record,
        sha256=_digest(_WORK_NAMESPACE, record),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C1StructuralTokenEditCohortResultV1:
    schema: str
    claim_boundary: str
    status: str
    spec: C1StructuralTokenEditCohortSpecV1
    candidate_sha256s: tuple[str, str, str, str]
    intervention_authority: intervention.RustActionInterventionAuthorityV1
    intervention_catalogue: intervention.RustActionInterventionCatalogueV1
    intervention_selections: tuple[
        intervention.RustActionInterventionSelectionV1,
        intervention.RustActionInterventionSelectionV1,
        intervention.RustActionInterventionSelectionV1,
        intervention.RustActionInterventionSelectionV1,
    ]
    executions: tuple[
        rust_executor.RustSelectedActionExecutionV2,
        rust_executor.RustSelectedActionExecutionV2,
        rust_executor.RustSelectedActionExecutionV2,
        rust_executor.RustSelectedActionExecutionV2,
    ]
    consequences: tuple[
        continual.TokenEditConsequenceV1,
        continual.TokenEditConsequenceV1,
        continual.TokenEditConsequenceV1,
        continual.TokenEditConsequenceV1,
    ]
    evidence: tuple[
        C1TokenEditExecutionEvidenceV1,
        C1TokenEditExecutionEvidenceV1,
        C1TokenEditExecutionEvidenceV1,
        C1TokenEditExecutionEvidenceV1,
    ]
    episodes: tuple[
        continual.FailedSuccessfulTokenEditEpisodeV1,
        continual.FailedSuccessfulTokenEditEpisodeV1,
    ]
    checkpoint_lineage_sha256s: tuple[str, str, str]
    checkpoint_array_lineage_sha256s: tuple[str, str, str]
    one_source_proposal: continual.TokenEditProposalV1
    fresh_source: continual.TokenSourceFrameV1
    fresh_proposal: continual.TokenEditProposalV1
    promoted_schema_indices: tuple[int, ...]
    promoted_schema_sha256s: tuple[str, ...]
    final_support_counts: tuple[int, ...]
    work: C1TokenEditCohortWorkV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "claim_boundary": self.claim_boundary,
            "status": self.status,
            "spec_sha256": self.spec.sha256,
            "candidate_sha256s": self.candidate_sha256s,
            "intervention_authority_sha256": self.intervention_authority.sha256,
            "intervention_catalogue_sha256": self.intervention_catalogue.sha256,
            "intervention_selection_sha256s": tuple(
                item.sha256 for item in self.intervention_selections
            ),
            "execution_sha256s": tuple(item.sha256 for item in self.executions),
            "consequence_sha256s": tuple(item.sha256 for item in self.consequences),
            "evidence_sha256s": tuple(item.sha256 for item in self.evidence),
            "episode_sha256s": tuple(item.sha256 for item in self.episodes),
            "checkpoint_lineage_sha256s": self.checkpoint_lineage_sha256s,
            "checkpoint_array_lineage_sha256s": self.checkpoint_array_lineage_sha256s,
            "one_source_proposal_sha256": self.one_source_proposal.sha256,
            "fresh_source_frame_sha256": self.fresh_source.sha256,
            "fresh_proposal_sha256": self.fresh_proposal.sha256,
            "promoted_schema_indices": self.promoted_schema_indices,
            "promoted_schema_sha256s": self.promoted_schema_sha256s,
            "final_support_counts": self.final_support_counts,
            "work_sha256": self.work.sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            self.schema != SCHEMA
            or self.claim_boundary != CLAIM_BOUNDARY
            or self.status != STATUS_C1_STRUCTURAL_PROPOSAL
            or type(self.spec) is not C1StructuralTokenEditCohortSpecV1
            or type(self.intervention_authority)
            is not intervention.RustActionInterventionAuthorityV1
            or type(self.intervention_catalogue)
            is not intervention.RustActionInterventionCatalogueV1
            or type(self.intervention_selections) is not tuple
            or len(self.intervention_selections) != 4
            or type(self.executions) is not tuple
            or len(self.executions) != 4
            or type(self.consequences) is not tuple
            or len(self.consequences) != 4
            or type(self.evidence) is not tuple
            or len(self.evidence) != 4
            or type(self.episodes) is not tuple
            or len(self.episodes) != 2
            or type(self.one_source_proposal) is not continual.TokenEditProposalV1
            or type(self.fresh_source) is not continual.TokenSourceFrameV1
            or type(self.fresh_proposal) is not continual.TokenEditProposalV1
            or type(self.work) is not C1TokenEditCohortWorkV1
        ):
            raise ContinualRustTokenEditSchemaCohortError("C1 token-edit result structure changed")
        self.spec.authenticate()
        expected_candidates = tuple(plan.candidate.sha256 for plan in _ATTEMPT_PLANS)
        for plan in _ATTEMPT_PLANS:
            plan.candidate.authenticate()
        if self.candidate_sha256s != expected_candidates:
            raise ContinualRustTokenEditSchemaCohortError("C1 opaque candidate projections changed")

        bindings = _canonical_bindings()
        expected_catalogue = intervention._make_catalogue(bindings)
        self.intervention_catalogue.authenticate_structure()
        self.intervention_authority.authenticate_structure()
        if (
            self.intervention_catalogue != expected_catalogue
            or self.intervention_authority.catalogue_sha256 != self.intervention_catalogue.sha256
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 intervention authority or catalogue binding changed"
            )
        binding_by_candidate = {binding.candidate.sha256: binding for binding in bindings}

        frames: list[continual.TokenSourceFrameV1] = []
        for index, plan in enumerate(_ATTEMPT_PLANS):
            material = self.spec.episodes[plan.episode_index]
            selection = self.intervention_selections[index]
            execution = self.executions[index]
            consequence = self.consequences[index]
            evidence = self.evidence[index]
            expected_frame = continual.project_token_source_v1(plan.source)
            frame = (
                self.episodes[plan.episode_index].failed_source
                if plan.phase == FAILED
                else self.episodes[plan.episode_index].successful_source
            )
            frames.append(frame)
            if type(selection) is not intervention.RustActionInterventionSelectionV1:
                raise TypeError("C1 selection receipts must be exact")
            if type(execution) is not rust_executor.RustSelectedActionExecutionV2:
                raise TypeError("C1 execution receipts must be exact V2")
            if type(consequence) is not continual.TokenEditConsequenceV1:
                raise TypeError("C1 consequences must be exact common receipts")
            if type(evidence) is not C1TokenEditExecutionEvidenceV1:
                raise TypeError("C1 evidence receipts must be exact")
            selection.authenticate_structure()
            execution.authenticate_structure()
            consequence.authenticate()
            evidence.authenticate()
            frame.authenticate()
            if frame != expected_frame:
                raise ContinualRustTokenEditSchemaCohortError(
                    "C1 retained physical source frame changed"
                )
            expected_intervention = _intervention_sha256(
                spec=self.spec,
                plan=plan,
                material=material,
                frame=frame,
            )
            binding = binding_by_candidate[plan.candidate.sha256]
            expected_release = intervention._make_release(selection, binding)
            batch_sha256, stdin_sha256, case_count = _executor_batch_commitments(material.batch)
            expected_outputs = material.batch if plan.phase == FAILED else material.expected_outputs
            expected_consequence = _make_consequence(
                execution=execution,
                frame=frame,
                material=material,
            )
            expected_evidence = _make_execution_evidence(
                ordinal=plan.ordinal,
                episode_label=material.label,
                phase=plan.phase,
                candidate=plan.candidate,
                intervention_sha256=expected_intervention,
                selection=selection,
                execution=execution,
                frame=frame,
                material=material,
                consequence=consequence,
            )
            if (
                selection.authority_sha256 != self.intervention_authority.sha256
                or selection.catalogue_sha256 != self.intervention_catalogue.sha256
                or selection.candidate_sha256 != plan.candidate.sha256
                or selection.intervention_sha256 != expected_intervention
                or execution.selected_source.authority_sha256 != self.intervention_authority.sha256
                or execution.selected_source.selection_sha256 != selection.sha256
                or execution.selected_source.release_sha256 != expected_release.sha256
                or execution.selected_source.action_handle != plan.candidate.candidate_id
                or execution.selected_source.source_sha256 != binding.rust_source_sha256
                or execution.selected_source.source != plan.source
                or execution.preparation.expected_authority_sha256
                != self.intervention_authority.sha256
                or execution.preparation.batch_sha256 != batch_sha256
                or execution.preparation.stdin_sha256 != stdin_sha256
                or execution.preparation.case_count != case_count
                or execution.run_receipt.batch_sha256 != batch_sha256
                or execution.run_receipt.stdin_sha256 != stdin_sha256
                or execution.outputs != expected_outputs
                or consequence != expected_consequence
                or evidence != expected_evidence
            ):
                raise ContinualRustTokenEditSchemaCohortError(
                    "C1 selected-source execution custody changed"
                )

        expected_episodes = tuple(
            continual.seal_failed_successful_token_edit_episode_v1(
                repository_sha256=material.repository_sha256,
                task_sha256=material.task_sha256,
                task_instance_sha256=material.task_instance_sha256,
                task_source_sha256=material.task_source_sha256,
                test_source_sha256=material.test_source_sha256,
                failed_source=frames[episode_index * 2],
                failed_consequence=self.consequences[episode_index * 2],
                successful_source=frames[episode_index * 2 + 1],
                successful_consequence=self.consequences[episode_index * 2 + 1],
            )
            for episode_index, material in enumerate(self.spec.episodes)
        )
        for episode in self.episodes:
            if type(episode) is not continual.FailedSuccessfulTokenEditEpisodeV1:
                raise TypeError("C1 episodes must be exact common receipts")
            episode.authenticate()
        if self.episodes != expected_episodes:
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 failed-to-successful episode custody changed"
            )

        empty = continual.start_continual_token_edit_schema_v1()
        empty.authenticate()
        empty_root = empty.checkpoint_sha256
        empty_array_root = empty.checkpoint_array_sha256
        after_a = continual.advance_continual_token_edit_schema_v1(empty, self.episodes[0])
        empty.authenticate()
        if (
            empty.checkpoint_sha256 != empty_root
            or empty.checkpoint_array_sha256 != empty_array_root
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 first advance mutated the empty parent"
            )
        expected_fresh = continual.project_token_source_v1(FRESH_STRUCTURAL_SOURCE)
        expected_one_source_proposal = after_a.propose(expected_fresh)
        after_a_root = after_a.checkpoint_sha256
        after_a_array_root = after_a.checkpoint_array_sha256
        after_b = continual.advance_continual_token_edit_schema_v1(
            after_a,
            self.episodes[1],
        )
        after_a.authenticate()
        if (
            after_a.checkpoint_sha256 != after_a_root
            or after_a.checkpoint_array_sha256 != after_a_array_root
        ):
            raise ContinualRustTokenEditSchemaCohortError("C1 second advance mutated its parent")
        expected_fresh_proposal = after_b.propose(expected_fresh)
        self.one_source_proposal.authenticate()
        self.fresh_proposal.authenticate()
        promoted = tuple(
            index
            for index, support in enumerate(after_b.support_counts)
            if support >= continual.MIN_INDEPENDENT_EPISODES
        )
        promoted_roots = tuple(continual.PARAMETERIZED_SCHEMAS[index].sha256 for index in promoted)
        seen_source_roots = {frame.source_sha256 for frame in frames}
        training_tokens = {token for frame in frames for token in frame.tokens}
        fresh_donor_token = expected_fresh.tokens[self.fresh_proposal.donor_token_index]
        if (
            self.checkpoint_lineage_sha256s
            != (empty.checkpoint_sha256, after_a.checkpoint_sha256, after_b.checkpoint_sha256)
            or self.checkpoint_array_lineage_sha256s
            != (
                empty.checkpoint_array_sha256,
                after_a.checkpoint_array_sha256,
                after_b.checkpoint_array_sha256,
            )
            or self.one_source_proposal.record() != expected_one_source_proposal.record()
            or self.one_source_proposal.edited_source != expected_one_source_proposal.edited_source
            or self.one_source_proposal.status != continual.STOP_INSUFFICIENT_INDEPENDENT_SOURCES
            or self.one_source_proposal.edited_source is not None
            or self.one_source_proposal.schema_raw_scores[3] != 2
            or self.one_source_proposal.race_simultaneous_spike_candidates != 2
            or self.fresh_source != expected_fresh
            or self.fresh_proposal.record() != expected_fresh_proposal.record()
            or self.fresh_proposal.edited_source != expected_fresh_proposal.edited_source
            or self.fresh_proposal.status != continual.READY
            or self.fresh_proposal.selected_schema_index != 3
            or self.fresh_proposal.edited_source != EXPECTED_FRESH_EDITED_SOURCE
            or self.fresh_proposal.edited_source_sha256
            != self.spec.expected_fresh_edited_source_sha256
            or fresh_donor_token in training_tokens
            or self.fresh_proposal.input_source_sha256 in seen_source_roots
            or self.fresh_proposal.edited_source_sha256 in seen_source_roots
            or promoted != (3,)
            or self.promoted_schema_indices != promoted
            or self.promoted_schema_sha256s != promoted_roots
            or self.final_support_counts != after_b.support_counts
            or after_b.provenance[0].matched_schema_indices != (3,)
            or after_b.provenance[1].matched_schema_indices != (3,)
            or after_b.provenance[0].independent_cell != 1
            or after_b.provenance[1].independent_cell != 1
            or after_b.support_counts[3] != 2
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 structural quorum, abstention, or fresh proposal changed"
            )

        receipt_root_groups = (
            tuple(item.intervention_sha256 for item in self.intervention_selections),
            tuple(item.sha256 for item in self.intervention_selections),
            tuple(item.sha256 for item in self.executions),
            tuple(item.feedback_sha256 for item in self.consequences),
        )
        if any(len(set(group)) != 4 for group in receipt_root_groups):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 execution or feedback receipt replay is forbidden"
            )
        self.work.authenticate()
        if self.work != _derive_work(
            selections=self.intervention_selections,
            executions=self.executions,
            consequences=self.consequences,
            episodes=self.episodes,
            fresh_source=self.fresh_source,
            fresh_proposal=self.fresh_proposal,
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 work differs from the exact transcript"
            )
        if self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False)):
            raise ContinualRustTokenEditSchemaCohortError(
                "C1 token-edit result authentication failed"
            )


def _seal_result(
    *,
    spec: C1StructuralTokenEditCohortSpecV1,
    authority: intervention.RustActionInterventionAuthorityV1,
    catalogue: intervention.RustActionInterventionCatalogueV1,
    selections: tuple[intervention.RustActionInterventionSelectionV1, ...],
    executions: tuple[rust_executor.RustSelectedActionExecutionV2, ...],
    consequences: tuple[continual.TokenEditConsequenceV1, ...],
    evidence: tuple[C1TokenEditExecutionEvidenceV1, ...],
    episodes: tuple[continual.FailedSuccessfulTokenEditEpisodeV1, ...],
    checkpoint_lineage_sha256s: tuple[str, str, str],
    checkpoint_array_lineage_sha256s: tuple[str, str, str],
    one_source_proposal: continual.TokenEditProposalV1,
    fresh_source: continual.TokenSourceFrameV1,
    fresh_proposal: continual.TokenEditProposalV1,
    promoted_schema_indices: tuple[int, ...],
    final_support_counts: tuple[int, ...],
) -> C1StructuralTokenEditCohortResultV1:
    work = _derive_work(
        selections=selections,
        executions=executions,
        consequences=consequences,
        episodes=episodes,
        fresh_source=fresh_source,
        fresh_proposal=fresh_proposal,
    )
    promoted_schema_sha256s = tuple(
        continual.PARAMETERIZED_SCHEMAS[index].sha256 for index in promoted_schema_indices
    )
    partial = C1StructuralTokenEditCohortResultV1(
        schema=SCHEMA,
        claim_boundary=CLAIM_BOUNDARY,
        status=STATUS_C1_STRUCTURAL_PROPOSAL,
        spec=spec,
        candidate_sha256s=tuple(plan.candidate.sha256 for plan in _ATTEMPT_PLANS),
        intervention_authority=authority,
        intervention_catalogue=catalogue,
        intervention_selections=selections,
        executions=executions,
        consequences=consequences,
        evidence=evidence,
        episodes=episodes,
        checkpoint_lineage_sha256s=checkpoint_lineage_sha256s,
        checkpoint_array_lineage_sha256s=checkpoint_array_lineage_sha256s,
        one_source_proposal=one_source_proposal,
        fresh_source=fresh_source,
        fresh_proposal=fresh_proposal,
        promoted_schema_indices=promoted_schema_indices,
        promoted_schema_sha256s=promoted_schema_sha256s,
        final_support_counts=final_support_counts,
        work=work,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def run_cohort_v1(
    spec: C1StructuralTokenEditCohortSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C1StructuralTokenEditCohortResultV1:
    """Execute four selected sources, learn two episodes, and propose once."""

    if type(spec) is not C1StructuralTokenEditCohortSpecV1:
        raise TypeError("run_cohort_v1 requires one exact C1 cohort specification")
    spec.authenticate()
    prepared = intervention.prepare_rust_action_intervention_v1(CANDIDATE_SOURCES)
    selections: list[intervention.RustActionInterventionSelectionV1] = []
    executions: list[rust_executor.RustSelectedActionExecutionV2] = []
    consequences: list[continual.TokenEditConsequenceV1] = []
    evidence: list[C1TokenEditExecutionEvidenceV1] = []
    episodes: list[continual.FailedSuccessfulTokenEditEpisodeV1] = []

    checkpoint = continual.start_continual_token_edit_schema_v1()
    checkpoint.authenticate()
    checkpoint_roots = [checkpoint.checkpoint_sha256]
    checkpoint_array_roots = [checkpoint.checkpoint_array_sha256]
    fresh_source = continual.project_token_source_v1(FRESH_STRUCTURAL_SOURCE)
    one_source_proposal: continual.TokenEditProposalV1 | None = None

    for episode_index, material in enumerate(spec.episodes):
        episode_frames: list[continual.TokenSourceFrameV1] = []
        episode_consequences: list[continual.TokenEditConsequenceV1] = []
        for plan in _ATTEMPT_PLANS[episode_index * 2 : episode_index * 2 + 2]:
            frame = continual.project_token_source_v1(plan.source)
            intervention_sha256 = _intervention_sha256(
                spec=spec,
                plan=plan,
                material=material,
                frame=frame,
            )
            selection = prepared.issue(plan.candidate.sha256, intervention_sha256)
            execution = prepared.execute_selected(selection, material.batch)
            execution.authenticate_structure()
            consequence = _make_consequence(
                execution=execution,
                frame=frame,
                material=material,
            )
            if consequence.public_test_cell != int(plan.phase == SUCCESSFUL):
                raise ContinualRustTokenEditSchemaCohortError(
                    f"episode {material.label} {plan.phase} consequence changed"
                )
            selections.append(selection)
            executions.append(execution)
            consequences.append(consequence)
            evidence.append(
                _make_execution_evidence(
                    ordinal=plan.ordinal,
                    episode_label=material.label,
                    phase=plan.phase,
                    candidate=plan.candidate,
                    intervention_sha256=intervention_sha256,
                    selection=selection,
                    execution=execution,
                    frame=frame,
                    material=material,
                    consequence=consequence,
                )
            )
            episode_frames.append(frame)
            episode_consequences.append(consequence)

        episode = continual.seal_failed_successful_token_edit_episode_v1(
            repository_sha256=material.repository_sha256,
            task_sha256=material.task_sha256,
            task_instance_sha256=material.task_instance_sha256,
            task_source_sha256=material.task_source_sha256,
            test_source_sha256=material.test_source_sha256,
            failed_source=episode_frames[0],
            failed_consequence=episode_consequences[0],
            successful_source=episode_frames[1],
            successful_consequence=episode_consequences[1],
        )
        parent_root = checkpoint.checkpoint_sha256
        parent_array_root = checkpoint.checkpoint_array_sha256
        successor = continual.advance_continual_token_edit_schema_v1(checkpoint, episode)
        checkpoint.authenticate()
        if (
            checkpoint.checkpoint_sha256 != parent_root
            or checkpoint.checkpoint_array_sha256 != parent_array_root
        ):
            raise ContinualRustTokenEditSchemaCohortError(
                "continual token-edit advance mutated its immutable parent"
            )
        checkpoint = successor
        episodes.append(episode)
        checkpoint_roots.append(checkpoint.checkpoint_sha256)
        checkpoint_array_roots.append(checkpoint.checkpoint_array_sha256)
        if episode_index == 0:
            one_source_proposal = checkpoint.propose(fresh_source)
            if one_source_proposal.status != continual.STOP_INSUFFICIENT_INDEPENDENT_SOURCES:
                raise ContinualRustTokenEditSchemaCohortError(
                    "one independent source did not physically abstain"
                )

    if one_source_proposal is None:
        raise RuntimeError("C1 curriculum did not produce the one-source control")
    final_proposal = checkpoint.propose(fresh_source)
    promoted = tuple(
        index
        for index, support in enumerate(checkpoint.support_counts)
        if support >= continual.MIN_INDEPENDENT_EPISODES
    )
    if (
        promoted != (3,)
        or final_proposal.status != continual.READY
        or final_proposal.selected_schema_index != 3
        or final_proposal.edited_source != EXPECTED_FRESH_EDITED_SOURCE
    ):
        raise ContinualRustTokenEditSchemaCohortError(
            "two-source C1 curriculum did not produce the structural proposal"
        )
    counters = prepared.work_counters
    if (
        counters.issued_selection_count,
        counters.spent_selection_count,
        counters.source_release_count,
        counters.execution_attempt_count,
        counters.successful_execution_count,
    ) != (4, 4, 4, 4, 4):
        raise ContinualRustTokenEditSchemaCohortError("selected-only intervention work changed")

    return _seal_result(
        spec=spec,
        authority=prepared.authority,
        catalogue=prepared.catalogue,
        selections=tuple(selections),
        executions=tuple(executions),
        consequences=tuple(consequences),
        evidence=tuple(evidence),
        episodes=tuple(episodes),
        checkpoint_lineage_sha256s=tuple(checkpoint_roots),
        checkpoint_array_lineage_sha256s=tuple(checkpoint_array_roots),
        one_source_proposal=one_source_proposal,
        fresh_source=fresh_source,
        fresh_proposal=final_proposal,
        promoted_schema_indices=promoted,
        final_support_counts=checkpoint.support_counts,
    )


__all__ = [
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
    "CANDIDATE_SOURCES",
    "CLAIM_BOUNDARY",
    "DEFAULT_PUBLIC_SPEC",
    "EXPECTED_FRESH_EDITED_SOURCE",
    "FAILED",
    "FRESH_STRUCTURAL_SOURCE",
    "PUBLIC_COHORT_NONCE_SHA256",
    "SCHEMA",
    "STATUS_C1_STRUCTURAL_PROPOSAL",
    "SUCCESSFUL",
    "C1StructuralTokenEditCohortResultV1",
    "C1StructuralTokenEditCohortSpecV1",
    "C1TokenEditCohortWorkV1",
    "C1TokenEditExecutionEvidenceV1",
    "ContinualRustTokenEditSchemaCohortError",
    "TokenEditEpisodeInputV1",
    "make_c1_structural_token_edit_cohort_spec_v1",
    "run_cohort_v1",
    "token_edit_cohort_output_sha256_v1",
]
