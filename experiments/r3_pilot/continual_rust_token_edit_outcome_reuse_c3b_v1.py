"""Fresh C3b consequence-reuse cohort for the fixed token-edit schema bank.

The cohort spends six selected Rust executions after an exclusive persistent
reservation.  One failed/successful seed pair gives schema 3 one support tick.
An adaptation query then needs explicit exploration; its proposal passes one
public feedback opening and advances the consequence policy exactly once.  A
fresh evaluation query is READY only from that successor, and evaluation never
updates the policy.

The Rust executor and both public-feedback openings are injected at the core
transaction boundary.  Production uses the selected-only intervention method;
tests can supply exact ``RustSelectedActionExecutionV2`` fakes without running
Rust.  A production result is selected-only Rust execution evidence with public,
caller-attested expected outputs.  It is not semantic diagnosis, generic repair,
protected transfer, SWE performance, or intelligence.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import common_continual_token_edit_consequence_policy_v1 as policy
from experiments.r3_pilot import common_continual_token_edit_schema_v1 as token_edits
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_actions
from experiments.r3_pilot import common_rust_action_intervention_v1 as interventions
from humanbrain import rust_selected_action_executor as rust_executor
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

SCHEMA = "humanbrain.continual-rust-token-edit-outcome-reuse-c3b.v1"
RESULT_DOCUMENT_SCHEMA = "humanbrain.continual-rust-token-edit-outcome-reuse-c3b-result-document.v1"
RESERVATION_SCHEMA = "humanbrain.continual-rust-token-edit-outcome-reuse-c3b-reservation.v1"
STATUS_PASS = "PASS_C3B_TOKEN_EDIT_OUTCOME_REUSE_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C3B_TOKEN_EDIT_OUTCOME_REUSE_V1"
STATUS_EXECUTION_ABORT = "ABORT_C3B_TOKEN_EDIT_OUTCOME_REUSE_V1"
CLAIM_BOUNDARY = (
    "one-fresh-c3b-public-seed-support-tick-plus-one-selected-only-rust-executed-"
    "caller-attested-adaptation-pass-reused-on-one-fresh-evaluation-query-through-"
    "the-fixed-token-edit-schema-bank-not-schema-discovery-semantic-diagnosis-"
    "generic-repair-protected-transfer-swe-or-intelligence"
)

FROZEN_MECHANISM_COMMIT = "d9a3018cebaadda28b524bd32c7d25c4c56970e6"
FROZEN_LEARNER_CONFIG_SHA256 = "84b60787deed505352885fad33abe58ba8b0d6c1d16cee688a7d19da0eb912a5"
FROZEN_MECHANISM_FILE_SHA256S = (
    (
        "experiments/r3_pilot/common_continual_action_outcome_memory_v1.py",
        "7aaf0a71eb968f48a1e683f064a0c2aa6a04c84e778f7cb510c0421ff78605f5",
    ),
    (
        "experiments/r3_pilot/common_continual_token_edit_schema_v1.py",
        "a0380e8ec30c55887a5e2b72c835d701d597a3ad90701d3e6c07cabed67ef443",
    ),
    (
        "experiments/r3_pilot/common_continual_token_edit_consequence_policy_v1.py",
        "dcc6f93c1ce599961e894178f8d609a000f07a0e26f2f6cb5e7d04d0dd7f48b6",
    ),
    (
        "experiments/r3_pilot/common_knowledge_action_memory_v1.py",
        "53ac814c18b98e047d0be267d616099f16db518a38c4cd8067f6344f338f2682",
    ),
    (
        "experiments/r3_pilot/common_rust_action_intervention_v1.py",
        "58ba951aad8eb10fc8b277025c4809748ec815ff430921f9c9c086078f06e43c",
    ),
    (
        "src/humanbrain/rust_selected_action_executor.py",
        "395c509bcbe66ef4a95e2a6c072766b38e58eb13409263c97bce4c4d8a431c29",
    ),
)

SCHEMA_INDEX = 3
SCHEMA_SHA256 = "7cf44511bced68635e3c77a199542061b404b5861fe6a51c7c6ba88b01c2b014"

SEED_BATCH = ((-6, 7, 11, 2), (8, -1, 5))
SEED_FAILED_OUTPUTS = ((7, 7, 7, 7), (7, 7, 7))
SEED_EXPECTED_OUTPUTS = ((-6, 7, 7, 2), (7, -1, 5))
ADAPTATION_BATCH = ((6, 1, 9, -2), (7, 5, 6))
ADAPTATION_BASELINE_OUTPUTS = ((0, 0, 0, 0), (0, 0, 0))
ADAPTATION_EXPECTED_OUTPUTS = ((0, 1, 0, 1), (0, 1, 0))
EVALUATION_BATCH = ((-3, -2, 4, -8), (0, -3, 9))
EVALUATION_BASELINE_OUTPUTS = ((0, 0, 0, 0), (0, 0, 0))
EVALUATION_EXPECTED_OUTPUTS = ((0, 1, 1, 0), (1, 0, 1))

SEED_FAILED_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let ceiling = 7_i64;
    samples
        .iter()
        .copied()
        .filter_map(|sample| {
            Some(if sample >= sample { ceiling } else { sample })
        })
        .collect()
}
"""

SEED_SUCCESSFUL_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let ceiling = 7_i64;
    samples
        .iter()
        .copied()
        .filter_map(|sample| {
            Some(if sample >= ceiling { ceiling } else { sample })
        })
        .collect()
}
"""

ADAPTATION_SOURCE = """\
fn stamp(condition: bool, _pivot: i64) -> i64 {
    condition as i64
}

pub fn solve(levels: &[i64]) -> Vec<i64> {
    let pivot = 6_i64;
    levels
        .iter()
        .copied()
        .scan((), |_, level| Some(stamp(level < level, pivot)))
        .collect()
}
"""

ADAPTATION_EDITED_SOURCE = """\
fn stamp(condition: bool, _pivot: i64) -> i64 {
    condition as i64
}

pub fn solve(levels: &[i64]) -> Vec<i64> {
    let pivot = 6_i64;
    levels
        .iter()
        .copied()
        .scan((), |_, level| Some(stamp(level < pivot, pivot)))
        .collect()
}
"""

EVALUATION_SOURCE = """\
fn mark(active: bool, _limit: i64) -> i64 {
    active as i64
}

pub fn solve(temperatures: &[i64]) -> Vec<i64> {
    let limit = -3_i64;
    let mut marks = Vec::with_capacity(temperatures.len());
    temperatures.iter().copied().for_each(|temperature| {
        marks.push(mark(temperature > temperature, limit));
    });
    marks
}
"""

EVALUATION_EDITED_SOURCE = """\
fn mark(active: bool, _limit: i64) -> i64 {
    active as i64
}

pub fn solve(temperatures: &[i64]) -> Vec<i64> {
    let limit = -3_i64;
    let mut marks = Vec::with_capacity(temperatures.len());
    temperatures.iter().copied().for_each(|temperature| {
        marks.push(mark(temperature > limit, limit));
    });
    marks
}
"""

SOURCE_SHA256S = (
    "bda397623caaf27ecef7e73e0b6052a0c12c3ff419333ef1f7231a1e011485d3",
    "514720a04d50420dbc4d1da9be1c24da37be8102009ecdcaa7259fc114795112",
    "be1bc451971f347f22d0072a3411d15a54f00c1d4c819f1be9d575330455625d",
    "d907300fcf3390c3b8697a8437f8448cc10057736ac1bdee3c27084f2a7b0b7f",
    "4588278d064b626f0a9c6e6c6067992d64ac547d09e1ff15e386f9d441f2d1b7",
    "b2ab91c571c3ab154735d98bb584502ba0039030372c33a2b75ad86992931319",
)

EXCLUDED_PRIOR_SOURCE_SHA256S = tuple(
    sorted(
        (
            "c5bc2f2af0a579325f857006e222f4f59f4f44bd58eceaa0dd8e82baba16f9c5",
            "16504bb28d57532c6621aa0a36c6db23204d54fe7ea0c56132efa8e38b234704",
            "8b225ebb888c282c0e6293215cdaecb36427efcbe3ca7317bbf322e5e1888c49",
            "126b0ce796c248396e311e31a332e28dcdfbb328e0b994d64b39a1f6e3e0877b",
            "a8102ce87ef9940510e2921ce401d3a66aa1b6ac08a6865fc84b733bc322cd71",
            "a5979ba178a792517f4e675fba66fe1a61f96c1a383e8453f32c5320f22fd04a",
            "d335b523ec67621ba6925c48c84f1d4cce2f38f00956099297db443d1fe80f50",
            "d00939b3ee635b6011a6e3296550dfef6fa1fa1d3211f42514822300bb37332c",
            "9156378419e5a875a0aa88632bbada13c3261e07ca979ce1ea9afbf650025975",
            "a35874ffedbaabac33977d554916c049a823434b35d57e5ba82be0770ab088a7",
            "66703237df4e8f68e7f9e678239970c93e218e7267bae1aeb4b8833e7a475a93",
            "ca96a9774be460daed557117e0b56fc334d29d709d9fac0a6f53003e3fd9be34",
            "118db3017e71bd5380cfa8bc3a06da5ea47fe038d4ccc3f8e5440ce51afb7fbc",
            "11fdcb5cf4d1ab242cabbc6ce844f0b2da3f5fa03a57f8aa2f2ecb7f7e410fb4",
            "a54d2fe9f99aaebfbe7cd98fe991c26142bea39a172f9ffa3c05117af17876a0",
            "09b937e053872172813c09637fa96b4f19775ecda6911f61b1431b413ccec8b3",
            "913097c4d9e858c41d80a8d671a5996340a9d3071d1b1902350bb4403a236f88",
            "967382da9e579783e7fe5cb5cbf3c4e1a92f5d8ca84948aa641d0f289f62fb8d",
        )
    )
)

_SOURCES = (
    SEED_FAILED_SOURCE,
    SEED_SUCCESSFUL_SOURCE,
    ADAPTATION_SOURCE,
    ADAPTATION_EDITED_SOURCE,
    EVALUATION_SOURCE,
    EVALUATION_EDITED_SOURCE,
)
_ROLES = (
    "SEED_FAILED",
    "SEED_SUCCESSFUL",
    "ADAPTATION_BASELINE",
    "ADAPTATION_PROPOSAL",
    "EVALUATION_BASELINE",
    "EVALUATION_PROPOSAL",
)

SEED_FAILED_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_b3", "opaque_c3b.slot_b3()"
)
SEED_SUCCESSFUL_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_h8", "opaque_c3b.slot_h8()"
)
ADAPTATION_BASELINE_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_m4", "opaque_c3b.slot_m4()"
)
ADAPTATION_PROPOSAL_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_q1", "opaque_c3b.slot_q1()"
)
EVALUATION_BASELINE_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_v7", "opaque_c3b.slot_v7()"
)
EVALUATION_PROPOSAL_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_z5", "opaque_c3b.slot_z5()"
)
_CANDIDATES = (
    SEED_FAILED_CANDIDATE,
    SEED_SUCCESSFUL_CANDIDATE,
    ADAPTATION_BASELINE_CANDIDATE,
    ADAPTATION_PROPOSAL_CANDIDATE,
    EVALUATION_BASELINE_CANDIDATE,
    EVALUATION_PROPOSAL_CANDIDATE,
)

_SPEC_NAMESPACE = b"c3b-token-edit-outcome-reuse-spec-v1"
_OUTPUT_NAMESPACE = b"c3b-token-edit-outcome-reuse-output-v1"
_INTERVENTION_NAMESPACE = b"c3b-token-edit-outcome-reuse-intervention-v1"
_FEEDBACK_NAMESPACE = b"c3b-token-edit-outcome-reuse-feedback-v1"
_SEED_CONTEXT_NAMESPACE = b"c3b-token-edit-outcome-reuse-seed-context-v1"
_RESERVATION_NAMESPACE = b"c3b-token-edit-outcome-reuse-reservation-v1"
_ATTEMPT_NAMESPACE = b"c3b-token-edit-outcome-reuse-attempt-v1"
_PROPOSAL_LINK_NAMESPACE = b"c3b-token-edit-outcome-reuse-proposal-link-v1"
_RESULT_NAMESPACE = b"c3b-token-edit-outcome-reuse-result-v1"
_DOCUMENT_NAMESPACE = b"c3b-token-edit-outcome-reuse-document-v1"
_ABORT_NAMESPACE = b"c3b-token-edit-outcome-reuse-abort-v1"


class ContinualRustTokenEditOutcomeReuseCohortError(ValueError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if type(value) is tuple:
        return tuple(_plain(item) for item in value)
    if type(value) is list:
        return [_plain(item) for item in value]
    if type(value) is dict:
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: _plain(item) for key, item in vars(value).items()}
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise TypeError(f"C3 transcript cannot serialize {type(value).__qualname__}")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_commit(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _current_mechanism_file_sha256s() -> tuple[tuple[str, str], ...]:
    repository = Path(__file__).resolve(strict=True).parents[2]
    return tuple(
        (relative, sha256((repository / relative).read_bytes()).hexdigest())
        for relative, _expected in FROZEN_MECHANISM_FILE_SHA256S
    )


def _validate_outputs(
    outputs: object,
    *,
    expected_cases: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if (
        type(outputs) is not tuple
        or (expected_cases is not None and len(outputs) != expected_cases)
        or any(
            type(row) is not tuple or any(type(value) is not int for value in row)
            for row in outputs
        )
    ):
        raise TypeError("C3 outputs must be one exact bounded integer tuple batch")
    return outputs


def outcome_reuse_output_sha256_v1(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(_OUTPUT_NAMESPACE, _validate_outputs(outputs))


def _validate_batch(batch: object) -> tuple[tuple[int, ...], ...]:
    checked = _validate_outputs(batch)
    if not checked:
        raise ContinualRustTokenEditOutcomeReuseCohortError("C3 batch must be nonempty")
    return checked


@dataclass(frozen=True, slots=True)
class C3OutcomeReuseCohortSpecV1:
    frozen_mechanism_commit: str
    frozen_learner_config_sha256: str
    frozen_mechanism_file_sha256s: tuple[tuple[str, str], ...]
    seed_batch: tuple[tuple[int, ...], ...]
    seed_expected_outputs: tuple[tuple[int, ...], ...]
    adaptation_batch: tuple[tuple[int, ...], ...]
    adaptation_expected_output_sha256: str
    evaluation_batch: tuple[tuple[int, ...], ...]
    evaluation_expected_output_sha256: str
    source_sha256s: tuple[str, ...]
    excluded_source_sha256s: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        result.update({"schema": SCHEMA, "claim_boundary": CLAIM_BOUNDARY})
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        source_roots = tuple(sha256(source.encode("ascii")).hexdigest() for source in _SOURCES)
        if (
            self.frozen_mechanism_commit != FROZEN_MECHANISM_COMMIT
            or self.frozen_learner_config_sha256 != FROZEN_LEARNER_CONFIG_SHA256
            or self.frozen_learner_config_sha256 != token_edits.CONFIG_SHA256
            or self.frozen_mechanism_file_sha256s != FROZEN_MECHANISM_FILE_SHA256S
            or self.frozen_mechanism_file_sha256s != _current_mechanism_file_sha256s()
            or _validate_batch(self.seed_batch) != SEED_BATCH
            or _validate_outputs(
                self.seed_expected_outputs,
                expected_cases=len(self.seed_batch),
            )
            != SEED_EXPECTED_OUTPUTS
            or _validate_batch(self.adaptation_batch) != ADAPTATION_BATCH
            or _validate_batch(self.evaluation_batch) != EVALUATION_BATCH
            or self.adaptation_expected_output_sha256
            != outcome_reuse_output_sha256_v1(ADAPTATION_EXPECTED_OUTPUTS)
            or self.evaluation_expected_output_sha256
            != outcome_reuse_output_sha256_v1(EVALUATION_EXPECTED_OUTPUTS)
            or self.source_sha256s != SOURCE_SHA256S
            or self.source_sha256s != source_roots
            or len(set(source_roots)) != 6
            or self.excluded_source_sha256s != EXCLUDED_PRIOR_SOURCE_SHA256S
            or set(source_roots).intersection(self.excluded_source_sha256s)
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditOutcomeReuseCohortError(
                "C3 public fixture specification changed"
            )


def make_c3_outcome_reuse_cohort_spec_v1() -> C3OutcomeReuseCohortSpecV1:
    partial = C3OutcomeReuseCohortSpecV1(
        frozen_mechanism_commit=FROZEN_MECHANISM_COMMIT,
        frozen_learner_config_sha256=FROZEN_LEARNER_CONFIG_SHA256,
        frozen_mechanism_file_sha256s=FROZEN_MECHANISM_FILE_SHA256S,
        seed_batch=SEED_BATCH,
        seed_expected_outputs=SEED_EXPECTED_OUTPUTS,
        adaptation_batch=ADAPTATION_BATCH,
        adaptation_expected_output_sha256=outcome_reuse_output_sha256_v1(
            ADAPTATION_EXPECTED_OUTPUTS
        ),
        evaluation_batch=EVALUATION_BATCH,
        evaluation_expected_output_sha256=outcome_reuse_output_sha256_v1(
            EVALUATION_EXPECTED_OUTPUTS
        ),
        source_sha256s=SOURCE_SHA256S,
        excluded_source_sha256s=EXCLUDED_PRIOR_SOURCE_SHA256S,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_SPEC_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


DEFAULT_PUBLIC_SPEC = make_c3_outcome_reuse_cohort_spec_v1()


@dataclass(frozen=True, slots=True)
class C3RunReservationV1:
    schema: str
    spec_sha256: str
    preregistration_commit: str
    destination: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            "schema": self.schema,
            "spec_sha256": self.spec_sha256,
            "preregistration_commit": self.preregistration_commit,
            "destination": self.destination,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self, *, destination: Path) -> None:
        if (
            self.schema != RESERVATION_SCHEMA
            or self.spec_sha256 != DEFAULT_PUBLIC_SPEC.sha256
            or not _valid_commit(self.preregistration_commit)
            or Path(self.destination) != destination
            or self.sha256 != _digest(_RESERVATION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditOutcomeReuseCohortError(
                "C3 run reservation authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C3ExecutionAttemptV1:
    ordinal: int
    role: str
    source_sha256: str
    selection: interventions.RustActionInterventionSelectionV1
    execution: RustSelectedActionExecutionV2
    consequence: token_edits.TokenEditConsequenceV1
    expected_output_sha256: str
    passed_cell: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            "ordinal": self.ordinal,
            "role": self.role,
            "source_sha256": self.source_sha256,
            "selection_sha256": self.selection.sha256,
            "execution_sha256": self.execution.sha256,
            "consequence_sha256": self.consequence.sha256,
            "observed_output_sha256": outcome_reuse_output_sha256_v1(self.execution.outputs),
            "expected_output_sha256": self.expected_output_sha256,
            "passed_cell": self.passed_cell,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.selection) is not interventions.RustActionInterventionSelectionV1:
            raise TypeError("C3 attempt selection must be exact")
        if type(self.execution) is not RustSelectedActionExecutionV2:
            raise TypeError("C3 attempt execution must be exact V2")
        if type(self.consequence) is not token_edits.TokenEditConsequenceV1:
            raise TypeError("C3 attempt consequence must be exact")
        self.selection.authenticate_structure()
        self.execution.authenticate_structure()
        self.consequence.authenticate()
        passed = int(self.execution.outputs == self.consequence.expected_outputs)
        candidate = _CANDIDATES[self.ordinal - 1] if self.ordinal in range(1, 7) else None
        if (
            self.ordinal not in range(1, 7)
            or self.role != _ROLES[self.ordinal - 1]
            or self.source_sha256 != SOURCE_SHA256S[self.ordinal - 1]
            or self.selection.candidate_sha256 != candidate.sha256
            or self.selection.authority_sha256 != self.execution.selected_source.authority_sha256
            or self.execution.selected_source.selection_sha256 != self.selection.sha256
            or self.execution.selected_source.action_handle != candidate.candidate_id
            or self.execution.selected_source.source_sha256 != self.source_sha256
            or self.consequence.source_sha256 != self.source_sha256
            or self.consequence.execution_sha256 != self.execution.sha256
            or self.consequence.batch_sha256 != self.execution.preparation.batch_sha256
            or self.consequence.observed_outputs != self.execution.outputs
            or self.expected_output_sha256
            != outcome_reuse_output_sha256_v1(self.consequence.expected_outputs)
            or self.passed_cell != passed
            or self.consequence.public_test_cell != passed
            or self.sha256 != _digest(_ATTEMPT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditOutcomeReuseCohortError(
                "C3 execution attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C3OutcomeReuseWorkV1:
    reservation_count: int
    issued_selection_count: int
    selected_source_execution_count: int
    compile_count: int
    run_count: int
    public_feedback_open_count: int
    schema_advance_count: int
    policy_advance_count: int
    evaluation_policy_advance_count: int

    def record(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

    def authenticate(self) -> None:
        if self.record() != {
            "reservation_count": 1,
            "issued_selection_count": 6,
            "selected_source_execution_count": 6,
            "compile_count": 6,
            "run_count": 6,
            "public_feedback_open_count": 2,
            "schema_advance_count": 1,
            "policy_advance_count": 1,
            "evaluation_policy_advance_count": 0,
        }:
            raise ContinualRustTokenEditOutcomeReuseCohortError("C3 work changed")


_EXPECTED_PAIR_WORK_COUNTERS = interventions.RustActionInterventionWorkCountersV1(
    issued_selection_count=2,
    spent_selection_count=2,
    source_release_count=2,
    execution_attempt_count=2,
    successful_execution_count=2,
)
_EXPECTED_PAIR_WORK_COUNTERS.authenticate_structure()


def _authenticate_pair_work(
    prepared: interventions.PreparedRustActionInterventionV1,
) -> interventions.RustActionInterventionWorkCountersV1:
    counters = prepared.work_counters
    counters.authenticate_structure()
    if counters != _EXPECTED_PAIR_WORK_COUNTERS:
        raise ContinualRustTokenEditOutcomeReuseCohortError(
            "C3 intervention authority work differs from 2/2/2/2/2"
        )
    return counters


def _proposal_link(
    proposal: policy.TokenEditConsequencePolicyProposalV1,
) -> str:
    return _digest(_PROPOSAL_LINK_NAMESPACE, proposal.record())


@dataclass(frozen=True, slots=True)
class C3OutcomeReuseCohortResultV1:
    status: str
    spec: C3OutcomeReuseCohortSpecV1
    reservation: C3RunReservationV1
    base_checkpoint: token_edits.FrozenContinualTokenEditSchemaV1
    parent_policy: policy.FrozenContinualTokenEditConsequencePolicyV1
    successor_policy: policy.FrozenContinualTokenEditConsequencePolicyV1
    adaptation_normal: policy.TokenEditConsequencePolicyProposalV1
    adaptation_exploration: policy.TokenEditConsequencePolicyProposalV1
    adaptation_outcome: policy.TokenEditConsequencePolicyOutcomeV1
    evaluation_parent: policy.TokenEditConsequencePolicyProposalV1
    evaluation_outcome_off: policy.TokenEditConsequencePolicyProposalV1
    evaluation_successor: policy.TokenEditConsequencePolicyProposalV1
    evaluation_outcome: policy.TokenEditConsequencePolicyOutcomeV1
    attempts: tuple[C3ExecutionAttemptV1, ...]
    authorities: tuple[interventions.RustActionInterventionAuthorityV1, ...]
    catalogues: tuple[interventions.RustActionInterventionCatalogueV1, ...]
    authority_work_counters: tuple[interventions.RustActionInterventionWorkCountersV1, ...]
    feedback_sha256s: tuple[str, str]
    work: C3OutcomeReuseWorkV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "status": self.status,
            "spec_sha256": self.spec.sha256,
            "reservation_sha256": self.reservation.sha256,
            "preregistration_commit": self.reservation.preregistration_commit,
            "base_checkpoint_sha256": self.base_checkpoint.checkpoint_sha256,
            "parent_policy_sha256": self.parent_policy.checkpoint_sha256,
            "successor_policy_sha256": self.successor_policy.checkpoint_sha256,
            "adaptation_normal_sha256": _proposal_link(self.adaptation_normal),
            "adaptation_exploration_sha256": _proposal_link(self.adaptation_exploration),
            "adaptation_outcome_sha256": self.adaptation_outcome.selected_outcome.sha256,
            "evaluation_parent_sha256": _proposal_link(self.evaluation_parent),
            "evaluation_outcome_off_sha256": _proposal_link(self.evaluation_outcome_off),
            "evaluation_successor_sha256": _proposal_link(self.evaluation_successor),
            "evaluation_outcome_sha256": self.evaluation_outcome.selected_outcome.sha256,
            "attempts": tuple(attempt.record() for attempt in self.attempts),
            "authority_sha256s": tuple(item.sha256 for item in self.authorities),
            "catalogue_sha256s": tuple(item.sha256 for item in self.catalogues),
            "authority_work_counters": tuple(
                dataclasses.asdict(item) for item in self.authority_work_counters
            ),
            "feedback_sha256s": self.feedback_sha256s,
            "work": self.work.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        self.spec.authenticate()
        self.reservation.authenticate(destination=Path(self.reservation.destination))
        self.base_checkpoint.authenticate()
        self.parent_policy.authenticate()
        self.successor_policy.authenticate()
        for proposal in (
            self.adaptation_normal,
            self.adaptation_exploration,
            self.evaluation_parent,
            self.evaluation_outcome_off,
            self.evaluation_successor,
        ):
            proposal.authenticate()
        self.adaptation_outcome.authenticate()
        self.evaluation_outcome.authenticate()
        for attempt in self.attempts:
            attempt.authenticate()
        for authority in self.authorities:
            authority.authenticate_structure()
        for catalogue in self.catalogues:
            catalogue.authenticate_structure()
        for counters in self.authority_work_counters:
            counters.authenticate_structure()
        self.work.authenticate()

        support = self.base_checkpoint.support_counts
        schema_root = token_edits.PARAMETERIZED_SCHEMAS[SCHEMA_INDEX].sha256
        action_row = self.successor_policy.outcome_memory.action_sha256s.index(schema_root)
        pass_cells = tuple(attempt.passed_cell for attempt in self.attempts)
        retained_pair_links = tuple(
            (
                attempt.selection.authority_sha256,
                attempt.selection.catalogue_sha256,
            )
            for attempt in self.attempts
        )
        expected_pair_links = tuple(
            (authority.sha256, catalogue.sha256)
            for authority, catalogue in zip(
                self.authorities,
                self.catalogues,
                strict=True,
            )
            for _ in range(2)
        )
        if (
            self.status
            != (STATUS_PASS if pass_cells == (0, 1, 0, 1, 0, 1) else STATUS_BEHAVIOR_FAIL)
            or len(self.attempts) != 6
            or tuple(attempt.ordinal for attempt in self.attempts) != tuple(range(1, 7))
            or len({attempt.execution.sha256 for attempt in self.attempts}) != 6
            or len({attempt.selection.sha256 for attempt in self.attempts}) != 6
            or len(self.authorities) != 3
            or len(self.catalogues) != 3
            or len(self.authority_work_counters) != 3
            or retained_pair_links != expected_pair_links
            or any(
                authority.catalogue_sha256 != catalogue.sha256
                for authority, catalogue in zip(
                    self.authorities,
                    self.catalogues,
                    strict=True,
                )
            )
            or any(
                counters != _EXPECTED_PAIR_WORK_COUNTERS
                for counters in self.authority_work_counters
            )
            or any(not _valid_sha256(value) for value in self.feedback_sha256s)
            or support[SCHEMA_INDEX] != 1
            or sum(support) != 1
            or self.base_checkpoint.provenance[0].matched_schema_indices != (SCHEMA_INDEX,)
            or self.parent_policy.base_checkpoint is not self.base_checkpoint
            or self.successor_policy.base_checkpoint is not self.base_checkpoint
            or self.adaptation_normal.policy is not self.parent_policy
            or self.adaptation_normal.status != policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
            or self.adaptation_exploration.policy is not self.parent_policy
            or self.adaptation_exploration.status != policy.EXPLORATION_READY
            or self.adaptation_exploration.selected_schema_index != SCHEMA_INDEX
            or self.adaptation_exploration.edited_source != ADAPTATION_EDITED_SOURCE
            or self.adaptation_outcome.proposal is not self.adaptation_exploration
            or self.adaptation_outcome.passed_cell != 1
            or len(self.successor_policy.outcome_memory.provenance) != 1
            or self.successor_policy.outcome_memory.excitatory_ticks[action_row] != 1
            or self.successor_policy.outcome_memory.inhibitory_ticks[action_row] != 0
            or self.evaluation_parent.policy is not self.parent_policy
            or self.evaluation_parent.status != policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
            or self.evaluation_outcome_off.policy is not self.successor_policy
            or self.evaluation_outcome_off.decision.outcome_gate
            or self.evaluation_outcome_off.status != policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
            or self.evaluation_successor.policy is not self.successor_policy
            or self.evaluation_successor.status != policy.READY
            or self.evaluation_successor.selected_schema_index != SCHEMA_INDEX
            or self.evaluation_successor.edited_source != EVALUATION_EDITED_SOURCE
            or self.evaluation_successor.decision.effective_scores[
                self.evaluation_successor.decision.activation.active_action_sha256s.index(
                    schema_root
                )
            ]
            != 4
            or self.evaluation_outcome.proposal is not self.evaluation_successor
            or self.evaluation_outcome.passed_cell != 1
            or self.attempts[2].consequence.test_manifest_sha256 != self.feedback_sha256s[0]
            or self.attempts[3].consequence.test_manifest_sha256 != self.feedback_sha256s[0]
            or self.attempts[4].consequence.test_manifest_sha256 != self.feedback_sha256s[1]
            or self.attempts[5].consequence.test_manifest_sha256 != self.feedback_sha256s[1]
            or self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditOutcomeReuseCohortError(
                "C3 cohort result authentication failed"
            )


ExecutionCallback = Callable[
    [
        interventions.PreparedRustActionInterventionV1,
        interventions.RustActionInterventionSelectionV1,
        tuple[tuple[int, ...], ...],
    ],
    RustSelectedActionExecutionV2,
]
FeedbackCallback = Callable[[], tuple[tuple[int, ...], ...]]


def _execute_selected(
    prepared: interventions.PreparedRustActionInterventionV1,
    selection: interventions.RustActionInterventionSelectionV1,
    batch: tuple[tuple[int, ...], ...],
) -> RustSelectedActionExecutionV2:
    return prepared.execute_selected(selection, batch)


def _prepare_pair(
    first_index: int,
) -> interventions.PreparedRustActionInterventionV1:
    return interventions.prepare_rust_action_intervention_v1(
        (
            (_CANDIDATES[first_index], _SOURCES[first_index]),
            (_CANDIDATES[first_index + 1], _SOURCES[first_index + 1]),
        )
    )


def _run_execution(
    *,
    prepared: interventions.PreparedRustActionInterventionV1,
    index: int,
    batch: tuple[tuple[int, ...], ...],
    execute: ExecutionCallback,
    spec_sha256: str,
) -> tuple[interventions.RustActionInterventionSelectionV1, RustSelectedActionExecutionV2]:
    candidate = _CANDIDATES[index]
    source = _SOURCES[index]
    intervention_sha256 = _digest(
        _INTERVENTION_NAMESPACE,
        {
            "spec_sha256": spec_sha256,
            "ordinal": index + 1,
            "role": _ROLES[index],
            "candidate_sha256": candidate.sha256,
            "source_sha256": SOURCE_SHA256S[index],
            "batch": batch,
        },
    )
    selection = prepared.issue(candidate.sha256, intervention_sha256)
    execution = execute(prepared, selection, batch)
    if type(execution) is not RustSelectedActionExecutionV2:
        raise TypeError("C3 executor must return one exact RustSelectedActionExecutionV2")
    execution.authenticate_structure()
    if (
        selection.authority_sha256 != prepared.authority_sha256
        or selection.catalogue_sha256 != prepared.catalogue.sha256
        or selection.candidate_sha256 != candidate.sha256
        or selection.intervention_sha256 != intervention_sha256
        or execution.selected_source.authority_sha256 != prepared.authority_sha256
        or execution.selected_source.selection_sha256 != selection.sha256
        or execution.selected_source.action_handle != candidate.candidate_id
        or execution.selected_source.source != source
        or execution.selected_source.source_sha256 != SOURCE_SHA256S[index]
        or execution.compile_receipt.compile_invocations != 1
        or execution.run_receipt.run_invocations != 1
        or len(execution.outputs) != len(batch)
    ):
        raise ContinualRustTokenEditOutcomeReuseCohortError("C3 selected execution custody changed")
    return selection, execution


def _make_consequence(
    *,
    execution: RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
    feedback_sha256: str,
) -> token_edits.TokenEditConsequenceV1:
    return token_edits.make_token_edit_consequence_v1(
        source_sha256=execution.selected_source.source_sha256,
        batch_sha256=execution.preparation.batch_sha256,
        test_manifest_sha256=feedback_sha256,
        execution_sha256=execution.sha256,
        observed_outputs=execution.outputs,
        expected_outputs=expected_outputs,
    )


def _make_attempt(
    *,
    index: int,
    selection: interventions.RustActionInterventionSelectionV1,
    execution: RustSelectedActionExecutionV2,
    consequence: token_edits.TokenEditConsequenceV1,
) -> C3ExecutionAttemptV1:
    partial = C3ExecutionAttemptV1(
        ordinal=index + 1,
        role=_ROLES[index],
        source_sha256=SOURCE_SHA256S[index],
        selection=selection,
        execution=execution,
        consequence=consequence,
        expected_output_sha256=outcome_reuse_output_sha256_v1(consequence.expected_outputs),
        passed_cell=consequence.public_test_cell,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_ATTEMPT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _open_feedback(
    *,
    phase: str,
    callback: FeedbackCallback,
    expected_output_sha256: str,
    expected_cases: int,
    execution_sha256s: tuple[str, str],
    spec_sha256: str,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    outputs = _validate_outputs(callback(), expected_cases=expected_cases)
    if outcome_reuse_output_sha256_v1(outputs) != expected_output_sha256:
        raise ContinualRustTokenEditOutcomeReuseCohortError(
            f"C3 {phase} public feedback differs from its commitment"
        )
    feedback_sha256 = _digest(
        _FEEDBACK_NAMESPACE,
        {
            "spec_sha256": spec_sha256,
            "phase": phase,
            "execution_sha256s": execution_sha256s,
            "expected_output_sha256": expected_output_sha256,
        },
    )
    return outputs, feedback_sha256


def _seed_context_root(spec_sha256: str, label: str) -> str:
    return _digest(
        _SEED_CONTEXT_NAMESPACE,
        {"spec_sha256": spec_sha256, "label": label},
    )


def _run_reserved_cohort_v1(
    *,
    spec: C3OutcomeReuseCohortSpecV1,
    reservation: C3RunReservationV1,
    execute: ExecutionCallback,
    adaptation_feedback: FeedbackCallback,
    evaluation_feedback: FeedbackCallback,
) -> C3OutcomeReuseCohortResultV1:
    spec.authenticate()
    reservation.authenticate(destination=Path(reservation.destination))
    if (
        not callable(execute)
        or not callable(adaptation_feedback)
        or not callable(evaluation_feedback)
    ):
        raise TypeError("C3 execution and feedback seams must be callable")
    if token_edits.PARAMETERIZED_SCHEMAS[SCHEMA_INDEX].sha256 != SCHEMA_SHA256:
        raise ContinualRustTokenEditOutcomeReuseCohortError("C3 schema 3 changed")

    authorities: list[interventions.RustActionInterventionAuthorityV1] = []
    catalogues: list[interventions.RustActionInterventionCatalogueV1] = []
    authority_work_counters: list[interventions.RustActionInterventionWorkCountersV1] = []
    selections: list[interventions.RustActionInterventionSelectionV1] = []
    executions: list[RustSelectedActionExecutionV2] = []

    seed_prepared = _prepare_pair(0)
    authorities.append(seed_prepared.authority)
    catalogues.append(seed_prepared.catalogue)
    for index in (0, 1):
        selection, execution = _run_execution(
            prepared=seed_prepared,
            index=index,
            batch=spec.seed_batch,
            execute=execute,
            spec_sha256=spec.sha256,
        )
        selections.append(selection)
        executions.append(execution)
    authority_work_counters.append(_authenticate_pair_work(seed_prepared))
    seed_feedback_sha256 = _seed_context_root(spec.sha256, "seed-public-feedback")
    seed_consequences = tuple(
        _make_consequence(
            execution=execution,
            expected_outputs=spec.seed_expected_outputs,
            feedback_sha256=seed_feedback_sha256,
        )
        for execution in executions[:2]
    )
    seed_frames = tuple(
        token_edits.project_token_source_v1(source)
        for source in (SEED_FAILED_SOURCE, SEED_SUCCESSFUL_SOURCE)
    )
    seed_episode = token_edits.seal_failed_successful_token_edit_episode_v1(
        repository_sha256=_seed_context_root(spec.sha256, "repository"),
        task_sha256=_seed_context_root(spec.sha256, "task"),
        task_instance_sha256=_seed_context_root(spec.sha256, "task-instance"),
        task_source_sha256=_seed_context_root(spec.sha256, "task-source"),
        test_source_sha256=seed_feedback_sha256,
        failed_source=seed_frames[0],
        failed_consequence=seed_consequences[0],
        successful_source=seed_frames[1],
        successful_consequence=seed_consequences[1],
    )
    base_checkpoint = token_edits.advance_continual_token_edit_schema_v1(
        token_edits.start_continual_token_edit_schema_v1(),
        seed_episode,
    )
    if (
        base_checkpoint.support_counts[SCHEMA_INDEX] != 1
        or sum(base_checkpoint.support_counts) != 1
    ):
        raise ContinualRustTokenEditOutcomeReuseCohortError(
            "C3 seed did not create exactly one schema-3 support tick"
        )
    parent_policy = policy.start_continual_token_edit_consequence_policy_v1(base_checkpoint)

    adaptation_frame = token_edits.project_token_source_v1(ADAPTATION_SOURCE)
    adaptation_normal = parent_policy.propose(adaptation_frame)
    adaptation_exploration = parent_policy.propose(
        adaptation_frame,
        exploration_gate=True,
    )
    if (
        adaptation_normal.status != policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
        or adaptation_exploration.status != policy.EXPLORATION_READY
        or adaptation_exploration.selected_schema_index != SCHEMA_INDEX
        or adaptation_exploration.edited_source != ADAPTATION_EDITED_SOURCE
    ):
        raise ContinualRustTokenEditOutcomeReuseCohortError("C3 adaptation selection changed")

    adaptation_prepared = _prepare_pair(2)
    authorities.append(adaptation_prepared.authority)
    catalogues.append(adaptation_prepared.catalogue)
    for index in (2, 3):
        selection, execution = _run_execution(
            prepared=adaptation_prepared,
            index=index,
            batch=spec.adaptation_batch,
            execute=execute,
            spec_sha256=spec.sha256,
        )
        selections.append(selection)
        executions.append(execution)
    authority_work_counters.append(_authenticate_pair_work(adaptation_prepared))
    adaptation_expected, adaptation_feedback_sha256 = _open_feedback(
        phase="adaptation",
        callback=adaptation_feedback,
        expected_output_sha256=spec.adaptation_expected_output_sha256,
        expected_cases=len(spec.adaptation_batch),
        execution_sha256s=(executions[2].sha256, executions[3].sha256),
        spec_sha256=spec.sha256,
    )
    adaptation_consequences = tuple(
        _make_consequence(
            execution=execution,
            expected_outputs=adaptation_expected,
            feedback_sha256=adaptation_feedback_sha256,
        )
        for execution in executions[2:4]
    )
    if tuple(item.public_test_cell for item in adaptation_consequences) != (0, 1):
        raise ContinualRustTokenEditOutcomeReuseCohortError(
            "C3 adaptation baseline/proposal consequence changed"
        )
    adaptation_outcome = policy.seal_token_edit_consequence_policy_outcome_v1(
        proposal=adaptation_exploration,
        consequence=adaptation_consequences[1],
    )
    successor_policy = policy.advance_continual_token_edit_consequence_policy_v1(
        parent_policy,
        adaptation_outcome,
    )
    parent_root = parent_policy.checkpoint_sha256
    successor_root = successor_policy.checkpoint_sha256

    evaluation_frame = token_edits.project_token_source_v1(EVALUATION_SOURCE)
    evaluation_parent = parent_policy.propose(evaluation_frame)
    evaluation_outcome_off = successor_policy.propose(
        evaluation_frame,
        outcome_gate=False,
    )
    evaluation_successor = successor_policy.propose(evaluation_frame)
    if (
        evaluation_parent.status != policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
        or evaluation_outcome_off.status != policy.STOP_INSUFFICIENT_OUTCOME_EVIDENCE
        or evaluation_successor.status != policy.READY
        or evaluation_successor.selected_schema_index != SCHEMA_INDEX
        or evaluation_successor.edited_source != EVALUATION_EDITED_SOURCE
    ):
        raise ContinualRustTokenEditOutcomeReuseCohortError("C3 evaluation reuse selection changed")

    evaluation_prepared = _prepare_pair(4)
    authorities.append(evaluation_prepared.authority)
    catalogues.append(evaluation_prepared.catalogue)
    for index in (4, 5):
        selection, execution = _run_execution(
            prepared=evaluation_prepared,
            index=index,
            batch=spec.evaluation_batch,
            execute=execute,
            spec_sha256=spec.sha256,
        )
        selections.append(selection)
        executions.append(execution)
    authority_work_counters.append(_authenticate_pair_work(evaluation_prepared))
    evaluation_expected, evaluation_feedback_sha256 = _open_feedback(
        phase="evaluation",
        callback=evaluation_feedback,
        expected_output_sha256=spec.evaluation_expected_output_sha256,
        expected_cases=len(spec.evaluation_batch),
        execution_sha256s=(executions[4].sha256, executions[5].sha256),
        spec_sha256=spec.sha256,
    )
    evaluation_consequences = tuple(
        _make_consequence(
            execution=execution,
            expected_outputs=evaluation_expected,
            feedback_sha256=evaluation_feedback_sha256,
        )
        for execution in executions[4:6]
    )
    if tuple(item.public_test_cell for item in evaluation_consequences) != (0, 1):
        raise ContinualRustTokenEditOutcomeReuseCohortError(
            "C3 evaluation baseline/proposal consequence changed"
        )
    evaluation_outcome = policy.seal_token_edit_consequence_policy_outcome_v1(
        proposal=evaluation_successor,
        consequence=evaluation_consequences[1],
    )
    parent_policy.authenticate()
    successor_policy.authenticate()
    if (
        parent_policy.checkpoint_sha256 != parent_root
        or successor_policy.checkpoint_sha256 != successor_root
        or len(successor_policy.outcome_memory.provenance) != 1
    ):
        raise RuntimeError("C3 evaluation mutated a policy checkpoint")

    consequences = (
        *seed_consequences,
        *adaptation_consequences,
        *evaluation_consequences,
    )
    attempts = tuple(
        _make_attempt(
            index=index,
            selection=selections[index],
            execution=executions[index],
            consequence=consequences[index],
        )
        for index in range(6)
    )
    work = C3OutcomeReuseWorkV1(
        reservation_count=1,
        issued_selection_count=6,
        selected_source_execution_count=len(executions),
        compile_count=sum(item.compile_receipt.compile_invocations for item in executions),
        run_count=sum(item.run_receipt.run_invocations for item in executions),
        public_feedback_open_count=2,
        schema_advance_count=1,
        policy_advance_count=1,
        evaluation_policy_advance_count=0,
    )
    pass_cells = tuple(attempt.passed_cell for attempt in attempts)
    partial = C3OutcomeReuseCohortResultV1(
        status=STATUS_PASS if pass_cells == (0, 1, 0, 1, 0, 1) else STATUS_BEHAVIOR_FAIL,
        spec=spec,
        reservation=reservation,
        base_checkpoint=base_checkpoint,
        parent_policy=parent_policy,
        successor_policy=successor_policy,
        adaptation_normal=adaptation_normal,
        adaptation_exploration=adaptation_exploration,
        adaptation_outcome=adaptation_outcome,
        evaluation_parent=evaluation_parent,
        evaluation_outcome_off=evaluation_outcome_off,
        evaluation_successor=evaluation_successor,
        evaluation_outcome=evaluation_outcome,
        attempts=attempts,
        authorities=tuple(authorities),
        catalogues=tuple(catalogues),
        authority_work_counters=tuple(authority_work_counters),
        feedback_sha256s=(adaptation_feedback_sha256, evaluation_feedback_sha256),
        work=work,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _production_evidence_paths_v1() -> tuple[Path, Path]:
    module_path = Path(__file__).resolve(strict=True)
    result_path = module_path.parent / f"{module_path.stem}.result.json"
    reservation_path = module_path.parent / f"{result_path.name}.reservation.json"
    return result_path, reservation_path


DEFAULT_RESULT_PATH, DEFAULT_RESERVATION_PATH = _production_evidence_paths_v1()


def _preflight_rustc_v1() -> str:
    """Resolve and authenticate the compiler before a persistent reservation exists."""

    identity = rust_executor._discover_rustc()
    identity.authenticate_structure()
    return identity.sha256


def _reserve_destination(
    spec: C3OutcomeReuseCohortSpecV1,
    *,
    preregistration_commit: str,
    destination: Path,
    reservation_path: Path,
) -> C3RunReservationV1:
    spec.authenticate()
    if not _valid_commit(preregistration_commit):
        raise ValueError("C3 preregistration commit must be 40 lowercase hex characters")
    destination = Path(destination).resolve()
    reservation_path = Path(reservation_path).resolve()
    expected_reservation = destination.with_name(f"{destination.name}.reservation.json")
    if reservation_path != expected_reservation:
        raise ValueError("C3 reservation must be the result's exact sibling")
    if os.path.lexists(destination) or os.path.lexists(reservation_path):
        raise FileExistsError(destination if os.path.lexists(destination) else reservation_path)
    partial = C3RunReservationV1(
        schema=RESERVATION_SCHEMA,
        spec_sha256=spec.sha256,
        preregistration_commit=preregistration_commit,
        destination=str(destination),
        sha256="",
    )
    reservation = replace(
        partial,
        sha256=_digest(_RESERVATION_NAMESPACE, partial.record(include_sha256=False)),
    )
    reservation.authenticate(destination=destination)
    payload = _canonical_json(reservation.record()) + b"\n"
    with reservation_path.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("C3 reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if reservation_path.read_bytes() != payload:
        raise OSError("C3 reservation readback changed")
    return reservation


def _result_document(result: C3OutcomeReuseCohortResultV1) -> dict[str, object]:
    result.authenticate()
    transcript = {
        "authorities": _plain(result.authorities),
        "catalogues": _plain(result.catalogues),
        "authority_work_counters": _plain(result.authority_work_counters),
        "attempts": tuple(
            {
                "ordinal": attempt.ordinal,
                "role": attempt.role,
                "selection": _plain(attempt.selection),
                "execution": _plain(attempt.execution),
                "consequence": _plain(attempt.consequence),
            }
            for attempt in result.attempts
        ),
    }
    body = {
        "schema": RESULT_DOCUMENT_SCHEMA,
        "terminal_status": result.status,
        "result_sha256": result.sha256,
        "authenticated_result_record": result.record(),
        "full_authenticated_transcript": transcript,
    }
    return {**body, "document_sha256": _digest(_DOCUMENT_NAMESPACE, body)}


def _abort_document(
    *,
    spec: C3OutcomeReuseCohortSpecV1,
    reservation: C3RunReservationV1,
    error: Exception,
) -> dict[str, object]:
    body = {
        "schema": RESULT_DOCUMENT_SCHEMA,
        "terminal_status": STATUS_EXECUTION_ABORT,
        "spec_sha256": spec.sha256,
        "reservation_sha256": reservation.sha256,
        "exception_type": type(error).__qualname__,
        "exception_message_sha256": sha256(str(error).encode("utf-8")).hexdigest(),
    }
    abort_sha256 = _digest(_ABORT_NAMESPACE, body)
    rooted = {**body, "abort_sha256": abort_sha256}
    return {**rooted, "document_sha256": _digest(_DOCUMENT_NAMESPACE, rooted)}


def _publish_document(destination: Path, document: dict[str, object]) -> None:
    payload = _canonical_json(document) + b"\n"
    with destination.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("C3 terminal document write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = destination.read_bytes()
    if (
        retained != payload
        or _canonical_json(json.loads(retained.decode("ascii"))) + b"\n" != payload
    ):
        raise OSError("C3 terminal document readback changed")
    decoded = json.loads(retained.decode("ascii"))
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if document_sha256 != _digest(_DOCUMENT_NAMESPACE, body):
        raise ContinualRustTokenEditOutcomeReuseCohortError("C3 terminal document root changed")
    if decoded["terminal_status"] != STATUS_EXECUTION_ABORT:
        record = dict(decoded["authenticated_result_record"])
        result_sha256 = record.pop("sha256", None)
        if result_sha256 != decoded["result_sha256"] or result_sha256 != _digest(
            _RESULT_NAMESPACE, record
        ):
            raise ContinualRustTokenEditOutcomeReuseCohortError("C3 persisted result root changed")


def _run_and_write_at_paths_for_test_v1(
    *,
    preregistration_commit: str,
    destination: Path,
    reservation_path: Path,
    execute: ExecutionCallback,
    adaptation_feedback: FeedbackCallback,
    evaluation_feedback: FeedbackCallback,
    spec: C3OutcomeReuseCohortSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C3OutcomeReuseCohortResultV1:
    """Private injected test seam; production evidence paths remain fixed."""

    destination = Path(destination).resolve()
    reservation = _reserve_destination(
        spec,
        preregistration_commit=preregistration_commit,
        destination=destination,
        reservation_path=reservation_path,
    )
    try:
        result = _run_reserved_cohort_v1(
            spec=spec,
            reservation=reservation,
            execute=execute,
            adaptation_feedback=adaptation_feedback,
            evaluation_feedback=evaluation_feedback,
        )
        document = _result_document(result)
    except Exception as error:
        document = _abort_document(spec=spec, reservation=reservation, error=error)
        _publish_document(destination, document)
        raise
    _publish_document(destination, document)
    return result


def run_and_write_cohort_v1(
    *,
    preregistration_commit: str,
    spec: C3OutcomeReuseCohortSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C3OutcomeReuseCohortResultV1:
    """Reserve and spend the exact production C3 cohort once."""

    _preflight_rustc_v1()
    destination, reservation_path = _production_evidence_paths_v1()
    return _run_and_write_at_paths_for_test_v1(
        destination=destination,
        reservation_path=reservation_path,
        preregistration_commit=preregistration_commit,
        execute=_execute_selected,
        adaptation_feedback=lambda: ADAPTATION_EXPECTED_OUTPUTS,
        evaluation_feedback=lambda: EVALUATION_EXPECTED_OUTPUTS,
        spec=spec,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration-commit",
        required=True,
        help="40-character commit that froze the external C3 preregistration",
    )
    arguments = parser.parse_args(argv)
    result = run_and_write_cohort_v1(
        preregistration_commit=arguments.preregistration_commit,
    )
    print(_canonical_json({"status": result.status, "sha256": result.sha256}).decode())
    return 0 if result.status == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTATION_BASELINE_OUTPUTS",
    "ADAPTATION_BATCH",
    "ADAPTATION_EDITED_SOURCE",
    "ADAPTATION_EXPECTED_OUTPUTS",
    "ADAPTATION_SOURCE",
    "CLAIM_BOUNDARY",
    "DEFAULT_PUBLIC_SPEC",
    "DEFAULT_RESERVATION_PATH",
    "DEFAULT_RESULT_PATH",
    "EVALUATION_BASELINE_OUTPUTS",
    "EVALUATION_BATCH",
    "EVALUATION_EDITED_SOURCE",
    "EVALUATION_EXPECTED_OUTPUTS",
    "EVALUATION_SOURCE",
    "EXCLUDED_PRIOR_SOURCE_SHA256S",
    "FROZEN_LEARNER_CONFIG_SHA256",
    "FROZEN_MECHANISM_COMMIT",
    "FROZEN_MECHANISM_FILE_SHA256S",
    "RESULT_DOCUMENT_SCHEMA",
    "SCHEMA",
    "SCHEMA_INDEX",
    "SCHEMA_SHA256",
    "SEED_BATCH",
    "SEED_EXPECTED_OUTPUTS",
    "SEED_FAILED_OUTPUTS",
    "SEED_FAILED_SOURCE",
    "SEED_SUCCESSFUL_SOURCE",
    "SOURCE_SHA256S",
    "STATUS_BEHAVIOR_FAIL",
    "STATUS_EXECUTION_ABORT",
    "STATUS_PASS",
    "C3ExecutionAttemptV1",
    "C3OutcomeReuseCohortResultV1",
    "C3OutcomeReuseCohortSpecV1",
    "C3OutcomeReuseWorkV1",
    "C3RunReservationV1",
    "ContinualRustTokenEditOutcomeReuseCohortError",
    "main",
    "make_c3_outcome_reuse_cohort_spec_v1",
    "outcome_reuse_output_sha256_v1",
    "run_and_write_cohort_v1",
]
