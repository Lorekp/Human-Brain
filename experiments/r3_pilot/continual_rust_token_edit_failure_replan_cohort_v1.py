"""Fresh C4 failure-conditioned rerace cohort for two supplied edit schemas.

The cohort executes schema A once, opens its public consequence, writes one
durable FAIL tick, and uses the generic ordinal-1 refractory trace to rerace the
unchanged A/B action tuple.  The selected schema B then executes and writes one
durable PASS tick.  A fresh evaluation executes the parent-selected A and the
final-memory-selected B before opening one shared public consequence.  Four
selected Rust executions are retained; evaluation does not update memory.
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

from experiments.r3_pilot import common_continual_action_outcome_memory_v1 as outcomes
from experiments.r3_pilot import common_continual_token_edit_schema_v1 as token_edits
from experiments.r3_pilot import common_failure_conditioned_action_replan_v1 as replans
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_actions
from experiments.r3_pilot import common_rust_action_intervention_v1 as interventions
from humanbrain import rust_selected_action_executor as rust_executor
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

SCHEMA = "humanbrain.continual-rust-token-edit-failure-replan-cohort.v1"
RESULT_DOCUMENT_SCHEMA = (
    "humanbrain.continual-rust-token-edit-failure-replan-cohort-result-document.v1"
)
RESERVATION_SCHEMA = "humanbrain.continual-rust-token-edit-failure-replan-cohort-reservation.v1"
STATUS_PASS = "PASS_C4_TOKEN_EDIT_FAILURE_REPLAN_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C4_TOKEN_EDIT_FAILURE_REPLAN_V1"
STATUS_EXECUTION_ABORT = "ABORT_C4_TOKEN_EDIT_FAILURE_REPLAN_V1"
CLAIM_BOUNDARY = (
    "one-fresh-supplied-two-schema-context-where-one-selected-only-rust-executed-"
    "caller-attested-failure-writes-one-durable-inhibitory-tick-and-one-deterministic-"
    "ordinal-one-refractory-pulse-reraces-the-unchanged-action-tuple-to-a-rival-whose-"
    "independent-pass-completes-durable-revision-not-failure-diagnosis-rival-correctness-"
    "schema-discovery-source-synthesis-generic-debugging-protected-transfer-swe-or-"
    "intelligence"
)

FROZEN_MECHANISM_COMMIT = "11e29b61aed6a5f6c381c7470b38bf5f2effe915"
FROZEN_LEARNER_CONFIG_SHA256 = "84b60787deed505352885fad33abe58ba8b0d6c1d16cee688a7d19da0eb912a5"
FROZEN_MECHANISM_FILE_SHA256S = (
    (
        "experiments/r3_pilot/common_failure_conditioned_action_replan_v1.py",
        "0f5f3ff4dbe4230aa76225d6d957c2a9db28edb2b16d55a8a98f306848444413",
    ),
    (
        "experiments/r3_pilot/common_continual_action_outcome_memory_v1.py",
        "7aaf0a71eb968f48a1e683f064a0c2aa6a04c84e778f7cb510c0421ff78605f5",
    ),
    (
        "experiments/r3_pilot/common_continual_token_edit_schema_v1.py",
        "a0380e8ec30c55887a5e2b72c835d701d597a3ad90701d3e6c07cabed67ef443",
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

ACTION_SCHEMA_INDICES = (2, 3)
ACTION_SHA256S = (
    "610269c84c10d687d7efcb0f435c982bcf39a97d295b58fbe43a14e41efb04f5",
    "7cf44511bced68635e3c77a199542061b404b5861fe6a51c7c6ba88b01c2b014",
)
ACTION_BASE_SCORES = (6, 4)
ADAPTATION_BINDINGS = ((2, 48, 49, 45), (3, 48, 49, 52))
EVALUATION_BINDINGS = ((2, 51, 52, 48), (3, 51, 52, 55))

ADAPTATION_BATCH = ((-2, 0, 1, 3, 5, 6, 9), (7, -4, 2))
ADAPTATION_SCHEMA_A_OUTPUTS = ((1, 1, 0, 0, 0, 0, 0), (0, 1, 0))
ADAPTATION_SCHEMA_B_OUTPUTS = ((1, 1, 1, 1, 1, 0, 0), (0, 1, 1))
ADAPTATION_EXPECTED_OUTPUTS = ADAPTATION_SCHEMA_B_OUTPUTS
EVALUATION_BATCH = ((-6, -3, -2, 0, 3, 4, 7), (5, -9, 1))
EVALUATION_SCHEMA_A_OUTPUTS = ((8, -8, -8, -8, -8, -8, -8), (-8, 8, -8))
EVALUATION_SCHEMA_B_OUTPUTS = ((8, 8, 8, 8, 8, -8, -8), (-8, 8, 8))
EVALUATION_EXPECTED_OUTPUTS = EVALUATION_SCHEMA_B_OUTPUTS

ADAPTATION_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let floor = 1_i64;
    let ceiling = 6_i64;
    samples
        .iter()
        .copied()
        .map(|datum| {
            if { floor; datum < datum && { ceiling; true } } {
                1
            } else {
                0
            }
        })
        .collect()
}
"""

ADAPTATION_SCHEMA_A_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let floor = 1_i64;
    let ceiling = 6_i64;
    samples
        .iter()
        .copied()
        .map(|datum| {
            if { floor; datum < floor && { ceiling; true } } {
                1
            } else {
                0
            }
        })
        .collect()
}
"""

ADAPTATION_SCHEMA_B_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let floor = 1_i64;
    let ceiling = 6_i64;
    samples
        .iter()
        .copied()
        .map(|datum| {
            if { floor; datum < ceiling && { ceiling; true } } {
                1
            } else {
                0
            }
        })
        .collect()
}
"""

EVALUATION_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let pivot = -3_i64;
    let limit = 4_i64;
    samples
        .iter()
        .copied()
        .filter_map(|signal| {
            Some(if { pivot; signal < signal && { limit; true } } {
                8
            } else {
                -8
            })
        })
        .collect()
}
"""

EVALUATION_SCHEMA_A_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let pivot = -3_i64;
    let limit = 4_i64;
    samples
        .iter()
        .copied()
        .filter_map(|signal| {
            Some(if { pivot; signal < pivot && { limit; true } } {
                8
            } else {
                -8
            })
        })
        .collect()
}
"""

EVALUATION_SCHEMA_B_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let pivot = -3_i64;
    let limit = 4_i64;
    samples
        .iter()
        .copied()
        .filter_map(|signal| {
            Some(if { pivot; signal < limit && { limit; true } } {
                8
            } else {
                -8
            })
        })
        .collect()
}
"""

SOURCE_SHA256S = (
    "652e21dfb435032e94d7f0b72e5c5ed3c28faf2f4ba32c09cb203a57d5c785c9",
    "b0cfe61f4d7c74758f0e7b165011ae660210f26079274a4179e585ad9279f742",
    "d34b307aa6cef6e403ee88698897f73dfdb21511aee7b1b629baf5631eaab6d6",
    "5ea76f081fe59993eaabd0da33cf9f2b1f8429ff3ea876c3f6a86fb2f6892c69",
    "e6f89d9d2be762471863ff3c6a8b00dd3eef59ad388eed55388883198385429f",
    "182c0e0084471185b6d0cd16271bc43b1eb7561a49a8a099bbbf4027abe9117f",
)
_ALL_SOURCES = (
    ADAPTATION_SOURCE,
    ADAPTATION_SCHEMA_A_SOURCE,
    ADAPTATION_SCHEMA_B_SOURCE,
    EVALUATION_SOURCE,
    EVALUATION_SCHEMA_A_SOURCE,
    EVALUATION_SCHEMA_B_SOURCE,
)
_EXECUTION_SOURCES = (
    ADAPTATION_SCHEMA_A_SOURCE,
    ADAPTATION_SCHEMA_B_SOURCE,
    EVALUATION_SCHEMA_A_SOURCE,
    EVALUATION_SCHEMA_B_SOURCE,
)
EXECUTION_SOURCE_SHA256S = (
    SOURCE_SHA256S[1],
    SOURCE_SHA256S[2],
    SOURCE_SHA256S[4],
    SOURCE_SHA256S[5],
)
_ROLES = (
    "ADAPTATION_SCHEMA_A",
    "ADAPTATION_SCHEMA_B",
    "EVALUATION_PARENT_SCHEMA_A",
    "EVALUATION_FINAL_SCHEMA_B",
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
            "bda397623caaf27ecef7e73e0b6052a0c12c3ff419333ef1f7231a1e011485d3",
            "514720a04d50420dbc4d1da9be1c24da37be8102009ecdcaa7259fc114795112",
            "be1bc451971f347f22d0072a3411d15a54f00c1d4c819f1be9d575330455625d",
            "d907300fcf3390c3b8697a8437f8448cc10057736ac1bdee3c27084f2a7b0b7f",
            "4588278d064b626f0a9c6e6c6067992d64ac547d09e1ff15e386f9d441f2d1b7",
            "b2ab91c571c3ab154735d98bb584502ba0039030372c33a2b75ad86992931319",
        )
    )
)

ADAPTATION_SCHEMA_A_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_d2", "opaque_c4.slot_d2()"
)
ADAPTATION_SCHEMA_B_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_k7", "opaque_c4.slot_k7()"
)
EVALUATION_SCHEMA_A_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_p4", "opaque_c4.slot_p4()"
)
EVALUATION_SCHEMA_B_CANDIDATE = knowledge_actions.project_knowledge_action_candidate_v1(
    "slot_x9", "opaque_c4.slot_x9()"
)
_CANDIDATES = (
    ADAPTATION_SCHEMA_A_CANDIDATE,
    ADAPTATION_SCHEMA_B_CANDIDATE,
    EVALUATION_SCHEMA_A_CANDIDATE,
    EVALUATION_SCHEMA_B_CANDIDATE,
)

_SPEC_NAMESPACE = b"c4-token-edit-failure-replan-spec-v1"
_OUTPUT_NAMESPACE = b"c4-token-edit-failure-replan-output-v1"
_INTERVENTION_NAMESPACE = b"c4-token-edit-failure-replan-intervention-v1"
_FEEDBACK_NAMESPACE = b"c4-token-edit-failure-replan-feedback-v1"
_CONTEXT_NAMESPACE = b"c4-token-edit-failure-replan-context-v1"
_RESERVATION_NAMESPACE = b"c4-token-edit-failure-replan-reservation-v1"
_ATTEMPT_NAMESPACE = b"c4-token-edit-failure-replan-attempt-v1"
_RESULT_NAMESPACE = b"c4-token-edit-failure-replan-result-v1"
_DOCUMENT_NAMESPACE = b"c4-token-edit-failure-replan-document-v1"
_ABORT_NAMESPACE = b"c4-token-edit-failure-replan-abort-v1"


class ContinualRustTokenEditFailureReplanCohortError(ValueError):
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
    raise TypeError(f"C4 transcript cannot serialize {type(value).__qualname__}")


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
        raise TypeError("C4 outputs must be one exact bounded integer tuple batch")
    return outputs


def failure_replan_output_sha256_v1(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(_OUTPUT_NAMESPACE, _validate_outputs(outputs))


def _validate_batch(batch: object) -> tuple[tuple[int, ...], ...]:
    checked = _validate_outputs(batch)
    if not checked:
        raise ContinualRustTokenEditFailureReplanCohortError("C4 batch must be nonempty")
    return checked


def _application_record(application: object) -> tuple[int, int, int, int]:
    if application is None:
        raise ContinualRustTokenEditFailureReplanCohortError("C4 supplied schema is inapplicable")
    return (
        application.schema_index,
        application.anchor_token_index,
        application.target_token_index,
        application.donor_token_index,
    )


def _project_action_sources(
    source: str,
) -> tuple[token_edits.TokenSourceFrameV1, tuple[object, object]]:
    frame = token_edits.project_token_source_v1(source)
    applications = tuple(
        token_edits._apply_schema(token_edits.PARAMETERIZED_SCHEMAS[index], frame)
        for index in ACTION_SCHEMA_INDICES
    )
    if any(application is None for application in applications):
        raise ContinualRustTokenEditFailureReplanCohortError(
            "C4 supplied schema pair is not uniquely applicable"
        )
    return frame, applications


@dataclass(frozen=True, slots=True)
class C4FailureReplanCohortSpecV1:
    frozen_mechanism_commit: str
    frozen_learner_config_sha256: str
    frozen_mechanism_file_sha256s: tuple[tuple[str, str], ...]
    action_schema_indices: tuple[int, int]
    action_sha256s: tuple[str, str]
    action_base_scores: tuple[int, int]
    adaptation_bindings: tuple[tuple[int, int, int, int], ...]
    evaluation_bindings: tuple[tuple[int, int, int, int], ...]
    adaptation_batch: tuple[tuple[int, ...], ...]
    adaptation_output_sha256s: tuple[str, str]
    evaluation_batch: tuple[tuple[int, ...], ...]
    evaluation_output_sha256s: tuple[str, str]
    source_sha256s: tuple[str, ...]
    candidate_sha256s: tuple[str, ...]
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
        adaptation_frame, adaptation_applications = _project_action_sources(ADAPTATION_SOURCE)
        evaluation_frame, evaluation_applications = _project_action_sources(EVALUATION_SOURCE)
        source_roots = tuple(sha256(source.encode("ascii")).hexdigest() for source in _ALL_SOURCES)
        candidate_roots = tuple(candidate.sha256 for candidate in _CANDIDATES)
        derived_adaptation_sources = tuple(
            application.edited_source.source for application in adaptation_applications
        )
        derived_evaluation_sources = tuple(
            application.edited_source.source for application in evaluation_applications
        )
        if (
            self.frozen_mechanism_commit != FROZEN_MECHANISM_COMMIT
            or self.frozen_learner_config_sha256 != FROZEN_LEARNER_CONFIG_SHA256
            or self.frozen_learner_config_sha256 != token_edits.CONFIG_SHA256
            or self.frozen_mechanism_file_sha256s != FROZEN_MECHANISM_FILE_SHA256S
            or self.frozen_mechanism_file_sha256s != _current_mechanism_file_sha256s()
            or self.action_schema_indices != ACTION_SCHEMA_INDICES
            or self.action_sha256s != ACTION_SHA256S
            or self.action_sha256s
            != tuple(
                token_edits.PARAMETERIZED_SCHEMAS[index].sha256 for index in ACTION_SCHEMA_INDICES
            )
            or self.action_base_scores != ACTION_BASE_SCORES
            or self.adaptation_bindings
            != tuple(_application_record(item) for item in adaptation_applications)
            or self.evaluation_bindings
            != tuple(_application_record(item) for item in evaluation_applications)
            or adaptation_frame.source != ADAPTATION_SOURCE
            or derived_adaptation_sources
            != (ADAPTATION_SCHEMA_A_SOURCE, ADAPTATION_SCHEMA_B_SOURCE)
            or evaluation_frame.source != EVALUATION_SOURCE
            or derived_evaluation_sources
            != (EVALUATION_SCHEMA_A_SOURCE, EVALUATION_SCHEMA_B_SOURCE)
            or _validate_batch(self.adaptation_batch) != ADAPTATION_BATCH
            or self.adaptation_output_sha256s
            != tuple(
                failure_replan_output_sha256_v1(outputs)
                for outputs in (ADAPTATION_SCHEMA_A_OUTPUTS, ADAPTATION_SCHEMA_B_OUTPUTS)
            )
            or _validate_batch(self.evaluation_batch) != EVALUATION_BATCH
            or self.evaluation_output_sha256s
            != tuple(
                failure_replan_output_sha256_v1(outputs)
                for outputs in (EVALUATION_SCHEMA_A_OUTPUTS, EVALUATION_SCHEMA_B_OUTPUTS)
            )
            or self.source_sha256s != SOURCE_SHA256S
            or self.source_sha256s != source_roots
            or len(set(source_roots)) != 6
            or self.candidate_sha256s != candidate_roots
            or len(set(candidate_roots)) != 4
            or self.excluded_source_sha256s != EXCLUDED_PRIOR_SOURCE_SHA256S
            or len(self.excluded_source_sha256s) != 24
            or set(source_roots).intersection(self.excluded_source_sha256s)
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditFailureReplanCohortError(
                "C4 public fixture specification changed"
            )


def make_c4_failure_replan_cohort_spec_v1() -> C4FailureReplanCohortSpecV1:
    partial = C4FailureReplanCohortSpecV1(
        frozen_mechanism_commit=FROZEN_MECHANISM_COMMIT,
        frozen_learner_config_sha256=FROZEN_LEARNER_CONFIG_SHA256,
        frozen_mechanism_file_sha256s=FROZEN_MECHANISM_FILE_SHA256S,
        action_schema_indices=ACTION_SCHEMA_INDICES,
        action_sha256s=ACTION_SHA256S,
        action_base_scores=ACTION_BASE_SCORES,
        adaptation_bindings=ADAPTATION_BINDINGS,
        evaluation_bindings=EVALUATION_BINDINGS,
        adaptation_batch=ADAPTATION_BATCH,
        adaptation_output_sha256s=tuple(
            failure_replan_output_sha256_v1(outputs)
            for outputs in (ADAPTATION_SCHEMA_A_OUTPUTS, ADAPTATION_SCHEMA_B_OUTPUTS)
        ),
        evaluation_batch=EVALUATION_BATCH,
        evaluation_output_sha256s=tuple(
            failure_replan_output_sha256_v1(outputs)
            for outputs in (EVALUATION_SCHEMA_A_OUTPUTS, EVALUATION_SCHEMA_B_OUTPUTS)
        ),
        source_sha256s=SOURCE_SHA256S,
        candidate_sha256s=tuple(candidate.sha256 for candidate in _CANDIDATES),
        excluded_source_sha256s=EXCLUDED_PRIOR_SOURCE_SHA256S,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_SPEC_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


DEFAULT_PUBLIC_SPEC = make_c4_failure_replan_cohort_spec_v1()


@dataclass(frozen=True, slots=True)
class C4RunReservationV1:
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
            raise ContinualRustTokenEditFailureReplanCohortError(
                "C4 run reservation authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C4ExecutionAttemptV1:
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
            "observed_output_sha256": failure_replan_output_sha256_v1(self.execution.outputs),
            "expected_output_sha256": self.expected_output_sha256,
            "passed_cell": self.passed_cell,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.selection) is not interventions.RustActionInterventionSelectionV1:
            raise TypeError("C4 attempt selection must be exact")
        if type(self.execution) is not RustSelectedActionExecutionV2:
            raise TypeError("C4 attempt execution must be exact V2")
        if type(self.consequence) is not token_edits.TokenEditConsequenceV1:
            raise TypeError("C4 attempt consequence must be exact")
        self.selection.authenticate_structure()
        self.execution.authenticate_structure()
        self.consequence.authenticate()
        passed = int(self.execution.outputs == self.consequence.expected_outputs)
        candidate = _CANDIDATES[self.ordinal - 1] if self.ordinal in range(1, 5) else None
        if (
            self.ordinal not in range(1, 5)
            or self.role != _ROLES[self.ordinal - 1]
            or self.source_sha256 != EXECUTION_SOURCE_SHA256S[self.ordinal - 1]
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
            != failure_replan_output_sha256_v1(self.consequence.expected_outputs)
            or self.passed_cell != passed
            or self.consequence.public_test_cell != passed
            or self.sha256 != _digest(_ATTEMPT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditFailureReplanCohortError(
                "C4 execution attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C4FailureReplanWorkV1:
    reservation_count: int
    issued_selection_count: int
    selected_source_execution_count: int
    compile_count: int
    run_count: int
    public_feedback_open_count: int
    outcome_advance_count: int
    replan_count: int
    combined_gate_off_control_count: int
    evaluation_outcome_advance_count: int

    def record(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

    def authenticate(self) -> None:
        if self.record() != {
            "reservation_count": 1,
            "issued_selection_count": 4,
            "selected_source_execution_count": 4,
            "compile_count": 4,
            "run_count": 4,
            "public_feedback_open_count": 2,
            "outcome_advance_count": 2,
            "replan_count": 1,
            "combined_gate_off_control_count": 1,
            "evaluation_outcome_advance_count": 0,
        }:
            raise ContinualRustTokenEditFailureReplanCohortError("C4 work changed")


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
        raise ContinualRustTokenEditFailureReplanCohortError(
            "C4 intervention authority work differs from 2/2/2/2/2"
        )
    return counters


def _decision_scores(decision: outcomes.ActionDecisionV1) -> dict[str, int]:
    return dict(
        zip(
            decision.activation.active_action_sha256s,
            decision.effective_scores,
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class C4FailureReplanCohortResultV1:
    status: str
    spec: C4FailureReplanCohortSpecV1
    reservation: C4RunReservationV1
    parent_memory: outcomes.FrozenContinualActionOutcomeMemoryV1
    failure_outcome: outcomes.SelectedActionOutcomeV1
    refractory_off: replans.FailureConditionedActionReplanV1
    gate_off: replans.FailureConditionedActionReplanV1
    replan: replans.FailureConditionedActionReplanV1
    success_outcome: outcomes.SelectedActionOutcomeV1
    final_memory: outcomes.FrozenContinualActionOutcomeMemoryV1
    evaluation_parent: outcomes.ActionDecisionV1
    evaluation_final: outcomes.ActionDecisionV1
    attempts: tuple[C4ExecutionAttemptV1, ...]
    authorities: tuple[interventions.RustActionInterventionAuthorityV1, ...]
    catalogues: tuple[interventions.RustActionInterventionCatalogueV1, ...]
    authority_work_counters: tuple[interventions.RustActionInterventionWorkCountersV1, ...]
    feedback_sha256s: tuple[str, str]
    work: C4FailureReplanWorkV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "status": self.status,
            "spec_sha256": self.spec.sha256,
            "reservation_sha256": self.reservation.sha256,
            "preregistration_commit": self.reservation.preregistration_commit,
            "parent_checkpoint_sha256": self.parent_memory.checkpoint_sha256,
            "failure_outcome_sha256": self.failure_outcome.sha256,
            "refractory_off_sha256": self.refractory_off.sha256,
            "gate_off_sha256": self.gate_off.sha256,
            "replan_sha256": self.replan.sha256,
            "success_outcome_sha256": self.success_outcome.sha256,
            "final_checkpoint_sha256": self.final_memory.checkpoint_sha256,
            "evaluation_parent_sha256": self.evaluation_parent.sha256,
            "evaluation_final_sha256": self.evaluation_final.sha256,
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
        self.parent_memory.authenticate()
        self.failure_outcome.authenticate()
        self.refractory_off.authenticate()
        self.gate_off.authenticate()
        self.replan.authenticate()
        self.success_outcome.authenticate()
        self.final_memory.authenticate()
        self.evaluation_parent.authenticate()
        self.evaluation_final.authenticate()
        for attempt in self.attempts:
            attempt.authenticate()
        for authority in self.authorities:
            authority.authenticate_structure()
        for catalogue in self.catalogues:
            catalogue.authenticate_structure()
        for counters in self.authority_work_counters:
            counters.authenticate_structure()
        self.work.authenticate()

        pass_cells = tuple(attempt.passed_cell for attempt in self.attempts)
        action_a, action_b = ACTION_SHA256S
        action_rows = {root: index for index, root in enumerate(self.final_memory.action_sha256s)}
        expected_final = outcomes.advance_continual_action_outcome_memory_v1(
            self.replan.successor,
            self.success_outcome,
        )
        retained_pair_links = tuple(
            (attempt.selection.authority_sha256, attempt.selection.catalogue_sha256)
            for attempt in self.attempts
        )
        expected_pair_links = tuple(
            (authority.sha256, catalogue.sha256)
            for authority, catalogue in zip(self.authorities, self.catalogues, strict=True)
            for _ in range(2)
        )
        if (
            self.status != (STATUS_PASS if pass_cells == (0, 1, 0, 1) else STATUS_BEHAVIOR_FAIL)
            or len(self.attempts) != 4
            or tuple(attempt.ordinal for attempt in self.attempts) != (1, 2, 3, 4)
            or len({attempt.execution.sha256 for attempt in self.attempts}) != 4
            or len({attempt.selection.sha256 for attempt in self.attempts}) != 4
            or len(self.authorities) != 2
            or len(self.catalogues) != 2
            or len(self.authority_work_counters) != 2
            or retained_pair_links != expected_pair_links
            or any(
                authority.catalogue_sha256 != catalogue.sha256
                for authority, catalogue in zip(self.authorities, self.catalogues, strict=True)
            )
            or any(
                counters != _EXPECTED_PAIR_WORK_COUNTERS
                for counters in self.authority_work_counters
            )
            or any(not _valid_sha256(value) for value in self.feedback_sha256s)
            or self.parent_memory.action_sha256s != ACTION_SHA256S
            or any(self.parent_memory.excitatory_ticks + self.parent_memory.inhibitory_ticks)
            or self.failure_outcome.passed_cell != 0
            or self.failure_outcome.decision.status != outcomes.READY
            or self.failure_outcome.selected_action_sha256 != action_a
            or _decision_scores(self.failure_outcome.decision) != {action_a: 6, action_b: 4}
            or self.failure_outcome.proposed_source_sha256 != self.attempts[0].source_sha256
            or self.failure_outcome.execution_sha256 != self.attempts[0].execution.sha256
            or self.failure_outcome.consequence_sha256 != self.attempts[0].consequence.sha256
            or self.refractory_off.parent is not self.parent_memory
            or self.refractory_off.status != replans.REFRACTORY_OFF
            or self.refractory_off.decision is None
            or self.refractory_off.decision.status != outcomes.STOP_PHYSICAL_RACE
            or _decision_scores(self.refractory_off.decision) != {action_a: 4, action_b: 4}
            or self.gate_off.parent is not self.parent_memory
            or self.gate_off.status != replans.STOP_GATE_OFF
            or self.gate_off.successor is not self.parent_memory
            or self.gate_off.retry_activation is not None
            or self.gate_off.decision is not None
            or self.replan.parent is not self.parent_memory
            or self.replan.status != replans.REFRACTORY_RERACED
            or self.replan.retry_activation != self.refractory_off.retry_activation
            or self.replan.successor.checkpoint_sha256
            != self.refractory_off.successor.checkpoint_sha256
            or self.replan.decision is None
            or self.replan.decision.status != outcomes.READY
            or self.replan.decision.selected_action_sha256 != action_b
            or self.replan.decision.selected_action_sha256
            == self.failure_outcome.selected_action_sha256
            or _decision_scores(self.replan.decision) != {action_a: 2, action_b: 4}
            or self.success_outcome.decision is not self.replan.decision
            or self.success_outcome.passed_cell != 1
            or self.success_outcome.selected_action_sha256 != action_b
            or self.success_outcome.proposed_source_sha256 != self.attempts[1].source_sha256
            or self.success_outcome.execution_sha256 != self.attempts[1].execution.sha256
            or self.success_outcome.consequence_sha256 != self.attempts[1].consequence.sha256
            or self.final_memory.checkpoint_sha256 != expected_final.checkpoint_sha256
            or len(self.final_memory.provenance) != 2
            or self.final_memory.inhibitory_ticks[action_rows[action_a]] != 1
            or self.final_memory.excitatory_ticks[action_rows[action_b]] != 1
            or sum(self.final_memory.inhibitory_ticks) != 1
            or sum(self.final_memory.excitatory_ticks) != 1
            or self.evaluation_parent.checkpoint_sha256 != self.parent_memory.checkpoint_sha256
            or self.evaluation_parent.status != outcomes.READY
            or self.evaluation_parent.selected_action_sha256 != action_a
            or _decision_scores(self.evaluation_parent) != {action_a: 6, action_b: 4}
            or self.evaluation_final.checkpoint_sha256 != self.final_memory.checkpoint_sha256
            or self.evaluation_final.status != outcomes.READY
            or self.evaluation_final.selected_action_sha256 != action_b
            or _decision_scores(self.evaluation_final) != {action_a: 4, action_b: 6}
            or self.evaluation_parent.activation != self.evaluation_final.activation
            or self.attempts[0].consequence.test_manifest_sha256 != self.feedback_sha256s[0]
            or self.attempts[1].consequence.test_manifest_sha256 != self.feedback_sha256s[0]
            or self.attempts[2].consequence.test_manifest_sha256 != self.feedback_sha256s[1]
            or self.attempts[3].consequence.test_manifest_sha256 != self.feedback_sha256s[1]
            or self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustTokenEditFailureReplanCohortError(
                "C4 cohort result authentication failed"
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


def _prepare_pair(first_index: int) -> interventions.PreparedRustActionInterventionV1:
    return interventions.prepare_rust_action_intervention_v1(
        (
            (_CANDIDATES[first_index], _EXECUTION_SOURCES[first_index]),
            (_CANDIDATES[first_index + 1], _EXECUTION_SOURCES[first_index + 1]),
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
    source = _EXECUTION_SOURCES[index]
    intervention_sha256 = _digest(
        _INTERVENTION_NAMESPACE,
        {
            "spec_sha256": spec_sha256,
            "ordinal": index + 1,
            "role": _ROLES[index],
            "candidate_sha256": candidate.sha256,
            "source_sha256": EXECUTION_SOURCE_SHA256S[index],
            "batch": batch,
        },
    )
    selection = prepared.issue(candidate.sha256, intervention_sha256)
    execution = execute(prepared, selection, batch)
    if type(execution) is not RustSelectedActionExecutionV2:
        raise TypeError("C4 executor must return one exact RustSelectedActionExecutionV2")
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
        or execution.selected_source.source_sha256 != EXECUTION_SOURCE_SHA256S[index]
        or execution.compile_receipt.compile_invocations != 1
        or execution.run_receipt.run_invocations != 1
        or len(execution.outputs) != len(batch)
    ):
        raise ContinualRustTokenEditFailureReplanCohortError(
            "C4 selected execution custody changed"
        )
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
) -> C4ExecutionAttemptV1:
    partial = C4ExecutionAttemptV1(
        ordinal=index + 1,
        role=_ROLES[index],
        source_sha256=EXECUTION_SOURCE_SHA256S[index],
        selection=selection,
        execution=execution,
        consequence=consequence,
        expected_output_sha256=failure_replan_output_sha256_v1(consequence.expected_outputs),
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
    expected_outputs: tuple[tuple[int, ...], ...],
    expected_cases: int,
    execution_sha256s: tuple[str, ...],
    spec_sha256: str,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    outputs = _validate_outputs(callback(), expected_cases=expected_cases)
    expected_root = failure_replan_output_sha256_v1(expected_outputs)
    if failure_replan_output_sha256_v1(outputs) != expected_root:
        raise ContinualRustTokenEditFailureReplanCohortError(
            f"C4 {phase} public feedback differs from its commitment"
        )
    feedback_sha256 = _digest(
        _FEEDBACK_NAMESPACE,
        {
            "spec_sha256": spec_sha256,
            "phase": phase,
            "execution_sha256s": execution_sha256s,
            "expected_output_sha256": expected_root,
        },
    )
    return outputs, feedback_sha256


def _context_root(spec_sha256: str, label: str) -> str:
    return _digest(_CONTEXT_NAMESPACE, {"spec_sha256": spec_sha256, "label": label})


def _activation(spec_sha256: str, label: str) -> outcomes.ActionActivationV1:
    return outcomes.seal_action_activation_v1(
        selection_context_sha256=_context_root(spec_sha256, label),
        active_action_sha256s=ACTION_SHA256S,
        base_scores=ACTION_BASE_SCORES,
    )


def _seal_generic_outcome(
    decision: outcomes.ActionDecisionV1,
    consequence: token_edits.TokenEditConsequenceV1,
) -> outcomes.SelectedActionOutcomeV1:
    return outcomes.seal_selected_action_outcome_v1(
        decision=decision,
        proposed_source_sha256=consequence.source_sha256,
        execution_sha256=consequence.execution_sha256,
        consequence_sha256=consequence.sha256,
        auxiliary_event_sha256s=(consequence.feedback_sha256,),
        passed_cell=consequence.public_test_cell,
    )


def _run_reserved_cohort_v1(
    *,
    spec: C4FailureReplanCohortSpecV1,
    reservation: C4RunReservationV1,
    execute: ExecutionCallback,
    adaptation_feedback: FeedbackCallback,
    evaluation_feedback: FeedbackCallback,
) -> C4FailureReplanCohortResultV1:
    spec.authenticate()
    reservation.authenticate(destination=Path(reservation.destination))
    if (
        not callable(execute)
        or not callable(adaptation_feedback)
        or not callable(evaluation_feedback)
    ):
        raise TypeError("C4 execution and feedback seams must be callable")
    _project_action_sources(ADAPTATION_SOURCE)
    _project_action_sources(EVALUATION_SOURCE)

    selections: list[interventions.RustActionInterventionSelectionV1] = []
    executions: list[RustSelectedActionExecutionV2] = []
    consequences: list[token_edits.TokenEditConsequenceV1] = []
    authorities: list[interventions.RustActionInterventionAuthorityV1] = []
    catalogues: list[interventions.RustActionInterventionCatalogueV1] = []
    authority_work_counters: list[interventions.RustActionInterventionWorkCountersV1] = []

    parent_memory = outcomes.start_continual_action_outcome_memory_v1(ACTION_SHA256S)
    adaptation_decision = parent_memory.decide(_activation(spec.sha256, "adaptation-initial"))
    if (
        adaptation_decision.status != outcomes.READY
        or adaptation_decision.selected_action_sha256 != ACTION_SHA256S[0]
    ):
        raise ContinualRustTokenEditFailureReplanCohortError(
            "C4 adaptation did not initially select schema A"
        )

    adaptation_prepared = _prepare_pair(0)
    authorities.append(adaptation_prepared.authority)
    catalogues.append(adaptation_prepared.catalogue)
    selection_a, execution_a = _run_execution(
        prepared=adaptation_prepared,
        index=0,
        batch=spec.adaptation_batch,
        execute=execute,
        spec_sha256=spec.sha256,
    )
    selections.append(selection_a)
    executions.append(execution_a)
    adaptation_expected, adaptation_feedback_sha256 = _open_feedback(
        phase="adaptation",
        callback=adaptation_feedback,
        expected_outputs=ADAPTATION_EXPECTED_OUTPUTS,
        expected_cases=len(spec.adaptation_batch),
        execution_sha256s=(execution_a.sha256,),
        spec_sha256=spec.sha256,
    )
    consequence_a = _make_consequence(
        execution=execution_a,
        expected_outputs=adaptation_expected,
        feedback_sha256=adaptation_feedback_sha256,
    )
    consequences.append(consequence_a)
    if consequence_a.public_test_cell != 0:
        raise ContinualRustTokenEditFailureReplanCohortError("C4 adaptation schema A did not fail")
    failure_outcome = _seal_generic_outcome(adaptation_decision, consequence_a)
    refractory_off = replans.advance_and_replan_after_failure_v1(
        parent_memory,
        failure_outcome,
        refractory_gate=False,
    )
    gate_off = replans.advance_and_replan_after_failure_v1(
        parent_memory,
        failure_outcome,
        plasticity_gate=False,
        consequence_gate=False,
    )
    replan = replans.advance_and_replan_after_failure_v1(parent_memory, failure_outcome)
    if (
        refractory_off.decision is None
        or refractory_off.decision.status != outcomes.STOP_PHYSICAL_RACE
        or replan.decision is None
        or replan.decision.status != outcomes.READY
        or replan.decision.selected_action_sha256 != ACTION_SHA256S[1]
    ):
        raise ContinualRustTokenEditFailureReplanCohortError(
            "C4 refractory rerace geometry changed"
        )

    selection_b, execution_b = _run_execution(
        prepared=adaptation_prepared,
        index=1,
        batch=spec.adaptation_batch,
        execute=execute,
        spec_sha256=spec.sha256,
    )
    selections.append(selection_b)
    executions.append(execution_b)
    consequence_b = _make_consequence(
        execution=execution_b,
        expected_outputs=adaptation_expected,
        feedback_sha256=adaptation_feedback_sha256,
    )
    consequences.append(consequence_b)
    if consequence_b.public_test_cell != 1:
        raise ContinualRustTokenEditFailureReplanCohortError("C4 adaptation schema B did not pass")
    success_outcome = _seal_generic_outcome(replan.decision, consequence_b)
    final_memory = outcomes.advance_continual_action_outcome_memory_v1(
        replan.successor,
        success_outcome,
    )
    authority_work_counters.append(_authenticate_pair_work(adaptation_prepared))

    evaluation_activation = _activation(spec.sha256, "evaluation-fresh")
    evaluation_parent = parent_memory.decide(evaluation_activation)
    evaluation_final = final_memory.decide(evaluation_activation)
    if (
        evaluation_parent.status != outcomes.READY
        or evaluation_parent.selected_action_sha256 != ACTION_SHA256S[0]
        or evaluation_final.status != outcomes.READY
        or evaluation_final.selected_action_sha256 != ACTION_SHA256S[1]
    ):
        raise ContinualRustTokenEditFailureReplanCohortError(
            "C4 fresh evaluation selection changed"
        )

    evaluation_prepared = _prepare_pair(2)
    authorities.append(evaluation_prepared.authority)
    catalogues.append(evaluation_prepared.catalogue)
    for index in (2, 3):
        decision = evaluation_parent if index == 2 else evaluation_final
        expected_action = ACTION_SHA256S[index - 2]
        if decision.selected_action_sha256 != expected_action:
            raise ContinualRustTokenEditFailureReplanCohortError(
                "C4 evaluation source is not selected by its decision"
            )
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
        expected_outputs=EVALUATION_EXPECTED_OUTPUTS,
        expected_cases=len(spec.evaluation_batch),
        execution_sha256s=(executions[2].sha256, executions[3].sha256),
        spec_sha256=spec.sha256,
    )
    for execution in executions[2:4]:
        consequences.append(
            _make_consequence(
                execution=execution,
                expected_outputs=evaluation_expected,
                feedback_sha256=evaluation_feedback_sha256,
            )
        )
    if tuple(item.public_test_cell for item in consequences[2:4]) != (0, 1):
        raise ContinualRustTokenEditFailureReplanCohortError(
            "C4 evaluation A/B consequences changed"
        )

    attempts = tuple(
        _make_attempt(
            index=index,
            selection=selections[index],
            execution=executions[index],
            consequence=consequences[index],
        )
        for index in range(4)
    )
    work = C4FailureReplanWorkV1(
        reservation_count=1,
        issued_selection_count=4,
        selected_source_execution_count=len(executions),
        compile_count=sum(item.compile_receipt.compile_invocations for item in executions),
        run_count=sum(item.run_receipt.run_invocations for item in executions),
        public_feedback_open_count=2,
        outcome_advance_count=2,
        replan_count=1,
        combined_gate_off_control_count=1,
        evaluation_outcome_advance_count=0,
    )
    pass_cells = tuple(attempt.passed_cell for attempt in attempts)
    partial = C4FailureReplanCohortResultV1(
        status=STATUS_PASS if pass_cells == (0, 1, 0, 1) else STATUS_BEHAVIOR_FAIL,
        spec=spec,
        reservation=reservation,
        parent_memory=parent_memory,
        failure_outcome=failure_outcome,
        refractory_off=refractory_off,
        gate_off=gate_off,
        replan=replan,
        success_outcome=success_outcome,
        final_memory=final_memory,
        evaluation_parent=evaluation_parent,
        evaluation_final=evaluation_final,
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
    spec: C4FailureReplanCohortSpecV1,
    *,
    preregistration_commit: str,
    destination: Path,
    reservation_path: Path,
) -> C4RunReservationV1:
    spec.authenticate()
    if not _valid_commit(preregistration_commit):
        raise ValueError("C4 preregistration commit must be 40 lowercase hex characters")
    destination = Path(destination).resolve()
    reservation_path = Path(reservation_path).resolve()
    expected_reservation = destination.with_name(f"{destination.name}.reservation.json")
    if reservation_path != expected_reservation:
        raise ValueError("C4 reservation must be the result's exact sibling")
    if os.path.lexists(destination) or os.path.lexists(reservation_path):
        raise FileExistsError(destination if os.path.lexists(destination) else reservation_path)
    partial = C4RunReservationV1(
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
            raise OSError("C4 reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if reservation_path.read_bytes() != payload:
        raise OSError("C4 reservation readback changed")
    return reservation


def _result_document(result: C4FailureReplanCohortResultV1) -> dict[str, object]:
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
    spec: C4FailureReplanCohortSpecV1,
    reservation: C4RunReservationV1,
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
            raise OSError("C4 terminal document write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = destination.read_bytes()
    if (
        retained != payload
        or _canonical_json(json.loads(retained.decode("ascii"))) + b"\n" != payload
    ):
        raise OSError("C4 terminal document readback changed")
    decoded = json.loads(retained.decode("ascii"))
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if document_sha256 != _digest(_DOCUMENT_NAMESPACE, body):
        raise ContinualRustTokenEditFailureReplanCohortError("C4 terminal document root changed")
    if decoded["terminal_status"] != STATUS_EXECUTION_ABORT:
        record = dict(decoded["authenticated_result_record"])
        result_sha256 = record.pop("sha256", None)
        if result_sha256 != decoded["result_sha256"] or result_sha256 != _digest(
            _RESULT_NAMESPACE, record
        ):
            raise ContinualRustTokenEditFailureReplanCohortError("C4 persisted result root changed")


def _run_and_write_at_paths_for_test_v1(
    *,
    preregistration_commit: str,
    destination: Path,
    reservation_path: Path,
    execute: ExecutionCallback,
    adaptation_feedback: FeedbackCallback,
    evaluation_feedback: FeedbackCallback,
    spec: C4FailureReplanCohortSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C4FailureReplanCohortResultV1:
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
    spec: C4FailureReplanCohortSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C4FailureReplanCohortResultV1:
    """Preflight, reserve, and spend the exact production C4 cohort once."""

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
        help="40-character commit that froze the external C4 preregistration",
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
    "ACTION_BASE_SCORES",
    "ACTION_SCHEMA_INDICES",
    "ACTION_SHA256S",
    "ADAPTATION_BATCH",
    "ADAPTATION_BINDINGS",
    "ADAPTATION_EXPECTED_OUTPUTS",
    "ADAPTATION_SCHEMA_A_OUTPUTS",
    "ADAPTATION_SCHEMA_A_SOURCE",
    "ADAPTATION_SCHEMA_B_OUTPUTS",
    "ADAPTATION_SCHEMA_B_SOURCE",
    "ADAPTATION_SOURCE",
    "CLAIM_BOUNDARY",
    "DEFAULT_PUBLIC_SPEC",
    "DEFAULT_RESERVATION_PATH",
    "DEFAULT_RESULT_PATH",
    "EVALUATION_BATCH",
    "EVALUATION_BINDINGS",
    "EVALUATION_EXPECTED_OUTPUTS",
    "EVALUATION_SCHEMA_A_OUTPUTS",
    "EVALUATION_SCHEMA_A_SOURCE",
    "EVALUATION_SCHEMA_B_OUTPUTS",
    "EVALUATION_SCHEMA_B_SOURCE",
    "EVALUATION_SOURCE",
    "EXCLUDED_PRIOR_SOURCE_SHA256S",
    "EXECUTION_SOURCE_SHA256S",
    "FROZEN_LEARNER_CONFIG_SHA256",
    "FROZEN_MECHANISM_COMMIT",
    "FROZEN_MECHANISM_FILE_SHA256S",
    "RESULT_DOCUMENT_SCHEMA",
    "SCHEMA",
    "SOURCE_SHA256S",
    "STATUS_BEHAVIOR_FAIL",
    "STATUS_EXECUTION_ABORT",
    "STATUS_PASS",
    "C4ExecutionAttemptV1",
    "C4FailureReplanCohortResultV1",
    "C4FailureReplanCohortSpecV1",
    "C4FailureReplanWorkV1",
    "C4RunReservationV1",
    "ContinualRustTokenEditFailureReplanCohortError",
    "failure_replan_output_sha256_v1",
    "main",
    "make_c4_failure_replan_cohort_spec_v1",
    "run_and_write_cohort_v1",
]
