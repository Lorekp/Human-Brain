"""Fresh C6 rustc-diagnostic factor selection followed by text materialization.

Two public TRAIN episodes teach diagonal error-class/source-region associations
inside the generic factorized memory.  On a fresh cross diagnostic, the memory
must select before a generic authority materializes exactly one supplied
operator/target edit and releases it to the V2 Rust executor.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_factorized_text_edit_materializer_v1 as materializer
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as adapter
from humanbrain import rust_selected_action_executor as rust_executor
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

SCHEMA = "humanbrain.continual-rust-diagnostic-factor-materialization-cohort.v1"
RESULT_DOCUMENT_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-factor-materialization-result-document.v1"
)
RESERVATION_SCHEMA = "humanbrain.continual-rust-diagnostic-factor-materialization-reservation.v1"
STATUS_PASS = "PASS_C6_RUSTC_DIAGNOSTIC_FACTOR_MATERIALIZATION_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C6_RUSTC_DIAGNOSTIC_FACTOR_MATERIALIZATION_V1"
STATUS_EXECUTION_ABORT = "ABORT_C6_RUSTC_DIAGNOSTIC_FACTOR_MATERIALIZATION_V1"
CLAIM_BOUNDARY = (
    "one-fresh-public-train-cohort-where-two-caller-attested-diagonal-rustc-code-"
    "and-hand-built-source-half-lessons-enable-one-unseen-cross-hypothesis-"
    "selection-before-a-generic-authority-post-selects-and-materializes-exactly-"
    "one-host-supplied-operator-target-text-edit-and-executes-it-successfully-"
    "with-no-alternative-materialization-or-evaluation-update-not-learned-cues-"
    "ontology-bindings-localization-diagnosis-edit-discovery-source-synthesis-"
    "search-debugging-transfer-swe-or-intelligence"
)

FROZEN_MECHANISM_COMMIT = "a6db0190d20fb9be8c5429691a3fefaab65b461b"
FROZEN_MECHANISM_FILE_SHA256S = (
    (
        "experiments/r3_pilot/common_factorized_feedback_hypothesis_memory_v1.py",
        "f17ef6861a160643216c955d170827d746fa58fad391b76d4fbdfa6b183377df",
    ),
    (
        "experiments/r3_pilot/common_rustc_factorized_feedback_adapter_v1.py",
        "e64d29d6173fb9a66692c1f9fc53885181004d56c842d5d44f61315c1958b40b",
    ),
    (
        "experiments/r3_pilot/common_continual_action_outcome_memory_v1.py",
        "7aaf0a71eb968f48a1e683f064a0c2aa6a04c84e778f7cb510c0421ff78605f5",
    ),
    (
        "experiments/r3_pilot/common_factorized_text_edit_materializer_v1.py",
        "4aac92d2cacba1e49d556d60263c55dc1d9a1ba8719d2abccf8910d40ba0d791",
    ),
    (
        "src/humanbrain/rust_selected_action_executor.py",
        "e1bfaa702b87f883d80b4c928f1dd067770b89e310d004b68a48f09f6a37227a",
    ),
)

SOURCE_FILE_NAME = "case.rs"
DIAGNOSTIC_ARGUMENTS = (
    "--edition",
    "2021",
    "--crate-name",
    "c6_diagnostic",
    "--crate-type",
    "lib",
    "--color",
    "never",
    "--error-format=json",
    "-Dwarnings",
    "--emit",
    "metadata",
    SOURCE_FILE_NAME,
)
ROLES = ("train-e0425-first", "train-e0308-second", "evaluation-e0425-second")
EXPECTED_DIAGNOSTICS = (
    ("E0425", adapter.FIRST_HALF),
    ("E0308", adapter.SECOND_HALF),
    ("E0425", adapter.SECOND_HALF),
)
EXPECTED_PRIMARY_BYTE_RANGES = ((190, 209), (383, 406), (331, 349))
ALLOW_EXPLORATION_BY_EPISODE = (True, True, False)
BATCHES = (((-9, -1, 5),), ((-12, 0, 7),), ((-20, 3, 10),))
EXPECTED_OUTPUTS = (((8, 16, 22),), ((1, 13, 20),), ((-3, 20, 27),))

TRAIN_E0425_FIRST_SOURCE = """\
struct TrainABox {
    _c6_train_a_missing: i64,
}

pub fn solve(samples: &[i64]) -> Vec<i64> {
    let first_bias: i64 = {
        let _c6_train_a_first_anchor = 6_i64;
        TrainABox { _c6_train_a_missing }._c6_train_a_missing
    };
    // C6_TRAIN_A_REGION_SPLIT
    let second_bias: i64 = {
        let _c6_train_a_second_anchor = 11_i64;
        _c6_train_a_second_anchor
    };
    samples
        .iter()
        .copied()
        .map(|value| value + first_bias + second_bias)
        .collect()
}
"""

TRAIN_E0308_SECOND_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let _c6_train_b_wrong_value = "thirteen";
    let first_bias: i64 = {
        let _c6_train_b_first_anchor = 4_i64;
        _c6_train_b_first_anchor
    };
    // C6_TRAIN_B_REGION_SPLIT
    let second_bias: i64 = {
        let _c6_train_b_second_anchor = 9_i64;
        let _c6_train_b_length = _c6_train_b_wrong_value.len();
        _c6_train_b_wrong_value
    };
    samples
        .iter()
        .copied()
        .map(|value| value + first_bias + second_bias)
        .collect()
}
"""

EVALUATION_E0425_SECOND_SOURCE = """\
struct EvalCBox {
    _c6_eval_c_missing: i64,
}

pub fn solve(samples: &[i64]) -> Vec<i64> {
    let first_bias: i64 = {
        let _c6_eval_c_first_anchor = 5_i64;
        _c6_eval_c_first_anchor
    };
    // C6_EVAL_C_REGION_SPLIT
    let second_bias: i64 = {
        let _c6_eval_c_second_anchor = 12_i64;
        EvalCBox { _c6_eval_c_missing }._c6_eval_c_missing
    };
    samples
        .iter()
        .copied()
        .map(|value| value + first_bias + second_bias)
        .collect()
}
"""

ORIGINAL_SOURCES = (
    TRAIN_E0425_FIRST_SOURCE,
    TRAIN_E0308_SECOND_SOURCE,
    EVALUATION_E0425_SECOND_SOURCE,
)
ORIGINAL_SOURCE_SHA256S = (
    "fbeac8ce5b168255e52c24318106a00194f7a5999c21bc453987227fddb5cb94",
    "3a76a6783fda0b2aef342124467e2fb1245ecea140052760dabed7b10d10799b",
    "cdddc8926d80232b494132e1687f0e344b92c0fdb5c667ed4a7051462ee72762",
)
TARGET_MARKERS = (
    ("_c6_train_a_first_anchor", "_c6_train_a_second_anchor"),
    ("_c6_train_b_first_anchor", "_c6_train_b_second_anchor"),
    ("_c6_eval_c_first_anchor", "_c6_eval_c_second_anchor"),
)

_SHA256_HEX = frozenset("0123456789abcdef")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_NAMESPACE = b"c6-rustc-diagnostic-factor-materialization-spec-v1"
_COMPONENT_NAMESPACE = b"c6-rustc-diagnostic-factor-component-v1"
_CONTEXT_NAMESPACE = b"c6-rustc-diagnostic-factor-context-v1"
_OUTPUT_NAMESPACE = b"c6-rustc-diagnostic-factor-output-v1"
_OUTPUT_FEEDBACK_NAMESPACE = b"c6-rustc-diagnostic-factor-output-feedback-v1"
_CONSEQUENCE_NAMESPACE = b"c6-rustc-diagnostic-factor-consequence-v1"
_ATTEMPT_NAMESPACE = b"c6-rustc-diagnostic-factor-selected-attempt-v1"
_WORK_NAMESPACE = b"c6-rustc-diagnostic-factor-work-v1"
_RESERVATION_NAMESPACE = b"c6-rustc-diagnostic-factor-reservation-v1"
_RESULT_NAMESPACE = b"c6-rustc-diagnostic-factor-result-v1"
_FAILURE_NAMESPACE = b"c6-rustc-diagnostic-factor-failure-v1"
_DOCUMENT_NAMESPACE = b"c6-rustc-diagnostic-factor-document-v1"


class ContinualRustDiagnosticFactorMaterializationCohortV1Error(ValueError):
    """The frozen C6 cohort or its authenticated transcript changed."""


class C6BehaviorFailure(RuntimeError):
    """A rooted scientific failure after the one-shot reservation was spent."""

    def __init__(self, reason: str, partial_transcript: dict[str, object]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial_transcript = partial_transcript


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


def _valid_commit(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in _SHA256_HEX for character in value)
    )


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
    raise TypeError(f"C6 transcript cannot serialize {type(value).__qualname__}")


def _behavior_failure(reason: str, **partial_transcript: object) -> C6BehaviorFailure:
    return C6BehaviorFailure(
        reason,
        {key: _plain(value) for key, value in partial_transcript.items()},
    )


OPERATOR_COMPONENT_SHA256S = (
    _digest(_COMPONENT_NAMESPACE, {"axis": 0, "value": "E0425"}),
    _digest(_COMPONENT_NAMESPACE, {"axis": 0, "value": "E0308"}),
)
TARGET_COMPONENT_SHA256S = (
    _digest(_COMPONENT_NAMESPACE, {"axis": 1, "value": adapter.FIRST_HALF}),
    _digest(_COMPONENT_NAMESPACE, {"axis": 1, "value": adapter.SECOND_HALF}),
)
HYPOTHESES_BY_KEY = {
    (operator_index, target_index): factors.seal_factorized_hypothesis_v1(
        (
            OPERATOR_COMPONENT_SHA256S[operator_index],
            TARGET_COMPONENT_SHA256S[target_index],
        )
    )
    for operator_index in range(2)
    for target_index in range(2)
}
HYPOTHESES = tuple(HYPOTHESES_BY_KEY[key] for key in ((0, 0), (0, 1), (1, 0), (1, 1)))
H00_SHA256 = HYPOTHESES_BY_KEY[(0, 0)].sha256
H01_SHA256 = HYPOTHESES_BY_KEY[(0, 1)].sha256
H10_SHA256 = HYPOTHESES_BY_KEY[(1, 0)].sha256
H11_SHA256 = HYPOTHESES_BY_KEY[(1, 1)].sha256
EXPLORATION_HYPOTHESIS_SHA256S = (H00_SHA256, H11_SHA256)
EVALUATION_HYPOTHESIS_SHA256 = H01_SHA256

OPERATOR_BINDINGS = tuple(
    sorted(
        (
            (
                OPERATOR_COMPONENT_SHA256S[0],
                materializer.DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR,
            ),
            (
                OPERATOR_COMPONENT_SHA256S[1],
                materializer.REPLACE_PRIMARY_SPAN_WITH_ANCHOR,
            ),
        )
    )
)
TARGET_BINDINGS_BY_EPISODE = tuple(
    tuple(
        sorted(
            (
                (TARGET_COMPONENT_SHA256S[0], first_marker),
                (TARGET_COMPONENT_SHA256S[1], second_marker),
            )
        )
    )
    for first_marker, second_marker in TARGET_MARKERS
)

EXCLUDED_PRIOR_SOURCE_SHA256S = (
    "08ffbca3d58b73e663e4516b0f9c564e1c6a2b4de33df0434c3ee933260c2312",
    "09b937e053872172813c09637fa96b4f19775ecda6911f61b1431b413ccec8b3",
    "0e131b7e04223b7ba8bbd504e8f4d1b158474d68fe0ec3a8f383a1cd76835739",
    "118db3017e71bd5380cfa8bc3a06da5ea47fe038d4ccc3f8e5440ce51afb7fbc",
    "11fdcb5cf4d1ab242cabbc6ce844f0b2da3f5fa03a57f8aa2f2ecb7f7e410fb4",
    "126b0ce796c248396e311e31a332e28dcdfbb328e0b994d64b39a1f6e3e0877b",
    "12ab6fb3efa6d3bf43b5d3903a743e004a1218656072dab32b08a3daf31fc7f8",
    "13405df1dce5d55512a918165f5102de5526b3fef5ad3c5fa95435a717bf712b",
    "1494cd3c8887257e8bc2cc82aa07c35ff8a7f1371887f1f517e63be8f77ea68d",
    "16504bb28d57532c6621aa0a36c6db23204d54fe7ea0c56132efa8e38b234704",
    "182c0e0084471185b6d0cd16271bc43b1eb7561a49a8a099bbbf4027abe9117f",
    "197a5fcbb6bb71b839994f70b43718610a8ebd43e35dd1050afb3bc7d1dc6572",
    "22d555f484f9aa33e1f5c184c5d91893aac1b02593ca5f1e4d28f1d9187418db",
    "4588278d064b626f0a9c6e6c6067992d64ac547d09e1ff15e386f9d441f2d1b7",
    "4ecf619642ddf0dee5466334c755544630f9d3656aea073c55e3a30777f4b79f",
    "514720a04d50420dbc4d1da9be1c24da37be8102009ecdcaa7259fc114795112",
    "5ea76f081fe59993eaabd0da33cf9f2b1f8429ff3ea876c3f6a86fb2f6892c69",
    "5efc39f66d45b1252a76ddd536a251d42e5956aaa84f61e3a2763b7d1d2a3d79",
    "652e21dfb435032e94d7f0b72e5c5ed3c28faf2f4ba32c09cb203a57d5c785c9",
    "66703237df4e8f68e7f9e678239970c93e218e7267bae1aeb4b8833e7a475a93",
    "767ac8ba99b72cfc2b148644c6e02f015dfe12bef19a1946288ccf4ae608f9cb",
    "83d7dbb8d55f0f9b06d6750ac4a3552981ec30700f9f65e11c753cf76a5d0f49",
    "8b225ebb888c282c0e6293215cdaecb36427efcbe3ca7317bbf322e5e1888c49",
    "913097c4d9e858c41d80a8d671a5996340a9d3071d1b1902350bb4403a236f88",
    "9156378419e5a875a0aa88632bbada13c3261e07ca979ce1ea9afbf650025975",
    "962a5bcd733f4026a07dea5342aa5145ea837bd282edbf67d41418da0f37f6dd",
    "9634b257c00cc1f0e2470cead58cc6ffec8136fb1a2388511b8c5f17b01dd9fc",
    "967382da9e579783e7fe5cb5cbf3c4e1a92f5d8ca84948aa641d0f289f62fb8d",
    "96f90f7f7d3e6a3677ce8cd1e7ddb3cc9986a49d42180a5f74b2375a1b6ed985",
    "a35874ffedbaabac33977d554916c049a823434b35d57e5ba82be0770ab088a7",
    "a54d2fe9f99aaebfbe7cd98fe991c26142bea39a172f9ffa3c05117af17876a0",
    "a5979ba178a792517f4e675fba66fe1a61f96c1a383e8453f32c5320f22fd04a",
    "a8102ce87ef9940510e2921ce401d3a66aa1b6ac08a6865fc84b733bc322cd71",
    "b0cfe61f4d7c74758f0e7b165011ae660210f26079274a4179e585ad9279f742",
    "b0e5bff5168441980089f2114b942d695b8d3496d6a67b86c96f7ffc8ca25ff7",
    "b2ab91c571c3ab154735d98bb584502ba0039030372c33a2b75ad86992931319",
    "b3cbeea34d564dd583a681142ad27d672c4ca64f9a7386fbb6c319b868c6cf0e",
    "b8bda9c33ba7d62f481387d0da28042c91a651a1e0c99b717b868a1f9078d81a",
    "bb29c7ac1a1b36d79131fc7dac4c3264b3fb1e053c5c82e4c49d706e0f412bd4",
    "bda397623caaf27ecef7e73e0b6052a0c12c3ff419333ef1f7231a1e011485d3",
    "be1bc451971f347f22d0072a3411d15a54f00c1d4c819f1be9d575330455625d",
    "c5bc2f2af0a579325f857006e222f4f59f4f44bd58eceaa0dd8e82baba16f9c5",
    "ca96a9774be460daed557117e0b56fc334d29d709d9fac0a6f53003e3fd9be34",
    "d00939b3ee635b6011a6e3296550dfef6fa1fa1d3211f42514822300bb37332c",
    "d23bee65de3f003933e9739add066b9c1973bb5760abfc6f15129b9b1fe169e3",
    "d335b523ec67621ba6925c48c84f1d4cce2f38f00956099297db443d1fe80f50",
    "d34b307aa6cef6e403ee88698897f73dfdb21511aee7b1b629baf5631eaab6d6",
    "d907300fcf3390c3b8697a8437f8448cc10057736ac1bdee3c27084f2a7b0b7f",
    "e6f89d9d2be762471863ff3c6a8b00dd3eef59ad388eed55388883198385429f",
    "e8dd412b73a6612ab6a70483ef738881ec09b2d9c429638cffab630648bcb3e8",
    "ee0b4c24a158c1d1e66bfbeaa5fcdccdaef641b5bd1b161f9a73054f9915e675",
)


def _current_mechanism_file_sha256s() -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, sha256((_REPO_ROOT / path).read_bytes()).hexdigest())
        for path, _ in FROZEN_MECHANISM_FILE_SHA256S
    )


def output_sha256_v1(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(_OUTPUT_NAMESPACE, {"outputs": outputs})


@dataclass(frozen=True, slots=True)
class C6DiagnosticFactorMaterializationSpecV1:
    frozen_mechanism_commit: str
    frozen_mechanism_file_sha256s: tuple[tuple[str, str], ...]
    diagnostic_arguments: tuple[str, ...]
    expected_diagnostics: tuple[tuple[str, str], ...]
    expected_primary_byte_ranges: tuple[tuple[int, int], ...]
    allow_exploration_by_episode: tuple[bool, bool, bool]
    batches: tuple[tuple[tuple[int, ...], ...], ...]
    expected_output_sha256s: tuple[str, ...]
    original_source_sha256s: tuple[str, ...]
    operator_bindings: tuple[tuple[str, str], ...]
    target_bindings_by_episode: tuple[tuple[tuple[str, str], ...], ...]
    hypothesis_sha256s: tuple[str, ...]
    exploration_hypothesis_sha256s: tuple[str, str]
    evaluation_hypothesis_sha256: str
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
        derived_source_roots = tuple(
            sha256(source.encode("ascii")).hexdigest() for source in ORIGINAL_SOURCES
        )
        hypothesis_roots = tuple(value.sha256 for value in HYPOTHESES)
        if (
            self.frozen_mechanism_commit != FROZEN_MECHANISM_COMMIT
            or not _valid_commit(self.frozen_mechanism_commit)
            or self.frozen_mechanism_file_sha256s != FROZEN_MECHANISM_FILE_SHA256S
            or self.frozen_mechanism_file_sha256s != _current_mechanism_file_sha256s()
            or self.diagnostic_arguments != DIAGNOSTIC_ARGUMENTS
            or self.expected_diagnostics != EXPECTED_DIAGNOSTICS
            or self.expected_primary_byte_ranges != EXPECTED_PRIMARY_BYTE_RANGES
            or self.allow_exploration_by_episode != ALLOW_EXPLORATION_BY_EPISODE
            or self.batches != BATCHES
            or self.expected_output_sha256s
            != tuple(output_sha256_v1(value) for value in EXPECTED_OUTPUTS)
            or self.original_source_sha256s != ORIGINAL_SOURCE_SHA256S
            or self.original_source_sha256s != derived_source_roots
            or len(set(derived_source_roots)) != 3
            or self.operator_bindings != OPERATOR_BINDINGS
            or self.target_bindings_by_episode != TARGET_BINDINGS_BY_EPISODE
            or len(self.target_bindings_by_episode) != 3
            or self.hypothesis_sha256s != hypothesis_roots
            or len(set(hypothesis_roots)) != 4
            or self.exploration_hypothesis_sha256s != EXPLORATION_HYPOTHESIS_SHA256S
            or self.evaluation_hypothesis_sha256 != EVALUATION_HYPOTHESIS_SHA256
            or self.excluded_source_sha256s != EXCLUDED_PRIOR_SOURCE_SHA256S
            or tuple(sorted(set(self.excluded_source_sha256s))) != self.excluded_source_sha256s
            or len(self.excluded_source_sha256s) != 51
            or set(derived_source_roots).intersection(self.excluded_source_sha256s)
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 public fixture specification changed"
            )


def make_c6_diagnostic_factor_materialization_spec_v1() -> C6DiagnosticFactorMaterializationSpecV1:
    partial = C6DiagnosticFactorMaterializationSpecV1(
        frozen_mechanism_commit=FROZEN_MECHANISM_COMMIT,
        frozen_mechanism_file_sha256s=FROZEN_MECHANISM_FILE_SHA256S,
        diagnostic_arguments=DIAGNOSTIC_ARGUMENTS,
        expected_diagnostics=EXPECTED_DIAGNOSTICS,
        expected_primary_byte_ranges=EXPECTED_PRIMARY_BYTE_RANGES,
        allow_exploration_by_episode=ALLOW_EXPLORATION_BY_EPISODE,
        batches=BATCHES,
        expected_output_sha256s=tuple(output_sha256_v1(value) for value in EXPECTED_OUTPUTS),
        original_source_sha256s=ORIGINAL_SOURCE_SHA256S,
        operator_bindings=OPERATOR_BINDINGS,
        target_bindings_by_episode=TARGET_BINDINGS_BY_EPISODE,
        hypothesis_sha256s=tuple(value.sha256 for value in HYPOTHESES),
        exploration_hypothesis_sha256s=EXPLORATION_HYPOTHESIS_SHA256S,
        evaluation_hypothesis_sha256=EVALUATION_HYPOTHESIS_SHA256,
        excluded_source_sha256s=EXCLUDED_PRIOR_SOURCE_SHA256S,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_SPEC_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


DEFAULT_PUBLIC_SPEC = make_c6_diagnostic_factor_materialization_spec_v1()


@dataclass(frozen=True, slots=True)
class C6RunReservationV1:
    schema: str
    spec_sha256: str
    preregistration_commit: str
    rustc_identity_sha256: str
    destination: str
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

    def authenticate(self, *, destination: Path) -> None:
        if (
            self.schema != RESERVATION_SCHEMA
            or not _valid_sha256(self.spec_sha256)
            or not _valid_commit(self.preregistration_commit)
            or not _valid_sha256(self.rustc_identity_sha256)
            or self.destination != str(Path(destination).resolve())
            or self.sha256 != _digest(_RESERVATION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 run reservation changed"
            )


@dataclass(frozen=True, slots=True)
class C6PublicConsequenceV1:
    source_sha256: str
    batch_sha256: str
    output_feedback_sha256: str
    execution_sha256: str
    observed_outputs: tuple[tuple[int, ...], ...]
    expected_outputs: tuple[tuple[int, ...], ...]
    passed_cell: int
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
            any(
                not _valid_sha256(value)
                for value in (
                    self.source_sha256,
                    self.batch_sha256,
                    self.output_feedback_sha256,
                    self.execution_sha256,
                )
            )
            or type(self.observed_outputs) is not tuple
            or type(self.expected_outputs) is not tuple
            or any(
                type(row) is not tuple or any(type(value) is not int for value in row)
                for outputs in (self.observed_outputs, self.expected_outputs)
                for row in outputs
            )
            or type(self.passed_cell) is not int
            or self.passed_cell not in (0, 1)
            or self.passed_cell != int(self.observed_outputs == self.expected_outputs)
            or self.sha256 != _digest(_CONSEQUENCE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 public consequence authentication failed"
            )


def _seal_consequence(
    *,
    execution: RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
    output_feedback_sha256: str,
) -> C6PublicConsequenceV1:
    payload = {
        "source_sha256": execution.selected_source.source_sha256,
        "batch_sha256": execution.preparation.batch_sha256,
        "output_feedback_sha256": output_feedback_sha256,
        "execution_sha256": execution.sha256,
        "observed_outputs": execution.outputs,
        "expected_outputs": expected_outputs,
        "passed_cell": int(execution.outputs == expected_outputs),
    }
    result = C6PublicConsequenceV1(
        **payload,
        sha256=_digest(_CONSEQUENCE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C6SelectedExecutionAttemptV1:
    ordinal: int
    role: str
    decision_sha256: str
    selected_hypothesis_sha256: str
    authority_receipt: materializer.FactorizedTextEditAuthorityReceiptV1
    pre_route_counts: tuple[int, int]
    pre_execute_counts: tuple[int, int]
    post_grant_counts: tuple[int, int]
    materialization: materializer.FactorizedTextEditMaterializationV1
    release: materializer.FactorizedTextEditReleaseV1
    execution: RustSelectedActionExecutionV2
    consequence: C6PublicConsequenceV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "ordinal": self.ordinal,
            "role": self.role,
            "decision_sha256": self.decision_sha256,
            "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
            "authority_receipt": self.authority_receipt.record(),
            "pre_route_counts": self.pre_route_counts,
            "pre_execute_counts": self.pre_execute_counts,
            "post_grant_counts": self.post_grant_counts,
            "materialization": self.materialization.record(),
            "release": self.release.record(),
            "execution_sha256": self.execution.sha256,
            "consequence": self.consequence.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        self.authority_receipt.authenticate()
        self.materialization.authenticate()
        self.release.authenticate()
        self.execution.authenticate_structure()
        self.consequence.authenticate()
        selected_source = self.execution.selected_source
        if (
            type(self.ordinal) is not int
            or not 1 <= self.ordinal <= 3
            or self.role != ROLES[self.ordinal - 1]
            or not _valid_sha256(self.decision_sha256)
            or not _valid_sha256(self.selected_hypothesis_sha256)
            or self.pre_route_counts != (0, 0)
            or self.pre_execute_counts != (0, 0)
            or self.post_grant_counts != (1, 1)
            or self.authority_receipt.sha256 != self.materialization.authority_sha256
            or self.decision_sha256 != self.materialization.decision_sha256
            or self.selected_hypothesis_sha256 != self.materialization.selected_hypothesis_sha256
            or self.release.authority_sha256 != self.authority_receipt.sha256
            or self.release.selection_sha256 != self.decision_sha256
            or self.release.materialization != self.materialization
            or selected_source.authority_sha256 != self.authority_receipt.sha256
            or selected_source.selection_sha256 != self.decision_sha256
            or selected_source.release_sha256 != self.release.sha256
            or selected_source.action_handle != self.release.action_handle
            or selected_source.source_sha256 != self.release.source_sha256
            or sha256(selected_source.source.encode("ascii")).hexdigest()
            != selected_source.source_sha256
            or self.execution.sha256 != self.consequence.execution_sha256
            or selected_source.source_sha256 != self.consequence.source_sha256
            or self.execution.preparation.batch_sha256 != self.consequence.batch_sha256
            or self.execution.outputs != self.consequence.observed_outputs
            or self.execution.compile_receipt.compile_invocations != 1
            or self.execution.run_receipt.run_invocations != 1
            or self.sha256 != _digest(_ATTEMPT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 selected execution attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C6WorkV1:
    diagnostic_compile_count: int
    materialization_count: int
    release_count: int
    selected_source_compile_count: int
    selected_source_run_count: int
    output_feedback_open_count: int
    memory_advance_count: int
    evaluation_memory_advance_count: int
    alternative_materialization_count: int
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
        expected = {
            "diagnostic_compile_count": 3,
            "materialization_count": 3,
            "release_count": 3,
            "selected_source_compile_count": 3,
            "selected_source_run_count": 3,
            "output_feedback_open_count": 3,
            "memory_advance_count": 2,
            "evaluation_memory_advance_count": 0,
            "alternative_materialization_count": 0,
        }
        if self.record(include_sha256=False) != expected or self.sha256 != _digest(
            _WORK_NAMESPACE, expected
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 work accounting changed"
            )


def _seal_work() -> C6WorkV1:
    payload = {
        "diagnostic_compile_count": 3,
        "materialization_count": 3,
        "release_count": 3,
        "selected_source_compile_count": 3,
        "selected_source_run_count": 3,
        "output_feedback_open_count": 3,
        "memory_advance_count": 2,
        "evaluation_memory_advance_count": 0,
        "alternative_materialization_count": 0,
    }
    result = C6WorkV1(**payload, sha256=_digest(_WORK_NAMESPACE, payload))
    result.authenticate()
    return result


def _memory_record(
    value: factors.FrozenFactorizedFeedbackHypothesisMemoryV1,
) -> dict[str, object]:
    return {
        "checkpoint_sha256": value.checkpoint_sha256,
        "hypotheses": tuple(item.record() for item in value.hypotheses),
        "provenance": tuple(item.record() for item in value.provenance),
        "axis_cue_sha256s": value.axis_cue_sha256s,
        "excitatory_ticks": value.excitatory_ticks,
        "inhibitory_ticks": value.inhibitory_ticks,
    }


def _decision_scores(decision: factors.FactorizedFeedbackDecisionV1) -> dict[str, int]:
    return {item.hypothesis.sha256: item.total_score for item in decision.hypothesis_scores}


@dataclass(frozen=True, slots=True)
class C6DiagnosticFactorMaterializationResultV1:
    status: str
    spec: C6DiagnosticFactorMaterializationSpecV1
    reservation: C6RunReservationV1
    diagnostic_attempts: tuple[adapter.RustcDiagnosticCompileAttemptV1, ...]
    observations: tuple[factors.FactorizedFeedbackObservationV1, ...]
    parent_memory: factors.FrozenFactorizedFeedbackHypothesisMemoryV1
    first_memory: factors.FrozenFactorizedFeedbackHypothesisMemoryV1
    final_memory: factors.FrozenFactorizedFeedbackHypothesisMemoryV1
    adaptation_normal_decisions: tuple[factors.FactorizedFeedbackDecisionV1, ...]
    exploration_decisions: tuple[factors.FactorizedFeedbackDecisionV1, ...]
    adaptation_outcomes: tuple[factors.FactorizedFeedbackOutcomeV1, ...]
    evaluation_decisions: tuple[factors.FactorizedFeedbackDecisionV1, ...]
    control_decisions: tuple[factors.FactorizedFeedbackDecisionV1, ...]
    selected_attempts: tuple[C6SelectedExecutionAttemptV1, ...]
    output_feedback_sha256s: tuple[str, ...]
    work: C6WorkV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "status": self.status,
            "spec_sha256": self.spec.sha256,
            "reservation_sha256": self.reservation.sha256,
            "diagnostic_attempts": tuple(item.record() for item in self.diagnostic_attempts),
            "observation_sha256s": tuple(item.sha256 for item in self.observations),
            "parent_memory": _memory_record(self.parent_memory),
            "first_memory": _memory_record(self.first_memory),
            "final_memory": _memory_record(self.final_memory),
            "adaptation_normal_decision_sha256s": tuple(
                item.sha256 for item in self.adaptation_normal_decisions
            ),
            "exploration_decision_sha256s": tuple(
                item.sha256 for item in self.exploration_decisions
            ),
            "adaptation_outcome_sha256s": tuple(item.sha256 for item in self.adaptation_outcomes),
            "evaluation_decision_sha256s": tuple(item.sha256 for item in self.evaluation_decisions),
            "control_decision_sha256s": tuple(item.sha256 for item in self.control_decisions),
            "selected_attempts": tuple(item.record() for item in self.selected_attempts),
            "output_feedback_sha256s": self.output_feedback_sha256s,
            "work": self.work.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        self.spec.authenticate()
        self.reservation.authenticate(destination=Path(self.reservation.destination))
        if (
            self.status != STATUS_PASS
            or self.reservation.spec_sha256 != self.spec.sha256
            or tuple(
                map(
                    len,
                    (
                        self.diagnostic_attempts,
                        self.observations,
                        self.adaptation_normal_decisions,
                        self.exploration_decisions,
                        self.adaptation_outcomes,
                        self.evaluation_decisions,
                        self.control_decisions,
                        self.selected_attempts,
                        self.output_feedback_sha256s,
                    ),
                )
            )
            != (3, 3, 2, 2, 2, 3, 3, 3, 3)
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 result tuple cardinality changed"
            )

        self.parent_memory.authenticate()
        self.first_memory.authenticate()
        self.final_memory.authenticate()
        self.work.authenticate()
        for value in self.diagnostic_attempts:
            value.authenticate()
        for value in self.observations:
            value.authenticate()
        for values in (
            self.adaptation_normal_decisions,
            self.exploration_decisions,
            self.evaluation_decisions,
            self.control_decisions,
        ):
            for value in values:
                value.authenticate()
        for value in self.adaptation_outcomes:
            value.authenticate()
        for value in self.selected_attempts:
            value.authenticate()

        expected_parent = factors.start_factorized_feedback_hypothesis_memory_v1(HYPOTHESES)
        expected_observations = tuple(
            adapter.project_rust_factorized_feedback_v1(
                attempt,
                selection_context_sha256=_context_sha256(
                    spec_sha256=self.spec.sha256,
                    role=role,
                    source_sha256=source_sha256,
                ),
                active_hypothesis_sha256s=expected_parent.hypothesis_sha256s,
            )
            for attempt, role, source_sha256 in zip(
                self.diagnostic_attempts,
                ROLES,
                ORIGINAL_SOURCE_SHA256S,
                strict=True,
            )
        )
        if _memory_record(self.parent_memory) != _memory_record(expected_parent):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 parent memory changed"
            )
        if self.observations != expected_observations:
            matches = tuple(
                observed == expected
                for observed, expected in zip(
                    self.observations,
                    expected_observations,
                    strict=True,
                )
            )
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                f"C6 observations changed: {matches}"
            )

        for index, attempt in enumerate(self.diagnostic_attempts):
            _authenticate_diagnostic_for_episode(
                index=index,
                attempt=attempt,
                rustc_identity_sha256=self.reservation.rustc_identity_sha256,
            )

        expected_normal_first = expected_parent.route(self.observations[0])
        expected_exploration_first = expected_parent.route(
            self.observations[0],
            exploration_hypothesis_sha256=H00_SHA256,
        )
        expected_first_outcome = factors.seal_factorized_feedback_outcome_v1(
            decision=expected_exploration_first,
            proposal_sha256=self.selected_attempts[0].materialization.sha256,
            execution_sha256=self.selected_attempts[0].execution.sha256,
            consequence_sha256=self.selected_attempts[0].consequence.sha256,
            feedback_sha256=self.output_feedback_sha256s[0],
            passed_cell=1,
        )
        expected_first_memory = factors.advance_factorized_feedback_hypothesis_memory_v1(
            expected_parent,
            expected_first_outcome,
        )
        expected_normal_second = expected_first_memory.route(self.observations[1])
        expected_exploration_second = expected_first_memory.route(
            self.observations[1],
            exploration_hypothesis_sha256=H11_SHA256,
        )
        expected_second_outcome = factors.seal_factorized_feedback_outcome_v1(
            decision=expected_exploration_second,
            proposal_sha256=self.selected_attempts[1].materialization.sha256,
            execution_sha256=self.selected_attempts[1].execution.sha256,
            consequence_sha256=self.selected_attempts[1].consequence.sha256,
            feedback_sha256=self.output_feedback_sha256s[1],
            passed_cell=1,
        )
        expected_final_memory = factors.advance_factorized_feedback_hypothesis_memory_v1(
            expected_first_memory,
            expected_second_outcome,
        )
        expected_evaluation_decisions = (
            expected_parent.route(self.observations[2]),
            expected_first_memory.route(self.observations[2]),
            expected_final_memory.route(self.observations[2]),
        )
        expected_controls = (
            expected_final_memory.route(self.observations[2], outcome_gate=False),
            expected_final_memory.route(self.observations[2], axis_gates=(False, True)),
            expected_final_memory.route(self.observations[2], axis_gates=(True, False)),
        )
        if (
            _memory_record(self.first_memory) != _memory_record(expected_first_memory)
            or _memory_record(self.final_memory) != _memory_record(expected_final_memory)
            or self.adaptation_normal_decisions != (expected_normal_first, expected_normal_second)
            or self.exploration_decisions
            != (expected_exploration_first, expected_exploration_second)
            or self.adaptation_outcomes != (expected_first_outcome, expected_second_outcome)
            or self.evaluation_decisions != expected_evaluation_decisions
            or self.control_decisions != expected_controls
            or any(
                control is not memory
                for memory, outcome in (
                    (expected_parent, expected_first_outcome),
                    (expected_first_memory, expected_second_outcome),
                )
                for control in (
                    factors.advance_factorized_feedback_hypothesis_memory_v1(
                        memory,
                        outcome,
                        plasticity_gate=False,
                    ),
                    factors.advance_factorized_feedback_hypothesis_memory_v1(
                        memory,
                        outcome,
                        consequence_gate=False,
                    ),
                )
            )
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 memory lineage or causal controls changed"
            )

        expected_selected = (H00_SHA256, H11_SHA256, H01_SHA256)
        expected_components = (
            (OPERATOR_COMPONENT_SHA256S[0], TARGET_COMPONENT_SHA256S[0]),
            (OPERATOR_COMPONENT_SHA256S[1], TARGET_COMPONENT_SHA256S[1]),
            (OPERATOR_COMPONENT_SHA256S[0], TARGET_COMPONENT_SHA256S[1]),
        )
        materialized_roots = tuple(
            item.materialization.materialized_source_sha256 for item in self.selected_attempts
        )
        for index, (attempt, selected_root, components) in enumerate(
            zip(self.selected_attempts, expected_selected, expected_components, strict=True)
        ):
            receipt = attempt.authority_receipt
            if (
                attempt.selected_hypothesis_sha256 != selected_root
                or (
                    attempt.materialization.operator_component_sha256,
                    attempt.materialization.target_component_sha256,
                )
                != components
                or receipt.source_sha256 != ORIGINAL_SOURCE_SHA256S[index]
                or receipt.source_size_bytes != len(ORIGINAL_SOURCES[index].encode("ascii"))
                or receipt.diagnostic_attempt_sha256 != self.diagnostic_attempts[index].sha256
                or receipt.selection_context_sha256
                != _context_sha256(
                    spec_sha256=self.spec.sha256,
                    role=ROLES[index],
                    source_sha256=ORIGINAL_SOURCE_SHA256S[index],
                )
                or receipt.operator_bindings != OPERATOR_BINDINGS
                or receipt.target_bindings != TARGET_BINDINGS_BY_EPISODE[index]
                or receipt.expected_checkpoint_sha256
                != (expected_parent, expected_first_memory, expected_final_memory)[
                    index
                ].checkpoint_sha256
                or receipt.allow_exploration != ALLOW_EXPLORATION_BY_EPISODE[index]
                or attempt.consequence.expected_outputs != EXPECTED_OUTPUTS[index]
                or attempt.consequence.output_feedback_sha256 != self.output_feedback_sha256s[index]
                or attempt.consequence.passed_cell != 1
                or attempt.execution.rustc_identity.sha256 != self.reservation.rustc_identity_sha256
            ):
                raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                    "C6 materializer or execution custody changed"
                )
        if (
            tuple(item.selected_hypothesis_sha256 for item in self.evaluation_decisions)
            != (None, None, H01_SHA256)
            or any(item.selected_hypothesis_sha256 is not None for item in self.control_decisions)
            or _decision_scores(self.evaluation_decisions[2])
            != {
                H00_SHA256: 2,
                H01_SHA256: 4,
                H10_SHA256: 0,
                H11_SHA256: 2,
            }
            or len(set(materialized_roots)) != 3
            or set(materialized_roots).intersection(ORIGINAL_SOURCE_SHA256S)
            or set(materialized_roots).intersection(EXCLUDED_PRIOR_SOURCE_SHA256S)
            or self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 final selection, freshness, or result root changed"
            )


DiagnosticCompileCallback = Callable[
    [str, str],
    adapter.RustcDiagnosticCompileAttemptV1,
]
ExecutionCallback = Callable[
    [
        materializer.PreparedFactorizedTextEditMaterializerV1,
        factors.FactorizedFeedbackDecisionV1,
        tuple[tuple[int, ...], ...],
    ],
    RustSelectedActionExecutionV2,
]
OutputFeedbackCallback = Callable[[str], tuple[tuple[int, ...], ...]]


def _context_sha256(*, spec_sha256: str, role: str, source_sha256: str) -> str:
    return _digest(
        _CONTEXT_NAMESPACE,
        {
            "spec_sha256": spec_sha256,
            "role": role,
            "source_sha256": source_sha256,
        },
    )


def _validate_outputs(
    outputs: tuple[tuple[int, ...], ...],
    *,
    batch: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if (
        type(outputs) is not tuple
        or len(outputs) != len(batch)
        or any(
            type(row) is not tuple
            or len(row) != len(inputs)
            or any(type(value) is not int for value in row)
            for row, inputs in zip(outputs, batch, strict=True)
        )
    ):
        raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
            "C6 public output feedback shape changed"
        )
    return outputs


def _authenticate_diagnostic_for_episode(
    *,
    index: int,
    attempt: adapter.RustcDiagnosticCompileAttemptV1,
    rustc_identity_sha256: str,
) -> None:
    if type(attempt) is not adapter.RustcDiagnosticCompileAttemptV1:
        raise TypeError("C6 diagnostic compiler must return one exact attempt")
    attempt.authenticate()
    source = ORIGINAL_SOURCES[index]
    expected_code, expected_region = EXPECTED_DIAGNOSTICS[index]
    spans = tuple(
        span for diagnostic in attempt.error_diagnostics for span in diagnostic.primary_spans
    )
    regions = {
        adapter.FIRST_HALF
        if 2 * ((span.byte_start + span.byte_end) // 2) < len(source.encode("ascii"))
        else adapter.SECOND_HALF
        for span in spans
    }
    if (
        attempt.status != adapter.COMPILE_FAILED
        or attempt.rustc_identity_sha256 != rustc_identity_sha256
        or attempt.source_file_name != SOURCE_FILE_NAME
        or attempt.source_sha256 != ORIGINAL_SOURCE_SHA256S[index]
        or attempt.source_size_bytes != len(source.encode("ascii"))
        or attempt.arguments != DIAGNOSTIC_ARGUMENTS
        or tuple(diagnostic.code for diagnostic in attempt.error_diagnostics) != (expected_code,)
        or len(spans) != 1
        or (spans[0].byte_start, spans[0].byte_end) != EXPECTED_PRIMARY_BYTE_RANGES[index]
        or regions != {expected_region}
    ):
        raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
            "C6 diagnostic attempt changed"
        )


def _compile_diagnostic_once(
    identity: rust_executor.RustcIdentityV1,
    source: str,
) -> adapter.RustcDiagnosticCompileAttemptV1:
    identity.authenticate_structure()
    source_bytes = source.encode("ascii")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory(prefix="humanbrain-c6-diagnostic-") as directory:
        source_path = Path(directory) / SOURCE_FILE_NAME
        source_path.write_bytes(source_bytes)
        completed = subprocess.run(
            (identity.path, *DIAGNOSTIC_ARGUMENTS),
            cwd=directory,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=30,
            creationflags=creationflags,
        )
    return adapter.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=identity.sha256,
        source_file_name=SOURCE_FILE_NAME,
        source=source_bytes,
        arguments=DIAGNOSTIC_ARGUMENTS,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        compile_invocations=1,
    )


def _execute_selected(
    prepared: materializer.PreparedFactorizedTextEditMaterializerV1,
    decision: factors.FactorizedFeedbackDecisionV1,
    batch: tuple[tuple[int, ...], ...],
) -> RustSelectedActionExecutionV2:
    executor = rust_executor.prepare_rust_selected_action_executor_v2(prepared, batch)
    return executor.execute_selected_once(decision)


def _prepare_episode(
    *,
    index: int,
    spec: C6DiagnosticFactorMaterializationSpecV1,
    attempt: adapter.RustcDiagnosticCompileAttemptV1,
    memory: factors.FrozenFactorizedFeedbackHypothesisMemoryV1,
) -> materializer.PreparedFactorizedTextEditMaterializerV1:
    return materializer.prepare_factorized_text_edit_materializer_v1(
        diagnostic_attempt=attempt,
        source=ORIGINAL_SOURCES[index],
        selection_context_sha256=_context_sha256(
            spec_sha256=spec.sha256,
            role=ROLES[index],
            source_sha256=ORIGINAL_SOURCE_SHA256S[index],
        ),
        expected_checkpoint_sha256=memory.checkpoint_sha256,
        allow_exploration=ALLOW_EXPLORATION_BY_EPISODE[index],
        operator_bindings=OPERATOR_BINDINGS,
        target_bindings=TARGET_BINDINGS_BY_EPISODE[index],
    )


def _run_selected_attempt(
    *,
    ordinal: int,
    spec: C6DiagnosticFactorMaterializationSpecV1,
    reservation: C6RunReservationV1,
    prepared: materializer.PreparedFactorizedTextEditMaterializerV1,
    pre_route_counts: tuple[int, int],
    decision: factors.FactorizedFeedbackDecisionV1,
    batch: tuple[tuple[int, ...], ...],
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> tuple[C6SelectedExecutionAttemptV1, str]:
    selected_hypothesis_sha256 = decision.selected_hypothesis_sha256
    if selected_hypothesis_sha256 is None:
        raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
            "C6 STOP decision cannot materialize source"
        )
    pre_execute_counts = (prepared.materialization_count, prepared.release_count)
    if pre_route_counts != (0, 0) or pre_execute_counts != (0, 0):
        raise _behavior_failure(
            "C6 source materialized before selected execution",
            stage=ROLES[ordinal - 1],
            decision=decision.record(),
            authority_receipt=prepared.receipt.record(),
            pre_route_counts=pre_route_counts,
            pre_execute_counts=pre_execute_counts,
        )

    execution = execute(prepared, decision, batch)
    if type(execution) is not RustSelectedActionExecutionV2:
        raise TypeError("C6 executor must return one exact RustSelectedActionExecutionV2")
    execution.authenticate_structure()
    materialization_record = prepared.materialization
    release = prepared.release
    post_grant_counts = (prepared.materialization_count, prepared.release_count)
    if materialization_record is None or release is None or post_grant_counts != (1, 1):
        raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
            "C6 selected execution lacks exact materialization custody"
        )
    selected_source = execution.selected_source
    if (
        execution.rustc_identity.sha256 != reservation.rustc_identity_sha256
        or selected_source.authority_sha256 != prepared.authority_sha256
        or selected_source.selection_sha256 != decision.sha256
        or selected_source.release_sha256 != release.sha256
        or selected_source.source_sha256 != release.source_sha256
        or execution.compile_receipt.compile_invocations != 1
        or execution.run_receipt.run_invocations != 1
        or len(execution.outputs) != len(batch)
    ):
        raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
            "C6 selected V2 execution custody changed"
        )

    role = ROLES[ordinal - 1]
    expected_outputs = _validate_outputs(output_feedback(role), batch=batch)
    output_feedback_sha256 = _digest(
        _OUTPUT_FEEDBACK_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "ordinal": ordinal,
            "role": role,
            "execution_sha256": execution.sha256,
            "expected_outputs": expected_outputs,
        },
    )
    consequence = _seal_consequence(
        execution=execution,
        expected_outputs=expected_outputs,
        output_feedback_sha256=output_feedback_sha256,
    )
    payload = {
        "ordinal": ordinal,
        "role": role,
        "decision_sha256": decision.sha256,
        "selected_hypothesis_sha256": selected_hypothesis_sha256,
        "authority_receipt": prepared.receipt,
        "pre_route_counts": pre_route_counts,
        "pre_execute_counts": pre_execute_counts,
        "post_grant_counts": post_grant_counts,
        "materialization": materialization_record,
        "release": release,
        "execution": execution,
        "consequence": consequence,
    }
    partial = C6SelectedExecutionAttemptV1(**payload, sha256="")
    attempt = replace(
        partial,
        sha256=_digest(_ATTEMPT_NAMESPACE, partial.record(include_sha256=False)),
    )
    attempt.authenticate()
    if consequence.passed_cell != 1:
        raise _behavior_failure(
            "C6 selected materialization failed public output feedback",
            stage=role,
            decision=decision.record(),
            selected_attempt={
                "attempt": attempt.record(),
                "authority_receipt": attempt.authority_receipt.record(),
                "materialization": attempt.materialization.record(),
                "release": attempt.release.record(),
                "execution": _plain(attempt.execution),
                "consequence": attempt.consequence.record(),
            },
        )
    return attempt, output_feedback_sha256


def _run_reserved_cohort_v1(
    *,
    spec: C6DiagnosticFactorMaterializationSpecV1,
    reservation: C6RunReservationV1,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> C6DiagnosticFactorMaterializationResultV1:
    spec.authenticate()
    reservation.authenticate(destination=Path(reservation.destination))
    if reservation.spec_sha256 != spec.sha256:
        raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
            "C6 reservation/spec custody changed"
        )

    parent = factors.start_factorized_feedback_hypothesis_memory_v1(HYPOTHESES)
    memories = [parent]
    diagnostic_attempts: list[adapter.RustcDiagnosticCompileAttemptV1] = []
    observations: list[factors.FactorizedFeedbackObservationV1] = []
    normal_decisions: list[factors.FactorizedFeedbackDecisionV1] = []
    exploration_decisions: list[factors.FactorizedFeedbackDecisionV1] = []
    outcomes: list[factors.FactorizedFeedbackOutcomeV1] = []
    selected_attempts: list[C6SelectedExecutionAttemptV1] = []
    output_feedback_sha256s: list[str] = []

    for index, exploration_root in enumerate(EXPLORATION_HYPOTHESIS_SHA256S):
        current_memory = memories[-1]
        attempt = diagnostic_compile(
            reservation.rustc_identity_sha256,
            ORIGINAL_SOURCES[index],
        )
        _authenticate_diagnostic_for_episode(
            index=index,
            attempt=attempt,
            rustc_identity_sha256=reservation.rustc_identity_sha256,
        )
        diagnostic_attempts.append(attempt)
        prepared = _prepare_episode(
            index=index,
            spec=spec,
            attempt=attempt,
            memory=current_memory,
        )
        pre_route_counts = (prepared.materialization_count, prepared.release_count)
        if pre_route_counts != (0, 0):
            raise _behavior_failure(
                "C6 materializer performed work before adaptation routing",
                stage=ROLES[index],
                authority_receipt=prepared.receipt.record(),
                pre_route_counts=pre_route_counts,
            )
        observation = adapter.project_rust_factorized_feedback_v1(
            attempt,
            selection_context_sha256=prepared.receipt.selection_context_sha256,
            active_hypothesis_sha256s=current_memory.hypothesis_sha256s,
        )
        observations.append(observation)
        normal = current_memory.route(observation)
        exploration = current_memory.route(
            observation,
            exploration_hypothesis_sha256=exploration_root,
        )
        normal_decisions.append(normal)
        exploration_decisions.append(exploration)
        if (
            normal.selected_hypothesis_sha256 is not None
            or exploration.selected_hypothesis_sha256 != exploration_root
        ):
            raise _behavior_failure(
                "C6 adaptation selection behavior changed",
                stage=ROLES[index],
                diagnostic_attempt=attempt.record(),
                observation=observation.record(),
                memory=_memory_record(current_memory),
                normal_decision=normal.record(),
                exploration_decision=exploration.record(),
                authority_receipt=prepared.receipt.record(),
                pre_route_counts=pre_route_counts,
            )
        selected_attempt, feedback_sha256 = _run_selected_attempt(
            ordinal=index + 1,
            spec=spec,
            reservation=reservation,
            prepared=prepared,
            pre_route_counts=pre_route_counts,
            decision=exploration,
            batch=BATCHES[index],
            execute=execute,
            output_feedback=output_feedback,
        )
        selected_attempts.append(selected_attempt)
        output_feedback_sha256s.append(feedback_sha256)
        outcome = factors.seal_factorized_feedback_outcome_v1(
            decision=exploration,
            proposal_sha256=selected_attempt.materialization.sha256,
            execution_sha256=selected_attempt.execution.sha256,
            consequence_sha256=selected_attempt.consequence.sha256,
            feedback_sha256=feedback_sha256,
            passed_cell=1,
        )
        controls = (
            factors.advance_factorized_feedback_hypothesis_memory_v1(
                current_memory,
                outcome,
                plasticity_gate=False,
            ),
            factors.advance_factorized_feedback_hypothesis_memory_v1(
                current_memory,
                outcome,
                consequence_gate=False,
            ),
        )
        if any(control is not current_memory for control in controls):
            raise _behavior_failure(
                "C6 adaptation learning gate control changed",
                stage=ROLES[index],
                exploration_decision=exploration.record(),
                selected_attempt=selected_attempt.record(),
                outcome=outcome.record(),
                memory=_memory_record(current_memory),
                control_memories=tuple(_memory_record(control) for control in controls),
            )
        outcomes.append(outcome)
        memories.append(
            factors.advance_factorized_feedback_hypothesis_memory_v1(
                current_memory,
                outcome,
            )
        )

    evaluation_index = 2
    evaluation_diagnostic = diagnostic_compile(
        reservation.rustc_identity_sha256,
        ORIGINAL_SOURCES[evaluation_index],
    )
    _authenticate_diagnostic_for_episode(
        index=evaluation_index,
        attempt=evaluation_diagnostic,
        rustc_identity_sha256=reservation.rustc_identity_sha256,
    )
    diagnostic_attempts.append(evaluation_diagnostic)
    evaluation_materializer = _prepare_episode(
        index=evaluation_index,
        spec=spec,
        attempt=evaluation_diagnostic,
        memory=memories[2],
    )
    evaluation_pre_route_counts = (
        evaluation_materializer.materialization_count,
        evaluation_materializer.release_count,
    )
    if evaluation_pre_route_counts != (0, 0):
        raise _behavior_failure(
            "C6 materializer performed work before evaluation routing",
            stage=ROLES[evaluation_index],
            authority_receipt=evaluation_materializer.receipt.record(),
            pre_route_counts=evaluation_pre_route_counts,
        )
    evaluation_observation = adapter.project_rust_factorized_feedback_v1(
        evaluation_diagnostic,
        selection_context_sha256=evaluation_materializer.receipt.selection_context_sha256,
        active_hypothesis_sha256s=memories[-1].hypothesis_sha256s,
    )
    observations.append(evaluation_observation)
    evaluation_decisions = (
        memories[0].route(evaluation_observation),
        memories[1].route(evaluation_observation),
        memories[2].route(evaluation_observation),
    )
    controls = (
        memories[2].route(evaluation_observation, outcome_gate=False),
        memories[2].route(evaluation_observation, axis_gates=(False, True)),
        memories[2].route(evaluation_observation, axis_gates=(True, False)),
    )
    if (
        tuple(value.selected_hypothesis_sha256 for value in evaluation_decisions)
        != (None, None, H01_SHA256)
        or any(value.selected_hypothesis_sha256 is not None for value in controls)
        or _decision_scores(evaluation_decisions[2])
        != {
            H00_SHA256: 2,
            H01_SHA256: 4,
            H10_SHA256: 0,
            H11_SHA256: 2,
        }
    ):
        raise _behavior_failure(
            "C6 evaluation recombination or causal-control behavior changed",
            stage="evaluation-selection",
            diagnostic_attempts=tuple(value.record() for value in diagnostic_attempts),
            observations=tuple(value.record() for value in observations),
            memories=tuple(_memory_record(value) for value in memories),
            adaptation_normal_decisions=tuple(value.record() for value in normal_decisions),
            exploration_decisions=tuple(value.record() for value in exploration_decisions),
            adaptation_outcomes=tuple(value.record() for value in outcomes),
            selected_attempts=tuple(value.record() for value in selected_attempts),
            evaluation_decisions=tuple(value.record() for value in evaluation_decisions),
            control_decisions=tuple(value.record() for value in controls),
            final_scores=_decision_scores(evaluation_decisions[2]),
            authority_receipt=evaluation_materializer.receipt.record(),
            pre_route_counts=evaluation_pre_route_counts,
        )
    evaluation_attempt, evaluation_feedback_sha256 = _run_selected_attempt(
        ordinal=3,
        spec=spec,
        reservation=reservation,
        prepared=evaluation_materializer,
        pre_route_counts=evaluation_pre_route_counts,
        decision=evaluation_decisions[2],
        batch=BATCHES[2],
        execute=execute,
        output_feedback=output_feedback,
    )
    selected_attempts.append(evaluation_attempt)
    output_feedback_sha256s.append(evaluation_feedback_sha256)

    payload = {
        "status": STATUS_PASS,
        "spec": spec,
        "reservation": reservation,
        "diagnostic_attempts": tuple(diagnostic_attempts),
        "observations": tuple(observations),
        "parent_memory": memories[0],
        "first_memory": memories[1],
        "final_memory": memories[2],
        "adaptation_normal_decisions": tuple(normal_decisions),
        "exploration_decisions": tuple(exploration_decisions),
        "adaptation_outcomes": tuple(outcomes),
        "evaluation_decisions": evaluation_decisions,
        "control_decisions": controls,
        "selected_attempts": tuple(selected_attempts),
        "output_feedback_sha256s": tuple(output_feedback_sha256s),
        "work": _seal_work(),
    }
    partial = C6DiagnosticFactorMaterializationResultV1(**payload, sha256="")
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _reserve_destination(
    spec: C6DiagnosticFactorMaterializationSpecV1,
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
) -> C6RunReservationV1:
    spec.authenticate()
    if not _valid_commit(preregistration_commit):
        raise ValueError("C6 preregistration commit must be 40 lowercase hex characters")
    if not _valid_sha256(rustc_identity_sha256):
        raise ValueError("C6 rustc identity root is invalid")
    destination = Path(destination).resolve()
    reservation_path = Path(reservation_path).resolve()
    expected_reservation = destination.with_name(f"{destination.name}.reservation.json")
    if reservation_path != expected_reservation:
        raise ValueError("C6 reservation must be the result's exact sibling")
    if os.path.lexists(destination) or os.path.lexists(reservation_path):
        raise FileExistsError(destination if os.path.lexists(destination) else reservation_path)
    partial = C6RunReservationV1(
        schema=RESERVATION_SCHEMA,
        spec_sha256=spec.sha256,
        preregistration_commit=preregistration_commit,
        rustc_identity_sha256=rustc_identity_sha256,
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
            raise OSError("C6 reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if reservation_path.read_bytes() != payload:
        raise OSError("C6 reservation readback changed")
    return reservation


def _result_document(
    result: C6DiagnosticFactorMaterializationResultV1,
) -> dict[str, object]:
    result.authenticate()
    body: dict[str, object] = {
        "schema": RESULT_DOCUMENT_SCHEMA,
        "terminal_status": result.status,
        "claim_boundary": CLAIM_BOUNDARY,
        "spec_sha256": result.spec.sha256,
        "reservation_sha256": result.reservation.sha256,
        "preregistration_commit": result.reservation.preregistration_commit,
        "rustc_identity_sha256": result.reservation.rustc_identity_sha256,
        "result_sha256": result.sha256,
        "authenticated_result_record": result.record(),
        "full_authenticated_transcript": {
            "spec": result.spec.record(),
            "reservation": result.reservation.record(),
            "diagnostic_attempts": tuple(item.record() for item in result.diagnostic_attempts),
            "observations": tuple(item.record() for item in result.observations),
            "memories": tuple(
                _memory_record(item)
                for item in (result.parent_memory, result.first_memory, result.final_memory)
            ),
            "adaptation_normal_decisions": tuple(
                item.record() for item in result.adaptation_normal_decisions
            ),
            "exploration_decisions": tuple(item.record() for item in result.exploration_decisions),
            "adaptation_outcomes": tuple(item.record() for item in result.adaptation_outcomes),
            "evaluation_decisions": tuple(item.record() for item in result.evaluation_decisions),
            "control_decisions": tuple(item.record() for item in result.control_decisions),
            "selected_attempts": tuple(
                {
                    "attempt": item.record(),
                    "authority_receipt": item.authority_receipt.record(),
                    "materialization": item.materialization.record(),
                    "release": item.release.record(),
                    "execution": _plain(item.execution),
                    "consequence": item.consequence.record(),
                }
                for item in result.selected_attempts
            ),
            "work": result.work.record(),
        },
    }
    return {**body, "document_sha256": _digest(_DOCUMENT_NAMESPACE, body)}


def _abort_document(
    *,
    spec: C6DiagnosticFactorMaterializationSpecV1,
    reservation: C6RunReservationV1,
    error: Exception,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": RESULT_DOCUMENT_SCHEMA,
        "terminal_status": STATUS_EXECUTION_ABORT,
        "claim_boundary": CLAIM_BOUNDARY,
        "spec_sha256": spec.sha256,
        "reservation_sha256": reservation.sha256,
        "preregistration_commit": reservation.preregistration_commit,
        "rustc_identity_sha256": reservation.rustc_identity_sha256,
        "error_type": type(error).__qualname__,
        "error_message": str(error)[:1000],
    }
    return {**body, "document_sha256": _digest(_DOCUMENT_NAMESPACE, body)}


def _failure_document(
    *,
    spec: C6DiagnosticFactorMaterializationSpecV1,
    reservation: C6RunReservationV1,
    error: C6BehaviorFailure,
) -> dict[str, object]:
    partial_transcript = _plain(error.partial_transcript)
    if type(partial_transcript) is not dict:
        raise TypeError("C6 behavior failure transcript must be one exact object")
    body: dict[str, object] = {
        "schema": RESULT_DOCUMENT_SCHEMA,
        "terminal_status": STATUS_BEHAVIOR_FAIL,
        "claim_boundary": CLAIM_BOUNDARY,
        "spec_sha256": spec.sha256,
        "reservation_sha256": reservation.sha256,
        "preregistration_commit": reservation.preregistration_commit,
        "rustc_identity_sha256": reservation.rustc_identity_sha256,
        "failure_type": type(error).__qualname__,
        "failure_reason": error.reason,
        "partial_transcript": partial_transcript,
        "partial_transcript_sha256": _digest(_FAILURE_NAMESPACE, partial_transcript),
    }
    return {**body, "document_sha256": _digest(_DOCUMENT_NAMESPACE, body)}


def _publish_document(destination: Path, document: dict[str, object]) -> None:
    payload = _canonical_json(document) + b"\n"
    with destination.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("C6 terminal document write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = destination.read_bytes()
    if (
        retained != payload
        or _canonical_json(json.loads(retained.decode("ascii"))) + b"\n" != payload
    ):
        raise OSError("C6 terminal document readback changed")
    decoded = json.loads(retained.decode("ascii"))
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if document_sha256 != _digest(_DOCUMENT_NAMESPACE, body):
        raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
            "C6 terminal document root changed"
        )
    terminal_status = decoded["terminal_status"]
    if terminal_status == STATUS_PASS:
        record = dict(decoded["authenticated_result_record"])
        result_sha256 = record.pop("sha256", None)
        if result_sha256 != decoded["result_sha256"] or result_sha256 != _digest(
            _RESULT_NAMESPACE,
            record,
        ):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 persisted result root changed"
            )
    elif terminal_status == STATUS_BEHAVIOR_FAIL:
        partial_transcript = decoded.get("partial_transcript")
        if type(partial_transcript) is not dict or decoded.get(
            "partial_transcript_sha256"
        ) != _digest(_FAILURE_NAMESPACE, partial_transcript):
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 persisted behavior-failure transcript root changed"
            )
    elif terminal_status != STATUS_EXECUTION_ABORT:
        raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
            "C6 terminal status changed"
        )


def _run_and_write_at_paths_for_test_v1(
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
    spec: C6DiagnosticFactorMaterializationSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C6DiagnosticFactorMaterializationResultV1:
    """Private injected test seam; production evidence paths remain fixed."""

    destination = Path(destination).resolve()
    reservation = _reserve_destination(
        spec,
        preregistration_commit=preregistration_commit,
        rustc_identity_sha256=rustc_identity_sha256,
        destination=destination,
        reservation_path=reservation_path,
    )
    try:
        result = _run_reserved_cohort_v1(
            spec=spec,
            reservation=reservation,
            diagnostic_compile=diagnostic_compile,
            execute=execute,
            output_feedback=output_feedback,
        )
        document = _result_document(result)
    except C6BehaviorFailure as error:
        document = _failure_document(spec=spec, reservation=reservation, error=error)
        _publish_document(destination, document)
        raise
    except Exception as error:
        document = _abort_document(spec=spec, reservation=reservation, error=error)
        _publish_document(destination, document)
        raise
    _publish_document(destination, document)
    return result


def _production_evidence_paths_v1() -> tuple[Path, Path]:
    destination = Path(__file__).with_suffix(".result.json").resolve()
    return destination, destination.with_name(f"{destination.name}.reservation.json")


def _preflight_rustc_v1() -> rust_executor.RustcIdentityV1:
    """Resolve and authenticate the compiler before a persistent reservation exists."""

    identity = rust_executor._discover_rustc()
    identity.authenticate_structure()
    return identity


def run_and_write_cohort_v1(
    *,
    preregistration_commit: str,
    spec: C6DiagnosticFactorMaterializationSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C6DiagnosticFactorMaterializationResultV1:
    """Preflight, reserve, and spend the exact production C6 cohort once."""

    identity = _preflight_rustc_v1()
    destination, reservation_path = _production_evidence_paths_v1()

    def diagnostic_compile(
        rustc_identity_sha256: str,
        source: str,
    ) -> adapter.RustcDiagnosticCompileAttemptV1:
        if rustc_identity_sha256 != identity.sha256:
            raise ContinualRustDiagnosticFactorMaterializationCohortV1Error(
                "C6 diagnostic compiler identity changed after reservation"
            )
        return _compile_diagnostic_once(identity, source)

    expected_by_role = dict(zip(ROLES, EXPECTED_OUTPUTS, strict=True))
    return _run_and_write_at_paths_for_test_v1(
        preregistration_commit=preregistration_commit,
        rustc_identity_sha256=identity.sha256,
        destination=destination,
        reservation_path=reservation_path,
        diagnostic_compile=diagnostic_compile,
        execute=_execute_selected,
        output_feedback=lambda role: expected_by_role[role],
        spec=spec,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-commit", required=True)
    arguments = parser.parse_args()
    result = run_and_write_cohort_v1(
        preregistration_commit=arguments.preregistration_commit,
    )
    print(_canonical_json(result.record()).decode("ascii"))


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_PUBLIC_SPEC",
    "STATUS_BEHAVIOR_FAIL",
    "STATUS_EXECUTION_ABORT",
    "STATUS_PASS",
    "C6BehaviorFailure",
    "C6DiagnosticFactorMaterializationResultV1",
    "C6DiagnosticFactorMaterializationSpecV1",
    "C6PublicConsequenceV1",
    "C6RunReservationV1",
    "C6SelectedExecutionAttemptV1",
    "C6WorkV1",
    "ContinualRustDiagnosticFactorMaterializationCohortV1Error",
    "make_c6_diagnostic_factor_materialization_spec_v1",
    "output_sha256_v1",
    "run_and_write_cohort_v1",
]


if __name__ == "__main__":
    main()
