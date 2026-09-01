"""Fresh C5 rustc-diagnostic factor recombination cohort.

Two public TRAIN episodes teach diagonal error-class/source-region associations
inside a supplied two-by-two repair bank.  A fresh third diagnostic must make
the generic factorized neuronal memory physically select the unseen cross
hypothesis.  Only the selected candidate is released, compiled, and run.
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
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_actions
from experiments.r3_pilot import common_rust_action_intervention_v1 as interventions
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as adapter
from humanbrain import rust_selected_action_executor as rust_executor
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

SCHEMA = "humanbrain.continual-rust-diagnostic-factor-recombination-cohort.v1"
RESULT_DOCUMENT_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-factor-recombination-result-document.v1"
)
RESERVATION_SCHEMA = "humanbrain.continual-rust-diagnostic-factor-recombination-reservation.v1"
STATUS_PASS = "PASS_C5_RUSTC_DIAGNOSTIC_FACTOR_RECOMBINATION_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C5_RUSTC_DIAGNOSTIC_FACTOR_RECOMBINATION_V1"
STATUS_EXECUTION_ABORT = "ABORT_C5_RUSTC_DIAGNOSTIC_FACTOR_RECOMBINATION_V1"
CLAIM_BOUNDARY = (
    "one-fresh-public-train-cohort-where-caller-attested-rustc-error-code-and-hand-"
    "built-relative-source-half-cues-teach-two-diagonal-associations-in-a-generic-"
    "factorized-neuronal-memory-whose-four-way-physical-race-selects-and-executes-"
    "one-unseen-cross-hypothesis-from-a-supplied-repair-bank-not-learned-cues-"
    "localization-diagnosis-repair-schema-discovery-source-synthesis-protected-"
    "transfer-swe-or-intelligence"
)

FROZEN_MECHANISM_COMMIT = "fe6500e810fb5f2e4d022b0b774ba31985e33429"
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

SOURCE_FILE_NAME = "case.rs"
DIAGNOSTIC_ARGUMENTS = (
    "--edition",
    "2021",
    "--crate-name",
    "c5_diagnostic",
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

ROLES = ("adaptation-e0425-first", "adaptation-e0308-second", "evaluation-e0425-second")
EXPECTED_DIAGNOSTICS = (
    ("E0425", adapter.FIRST_HALF),
    ("E0308", adapter.SECOND_HALF),
    ("E0425", adapter.SECOND_HALF),
)
BATCHES = (((-3, 0, 4),), ((-5, 1, 6),), ((-8, 2, 7),))
EXPECTED_OUTPUTS = (((4, 7, 11),), ((6, 12, 17),), ((5, 15, 20),))

ADAPTATION_E0425_FIRST_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let first_bias: i64 = {
        let _first_anchor = 2_i64;
        _missing_value
    };
    // C5_REGION_SPLIT
    let second_bias: i64 = {
        let _second_anchor = 5_i64;
        5_i64
    };
    samples
        .iter()
        .copied()
        .map(|value| value + first_bias + second_bias)
        .collect()
}
"""

ADAPTATION_E0308_SECOND_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let _wrong_value = "eleven";
    let first_bias: i64 = {
        let _first_anchor = 3_i64;
        3_i64
    };
    // C5_REGION_SPLIT
    let second_bias: i64 = {
        let _second_anchor = 8_i64;
        _wrong_value
    };
    samples
        .iter()
        .copied()
        .map(|value| value + first_bias + second_bias)
        .collect()
}
"""

EVALUATION_E0425_SECOND_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let first_bias: i64 = {
        let _first_anchor = 4_i64;
        4_i64
    };
    // C5_REGION_SPLIT
    let second_bias: i64 = {
        let _second_anchor = 9_i64;
        _missing_value
    };
    samples
        .iter()
        .copied()
        .map(|value| value + first_bias + second_bias)
        .collect()
}
"""

ORIGINAL_SOURCES = (
    ADAPTATION_E0425_FIRST_SOURCE,
    ADAPTATION_E0308_SECOND_SOURCE,
    EVALUATION_E0425_SECOND_SOURCE,
)

_SHA256_HEX = frozenset("0123456789abcdef")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_NAMESPACE = b"c5-rustc-diagnostic-factor-recombination-spec-v1"
_COMPONENT_NAMESPACE = b"c5-rustc-diagnostic-factor-component-v1"
_CONTEXT_NAMESPACE = b"c5-rustc-diagnostic-factor-context-v1"
_CATALOGUE_BINDING_NAMESPACE = b"c5-rustc-diagnostic-factor-catalogue-binding-v1"
_SELECTED_BINDING_NAMESPACE = b"c5-rustc-diagnostic-factor-selected-binding-v1"
_INTERVENTION_NAMESPACE = b"c5-rustc-diagnostic-factor-intervention-v1"
_OUTPUT_FEEDBACK_NAMESPACE = b"c5-rustc-diagnostic-factor-output-feedback-v1"
_CONSEQUENCE_NAMESPACE = b"c5-rustc-diagnostic-factor-consequence-v1"
_ATTEMPT_NAMESPACE = b"c5-rustc-diagnostic-factor-selected-attempt-v1"
_WORK_NAMESPACE = b"c5-rustc-diagnostic-factor-work-v1"
_RESERVATION_NAMESPACE = b"c5-rustc-diagnostic-factor-reservation-v1"
_RESULT_NAMESPACE = b"c5-rustc-diagnostic-factor-result-v1"
_FAILURE_NAMESPACE = b"c5-rustc-diagnostic-factor-failure-v1"
_DOCUMENT_NAMESPACE = b"c5-rustc-diagnostic-factor-document-v1"


class ContinualRustDiagnosticFactorRecombinationCohortV1Error(ValueError):
    """The frozen C5 cohort or its authenticated transcript changed."""


class C5BehaviorFailure(RuntimeError):
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
    raise TypeError(f"C5 transcript cannot serialize {type(value).__qualname__}")


def _behavior_failure(reason: str, **partial_transcript: object) -> C5BehaviorFailure:
    return C5BehaviorFailure(
        reason,
        {key: _plain(value) for key, value in partial_transcript.items()},
    )


def _insert_candidate_line(source: str, *, class_index: int, region_index: int) -> str:
    region = "first" if region_index == 0 else "second"
    anchor = f"        let _{region}_anchor = "
    if source.count(anchor) != 1:
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error("C5 source anchor changed")
    start = source.index(anchor)
    line_end = source.index("\n", start) + 1
    variable = "_missing_value" if class_index == 0 else "_wrong_value"
    insertion = f"        let {variable} = _{region}_anchor;\n"
    return f"{source[:line_end]}{insertion}{source[line_end:]}"


EPISODE_CANDIDATE_SOURCES = tuple(
    tuple(
        _insert_candidate_line(source, class_index=class_index, region_index=region_index)
        for class_index in range(2)
        for region_index in range(2)
    )
    for source in ORIGINAL_SOURCES
)
ALL_SOURCES = tuple(
    source
    for original, candidates in zip(
        ORIGINAL_SOURCES,
        EPISODE_CANDIDATE_SOURCES,
        strict=True,
    )
    for source in (original, *candidates)
)
SOURCE_SHA256S = (
    "767ac8ba99b72cfc2b148644c6e02f015dfe12bef19a1946288ccf4ae608f9cb",
    "962a5bcd733f4026a07dea5342aa5145ea837bd282edbf67d41418da0f37f6dd",
    "b8bda9c33ba7d62f481387d0da28042c91a651a1e0c99b717b868a1f9078d81a",
    "bb29c7ac1a1b36d79131fc7dac4c3264b3fb1e053c5c82e4c49d706e0f412bd4",
    "0e131b7e04223b7ba8bbd504e8f4d1b158474d68fe0ec3a8f383a1cd76835739",
    "e8dd412b73a6612ab6a70483ef738881ec09b2d9c429638cffab630648bcb3e8",
    "d23bee65de3f003933e9739add066b9c1973bb5760abfc6f15129b9b1fe169e3",
    "22d555f484f9aa33e1f5c184c5d91893aac1b02593ca5f1e4d28f1d9187418db",
    "5efc39f66d45b1252a76ddd536a251d42e5956aaa84f61e3a2763b7d1d2a3d79",
    "ee0b4c24a158c1d1e66bfbeaa5fcdccdaef641b5bd1b161f9a73054f9915e675",
    "08ffbca3d58b73e663e4516b0f9c564e1c6a2b4de33df0434c3ee933260c2312",
    "b0e5bff5168441980089f2114b942d695b8d3496d6a67b86c96f7ffc8ca25ff7",
    "13405df1dce5d55512a918165f5102de5526b3fef5ad3c5fa95435a717bf712b",
    "12ab6fb3efa6d3bf43b5d3903a743e004a1218656072dab32b08a3daf31fc7f8",
    "4ecf619642ddf0dee5466334c755544630f9d3656aea073c55e3a30777f4b79f",
)

CLASS_COMPONENT_SHA256S = (
    _digest(_COMPONENT_NAMESPACE, {"axis": 0, "value": "E0425"}),
    _digest(_COMPONENT_NAMESPACE, {"axis": 0, "value": "E0308"}),
)
REGION_COMPONENT_SHA256S = (
    _digest(_COMPONENT_NAMESPACE, {"axis": 1, "value": adapter.FIRST_HALF}),
    _digest(_COMPONENT_NAMESPACE, {"axis": 1, "value": adapter.SECOND_HALF}),
)
HYPOTHESES_BY_KEY = {
    (class_index, region_index): factors.seal_factorized_hypothesis_v1(
        (
            CLASS_COMPONENT_SHA256S[class_index],
            REGION_COMPONENT_SHA256S[region_index],
        )
    )
    for class_index in range(2)
    for region_index in range(2)
}
HYPOTHESES = tuple(HYPOTHESES_BY_KEY[key] for key in ((0, 0), (0, 1), (1, 0), (1, 1)))
H00_SHA256 = HYPOTHESES_BY_KEY[(0, 0)].sha256
H01_SHA256 = HYPOTHESES_BY_KEY[(0, 1)].sha256
H10_SHA256 = HYPOTHESES_BY_KEY[(1, 0)].sha256
H11_SHA256 = HYPOTHESES_BY_KEY[(1, 1)].sha256
EXPLORATION_HYPOTHESIS_SHA256S = (H00_SHA256, H11_SHA256)
EVALUATION_HYPOTHESIS_SHA256 = H01_SHA256

CANDIDATES = tuple(
    knowledge_actions.project_knowledge_action_candidate_v1(
        f"c5_h{class_index}{region_index}",
        f"opaque_c5.h{class_index}{region_index}()",
    )
    for class_index in range(2)
    for region_index in range(2)
)
HYPOTHESIS_CANDIDATE_ROWS = tuple(
    (hypothesis.sha256, candidate.sha256)
    for hypothesis, candidate in zip(HYPOTHESES, CANDIDATES, strict=True)
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
            "652e21dfb435032e94d7f0b72e5c5ed3c28faf2f4ba32c09cb203a57d5c785c9",
            "b0cfe61f4d7c74758f0e7b165011ae660210f26079274a4179e585ad9279f742",
            "d34b307aa6cef6e403ee88698897f73dfdb21511aee7b1b629baf5631eaab6d6",
            "5ea76f081fe59993eaabd0da33cf9f2b1f8429ff3ea876c3f6a86fb2f6892c69",
            "e6f89d9d2be762471863ff3c6a8b00dd3eef59ad388eed55388883198385429f",
            "182c0e0084471185b6d0cd16271bc43b1eb7561a49a8a099bbbf4027abe9117f",
            "9634b257c00cc1f0e2470cead58cc6ffec8136fb1a2388511b8c5f17b01dd9fc",
            "1494cd3c8887257e8bc2cc82aa07c35ff8a7f1371887f1f517e63be8f77ea68d",
            "197a5fcbb6bb71b839994f70b43718610a8ebd43e35dd1050afb3bc7d1dc6572",
            "96f90f7f7d3e6a3677ce8cd1e7ddb3cc9986a49d42180a5f74b2375a1b6ed985",
            "b3cbeea34d564dd583a681142ad27d672c4ca64f9a7386fbb6c319b868c6cf0e",
            "83d7dbb8d55f0f9b06d6750ac4a3552981ec30700f9f65e11c753cf76a5d0f49",
        )
    )
)


def _current_mechanism_file_sha256s() -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, sha256((_REPO_ROOT / path).read_bytes()).hexdigest())
        for path, _ in FROZEN_MECHANISM_FILE_SHA256S
    )


def output_sha256_v1(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(b"c5-rustc-diagnostic-factor-output-v1", {"outputs": outputs})


@dataclass(frozen=True, slots=True)
class C5DiagnosticFactorRecombinationSpecV1:
    frozen_mechanism_commit: str
    frozen_mechanism_file_sha256s: tuple[tuple[str, str], ...]
    diagnostic_arguments: tuple[str, ...]
    expected_diagnostics: tuple[tuple[str, str], ...]
    batches: tuple[tuple[tuple[int, ...], ...], ...]
    expected_output_sha256s: tuple[str, ...]
    source_sha256s: tuple[str, ...]
    hypothesis_sha256s: tuple[str, ...]
    candidate_sha256s: tuple[str, ...]
    hypothesis_candidate_rows: tuple[tuple[str, str], ...]
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
        derived_sources = tuple(
            sha256(source.encode("ascii")).hexdigest() for source in ALL_SOURCES
        )
        hypothesis_roots = tuple(hypothesis.sha256 for hypothesis in HYPOTHESES)
        candidate_roots = tuple(candidate.sha256 for candidate in CANDIDATES)
        if (
            self.frozen_mechanism_commit != FROZEN_MECHANISM_COMMIT
            or not _valid_commit(self.frozen_mechanism_commit)
            or self.frozen_mechanism_file_sha256s != FROZEN_MECHANISM_FILE_SHA256S
            or self.frozen_mechanism_file_sha256s != _current_mechanism_file_sha256s()
            or self.diagnostic_arguments != DIAGNOSTIC_ARGUMENTS
            or self.expected_diagnostics != EXPECTED_DIAGNOSTICS
            or self.batches != BATCHES
            or self.expected_output_sha256s
            != tuple(output_sha256_v1(outputs) for outputs in EXPECTED_OUTPUTS)
            or self.source_sha256s != SOURCE_SHA256S
            or self.source_sha256s != derived_sources
            or len(derived_sources) != 15
            or len(set(derived_sources)) != 15
            or self.hypothesis_sha256s != hypothesis_roots
            or len(set(hypothesis_roots)) != 4
            or self.candidate_sha256s != candidate_roots
            or len(set(candidate_roots)) != 4
            or self.hypothesis_candidate_rows != HYPOTHESIS_CANDIDATE_ROWS
            or dict(self.hypothesis_candidate_rows).keys() != set(hypothesis_roots)
            or set(dict(self.hypothesis_candidate_rows).values()) != set(candidate_roots)
            or self.exploration_hypothesis_sha256s != EXPLORATION_HYPOTHESIS_SHA256S
            or self.evaluation_hypothesis_sha256 != EVALUATION_HYPOTHESIS_SHA256
            or self.excluded_source_sha256s != EXCLUDED_PRIOR_SOURCE_SHA256S
            or len(self.excluded_source_sha256s) != 36
            or set(derived_sources).intersection(self.excluded_source_sha256s)
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 public fixture specification changed"
            )


def make_c5_diagnostic_factor_recombination_spec_v1() -> C5DiagnosticFactorRecombinationSpecV1:
    partial = C5DiagnosticFactorRecombinationSpecV1(
        frozen_mechanism_commit=FROZEN_MECHANISM_COMMIT,
        frozen_mechanism_file_sha256s=FROZEN_MECHANISM_FILE_SHA256S,
        diagnostic_arguments=DIAGNOSTIC_ARGUMENTS,
        expected_diagnostics=EXPECTED_DIAGNOSTICS,
        batches=BATCHES,
        expected_output_sha256s=tuple(output_sha256_v1(outputs) for outputs in EXPECTED_OUTPUTS),
        source_sha256s=SOURCE_SHA256S,
        hypothesis_sha256s=tuple(hypothesis.sha256 for hypothesis in HYPOTHESES),
        candidate_sha256s=tuple(candidate.sha256 for candidate in CANDIDATES),
        hypothesis_candidate_rows=HYPOTHESIS_CANDIDATE_ROWS,
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


DEFAULT_PUBLIC_SPEC = make_c5_diagnostic_factor_recombination_spec_v1()


@dataclass(frozen=True, slots=True)
class C5HypothesisCandidateCatalogueBindingV1:
    catalogue_sha256: str
    rows: tuple[tuple[str, str], ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "catalogue_sha256": self.catalogue_sha256,
            "rows": self.rows,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            not _valid_sha256(self.catalogue_sha256)
            or type(self.rows) is not tuple
            or self.rows != tuple(sorted(HYPOTHESIS_CANDIDATE_ROWS))
            or len(self.rows) != 4
            or len({left for left, _ in self.rows}) != 4
            or len({right for _, right in self.rows}) != 4
            or self.sha256
            != _digest(
                _CATALOGUE_BINDING_NAMESPACE,
                self.record(include_sha256=False),
            )
        ):
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 hypothesis/candidate catalogue binding changed"
            )

    def candidate_for(self, hypothesis_sha256: str) -> str:
        self.authenticate()
        try:
            return dict(self.rows)[hypothesis_sha256]
        except KeyError as exc:
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 selected hypothesis is outside the frozen binding"
            ) from exc


def _seal_catalogue_binding(
    prepared: interventions.PreparedRustActionInterventionV1,
) -> C5HypothesisCandidateCatalogueBindingV1:
    if set(prepared.candidate_sha256s) != {candidate for _, candidate in HYPOTHESIS_CANDIDATE_ROWS}:
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
            "C5 intervention catalogue is not the full frozen candidate bank"
        )
    payload = {
        "catalogue_sha256": prepared.catalogue_sha256,
        "rows": tuple(sorted(HYPOTHESIS_CANDIDATE_ROWS)),
    }
    result = C5HypothesisCandidateCatalogueBindingV1(
        **payload,
        sha256=_digest(_CATALOGUE_BINDING_NAMESPACE, payload),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C5RunReservationV1:
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
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 run reservation changed"
            )


@dataclass(frozen=True, slots=True)
class C5PublicConsequenceV1:
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
            not _valid_sha256(self.source_sha256)
            or not _valid_sha256(self.batch_sha256)
            or not _valid_sha256(self.output_feedback_sha256)
            or not _valid_sha256(self.execution_sha256)
            or type(self.observed_outputs) is not tuple
            or type(self.expected_outputs) is not tuple
            or any(
                type(batch) is not tuple or any(type(value) is not int for value in batch)
                for outputs in (self.observed_outputs, self.expected_outputs)
                for batch in outputs
            )
            or type(self.passed_cell) is not int
            or self.passed_cell not in (0, 1)
            or self.passed_cell != int(self.observed_outputs == self.expected_outputs)
            or self.sha256 != _digest(_CONSEQUENCE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 public consequence authentication failed"
            )


def _seal_consequence(
    *,
    execution: RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
    output_feedback_sha256: str,
) -> C5PublicConsequenceV1:
    payload = {
        "source_sha256": execution.selected_source.source_sha256,
        "batch_sha256": execution.preparation.batch_sha256,
        "output_feedback_sha256": output_feedback_sha256,
        "execution_sha256": execution.sha256,
        "observed_outputs": execution.outputs,
        "expected_outputs": expected_outputs,
        "passed_cell": int(execution.outputs == expected_outputs),
    }
    result = C5PublicConsequenceV1(
        **payload,
        sha256=_digest(_CONSEQUENCE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C5SelectedExecutionAttemptV1:
    ordinal: int
    role: str
    selected_hypothesis_sha256: str
    candidate_sha256: str
    binding_receipt_sha256: str
    selected_binding_sha256: str
    selection: interventions.RustActionInterventionSelectionV1
    execution: RustSelectedActionExecutionV2
    consequence: C5PublicConsequenceV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "ordinal": self.ordinal,
            "role": self.role,
            "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
            "candidate_sha256": self.candidate_sha256,
            "binding_receipt_sha256": self.binding_receipt_sha256,
            "selected_binding_sha256": self.selected_binding_sha256,
            "selection_sha256": self.selection.sha256,
            "execution_sha256": self.execution.sha256,
            "consequence": self.consequence.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        self.selection.authenticate_structure()
        self.execution.authenticate_structure()
        self.consequence.authenticate()
        candidate_by_root = {candidate.sha256: candidate for candidate in CANDIDATES}
        if self.candidate_sha256 not in candidate_by_root:
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 selected candidate is outside the frozen bank"
            )
        expected_candidate = candidate_by_root[self.candidate_sha256]
        expected_binding = _digest(
            _SELECTED_BINDING_NAMESPACE,
            {
                "binding_receipt_sha256": self.binding_receipt_sha256,
                "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
                "candidate_sha256": self.candidate_sha256,
                "selection_sha256": self.selection.sha256,
            },
        )
        if (
            type(self.ordinal) is not int
            or not 1 <= self.ordinal <= 3
            or self.role != ROLES[self.ordinal - 1]
            or not _valid_sha256(self.selected_hypothesis_sha256)
            or not _valid_sha256(self.candidate_sha256)
            or not _valid_sha256(self.binding_receipt_sha256)
            or self.selected_binding_sha256 != expected_binding
            or self.selection.candidate_sha256 != self.candidate_sha256
            or self.execution.selected_source.selection_sha256 != self.selection.sha256
            or self.execution.selected_source.action_handle != expected_candidate.candidate_id
            or self.execution.selected_source.source_sha256 != self.consequence.source_sha256
            or self.execution.sha256 != self.consequence.execution_sha256
            or self.execution.preparation.batch_sha256 != self.consequence.batch_sha256
            or self.execution.outputs != self.consequence.observed_outputs
            or self.execution.compile_receipt.compile_invocations != 1
            or self.execution.run_receipt.run_invocations != 1
            or self.sha256 != _digest(_ATTEMPT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 selected execution attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C5WorkV1:
    diagnostic_compile_count: int
    selected_candidate_compile_count: int
    selected_candidate_run_count: int
    output_feedback_open_count: int
    memory_advance_count: int
    evaluation_memory_advance_count: int
    unselected_source_release_count: int
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

    def _authenticate_result(self) -> None:
        self.spec.authenticate()
        self.reservation.authenticate(destination=Path(self.reservation.destination))
        objects = (
            self.catalogue_bindings,
            self.diagnostic_attempts,
            self.observations,
            self.adaptation_normal_decisions,
            self.exploration_decisions,
            self.adaptation_outcomes,
            self.evaluation_decisions,
            self.control_decisions,
            self.selected_attempts,
            self.authorities,
            self.catalogues,
            self.authority_work_counters,
        )
        expected_lengths = (3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3)
        if tuple(map(len, objects)) != expected_lengths:
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 result tuple cardinality changed"
            )

        self.parent_memory.authenticate()
        self.first_memory.authenticate()
        self.final_memory.authenticate()
        self.work.authenticate()
        for value in self.catalogue_bindings:
            value.authenticate()
        for value in self.diagnostic_attempts:
            value.authenticate()
        for value in self.observations:
            value.authenticate()
        for decisions in (
            self.adaptation_normal_decisions,
            self.exploration_decisions,
            self.evaluation_decisions,
            self.control_decisions,
        ):
            for value in decisions:
                value.authenticate()
        for value in self.adaptation_outcomes:
            value.authenticate()
        for value in self.selected_attempts:
            value.authenticate()
        for value in self.authorities:
            value.authenticate_structure()
        for value in self.catalogues:
            value.authenticate_structure()
        for value in self.authority_work_counters:
            value.authenticate_structure()

        original_source_roots = SOURCE_SHA256S[0::5]
        expected_observations = tuple(
            adapter.project_rust_factorized_feedback_v1(
                attempt,
                selection_context_sha256=_context_sha256(
                    spec_sha256=self.spec.sha256,
                    role=role,
                    source_sha256=source_sha256,
                ),
                active_hypothesis_sha256s=self.parent_memory.hypothesis_sha256s,
            )
            for attempt, role, source_sha256 in zip(
                self.diagnostic_attempts,
                ROLES,
                original_source_roots,
                strict=True,
            )
        )
        for attempt, (expected_code, expected_region), source, source_sha256 in zip(
            self.diagnostic_attempts,
            EXPECTED_DIAGNOSTICS,
            ORIGINAL_SOURCES,
            original_source_roots,
            strict=True,
        ):
            spans = tuple(
                span
                for diagnostic in attempt.error_diagnostics
                for span in diagnostic.primary_spans
            )
            regions = {
                adapter.FIRST_HALF
                if 2 * ((span.byte_start + span.byte_end) // 2) < len(source.encode("ascii"))
                else adapter.SECOND_HALF
                for span in spans
            }
            if (
                attempt.status != adapter.COMPILE_FAILED
                or attempt.rustc_identity_sha256 != self.reservation.rustc_identity_sha256
                or attempt.source_sha256 != source_sha256
                or attempt.source_size_bytes != len(source.encode("ascii"))
                or attempt.arguments != DIAGNOSTIC_ARGUMENTS
                or tuple(diagnostic.code for diagnostic in attempt.error_diagnostics)
                != (expected_code,)
                or regions != {expected_region}
            ):
                raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                    "C5 diagnostic attempt changed"
                )

        expected_parent = factors.start_factorized_feedback_hypothesis_memory_v1(HYPOTHESES)
        expected_normal_first = self.parent_memory.route(self.observations[0])
        expected_exploration_first = self.parent_memory.route(
            self.observations[0],
            exploration_hypothesis_sha256=H00_SHA256,
        )
        expected_first_outcome = factors.seal_factorized_feedback_outcome_v1(
            decision=expected_exploration_first,
            proposal_sha256=self.selected_attempts[0].selected_binding_sha256,
            execution_sha256=self.selected_attempts[0].execution.sha256,
            consequence_sha256=self.selected_attempts[0].consequence.sha256,
            feedback_sha256=self.output_feedback_sha256s[0],
            passed_cell=1,
        )
        expected_first = factors.advance_factorized_feedback_hypothesis_memory_v1(
            self.parent_memory,
            expected_first_outcome,
        )
        expected_normal_second = self.first_memory.route(self.observations[1])
        expected_exploration_second = self.first_memory.route(
            self.observations[1],
            exploration_hypothesis_sha256=H11_SHA256,
        )
        expected_second_outcome = factors.seal_factorized_feedback_outcome_v1(
            decision=expected_exploration_second,
            proposal_sha256=self.selected_attempts[1].selected_binding_sha256,
            execution_sha256=self.selected_attempts[1].execution.sha256,
            consequence_sha256=self.selected_attempts[1].consequence.sha256,
            feedback_sha256=self.output_feedback_sha256s[1],
            passed_cell=1,
        )
        expected_final = factors.advance_factorized_feedback_hypothesis_memory_v1(
            self.first_memory,
            expected_second_outcome,
        )
        expected_evaluation = (
            self.parent_memory.route(self.observations[2]),
            self.first_memory.route(self.observations[2]),
            self.final_memory.route(self.observations[2]),
        )
        expected_controls = (
            self.final_memory.route(self.observations[2], outcome_gate=False),
            self.final_memory.route(self.observations[2], axis_gates=(False, True)),
            self.final_memory.route(self.observations[2], axis_gates=(True, False)),
        )

        candidate_mapping = dict(self.spec.hypothesis_candidate_rows)
        selected_hypotheses = (H00_SHA256, H11_SHA256, H01_SHA256)
        expected_candidate_sources = tuple(
            EPISODE_CANDIDATE_SOURCES[episode][
                self.spec.hypothesis_sha256s.index(hypothesis_sha256)
            ]
            for episode, hypothesis_sha256 in enumerate(selected_hypotheses)
        )
        expected_final_scores = {
            H00_SHA256: 2,
            H01_SHA256: 4,
            H10_SHA256: 0,
            H11_SHA256: 2,
        }
        if (
            self.status != STATUS_PASS
            or self.spec.sha256 != DEFAULT_PUBLIC_SPEC.sha256
            or self.reservation.spec_sha256 != self.spec.sha256
            or self.observations != expected_observations
            or self.parent_memory.checkpoint_sha256 != expected_parent.checkpoint_sha256
            or self.adaptation_normal_decisions != (expected_normal_first, expected_normal_second)
            or any(
                decision.selected_hypothesis_sha256 is not None
                for decision in self.adaptation_normal_decisions
            )
            or self.exploration_decisions
            != (expected_exploration_first, expected_exploration_second)
            or tuple(decision.selected_hypothesis_sha256 for decision in self.exploration_decisions)
            != (H00_SHA256, H11_SHA256)
            or self.adaptation_outcomes != (expected_first_outcome, expected_second_outcome)
            or self.first_memory.checkpoint_sha256 != expected_first.checkpoint_sha256
            or self.final_memory.checkpoint_sha256 != expected_final.checkpoint_sha256
            or len(self.final_memory.provenance) != 2
            or self.evaluation_decisions != expected_evaluation
            or tuple(decision.selected_hypothesis_sha256 for decision in self.evaluation_decisions)
            != (None, None, H01_SHA256)
            or _decision_scores(self.evaluation_decisions[2]) != expected_final_scores
            or self.control_decisions != expected_controls
            or any(
                decision.selected_hypothesis_sha256 is not None
                for decision in self.control_decisions
            )
            or tuple(attempt.selected_hypothesis_sha256 for attempt in self.selected_attempts)
            != selected_hypotheses
            or tuple(attempt.ordinal for attempt in self.selected_attempts) != (1, 2, 3)
            or tuple(attempt.role for attempt in self.selected_attempts) != ROLES
            or tuple(attempt.binding_receipt_sha256 for attempt in self.selected_attempts)
            != tuple(binding.sha256 for binding in self.catalogue_bindings)
            or tuple(attempt.candidate_sha256 for attempt in self.selected_attempts)
            != tuple(candidate_mapping[value] for value in selected_hypotheses)
            or tuple(attempt.execution.selected_source.source for attempt in self.selected_attempts)
            != expected_candidate_sources
            or tuple(attempt.execution.rustc_identity.sha256 for attempt in self.selected_attempts)
            != (self.reservation.rustc_identity_sha256,) * 3
            or tuple(attempt.consequence.expected_outputs for attempt in self.selected_attempts)
            != EXPECTED_OUTPUTS
            or any(attempt.consequence.passed_cell != 1 for attempt in self.selected_attempts)
            or tuple(binding.catalogue_sha256 for binding in self.catalogue_bindings)
            != tuple(catalogue.sha256 for catalogue in self.catalogues)
            or tuple(
                (
                    attempt.selection.authority_sha256,
                    attempt.selection.catalogue_sha256,
                )
                for attempt in self.selected_attempts
            )
            != tuple(
                (authority.sha256, catalogue.sha256)
                for authority, catalogue in zip(
                    self.authorities,
                    self.catalogues,
                    strict=True,
                )
            )
            or any(
                authority.catalogue_sha256 != catalogue.sha256
                for authority, catalogue in zip(
                    self.authorities,
                    self.catalogues,
                    strict=True,
                )
            )
            or any(
                set(catalogue.candidate_sha256s) != set(self.spec.candidate_sha256s)
                for catalogue in self.catalogues
            )
            or tuple(set(catalogue.rust_source_sha256s) for catalogue in self.catalogues)
            != tuple(
                {sha256(source.encode("ascii")).hexdigest() for source in candidate_sources}
                for candidate_sources in EPISODE_CANDIDATE_SOURCES
            )
            or any(
                counters != _EXPECTED_INTERVENTION_COUNTERS
                for counters in self.authority_work_counters
            )
            or len(set(self.output_feedback_sha256s)) != 3
            or any(not _valid_sha256(value) for value in self.output_feedback_sha256s)
            or self.output_feedback_sha256s
            != tuple(
                attempt.consequence.output_feedback_sha256 for attempt in self.selected_attempts
            )
            or any(
                factors.advance_factorized_feedback_hypothesis_memory_v1(
                    parent,
                    outcome,
                    plasticity_gate=False,
                )
                is not parent
                or factors.advance_factorized_feedback_hypothesis_memory_v1(
                    parent,
                    outcome,
                    consequence_gate=False,
                )
                is not parent
                for parent, outcome in (
                    (self.parent_memory, self.adaptation_outcomes[0]),
                    (self.first_memory, self.adaptation_outcomes[1]),
                )
            )
            or self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 cohort result authentication failed"
            )

    def authenticate(self) -> None:
        if (
            self.diagnostic_compile_count,
            self.selected_candidate_compile_count,
            self.selected_candidate_run_count,
            self.output_feedback_open_count,
            self.memory_advance_count,
            self.evaluation_memory_advance_count,
            self.unselected_source_release_count,
        ) != (3, 3, 3, 3, 2, 0, 0) or self.sha256 != _digest(
            _WORK_NAMESPACE, self.record(include_sha256=False)
        ):
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 work accounting changed"
            )


DiagnosticCompileCallback = Callable[
    [str, str],
    adapter.RustcDiagnosticCompileAttemptV1,
]
ExecutionCallback = Callable[
    [
        interventions.PreparedRustActionInterventionV1,
        interventions.RustActionInterventionSelectionV1,
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


def _prepare_episode(
    episode: int,
) -> interventions.PreparedRustActionInterventionV1:
    return interventions.prepare_rust_action_intervention_v1(
        tuple(
            (candidate, source)
            for candidate, source in zip(
                CANDIDATES,
                EPISODE_CANDIDATE_SOURCES[episode],
                strict=True,
            )
        )
    )


def _validate_outputs(
    outputs: object,
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
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
            "C5 public output feedback shape changed"
        )
    return outputs


def _compile_diagnostic_once(
    identity: rust_executor.RustcIdentityV1,
    source: str,
) -> adapter.RustcDiagnosticCompileAttemptV1:
    identity.authenticate_structure()
    source_bytes = source.encode("ascii")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory(prefix="humanbrain-c5-diagnostic-") as directory:
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
    prepared: interventions.PreparedRustActionInterventionV1,
    selection: interventions.RustActionInterventionSelectionV1,
    batch: tuple[tuple[int, ...], ...],
) -> RustSelectedActionExecutionV2:
    return prepared.execute_selected(selection, batch)


def _run_selected_attempt(
    *,
    ordinal: int,
    spec: C5DiagnosticFactorRecombinationSpecV1,
    prepared: interventions.PreparedRustActionInterventionV1,
    binding: C5HypothesisCandidateCatalogueBindingV1,
    decision: factors.FactorizedFeedbackDecisionV1,
    batch: tuple[tuple[int, ...], ...],
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> tuple[C5SelectedExecutionAttemptV1, str]:
    hypothesis_sha256 = decision.selected_hypothesis_sha256
    if hypothesis_sha256 is None:
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
            "C5 STOP decision cannot release a source"
        )
    role = ROLES[ordinal - 1]
    candidate_sha256 = binding.candidate_for(hypothesis_sha256)
    intervention_sha256 = _digest(
        _INTERVENTION_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "ordinal": ordinal,
            "role": role,
            "decision_sha256": decision.sha256,
            "binding_receipt_sha256": binding.sha256,
            "hypothesis_sha256": hypothesis_sha256,
            "candidate_sha256": candidate_sha256,
            "batch": batch,
        },
    )
    selection = prepared.issue(candidate_sha256, intervention_sha256)
    execution = execute(prepared, selection, batch)
    if type(execution) is not RustSelectedActionExecutionV2:
        raise TypeError("C5 executor must return one exact RustSelectedActionExecutionV2")
    execution.authenticate_structure()
    if (
        selection.authority_sha256 != prepared.authority_sha256
        or selection.catalogue_sha256 != prepared.catalogue_sha256
        or execution.selected_source.authority_sha256 != prepared.authority_sha256
        or execution.selected_source.selection_sha256 != selection.sha256
        or execution.compile_receipt.compile_invocations != 1
        or execution.run_receipt.run_invocations != 1
        or len(execution.outputs) != len(batch)
    ):
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
            "C5 selected execution custody changed"
        )

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
    selected_binding_sha256 = _digest(
        _SELECTED_BINDING_NAMESPACE,
        {
            "binding_receipt_sha256": binding.sha256,
            "selected_hypothesis_sha256": hypothesis_sha256,
            "candidate_sha256": candidate_sha256,
            "selection_sha256": selection.sha256,
        },
    )
    payload = {
        "ordinal": ordinal,
        "role": role,
        "selected_hypothesis_sha256": hypothesis_sha256,
        "candidate_sha256": candidate_sha256,
        "binding_receipt_sha256": binding.sha256,
        "selected_binding_sha256": selected_binding_sha256,
        "selection": selection,
        "execution": execution,
        "consequence": consequence,
    }
    attempt = C5SelectedExecutionAttemptV1(
        **payload,
        sha256=_digest(
            _ATTEMPT_NAMESPACE,
            {
                "ordinal": ordinal,
                "role": role,
                "selected_hypothesis_sha256": hypothesis_sha256,
                "candidate_sha256": candidate_sha256,
                "binding_receipt_sha256": binding.sha256,
                "selected_binding_sha256": selected_binding_sha256,
                "selection_sha256": selection.sha256,
                "execution_sha256": execution.sha256,
                "consequence": consequence.record(),
            },
        ),
    )
    attempt.authenticate()
    if consequence.passed_cell != 1:
        raise _behavior_failure(
            "C5 selected candidate failed public output feedback",
            stage=role,
            decision=decision.record(),
            catalogue_binding=binding.record(),
            selected_attempt={
                "attempt": attempt.record(),
                "selection": _plain(attempt.selection),
                "execution": _plain(attempt.execution),
                "consequence": attempt.consequence.record(),
            },
            authority_work_counters=prepared.work_counters,
        )
    return attempt, output_feedback_sha256


def _seal_work() -> C5WorkV1:
    payload = {
        "diagnostic_compile_count": 3,
        "selected_candidate_compile_count": 3,
        "selected_candidate_run_count": 3,
        "output_feedback_open_count": 3,
        "memory_advance_count": 2,
        "evaluation_memory_advance_count": 0,
        "unselected_source_release_count": 0,
    }
    result = C5WorkV1(
        **payload,
        sha256=_digest(_WORK_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _run_reserved_cohort_v1(
    *,
    spec: C5DiagnosticFactorRecombinationSpecV1,
    reservation: C5RunReservationV1,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> C5DiagnosticFactorRecombinationResultV1:
    spec.authenticate()
    reservation.authenticate(destination=Path(reservation.destination))
    if reservation.spec_sha256 != spec.sha256:
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
            "C5 reservation/spec custody changed"
        )

    parent = factors.start_factorized_feedback_hypothesis_memory_v1(HYPOTHESES)
    prepared = tuple(_prepare_episode(index) for index in range(3))
    bindings = tuple(_seal_catalogue_binding(value) for value in prepared)
    if any(
        value.work_counters != interventions.RustActionInterventionWorkCountersV1(0, 0, 0, 0, 0)
        for value in prepared
    ):
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
            "C5 authority performed work before the first diagnostic"
        )

    diagnostic_attempts: list[adapter.RustcDiagnosticCompileAttemptV1] = []
    observations: list[factors.FactorizedFeedbackObservationV1] = []
    selected_attempts: list[C5SelectedExecutionAttemptV1] = []
    output_feedback_sha256s: list[str] = []

    first_diagnostic = diagnostic_compile(
        reservation.rustc_identity_sha256,
        ORIGINAL_SOURCES[0],
    )
    diagnostic_attempts.append(first_diagnostic)
    first_observation = adapter.project_rust_factorized_feedback_v1(
        first_diagnostic,
        selection_context_sha256=_context_sha256(
            spec_sha256=spec.sha256,
            role=ROLES[0],
            source_sha256=SOURCE_SHA256S[0],
        ),
        active_hypothesis_sha256s=parent.hypothesis_sha256s,
    )
    observations.append(first_observation)
    normal_first = parent.route(first_observation)
    exploration_first = parent.route(
        first_observation,
        exploration_hypothesis_sha256=H00_SHA256,
    )
    if (
        normal_first.selected_hypothesis_sha256 is not None
        or exploration_first.selected_hypothesis_sha256 != H00_SHA256
    ):
        raise _behavior_failure(
            "C5 first adaptation selection behavior changed",
            stage="adaptation-first-selection",
            diagnostic_attempt=first_diagnostic.record(),
            observation=first_observation.record(),
            parent_memory=_memory_record(parent),
            normal_decision=normal_first.record(),
            exploration_decision=exploration_first.record(),
        )
    first_attempt, first_feedback_sha256 = _run_selected_attempt(
        ordinal=1,
        spec=spec,
        prepared=prepared[0],
        binding=bindings[0],
        decision=exploration_first,
        batch=BATCHES[0],
        execute=execute,
        output_feedback=output_feedback,
    )
    selected_attempts.append(first_attempt)
    output_feedback_sha256s.append(first_feedback_sha256)
    first_outcome = factors.seal_factorized_feedback_outcome_v1(
        decision=exploration_first,
        proposal_sha256=first_attempt.selected_binding_sha256,
        execution_sha256=first_attempt.execution.sha256,
        consequence_sha256=first_attempt.consequence.sha256,
        feedback_sha256=first_feedback_sha256,
        passed_cell=1,
    )
    first_learning_controls = (
        factors.advance_factorized_feedback_hypothesis_memory_v1(
            parent,
            first_outcome,
            plasticity_gate=False,
        ),
        factors.advance_factorized_feedback_hypothesis_memory_v1(
            parent,
            first_outcome,
            consequence_gate=False,
        ),
    )
    if any(control is not parent for control in first_learning_controls):
        raise _behavior_failure(
            "C5 first learning gate control changed",
            stage="adaptation-first-learning-controls",
            diagnostic_attempt=first_diagnostic.record(),
            observation=first_observation.record(),
            exploration_decision=exploration_first.record(),
            selected_attempt=first_attempt.record(),
            outcome=first_outcome.record(),
            parent_memory=_memory_record(parent),
            control_memories=tuple(_memory_record(control) for control in first_learning_controls),
        )
    first_memory = factors.advance_factorized_feedback_hypothesis_memory_v1(
        parent,
        first_outcome,
    )

    second_diagnostic = diagnostic_compile(
        reservation.rustc_identity_sha256,
        ORIGINAL_SOURCES[1],
    )
    diagnostic_attempts.append(second_diagnostic)
    second_observation = adapter.project_rust_factorized_feedback_v1(
        second_diagnostic,
        selection_context_sha256=_context_sha256(
            spec_sha256=spec.sha256,
            role=ROLES[1],
            source_sha256=SOURCE_SHA256S[5],
        ),
        active_hypothesis_sha256s=first_memory.hypothesis_sha256s,
    )
    observations.append(second_observation)
    normal_second = first_memory.route(second_observation)
    exploration_second = first_memory.route(
        second_observation,
        exploration_hypothesis_sha256=H11_SHA256,
    )
    if (
        normal_second.selected_hypothesis_sha256 is not None
        or exploration_second.selected_hypothesis_sha256 != H11_SHA256
    ):
        raise _behavior_failure(
            "C5 second adaptation selection behavior changed",
            stage="adaptation-second-selection",
            diagnostic_attempts=tuple(item.record() for item in diagnostic_attempts),
            observations=tuple(item.record() for item in observations),
            first_memory=_memory_record(first_memory),
            normal_decision=normal_second.record(),
            exploration_decision=exploration_second.record(),
            prior_selected_attempt=first_attempt.record(),
            prior_outcome=first_outcome.record(),
        )
    second_attempt, second_feedback_sha256 = _run_selected_attempt(
        ordinal=2,
        spec=spec,
        prepared=prepared[1],
        binding=bindings[1],
        decision=exploration_second,
        batch=BATCHES[1],
        execute=execute,
        output_feedback=output_feedback,
    )
    selected_attempts.append(second_attempt)
    output_feedback_sha256s.append(second_feedback_sha256)
    second_outcome = factors.seal_factorized_feedback_outcome_v1(
        decision=exploration_second,
        proposal_sha256=second_attempt.selected_binding_sha256,
        execution_sha256=second_attempt.execution.sha256,
        consequence_sha256=second_attempt.consequence.sha256,
        feedback_sha256=second_feedback_sha256,
        passed_cell=1,
    )
    second_learning_controls = (
        factors.advance_factorized_feedback_hypothesis_memory_v1(
            first_memory,
            second_outcome,
            plasticity_gate=False,
        ),
        factors.advance_factorized_feedback_hypothesis_memory_v1(
            first_memory,
            second_outcome,
            consequence_gate=False,
        ),
    )
    if any(control is not first_memory for control in second_learning_controls):
        raise _behavior_failure(
            "C5 second learning gate control changed",
            stage="adaptation-second-learning-controls",
            diagnostic_attempts=tuple(item.record() for item in diagnostic_attempts),
            observations=tuple(item.record() for item in observations),
            exploration_decisions=(exploration_first.record(), exploration_second.record()),
            selected_attempts=(first_attempt.record(), second_attempt.record()),
            outcomes=(first_outcome.record(), second_outcome.record()),
            first_memory=_memory_record(first_memory),
            control_memories=tuple(_memory_record(control) for control in second_learning_controls),
        )
    final_memory = factors.advance_factorized_feedback_hypothesis_memory_v1(
        first_memory,
        second_outcome,
    )

    evaluation_diagnostic = diagnostic_compile(
        reservation.rustc_identity_sha256,
        ORIGINAL_SOURCES[2],
    )
    diagnostic_attempts.append(evaluation_diagnostic)
    evaluation_observation = adapter.project_rust_factorized_feedback_v1(
        evaluation_diagnostic,
        selection_context_sha256=_context_sha256(
            spec_sha256=spec.sha256,
            role=ROLES[2],
            source_sha256=SOURCE_SHA256S[10],
        ),
        active_hypothesis_sha256s=final_memory.hypothesis_sha256s,
    )
    observations.append(evaluation_observation)
    evaluation_decisions = (
        parent.route(evaluation_observation),
        first_memory.route(evaluation_observation),
        final_memory.route(evaluation_observation),
    )
    controls = (
        final_memory.route(evaluation_observation, outcome_gate=False),
        final_memory.route(evaluation_observation, axis_gates=(False, True)),
        final_memory.route(evaluation_observation, axis_gates=(True, False)),
    )
    expected_evaluation_selections = (None, None, H01_SHA256)
    expected_final_scores = {
        H00_SHA256: 2,
        H01_SHA256: 4,
        H10_SHA256: 0,
        H11_SHA256: 2,
    }
    if (
        tuple(decision.selected_hypothesis_sha256 for decision in evaluation_decisions)
        != expected_evaluation_selections
        or any(decision.selected_hypothesis_sha256 is not None for decision in controls)
        or _decision_scores(evaluation_decisions[2]) != expected_final_scores
    ):
        raise _behavior_failure(
            "C5 evaluation recombination or causal-control behavior changed",
            stage="evaluation-selection",
            diagnostic_attempts=tuple(item.record() for item in diagnostic_attempts),
            observations=tuple(item.record() for item in observations),
            memories=tuple(
                _memory_record(memory) for memory in (parent, first_memory, final_memory)
            ),
            adaptation_normal_decisions=(normal_first.record(), normal_second.record()),
            exploration_decisions=(exploration_first.record(), exploration_second.record()),
            adaptation_outcomes=(first_outcome.record(), second_outcome.record()),
            selected_attempts=(first_attempt.record(), second_attempt.record()),
            evaluation_decisions=tuple(decision.record() for decision in evaluation_decisions),
            control_decisions=tuple(decision.record() for decision in controls),
            final_scores=_decision_scores(evaluation_decisions[2]),
        )
    evaluation_attempt, evaluation_feedback_sha256 = _run_selected_attempt(
        ordinal=3,
        spec=spec,
        prepared=prepared[2],
        binding=bindings[2],
        decision=evaluation_decisions[2],
        batch=BATCHES[2],
        execute=execute,
        output_feedback=output_feedback,
    )
    selected_attempts.append(evaluation_attempt)
    output_feedback_sha256s.append(evaluation_feedback_sha256)

    work = _seal_work()
    payload = {
        "status": STATUS_PASS,
        "spec": spec,
        "reservation": reservation,
        "catalogue_bindings": bindings,
        "diagnostic_attempts": tuple(diagnostic_attempts),
        "observations": tuple(observations),
        "parent_memory": parent,
        "first_memory": first_memory,
        "final_memory": final_memory,
        "adaptation_normal_decisions": (normal_first, normal_second),
        "exploration_decisions": (exploration_first, exploration_second),
        "adaptation_outcomes": (first_outcome, second_outcome),
        "evaluation_decisions": evaluation_decisions,
        "control_decisions": controls,
        "selected_attempts": tuple(selected_attempts),
        "authorities": tuple(value.authority for value in prepared),
        "catalogues": tuple(value.catalogue for value in prepared),
        "authority_work_counters": tuple(value.work_counters for value in prepared),
        "output_feedback_sha256s": tuple(output_feedback_sha256s),
        "work": work,
    }
    partial = C5DiagnosticFactorRecombinationResultV1(**payload, sha256="")
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _reserve_destination(
    spec: C5DiagnosticFactorRecombinationSpecV1,
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
) -> C5RunReservationV1:
    spec.authenticate()
    if not _valid_commit(preregistration_commit):
        raise ValueError("C5 preregistration commit must be 40 lowercase hex characters")
    if not _valid_sha256(rustc_identity_sha256):
        raise ValueError("C5 rustc identity root is invalid")
    destination = Path(destination).resolve()
    reservation_path = Path(reservation_path).resolve()
    expected_reservation = destination.with_name(f"{destination.name}.reservation.json")
    if reservation_path != expected_reservation:
        raise ValueError("C5 reservation must be the result's exact sibling")
    if os.path.lexists(destination) or os.path.lexists(reservation_path):
        raise FileExistsError(destination if os.path.lexists(destination) else reservation_path)
    partial = C5RunReservationV1(
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
            raise OSError("C5 reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if reservation_path.read_bytes() != payload:
        raise OSError("C5 reservation readback changed")
    return reservation


def _result_document(
    result: C5DiagnosticFactorRecombinationResultV1,
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
            "catalogue_bindings": tuple(item.record() for item in result.catalogue_bindings),
            "diagnostic_attempts": tuple(item.record() for item in result.diagnostic_attempts),
            "observations": tuple(item.record() for item in result.observations),
            "memories": tuple(
                _memory_record(item)
                for item in (
                    result.parent_memory,
                    result.first_memory,
                    result.final_memory,
                )
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
                    "selection": _plain(item.selection),
                    "execution": _plain(item.execution),
                    "consequence": item.consequence.record(),
                }
                for item in result.selected_attempts
            ),
            "authorities": tuple(_plain(item) for item in result.authorities),
            "catalogues": tuple(_plain(item) for item in result.catalogues),
            "authority_work_counters": tuple(
                _plain(item) for item in result.authority_work_counters
            ),
            "work": result.work.record(),
        },
    }
    return {
        **body,
        "document_sha256": _digest(_DOCUMENT_NAMESPACE, body),
    }


def _abort_document(
    *,
    spec: C5DiagnosticFactorRecombinationSpecV1,
    reservation: C5RunReservationV1,
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
    return {
        **body,
        "document_sha256": _digest(_DOCUMENT_NAMESPACE, body),
    }


def _failure_document(
    *,
    spec: C5DiagnosticFactorRecombinationSpecV1,
    reservation: C5RunReservationV1,
    error: C5BehaviorFailure,
) -> dict[str, object]:
    partial_transcript = _plain(error.partial_transcript)
    if type(partial_transcript) is not dict:
        raise TypeError("C5 behavior failure transcript must be one exact object")
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
    return {
        **body,
        "document_sha256": _digest(_DOCUMENT_NAMESPACE, body),
    }


def _publish_document(destination: Path, document: dict[str, object]) -> None:
    payload = _canonical_json(document) + b"\n"
    with destination.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("C5 terminal document write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = destination.read_bytes()
    if (
        retained != payload
        or _canonical_json(json.loads(retained.decode("ascii"))) + b"\n" != payload
    ):
        raise OSError("C5 terminal document readback changed")
    decoded = json.loads(retained.decode("ascii"))
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if document_sha256 != _digest(_DOCUMENT_NAMESPACE, body):
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
            "C5 terminal document root changed"
        )
    terminal_status = decoded["terminal_status"]
    if terminal_status == STATUS_PASS:
        record = dict(decoded["authenticated_result_record"])
        result_sha256 = record.pop("sha256", None)
        if result_sha256 != decoded["result_sha256"] or result_sha256 != _digest(
            _RESULT_NAMESPACE,
            record,
        ):
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 persisted result root changed"
            )
    elif terminal_status == STATUS_BEHAVIOR_FAIL:
        partial_transcript = decoded.get("partial_transcript")
        if type(partial_transcript) is not dict or decoded.get(
            "partial_transcript_sha256"
        ) != _digest(_FAILURE_NAMESPACE, partial_transcript):
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 persisted behavior-failure transcript root changed"
            )
    elif terminal_status != STATUS_EXECUTION_ABORT:
        raise ContinualRustDiagnosticFactorRecombinationCohortV1Error("C5 terminal status changed")


def _run_and_write_at_paths_for_test_v1(
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
    spec: C5DiagnosticFactorRecombinationSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C5DiagnosticFactorRecombinationResultV1:
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
    except C5BehaviorFailure as error:
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
    spec: C5DiagnosticFactorRecombinationSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C5DiagnosticFactorRecombinationResultV1:
    """Preflight, reserve, and spend the exact production C5 cohort once."""

    identity = _preflight_rustc_v1()
    destination, reservation_path = _production_evidence_paths_v1()

    def diagnostic_compile(
        rustc_identity_sha256: str,
        source: str,
    ) -> adapter.RustcDiagnosticCompileAttemptV1:
        if rustc_identity_sha256 != identity.sha256:
            raise ContinualRustDiagnosticFactorRecombinationCohortV1Error(
                "C5 diagnostic compiler identity changed after reservation"
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
    "C5BehaviorFailure",
    "C5DiagnosticFactorRecombinationResultV1",
    "C5DiagnosticFactorRecombinationSpecV1",
    "C5HypothesisCandidateCatalogueBindingV1",
    "C5PublicConsequenceV1",
    "C5RunReservationV1",
    "C5SelectedExecutionAttemptV1",
    "C5WorkV1",
    "ContinualRustDiagnosticFactorRecombinationCohortV1Error",
    "make_c5_diagnostic_factor_recombination_spec_v1",
    "output_sha256_v1",
    "run_and_write_cohort_v1",
]


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


def _decision_scores(
    decision: factors.FactorizedFeedbackDecisionV1,
) -> dict[str, int]:
    return {score.hypothesis.sha256: score.total_score for score in decision.hypothesis_scores}


_EXPECTED_INTERVENTION_COUNTERS = interventions.RustActionInterventionWorkCountersV1(
    issued_selection_count=1,
    spent_selection_count=1,
    source_release_count=1,
    execution_attempt_count=1,
    successful_execution_count=1,
)


@dataclass(frozen=True, slots=True)
class C5DiagnosticFactorRecombinationResultV1:
    status: str
    spec: C5DiagnosticFactorRecombinationSpecV1
    reservation: C5RunReservationV1
    catalogue_bindings: tuple[C5HypothesisCandidateCatalogueBindingV1, ...]
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
    selected_attempts: tuple[C5SelectedExecutionAttemptV1, ...]
    authorities: tuple[interventions.RustActionInterventionAuthorityV1, ...]
    catalogues: tuple[interventions.RustActionInterventionCatalogueV1, ...]
    authority_work_counters: tuple[interventions.RustActionInterventionWorkCountersV1, ...]
    output_feedback_sha256s: tuple[str, ...]
    work: C5WorkV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "status": self.status,
            "spec_sha256": self.spec.sha256,
            "reservation_sha256": self.reservation.sha256,
            "catalogue_bindings": tuple(item.record() for item in self.catalogue_bindings),
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
            "authority_sha256s": tuple(item.sha256 for item in self.authorities),
            "catalogue_sha256s": tuple(item.sha256 for item in self.catalogues),
            "authority_work_counters": tuple(
                dataclasses.asdict(item) for item in self.authority_work_counters
            ),
            "output_feedback_sha256s": self.output_feedback_sha256s,
            "work": self.work.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        C5WorkV1._authenticate_result(self)


if __name__ == "__main__":
    main()
