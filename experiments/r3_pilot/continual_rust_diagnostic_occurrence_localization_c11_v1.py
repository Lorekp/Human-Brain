"""C11 consequence-conditioned selection of a relative source occurrence.

Four public TRAIN lessons cover the complete concrete operation x occurrence-
relation grid.  Each lesson supplies the correct operation only.  From one
immutable pre-lesson checkpoint, both FARTHER and NEARER target hypotheses are
selected by explicit exploration and executed in a frozen, counterbalanced
order.  One feedback opening follows both executions; exactly one consequence
must pass, and only that consequence advances the main memory.

Two fresh diagnostics are then routed normally.  Both selected edits execute
before one shared evaluation oracle opens, and evaluation cannot update memory.

This is a narrow occurrence-selection experiment.  Tokenization, the simple-let
grammar, primary spans, the FARTHER/NEARER ontology, the exact-two roster policy,
operation interventions, word cues, edit semantics, compiler/executor, and
output oracle remain supplied.  All four hypotheses are trained, so this is not
held-out composition, semantic localization, diagnosis, synthesis, debugging,
transfer, SWE performance, or intelligence.
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

from experiments.r3_pilot import common_ascii_word_unigram_cues_v1 as ascii_cues
from experiments.r3_pilot import common_factorized_feedback_hypothesis_memory_v1 as factors
from experiments.r3_pilot import common_factorized_text_edit_operator_catalogue_v1 as catalogue
from experiments.r3_pilot import common_preceding_let_declaration_occurrences_v1 as occurrences
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import (
    common_rustc_message_line_occurrence_text_edit_authority_v1 as authority,
)
from experiments.r3_pilot import common_rustc_message_line_unordered_cue_adapter_v1 as message_line
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from experiments.r3_pilot import (
    continual_rust_diagnostic_source_context_cue_learning_c10b_v1 as c10b,
)
from humanbrain import rust_selected_action_executor as rust_executor
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

SCHEMA = "humanbrain.continual-rust-diagnostic-occurrence-localization-c11.v1"
RESULT_DOCUMENT_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-occurrence-localization-c11-result-document.v1"
)
RESERVATION_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-occurrence-localization-c11-reservation.v1"
)
STATUS_PASS = "PASS_C11_RUSTC_DIAGNOSTIC_OCCURRENCE_LOCALIZATION_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C11_RUSTC_DIAGNOSTIC_OCCURRENCE_LOCALIZATION_V1"
STATUS_EXECUTION_ABORT = "ABORT_C11_RUSTC_DIAGNOSTIC_OCCURRENCE_LOCALIZATION_V1"
CLAIM_BOUNDARY = (
    "one-fresh-public-six-source-cohort-where-four-paired-outcome-blind-"
    "lessons-condition-selection-of-one-authenticated-relative-preceding-let-"
    "declaration-occurrence-from-parent-message-plus-primary-line-word-cues-"
    "inside-one-supplied-2x2-operation-relation-bank-and-enable-two-fresh-"
    "selected-only-edits-before-one-shared-oracle-with-no-evaluation-update-"
    "all-four-hypotheses-trained-not-held-out-composition-tokenization-simple-"
    "let-grammar-primary-span-farther-nearer-ontology-exact-two-policy-"
    "operation-interventions-word-cues-exploration-edit-semantics-compiler-"
    "executor-and-output-oracle-supplied-not-semantics-diagnosis-synthesis-"
    "debugging-transfer-swe-or-intelligence"
)

FROZEN_MECHANISM_COMMIT = "1ff07c2b5458cc5860cecbf02afc5ab8e62bff3c"
FROZEN_MECHANISM_FILE_SHA256S = (
    (
        "experiments/r3_pilot/common_factorized_feedback_hypothesis_memory_v1.py",
        "999814c34b8db1fa89977d6e2fcf11b1bc8b46e7806ff831fa1655a41eabedad",
    ),
    (
        "experiments/r3_pilot/common_continual_action_outcome_memory_v1.py",
        "7aaf0a71eb968f48a1e683f064a0c2aa6a04c84e778f7cb510c0421ff78605f5",
    ),
    (
        "experiments/r3_pilot/common_rustc_factorized_feedback_adapter_v1.py",
        "e64d29d6173fb9a66692c1f9fc53885181004d56c842d5d44f61315c1958b40b",
    ),
    (
        "experiments/r3_pilot/common_unordered_cue_factor_memory_v1.py",
        "10e7f5f77790308735edca7b73e899614410aa8f8be996ac141069ac30a7c646",
    ),
    (
        "experiments/r3_pilot/common_ascii_word_unigram_cues_v1.py",
        "fa4cd2d6fe6247621eec15927ff89a1997ff81463d3a25da7974632ffa5130d0",
    ),
    (
        "experiments/r3_pilot/common_rustc_lexical_unordered_cue_adapter_v1.py",
        "acea9cc5b103efcea207948f8422d89fce34d4b74a939d990bb11f370c945d6d",
    ),
    (
        "experiments/r3_pilot/common_rustc_message_line_unordered_cue_adapter_v1.py",
        "0018cad4716336146a5c074cf4c1410e8964f1258504f8593ce2dd734f64060f",
    ),
    (
        "experiments/r3_pilot/common_factorized_text_edit_materializer_v1.py",
        "4aac92d2cacba1e49d556d60263c55dc1d9a1ba8719d2abccf8910d40ba0d791",
    ),
    (
        "experiments/r3_pilot/common_factorized_text_edit_operator_catalogue_v1.py",
        "adb8b9caf802f9381e351eed76cbee8ce243a854d563469f1e6686f26d7bfb32",
    ),
    (
        "experiments/r3_pilot/common_rust_action_intervention_v1.py",
        "58ba951aad8eb10fc8b277025c4809748ec815ff430921f9c9c086078f06e43c",
    ),
    (
        "experiments/r3_pilot/common_rustc_message_line_unordered_cue_text_edit_authority_v1.py",
        "b1ac7a91e565e81dd5d0b5ed003e80ee685d1d3effcc29642df805d402d5e6db",
    ),
    (
        "src/humanbrain/rust_selected_action_executor.py",
        "e1bfaa702b87f883d80b4c928f1dd067770b89e310d004b68a48f09f6a37227a",
    ),
    (
        "experiments/r3_pilot/common_continual_token_edit_schema_v1.py",
        "a0380e8ec30c55887a5e2b72c835d701d597a3ad90701d3e6c07cabed67ef443",
    ),
    (
        "experiments/r3_pilot/common_preceding_let_declaration_occurrences_v1.py",
        "767e2a62ed22135c14389bce3ff6952186487fc477970047ea5164617fee9d36",
    ),
    (
        "experiments/r3_pilot/common_rustc_message_line_occurrence_text_edit_authority_v1.py",
        "25571e3ed442292af964436263702453a8e886824d6787be752c13e9898378da",
    ),
)

SOURCE_FILE_NAME = "case.rs"
DIAGNOSTIC_ARGUMENTS = (
    "--edition",
    "2021",
    "--crate-name",
    "c11_occurrence_localization",
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
TRAIN_COUNT = 4
EPISODE_ROLES = (
    "train-declare-nuvor-a",
    "train-replace-selqim-a",
    "train-declare-selqim-b",
    "train-replace-nuvor-b",
    "evaluation-declare-selqim-c",
    "evaluation-replace-nuvor-c",
)
TRAIN_FEEDBACK_ROLES = tuple(f"c11-{role}-paired-feedback" for role in EPISODE_ROLES[:4])
EVALUATION_ORACLE_ROLE = "c11-evaluation-oracle-after-both-executions"
EXPECTED_DIAGNOSTICS = ("E0425", "E0308", "E0425", "E0308", "E0425", "E0308")
CONTEXT_WORDS = ("nuvor", "selqim", "selqim", "nuvor", "selqim", "nuvor")
EXPECTED_PRIMARY_TEXTS = (
    "murad00vek",
    "pelin11qor",
    "virel01mox",
    "zoran10piv",
    "cavin01lur",
    "tarem10qix",
)
EXPECTED_PARENT_MESSAGES = tuple(
    (
        f"cannot find value `{primary}` in this scope"
        if code == "E0425"
        else "mismatched types"
    )
    for code, primary in zip(EXPECTED_DIAGNOSTICS, EXPECTED_PRIMARY_TEXTS, strict=True)
)

TRAIN_DECLARE_FAR_SOURCE = """\
struct Orvak00Nemi {
    murad00vek: i64,
}

fn nuvor(sorin00pax: i64) -> i64 {
    sorin00pax
}

pub fn solve(talor00vix: &[i64]) -> Vec<i64> {
    let fesal00dum = 61_i64;
    let horin00qap = 37_i64;
    talor00vix
        .iter()
        .copied()
        .map(|wurel00kaz| {
            wurel00kaz + nuvor(Orvak00Nemi { murad00vek }.murad00vek)
                + (fesal00dum - fesal00dum)
                + (horin00qap - horin00qap)
        })
        .collect()
}"""

TRAIN_REPLACE_NEAR_SOURCE = """\
fn selqim(cador11mep: i64) -> i64 {
    cador11mep
}

pub fn solve(dovar11kest: &[i64]) -> Vec<i64> {
    let ruvan11pox = 43_i64;
    let belar11tix = 67_i64;
    let pelin11qor = "lum11vexa";
    assert!(!pelin11qor.is_empty());
    dovar11kest
        .iter()
        .copied()
        .map(|juren11sav| {
            juren11sav + selqim(pelin11qor)
                + (ruvan11pox - ruvan11pox)
                + (belar11tix - belar11tix)
        })
        .collect()
}"""

TRAIN_DECLARE_NEAR_SOURCE = """\
struct Kirel01Pavo {
    virel01mox: i64,
}

fn selqim(mavor01tec: i64) -> i64 {
    mavor01tec
}

pub fn solve(nurel01zaf: &[i64]) -> Vec<i64> {
    let jorin01dex = 71_i64;
    let kelar01vup = 47_i64;
    nurel01zaf
        .iter()
        .copied()
        .map(|qorin01bes| {
            qorin01bes + selqim(Kirel01Pavo { virel01mox }.virel01mox)
                + (jorin01dex - jorin01dex)
                + (kelar01vup - kelar01vup)
        })
        .collect()
}"""

TRAIN_REPLACE_FAR_SOURCE = """\
fn nuvor(ravik10mep: i64) -> i64 {
    ravik10mep
}

pub fn solve(seldor10pav: &[i64]) -> Vec<i64> {
    let lumar10qex = 53_i64;
    let nexpi10vor = 73_i64;
    let zoran10piv = "nur10vexa";
    assert!(!zoran10piv.is_empty());
    seldor10pav
        .iter()
        .copied()
        .map(|xalen10qor| {
            xalen10qor + nuvor(zoran10piv)
                + (lumar10qex - lumar10qex)
                + (nexpi10vor - nexpi10vor)
        })
        .collect()
}"""

EVALUATION_DECLARE_NEAR_SOURCE = """\
struct Yavin01Cero {
    cavin01lur: i64,
}

fn selqim(abren01toq: i64) -> i64 {
    abren01toq
}

pub fn solve(cavor01lumi: &[i64]) -> Vec<i64> {
    let qusel01fen = 79_i64;
    let pavri01zod = 59_i64;
    cavor01lumi
        .iter()
        .copied()
        .map(|elmar01qix| {
            elmar01qix + selqim(Yavin01Cero { cavin01lur }.cavin01lur)
                + (qusel01fen - qusel01fen)
                + (pavri01zod - pavri01zod)
        })
        .collect()
}"""

EVALUATION_REPLACE_FAR_SOURCE = """\
fn nuvor(faren10zup: i64) -> i64 {
    faren10zup
}

pub fn solve(gorel10mavi: &[i64]) -> Vec<i64> {
    let seldi10rax = 89_i64;
    let ravun10pel = 83_i64;
    let tarem10qix = "lus10qena";
    assert!(!tarem10qix.is_empty());
    gorel10mavi
        .iter()
        .copied()
        .map(|kiren10voq| {
            kiren10voq + nuvor(tarem10qix)
                + (seldi10rax - seldi10rax)
                + (ravun10pel - ravun10pel)
        })
        .collect()
}"""

ORIGINAL_SOURCES = (
    TRAIN_DECLARE_FAR_SOURCE,
    TRAIN_REPLACE_NEAR_SOURCE,
    TRAIN_DECLARE_NEAR_SOURCE,
    TRAIN_REPLACE_FAR_SOURCE,
    EVALUATION_DECLARE_NEAR_SOURCE,
    EVALUATION_REPLACE_FAR_SOURCE,
)


def _primary_range(source: str, *, code: str, context: str, primary: str) -> tuple[int, int]:
    if code == "E0425":
        needle = f"{{ {primary} }}"
        start = source.index(needle) + 2
    else:
        needle = f"{context}({primary})"
        start = source.index(needle) + len(context) + 1
    return start, start + len(primary)


EXPECTED_PRIMARY_BYTE_RANGES = tuple(
    _primary_range(source, code=code, context=context, primary=primary)
    for source, code, context, primary in zip(
        ORIGINAL_SOURCES,
        EXPECTED_DIAGNOSTICS,
        CONTEXT_WORDS,
        EXPECTED_PRIMARY_TEXTS,
        strict=True,
    )
)
ORIGINAL_SOURCE_SHA256S = tuple(
    sha256(source.encode("ascii")).hexdigest() for source in ORIGINAL_SOURCES
)
ORIGINAL_SOURCE_SIZES = tuple(len(source.encode("ascii")) for source in ORIGINAL_SOURCES)
EXPECTED_MESSAGE_TOKENS = tuple(
    ascii_cues.ascii_word_unigrams_v1(message) for message in EXPECTED_PARENT_MESSAGES
)
EXPECTED_SOURCE_LINE_TOKENS = tuple(
    ascii_cues.ascii_word_unigrams_v1(
        source[
            source.rfind("\n", 0, byte_start) + 1 : (
                len(source) if source.find("\n", byte_end) < 0 else source.find("\n", byte_end)
            )
        ]
    )
    for source, (byte_start, byte_end) in zip(
        ORIGINAL_SOURCES, EXPECTED_PRIMARY_BYTE_RANGES, strict=True
    )
)
EXPECTED_MERGED_CUE_COUNTS = tuple(
    len({*message_tokens, *line_tokens})
    for message_tokens, line_tokens in zip(
        EXPECTED_MESSAGE_TOKENS, EXPECTED_SOURCE_LINE_TOKENS, strict=True
    )
)
MERGED_TOKEN_INVENTORIES = tuple(
    tuple(sorted({*message_tokens, *line_tokens}))
    for message_tokens, line_tokens in zip(
        EXPECTED_MESSAGE_TOKENS, EXPECTED_SOURCE_LINE_TOKENS, strict=True
    )
)
_TRAIN_TOKEN_INVENTORY = {
    token for inventory in MERGED_TOKEN_INVENTORIES[:TRAIN_COUNT] for token in inventory
}
EXPECTED_EVALUATION_SUPPORTED_CUE_INTERSECTIONS = tuple(
    tuple(sorted(_TRAIN_TOKEN_INVENTORY.intersection(inventory)))
    for inventory in MERGED_TOKEN_INVENTORIES[TRAIN_COUNT:]
)
EVALUATION_IDENTIFIER_TOKENS = EXPECTED_PRIMARY_TEXTS[TRAIN_COUNT:]
EXPECTED_EVALUATION_IDENTIFIER_SUPPORT_TICKS = 0

BATCHES = (
    ((-211, -17, 31),),
    ((-193, 13, 41),),
    ((-181, 19, 43),),
    ((-173, 23, 47),),
    ((-167, 29, 53),),
    ((-157, 31, 59),),
)
EXPECTED_OUTPUTS = (
    ((-150, 44, 92),),
    ((-126, 80, 108),),
    ((-134, 66, 90),),
    ((-120, 76, 100),),
    ((-108, 88, 112),),
    ((-68, 120, 148),),
)

OPERATOR_ACTION_SHA256S = (
    catalogue.DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR_ACTION_V1.sha256,
    catalogue.REPLACE_PRIMARY_SPAN_WITH_ANCHOR_ACTION_V1.sha256,
)
RELATIONS = occurrences.PRECEDING_DECLARATION_RELATIONS
RELATION_COMPONENT_SHA256S = tuple(
    occurrences.preceding_declaration_relation_component_sha256_v1(relation)
    for relation in RELATIONS
)
HYPOTHESES_BY_KEY = {
    (operator_index, relation_index): factors.seal_factorized_hypothesis_v1(
        (
            OPERATOR_ACTION_SHA256S[operator_index],
            RELATION_COMPONENT_SHA256S[relation_index],
        )
    )
    for operator_index in range(2)
    for relation_index in range(2)
}
HYPOTHESES = tuple(
    HYPOTHESES_BY_KEY[key] for key in ((0, 0), (0, 1), (1, 0), (1, 1))
)
TRAIN_OPERATOR_INDEXES = (0, 1, 0, 1)
TRAIN_RELATION_ORDER_INDEXES = ((0, 1), (1, 0), (0, 1), (1, 0))
EVALUATION_HYPOTHESIS_SHA256S = (
    HYPOTHESES_BY_KEY[(0, 1)].sha256,
    HYPOTHESES_BY_KEY[(1, 0)].sha256,
)
OPERATOR_BINDINGS = catalogue.canonical_factorized_text_edit_operator_bindings_v1()

# C10b's 99 exclusions plus its six original and six materialized source roots.
EXCLUDED_PRIOR_SOURCE_SHA256S = tuple(
    sorted(
        {
            *c10b.EXCLUDED_PRIOR_SOURCE_SHA256S,
            *c10b.ORIGINAL_SOURCE_SHA256S,
            *c10b.EXPECTED_MATERIALIZED_SOURCE_SHA256S,
        }
    )
)

EXPECTED_WORK_COUNTS = (
    ("diagnostic_compile_count", 6),
    ("retained_message_line_envelope_count", 6),
    ("retained_occurrence_roster_count", 6),
    ("retained_occurrence_certificate_count", 12),
    ("materialization_count", 10),
    ("release_count", 10),
    ("selected_source_compile_count", 10),
    ("selected_source_run_count", 10),
    ("training_feedback_open_count", 4),
    ("evaluation_oracle_open_count", 1),
    ("memory_advance_count", 4),
    ("evaluation_memory_advance_count", 0),
    ("training_pair_decision_count", 8),
    ("evaluation_decision_count", 2),
    ("control_decision_count", 8),
    ("evaluation_execution_before_oracle_count", 2),
    ("alternative_materialization_count", 0),
)

_SHA256_HEX = frozenset("0123456789abcdef")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-spec-v1"
_CONTEXT_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-context-v1"
_OUTPUT_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-output-v1"
_FEEDBACK_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-feedback-v1"
_CONSEQUENCE_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-consequence-v1"
_ATTEMPT_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-attempt-v1"
_WORK_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-work-v1"
_RESERVATION_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-reservation-v1"
_RESULT_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-result-v1"
_FAILURE_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-failure-v1"
_DOCUMENT_NAMESPACE = b"c11-rustc-diagnostic-occurrence-localization-document-v1"


class ContinualRustDiagnosticOccurrenceLocalizationC11Error(ValueError):
    """The frozen C11 cohort or its authenticated transcript changed."""


class C11BehaviorFailure(RuntimeError):
    """A rooted scientific failure after the one-shot reservation was spent."""

    def __init__(self, reason: str, partial_transcript: dict[str, object]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial_transcript = partial_transcript


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


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
    if value is None or type(value) in (str, int, bool, float):
        return value
    if hasattr(value, "record"):
        return _plain(value.record())
    if hasattr(value, "__dict__"):
        return {key: _plain(item) for key, item in vars(value).items()}
    raise TypeError(f"C11 transcript cannot serialize {type(value).__qualname__}")


def _behavior_failure(reason: str, **partial_transcript: object) -> C11BehaviorFailure:
    return C11BehaviorFailure(
        reason,
        {key: _plain(value) for key, value in partial_transcript.items()},
    )


def _current_mechanism_file_sha256s() -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, sha256((_REPO_ROOT / path).read_bytes()).hexdigest())
        for path, _expected in FROZEN_MECHANISM_FILE_SHA256S
    )


def output_sha256_c11_v1(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(_OUTPUT_NAMESPACE, {"outputs": outputs})


@dataclass(frozen=True, slots=True)
class C11DiagnosticOccurrenceLocalizationSpecV1:
    frozen_mechanism_commit: str
    frozen_mechanism_file_sha256s: tuple[tuple[str, str], ...]
    diagnostic_arguments: tuple[str, ...]
    expected_diagnostics: tuple[str, ...]
    expected_primary_byte_ranges: tuple[tuple[int, int], ...]
    expected_parent_messages: tuple[str, ...]
    expected_message_tokens: tuple[tuple[str, ...], ...]
    expected_source_line_tokens: tuple[tuple[str, ...], ...]
    expected_merged_cue_counts: tuple[int, ...]
    expected_evaluation_supported_cue_intersections: tuple[tuple[str, ...], ...]
    evaluation_identifier_tokens: tuple[str, str]
    expected_evaluation_identifier_support_ticks: int
    batches: tuple[tuple[tuple[int, ...], ...], ...]
    expected_output_sha256s: tuple[str, ...]
    original_source_sha256s: tuple[str, ...]
    operator_bindings: tuple[tuple[str, str], ...]
    relation_component_sha256s: tuple[str, str]
    hypothesis_sha256s: tuple[str, ...]
    train_operator_indexes: tuple[int, ...]
    train_relation_order_indexes: tuple[tuple[int, int], ...]
    evaluation_hypothesis_sha256s: tuple[str, str]
    expected_work_counts: tuple[tuple[str, int], ...]
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

    def authenticate(self, *, require_frozen_commit: bool = False) -> None:
        source_bytes = tuple(source.encode("ascii") for source in ORIGINAL_SOURCES)
        if (
            not _valid_commit(self.frozen_mechanism_commit)
            or (require_frozen_commit and self.frozen_mechanism_commit == "0" * 40)
            or self.frozen_mechanism_file_sha256s != FROZEN_MECHANISM_FILE_SHA256S
            or self.frozen_mechanism_file_sha256s != _current_mechanism_file_sha256s()
            or len(self.frozen_mechanism_file_sha256s) != 15
            or self.diagnostic_arguments != DIAGNOSTIC_ARGUMENTS
            or self.expected_diagnostics != EXPECTED_DIAGNOSTICS
            or self.expected_primary_byte_ranges != EXPECTED_PRIMARY_BYTE_RANGES
            or self.expected_parent_messages != EXPECTED_PARENT_MESSAGES
            or self.expected_message_tokens != EXPECTED_MESSAGE_TOKENS
            or self.expected_source_line_tokens != EXPECTED_SOURCE_LINE_TOKENS
            or self.expected_merged_cue_counts != EXPECTED_MERGED_CUE_COUNTS
            or self.expected_evaluation_supported_cue_intersections
            != EXPECTED_EVALUATION_SUPPORTED_CUE_INTERSECTIONS
            or self.evaluation_identifier_tokens != EVALUATION_IDENTIFIER_TOKENS
            or any(token in _TRAIN_TOKEN_INVENTORY for token in self.evaluation_identifier_tokens)
            or any(
                token not in inventory
                for token, inventory in zip(
                    self.evaluation_identifier_tokens,
                    MERGED_TOKEN_INVENTORIES[TRAIN_COUNT:],
                    strict=True,
                )
            )
            or self.expected_evaluation_identifier_support_ticks
            != EXPECTED_EVALUATION_IDENTIFIER_SUPPORT_TICKS
            or self.batches != BATCHES
            or self.expected_output_sha256s
            != tuple(output_sha256_c11_v1(value) for value in EXPECTED_OUTPUTS)
            or self.original_source_sha256s != ORIGINAL_SOURCE_SHA256S
            or self.original_source_sha256s
            != tuple(sha256(value).hexdigest() for value in source_bytes)
            or tuple(map(len, source_bytes)) != ORIGINAL_SOURCE_SIZES
            or any(b"\r" in value or value.endswith(b"\n") for value in source_bytes)
            or self.operator_bindings != OPERATOR_BINDINGS
            or self.relation_component_sha256s != RELATION_COMPONENT_SHA256S
            or self.hypothesis_sha256s != tuple(item.sha256 for item in HYPOTHESES)
            or len(set(self.hypothesis_sha256s)) != 4
            or self.train_operator_indexes != TRAIN_OPERATOR_INDEXES
            or self.train_relation_order_indexes != TRAIN_RELATION_ORDER_INDEXES
            or any(tuple(sorted(order)) != (0, 1) for order in self.train_relation_order_indexes)
            or self.evaluation_hypothesis_sha256s != EVALUATION_HYPOTHESIS_SHA256S
            or self.expected_work_counts != EXPECTED_WORK_COUNTS
            or self.excluded_source_sha256s != EXCLUDED_PRIOR_SOURCE_SHA256S
            or tuple(sorted(self.excluded_source_sha256s)) != self.excluded_source_sha256s
            or len(set(self.excluded_source_sha256s)) != 111
            or set(self.original_source_sha256s).intersection(self.excluded_source_sha256s)
            or any(count > message_line.MAX_TOTAL_CUES for count in self.expected_merged_cue_counts)
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 public fixture specification changed"
            )


def make_c11_diagnostic_occurrence_localization_spec_v1() -> (
    C11DiagnosticOccurrenceLocalizationSpecV1
):
    partial = C11DiagnosticOccurrenceLocalizationSpecV1(
        frozen_mechanism_commit=FROZEN_MECHANISM_COMMIT,
        frozen_mechanism_file_sha256s=FROZEN_MECHANISM_FILE_SHA256S,
        diagnostic_arguments=DIAGNOSTIC_ARGUMENTS,
        expected_diagnostics=EXPECTED_DIAGNOSTICS,
        expected_primary_byte_ranges=EXPECTED_PRIMARY_BYTE_RANGES,
        expected_parent_messages=EXPECTED_PARENT_MESSAGES,
        expected_message_tokens=EXPECTED_MESSAGE_TOKENS,
        expected_source_line_tokens=EXPECTED_SOURCE_LINE_TOKENS,
        expected_merged_cue_counts=EXPECTED_MERGED_CUE_COUNTS,
        expected_evaluation_supported_cue_intersections=(
            EXPECTED_EVALUATION_SUPPORTED_CUE_INTERSECTIONS
        ),
        evaluation_identifier_tokens=EVALUATION_IDENTIFIER_TOKENS,
        expected_evaluation_identifier_support_ticks=(
            EXPECTED_EVALUATION_IDENTIFIER_SUPPORT_TICKS
        ),
        batches=BATCHES,
        expected_output_sha256s=tuple(output_sha256_c11_v1(value) for value in EXPECTED_OUTPUTS),
        original_source_sha256s=ORIGINAL_SOURCE_SHA256S,
        operator_bindings=OPERATOR_BINDINGS,
        relation_component_sha256s=RELATION_COMPONENT_SHA256S,
        hypothesis_sha256s=tuple(item.sha256 for item in HYPOTHESES),
        train_operator_indexes=TRAIN_OPERATOR_INDEXES,
        train_relation_order_indexes=TRAIN_RELATION_ORDER_INDEXES,
        evaluation_hypothesis_sha256s=EVALUATION_HYPOTHESIS_SHA256S,
        expected_work_counts=EXPECTED_WORK_COUNTS,
        excluded_source_sha256s=EXCLUDED_PRIOR_SOURCE_SHA256S,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_SPEC_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


DEFAULT_PUBLIC_SPEC = make_c11_diagnostic_occurrence_localization_spec_v1()


@dataclass(frozen=True, slots=True)
class C11RunReservationV1:
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 run reservation changed"
            )


@dataclass(frozen=True, slots=True)
class C11PublicConsequenceV1:
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 public consequence authentication failed"
            )


def _seal_consequence(
    *,
    execution: RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
    output_feedback_sha256: str,
) -> C11PublicConsequenceV1:
    payload = {
        "source_sha256": execution.selected_source.source_sha256,
        "batch_sha256": execution.preparation.batch_sha256,
        "output_feedback_sha256": output_feedback_sha256,
        "execution_sha256": execution.sha256,
        "observed_outputs": execution.outputs,
        "expected_outputs": expected_outputs,
        "passed_cell": int(execution.outputs == expected_outputs),
    }
    result = C11PublicConsequenceV1(
        **payload,
        sha256=_digest(_CONSEQUENCE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _authority_work_counts(
    prepared: authority.PreparedRustcMessageLineOccurrenceTextEditAuthorityV1,
) -> tuple[int, int, int, int]:
    work = prepared.work_counters
    return (
        work.materialization_count,
        work.release_count,
        work.execution_attempt_count,
        work.successful_execution_count,
    )


@dataclass(frozen=True, slots=True)
class C11SelectedExecutionAttemptV1:
    execution_ordinal: int
    episode_index: int
    phase: str
    relation: str
    decision_sha256: str
    selected_hypothesis_sha256: str
    authority_receipt: authority.RustcMessageLineOccurrenceTextEditAuthorityReceiptV1
    message_line_envelope_sha256: str
    occurrence_roster_sha256: str
    pre_execute_work: tuple[int, int, int, int]
    post_execute_work: tuple[int, int, int, int]
    materialization: authority.UnorderedCueFactorTextEditMaterializationV1
    release: authority.UnorderedCueFactorTextEditReleaseV1
    execution: RustSelectedActionExecutionV2
    consequence: C11PublicConsequenceV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "execution_ordinal": self.execution_ordinal,
            "episode_index": self.episode_index,
            "phase": self.phase,
            "relation": self.relation,
            "decision_sha256": self.decision_sha256,
            "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
            "authority_receipt": self.authority_receipt.record(),
            "message_line_envelope_sha256": self.message_line_envelope_sha256,
            "occurrence_roster_sha256": self.occurrence_roster_sha256,
            "pre_execute_work": self.pre_execute_work,
            "post_execute_work": self.post_execute_work,
            "materialization": self.materialization.record(),
            "release": self.release.record(),
            "execution": _plain(self.execution),
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
        if self.phase == "TRAIN":
            expected_operator_index = TRAIN_OPERATOR_INDEXES[self.episode_index]
        elif self.phase == "EVALUATION":
            expected_operator_index = (0, 1)[self.episode_index - TRAIN_COUNT]
        else:
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 selected attempt phase changed"
            )
        try:
            relation_index = RELATIONS.index(self.relation)
        except ValueError as exc:
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 selected attempt relation changed"
            ) from exc
        expected_hypothesis = HYPOTHESES_BY_KEY[(expected_operator_index, relation_index)]
        selected_source = self.execution.selected_source
        if (
            type(self.execution_ordinal) is not int
            or not 1 <= self.execution_ordinal <= 10
            or type(self.episode_index) is not int
            or not 0 <= self.episode_index < len(ORIGINAL_SOURCES)
            or self.selected_hypothesis_sha256 != expected_hypothesis.sha256
            or self.pre_execute_work != (0, 0, 0, 0)
            or self.post_execute_work != (1, 1, 1, 1)
            or self.authority_receipt.message_line_envelope_sha256
            != self.message_line_envelope_sha256
            or self.authority_receipt.occurrence_roster_sha256
            != self.occurrence_roster_sha256
            or self.authority_receipt.delegate_authority_sha256
            != self.materialization.authority_sha256
            or self.decision_sha256 != self.materialization.decision_sha256
            or self.selected_hypothesis_sha256
            != self.materialization.selected_hypothesis_sha256
            or self.materialization.operator_action_sha256
            != expected_hypothesis.component_sha256s[0]
            or self.materialization.target_component_sha256
            != expected_hypothesis.component_sha256s[1]
            or self.release.authority_sha256
            != self.authority_receipt.delegate_authority_sha256
            or self.release.selection_sha256 != self.decision_sha256
            or self.release.materialization != self.materialization
            or selected_source.authority_sha256
            != self.authority_receipt.delegate_authority_sha256
            or selected_source.selection_sha256 != self.decision_sha256
            or selected_source.release_sha256 != self.release.sha256
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 selected execution attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C11WorkV1:
    diagnostic_compile_count: int
    retained_message_line_envelope_count: int
    retained_occurrence_roster_count: int
    retained_occurrence_certificate_count: int
    materialization_count: int
    release_count: int
    selected_source_compile_count: int
    selected_source_run_count: int
    training_feedback_open_count: int
    evaluation_oracle_open_count: int
    memory_advance_count: int
    evaluation_memory_advance_count: int
    training_pair_decision_count: int
    evaluation_decision_count: int
    control_decision_count: int
    evaluation_execution_before_oracle_count: int
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
        expected = dict(EXPECTED_WORK_COUNTS)
        if self.record(include_sha256=False) != expected or self.sha256 != _digest(
            _WORK_NAMESPACE, expected
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 work accounting changed"
            )


def _seal_work() -> C11WorkV1:
    payload = dict(EXPECTED_WORK_COUNTS)
    result = C11WorkV1(**payload, sha256=_digest(_WORK_NAMESPACE, payload))
    result.authenticate()
    return result


def _require_unique_pair_pass(
    consequences: tuple[C11PublicConsequenceV1, C11PublicConsequenceV1],
    *,
    episode_index: int,
    checkpoint_sha256: str,
) -> int:
    """Return the unique passing index; 0/0 and 1/1 fail before memory advance."""

    for consequence in consequences:
        consequence.authenticate()
    passed = tuple(item.passed_cell for item in consequences)
    return _unique_pair_pass_index(
        passed,
        episode_index=episode_index,
        checkpoint_sha256=checkpoint_sha256,
        consequences=tuple(item.record() for item in consequences),
    )


def _unique_pair_pass_index(
    passed: tuple[int, int],
    *,
    episode_index: int,
    checkpoint_sha256: str,
    consequences: object = (),
) -> int:
    """Pure transaction gate used before any C11 memory advance."""

    if (
        type(passed) is not tuple
        or len(passed) != 2
        or any(type(value) is not int or value not in (0, 1) for value in passed)
        or not _valid_sha256(checkpoint_sha256)
    ):
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 paired pass-cell gate input changed"
        )
    if sum(passed) != 1:
        raise _behavior_failure(
            "C11 paired feedback did not contain exactly one pass and one failure",
            episode_index=episode_index,
            checkpoint_sha256=checkpoint_sha256,
            passed_cells=passed,
            consequences=consequences,
            memory_advance_count=0,
        )
    return passed.index(1)


def _context_sha256(*, spec_sha256: str, role: str, source_sha256: str) -> str:
    return _digest(
        _CONTEXT_NAMESPACE,
        {
            "spec_sha256": spec_sha256,
            "role": role,
            "source_sha256": source_sha256,
        },
    )


def _ablation_observation(
    envelope: message_line.RustcMessageLineUnorderedCueEnvelopeV1,
    cue_sha256s: tuple[str, ...],
) -> unordered.UnorderedCueFactorObservationV1:
    return unordered.seal_unordered_cue_factor_observation_v1(
        selection_context_sha256=envelope.observation.selection_context_sha256,
        cue_sha256s=cue_sha256s,
    )


def _control_decisions(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    envelopes: tuple[message_line.RustcMessageLineUnorderedCueEnvelopeV1, ...],
) -> tuple[unordered.UnorderedCueFactorDecisionV1, ...]:
    return tuple(
        decision
        for envelope in envelopes[TRAIN_COUNT:]
        for decision in (
            memory.route(envelope.observation, outcome_gate=False),
            memory.route(_ablation_observation(envelope, envelope.message_cue_sha256s)),
            memory.route(_ablation_observation(envelope, envelope.source_line_cue_sha256s)),
            memory.route(envelope.observation, axis_gates=(True, False)),
        )
    )


def _deranged_context_decisions(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    envelopes: tuple[message_line.RustcMessageLineUnorderedCueEnvelopeV1, ...],
) -> tuple[unordered.UnorderedCueFactorDecisionV1, ...]:
    evaluation = envelopes[TRAIN_COUNT:]
    context_cues = tuple(
        ascii_cues.ascii_word_unigram_cue_sha256_v1(word)
        for word in ("selqim", "nuvor")
    )
    replacements = tuple(reversed(context_cues))
    observations = tuple(
        _ablation_observation(
            envelope,
            tuple(
                sorted(
                    {
                        *(cue for cue in envelope.observation.cue_sha256s if cue != context_cue),
                        replacement,
                    }
                )
            ),
        )
        for envelope, context_cue, replacement in zip(
            evaluation, context_cues, replacements, strict=True
        )
    )
    return tuple(memory.route(item) for item in observations)


def _authenticate_permutation_and_relation_controls(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    envelopes: tuple[message_line.RustcMessageLineUnorderedCueEnvelopeV1, ...],
    rosters: tuple[occurrences.PrecedingLetDeclarationRosterV1, ...],
) -> None:
    evaluation = envelopes[TRAIN_COUNT:]
    permuted = tuple(
        unordered.seal_unordered_cue_factor_observation_v1(
            selection_context_sha256=envelope.observation.selection_context_sha256,
            cue_sha256s=tuple(reversed(envelope.observation.cue_sha256s)),
        )
        for envelope in evaluation
    )
    if (
        permuted != tuple(item.observation for item in evaluation)
        or tuple(memory.route(item) for item in permuted)
        != tuple(memory.route(item.observation) for item in evaluation)
    ):
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 unordered cue permutation changed routing"
        )

    for source, roster in zip(ORIGINAL_SOURCES, rosters, strict=True):
        renamed = source
        for item in roster.occurrences:
            renamed = renamed.replace(item.identifier, item.identifier[::-1])
        renamed_roster = occurrences.project_preceding_let_declaration_occurrences_v1(
            renamed,
            primary_byte_start=roster.primary_byte_start,
            primary_byte_end=roster.primary_byte_end,
        )
        padding = "// c11 relation-invariant source padding\n"
        padded_roster = occurrences.project_preceding_let_declaration_occurrences_v1(
            padding + source,
            primary_byte_start=roster.primary_byte_start + len(padding),
            primary_byte_end=roster.primary_byte_end + len(padding),
        )
        signature = tuple(
            (item.relation, item.relation_component_sha256) for item in roster.occurrences
        )
        if signature != tuple(
            (item.relation, item.relation_component_sha256)
            for item in renamed_roster.occurrences
        ) or signature != tuple(
            (item.relation, item.relation_component_sha256)
            for item in padded_roster.occurrences
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 identifier relabel or source padding changed relation identity"
            )


def _authenticate_malformed_roster_controls_v1() -> None:
    """Development control: exact 0/1/3 eligible rosters all fail closed."""

    malformed = (
        "fn f() {\n    missing;\n}\n",
        "fn f() {\n    let only = 1_i64;\n    missing;\n}\n",
        (
            "fn f() {\n"
            "    let first = 1_i64;\n"
            "    let second = 2_i64;\n"
            "    let third = 3_i64;\n"
            "    missing;\n"
            "}\n"
        ),
    )
    for source in malformed:
        start = source.index("missing")
        try:
            occurrences.project_preceding_let_declaration_occurrences_v1(
                source,
                primary_byte_start=start,
                primary_byte_end=start + len("missing"),
            )
        except occurrences.PrecedingLetDeclarationOccurrenceError:
            continue
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 malformed occurrence roster did not fail closed"
        )


@dataclass(frozen=True, slots=True)
class C11DiagnosticOccurrenceLocalizationResultV1:
    status: str
    spec: C11DiagnosticOccurrenceLocalizationSpecV1
    reservation: C11RunReservationV1
    diagnostic_attempts: tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, ...]
    message_line_envelopes: tuple[message_line.RustcMessageLineUnorderedCueEnvelopeV1, ...]
    occurrence_rosters: tuple[occurrences.PrecedingLetDeclarationRosterV1, ...]
    observations: tuple[unordered.UnorderedCueFactorObservationV1, ...]
    memories: tuple[unordered.FrozenUnorderedCueFactorMemoryV1, ...]
    training_normal_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    training_pair_decisions: tuple[
        tuple[unordered.UnorderedCueFactorDecisionV1, unordered.UnorderedCueFactorDecisionV1],
        ...,
    ]
    training_attempt_pairs: tuple[
        tuple[C11SelectedExecutionAttemptV1, C11SelectedExecutionAttemptV1], ...
    ]
    training_outcome_pairs: tuple[
        tuple[unordered.UnorderedCueFactorOutcomeV1, unordered.UnorderedCueFactorOutcomeV1],
        ...,
    ]
    passing_outcome_indexes: tuple[int, ...]
    evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    control_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    deranged_context_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    evaluation_attempts: tuple[C11SelectedExecutionAttemptV1, ...]
    training_feedback_sha256s: tuple[str, ...]
    evaluation_oracle_sha256: str
    evaluation_checkpoint_before_sha256: str
    evaluation_checkpoint_after_sha256: str
    work: C11WorkV1
    sha256: str

    @property
    def parent_memory(self) -> unordered.FrozenUnorderedCueFactorMemoryV1:
        return self.memories[0]

    @property
    def final_memory(self) -> unordered.FrozenUnorderedCueFactorMemoryV1:
        return self.memories[-1]

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "status": self.status,
            "spec_sha256": self.spec.sha256,
            "reservation_sha256": self.reservation.sha256,
            "diagnostic_attempts": tuple(item.record() for item in self.diagnostic_attempts),
            "message_line_envelopes": tuple(
                item.record() for item in self.message_line_envelopes
            ),
            "occurrence_rosters": tuple(item.record() for item in self.occurrence_rosters),
            "observations": tuple(item.record() for item in self.observations),
            "memories": tuple(item.record() for item in self.memories),
            "training_normal_decisions": tuple(
                item.record() for item in self.training_normal_decisions
            ),
            "training_pair_decisions": tuple(
                tuple(item.record() for item in pair) for pair in self.training_pair_decisions
            ),
            "training_attempt_pairs": tuple(
                tuple(item.record() for item in pair) for pair in self.training_attempt_pairs
            ),
            "training_outcome_pairs": tuple(
                tuple(item.record() for item in pair) for pair in self.training_outcome_pairs
            ),
            "passing_outcome_indexes": self.passing_outcome_indexes,
            "evaluation_decisions": tuple(item.record() for item in self.evaluation_decisions),
            "control_decisions": tuple(item.record() for item in self.control_decisions),
            "deranged_context_decisions": tuple(
                item.record() for item in self.deranged_context_decisions
            ),
            "evaluation_attempts": tuple(item.record() for item in self.evaluation_attempts),
            "training_feedback_sha256s": self.training_feedback_sha256s,
            "evaluation_oracle_sha256": self.evaluation_oracle_sha256,
            "evaluation_checkpoint_before_sha256": self.evaluation_checkpoint_before_sha256,
            "evaluation_checkpoint_after_sha256": self.evaluation_checkpoint_after_sha256,
            "work": self.work.record(),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        self.spec.authenticate()
        self.reservation.authenticate(destination=Path(self.reservation.destination))
        cardinalities = tuple(
            map(
                len,
                (
                    self.diagnostic_attempts,
                    self.message_line_envelopes,
                    self.occurrence_rosters,
                    self.observations,
                    self.memories,
                    self.training_normal_decisions,
                    self.training_pair_decisions,
                    self.training_attempt_pairs,
                    self.training_outcome_pairs,
                    self.passing_outcome_indexes,
                    self.evaluation_decisions,
                    self.control_decisions,
                    self.deranged_context_decisions,
                    self.evaluation_attempts,
                    self.training_feedback_sha256s,
                ),
            )
        )
        if (
            self.status != STATUS_PASS
            or self.reservation.spec_sha256 != self.spec.sha256
            or cardinalities != (6, 6, 6, 6, 5, 4, 4, 4, 4, 4, 2, 8, 2, 2, 4)
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 result tuple cardinality changed"
            )
        self.work.authenticate()
        for item in self.diagnostic_attempts:
            item.authenticate()
        for item in self.message_line_envelopes:
            item.authenticate()
        for item in self.occurrence_rosters:
            item.authenticate()
        for item in self.observations:
            item.authenticate()
        for item in self.memories:
            item.authenticate()
        for group in (
            self.training_normal_decisions,
            *(pair for pair in self.training_pair_decisions),
            self.evaluation_decisions,
            self.control_decisions,
            self.deranged_context_decisions,
        ):
            for item in group:
                item.authenticate()
        for pair in self.training_attempt_pairs:
            for item in pair:
                item.authenticate()
        for pair in self.training_outcome_pairs:
            for item in pair:
                item.authenticate()
        for item in self.evaluation_attempts:
            item.authenticate()

        for index, (attempt, envelope, roster, observation) in enumerate(
            zip(
                self.diagnostic_attempts,
                self.message_line_envelopes,
                self.occurrence_rosters,
                self.observations,
                strict=True,
            )
        ):
            _authenticate_diagnostic_for_episode(
                index=index,
                attempt=attempt,
                stderr=envelope.stderr,
                rustc_identity_sha256=self.reservation.rustc_identity_sha256,
            )
            expected_context = _context_sha256(
                spec_sha256=self.spec.sha256,
                role=EPISODE_ROLES[index],
                source_sha256=ORIGINAL_SOURCE_SHA256S[index],
            )
            if (
                envelope.attempt != attempt
                or envelope.source != ORIGINAL_SOURCES[index]
                or envelope.observation != observation
                or envelope.selection_context_sha256 != expected_context
                or envelope.message_tokens != EXPECTED_MESSAGE_TOKENS[index]
                or envelope.source_line_tokens != EXPECTED_SOURCE_LINE_TOKENS[index]
                or len(observation.cue_sha256s) != EXPECTED_MERGED_CUE_COUNTS[index]
                or roster.source != ORIGINAL_SOURCES[index]
                or roster.source_sha256 != ORIGINAL_SOURCE_SHA256S[index]
                or (roster.primary_byte_start, roster.primary_byte_end)
                != EXPECTED_PRIMARY_BYTE_RANGES[index]
                or roster.primary_span_bytes.decode("ascii") != EXPECTED_PRIMARY_TEXTS[index]
                or tuple(item.relation for item in roster.occurrences) != RELATIONS
            ):
                raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                    "C11 diagnostic, envelope, roster, or observation cross-custody changed"
                )

        expected_parent = unordered.start_unordered_cue_factor_memory_v1(HYPOTHESES)
        if self.memories[0].record() != expected_parent.record():
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 parent-memory custody changed"
            )
        current = expected_parent
        expected_memories = [current]
        for episode_index in range(TRAIN_COUNT):
            observation = self.observations[episode_index]
            normal = current.route(observation)
            expected_pair = tuple(
                current.route(
                    observation,
                    exploration_hypothesis_sha256=HYPOTHESES_BY_KEY[
                        (TRAIN_OPERATOR_INDEXES[episode_index], relation_index)
                    ].sha256,
                )
                for relation_index in TRAIN_RELATION_ORDER_INDEXES[episode_index]
            )
            attempts = self.training_attempt_pairs[episode_index]
            expected_outputs = EXPECTED_OUTPUTS[episode_index]
            expected_feedback_sha256 = _digest(
                _FEEDBACK_NAMESPACE,
                {
                    "spec_sha256": self.spec.sha256,
                    "role": TRAIN_FEEDBACK_ROLES[episode_index],
                    "checkpoint_sha256": current.checkpoint_sha256,
                    "decision_sha256s": tuple(item.sha256 for item in expected_pair),
                    "execution_sha256s": tuple(item.execution.sha256 for item in attempts),
                    "expected_outputs": expected_outputs,
                },
            )
            envelope = self.message_line_envelopes[episode_index]
            roster = self.occurrence_rosters[episode_index]
            if (
                self.training_feedback_sha256s[episode_index]
                != expected_feedback_sha256
                or any(
                    attempt.authority_receipt.message_line_envelope_sha256
                    != envelope.sha256
                    or attempt.authority_receipt.occurrence_roster_sha256 != roster.sha256
                    or attempt.authority_receipt.source_sha256
                    != ORIGINAL_SOURCE_SHA256S[episode_index]
                    or attempt.authority_receipt.diagnostic_attempt_sha256
                    != self.diagnostic_attempts[episode_index].sha256
                    or attempt.consequence.expected_outputs != expected_outputs
                    or attempt.consequence.output_feedback_sha256
                    != expected_feedback_sha256
                    for attempt in attempts
                )
            ):
                raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                    "C11 TRAIN output, feedback, envelope, or roster cross-custody changed"
                )
            outcomes = tuple(
                unordered.seal_unordered_cue_factor_outcome_v1(
                    decision=decision,
                    proposal_sha256=attempt.materialization.sha256,
                    execution_sha256=attempt.execution.sha256,
                    consequence_sha256=attempt.consequence.sha256,
                    feedback_sha256=self.training_feedback_sha256s[episode_index],
                    passed_cell=attempt.consequence.passed_cell,
                )
                for decision, attempt in zip(expected_pair, attempts, strict=True)
            )
            passing_index = _require_unique_pair_pass(
                tuple(item.consequence for item in attempts),
                episode_index=episode_index,
                checkpoint_sha256=current.checkpoint_sha256,
            )
            if (
                normal != self.training_normal_decisions[episode_index]
                or expected_pair != self.training_pair_decisions[episode_index]
                or any(
                    attempt.decision_sha256 != decision.sha256
                    for attempt, decision in zip(attempts, expected_pair, strict=True)
                )
                or outcomes != self.training_outcome_pairs[episode_index]
                or passing_index != self.passing_outcome_indexes[episode_index]
            ):
                raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                    "C11 paired adaptation transcript changed"
                )
            current = unordered.advance_unordered_cue_factor_memory_v1(
                current, outcomes[passing_index]
            )
            expected_memories.append(current)
        if tuple(item.record() for item in self.memories) != tuple(
            item.record() for item in expected_memories
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 checkpoint lineage changed"
            )

        expected_evaluations = tuple(
            current.route(item) for item in self.observations[TRAIN_COUNT:]
        )
        expected_controls = _control_decisions(current, self.message_line_envelopes)
        expected_deranged = _deranged_context_decisions(current, self.message_line_envelopes)
        executed_decision_roots = {
            item.decision_sha256
            for pair in self.training_attempt_pairs
            for item in pair
        } | {item.decision_sha256 for item in self.evaluation_attempts}
        read_only = (
            *self.training_normal_decisions,
            *self.control_decisions,
            *self.deranged_context_decisions,
        )
        if (
            self.evaluation_decisions != expected_evaluations
            or tuple(item.selected_hypothesis_sha256 for item in expected_evaluations)
            != EVALUATION_HYPOTHESIS_SHA256S
            or self.control_decisions != expected_controls
            or any(item.selected_hypothesis_sha256 is not None for item in expected_controls)
            or self.deranged_context_decisions != expected_deranged
            or tuple(item.selected_hypothesis_sha256 for item in expected_deranged)
            != (HYPOTHESES_BY_KEY[(0, 0)].sha256, HYPOTHESES_BY_KEY[(1, 1)].sha256)
            or any(item.sha256 in executed_decision_roots for item in read_only)
            or tuple(item.decision_sha256 for item in self.evaluation_attempts)
            != tuple(item.sha256 for item in self.evaluation_decisions)
            or any(item.consequence.passed_cell != 1 for item in self.evaluation_attempts)
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 final selection, control, or selected-only execution changed"
            )
        expected_evaluation_oracle_sha256 = _digest(
            _FEEDBACK_NAMESPACE,
            {
                "spec_sha256": self.spec.sha256,
                "role": EVALUATION_ORACLE_ROLE,
                "execution_sha256s": tuple(
                    item.execution.sha256 for item in self.evaluation_attempts
                ),
                "expected_outputs": EXPECTED_OUTPUTS[TRAIN_COUNT:],
            },
        )
        if self.evaluation_oracle_sha256 != expected_evaluation_oracle_sha256:
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 shared evaluation oracle root changed"
            )
        for ordinal, (episode_index, attempt, expected_outputs) in enumerate(
            zip(
                range(TRAIN_COUNT, len(ORIGINAL_SOURCES)),
                self.evaluation_attempts,
                EXPECTED_OUTPUTS[TRAIN_COUNT:],
                strict=True,
            ),
            start=9,
        ):
            expected_feedback_sha256 = _digest(
                _FEEDBACK_NAMESPACE,
                {
                    "evaluation_oracle_sha256": expected_evaluation_oracle_sha256,
                    "execution_ordinal": ordinal,
                },
            )
            envelope = self.message_line_envelopes[episode_index]
            roster = self.occurrence_rosters[episode_index]
            if (
                attempt.execution_ordinal != ordinal
                or attempt.episode_index != episode_index
                or attempt.authority_receipt.message_line_envelope_sha256 != envelope.sha256
                or attempt.authority_receipt.occurrence_roster_sha256 != roster.sha256
                or attempt.authority_receipt.source_sha256 != ORIGINAL_SOURCE_SHA256S[episode_index]
                or attempt.authority_receipt.diagnostic_attempt_sha256
                != self.diagnostic_attempts[episode_index].sha256
                or attempt.consequence.expected_outputs != expected_outputs
                or attempt.consequence.output_feedback_sha256 != expected_feedback_sha256
            ):
                raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                    "C11 evaluation output, feedback, envelope, or roster cross-custody changed"
                )
        _authenticate_permutation_and_relation_controls(
            current, self.message_line_envelopes, self.occurrence_rosters
        )
        merged_token_inventories = tuple(
            tuple(sorted({*item.message_tokens, *item.source_line_tokens}))
            for item in self.message_line_envelopes
        )
        training_tokens = {
            token
            for inventory in merged_token_inventories[:TRAIN_COUNT]
            for token in inventory
        }
        actual_intersections = tuple(
            tuple(sorted(training_tokens.intersection(inventory)))
            for inventory in merged_token_inventories[TRAIN_COUNT:]
        )
        identifier_cues = tuple(
            ascii_cues.ascii_word_unigram_cue_sha256_v1(token)
            for token in EVALUATION_IDENTIFIER_TOKENS
        )
        identifier_support_ticks = tuple(
            sum(
                row.excitatory_ticks + row.inhibitory_ticks
                for row in current.binding_evidence
                if row.cue_sha256 == cue
            )
            for cue in identifier_cues
        )
        if (
            actual_intersections != EXPECTED_EVALUATION_SUPPORTED_CUE_INTERSECTIONS
            or identifier_support_ticks
            != (EXPECTED_EVALUATION_IDENTIFIER_SUPPORT_TICKS,) * 2
            or any(cue in current.cue_sha256s for cue in identifier_cues)
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 TRAIN/evaluation cue overlap or fresh-identifier u=0 evidence changed"
            )
        materialized_roots = {
            item.materialization.materialized_source_sha256
            for pair in self.training_attempt_pairs
            for item in pair
        } | {
            item.materialization.materialized_source_sha256
            for item in self.evaluation_attempts
        }
        derived_work = {
            "diagnostic_compile_count": len(self.diagnostic_attempts),
            "retained_message_line_envelope_count": len(self.message_line_envelopes),
            "retained_occurrence_roster_count": len(self.occurrence_rosters),
            "retained_occurrence_certificate_count": sum(
                len(item.occurrences) for item in self.occurrence_rosters
            ),
            "materialization_count": sum(len(pair) for pair in self.training_attempt_pairs)
            + len(self.evaluation_attempts),
            "release_count": sum(len(pair) for pair in self.training_attempt_pairs)
            + len(self.evaluation_attempts),
            "selected_source_compile_count": sum(
                item.execution.compile_receipt.compile_invocations
                for pair in self.training_attempt_pairs
                for item in pair
            )
            + sum(
                item.execution.compile_receipt.compile_invocations
                for item in self.evaluation_attempts
            ),
            "selected_source_run_count": sum(
                item.execution.run_receipt.run_invocations
                for pair in self.training_attempt_pairs
                for item in pair
            )
            + sum(
                item.execution.run_receipt.run_invocations
                for item in self.evaluation_attempts
            ),
            "training_feedback_open_count": len(self.training_feedback_sha256s),
            "evaluation_oracle_open_count": int(bool(self.evaluation_oracle_sha256)),
            "memory_advance_count": len(self.memories) - 1,
            "evaluation_memory_advance_count": 0,
            "training_pair_decision_count": sum(
                len(pair) for pair in self.training_pair_decisions
            ),
            "evaluation_decision_count": len(self.evaluation_decisions),
            "control_decision_count": len(self.control_decisions),
            "evaluation_execution_before_oracle_count": len(self.evaluation_attempts),
            "alternative_materialization_count": 0,
        }
        if (
            derived_work != dict(EXPECTED_WORK_COUNTS)
            or self.work.record(include_sha256=False) != derived_work
            or len(materialized_roots) != 10
            or materialized_roots.intersection(
                {*self.spec.excluded_source_sha256s, *self.spec.original_source_sha256s}
            )
            or self.evaluation_checkpoint_before_sha256 != current.checkpoint_sha256
            or self.evaluation_checkpoint_after_sha256
            != self.evaluation_checkpoint_before_sha256
            or self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 source freshness, evaluation immutability, or result root changed"
            )


DiagnosticCompileCallback = Callable[
    [str, str],
    tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes],
]
ExecutionCallback = Callable[
    [
        authority.PreparedRustcMessageLineOccurrenceTextEditAuthorityV1,
        unordered.UnorderedCueFactorDecisionV1,
        tuple[tuple[int, ...], ...],
    ],
    RustSelectedActionExecutionV2,
]
OutputFeedbackCallback = Callable[[str], object]


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
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 public output feedback shape changed"
        )
    return outputs


def _validate_evaluation_oracle(
    value: object,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if type(value) is not tuple or len(value) != 2:
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 evaluation oracle must open both results together"
        )
    return tuple(
        _validate_outputs(outputs, batch=BATCHES[index])
        for index, outputs in enumerate(value, start=TRAIN_COUNT)
    )


def _coded_parent_error_records(stderr: bytes) -> tuple[dict[str, object], ...]:
    try:
        rows = tuple(json.loads(line) for line in stderr.decode("utf-8").splitlines() if line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 rustc stderr is not exact JSON-lines UTF-8"
        ) from exc
    return tuple(
        row
        for row in rows
        if type(row) is dict
        and row.get("$message_type") == "diagnostic"
        and row.get("level") == "error"
        and type(row.get("code")) is dict
        and type(row["code"].get("code")) is str
    )


def _authenticate_diagnostic_for_episode(
    *,
    index: int,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    stderr: bytes,
    rustc_identity_sha256: str,
) -> None:
    if type(attempt) is not rustc_feedback.RustcDiagnosticCompileAttemptV1:
        raise TypeError("C11 diagnostic compiler must return one exact attempt")
    if type(stderr) is not bytes:
        raise TypeError("C11 diagnostic compiler must retain exact stderr bytes")
    attempt.authenticate()
    source_bytes = ORIGINAL_SOURCES[index].encode("ascii")
    expected_code = EXPECTED_DIAGNOSTICS[index]
    spans = tuple(
        span for diagnostic in attempt.error_diagnostics for span in diagnostic.primary_spans
    )
    parent_records = _coded_parent_error_records(stderr)
    if (
        attempt.status != rustc_feedback.COMPILE_FAILED
        or attempt.rustc_identity_sha256 != rustc_identity_sha256
        or attempt.source_file_name != SOURCE_FILE_NAME
        or attempt.source_sha256 != ORIGINAL_SOURCE_SHA256S[index]
        or attempt.source_size_bytes != ORIGINAL_SOURCE_SIZES[index]
        or attempt.arguments != DIAGNOSTIC_ARGUMENTS
        or sha256(stderr).hexdigest() != attempt.stderr_sha256
        or tuple(diagnostic.code for diagnostic in attempt.error_diagnostics) != (expected_code,)
        or len(spans) != 1
        or (spans[0].byte_start, spans[0].byte_end) != EXPECTED_PRIMARY_BYTE_RANGES[index]
        or source_bytes[spans[0].byte_start : spans[0].byte_end].decode("ascii")
        != EXPECTED_PRIMARY_TEXTS[index]
        or len(parent_records) != 1
        or parent_records[0]["code"]["code"] != expected_code
        or parent_records[0].get("message") != EXPECTED_PARENT_MESSAGES[index]
    ):
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 diagnostic attempt or parent-message fixture changed"
        )


def _compile_diagnostic_once(
    identity: rust_executor.RustcIdentityV1,
    source: str,
) -> tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes]:
    identity.authenticate_structure()
    source_bytes = source.encode("ascii")
    with tempfile.TemporaryDirectory(prefix="humanbrain-c11-diagnostic-") as raw_directory:
        directory = Path(raw_directory)
        source_path = directory / SOURCE_FILE_NAME
        source_path.write_bytes(source_bytes)
        completed = subprocess.run(
            [str(identity.path), *DIAGNOSTIC_ARGUMENTS],
            cwd=directory,
            capture_output=True,
            check=False,
            timeout=rust_executor.COMPILE_TIMEOUT_SECONDS,
        )
    attempt = rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
        rustc_identity_sha256=identity.sha256,
        source_file_name=SOURCE_FILE_NAME,
        source=source_bytes,
        arguments=DIAGNOSTIC_ARGUMENTS,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        compile_invocations=1,
    )
    return attempt, completed.stderr


def _execute_selected(
    prepared: authority.PreparedRustcMessageLineOccurrenceTextEditAuthorityV1,
    decision: unordered.UnorderedCueFactorDecisionV1,
    batch: tuple[tuple[int, ...], ...],
) -> RustSelectedActionExecutionV2:
    return prepared.execute_selected(decision, batch)


def _prepare_episode(
    *,
    index: int,
    spec: C11DiagnosticOccurrenceLocalizationSpecV1,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    stderr: bytes,
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    allow_exploration: bool,
) -> authority.PreparedRustcMessageLineOccurrenceTextEditAuthorityV1:
    return authority.prepare_rustc_message_line_occurrence_text_edit_authority_v1(
        diagnostic_attempt=attempt,
        stderr=stderr,
        source=ORIGINAL_SOURCES[index],
        selection_context_sha256=_context_sha256(
            spec_sha256=spec.sha256,
            role=EPISODE_ROLES[index],
            source_sha256=ORIGINAL_SOURCE_SHA256S[index],
        ),
        expected_checkpoint_sha256=memory.checkpoint_sha256,
        expected_factorized_checkpoint_sha256=memory.factorized_checkpoint_sha256,
        allow_exploration=allow_exploration,
    )


@dataclass(frozen=True, slots=True)
class _SelectedGrantV1:
    prepared: authority.PreparedRustcMessageLineOccurrenceTextEditAuthorityV1
    decision: unordered.UnorderedCueFactorDecisionV1
    pre_execute_work: tuple[int, int, int, int]
    post_execute_work: tuple[int, int, int, int]
    materialization: authority.UnorderedCueFactorTextEditMaterializationV1
    release: authority.UnorderedCueFactorTextEditReleaseV1
    execution: RustSelectedActionExecutionV2


def _execute_selected_grant(
    *,
    prepared: authority.PreparedRustcMessageLineOccurrenceTextEditAuthorityV1,
    decision: unordered.UnorderedCueFactorDecisionV1,
    batch: tuple[tuple[int, ...], ...],
    execute: ExecutionCallback,
) -> _SelectedGrantV1:
    pre_work = _authority_work_counts(prepared)
    execution = execute(prepared, decision, batch)
    post_work = _authority_work_counts(prepared)
    materialization = prepared.materialization
    release = prepared.release
    if (
        pre_work != (0, 0, 0, 0)
        or post_work != (1, 1, 1, 1)
        or materialization is None
        or release is None
    ):
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 selected authority work changed"
        )
    return _SelectedGrantV1(
        prepared=prepared,
        decision=decision,
        pre_execute_work=pre_work,
        post_execute_work=post_work,
        materialization=materialization,
        release=release,
        execution=execution,
    )


def _seal_selected_attempt(
    *,
    execution_ordinal: int,
    episode_index: int,
    phase: str,
    relation: str,
    grant: _SelectedGrantV1,
    expected_outputs: tuple[tuple[int, ...], ...],
    output_feedback_sha256: str,
) -> C11SelectedExecutionAttemptV1:
    consequence = _seal_consequence(
        execution=grant.execution,
        expected_outputs=expected_outputs,
        output_feedback_sha256=output_feedback_sha256,
    )
    payload = {
        "execution_ordinal": execution_ordinal,
        "episode_index": episode_index,
        "phase": phase,
        "relation": relation,
        "decision_sha256": grant.decision.sha256,
        "selected_hypothesis_sha256": grant.decision.selected_hypothesis_sha256,
        "authority_receipt": grant.prepared.receipt,
        "message_line_envelope_sha256": grant.prepared.message_line_envelope.sha256,
        "occurrence_roster_sha256": grant.prepared.occurrence_roster.sha256,
        "pre_execute_work": grant.pre_execute_work,
        "post_execute_work": grant.post_execute_work,
        "materialization": grant.materialization,
        "release": grant.release,
        "execution": grant.execution,
        "consequence": consequence,
    }
    partial = C11SelectedExecutionAttemptV1(**payload, sha256="")
    result = replace(
        partial,
        sha256=_digest(_ATTEMPT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _run_reserved_cohort_v1(
    *,
    spec: C11DiagnosticOccurrenceLocalizationSpecV1,
    reservation: C11RunReservationV1,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> C11DiagnosticOccurrenceLocalizationResultV1:
    spec.authenticate()
    reservation.authenticate(destination=Path(reservation.destination))
    if reservation.spec_sha256 != spec.sha256:
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 reservation/spec custody changed"
        )

    parent = unordered.start_unordered_cue_factor_memory_v1(HYPOTHESES)
    compiled = tuple(
        diagnostic_compile(reservation.rustc_identity_sha256, source)
        for source in ORIGINAL_SOURCES
    )
    if any(type(item) is not tuple or len(item) != 2 for item in compiled):
        raise TypeError("C11 diagnostic compiler must return exact (attempt, stderr) tuples")
    diagnostic_attempts = tuple(item[0] for item in compiled)
    diagnostic_stderr = tuple(item[1] for item in compiled)
    for index, (attempt, stderr) in enumerate(
        zip(diagnostic_attempts, diagnostic_stderr, strict=True)
    ):
        _authenticate_diagnostic_for_episode(
            index=index,
            attempt=attempt,
            stderr=stderr,
            rustc_identity_sha256=reservation.rustc_identity_sha256,
        )

    parent_authorities = tuple(
        _prepare_episode(
            index=index,
            spec=spec,
            attempt=attempt,
            stderr=stderr,
            memory=parent,
            allow_exploration=index < TRAIN_COUNT,
        )
        for index, (attempt, stderr) in enumerate(
            zip(diagnostic_attempts, diagnostic_stderr, strict=True)
        )
    )
    message_line_envelopes = tuple(item.message_line_envelope for item in parent_authorities)
    occurrence_rosters = tuple(item.occurrence_roster for item in parent_authorities)
    observations = tuple(item.projected_observation for item in parent_authorities)
    if (
        tuple(item.message_tokens for item in message_line_envelopes) != EXPECTED_MESSAGE_TOKENS
        or tuple(item.source_line_tokens for item in message_line_envelopes)
        != EXPECTED_SOURCE_LINE_TOKENS
        or tuple(len(item.observation.cue_sha256s) for item in message_line_envelopes)
        != EXPECTED_MERGED_CUE_COUNTS
        or tuple(item.source_sha256 for item in occurrence_rosters) != ORIGINAL_SOURCE_SHA256S
        or any(len(item.occurrences) != 2 for item in occurrence_rosters)
        or any(
            tuple(occurrence.relation for occurrence in item.occurrences) != RELATIONS
            for item in occurrence_rosters
        )
        or len({item.sha256 for item in occurrence_rosters}) != 6
    ):
        raise _behavior_failure(
            "C11 message-line or complete occurrence-roster projection changed",
            message_line_envelopes=tuple(item.record() for item in message_line_envelopes),
            occurrence_rosters=tuple(item.record() for item in occurrence_rosters),
        )

    parent_evaluation_decisions = tuple(parent.route(item) for item in observations[TRAIN_COUNT:])
    if any(item.selected_hypothesis_sha256 is not None for item in parent_evaluation_decisions):
        raise _behavior_failure(
            "C11 parent checkpoint did not STOP on fresh evaluations",
            decisions=tuple(item.record() for item in parent_evaluation_decisions),
        )

    memories = [parent]
    training_normal_decisions: list[unordered.UnorderedCueFactorDecisionV1] = []
    training_pair_decisions: list[
        tuple[unordered.UnorderedCueFactorDecisionV1, unordered.UnorderedCueFactorDecisionV1]
    ] = []
    training_attempt_pairs: list[
        tuple[C11SelectedExecutionAttemptV1, C11SelectedExecutionAttemptV1]
    ] = []
    training_outcome_pairs: list[
        tuple[unordered.UnorderedCueFactorOutcomeV1, unordered.UnorderedCueFactorOutcomeV1]
    ] = []
    passing_outcome_indexes: list[int] = []
    training_feedback_sha256s: list[str] = []
    execution_ordinal = 0

    for episode_index in range(TRAIN_COUNT):
        current = memories[-1]
        observation = observations[episode_index]
        normal = current.route(observation)
        relation_order = TRAIN_RELATION_ORDER_INDEXES[episode_index]
        prepared_pair = tuple(
            _prepare_episode(
                index=episode_index,
                spec=spec,
                attempt=diagnostic_attempts[episode_index],
                stderr=diagnostic_stderr[episode_index],
                memory=current,
                allow_exploration=True,
            )
            for _relation_index in relation_order
        )
        if any(
            prepared.projected_observation != observation
            or prepared.occurrence_roster != occurrence_rosters[episode_index]
            or _authority_work_counts(prepared) != (0, 0, 0, 0)
            for prepared in prepared_pair
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 paired authorities changed projection or began with work"
            )
        decisions = tuple(
            current.route(
                observation,
                exploration_hypothesis_sha256=HYPOTHESES_BY_KEY[
                    (TRAIN_OPERATOR_INDEXES[episode_index], relation_index)
                ].sha256,
            )
            for relation_index in relation_order
        )
        if any(
            decision.selected_hypothesis_sha256
            != HYPOTHESES_BY_KEY[
                (TRAIN_OPERATOR_INDEXES[episode_index], relation_index)
            ].sha256
            for decision, relation_index in zip(decisions, relation_order, strict=True)
        ):
            raise _behavior_failure(
                "C11 explicit operation-fixed occurrence exploration changed",
                episode_index=episode_index,
                decisions=tuple(item.record() for item in decisions),
            )

        grants = tuple(
            _execute_selected_grant(
                prepared=prepared,
                decision=decision,
                batch=BATCHES[episode_index],
                execute=execute,
            )
            for prepared, decision in zip(prepared_pair, decisions, strict=True)
        )
        expected_outputs = _validate_outputs(
            output_feedback(TRAIN_FEEDBACK_ROLES[episode_index]),
            batch=BATCHES[episode_index],
        )
        feedback_sha256 = _digest(
            _FEEDBACK_NAMESPACE,
            {
                "spec_sha256": spec.sha256,
                "role": TRAIN_FEEDBACK_ROLES[episode_index],
                "checkpoint_sha256": current.checkpoint_sha256,
                "decision_sha256s": tuple(item.sha256 for item in decisions),
                "execution_sha256s": tuple(item.execution.sha256 for item in grants),
                "expected_outputs": expected_outputs,
            },
        )
        attempts: list[C11SelectedExecutionAttemptV1] = []
        for relation_index, grant in zip(relation_order, grants, strict=True):
            execution_ordinal += 1
            attempts.append(
                _seal_selected_attempt(
                    execution_ordinal=execution_ordinal,
                    episode_index=episode_index,
                    phase="TRAIN",
                    relation=RELATIONS[relation_index],
                    grant=grant,
                    expected_outputs=expected_outputs,
                    output_feedback_sha256=feedback_sha256,
                )
            )
        attempt_pair = (attempts[0], attempts[1])
        passing_index = _require_unique_pair_pass(
            tuple(item.consequence for item in attempt_pair),
            episode_index=episode_index,
            checkpoint_sha256=current.checkpoint_sha256,
        )
        outcomes = tuple(
            unordered.seal_unordered_cue_factor_outcome_v1(
                decision=decision,
                proposal_sha256=attempt.materialization.sha256,
                execution_sha256=attempt.execution.sha256,
                consequence_sha256=attempt.consequence.sha256,
                feedback_sha256=feedback_sha256,
                passed_cell=attempt.consequence.passed_cell,
            )
            for decision, attempt in zip(decisions, attempt_pair, strict=True)
        )
        passing_outcome = outcomes[passing_index]
        if (
            unordered.advance_unordered_cue_factor_memory_v1(
                current, passing_outcome, plasticity_gate=False
            )
            is not current
            or unordered.advance_unordered_cue_factor_memory_v1(
                current, passing_outcome, consequence_gate=False
            )
            is not current
        ):
            raise _behavior_failure(
                "C11 adaptation learning-gate control changed",
                episode_index=episode_index,
                passing_outcome=passing_outcome.record(),
            )
        training_normal_decisions.append(normal)
        training_pair_decisions.append((decisions[0], decisions[1]))
        training_attempt_pairs.append(attempt_pair)
        training_outcome_pairs.append((outcomes[0], outcomes[1]))
        passing_outcome_indexes.append(passing_index)
        training_feedback_sha256s.append(feedback_sha256)
        memories.append(
            unordered.advance_unordered_cue_factor_memory_v1(current, passing_outcome)
        )

    final = memories[-1]
    evaluation_checkpoint_before_sha256 = final.checkpoint_sha256
    evaluation_decisions = tuple(final.route(item) for item in observations[TRAIN_COUNT:])
    control_decisions = _control_decisions(final, message_line_envelopes)
    deranged_context_decisions = _deranged_context_decisions(final, message_line_envelopes)
    if (
        tuple(item.selected_hypothesis_sha256 for item in evaluation_decisions)
        != EVALUATION_HYPOTHESIS_SHA256S
        or any(item.selected_hypothesis_sha256 is not None for item in control_decisions)
        or tuple(item.selected_hypothesis_sha256 for item in deranged_context_decisions)
        != (HYPOTHESES_BY_KEY[(0, 0)].sha256, HYPOTHESES_BY_KEY[(1, 1)].sha256)
    ):
        raise _behavior_failure(
            "C11 final routing or causal-control behavior changed",
            evaluation_decisions=tuple(item.record() for item in evaluation_decisions),
            control_decisions=tuple(item.record() for item in control_decisions),
            deranged_context_decisions=tuple(
                item.record() for item in deranged_context_decisions
            ),
        )
    _authenticate_permutation_and_relation_controls(
        final, message_line_envelopes, occurrence_rosters
    )

    evaluation_grants: list[tuple[int, _SelectedGrantV1, str]] = []
    for episode_index, decision in enumerate(
        evaluation_decisions, start=TRAIN_COUNT
    ):
        prepared = _prepare_episode(
            index=episode_index,
            spec=spec,
            attempt=diagnostic_attempts[episode_index],
            stderr=diagnostic_stderr[episode_index],
            memory=final,
            allow_exploration=False,
        )
        if (
            prepared.projected_observation != observations[episode_index]
            or prepared.occurrence_roster != occurrence_rosters[episode_index]
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 evaluation authority projection changed"
            )
        selected = next(
            item
            for item in HYPOTHESES
            if item.sha256 == decision.selected_hypothesis_sha256
        )
        relation_index = RELATION_COMPONENT_SHA256S.index(selected.component_sha256s[1])
        evaluation_grants.append(
            (
                episode_index,
                _execute_selected_grant(
                    prepared=prepared,
                    decision=decision,
                    batch=BATCHES[episode_index],
                    execute=execute,
                ),
                RELATIONS[relation_index],
            )
        )

    evaluation_expected_outputs = _validate_evaluation_oracle(
        output_feedback(EVALUATION_ORACLE_ROLE)
    )
    evaluation_oracle_sha256 = _digest(
        _FEEDBACK_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "role": EVALUATION_ORACLE_ROLE,
            "execution_sha256s": tuple(item[1].execution.sha256 for item in evaluation_grants),
            "expected_outputs": evaluation_expected_outputs,
        },
    )
    evaluation_attempts: list[C11SelectedExecutionAttemptV1] = []
    for (episode_index, grant, relation), expected_outputs in zip(
        evaluation_grants, evaluation_expected_outputs, strict=True
    ):
        execution_ordinal += 1
        feedback_sha256 = _digest(
            _FEEDBACK_NAMESPACE,
            {
                "evaluation_oracle_sha256": evaluation_oracle_sha256,
                "execution_ordinal": execution_ordinal,
            },
        )
        evaluation_attempts.append(
            _seal_selected_attempt(
                execution_ordinal=execution_ordinal,
                episode_index=episode_index,
                phase="EVALUATION",
                relation=relation,
                grant=grant,
                expected_outputs=expected_outputs,
                output_feedback_sha256=feedback_sha256,
            )
        )
    if any(item.consequence.passed_cell != 1 for item in evaluation_attempts):
        raise _behavior_failure(
            "C11 normally selected evaluation occurrence failed public output",
            attempts=tuple(item.record() for item in evaluation_attempts),
        )
    evaluation_checkpoint_after_sha256 = final.checkpoint_sha256

    payload = {
        "status": STATUS_PASS,
        "spec": spec,
        "reservation": reservation,
        "diagnostic_attempts": diagnostic_attempts,
        "message_line_envelopes": message_line_envelopes,
        "occurrence_rosters": occurrence_rosters,
        "observations": observations,
        "memories": tuple(memories),
        "training_normal_decisions": tuple(training_normal_decisions),
        "training_pair_decisions": tuple(training_pair_decisions),
        "training_attempt_pairs": tuple(training_attempt_pairs),
        "training_outcome_pairs": tuple(training_outcome_pairs),
        "passing_outcome_indexes": tuple(passing_outcome_indexes),
        "evaluation_decisions": evaluation_decisions,
        "control_decisions": control_decisions,
        "deranged_context_decisions": deranged_context_decisions,
        "evaluation_attempts": tuple(evaluation_attempts),
        "training_feedback_sha256s": tuple(training_feedback_sha256s),
        "evaluation_oracle_sha256": evaluation_oracle_sha256,
        "evaluation_checkpoint_before_sha256": evaluation_checkpoint_before_sha256,
        "evaluation_checkpoint_after_sha256": evaluation_checkpoint_after_sha256,
        "work": _seal_work(),
    }
    partial = C11DiagnosticOccurrenceLocalizationResultV1(**payload, sha256="")
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _reserve_destination(
    spec: C11DiagnosticOccurrenceLocalizationSpecV1,
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
) -> C11RunReservationV1:
    spec.authenticate(require_frozen_commit=True)
    if not _valid_commit(preregistration_commit):
        raise ValueError("C11 preregistration commit must be 40 lowercase hex characters")
    if not _valid_sha256(rustc_identity_sha256):
        raise ValueError("C11 rustc identity root is invalid")
    destination = Path(destination).resolve()
    reservation_path = Path(reservation_path).resolve()
    expected_reservation = destination.with_name(f"{destination.name}.reservation.json")
    if reservation_path != expected_reservation:
        raise ValueError("C11 reservation must be the result's exact sibling")
    if os.path.lexists(destination) or os.path.lexists(reservation_path):
        raise FileExistsError(destination if os.path.lexists(destination) else reservation_path)
    partial = C11RunReservationV1(
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
            raise OSError("C11 reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if reservation_path.read_bytes() != payload:
        raise OSError("C11 reservation readback changed")
    return reservation


def _result_document(result: C11DiagnosticOccurrenceLocalizationResultV1) -> dict[str, object]:
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
    }
    return {**body, "document_sha256": _digest(_DOCUMENT_NAMESPACE, body)}


def _abort_document(
    *,
    spec: C11DiagnosticOccurrenceLocalizationSpecV1,
    reservation: C11RunReservationV1,
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
    spec: C11DiagnosticOccurrenceLocalizationSpecV1,
    reservation: C11RunReservationV1,
    error: C11BehaviorFailure,
) -> dict[str, object]:
    partial_transcript = _plain(error.partial_transcript)
    if type(partial_transcript) is not dict:
        raise TypeError("C11 behavior failure transcript must be one exact object")
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
            raise OSError("C11 terminal document write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = destination.read_bytes()
    if retained != payload or _canonical_json(json.loads(retained.decode("ascii"))) + b"\n" != (
        payload
    ):
        raise OSError("C11 terminal document readback changed")
    decoded = json.loads(retained.decode("ascii"))
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if document_sha256 != _digest(_DOCUMENT_NAMESPACE, body):
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 terminal document root changed"
        )
    if decoded["terminal_status"] == STATUS_PASS:
        record = dict(decoded["authenticated_result_record"])
        result_sha256 = record.pop("sha256", None)
        if result_sha256 != decoded["result_sha256"] or result_sha256 != _digest(
            _RESULT_NAMESPACE, record
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 persisted result root changed"
            )
    elif decoded["terminal_status"] == STATUS_BEHAVIOR_FAIL:
        partial = decoded.get("partial_transcript")
        if type(partial) is not dict or decoded.get("partial_transcript_sha256") != _digest(
            _FAILURE_NAMESPACE, partial
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 persisted behavior-failure transcript root changed"
            )
    elif decoded["terminal_status"] != STATUS_EXECUTION_ABORT:
        raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
            "C11 terminal status changed"
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
    spec: C11DiagnosticOccurrenceLocalizationSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C11DiagnosticOccurrenceLocalizationResultV1:
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
    except C11BehaviorFailure as error:
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


def run_and_write_c11_v1(
    *,
    preregistration_commit: str,
    spec: C11DiagnosticOccurrenceLocalizationSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C11DiagnosticOccurrenceLocalizationResultV1:
    """Preflight, reserve, and spend the exact production C11 cohort once."""

    spec.authenticate(require_frozen_commit=True)
    identity = _preflight_rustc_v1()
    destination, reservation_path = _production_evidence_paths_v1()

    def diagnostic_compile(
        rustc_identity_sha256: str,
        source: str,
    ) -> tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes]:
        if rustc_identity_sha256 != identity.sha256:
            raise ContinualRustDiagnosticOccurrenceLocalizationC11Error(
                "C11 diagnostic compiler identity changed after reservation"
            )
        return _compile_diagnostic_once(identity, source)

    expected_by_role = dict(zip(TRAIN_FEEDBACK_ROLES, EXPECTED_OUTPUTS[:TRAIN_COUNT], strict=True))

    def output_feedback(role: str) -> object:
        if role == EVALUATION_ORACLE_ROLE:
            return EXPECTED_OUTPUTS[TRAIN_COUNT:]
        return expected_by_role[role]

    return _run_and_write_at_paths_for_test_v1(
        preregistration_commit=preregistration_commit,
        rustc_identity_sha256=identity.sha256,
        destination=destination,
        reservation_path=reservation_path,
        diagnostic_compile=diagnostic_compile,
        execute=_execute_selected,
        output_feedback=output_feedback,
        spec=spec,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-commit", required=True)
    arguments = parser.parse_args()
    result = run_and_write_c11_v1(preregistration_commit=arguments.preregistration_commit)
    print(_canonical_json(result.record()).decode("ascii"))


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_PUBLIC_SPEC",
    "STATUS_BEHAVIOR_FAIL",
    "STATUS_EXECUTION_ABORT",
    "STATUS_PASS",
    "C11BehaviorFailure",
    "C11DiagnosticOccurrenceLocalizationResultV1",
    "C11DiagnosticOccurrenceLocalizationSpecV1",
    "C11PublicConsequenceV1",
    "C11RunReservationV1",
    "C11SelectedExecutionAttemptV1",
    "C11WorkV1",
    "ContinualRustDiagnosticOccurrenceLocalizationC11Error",
    "make_c11_diagnostic_occurrence_localization_spec_v1",
    "output_sha256_c11_v1",
    "run_and_write_c11_v1",
]


if __name__ == "__main__":
    main()
