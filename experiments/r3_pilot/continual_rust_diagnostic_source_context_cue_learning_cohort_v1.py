"""Fresh C10 target binding from authenticated rustc message and source-line words.

Four public TRAIN consequences cover the complete concrete 2x2 action grid.
Lowercase ASCII-alphanumeric unigrams from the parent diagnostic message and
the complete primary-span-containing source line enter one role-neutral set and
are projected uniformly to both action axes. Two fresh diagnostics are routed
by the final immutable memory before both selected edits execute and one shared
oracle opens.

This is consequence-conditioned target binding reuse, not held-out composition.
Tokenization, primary spans, operations, target markers, exploration answers,
edits, and outputs remain supplied. It is not semantics, localization,
diagnosis, synthesis, debugging, transfer, SWE performance, or intelligence.
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
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_rustc_message_line_unordered_cue_adapter_v1 as message_line
from experiments.r3_pilot import (
    common_rustc_message_line_unordered_cue_text_edit_authority_v1 as authority,
)
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from humanbrain import rust_selected_action_executor as rust_executor
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

SCHEMA = "humanbrain.continual-rust-diagnostic-source-context-cue-learning-cohort.v1"
RESULT_DOCUMENT_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-source-context-cue-learning-result-document.v1"
)
RESERVATION_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-source-context-cue-learning-reservation.v1"
)
STATUS_PASS = "PASS_C10_RUSTC_DIAGNOSTIC_SOURCE_CONTEXT_CUE_LEARNING_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C10_RUSTC_DIAGNOSTIC_SOURCE_CONTEXT_CUE_LEARNING_V1"
STATUS_EXECUTION_ABORT = "ABORT_C10_RUSTC_DIAGNOSTIC_SOURCE_CONTEXT_CUE_LEARNING_V1"
CLAIM_BOUNDARY = (
    "one-fresh-public-six-source-cohort-where-four-caller-attested-successful-"
    "rustc-consequences-bind-authenticated-parent-message-plus-complete-primary-"
    "span-containing-source-line-role-neutral-generic-word-unigrams-to-one-"
    "supplied-2x2-concrete-operation-target-bank-and-enable-normal-selection-on-"
    "two-fresh-identifier-disjoint-diagnostics-before-selected-only-"
    "materialization-with-no-evaluation-update-all-four-hypotheses-trained-not-"
    "held-out-composition-tokenization-primary-span-operation-target-bank-target-"
    "markers-exploration-answers-edits-and-outputs-supplied-not-semantics-"
    "localization-diagnosis-synthesis-debugging-transfer-swe-or-intelligence"
)

FROZEN_MECHANISM_COMMIT = "4ba0cf5caedd48b02bf5afdcb1fe0962cb439270"
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
)

SOURCE_FILE_NAME = "case.rs"
DIAGNOSTIC_ARGUMENTS = (
    "--edition",
    "2021",
    "--crate-name",
    "c10_source_context_cue_learning",
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
ROLES = (
    "train-h00-e0425-orlune",
    "train-h11-e0308-vescar",
    "train-h01-e0425-vescar",
    "train-h10-e0308-orlune",
    "evaluation-h01-e0425-vescar",
    "evaluation-h10-e0308-orlune",
)
EVALUATION_ORACLE_ROLE = "evaluation-oracle-after-both-executions"
TRAIN_COUNT = 4
EXPECTED_DIAGNOSTICS = (
    "E0425",
    "E0308",
    "E0425",
    "E0308",
    "E0425",
    "E0308",
)
EXPECTED_PRIMARY_BYTE_RANGES = (
    (364, 374),
    (380, 390),
    (364, 374),
    (380, 390),
    (364, 374),
    (383, 393),
)
EXPECTED_PRIMARY_TEXTS = (
    "peluc10t00",
    "safoc10t11",
    "niruc10t01",
    "wexoc10t10",
    "jiyac10e01",
    "bemoc10e10",
)
EXPECTED_PARENT_MESSAGES = (
    "cannot find value `peluc10t00` in this scope",
    "mismatched types",
    "cannot find value `niruc10t01` in this scope",
    "mismatched types",
    "cannot find value `jiyac10e01` in this scope",
    "mismatched types",
)
EXPECTED_MESSAGE_TOKENS = (
    ("cannot", "find", "in", "peluc10t00", "scope", "this", "value"),
    ("mismatched", "types"),
    ("cannot", "find", "in", "niruc10t01", "scope", "this", "value"),
    ("mismatched", "types"),
    ("cannot", "find", "in", "jiyac10e01", "scope", "this", "value"),
    ("mismatched", "types"),
)
EXPECTED_SOURCE_LINE_TOKENS = (
    ("c10t00vault", "guardc10t00", "itemc10t00", "map", "orlune", "peluc10t00"),
    ("guardc10t11", "itemc10t11", "map", "safoc10t11", "vescar"),
    ("c10t01vault", "guardc10t01", "itemc10t01", "map", "niruc10t01", "vescar"),
    ("guardc10t10", "itemc10t10", "map", "orlune", "wexoc10t10"),
    ("c10e01vault", "guardc10e01", "itemc10e01", "jiyac10e01", "map", "vescar"),
    ("bemoc10e10", "guardc10e10", "itemc10e10", "map", "orlune"),
)
EXPECTED_MERGED_CUE_COUNTS = (12, 7, 12, 7, 12, 7)
EXPECTED_EVALUATION_SUPPORTED_CUE_INTERSECTIONS = (
    ("cannot", "find", "in", "map", "scope", "this", "value", "vescar"),
    ("map", "mismatched", "orlune", "types"),
)
EVALUATION_IDENTIFIER_TOKENS = ("jiyac10e01", "bemoc10e10")
EXPECTED_EVALUATION_IDENTIFIER_SUPPORT_TICKS = 0
ALLOW_EXPLORATION_BY_EPISODE = (True, True, True, True, False, False)
BATCHES = (
    ((-101, -7, 23),),
    ((-83, 11, 29),),
    ((-97, 13, 31),),
    ((-109, 17, 41),),
    ((-127, 19, 46),),
    ((-131, 25, 49),),
)
EXPECTED_OUTPUTS = (
    ((-64, 30, 60),),
    ((-16, 78, 96),),
    ((-26, 84, 102),),
    ((-56, 70, 94),),
    ((-48, 98, 125),),
    ((-48, 108, 132),),
)

TRAIN_H00_SOURCE = """\
struct C10T00Vault {
    peluc10t00: i64,
}

fn orlune(flowc10t00: i64) -> i64 {
    flowc10t00
}

pub fn solve(seriesc10t00: &[i64]) -> Vec<i64> {
    let prixa = 61_i64;
    let nalvo = 37_i64;
    let guardc10t00 = (nalvo - nalvo) + (prixa - prixa);
    seriesc10t00
        .iter()
        .copied()
        .map(|itemc10t00| itemc10t00 + orlune(C10T00Vault { peluc10t00 }.peluc10t00) + guardc10t00)
        .collect()
}"""

TRAIN_H11_SOURCE = """\
fn vescar(flowc10t11: i64) -> i64 {
    flowc10t11
}

pub fn solve(seriesc10t11: &[i64]) -> Vec<i64> {
    let sorem = 43_i64;
    let talvi = 67_i64;
    let guardc10t11 = (sorem - sorem) + (talvi - talvi);
    let safoc10t11 = "inkc10t11";
    let _widthc10t11 = safoc10t11.len();
    seriesc10t11
        .iter()
        .copied()
        .map(|itemc10t11| itemc10t11 + vescar(safoc10t11) + guardc10t11)
        .collect()
}"""

TRAIN_H01_SOURCE = """\
struct C10T01Vault {
    niruc10t01: i64,
}

fn vescar(flowc10t01: i64) -> i64 {
    flowc10t01
}

pub fn solve(seriesc10t01: &[i64]) -> Vec<i64> {
    let lumra = 71_i64;
    let vekon = 47_i64;
    let guardc10t01 = (vekon - vekon) + (lumra - lumra);
    seriesc10t01
        .iter()
        .copied()
        .map(|itemc10t01| itemc10t01 + vescar(C10T01Vault { niruc10t01 }.niruc10t01) + guardc10t01)
        .collect()
}"""

TRAIN_H10_SOURCE = """\
fn orlune(flowc10t10: i64) -> i64 {
    flowc10t10
}

pub fn solve(seriesc10t10: &[i64]) -> Vec<i64> {
    let dorin = 53_i64;
    let kaspe = 73_i64;
    let guardc10t10 = (dorin - dorin) + (kaspe - kaspe);
    let wexoc10t10 = "inkc10t10";
    let _widthc10t10 = wexoc10t10.len();
    seriesc10t10
        .iter()
        .copied()
        .map(|itemc10t10| itemc10t10 + orlune(wexoc10t10) + guardc10t10)
        .collect()
}"""

EVALUATION_H01_SOURCE = """\
struct C10E01Vault {
    jiyac10e01: i64,
}

fn vescar(flowc10e01: i64) -> i64 {
    flowc10e01
}

pub fn solve(seriesc10e01: &[i64]) -> Vec<i64> {
    let zelun = 79_i64;
    let mavri = 59_i64;
    let guardc10e01 = (mavri - mavri) + (zelun - zelun);
    seriesc10e01
        .iter()
        .copied()
        .map(|itemc10e01| itemc10e01 + vescar(C10E01Vault { jiyac10e01 }.jiyac10e01) + guardc10e01)
        .collect()
}"""

EVALUATION_H10_SOURCE = """\
fn orlune(flowc10e10: i64) -> i64 {
    flowc10e10
}

pub fn solve(seriesc10e10: &[i64]) -> Vec<i64> {
    let beska = 89_i64;
    let quorin = 83_i64;
    let guardc10e10 = (quorin - quorin) + (beska - beska);
    let bemoc10e10 = "inkc10e10";
    let _widthc10e10 = bemoc10e10.len();
    seriesc10e10
        .iter()
        .copied()
        .map(|itemc10e10| itemc10e10 + orlune(bemoc10e10) + guardc10e10)
        .collect()
}"""

ORIGINAL_SOURCES = (
    TRAIN_H00_SOURCE,
    TRAIN_H11_SOURCE,
    TRAIN_H01_SOURCE,
    TRAIN_H10_SOURCE,
    EVALUATION_H01_SOURCE,
    EVALUATION_H10_SOURCE,
)
ORIGINAL_SOURCE_SIZES = (424, 427, 424, 427, 424, 430)
ORIGINAL_SOURCE_SHA256S = (
    "b6c0d6eb0128523f5135c0497453aa89ab42914e007eba8263bdc6bc64b14eb0",
    "7b7c814f1b0c3a55cce059eedefa3fc6ec389078aec0b7e6da32c15ef4e3890e",
    "80a51272bb0f3f2dab103916f34025c1efb384f33cf2d1176b276e7149bd38fc",
    "b1235cd4490497f9e3d17db82408f3d2d47355261dd32792360fa4ff19c2e458",
    "55466beed71c9e57c2cc1d8c31835d29500e187327903b5fde4038724d8e0f5d",
    "24bab5e10a02ac039b1fec33de22be26b27af3524b360ce24947a5ab78df880c",
)
EXPECTED_MATERIALIZED_SOURCE_SHA256S = (
    "5fb8edd3d42c1dd2c8e7fd95aec4cf6809cb8158cc92929b371653718cc03fc2",
    "58fc380502a2e3ba06debbd720a8dad7d4dd223183ac8d1eb4cd8dcc324d77b1",
    "0e53f26ec756454b046785af357fee1be1db7e9e8fce8c1b4683d2cc6217ac44",
    "7cfdf714f80dba936574bca220d6a8e82fde7907c8feab8601f5315105c460ed",
    "efe7f26661853a16930fb596374851727c3d0f32a02ca9c3e3bd610eca91fe4d",
    "becaa22a3e161cb8c825d6f70a6235604971d33f6a05e704a8553abf3b741bc9",
)
TARGET_MARKERS = (
    ("nalvo", "prixa"),
    ("sorem", "talvi"),
    ("vekon", "lumra"),
    ("dorin", "kaspe"),
    ("mavri", "zelun"),
    ("quorin", "beska"),
)

_SHA256_HEX = frozenset("0123456789abcdef")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-spec-v1"
_COMPONENT_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-component-v1"
_CONTEXT_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-context-v1"
_OUTPUT_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-output-v1"
_OUTPUT_FEEDBACK_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-feedback-v1"
_CONSEQUENCE_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-consequence-v1"
_ATTEMPT_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-attempt-v1"
_WORK_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-work-v1"
_RESERVATION_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-reservation-v1"
_RESULT_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-result-v1"
_FAILURE_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-failure-v1"
_DOCUMENT_NAMESPACE = b"c10-rustc-diagnostic-source-context-cue-learning-document-v1"


class ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(ValueError):
    """The frozen C10 cohort or its authenticated transcript changed."""


class C10BehaviorFailure(RuntimeError):
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
    raise TypeError(f"C10 transcript cannot serialize {type(value).__qualname__}")


def _behavior_failure(reason: str, **partial_transcript: object) -> C10BehaviorFailure:
    return C10BehaviorFailure(
        reason,
        {key: _plain(value) for key, value in partial_transcript.items()},
    )


OPERATOR_ACTION_SHA256S = (
    catalogue.DECLARE_PRIMARY_IDENTIFIER_FROM_ANCHOR_ACTION_V1.sha256,
    catalogue.REPLACE_PRIMARY_SPAN_WITH_ANCHOR_ACTION_V1.sha256,
)
TARGET_COMPONENT_SHA256S = (
    _digest(_COMPONENT_NAMESPACE, {"axis": 1, "value": "target-zero"}),
    _digest(_COMPONENT_NAMESPACE, {"axis": 1, "value": "target-one"}),
)
HYPOTHESES_BY_KEY = {
    (operator_index, target_index): factors.seal_factorized_hypothesis_v1(
        (
            OPERATOR_ACTION_SHA256S[operator_index],
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
EXPLORATION_HYPOTHESIS_SHA256S = (
    H00_SHA256,
    H11_SHA256,
    H01_SHA256,
    H10_SHA256,
)
EVALUATION_HYPOTHESIS_SHA256S = (H01_SHA256, H10_SHA256)
EXPECTED_NORMAL_TRAIN_SELECTIONS = (None, H00_SHA256, H00_SHA256, None)
EXPECTED_DIAGONAL_EVALUATION_SELECTIONS = (H00_SHA256, H11_SHA256)
EXPECTED_THREE_LESSON_EVALUATION_SELECTIONS = (H01_SHA256, None)
EXPECTED_SCORE_MATRICES = (
    # code, correct, same-operation/wrong-target, wrong-operation/same-target, both-wrong
    ("E0425", 50, 46, 26, 22),
    ("E0308", 26, 22, 18, 14),
)
EXPECTED_DIAGONAL_SCORE_MAPS = (
    tuple(sorted(((H00_SHA256, 28), (H01_SHA256, 18), (H10_SHA256, 18), (H11_SHA256, 8)))),
    tuple(sorted(((H00_SHA256, 8), (H01_SHA256, 10), (H10_SHA256, 10), (H11_SHA256, 12)))),
)
EXPECTED_THREE_LESSON_SCORE_MAPS = (
    tuple(sorted(((H00_SHA256, 44), (H01_SHA256, 50), (H10_SHA256, 18), (H11_SHA256, 24)))),
    tuple(sorted(((H00_SHA256, 10), (H01_SHA256, 14), (H10_SHA256, 10), (H11_SHA256, 14)))),
)
EXPECTED_FINAL_SCORE_MAPS = (
    tuple(sorted(((H00_SHA256, 46), (H01_SHA256, 50), (H10_SHA256, 22), (H11_SHA256, 26)))),
    tuple(sorted(((H00_SHA256, 18), (H01_SHA256, 14), (H10_SHA256, 26), (H11_SHA256, 22)))),
)
OPERATOR_BINDINGS = catalogue.canonical_factorized_text_edit_operator_bindings_v1()
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

# C9's 75 prior exclusions plus its six original and six materialized roots.
EXCLUDED_PRIOR_SOURCE_SHA256S = (
    "077283306f7ecdd8e80fc072fa23db8902593deff7e67543c0c405a23d0009f6",
    "082120df1274f6b90bccb3c064810a37079e892560b9429cda63f68554020338",
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
    "178e9ffa2a3c9faf01739db95195c5575d9da57c6835e9a4945269bdb7ee0fc8",
    "182c0e0084471185b6d0cd16271bc43b1eb7561a49a8a099bbbf4027abe9117f",
    "197a5fcbb6bb71b839994f70b43718610a8ebd43e35dd1050afb3bc7d1dc6572",
    "1f0fa31aafb6d3737b0a394c4a9f31140aff3f543d35911c80ac184ba828aeb9",
    "2018ff95b0b6c90abe9a754d1490b56ef6a854cd3801c4a9973ead72e57926d9",
    "22d555f484f9aa33e1f5c184c5d91893aac1b02593ca5f1e4d28f1d9187418db",
    "2cb5e3555404d53084672be16652220a61e0c448ee9b9f97bd1084e4a892e2ef",
    "30b0f5e5d23fc2b8ede3c6fcb3892078b89be2fec1ddb530a50251fc48a0d33e",
    "319f315176807b6f4db650e74d0775f65af626a06d5f4caf87031dd692ee8c7a",
    "320434b33f67afd875de8232703e25da7c4032f367c3a76dfccbd572df226243",
    "3598b6e3bfdc014c579d1c85bb38748506f83115c9f61ed9080b2b850d8745fc",
    "3909b3793cc81ebf2cc672a7021cfff4fa340eca655b26e52d0e55481d403670",
    "3a76a6783fda0b2aef342124467e2fb1245ecea140052760dabed7b10d10799b",
    "3c354eda2adea9f7adf50ac6368394e00370555d7a999d7da96c873dd45947f6",
    "4588278d064b626f0a9c6e6c6067992d64ac547d09e1ff15e386f9d441f2d1b7",
    "4ecf619642ddf0dee5466334c755544630f9d3656aea073c55e3a30777f4b79f",
    "514720a04d50420dbc4d1da9be1c24da37be8102009ecdcaa7259fc114795112",
    "532e90d9739ec16748ad152d53509d34de7ccbf5efccfb067aa915789210f662",
    "55ec139329ee99ccebde3c3ffcc5d7dac5515075c9ff0095e6068517009fb1ff",
    "5c439c971363f6863f0386b16c90ac9258d0d8d4f82fccfab300bfeec91b534a",
    "5ea76f081fe59993eaabd0da33cf9f2b1f8429ff3ea876c3f6a86fb2f6892c69",
    "5efc39f66d45b1252a76ddd536a251d42e5956aaa84f61e3a2763b7d1d2a3d79",
    "61874992d6db72fcc87f6101b1a9e8deb8cf46904c4617596e04b1481b1c38d2",
    "62b3a80e76db43ce2c726ae246bf487eb26ba4c40e02b7c15a2fcedee36a3e3f",
    "652e21dfb435032e94d7f0b72e5c5ed3c28faf2f4ba32c09cb203a57d5c785c9",
    "66703237df4e8f68e7f9e678239970c93e218e7267bae1aeb4b8833e7a475a93",
    "67db118afe3d6d529adb81835b058771d922b86b2f17b11270d0e37f2044bad0",
    "737b90128d66b22d7f5f15e04780eaa2920c85612a9f8bb24f4f814c0afe8d70",
    "767ac8ba99b72cfc2b148644c6e02f015dfe12bef19a1946288ccf4ae608f9cb",
    "7ea07d6438adf64d57210c35a69d88c4a48c5c8f5cc653246d00b32e3924d9e2",
    "829d367e3845f66d4c3c1a9bafa43baaec733b618c600ae9da505479f6b332c7",
    "83d7dbb8d55f0f9b06d6750ac4a3552981ec30700f9f65e11c753cf76a5d0f49",
    "875e1e20df98385178b3af24600bfd9bed7973b6333ebbdad65fe19b07d275aa",
    "8b225ebb888c282c0e6293215cdaecb36427efcbe3ca7317bbf322e5e1888c49",
    "900e4f8cc9648f00e4835b50185f24da64d12babb35dbf5bf65935f575398362",
    "913097c4d9e858c41d80a8d671a5996340a9d3071d1b1902350bb4403a236f88",
    "9156378419e5a875a0aa88632bbada13c3261e07ca979ce1ea9afbf650025975",
    "962a5bcd733f4026a07dea5342aa5145ea837bd282edbf67d41418da0f37f6dd",
    "9634b257c00cc1f0e2470cead58cc6ffec8136fb1a2388511b8c5f17b01dd9fc",
    "967382da9e579783e7fe5cb5cbf3c4e1a92f5d8ca84948aa641d0f289f62fb8d",
    "96f90f7f7d3e6a3677ce8cd1e7ddb3cc9986a49d42180a5f74b2375a1b6ed985",
    "a0c554ee679a10b37bad6fe3e7b6d6b905777d579302ab1cde2cf7f69f43b5dc",
    "a30f2d85a0f2fcd28458f01e2353b20bd7567478c9f0e14ba900becdcca0da15",
    "a35874ffedbaabac33977d554916c049a823434b35d57e5ba82be0770ab088a7",
    "a54d2fe9f99aaebfbe7cd98fe991c26142bea39a172f9ffa3c05117af17876a0",
    "a5979ba178a792517f4e675fba66fe1a61f96c1a383e8453f32c5320f22fd04a",
    "a8102ce87ef9940510e2921ce401d3a66aa1b6ac08a6865fc84b733bc322cd71",
    "b037904c58d0ab527c2d9e20595a6071ad786a0748165a2684f3af54448a417f",
    "b05b9acb3804dce03571ccb1c42e707cb799ca29ff4c4dc0bcdce0fefa0344af",
    "b0cfe61f4d7c74758f0e7b165011ae660210f26079274a4179e585ad9279f742",
    "b0e5bff5168441980089f2114b942d695b8d3496d6a67b86c96f7ffc8ca25ff7",
    "b2ab91c571c3ab154735d98bb584502ba0039030372c33a2b75ad86992931319",
    "b3cbeea34d564dd583a681142ad27d672c4ca64f9a7386fbb6c319b868c6cf0e",
    "b8bda9c33ba7d62f481387d0da28042c91a651a1e0c99b717b868a1f9078d81a",
    "b9f5dae974ca36de1fdfb9d6e3a3a7cca962ccde9bda4dda57cc017468467bb8",
    "bb29c7ac1a1b36d79131fc7dac4c3264b3fb1e053c5c82e4c49d706e0f412bd4",
    "bda397623caaf27ecef7e73e0b6052a0c12c3ff419333ef1f7231a1e011485d3",
    "be1bc451971f347f22d0072a3411d15a54f00c1d4c819f1be9d575330455625d",
    "be1cddea32682fb8f533adfbb1e78621b8d67a024594317aace73e6d87befe9f",
    "c199578bb5c3d6df27a533cacf04d0d14746fe12b5ae556b9c805f2d746df5d0",
    "c5bc2f2af0a579325f857006e222f4f59f4f44bd58eceaa0dd8e82baba16f9c5",
    "ca96a9774be460daed557117e0b56fc334d29d709d9fac0a6f53003e3fd9be34",
    "cdddc8926d80232b494132e1687f0e344b92c0fdb5c667ed4a7051462ee72762",
    "d00939b3ee635b6011a6e3296550dfef6fa1fa1d3211f42514822300bb37332c",
    "d23bee65de3f003933e9739add066b9c1973bb5760abfc6f15129b9b1fe169e3",
    "d335b523ec67621ba6925c48c84f1d4cce2f38f00956099297db443d1fe80f50",
    "d34b307aa6cef6e403ee88698897f73dfdb21511aee7b1b629baf5631eaab6d6",
    "d907300fcf3390c3b8697a8437f8448cc10057736ac1bdee3c27084f2a7b0b7f",
    "e45e23ac3e82c4146d5aafee56590c9368b758ce744b04e52547e30a7374b855",
    "e6f89d9d2be762471863ff3c6a8b00dd3eef59ad388eed55388883198385429f",
    "e8dd412b73a6612ab6a70483ef738881ec09b2d9c429638cffab630648bcb3e8",
    "ee0b4c24a158c1d1e66bfbeaa5fcdccdaef641b5bd1b161f9a73054f9915e675",
    "ef114a5c9493d7f6cc5726882217322dd7d8ccee527cd4811a01e1ed1bc3ba55",
    "fbeac8ce5b168255e52c24318106a00194f7a5999c21bc453987227fddb5cb94",
    "fe2047e404f00495f1a10493b2261d03709b393387e21cc5ed813efd9df2f7cd",
)

EXPECTED_WORK_COUNTS = (
    ("diagnostic_compile_count", 6),
    ("retained_message_line_envelope_count", 6),
    ("materialization_count", 6),
    ("release_count", 6),
    ("selected_source_compile_count", 6),
    ("selected_source_run_count", 6),
    ("training_feedback_open_count", 4),
    ("evaluation_oracle_open_count", 1),
    ("memory_advance_count", 4),
    ("evaluation_memory_advance_count", 0),
    ("parent_decision_count", 2),
    ("incomplete_grid_decision_count", 4),
    ("evaluation_decision_count", 2),
    ("control_decision_count", 6),
    ("evaluation_execution_before_oracle_count", 2),
    ("alternative_materialization_count", 0),
)


def _current_mechanism_file_sha256s() -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, sha256((_REPO_ROOT / path).read_bytes()).hexdigest())
        for path, _root in FROZEN_MECHANISM_FILE_SHA256S
    )


def output_sha256_v1(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(_OUTPUT_NAMESPACE, {"outputs": outputs})


def _class_token_overlap(
    token_inventories: tuple[tuple[str, ...], ...],
    train_indexes: tuple[int, ...],
    evaluation_index: int,
) -> tuple[str, ...]:
    training = {token for index in train_indexes for token in token_inventories[index]}
    return tuple(sorted(training.intersection(token_inventories[evaluation_index])))


@dataclass(frozen=True, slots=True)
class C10DiagnosticSourceContextCueLearningSpecV1:
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
    allow_exploration_by_episode: tuple[bool, ...]
    batches: tuple[tuple[tuple[int, ...], ...], ...]
    expected_output_sha256s: tuple[str, ...]
    original_source_sha256s: tuple[str, ...]
    expected_materialized_source_sha256s: tuple[str, ...]
    operator_bindings: tuple[tuple[str, str], ...]
    target_bindings_by_episode: tuple[tuple[tuple[str, str], ...], ...]
    hypothesis_sha256s: tuple[str, ...]
    exploration_hypothesis_sha256s: tuple[str, ...]
    evaluation_hypothesis_sha256s: tuple[str, ...]
    expected_score_matrices: tuple[tuple[str | int, ...], ...]
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

    def authenticate(self) -> None:
        source_bytes = tuple(source.encode("ascii") for source in ORIGINAL_SOURCES)
        derived_source_sha256s = tuple(sha256(value).hexdigest() for value in source_bytes)
        message_tokens = tuple(
            ascii_cues.ascii_word_unigrams_v1(message) for message in EXPECTED_PARENT_MESSAGES
        )
        line_texts = tuple(
            source[
                source.rfind(b"\n", 0, byte_start) + 1 : (
                    len(source)
                    if source.find(b"\n", byte_end) < 0
                    else source.find(b"\n", byte_end)
                )
            ].decode("ascii")
            for source, (byte_start, byte_end) in zip(
                source_bytes, EXPECTED_PRIMARY_BYTE_RANGES, strict=True
            )
        )
        source_line_tokens = tuple(ascii_cues.ascii_word_unigrams_v1(line) for line in line_texts)
        merged_tokens = tuple(
            tuple(sorted({*message, *line}))
            for message, line in zip(message_tokens, source_line_tokens, strict=True)
        )
        training_tokens = {token for row in merged_tokens[:TRAIN_COUNT] for token in row}
        supported_eval = (
            tuple(sorted(training_tokens.intersection(merged_tokens[4]))),
            tuple(sorted(training_tokens.intersection(merged_tokens[5]))),
        )
        e0425_tokens = {token for index in (0, 2, 4) for token in message_tokens[index]}
        e0308_tokens = {token for index in (1, 3, 5) for token in message_tokens[index]}
        if (
            self.frozen_mechanism_commit != FROZEN_MECHANISM_COMMIT
            or not _valid_commit(self.frozen_mechanism_commit)
            or self.frozen_mechanism_file_sha256s != FROZEN_MECHANISM_FILE_SHA256S
            or self.frozen_mechanism_file_sha256s != _current_mechanism_file_sha256s()
            or len(self.frozen_mechanism_file_sha256s) != 12
            or self.diagnostic_arguments != DIAGNOSTIC_ARGUMENTS
            or self.expected_diagnostics != EXPECTED_DIAGNOSTICS
            or self.expected_primary_byte_ranges != EXPECTED_PRIMARY_BYTE_RANGES
            or self.expected_parent_messages != EXPECTED_PARENT_MESSAGES
            or self.expected_message_tokens != EXPECTED_MESSAGE_TOKENS
            or self.expected_message_tokens != message_tokens
            or self.expected_source_line_tokens != EXPECTED_SOURCE_LINE_TOKENS
            or self.expected_source_line_tokens != source_line_tokens
            or self.expected_merged_cue_counts != EXPECTED_MERGED_CUE_COUNTS
            or self.expected_merged_cue_counts != tuple(map(len, merged_tokens))
            or self.expected_evaluation_supported_cue_intersections
            != EXPECTED_EVALUATION_SUPPORTED_CUE_INTERSECTIONS
            or self.expected_evaluation_supported_cue_intersections != supported_eval
            or e0425_tokens.intersection(e0308_tokens)
            or self.evaluation_identifier_tokens != EVALUATION_IDENTIFIER_TOKENS
            or any(token in training_tokens for token in self.evaluation_identifier_tokens)
            or self.evaluation_identifier_tokens[0] not in merged_tokens[4]
            or self.evaluation_identifier_tokens[1] not in merged_tokens[5]
            or self.expected_evaluation_identifier_support_ticks
            != EXPECTED_EVALUATION_IDENTIFIER_SUPPORT_TICKS
            or self.allow_exploration_by_episode != ALLOW_EXPLORATION_BY_EPISODE
            or self.batches != BATCHES
            or self.expected_output_sha256s
            != tuple(output_sha256_v1(value) for value in EXPECTED_OUTPUTS)
            or self.original_source_sha256s != ORIGINAL_SOURCE_SHA256S
            or self.original_source_sha256s != derived_source_sha256s
            or tuple(map(len, source_bytes)) != ORIGINAL_SOURCE_SIZES
            or any(b"\r" in value or value.endswith(b"\n") for value in source_bytes)
            or self.expected_materialized_source_sha256s != EXPECTED_MATERIALIZED_SOURCE_SHA256S
            or self.operator_bindings != OPERATOR_BINDINGS
            or self.target_bindings_by_episode != TARGET_BINDINGS_BY_EPISODE
            or self.hypothesis_sha256s != tuple(value.sha256 for value in HYPOTHESES)
            or len(set(self.hypothesis_sha256s)) != 4
            or self.exploration_hypothesis_sha256s != EXPLORATION_HYPOTHESIS_SHA256S
            or self.evaluation_hypothesis_sha256s != EVALUATION_HYPOTHESIS_SHA256S
            or self.expected_score_matrices != EXPECTED_SCORE_MATRICES
            or self.expected_work_counts != EXPECTED_WORK_COUNTS
            or self.excluded_source_sha256s != EXCLUDED_PRIOR_SOURCE_SHA256S
            or tuple(sorted(self.excluded_source_sha256s)) != self.excluded_source_sha256s
            or len(set(self.excluded_source_sha256s)) != 87
            or set(derived_source_sha256s).intersection(self.excluded_source_sha256s)
            or set(EXPECTED_MATERIALIZED_SOURCE_SHA256S).intersection(
                {*derived_source_sha256s, *self.excluded_source_sha256s}
            )
            or any(len(tokens) > message_line.MAX_TOTAL_CUES for tokens in merged_tokens)
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 public fixture specification changed"
            )


def make_c10_diagnostic_source_context_cue_learning_spec_v1() -> (
    C10DiagnosticSourceContextCueLearningSpecV1
):
    partial = C10DiagnosticSourceContextCueLearningSpecV1(
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
        expected_evaluation_identifier_support_ticks=(EXPECTED_EVALUATION_IDENTIFIER_SUPPORT_TICKS),
        allow_exploration_by_episode=ALLOW_EXPLORATION_BY_EPISODE,
        batches=BATCHES,
        expected_output_sha256s=tuple(output_sha256_v1(value) for value in EXPECTED_OUTPUTS),
        original_source_sha256s=ORIGINAL_SOURCE_SHA256S,
        expected_materialized_source_sha256s=EXPECTED_MATERIALIZED_SOURCE_SHA256S,
        operator_bindings=OPERATOR_BINDINGS,
        target_bindings_by_episode=TARGET_BINDINGS_BY_EPISODE,
        hypothesis_sha256s=tuple(value.sha256 for value in HYPOTHESES),
        exploration_hypothesis_sha256s=EXPLORATION_HYPOTHESIS_SHA256S,
        evaluation_hypothesis_sha256s=EVALUATION_HYPOTHESIS_SHA256S,
        expected_score_matrices=EXPECTED_SCORE_MATRICES,
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


DEFAULT_PUBLIC_SPEC = make_c10_diagnostic_source_context_cue_learning_spec_v1()


@dataclass(frozen=True, slots=True)
class C10RunReservationV1:
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
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 run reservation changed"
            )


@dataclass(frozen=True, slots=True)
class C10PublicConsequenceV1:
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
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 public consequence authentication failed"
            )


def _seal_consequence(
    *,
    execution: RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
    output_feedback_sha256: str,
) -> C10PublicConsequenceV1:
    payload = {
        "source_sha256": execution.selected_source.source_sha256,
        "batch_sha256": execution.preparation.batch_sha256,
        "output_feedback_sha256": output_feedback_sha256,
        "execution_sha256": execution.sha256,
        "observed_outputs": execution.outputs,
        "expected_outputs": expected_outputs,
        "passed_cell": int(execution.outputs == expected_outputs),
    }
    result = C10PublicConsequenceV1(
        **payload,
        sha256=_digest(_CONSEQUENCE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _authority_work_counts(
    prepared: authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1,
) -> tuple[int, int, int, int]:
    work = prepared.work_counters
    return (
        work.materialization_count,
        work.release_count,
        work.execution_attempt_count,
        work.successful_execution_count,
    )


@dataclass(frozen=True, slots=True)
class C10SelectedExecutionAttemptV1:
    ordinal: int
    role: str
    decision_sha256: str
    selected_hypothesis_sha256: str
    authority_receipt: authority.RustcMessageLineUnorderedCueTextEditAuthorityReceiptV1
    message_line_envelope_sha256: str
    pre_execute_work: tuple[int, int, int, int]
    post_execute_work: tuple[int, int, int, int]
    materialization: authority.UnorderedCueFactorTextEditMaterializationV1
    release: authority.UnorderedCueFactorTextEditReleaseV1
    execution: RustSelectedActionExecutionV2
    consequence: C10PublicConsequenceV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "ordinal": self.ordinal,
            "role": self.role,
            "decision_sha256": self.decision_sha256,
            "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
            "authority_receipt": self.authority_receipt.record(),
            "message_line_envelope_sha256": self.message_line_envelope_sha256,
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
        index = self.ordinal - 1
        selected_source = self.execution.selected_source
        expected_key = ((0, 0), (1, 1), (0, 1), (1, 0), (0, 1), (1, 0))[index]
        if (
            type(self.ordinal) is not int
            or not 1 <= self.ordinal <= len(ROLES)
            or self.role != ROLES[index]
            or self.selected_hypothesis_sha256
            != (*EXPLORATION_HYPOTHESIS_SHA256S, *EVALUATION_HYPOTHESIS_SHA256S)[index]
            or self.pre_execute_work != (0, 0, 0, 0)
            or self.post_execute_work != (1, 1, 1, 1)
            or self.authority_receipt.allow_exploration != ALLOW_EXPLORATION_BY_EPISODE[index]
            or self.authority_receipt.message_line_envelope_sha256
            != self.message_line_envelope_sha256
            or self.authority_receipt.sha256 != self.materialization.authority_sha256
            or self.decision_sha256 != self.materialization.decision_sha256
            or self.selected_hypothesis_sha256 != self.materialization.selected_hypothesis_sha256
            or self.materialization.operator_action_sha256
            != HYPOTHESES_BY_KEY[expected_key].component_sha256s[0]
            or self.release.authority_sha256 != self.authority_receipt.sha256
            or self.release.selection_sha256 != self.decision_sha256
            or self.release.materialization != self.materialization
            or selected_source.authority_sha256 != self.authority_receipt.sha256
            or selected_source.selection_sha256 != self.decision_sha256
            or selected_source.release_sha256 != self.release.sha256
            or selected_source.source_sha256 != self.release.source_sha256
            or selected_source.source_sha256 != EXPECTED_MATERIALIZED_SOURCE_SHA256S[index]
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
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 selected execution attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C10WorkV1:
    diagnostic_compile_count: int
    retained_message_line_envelope_count: int
    materialization_count: int
    release_count: int
    selected_source_compile_count: int
    selected_source_run_count: int
    training_feedback_open_count: int
    evaluation_oracle_open_count: int
    memory_advance_count: int
    evaluation_memory_advance_count: int
    parent_decision_count: int
    incomplete_grid_decision_count: int
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
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 work accounting changed"
            )


def _seal_work() -> C10WorkV1:
    payload = dict(EXPECTED_WORK_COUNTS)
    result = C10WorkV1(**payload, sha256=_digest(_WORK_NAMESPACE, payload))
    result.authenticate()
    return result


def _decision_scores(decision: unordered.UnorderedCueFactorDecisionV1) -> dict[str, int]:
    return {item.hypothesis.sha256: item.total_score for item in decision.hypothesis_scores}


def _expected_binding_evidence(
    observations: tuple[unordered.UnorderedCueFactorObservationV1, ...],
) -> dict[tuple[str, int, str], tuple[int, int]]:
    expected: dict[tuple[str, int, str], tuple[int, int]] = {}
    lesson_keys = ((0, 0), (1, 1), (0, 1), (1, 0))
    for cue in sorted({cue for item in observations[:TRAIN_COUNT] for cue in item.cue_sha256s}):
        for action_axis, components in enumerate(
            (OPERATOR_ACTION_SHA256S, TARGET_COMPONENT_SHA256S)
        ):
            for component in components:
                excitatory = sum(
                    cue in observation.cue_sha256s
                    and HYPOTHESES_BY_KEY[key].component_sha256s[action_axis] == component
                    for observation, key in zip(
                        observations[:TRAIN_COUNT], lesson_keys, strict=True
                    )
                )
                expected[(cue, action_axis, component)] = (excitatory, 0)
    return expected


def _authenticate_binding_evidence(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    observations: tuple[unordered.UnorderedCueFactorObservationV1, ...],
    envelopes: tuple[message_line.RustcMessageLineUnorderedCueEnvelopeV1, ...],
) -> None:
    actual = {
        (row.cue_sha256, row.action_axis, row.component_sha256): (
            row.excitatory_ticks,
            row.inhibitory_ticks,
        )
        for row in memory.binding_evidence
    }
    expected = _expected_binding_evidence(observations)
    trained_cues = {cue for item in observations[:TRAIN_COUNT] for cue in item.cue_sha256s}
    training_tokens = {
        token
        for index in range(TRAIN_COUNT)
        for token in (*EXPECTED_MESSAGE_TOKENS[index], *EXPECTED_SOURCE_LINE_TOKENS[index])
    }
    expected_evaluation_only_tokens = tuple(
        tuple(
            sorted(
                {
                    *EXPECTED_MESSAGE_TOKENS[index],
                    *EXPECTED_SOURCE_LINE_TOKENS[index],
                }
                - training_tokens
            )
        )
        for index in range(TRAIN_COUNT, len(ROLES))
    )
    expected_evaluation_only_cues = tuple(
        {ascii_cues.ascii_word_unigram_cue_sha256_v1(token) for token in episode_tokens}
        for episode_tokens in expected_evaluation_only_tokens
    )
    actual_evaluation_only_cues = tuple(
        set(envelope.observation.cue_sha256s) - trained_cues for envelope in envelopes[TRAIN_COUNT:]
    )
    all_evaluation_only_cues = set().union(*actual_evaluation_only_cues)
    identifier_cues = {
        ascii_cues.ascii_word_unigram_cue_sha256_v1(token) for token in EVALUATION_IDENTIFIER_TOKENS
    }
    unsupported_ticks = {
        cue: sum(
            row.excitatory_ticks + row.inhibitory_ticks
            for row in memory.binding_evidence
            if row.cue_sha256 == cue
        )
        for cue in all_evaluation_only_cues
    }
    if (
        actual != expected
        or len(actual) != 100
        or len(trained_cues) != 25
        or tuple(map(len, actual_evaluation_only_cues)) != (4, 3)
        or actual_evaluation_only_cues != expected_evaluation_only_cues
        or len(all_evaluation_only_cues) != 7
        or identifier_cues - all_evaluation_only_cues
        or any(unsupported_ticks.get(cue) != 0 for cue in identifier_cues)
        or any(unsupported_ticks.values())
        or any(key[0] in all_evaluation_only_cues for key in actual)
    ):
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 message-line binding or evaluation-only u=0 evidence changed"
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
        )
    )


def _authenticate_unordered_and_context_controls(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    envelopes: tuple[message_line.RustcMessageLineUnorderedCueEnvelopeV1, ...],
) -> None:
    evaluation_envelopes = envelopes[TRAIN_COUNT:]
    permuted_observations = tuple(
        unordered.seal_unordered_cue_factor_observation_v1(
            selection_context_sha256=envelope.observation.selection_context_sha256,
            cue_sha256s=tuple(reversed(envelope.observation.cue_sha256s)),
        )
        for envelope in evaluation_envelopes
    )
    context_tokens = ("vescar", "orlune")
    context_cues = tuple(
        ascii_cues.ascii_word_unigram_cue_sha256_v1(token) for token in context_tokens
    )
    ablated_observations = tuple(
        _ablation_observation(
            envelope,
            tuple(cue for cue in envelope.observation.cue_sha256s if cue != context_cue),
        )
        for envelope, context_cue in zip(evaluation_envelopes, context_cues, strict=True)
    )
    deranged_observations = tuple(
        _ablation_observation(
            envelope,
            tuple(
                sorted(
                    {
                        *(cue for cue in envelope.observation.cue_sha256s if cue != context_cue),
                        replacement_cue,
                    }
                )
            ),
        )
        for envelope, context_cue, replacement_cue in zip(
            evaluation_envelopes,
            context_cues,
            reversed(context_cues),
            strict=True,
        )
    )
    if (
        permuted_observations != tuple(envelope.observation for envelope in evaluation_envelopes)
        or tuple(memory.route(item) for item in permuted_observations)
        != tuple(memory.route(envelope.observation) for envelope in evaluation_envelopes)
        or any(
            memory.route(item).selected_hypothesis_sha256 is not None
            for item in ablated_observations
        )
        or tuple(memory.route(item).selected_hypothesis_sha256 for item in deranged_observations)
        != (H00_SHA256, H11_SHA256)
    ):
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 unordered permutation, context ablation, or derangement changed"
        )


@dataclass(frozen=True, slots=True)
class C10DiagnosticSourceContextCueLearningResultV1:
    status: str
    spec: C10DiagnosticSourceContextCueLearningSpecV1
    reservation: C10RunReservationV1
    diagnostic_attempts: tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, ...]
    message_line_envelopes: tuple[message_line.RustcMessageLineUnorderedCueEnvelopeV1, ...]
    observations: tuple[unordered.UnorderedCueFactorObservationV1, ...]
    memories: tuple[unordered.FrozenUnorderedCueFactorMemoryV1, ...]
    adaptation_normal_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    exploration_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    adaptation_outcomes: tuple[unordered.UnorderedCueFactorOutcomeV1, ...]
    parent_evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    diagonal_evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    three_lesson_evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    control_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    selected_attempts: tuple[C10SelectedExecutionAttemptV1, ...]
    output_feedback_sha256s: tuple[str, ...]
    evaluation_oracle_sha256: str
    evaluation_checkpoint_before_sha256: str
    evaluation_checkpoint_after_sha256: str
    work: C10WorkV1
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
            "message_line_envelopes": tuple(item.record() for item in self.message_line_envelopes),
            "observations": tuple(item.record() for item in self.observations),
            "memories": tuple(item.record() for item in self.memories),
            "adaptation_normal_decisions": tuple(
                item.record() for item in self.adaptation_normal_decisions
            ),
            "exploration_decisions": tuple(item.record() for item in self.exploration_decisions),
            "adaptation_outcomes": tuple(item.record() for item in self.adaptation_outcomes),
            "parent_evaluation_decisions": tuple(
                item.record() for item in self.parent_evaluation_decisions
            ),
            "diagonal_evaluation_decisions": tuple(
                item.record() for item in self.diagonal_evaluation_decisions
            ),
            "three_lesson_evaluation_decisions": tuple(
                item.record() for item in self.three_lesson_evaluation_decisions
            ),
            "evaluation_decisions": tuple(item.record() for item in self.evaluation_decisions),
            "control_decisions": tuple(item.record() for item in self.control_decisions),
            "selected_attempts": tuple(item.record() for item in self.selected_attempts),
            "output_feedback_sha256s": self.output_feedback_sha256s,
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
                    self.observations,
                    self.memories,
                    self.adaptation_normal_decisions,
                    self.exploration_decisions,
                    self.adaptation_outcomes,
                    self.parent_evaluation_decisions,
                    self.diagonal_evaluation_decisions,
                    self.three_lesson_evaluation_decisions,
                    self.evaluation_decisions,
                    self.control_decisions,
                    self.selected_attempts,
                    self.output_feedback_sha256s,
                ),
            )
        )
        if (
            self.status != STATUS_PASS
            or self.reservation.spec_sha256 != self.spec.sha256
            or cardinalities != (6, 6, 6, 5, 4, 4, 4, 2, 2, 2, 2, 6, 6, 6)
        ):
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 result tuple cardinality changed"
            )
        self.work.authenticate()
        for attempt in self.diagnostic_attempts:
            attempt.authenticate()
        for envelope in self.message_line_envelopes:
            envelope.authenticate()
        for item in self.observations:
            item.authenticate()
        for item in self.memories:
            item.authenticate()
        for group in (
            self.adaptation_normal_decisions,
            self.exploration_decisions,
            self.parent_evaluation_decisions,
            self.diagonal_evaluation_decisions,
            self.three_lesson_evaluation_decisions,
            self.evaluation_decisions,
            self.control_decisions,
        ):
            for item in group:
                item.authenticate()
        for item in self.adaptation_outcomes:
            item.authenticate()
        for item in self.selected_attempts:
            item.authenticate()

        expected_parent = unordered.start_unordered_cue_factor_memory_v1(HYPOTHESES)
        for index, (attempt, envelope) in enumerate(
            zip(self.diagnostic_attempts, self.message_line_envelopes, strict=True)
        ):
            _authenticate_diagnostic_for_episode(
                index=index,
                attempt=attempt,
                stderr=envelope.stderr,
                rustc_identity_sha256=self.reservation.rustc_identity_sha256,
            )
            expected_message_cues = tuple(
                ascii_cues.ascii_word_unigram_cue_sha256_v1(token)
                for token in EXPECTED_MESSAGE_TOKENS[index]
            )
            expected_source_line_cues = tuple(
                ascii_cues.ascii_word_unigram_cue_sha256_v1(token)
                for token in EXPECTED_SOURCE_LINE_TOKENS[index]
            )
            if (
                envelope.attempt != attempt
                or envelope.source != ORIGINAL_SOURCES[index]
                or envelope.message_tokens != EXPECTED_MESSAGE_TOKENS[index]
                or envelope.message_cue_sha256s != expected_message_cues
                or envelope.source_line_tokens != EXPECTED_SOURCE_LINE_TOKENS[index]
                or envelope.source_line_cue_sha256s != expected_source_line_cues
                or len(envelope.observation.cue_sha256s) != EXPECTED_MERGED_CUE_COUNTS[index]
                or envelope.observation != self.observations[index]
                or set(envelope.observation.cue_sha256s)
                != {*envelope.message_cue_sha256s, *envelope.source_line_cue_sha256s}
            ):
                raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                    "C10 message-line envelope custody changed"
                )
        if self.memories[0].record() != expected_parent.record():
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 parent-memory custody changed"
            )

        current = expected_parent
        expected_memories = [current]
        for index, exploration_root in enumerate(EXPLORATION_HYPOTHESIS_SHA256S):
            normal = current.route(self.observations[index])
            exploration = current.route(
                self.observations[index],
                exploration_hypothesis_sha256=exploration_root,
            )
            selected = self.selected_attempts[index]
            expected_outcome = unordered.seal_unordered_cue_factor_outcome_v1(
                decision=exploration,
                proposal_sha256=selected.materialization.sha256,
                execution_sha256=selected.execution.sha256,
                consequence_sha256=selected.consequence.sha256,
                feedback_sha256=self.output_feedback_sha256s[index],
                passed_cell=1,
            )
            if (
                normal != self.adaptation_normal_decisions[index]
                or normal.selected_hypothesis_sha256 != EXPECTED_NORMAL_TRAIN_SELECTIONS[index]
                or exploration != self.exploration_decisions[index]
                or exploration.selected_hypothesis_sha256 != exploration_root
                or selected.decision_sha256 != exploration.sha256
                or expected_outcome != self.adaptation_outcomes[index]
            ):
                raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                    "C10 adaptation transcript changed"
                )
            current = unordered.advance_unordered_cue_factor_memory_v1(
                current,
                expected_outcome,
            )
            expected_memories.append(current)
        if tuple(item.record() for item in self.memories) != tuple(
            item.record() for item in expected_memories
        ):
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 checkpoint lineage changed"
            )

        evaluation_observations = self.observations[TRAIN_COUNT:]
        parent_decisions = tuple(
            expected_memories[0].route(item) for item in evaluation_observations
        )
        diagonal_decisions = tuple(
            expected_memories[2].route(item) for item in evaluation_observations
        )
        three_lesson_decisions = tuple(
            expected_memories[3].route(item) for item in evaluation_observations
        )
        evaluations = tuple(expected_memories[4].route(item) for item in evaluation_observations)
        controls = _control_decisions(expected_memories[4], self.message_line_envelopes)
        read_only_decisions = (
            *parent_decisions,
            *diagonal_decisions,
            *three_lesson_decisions,
            *self.adaptation_normal_decisions,
            *controls,
        )
        executed_decision_sha256s = {item.decision_sha256 for item in self.selected_attempts}
        if (
            self.parent_evaluation_decisions != parent_decisions
            or self.diagonal_evaluation_decisions != diagonal_decisions
            or self.three_lesson_evaluation_decisions != three_lesson_decisions
            or self.evaluation_decisions != evaluations
            or self.control_decisions != controls
            or any(item.selected_hypothesis_sha256 is not None for item in parent_decisions)
            or tuple(item.selected_hypothesis_sha256 for item in diagonal_decisions)
            != EXPECTED_DIAGONAL_EVALUATION_SELECTIONS
            or tuple(item.selected_hypothesis_sha256 for item in three_lesson_decisions)
            != EXPECTED_THREE_LESSON_EVALUATION_SELECTIONS
            or tuple(item.selected_hypothesis_sha256 for item in evaluations)
            != EVALUATION_HYPOTHESIS_SHA256S
            or any(item.selected_hypothesis_sha256 is not None for item in controls)
            or any(item.sha256 in executed_decision_sha256s for item in read_only_decisions)
            or tuple(tuple(sorted(_decision_scores(item).items())) for item in diagonal_decisions)
            != EXPECTED_DIAGONAL_SCORE_MAPS
            or tuple(tuple(sorted(_decision_scores(item).items())) for item in evaluations)
            != EXPECTED_FINAL_SCORE_MAPS
            or tuple(
                tuple(sorted(_decision_scores(item).items())) for item in three_lesson_decisions
            )
            != EXPECTED_THREE_LESSON_SCORE_MAPS
        ):
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 incomplete-grid, final selection, or causal-control behavior changed"
            )

        merged_token_inventories = tuple(
            tuple(sorted({*item.message_tokens, *item.source_line_tokens}))
            for item in self.message_line_envelopes
        )
        training_tokens = {
            token for inventory in merged_token_inventories[:TRAIN_COUNT] for token in inventory
        }
        actual_overlaps = tuple(
            tuple(sorted(training_tokens.intersection(inventory)))
            for inventory in merged_token_inventories[TRAIN_COUNT:]
        )
        if actual_overlaps != EXPECTED_EVALUATION_SUPPORTED_CUE_INTERSECTIONS:
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 TRAIN/evaluation message-line overlap changed"
            )
        _authenticate_binding_evidence(
            self.final_memory,
            self.observations,
            self.message_line_envelopes,
        )
        _authenticate_unordered_and_context_controls(
            self.final_memory,
            self.message_line_envelopes,
        )

        expected_oracle_sha256 = _digest(
            _OUTPUT_FEEDBACK_NAMESPACE,
            {
                "spec_sha256": self.spec.sha256,
                "role": EVALUATION_ORACLE_ROLE,
                "execution_sha256s": tuple(
                    item.execution.sha256 for item in self.selected_attempts[TRAIN_COUNT:]
                ),
                "expected_outputs": EXPECTED_OUTPUTS[TRAIN_COUNT:],
            },
        )
        expected_evaluation_feedback_roots = tuple(
            _digest(
                _OUTPUT_FEEDBACK_NAMESPACE,
                {
                    "evaluation_oracle_sha256": expected_oracle_sha256,
                    "ordinal": ordinal,
                },
            )
            for ordinal in range(TRAIN_COUNT + 1, len(ROLES) + 1)
        )
        if (
            self.evaluation_oracle_sha256 != expected_oracle_sha256
            or self.output_feedback_sha256s[TRAIN_COUNT:] != expected_evaluation_feedback_roots
            or tuple(item.consequence.expected_outputs for item in self.selected_attempts)
            != EXPECTED_OUTPUTS
            or tuple(
                item.materialization.materialized_source_sha256 for item in self.selected_attempts
            )
            != EXPECTED_MATERIALIZED_SOURCE_SHA256S
            or self.evaluation_checkpoint_before_sha256 != self.final_memory.checkpoint_sha256
            or self.evaluation_checkpoint_after_sha256 != self.evaluation_checkpoint_before_sha256
            or self.sha256 != _digest(_RESULT_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 result root, materialized source, oracle, or evaluation immutability changed"
            )


DiagnosticCompileCallback = Callable[
    [str, str],
    tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes],
]
ExecutionCallback = Callable[
    [
        authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1,
        unordered.UnorderedCueFactorDecisionV1,
        tuple[tuple[int, ...], ...],
    ],
    RustSelectedActionExecutionV2,
]
OutputFeedbackCallback = Callable[[str], object]


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
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 public output feedback shape changed"
        )
    return outputs


def _validate_evaluation_oracle(
    value: object,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if type(value) is not tuple or len(value) != 2:
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 evaluation oracle must open both results together"
        )
    return tuple(
        _validate_outputs(outputs, batch=BATCHES[index])
        for index, outputs in enumerate(value, start=TRAIN_COUNT)
    )


def _coded_parent_error_records(stderr: bytes) -> tuple[dict[str, object], ...]:
    try:
        rows = tuple(json.loads(line) for line in stderr.decode("utf-8").splitlines() if line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 rustc stderr is not exact JSON-lines UTF-8"
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
        raise TypeError("C10 diagnostic compiler must return one exact attempt")
    if type(stderr) is not bytes:
        raise TypeError("C10 diagnostic compiler must retain exact stderr bytes")
    attempt.authenticate()
    source = ORIGINAL_SOURCES[index]
    source_bytes = source.encode("ascii")
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
        or parent_records[0].get("children") != []
    ):
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 diagnostic attempt or parent-message fixture changed"
        )


def _compile_diagnostic_once(
    identity: rust_executor.RustcIdentityV1,
    source: str,
) -> tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes]:
    identity.authenticate_structure()
    source_bytes = source.encode("ascii")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory(
        prefix="humanbrain-c10-source-context-diagnostic-"
    ) as directory:
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
    prepared: authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1,
    decision: unordered.UnorderedCueFactorDecisionV1,
    batch: tuple[tuple[int, ...], ...],
) -> RustSelectedActionExecutionV2:
    return prepared.execute_selected(decision, batch)


def _prepare_episode(
    *,
    index: int,
    spec: C10DiagnosticSourceContextCueLearningSpecV1,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    stderr: bytes,
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
) -> authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1:
    return authority.prepare_rustc_message_line_unordered_cue_text_edit_authority_v1(
        diagnostic_attempt=attempt,
        stderr=stderr,
        source=ORIGINAL_SOURCES[index],
        selection_context_sha256=_context_sha256(
            spec_sha256=spec.sha256,
            role=ROLES[index],
            source_sha256=ORIGINAL_SOURCE_SHA256S[index],
        ),
        expected_checkpoint_sha256=memory.checkpoint_sha256,
        expected_factorized_checkpoint_sha256=memory.factorized_checkpoint_sha256,
        allow_exploration=ALLOW_EXPLORATION_BY_EPISODE[index],
        target_bindings=TARGET_BINDINGS_BY_EPISODE[index],
    )


@dataclass(frozen=True, slots=True)
class _SelectedGrantV1:
    prepared: authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1
    decision: unordered.UnorderedCueFactorDecisionV1
    pre_execute_work: tuple[int, int, int, int]
    post_execute_work: tuple[int, int, int, int]
    materialization: authority.UnorderedCueFactorTextEditMaterializationV1
    release: authority.UnorderedCueFactorTextEditReleaseV1
    execution: RustSelectedActionExecutionV2


def _execute_selected_grant(
    *,
    ordinal: int,
    reservation: C10RunReservationV1,
    prepared: authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1,
    decision: unordered.UnorderedCueFactorDecisionV1,
    batch: tuple[tuple[int, ...], ...],
    execute: ExecutionCallback,
) -> _SelectedGrantV1:
    if decision.selected_hypothesis_sha256 is None:
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 STOP decision cannot materialize source"
        )
    pre_execute_work = _authority_work_counts(prepared)
    if pre_execute_work != (0, 0, 0, 0):
        raise _behavior_failure(
            "C10 source materialized before selected execution",
            stage=ROLES[ordinal - 1],
            decision=decision.record(),
            authority_receipt=prepared.receipt.record(),
            pre_execute_work=pre_execute_work,
        )
    execution = execute(prepared, decision, batch)
    if type(execution) is not RustSelectedActionExecutionV2:
        raise TypeError("C10 executor must return one exact RustSelectedActionExecutionV2")
    execution.authenticate_structure()
    materialization = prepared.materialization
    release = prepared.release
    post_execute_work = _authority_work_counts(prepared)
    if materialization is None or release is None or post_execute_work != (1, 1, 1, 1):
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 selected execution lacks exact authority work custody"
        )
    selected_source = execution.selected_source
    if (
        execution.rustc_identity.sha256 != reservation.rustc_identity_sha256
        or selected_source.authority_sha256 != prepared.authority_sha256
        or selected_source.selection_sha256 != decision.sha256
        or selected_source.release_sha256 != release.sha256
        or selected_source.source_sha256 != release.source_sha256
        or selected_source.source_sha256 != EXPECTED_MATERIALIZED_SOURCE_SHA256S[ordinal - 1]
        or execution.compile_receipt.compile_invocations != 1
        or execution.run_receipt.run_invocations != 1
        or len(execution.outputs) != len(batch)
    ):
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 selected V2 execution custody changed"
        )
    return _SelectedGrantV1(
        prepared=prepared,
        decision=decision,
        pre_execute_work=pre_execute_work,
        post_execute_work=post_execute_work,
        materialization=materialization,
        release=release,
        execution=execution,
    )


def _seal_selected_attempt(
    *,
    ordinal: int,
    grant: _SelectedGrantV1,
    expected_outputs: tuple[tuple[int, ...], ...],
    output_feedback_sha256: str,
) -> C10SelectedExecutionAttemptV1:
    selected_hypothesis_sha256 = grant.decision.selected_hypothesis_sha256
    if selected_hypothesis_sha256 is None:
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 selected grant lost its hypothesis"
        )
    consequence = _seal_consequence(
        execution=grant.execution,
        expected_outputs=expected_outputs,
        output_feedback_sha256=output_feedback_sha256,
    )
    payload = {
        "ordinal": ordinal,
        "role": ROLES[ordinal - 1],
        "decision_sha256": grant.decision.sha256,
        "selected_hypothesis_sha256": selected_hypothesis_sha256,
        "authority_receipt": grant.prepared.receipt,
        "message_line_envelope_sha256": grant.prepared.message_line_envelope.sha256,
        "pre_execute_work": grant.pre_execute_work,
        "post_execute_work": grant.post_execute_work,
        "materialization": grant.materialization,
        "release": grant.release,
        "execution": grant.execution,
        "consequence": consequence,
    }
    partial = C10SelectedExecutionAttemptV1(**payload, sha256="")
    attempt = replace(
        partial,
        sha256=_digest(_ATTEMPT_NAMESPACE, partial.record(include_sha256=False)),
    )
    attempt.authenticate()
    if consequence.passed_cell != 1:
        raise _behavior_failure(
            "C10 selected materialization failed public output feedback",
            stage=ROLES[ordinal - 1],
            decision=grant.decision.record(),
            selected_attempt=attempt.record(),
        )
    return attempt


def _run_selected_training_attempt(
    *,
    ordinal: int,
    spec: C10DiagnosticSourceContextCueLearningSpecV1,
    reservation: C10RunReservationV1,
    prepared: authority.PreparedRustcMessageLineUnorderedCueTextEditAuthorityV1,
    decision: unordered.UnorderedCueFactorDecisionV1,
    batch: tuple[tuple[int, ...], ...],
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> tuple[C10SelectedExecutionAttemptV1, str]:
    grant = _execute_selected_grant(
        ordinal=ordinal,
        reservation=reservation,
        prepared=prepared,
        decision=decision,
        batch=batch,
        execute=execute,
    )
    role = ROLES[ordinal - 1]
    expected_outputs = _validate_outputs(output_feedback(role), batch=batch)
    output_feedback_sha256 = _digest(
        _OUTPUT_FEEDBACK_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "ordinal": ordinal,
            "role": role,
            "execution_sha256": grant.execution.sha256,
            "expected_outputs": expected_outputs,
        },
    )
    return (
        _seal_selected_attempt(
            ordinal=ordinal,
            grant=grant,
            expected_outputs=expected_outputs,
            output_feedback_sha256=output_feedback_sha256,
        ),
        output_feedback_sha256,
    )


def _run_reserved_cohort_v1(
    *,
    spec: C10DiagnosticSourceContextCueLearningSpecV1,
    reservation: C10RunReservationV1,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> C10DiagnosticSourceContextCueLearningResultV1:
    spec.authenticate()
    reservation.authenticate(destination=Path(reservation.destination))
    if reservation.spec_sha256 != spec.sha256:
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 reservation/spec custody changed"
        )

    parent = unordered.start_unordered_cue_factor_memory_v1(HYPOTHESES)
    compiled = tuple(
        diagnostic_compile(reservation.rustc_identity_sha256, source) for source in ORIGINAL_SOURCES
    )
    if any(type(item) is not tuple or len(item) != 2 for item in compiled):
        raise TypeError("C10 diagnostic compiler must return exact (attempt, stderr) tuples")
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
        )
        for index, (attempt, stderr) in enumerate(
            zip(diagnostic_attempts, diagnostic_stderr, strict=True)
        )
    )
    message_line_envelopes = tuple(item.message_line_envelope for item in parent_authorities)
    observations = tuple(item.observation for item in message_line_envelopes)
    if (
        tuple(item.message_tokens for item in message_line_envelopes) != EXPECTED_MESSAGE_TOKENS
        or tuple(item.source_line_tokens for item in message_line_envelopes)
        != EXPECTED_SOURCE_LINE_TOKENS
        or tuple(len(item.observation.cue_sha256s) for item in message_line_envelopes)
        != EXPECTED_MERGED_CUE_COUNTS
    ):
        raise _behavior_failure(
            "C10 parent-message or source-line token inventories changed",
            message_line_envelopes=tuple(item.record() for item in message_line_envelopes),
        )

    evaluation_observations = observations[TRAIN_COUNT:]
    parent_evaluation_decisions = tuple(parent.route(item) for item in evaluation_observations)
    if any(item.selected_hypothesis_sha256 is not None for item in parent_evaluation_decisions):
        raise _behavior_failure(
            "C10 parent checkpoint did not STOP on fresh evaluations",
            decisions=tuple(item.record() for item in parent_evaluation_decisions),
        )

    memories = [parent]
    normal_decisions: list[unordered.UnorderedCueFactorDecisionV1] = []
    exploration_decisions: list[unordered.UnorderedCueFactorDecisionV1] = []
    outcomes: list[unordered.UnorderedCueFactorOutcomeV1] = []
    selected_attempts: list[C10SelectedExecutionAttemptV1] = []
    output_feedback_sha256s: list[str] = []
    diagonal_evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...] = ()
    three_lesson_evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...] = ()

    for index, exploration_root in enumerate(EXPLORATION_HYPOTHESIS_SHA256S):
        current = memories[-1]
        prepared = (
            parent_authorities[index]
            if current is parent
            else _prepare_episode(
                index=index,
                spec=spec,
                attempt=diagnostic_attempts[index],
                stderr=diagnostic_stderr[index],
                memory=current,
            )
        )
        observation = prepared.projected_observation
        if observation != observations[index]:
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 authority projection changed across checkpoints"
            )
        normal = current.route(observation)
        exploration = current.route(
            observation,
            exploration_hypothesis_sha256=exploration_root,
        )
        if (
            normal.selected_hypothesis_sha256 != EXPECTED_NORMAL_TRAIN_SELECTIONS[index]
            or exploration.selected_hypothesis_sha256 != exploration_root
        ):
            raise _behavior_failure(
                "C10 adaptation normal or explicit-exploration selection changed",
                stage=ROLES[index],
                memory=current.record(),
                observation=observation.record(),
                normal_decision=normal.record(),
                exploration_decision=exploration.record(),
            )
        selected, feedback_root = _run_selected_training_attempt(
            ordinal=index + 1,
            spec=spec,
            reservation=reservation,
            prepared=prepared,
            decision=exploration,
            batch=BATCHES[index],
            execute=execute,
            output_feedback=output_feedback,
        )
        outcome = unordered.seal_unordered_cue_factor_outcome_v1(
            decision=exploration,
            proposal_sha256=selected.materialization.sha256,
            execution_sha256=selected.execution.sha256,
            consequence_sha256=selected.consequence.sha256,
            feedback_sha256=feedback_root,
            passed_cell=1,
        )
        if (
            unordered.advance_unordered_cue_factor_memory_v1(
                current,
                outcome,
                plasticity_gate=False,
            )
            is not current
            or unordered.advance_unordered_cue_factor_memory_v1(
                current,
                outcome,
                consequence_gate=False,
            )
            is not current
        ):
            raise _behavior_failure(
                "C10 adaptation learning-gate control changed",
                stage=ROLES[index],
                outcome=outcome.record(),
                memory=current.record(),
            )
        normal_decisions.append(normal)
        exploration_decisions.append(exploration)
        selected_attempts.append(selected)
        output_feedback_sha256s.append(feedback_root)
        outcomes.append(outcome)
        memories.append(unordered.advance_unordered_cue_factor_memory_v1(current, outcome))
        if index == 1:
            diagonal_evaluation_decisions = tuple(
                memories[-1].route(item) for item in evaluation_observations
            )
            if (
                tuple(item.selected_hypothesis_sha256 for item in diagonal_evaluation_decisions)
                != EXPECTED_DIAGONAL_EVALUATION_SELECTIONS
            ):
                raise _behavior_failure(
                    "C10 diagonal checkpoint lost wrong same-class negative telemetry",
                    decisions=tuple(item.record() for item in diagonal_evaluation_decisions),
                )
        elif index == 2:
            three_lesson_evaluation_decisions = tuple(
                memories[-1].route(item) for item in evaluation_observations
            )
            if (
                tuple(item.selected_hypothesis_sha256 for item in three_lesson_evaluation_decisions)
                != EXPECTED_THREE_LESSON_EVALUATION_SELECTIONS
                or tuple(
                    tuple(sorted(_decision_scores(item).items()))
                    for item in three_lesson_evaluation_decisions
                )
                != EXPECTED_THREE_LESSON_SCORE_MAPS
            ):
                raise _behavior_failure(
                    "C10 three-lesson checkpoint lost missing-class negative telemetry",
                    decisions=tuple(item.record() for item in three_lesson_evaluation_decisions),
                )

    final = memories[-1]
    evaluation_checkpoint_before_sha256 = final.checkpoint_sha256
    evaluation_decisions = tuple(final.route(item) for item in evaluation_observations)
    control_decisions = _control_decisions(final, message_line_envelopes)
    if (
        tuple(item.selected_hypothesis_sha256 for item in evaluation_decisions)
        != EVALUATION_HYPOTHESIS_SHA256S
        or tuple(tuple(sorted(_decision_scores(item).items())) for item in evaluation_decisions)
        != EXPECTED_FINAL_SCORE_MAPS
        or any(item.selected_hypothesis_sha256 is not None for item in control_decisions)
    ):
        raise _behavior_failure(
            "C10 final message-line routing or causal-control behavior changed",
            evaluation_decisions=tuple(item.record() for item in evaluation_decisions),
            control_decisions=tuple(item.record() for item in control_decisions),
        )
    _authenticate_binding_evidence(final, observations, message_line_envelopes)
    _authenticate_unordered_and_context_controls(final, message_line_envelopes)

    evaluation_grants: list[_SelectedGrantV1] = []
    for index, decision in enumerate(evaluation_decisions, start=TRAIN_COUNT):
        prepared = _prepare_episode(
            index=index,
            spec=spec,
            attempt=diagnostic_attempts[index],
            stderr=diagnostic_stderr[index],
            memory=final,
        )
        if prepared.projected_observation != observations[index]:
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 evaluation authority projection changed"
            )
        evaluation_grants.append(
            _execute_selected_grant(
                ordinal=index + 1,
                reservation=reservation,
                prepared=prepared,
                decision=decision,
                batch=BATCHES[index],
                execute=execute,
            )
        )

    evaluation_expected_outputs = _validate_evaluation_oracle(
        output_feedback(EVALUATION_ORACLE_ROLE)
    )
    evaluation_oracle_sha256 = _digest(
        _OUTPUT_FEEDBACK_NAMESPACE,
        {
            "spec_sha256": spec.sha256,
            "role": EVALUATION_ORACLE_ROLE,
            "execution_sha256s": tuple(item.execution.sha256 for item in evaluation_grants),
            "expected_outputs": evaluation_expected_outputs,
        },
    )
    for index, (grant, expected_outputs) in enumerate(
        zip(evaluation_grants, evaluation_expected_outputs, strict=True),
        start=TRAIN_COUNT,
    ):
        feedback_root = _digest(
            _OUTPUT_FEEDBACK_NAMESPACE,
            {
                "evaluation_oracle_sha256": evaluation_oracle_sha256,
                "ordinal": index + 1,
            },
        )
        selected_attempts.append(
            _seal_selected_attempt(
                ordinal=index + 1,
                grant=grant,
                expected_outputs=expected_outputs,
                output_feedback_sha256=feedback_root,
            )
        )
        output_feedback_sha256s.append(feedback_root)
    evaluation_checkpoint_after_sha256 = final.checkpoint_sha256

    payload = {
        "status": STATUS_PASS,
        "spec": spec,
        "reservation": reservation,
        "diagnostic_attempts": diagnostic_attempts,
        "message_line_envelopes": message_line_envelopes,
        "observations": observations,
        "memories": tuple(memories),
        "adaptation_normal_decisions": tuple(normal_decisions),
        "exploration_decisions": tuple(exploration_decisions),
        "adaptation_outcomes": tuple(outcomes),
        "parent_evaluation_decisions": parent_evaluation_decisions,
        "diagonal_evaluation_decisions": diagonal_evaluation_decisions,
        "three_lesson_evaluation_decisions": three_lesson_evaluation_decisions,
        "evaluation_decisions": evaluation_decisions,
        "control_decisions": control_decisions,
        "selected_attempts": tuple(selected_attempts),
        "output_feedback_sha256s": tuple(output_feedback_sha256s),
        "evaluation_oracle_sha256": evaluation_oracle_sha256,
        "evaluation_checkpoint_before_sha256": evaluation_checkpoint_before_sha256,
        "evaluation_checkpoint_after_sha256": evaluation_checkpoint_after_sha256,
        "work": _seal_work(),
    }
    partial = C10DiagnosticSourceContextCueLearningResultV1(**payload, sha256="")
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _reserve_destination(
    spec: C10DiagnosticSourceContextCueLearningSpecV1,
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
) -> C10RunReservationV1:
    spec.authenticate()
    if not _valid_commit(preregistration_commit):
        raise ValueError("C10 preregistration commit must be 40 lowercase hex characters")
    if not _valid_sha256(rustc_identity_sha256):
        raise ValueError("C10 rustc identity root is invalid")
    destination = Path(destination).resolve()
    reservation_path = Path(reservation_path).resolve()
    expected_reservation = destination.with_name(f"{destination.name}.reservation.json")
    if reservation_path != expected_reservation:
        raise ValueError("C10 reservation must be the result's exact sibling")
    if os.path.lexists(destination) or os.path.lexists(reservation_path):
        raise FileExistsError(destination if os.path.lexists(destination) else reservation_path)
    partial = C10RunReservationV1(
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
            raise OSError("C10 reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if reservation_path.read_bytes() != payload:
        raise OSError("C10 reservation readback changed")
    return reservation


def _result_document(
    result: C10DiagnosticSourceContextCueLearningResultV1,
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
    }
    return {**body, "document_sha256": _digest(_DOCUMENT_NAMESPACE, body)}


def _abort_document(
    *,
    spec: C10DiagnosticSourceContextCueLearningSpecV1,
    reservation: C10RunReservationV1,
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
    spec: C10DiagnosticSourceContextCueLearningSpecV1,
    reservation: C10RunReservationV1,
    error: C10BehaviorFailure,
) -> dict[str, object]:
    partial_transcript = _plain(error.partial_transcript)
    if type(partial_transcript) is not dict:
        raise TypeError("C10 behavior failure transcript must be one exact object")
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
            raise OSError("C10 terminal document write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = destination.read_bytes()
    if retained != payload or _canonical_json(json.loads(retained.decode("ascii"))) + b"\n" != (
        payload
    ):
        raise OSError("C10 terminal document readback changed")
    decoded = json.loads(retained.decode("ascii"))
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if document_sha256 != _digest(_DOCUMENT_NAMESPACE, body):
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 terminal document root changed"
        )
    if decoded["terminal_status"] == STATUS_PASS:
        record = dict(decoded["authenticated_result_record"])
        result_sha256 = record.pop("sha256", None)
        if result_sha256 != decoded["result_sha256"] or result_sha256 != _digest(
            _RESULT_NAMESPACE,
            record,
        ):
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 persisted result root changed"
            )
    elif decoded["terminal_status"] == STATUS_BEHAVIOR_FAIL:
        partial = decoded.get("partial_transcript")
        if type(partial) is not dict or decoded.get("partial_transcript_sha256") != _digest(
            _FAILURE_NAMESPACE,
            partial,
        ):
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 persisted behavior-failure transcript root changed"
            )
    elif decoded["terminal_status"] != STATUS_EXECUTION_ABORT:
        raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
            "C10 terminal status changed"
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
    spec: C10DiagnosticSourceContextCueLearningSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C10DiagnosticSourceContextCueLearningResultV1:
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
    except C10BehaviorFailure as error:
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
    spec: C10DiagnosticSourceContextCueLearningSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C10DiagnosticSourceContextCueLearningResultV1:
    """Preflight, reserve, and spend the exact production C10 cohort once."""

    identity = _preflight_rustc_v1()
    destination, reservation_path = _production_evidence_paths_v1()

    def diagnostic_compile(
        rustc_identity_sha256: str,
        source: str,
    ) -> tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes]:
        if rustc_identity_sha256 != identity.sha256:
            raise ContinualRustDiagnosticSourceContextCueLearningCohortV1Error(
                "C10 diagnostic compiler identity changed after reservation"
            )
        return _compile_diagnostic_once(identity, source)

    expected_by_role = dict(zip(ROLES, EXPECTED_OUTPUTS, strict=True))

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
    "C10BehaviorFailure",
    "C10DiagnosticSourceContextCueLearningResultV1",
    "C10DiagnosticSourceContextCueLearningSpecV1",
    "C10PublicConsequenceV1",
    "C10RunReservationV1",
    "C10SelectedExecutionAttemptV1",
    "C10WorkV1",
    "ContinualRustDiagnosticSourceContextCueLearningCohortV1Error",
    "make_c10_diagnostic_source_context_cue_learning_spec_v1",
    "output_sha256_v1",
    "run_and_write_cohort_v1",
]


if __name__ == "__main__":
    main()
