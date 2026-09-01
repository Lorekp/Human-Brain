"""Fresh C8 rustc-diagnostic learning from one unordered cue set.

Four public TRAIN consequences cover the complete 2x2 operator/target grid.
The learner receives the diagnostic and source-region cues as one unordered set,
must remain ambiguous after the two diagonals and after the missing fourth cell,
then must reuse the learned bindings on two fresh sources.  The edit authority
materializes only the selected composite action.
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
from experiments.r3_pilot import common_factorized_text_edit_operator_catalogue_v1 as catalogue
from experiments.r3_pilot import common_rustc_factorized_feedback_adapter_v1 as rustc_feedback
from experiments.r3_pilot import common_unordered_cue_factor_memory_v1 as unordered
from experiments.r3_pilot import common_unordered_cue_factor_text_edit_authority_v1 as authority
from humanbrain import rust_selected_action_executor as rust_executor
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

SCHEMA = "humanbrain.continual-rust-diagnostic-unordered-cue-binding-cohort.v1"
RESULT_DOCUMENT_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-unordered-cue-binding-result-document.v1"
)
RESERVATION_SCHEMA = "humanbrain.continual-rust-diagnostic-unordered-cue-binding-reservation.v1"
STATUS_PASS = "PASS_C8_RUSTC_DIAGNOSTIC_UNORDERED_CUE_BINDING_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C8_RUSTC_DIAGNOSTIC_UNORDERED_CUE_BINDING_V1"
STATUS_EXECUTION_ABORT = "ABORT_C8_RUSTC_DIAGNOSTIC_UNORDERED_CUE_BINDING_V1"
CLAIM_BOUNDARY = (
    "one-fresh-public-full-grid-cohort-where-four-caller-attested-successful-"
    "rustc-consequences-associate-one-unordered-two-cue-set-with-two-components-"
    "of-one-supplied-2x2-concrete-action-target-bank-and-enable-normal-selection-"
    "on-two-fresh-sources-before-selected-only-materialization-with-no-evaluation-"
    "update-not-cue-discovery-action-or-target-discovery-edit-semantics-"
    "localization-diagnosis-source-synthesis-compositional-generalization-"
    "transfer-debugging-swe-or-intelligence"
)

FROZEN_MECHANISM_COMMIT = "2eae160c645b1e5fb0be996b7d1d4bac7c862508"
FROZEN_MECHANISM_FILE_SHA256S = (
    (
        "experiments/r3_pilot/common_factorized_feedback_hypothesis_memory_v1.py",
        "f17ef6861a160643216c955d170827d746fa58fad391b76d4fbdfa6b183377df",
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
        "experiments/r3_pilot/common_rustc_unordered_cue_factor_adapter_v1.py",
        "f4d57360d2adb757a800159d7e89201db98fdae42d1d0384dbe36eaec3366adc",
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
        "experiments/r3_pilot/common_unordered_cue_factor_text_edit_authority_v1.py",
        "4bd9f95fab8b37b78458bafd611fe2a5fcc1ce1dfe21fa5a3f4fbeb0d1dea6a1",
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
    "c8_unordered_binding",
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
    "train-h00-e0425-first",
    "train-h11-e0308-second",
    "train-h01-e0425-second",
    "train-h10-e0308-first",
    "evaluation-h01-e0425-second",
    "evaluation-h10-e0308-first",
)
EVALUATION_ORACLE_ROLE = "evaluation-oracle-after-both-executions"
TRAIN_COUNT = 4
EXPECTED_DIAGNOSTICS = (
    ("E0425", rustc_feedback.FIRST_HALF),
    ("E0308", rustc_feedback.SECOND_HALF),
    ("E0425", rustc_feedback.SECOND_HALF),
    ("E0308", rustc_feedback.FIRST_HALF),
    ("E0425", rustc_feedback.SECOND_HALF),
    ("E0308", rustc_feedback.FIRST_HALF),
)
EXPECTED_PRIMARY_BYTE_RANGES = (
    (200, 210),
    (357, 368),
    (333, 343),
    (222, 233),
    (334, 344),
    (221, 232),
)
ALLOW_EXPLORATION_BY_EPISODE = (True, True, True, True, False, False)
BATCHES = (
    ((-31, -4, 9),),
    ((-14, 1, 30),),
    ((-18, 2, 25),),
    ((-25, -6, 12),),
    ((-40, -3, 6),),
    ((-32, 4, 15),),
)
EXPECTED_OUTPUTS = (
    ((-1, 26, 39),),
    ((7, 22, 51),),
    ((8, 28, 51),),
    ((-3, 16, 34),),
    ((-6, 31, 40),),
    ((-5, 31, 42),),
)

TRAIN_H00_SOURCE = """\
struct C8T00Box {
    c8_t00_gap: i64,
}

pub fn solve(samples: &[i64]) -> Vec<i64> {
    let c8_t00_first_anchor = 13_i64;
    let c8_t00_second_anchor = 17_i64;
    let first_bias: i64 = C8T00Box { c8_t00_gap }.c8_t00_gap;
    // C8_T00_REGION_DIVIDER
    let second_bias: i64 = c8_t00_second_anchor;
    samples
        .iter()
        .copied()
        .map(|value| {
            value
                + first_bias
                + second_bias
                + (c8_t00_first_anchor - c8_t00_first_anchor)
                + (c8_t00_second_anchor - c8_t00_second_anchor)
        })
        .collect()
}
"""

TRAIN_H11_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let c8_t11_text = "obsidian";
    let c8_t11_first_anchor = 5_i64;
    let c8_t11_second_anchor = 16_i64;
    let first_bias: i64 = c8_t11_first_anchor;
    // C8_T11_REGION_DIVIDER
    let _c8_t11_prefix_probe = samples.first().copied();
    let _c8_t11_width = c8_t11_text.len();
    let second_bias: i64 = c8_t11_text;
    samples
        .iter()
        .copied()
        .map(|value| {
            value
                + first_bias
                + second_bias
                + (c8_t11_first_anchor - c8_t11_first_anchor)
                + (c8_t11_second_anchor - c8_t11_second_anchor)
        })
        .collect()
}
"""

TRAIN_H01_SOURCE = """\
struct C8T01Box {
    c8_t01_gap: i64,
}

pub fn solve(samples: &[i64]) -> Vec<i64> {
    let c8_t01_first_anchor = 7_i64;
    let c8_t01_second_anchor = 19_i64;
    let first_bias: i64 = c8_t01_first_anchor;
    // C8_T01_REGION_DIVIDER
    let _c8_t01_prefix_probe = samples.first().copied();
    let second_bias: i64 = C8T01Box { c8_t01_gap }.c8_t01_gap;
    samples
        .iter()
        .copied()
        .map(|value| {
            value
                + first_bias
                + second_bias
                + (c8_t01_first_anchor - c8_t01_first_anchor)
                + (c8_t01_second_anchor - c8_t01_second_anchor)
        })
        .collect()
}
"""

TRAIN_H10_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let c8_t10_text = "granite";
    let c8_t10_first_anchor = 8_i64;
    let c8_t10_second_anchor = 14_i64;
    let _c8_t10_width = c8_t10_text.len();
    let first_bias: i64 = c8_t10_text;
    // C8_T10_REGION_DIVIDER
    let second_bias: i64 = c8_t10_second_anchor;
    samples
        .iter()
        .copied()
        .map(|value| {
            value
                + first_bias
                + second_bias
                + (c8_t10_first_anchor - c8_t10_first_anchor)
                + (c8_t10_second_anchor - c8_t10_second_anchor)
        })
        .collect()
}
"""

EVALUATION_H01_SOURCE = """\
struct C8E01Box {
    c8_e01_gap: i64,
}

pub fn solve(samples: &[i64]) -> Vec<i64> {
    let c8_e01_first_anchor = 11_i64;
    let c8_e01_second_anchor = 23_i64;
    let first_bias: i64 = c8_e01_first_anchor;
    // C8_E01_REGION_DIVIDER
    let _c8_e01_prefix_probe = samples.first().copied();
    let second_bias: i64 = C8E01Box { c8_e01_gap }.c8_e01_gap;
    samples
        .iter()
        .copied()
        .map(|value| {
            value
                + first_bias
                + second_bias
                + (c8_e01_first_anchor - c8_e01_first_anchor)
                + (c8_e01_second_anchor - c8_e01_second_anchor)
        })
        .collect()
}
"""

EVALUATION_H10_SOURCE = """\
pub fn solve(samples: &[i64]) -> Vec<i64> {
    let c8_e10_text = "basalt";
    let c8_e10_first_anchor = 9_i64;
    let c8_e10_second_anchor = 18_i64;
    let _c8_e10_width = c8_e10_text.len();
    let first_bias: i64 = c8_e10_text;
    // C8_E10_REGION_DIVIDER
    let second_bias: i64 = c8_e10_second_anchor;
    samples
        .iter()
        .copied()
        .map(|value| {
            value
                + first_bias
                + second_bias
                + (c8_e10_first_anchor - c8_e10_first_anchor)
                + (c8_e10_second_anchor - c8_e10_second_anchor)
        })
        .collect()
}
"""

ORIGINAL_SOURCES = (
    TRAIN_H00_SOURCE,
    TRAIN_H11_SOURCE,
    TRAIN_H01_SOURCE,
    TRAIN_H10_SOURCE,
    EVALUATION_H01_SOURCE,
    EVALUATION_H10_SOURCE,
)
ORIGINAL_SOURCE_SHA256S = (
    "c199578bb5c3d6df27a533cacf04d0d14746fe12b5ae556b9c805f2d746df5d0",
    "a0c554ee679a10b37bad6fe3e7b6d6b905777d579302ab1cde2cf7f69f43b5dc",
    "077283306f7ecdd8e80fc072fa23db8902593deff7e67543c0c405a23d0009f6",
    "b037904c58d0ab527c2d9e20595a6071ad786a0748165a2684f3af54448a417f",
    "b05b9acb3804dce03571ccb1c42e707cb799ca29ff4c4dc0bcdce0fefa0344af",
    "532e90d9739ec16748ad152d53509d34de7ccbf5efccfb067aa915789210f662",
)
EXPECTED_MATERIALIZED_SOURCE_SHA256S = (
    "082120df1274f6b90bccb3c064810a37079e892560b9429cda63f68554020338",
    "3598b6e3bfdc014c579d1c85bb38748506f83115c9f61ed9080b2b850d8745fc",
    "319f315176807b6f4db650e74d0775f65af626a06d5f4caf87031dd692ee8c7a",
    "178e9ffa2a3c9faf01739db95195c5575d9da57c6835e9a4945269bdb7ee0fc8",
    "62b3a80e76db43ce2c726ae246bf487eb26ba4c40e02b7c15a2fcedee36a3e3f",
    "ef114a5c9493d7f6cc5726882217322dd7d8ccee527cd4811a01e1ed1bc3ba55",
)
TARGET_MARKERS = (
    ("c8_t00_first_anchor", "c8_t00_second_anchor"),
    ("c8_t11_first_anchor", "c8_t11_second_anchor"),
    ("c8_t01_first_anchor", "c8_t01_second_anchor"),
    ("c8_t10_first_anchor", "c8_t10_second_anchor"),
    ("c8_e01_first_anchor", "c8_e01_second_anchor"),
    ("c8_e10_first_anchor", "c8_e10_second_anchor"),
)

_SHA256_HEX = frozenset("0123456789abcdef")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-spec-v1"
_COMPONENT_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-component-v1"
_CONTEXT_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-context-v1"
_OUTPUT_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-output-v1"
_OUTPUT_FEEDBACK_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-feedback-v1"
_CONSEQUENCE_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-consequence-v1"
_ATTEMPT_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-attempt-v1"
_WORK_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-work-v1"
_RESERVATION_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-reservation-v1"
_RESULT_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-result-v1"
_FAILURE_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-failure-v1"
_DOCUMENT_NAMESPACE = b"c8-rustc-diagnostic-unordered-cue-binding-document-v1"


class ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(ValueError):
    """The frozen C8 cohort or its authenticated transcript changed."""


class C8BehaviorFailure(RuntimeError):
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
    raise TypeError(f"C8 transcript cannot serialize {type(value).__qualname__}")


def _behavior_failure(reason: str, **partial_transcript: object) -> C8BehaviorFailure:
    return C8BehaviorFailure(
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

# C7's 57 prior roots, its three original roots, and its three materialized roots.
EXCLUDED_PRIOR_SOURCE_SHA256S = tuple(
    sorted(
        (
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
            "2cb5e3555404d53084672be16652220a61e0c448ee9b9f97bd1084e4a892e2ef",
            "3a76a6783fda0b2aef342124467e2fb1245ecea140052760dabed7b10d10799b",
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
            "900e4f8cc9648f00e4835b50185f24da64d12babb35dbf5bf65935f575398362",
            "913097c4d9e858c41d80a8d671a5996340a9d3071d1b1902350bb4403a236f88",
            "9156378419e5a875a0aa88632bbada13c3261e07ca979ce1ea9afbf650025975",
            "962a5bcd733f4026a07dea5342aa5145ea837bd282edbf67d41418da0f37f6dd",
            "9634b257c00cc1f0e2470cead58cc6ffec8136fb1a2388511b8c5f17b01dd9fc",
            "967382da9e579783e7fe5cb5cbf3c4e1a92f5d8ca84948aa641d0f289f62fb8d",
            "96f90f7f7d3e6a3677ce8cd1e7ddb3cc9986a49d42180a5f74b2375a1b6ed985",
            "a30f2d85a0f2fcd28458f01e2353b20bd7567478c9f0e14ba900becdcca0da15",
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
            "cdddc8926d80232b494132e1687f0e344b92c0fdb5c667ed4a7051462ee72762",
            "d00939b3ee635b6011a6e3296550dfef6fa1fa1d3211f42514822300bb37332c",
            "d23bee65de3f003933e9739add066b9c1973bb5760abfc6f15129b9b1fe169e3",
            "d335b523ec67621ba6925c48c84f1d4cce2f38f00956099297db443d1fe80f50",
            "d34b307aa6cef6e403ee88698897f73dfdb21511aee7b1b629baf5631eaab6d6",
            "d907300fcf3390c3b8697a8437f8448cc10057736ac1bdee3c27084f2a7b0b7f",
            "e6f89d9d2be762471863ff3c6a8b00dd3eef59ad388eed55388883198385429f",
            "e8dd412b73a6612ab6a70483ef738881ec09b2d9c429638cffab630648bcb3e8",
            "ee0b4c24a158c1d1e66bfbeaa5fcdccdaef641b5bd1b161f9a73054f9915e675",
            "fbeac8ce5b168255e52c24318106a00194f7a5999c21bc453987227fddb5cb94",
            "7ea07d6438adf64d57210c35a69d88c4a48c5c8f5cc653246d00b32e3924d9e2",
            "737b90128d66b22d7f5f15e04780eaa2920c85612a9f8bb24f4f814c0afe8d70",
            "2018ff95b0b6c90abe9a754d1490b56ef6a854cd3801c4a9973ead72e57926d9",
            "61874992d6db72fcc87f6101b1a9e8deb8cf46904c4617596e04b1481b1c38d2",
            "1f0fa31aafb6d3737b0a394c4a9f31140aff3f543d35911c80ac184ba828aeb9",
            "875e1e20df98385178b3af24600bfd9bed7973b6333ebbdad65fe19b07d275aa",
        )
    )
)


def _current_mechanism_file_sha256s() -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, sha256((_REPO_ROOT / path).read_bytes()).hexdigest())
        for path, _root in FROZEN_MECHANISM_FILE_SHA256S
    )


def output_sha256_v1(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(_OUTPUT_NAMESPACE, {"outputs": outputs})


@dataclass(frozen=True, slots=True)
class C8DiagnosticUnorderedCueBindingSpecV1:
    frozen_mechanism_commit: str
    frozen_mechanism_file_sha256s: tuple[tuple[str, str], ...]
    diagnostic_arguments: tuple[str, ...]
    expected_diagnostics: tuple[tuple[str, str], ...]
    expected_primary_byte_ranges: tuple[tuple[int, int], ...]
    allow_exploration_by_episode: tuple[bool, ...]
    batches: tuple[tuple[tuple[int, ...], ...], ...]
    expected_output_sha256s: tuple[str, ...]
    original_source_sha256s: tuple[str, ...]
    operator_bindings: tuple[tuple[str, str], ...]
    target_bindings_by_episode: tuple[tuple[tuple[str, str], ...], ...]
    hypothesis_sha256s: tuple[str, ...]
    exploration_hypothesis_sha256s: tuple[str, ...]
    evaluation_hypothesis_sha256s: tuple[str, ...]
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
            sha256(source.encode("ascii")).hexdigest() for source in ORIGINAL_SOURCES
        )
        expected_outputs = tuple(output_sha256_v1(value) for value in EXPECTED_OUTPUTS)
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
            or self.expected_output_sha256s != expected_outputs
            or self.original_source_sha256s != ORIGINAL_SOURCE_SHA256S
            or self.original_source_sha256s != derived_sources
            or len(set(derived_sources)) != 6
            or self.operator_bindings != OPERATOR_BINDINGS
            or self.target_bindings_by_episode != TARGET_BINDINGS_BY_EPISODE
            or len(self.target_bindings_by_episode) != 6
            or self.hypothesis_sha256s != tuple(value.sha256 for value in HYPOTHESES)
            or len(set(self.hypothesis_sha256s)) != 4
            or self.exploration_hypothesis_sha256s != EXPLORATION_HYPOTHESIS_SHA256S
            or self.evaluation_hypothesis_sha256s != EVALUATION_HYPOTHESIS_SHA256S
            or self.excluded_source_sha256s != EXCLUDED_PRIOR_SOURCE_SHA256S
            or len(self.excluded_source_sha256s) != 63
            or len(set(self.excluded_source_sha256s)) != 63
            or set(derived_sources).intersection(self.excluded_source_sha256s)
            or set(EXPECTED_MATERIALIZED_SOURCE_SHA256S).intersection(
                {*derived_sources, *self.excluded_source_sha256s}
            )
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 public fixture specification changed"
            )


def make_c8_diagnostic_unordered_cue_binding_spec_v1() -> C8DiagnosticUnorderedCueBindingSpecV1:
    partial = C8DiagnosticUnorderedCueBindingSpecV1(
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
        evaluation_hypothesis_sha256s=EVALUATION_HYPOTHESIS_SHA256S,
        excluded_source_sha256s=EXCLUDED_PRIOR_SOURCE_SHA256S,
        sha256="",
    )
    result = replace(
        partial,
        sha256=_digest(_SPEC_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


DEFAULT_PUBLIC_SPEC = make_c8_diagnostic_unordered_cue_binding_spec_v1()


@dataclass(frozen=True, slots=True)
class C8RunReservationV1:
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
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 run reservation changed"
            )


@dataclass(frozen=True, slots=True)
class C8PublicConsequenceV1:
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
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 public consequence authentication failed"
            )


def _seal_consequence(
    *,
    execution: RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
    output_feedback_sha256: str,
) -> C8PublicConsequenceV1:
    payload = {
        "source_sha256": execution.selected_source.source_sha256,
        "batch_sha256": execution.preparation.batch_sha256,
        "output_feedback_sha256": output_feedback_sha256,
        "execution_sha256": execution.sha256,
        "observed_outputs": execution.outputs,
        "expected_outputs": expected_outputs,
        "passed_cell": int(execution.outputs == expected_outputs),
    }
    result = C8PublicConsequenceV1(
        **payload,
        sha256=_digest(_CONSEQUENCE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class C8SelectedExecutionAttemptV1:
    ordinal: int
    role: str
    decision_sha256: str
    selected_hypothesis_sha256: str
    authority_receipt: authority.UnorderedCueFactorTextEditAuthorityReceiptV1
    pre_execute_counts: tuple[int, int]
    post_grant_counts: tuple[int, int]
    materialization: authority.UnorderedCueFactorTextEditMaterializationV1
    release: authority.UnorderedCueFactorTextEditReleaseV1
    execution: RustSelectedActionExecutionV2
    consequence: C8PublicConsequenceV1
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "ordinal": self.ordinal,
            "role": self.role,
            "decision_sha256": self.decision_sha256,
            "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
            "authority_receipt": self.authority_receipt.record(),
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
        index = self.ordinal - 1
        if (
            type(self.ordinal) is not int
            or not 1 <= self.ordinal <= 6
            or self.role != ROLES[index]
            or self.selected_hypothesis_sha256
            != (*EXPLORATION_HYPOTHESIS_SHA256S, *EVALUATION_HYPOTHESIS_SHA256S)[index]
            or self.pre_execute_counts != (0, 0)
            or self.post_grant_counts != (1, 1)
            or self.authority_receipt.allow_exploration != ALLOW_EXPLORATION_BY_EPISODE[index]
            or self.authority_receipt.sha256 != self.materialization.authority_sha256
            or self.decision_sha256 != self.materialization.decision_sha256
            or self.selected_hypothesis_sha256 != self.materialization.selected_hypothesis_sha256
            or self.materialization.operator_action_sha256
            != HYPOTHESES_BY_KEY[
                ((0, 0), (1, 1), (0, 1), (1, 0), (0, 1), (1, 0))[index]
            ].component_sha256s[0]
            or self.release.authority_sha256 != self.authority_receipt.sha256
            or self.release.selection_sha256 != self.decision_sha256
            or self.release.materialization != self.materialization
            or selected_source.authority_sha256 != self.authority_receipt.sha256
            or selected_source.selection_sha256 != self.decision_sha256
            or selected_source.release_sha256 != self.release.sha256
            or selected_source.action_handle != self.release.action_handle
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
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 selected execution attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C8WorkV1:
    diagnostic_compile_count: int
    materialization_count: int
    release_count: int
    selected_source_compile_count: int
    selected_source_run_count: int
    output_feedback_open_count: int
    training_feedback_open_count: int
    evaluation_oracle_open_count: int
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
            "diagnostic_compile_count": 6,
            "materialization_count": 6,
            "release_count": 6,
            "selected_source_compile_count": 6,
            "selected_source_run_count": 6,
            "output_feedback_open_count": 5,
            "training_feedback_open_count": 4,
            "evaluation_oracle_open_count": 1,
            "memory_advance_count": 4,
            "evaluation_memory_advance_count": 0,
            "alternative_materialization_count": 0,
        }
        if self.record(include_sha256=False) != expected or self.sha256 != _digest(
            _WORK_NAMESPACE, expected
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 work accounting changed"
            )


def _seal_work() -> C8WorkV1:
    payload = {
        "diagnostic_compile_count": 6,
        "materialization_count": 6,
        "release_count": 6,
        "selected_source_compile_count": 6,
        "selected_source_run_count": 6,
        "output_feedback_open_count": 5,
        "training_feedback_open_count": 4,
        "evaluation_oracle_open_count": 1,
        "memory_advance_count": 4,
        "evaluation_memory_advance_count": 0,
        "alternative_materialization_count": 0,
    }
    result = C8WorkV1(**payload, sha256=_digest(_WORK_NAMESPACE, payload))
    result.authenticate()
    return result


def _decision_scores(decision: unordered.UnorderedCueFactorDecisionV1) -> dict[str, int]:
    return {item.hypothesis.sha256: item.total_score for item in decision.hypothesis_scores}


def _expected_binding_evidence(
    observations: tuple[unordered.UnorderedCueFactorObservationV1, ...],
) -> dict[tuple[str, int, str], tuple[int, int]]:
    expected: dict[tuple[str, int, str], tuple[int, int]] = {}
    for cue in sorted({cue for item in observations[:TRAIN_COUNT] for cue in item.cue_sha256s}):
        for action_axis, components in enumerate(
            (OPERATOR_ACTION_SHA256S, TARGET_COMPONENT_SHA256S)
        ):
            for component in components:
                excitatory = sum(
                    cue in observation.cue_sha256s
                    and HYPOTHESES_BY_KEY[key].component_sha256s[action_axis] == component
                    for observation, key in zip(
                        observations[:TRAIN_COUNT],
                        ((0, 0), (1, 1), (0, 1), (1, 0)),
                        strict=True,
                    )
                )
                expected[(cue, action_axis, component)] = (excitatory, 0)
    return expected


def _authenticate_binding_evidence(
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
    observations: tuple[unordered.UnorderedCueFactorObservationV1, ...],
) -> None:
    actual = {
        (row.cue_sha256, row.action_axis, row.component_sha256): (
            row.excitatory_ticks,
            row.inhibitory_ticks,
        )
        for row in memory.binding_evidence
    }
    expected = _expected_binding_evidence(observations)
    if actual != expected or len(actual) != 16:
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 final binding evidence changed"
        )
    cue_roots = sorted({key[0] for key in actual})
    bound_axes: list[int] = []
    for cue in cue_roots:
        patterns = tuple(
            sorted(
                actual[(cue, action_axis, component)][0]
                for component in (OPERATOR_ACTION_SHA256S, TARGET_COMPONENT_SHA256S)[action_axis]
            )
            for action_axis in range(2)
        )
        if sorted(patterns) != [[0, 2], [1, 1]]:
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 cue lacks one unique and one tied axis association"
            )
        bound_axes.append(patterns.index([0, 2]))
    if sorted(bound_axes) != [0, 0, 1, 1]:
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 learned cue-axis binding balance changed"
        )


@dataclass(frozen=True, slots=True)
class C8DiagnosticUnorderedCueBindingResultV1:
    status: str
    spec: C8DiagnosticUnorderedCueBindingSpecV1
    reservation: C8RunReservationV1
    diagnostic_attempts: tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, ...]
    observations: tuple[unordered.UnorderedCueFactorObservationV1, ...]
    memories: tuple[unordered.FrozenUnorderedCueFactorMemoryV1, ...]
    adaptation_normal_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    exploration_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    adaptation_outcomes: tuple[unordered.UnorderedCueFactorOutcomeV1, ...]
    parent_cross_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    diagonal_cross_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    three_lesson_cross_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    control_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    selected_attempts: tuple[C8SelectedExecutionAttemptV1, ...]
    output_feedback_sha256s: tuple[str, ...]
    evaluation_oracle_sha256: str
    work: C8WorkV1
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
            "observations": tuple(item.record() for item in self.observations),
            "memories": tuple(item.record() for item in self.memories),
            "adaptation_normal_decisions": tuple(
                item.record() for item in self.adaptation_normal_decisions
            ),
            "exploration_decisions": tuple(item.record() for item in self.exploration_decisions),
            "adaptation_outcomes": tuple(item.record() for item in self.adaptation_outcomes),
            "parent_cross_decisions": tuple(item.record() for item in self.parent_cross_decisions),
            "diagonal_cross_decisions": tuple(
                item.record() for item in self.diagonal_cross_decisions
            ),
            "three_lesson_cross_decisions": tuple(
                item.record() for item in self.three_lesson_cross_decisions
            ),
            "evaluation_decisions": tuple(item.record() for item in self.evaluation_decisions),
            "control_decisions": tuple(item.record() for item in self.control_decisions),
            "selected_attempts": tuple(item.record() for item in self.selected_attempts),
            "output_feedback_sha256s": self.output_feedback_sha256s,
            "evaluation_oracle_sha256": self.evaluation_oracle_sha256,
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
                    self.observations,
                    self.memories,
                    self.adaptation_normal_decisions,
                    self.exploration_decisions,
                    self.adaptation_outcomes,
                    self.parent_cross_decisions,
                    self.diagonal_cross_decisions,
                    self.three_lesson_cross_decisions,
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
            or cardinalities != (6, 6, 5, 4, 4, 4, 2, 2, 2, 2, 6, 6, 6)
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 result tuple cardinality changed"
            )
        self.work.authenticate()
        for item in self.diagnostic_attempts:
            item.authenticate()
        for item in self.observations:
            item.authenticate()
        for item in self.memories:
            item.authenticate()
        for group in (
            self.adaptation_normal_decisions,
            self.exploration_decisions,
            self.parent_cross_decisions,
            self.diagonal_cross_decisions,
            self.three_lesson_cross_decisions,
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
        expected_observations = tuple(
            _prepare_episode(
                index=index,
                spec=self.spec,
                attempt=attempt,
                memory=expected_parent,
            ).projected_observation
            for index, attempt in enumerate(self.diagnostic_attempts)
        )
        if self.observations != expected_observations or self.memories[0].record() != (
            expected_parent.record()
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 observation or parent-memory custody changed"
            )
        for index, attempt in enumerate(self.diagnostic_attempts):
            _authenticate_diagnostic_for_episode(
                index=index,
                attempt=attempt,
                rustc_identity_sha256=self.reservation.rustc_identity_sha256,
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
                or normal.selected_hypothesis_sha256 is not None
                or exploration != self.exploration_decisions[index]
                or exploration.selected_hypothesis_sha256 != exploration_root
                or expected_outcome != self.adaptation_outcomes[index]
            ):
                raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                    "C8 adaptation transcript changed"
                )
            current = unordered.advance_unordered_cue_factor_memory_v1(
                current,
                expected_outcome,
            )
            expected_memories.append(current)
        if tuple(item.record() for item in self.memories) != tuple(
            item.record() for item in expected_memories
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 checkpoint lineage changed"
            )

        parent_cross = tuple(expected_memories[0].route(item) for item in self.observations[2:4])
        diagonal_cross = tuple(expected_memories[2].route(item) for item in self.observations[2:4])
        three_lesson = tuple(expected_memories[3].route(item) for item in self.observations[2:4])
        evaluations = tuple(expected_memories[4].route(item) for item in self.observations[4:6])
        controls = tuple(
            decision
            for observation in self.observations[4:6]
            for decision in (
                expected_memories[4].route(observation, outcome_gate=False),
                expected_memories[4].route(observation, axis_gates=(False, True)),
                expected_memories[4].route(observation, axis_gates=(True, False)),
            )
        )
        if (
            self.parent_cross_decisions != parent_cross
            or self.diagonal_cross_decisions != diagonal_cross
            or self.three_lesson_cross_decisions != three_lesson
            or self.evaluation_decisions != evaluations
            or self.control_decisions != controls
            or any(item.selected_hypothesis_sha256 is not None for item in parent_cross)
            or any(item.selected_hypothesis_sha256 is not None for item in diagonal_cross)
            or tuple(item.selected_hypothesis_sha256 for item in three_lesson) != (H01_SHA256, None)
            or tuple(item.selected_hypothesis_sha256 for item in evaluations)
            != EVALUATION_HYPOTHESIS_SHA256S
            or any(item.selected_hypothesis_sha256 is not None for item in controls)
            or _decision_scores(evaluations[0])
            != {
                H00_SHA256: 8,
                H01_SHA256: 12,
                H10_SHA256: 4,
                H11_SHA256: 8,
            }
            or _decision_scores(evaluations[1])
            != {
                H00_SHA256: 8,
                H01_SHA256: 4,
                H10_SHA256: 12,
                H11_SHA256: 8,
            }
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 crossed-grid selection behavior changed"
            )
        _authenticate_binding_evidence(self.final_memory, self.observations)
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
            or self.sha256
            != _digest(
                _RESULT_NAMESPACE,
                self.record(include_sha256=False),
            )
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 result root or materialized-source custody changed"
            )


DiagnosticCompileCallback = Callable[
    [str, str],
    rustc_feedback.RustcDiagnosticCompileAttemptV1,
]
ExecutionCallback = Callable[
    [
        authority.PreparedUnorderedCueFactorTextEditAuthorityV1,
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
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 public output feedback shape changed"
        )
    return outputs


def _validate_evaluation_oracle(
    value: object,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if type(value) is not tuple or len(value) != 2:
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 evaluation oracle must open both results together"
        )
    return tuple(
        _validate_outputs(outputs, batch=BATCHES[index])
        for index, outputs in enumerate(value, start=TRAIN_COUNT)
    )


def _authenticate_diagnostic_for_episode(
    *,
    index: int,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    rustc_identity_sha256: str,
) -> None:
    if type(attempt) is not rustc_feedback.RustcDiagnosticCompileAttemptV1:
        raise TypeError("C8 diagnostic compiler must return one exact attempt")
    attempt.authenticate()
    source = ORIGINAL_SOURCES[index]
    expected_code, expected_region = EXPECTED_DIAGNOSTICS[index]
    spans = tuple(
        span for diagnostic in attempt.error_diagnostics for span in diagnostic.primary_spans
    )
    regions = {
        rustc_feedback.FIRST_HALF
        if 2 * ((span.byte_start + span.byte_end) // 2) < len(source.encode("ascii"))
        else rustc_feedback.SECOND_HALF
        for span in spans
    }
    if (
        attempt.status != rustc_feedback.COMPILE_FAILED
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
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 diagnostic attempt changed"
        )


def _compile_diagnostic_once(
    identity: rust_executor.RustcIdentityV1,
    source: str,
) -> rustc_feedback.RustcDiagnosticCompileAttemptV1:
    identity.authenticate_structure()
    source_bytes = source.encode("ascii")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory(prefix="humanbrain-c8-unordered-diagnostic-") as directory:
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
    return rustc_feedback.seal_rustc_diagnostic_compile_attempt_v1(
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
    prepared: authority.PreparedUnorderedCueFactorTextEditAuthorityV1,
    decision: unordered.UnorderedCueFactorDecisionV1,
    batch: tuple[tuple[int, ...], ...],
) -> RustSelectedActionExecutionV2:
    return prepared.execute_selected(decision, batch)


def _prepare_episode(
    *,
    index: int,
    spec: C8DiagnosticUnorderedCueBindingSpecV1,
    attempt: rustc_feedback.RustcDiagnosticCompileAttemptV1,
    memory: unordered.FrozenUnorderedCueFactorMemoryV1,
) -> authority.PreparedUnorderedCueFactorTextEditAuthorityV1:
    return authority.prepare_unordered_cue_factor_text_edit_authority_v1(
        diagnostic_attempt=attempt,
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
    prepared: authority.PreparedUnorderedCueFactorTextEditAuthorityV1
    decision: unordered.UnorderedCueFactorDecisionV1
    pre_execute_counts: tuple[int, int]
    post_grant_counts: tuple[int, int]
    materialization: authority.UnorderedCueFactorTextEditMaterializationV1
    release: authority.UnorderedCueFactorTextEditReleaseV1
    execution: RustSelectedActionExecutionV2


def _execute_selected_grant(
    *,
    ordinal: int,
    reservation: C8RunReservationV1,
    prepared: authority.PreparedUnorderedCueFactorTextEditAuthorityV1,
    decision: unordered.UnorderedCueFactorDecisionV1,
    batch: tuple[tuple[int, ...], ...],
    execute: ExecutionCallback,
) -> _SelectedGrantV1:
    selected_hypothesis_sha256 = decision.selected_hypothesis_sha256
    if selected_hypothesis_sha256 is None:
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 STOP decision cannot materialize source"
        )
    pre_execute_counts = (prepared.materialization_count, prepared.release_count)
    if pre_execute_counts != (0, 0):
        raise _behavior_failure(
            "C8 source materialized before selected execution",
            stage=ROLES[ordinal - 1],
            decision=decision.record(),
            authority_receipt=prepared.receipt.record(),
            pre_execute_counts=pre_execute_counts,
        )
    execution = execute(prepared, decision, batch)
    if type(execution) is not RustSelectedActionExecutionV2:
        raise TypeError("C8 executor must return one exact RustSelectedActionExecutionV2")
    execution.authenticate_structure()
    materialization = prepared.materialization
    release = prepared.release
    post_grant_counts = (prepared.materialization_count, prepared.release_count)
    if materialization is None or release is None or post_grant_counts != (1, 1):
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 selected execution lacks exact materialization custody"
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
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 selected V2 execution custody changed"
        )
    return _SelectedGrantV1(
        prepared=prepared,
        decision=decision,
        pre_execute_counts=pre_execute_counts,
        post_grant_counts=post_grant_counts,
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
) -> C8SelectedExecutionAttemptV1:
    decision = grant.decision
    execution = grant.execution
    role = ROLES[ordinal - 1]
    selected_hypothesis_sha256 = decision.selected_hypothesis_sha256
    if selected_hypothesis_sha256 is None:
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 selected grant lost its hypothesis"
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
        "authority_receipt": grant.prepared.receipt,
        "pre_execute_counts": grant.pre_execute_counts,
        "post_grant_counts": grant.post_grant_counts,
        "materialization": grant.materialization,
        "release": grant.release,
        "execution": execution,
        "consequence": consequence,
    }
    partial = C8SelectedExecutionAttemptV1(**payload, sha256="")
    attempt = replace(
        partial,
        sha256=_digest(_ATTEMPT_NAMESPACE, partial.record(include_sha256=False)),
    )
    attempt.authenticate()
    if consequence.passed_cell != 1:
        raise _behavior_failure(
            "C8 selected materialization failed public output feedback",
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
    return attempt


def _run_selected_attempt(
    *,
    ordinal: int,
    spec: C8DiagnosticUnorderedCueBindingSpecV1,
    reservation: C8RunReservationV1,
    prepared: authority.PreparedUnorderedCueFactorTextEditAuthorityV1,
    decision: unordered.UnorderedCueFactorDecisionV1,
    batch: tuple[tuple[int, ...], ...],
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> tuple[C8SelectedExecutionAttemptV1, str]:
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
    spec: C8DiagnosticUnorderedCueBindingSpecV1,
    reservation: C8RunReservationV1,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> C8DiagnosticUnorderedCueBindingResultV1:
    spec.authenticate()
    reservation.authenticate(destination=Path(reservation.destination))
    if reservation.spec_sha256 != spec.sha256:
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 reservation/spec custody changed"
        )

    parent = unordered.start_unordered_cue_factor_memory_v1(HYPOTHESES)
    diagnostic_attempts = tuple(
        diagnostic_compile(reservation.rustc_identity_sha256, source) for source in ORIGINAL_SOURCES
    )
    for index, attempt in enumerate(diagnostic_attempts):
        _authenticate_diagnostic_for_episode(
            index=index,
            attempt=attempt,
            rustc_identity_sha256=reservation.rustc_identity_sha256,
        )
    parent_authorities = tuple(
        _prepare_episode(index=index, spec=spec, attempt=attempt, memory=parent)
        for index, attempt in enumerate(diagnostic_attempts)
    )
    observations = tuple(item.projected_observation for item in parent_authorities)

    memories = [parent]
    normal_decisions: list[unordered.UnorderedCueFactorDecisionV1] = []
    exploration_decisions: list[unordered.UnorderedCueFactorDecisionV1] = []
    outcomes: list[unordered.UnorderedCueFactorOutcomeV1] = []
    selected_attempts: list[C8SelectedExecutionAttemptV1] = []
    output_feedback_sha256s: list[str] = []
    parent_cross_decisions = tuple(parent.route(item) for item in observations[2:4])
    diagonal_cross_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...] = ()
    three_lesson_cross_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...] = ()

    for index, exploration_root in enumerate(EXPLORATION_HYPOTHESIS_SHA256S):
        current = memories[-1]
        prepared = (
            parent_authorities[index]
            if current is parent
            else _prepare_episode(
                index=index,
                spec=spec,
                attempt=diagnostic_attempts[index],
                memory=current,
            )
        )
        observation = prepared.projected_observation
        if observation != observations[index]:
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 authority projection changed across checkpoints"
            )
        normal = current.route(observation)
        exploration = current.route(
            observation,
            exploration_hypothesis_sha256=exploration_root,
        )
        if (
            normal.selected_hypothesis_sha256 is not None
            or exploration.selected_hypothesis_sha256 != exploration_root
        ):
            raise _behavior_failure(
                "C8 adaptation selection behavior changed",
                stage=ROLES[index],
                memory=current.record(),
                observation=observations[index].record(),
                normal_decision=normal.record(),
                exploration_decision=exploration.record(),
            )
        selected, feedback_root = _run_selected_attempt(
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
                "C8 adaptation learning-gate control changed",
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
            diagonal_cross_decisions = tuple(memories[-1].route(item) for item in observations[2:4])
            if any(
                item.selected_hypothesis_sha256 is not None for item in diagonal_cross_decisions
            ):
                raise _behavior_failure(
                    "C8 diagonal checkpoint resolved a confounded cross",
                    decisions=tuple(item.record() for item in diagonal_cross_decisions),
                )
        elif index == 2:
            three_lesson_cross_decisions = tuple(
                memories[-1].route(item) for item in observations[2:4]
            )
            if tuple(item.selected_hypothesis_sha256 for item in three_lesson_cross_decisions) != (
                H01_SHA256,
                None,
            ):
                raise _behavior_failure(
                    "C8 three-lesson checkpoint failed missing-cross control",
                    decisions=tuple(item.record() for item in three_lesson_cross_decisions),
                )

    final = memories[-1]
    evaluation_decisions = tuple(final.route(item) for item in observations[4:6])
    control_decisions = tuple(
        decision
        for observation in observations[4:6]
        for decision in (
            final.route(observation, outcome_gate=False),
            final.route(observation, axis_gates=(False, True)),
            final.route(observation, axis_gates=(True, False)),
        )
    )
    if (
        any(item.selected_hypothesis_sha256 is not None for item in parent_cross_decisions)
        or tuple(item.selected_hypothesis_sha256 for item in evaluation_decisions)
        != EVALUATION_HYPOTHESIS_SHA256S
        or any(item.selected_hypothesis_sha256 is not None for item in control_decisions)
    ):
        raise _behavior_failure(
            "C8 evaluation binding or causal-control behavior changed",
            parent_cross_decisions=tuple(item.record() for item in parent_cross_decisions),
            evaluation_decisions=tuple(item.record() for item in evaluation_decisions),
            control_decisions=tuple(item.record() for item in control_decisions),
        )
    _authenticate_binding_evidence(final, observations)

    evaluation_grants: list[_SelectedGrantV1] = []
    for index, decision in enumerate(evaluation_decisions, start=TRAIN_COUNT):
        prepared = _prepare_episode(
            index=index,
            spec=spec,
            attempt=diagnostic_attempts[index],
            memory=final,
        )
        if prepared.projected_observation != observations[index]:
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 evaluation authority projection changed"
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
        selected = _seal_selected_attempt(
            ordinal=index + 1,
            grant=grant,
            expected_outputs=expected_outputs,
            output_feedback_sha256=feedback_root,
        )
        selected_attempts.append(selected)
        output_feedback_sha256s.append(feedback_root)

    payload = {
        "status": STATUS_PASS,
        "spec": spec,
        "reservation": reservation,
        "diagnostic_attempts": diagnostic_attempts,
        "observations": observations,
        "memories": tuple(memories),
        "adaptation_normal_decisions": tuple(normal_decisions),
        "exploration_decisions": tuple(exploration_decisions),
        "adaptation_outcomes": tuple(outcomes),
        "parent_cross_decisions": parent_cross_decisions,
        "diagonal_cross_decisions": diagonal_cross_decisions,
        "three_lesson_cross_decisions": three_lesson_cross_decisions,
        "evaluation_decisions": evaluation_decisions,
        "control_decisions": control_decisions,
        "selected_attempts": tuple(selected_attempts),
        "output_feedback_sha256s": tuple(output_feedback_sha256s),
        "evaluation_oracle_sha256": evaluation_oracle_sha256,
        "work": _seal_work(),
    }
    partial = C8DiagnosticUnorderedCueBindingResultV1(**payload, sha256="")
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _reserve_destination(
    spec: C8DiagnosticUnorderedCueBindingSpecV1,
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
) -> C8RunReservationV1:
    spec.authenticate()
    if not _valid_commit(preregistration_commit):
        raise ValueError("C8 preregistration commit must be 40 lowercase hex characters")
    if not _valid_sha256(rustc_identity_sha256):
        raise ValueError("C8 rustc identity root is invalid")
    destination = Path(destination).resolve()
    reservation_path = Path(reservation_path).resolve()
    expected_reservation = destination.with_name(f"{destination.name}.reservation.json")
    if reservation_path != expected_reservation:
        raise ValueError("C8 reservation must be the result's exact sibling")
    if os.path.lexists(destination) or os.path.lexists(reservation_path):
        raise FileExistsError(destination if os.path.lexists(destination) else reservation_path)
    partial = C8RunReservationV1(
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
            raise OSError("C8 reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if reservation_path.read_bytes() != payload:
        raise OSError("C8 reservation readback changed")
    return reservation


def _result_document(
    result: C8DiagnosticUnorderedCueBindingResultV1,
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
            "memories": tuple(item.record() for item in result.memories),
            "adaptation_normal_decisions": tuple(
                item.record() for item in result.adaptation_normal_decisions
            ),
            "exploration_decisions": tuple(item.record() for item in result.exploration_decisions),
            "adaptation_outcomes": tuple(item.record() for item in result.adaptation_outcomes),
            "parent_cross_decisions": tuple(
                item.record() for item in result.parent_cross_decisions
            ),
            "diagonal_cross_decisions": tuple(
                item.record() for item in result.diagonal_cross_decisions
            ),
            "three_lesson_cross_decisions": tuple(
                item.record() for item in result.three_lesson_cross_decisions
            ),
            "evaluation_decisions": tuple(item.record() for item in result.evaluation_decisions),
            "control_decisions": tuple(item.record() for item in result.control_decisions),
            "evaluation_oracle_sha256": result.evaluation_oracle_sha256,
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
    spec: C8DiagnosticUnorderedCueBindingSpecV1,
    reservation: C8RunReservationV1,
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
    spec: C8DiagnosticUnorderedCueBindingSpecV1,
    reservation: C8RunReservationV1,
    error: C8BehaviorFailure,
) -> dict[str, object]:
    partial_transcript = _plain(error.partial_transcript)
    if type(partial_transcript) is not dict:
        raise TypeError("C8 behavior failure transcript must be one exact object")
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
            raise OSError("C8 terminal document write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = destination.read_bytes()
    if retained != payload or _canonical_json(json.loads(retained.decode("ascii"))) + b"\n" != (
        payload
    ):
        raise OSError("C8 terminal document readback changed")
    decoded = json.loads(retained.decode("ascii"))
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if document_sha256 != _digest(_DOCUMENT_NAMESPACE, body):
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
            "C8 terminal document root changed"
        )
    if decoded["terminal_status"] == STATUS_PASS:
        record = dict(decoded["authenticated_result_record"])
        result_sha256 = record.pop("sha256", None)
        if result_sha256 != decoded["result_sha256"] or result_sha256 != _digest(
            _RESULT_NAMESPACE,
            record,
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 persisted result root changed"
            )
    elif decoded["terminal_status"] == STATUS_BEHAVIOR_FAIL:
        partial = decoded.get("partial_transcript")
        if type(partial) is not dict or decoded.get("partial_transcript_sha256") != _digest(
            _FAILURE_NAMESPACE,
            partial,
        ):
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 persisted behavior-failure transcript root changed"
            )
    elif decoded["terminal_status"] != STATUS_EXECUTION_ABORT:
        raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error("C8 terminal status changed")


def _run_and_write_at_paths_for_test_v1(
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
    spec: C8DiagnosticUnorderedCueBindingSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C8DiagnosticUnorderedCueBindingResultV1:
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
    except C8BehaviorFailure as error:
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
    spec: C8DiagnosticUnorderedCueBindingSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C8DiagnosticUnorderedCueBindingResultV1:
    """Preflight, reserve, and spend the exact production C8 cohort once."""

    identity = _preflight_rustc_v1()
    destination, reservation_path = _production_evidence_paths_v1()

    def diagnostic_compile(
        rustc_identity_sha256: str,
        source: str,
    ) -> rustc_feedback.RustcDiagnosticCompileAttemptV1:
        if rustc_identity_sha256 != identity.sha256:
            raise ContinualRustDiagnosticUnorderedCueBindingCohortV1Error(
                "C8 diagnostic compiler identity changed after reservation"
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
    "C8BehaviorFailure",
    "C8DiagnosticUnorderedCueBindingResultV1",
    "C8DiagnosticUnorderedCueBindingSpecV1",
    "C8PublicConsequenceV1",
    "C8RunReservationV1",
    "C8SelectedExecutionAttemptV1",
    "C8WorkV1",
    "ContinualRustDiagnosticUnorderedCueBindingCohortV1Error",
    "make_c8_diagnostic_unordered_cue_binding_spec_v1",
    "output_sha256_v1",
    "run_and_write_cohort_v1",
]


if __name__ == "__main__":
    main()
