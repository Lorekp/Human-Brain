"""C11B consequence-conditioned selection of a relative source occurrence.

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
from humanbrain import rust_selected_action_executor as rust_executor
from humanbrain.rust_selected_action_executor import RustSelectedActionExecutionV2

SCHEMA = "humanbrain.continual-rust-diagnostic-occurrence-localization-c11b.v1"
RESULT_DOCUMENT_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-occurrence-localization-c11b-result-document.v1"
)
RESERVATION_SCHEMA = (
    "humanbrain.continual-rust-diagnostic-occurrence-localization-c11b-reservation.v1"
)
STATUS_PASS = "PASS_C11B_RUSTC_DIAGNOSTIC_OCCURRENCE_LOCALIZATION_V1"
STATUS_BEHAVIOR_FAIL = "FAIL_C11B_RUSTC_DIAGNOSTIC_OCCURRENCE_LOCALIZATION_V1"
STATUS_EXECUTION_ABORT = "ABORT_C11B_RUSTC_DIAGNOSTIC_OCCURRENCE_LOCALIZATION_V1"
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
    "c11b_occurrence_localization",
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
TRAIN_FEEDBACK_ROLES = tuple(f"c11b-{role}-paired-feedback" for role in EPISODE_ROLES[:4])
EVALUATION_ORACLE_ROLE = "c11b-evaluation-oracle-after-both-executions"
EXPECTED_DIAGNOSTICS = ("E0425", "E0308", "E0425", "E0308", "E0425", "E0308")
CONTEXT_WORDS = ("nuvor", "selqim", "selqim", "nuvor", "selqim", "nuvor")
EXPECTED_PRIMARY_TEXTS = (
    "xevan20qor",
    "varem21dox",
    "belun22qaz",
    "mavet23qor",
    "xorin24qep",
    "jorin25vud",
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
struct Brask20Tuli {
    xevan20qor: i64,
}

fn nuvor(dalek20vim: i64) -> i64 {
    dalek20vim
}

pub fn solve(perun20zaf: &[i64]) -> Vec<i64> {
    let kirad20mox = 61_i64;
    let vulen20pex = 37_i64;
    perun20zaf
        .iter()
        .copied()
        .map(|jomar20tic| {
            jomar20tic + nuvor(Brask20Tuli { xevan20qor }.xevan20qor)
                + (kirad20mox - kirad20mox)
                + (vulen20pex - vulen20pex)
        })
        .collect()
}"""

TRAIN_REPLACE_NEAR_SOURCE = """\
fn selqim(fenar21qov: i64) -> i64 {
    fenar21qov
}

pub fn solve(lurin21pask: &[i64]) -> Vec<i64> {
    let xador21miv = 43_i64;
    let qesin21zup = 67_i64;
    let varem21dox = "sar21nivo";
    assert!(!varem21dox.is_empty());
    lurin21pask
        .iter()
        .copied()
        .map(|koral21tex| {
            koral21tex + selqim(varem21dox)
                + (xador21miv - xador21miv)
                + (qesin21zup - qesin21zup)
        })
        .collect()
}"""

TRAIN_DECLARE_NEAR_SOURCE = """\
struct Tovan22Rumi {
    belun22qaz: i64,
}

fn selqim(xorin22pev: i64) -> i64 {
    xorin22pev
}

pub fn solve(damar22kix: &[i64]) -> Vec<i64> {
    let fesul22vor = 71_i64;
    let pavin22doq = 47_i64;
    damar22kix
        .iter()
        .copied()
        .map(|lurad22mex| {
            lurad22mex + selqim(Tovan22Rumi { belun22qaz }.belun22qaz)
                + (fesul22vor - fesul22vor)
                + (pavin22doq - pavin22doq)
        })
        .collect()
}"""

TRAIN_REPLACE_FAR_SOURCE = """\
fn nuvor(xeral23qim: i64) -> i64 {
    xeral23qim
}

pub fn solve(bunar23vex: &[i64]) -> Vec<i64> {
    let korin23daf = 53_i64;
    let jelar23pux = 73_i64;
    let mavet23qor = "zol23rima";
    assert!(!mavet23qor.is_empty());
    bunar23vex
        .iter()
        .copied()
        .map(|perid23sav| {
            perid23sav + nuvor(mavet23qor)
                + (korin23daf - korin23daf)
                + (jelar23pux - jelar23pux)
        })
        .collect()
}"""

EVALUATION_DECLARE_NEAR_SOURCE = """\
struct Brum24Keli {
    xorin24qep: i64,
}

fn selqim(dalum24vix: i64) -> i64 {
    dalum24vix
}

pub fn solve(fesad24por: &[i64]) -> Vec<i64> {
    let jorin24mav = 79_i64;
    let kelun24tex = 59_i64;
    fesad24por
        .iter()
        .copied()
        .map(|ravid24bos| {
            ravid24bos + selqim(Brum24Keli { xorin24qep }.xorin24qep)
                + (jorin24mav - jorin24mav)
                + (kelun24tex - kelun24tex)
        })
        .collect()
}"""

EVALUATION_REPLACE_FAR_SOURCE = """\
fn nuvor(xedal25qor: i64) -> i64 {
    xedal25qor
}

pub fn solve(murin25pax: &[i64]) -> Vec<i64> {
    let kavor25zim = 89_i64;
    let belar25qex = 83_i64;
    let jorin25vud = "pir25sova";
    assert!(!jorin25vud.is_empty());
    murin25pax
        .iter()
        .copied()
        .map(|dalex25mur| {
            dalex25mur + nuvor(jorin25vud)
                + (kavor25zim - kavor25zim)
                + (belar25qex - belar25qex)
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

# C11's 111 prior exclusions plus every source root exposed while developing
# and preregistering C11: six stale-fake originals, ten stale-fake
# materializations, six committed originals, and ten committed-fixture
# materializations. Literal custody avoids importing the spent predecessor.
EXCLUDED_PRIOR_SOURCE_SHA256S = (
    "01c495a14459751c531bd93f7d6c148c94ba1c07d945a54554753146beb0833d",
    "043654a72e840728743e816026e3e2168472c58d99dace9f92486cdbb9dccf4c",
    "077283306f7ecdd8e80fc072fa23db8902593deff7e67543c0c405a23d0009f6",
    "07f4e3f8f6e1e932319146df507252815797acec809ba3a477b0f50604cfa3e6",
    "082120df1274f6b90bccb3c064810a37079e892560b9429cda63f68554020338",
    "085613a2a90d104d2011b0ad1f43840dbffb72c7823c397b43fef10297c10770",
    "08ffbca3d58b73e663e4516b0f9c564e1c6a2b4de33df0434c3ee933260c2312",
    "099d23abb6fd2cf68894b9b0739b5a134b4be172368e1938b2577f69e6c8caf9",
    "09b937e053872172813c09637fa96b4f19775ecda6911f61b1431b413ccec8b3",
    "0a4a5f19ae15bf863e122c0763b1f8345d50c34c05fd6949cb3a6cbd2631759f",
    "0e131b7e04223b7ba8bbd504e8f4d1b158474d68fe0ec3a8f383a1cd76835739",
    "0e53f26ec756454b046785af357fee1be1db7e9e8fce8c1b4683d2cc6217ac44",
    "0e64010a82f448322b8c42ed6e73bee82391b0d069a5e6ff349935e4fe1ccf47",
    "118db3017e71bd5380cfa8bc3a06da5ea47fe038d4ccc3f8e5440ce51afb7fbc",
    "11fdcb5cf4d1ab242cabbc6ce844f0b2da3f5fa03a57f8aa2f2ecb7f7e410fb4",
    "126b0ce796c248396e311e31a332e28dcdfbb328e0b994d64b39a1f6e3e0877b",
    "12ab6fb3efa6d3bf43b5d3903a743e004a1218656072dab32b08a3daf31fc7f8",
    "13405df1dce5d55512a918165f5102de5526b3fef5ad3c5fa95435a717bf712b",
    "1494cd3c8887257e8bc2cc82aa07c35ff8a7f1371887f1f517e63be8f77ea68d",
    "1501ca341da74b246e1a59e006c56ed2a7251f39e7c78ae7c8ed36ed047d5e66",
    "15475c869b7c5dbf90030264145f59d462fa66d5f49c3f17543d12270d55cb87",
    "16504bb28d57532c6621aa0a36c6db23204d54fe7ea0c56132efa8e38b234704",
    "178e9ffa2a3c9faf01739db95195c5575d9da57c6835e9a4945269bdb7ee0fc8",
    "180ff462710fde6978682cf9de58c2e412645d36e616959fbf66447b1532f7b2",
    "182c0e0084471185b6d0cd16271bc43b1eb7561a49a8a099bbbf4027abe9117f",
    "186a019e4442da249d1e6c73e840b1acb18be9d4fca7a3cf49d341a0ae049f7a",
    "197a5fcbb6bb71b839994f70b43718610a8ebd43e35dd1050afb3bc7d1dc6572",
    "1cb28ca58e88b9641fc8416aaf6a076c12499677d8f25181c37c770b2e84105d",
    "1d41a1e71de3d87d7a2ea2e54beedad232fc4c5c0e8af88ec19027e975ba18b9",
    "1f0fa31aafb6d3737b0a394c4a9f31140aff3f543d35911c80ac184ba828aeb9",
    "2018ff95b0b6c90abe9a754d1490b56ef6a854cd3801c4a9973ead72e57926d9",
    "205b3934fe85ed23676639dfcba907f0d81522c02bf9c95c4d32f7a4d71511b3",
    "22d555f484f9aa33e1f5c184c5d91893aac1b02593ca5f1e4d28f1d9187418db",
    "24bab5e10a02ac039b1fec33de22be26b27af3524b360ce24947a5ab78df880c",
    "2cb5e3555404d53084672be16652220a61e0c448ee9b9f97bd1084e4a892e2ef",
    "30b0f5e5d23fc2b8ede3c6fcb3892078b89be2fec1ddb530a50251fc48a0d33e",
    "30d0ce2d3985e8c3848aa4d8bc91a44c9e9e612a98a0db7ce8382e5b28de6f50",
    "319f315176807b6f4db650e74d0775f65af626a06d5f4caf87031dd692ee8c7a",
    "320434b33f67afd875de8232703e25da7c4032f367c3a76dfccbd572df226243",
    "3598b6e3bfdc014c579d1c85bb38748506f83115c9f61ed9080b2b850d8745fc",
    "3909b3793cc81ebf2cc672a7021cfff4fa340eca655b26e52d0e55481d403670",
    "3a76a6783fda0b2aef342124467e2fb1245ecea140052760dabed7b10d10799b",
    "3c354eda2adea9f7adf50ac6368394e00370555d7a999d7da96c873dd45947f6",
    "41d88c29b99af965637d6fec963fb177f79bfc8237fee36fc9aa734553617b54",
    "4588278d064b626f0a9c6e6c6067992d64ac547d09e1ff15e386f9d441f2d1b7",
    "4ecf619642ddf0dee5466334c755544630f9d3656aea073c55e3a30777f4b79f",
    "511caf6c304c7f64101af4b868e49360067bfa7a6eed2fc20b5c68b37d3b68d2",
    "514720a04d50420dbc4d1da9be1c24da37be8102009ecdcaa7259fc114795112",
    "532e90d9739ec16748ad152d53509d34de7ccbf5efccfb067aa915789210f662",
    "54a4a279c62c5dfb2c5c1ffb5bd7944ec8ac56ff3fb4dbd2eb9223e0ef8452bd",
    "55466beed71c9e57c2cc1d8c31835d29500e187327903b5fde4038724d8e0f5d",
    "55ec139329ee99ccebde3c3ffcc5d7dac5515075c9ff0095e6068517009fb1ff",
    "58fc380502a2e3ba06debbd720a8dad7d4dd223183ac8d1eb4cd8dcc324d77b1",
    "5bce3fd6d5e55d66da2bb7f55adb0fc6d9ef1c66d17c639ab88d2e5d587ed057",
    "5c439c971363f6863f0386b16c90ac9258d0d8d4f82fccfab300bfeec91b534a",
    "5ea76f081fe59993eaabd0da33cf9f2b1f8429ff3ea876c3f6a86fb2f6892c69",
    "5efc39f66d45b1252a76ddd536a251d42e5956aaa84f61e3a2763b7d1d2a3d79",
    "5fb8edd3d42c1dd2c8e7fd95aec4cf6809cb8158cc92929b371653718cc03fc2",
    "61874992d6db72fcc87f6101b1a9e8deb8cf46904c4617596e04b1481b1c38d2",
    "6255543848aa1646402de9eb7a9780a5dd67c67f95400a27a59118292c691076",
    "62b3a80e76db43ce2c726ae246bf487eb26ba4c40e02b7c15a2fcedee36a3e3f",
    "652e0453283971e31e42a27ead3822743e17d0fc75e8ea5ae6afe6ee8c499fa4",
    "652e21dfb435032e94d7f0b72e5c5ed3c28faf2f4ba32c09cb203a57d5c785c9",
    "66703237df4e8f68e7f9e678239970c93e218e7267bae1aeb4b8833e7a475a93",
    "6745cd76302ab4b9f8f0d393783804604ea14c0824b3e5f3abffd854056b5937",
    "67db118afe3d6d529adb81835b058771d922b86b2f17b11270d0e37f2044bad0",
    "737b90128d66b22d7f5f15e04780eaa2920c85612a9f8bb24f4f814c0afe8d70",
    "767ac8ba99b72cfc2b148644c6e02f015dfe12bef19a1946288ccf4ae608f9cb",
    "77b0f6c81f38a4edc68f9ab5212bf281fa7c08576c6241937fb1c6d8836cfaa5",
    "78b547d8925c6f9e5089fd98b15b68c1d770cc57550218b6844013348a515237",
    "7963786ba604c2da0ab21f8d9bbb032436c9abfff71b964b6a1c1e5da0613bd4",
    "7b7c814f1b0c3a55cce059eedefa3fc6ec389078aec0b7e6da32c15ef4e3890e",
    "7bf81394175578ae1ccd2ac1aa8a0e63d008498b5ee3705b33913a38055c62a3",
    "7cfdf714f80dba936574bca220d6a8e82fde7907c8feab8601f5315105c460ed",
    "7ea07d6438adf64d57210c35a69d88c4a48c5c8f5cc653246d00b32e3924d9e2",
    "80a51272bb0f3f2dab103916f34025c1efb384f33cf2d1176b276e7149bd38fc",
    "829d367e3845f66d4c3c1a9bafa43baaec733b618c600ae9da505479f6b332c7",
    "83d7dbb8d55f0f9b06d6750ac4a3552981ec30700f9f65e11c753cf76a5d0f49",
    "875e1e20df98385178b3af24600bfd9bed7973b6333ebbdad65fe19b07d275aa",
    "8b225ebb888c282c0e6293215cdaecb36427efcbe3ca7317bbf322e5e1888c49",
    "8b36d4ffe2dfb48c3f8f3bf3eb8ebcfa62e32cee4b18a12f46f12318176fedbb",
    "900e4f8cc9648f00e4835b50185f24da64d12babb35dbf5bf65935f575398362",
    "909dce5a418c3118bfb4479a3551aefde5b8828a6802b9566af4807ea3d7776b",
    "913097c4d9e858c41d80a8d671a5996340a9d3071d1b1902350bb4403a236f88",
    "9156378419e5a875a0aa88632bbada13c3261e07ca979ce1ea9afbf650025975",
    "922b4edcfbac20cc185dc885798a0df53e2d0678f6fbce6b761f6b02b03cdb1f",
    "93c64d8f9ee885311950b9c81ee793222ebba5abc72ded612b53c66d70a5bfa1",
    "962a5bcd733f4026a07dea5342aa5145ea837bd282edbf67d41418da0f37f6dd",
    "9634b257c00cc1f0e2470cead58cc6ffec8136fb1a2388511b8c5f17b01dd9fc",
    "967382da9e579783e7fe5cb5cbf3c4e1a92f5d8ca84948aa641d0f289f62fb8d",
    "96f90f7f7d3e6a3677ce8cd1e7ddb3cc9986a49d42180a5f74b2375a1b6ed985",
    "97ad17997290a4a4ec42f3721299ee937de50a70c57e8b0738a6cba5da579c9c",
    "991c87864d821cefcf10e2a9dd9a383540cb1597d36b5ccbd939058567d482c0",
    "a0c554ee679a10b37bad6fe3e7b6d6b905777d579302ab1cde2cf7f69f43b5dc",
    "a30f2d85a0f2fcd28458f01e2353b20bd7567478c9f0e14ba900becdcca0da15",
    "a35874ffedbaabac33977d554916c049a823434b35d57e5ba82be0770ab088a7",
    "a54d2fe9f99aaebfbe7cd98fe991c26142bea39a172f9ffa3c05117af17876a0",
    "a5979ba178a792517f4e675fba66fe1a61f96c1a383e8453f32c5320f22fd04a",
    "a8102ce87ef9940510e2921ce401d3a66aa1b6ac08a6865fc84b733bc322cd71",
    "a89deaef69ee09e4511615ba8da7f5246b82b334854ce33b4f653b94abcbd85b",
    "b037904c58d0ab527c2d9e20595a6071ad786a0748165a2684f3af54448a417f",
    "b058aff0828a478f2d195e965ad0860362343af0271b860a2ac9f7ca9195229b",
    "b05b9acb3804dce03571ccb1c42e707cb799ca29ff4c4dc0bcdce0fefa0344af",
    "b0cfe61f4d7c74758f0e7b165011ae660210f26079274a4179e585ad9279f742",
    "b0e5bff5168441980089f2114b942d695b8d3496d6a67b86c96f7ffc8ca25ff7",
    "b1235cd4490497f9e3d17db82408f3d2d47355261dd32792360fa4ff19c2e458",
    "b2ab91c571c3ab154735d98bb584502ba0039030372c33a2b75ad86992931319",
    "b3cbeea34d564dd583a681142ad27d672c4ca64f9a7386fbb6c319b868c6cf0e",
    "b6c0d6eb0128523f5135c0497453aa89ab42914e007eba8263bdc6bc64b14eb0",
    "b8bda9c33ba7d62f481387d0da28042c91a651a1e0c99b717b868a1f9078d81a",
    "b9f5dae974ca36de1fdfb9d6e3a3a7cca962ccde9bda4dda57cc017468467bb8",
    "bb29c7ac1a1b36d79131fc7dac4c3264b3fb1e053c5c82e4c49d706e0f412bd4",
    "bda397623caaf27ecef7e73e0b6052a0c12c3ff419333ef1f7231a1e011485d3",
    "be1bc451971f347f22d0072a3411d15a54f00c1d4c819f1be9d575330455625d",
    "be1cddea32682fb8f533adfbb1e78621b8d67a024594317aace73e6d87befe9f",
    "becaa22a3e161cb8c825d6f70a6235604971d33f6a05e704a8553abf3b741bc9",
    "c199578bb5c3d6df27a533cacf04d0d14746fe12b5ae556b9c805f2d746df5d0",
    "c25990cc50879cfd1bcb2f0f956260233efaaca98ea28a928b539febbbea323b",
    "c32cf1d3398cf09cdbba91529789049b67f9113a96e1ac3ee6e2889169b4ef8d",
    "c5bc2f2af0a579325f857006e222f4f59f4f44bd58eceaa0dd8e82baba16f9c5",
    "c933d69fc571aebf24eaa5cba0255e6b5767a3316a4884337c44725f2519c56d",
    "ca96a9774be460daed557117e0b56fc334d29d709d9fac0a6f53003e3fd9be34",
    "cc9cbd482f93a7a2c761bb22b01c121ee9670f49cdaea0242654af997e6766ff",
    "cdddc8926d80232b494132e1687f0e344b92c0fdb5c667ed4a7051462ee72762",
    "d00939b3ee635b6011a6e3296550dfef6fa1fa1d3211f42514822300bb37332c",
    "d170b638fb72ec15d8414e61b397b7e59165d1d89958bb05e6658ad0c20358a8",
    "d23bee65de3f003933e9739add066b9c1973bb5760abfc6f15129b9b1fe169e3",
    "d335b523ec67621ba6925c48c84f1d4cce2f38f00956099297db443d1fe80f50",
    "d34b307aa6cef6e403ee88698897f73dfdb21511aee7b1b629baf5631eaab6d6",
    "d67a08288db55c84a4e7df58c68c9a62819d118c4f80aa4ff196606c21166046",
    "d907300fcf3390c3b8697a8437f8448cc10057736ac1bdee3c27084f2a7b0b7f",
    "e45e23ac3e82c4146d5aafee56590c9368b758ce744b04e52547e30a7374b855",
    "e6f89d9d2be762471863ff3c6a8b00dd3eef59ad388eed55388883198385429f",
    "e8dd412b73a6612ab6a70483ef738881ec09b2d9c429638cffab630648bcb3e8",
    "ed19d4db7c290cd81eb57f39e3cb818ca2a5f51770e3b9b8dc71252f9cf496df",
    "ee0b4c24a158c1d1e66bfbeaa5fcdccdaef641b5bd1b161f9a73054f9915e675",
    "ef114a5c9493d7f6cc5726882217322dd7d8ccee527cd4811a01e1ed1bc3ba55",
    "efe7f26661853a16930fb596374851727c3d0f32a02ca9c3e3bd610eca91fe4d",
    "f190adc6241af291b5f64256eb813bbdb75c9d2bf6f6d354b70b5375751dd13b",
    "f7d4054296cf905af537ef45f98301a30f109e2b84cf286ec1393eda2d91d970",
    "fb287c91a51e25ff28d3c4a368f2cc297e05c81b81da73acdec44b2a95af2d21",
    "fbeac8ce5b168255e52c24318106a00194f7a5999c21bc453987227fddb5cb94",
    "fe2047e404f00495f1a10493b2261d03709b393387e21cc5ed813efd9df2f7cd",
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
_SPEC_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-spec-v1"
_CONTEXT_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-context-v1"
_OUTPUT_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-output-v1"
_FEEDBACK_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-feedback-v1"
_CONSEQUENCE_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-consequence-v1"
_ATTEMPT_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-attempt-v1"
_WORK_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-work-v1"
_RESERVATION_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-reservation-v1"
_RESULT_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-result-v1"
_FAILURE_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-failure-v1"
_DOCUMENT_NAMESPACE = b"c11b-rustc-diagnostic-occurrence-localization-document-v1"


class ContinualRustDiagnosticOccurrenceLocalizationC11BError(ValueError):
    """The frozen C11B cohort or its authenticated transcript changed."""


class C11BBehaviorFailure(RuntimeError):
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
    raise TypeError(f"C11B transcript cannot serialize {type(value).__qualname__}")


def _behavior_failure(reason: str, **partial_transcript: object) -> C11BBehaviorFailure:
    return C11BBehaviorFailure(
        reason,
        {key: _plain(value) for key, value in partial_transcript.items()},
    )


def _current_mechanism_file_sha256s() -> tuple[tuple[str, str], ...]:
    return tuple(
        (path, sha256((_REPO_ROOT / path).read_bytes()).hexdigest())
        for path, _expected in FROZEN_MECHANISM_FILE_SHA256S
    )


def output_sha256_c11b_v1(outputs: tuple[tuple[int, ...], ...]) -> str:
    return _digest(_OUTPUT_NAMESPACE, {"outputs": outputs})


@dataclass(frozen=True, slots=True)
class C11BDiagnosticOccurrenceLocalizationSpecV1:
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
            != tuple(output_sha256_c11b_v1(value) for value in EXPECTED_OUTPUTS)
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
            or len(set(self.excluded_source_sha256s)) != 143
            or set(self.original_source_sha256s).intersection(self.excluded_source_sha256s)
            or any(count > message_line.MAX_TOTAL_CUES for count in self.expected_merged_cue_counts)
            or self.sha256 != _digest(_SPEC_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B public fixture specification changed"
            )


def make_c11b_diagnostic_occurrence_localization_spec_v1() -> (
    C11BDiagnosticOccurrenceLocalizationSpecV1
):
    partial = C11BDiagnosticOccurrenceLocalizationSpecV1(
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
        expected_output_sha256s=tuple(output_sha256_c11b_v1(value) for value in EXPECTED_OUTPUTS),
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


DEFAULT_PUBLIC_SPEC = make_c11b_diagnostic_occurrence_localization_spec_v1()


@dataclass(frozen=True, slots=True)
class C11BRunReservationV1:
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B run reservation changed"
            )


@dataclass(frozen=True, slots=True)
class C11BPublicConsequenceV1:
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B public consequence authentication failed"
            )


def _seal_consequence(
    *,
    execution: RustSelectedActionExecutionV2,
    expected_outputs: tuple[tuple[int, ...], ...],
    output_feedback_sha256: str,
) -> C11BPublicConsequenceV1:
    payload = {
        "source_sha256": execution.selected_source.source_sha256,
        "batch_sha256": execution.preparation.batch_sha256,
        "output_feedback_sha256": output_feedback_sha256,
        "execution_sha256": execution.sha256,
        "observed_outputs": execution.outputs,
        "expected_outputs": expected_outputs,
        "passed_cell": int(execution.outputs == expected_outputs),
    }
    result = C11BPublicConsequenceV1(
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
class C11BSelectedExecutionAttemptV1:
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
    consequence: C11BPublicConsequenceV1
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B selected attempt phase changed"
            )
        try:
            relation_index = RELATIONS.index(self.relation)
        except ValueError as exc:
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B selected attempt relation changed"
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B selected execution attempt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class C11BWorkV1:
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B work accounting changed"
            )


def _seal_work() -> C11BWorkV1:
    payload = dict(EXPECTED_WORK_COUNTS)
    result = C11BWorkV1(**payload, sha256=_digest(_WORK_NAMESPACE, payload))
    result.authenticate()
    return result


def _require_unique_pair_pass(
    consequences: tuple[C11BPublicConsequenceV1, C11BPublicConsequenceV1],
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
    """Pure transaction gate used before any C11B memory advance."""

    if (
        type(passed) is not tuple
        or len(passed) != 2
        or any(type(value) is not int or value not in (0, 1) for value in passed)
        or not _valid_sha256(checkpoint_sha256)
    ):
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B paired pass-cell gate input changed"
        )
    if sum(passed) != 1:
        raise _behavior_failure(
            "C11B paired feedback did not contain exactly one pass and one failure",
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
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B unordered cue permutation changed routing"
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
        padding = "// c11b relation-invariant source padding\n"
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B identifier relabel or source padding changed relation identity"
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
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B malformed occurrence roster did not fail closed"
        )


@dataclass(frozen=True, slots=True)
class C11BDiagnosticOccurrenceLocalizationResultV1:
    status: str
    spec: C11BDiagnosticOccurrenceLocalizationSpecV1
    reservation: C11BRunReservationV1
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
        tuple[C11BSelectedExecutionAttemptV1, C11BSelectedExecutionAttemptV1], ...
    ]
    training_outcome_pairs: tuple[
        tuple[unordered.UnorderedCueFactorOutcomeV1, unordered.UnorderedCueFactorOutcomeV1],
        ...,
    ]
    passing_outcome_indexes: tuple[int, ...]
    evaluation_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    control_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    deranged_context_decisions: tuple[unordered.UnorderedCueFactorDecisionV1, ...]
    evaluation_attempts: tuple[C11BSelectedExecutionAttemptV1, ...]
    training_feedback_sha256s: tuple[str, ...]
    evaluation_oracle_sha256: str
    evaluation_checkpoint_before_sha256: str
    evaluation_checkpoint_after_sha256: str
    work: C11BWorkV1
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B result tuple cardinality changed"
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
                raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                    "C11B diagnostic, envelope, roster, or observation cross-custody changed"
                )

        expected_parent = unordered.start_unordered_cue_factor_memory_v1(HYPOTHESES)
        if self.memories[0].record() != expected_parent.record():
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B parent-memory custody changed"
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
                raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                    "C11B TRAIN output, feedback, envelope, or roster cross-custody changed"
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
                raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                    "C11B paired adaptation transcript changed"
                )
            current = unordered.advance_unordered_cue_factor_memory_v1(
                current, outcomes[passing_index]
            )
            expected_memories.append(current)
        if tuple(item.record() for item in self.memories) != tuple(
            item.record() for item in expected_memories
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B checkpoint lineage changed"
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B final selection, control, or selected-only execution changed"
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B shared evaluation oracle root changed"
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
                raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                    "C11B evaluation output, feedback, envelope, or roster cross-custody changed"
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B TRAIN/evaluation cue overlap or fresh-identifier u=0 evidence changed"
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B source freshness, evaluation immutability, or result root changed"
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
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B public output feedback shape changed"
        )
    return outputs


def _validate_evaluation_oracle(
    value: object,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if type(value) is not tuple or len(value) != 2:
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B evaluation oracle must open both results together"
        )
    return tuple(
        _validate_outputs(outputs, batch=BATCHES[index])
        for index, outputs in enumerate(value, start=TRAIN_COUNT)
    )


def _coded_parent_error_records(stderr: bytes) -> tuple[dict[str, object], ...]:
    try:
        rows = tuple(json.loads(line) for line in stderr.decode("utf-8").splitlines() if line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B rustc stderr is not exact JSON-lines UTF-8"
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
        raise TypeError("C11B diagnostic compiler must return one exact attempt")
    if type(stderr) is not bytes:
        raise TypeError("C11B diagnostic compiler must retain exact stderr bytes")
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
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B diagnostic attempt or parent-message fixture changed"
        )


def _compile_diagnostic_once(
    identity: rust_executor.RustcIdentityV1,
    source: str,
) -> tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes]:
    identity.authenticate_structure()
    source_bytes = source.encode("ascii")
    with tempfile.TemporaryDirectory(prefix="humanbrain-c11b-diagnostic-") as raw_directory:
        directory = Path(raw_directory)
        source_path = directory / SOURCE_FILE_NAME
        source_path.write_bytes(source_bytes)
        completed = subprocess.run(
            [str(identity.path), *DIAGNOSTIC_ARGUMENTS],
            cwd=directory,
            capture_output=True,
            check=False,
            timeout=30,
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
    spec: C11BDiagnosticOccurrenceLocalizationSpecV1,
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
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B selected authority work changed"
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
) -> C11BSelectedExecutionAttemptV1:
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
    partial = C11BSelectedExecutionAttemptV1(**payload, sha256="")
    result = replace(
        partial,
        sha256=_digest(_ATTEMPT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _run_reserved_cohort_v1(
    *,
    spec: C11BDiagnosticOccurrenceLocalizationSpecV1,
    reservation: C11BRunReservationV1,
    diagnostic_compile: DiagnosticCompileCallback,
    execute: ExecutionCallback,
    output_feedback: OutputFeedbackCallback,
) -> C11BDiagnosticOccurrenceLocalizationResultV1:
    spec.authenticate()
    reservation.authenticate(destination=Path(reservation.destination))
    if reservation.spec_sha256 != spec.sha256:
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B reservation/spec custody changed"
        )

    parent = unordered.start_unordered_cue_factor_memory_v1(HYPOTHESES)
    compiled = tuple(
        diagnostic_compile(reservation.rustc_identity_sha256, source)
        for source in ORIGINAL_SOURCES
    )
    if any(type(item) is not tuple or len(item) != 2 for item in compiled):
        raise TypeError("C11B diagnostic compiler must return exact (attempt, stderr) tuples")
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
            "C11B message-line or complete occurrence-roster projection changed",
            message_line_envelopes=tuple(item.record() for item in message_line_envelopes),
            occurrence_rosters=tuple(item.record() for item in occurrence_rosters),
        )

    parent_evaluation_decisions = tuple(parent.route(item) for item in observations[TRAIN_COUNT:])
    if any(item.selected_hypothesis_sha256 is not None for item in parent_evaluation_decisions):
        raise _behavior_failure(
            "C11B parent checkpoint did not STOP on fresh evaluations",
            decisions=tuple(item.record() for item in parent_evaluation_decisions),
        )

    memories = [parent]
    training_normal_decisions: list[unordered.UnorderedCueFactorDecisionV1] = []
    training_pair_decisions: list[
        tuple[unordered.UnorderedCueFactorDecisionV1, unordered.UnorderedCueFactorDecisionV1]
    ] = []
    training_attempt_pairs: list[
        tuple[C11BSelectedExecutionAttemptV1, C11BSelectedExecutionAttemptV1]
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B paired authorities changed projection or began with work"
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
                "C11B explicit operation-fixed occurrence exploration changed",
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
        attempts: list[C11BSelectedExecutionAttemptV1] = []
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
                "C11B adaptation learning-gate control changed",
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
            "C11B final routing or causal-control behavior changed",
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
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B evaluation authority projection changed"
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
    evaluation_attempts: list[C11BSelectedExecutionAttemptV1] = []
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
            "C11B normally selected evaluation occurrence failed public output",
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
    partial = C11BDiagnosticOccurrenceLocalizationResultV1(**payload, sha256="")
    result = replace(
        partial,
        sha256=_digest(_RESULT_NAMESPACE, partial.record(include_sha256=False)),
    )
    result.authenticate()
    return result


def _reserve_destination(
    spec: C11BDiagnosticOccurrenceLocalizationSpecV1,
    *,
    preregistration_commit: str,
    rustc_identity_sha256: str,
    destination: Path,
    reservation_path: Path,
) -> C11BRunReservationV1:
    spec.authenticate(require_frozen_commit=True)
    if not _valid_commit(preregistration_commit):
        raise ValueError("C11B preregistration commit must be 40 lowercase hex characters")
    if not _valid_sha256(rustc_identity_sha256):
        raise ValueError("C11B rustc identity root is invalid")
    destination = Path(destination).resolve()
    reservation_path = Path(reservation_path).resolve()
    expected_reservation = destination.with_name(f"{destination.name}.reservation.json")
    if reservation_path != expected_reservation:
        raise ValueError("C11B reservation must be the result's exact sibling")
    if os.path.lexists(destination) or os.path.lexists(reservation_path):
        raise FileExistsError(destination if os.path.lexists(destination) else reservation_path)
    partial = C11BRunReservationV1(
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
            raise OSError("C11B reservation write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if reservation_path.read_bytes() != payload:
        raise OSError("C11B reservation readback changed")
    return reservation


def _result_document(result: C11BDiagnosticOccurrenceLocalizationResultV1) -> dict[str, object]:
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
    spec: C11BDiagnosticOccurrenceLocalizationSpecV1,
    reservation: C11BRunReservationV1,
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
    spec: C11BDiagnosticOccurrenceLocalizationSpecV1,
    reservation: C11BRunReservationV1,
    error: C11BBehaviorFailure,
) -> dict[str, object]:
    partial_transcript = _plain(error.partial_transcript)
    if type(partial_transcript) is not dict:
        raise TypeError("C11B behavior failure transcript must be one exact object")
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
            raise OSError("C11B terminal document write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    retained = destination.read_bytes()
    if retained != payload or _canonical_json(json.loads(retained.decode("ascii"))) + b"\n" != (
        payload
    ):
        raise OSError("C11B terminal document readback changed")
    decoded = json.loads(retained.decode("ascii"))
    body = dict(decoded)
    document_sha256 = body.pop("document_sha256", None)
    if document_sha256 != _digest(_DOCUMENT_NAMESPACE, body):
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B terminal document root changed"
        )
    if decoded["terminal_status"] == STATUS_PASS:
        record = dict(decoded["authenticated_result_record"])
        result_sha256 = record.pop("sha256", None)
        if result_sha256 != decoded["result_sha256"] or result_sha256 != _digest(
            _RESULT_NAMESPACE, record
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B persisted result root changed"
            )
    elif decoded["terminal_status"] == STATUS_BEHAVIOR_FAIL:
        partial = decoded.get("partial_transcript")
        if type(partial) is not dict or decoded.get("partial_transcript_sha256") != _digest(
            _FAILURE_NAMESPACE, partial
        ):
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B persisted behavior-failure transcript root changed"
            )
    elif decoded["terminal_status"] != STATUS_EXECUTION_ABORT:
        raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
            "C11B terminal status changed"
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
    spec: C11BDiagnosticOccurrenceLocalizationSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C11BDiagnosticOccurrenceLocalizationResultV1:
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
    except C11BBehaviorFailure as error:
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


def run_and_write_c11b_v1(
    *,
    preregistration_commit: str,
    spec: C11BDiagnosticOccurrenceLocalizationSpecV1 = DEFAULT_PUBLIC_SPEC,
) -> C11BDiagnosticOccurrenceLocalizationResultV1:
    """Preflight, reserve, and spend the exact production C11B cohort once."""

    spec.authenticate(require_frozen_commit=True)
    identity = _preflight_rustc_v1()
    destination, reservation_path = _production_evidence_paths_v1()

    def diagnostic_compile(
        rustc_identity_sha256: str,
        source: str,
    ) -> tuple[rustc_feedback.RustcDiagnosticCompileAttemptV1, bytes]:
        if rustc_identity_sha256 != identity.sha256:
            raise ContinualRustDiagnosticOccurrenceLocalizationC11BError(
                "C11B diagnostic compiler identity changed after reservation"
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
    result = run_and_write_c11b_v1(preregistration_commit=arguments.preregistration_commit)
    print(_canonical_json(result.record()).decode("ascii"))


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_PUBLIC_SPEC",
    "STATUS_BEHAVIOR_FAIL",
    "STATUS_EXECUTION_ABORT",
    "STATUS_PASS",
    "C11BBehaviorFailure",
    "C11BDiagnosticOccurrenceLocalizationResultV1",
    "C11BDiagnosticOccurrenceLocalizationSpecV1",
    "C11BPublicConsequenceV1",
    "C11BRunReservationV1",
    "C11BSelectedExecutionAttemptV1",
    "C11BWorkV1",
    "ContinualRustDiagnosticOccurrenceLocalizationC11BError",
    "make_c11b_diagnostic_occurrence_localization_spec_v1",
    "output_sha256_c11b_v1",
    "run_and_write_c11b_v1",
]


if __name__ == "__main__":
    main()
