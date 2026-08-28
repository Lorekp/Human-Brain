"""Pre-outcome protocol manifest for R3 calibration v3.

This module contains no calibration replay.  It freezes the public protocol
surface that future outcome-bearing runners must consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from humanbrain.sdr_union import token_sdr_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE_BYTE_RULE = (
    "sha256(raw Git blob or live file after CRLF-to-LF normalization; lone CR forbidden)"
)
ARTIFACT_ROOT = "experiments/r3_pilot/calibration_v3_artifacts"
THRESHOLD_ARMS = ((10, 13), (16, 18), (18, 20), (20, 20))
DEVELOPMENT_SEEDS = (3, 7, 17, 29)
CONFIRMATION_SEEDS = (43, 59, 83, 101)
FINAL_VERDICT_SEEDS = tuple(dict.fromkeys(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS))
FINAL_PRE_SELECTION_FAILURE_GATES = (
    "electrical_v3",
    "compressed_reserve_q4",
    "full_cohort_q255",
    "r3_f0_development",
    "development_selection",
)
FINAL_POST_SELECTION_FAILURE_GATES = (
    "independent_confirmation",
    "deterministic_confirmation",
)
RESUME_TRANSITION_EVENTS = (529_340, 1_058_681, 1_588_022)
PRODUCTION_TOKEN_IDS = tuple(range(256)) + tuple(range(1024, 1536))
PRODUCTION_TARGET_TOKEN = 32
DISJOINT_TARGET_TOKEN = 2_041
PROJECTION_SEED = 17
COMPRESSED_RESERVE_CONTEXT_IDS = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    57,
    58,
    59,
    60,
    61,
    62,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
)


class ProtocolError(RuntimeError):
    """The preregistration or its source binding is not safe to execute."""


_COMPRESSED_RESERVE_SCHEDULE = (
    (32, "base_full", 8, 32_768),
    (33, "reserve_cohort_1_recruited", 16, 32_856),
    (64, "reserve_cohort_1_full", 16, 32_856),
    (65, "reserve_cohort_2_recruited", 24, 32_944),
    (96, "reserve_cohort_2_full", 24, 32_944),
    (97, "reserve_cohort_3_recruited", 32, 33_032),
    (128, "reserve_cohort_3_full", 32, 33_032),
    (129, "reserve_cohort_4_sentinel_recruited", 40, 33_120),
)


def compressed_reserve_schedule() -> list[dict[str, int | str]]:
    """Return the literal q=4 reserve transition checkpoints."""

    return [
        {
            "event": event,
            "exposure": exposure,
            "reserve_cursor": reserve_cursor,
            "target_column_depth": depth,
        }
        for exposure, event, depth, reserve_cursor in _COMPRESSED_RESERVE_SCHEDULE
    ]


def production_overlap_pairs() -> tuple[tuple[int, int], ...]:
    """Return the balanced actual-codebook preludes for q=255."""

    vocabulary = tuple(
        token_id for token_id in PRODUCTION_TOKEN_IDS if token_id != PRODUCTION_TARGET_TOKEN
    )
    pairs: list[tuple[int, int]] = []
    for exposure in range(2_041):
        lag_block, left_index = divmod(exposure, len(vocabulary))
        right_index = (left_index + lag_block + 1) % len(vocabulary)
        pairs.append((vocabulary[left_index], vocabulary[right_index]))
    return tuple(pairs)


def production_codebook_preimage() -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return the exact token/column rows hashed by the q=255 protocol."""

    return tuple(
        (
            token_id,
            tuple(
                sorted(
                    map(
                        int,
                        token_sdr_indices(
                            token_id,
                            seed=PROJECTION_SEED,
                            n_bits=4_096,
                            w_ones=11,
                        ),
                    )
                )
            ),
        )
        for token_id in PRODUCTION_TOKEN_IDS
    )


def target_disjoint_context_ids() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return frozen learned/novel IDs disjoint from synthetic target 2041."""

    target_columns = set(
        map(
            int,
            token_sdr_indices(
                DISJOINT_TARGET_TOKEN,
                seed=PROJECTION_SEED,
                n_bits=4_096,
                w_ones=11,
            ),
        )
    )
    learned: list[int] = []
    novel: list[int] = []
    token_id = 0
    while len(learned) < 2_041 or len(novel) < 128:
        columns = set(
            map(
                int,
                token_sdr_indices(
                    token_id,
                    seed=PROJECTION_SEED,
                    n_bits=4_096,
                    w_ones=11,
                ),
            )
        )
        if token_id != DISJOINT_TARGET_TOKEN and columns.isdisjoint(target_columns):
            if len(learned) < 2_041:
                learned.append(token_id)
            else:
                novel.append(token_id)
        token_id += 1
    return tuple(learned), tuple(novel)


def _arm_slug(arm: tuple[int, int]) -> str:
    theta_match, theta_active = arm
    return f"theta-m-{theta_match}_theta-a-{theta_active}"


def select_development_arm(verdicts: dict[str, bool]) -> tuple[int, int]:
    """Select the first passing arm without outcome-dependent reordering."""

    expected = {_arm_slug(arm) for arm in THRESHOLD_ARMS}
    if set(verdicts) != expected or any(type(value) is not bool for value in verdicts.values()):
        raise ProtocolError("development selection requires exactly one verdict per arm")
    for arm in THRESHOLD_ARMS:
        if verdicts[_arm_slug(arm)]:
            return arm
    raise ProtocolError("no candidate arm passed every development gate")


def _seed_artifact_path(*parts: str, seed: int) -> str:
    return "/".join((ARTIFACT_ROOT, *parts, f"seed-{seed:010d}.json"))


def artifact_contract() -> dict[str, Any]:
    """Return every possible result path before any outcome is observed."""

    arm_slugs = tuple((_arm_slug(arm), arm) for arm in THRESHOLD_ARMS)

    def arm_seed_grid(path_parts: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, list[str]]:
        return {
            slug: [_seed_artifact_path(*path_parts, slug, seed=seed) for seed in seeds]
            for slug, _arm in arm_slugs
        }

    regimes = (
        "production-overlap-ordered-pairs",
        "target-disjoint-isolation-control",
    )
    deterministic_confirmation_resumes = [
        "/".join(
            (
                ARTIFACT_ROOT,
                "r3-f0",
                "deterministic-confirmation",
                slug,
                "resume",
                f"transition-{transition_event:010d}{extension}",
            )
        )
        for slug, _arm in arm_slugs
        for transition_event in RESUME_TRANSITION_EVENTS
        for extension in (".r3ntcp1", ".resume.json")
    ]
    contract = {
        "root": ARTIFACT_ROOT,
        "write_policy": {
            "checkpoint_api": (
                "humanbrain.native_temporal_runtime.NativeTemporalCircuit.publish_checkpoint"
            ),
            "checkpoint_without_manifest": "manual_STOP",
            "envelope_api": "humanbrain.native_temporal_runtime.publish_immutable_bytes",
            "envelope_max_bytes": 1_048_576,
            "existing_target": "STOP",
            "fallback": False,
            "force_or_overwrite_option": False,
            "manifest_commit_evidence": [
                "same_process_TemporalPublicationOutcome.DURABLE_PUBLISHED_receipt",
                "later_open_resume_TemporalDurabilityOutcome.DURABLE_exact_bytes_"
                "reconciliation_receipt",
            ],
            "manifest_presence_alone_is_commit_evidence": False,
            "mode": "exclusive_durable_publish",
            "overwrite": False,
            "publisher_directory_creation": False,
            "retry": False,
        },
        "path_inventory": {
            "binary_checkpoint_paths": 12,
            "direct_parent_directories": 39,
            "directory_list_canonical_json_bytes": 5_305,
            "directory_list_sha256": (
                "20be5eb10a661a1ec91c693d336a9e024367cb4a2165285eab8fc6f812fd9055"
            ),
            "full_parent_directory_closure": 54,
            "json_paths": 126,
            "resume_path_list_canonical_json_bytes": 3_409,
            "resume_path_list_sha256": (
                "82581f614b7ef2766f0c8264d7ea44f59818bc2baca8661ea241d24400b09bb8"
            ),
            "total_paths": 138,
        },
        "electrical_v3": {
            "development": [
                _seed_artifact_path("electrical-v3", "development", seed=seed)
                for seed in DEVELOPMENT_SEEDS
            ],
            "independent_confirmation": [
                _seed_artifact_path("electrical-v3", "independent-confirmation", seed=seed)
                for seed in CONFIRMATION_SEEDS
            ],
        },
        "compressed_reserve_q4": {
            "development": arm_seed_grid(("q4", "development"), DEVELOPMENT_SEEDS),
            "independent_confirmation": arm_seed_grid(
                ("q4", "independent-confirmation"), CONFIRMATION_SEEDS
            ),
        },
        "full_cohort_q255": {
            "development": {
                regime: arm_seed_grid(("q255", "development", regime), DEVELOPMENT_SEEDS)
                for regime in regimes
            },
            "independent_confirmation": {
                regime: arm_seed_grid(
                    ("q255", "independent-confirmation", regime),
                    CONFIRMATION_SEEDS,
                )
                for regime in regimes
            },
        },
        "r3_f0_train": {
            "development": {
                slug: "/".join(
                    (
                        ARTIFACT_ROOT,
                        "r3-f0",
                        "development",
                        slug,
                        "construction-seed-0000000017.json",
                    )
                )
                for slug, _arm in arm_slugs
            },
            "deterministic_confirmation": {
                slug: "/".join(
                    (
                        ARTIFACT_ROOT,
                        "r3-f0",
                        "deterministic-confirmation",
                        slug,
                        "construction-seed-0000000017.json",
                    )
                )
                for slug, _arm in arm_slugs
            },
            "deterministic_confirmation_resumes": deterministic_confirmation_resumes,
        },
        "selection": {
            "development": (f"{ARTIFACT_ROOT}/selection/development-selection.json"),
            "final": f"{ARTIFACT_ROOT}/selection/final-verdict.json",
        },
    }
    parent_directories = artifact_parent_directories(contract)
    contract["directory_preparation"] = {
        "active_time_directory_creation": "forbidden",
        "authority_source": "exact_frozen_artifact_path_parent_closure",
        "clock_order": "complete_before_first_fresh_or_resume_clock_sample",
        "derivation": (
            "all POSIX parent ancestors of every frozen artifact path, stopping at and "
            "including artifact root"
        ),
        "directories": list(parent_directories),
        "failure_rule": "STOP_before_session_start_or_checkpoint_import",
        "native_api": "humanbrain.native_temporal_runtime.prepare_durable_directory",
        "order_rule": "ascending_component_count_then_UTF-8_bytewise_path",
        "required_outcome": "TemporalDurabilityOutcome.DURABLE",
    }
    contract["resume_admission"] = {
        "exact_bytes": (
            "bounded stable-read manifest bytes accepted by the decoder; no reserialization"
        ),
        "manifest_presence_alone_is_sufficient": False,
        "native_api": ("humanbrain.native_temporal_runtime.reconcile_existing_immutable_bytes"),
        "ordering": "after_resume_clock_start_and_before_checkpoint_open_or_import",
        "persistent_sidecar_receipt": False,
        "required_outcome": "TemporalDurabilityOutcome.DURABLE",
        "same_process_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE_rule": ("terminal_STOP_without_retry"),
        "scope": "every_open_resume",
    }
    contract["filesystem_threat_model"] = {
        "in_scope": [
            "process_crash",
            "power_loss",
            "benign_concurrent_directory_preparation",
            "symlink_or_reparse_point_traversal_and_substitution_detection",
        ],
        "out_of_scope": [
            "malicious_concurrent_namespace_writer_with_repository_write_or_delete_access"
        ],
    }
    return contract


def artifact_paths(contract: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Flatten every exact artifact destination and reject duplicate names."""

    frozen = artifact_contract() if contract is None else contract
    paths: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str) and value.endswith((".json", ".r3ntcp1")):
            paths.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(frozen)
    if len(paths) != len(set(paths)):
        raise ProtocolError("artifact contract contains duplicate paths")
    return tuple(paths)


def artifact_parent_directories(
    contract: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Derive the exact root-first directory authority from frozen artifact paths."""

    frozen = artifact_contract() if contract is None else contract
    root = frozen.get("root")
    if type(root) is not str or not root or root.startswith("/"):
        raise ProtocolError("artifact root must be a non-empty relative POSIX path")
    root_parts = root.split("/")
    if any(not part or part in {".", ".."} for part in root_parts):
        raise ProtocolError("artifact root contains an invalid path component")

    directories: set[str] = set()
    for path in artifact_paths(frozen):
        parts = path.split("/")
        if parts[: len(root_parts)] != root_parts or len(parts) <= len(root_parts):
            raise ProtocolError("artifact path escapes the frozen artifact root")
        if any(not part or part in {".", ".."} for part in parts):
            raise ProtocolError("artifact path contains an invalid path component")
        for component_count in range(len(root_parts), len(parts)):
            directories.add("/".join(parts[:component_count]))

    return tuple(sorted(directories, key=lambda path: (path.count("/"), path.encode("utf-8"))))


def cohort_rounded_depth(context_demand: int) -> int:
    """Return the full-cohort column depth required for one TRAIN demand."""

    if type(context_demand) is not int or context_demand < 0:
        raise ValueError("context_demand must be a non-negative int")
    capacity_per_cohort = 8 * 255
    cohorts = max(1, (context_demand + capacity_per_cohort - 1) // capacity_per_cohort)
    return 8 * cohorts


def capacity_summary() -> dict[str, Any]:
    """Return the schema-seven TRAIN-only capacity binding without opening inputs."""

    return {
        "artifact_id": "r3-relay-pool-capacity-v6-cohort-rounded",
        "artifact_path": "experiments/r3_pilot/relay_pool_capacity.json",
        "provenance": {
            "held_out_bytes_live_verified": False,
            "historical_full_corpus_sha256_not_live_verified": (
                "d585e5644505b34ca516446f9b4fa2e95ee8ffa860d27858a7ca6450ee9ad3aa"
            ),
            "train_corpus_bytes_live_verified": 3_656_777,
            "train_input_authority_sha256": (
                "29304cf71cce83ab34ee8253117043aec8413806ba0047ff850bb3c93a31d966"
            ),
            "train_stream_sha256": (
                "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"
            ),
        },
        "schema_version": 7,
        "scientific_payload": {
            "canonical_hash_rule": (
                "sha256(canonical JSON of relay_pool_capacity without schema_version and scope)"
            ),
            "schema_seven_change_scope": "provenance_only",
            "sha256": "72665d2fc7360cc1dd32367211d9477e64a61e14e059714b0d1d1274297eb7d1",
        },
        "structure": {
            "base_cells_per_column": 8,
            "cohort_cells": 8,
            "column_count": 4_096,
            "conditional_on_gate": "r3_f0_train",
            "configured_physical_cells": 274_432,
            "distinct_train_contexts": 2_116_946,
            "ideal_pyramidal_cells": 109_216,
            "logical_pair_slots": 232_864_060,
            "max_column_depth": 1_128,
            "max_context_demand": 286_874,
            "max_segments_per_cell": 255,
            "physical_contacts": 465_728_120,
            "pyramidal_cells_recruited": 81_632,
            "pyramidal_cells_remaining": 16_672,
            "pyramidal_cells_required": 114_400,
            "recruited_columns": 1_286,
            "rounding_overhead_cells": 5_184,
            "segment_count": 23_286_406,
        },
    }


def _source_binding() -> dict[str, Any]:
    return {
        "canonical_byte_rule": CANONICAL_SOURCE_BYTE_RULE,
        "baseline_commits": {
            "explicit_threshold_oracle": ("c67c6c4b960a26ca8227f78d80da306076d1a931"),
            "procedural_gate3_refreeze": ("600101424df9e934da4773a5cf1046a0c2419e87"),
        },
        "blockers": [],
        "files": {
            "benchmarks/procedural_runtime_gate3_slice_protocol_v1.json": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "b740b1159f5b44bd4f98927668567d2b42ad2a16a282aad113145a07183149fd",
            },
            "experiments/r3_pilot/calibrate_minicolumn_electrical.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "deccc519ab66ac3a53de3c20c60a5a078ca9b9c75038226454aa5af434a17de2",
            },
            "experiments/r3_pilot/calibration_v3_authority.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "d1d790d5cecdb3b0d3b06d7112e6b98434f1c828cc84cb071dae35f9c785cff1",
            },
            "experiments/r3_pilot/calibration_v3_definitions.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "c8fec2ce273726fb9d0de13818153469efb71b4cee762581b1c8155ffac6e950",
            },
            "experiments/r3_pilot/calibration_v3_gate3_authority.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "b4b423e5cc5dbf4f03c51832812815838f204dd15635abbba64d04f712eb6881",
            },
            "experiments/r3_pilot/calibration_v3_overlap_scoring.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "0854f67ab0363d332edfd615ae8b66357d94dd4c22933a1a24290d1a64879d5c",
            },
            "experiments/r3_pilot/calibration_v3_persistence.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "c893bb3c79febfcf82d96af6474c3ce772fcfc8e6b17705d626542de5196b5a3",
            },
            "experiments/r3_pilot/calibration_v3_train_scoring.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "7daddd820c13cc39ec259d52da6a710582903c94b9106f44b644ee70a6d6b234",
            },
            "experiments/r3_pilot/calibration_v3_transaction.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "5542a09fb62a78ffc71d53527d94410c2a2c6054e8336015de3cd595930e5763",
            },
            "experiments/r3_pilot/r3_f0_train_input.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "4837fcfd8aa2baa9f58de95e0ab5bf31674513f20d641d39f6e1ead9b7c1560e",
            },
            "experiments/r3_pilot/r3_f0_train_input_v1.json": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "29304cf71cce83ab34ee8253117043aec8413806ba0047ff850bb3c93a31d966",
            },
            "experiments/r3_pilot/relay_pool_capacity.json": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "45c010fdba9a5a2cb301d70d6e98dd004723aff8d4293aa8f44c7425a74b801b",
            },
            "experiments/r3_pilot/run_calibration_v3_fixtures.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "281b478d5139337ac4553b4a69dfb487cd2695c43152e8a063c6b6e2c2fee188",
            },
            "src/humanbrain/__init__.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "9019d88aa76d762c8fe7a63339f53f7e73849fa2a6658a8a762354eeb2cc67d7",
            },
            "src/humanbrain/concept_tokens.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "9070deaf7d9a8a54add24e3a213d8b6aa7bbb24b3d35ad66a7b68b3d1148f43e",
            },
            "src/humanbrain/cortex.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "7a7704a14324203d2cce427a11fbd58ffbbc345185ba34633449f59cf294624d",
            },
            "src/humanbrain/native_temporal_runtime.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "b7acf7dd103f3b2ef9ef1e8f423daf8ca13e21cabdc16aff6e9b6f8cca82b22a",
            },
            "src/humanbrain/race_bias.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "8d9dff4b65a28447840429d688d551d6b2bb8f963b97dcd95e0420c8c33a0100",
            },
            "src/humanbrain/sdr_union.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "1de7f6edd914584c4c72d7e42426254590049cfb69cbbfa730afa3179afa63f9",
            },
            "src/humanbrain/sparse_temporal_cortex.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "15e44e8250f69ae77774c781a6dee3da5dc8607ddbfd31bfc5fa805e8ffddac0",
            },
            "src/humanbrain/tokenizer.py": {
                "commit": "f8682943a1cfdf6c8655d10c888689c8235c464e",
                "sha256": "08ef114a758006868944ad3091ca66619d3c082086ea2d44d93211d82a9791ce",
            },
        },
        "governing_gate3_path": ("benchmarks/procedural_runtime_gate3_slice_protocol_v1.json"),
        "runnable": True,
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one protocol value using the exact canonical JSON rule."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_source_bytes(raw: bytes) -> bytes:
    """Normalize only checkout line endings before a source SHA-256."""

    without_crlf = raw.replace(b"\r\n", b"\n")
    if b"\r" in without_crlf:
        raise ProtocolError("canonical source contains a lone CR byte")
    return without_crlf


def build_protocol() -> dict[str, Any]:
    """Return the canonical pre-outcome protocol manifest."""

    return {
        "artifact_id": "r3-calibration-v3-preregistration",
        "schema_version": 7,
        "change_notes": [
            "docs/research/r3-calibration-v3-change-note-1.md",
        ],
        "circuit_contract": {
            "active_columns": 11,
            "base_cells_per_column": 8,
            "cohort_cells": 8,
            "column_count": 4_096,
            "contacts_per_segment": 20,
            "context_physical_width": 22,
            "max_pyramidal_cells": 131_072,
            "novel_race_cap": 8,
            "projection_seed": PROJECTION_SEED,
            "source_relay_pairs_per_segment": 10,
        },
        "native_definition_contract": {
            "config": {
                "base_cells_per_column": 8,
                "column_count": 4_096,
                "connected_permanence_ticks": 250,
                "construction_seed_scope": [17],
                "context_policy": "BOUND_WINNERS",
                "initial_permanence_ticks": 255,
                "max_pyramidal_cells": 131_072,
                "max_segments_per_cell": 255,
                "negative_decrement_ticks": 2,
                "novel_race_cap": 8,
                "permanence_increment_ticks": 50,
                "plasticity_policy": "PLASTIC",
                "prediction_quorum": 11,
                "segment_capacity": 23_286_406,
                "theta_arm_scope": [list(arm) for arm in THRESHOLD_ARMS],
            },
            "threshold_offsets_mv": {
                "dtype": "float64",
                "generator": "numpy.random.default_rng(seed).normal",
                "mean_microvolts": 0,
                "ordering": "ascending_physical_gid",
                "physical_cell_count_rule": ("2 * max_pyramidal_cells + 3 * column_count"),
                "seed_source": "construction_seed",
                "standard_deviation_microvolts": 25,
            },
            "token_codebook": {
                "active_columns_per_token": 11,
                "builder": "production_codebook_preimage",
                "canonical_json_sha256": (
                    "388177db5f1d8cba9f0606d5e065148498418b10d27d7abf6117d09b44238a9c"
                ),
                "projection_seed": 17,
                "token_count": 768,
            },
        },
        "execution_status": "RUNNABLE_PRE_OUTCOME",
        "timing_contract": {
            "cap_ns": 54_000_000_000_000,
            "boundaries": {
                "clean_stop": (
                    "after all scientific work and, at a checkpoint, durable checkpoint "
                    "publication plus fresh reopen, full import validation, derived-index "
                    "rebuild, and identity/parity checks; before bounded manifest or result "
                    "publication"
                ),
                "fresh_start": (
                    "after durable preparation of the full authority-derived output-directory "
                    "closure; immediately before the first of production circuit construction "
                    "or production TRAIN-source access"
                ),
                "resume_start": (
                    "after durable preparation of the full authority-derived output-directory "
                    "closure; immediately before opening the exact frozen resume manifest"
                ),
            },
            "cap_check_rule": (
                "check within books and immediately before and after every atomic persistence "
                "operation"
            ),
            "charged_operations": [
                "production runtime allocation/construction and exact TRAIN reads and hashes",
                "simulation, activity transfer, recognition, race, plasticity, evaluation, "
                "scoring, stream-hash updates, and final state identity",
                "checkpoint traversal, encoding, staging copy, durable publication, cleanup, "
                "and parent-directory durability",
                "resume-manifest open and validation plus exact existing-byte file-and-parent "
                "durability reconciliation",
                "fresh checkpoint open/read, full import validation, derived-index rebuild, "
                "and identity/parity checks",
                "asynchronous persistence waits and all incidental elapsed time inside the "
                "closed session",
            ],
            "clock": "CPython 3.11 time.perf_counter_ns",
            "comparison_input": "max(uninterrupted_cumulative_ns,resumed_cumulative_ns)",
            "contract_id": "r3-cumulative-active-wall-v2",
            "cumulative_rule": (
                "first prior_cumulative_ns=0; "
                "cumulative_active_ns=prior_cumulative_ns+session_active_ns exactly once"
            ),
            "excluded_operations": [
                "authority-derived output-directory closure derivation, durable preparation, "
                "and confirmation before the first fresh or resume clock sample",
                "bounded resume-manifest construction, publication, reopen, and confirmation",
                "bounded result-envelope construction, publication, reopen, and confirmation",
            ],
            "implementation_status": "IMPLEMENTED_SOURCE_BOUND_PRE_OUTCOME",
            "integer_rule": "timing fields are exact non-negative integer nanoseconds",
            "session_rule": (
                "one closed active interval with no internal pause; scheduler delay, sleep, "
                "locks, garbage collection, synchronous I/O, asynchronous waits, and incidental "
                "error handling remain charged"
            ),
            "status": "ADOPTED_PRE_OUTCOME",
            "stop_rules": {
                "cap_crossed": (
                    "finish any non-cancellable persistence operation and cleanup, record exact "
                    "overshoot, then STOP"
                ),
                "unclean_or_open_session": "TIMING_INDETERMINATE",
            },
        },
        "execution_order": [
            "electrical_v3_all_development_and_independent_confirmation_seeds",
            "q4_all_candidate_arms_on_development_seeds",
            ("q255_both_source_regimes_for_every_q4_passing_arm_on_development_seeds"),
            "r3_f0_development_fresh_seed17_for_fixture_passing_arms_in_frozen_order",
            "write_development_selection_exclusively_for_first_passing_arm",
            "confirm_selected_arm_q4_and_q255_only_on_independent_seeds",
            (
                "r3_f0_deterministic_confirmation_selected_arm_seed17_second_"
                "fresh_plus_25_50_75_resumes"
            ),
            "reject_v3_without_fallback_if_any_confirmation_gate_fails",
            "write_final_verdict_exclusively",
            "write_CN_2_selected_literals_only_after_confirmed_PASS",
        ],
        "threshold_selection": {
            "candidate_arms": [list(arm) for arm in THRESHOLD_ARMS],
            "confirmation_failure_rule": "reject_v3_without_fallback",
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "independent_confirmation_seeds": list(CONFIRMATION_SEEDS),
            "seed_scope": {
                "electrical_q4_q255_development": list(DEVELOPMENT_SEEDS),
                "electrical_q4_q255_independent_confirmation": list(CONFIRMATION_SEEDS),
                "r3_f0_development_and_deterministic_confirmation": [17],
            },
            "selection_rule": (
                "first listed arm passing every development fixture and the "
                "TRAIN-order R3-F0 structural gate"
            ),
        },
        "artifact_contract": artifact_contract(),
        "screens": {
            "electrical_v3": {
                "basket_domain": {
                    "count": 4_096,
                    "coverage_rule": "per-seed global-worst basket threshold",
                    "ids": [262_144, 266_240],
                },
                "claim_scope": "synthetic_fixture_only",
                "construction_seeds": list(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS),
                "boundary_scenarios_pa": {
                    "active_vs_matching": [650, 639, 590, 543, 496, 449, 402, 355],
                    "matching_vs_novel": [600, 590, 543, 496, 449, 402, 355, 308],
                    "novel_rank_boundary": [550, 542, 496, 449, 402, 355, 308, 261],
                },
                "coupled_race_gate": {
                    "basket_current_pa": 200_000,
                    "basket_to_gaba_ticks": 1,
                    "control_pyramidal_spikes": 2,
                    "dt_us": 50,
                    "gaba_current_pa": 500,
                    "inhibited_pyramidal_spikes": 1,
                    "minimum_natural_latency_strictly_greater_us": 100,
                    "winner_to_basket_ticks": 1,
                },
                "generic_proximal_domain": {
                    "count": 4_096,
                    "current_pa": 570,
                    "ids": [266_240, 270_336],
                },
                "generic_proximal_relay_domain": {
                    "count": 4_096,
                    "current_pa": 20_000,
                    "ids": [270_336, 274_432],
                },
                "generic_proximal_relay_trials": 32_768,
                "pyramidal_domain": {
                    "base_ids": [0, 32_768],
                    "cohort_width": 8,
                    "configured_ids": [0, 131_072],
                    "coverage_rule": (
                        "per-seed global-worst threshold assignment over all "
                        "configured IDs; replay every recruited mapping if "
                        "monotonicity is not proved"
                    ),
                },
                "rank_assignment": (
                    "rank 0 receives the maximum threshold; rank 1 receives the "
                    "minimum; remaining descending currents map to ascending "
                    "thresholds"
                ),
                "same_tick_tie_break": (
                    "descending voltage-minus-threshold margin, then ascending physical GID"
                ),
                "monotonicity_criterion": (
                    "at fixed current and rest state, first-spike tick is "
                    "nondecreasing in threshold offset over the full configured "
                    "domain; otherwise replay every recruited cohort mapping"
                ),
                "relay_claim": (
                    "orchestrated source-then-relay latency ordering, not synaptic causality"
                ),
                "required_verdict": "PASS_RESERVE_VALID_ELECTRICAL_V3",
                "winner_relay_domain": {
                    "count": 131_072,
                    "coverage_rule": "full-domain global-worst singleton",
                    "current_pa": 20_000,
                    "ids": [131_072, 262_144],
                },
            },
            "compressed_reserve_q4": {
                "claim_scope": "synthetic_fixture_only",
                "native_vertical_slice": {
                    "construction_seed": 17,
                    "circuit_identity_sha256": (
                        "1e15d708c81a07c3949097babf94c2c5d7922c3545dac487f8e53e84a3b92220"
                    ),
                    "theta_active": 13,
                    "theta_match": 10,
                },
                "native_circuit_identity_sha256_by_arm_and_seed": {
                    "theta-m-10_theta-a-13": {
                        "construction-seed-0000000003": (
                            "790e907a559baab23753d10dd8d1c8612c080eea7ea3298b0349cf888a537e1a"
                        ),
                        "construction-seed-0000000007": (
                            "0d944f001af3024f5eeaec14a52111ff9d7b7f9c74ed7f9208431dac3a1c87f2"
                        ),
                        "construction-seed-0000000017": (
                            "1e15d708c81a07c3949097babf94c2c5d7922c3545dac487f8e53e84a3b92220"
                        ),
                        "construction-seed-0000000029": (
                            "8076a1fefd6433adbd9e0ad6bb77d68ecbccc253ed648f938b41e6919796a256"
                        ),
                    },
                    "theta-m-16_theta-a-18": {
                        "construction-seed-0000000003": (
                            "7d80d34955b24987e18b094966c6ad6e097b763dd9a7de47b9ed1e7f20fcba08"
                        ),
                        "construction-seed-0000000007": (
                            "c1278adab762e9824d553d392adc3106c8da91842b433911e0c1d1002834b955"
                        ),
                        "construction-seed-0000000017": (
                            "cb0ee4ceb117b5b409f871940be45948234c79514d3654592f4ba97357507708"
                        ),
                        "construction-seed-0000000029": (
                            "d46b35d47d9c6fd4396fd9b0bd26c0d798fc2840f18ef9daf22b9ec95416fb0e"
                        ),
                    },
                    "theta-m-18_theta-a-20": {
                        "construction-seed-0000000003": (
                            "5dab073944f5f0c7dcac3c15f19f1ef3bf80ff286588941edd18fe4e70297560"
                        ),
                        "construction-seed-0000000007": (
                            "ccbdc1936dd338eefe8a391be95bffa87e0cb69acde49861f386271d62cc596b"
                        ),
                        "construction-seed-0000000017": (
                            "976e6be49cc4e8e8edf7710992e6ec49ab80bf7427a1ae04a1f74c71281c7f78"
                        ),
                        "construction-seed-0000000029": (
                            "28761bffb843fdc20ef996c0cc3573921da0def4850727ac1d3c3175d94d253c"
                        ),
                    },
                    "theta-m-20_theta-a-20": {
                        "construction-seed-0000000003": (
                            "d96a7d9c73526757ef74e264b989f2c9e72cc7638d913f84bc1ebb6ddf4010a1"
                        ),
                        "construction-seed-0000000007": (
                            "b1c8c15022d6b0055db883a451b00215e35ba6bf6c655560a2f68b8a5644d67a"
                        ),
                        "construction-seed-0000000017": (
                            "ad9ea9d8a660a22457d67e10054d458b2a86650c44008e6e2a2cfea7b5e3d2ec"
                        ),
                        "construction-seed-0000000029": (
                            "d3c8d459260cd0a6e1d6a54c297b4e490d4c89c178bf750dc0973c82432eeec7"
                        ),
                    },
                },
                "context_id_rule": (
                    "literal first 129 target-disjoint IDs in ascending search order"
                ),
                "context_ids": list(COMPRESSED_RESERVE_CONTEXT_IDS),
                "context_ids_sha256": (
                    "8ad1663264ca08bfa18830f285d65457fdc499393babefbe098f49223c75feef"
                ),
                "episode_steps": [
                    "reset_transient",
                    "recognize_context",
                    "diagnose_context",
                    "learn_target",
                    "reset_transient",
                ],
                "expected_target_segments": 1_419,
                "full_reserve_cohorts_filled": 3,
                "gates": [
                    "all target columns recruit complete eight-cell cohorts atomically",
                    ("no cohort is used before its predecessor reaches q=4 on every cell"),
                    "novel driven candidates never exceed eight",
                    "every novel exposure allocates exactly eleven segments",
                    ("pre-exposure matching active and predicted-column counts are zero"),
                    "fresh and milestone-resumed logical snapshots are identical",
                ],
                "max_segments_per_cell": 4,
                "plastic_steps_per_exposure": 1,
                "sentinel_reserve_cohort": 4,
                "target_exposures": 129,
                "target_token": DISJOINT_TARGET_TOKEN,
                "transition_schedule": compressed_reserve_schedule(),
            },
            "full_cohort_q255": {
                "base_full_exposure": 2_040,
                "claim_scope": "synchronous_fixture_only",
                "disjoint_context_ids_sha256": (
                    "495e9fdf8d501617bc392cfa21ee8c31538fef700aaeef6cd5c446e268d399e3"
                ),
                "disjoint_input_facts": {
                    "learned_contexts": 2_041,
                    "learned_max_column_overlap": 3,
                    "novel_contexts": 128,
                    "novel_internal_max_column_overlap": 2,
                    "novel_vs_learned_max_column_overlap": 2,
                    "target_overlap": 0,
                    "unique_rows": 2_169,
                },
                "expected_reserve_cursor": 32_856,
                "expected_target_segments": 22_451,
                "first_reserve_transition_exposure": 2_041,
                "gates": [
                    "2041 source contexts are unique",
                    "2041 target winner tuples are unique",
                    "maximum winner overlap is below ceil(theta_match/2)",
                    "winner/relay physical overlap is exactly twice winner overlap",
                    ("pre-exposure matching active and predicted-column counts are zero"),
                    "every novel exposure allocates exactly eleven segments",
                    "direct learned-source recurrence selects all eleven stored owners",
                    "128 frozen novel probes produce no false match or prediction",
                    "exposure 2040 leaves every base target owner at load 255",
                    ("exposure 2041 recruits eleven complete eight-cell cohorts atomically"),
                    ("fresh and resumed logical plus recognition-index hashes are identical"),
                ],
                "max_segments_per_cell": 255,
                "production_codebook": {
                    "hash_preimage": (
                        "canonical JSON [[token_id, sorted_column_ids], ...] in "
                        "ascending frozen token-ID order"
                    ),
                    "ids": "0..255 plus 1024..1535",
                    "input_max_column_overlap": 2,
                    "sha256": ("388177db5f1d8cba9f0606d5e065148498418b10d27d7abf6117d09b44238a9c"),
                    "target_token": 32,
                    "token_count": 768,
                },
                "production_episode_steps": [
                    "reset_transient",
                    "recognize_a",
                    "recognize_b",
                    "diagnose_context",
                    "learn_target",
                    "reset_transient",
                ],
                "disjoint_episode_steps": [
                    "reset_transient",
                    "recognize_context",
                    "diagnose_context",
                    "learn_target",
                    "reset_transient",
                ],
                "production_pairs_sha256": (
                    "8ef9805b578c5f55485473f62ff577f0bcf319565f134788ea43902c4dfa7edf"
                ),
                "source_regimes": [
                    "production_overlap_ordered_pairs",
                    "target_disjoint_isolation_control",
                ],
                "target_exposures": 2_041,
            },
            "r3_f0_train": {
                "claim_scope": "sole_c1_scale_identity_collision_gate",
                "construction_seed_scope": {
                    "development": [17],
                    "deterministic_confirmation": [17],
                    "independent_seed_confirmation": ("forbidden_by_governing_C1_protocol"),
                },
                "native_circuit_identity_sha256_by_arm_and_seed": {
                    "theta-m-10_theta-a-13": {
                        "construction-seed-0000000017": (
                            "be6d5470a8cab36cd8a8d52b456c2d7601012a71d4b689999233b6c37e84874a"
                        ),
                    },
                    "theta-m-16_theta-a-18": {
                        "construction-seed-0000000017": (
                            "efcd70c8f0a1deeb04ffa32b7d905f42c01000caa33ed18d61ac366594ed9502"
                        ),
                    },
                    "theta-m-18_theta-a-20": {
                        "construction-seed-0000000017": (
                            "ec92505afdccf245b8b61e8fdbd0dbb52efecd73a29a78c28df6a7bb934aaf80"
                        ),
                    },
                    "theta-m-20_theta-a-20": {
                        "construction-seed-0000000017": (
                            "095b58957ecc2a25a73cebe57c5ee17183936c202d3b2481b67b5e7fb563915a"
                        ),
                    },
                },
                "cohort_invariants": {
                    "cohort_width": 8,
                    "final_max_depth": 1_128,
                    "final_pyramidal_cells": 114_400,
                    "final_recruited_columns": 1_286,
                    "max_cell_load": 255,
                    "partial_cohorts_allowed": False,
                    "prior_cohort_must_fill": True,
                },
                "determinism": {
                    "development_fresh_runs_per_passing_arm": 1,
                    "deterministic_confirmation_fresh_runs_for_selected_arm": 1,
                    "fresh_runs": 2,
                    "resume_after_transition_events": [529_340, 1_058_681, 1_588_022],
                    "resume_fraction_rule": ("floor(total_transition_events * numerator / 4)"),
                    "resume_numerators": [1, 2, 3],
                    "resume_milestones": [
                        {
                            "after_transition_event": 529_340,
                            "book_name": "Joshua",
                            "book_ordinal_zero_based": 5,
                            "event_count": 529_346,
                            "future_book_resets": 43,
                            "last_token_index_zero_based": 42_123,
                            "learning_step": 529_346,
                            "next_token_index_zero_based": 42_124,
                            "remaining_transition_events": 1_588_023,
                            "reset_before_next_token": False,
                        },
                        {
                            "after_transition_event": 1_058_681,
                            "book_name": "2 Chronicles",
                            "book_ordinal_zero_based": 13,
                            "event_count": 1_058_695,
                            "future_book_resets": 35,
                            "last_token_index_zero_based": 82_661,
                            "learning_step": 1_058_695,
                            "next_token_index_zero_based": 82_662,
                            "remaining_transition_events": 1_058_682,
                            "reset_before_next_token": False,
                        },
                        {
                            "after_transition_event": 1_588_022,
                            "book_name": "Jeremiah",
                            "book_ordinal_zero_based": 23,
                            "event_count": 1_588_046,
                            "future_book_resets": 25,
                            "last_token_index_zero_based": 33_073,
                            "learning_step": 1_588_046,
                            "next_token_index_zero_based": 33_074,
                            "remaining_transition_events": 529_341,
                            "reset_before_next_token": False,
                        },
                    ],
                    "snapshot_hash_includes": [
                        "source_and_protocol_bindings",
                        "train_position_and_previous_context",
                        "column_cohorts_and_owner_loads",
                        "segment_owners_contacts_and_permanence_ticks",
                        "learning_and_event_counters",
                        "recognition_index_or_canonical_rebuild_digest",
                        "ordered_winner_tuple_stream",
                    ],
                },
                "identity_invariants": [
                    "distinct temporal keys have injective winner tuples",
                    "repeated temporal keys reproduce the same winner tuple",
                    "winner/relay physical overlap is exactly twice winner overlap",
                    "maximum pair overlap is below ceil(theta_match/2)",
                ],
                "input": {
                    "book_count": 49,
                    "distinct_contexts": 2_116_946,
                    "held_out_access": "forbidden",
                    "segment_count": 23_286_406,
                    "stream": "frozen_C1_TRAIN_allocation_order",
                    "token_count": 2_117_412,
                    "transition_events": 2_117_363,
                    "train_stream_sha256": (
                        "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"
                    ),
                },
                "pass_schedule": {
                    "book_order": "canonical TRAIN order",
                    "book_resets": 49,
                    "passes_per_replay": 1,
                    "per_book_sequence": [
                        "reset_transient",
                        "learn_first_token_as_zero_allocation_context_seed",
                        "learn_each_remaining_token_exactly_once_in_order",
                    ],
                    "tokens": 2_117_412,
                    "transition_events": 2_117_363,
                    "zero_allocation_context_seed_steps": 49,
                },
                "overlap_algorithm": (
                    "bucket winner tuples by current token and reject duplicate "
                    "r-subsets; different production tokens share at most two "
                    "columns"
                ),
                "overlap_rejection_bounds": {
                    "theta_m10": 5,
                    "theta_m16": 8,
                    "theta_m18": 9,
                    "theta_m20": 10,
                },
                "recurrence_invariants": [
                    "an exact stored source may legitimately predict multiple next tokens",
                    "a segment whose stored source differs from W_prev must not match",
                    "a repeated observed transition selects its previously stored owners",
                    "a novel observed transition allocates one segment per active column",
                ],
                "required_verdict": "PASS_R3_F0_TRAIN_STRUCTURE",
            },
        },
        "capacity": capacity_summary(),
        "result_contract": {
            "artifact_creation_requires_runnable_source_binding": True,
            "confirmation_artifacts_must_contain_selected_arm_only": True,
            "development_selection_path": artifact_contract()["selection"]["development"],
            "development_artifacts_must_report_every_executed_arm": True,
            "final_verdict_path": artifact_contract()["selection"]["final"],
            "final_verdict": {
                "construction_seeds": list(FINAL_VERDICT_SEEDS),
                "phase": "final",
                "result_keys": [
                    "selected_arm_slug",
                    "verdict",
                    "first_failed_gate",
                ],
                "screen_id": "threshold_selection",
                "verdicts": {
                    "PASS_R3_CALIBRATION_V3": {
                        "first_failed_gate": None,
                        "selected_arm_slug": "candidate_arm_required",
                        "thresholds": "selected_arm_required",
                    },
                    "REJECT_R3_CALIBRATION_V3": {
                        "post_selection": {
                            "first_failed_gates": list(FINAL_POST_SELECTION_FAILURE_GATES),
                            "selected_arm_slug": "candidate_arm_required",
                            "thresholds": "selected_arm_required",
                        },
                        "pre_selection": {
                            "first_failed_gates": list(FINAL_PRE_SELECTION_FAILURE_GATES),
                            "selected_arm_slug": None,
                            "thresholds": None,
                        },
                    },
                },
            },
            "preterminal_r3_stop": {
                "final_first_failed_gate_by_phase": {
                    "deterministic_confirmation": "deterministic_confirmation",
                    "development": "r3_f0_development",
                },
                "stop_sources": ["native", "transaction", "scorer"],
                "terminal_pass_r3_result": "required",
                "terminal_r3_result_publication": "forbidden",
            },
            "protocol_digest": "sha256(canonical_json(build_protocol()))",
            "envelopes": {
                "outer_keys": ["artifact_id", "payload", "payload_sha256", "schema_version"],
                "result": {
                    "artifact_id": "r3-calibration-v3-result-envelope-v1",
                    "payload_keys": [
                        "artifact_path",
                        "claim_boundary",
                        "construction_seeds",
                        "phase",
                        "protocol_sha256",
                        "result",
                        "screen_id",
                        "source_binding",
                        "source_bundle_identity_sha256",
                        "thresholds",
                    ],
                    "r3_f0_result_keys": [
                        "arm_slug",
                        "comparisons",
                        "kind",
                        "reference",
                        "resumes",
                        "structural_metrics",
                        "train_input",
                        "verdict",
                    ],
                },
                "resume": {
                    "artifact_id": "r3-f0-resume-manifest-v1",
                    "payload_keys": [
                        "arm_slug",
                        "artifact_path",
                        "checkpoint",
                        "checkpoint_publication",
                        "circuit_contract",
                        "claim_boundary",
                        "construction_seed",
                        "cursor",
                        "phase",
                        "previous_state",
                        "protocol_sha256",
                        "reference_trajectory_id",
                        "screen_id",
                        "source_binding",
                        "source_bundle_identity_sha256",
                        "stream_prefixes",
                        "thresholds",
                        "timing",
                        "train_input",
                    ],
                },
            },
            "required_claim_fields": {
                "capability_evidence": False,
                "held_out_data_accessed": False,
                "transfer_evidence": False,
            },
            "required_identity_fields": [
                "artifact_path",
                "protocol_sha256",
                "source_binding",
                "screen_id",
                "phase",
                "construction_seeds",
                "thresholds",
            ],
            "selection_publication_order": [
                "development-selection.json after all development gates",
                "final-verdict.json once after confirmed PASS or the first fail-fast gate",
            ],
            "writes_must_be_exclusive": True,
        },
        "source_binding": _source_binding(),
        "claim_boundary": {
            "capability_evidence": False,
            "held_out_data_accessed": False,
            "transfer_evidence": False,
        },
    }


def protocol_sha256() -> str:
    """Return the result-envelope digest for the exact frozen manifest."""

    return hashlib.sha256(canonical_json_bytes(build_protocol())).hexdigest()


def validate_protocol(value: dict[str, Any], *, require_runnable: bool) -> None:
    """Validate exact preregistration identity and its execution state."""

    if value != build_protocol():
        raise ProtocolError("value does not match the canonical protocol")
    binding = value["source_binding"]
    if binding.get("canonical_byte_rule") != CANONICAL_SOURCE_BYTE_RULE:
        raise ProtocolError("source canonical-byte rule is missing or changed")
    for path, row in binding["files"].items():
        commit = row["commit"]
        digest = row["sha256"]
        if (
            len(commit) != 40
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in commit)
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ProtocolError(f"source binding is incomplete for {path}")
    if binding.get("governing_gate3_path") not in binding["files"]:
        raise ProtocolError("governing Gate3 path is not source-bound")
    artifact_paths(value["artifact_contract"])
    if binding["runnable"] and binding["blockers"]:
        raise ProtocolError("runnable source binding still declares blockers")
    if require_runnable and not binding["runnable"]:
        blockers = "; ".join(binding["blockers"])
        raise ProtocolError(f"protocol is not runnable: {blockers}")


def _bound_source_path(root: Path, relative_path: str) -> Path:
    if (
        not relative_path
        or "\\" in relative_path
        or ":" in relative_path
        or relative_path.startswith("/")
        or any(part in {"", ".", ".."} for part in relative_path.split("/"))
    ):
        raise ProtocolError(f"invalid bound source path: {relative_path}")
    resolved_root = root.resolve()
    path = resolved_root.joinpath(*relative_path.split("/")).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ProtocolError(f"source path escapes repository: {relative_path}") from exc
    return path


def validate_committed_source_blobs(root: Path, binding: dict[str, Any]) -> None:
    """Verify each commit:path blob against the cross-checkout canonical hash."""

    if binding.get("canonical_byte_rule") != CANONICAL_SOURCE_BYTE_RULE:
        raise ProtocolError("source canonical-byte rule is missing or changed")
    resolved_root = root.resolve()
    for relative_path, row in binding["files"].items():
        _bound_source_path(resolved_root, relative_path)
        object_name = f"{row['commit']}:{relative_path}"
        try:
            completed = subprocess.run(
                ["git", "cat-file", "blob", object_name],
                cwd=resolved_root,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise ProtocolError("git is unavailable for commit:path validation") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ProtocolError(f"bound Git blob is unavailable for {object_name}: {detail}")
        observed = hashlib.sha256(canonical_source_bytes(completed.stdout)).hexdigest()
        if observed != row["sha256"]:
            raise ProtocolError(
                f"Git blob hash mismatch for {object_name}: {observed} != {row['sha256']}"
            )


def validate_source_bytes(root: Path, binding: dict[str, Any]) -> None:
    """Fail if any live canonical file differs from its bound Git blob hash."""

    if binding.get("canonical_byte_rule") != CANONICAL_SOURCE_BYTE_RULE:
        raise ProtocolError("source canonical-byte rule is missing or changed")
    for relative_path, row in binding["files"].items():
        path = _bound_source_path(root, relative_path)
        if not path.is_file():
            raise ProtocolError(f"bound source is missing: {relative_path}")
        observed = hashlib.sha256(canonical_source_bytes(path.read_bytes())).hexdigest()
        if observed != row["sha256"]:
            raise ProtocolError(
                f"live source hash drift for {relative_path}: {observed} != {row['sha256']}"
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check-draft", action="store_true")
    actions.add_argument("--check-runnable", action="store_true")
    actions.add_argument("--print-protocol", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    frozen = build_protocol()
    if args.print_protocol:
        print(json.dumps(frozen, indent=2, sort_keys=True))
        return 0
    try:
        validate_protocol(frozen, require_runnable=bool(args.check_runnable))
        validate_committed_source_blobs(REPO_ROOT, frozen["source_binding"])
        if args.check_runnable:
            validate_source_bytes(REPO_ROOT, frozen["source_binding"])
    except ProtocolError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.check_draft:
        print("protocol draft check: PASS (RUNNABLE PRE-OUTCOME)")
    else:
        print("protocol runnable check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
