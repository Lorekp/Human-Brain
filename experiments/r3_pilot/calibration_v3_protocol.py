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
    return {
        "root": ARTIFACT_ROOT,
        "write_policy": {
            "existing_target": "STOP",
            "force_or_overwrite_option": False,
            "mode": "exclusive_create",
            "python_file_mode": "xb",
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
        },
        "selection": {
            "development": (f"{ARTIFACT_ROOT}/selection/development-selection.json"),
            "final": f"{ARTIFACT_ROOT}/selection/final-verdict.json",
        },
    }


def artifact_paths(contract: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Flatten the exact JSON destinations and reject duplicate names."""

    frozen = artifact_contract() if contract is None else contract
    paths: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str) and value.endswith(".json"):
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


def cohort_rounded_depth(context_demand: int) -> int:
    """Return the full-cohort column depth required for one TRAIN demand."""

    if type(context_demand) is not int or context_demand < 0:
        raise ValueError("context_demand must be a non-negative int")
    capacity_per_cohort = 8 * 255
    cohorts = max(1, (context_demand + capacity_per_cohort - 1) // capacity_per_cohort)
    return 8 * cohorts


def capacity_v6_summary() -> dict[str, Any]:
    """Return static cohort-rounded sizing; no corpus or outcome is opened."""

    return {
        "base_cells_per_column": 8,
        "cohort_cells": 8,
        "column_count": 4_096,
        "conditional_on_gate": "r3_f0_train",
        "configured_physical_cells": 274_432,
        "distinct_train_contexts": 2_116_946,
        "ideal_pyramidal_cells": 109_216,
        "lazy_committed_physical_cells_projection": 241_088,
        "lazy_projection_implemented": False,
        "max_column_depth": 1_128,
        "max_context_demand": 286_874,
        "max_segments_per_cell": 255,
        "native_temporal_memory": {
            "a1": {
                "index_bytes": 1_864_485_344,
                "requested_payload_bytes": 3_590_431_900,
                "source_scratch_bytes": 524_288,
            },
            "a3": {
                "index_bytes": 1_862_961_632,
                "requested_payload_bytes": 3_588_400_284,
                "source_scratch_bytes": 16_384,
            },
            "allocator_metadata_excluded": True,
            "claim_scope": "implemented_intrusive_variable_live_pair_payload",
            "fitted_physics_excluded": True,
            "learned_soa_bytes": 1_490_461_056,
            "mutation_transaction_checkpoint_runtime_implemented": False,
            "owner_scratch_bytes": 2_097_152,
            "python_diagnostics_bytes_excluded": 2_097_152,
            "segment_scratch_bytes": 232_864_060,
            "source_commit": "3dd2c4fa9eefb7ae0d99da2d5363053ba540a838",
            "store_header_bytes_excluded": 432,
        },
        "pyramidal_cells_remaining": 16_672,
        "pyramidal_cells_required": 114_400,
        "recruited_columns": 1_286,
        "rounding_overhead_cells": 5_184,
        "segment_count": 23_286_406,
        "train_stream_sha256": ("f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"),
    }


def _source_binding() -> dict[str, Any]:
    return {
        "canonical_byte_rule": CANONICAL_SOURCE_BYTE_RULE,
        "baseline_commits": {
            "explicit_threshold_oracle": ("c67c6c4b960a26ca8227f78d80da306076d1a931"),
            "native_random123": "15963daf1ea74e9e3a2a41c7168ac738d3b61e2e",
            "native_temporal_physics": "612b2f55373550f6abd39a6b89d9f6cd978cd7eb",
            "native_temporal_store": "3dd2c4fa9eefb7ae0d99da2d5363053ba540a838",
            "procedural_gate3_refreeze": ("1a262b6d11c0c14bd82c4923e8499befb169576c"),
        },
        "blockers": [
            ("native temporal mutation/integration runner is not implemented and source-bound"),
            "outcome-bearing electrical, q4, q255, and R3-F0 runners remain disabled",
        ],
        "files": {
            "benchmarks/procedural_runtime_gate3_slice_protocol_v1.json": {
                "commit": "1a262b6d11c0c14bd82c4923e8499befb169576c",
                "sha256": ("0adecfa53b94ca4f47294e307849b9ae7dd4b6d0b573a4352be5574d6c32d534"),
            },
            "experiments/r3_pilot/relay_pool_capacity.json": {
                "commit": "a745826f50c7a5a65c2885f5f66de78d44dfdb74",
                "sha256": ("ed316960a1088f936fcb4267107b8947af01f1ac736214839233d7cc27e8a844"),
            },
            "src/humanbrain/_native_src/src/lib.rs": {
                "commit": "612b2f55373550f6abd39a6b89d9f6cd978cd7eb",
                "sha256": ("02ed78dcb7979264a5ea1be2be2308d075e9031e0dac129860ca8c7280cdf707"),
            },
            "src/humanbrain/_native_src/src/procedural.rs": {
                "commit": "15963daf1ea74e9e3a2a41c7168ac738d3b61e2e",
                "sha256": ("7c0f83c9ac5825d4d36784edb27544edd4ad90e9b74967b0c7b1d6e55088e48f"),
            },
            "src/humanbrain/_native_src/src/random123.rs": {
                "commit": "15963daf1ea74e9e3a2a41c7168ac738d3b61e2e",
                "sha256": ("0a229d96f549ae12f4dc8d7b6765a129fdd3cd0834a4aca0403e80d3a2397435"),
            },
            "src/humanbrain/_native_src/src/temporal.rs": {
                "commit": "3dd2c4fa9eefb7ae0d99da2d5363053ba540a838",
                "sha256": ("e9ecb27710ec6ca27ac7f32a6ae9b534ffdca0c3208cb48c1eb858474bae7ef9"),
            },
            "src/humanbrain/_native_src/src/temporal_physics.rs": {
                "commit": "612b2f55373550f6abd39a6b89d9f6cd978cd7eb",
                "sha256": ("41048e560ecb80c67f0e822edad12133b4e37c9c613933c796765521551df511"),
            },
            "src/humanbrain/cells.py": {
                "commit": "c323575299637a1e34141277612e6023c6b9e617",
                "sha256": ("5ca05a64a89f930db3ea351e3e236e3242bf21652fff127747a36d9ca488d2e6"),
            },
            "src/humanbrain/cortex.py": {
                "commit": "6a153c7a2bea5636035e144494ec929f60f50682",
                "sha256": ("16545b390c516b6f5817366946f0f516cec5b049bb9e1f6021184d9e156e0a8c"),
            },
            "src/humanbrain/native_temporal.py": {
                "commit": "3dd2c4fa9eefb7ae0d99da2d5363053ba540a838",
                "sha256": ("d9034d542755a3747d588e15cf89f3455a265fb6a89c9afd002262d620ffd84b"),
            },
            "src/humanbrain/sdr_union.py": {
                "commit": "40a29341dadc3bac903bf62b2c68763f5f5c493f",
                "sha256": ("1de7f6edd914584c4c72d7e42426254590049cfb69cbbfa730afa3179afa63f9"),
            },
            "src/humanbrain/sparse_temporal_cortex.py": {
                "commit": "c67c6c4b960a26ca8227f78d80da306076d1a931",
                "sha256": ("22e5ab6bb9e3b1bc7010e8c3ffd11d9c2dd9ea01a33ba15046a32db2228357a8"),
            },
        },
        "runnable": False,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


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
        "schema_version": 3,
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
        "execution_status": "NOT_RUNNABLE_PRE_OUTCOME",
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
                    "active_vs_matching": [650, 640, 590, 543, 496, 449, 402, 355],
                    "matching_vs_novel": [600, 590, 543, 496, 449, 402, 355, 308],
                    "novel_rank_boundary": [550, 543, 496, 449, 402, 355, 308, 261],
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
        "capacity_v6": capacity_v6_summary(),
        "result_contract": {
            "artifact_creation_requires_runnable_source_binding": True,
            "confirmation_artifacts_must_contain_selected_arm_only": True,
            "development_selection_path": artifact_contract()["selection"]["development"],
            "development_artifacts_must_report_every_executed_arm": True,
            "final_verdict_path": artifact_contract()["selection"]["final"],
            "protocol_digest": "sha256(canonical_json(build_protocol()))",
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
                "final-verdict.json after all confirmation gates",
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

    return hashlib.sha256(_canonical_json_bytes(build_protocol())).hexdigest()


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
        print("protocol draft check: PASS (NOT RUNNABLE)")
    else:  # pragma: no cover - unreachable until the fail-closed flag is refrozen.
        print("protocol runnable check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
