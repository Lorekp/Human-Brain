from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
from experiments.r3_pilot import calibration_v3_protocol as protocol
from humanbrain.sdr_union import token_sdr_indices

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "r3_pilot" / "calibration_v3_protocol.py"
EXPECTED_RESUME_PATHS = (
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-10_theta-a-13/resume/transition-0000529340.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-10_theta-a-13/resume/transition-0000529340.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-10_theta-a-13/resume/transition-0001058681.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-10_theta-a-13/resume/transition-0001058681.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-10_theta-a-13/resume/transition-0001588022.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-10_theta-a-13/resume/transition-0001588022.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-16_theta-a-18/resume/transition-0000529340.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-16_theta-a-18/resume/transition-0000529340.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-16_theta-a-18/resume/transition-0001058681.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-16_theta-a-18/resume/transition-0001058681.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-16_theta-a-18/resume/transition-0001588022.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-16_theta-a-18/resume/transition-0001588022.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-18_theta-a-20/resume/transition-0000529340.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-18_theta-a-20/resume/transition-0000529340.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-18_theta-a-20/resume/transition-0001058681.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-18_theta-a-20/resume/transition-0001058681.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-18_theta-a-20/resume/transition-0001588022.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-18_theta-a-20/resume/transition-0001588022.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-20_theta-a-20/resume/transition-0000529340.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-20_theta-a-20/resume/transition-0000529340.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-20_theta-a-20/resume/transition-0001058681.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-20_theta-a-20/resume/transition-0001058681.resume.json",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-20_theta-a-20/resume/transition-0001588022.r3ntcp1",
    "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/deterministic-confirmation/theta-m-20_theta-a-20/resume/transition-0001588022.resume.json",
)


def test_protocol_separates_fixture_selection_from_the_train_scale_gate() -> None:
    frozen = protocol.build_protocol()

    assert frozen["artifact_id"] == "r3-calibration-v3-preregistration"
    assert frozen["schema_version"] == 6
    assert frozen["change_notes"] == ["docs/research/r3-calibration-v3-change-note-1.md"]
    assert frozen["circuit_contract"] == {
        "active_columns": 11,
        "base_cells_per_column": 8,
        "cohort_cells": 8,
        "column_count": 4096,
        "contacts_per_segment": 20,
        "context_physical_width": 22,
        "max_pyramidal_cells": 131072,
        "novel_race_cap": 8,
        "projection_seed": 17,
        "source_relay_pairs_per_segment": 10,
    }
    assert frozen["threshold_selection"] == {
        "candidate_arms": [[10, 13], [16, 18], [18, 20], [20, 20]],
        "confirmation_failure_rule": "reject_v3_without_fallback",
        "development_seeds": [3, 7, 17, 29],
        "independent_confirmation_seeds": [43, 59, 83, 101],
        "seed_scope": {
            "electrical_q4_q255_development": [3, 7, 17, 29],
            "electrical_q4_q255_independent_confirmation": [43, 59, 83, 101],
            "r3_f0_development_and_deterministic_confirmation": [17],
        },
        "selection_rule": (
            "first listed arm passing every development fixture and the "
            "TRAIN-order R3-F0 structural gate"
        ),
    }
    assert frozen["screens"]["compressed_reserve_q4"]["claim_scope"] == ("synthetic_fixture_only")
    assert frozen["screens"]["full_cohort_q255"]["claim_scope"] == ("synchronous_fixture_only")
    assert frozen["screens"]["r3_f0_train"]["claim_scope"] == (
        "sole_c1_scale_identity_collision_gate"
    )
    assert frozen["claim_boundary"]["held_out_data_accessed"] is False
    assert frozen["claim_boundary"]["capability_evidence"] is False


def test_q4_schedule_forces_three_complete_reserve_cohorts_and_a_sentinel() -> None:
    assert protocol.compressed_reserve_schedule() == [
        {
            "event": "base_full",
            "exposure": 32,
            "reserve_cursor": 32768,
            "target_column_depth": 8,
        },
        {
            "event": "reserve_cohort_1_recruited",
            "exposure": 33,
            "reserve_cursor": 32856,
            "target_column_depth": 16,
        },
        {
            "event": "reserve_cohort_1_full",
            "exposure": 64,
            "reserve_cursor": 32856,
            "target_column_depth": 16,
        },
        {
            "event": "reserve_cohort_2_recruited",
            "exposure": 65,
            "reserve_cursor": 32944,
            "target_column_depth": 24,
        },
        {
            "event": "reserve_cohort_2_full",
            "exposure": 96,
            "reserve_cursor": 32944,
            "target_column_depth": 24,
        },
        {
            "event": "reserve_cohort_3_recruited",
            "exposure": 97,
            "reserve_cursor": 33032,
            "target_column_depth": 32,
        },
        {
            "event": "reserve_cohort_3_full",
            "exposure": 128,
            "reserve_cursor": 33032,
            "target_column_depth": 32,
        },
        {
            "event": "reserve_cohort_4_sentinel_recruited",
            "exposure": 129,
            "reserve_cursor": 33120,
            "target_column_depth": 40,
        },
    ]

    screen = protocol.build_protocol()["screens"]["compressed_reserve_q4"]
    assert screen["native_vertical_slice"] == {
        "construction_seed": 17,
        "circuit_identity_sha256": (
            "1e15d708c81a07c3949097babf94c2c5d7922c3545dac487f8e53e84a3b92220"
        ),
        "theta_active": 13,
        "theta_match": 10,
    }
    assert screen["max_segments_per_cell"] == 4
    assert screen["target_exposures"] == 129
    assert screen["target_token"] == 2041
    assert screen["context_id_rule"] == (
        "literal first 129 target-disjoint IDs in ascending search order"
    )
    assert screen["context_ids"] == list(protocol.COMPRESSED_RESERVE_CONTEXT_IDS)
    assert tuple(screen["context_ids"]) == protocol.target_disjoint_context_ids()[0][:129]
    assert len(screen["context_ids"]) == 129
    assert screen["context_ids"][-1] == 133
    assert screen["context_ids_sha256"] == (
        "8ad1663264ca08bfa18830f285d65457fdc499393babefbe098f49223c75feef"
    )
    assert _canonical_sha256(screen["context_ids"]) == screen["context_ids_sha256"]
    assert screen["plastic_steps_per_exposure"] == 1
    assert screen["episode_steps"] == [
        "reset_transient",
        "recognize_context",
        "diagnose_context",
        "learn_target",
        "reset_transient",
    ]
    assert screen["full_reserve_cohorts_filled"] == 3
    assert screen["sentinel_reserve_cohort"] == 4
    assert screen["expected_target_segments"] == 1419


def test_q255_source_regimes_are_frozen_without_temporal_outcomes() -> None:
    production_pairs = protocol.production_overlap_pairs()
    production_codebook = protocol.production_codebook_preimage()
    disjoint_contexts, disjoint_novel = protocol.target_disjoint_context_ids()

    assert len(production_pairs) == 2041
    assert len(set(production_pairs)) == 2041
    assert all(left != right for left, right in production_pairs)
    assert production_pairs[:2] == ((0, 1), (1, 2))
    assert production_pairs[766] == (1535, 0)
    assert production_pairs[767] == (0, 2)
    assert production_pairs[1534] == (0, 3)
    assert production_pairs[-1] == (1275, 1278)
    assert _canonical_sha256(production_pairs) == (
        "8ef9805b578c5f55485473f62ff577f0bcf319565f134788ea43902c4dfa7edf"
    )
    assert len(production_codebook) == 768
    assert production_codebook[0] == (
        0,
        (363, 535, 752, 1327, 1601, 1992, 2130, 2438, 3144, 3682, 4054),
    )
    assert production_codebook[-1] == (
        1535,
        (658, 854, 1530, 1623, 2268, 2428, 3244, 3510, 3696, 3744, 4031),
    )
    assert all(columns == tuple(sorted(columns)) for _, columns in production_codebook)
    assert _canonical_sha256(production_codebook) == (
        "388177db5f1d8cba9f0606d5e065148498418b10d27d7abf6117d09b44238a9c"
    )

    target_columns = set(
        map(
            int,
            token_sdr_indices(2041, seed=17, n_bits=4096, w_ones=11),
        )
    )
    assert len(disjoint_contexts) == 2041
    assert len(disjoint_novel) == 128
    assert len(set(disjoint_contexts + disjoint_novel)) == 2169
    assert disjoint_contexts[-1] == 2101
    assert disjoint_novel[0] == 2102
    assert disjoint_novel[-1] == 2233
    for token_id in disjoint_contexts + disjoint_novel:
        columns = set(
            map(
                int,
                token_sdr_indices(token_id, seed=17, n_bits=4096, w_ones=11),
            )
        )
        assert columns.isdisjoint(target_columns)
    assert (
        _canonical_sha256(
            {
                "learned": disjoint_contexts,
                "novel": disjoint_novel,
                "target": 2041,
            }
        )
        == "495e9fdf8d501617bc392cfa21ee8c31538fef700aaeef6cd5c446e268d399e3"
    )

    screen = protocol.build_protocol()["screens"]["full_cohort_q255"]
    assert screen["max_segments_per_cell"] == 255
    assert screen["target_exposures"] == 2041
    assert screen["base_full_exposure"] == 2040
    assert screen["first_reserve_transition_exposure"] == 2041
    assert screen["expected_target_segments"] == 22451
    assert screen["expected_reserve_cursor"] == 32856
    assert screen["source_regimes"] == [
        "production_overlap_ordered_pairs",
        "target_disjoint_isolation_control",
    ]


def test_capacity_binds_schema_seven_train_only_provenance_and_unchanged_structure() -> None:
    assert [
        protocol.cohort_rounded_depth(demand) for demand in (0, 1, 2040, 2041, 4080, 4081, 286874)
    ] == [8, 8, 8, 16, 16, 24, 1128]

    assert protocol.capacity_summary() == {
        "artifact_id": "r3-relay-pool-capacity-v6-cohort-rounded",
        "artifact_path": "experiments/r3_pilot/relay_pool_capacity.json",
        "provenance": {
            "held_out_bytes_live_verified": False,
            "historical_full_corpus_sha256_not_live_verified": (
                "d585e5644505b34ca516446f9b4fa2e95ee8ffa860d27858a7ca6450ee9ad3aa"
            ),
            "train_corpus_bytes_live_verified": 3656777,
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
            "column_count": 4096,
            "conditional_on_gate": "r3_f0_train",
            "configured_physical_cells": 274432,
            "distinct_train_contexts": 2116946,
            "ideal_pyramidal_cells": 109216,
            "logical_pair_slots": 232864060,
            "max_column_depth": 1128,
            "max_context_demand": 286874,
            "max_segments_per_cell": 255,
            "physical_contacts": 465728120,
            "pyramidal_cells_recruited": 81632,
            "pyramidal_cells_remaining": 16672,
            "pyramidal_cells_required": 114400,
            "recruited_columns": 1286,
            "rounding_overhead_cells": 5184,
            "segment_count": 23286406,
        },
    }

    capacity = protocol.build_protocol()["capacity"]
    assert capacity == protocol.capacity_summary()
    assert capacity["structure"]["pyramidal_cells_required"] < 131072
    assert "capacity_v6" not in protocol.build_protocol()
    assert not hasattr(protocol, "capacity_v6_summary")

    artifact = json.loads(
        (ROOT / "experiments" / "r3_pilot" / "relay_pool_capacity.json").read_text(encoding="utf-8")
    )
    scientific_payload = dict(artifact)
    scientific_payload.pop("schema_version")
    scientific_payload.pop("scope")
    assert artifact["artifact_id"] == capacity["artifact_id"]
    assert artifact["schema_version"] == capacity["schema_version"]
    assert _canonical_sha256(scientific_payload) == capacity["scientific_payload"]["sha256"]
    assert {key: artifact["scope"][key] for key in capacity["provenance"]} == capacity["provenance"]
    assert capacity["artifact_path"] in protocol.build_protocol()["source_binding"]["files"]


def test_electrical_v3_covers_reserve_and_winner_relay_domains() -> None:
    screen = protocol.build_protocol()["screens"]["electrical_v3"]

    assert screen["claim_scope"] == "synthetic_fixture_only"
    assert screen["construction_seeds"] == [3, 7, 17, 29, 43, 59, 83, 101]
    assert screen["pyramidal_domain"] == {
        "base_ids": [0, 32768],
        "configured_ids": [0, 131072],
        "cohort_width": 8,
        "coverage_rule": (
            "per-seed global-worst threshold assignment over all configured IDs; "
            "replay every recruited mapping if monotonicity is not proved"
        ),
    }
    assert screen["basket_domain"] == {
        "count": 4096,
        "ids": [262144, 266240],
        "coverage_rule": "per-seed global-worst basket threshold",
    }
    assert screen["winner_relay_domain"] == {
        "count": 131072,
        "ids": [131072, 262144],
        "current_pa": 20000,
        "coverage_rule": "full-domain global-worst singleton",
    }
    assert screen["generic_proximal_domain"] == {
        "count": 4096,
        "ids": [266240, 270336],
        "current_pa": 570,
    }
    assert screen["generic_proximal_relay_domain"] == {
        "count": 4096,
        "ids": [270336, 274432],
        "current_pa": 20000,
    }
    assert screen["generic_proximal_relay_trials"] == 32768
    assert screen["boundary_scenarios_pa"] == {
        "active_vs_matching": [650, 639, 590, 543, 496, 449, 402, 355],
        "matching_vs_novel": [600, 590, 543, 496, 449, 402, 355, 308],
        "novel_rank_boundary": [550, 542, 496, 449, 402, 355, 308, 261],
    }
    assert screen["coupled_race_gate"] == {
        "basket_current_pa": 200000,
        "basket_to_gaba_ticks": 1,
        "control_pyramidal_spikes": 2,
        "dt_us": 50,
        "gaba_current_pa": 500,
        "inhibited_pyramidal_spikes": 1,
        "minimum_natural_latency_strictly_greater_us": 100,
        "winner_to_basket_ticks": 1,
    }
    assert screen["rank_assignment"] == (
        "rank 0 receives the maximum threshold; rank 1 receives the minimum; "
        "remaining descending currents map to ascending thresholds"
    )
    assert screen["same_tick_tie_break"] == (
        "descending voltage-minus-threshold margin, then ascending physical GID"
    )
    assert screen["monotonicity_criterion"] == (
        "at fixed current and rest state, first-spike tick is nondecreasing in "
        "threshold offset over the full configured domain; otherwise replay "
        "every recruited cohort mapping"
    )
    assert screen["relay_claim"] == (
        "orchestrated source-then-relay latency ordering, not synaptic causality"
    )
    assert screen["required_verdict"] == "PASS_RESERVE_VALID_ELECTRICAL_V3"


def test_r3_f0_is_the_train_order_identity_and_collision_gate() -> None:
    screen = protocol.build_protocol()["screens"]["r3_f0_train"]

    assert screen["input"] == {
        "book_count": 49,
        "distinct_contexts": 2116946,
        "held_out_access": "forbidden",
        "segment_count": 23286406,
        "stream": "frozen_C1_TRAIN_allocation_order",
        "token_count": 2117412,
        "transition_events": 2117363,
        "train_stream_sha256": ("f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"),
    }
    assert screen["construction_seed_scope"] == {
        "development": [17],
        "deterministic_confirmation": [17],
        "independent_seed_confirmation": "forbidden_by_governing_C1_protocol",
    }
    assert screen["pass_schedule"] == {
        "book_order": "canonical TRAIN order",
        "book_resets": 49,
        "passes_per_replay": 1,
        "per_book_sequence": [
            "reset_transient",
            "learn_first_token_as_zero_allocation_context_seed",
            "learn_each_remaining_token_exactly_once_in_order",
        ],
        "tokens": 2117412,
        "transition_events": 2117363,
        "zero_allocation_context_seed_steps": 49,
    }
    assert screen["overlap_rejection_bounds"] == {
        "theta_m10": 5,
        "theta_m16": 8,
        "theta_m18": 9,
        "theta_m20": 10,
    }
    assert screen["overlap_algorithm"] == (
        "bucket winner tuples by current token and reject duplicate r-subsets; "
        "different production tokens share at most two columns"
    )
    assert screen["identity_invariants"] == [
        "distinct temporal keys have injective winner tuples",
        "repeated temporal keys reproduce the same winner tuple",
        "winner/relay physical overlap is exactly twice winner overlap",
        "maximum pair overlap is below ceil(theta_match/2)",
    ]
    assert screen["recurrence_invariants"] == [
        "an exact stored source may legitimately predict multiple next tokens",
        "a segment whose stored source differs from W_prev must not match",
        "a repeated observed transition selects its previously stored owners",
        "a novel observed transition allocates one segment per active column",
    ]
    assert screen["cohort_invariants"] == {
        "cohort_width": 8,
        "final_max_depth": 1128,
        "final_pyramidal_cells": 114400,
        "final_recruited_columns": 1286,
        "max_cell_load": 255,
        "partial_cohorts_allowed": False,
        "prior_cohort_must_fill": True,
    }
    assert screen["determinism"] == {
        "fresh_runs": 2,
        "development_fresh_runs_per_passing_arm": 1,
        "deterministic_confirmation_fresh_runs_for_selected_arm": 1,
        "resume_after_transition_events": [529340, 1058681, 1588022],
        "resume_fraction_rule": "floor(total_transition_events * numerator / 4)",
        "resume_numerators": [1, 2, 3],
        "resume_milestones": [
            {
                "after_transition_event": 529340,
                "book_name": "Joshua",
                "book_ordinal_zero_based": 5,
                "event_count": 529346,
                "future_book_resets": 43,
                "last_token_index_zero_based": 42123,
                "learning_step": 529346,
                "next_token_index_zero_based": 42124,
                "remaining_transition_events": 1588023,
                "reset_before_next_token": False,
            },
            {
                "after_transition_event": 1058681,
                "book_name": "2 Chronicles",
                "book_ordinal_zero_based": 13,
                "event_count": 1058695,
                "future_book_resets": 35,
                "last_token_index_zero_based": 82661,
                "learning_step": 1058695,
                "next_token_index_zero_based": 82662,
                "remaining_transition_events": 1058682,
                "reset_before_next_token": False,
            },
            {
                "after_transition_event": 1588022,
                "book_name": "Jeremiah",
                "book_ordinal_zero_based": 23,
                "event_count": 1588046,
                "future_book_resets": 25,
                "last_token_index_zero_based": 33073,
                "learning_step": 1588046,
                "next_token_index_zero_based": 33074,
                "remaining_transition_events": 529341,
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
    }
    assert screen["required_verdict"] == "PASS_R3_F0_TRAIN_STRUCTURE"


def test_timing_contract_is_adopted_without_execution_authority() -> None:
    frozen = protocol.build_protocol()

    assert frozen["timing_contract"] == {
        "cap_ns": 54000000000000,
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
            "check within books and immediately before and after every atomic persistence operation"
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
        "implementation_status": "DISABLED_BEHIND_EXECUTE_FROZEN_PLAN",
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
    }
    assert frozen["source_binding"]["runnable"] is False
    assert frozen["execution_status"] == "NOT_RUNNABLE_PRE_OUTCOME"


def test_artifact_paths_are_exact_unique_and_exclusive() -> None:
    contract = protocol.artifact_contract()

    assert contract["root"] == "experiments/r3_pilot/calibration_v3_artifacts"
    assert contract["write_policy"] == {
        "checkpoint_api": (
            "humanbrain.native_temporal_runtime.NativeTemporalCircuit.publish_checkpoint"
        ),
        "checkpoint_without_manifest": "manual_STOP",
        "envelope_api": "humanbrain.native_temporal_runtime.publish_immutable_bytes",
        "envelope_max_bytes": 1048576,
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
    }
    assert contract["electrical_v3"]["development"][0] == (
        "experiments/r3_pilot/calibration_v3_artifacts/electrical-v3/"
        "development/seed-0000000003.json"
    )
    assert contract["compressed_reserve_q4"]["development"]["theta-m-10_theta-a-13"][0] == (
        "experiments/r3_pilot/calibration_v3_artifacts/q4/development/"
        "theta-m-10_theta-a-13/seed-0000000003.json"
    )
    assert contract["full_cohort_q255"]["independent_confirmation"][
        "target-disjoint-isolation-control"
    ]["theta-m-20_theta-a-20"][-1] == (
        "experiments/r3_pilot/calibration_v3_artifacts/q255/"
        "independent-confirmation/target-disjoint-isolation-control/"
        "theta-m-20_theta-a-20/seed-0000000101.json"
    )
    assert contract["r3_f0_train"]["development"]["theta-m-18_theta-a-20"] == (
        "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/development/"
        "theta-m-18_theta-a-20/construction-seed-0000000017.json"
    )
    assert contract["r3_f0_train"]["deterministic_confirmation"]["theta-m-18_theta-a-20"] == (
        "experiments/r3_pilot/calibration_v3_artifacts/r3-f0/"
        "deterministic-confirmation/theta-m-18_theta-a-20/"
        "construction-seed-0000000017.json"
    )
    assert tuple(contract["r3_f0_train"]["deterministic_confirmation_resumes"]) == (
        EXPECTED_RESUME_PATHS
    )
    resume_preimage = json.dumps(
        contract["r3_f0_train"]["deterministic_confirmation_resumes"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert len(resume_preimage) == 3409
    assert hashlib.sha256(resume_preimage).hexdigest() == (
        "82581f614b7ef2766f0c8264d7ea44f59818bc2baca8661ea241d24400b09bb8"
    )
    assert contract["path_inventory"] == {
        "binary_checkpoint_paths": 12,
        "direct_parent_directories": 39,
        "directory_list_canonical_json_bytes": 5305,
        "directory_list_sha256": (
            "20be5eb10a661a1ec91c693d336a9e024367cb4a2165285eab8fc6f812fd9055"
        ),
        "full_parent_directory_closure": 54,
        "json_paths": 126,
        "resume_path_list_canonical_json_bytes": 3409,
        "resume_path_list_sha256": (
            "82581f614b7ef2766f0c8264d7ea44f59818bc2baca8661ea241d24400b09bb8"
        ),
        "total_paths": 138,
    }
    assert contract["selection"] == {
        "development": (
            "experiments/r3_pilot/calibration_v3_artifacts/selection/development-selection.json"
        ),
        "final": ("experiments/r3_pilot/calibration_v3_artifacts/selection/final-verdict.json"),
    }

    paths = protocol.artifact_paths(contract)
    assert len(paths) == 138
    assert len(set(paths)) == len(paths)
    assert sum(path.endswith(".json") for path in paths) == 126
    assert sum(path.endswith(".r3ntcp1") for path in paths) == 12
    assert tuple(path for path in paths if "/resume/" in path) == EXPECTED_RESUME_PATHS
    direct_parents = {str(PurePosixPath(path).parent) for path in paths}
    ancestor_closure: set[str] = set()
    for parent in direct_parents:
        cursor = PurePosixPath(parent)
        while True:
            ancestor_closure.add(str(cursor))
            if str(cursor) == protocol.ARTIFACT_ROOT:
                break
            cursor = cursor.parent
    assert len(direct_parents) == 39
    assert len(ancestor_closure) == 54
    ordered_parent_directories = tuple(
        sorted(
            ancestor_closure,
            key=lambda path: (path.count("/"), path.encode("utf-8")),
        )
    )
    assert protocol.artifact_parent_directories(contract) == ordered_parent_directories
    assert contract["directory_preparation"] == {
        "active_time_directory_creation": "forbidden",
        "authority_source": "exact_frozen_artifact_path_parent_closure",
        "clock_order": "complete_before_first_fresh_or_resume_clock_sample",
        "derivation": (
            "all POSIX parent ancestors of every frozen artifact path, stopping at and "
            "including artifact root"
        ),
        "directories": list(ordered_parent_directories),
        "failure_rule": "STOP_before_session_start_or_checkpoint_import",
        "native_api": "humanbrain.native_temporal_runtime.prepare_durable_directory",
        "order_rule": "ascending_component_count_then_UTF-8_bytewise_path",
        "required_outcome": "TemporalDurabilityOutcome.DURABLE",
    }
    directory_preimage = protocol.canonical_json_bytes(list(ordered_parent_directories))
    assert len(directory_preimage) == 5305
    assert hashlib.sha256(directory_preimage).hexdigest() == (
        "20be5eb10a661a1ec91c693d336a9e024367cb4a2165285eab8fc6f812fd9055"
    )
    assert contract["resume_admission"] == {
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
    assert contract["filesystem_threat_model"] == {
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


def test_development_selection_is_first_pass_in_frozen_arm_order() -> None:
    verdicts = {
        "theta-m-10_theta-a-13": False,
        "theta-m-16_theta-a-18": False,
        "theta-m-18_theta-a-20": True,
        "theta-m-20_theta-a-20": True,
    }
    assert protocol.select_development_arm(verdicts) == (18, 20)

    verdicts["theta-m-18_theta-a-20"] = False
    assert protocol.select_development_arm(verdicts) == (20, 20)

    verdicts["theta-m-20_theta-a-20"] = False
    with pytest.raises(protocol.ProtocolError, match="no candidate arm passed"):
        protocol.select_development_arm(verdicts)

    with pytest.raises(protocol.ProtocolError, match="exactly one verdict"):
        protocol.select_development_arm({"theta-m-10_theta-a-13": True})


def test_source_binding_is_exact_nested_gate3_bound_and_non_runnable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = protocol.build_protocol()
    binding = frozen["source_binding"]

    assert binding["runnable"] is False
    assert binding["governing_gate3_path"] == (
        "benchmarks/procedural_runtime_gate3_slice_protocol_v1.json"
    )
    assert binding["canonical_byte_rule"] == (
        "sha256(raw Git blob or live file after CRLF-to-LF normalization; lone CR forbidden)"
    )
    assert binding["baseline_commits"] == {
        "explicit_threshold_oracle": "c67c6c4b960a26ca8227f78d80da306076d1a931",
        "procedural_gate3_refreeze": "1a262b6d11c0c14bd82c4923e8499befb169576c",
    }
    assert binding["blockers"] == [
        "Official calibration-v3 screen execution, checkpoint/resume orchestration, "
        "and immutable result publication remain disabled behind `_execute_frozen_plan`."
    ]
    assert binding["files"] == {
        "benchmarks/procedural_runtime_gate3_slice_protocol_v1.json": {
            "commit": "1a262b6d11c0c14bd82c4923e8499befb169576c",
            "sha256": "0adecfa53b94ca4f47294e307849b9ae7dd4b6d0b573a4352be5574d6c32d534",
        },
        "experiments/r3_pilot/relay_pool_capacity.json": {
            "commit": "5882bb9d59250c644e8c462428a3908def0f1ee5",
            "sha256": "45c010fdba9a5a2cb301d70d6e98dd004723aff8d4293aa8f44c7425a74b801b",
        },
        "src/humanbrain/cortex.py": {
            "commit": "6a153c7a2bea5636035e144494ec929f60f50682",
            "sha256": "16545b390c516b6f5817366946f0f516cec5b049bb9e1f6021184d9e156e0a8c",
        },
        "src/humanbrain/sdr_union.py": {
            "commit": "40a29341dadc3bac903bf62b2c68763f5f5c493f",
            "sha256": "1de7f6edd914584c4c72d7e42426254590049cfb69cbbfa730afa3179afa63f9",
        },
        "src/humanbrain/sparse_temporal_cortex.py": {
            "commit": "c67c6c4b960a26ca8227f78d80da306076d1a931",
            "sha256": "22e5ab6bb9e3b1bc7010e8c3ffd11d9c2dd9ea01a33ba15046a32db2228357a8",
        },
    }
    assert not any(path.startswith("src/humanbrain/_native_src/") for path in binding["files"])
    assert "src/humanbrain/native_temporal.py" not in binding["files"]
    assert "src/humanbrain/cells.py" not in binding["files"]
    assert all(
        len(row["commit"]) == 40 and len(row["sha256"]) == 64 for row in binding["files"].values()
    )
    protocol.validate_protocol(frozen, require_runnable=False)
    with pytest.raises(protocol.ProtocolError, match="not runnable"):
        protocol.validate_protocol(frozen, require_runnable=True)

    tampered = json.loads(json.dumps(frozen))
    tampered["source_binding"]["files"]["src/humanbrain/sparse_temporal_cortex.py"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(protocol.ProtocolError, match="canonical protocol"):
        protocol.validate_protocol(tampered, require_runnable=False)

    unbound_gate3 = json.loads(json.dumps(frozen))
    unbound_gate3["source_binding"]["governing_gate3_path"] = "benchmarks/not-bound.json"
    monkeypatch.setattr(protocol, "build_protocol", lambda: unbound_gate3)
    with pytest.raises(protocol.ProtocolError, match="governing Gate3 path is not source-bound"):
        protocol.validate_protocol(unbound_gate3, require_runnable=False)


def test_source_byte_validator_and_cli_fail_closed() -> None:
    source = ROOT / "tests" / "test_r3_calibration_v3_protocol.py"
    source_bytes = source.read_bytes()
    binding = {
        "canonical_byte_rule": (
            "sha256(raw Git blob or live file after CRLF-to-LF normalization; lone CR forbidden)"
        ),
        "files": {
            "tests/test_r3_calibration_v3_protocol.py": {
                "commit": "a" * 40,
                "sha256": hashlib.sha256(protocol.canonical_source_bytes(source_bytes)).hexdigest(),
            }
        },
    }

    protocol.validate_source_bytes(ROOT, binding)
    binding["files"]["tests/test_r3_calibration_v3_protocol.py"]["sha256"] = "0" * 64
    with pytest.raises(protocol.ProtocolError, match="source hash drift"):
        protocol.validate_source_bytes(ROOT, binding)

    draft = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-draft"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    runnable = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-runnable"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    printed = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-protocol"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert draft.returncode == 0, draft.stderr
    assert "protocol draft check: PASS (NOT RUNNABLE)" in draft.stdout
    assert runnable.returncode == 2
    assert "protocol is not runnable" in runnable.stderr
    assert printed.returncode == 0, printed.stderr
    assert json.loads(printed.stdout) == protocol.build_protocol()


def test_canonical_json_bytes_are_ascii_and_reject_nonfinite_numbers() -> None:
    assert protocol.canonical_json_bytes({"snowman": "☃"}) == (b'{"snowman":"\\u2603"}')
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        protocol.canonical_json_bytes({"invalid": float("nan")})


def test_git_blob_binding_is_cross_checkout_and_draft_tolerates_live_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "fixture-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    source = repo / "source.txt"
    source.write_bytes(b"alpha\nbeta\n")
    subprocess.run(["git", "add", "source.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Calibration Fixture",
            "-c",
            "user.email=calibration-fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    binding = {
        "canonical_byte_rule": (
            "sha256(raw Git blob or live file after CRLF-to-LF normalization; lone CR forbidden)"
        ),
        "files": {
            "source.txt": {
                "commit": commit,
                "sha256": hashlib.sha256(b"alpha\nbeta\n").hexdigest(),
            }
        },
    }

    protocol.validate_committed_source_blobs(repo, binding)
    source.write_bytes(b"alpha\r\nbeta\r\n")
    protocol.validate_committed_source_blobs(repo, binding)
    protocol.validate_source_bytes(repo, binding)

    source.write_bytes(b"live drift\n")
    protocol.validate_committed_source_blobs(repo, binding)
    with pytest.raises(protocol.ProtocolError, match="live source hash drift"):
        protocol.validate_source_bytes(repo, binding)

    binding["files"]["source.txt"]["sha256"] = "0" * 64
    with pytest.raises(protocol.ProtocolError, match="Git blob hash mismatch"):
        protocol.validate_committed_source_blobs(repo, binding)

    with pytest.raises(protocol.ProtocolError, match="lone CR"):
        protocol.canonical_source_bytes(b"alpha\rbeta\n")


def test_future_runnable_cli_must_validate_live_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = json.loads(json.dumps(protocol.build_protocol()))
    frozen["source_binding"]["runnable"] = True
    frozen["source_binding"]["blockers"] = []
    committed: list[tuple[Path, dict[str, object]]] = []
    live: list[tuple[Path, dict[str, object]]] = []

    monkeypatch.setattr(protocol, "build_protocol", lambda: frozen)
    monkeypatch.setattr(
        protocol,
        "validate_committed_source_blobs",
        lambda root, binding: committed.append((root, binding)),
    )
    monkeypatch.setattr(
        protocol,
        "validate_source_bytes",
        lambda root, binding: live.append((root, binding)),
    )

    assert protocol.main(["--check-draft"]) == 0
    assert committed == [(protocol.REPO_ROOT, frozen["source_binding"])]
    assert live == []

    committed.clear()
    assert protocol.main(["--check-runnable"]) == 0
    assert committed == [(protocol.REPO_ROOT, frozen["source_binding"])]
    assert live == [(protocol.REPO_ROOT, frozen["source_binding"])]


def test_execution_order_and_future_result_contract_are_predeclared() -> None:
    frozen = protocol.build_protocol()

    assert frozen["execution_status"] == "NOT_RUNNABLE_PRE_OUTCOME"
    assert frozen["execution_order"] == [
        "electrical_v3_all_development_and_independent_confirmation_seeds",
        "q4_all_candidate_arms_on_development_seeds",
        "q255_both_source_regimes_for_every_q4_passing_arm_on_development_seeds",
        "r3_f0_development_fresh_seed17_for_fixture_passing_arms_in_frozen_order",
        "write_development_selection_exclusively_for_first_passing_arm",
        "confirm_selected_arm_q4_and_q255_only_on_independent_seeds",
        ("r3_f0_deterministic_confirmation_selected_arm_seed17_second_fresh_plus_25_50_75_resumes"),
        "reject_v3_without_fallback_if_any_confirmation_gate_fails",
        "write_final_verdict_exclusively",
        "write_CN_2_selected_literals_only_after_confirmed_PASS",
    ]
    assert frozen["result_contract"] == {
        "artifact_creation_requires_runnable_source_binding": True,
        "development_artifacts_must_report_every_executed_arm": True,
        "confirmation_artifacts_must_contain_selected_arm_only": True,
        "development_selection_path": (
            "experiments/r3_pilot/calibration_v3_artifacts/selection/development-selection.json"
        ),
        "final_verdict_path": (
            "experiments/r3_pilot/calibration_v3_artifacts/selection/final-verdict.json"
        ),
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
            "final-verdict.json after all confirmation gates",
        ],
        "writes_must_be_exclusive": True,
    }
    assert len(protocol.protocol_sha256()) == 64
    assert protocol.protocol_sha256() == _canonical_sha256(frozen)


def test_fixture_falsifiers_are_literal_and_arm_specific() -> None:
    screens = protocol.build_protocol()["screens"]

    assert screens["compressed_reserve_q4"]["gates"] == [
        "all target columns recruit complete eight-cell cohorts atomically",
        "no cohort is used before its predecessor reaches q=4 on every cell",
        "novel driven candidates never exceed eight",
        "every novel exposure allocates exactly eleven segments",
        "pre-exposure matching active and predicted-column counts are zero",
        "fresh and milestone-resumed logical snapshots are identical",
    ]
    assert screens["full_cohort_q255"]["production_codebook"] == {
        "hash_preimage": (
            "canonical JSON [[token_id, sorted_column_ids], ...] in ascending frozen token-ID order"
        ),
        "ids": "0..255 plus 1024..1535",
        "input_max_column_overlap": 2,
        "sha256": "388177db5f1d8cba9f0606d5e065148498418b10d27d7abf6117d09b44238a9c",
        "target_token": 32,
        "token_count": 768,
    }
    assert screens["full_cohort_q255"]["production_episode_steps"] == [
        "reset_transient",
        "recognize_a",
        "recognize_b",
        "diagnose_context",
        "learn_target",
        "reset_transient",
    ]
    assert screens["full_cohort_q255"]["disjoint_episode_steps"] == [
        "reset_transient",
        "recognize_context",
        "diagnose_context",
        "learn_target",
        "reset_transient",
    ]
    assert screens["full_cohort_q255"]["disjoint_input_facts"] == {
        "learned_contexts": 2041,
        "learned_max_column_overlap": 3,
        "novel_contexts": 128,
        "novel_internal_max_column_overlap": 2,
        "novel_vs_learned_max_column_overlap": 2,
        "target_overlap": 0,
        "unique_rows": 2169,
    }
    assert screens["full_cohort_q255"]["gates"] == [
        "2041 source contexts are unique",
        "2041 target winner tuples are unique",
        "maximum winner overlap is below ceil(theta_match/2)",
        "winner/relay physical overlap is exactly twice winner overlap",
        "pre-exposure matching active and predicted-column counts are zero",
        "every novel exposure allocates exactly eleven segments",
        "direct learned-source recurrence selects all eleven stored owners",
        "128 frozen novel probes produce no false match or prediction",
        "exposure 2040 leaves every base target owner at load 255",
        "exposure 2041 recruits eleven complete eight-cell cohorts atomically",
        "fresh and resumed logical plus recognition-index hashes are identical",
    ]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
