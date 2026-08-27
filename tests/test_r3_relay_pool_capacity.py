from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.r3_pilot.calibrate_relay_pool_capacity import (
    cohort_rounded_depth,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "r3_pilot" / "calibrate_relay_pool_capacity.py"
ARTIFACT = ROOT / "experiments" / "r3_pilot" / "relay_pool_capacity.json"


def test_committed_relay_pool_capacity_passes_fast_check() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "relay-pool capacity artifact check: PASS" in completed.stdout


def test_artifact_freezes_the_shared_column_contract() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact_id"] == "r3-relay-pool-capacity-v6-cohort-rounded"
    assert artifact["schema_version"] == 6
    assert artifact["architecture"] == {
        "basket_cells": 4096,
        "basket_offset": 262144,
        "cells_per_column": 32,
        "columns": 4096,
        "contact_source_pairs": {
            "A1": "winner_pyramidal + injective_winner_relay",
            "A3": "generic_proximal + injective_generic_relay",
        },
        "logical_pair_slots_per_segment": 10,
        "physical_contacts_per_pair": 2,
        "physical_contacts_per_segment": 20,
        "generic_proximal_cells": 4096,
        "generic_proximal_offset": 266240,
        "generic_relay_cells": 4096,
        "generic_relay_offset": 270336,
        "initial_permanence": "0.51",
        "max_segments_per_cell": 255,
        "physical_gid_bits": 19,
        "pyramidal_cell_cap": 131072,
        "sdr_seed": 17,
        "token_sdr": "token_sdr_indices(seed=17,n_bits=4096,w_ones=K)",
        "total_physical_glif3_cells": 274432,
        "winner_relay_cells": 131072,
        "winner_relay_offset": 131072,
    }
    assert artifact["shared_column_arms"]["K20"]["winner_cells_per_context"] == 20
    assert artifact["shared_column_arms"]["K20"]["physical_relay_multiplier"] == 1
    assert artifact["shared_column_arms"]["K11"]["physical_relay_multiplier"] == 2
    assert artifact["shared_column_arms"]["K40"]["winner_cells_per_context"] == 40
    assert artifact["permanence_encoding"] == {
        "connected_ticks": 250,
        "initial_ticks": 255,
        "minimum_bits": 9,
        "native_hot_dtype": "uint16",
        "negative_decrement_ticks": 2,
        "permanence_increment_ticks": 50,
        "reachable_tick_count": 501,
        "reachable_tick_range": [0, 500],
        "ticks_per_unit": 500,
    }
    assert artifact["claim_boundary"] == {
        "capability_evidence": False,
        "cohort_rounded_train_structural_sizing_only": True,
        "physical_race_evidence": False,
    }


def test_cohort_rounded_depth_freezes_full_eight_cell_reserve_boundaries() -> None:
    assert [
        cohort_rounded_depth(demand, base_depth=8)
        for demand in (0, 1, 2040, 2041, 4080, 4081, 286874)
    ] == [8, 8, 8, 16, 16, 24, 1128]


def test_artifact_freezes_train_capacity_stops_and_developmental_fit() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["scope"]["held_out_books_decoded"] == 0
    assert artifact["scope"]["train_stream_sha256"] == (
        "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"
    )
    assert artifact["context_demand"] == {
        "distinct_contexts": 2116946,
        "mapping_sha256": (
            "c21745a4ef04404dc6b4e84f4619b273817096ce6647d19e01bffbe5e9617987"
        ),
        "max_context_token_ids": [32],
        "max_contexts_per_token": 187263,
    }
    assert artifact["schedule"]["exposure_passes"] == 1
    assert artifact["schedule"]["book_boundary_resets"] == 49
    assert artifact["schedule"]["pass_replay_method"] == "single_train_pass"
    assert [row["novel_contexts"] for row in artifact["schedule"]["passes"]] == [
        2116946,
    ]

    fixed = artifact["shared_column_arms"]
    assert fixed["K20"]["fits_segment_capacity"] is False
    assert fixed["K20"]["train_pass_segments"] == 42338920
    assert fixed["K20"]["train_pass_segments_created"] == 42338920
    assert fixed["K20"]["max_segments_per_cell"] == 7097
    assert fixed["K20"]["capacity_stop"] == {
        "allocation_index_zero_based": 71211,
        "cell_id": 108561,
        "load": 256,
        "token_context_ordinal": 5056,
        "token_id": 97,
    }

    developmental = artifact["developmental_variable_depth"]
    assert developmental["base4_K11"]["all_train_unique_states_fit"] is True
    assert developmental["base4_K11"]["ideal_unrounded_pyramidal_cells_required"] == 99398
    assert developmental["base4_K11"]["cohort_rounding_overhead_cells"] == 7138
    assert developmental["base4_K11"]["pyramidal_cells_required"] == 106536
    assert developmental["base4_K11"]["max_column_depth"] == 1132
    assert developmental["base8_K11"]["all_train_unique_states_fit"] is True
    assert developmental["base8_K11"]["ideal_unrounded_pyramidal_cells_required"] == 109216
    assert developmental["base8_K11"]["cohort_rounding_overhead_cells"] == 5184
    assert developmental["base8_K11"]["pyramidal_cells_required"] == 114400
    assert developmental["base8_K11"]["pyramidal_cells_recruited"] == 81632
    assert developmental["base8_K11"]["pyramidal_cells_remaining"] == 16672
    assert developmental["base8_K11"]["max_column_depth"] == 1128
    assert developmental["base8_K11"]["logical_pair_slots"] == 232864060
    assert developmental["base8_K11"]["physical_contacts"] == 465728120
    assert developmental["base8_K11"]["relay_population_cells"] == 131072
    assert developmental["base8_K11"]["relay_population_cells_sized"] is True
    memory = developmental["base8_K11"]["memory_estimate"]
    assert memory["segment_store"]["segment_count"] == 23286406
    assert memory["segment_store"]["pair_slot_count"] == 232864060
    assert memory["segment_store"]["pair_slots_per_segment"] == 10
    assert memory["segment_store"]["physical_contacts_per_pair"] == 2
    assert memory["segment_store"]["permanence_bits"] == 9
    assert memory["segment_store"]["relay_projection_bits"] == 0
    assert memory["segment_store"]["segment_contact_offset_bytes"] == 0
    assert memory["segment_store"]["compact_bitpacked"]["A1"] == {
        "mib": "768.939789",
        "pair_permanence_bits": 2095776540,
        "pair_record_bits": 6054465560,
        "pair_record_bytes": 756808195,
        "pair_source_bits": 3958689020,
        "segment_owner_bits": 395868902,
        "segment_owner_bytes": 49483613,
        "source_bits_per_pair": 17,
        "total_bits": 6450334462,
        "total_bytes": 806291808,
    }
    assert memory["segment_store"]["compact_bitpacked"]["A3"] == {
        "mib": "630.141994",
        "pair_permanence_bits": 2095776540,
        "pair_record_bits": 4890145260,
        "pair_record_bytes": 611268158,
        "pair_source_bits": 2794368720,
        "segment_owner_bits": 395868902,
        "segment_owner_bytes": 49483613,
        "source_bits_per_pair": 12,
        "total_bits": 5286014162,
        "total_bytes": 660751771,
    }
    assert memory["segment_store"]["native_hot_soa"] == {
        "mib": "1421.414429",
        "owner_segment_count_bytes": 131072,
        "owner_segment_count_dtype": "uint8",
        "pair_permanence_bytes": 465728120,
        "pair_permanence_dtype": "uint16",
        "pair_source_bytes": 931456240,
        "pair_source_dtype": "uint32",
        "segment_owner_bytes": 93145624,
        "segment_owner_dtype": "uint32",
        "total_bytes": 1490461056,
    }
    assert memory["developmental_column_membership"] == {
        "base_memberships_arithmetic": 32768,
        "compact_recruited_column_bits": 979584,
        "compact_recruited_column_bytes": 122448,
        "compact_recruited_column_dtype_bits": 12,
        "hot_cell_gid_bytes": 457600,
        "hot_column_offset_bytes": 16388,
        "native_hot_soa_bytes": 473988,
        "recruited_memberships": 81632,
        "restore_rule": (
            "base memberships are arithmetic; recruited physical ids are contiguous "
            "and store one packed column id each; rebuild hot CSR on restore"
        ),
    }
    assert memory["total_excluding_recognition_index"] == {
        "A1_compact_bitpacked_bytes": 806414256,
        "A3_compact_bitpacked_bytes": 660874219,
        "native_hot_soa_bytes": 1490935044,
    }
    assert memory["recognition_inverted_index"] == {
        "excluded_mutable_64_entry_block_ceiling_estimate_bytes": 996199200,
        "included_in_store_totals": False,
        "required_static_csr_floor_bytes": 931980532,
        "static_offset_bytes": 524292,
        "static_posting_bytes": 931456240,
    }
    assert artifact["token_pool_comparisons"]["q4"]["capacity_stop"] is True

    w22_theta_m = artifact["collision_tails"]["W22"]["theta_m_10"]
    assert w22_theta_m["population_size"] == 131072
    assert w22_theta_m["paired_sources"] is True
    assert w22_theta_m["physical_threshold"] == 10
    assert w22_theta_m["pair_threshold"] == 5

    theta_m = artifact["collision_tails"]["W40"]["theta_m_10"]
    assert theta_m["exact_numerator"] == (
        "16171801017287840879306357661200267260741661"
    )
    assert theta_m["exact_denominator"] == (
        "42660784815681191813942764671745421857555991066929013254491293765794523136"
    )
    assert theta_m["approx_probability"] == "3.79078844591331400089E-31"
