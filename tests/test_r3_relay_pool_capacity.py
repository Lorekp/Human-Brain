from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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

    assert artifact["architecture"] == {
        "basket_cells": 4096,
        "cells_per_column": 32,
        "columns": 4096,
        "contacts_per_segment": 20,
        "initial_permanence": "0.51",
        "max_segments_per_cell": 255,
        "presynaptic_source_cells_with_relays": 262144,
        "pyramidal_cell_cap": 131072,
        "relay_cells": 131072,
        "sdr_seed": 17,
        "token_sdr": "token_sdr_indices(seed=17,n_bits=4096,w_ones=K)",
        "total_physical_glif3_cells": 266240,
    }
    assert artifact["shared_column_arms"]["K20"]["winner_cells_per_context"] == 20
    assert artifact["shared_column_arms"]["K20"]["physical_relay_multiplier"] == 1
    assert artifact["shared_column_arms"]["K11"]["physical_relay_multiplier"] == 2
    assert artifact["shared_column_arms"]["K40"]["winner_cells_per_context"] == 40
    assert artifact["claim_boundary"] == {
        "capability_evidence": False,
        "ideal_structural_sizing_only": True,
        "physical_race_evidence": False,
    }


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
    assert artifact["schedule"]["pass_replay_method"] == "literal_three_pass"
    assert [row["novel_contexts"] for row in artifact["schedule"]["passes"]] == [
        2116946,
        0,
        0,
    ]

    fixed = artifact["shared_column_arms"]
    assert fixed["K20"]["fits_segment_capacity"] is False
    assert fixed["K20"]["first_pass_segments"] == 42338920
    assert fixed["K20"]["three_pass_segments_created"] == 42338920
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
    assert developmental["base4_K11"]["pyramidal_cells_required"] == 99398
    assert developmental["base4_K11"]["max_column_depth"] == 1125
    assert developmental["base8_K11"]["all_train_unique_states_fit"] is True
    assert developmental["base8_K11"]["pyramidal_cells_required"] == 109216
    assert developmental["base8_K11"]["relay_population_cells"] == 131072
    assert developmental["base8_K11"]["relay_population_cells_sized"] is True
    memory = developmental["base8_K11"]["memory_estimate"]
    assert memory["assumptions"]["contact_cell_gid_bits"] == 18
    assert memory["assumptions"]["relay_mapping"] == (
        "injective arithmetic offset; zero resident projection bits"
    )
    assert memory["assumptions"]["cell_segment_index"] == "CSR u32 offsets"
    assert memory["recognition_inverted_index_included"] is False
    assert memory["relay_projection_bits"] == 0
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
