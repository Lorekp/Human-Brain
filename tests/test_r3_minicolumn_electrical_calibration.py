from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.r3_pilot import calibrate_minicolumn_electrical as calibration


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "r3_pilot" / "calibrate_minicolumn_electrical.py"
ARTIFACT = ROOT / "experiments" / "r3_pilot" / "minicolumn_electrical_calibration.json"


def test_electrical_protocol_freezes_physical_defaults_and_adversaries() -> None:
    assert calibration._protocol() == {
        "active_current_pa": 670,
        "basket_current_pa": 200000,
        "basket_offset": 262144,
        "base_pyramidal_cells": 32768,
        "cells_per_column": 8,
        "columns": 4096,
        "construction_seeds": [3, 7, 17, 29, 43],
        "dt_us": 50,
        "gaba_current_pa": 500,
        "homeostasis_adversary_rule": (
            "boundary winner receives -20 pA; every competitor receives +20 pA"
        ),
        "homeostasis_adversarial_pa": 20,
        "latency_must_exceed_us": 100,
        "matching_current_pa": 620,
        "max_pyramidal_cells": 131072,
        "novel_current_ladder_pa": [570, 523, 476, 429, 382, 335, 288, 241],
        "physical_cell_count": 274432,
        "proximal_current_pa": 570,
        "proximal_offset": 266240,
        "proximal_relay_offset": 270336,
        "rank_assignment": (
            "rank 0 is assigned to each column's maximum threshold offset, rank 1 "
            "to its minimum, and descending remainder currents to ascending "
            "remaining thresholds"
        ),
        "relay_offset": 131072,
        "relay_current_pa": 20000,
        "scenario_currents_pa": {
            "active_vs_matching": [650, 640, 590, 543, 496, 449, 402, 355],
            "matching_vs_novel": [600, 590, 543, 496, 449, 402, 355, 308],
            "novel_rank_boundary": [550, 543, 496, 449, 402, 355, 308, 261],
        },
        "singleton_matching_active_assumption": True,
    }


def test_one_trial_replays_coupled_latency_and_inhibition() -> None:
    cortex = calibration._new_cortex(3)
    observed = calibration._probe_trial(
        cortex,
        column=0,
        currents=calibration._scenario_currents()["novel_rank_boundary"],
    )

    assert observed["control_spike_count"] == 2
    assert observed["natural_latency_ticks"] > 2
    assert observed["control_winner"] == observed["expected_winner"]
    assert observed["inhibited_spike_count"] == 1
    assert observed["inhibited_winner"] == observed["expected_winner"]
    assert observed["winner_to_basket_ticks"] == 1
    assert observed["basket_to_gaba_ticks"] == 1


def test_historical_electrical_artifact_fails_closed_after_source_refreeze() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "FAIL: sparse source changed" in completed.stdout


def test_artifact_freezes_exhaustive_coupled_evidence() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact_id"] == "r3-minicolumn-electrical-v1"
    assert artifact["schema_version"] == 1
    assert artifact["source_commit"] == "df26cc2"
    assert artifact["protocol"] == calibration._protocol()
    assert artifact["claim_boundary"] == {
        "capability_evidence": False,
        "sparse_temporal_integration_evidence": False,
        "synthetic_electrical_calibration": True,
        "transfer_evidence": False,
    }
    assert set(artifact["boundaries"]) == {
        "active_vs_matching",
        "matching_vs_novel",
        "novel_rank_boundary",
    }
    proximal = artifact["proximal_channel"]
    assert proximal["trials"] == 20480
    assert proximal["source_single_spike_trials"] == 20480
    assert proximal["dedicated_relay_single_spike_trials"] == 20480
    assert proximal["expected_source_winner_trials"] == 20480
    assert proximal["expected_relay_winner_trials"] == 20480
    assert proximal["causal_relay_after_source_trials"] == 20480
    assert proximal["verdict"] == "PASS_EXHAUSTIVE_PROXIMAL_RELAY_SCREEN"
    for boundary in artifact["boundaries"].values():
        assert boundary["trials"] == 20480
        assert boundary["control_two_spike_trials"] == 20480
        assert boundary["inhibited_single_spike_trials"] == 20480
        assert boundary["expected_control_winner_trials"] == 20480
        assert boundary["expected_inhibited_winner_trials"] == 20480
        assert boundary["minimum_natural_latency_ticks"] > 2
        assert boundary["minimum_winner_to_basket_ticks"] == 1
        assert boundary["maximum_winner_to_basket_ticks"] == 1
        assert boundary["minimum_basket_to_gaba_ticks"] == 1
        assert boundary["maximum_basket_to_gaba_ticks"] == 1
        assert boundary["verdict"] == "PASS_EXHAUSTIVE_COUPLED_SCREEN"
        assert len(boundary["seed_hashes_sha256"]) == 5
