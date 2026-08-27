from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.r3_pilot import calibrate_minicolumn_electrical as calibration
from experiments.r3_pilot import calibration_v3_protocol

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "r3_pilot" / "calibrate_minicolumn_electrical.py"
ARTIFACT = ROOT / "experiments" / "r3_pilot" / "minicolumn_electrical_calibration.json"


def test_electrical_module_executes_the_exact_frozen_v3_screen() -> None:
    screen = calibration_v3_protocol.build_protocol()["screens"]["electrical_v3"]

    assert calibration._protocol() == screen
    assert screen["construction_seeds"] == [3, 7, 17, 29, 43, 59, 83, 101]
    assert screen["pyramidal_domain"]["configured_ids"] == [0, 131072]
    assert screen["winner_relay_domain"]["ids"] == [131072, 262144]
    assert screen["generic_proximal_relay_trials"] == 32768
    assert screen["required_verdict"] == calibration.PASS_VERDICT


def test_global_extrema_trial_replays_physical_inhibition_and_exposes_latency_reject() -> None:
    cortex = calibration._new_cortex(3)
    candidates, basket_gid, fixture = calibration._boundary_fixture(cortex)
    observed = calibration._probe_boundary(
        cortex,
        basket_gid=basket_gid,
        candidate_gids=candidates,
        currents=calibration._scenario_currents()["novel_rank_boundary"],
    )

    assert candidates[0] == fixture["pyramidal"]["maximum_gid"]
    assert candidates[1] == fixture["pyramidal"]["minimum_gid"]
    assert basket_gid == fixture["basket"]["maximum_gid"]
    assert len(observed["control_spike_gids"]) == 2
    assert observed["control_winner"] == observed["expected_winner"]
    assert len(observed["inhibited_spike_gids"]) == 1
    assert observed["inhibited_winner"] == observed["expected_winner"]
    assert observed["winner_to_basket_ticks"] == 1
    assert observed["basket_to_gaba_ticks"] == 1
    assert observed["natural_latency_ticks"] == 2
    assert observed["passed"] is False


def test_full_v3_replay_covers_all_domains_and_returns_the_measured_reject() -> None:
    replay = calibration.run_replay()
    calibration.validate_artifact(replay)

    assert replay["verdict"] == calibration.REJECT_VERDICT
    assert set(replay["seeds"]) == {"3", "7", "17", "29", "43", "59", "83", "101"}
    summary = replay["summary"]
    assert summary["monotonicity"] == {
        "pyramidal": {
            "cells_covered": 1_048_576,
            "comparisons": 12_582_816,
            "passed_seeds": 8,
        },
        "winner_relay": {
            "cells_covered": 1_048_576,
            "comparisons": 1_048_568,
            "passed_seeds": 8,
        },
    }
    assert summary["winner_relay"] == {
        "domain_cells_covered": 1_048_576,
        "passed_global_worst_singletons": 8,
        "singleton_trials": 8,
    }
    assert summary["proximal_channel"] == {
        "causal_relay_after_source_trials": 32_768,
        "dedicated_relay_single_spike_trials": 32_768,
        "expected_relay_winner_trials": 32_768,
        "expected_source_winner_trials": 32_768,
        "passed_seeds": 8,
        "source_single_spike_trials": 32_768,
        "trials": 32_768,
    }
    assert summary["boundaries"]["matching_vs_novel"]["passed_trials"] == 8
    assert summary["boundaries"]["active_vs_matching"]["passed_trials"] == 0
    assert summary["boundaries"]["novel_rank_boundary"]["passed_trials"] == 0
    assert len(replay["rejection_reasons"]) == 16


def test_current_artifact_is_source_bound_and_reports_scientific_reject() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    calibration.validate_artifact(artifact)

    assert artifact["artifact_id"] == "r3-minicolumn-electrical-v3"
    assert artifact["schema_version"] == 3
    assert artifact["source_sha256"] == calibration._source_hashes()
    assert artifact["verdict"] == calibration.REJECT_VERDICT

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "scientific verdict: REJECT_RESERVE_VALID_ELECTRICAL_V3" in completed.stdout
