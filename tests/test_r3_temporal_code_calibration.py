from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from experiments.r3_pilot import calibrate_temporal_code as calibration


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "r3_pilot" / "calibrate_temporal_code.py"
ARTIFACT = ROOT / "experiments" / "r3_pilot" / "temporal_code_calibration.json"


def test_committed_temporal_code_artifact_passes_fast_check() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "temporal-code calibration artifact check: PASS" in completed.stdout


def test_artifact_freezes_the_synthetic_threshold_grid() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact_id"] == "r3-temporal-code-collision-v2"
    assert artifact["schema_version"] == 2
    assert artifact["source_commit"] == "df26cc2"
    assert artifact["protocol"] == {
        "active_columns": 11,
        "base_cells_per_column": 8,
        "column_count": 4096,
        "contacts_per_segment": 20,
        "construction_seeds": [3, 7, 17, 29],
        "context_physical_width": 22,
        "context_source_groups": 11,
        "context_source_structure": "injective (pyramidal winner, relay twin) pairs",
        "direct_recurrent_source_probe": (
            "diagnose each emitted target context as the next step's W_prev"
        ),
        "learned_contexts": 128,
        "learned_target_pair_overlap_gate": (
            "reject max winner-pair overlap >= ceil(theta_match/2)"
        ),
        "novel_contexts": 128,
        "projection_seed": 17,
        "reject_token_sdr_overlap_at_or_above": 5,
        "same_token_collision_model": (
            "measured grouped-pair intersections; physical overlap is exactly twice "
            "winner-pair overlap"
        ),
        "target_token": 256,
        "threshold_arms": [[10, 13], [16, 18], [18, 20]],
        "token_codebook": "token_sdr_indices(seed=17,n_bits=4096,w_ones=11)",
    }
    assert set(artifact["arms"]) == {
        "theta_m10_theta_a13",
        "theta_m16_theta_a18",
        "theta_m18_theta_a20",
    }
    assert artifact["claim_boundary"] == {
        "capability_evidence": False,
        "held_out_data_accessed": False,
        "independent_confirmatory_seed_set": False,
        "synthetic_collision_calibration": True,
        "threshold_selection_is_evaluation": False,
        "transfer_evidence": False,
    }
    assert artifact["selection"]["rule_frozen_before_stable_source_replay"] is True
    assert artifact["selection"]["rule_revision_basis"] == (
        "paired-source structural correction after a preliminary synthetic artifact "
        "exposed unsafe learned-code overlap"
    )


def test_artifact_freezes_collision_and_read_only_evidence() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["selection"]["selected_thresholds"] == [18, 20]
    expected_gate = {
        "theta_m10_theta_a13": (5, 0, "REJECT_SYNTHETIC_COLLISION_SCREEN"),
        "theta_m16_theta_a18": (8, 3, "REJECT_SYNTHETIC_COLLISION_SCREEN"),
        "theta_m18_theta_a20": (9, 4, "PASS_SYNTHETIC_COLLISION_SCREEN"),
    }
    for arm_name, arm in artifact["arms"].items():
        bound, safe_seeds, verdict = expected_gate[arm_name]
        assert arm["aggregate"] == {
            "learned_code_collisions": 0,
            "learned_code_max_pair_overlap": 8,
            "learned_code_pair_overlap_rejection_bound": bound,
            "learned_code_seeds_below_rejection_bound": safe_seeds,
            "learned_contexts_active": 512,
            "learned_contexts_expected": 512,
            "novel_active_segments": 0,
            "novel_false_predictions": 0,
            "novel_matching_segments": 0,
            "pre_exposure_active_segments": 0,
            "pre_exposure_matching_segments": 0,
            "pre_exposure_predicted_columns": 0,
            "pre_exposure_predicted_contexts": 0,
            "recurrent_source_active_segments": 0,
            "recurrent_source_matching_segments": 0,
            "screening_verdict": verdict,
            "seed_runs": 4,
        }
        assert [
            row["learned_codes"]["max_pairwise_winner_intersection"]
            for row in arm["seeds"]
        ] == [6, 8, 6, 6]
        for row in arm["seeds"]:
            assert row["learned_codes"]["unique_codes"] == 128
            assert row["learned_codes"][
                "physical_overlap_is_exactly_twice_winner_overlap"
            ] is True
            assert row["learned_source_context_intersections"][
                "pairwise_winner_pairs"
            ]["unique_rows"] == 128
            assert row["learned_source_context_intersections"][
                "physical_overlap_is_exactly_twice_pair_overlap"
            ] is True
            assert row["learned_context_diagnostics"]["contexts_with_active_segments"] == 128
            assert row["target_pre_exposure_diagnostics"]["contexts"] == 128
            assert row["target_pre_exposure_diagnostics"]["matching_segments"] == 0
            assert row["target_pre_exposure_diagnostics"]["active_segments"] == 0
            assert row["learned_codes"]["predicted_columns_before_exposure"] == 0
            assert row["novel_probe"]["grouped_pair_overlap"][
                "physical_overlap_is_exactly_twice_pair_overlap"
            ] is True
            assert all(row["parity"].values())


def test_real_shared_token_codebook_is_unique_and_not_disjoint() -> None:
    codebook, codebook_hash, overlap = calibration._build_codebook()

    assert len(codebook) == 257
    assert all(len(columns) == 11 for columns in codebook.values())
    assert codebook_hash == "4ab66dab1657c4e4353267e7fe58371cfc7e833fe9c50f9bb7e9ce3bc34077a9"
    assert overlap == {
        "histogram": {"0": 31959, "1": 931, "2": 6},
        "max_intersection": 2,
        "pairs": 32896,
        "total_intersection": 943,
        "unique_rows": 257,
    }


def test_grouped_pair_overlap_is_not_treated_as_independent_cells() -> None:
    winner_rows = [(1, 2, 3), (1, 2, 4), (1, 5, 6)]
    paired_physical_rows = [
        row + tuple(cell + 100 for cell in row) for row in winner_rows
    ]
    pair_summary = calibration._intersection_summary(winner_rows)
    physical_summary = calibration._intersection_summary(paired_physical_rows)

    assert pair_summary["max_intersection"] == 2
    assert physical_summary["max_intersection"] == 4
    assert calibration._paired_overlap_identity(pair_summary, physical_summary)
    assert not calibration._paired_overlap_identity(pair_summary, pair_summary)


def test_threshold_gate_rejects_a_target_code_overlap_at_the_paired_bound() -> None:
    seed_row = {
        "learned_codes": {
            "collision_count": 0,
            "max_pairwise_winner_intersection": 5,
            "predicted_columns_before_exposure": 0,
            "predicted_contexts_before_exposure": 0,
        },
        "learned_context_diagnostics": {"contexts_with_active_segments": 128},
        "novel_probe": {
            "active_segments": 0,
            "false_predictions": 0,
            "matching_segments": 0,
        },
        "recurrent_source_diagnostics": {
            "active_segments": 0,
            "matching_segments": 0,
        },
        "target_pre_exposure_diagnostics": {
            "active_segments": 0,
            "matching_segments": 0,
        },
    }

    rejected = calibration._aggregate([seed_row], theta_match=10)
    seed_row["learned_codes"]["max_pairwise_winner_intersection"] = 4
    accepted = calibration._aggregate([seed_row], theta_match=10)
    seed_row["recurrent_source_diagnostics"]["matching_segments"] = 1
    recurrent_rejected = calibration._aggregate([seed_row], theta_match=10)
    seed_row["recurrent_source_diagnostics"]["matching_segments"] = 0
    seed_row["target_pre_exposure_diagnostics"]["matching_segments"] = 1
    pre_exposure_rejected = calibration._aggregate([seed_row], theta_match=10)

    assert rejected["learned_code_pair_overlap_rejection_bound"] == 5
    assert rejected["learned_code_max_pair_overlap"] == 5
    assert rejected["learned_code_seeds_below_rejection_bound"] == 0
    assert rejected["screening_verdict"] == "REJECT_SYNTHETIC_COLLISION_SCREEN"
    assert accepted["learned_code_seeds_below_rejection_bound"] == 1
    assert accepted["screening_verdict"] == "PASS_SYNTHETIC_COLLISION_SCREEN"
    assert recurrent_rejected["recurrent_source_matching_segments"] == 1
    assert (
        recurrent_rejected["screening_verdict"]
        == "REJECT_SYNTHETIC_COLLISION_SCREEN"
    )
    assert pre_exposure_rejected["pre_exposure_matching_segments"] == 1
    assert (
        pre_exposure_rejected["screening_verdict"]
        == "REJECT_SYNTHETIC_COLLISION_SCREEN"
    )


def test_deep_snapshot_hash_includes_the_reserve_cursor() -> None:
    learned = SimpleNamespace(
        column_cells=((0,),),
        segments=(),
        learning_step=1,
        next_reserve_cell=8,
    )
    snapshot = SimpleNamespace(
        circuit_identity="a" * 64,
        learned=learned,
        previous_context=(),
        previous_winner_context=(),
        event_step=0,
    )
    recruited = SimpleNamespace(
        circuit_identity=snapshot.circuit_identity,
        learned=SimpleNamespace(
            column_cells=learned.column_cells,
            segments=learned.segments,
            learning_step=learned.learning_step,
            next_reserve_cell=9,
        ),
        previous_context=snapshot.previous_context,
        previous_winner_context=snapshot.previous_winner_context,
        event_step=snapshot.event_step,
    )
    winner_context_changed = SimpleNamespace(
        circuit_identity=snapshot.circuit_identity,
        learned=learned,
        previous_context=snapshot.previous_context,
        previous_winner_context=(7,),
        event_step=snapshot.event_step,
    )

    assert calibration._snapshot_hash(snapshot) != calibration._snapshot_hash(recruited)
    assert (
        calibration._snapshot_hash(snapshot)
        != calibration._snapshot_hash(winner_context_changed)
    )
