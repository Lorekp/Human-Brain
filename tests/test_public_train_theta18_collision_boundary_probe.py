from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.r3_pilot import public_train_theta18_collision_boundary_probe as probe


EXPECTED_ARTIFACT_SHA256_CANONICAL_LF = (
    "2420398075188b134271a0a1597787d5e44afce4e840e85945aa665c1e673e36"
)


def test_reference_authentication_pins_the_committed_theta16_localization(
    tmp_path: Path,
) -> None:
    reference = probe.load_authenticated_reference()
    result = reference["result"]
    group = result["lifecycle"]["segment_groups"][0]

    assert probe.REFERENCE_COMMIT == "0a6488ac52d17deb384aec4e20c67883ebef5f2e"
    assert probe.REFERENCE_ARTIFACT_SHA256_CANONICAL_LF == (
        "1197fcd76b9257d653f91288f6313eaf75649239269d2511dd50aba2b21e9dbe"
    )
    assert reference["definition"]["arm"] == [16, 18]
    assert result["global_zero_based_step_index"] == 7_933
    assert result["token_id"] == 32
    assert result["expected_source_fanout"] == 0
    assert result["observed"]["matching_segments"] == 11
    assert result["observed"]["active_segments"] == 0
    assert group["segment_ids"] == list(range(2_728, 2_739))
    assert group["origin_global_step_index"] == 249
    assert group["origin_current_source_overlap_cells"] == 8
    assert group["potential_contacts"] == 16

    tampered = tmp_path / "tampered-reference.json"
    raw = probe.REFERENCE_ARTIFACT_PATH.read_bytes()
    tampered.write_bytes(raw.replace(b'"schema_version": 1', b'"schema_version": 2'))
    with pytest.raises(RuntimeError, match="reference artifact SHA-256 mismatch"):
        probe.load_authenticated_reference(tampered)


def test_banked_theta18_boundary_artifact_is_spent_without_rerunning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_run() -> object:
        raise AssertionError("ordinary artifact guard reran the physical probe")

    monkeypatch.setattr(probe, "_run_boundary", forbidden_run)
    raw = probe.OUTPUT_PATH.read_bytes()
    payload = probe.read_banked_artifact()
    result = payload["result"]
    row = result["row_provenance"]

    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == (
        EXPECTED_ARTIFACT_SHA256_CANONICAL_LF
    )
    assert payload["status"] == "PASS_LOCALIZED_THETA18_COLLISION_BOUNDARY"
    assert payload["claim_boundary"] == (
        "one actual public TRAIN-only Genesis replay through zero-based token 7,933 "
        "at the preregistered arm (18,20), seed 17; known arm-(16,18) collision "
        "boundary only; no calibration-v3 reinterpretation/rerun, threshold tuning, "
        "later TRAIN tokens, threshold-selection/candidate-repair, confirmation, held-out, "
        "C1, capability, or transfer evidence"
    )
    assert payload["interpretation"] == {
        "result_scope": "localized_counterfactual_only",
        "what_pass_means": (
            "the authenticated 8/11 foreign row remains candidate-reachable but is below "
            "theta_match=18 at this one boundary"
        ),
        "threshold_selection_evidence": False,
        "candidate_repair_evidence": False,
        "preexisting_q255_constraint": (
            "independent public q255 production-overlap evidence already STOPs arm (18,20) "
            "at recurrence 82/123 with matching=11, active=0, predicted_columns=0, and "
            "stored-owner drift; this probe neither reruns nor reinterprets it"
        ),
    }
    assert payload["reference"] == {
        "artifact_id": "public-train-first-recurrent-key-failure-v1",
        "artifact_path": "experiments/r3_pilot/public_train_first_key_failure_probe.json",
        "artifact_sha256_canonical_lf": (
            "1197fcd76b9257d653f91288f6313eaf75649239269d2511dd50aba2b21e9dbe"
        ),
        "commit": "0a6488ac52d17deb384aec4e20c67883ebef5f2e",
    }
    assert payload["train_input"]["held_out_books_materialized"] == 0
    assert payload["definition"]["arm"] == [18, 20]
    assert payload["probe_bound"] == {
        "book_ordinal": 0,
        "book_short_name": "Genesis",
        "executed_steps": 7_934,
        "last_zero_based_token_index": 7_933,
    }
    assert result["token_id"] == 32
    assert result["reference_current_source_reproduced"] is True
    assert result["observed"] == {
        "active_segments": 0,
        "allocated_segments": 11,
        "candidate_segments": 2_233,
        "matching_segments": 0,
        "postings_visited": 3_751,
        "predicted": False,
        "predicted_columns": 0,
        "punished_segments": 0,
    }
    assert row["segment_ids"] == list(range(2_728, 2_739))
    assert row["origin_global_step_index"] == 249
    assert row["origin_current_source_overlap_cells"] == 8
    assert row["potential_contacts"] == 16
    assert row["connected_contacts"] == 16
    assert row["candidate_reachable"] is True
    assert row["below_theta_match"] is True
    assert row["matching"] is False
    assert row["active"] is False
    assert row["owners_in_current_token_columns"] is False
    assert result["checks"] == {
        "current_key_outcome_preserved": True,
        "current_source_reproduced": True,
        "global_recognition_below_boundary": True,
        "known_origin_reproduced": True,
        "known_rows_below_theta_match": True,
        "known_rows_candidate_reachable": True,
        "known_rows_reproduced": True,
        "reference_authenticated": True,
        "stopped_at_exact_public_boundary": True,
    }
    assert result["failures"] == []
