from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "experiments" / "r3_pilot"
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_role_chunk_feedback_action_probe as probe


@pytest.fixture(scope="module")
def prepared():
    return probe._prepare_probe()


def _physical_generation(token: int) -> dict[str, object]:
    return {
        "generation": {
            "surface_token_id": token,
            "path": "TEMPORAL",
            "activated_columns": 11,
            "winner_input_columns": 11,
            "candidate_tokens": 1,
            "physical_election_count": 1,
            "temporal_event_count": 1,
        },
        "native_stop": None,
    }


def _dispatch(token: int, action: str, *, changed: bool) -> dict[str, object]:
    return {
        "observed_token_id": token,
        "dispatched": True,
        "action": action,
        "source_changed": changed,
    }


def test_previous_action_and_historical_prompt_disclosure_are_exact() -> None:
    anchors = probe._assert_anchors()
    probe._assert_frozen_sources()
    inventory = probe._durable_marker_inventory()

    assert anchors == {
        "previous_action_outcome": "PASS_ROLE_ROUTE_LAYER0_ACTION",
        "previous_action_artifact_sha256": probe.EXPECTED_PREVIOUS_ACTION_ARTIFACT_SHA256,
        "previous_action_source_sha256": probe.EXPECTED_PREVIOUS_ACTION_SOURCE_SHA256,
        "historical_baseline_sha256": probe.EXPECTED_HISTORICAL_BASELINE_SHA256,
        "historical_prompt_disclosure_count": 2,
        "freshness_claimed": False,
    }
    assert all(probe.TARGET_TASK_ID not in record["path"] for record in inventory)


def test_fresh_circuit_uses_only_two_scalar_feedback_sequences(prepared) -> None:
    (
        _protocol,
        target,
        episode,
        definition,
        checkpoint,
        _identity,
        train,
        capacity,
    ) = prepared

    assert target.public.task_id == probe.TARGET_TASK_ID
    assert target.public.task_sha256 == probe.TARGET_TASK_SHA256
    assert episode.structural_sha256 == probe.TARGET_STRUCTURAL_SHA256
    assert probe._sha256_json(tuple(episode.token_ids)) == (
        probe.TARGET_CONCEPT_STREAM_SHA256
    )
    assert len(checkpoint) > 0
    assert tuple(record["task_id"] for record in train["episodes"]) == (
        "simple-swe-v1/train/tier-0/t0_scalar_nonnegative/005",
        "simple-swe-v1/train/tier-0/t0_scalar_nonnegative/006",
    )
    assert tuple(train["episodes"][0]["sequence_token_ids"][-3:]) == (
        probe.TEST_TOKEN_ID,
        probe.OBS_FAIL_TOKEN_ID,
        probe.FIX_TOKEN_ID,
    )
    assert tuple(train["episodes"][1]["sequence_token_ids"][-3:]) == (
        probe.TEST_TOKEN_ID,
        probe.OBS_PASS_TOKEN_ID,
        probe.DONE_TOKEN_ID,
    )
    assert capacity == {
        "training_sequence_count": 2,
        "noninitial_arrivals": 118,
        "segment_capacity": 1_298,
        "max_pyramidal_cells": 34_066,
    }
    assert definition.config.column_count == 4_096
    assert definition.config.theta_match == 16
    assert definition.config.theta_active == 18
    token_codes = dict(definition.token_codes)
    assert all(
        len(token_codes[token]) == 11
        for token in (
            probe.TEST_TOKEN_ID,
            probe.DONE_TOKEN_ID,
            probe.FIX_TOKEN_ID,
            probe.OBS_FAIL_TOKEN_ID,
            probe.OBS_PASS_TOKEN_ID,
        )
    )


def test_public_diagnostic_bit_and_observed_action_dispatch_are_literal(prepared) -> None:
    _protocol, target, episode, *_rest = prepared
    signature = probe.action_probe._layer0_signature(episode, target.public)
    identity = probe._compile_environment_source(
        probe.IDENTITY_LAYER0_TOKENS, episode, target
    )
    correct = probe._compile_environment_source(
        probe.FIX_LAYER0_TOKENS, episode, target
    )

    failed = probe._run_public_diagnostic(identity, signature.entry_point)
    passed = probe._run_public_diagnostic(correct, signature.entry_point)
    assert failed == {
        "input": -7,
        "expected": 0,
        "observed": -7,
        "passed": False,
        "error": None,
    }
    assert passed == {
        "input": -7,
        "expected": 0,
        "observed": 0,
        "passed": True,
        "error": None,
    }
    assert probe._observation_token(failed) == probe.OBS_FAIL_TOKEN_ID
    assert probe._observation_token(passed) == probe.OBS_PASS_TOKEN_ID
    with pytest.raises(RuntimeError, match="cannot become feedback"):
        probe._observation_token({**failed, "error": "SyntaxError"})
    with pytest.raises(RuntimeError, match="genuine scalar"):
        probe._observation_token({**failed, "observed": "-7"})

    after_test, test_record = probe._dispatch_observed_action(
        probe.TEST_TOKEN_ID, identity, episode, target
    )
    after_fix, fix_record = probe._dispatch_observed_action(
        probe.FIX_TOKEN_ID, identity, episode, target
    )
    after_done, done_record = probe._dispatch_observed_action(
        probe.DONE_TOKEN_ID, correct, episode, target
    )
    assert after_test == identity and test_record["source_changed"] is False
    assert after_fix == correct and fix_record["source_changed"] is True
    assert after_done == correct and done_record["source_changed"] is False


def test_posthoc_gate_requires_test_fail_fix_and_test_pass_done() -> None:
    scored, gates = probe._score_after_all_calls(
        actual_first=_physical_generation(probe.TEST_TOKEN_ID),
        control_first=_physical_generation(probe.TEST_TOKEN_ID),
        actual_second=_physical_generation(probe.FIX_TOKEN_ID),
        control_second=_physical_generation(probe.DONE_TOKEN_ID),
        actual_first_dispatch=_dispatch(probe.TEST_TOKEN_ID, "TEST", changed=False),
        control_first_dispatch=_dispatch(probe.TEST_TOKEN_ID, "TEST", changed=False),
        actual_second_dispatch=_dispatch(probe.FIX_TOKEN_ID, "FIX", changed=True),
        control_second_dispatch=_dispatch(probe.DONE_TOKEN_ID, "DONE", changed=False),
        actual_diagnostic={"passed": False},
        control_diagnostic={"passed": True},
        sandbox={
            "passed": True,
            "sandboxed": True,
            "outcome": "passed",
            "regression_case_count": 4,
            "property_case_count": 32,
        },
    )

    assert all(gates.values())
    assert all(record["physical_route_election_valid"] for record in scored.values())


def test_one_marker_precedes_four_calls_and_hidden_score_is_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared
) -> None:
    _protocol, target, _episode, _definition, checkpoint, *_rest = prepared
    marker_path = tmp_path / "target.spent.json"
    monkeypatch.setattr(probe, "SPEND_MARKER_PATH", marker_path)
    monkeypatch.setattr(probe, "OUTPUT_PATH", tmp_path / "artifact.json")
    monkeypatch.setattr(probe, "_display_path", lambda path: Path(path).name)

    marker = probe._create_spend_marker(target, probe._sha256_bytes(checkpoint))
    assert marker["exclusive_create"] is True
    assert marker["flushed_and_fsynced_before_all_four_generate_calls"] is True
    with pytest.raises(RuntimeError, match="already spent"):
        probe._create_spend_marker(target, probe._sha256_bytes(checkpoint))

    source = inspect.getsource(probe.main)
    marker_position = source.index("marker = _create_spend_marker(")
    call_positions = []
    start = 0
    needle = "route_probe._generate_once("
    while (position := source.find(needle, start)) >= 0:
        call_positions.append(position)
        start = position + len(needle)
    sandbox_position = source.index("sandbox = action_probe._sandbox_record(")
    score_position = source.index("scored_calls, gates = _score_after_all_calls(")

    assert len(call_positions) == 4
    assert marker_position < min(call_positions)
    assert max(call_positions) < sandbox_position < score_position
    assert "reset_before=False" in inspect.getsource(probe._advance_feedback)
