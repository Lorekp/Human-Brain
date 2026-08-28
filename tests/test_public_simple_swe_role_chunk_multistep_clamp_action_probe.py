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

import public_simple_swe_role_chunk_multistep_clamp_action_probe as probe


@pytest.fixture(scope="module")
def prepared_training():
    return probe._prepare_training()


def _physical(token: int) -> dict[str, object]:
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


def _feedback(token: int) -> dict[str, object]:
    return {
        "advanced": True,
        "observation_token_id": token,
        "native_stop": None,
    }


def _dispatch(token: int, action: str, changed: bool) -> dict[str, object]:
    return {
        "observed_token_id": token,
        "dispatched": True,
        "action": action,
        "source_changed": changed,
    }


def test_previous_results_disclosures_and_target_absence_are_exact() -> None:
    anchors = probe._assert_anchors()
    probe._assert_frozen_sources()
    inventory = probe._prior_public_artifact_inventory()

    assert anchors == {
        "signed_residual_outcome": "PASS_FIXED_SIGNED_RESIDUAL_ACTION_ROUTING",
        "signed_residual_source_sha256": probe.EXPECTED_SIGNED_SOURCE_SHA256,
        "signed_residual_artifact_sha256": probe.EXPECTED_SIGNED_ARTIFACT_SHA256,
        "donor_007_previously_used_as_native_typed_target": True,
        "prior_native_clamp_artifact_sha256": probe.EXPECTED_PRIOR_CLAMP_ARTIFACT_SHA256,
        "target_historical_prompt_disclosure_count": 2,
        "freshness_claimed": False,
    }
    assert inventory
    assert all(probe.TARGET_TASK_ID not in record["path"] for record in inventory)


def test_real_donor_checkers_derive_both_multistep_trajectories_before_learn(
    prepared_training,
) -> None:
    (
        _protocol,
        training,
        target,
        _encoder,
        episodes,
        definition,
        checkpoint,
        _identity,
        train,
        capacity,
    ) = prepared_training

    assert tuple(task.public.ordinal for task in training) == (2, 7)
    assert target.public.ordinal == 6
    assert target.public.task_sha256 == probe.TASK_RECORDS[6]["task_sha256"]
    assert set(episodes) == {2, 7}
    assert {len(episode.token_ids) for episode in episodes.values()} == {88}
    assert {
        probe._sha256_json(tuple(episode.token_ids)) for episode in episodes.values()
    } == {probe.CONCEPT_STREAM_SHA256}
    assert len(checkpoint) > 0
    assert capacity == probe.EXPECTED_CAPACITY
    assert definition.config.segment_capacity == 2_046
    assert definition.config.max_pyramidal_cells == 34_814

    identity, lower = train["episodes"]
    assert identity["initial_source_kind"] == "identity"
    assert identity["derived_observation_token_ids"] == [
        probe.OBS_LOW_TOKEN_ID,
        probe.OBS_HIGH_TOKEN_ID,
        probe.OBS_PASS_TOKEN_ID,
    ]
    assert identity["sequence_token_ids"][-7:] == [
        probe.TEST_TOKEN_ID,
        probe.OBS_LOW_TOKEN_ID,
        probe.FIX_LOWER_TOKEN_ID,
        probe.OBS_HIGH_TOKEN_ID,
        probe.FIX_FULL_TOKEN_ID,
        probe.OBS_PASS_TOKEN_ID,
        probe.DONE_TOKEN_ID,
    ]
    assert [record["classification"] for record in identity["visible_checkers"]] == [
        "LOW",
        "HIGH",
        "PASS",
    ]
    assert lower["initial_source_kind"] == "lower_only"
    assert lower["derived_observation_token_ids"] == [
        probe.OBS_HIGH_TOKEN_ID,
        probe.OBS_PASS_TOKEN_ID,
    ]
    assert lower["sequence_token_ids"][-5:] == [
        probe.TEST_TOKEN_ID,
        probe.OBS_HIGH_TOKEN_ID,
        probe.FIX_FULL_TOKEN_ID,
        probe.OBS_PASS_TOKEN_ID,
        probe.DONE_TOKEN_ID,
    ]
    assert [record["classification"] for record in lower["visible_checkers"]] == [
        "HIGH",
        "PASS",
    ]
    assert identity["all_observations_derived_before_learn"] is True
    assert lower["all_observations_derived_before_learn"] is True


def test_checker_stops_sequentially_and_fixed_actuators_are_literal(
    prepared_training,
) -> None:
    _protocol, training, _target, _encoder, episodes, *_rest = prepared_training
    donor = training[0]
    episode = episodes[probe.IDENTITY_DONOR_ORDINAL]
    signature = probe._layer0_signature(episode, donor.public)
    identity = probe._compile_source(probe.IDENTITY_LAYER0_TOKENS, episode, donor.public)
    lower = probe._compile_source(probe.LOWER_LAYER0_TOKENS, episode, donor.public)
    full = probe._compile_source(probe.FULL_LAYER0_TOKENS, episode, donor.public)

    low = probe._visible_checker(identity, signature.entry_point)
    high = probe._visible_checker(lower, signature.entry_point)
    passed = probe._visible_checker(full, signature.entry_point)
    assert low["classification"] == "LOW" and len(low["executed_cases"]) == 1
    assert high["classification"] == "HIGH" and len(high["executed_cases"]) == 2
    assert passed["classification"] == "PASS" and len(passed["executed_cases"]) == 3
    assert probe._observation_token(low) == probe.OBS_LOW_TOKEN_ID
    assert probe._observation_token(high) == probe.OBS_HIGH_TOKEN_ID
    assert probe._observation_token(passed) == probe.OBS_PASS_TOKEN_ID
    assert "max(" in lower
    assert "min(max(" in full

    unexpected = probe._visible_checker(
        f"def {signature.entry_point}({signature.parameter_name}):\n    return 99\n",
        signature.entry_point,
    )
    crashed = probe._visible_checker("not python", signature.entry_point)
    with pytest.raises(RuntimeError, match="cannot become feedback"):
        probe._observation_token(unexpected)
    with pytest.raises(RuntimeError, match="cannot become feedback"):
        probe._observation_token(crashed)

    after_test, test_dispatch = probe._dispatch_observed_action(
        probe.TEST_TOKEN_ID, identity, episode, donor.public
    )
    after_lower, lower_dispatch = probe._dispatch_observed_action(
        probe.FIX_LOWER_TOKEN_ID, identity, episode, donor.public
    )
    after_full, full_dispatch = probe._dispatch_observed_action(
        probe.FIX_FULL_TOKEN_ID, lower, episode, donor.public
    )
    after_done, done_dispatch = probe._dispatch_observed_action(
        probe.DONE_TOKEN_ID, full, episode, donor.public
    )
    assert after_test == identity and test_dispatch["source_changed"] is False
    assert after_lower == lower and lower_dispatch["source_changed"] is True
    assert after_full == full and full_dispatch["source_changed"] is True
    assert after_done == full and done_dispatch["source_changed"] is False


def test_posthoc_gate_requires_exact_four_plus_three_routes(prepared_training) -> None:
    _protocol, training, _target, _encoder, episodes, *_rest = prepared_training
    donor = training[0]
    episode = episodes[probe.IDENTITY_DONOR_ORDINAL]
    signature = probe._layer0_signature(episode, donor.public)
    identity = probe._compile_source(probe.IDENTITY_LAYER0_TOKENS, episode, donor.public)
    lower = probe._compile_source(probe.LOWER_LAYER0_TOKENS, episode, donor.public)
    full = probe._compile_source(probe.FULL_LAYER0_TOKENS, episode, donor.public)
    actual_checkers = (
        probe._visible_checker(identity, signature.entry_point),
        probe._visible_checker(lower, signature.entry_point),
        probe._visible_checker(full, signature.entry_point),
    )
    control_checkers = (
        probe._visible_checker(lower, signature.entry_point),
        probe._visible_checker(full, signature.entry_point),
    )
    actual_calls = (
        _physical(probe.TEST_TOKEN_ID),
        _physical(probe.FIX_LOWER_TOKEN_ID),
        _physical(probe.FIX_FULL_TOKEN_ID),
        _physical(probe.DONE_TOKEN_ID),
    )
    control_calls = (
        _physical(probe.TEST_TOKEN_ID),
        _physical(probe.FIX_FULL_TOKEN_ID),
        _physical(probe.DONE_TOKEN_ID),
    )
    scored, gates = probe._score_after_all_calls(
        actual_calls=actual_calls,
        control_calls=control_calls,
        actual_observations=(
            probe.OBS_LOW_TOKEN_ID,
            probe.OBS_HIGH_TOKEN_ID,
            probe.OBS_PASS_TOKEN_ID,
        ),
        control_observations=(probe.OBS_HIGH_TOKEN_ID, probe.OBS_PASS_TOKEN_ID),
        actual_feedback=(
            _feedback(probe.OBS_LOW_TOKEN_ID),
            _feedback(probe.OBS_HIGH_TOKEN_ID),
            _feedback(probe.OBS_PASS_TOKEN_ID),
        ),
        control_feedback=(
            _feedback(probe.OBS_HIGH_TOKEN_ID),
            _feedback(probe.OBS_PASS_TOKEN_ID),
        ),
        dispatches=(
            _dispatch(probe.TEST_TOKEN_ID, "TEST", False),
            _dispatch(probe.TEST_TOKEN_ID, "TEST", False),
            _dispatch(probe.FIX_LOWER_TOKEN_ID, "FIX_LOWER", True),
            _dispatch(probe.FIX_FULL_TOKEN_ID, "FIX_FULL", True),
            _dispatch(probe.FIX_FULL_TOKEN_ID, "FIX_FULL", True),
            _dispatch(probe.DONE_TOKEN_ID, "DONE", False),
            _dispatch(probe.DONE_TOKEN_ID, "DONE", False),
        ),
        actual_sources=(identity, identity, lower, full, full),
        control_sources=(lower, lower, full, full),
        identity_source=identity,
        lower_source=lower,
        full_source=full,
        actual_checkers=actual_checkers,
        control_checkers=control_checkers,
        sandbox={
            "passed": True,
            "sandboxed": True,
            "outcome": "passed",
            "regression_case_count": 4,
            "property_case_count": 32,
        },
    )

    assert all(gates.values())
    assert len(scored) == 7
    assert all(record["physical_route_election_valid"] for record in scored.values())


def test_one_marker_precedes_seven_calls_and_hidden_score_is_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared_training
) -> None:
    _protocol, _training, target, _encoder, _episodes, _definition, checkpoint, *_rest = (
        prepared_training
    )
    marker_path = tmp_path / "target.spent.json"
    monkeypatch.setattr(probe, "SPEND_MARKER_PATH", marker_path)
    monkeypatch.setattr(probe, "OUTPUT_PATH", tmp_path / "artifact.json")
    monkeypatch.setattr(probe, "_display_path", lambda path: Path(path).name)

    marker = probe._create_spend_marker(target, probe._sha256_bytes(checkpoint))
    assert marker["exclusive_create"] is True
    assert marker["flushed_and_fsynced_before_all_seven_generate_calls"] is True
    with pytest.raises(RuntimeError, match="already spent"):
        probe._create_spend_marker(target, probe._sha256_bytes(checkpoint))

    source = inspect.getsource(probe.main)
    marker_position = source.index("marker = _create_spend_marker(")
    calls = []
    start = 0
    needle = "route_probe._generate_once("
    while (position := source.find(needle, start)) >= 0:
        calls.append(position)
        start = position + len(needle)
    sandbox_position = source.index("sandbox = action_probe._sandbox_record(")
    score_position = source.index("scored_calls, gates = _score_after_all_calls(")

    assert len(calls) == 7
    assert marker_position < min(calls)
    assert max(calls) < sandbox_position < score_position
    assert "reset_before=False" in inspect.getsource(probe.feedback_probe._advance_feedback)
    assert "encoder.encode(target.public)" not in inspect.getsource(probe._prepare_training)
    assert "encoder.encode(target.public)" in inspect.getsource(probe._prepare_probe)
