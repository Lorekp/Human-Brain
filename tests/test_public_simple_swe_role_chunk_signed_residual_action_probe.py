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

import public_simple_swe_role_chunk_signed_residual_action_probe as probe


@pytest.fixture(scope="module")
def prepared():
    return probe._prepare_probe()


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


def _dispatch(token: int, action: str, changed: bool) -> dict[str, object]:
    return {
        "observed_token_id": token,
        "dispatched": True,
        "action": action,
        "source_changed": changed,
    }


def test_feedback_anchor_and_reuse_disclosures_are_frozen() -> None:
    anchors = probe._assert_anchors()
    probe._assert_frozen_sources()
    inventory = probe._durable_marker_inventory()

    assert anchors == {
        "feedback_action_outcome": "PASS_REPETITION_CONDITIONED_BINARY_FEEDBACK_ACTION",
        "feedback_action_source_sha256": probe.EXPECTED_FEEDBACK_SOURCE_SHA256,
        "feedback_action_artifact_sha256": probe.EXPECTED_FEEDBACK_ARTIFACT_SHA256,
        "minus_donor_008_previously_used_as_route_target": True,
        "target_historical_prompt_disclosure_count": 2,
        "freshness_claimed": False,
    }
    assert all(probe.TARGET_TASK_ID not in record["path"] for record in inventory)


def test_donor_checkers_derive_two_trajectories_before_learn(prepared) -> None:
    (
        _protocol,
        training,
        target,
        episodes,
        definition,
        checkpoint,
        _identity,
        train,
        capacity,
    ) = prepared

    assert tuple(task.public.ordinal for task in training) == (0, 8)
    assert target.public.ordinal == 7
    assert target.public.task_sha256 == probe.TASK_RECORDS[7]["task_sha256"]
    assert {len(episode.token_ids) for episode in episodes.values()} == {56}
    assert {
        probe._sha256_json(tuple(episode.token_ids)) for episode in episodes.values()
    } == {probe.CONCEPT_STREAM_SHA256}
    assert len(checkpoint) > 0
    assert capacity == probe.EXPECTED_CAPACITY
    assert definition.config.segment_capacity == 1_276
    assert definition.config.max_pyramidal_cells == 34_044

    plus, minus = train["episodes"]
    assert plus["public_offset_from_sidecar"] == 5
    assert plus["visible_checker"]["residual"] == [5, 5, 5]
    assert plus["derived_observation_token_id"] == probe.OBS_PLUS5_TOKEN_ID
    assert plus["teacher_action_token_id"] == probe.FIX_ADD5_TOKEN_ID
    assert plus["sequence_token_ids"][-3:] == [
        probe.TEST_TOKEN_ID,
        probe.OBS_PLUS5_TOKEN_ID,
        probe.FIX_ADD5_TOKEN_ID,
    ]
    assert minus["public_offset_from_sidecar"] == -5
    assert minus["visible_checker"]["residual"] == [-5, -5, -5]
    assert minus["derived_observation_token_id"] == probe.OBS_MINUS5_TOKEN_ID
    assert minus["teacher_action_token_id"] == probe.FIX_SUB5_TOKEN_ID
    assert minus["sequence_token_ids"][-3:] == [
        probe.TEST_TOKEN_ID,
        probe.OBS_MINUS5_TOKEN_ID,
        probe.FIX_SUB5_TOKEN_ID,
    ]
    assert plus["observation_derived_before_learn"] is True
    assert minus["observation_derived_before_learn"] is True


def test_signed_residual_mapping_and_fixed_layer0_actions_are_literal(prepared) -> None:
    _protocol, _training, target, episodes, *_rest = prepared
    episode = episodes[probe.TARGET_ORDINAL]
    signature = probe._layer0_signature(episode, target.public)
    identity = probe._compile_source(
        probe.IDENTITY_LAYER0_TOKENS, episode, target.public
    )
    add5 = probe._compile_source(probe.ADD5_LAYER0_TOKENS, episode, target.public)
    sub5 = probe._compile_source(probe.SUB5_LAYER0_TOKENS, episode, target.public)

    minus = probe._visible_checker(identity, signature.entry_point, -5)
    plus = probe._visible_checker(identity, signature.entry_point, 5)
    assert minus["residual"] == [-5, -5, -5]
    assert plus["residual"] == [5, 5, 5]
    assert probe._observation_token(minus) == probe.OBS_MINUS5_TOKEN_ID
    assert probe._observation_token(plus) == probe.OBS_PLUS5_TOKEN_ID
    with pytest.raises(RuntimeError, match="mixed or unsupported"):
        probe._observation_token(
            {
                **plus,
                "observed": [-4, 10, 7],
                "residual": [5, -5, 5],
                "passed": False,
            }
        )
    with pytest.raises(RuntimeError, match="cannot become cell feedback"):
        probe._observation_token({**plus, "error": "SyntaxError"})

    assert probe._visible_checker(sub5, signature.entry_point, -5)["passed"] is True
    assert probe._visible_checker(add5, signature.entry_point, 5)["passed"] is True
    assert " - " in sub5
    assert " + " in add5


def test_posthoc_gate_requires_minus_to_sub5_and_plus_to_add5() -> None:
    actual_first = _physical(probe.TEST_TOKEN_ID)
    control_first = _physical(probe.TEST_TOKEN_ID)
    scored, gates = probe._score_after_all_calls(
        actual_first=actual_first,
        control_first=control_first,
        actual_second=_physical(probe.FIX_SUB5_TOKEN_ID),
        control_second=_physical(probe.FIX_ADD5_TOKEN_ID),
        actual_observation=probe.OBS_MINUS5_TOKEN_ID,
        control_observation=probe.OBS_PLUS5_TOKEN_ID,
        actual_feedback={
            "advanced": True,
            "observation_token_id": probe.OBS_MINUS5_TOKEN_ID,
        },
        control_feedback={
            "advanced": True,
            "observation_token_id": probe.OBS_PLUS5_TOKEN_ID,
        },
        dispatches=(
            _dispatch(probe.TEST_TOKEN_ID, "TEST", False),
            _dispatch(probe.TEST_TOKEN_ID, "TEST", False),
            _dispatch(probe.FIX_SUB5_TOKEN_ID, "FIX_SUB5", True),
            _dispatch(probe.FIX_ADD5_TOKEN_ID, "FIX_ADD5", True),
        ),
        actual_final_checker={"passed": True},
        control_final_checker={"passed": True},
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
    _protocol, _training, target, _episodes, _definition, checkpoint, *_rest = prepared
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
    calls = []
    start = 0
    needle = "route_probe._generate_once("
    while (position := source.find(needle, start)) >= 0:
        calls.append(position)
        start = position + len(needle)
    sandbox_position = source.index("sandbox = action_probe._sandbox_record(")
    score_position = source.index("scored_calls, gates = _score_after_all_calls(")

    assert len(calls) == 4
    assert marker_position < min(calls)
    assert max(calls) < sandbox_position < score_position
    assert "reset_before=False" in inspect.getsource(
        probe.feedback_probe._advance_feedback
    )
