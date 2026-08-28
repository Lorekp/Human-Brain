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

import public_simple_swe_role_chunk_parameterized_action_probe as probe
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    TemporalStepMode,
)


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


def test_exact_anchors_historical_drift_and_no_prior_routed_target() -> None:
    anchors = probe._assert_anchors()
    probe._assert_frozen_sources()
    inventory = probe._prior_routed_target_inventory()

    assert anchors == {
        "signed_residual_outcome": "PASS_FIXED_SIGNED_RESIDUAL_ACTION_ROUTING",
        "signed_residual_artifact_sha256": probe.EXPECTED_SIGNED_ARTIFACT_SHA256,
        "role_place_outcome": "PASS_TWO_ROUTE_VARIABLE_LITERAL_SCALAR_SEMANTICS",
        "role_place_artifact_sha256": probe.EXPECTED_ROLE_PLACE_ARTIFACT_SHA256,
        "route_artifact_sha256": probe.EXPECTED_ROUTE_ARTIFACT_SHA256,
        "donors_005_and_006_previously_used_in_role_chunk_routing": True,
        "target_historical_prompt_disclosure_count": 2,
        "historical_target_task_sha256": probe.HISTORICAL_TARGET_TASK_SHA256,
        "current_target_task_sha256": probe.TASK_RECORDS[4]["task_sha256"],
        "historical_task_content_drift": True,
        "freshness_claimed": False,
    }
    assert inventory
    assert all(probe.TARGET_TASK_ID not in record["path"] for record in inventory)


def test_donor_only_training_derives_sign_and_exact_sequence_capacities(
    prepared_training,
) -> None:
    prepared = prepared_training
    training = prepared["training"]
    target = prepared["target"]
    episodes = prepared["episodes"]

    assert tuple(task.public.ordinal for task in training) == (6, 5)
    assert target.public.ordinal == 4
    assert target.public.task_sha256 == probe.TASK_RECORDS[4]["task_sha256"]
    assert "target_episode" not in prepared
    assert set(episodes) == {6, 5}
    assert {len(episode.token_ids) for episode in episodes.values()} == {56}
    assert {
        probe._sha256_json(tuple(episode.token_ids)) for episode in episodes.values()
    } == {probe.CONCEPT_STREAM_SHA256}

    assert prepared["router_capacity"] == probe.EXPECTED_ROUTER_CAPACITY
    assert prepared["builder_capacity"] == probe.EXPECTED_BUILDER_CAPACITY
    assert prepared["router_definition"].config.segment_capacity == 1_276
    assert prepared["router_definition"].config.max_pyramidal_cells == 34_044
    assert prepared["builder_definition"].config.segment_capacity == 1_936
    assert prepared["builder_definition"].config.max_pyramidal_cells == 6_032

    add, sub = prepared["router_train"]
    assert add["visible_residual_checker"]["residual"] == [2, 2, 2]
    assert add["derived_sign_token_id"] == probe.OBS_POS_TOKEN_ID
    assert add["teacher_build_token_id"] == probe.BUILD_ADD_TOKEN_ID
    assert add["sequence_token_ids"][-3:] == [
        probe.TEST_TOKEN_ID,
        probe.OBS_POS_TOKEN_ID,
        probe.BUILD_ADD_TOKEN_ID,
    ]
    assert sub["visible_residual_checker"]["residual"] == [-8, -8, -8]
    assert sub["derived_sign_token_id"] == probe.OBS_NEG_TOKEN_ID
    assert sub["teacher_build_token_id"] == probe.BUILD_SUB_TOKEN_ID
    assert sub["sequence_token_ids"][-3:] == [
        probe.TEST_TOKEN_ID,
        probe.OBS_NEG_TOKEN_ID,
        probe.BUILD_SUB_TOKEN_ID,
    ]
    assert add["only_sign_token_entered_cells_after_test"] is True
    assert sub["only_sign_token_entered_cells_after_test"] is True

    add_builder, sub_builder = prepared["builder_train"]
    assert (add_builder["magnitude_slot"], add_builder["teacher_build_token_id"]) == (
        "2",
        probe.BUILD_ADD_TOKEN_ID,
    )
    assert (sub_builder["magnitude_slot"], sub_builder["teacher_build_token_id"]) == (
        "8",
        probe.BUILD_SUB_TOKEN_ID,
    )
    assert add_builder["temporal_target_contains_literal_magnitude"] is False
    assert sub_builder["temporal_target_contains_literal_magnitude"] is False


def test_router_checkpoint_replays_both_donor_sign_routes(prepared_training) -> None:
    prepared = prepared_training
    with probe.route_probe._public_train_authority():
        for task, record in zip(
            prepared["training"], prepared["router_train"], strict=True
        ):
            episode = prepared["episodes"][task.public.ordinal]
            with NativeTemporalCircuit.from_checkpoint(
                prepared["router_definition"], prepared["router_checkpoint"]
            ) as circuit:
                circuit.advance_block(
                    tuple(episode.token_ids),
                    mode=TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                first = probe.route_probe._generate_once(circuit)
                feedback = probe.feedback_probe._advance_feedback(
                    circuit, record["derived_sign_token_id"]
                )
                second = probe.route_probe._generate_once(circuit)

            first_score = probe.route_probe._score_generation(
                first, probe.TEST_TOKEN_ID
            )
            second_score = probe.route_probe._score_generation(
                second, record["teacher_build_token_id"]
            )
            assert first_score["correct"] is True
            assert feedback["advanced"] is True
            assert second_score["correct"] is True


def test_builder_checkpoint_replays_donors_through_physical_copy(prepared_training) -> None:
    prepared = prepared_training
    routes = (probe.BUILD_ADD_TOKEN_ID, probe.BUILD_SUB_TOKEN_ID)
    expected_sources = (
        "def solve_2c9984a52db14adc(n):\n    return n + 2\n",
        "def solve_6acb574ebf911e92(n):\n    return n - 8\n",
    )
    with probe.route_probe._public_train_authority():
        for task, route, expected_source in zip(
            prepared["training"], routes, expected_sources, strict=True
        ):
            episode = prepared["episodes"][task.public.ordinal]
            payload = probe._builder_payload(task, episode)
            prompt = probe._builder_prompt(payload, route)
            teacher_target = probe._builder_internal_target(payload, route)
            assert prompt.count(route) == 1
            assert route not in teacher_target
            assert ord("+") in teacher_target if route == probe.BUILD_ADD_TOKEN_ID else ord("-") in teacher_target

            with NativeTemporalCircuit.from_checkpoint(
                prepared["builder_definition"], prepared["builder_checkpoint"]
            ) as circuit:
                circuit.advance_block(
                    prompt,
                    mode=TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                run = probe.role_place_probe._generate_teacher_free(circuit)
            analysis = probe._analyze_builder_run(payload, route, run)

            assert analysis["passed"] is True
            assert analysis["source"] == expected_source
            assert analysis["copy_transactions_exact"] is True
            assert [record["slot"] for record in analysis["copy_transactions"]] == [
                0,
                2,
            ]
            assert analysis["operator_elected_physically"] is True
            assert analysis["ordinary_eos"] is True


def test_posthoc_gate_requires_operation_and_payload_interventions() -> None:
    payload9 = {
        "identifier": b"solve_5899fcfd152e762e",
        "parameter": b"n",
        "magnitude": b"9",
    }
    payload8 = {**payload9, "magnitude": b"8"}
    actual_prompt = probe._builder_prompt(payload9, probe.BUILD_SUB_TOKEN_ID)
    operation_prompt = probe._builder_prompt(payload9, probe.BUILD_ADD_TOKEN_ID)
    payload_prompt = probe._builder_prompt(payload8, probe.BUILD_SUB_TOKEN_ID)
    first = _physical(probe.TEST_TOKEN_ID)
    _scored, gates = probe._score_after_all_target_calls(
        actual_first=first,
        intervention_first=dict(first),
        actual_second=_physical(probe.BUILD_SUB_TOKEN_ID),
        intervention_second=_physical(probe.BUILD_ADD_TOKEN_ID),
        actual_observation=probe.OBS_NEG_TOKEN_ID,
        intervention_observation=probe.OBS_POS_TOKEN_ID,
        actual_feedback=_feedback(probe.OBS_NEG_TOKEN_ID),
        intervention_feedback=_feedback(probe.OBS_POS_TOKEN_ID),
        actual_build=probe.BUILD_SUB_TOKEN_ID,
        intervention_build=probe.BUILD_ADD_TOKEN_ID,
        builder_prompts=(actual_prompt, operation_prompt, payload_prompt),
        actual_analysis={"passed": True, "signed_offset": -9},
        operation_analysis={"passed": True, "signed_offset": 9},
        payload_analysis={"passed": True, "signed_offset": -8},
        actual_checker={"passed": True, "residual": [0, 0, 0]},
        operation_checker={"passed": False, "residual": [-18, -18, -18]},
        payload_checker={"passed": False, "residual": [-1, -1, -1]},
        sandbox={
            "passed": True,
            "sandboxed": True,
            "outcome": "passed",
            "regression_case_count": 4,
            "property_case_count": 32,
        },
    )

    assert all(gates.values())


def test_one_marker_precedes_every_target_generate_and_training_never_encodes_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared_training
) -> None:
    prepared = prepared_training
    marker_path = tmp_path / "target.spent.json"
    monkeypatch.setattr(probe, "SPEND_MARKER_PATH", marker_path)
    monkeypatch.setattr(probe, "OUTPUT_PATH", tmp_path / "artifact.json")
    monkeypatch.setattr(probe, "_display_path", lambda path: Path(path).name)

    marker = probe._create_spend_marker(
        prepared["target"],
        probe._sha256_bytes(prepared["router_checkpoint"]),
        probe._sha256_bytes(prepared["builder_checkpoint"]),
    )
    assert marker["exclusive_create"] is True
    assert (
        marker[
            "flushed_and_fsynced_before_all_target_router_and_builder_generate_calls"
        ]
        is True
    )
    with pytest.raises(RuntimeError, match="already spent"):
        probe._create_spend_marker(
            prepared["target"],
            probe._sha256_bytes(prepared["router_checkpoint"]),
            probe._sha256_bytes(prepared["builder_checkpoint"]),
        )

    source = inspect.getsource(probe.main)
    marker_position = source.index("marker = _create_spend_marker(")
    router_calls = []
    start = 0
    router_needle = "route_probe._generate_once("
    while (position := source.find(router_needle, start)) >= 0:
        router_calls.append(position)
        start = position + len(router_needle)
    builder_calls = []
    start = 0
    builder_needle = "role_place_probe._generate_teacher_free("
    while (position := source.find(builder_needle, start)) >= 0:
        builder_calls.append(position)
        start = position + len(builder_needle)
    sandbox_position = source.index("sandbox = action_probe._sandbox_record(")
    score_position = source.index("router_scored, gates = _score_after_all_target_calls(")

    assert len(router_calls) == 4
    assert len(builder_calls) == 3
    assert marker_position < min((*router_calls, *builder_calls))
    assert max((*router_calls, *builder_calls)) < sandbox_position < score_position
    assert "encoder.encode(target.public)" not in inspect.getsource(
        probe._prepare_training
    )
    assert "encoder.encode(target.public)" in inspect.getsource(probe._prepare_probe)
    assert "_builder_internal_target(" not in source
    assert "_expected_source_bytes(" not in source
