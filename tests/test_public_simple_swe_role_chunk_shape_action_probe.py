from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "experiments" / "r3_pilot"
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_role_chunk_shape_action_probe as probe
from humanbrain.native_temporal_runtime import NativeTemporalCircuit, TemporalStepMode


@pytest.fixture(scope="module")
def donor_material():
    return probe._prepare_donor_material()


@pytest.fixture(scope="module")
def trained_donors(donor_material):
    return probe._train_material(donor_material)


def _physical(token: int) -> dict[str, object]:
    return {
        "generation": {
            "surface_token_id": token,
            "path": "TEMPORAL",
            "activated_columns": probe.ACTIVE_COLUMNS,
            "physical_election_count": 1,
            "winner_input_columns": probe.ACTIVE_COLUMNS,
            "candidate_tokens": 1,
        },
        "native_stop": None,
    }


def test_only_four_public_train_donors_are_prepared(donor_material) -> None:
    training = donor_material["training"]
    assert [(task.public.family, task.public.ordinal) for task in training] == [
        (probe.SCALAR_FAMILY, 5),
        (probe.SCALAR_FAMILY, 6),
        (probe.MAP_FAMILY, 5),
        (probe.MAP_FAMILY, 6),
    ]
    assert all(task.public.task_id != probe.TARGET_TASK_ID for task in training)
    assert set(donor_material["episodes"]) == {
        (probe.SCALAR_FAMILY, 5),
        (probe.SCALAR_FAMILY, 6),
        (probe.MAP_FAMILY, 5),
        (probe.MAP_FAMILY, 6),
    }
    assert {
        len(episode.token_ids)
        for (family, _ordinal), episode in donor_material["episodes"].items()
        if family == probe.SCALAR_FAMILY
    } == {56}
    assert {
        len(episode.token_ids)
        for (family, _ordinal), episode in donor_material["episodes"].items()
        if family == probe.MAP_FAMILY
    } == {66}


def test_list_feedback_is_real_uniform_execution_residual(donor_material) -> None:
    plus_one = donor_material["feedback"][(probe.MAP_FAMILY, 5)]
    plus_five = donor_material["feedback"][(probe.MAP_FAMILY, 6)]
    assert probe._uniform_list_signed_residual(plus_one) == 1
    assert probe._uniform_list_signed_residual(plus_five) == 5
    assert probe._list_feedback_slot_block(plus_one) == (
        probe.slot_start_token_id(2),
        ord("1"),
        probe.slot_end_token_id(2),
    )
    assert probe._list_feedback_slot_block(plus_five) == (
        probe.slot_start_token_id(2),
        ord("5"),
        probe.slot_end_token_id(2),
    )
    assert all(case["observed"] == case["input"] for case in plus_one["cases"])

    mixed = dict(plus_one)
    mixed["residual"] = [1, 1, 2, 1, 1, 1]
    with pytest.raises(RuntimeError, match="mixed, zero"):
        probe._list_feedback_slot_block(mixed)
    passing = dict(plus_one)
    passing["passed"] = True
    with pytest.raises(RuntimeError, match="mixed, zero"):
        probe._list_feedback_slot_block(passing)


def test_exact_shape_router_and_builder_capacity_guards(donor_material) -> None:
    assert donor_material["router_capacity"] == probe.EXPECTED_SHAPE_ROUTER_CAPACITY
    assert donor_material["builder_capacity"] == probe.EXPECTED_BUILDER_CAPACITY
    assert probe.EXPECTED_SHAPE_ROUTER_CAPACITY == {
        "training_sequence_lengths": [59, 59, 69, 69],
        "noninitial_arrivals": 252,
        "segment_capacity": 2_772,
        "max_pyramidal_cells": 35_540,
        "threshold_offset_cells": 83_368,
    }
    assert probe.EXPECTED_BUILDER_CAPACITY == {
        "training_sequence_lengths": [60, 60, 88, 88],
        "noninitial_arrivals": 292,
        "segment_capacity": 3_212,
        "max_pyramidal_cells": 7_308,
        "threshold_offset_cells": 16_152,
    }
    assert [sequence[-3:] for sequence in donor_material["router_sequences"]] == [
        (probe.TEST_TOKEN_ID, probe.OBS_SCALAR_TOKEN_ID, probe.SHAPE_SCALAR_TOKEN_ID),
        (probe.TEST_TOKEN_ID, probe.OBS_SCALAR_TOKEN_ID, probe.SHAPE_SCALAR_TOKEN_ID),
        (probe.TEST_TOKEN_ID, probe.OBS_LIST_TOKEN_ID, probe.SHAPE_MAP_TOKEN_ID),
        (probe.TEST_TOKEN_ID, probe.OBS_LIST_TOKEN_ID, probe.SHAPE_MAP_TOKEN_ID),
    ]
    assert probe._assert_generation_budget() == {
        "longest_expected_source_bytes": 81,
        "required_calls_including_eos": 82,
        "generation_budget": 128,
    }


def test_builder_teacher_paths_copy_parameter_and_map_payload_physically(donor_material) -> None:
    scalar_sub, scalar_add, map_one, map_five = donor_material["builder_sequences"]
    assert [len(sequence) for sequence in (scalar_sub, scalar_add, map_one, map_five)] == [
        60,
        60,
        88,
        88,
    ]
    scalar_target = probe._builder_teacher_target(
        probe.SHAPE_SCALAR_TOKEN_ID, probe.BUILD_ADD_TOKEN_ID
    )
    map_target = probe._builder_teacher_target(
        probe.SHAPE_MAP_TOKEN_ID, probe.BUILD_ADD_TOKEN_ID
    )
    assert [
        token
        for token in scalar_target
        if token in {probe.copy_slot_token_id(slot) for slot in range(probe.SLOT_COUNT)}
    ] == [
        probe.copy_slot_token_id(0),
        probe.copy_slot_token_id(1),
        probe.copy_slot_token_id(1),
        probe.copy_slot_token_id(2),
    ]
    assert [
        token
        for token in map_target
        if token in {probe.copy_slot_token_id(slot) for slot in range(probe.SLOT_COUNT)}
    ] == [
        probe.copy_slot_token_id(0),
        probe.copy_slot_token_id(1),
        probe.copy_slot_token_id(2),
        probe.copy_slot_token_id(1),
    ]
    assert probe._expected_source_bytes(
        "solve_26ac026192446751",
        "values",
        b"3",
        probe.SHAPE_MAP_TOKEN_ID,
        probe.BUILD_ADD_TOKEN_ID,
    ) == (
        b"def solve_26ac026192446751(values):\n"
        b"    return [item_0 + 3 for item_0 in values]\n"
    )


def test_native_checkpoints_replay_all_four_donor_routes_and_sources(trained_donors) -> None:
    with probe.route_probe._public_train_authority():
        for task in trained_donors["training"]:
            key = (task.public.family, task.public.ordinal)
            checker = trained_donors["feedback"][key]
            observation, shape = probe._shape_tokens_from_checker(checker)
            with NativeTemporalCircuit.from_checkpoint(
                trained_donors["router_definition"], trained_donors["router_checkpoint"]
            ) as router:
                assert router.checkpoint_identity() == trained_donors["router_checkpoint_identity"]
                router.advance_block(
                    tuple(trained_donors["episodes"][key].token_ids),
                    mode=TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                first = probe.route_probe._generate_once(router)
                probe.feedback_probe._advance_feedback(router, observation)
                second = probe.route_probe._generate_once(router)
            assert probe._observed_surface_token(first) == probe.TEST_TOKEN_ID
            assert probe._observed_surface_token(second) == shape

            build = probe._teacher_build_label_for_checker(checker)
            sign_observation = probe._sign_observation_from_checker(checker)
            teacher = probe._builder_teacher_target(shape, build)
            sequence = probe._builder_train_episode(task, checker)
            prompt = sequence[: -len(teacher)]
            assert prompt[-1] == sign_observation
            assert prompt[-1] not in {probe.BUILD_ADD_TOKEN_ID, probe.BUILD_SUB_TOKEN_ID}
            with NativeTemporalCircuit.from_checkpoint(
                trained_donors["builder_definition"], trained_donors["builder_checkpoint"]
            ) as builder:
                assert builder.checkpoint_identity() == trained_donors["builder_checkpoint_identity"]
                builder.advance_block(prompt, mode=TemporalStepMode.READ_ONLY, reset_before=True)
                run = probe._generate_teacher_free(builder)
            magnitude = (
                probe.pac1_probe._feedback_magnitude_bytes(checker)
                if task.public.family == probe.SCALAR_FAMILY
                else bytes(probe._list_feedback_slot_block(checker)[1:-1])
            )
            analysis = probe._analyze_builder_run(
                entrypoint=task.public.signature.entry_point,
                parameter=task.public.signature.parameters[0].name,
                magnitude=magnitude,
                shape_token=shape,
                build_token=build,
                run=run,
            )
            assert analysis["passed"] is True
            assert run["terminated_by_eos"] is True
            assert run["stop"] is None
            assert len(run["generations"]) < probe.MAX_GENERATIONS
            assert analysis["copy_slots"] == (
                [0, 1, 1, 2]
                if task.public.family == probe.SCALAR_FAMILY
                else [0, 1, 2, 1]
            )


def test_target_remains_unspent_and_posthoc_controls_are_single_factor(donor_material) -> None:
    upstream = probe._assert_upstream_pass()
    assert upstream == {
        "source_sha256": probe.EXPECTED_PAC1_SOURCE_SHA256,
        "artifact_sha256": probe.EXPECTED_PAC1_ARTIFACT_SHA256,
        "outcome": "PASS_R3_PAC1_EXECUTION_FEEDBACK_CONSTRUCTION",
    }
    history = probe._assert_history_anchor()
    assert history["historical_target_occurrences"] == 2
    assert history["historical_task_content_drift"] is True
    main_source = inspect.getsource(probe.main)
    assert main_source.index("_assert_upstream_pass()") < main_source.index("_select_target")
    assert main_source.index("_select_target") < main_source.index('["encoder"].encode')
    assert "_teacher_build_label_for_checker" not in main_source
    assert "_sign_observation_from_checker(identity_checker)" in main_source
    assert "_shape_observation_from_checker(identity_checker)" in main_source

    map_analysis = {"passed": True, "copy_slots": [0, 1, 2, 1]}
    scalar_analysis = {"passed": True, "copy_slots": [0, 1, 1, 2]}
    checker_pass = {"passed": True}
    checker_fail = {"passed": False}
    sandbox = {
        "passed": True,
        "sandboxed": True,
        "outcome": "passed",
        "regression_case_count": 4,
        "property_case_count": 32,
    }
    gates = probe._score_posthoc(
        actual_first=_physical(probe.TEST_TOKEN_ID),
        identity_checker=donor_material["feedback"][(probe.MAP_FAMILY, 5)],
        actual_feedback={
            "advanced": True,
            "observation_token_id": probe.OBS_LIST_TOKEN_ID,
            "native_stop": None,
        },
        actual_second=_physical(probe.SHAPE_MAP_TOKEN_ID),
        actual_analysis=map_analysis,
        shape_analysis=scalar_analysis,
        payload_analysis=map_analysis,
        actual_checker=checker_pass,
        shape_checker=checker_fail,
        payload_checker=checker_fail,
        sandbox=sandbox,
    )
    assert all(gates.values())
    assert not probe.OUTPUT_PATH.exists()
    assert not probe.SPEND_MARKER_PATH.exists()


def test_post_marker_model_failure_builds_terminal_stop_without_target_retry(
    trained_donors,
) -> None:
    target = probe._select_target(trained_donors["protocol"])
    identity_checker = trained_donors["feedback"][(probe.MAP_FAMILY, 5)]
    actual_feedback = {
        "advanced": True,
        "observation_token_id": probe.OBS_LIST_TOKEN_ID,
        "native_stop": None,
    }
    payload = probe._terminal_router_stop_payload(
        upstream={"outcome": "PASS_R3_PAC1_EXECUTION_FEEDBACK_CONSTRUCTION"},
        history_anchor={"historical_prompt_exposure": True},
        generation_budget=probe._assert_generation_budget(),
        prior_inventory=[],
        prepared=trained_donors,
        target=target,
        marker={"state": "TARGET_ATTEMPT_SPENT"},
        actual_prompt=(),
        actual_first=_physical(probe.TEST_TOKEN_ID),
        identity_checker=identity_checker,
        actual_feedback=actual_feedback,
        actual_second=_physical(9_999),
        actual_shape=9_999,
        started_ns=0,
    )
    assert payload["outcome"] == "STOP_R3_PAC2_TWO_SHAPE_LIST_MAP_CONSTRUCTION"
    assert payload["builder_target"]["prompt_steps"] == []
    assert payload["sandbox"]["outcome"] == "not_attempted"
    assert payload["gates"]["router_physically_selected_test"] is True
    assert payload["gates"]["target_prompt_plus_list_observation_selected_map_shape"] is False
    assert not all(payload["gates"].values())
    json.dumps(payload, sort_keys=True, allow_nan=False)

    assert probe._contains_task_id(
        {"task_ids": ["unrelated", probe.TARGET_TASK_ID]}, probe.TARGET_TASK_ID
    )
    main_source = inspect.getsource(probe.main)
    assert "shape router did not emit a physical shape token" not in main_source
    assert "builder output did not decode to source" not in main_source
    assert not probe.OUTPUT_PATH.exists()
    assert not probe.SPEND_MARKER_PATH.exists()
