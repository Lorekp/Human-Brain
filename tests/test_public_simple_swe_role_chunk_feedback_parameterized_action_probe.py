from __future__ import annotations

import hashlib
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

import public_simple_swe_role_chunk_feedback_parameterized_action_probe as probe


@pytest.fixture(scope="module")
def prepared_training():
    return probe._prepare_training()


def _physical(token: int) -> dict[str, object]:
    return {
        "generation": {
            "surface_token_id": token,
            "path": "TEMPORAL",
            "physical_election_count": 1,
            "winner_input_columns": probe.ACTIVE_COLUMNS,
            "predicted": True,
            "novel_race": False,
            "copy_source_slot": None,
            "copy_place": None,
        },
        "native_stop": None,
    }


def test_exact_anchors_history_and_unspent_target_identity() -> None:
    anchors = probe._assert_anchors()
    assert anchors["pac0_outcome"] == "PASS_R3_PAC0_ROLE_BOUND_CONSTRUCTION"
    assert anchors["pac0_source_sha256"] == probe.EXPECTED_PAC0_SOURCE_SHA256
    assert anchors["pac0_artifact_sha256"] == probe.EXPECTED_PAC0_ARTIFACT_SHA256
    assert anchors["historical_target_occurrences"] == 2
    assert anchors["historical_task_content_drift"] is True
    assert probe._prior_routed_target_inventory() == []
    assert not probe.OUTPUT_PATH.exists()
    assert not probe.SPEND_MARKER_PATH.exists()

    _protocol, fit_donors, training, target = probe._select_tasks()
    assert len(fit_donors) == 8
    assert [task.public.ordinal for task in training] == [0, 1]
    assert target.public.ordinal == 2
    assert target.public.task_id == probe.TARGET_TASK_ID
    assert target.public.task_sha256 == probe.TASK_RECORDS[2]["task_sha256"]
    assert hashlib.sha256(target.public.prompt.encode()).hexdigest() == probe.TASK_RECORDS[2][
        "prompt_sha256"
    ]


def test_real_donor_checkers_are_the_only_feedback_byte_source(prepared_training) -> None:
    checkers = prepared_training["feedback_by_ordinal"]
    plus = checkers[probe.ADD_DONOR_ORDINAL]
    minus = checkers[probe.SUB_DONOR_ORDINAL]
    assert probe._uniform_signed_residual(plus) == 5
    assert probe._uniform_signed_residual(minus) == -7
    assert probe._sign_observation(plus) == probe.OBS_POS_TOKEN_ID
    assert probe._sign_observation(minus) == probe.OBS_NEG_TOKEN_ID
    assert probe._feedback_magnitude_bytes(plus) == b"5"
    assert probe._feedback_magnitude_bytes(minus) == b"7"
    assert probe._feedback_slot_block(plus) == (
        probe.slot_start_token_id(2),
        ord("5"),
        probe.slot_end_token_id(2),
    )
    assert probe._feedback_slot_block(minus) == (
        probe.slot_start_token_id(2),
        ord("7"),
        probe.slot_end_token_id(2),
    )

    malformed = dict(minus)
    malformed["error"] = "SyntaxError: diagnostic failed"
    with pytest.raises(RuntimeError, match="malformed or passing"):
        probe._feedback_slot_block(malformed)
    mixed = dict(minus)
    mixed["residual"] = [-7, -6, -7]
    with pytest.raises(RuntimeError, match="mixed, zero"):
        probe._feedback_slot_block(mixed)
    inconsistent = dict(minus)
    inconsistent["residual"] = [-8, -8, -8]
    with pytest.raises(RuntimeError, match="inconsistent"):
        probe._feedback_slot_block(inconsistent)
    passing = dict(minus)
    passing["passed"] = True
    with pytest.raises(RuntimeError, match="malformed or passing"):
        probe._feedback_slot_block(passing)


def test_exact_router_and_feedback_builder_training_arithmetic(prepared_training) -> None:
    assert prepared_training["router_capacity"] == probe.EXPECTED_ROUTER_CAPACITY
    assert prepared_training["builder_capacity"] == probe.EXPECTED_BUILDER_CAPACITY
    assert probe.EXPECTED_BUILDER_CAPACITY == {
        "training_sequence_lengths": [56, 56],
        "noninitial_arrivals": 110,
        "segment_capacity": 1_210,
        "max_pyramidal_cells": 5_306,
        "threshold_offset_cells": 12_148,
    }
    assert [record["uniform_signed_residual"] for record in prepared_training["router_train"]] == [
        5,
        -7,
    ]
    assert [record["feedback_magnitude_slot"] for record in prepared_training["builder_train"]] == [
        "5",
        "7",
    ]
    assert all(
        record["feedback_lineage"] == "real_visible_identity_checker_uniform_residual"
        and record["temporal_target_contains_literal_magnitude"] is False
        for record in prepared_training["builder_train"]
    )
    for record in prepared_training["builder_train"]:
        sequence = tuple(record["sequence_token_ids"])
        assert len(sequence) == 56
        assert sequence[:24] == probe._entrypoint_slot_block(record["entrypoint_slot"])
        assert sequence[24:27] == (
            probe.slot_start_token_id(2),
            ord(record["feedback_magnitude_slot"]),
            probe.slot_end_token_id(2),
        )
        assert sequence[27] in (probe.BUILD_ADD_TOKEN_ID, probe.BUILD_SUB_TOKEN_ID)
        assert sequence[28:] == probe._builder_teacher_target(sequence[27])


def test_router_checkpoint_replays_both_donor_sign_routes(prepared_training) -> None:
    definition = prepared_training["router_definition"]
    checkpoint = prepared_training["router_checkpoint"]
    identity = prepared_training["router_checkpoint_identity"]
    for task, expected_observation, expected_build in (
        (
            prepared_training["training"][0],
            probe.OBS_POS_TOKEN_ID,
            probe.BUILD_ADD_TOKEN_ID,
        ),
        (
            prepared_training["training"][1],
            probe.OBS_NEG_TOKEN_ID,
            probe.BUILD_SUB_TOKEN_ID,
        ),
    ):
        episode = prepared_training["episodes"][task.public.ordinal]
        checker = prepared_training["feedback_by_ordinal"][task.public.ordinal]
        with probe.route_probe._public_train_authority():
            with probe.NativeTemporalCircuit.from_checkpoint(definition, checkpoint) as circuit:
                assert circuit.checkpoint_identity() == identity
                circuit.advance_block(
                    tuple(episode.token_ids),
                    mode=probe.TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                first = probe.route_probe._generate_once(circuit)
                feedback = probe.feedback_probe._advance_feedback(
                    circuit, probe._sign_observation(checker)
                )
                second = probe.route_probe._generate_once(circuit)
        assert probe.route_probe._score_generation(first, probe.TEST_TOKEN_ID)["correct"]
        assert feedback["observation_token_id"] == expected_observation
        assert probe.route_probe._score_generation(second, expected_build)["correct"]


def test_builder_checkpoint_replays_donors_with_split_persistent_slots(
    prepared_training,
) -> None:
    definition = prepared_training["builder_definition"]
    checkpoint = prepared_training["builder_checkpoint"]
    identity = prepared_training["builder_checkpoint_identity"]
    for task, build in zip(
        prepared_training["training"],
        (probe.BUILD_ADD_TOKEN_ID, probe.BUILD_SUB_TOKEN_ID),
        strict=True,
    ):
        checker = prepared_training["feedback_by_ordinal"][task.public.ordinal]
        entrypoint = task.public.signature.entry_point
        with probe.route_probe._public_train_authority():
            with probe.NativeTemporalCircuit.from_checkpoint(definition, checkpoint) as circuit:
                assert circuit.checkpoint_identity() == identity
                circuit.advance_block(
                    probe._entrypoint_slot_block(entrypoint),
                    mode=probe.TemporalStepMode.READ_ONLY,
                    reset_before=True,
                )
                circuit.advance_block(
                    probe._feedback_slot_block(checker),
                    mode=probe.TemporalStepMode.READ_ONLY,
                    reset_before=False,
                )
                circuit.advance_block(
                    (build,),
                    mode=probe.TemporalStepMode.READ_ONLY,
                    reset_before=False,
                )
                run = probe.role_place_probe._generate_teacher_free(circuit)
        analysis = probe._analyze_builder_run(entrypoint, checker, build, run)
        assert analysis["passed"] is True
        assert analysis["copy_transactions_exact"] is True
        assert analysis["operator_elected_physically"] is True
        assert analysis["ordinary_eos"] is True
        assert analysis["feedback_magnitude"] == abs(probe._uniform_signed_residual(checker))


def test_target_path_has_one_marker_then_checker_feedback_without_sidecar_builder_shortcut() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    main_source = inspect.getsource(probe.main)
    assert "pac0_probe._builder_payload" not in source
    assert "pac0_probe._offset" not in source
    assert "magnitude_override" not in source
    assert "_environment_expected_offset" not in inspect.getsource(
        probe._feedback_slot_block
    )
    assert list(inspect.signature(probe._entrypoint_slot_block).parameters) == ["entrypoint"]
    assert list(inspect.signature(probe._feedback_slot_block).parameters) == ["checker"]
    assert list(inspect.signature(probe._builder_teacher_target).parameters) == ["build_token"]
    assert list(inspect.signature(probe._analyze_builder_run).parameters) == [
        "entrypoint",
        "checker",
        "build_token",
        "run",
    ]

    pre_capture = main_source.index("pre_steps = tuple(")
    marker = main_source.index("marker = _create_spend_marker(")
    first_generate = main_source.index("actual_first = route_probe._generate_once")
    identity_execution = main_source.index(
        "target_identity_checker = _run_visible_identity_environment"
    )
    post_slot = main_source.index("observation_blocks = (")
    no_reset_slot = main_source.index("for builder, block in zip(builders, observation_blocks")
    builder_runs = main_source.index("builder_runs = tuple(")
    analysis = main_source.index("analyses = (")
    sandbox = main_source.index("sandbox = action_probe._sandbox_record(")
    score = main_source.index("_score_after_all_target_calls(")
    assert pre_capture < marker < first_generate < identity_execution < post_slot
    assert post_slot < no_reset_slot < builder_runs < analysis < sandbox < score
    assert main_source.count("_create_spend_marker(") == 1
    assert main_source.count("_run_visible_identity_environment(target_episode, target)") == 1
    assert "reset_before=False" in main_source[no_reset_slot:builder_runs]
    assert "_feedback_slot_block(target_identity_checker)" in main_source
    assert "_feedback_slot_block(payload_donor_checker)" in main_source
    assert "target.public.hidden" not in main_source
    assert "_builder_teacher_target(" not in main_source

    marker_source = inspect.getsource(probe._create_spend_marker)
    assert "router_target_no_argument_generation_calls_planned\": 4" in marker_source
    assert "builder_target_runs_planned\": 3" in marker_source
    assert "canonical_target_sandbox_calls_planned\": 1" in marker_source
    assert "open(\"x\"" in marker_source

    assert probe.EXPECTED_SOURCE_SHA256 == {
        "actual": "4e5ae71905f25d84cff5ff3932958af9d5add775e79e1bdae8b4501b92de700b",
        "sign_intervention": "1b114698a12473eede038496bf4d9f1fc45960a75c55b51c429c7769e3e92eba",
        "payload_intervention": "04b29bc752f6b4c5f5604e1210b630399607d819f0e812c675d26d248011b00e",
    }


def test_posthoc_gate_requires_independent_sign_and_real_payload_controls() -> None:
    actual_checker = {
        "inputs": list(probe.signed_probe.VISIBLE_INPUTS),
        "expected": [-6, -2, 5],
        "observed": [-4, 0, 7],
        "residual": [-2, -2, -2],
        "passed": False,
        "error": None,
    }
    donor_checker = {
        "inputs": list(probe.signed_probe.VISIBLE_INPUTS),
        "expected": [-11, -7, 0],
        "observed": [-4, 0, 7],
        "residual": [-7, -7, -7],
        "passed": False,
        "error": None,
    }
    entrypoint = probe.TASK_RECORDS[probe.TARGET_ORDINAL]["entry_point"]
    actual = probe._expected_source_bytes(entrypoint, actual_checker, probe.BUILD_SUB_TOKEN_ID)
    sign = probe._expected_source_bytes(entrypoint, actual_checker, probe.BUILD_ADD_TOKEN_ID)
    payload = probe._expected_source_bytes(entrypoint, donor_checker, probe.BUILD_SUB_TOKEN_ID)
    assert hashlib.sha256(actual).hexdigest() == probe.EXPECTED_SOURCE_SHA256["actual"]
    assert hashlib.sha256(sign).hexdigest() == probe.EXPECTED_SOURCE_SHA256["sign_intervention"]
    assert hashlib.sha256(payload).hexdigest() == probe.EXPECTED_SOURCE_SHA256[
        "payload_intervention"
    ]
    assert actual.replace(b"-", b"+", 1) == sign
    assert actual.replace(b"2", b"7", 1) == payload
    assert probe._feedback_slot_block(actual_checker) == (
        probe.slot_start_token_id(2),
        ord("2"),
        probe.slot_end_token_id(2),
    )
    assert probe._feedback_slot_block(donor_checker) == (
        probe.slot_start_token_id(2),
        ord("7"),
        probe.slot_end_token_id(2),
    )

