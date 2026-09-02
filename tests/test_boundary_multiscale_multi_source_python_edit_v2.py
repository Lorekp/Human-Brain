from __future__ import annotations

import inspect
from pathlib import PurePosixPath

from experiments.r4_integrated.boundary_multiscale_multi_source_python_edit_v2 import (
    _issue_related_paths,
    public_fresh_tasks,
    public_inference_schema,
    public_training_experiences,
    run_experiment,
)
from humanbrain.boundary_multiscale_source_motor_controller import (
    BoundaryMultiscaleSourceMotorController,
)
from humanbrain.multiscale_opaque_event_tape import MultiscaleOpaqueEventTape


def test_fresh_boundary_integrated_result_passes_first_middle_and_last(tmp_path) -> None:
    result = run_experiment(tmp_path / "integrated")

    assert result["gate_passed"] is True
    assert result["result_sha256"] == (
        "7eac8918de381657149e9cfa0aee9ddac3d8ab677023a0cc6b1ede8cecefa07b"
    )
    assert result["public_training_experience_count"] == 2
    assert result["fresh_action_task_count"] == 3
    assert result["fresh_ambiguity_task_count"] == 1
    assert result["integrity"] == {
        "fresh_task_fields": ("task_id", "issue", "files"),
        "fresh_tasks_have_four_files": True,
        "fresh_action_target_positions": (0, 2, 3),
        "all_queries_single_shot": True,
        "all_workspaces_lossless": True,
        "all_emissions_live": True,
        "all_neutral_checkpoints_retained": True,
    }

    treatment = result["arms"]["none"]
    assert treatment["action_required_success_count"] == 3
    assert treatment["ambiguity_stop_success_count"] == 1
    assert treatment["emit_count"] == 3
    assert treatment["stop_count"] == 1
    assert treatment["checkpoint_retained"] is True
    treatment_by_task = {row["task_id"]: row for row in treatment["tasks"]}

    expected_actions = {
        "fresh-auric-first": ("auric_worker.py", 0),
        "fresh-selene-middle": ("selene_worker.py", 2),
        "fresh-wyvern-last": ("wyvern_worker.py", 3),
    }
    for task_id, (expected_path, expected_position) in expected_actions.items():
        row = treatment_by_task[task_id]
        assert row["decision"] == "EMIT"
        assert row["outcome"] == "correct_edit"
        assert row["before_behavior_passed"] is False
        assert row["after_behavior_passed"] is True
        assert row["changed_paths"] == (expected_path,)
        assert row["intended_position"] == expected_position
        assert row["boundary_match_counts"] == (2,)
        assert row["decoys_preserved"] is True
        assert row["emission_count"] == 4
        assert row["emissions_from_current_tape"] is True
        assert row["materialization_valid"] is True
        assert row["neutral_checkpoint_retained"] is True

    ambiguity = treatment_by_task["fresh-ambiguous-datura-xiphos"]
    assert ambiguity["decision"] == "STOP"
    assert ambiguity["outcome"] == "correct_ambiguity_stop"
    assert ambiguity["related_path_count"] == 2
    assert ambiguity["changed_paths"] == ()
    assert ambiguity["stop_stage"] == "role_binder_0"
    assert ambiguity["stop_reason"] == "equality:collision"

    expected_action_stop_reasons = {
        "equality_off": "equality:tonic_stop",
        "propagation_off": "propagation:tonic_stop",
        "boundary_off": "boundary:left_missing_boundary",
    }
    for control, expected_reason in expected_action_stop_reasons.items():
        arm = result["arms"][control]
        assert arm["action_required_success_count"] == 0
        assert arm["emit_count"] == 0
        assert arm["stop_count"] == 4
        assert arm["checkpoint_retained"] is True
        action_rows = tuple(row for row in arm["tasks"] if not row["ambiguous"])
        assert all(row["stop_stage"] == "role_binder_0" for row in action_rows)
        assert all(row["stop_reason"] == expected_reason for row in action_rows)
        assert all(row["changed_paths"] == () for row in arm["tasks"])


def test_public_v2_boundary_is_answer_free_fresh_and_position_varied() -> None:
    assert public_inference_schema() == ("task_id", "issue", "files")
    signature = inspect.signature(BoundaryMultiscaleSourceMotorController.respond)
    assert tuple(signature.parameters) == ("self", "query", "files", "ablation")

    training = public_training_experiences()
    fresh = public_fresh_tasks()
    trained_path_stems = {
        PurePosixPath(item.path).stem for experience in training for item in experience.before_files
    }
    fresh_path_stems = {PurePosixPath(item.path).stem for task in fresh for item in task.files}
    assert trained_path_stems.isdisjoint(fresh_path_stems)

    assert tuple(len(task.files) for task in fresh) == (4, 4, 4, 4)
    assert tuple(len(_issue_related_paths(task.issue, task.files)) for task in fresh) == (
        1,
        1,
        1,
        2,
    )
    positions = tuple(
        tuple(item.path for item in task.files).index(related[0])
        if len(related := _issue_related_paths(task.issue, task.files)) == 1
        else None
        for task in fresh
    )
    assert positions == (0, 2, 3, None)

    for task in fresh:
        assert len({PurePosixPath(item.path).suffix for item in task.files}) >= 3
        observation = MultiscaleOpaqueEventTape.observe(task.issue, task.files)
        assert observation.reconstruct_query() == task.issue
        assert observation.reconstruct_workspace() == task.files
