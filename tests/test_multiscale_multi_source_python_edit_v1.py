from __future__ import annotations

import inspect
from pathlib import PurePosixPath

import pytest
from experiments.r4_integrated.multiscale_multi_source_python_edit_v1 import (
    _issue_related_paths,
    _run_python_file,
    public_fresh_tasks,
    public_inference_schema,
    public_training_experiences,
    run_experiment,
)
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueEventTapeError,
)
from humanbrain.multiscale_source_motor_controller import (
    MultiscaleSourceMotorController,
)
from humanbrain.source_relative_action_tape import TapeValence


def test_fresh_integrated_result_localizes_boundary_position_overfit(tmp_path) -> None:
    result = run_experiment(tmp_path / "integrated")

    assert result["gate_passed"] is False
    assert result["public_training_experience_count"] == 2
    assert result["fresh_action_task_count"] == 2
    assert result["fresh_ambiguity_task_count"] == 1
    assert all(result["integrity"].values())

    treatment = result["arms"]["none"]
    assert treatment["action_required_success_count"] == 1
    assert treatment["ambiguity_stop_success_count"] == 1
    assert treatment["emit_count"] == 1
    assert treatment["stop_count"] == 2
    treatment_by_task = {row["task_id"]: row for row in treatment["tasks"]}

    lumen = treatment_by_task["fresh-lumen"]
    assert lumen["decision"] == "EMIT"
    assert lumen["outcome"] == "correct_edit"
    assert lumen["before_behavior_passed"] is False
    assert lumen["after_behavior_passed"] is True
    assert lumen["changed_paths"] == ("lumen_worker.py",)
    assert lumen["decoys_preserved"] is True
    assert lumen["emission_count"] == 4
    assert lumen["emissions_from_current_tape"] is True

    vireo = treatment_by_task["fresh-vireo"]
    assert vireo["decision"] == "STOP"
    assert vireo["outcome"] == "grounding_or_motor_stop"
    assert vireo["stop_stage"] == "role_binder_0"
    assert vireo["stop_reason"] == "containment:tonic_stop"
    assert vireo["before_behavior_passed"] is False
    assert vireo["after_behavior_passed"] is False
    assert vireo["changed_paths"] == ()

    ambiguity = treatment_by_task["fresh-ambiguous-prism-quartz"]
    assert ambiguity["decision"] == "STOP"
    assert ambiguity["outcome"] == "correct_ambiguity_stop"
    assert ambiguity["related_path_count"] == 2
    assert ambiguity["stop_stage"] == "role_binder_0"
    assert ambiguity["stop_reason"] == "equality:collision"

    for control in ("equality_off", "propagation_off"):
        arm = result["arms"][control]
        assert arm["action_required_success_count"] == 0
        assert arm["emit_count"] == 0
        assert arm["stop_count"] == 3
        assert arm["checkpoint_retained"] is True


def test_public_boundary_is_complete_cue_free_and_fresh() -> None:
    assert public_inference_schema() == ("task_id", "issue", "files")
    signature = inspect.signature(MultiscaleSourceMotorController.respond)
    assert tuple(signature.parameters) == ("self", "query", "files", "ablation")

    training = public_training_experiences()
    fresh = public_fresh_tasks()
    trained_path_stems = {
        PurePosixPath(item.path).stem for experience in training for item in experience.before_files
    }
    fresh_path_stems = {PurePosixPath(item.path).stem for task in fresh for item in task.files}
    assert trained_path_stems.isdisjoint(fresh_path_stems)

    for task in fresh:
        assert len(task.files) == 3
        assert all(item.path.endswith("_worker.py") for item in task.files)
        assert all(item.content.count(b"\n") == 4 for item in task.files)
        observation = MultiscaleOpaqueEventTape.observe(task.issue, task.files)
        assert observation.reconstruct_query() == task.issue
        assert observation.reconstruct_workspace() == tuple(
            sorted(task.files, key=lambda item: item.path)
        )

    assert tuple(len(_issue_related_paths(task.issue, task.files)) for task in fresh) == (
        1,
        1,
        2,
    )


def test_path_only_or_blank_emission_cannot_pass_exact_execution(tmp_path) -> None:
    controller = MultiscaleSourceMotorController.grow(public_training_experiences())
    task = public_fresh_tasks()[0]
    observation = MultiscaleOpaqueEventTape.observe(task.issue, task.files)
    decision = controller.respond(task.issue, task.files)
    assert decision.decision == "EMIT"
    assert decision.emissions
    controller.resolve(TapeValence.NEUTRAL)

    intended_path = _issue_related_paths(task.issue, task.files)[0]
    path_only_workspace = observation.materialize_motor_emission((decision.emissions[0],))
    assert not _run_python_file(
        path_only_workspace,
        intended_path,
        tmp_path / "path-only",
    )
    with pytest.raises(OpaqueEventTapeError, match="non-empty tuple"):
        observation.materialize_motor_emission(())
