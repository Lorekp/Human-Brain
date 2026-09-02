from __future__ import annotations

from dataclasses import fields

import pytest
from experiments.r4_integrated import cue_free_python_source_edit_pilot_v1 as pilot


def test_one_whole_workspace_query_rewrites_fresh_python_and_controls_stop(
    tmp_path,
) -> None:
    result = pilot.run_pilot(tmp_path)

    assert result["status"] == pilot.STATUS_PASS
    assert all(result["checks"].values())
    assert result["training_repository_count"] == 2
    assert result["fresh_task_count"] == 4

    treatment = result["arms"]["treatment"]
    assert treatment["action_emits"] == 3
    assert treatment["action_pre_edit_failures"] == 3
    assert treatment["action_behavior_passes"] == 3
    assert treatment["all_emissions_from_current_tape"]
    assert treatment["total_stops"] == 1
    assert treatment["binder_stops"] == 1

    binder_off = result["arms"]["binder_off"]
    assert binder_off["total_emits"] == 0
    assert binder_off["binder_stops"] == 4

    tape_off = result["arms"]["tape_off"]
    assert tape_off["total_emits"] == 0
    assert tape_off["tape_stops"] == 3
    assert tape_off["binder_stops"] == 1

    ambiguity = next(
        record
        for record in treatment["task_records"]
        if record["task_id"] == "fresh-ambiguous-stop"
    )
    assert ambiguity["decision"] == "STOP"
    assert ambiguity["stop_stage"] == "binder"
    assert "collision" in ambiguity["stop_reason"]
    assert not ambiguity["workspace_changed"]


def test_public_frames_are_lossless_and_contain_no_target_or_gold_field() -> None:
    assert tuple(field.name for field in fields(pilot.PublicPythonRepositoryTask)) == (
        "task_id",
        "issue",
        "files",
    )

    for task in pilot.public_train_tasks():
        issue_tokens = pilot.tokenize_issue_losslessly(task.issue)
        workspace_tape = pilot.canonical_workspace_tape(task.files)

        assert b"".join(issue_tokens) == task.issue
        assert pilot.decode_canonical_workspace_tape(workspace_tape) == tuple(
            sorted(task.files, key=lambda item: item.path)
        )
        expected_python_files = 2 if task.task_id == "fresh-ambiguous-stop" else 1
        assert sum(item.path.endswith(".py") for item in task.files) == expected_python_files


def test_path_only_is_rejected_and_blank_replacement_fails_behavior(tmp_path) -> None:
    task = pilot.public_train_tasks()[0]
    python_file = next(item for item in task.files if item.path.endswith(".py"))
    path_tokens = pilot._path_tokens(python_file.path)

    with pytest.raises(ValueError, match="at least one source token"):
        pilot.apply_emitted_source_action(task.files, path_tokens)

    blank_files = pilot.apply_emitted_source_action(
        task.files,
        (*path_tokens, *pilot._source_tokens(b"\n")),
    )
    assert not pilot._run_python_behavior(blank_files, tmp_path / "blank-workspace")
