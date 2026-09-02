from __future__ import annotations

from dataclasses import fields

from experiments.r4_integrated import cue_free_python_online_growth_v1 as experiment
from experiments.r4_integrated import cue_free_python_source_edit_pilot_v1 as transport


def test_receipt_bound_growth_transfers_while_no_update_stops(tmp_path) -> None:
    result = experiment.run_experiment(tmp_path)

    assert result["status"] == experiment.STATUS_PASS
    assert all(result["checks"].values())
    assert result["fresh_transfer_count"] == 2

    treatment = result["arms"]["treatment"]
    assert treatment["emits"] == 2
    assert treatment["behavior_passes"] == 2
    assert all(
        record["current_tape_occurrences_only"] and record["query_count"] == 1
        for record in treatment["records"]
    )

    no_update = result["arms"]["no_update"]
    assert no_update["emits"] == 0
    assert no_update["stops"] == 2
    assert all(record["physical_stop"] for record in no_update["records"])

    growth = result["growth"]
    assert growth["reserved_cell_capacity_added"] == 4_387
    assert growth["reserved_synapse_capacity_estimate_added"] == 61
    assert growth["coactivation_support_tick_estimate_added"] == 61
    assert growth["positive_success_support_tick_estimate_added"] == 49
    assert growth["excitatory_support_tick_estimate_added"] == 110
    assert growth["parent_experience_replay_count"] == 0
    assert growth["raw_training_sequences_retained"] == 0


def test_public_stream_has_no_fresh_target_and_shifts_live_path_position() -> None:
    assert tuple(field.name for field in fields(transport.PublicPythonRepositoryTask)) == (
        "task_id",
        "issue",
        "files",
    )
    tasks = experiment.public_online_growth_tasks()
    assert len(tasks) == 3

    python_paths: list[str] = []
    sorted_path_positions: list[int] = []
    python_sources: list[bytes] = []
    for task in tasks:
        assert not any(
            word in field.name.lower()
            for field in fields(task)
            for word in ("target", "gold", "after", "expected")
        )
        assert b"".join(transport.tokenize_issue_losslessly(task.issue)) == task.issue
        tape = transport.canonical_workspace_tape(task.files)
        assert transport.decode_canonical_workspace_tape(tape) == tuple(
            sorted(task.files, key=lambda item: item.path)
        )
        python_file = next(item for item in task.files if item.path.endswith(".py"))
        ordered_paths = tuple(item.path for item in sorted(task.files, key=lambda item: item.path))
        python_paths.append(python_file.path)
        python_sources.append(python_file.source)
        sorted_path_positions.append(ordered_paths.index(python_file.path))

    assert len(set(python_paths)) == len(tasks)
    assert len(set(python_sources)) == len(tasks)
    assert sorted_path_positions == [1, 0, 3]
