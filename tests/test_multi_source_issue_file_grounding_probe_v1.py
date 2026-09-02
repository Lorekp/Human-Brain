from __future__ import annotations

from dataclasses import fields

from experiments.r4_integrated import cue_free_python_source_edit_pilot_v1 as transport
from experiments.r4_integrated import multi_source_issue_file_grounding_probe_v1 as probe


def test_fixed_probe_records_multi_source_failure_without_failing_execution(
    tmp_path,
) -> None:
    result = probe.run_probe(tmp_path)

    assert result["status"] == probe.STATUS_PASS
    assert all(result["integrity_checks"].values())
    assert result["training_repository_count"] == 2
    assert result["fresh_task_count"] == 3
    assert result["dependency_custody"] == probe._EXPECTED_DEPENDENCY_CUSTODY
    assert result["public_inference_fields"] == ("issue_tokens", "workspace_tape")
    assert result["result_sha256"] == probe._record_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )

    records = result["task_records"]
    assert tuple(record["query_count"] for record in records) == (1, 1, 1)
    assert sum(result["failure_distribution"].values()) == 3
    assert result["failure_distribution"] == {
        "ambiguity_stop": 1,
        "binder_stop": 2,
    }
    assert result["scientific_outcome"] == probe.OUTCOME_LOCALIZED
    assert not result["capability_checks"][
        "all_named_relations_grounded_and_executed"
    ]
    assert result["capability_checks"]["genuine_ambiguity_stopped"]


def test_public_frames_are_complete_matched_and_contain_no_routing_fields() -> None:
    assert tuple(field.name for field in fields(probe.PublicMultiSourceTask)) == (
        "task_id",
        "issue",
        "files",
    )
    assert tuple(field.name for field in fields(probe.PublicInferenceFrame)) == (
        "issue_tokens",
        "workspace_tape",
    )

    tasks = probe.public_train_tasks()
    for index, task in enumerate(tasks):
        frame = probe.public_inference_frame(task)
        assert b"".join(frame.issue_tokens) == task.issue
        assert transport.decode_canonical_workspace_tape(frame.workspace_tape) == tuple(
            sorted(task.files, key=lambda item: item.path)
        )
        assert 3 <= len(task.files) <= 5
        assert all(item.path.endswith("_worker.py") for item in task.files)
        assert len({probe._layout_shape(item.source) for item in task.files}) == 1
        expected_relation_count = 2 if index == len(tasks) - 1 else 1
        assert len(probe._issue_related_paths(task.issue, task.files)) == expected_relation_count


def test_observable_evaluator_rejects_noop_blank_target_and_decoy_breakage(
    tmp_path,
) -> None:
    task = probe.public_train_tasks()[0]
    intended_path = probe._issue_related_paths(task.issue, task.files)[0]
    decoy_path = next(item.path for item in task.files if item.path != intended_path)

    assert not probe._run_python_file(task.files, intended_path, tmp_path / "noop")

    blank_target = transport.apply_emitted_source_action(
        task.files,
        (*transport._path_tokens(intended_path), *transport._source_tokens(b"\n")),
    )
    assert not probe._run_python_file(
        blank_target,
        intended_path,
        tmp_path / "blank-target",
    )

    blank_decoy = transport.apply_emitted_source_action(
        task.files,
        (*transport._path_tokens(decoy_path), *transport._source_tokens(b"\n")),
    )
    assert not probe._decoys_preserved(task.files, blank_decoy, intended_path)
    assert not probe._run_python_file(
        blank_decoy,
        intended_path,
        tmp_path / "broken-decoy",
    )
