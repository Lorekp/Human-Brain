from __future__ import annotations

import inspect
import re
from dataclasses import fields
from pathlib import PurePosixPath

import experiments.r4_integrated.persistent_variability_multiscale_python_learning_v2 as cohort
from humanbrain.variability_multiscale_source_motor_online_growth import (
    AppendOnlyVariabilityMultiscaleSourceMotorStack,
)

_EXPECTED_SOURCE_SHA256 = (
    "b48c705d6877c6e031963ed2fddcceae32620bf51e984ee64430561a79b4a4ec"
)
_EXPECTED_COHORT_SHA256 = (
    "8a7048abff67f1ae3a2c45072fa0a5887977cd3de2eb41d127a488cddbadcaff"
)
_EXPECTED_DEPENDENCY_CUSTODY_SHA256 = (
    "649422f13ca04da23d42f005fcb576b44403c455714538559df35501098f6c89"
)
_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")


def _related_paths(
    issue: bytes,
    files: tuple[object, ...],
) -> tuple[str, ...]:
    identities = frozenset(_IDENTIFIER.findall(issue))
    return tuple(
        item.path
        for item in files
        if PurePosixPath(item.path).stem.encode("utf-8") in identities
    )


def test_persistent_variability_v2_inert_custody_spec_and_api_freeze() -> None:
    assert cohort.ARCHITECTURE_COMMIT == (
        "fd38ef6981f3b0c00d5c76557527b3aed69b9d1b"
    )
    assert cohort.experiment_source_custody_sha256() == (
        cohort.EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256
    )
    assert cohort.experiment_source_sha256() == _EXPECTED_SOURCE_SHA256
    assert cohort.cohort_sha256() == _EXPECTED_COHORT_SHA256
    assert cohort.dependency_custody()["sha256"] == (
        _EXPECTED_DEPENDENCY_CUSTODY_SHA256
    )
    assert cohort.public_inference_schema() == ("task_id", "issue", "files")
    assert tuple(field.name for field in fields(cohort.PublicPersistentVariabilityTask)) == (
        "task_id",
        "issue",
        "files",
    )
    assert tuple(
        inspect.signature(
            AppendOnlyVariabilityMultiscaleSourceMotorStack.respond
        ).parameters
    ) == ("self", "query", "files", "ablation")
    assert tuple(
        inspect.signature(
            AppendOnlyVariabilityMultiscaleSourceMotorStack.extend_successes
        ).parameters
    ) == ("self", "successes", "ablation")
    assert tuple(inspect.signature(cohort.run_and_write_artifact).parameters) == ()

    manifest = cohort.public_cohort_manifest()
    assert manifest == {
        "architecture_commit": "fd38ef6981f3b0c00d5c76557527b3aed69b9d1b",
        "causal_ablation_task_ids": (
            "persistent-v2-ablation-equinox",
            "persistent-v2-ablation-pulse",
            "persistent-v2-ablation-keystone",
        ),
        "current_success_task_ids": (
            "persistent-v2-current-cinder",
            "persistent-v2-current-umber",
        ),
        "mode": "public_train_one_shot",
        "protected_splits_opened": (),
        "seed_success_count": 2,
        "stream_task_ids": tuple(task.task_id for task in cohort.public_stream_tasks()),
        "treatment_stack_count": 1,
        "frozen_control_stack_count": 1,
    }
    assert cohort.benchmark_artifact_path().name == (
        "r4_integrated_persistent_variability_multiscale_python_learning_v2.json"
    )

    writer_source = inspect.getsource(cohort.run_and_write_artifact)
    assert ".exists()" in writer_source
    assert '.open("x"' in writer_source
    assert "_run_experiment_at" in writer_source
    model_source = inspect.getsource(cohort._model_attempt)
    assert "successful_after" not in model_source
    assert model_source.index("learner.respond") < model_source.index(
        "_issue_related_paths"
    )


def test_persistent_variability_v2_inert_public_cohort_is_fresh_and_structured() -> None:
    seed = cohort.public_seed_training_experiences()
    stream = cohort.public_stream_tasks()
    assert len(seed) == 2
    assert len(stream) == 11
    assert all(len(row.before_files) == len(row.after_files) == 5 for row in seed)
    assert all(len(task.files) == 5 for task in stream)
    assert len({row.provenance_id for row in seed}) == 2
    assert len({row.query for row in seed}) == 2
    assert len({row.before_files for row in seed}) == 2
    assert len({task.task_id for task in stream}) == len(stream)
    assert all(
        tuple(item.path for item in task.files)
        == tuple(sorted(item.path for item in task.files))
        for task in stream
    )

    seed_related = tuple(
        _related_paths(row.query, row.before_files) for row in seed
    )
    stream_related = tuple(_related_paths(task.issue, task.files) for task in stream)
    assert tuple(len(paths) for paths in seed_related) == (1, 1)
    assert tuple(len(paths) for paths in stream_related) == (
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
    )
    assert tuple(len(PurePosixPath(paths[0]).parts) for paths in seed_related) == (
        2,
        2,
    )
    assert tuple(
        None if len(paths) != 1 else len(PurePosixPath(paths[0]).parts)
        for paths in stream_related
    ) == (2, 2, 3, 3, 3, 2, None, 4, 3, 3, 3)
    assert tuple(
        None
        if len(paths) != 1
        else tuple(item.path for item in task.files).index(paths[0])
        for task, paths in zip(stream, stream_related, strict=True)
    ) == (2, 3, 0, 4, 2, 3, None, 2, 1, 1, 3)

    seed_paths = {item.path for row in seed for item in row.before_files}
    stream_paths = {item.path for task in stream for item in task.files}
    seed_directories = {PurePosixPath(path).parts[0] for path in seed_paths}
    stream_directories = {PurePosixPath(path).parts[0] for path in stream_paths}
    assert seed_paths.isdisjoint(stream_paths)
    assert seed_directories.isdisjoint(stream_directories)

    seed_functions = {
        function
        for row in seed
        for item in row.before_files
        for function in re.findall(
            rb"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\(", item.content
        )
    }
    stream_functions = {
        function
        for task in stream
        for item in task.files
        for function in re.findall(
            rb"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\(", item.content
        )
    }
    assert len(seed_functions) == 2
    assert len(stream_functions) == 12
    assert seed_functions.isdisjoint(stream_functions)
