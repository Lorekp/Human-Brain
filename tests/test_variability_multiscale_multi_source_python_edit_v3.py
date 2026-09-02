from __future__ import annotations

import inspect
import re
from dataclasses import fields
from pathlib import PurePosixPath

from experiments.r4_integrated.variability_multiscale_multi_source_python_edit_v3 import (
    ARCHITECTURE_COMMIT,
    EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256,
    PublicVariabilityMultiSourceTask,
    benchmark_artifact_path,
    cohort_sha256,
    dependency_custody,
    experiment_source_custody_sha256,
    experiment_source_sha256,
    public_fresh_tasks,
    public_inference_schema,
    public_training_experiences,
    run_and_write_artifact,
)
from humanbrain.variability_multiscale_source_motor_controller import (
    VariabilityMultiscaleSourceMotorController,
)

_EXPECTED_SOURCE_SHA256 = (
    "22f4200df5cce8991d72f2007c9a94e8c8f7793b35d6be5897b0d9beccb5c467"
)
_EXPECTED_COHORT_SHA256 = (
    "60bae080e6ade42b507b3a0525540c00ab98fe8b8550d0edbc6951aedb4560fb"
)
_EXPECTED_DEPENDENCY_CUSTODY_SHA256 = (
    "cfaee1dac619d091d4545d1a816a176a321e3f4656b01b8287cdbce831385457"
)
_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")


def _related_paths(task: PublicVariabilityMultiSourceTask) -> tuple[str, ...]:
    identities = frozenset(_IDENTIFIER.findall(task.issue))
    return tuple(
        item.path
        for item in task.files
        if PurePosixPath(item.path).stem.encode("utf-8") in identities
    )


def test_v3_inert_source_dependency_cohort_and_api_freeze() -> None:
    assert ARCHITECTURE_COMMIT == "101edfb86470f0124f0859f782f7f952671f5115"
    assert experiment_source_custody_sha256() == (
        EXPECTED_EXPERIMENT_SOURCE_CUSTODY_SHA256
    )
    assert experiment_source_sha256() == _EXPECTED_SOURCE_SHA256
    assert cohort_sha256() == _EXPECTED_COHORT_SHA256
    assert dependency_custody()["sha256"] == _EXPECTED_DEPENDENCY_CUSTODY_SHA256
    assert public_inference_schema() == ("task_id", "issue", "files")
    assert tuple(field.name for field in fields(PublicVariabilityMultiSourceTask)) == (
        "task_id",
        "issue",
        "files",
    )
    assert tuple(
        inspect.signature(VariabilityMultiscaleSourceMotorController.respond).parameters
    ) == ("self", "query", "files", "ablation")
    assert tuple(inspect.signature(run_and_write_artifact).parameters) == ()
    assert benchmark_artifact_path().name == (
        "r4_integrated_variability_multiscale_multi_source_python_edit_v3.json"
    )
    writer_source = inspect.getsource(run_and_write_artifact)
    assert ".exists()" in writer_source
    assert '.open("x"' in writer_source


def test_v3_inert_public_cohort_is_independent_fresh_and_position_varied() -> None:
    training = public_training_experiences()
    fresh = public_fresh_tasks()

    assert len(training) == 2
    assert len({row.provenance_id for row in training}) == 2
    assert len({row.query for row in training}) == 2
    assert len({row.before_files for row in training}) == 2
    assert len(fresh) == 4
    assert tuple(len(task.files) for task in fresh) == (4, 4, 4, 4)

    trained_paths = {
        item.path for experience in training for item in experience.before_files
    }
    fresh_paths = {item.path for task in fresh for item in task.files}
    trained_directories = {PurePosixPath(path).parts[0] for path in trained_paths}
    fresh_directories = {PurePosixPath(path).parts[0] for path in fresh_paths}
    trained_stems = {PurePosixPath(path).stem for path in trained_paths}
    fresh_stems = {PurePosixPath(path).stem for path in fresh_paths}
    trained_functions = {
        function
        for experience in training
        for item in experience.before_files
        for function in re.findall(rb"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\(", item.content)
    }
    fresh_functions = {
        function
        for task in fresh
        for item in task.files
        for function in re.findall(rb"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\(", item.content)
    }
    assert trained_paths.isdisjoint(fresh_paths)
    assert trained_directories.isdisjoint(fresh_directories)
    assert trained_stems.isdisjoint(fresh_stems)
    assert trained_functions.isdisjoint(fresh_functions)
    assert all(
        len(PurePosixPath(path).parts) == 2 for path in trained_paths | fresh_paths
    )

    related = tuple(_related_paths(task) for task in fresh)
    assert tuple(len(paths) for paths in related) == (1, 1, 1, 2)
    assert tuple(
        tuple(item.path for item in task.files).index(paths[0])
        if len(paths) == 1
        else None
        for task, paths in zip(fresh, related, strict=True)
    ) == (0, 2, 3, None)
