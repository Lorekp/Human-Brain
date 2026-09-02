from __future__ import annotations

import ast
from collections import Counter
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated import continual_toy_swe_tasks_v5 as tasks_v5

_SOURCE_SHA256 = "c73b173265f2d4478fc03a9fffbd6d4d657e4f21aa29e0a47fcb89152ca3d8a4"
_STREAM_SHA256 = "f5abad0bd725b6946ea9ee98e74752c03940bcfe615de8ef1caa0c598b51d1e2"
_CUSTODY_SHA256 = "4081b581af922b78c0cffbc6d4241b474b3a65f1c571e998d0a90bdeb08ad495"


def test_v5_source_stream_and_custody_are_frozen() -> None:
    stream = tasks_v5.continual_toy_swe_train_stream_v5()
    source = Path(tasks_v5.__file__).resolve().read_bytes()

    assert sha256(source).hexdigest() == _SOURCE_SHA256
    assert tasks_v5.TRAIN_STREAM_SHA256 == _STREAM_SHA256
    assert tasks_v5.train_stream_sha256_v5(stream) == _STREAM_SHA256
    assert tasks_v5.TRAIN_STREAM_CUSTODY_SHA256 == _CUSTODY_SHA256
    assert tasks_v5.train_stream_custody_v5()["train_stream_sha256"] == (
        _STREAM_SHA256
    )


def test_v5_public_projection_contains_only_target_free_inputs() -> None:
    expected_fields = {
        "partition",
        "prompt",
        "relative_path",
        "schema",
        "signature",
        "starter_source",
        "task_commitment_sha256",
    }
    forbidden_fields = {
        "attempt_budget",
        "capability_group",
        "expected_action",
        "feedback_pressure",
        "generated_task",
        "oracle",
        "private_test_sha256",
        "property_cases",
        "regression_cases",
        "retention_anchor_ordinal",
        "stage",
        "successful_source",
    }

    for task in tasks_v5.continual_toy_swe_train_stream_v5():
        public = tasks_v5.learner_visible_observation_v5(task)
        record = public.record()
        assert set(record) == expected_fields
        assert not forbidden_fields & set(record)

        tree = ast.parse(public.starter_source)
        target = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == public.signature.entry_point
        ]
        assert len(target) == 1
        assert not any(isinstance(node, ast.Return) for node in ast.walk(target[0]))
        assert any(isinstance(node, ast.Raise) for node in ast.walk(target[0]))


def test_v5_integrated_pressure_distribution_and_retention_links_are_frozen() -> None:
    stream = tasks_v5.continual_toy_swe_train_stream_v5()

    assert len(stream) == 16
    assert Counter(len(task.public.signature.parameters) for task in stream) == {
        1: 4,
        2: 4,
        4: 4,
        5: 4,
    }
    assert Counter(task.private.stage for task in stream) == {
        "renamed-transfer": 4,
        "novel-factor-recombination": 4,
        "execution-feedback-revision": 4,
        "delayed-retention-recheck": 2,
        "insufficient-evidence": 2,
    }
    assert Counter(task.private.expected_action for task in stream) == {
        "execute": 14,
        "stop": 2,
    }
    assert {
        index: task.private.retention_anchor_ordinal
        for index, task in enumerate(stream)
        if task.private.retention_anchor_ordinal is not None
    } == {12: 0, 13: 7}

    for task in stream:
        private = task.private.generated_task.private
        expected_counts = (
            (0, 0)
            if task.private.expected_action == "stop"
            else (
                tasks_v5.EXECUTION_REGRESSION_CASES,
                tasks_v5.EXECUTION_PROPERTY_CASES,
            )
        )
        assert (len(private.regression_cases), len(private.property_cases)) == (
            expected_counts
        )
