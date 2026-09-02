from __future__ import annotations

import ast
from collections import Counter
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated import continual_toy_swe_tasks_v6 as tasks_v6

_SOURCE_SHA256 = "34a19e89e8893f2a496cda422e7e2d5693c7d65cb68a4ebc1a0699e8b9049e24"
_STREAM_SHA256 = "4291faf1c0035dbff79b40250ab8cfd0e718d0a610ffd73684481a93fed60303"
_CUSTODY_SHA256 = "e19d14234b9d3bef90927e176590ede49f17c8280208712ee0a455b743cf672b"


def test_v6_source_stream_and_custody_are_frozen() -> None:
    stream = tasks_v6.continual_toy_swe_train_stream_v6()
    source = Path(tasks_v6.__file__).resolve().read_bytes()

    assert sha256(source).hexdigest() == _SOURCE_SHA256
    assert tasks_v6.TRAIN_STREAM_SHA256 == _STREAM_SHA256
    assert tasks_v6.train_stream_sha256_v6(stream) == _STREAM_SHA256
    assert tasks_v6.TRAIN_STREAM_CUSTODY_SHA256 == _CUSTODY_SHA256
    custody = tasks_v6.train_stream_custody_v6()
    assert custody["train_stream_sha256"] == _STREAM_SHA256
    assert custody["assembly_model_imports_during_authoring"] == 0
    assert custody["assembly_model_outputs_used_during_authoring"] == 0


def test_v6_public_projection_contains_only_target_free_inputs() -> None:
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
        "candidate_patch",
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
        "target_span",
    }

    for task in tasks_v6.continual_toy_swe_train_stream_v6():
        public = tasks_v6.learner_visible_observation_v6(task)
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


def test_v6_pressure_balance_private_cases_and_retention_links_are_frozen() -> None:
    stream = tasks_v6.continual_toy_swe_train_stream_v6()

    assert len(stream) == 16
    assert tasks_v6.MAX_ATTEMPTS == 3
    assert Counter(len(task.public.signature.parameters) for task in stream) == {
        1: 4,
        2: 4,
        4: 4,
        5: 4,
    }
    assert Counter(task.private.stage for task in stream) == {
        "renamed-transfer": 4,
        "nested-multifragment-recombination": 4,
        "execution-feedback-revision": 4,
        "delayed-retention-recheck": 2,
        "insufficient-evidence": 2,
    }
    assert Counter(task.private.expected_action for task in stream) == {
        "execute": 14,
        "stop": 2,
    }
    assert Counter(task.private.attempt_budget for task in stream) == {
        1: 2,
        2: 6,
        3: 8,
    }
    assert {
        index: task.private.retention_anchor_ordinal
        for index, task in enumerate(stream)
        if task.private.retention_anchor_ordinal is not None
    } == {12: 1, 13: 6}
    assert len({task.public.signature.entry_point for task in stream}) == 16
    assert len({task.public.prompt for task in stream}) == 16
    assert len({task.public.starter_source for task in stream}) == 16

    for task in stream:
        private = task.private.generated_task.private
        expected_counts = (
            (0, 0)
            if task.private.expected_action == "stop"
            else (
                tasks_v6.EXECUTION_REGRESSION_CASES,
                tasks_v6.EXECUTION_PROPERTY_CASES,
            )
        )
        assert (len(private.regression_cases), len(private.property_cases)) == (
            expected_counts
        )
