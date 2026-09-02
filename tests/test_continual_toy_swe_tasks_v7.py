"""Frozen fixture checks for the unspent public-TRAIN V7 task stream.

These tests authenticate task construction and custody only.  They never import,
instantiate, or run a learner and never create a benchmark artifact.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated import continual_toy_swe_tasks_v7 as fixture

FROZEN_TASK_FILE_SHA256 = (
    "4e3f3f9cc1ccaf20f7b5b0fda1730e5a7fe532f7c11e3ef0dafdd5327970d8b0"
)


def _signature_shape(task: fixture.ContinualToySweTrainTaskV7) -> object:
    record = task.public.signature.to_record()
    return (
        tuple(parameter["type"] for parameter in record["parameters"]),
        record["result"],
    )


def test_v7_frozen_logical_source_and_custody_roots() -> None:
    source = Path(fixture.__file__).read_bytes()
    assert sha256(source).hexdigest() == FROZEN_TASK_FILE_SHA256
    assert fixture.train_stream_sha256_v7() == fixture.TRAIN_STREAM_SHA256
    assert fixture.task_source_sha256_v7() == fixture.TASK_SOURCE_SHA256
    custody = fixture.train_stream_custody_v7()
    assert custody["train_stream_sha256"] == fixture.TRAIN_STREAM_SHA256
    assert custody["task_source_sha256"] == fixture.TASK_SOURCE_SHA256
    assert fixture.TRAIN_STREAM_SHA256 == (
        "63227e99f894654a32873b100682f30baa786d3a5e06018d2083f1b58820a117"
    )
    assert fixture.TASK_SOURCE_SHA256 == (
        "a73bdc1fb15fceccdc2419ca18a279de0d023a57e4ce4c39497117f637c8f2cc"
    )
    assert fixture.TRAIN_STREAM_CUSTODY_SHA256 == (
        "25d7db8cb294f5aec1f5953a336b1ada1bd36917fe0e7052081b86bfb352f35e"
    )


def test_v7_learner_projection_is_minimal_and_requires_source_inspection() -> None:
    tasks = fixture.continual_toy_swe_train_stream_v7()
    public_keys = {
        "schema",
        "partition",
        "task_commitment_sha256",
        "relative_path",
        "signature",
        "prompt",
        "starter_source",
    }
    forbidden = {
        "answer",
        "attempt_budget",
        "candidate_patch",
        "capability_group",
        "diagnostic_class",
        "expected_action",
        "feedback_pressure",
        "oracle",
        "private_test_sha256",
        "property_cases",
        "regression_cases",
        "repaired_source",
        "retention_anchor_ordinal",
        "stage",
        "target_span",
        "task_ontology",
    }
    assert [field.name for field in fields(fixture.LearnerVisibleToySweObservationV7)] == [
        "task_commitment_sha256",
        "relative_path",
        "signature",
        "prompt",
        "starter_source",
        "observation_sha256",
    ]
    for task in tasks:
        public = fixture.learner_visible_observation_v7(task)
        record = public.record()
        assert set(record) == public_keys
        assert record["partition"] == "train"
        assert forbidden.isdisjoint(record)
        serialized = json.dumps(record, sort_keys=True)
        assert all(f'"{name}"' not in serialized for name in forbidden)

        tree = ast.parse(public.starter_source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 3
        assert functions[1].name == public.signature.entry_point
        assert isinstance(functions[1].body[-1], ast.Raise)
        assert "NotImplementedError" in ast.dump(functions[1].body[-1])
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)

    calibration = tasks[3].public
    assert calibration.signature.to_record()["result"]["constructor"] == "Int"
    assert "Return the sum of the clamped results" in calibration.prompt
    assert "or 0 when lower is greater than upper" in calibration.prompt


def test_v7_stream_distribution_feedback_and_retention_are_fixed() -> None:
    tasks = fixture.continual_toy_swe_train_stream_v7()
    assert len(tasks) == fixture.TASK_COUNT == 16
    assert Counter(task.private.stage for task in tasks) == {
        "public-family-acquisition": 4,
        "cross-family-recombination": 4,
        "execution-feedback-revision": 4,
        "delayed-retention-recheck": 2,
        "insufficient-evidence": 2,
    }
    assert Counter(task.private.expected_action for task in tasks) == {
        "execute": 14,
        "stop": 2,
    }
    assert Counter(task.private.attempt_budget for task in tasks) == {1: 2, 2: 6, 3: 8}
    assert Counter(len(task.public.signature.parameters) for task in tasks) == {
        1: 4,
        2: 4,
        3: 4,
        5: 4,
    }
    assert {
        ordinal: task.private.retention_anchor_ordinal
        for ordinal, task in enumerate(tasks)
        if task.private.retention_anchor_ordinal is not None
    } == {12: 0, 13: 1}
    assert sum(task.private.attempt_budget - 1 for task in tasks) == 22
    assert sum(task.private.attempt_budget == 3 for task in tasks) == 8
    assert len({task.private.capability_group for task in tasks}) == 16
    assert all(task.private.capability_group.startswith("g7-") for task in tasks)

    assert _signature_shape(tasks[12]) == _signature_shape(tasks[0])
    assert _signature_shape(tasks[13]) == _signature_shape(tasks[1])
    assert tasks[12].public.prompt != tasks[0].public.prompt
    assert tasks[13].public.prompt != tasks[1].public.prompt


def test_v7_private_cases_are_deterministic_and_stop_is_evidence_limited() -> None:
    tasks = fixture.continual_toy_swe_train_stream_v7()
    hidden_case_total = 0
    for task in tasks:
        task.authenticate()
        generated = task.private.generated_task
        assert generated.public.partition == fixture.PARTITION
        assert generated.public.family == "F7"
        if task.private.expected_action == "execute":
            assert len(generated.private.regression_cases) == 5
            assert len(generated.private.property_cases) == 9
            assert {
                case.distribution for case in generated.private.property_cases
            } == {"v7-deterministic-property"}
            hidden_case_total += 14
        else:
            assert task.private.stage == "insufficient-evidence"
            assert task.private.attempt_budget == 1
            assert not generated.private.regression_cases
            assert not generated.private.property_cases
            assert (
                "absent" in task.public.prompt
                or "No coefficients" in task.public.prompt
            )
            assert "Do not guess" in task.public.prompt
    assert hidden_case_total == 196

    # Shared inputs make the order-sensitive revision opportunities falsifiable.
    for earlier, later in ((4, 8), (5, 9), (7, 11)):
        earlier_cases = tasks[earlier].private.generated_task.private.regression_cases
        later_cases = tasks[later].private.generated_task.private.regression_cases
        earlier_by_args = {
            json.dumps(case.args, separators=(",", ":")): case.expected
            for case in earlier_cases
        }
        later_by_args = {
            json.dumps(case.args, separators=(",", ":")): case.expected
            for case in later_cases
        }
        shared = earlier_by_args.keys() & later_by_args.keys()
        assert shared
        assert any(earlier_by_args[key] != later_by_args[key] for key in shared)


def test_v7_custody_denies_answer_scaffolds_and_benchmark_claims() -> None:
    custody = fixture.train_stream_custody_v7()
    assert custody["public_train_only"] is True
    assert custody["task_count"] == 16
    assert custody["executable_hidden_cases_per_task"] == 14
    assert custody["three_attempt_feedback_tasks"] == 8
    assert custody["later_attempt_opportunities"] == 22
    assert custody["learner_visible_fields"] == [
        "task_commitment_sha256",
        "relative_path",
        "signature",
        "prompt",
        "starter_source",
    ]
    for key in (
        "learner_or_model_imports_during_authoring",
        "learner_or_model_outputs_used_during_authoring",
        "learner_runs_during_authoring",
        "benchmark_artifacts_created",
        "successful_target_sources_stored",
        "candidate_patches_stored",
        "target_spans_stored",
        "answer_bearing_diagnostic_classes_stored",
        "task_specific_learner_ontologies_stored",
        "protected_dev_or_evaluation_inputs_used",
        "exact_v6_task_commitments_reused",
    ):
        assert custody[key] == 0

    source = Path(fixture.__file__).read_text(encoding="utf-8")
    assert "PersistentCodingLearner" not in source
    for prohibited_module in (
        "assembly_factorized_pointer",
        "factorized_fragment_memory",
        "recurrent_assembly_stack",
        "role_relative_compositor",
    ):
        assert prohibited_module not in source
