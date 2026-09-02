from __future__ import annotations

import ast
from collections import Counter

from experiments.r4_integrated import continual_toy_swe_tasks_v1 as tasks


def test_continual_toy_swe_train_stream_is_fresh_private_and_deterministic() -> None:
    first = tasks.continual_toy_swe_train_stream_v1()
    second = tasks.continual_toy_swe_train_stream_v1()

    assert first is second
    assert len(first) == tasks.TASK_COUNT == 10
    assert tasks.train_stream_sha256_v1(first) == tasks.TRAIN_STREAM_SHA256
    assert len({task.task_sha256 for task in first}) == 10
    assert len({task.public.signature.entry_point for task in first}) == 10
    assert len({task.public.relative_path for task in first}) == 10

    private_stages = Counter(task.private.stage for task in first)
    private_groups = Counter(task.private.semantic_group for task in first)
    assert private_stages == {
        "first-exposure": 3,
        "fresh-replay": 3,
        "paraphrased-transfer": 3,
        "unseen": 1,
    }
    assert sorted(private_groups.values()) == [1, 3, 3, 3]

    forbidden_public_keys = {
        "family",
        "semantic_group",
        "stage",
        "teacher_source",
        "regression_cases",
        "property_cases",
        "expected",
    }
    for ordinal, task in enumerate(first):
        task.authenticate()
        visible = tasks.learner_visible_observation_v1(task)
        visible_record = visible.record()
        assert forbidden_public_keys.isdisjoint(visible_record)
        assert visible is task.public
        assert visible.relative_path == f"solutions/challenge_{ordinal:02d}.py"
        assert task.private.generated_task.public.partition == "train"
        assert len(task.private.generated_task.private.regression_cases) == 4
        assert len(task.private.generated_task.private.property_cases) == 12

        starter = ast.parse(visible.starter_source)
        assert len(starter.body) == 1 and isinstance(starter.body[0], ast.FunctionDef)
        assert starter.body[0].name == visible.signature.entry_point
