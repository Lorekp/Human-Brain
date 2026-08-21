from __future__ import annotations

from pathlib import Path

import pytest

from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import generate_visible_partition


PROTOCOL = (
    Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"
)


def test_train_and_dev_tasks_are_deterministic_and_family_disjoint() -> None:
    protocol = load_protocol(PROTOCOL)

    first_train = generate_visible_partition(protocol, "train")
    second_train = generate_visible_partition(protocol, "train")
    dev = generate_visible_partition(protocol, "dev")

    assert first_train == second_train
    assert len(first_train) == 90
    assert len(dev) == 27
    assert {task.public.family for task in first_train}.isdisjoint(
        task.public.family for task in dev
    )
    assert len({task.public.task_sha256 for task in first_train + dev}) == 117


def test_every_visible_task_has_separate_regression_and_property_tests() -> None:
    protocol = load_protocol(PROTOCOL)

    for task in generate_visible_partition(protocol, "dev"):
        assert len(task.private.regression_cases) == 4
        assert len(task.private.property_cases) == 32
        assert task.private.test_sha256
        assert task.public.task_sha256
        assert "assert " not in task.public.prompt
        assert "Example" not in task.public.prompt
        assert task.private.test_sha256 not in task.public.prompt
        assert {case.distribution for case in task.private.property_cases}


def test_eval_tasks_cannot_be_materialized_by_the_visible_generator() -> None:
    protocol = load_protocol(PROTOCOL)

    with pytest.raises(PermissionError, match="post-freeze reveal"):
        generate_visible_partition(protocol, "eval")


def test_protocol_has_no_eval_seed_and_preregisters_all_eval_families() -> None:
    protocol = load_protocol(PROTOCOL)
    evaluation = protocol.partitions["eval"]

    assert evaluation.seed is None
    assert evaluation.seed_strategy == "post-freeze-256-bit-os-entropy"
    assert evaluation.task_count == 30
    assert sum(evaluation.family_quotas.values()) == 30
    assert set(protocol.partitions).issuperset({"train", "dev", "eval"})
