from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import _generate_partition, generate_visible_partition

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
        assert task.public.family.removeprefix(f"t{task.public.tier}_") not in (
            task.public.entry_point
        )


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


def test_cell_prompt_ends_in_task_specific_information() -> None:
    protocol = load_protocol(PROTOCOL)
    tasks = generate_visible_partition(protocol, "dev")
    first_per_family = {}
    for task in tasks:
        first_per_family.setdefault(task.public.family, task)
    tails = {
        task.public.prompt.encode("utf-8")[-64:]
        for task in first_per_family.values()
    }

    assert len(tails) == len(first_per_family)
    assert all(b"no Markdown" not in tail for tail in tails)


def test_entry_points_depend_on_the_unpredictable_partition_seed() -> None:
    protocol = load_protocol(PROTOCOL)
    first_seed = int.from_bytes(sha256(b"first hidden eval seed").digest(), "big")
    second_seed = int.from_bytes(sha256(b"second hidden eval seed").digest(), "big")

    first = _generate_partition(protocol, "eval", first_seed)
    second = _generate_partition(protocol, "eval", second_seed)

    assert [task.public.task_id for task in first] == [
        task.public.task_id for task in second
    ]
    assert all(
        left.public.entry_point != right.public.entry_point
        for left, right in zip(first, second, strict=True)
    )
    assert all(
        task.public.entry_point
        != f"solve_{sha256(task.public.task_id.encode('utf-8')).hexdigest()[:12]}"
        for task in first
    )


@pytest.mark.parametrize(
    ("path", "weakened"),
    [
        (("metric", "samples_per_task"), 2),
        (("metric", "decoding"), "external-model"),
        (("metric", "repairs"), 1),
        (("metric", "test_feedback"), True),
        (("prompt_contract", "response"), "markdown"),
        (("prompt_contract", "imports"), True),
        (("prompt_contract", "examples_in_prompt"), True),
        (("hidden_tests", "storage"), "source-tree"),
        (("freeze", "one_checkpoint_for_all_tasks"), False),
        (("freeze", "reload_checkpoint_before_each_task"), False),
        (("freeze", "verify_checkpoint_state_before_and_after"), False),
        (("freeze", "learning_during_dev_or_eval"), True),
        (("freeze", "task_specific_replay"), True),
        (("freeze", "require_clean_source_tree"), False),
        (("freeze", "eval_seed_created_after_freeze"), False),
        (("freeze", "sandbox_runtime_frozen_before_eval_seed"), False),
        (("freeze", "compiler_registry_frozen"), False),
        (("freeze", "canonical_runner_only"), False),
        (("freeze", "eval_seed_reserved_before_task_generation"), False),
        (("freeze", "external_anchor_before_eval_task_generation"), False),
        (("reporting", "baselines"), "available-during-cell-inference"),
    ],
)
def test_protocol_rejects_weakened_anti_leakage_invariants(
    tmp_path: Path, path: tuple[str, str], weakened: object
) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = weakened
    candidate = tmp_path / "weakened.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invariant"):
        load_protocol(candidate)


def test_protocol_rejects_incomplete_failure_taxonomy(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["reporting"]["failure_taxonomy"].remove("property_failure")
    candidate = tmp_path / "weakened.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="failure taxonomy"):
        load_protocol(candidate)
