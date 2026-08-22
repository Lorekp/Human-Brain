from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest
from humanbrain.simple_swe import tasks as task_module
from humanbrain.simple_swe import family_callable_types
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import (
    CallableType,
    _generate_partition,
    canonical_json_bytes,
    generate_visible_partition,
)
from humanbrain.typed_program import ProgramParameter, ProgramType

PROTOCOL = (
    Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"
)


def _unary_callable(
    parameter_name: str, parameter_type: ProgramType, result: ProgramType
) -> CallableType:
    return CallableType(
        parameters=(ProgramParameter(parameter_name, parameter_type),),
        result=result,
    )


def test_every_family_exposes_one_explicit_callable_type() -> None:
    integer = ProgramType.Int()
    string = ProgramType.String()
    integers = ProgramType.List(integer)
    strings = ProgramType.List(string)
    expected = {
        "t0_scalar_clamp": _unary_callable("n", integer, integer),
        "t0_scalar_offset": _unary_callable("n", integer, integer),
        "t0_scalar_nonnegative": _unary_callable("n", integer, integer),
        "t1_list_map_add": _unary_callable("values", integers, integers),
        "t1_list_map_abs": _unary_callable("values", integers, integers),
        "t1_string_filter_class": _unary_callable("text", string, string),
        "t2_filter_then_reverse": _unary_callable("text", string, string),
        "t2_clamp_then_sum": _unary_callable("values", integers, integer),
        "t2_map_then_count": _unary_callable("values", integers, integer),
        "t0_digit_sum": _unary_callable("n", integer, integer),
        "t0_stable_parity_select": _unary_callable("values", integers, integers),
        "t0_longest_character_run": _unary_callable("text", string, integer),
        "t1_nonzero_digit_product": _unary_callable("n", integer, integer),
        "t1_stable_range_select": _unary_callable("values", integers, integers),
        "t1_longest_target_run": _unary_callable("text", string, integer),
        "t2_digit_reduce_mod": _unary_callable("n", integer, integer),
        "t2_select_then_span": _unary_callable("values", integers, integer),
        "t2_run_length_signature": _unary_callable("text", string, integers),
        "t0_bounded_scalar_classifier": _unary_callable("n", integer, integer),
        "t0_adjacent_equal_count": _unary_callable("values", integers, integer),
        "t0_token_boundary_length": _unary_callable("text", string, integer),
        "t1_bounded_divisibility_label": _unary_callable("n", integer, string),
        "t1_adjacent_differences": _unary_callable("values", integers, integers),
        "t1_word_length_filter": _unary_callable("text", string, strings),
        "t2_clamped_window_sums": _unary_callable("values", integers, integers),
        "t2_filtered_adjacent_changes": _unary_callable("values", integers, integer),
        "t2_token_run_signature": _unary_callable("text", string, integers),
    }

    observed = family_callable_types()

    assert observed == expected
    with pytest.raises(TypeError):
        observed["unknown"] = _unary_callable("value", integer, integer)  # type: ignore[index]


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


@pytest.mark.parametrize(
    ("partition", "expected_sha256"),
    [
        ("train", "630feb71737bf67e990499dfc2aaa279f2caacc6282cf1a8ff2bf4160d73a666"),
        ("dev", "4006db794aad2de59be21d1148413bc1ddf50e2de1abc401fce6e6c3314e9235"),
    ],
)
def test_visible_partition_generation_bytes_are_frozen(
    partition: str, expected_sha256: str
) -> None:
    protocol = load_protocol(PROTOCOL)
    tasks = generate_visible_partition(protocol, partition)
    payload = canonical_json_bytes([asdict(task) for task in tasks])

    assert sha256(payload).hexdigest() == expected_sha256


def test_every_protocol_family_has_one_explicit_callable_registration() -> None:
    protocol = load_protocol(PROTOCOL)
    expected_families = {
        family
        for partition in protocol.partitions.values()
        for family in partition.family_quotas
    }
    registry = family_callable_types()

    assert set(registry) == expected_families
    assert len(registry) == 27


def test_unknown_task_family_is_rejected_by_the_registry() -> None:
    protocol = load_protocol(PROTOCOL)

    with pytest.raises(
        ValueError, match="protocol references unknown task family unknown_family"
    ):
        task_module._build_task(protocol, "train", "unknown_family", 0, 123)


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
            task.public.signature.entry_point
        )
        assert CallableType(
            task.public.signature.parameters, task.public.signature.result
        ) == family_callable_types()[task.public.family]


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
        left.public.signature.entry_point != right.public.signature.entry_point
        for left, right in zip(first, second, strict=True)
    )
    assert all(
        task.public.signature.entry_point
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
