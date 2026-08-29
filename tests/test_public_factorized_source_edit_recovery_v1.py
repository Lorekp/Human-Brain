from __future__ import annotations

from pathlib import Path

from experiments.r3_pilot import public_factorized_source_edit_recovery_v1 as recovery
from experiments.r3_pilot import public_typed_prompt_factors as prompt_factors
from humanbrain import typed_program as tp


def test_recovery_protocol_is_fresh_public_and_mechanically_exact() -> None:
    recovery._assert_recovery_protocol()
    assert tuple(task.name for task in recovery.PUBLIC_TASKS) == (
        "four-minus-abs-datum",
        "list-map-add-four",
    )
    assert all(
        task.score_task.public.partition == "public-canary"
        for task in recovery.PUBLIC_TASKS
    )
    assert recovery._assert_v2_boundary() == {
        "verdict": "STOP_PUBLIC_FACTORIZED_SOURCE_EDIT_CANARY_V2",
        "artifact_sha256": recovery._V2_ARTIFACT_SHA256,
        "passed_tasks": 3,
        "failed_task": "list-map-subtract-two",
    }


def test_recovery_prompts_bind_distinct_generic_compositions() -> None:
    scalar, mapped = recovery.PUBLIC_TASKS
    scalar_observation = prompt_factors.observe_prompt_compositions(
        scalar.prompt, scalar.signature
    )
    map_observation = prompt_factors.observe_prompt_compositions(
        mapped.prompt, mapped.signature
    )

    assert scalar_observation.pattern is not None
    assert scalar_observation.pattern.operator == "subtract"
    assert scalar_observation.pattern.container_schema is None
    assert scalar_observation.pattern.left.source == "LITERAL"
    assert scalar_observation.pattern.right.unary_operators == ("absolute",)

    assert map_observation.pattern is not None
    assert map_observation.pattern.operator == "add"
    assert map_observation.pattern.container_schema == "MAP_BIND"
    assert map_observation.pattern.left.source == "BOUND0"
    assert map_observation.pattern.right.source == "LITERAL"


def test_recovery_cohort_is_prepared_but_unspent() -> None:
    assert not Path(recovery.__file__).with_suffix(".json").exists()
    for task in recovery.PUBLIC_TASKS:
        one_edit = tp.apply_factor_action(task.identity, task.expected_actions[0])
        assert one_edit.tokens != task.expected_tokens
