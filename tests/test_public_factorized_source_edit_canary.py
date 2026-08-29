from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import pytest
from experiments.r3_pilot import public_factorized_source_edit_canary as canary
from experiments.r3_pilot import public_typed_prompt_factors as prompt_factors
from humanbrain import typed_program as tp


def _desired(task: canary.PublicSourceEditCanaryTask) -> tuple[object, ...]:
    return tuple(canary._snapshot(case.expected) for case in task.visible_cases)


def test_public_tasks_are_disjoint_and_actions_compile_execute_exactly() -> None:
    aliases = canary._assert_public_protocol()
    assert aliases == {
        "abs-value-minus-three": True,
        "three-minus-abs-value": True,
        "list-map-absolute": False,
        "list-map-subtract-two": False,
    }

    for task in canary.PUBLIC_TASKS:
        final = canary._apply_actions(task.identity, task.expected_actions)
        assert final.tokens == task.expected_tokens
        assert tp.compile(final).source == task.expected_source
        assert canary._visible_outputs(final, task.visible_cases) == _desired(task)

        visible = {json.dumps(case.arguments) for case in task.visible_cases}
        scoring = {
            json.dumps(case.args)
            for case in (
                *task.score_task.private.regression_cases,
                *task.score_task.private.property_cases,
            )
        }
        assert visible.isdisjoint(scoring)
        assert task.score_task.public.partition == "public-canary"


def test_identity_one_edit_and_every_h1_action_are_insufficient() -> None:
    for task in canary.PUBLIC_TASKS:
        assert canary._visible_outputs(task.identity, task.visible_cases) != _desired(task)
        first = canary._apply_actions(task.identity, task.expected_actions[:1])
        assert canary._visible_outputs(first, task.visible_cases) != _desired(task)

        exact = []
        for action in tp.factor_action_candidates(
            task.identity, literals=canary.PUBLIC_LITERAL_BANK
        ).actions:
            program = (
                task.identity
                if action.schema == "DONE"
                else tp.apply_factor_action(task.identity, action)
            )
            if canary._visible_outputs(program, task.visible_cases) == _desired(task):
                exact.append(action)
        assert exact == []


def test_scalar_order_alias_is_real_but_list_body_must_be_created() -> None:
    for task in canary.PUBLIC_TASKS[:2]:
        assert task.reverse_equivalent_actions is not None
        assert canary._apply_actions(task.identity, task.expected_actions) == (
            canary._apply_actions(task.identity, task.reverse_equivalent_actions)
        )

    for task in canary.PUBLIC_TASKS[2:]:
        assert task.reverse_equivalent_actions is None
        initial = tp.factor_action_candidates(
            task.identity, literals=canary.PUBLIC_LITERAL_BANK
        ).actions
        assert task.expected_actions[1] not in initial
        mapped = canary._apply_actions(task.identity, task.expected_actions[:1])
        assert task.expected_actions[1] in tp.factor_action_candidates(
            mapped, literals=canary.PUBLIC_LITERAL_BANK
        ).actions


def test_scalar_prompts_express_one_supported_nested_absolute_pattern() -> None:
    for task in canary.PUBLIC_TASKS[:2]:
        observed = prompt_factors.observe_prompt_compositions(
            task.prompt, task.signature
        )
        assert observed.supported
        assert observed.pattern is not None
        assert observed.pattern.operator == "subtract"
        assert sum(
            operand.unary_operators == ("absolute",)
            for operand in (observed.pattern.left, observed.pattern.right)
        ) == 1


def test_created_child_v3_prerequisite_requires_exact_pass_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    banked = canary._assert_created_child_prerequisite()
    assert banked == {
        "verdict": "PASS_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V3",
        "artifact_sha256": canary._REQUIRED_CREATED_CHILD_ARTIFACT_SHA256,
        "checkpoint_sha256": (
            "7820c258c0d7d243008c5ec95b7ed19ceb91b77729396877b0d474af5a272f85"
        ),
    }

    prerequisite = tmp_path / "created-child-v3.json"
    prerequisite.write_text(
        json.dumps(
            {
                "schema": "public-typed-factor-horizon2-int-created-child-v3",
                "verdict": "PASS_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V3",
                "passed_tasks": 2,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "canary.json"
    monkeypatch.setattr(canary, "_CREATED_CHILD_V3_PATH", prerequisite)
    monkeypatch.setattr(
        canary,
        "_parse_args",
        lambda: argparse.Namespace(json=output, python_executable=None),
    )
    monkeypatch.setattr(
        canary.h2,
        "build_public_factorized_horizon2_int",
        lambda **_: pytest.fail("model must not build after prerequisite STOP"),
    )

    with pytest.raises(RuntimeError, match="exact created-child V3 prerequisite"):
        canary.main()
    assert not output.exists()


def test_artifact_write_is_exclusive_and_runner_delegates_real_edits(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canary.json"
    canary.write_public_artifact({"schema": canary.SCHEMA}, output)
    with pytest.raises(RuntimeError, match="artifact already exists"):
        canary.write_public_artifact({"schema": canary.SCHEMA}, output)

    source = inspect.getsource(canary.run_public_task)
    assert "FactorizedSourceEditor" in source
    assert "apply_factor_action" not in source
    assert "generate_visible_partition" not in inspect.getsource(canary)
