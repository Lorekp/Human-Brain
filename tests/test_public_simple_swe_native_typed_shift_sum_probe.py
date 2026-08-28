from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (
    REPO_ROOT / "src",
    REPO_ROOT,
    REPO_ROOT / "experiments" / "r3_pilot",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# The focused test imports the directly executable probe after installing its roots.
# ruff: noqa: E402
import public_simple_swe_native_typed_shift_sum_probe as probe
from humanbrain.typed_program import Program
from humanbrain.typed_program import compile as compile_typed_program

LIST_ADD_TOKENS = (
    "map",
    "bind",
    "add",
    "variable-0",
    "integer",
    "+",
    "5",
    "end",
    "variable-0",
)
CLAMP_SUM_TOKENS = (
    "sum",
    "map",
    "bind",
    "if",
    "less-than",
    "variable-0",
    "integer",
    "-",
    "4",
    "end",
    "integer",
    "-",
    "4",
    "end",
    "if",
    "less-than",
    "integer",
    "+",
    "4",
    "end",
    "variable-0",
    "integer",
    "+",
    "4",
    "end",
    "variable-0",
    "variable-0",
)
TARGET_TOKENS = (
    "sum",
    "map",
    "bind",
    "add",
    "variable-0",
    "integer",
    "+",
    "6",
    "end",
    "variable-0",
)


def _selected() -> tuple[object, tuple[object, ...], object]:
    return probe._select_tasks()


def test_frozen_donors_and_self_posed_target_are_exact() -> None:
    _, donors, target = _selected()
    list_add, clamp_sum = donors

    assert tuple(
        (task.public.family, task.public.ordinal) for task in donors
    ) == probe.TRAIN_KEYS
    assert (target.public.family, target.public.ordinal) == probe.TARGET_KEY
    for task in (*donors, target):
        key = (task.public.family, task.public.ordinal)
        expected = probe.EXPECTED_SOURCE_RECORDS[key]
        assert task.public.task_id == expected["task_id"]
        assert task.public.task_sha256 == expected["task_sha256"]
        assert task.public.hidden_test_sha256 == expected["hidden_test_sha256"]
        assert len(probe._tokenized_public_prompt(task.public)) == (
            probe.EXPECTED_PROMPT_TOKEN_COUNTS[key]
        )
        assert len(task.private.regression_cases) == 4
        assert len(task.private.property_cases) == 32

    assert probe._expected_program_tokens(list_add.public) == LIST_ADD_TOKENS
    assert probe._expected_program_tokens(clamp_sum.public) == CLAMP_SUM_TOKENS
    assert probe._expected_program_tokens(target.public) == TARGET_TOKENS
    assert TARGET_TOKENS == probe.EXPECTED_TARGET_TOKENS
    assert all(
        probe._expected_program_tokens(task.public) != TARGET_TOKENS for task in donors
    )
    assert probe._task_fields(list_add.public).literal == b"5"
    assert probe._task_fields(clamp_sum.public).lower == b"-4"
    assert probe._task_fields(clamp_sum.public).upper == b"4"
    assert probe._task_fields(target.public).literal == b"6"

    assert compile_typed_program(
        Program(LIST_ADD_TOKENS, list_add.public.signature)
    ).source == (
        "def solve_fa9aceee8d7adbba(values):\n"
        "    return [item_0 + 5 for item_0 in values]\n"
    )
    assert compile_typed_program(
        Program(CLAMP_SUM_TOKENS, clamp_sum.public.signature)
    ).source == (
        "def solve_7994ea445e2aa4d7(values):\n"
        "    return sum([-4 if item_0 < -4 else 4 if 4 < item_0 else "
        "item_0 for item_0 in values])\n"
    )
    assert compile_typed_program(
        Program(TARGET_TOKENS, target.public.signature)
    ).source == probe.EXPECTED_TARGET_SOURCE
    assert target.private.regression_cases[0].args == ([],)
    assert target.private.regression_cases[0].expected == 0
    assert target.private.property_cases[0].args == ([3, 35, 78, -64, -66, 59],)
    assert target.private.property_cases[0].expected == 81


def test_teacher_copy_schedule_and_capacity_are_frozen_without_native_execution() -> None:
    _, donors, target = _selected()
    list_add, clamp_sum = donors
    teachers = {key: probe._internal_teacher_tokens(key) for key in probe.TRAIN_KEYS}

    assert tuple(
        probe._teacher_label(token_id) for token_id in teachers[probe.LIST_ADD_KEY]
    ) == (
        "map",
        "bind",
        "add",
        "variable-0",
        "integer",
        "COPY_SLOT_2",
        "end",
        "variable-0",
        "<EOS>",
    )
    assert probe._expected_copy_schedule(list_add.public) == ((2, ("+", "5")),)
    assert probe._expected_copy_schedule(clamp_sum.public) == (
        (2, ("-", "4")),
        (2, ("-", "4")),
        (3, ("+", "4")),
        (3, ("+", "4")),
    )
    assert probe._expected_copy_schedule(target.public) == ((2, ("+", "6")),)

    prompts = {
        key: probe._tokenized_public_prompt(task.public)
        for key, task in zip(probe.TRAIN_KEYS, donors, strict=True)
    }
    accounting = probe._learning_arrival_accounting(prompts, teachers)
    assert accounting["total_learning_arrivals"] == 640
    assert accounting["noninitial_learning_arrivals"] == 638
    assert accounting["segment_capacity"] == 7018
    assert accounting["episodes"]["t1_list_map_add/004"]["physical_teacher_arrivals"] == 17
    assert accounting["episodes"]["t2_clamp_then_sum/002"]["physical_teacher_arrivals"] == 47

    definition, definition_accounting = probe._definition(
        donors, probe._target_runtime_input(target.public)
    )
    assert definition.config.segment_capacity == 7018
    assert definition_accounting == accounting
    assert probe.ADDRESS_OFFSETS == (
        0,
        1,
        2,
        4,
        5,
        13,
        14,
        40,
        41,
        42,
        121,
        122,
        127,
        128,
        129,
        382,
        383,
    )


def test_target_runtime_input_excludes_solution_and_private_cases() -> None:
    _, _, target = _selected()
    runtime_input = probe._target_runtime_input(target.public)

    assert tuple(field.name for field in fields(runtime_input)) == (
        "prompt_tokens",
        "parameter_type",
        "result_type",
    )
    assert probe._detokenize_prompt(runtime_input.prompt_tokens) == (
        target.public.prompt.encode("ascii")
    )


def test_native_stop_is_terminal_before_compile_or_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _, target = _selected()
    native_failure = {
        "kind": "model",
        "gate": "GENERATION_STOP",
        "detail": {"status": "SCIENTIFIC_STOP"},
    }

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("terminal native failure must block compile and sandbox")

    monkeypatch.setattr(probe, "compile_typed_program", forbidden)
    monkeypatch.setattr(probe, "execute_completion", forbidden)
    record, first_failure = probe._score_target(
        episode=probe._WorkspaceEpisode(prompt_steps=(), workspace_steps=(object(),)),
        raw_generations=None,
        native_failure=native_failure,
        task=target,
        protocol=protocol,
        python_executable=Path("unused-python"),
        runtime_identity={"executable_sha256": "unused"},
        worker_source=b"",
        worker_sha256="unused",
    )

    assert first_failure is native_failure
    assert record["first_failure"] is native_failure
    assert record["generations"] == []
    assert record["fitted_cell_races_exact"] is False
    assert record["sandbox"] == {
        "attempted": False,
        "passed": False,
        "detail": "not run because an earlier gate stopped",
    }


def test_uncommitted_probe_source_cannot_spend_the_one_shot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DirtyStatus:
        stdout = (
            "?? experiments/r3_pilot/"
            "public_simple_swe_native_typed_shift_sum_probe.py\n"
        )

    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: DirtyStatus())
    with pytest.raises(RuntimeError, match="committed and unmodified"):
        probe._assert_source_committed_and_output_absent(tmp_path / "result.json")
