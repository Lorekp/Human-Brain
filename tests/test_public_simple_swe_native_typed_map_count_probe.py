from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (
    REPO_ROOT / "src",
    REPO_ROOT,
    REPO_ROOT / "experiments" / "r3_pilot",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_native_typed_map_count_probe as probe
from humanbrain.typed_program import Program, compile as compile_typed_program


TRAIN_TOKENS = (
    "sum",
    "map",
    "bind",
    "if",
    "less-than",
    "integer",
    "+",
    "2",
    "end",
    "add",
    "variable-0",
    "integer",
    "-",
    "4",
    "end",
    "integer",
    "+",
    "1",
    "end",
    "integer",
    "+",
    "0",
    "end",
    "variable-0",
)
FOLLOW_ON_TOKENS = (
    "sum",
    "map",
    "bind",
    "if",
    "less-than",
    "integer",
    "+",
    "2",
    "end",
    "add",
    "variable-0",
    "integer",
    "+",
    "4",
    "end",
    "integer",
    "+",
    "1",
    "end",
    "integer",
    "+",
    "0",
    "end",
    "variable-0",
)


def _selected() -> tuple[object, object, object]:
    return probe._select_tasks()


def test_frozen_map_count_tasks_prompts_and_programs_are_exact() -> None:
    _, train, follow_on = _selected()

    assert train.public.task_id == probe.EXPECTED_TASK_IDS[probe.TRAIN_KEY]
    assert follow_on.public.task_id == probe.EXPECTED_TASK_IDS[probe.FOLLOW_ON_KEY]
    assert probe._task_fields(train.public).delta == b"-4"
    assert probe._task_fields(train.public).threshold == b"2"
    assert probe._task_fields(follow_on.public).delta == b"4"
    assert probe._task_fields(follow_on.public).threshold == b"2"
    assert len(probe._tokenized_public_prompt(train.public)) == 290
    assert len(probe._tokenized_public_prompt(follow_on.public)) == 290
    assert probe._expected_program_tokens(train.public) == TRAIN_TOKENS
    assert probe._expected_program_tokens(follow_on.public) == FOLLOW_ON_TOKENS

    assert compile_typed_program(Program(TRAIN_TOKENS, train.public.signature)).source == (
        "def solve_419286ab7e1266d1(values):\n"
        "    return sum([1 if 2 < item_0 + -4 else 0 for item_0 in values])\n"
    )
    assert compile_typed_program(
        Program(FOLLOW_ON_TOKENS, follow_on.public.signature)
    ).source == (
        "def solve_a9c77fb740b2d54f(values):\n"
        "    return sum([1 if 2 < item_0 + 4 else 0 for item_0 in values])\n"
    )
    assert len(train.private.regression_cases) == 4
    assert len(train.private.property_cases) == 32
    assert len(follow_on.private.regression_cases) == 4
    assert len(follow_on.private.property_cases) == 32


def test_teacher_copy_schedule_and_capacity_are_frozen_without_native_execution() -> None:
    _, train, follow_on = _selected()
    teacher = probe._internal_teacher_tokens()

    assert tuple(probe._teacher_label(token_id) for token_id in teacher) == (
        "sum",
        "map",
        "bind",
        "if",
        "less-than",
        "integer",
        "COPY_SLOT_3",
        "end",
        "add",
        "variable-0",
        "integer",
        "COPY_SLOT_2",
        "end",
        "integer",
        "+",
        "1",
        "end",
        "integer",
        "+",
        "0",
        "end",
        "variable-0",
        "<EOS>",
    )
    assert probe._expected_copy_schedule(train.public) == (
        (3, ("+", "2")),
        (2, ("-", "4")),
    )
    assert probe._expected_copy_schedule(follow_on.public) == (
        (3, ("+", "2")),
        (2, ("+", "4")),
    )
    accounting = probe._learning_arrival_accounting(
        probe._tokenized_public_prompt(train.public), teacher
    )
    assert accounting["total_learning_arrivals"] == 341
    assert accounting["physical_teacher_arrivals"] == 45
    assert accounting["segment_capacity"] == 3740
    definition, definition_accounting = probe._definition(
        train, probe._follow_on_runtime_input(follow_on.public)
    )
    assert definition.config.segment_capacity == 3740
    assert definition_accounting == accounting
    assert probe.ADDRESS_OFFSETS == (
        0,
        1,
        2,
        4,
        5,
        13,
        40,
        41,
        42,
        121,
        122,
        367,
        368,
    )


def test_follow_on_runtime_input_excludes_target_and_private_cases() -> None:
    _, _, follow_on = _selected()
    runtime_input = probe._follow_on_runtime_input(follow_on.public)

    assert tuple(field.name for field in fields(runtime_input)) == (
        "prompt_tokens",
        "parameter_type",
        "result_type",
    )
    assert probe._detokenize_prompt(runtime_input.prompt_tokens) == (
        follow_on.public.prompt.encode("ascii")
    )


def test_native_stop_is_terminal_before_compile_or_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, _, follow_on = _selected()
    native_failure = {
        "kind": "model",
        "gate": "GENERATION_STOP",
        "detail": {"status": "SCIENTIFIC_STOP"},
    }

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("terminal native failure must block compile and sandbox")

    monkeypatch.setattr(probe, "compile_typed_program", forbidden)
    monkeypatch.setattr(probe, "execute_completion", forbidden)
    record, first_failure = probe._score_follow_on(
        episode=probe._WorkspaceEpisode(prompt_steps=(), workspace_steps=(object(),)),
        raw_generations=None,
        native_failure=native_failure,
        task=follow_on,
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
        stdout = "?? experiments/r3_pilot/public_simple_swe_native_typed_map_count_probe.py\n"

    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: DirtyStatus())
    with pytest.raises(RuntimeError, match="committed and unmodified"):
        probe._assert_source_committed_and_output_absent(tmp_path / "result.json")
