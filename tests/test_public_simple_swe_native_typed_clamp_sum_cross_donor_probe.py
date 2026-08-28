from __future__ import annotations

import json
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

# The focused test imports the directly executable probe after installing roots.
# ruff: noqa: E402
import public_simple_swe_native_typed_clamp_sum_cross_donor_probe as probe
from humanbrain.native_temporal_runtime import (
    NativeTemporalRuntimeError,
    NativeTemporalRuntimeStatus,
)
from humanbrain.typed_program import Program
from humanbrain.typed_program import compile as compile_typed_program

SCALAR_CLAMP_TOKENS = (
    "if",
    "less-than",
    "variable-0",
    "integer",
    "-",
    "8",
    "end",
    "integer",
    "-",
    "8",
    "end",
    "if",
    "less-than",
    "integer",
    "+",
    "8",
    "end",
    "variable-0",
    "integer",
    "+",
    "8",
    "end",
    "variable-0",
)
MAP_COUNT_TOKENS = (
    "sum",
    "map",
    "bind",
    "if",
    "less-than",
    "integer",
    "+",
    "0",
    "end",
    "add",
    "variable-0",
    "integer",
    "-",
    "2",
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


def _selected() -> tuple[object, tuple[object, ...], object]:
    return probe._select_tasks()


def test_visible_train_tasks_prompts_and_expected_programs_are_exact() -> None:
    _, donors, target = _selected()
    scalar_clamp, map_count = donors

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
        prompt_tokens = probe._tokenized_public_prompt(task.public)
        assert len(prompt_tokens) == probe.EXPECTED_PROMPT_TOKEN_COUNTS[key]
        assert probe._detokenize_prompt(prompt_tokens) == task.public.prompt.encode(
            "ascii"
        )
        assert len(task.private.regression_cases) == 4
        assert len(task.private.property_cases) == 32

    assert probe._expected_program_tokens(scalar_clamp.public) == SCALAR_CLAMP_TOKENS
    assert probe._expected_program_tokens(map_count.public) == MAP_COUNT_TOKENS
    assert probe._expected_program_tokens(target.public) == probe.EXPECTED_TARGET_TOKENS
    assert all(
        probe._expected_program_tokens(task.public) != probe.EXPECTED_TARGET_TOKENS
        for task in donors
    )
    assert compile_typed_program(
        Program(SCALAR_CLAMP_TOKENS, scalar_clamp.public.signature)
    ).source == probe._expected_compiled_source(scalar_clamp.public)
    assert compile_typed_program(
        Program(MAP_COUNT_TOKENS, map_count.public.signature)
    ).source == probe._expected_compiled_source(map_count.public)
    assert compile_typed_program(
        Program(probe.EXPECTED_TARGET_TOKENS, target.public.signature)
    ).source == probe.EXPECTED_TARGET_SOURCE


def test_copy_schedules_and_pointer_capacity_are_frozen_without_a_circuit() -> None:
    _, donors, target = _selected()
    scalar_clamp, map_count = donors
    teachers = {key: probe._internal_teacher_tokens(key) for key in probe.TRAIN_KEYS}
    prompts = {
        key: probe._tokenized_public_prompt(task.public)
        for key, task in zip(probe.TRAIN_KEYS, donors, strict=True)
    }

    assert probe._expected_copy_schedule(scalar_clamp.public) == (
        (2, ("-", "8")),
        (2, ("-", "8")),
        (3, ("+", "8")),
        (3, ("+", "8")),
    )
    assert probe._expected_copy_schedule(map_count.public) == (
        (3, ("+", "0")),
        (2, ("-", "2")),
    )
    assert probe._expected_copy_schedule(target.public) == (
        (2, ("-", "3")),
        (2, ("-", "3")),
        (3, ("+", "3")),
        (3, ("+", "3")),
    )

    accounting = probe._learning_arrival_accounting(prompts, teachers)
    assert accounting["base_learning_arrivals"] == 659
    assert accounting["constructor_pointer_cue_arrivals"] == 10
    assert accounting["total_learning_arrivals"] == 669
    assert accounting["noninitial_learning_arrivals"] == 667
    assert accounting["base_temporal_segment_rows_upper_bound"] == 7_227
    assert (
        accounting["ordinary_segment_rows_upper_bound_including_pointer_cues"]
        == 7_337
    )
    assert accounting["constructor_pointer"] == {
        "role_bearing_constructors": [
            "add",
            "absolute",
            "less-than",
            "if",
            "map",
            "sum",
            "bind",
        ],
        "identity": (
            "opaque surface winners + semantic cue/role + parent-relative "
            "address + parent pointer when active"
        ),
        "exact_role_opening_call_count": 10,
        "rows_per_cue_call_upper_bound": 22,
        "cue_rows_per_novel_call": 11,
        "recurrence_rows_per_novel_call": 11,
        "exact_replay_rows": 0,
        "cue_rows_upper_bound": 110,
        "recurrence_rows_upper_bound": 110,
        "total_rows_upper_bound": 220,
        "frozen_occurrence_labels_are_unique": True,
        "physical_pointer_identity_uniqueness_claimed": False,
        "capacity_assumes_every_call_recruits": True,
        "root_election_is_upstream": True,
    }
    assert accounting["segment_capacity"] == 7_447
    assert accounting["episodes"]["t0_scalar_clamp/001"][
        "constructor_pointer_row_capacity_upper_bound"
    ] == 88
    assert accounting["episodes"]["t2_map_then_count/002"][
        "constructor_pointer_row_capacity_upper_bound"
    ] == 132
    scalar_occurrences = accounting["episodes"]["t0_scalar_clamp/001"][
        "constructor_pointer_occurrences"
    ]
    map_occurrences = accounting["episodes"]["t2_map_then_count/002"][
        "constructor_pointer_occurrences"
    ]
    assert len(scalar_occurrences) == 4
    assert len(map_occurrences) == 6
    assert scalar_occurrences[2] == {
        "occurrence": 2,
        "teacher_position": 9,
        "surface": "if",
        "surface_feedback_source": "t0_scalar_clamp/001@9",
        "semantic_cue": "expression-int-child-2",
        "absolute_address": [2],
        "absolute_address_offset": 3,
        "relative_address": [2],
        "relative_address_offset": 3,
        "parent_occurrence": 0,
        "parent_pointer_source": "t0_scalar_clamp/001#0",
        "maximum_persistent_rows": 22,
    }
    assert map_occurrences[5]["absolute_address_offset"] == 122
    assert map_occurrences[5]["relative_address_offset"] == 2
    assert map_occurrences[5]["parent_occurrence"] == 4
    assert probe.COLUMN_COUNT == 768
    assert probe._RESERVED_COLUMN_COUNT == 561
    assert len(probe.ADDRESS_OFFSETS) == 24

    assert probe._infer_pointer_rows_from_teacher_delta(
        learning_step_delta=3,
        segment_row_delta=33,
        returned_exact_rows=11,
    ) == (0, 22)
    assert probe._infer_pointer_rows_from_teacher_delta(
        learning_step_delta=3,
        segment_row_delta=22,
        returned_exact_rows=11,
    ) == (11, 0)
    assert probe._infer_pointer_rows_from_teacher_delta(
        learning_step_delta=3,
        segment_row_delta=33,
        returned_exact_rows=4,
    ) == (7, 22)


def test_target_runtime_input_and_failure_localization_exclude_solution_data() -> None:
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

    root = NativeTemporalRuntimeError(
        NativeTemporalRuntimeStatus.SCIENTIFIC_STOP,
        "root stopped",
        completed_prefix=0,
        poison_code=7,
        stage_code=3,
    )
    descendant = NativeTemporalRuntimeError(
        NativeTemporalRuntimeStatus.SCIENTIFIC_STOP,
        "descendant stopped",
        completed_prefix=3,
        poison_code=7,
        stage_code=3,
    )
    root_record = probe._generation_failure(root, probe.EXPECTED_TARGET_TOKENS)
    descendant_record = probe._generation_failure(
        descendant, probe.EXPECTED_TARGET_TOKENS
    )
    assert root_record["localization"] == {
        "gate": "GENERATION_STOP",
        "scope": "root",
        "generation_position": 0,
        "expected_surface": "sum",
        "root_expected_surface": "sum",
        "root_election_is_upstream_of_constructor_pointer": True,
    }
    assert descendant_record["localization"]["scope"] == "descendant"
    assert descendant_record["localization"]["generation_position"] == 3
    assert descendant_record["localization"]["expected_surface"] == "if"


def test_terminal_native_stop_blocks_compile_sandbox_and_source_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _, target = _selected()
    native_failure = probe._generation_failure(
        NativeTemporalRuntimeError(
            NativeTemporalRuntimeStatus.SCIENTIFIC_STOP,
            "root stopped",
            completed_prefix=0,
        ),
        probe.EXPECTED_TARGET_TOKENS,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("terminal native failure must block compiler and sandbox")

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
    assert record["first_failure_localization"]["scope"] == "root"
    assert record["sandbox"]["attempted"] is False

    class DirtyStatus:
        stdout = (
            "?? experiments/r3_pilot/"
            "public_simple_swe_native_typed_clamp_sum_cross_donor_probe.py\n"
        )

    expected_native = {
        path.relative_to(probe.REPO_ROOT).as_posix(): "pinned"
        for path in probe.NATIVE_MECHANISM_PATHS
    }
    monkeypatch.setattr(probe, "EXPECTED_NATIVE_MECHANISM_SHA256", expected_native)

    def pinned_sha(path: Path) -> str:
        if path == probe.HELPER_PATH:
            return probe.EXPECTED_HELPER_SHA256
        return "pinned"

    monkeypatch.setattr(probe, "_sha256_file", pinned_sha)
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: DirtyStatus())
    with pytest.raises(RuntimeError, match="committed and unmodified"):
        probe._assert_source_committed_and_output_absent()


def test_spend_marker_is_durable_exclusive_and_checked_without_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_dir = (
        probe.REPO_ROOT
        / ".pytest_tmp"
        / f"cross_donor_spend_marker_{probe.os.getpid()}"
    )
    test_dir.mkdir(parents=True, exist_ok=False)
    artifact = test_dir / "probe.json"
    marker = test_dir / "probe.spent.json"
    try:
        monkeypatch.setattr(probe, "OUTPUT_PATH", artifact)
        monkeypatch.setattr(probe, "SPEND_MARKER_PATH", marker)
        fsync_calls: list[int] = []
        real_fsync = probe.os.fsync

        def recording_fsync(file_descriptor: int) -> None:
            fsync_calls.append(file_descriptor)
            real_fsync(file_descriptor)

        monkeypatch.setattr(probe.os, "fsync", recording_fsync)
        record = probe._create_spend_marker()
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        assert fsync_calls
        assert marker_payload["state"] == "TARGET_ATTEMPT_SPENT"
        assert record == {
            "path": marker.relative_to(probe.REPO_ROOT).as_posix(),
            "sha256": probe._sha256_file(marker),
            "state": "TARGET_ATTEMPT_SPENT",
            "exclusive_create": True,
            "flushed_and_fsynced_before_generate": True,
        }
        with pytest.raises(RuntimeError, match="spend marker exists"):
            probe._create_spend_marker()

        marker.unlink()
        artifact.write_text("{}\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="one-shot artifact exists"):
            probe._create_spend_marker()
        assert not marker.exists()
        artifact.unlink()
        assert not artifact.exists()
    finally:
        marker.unlink(missing_ok=True)
        artifact.unlink(missing_ok=True)
        test_dir.rmdir()
    assert not test_dir.exists()
