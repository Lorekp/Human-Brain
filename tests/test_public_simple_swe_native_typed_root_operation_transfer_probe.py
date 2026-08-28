from __future__ import annotations

import inspect
import json
import sys
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

# ruff: noqa: E402
import public_simple_swe_native_typed_root_operation_transfer_probe as probe
from humanbrain.typed_program import Program
from humanbrain.typed_program import compile as compile_typed_program


DONOR_TOKENS = (
    "sum",
    "map",
    "bind",
    "if",
    "less-than",
    "variable-0",
    "integer",
    "-",
    "7",
    "end",
    "integer",
    "-",
    "7",
    "end",
    "if",
    "less-than",
    "integer",
    "+",
    "7",
    "end",
    "variable-0",
    "integer",
    "+",
    "7",
    "end",
    "variable-0",
    "variable-0",
)


def _selected() -> tuple[object, tuple[object, ...], object]:
    return probe._select_tasks()


def test_fresh_visible_train_donor_and_target_are_exact_and_family_disjoint() -> None:
    _, donors, target = _selected()
    assert len(donors) == 1
    donor = donors[0]
    assert (donor.public.family, donor.public.ordinal) == probe.CLAMP_SUM_KEY
    assert (target.public.family, target.public.ordinal) == probe.TARGET_KEY
    assert donor.public.family != target.public.family
    assert probe.CLAMP_SUM_KEY not in probe.KNOWN_USED_T2_CLAMP_THEN_SUM_KEYS
    assert probe.TARGET_KEY not in probe.KNOWN_SPENT_VISIBLE_TRAIN_TARGET_KEYS

    for task in (donor, target):
        key = (task.public.family, task.public.ordinal)
        expected = probe.EXPECTED_SOURCE_RECORDS[key]
        assert task.public.partition == "train"
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


def test_root_operation_transfer_programs_and_copy_schedules_are_frozen() -> None:
    _, donors, target = _selected()
    donor = donors[0]

    assert probe._expected_program_tokens(donor.public) == DONOR_TOKENS
    assert probe._expected_program_tokens(target.public) == probe.EXPECTED_TARGET_TOKENS
    assert probe._expected_program_tokens(donor.public) != probe.EXPECTED_TARGET_TOKENS
    assert compile_typed_program(
        Program(DONOR_TOKENS, donor.public.signature)
    ).source == probe._expected_compiled_source(donor.public)
    assert compile_typed_program(
        Program(probe.EXPECTED_TARGET_TOKENS, target.public.signature)
    ).source == probe.EXPECTED_TARGET_SOURCE

    assert probe._expected_copy_schedule(donor.public) == (
        (2, ("-", "7")),
        (2, ("-", "7")),
        (3, ("+", "7")),
        (3, ("+", "7")),
    )
    assert probe._expected_copy_schedule(target.public) == (
        (2, ("-", "2")),
        (2, ("-", "2")),
        (3, ("+", "2")),
        (3, ("+", "2")),
    )


def test_nested_if_operation_transfer_source_and_capacity_are_frozen() -> None:
    _, donors, target = _selected()
    donor = donors[0]
    teachers = {probe.CLAMP_SUM_KEY: probe._internal_teacher_tokens(probe.CLAMP_SUM_KEY)}
    prompts = {
        probe.CLAMP_SUM_KEY: probe._tokenized_public_prompt(donor.public),
    }
    accounting = probe._learning_arrival_accounting(prompts, teachers)

    assert accounting["episodes"]["t2_clamp_then_sum/004"][
        "physical_teacher_arrivals"
    ] == 48
    assert accounting["base_learning_arrivals"] == 338
    assert accounting["constructor_pointer_cue_arrivals"] == 11
    assert accounting["total_learning_arrivals"] == 349
    assert accounting["noninitial_learning_arrivals"] == 348
    assert accounting["base_temporal_segment_rows_upper_bound"] == 3_707
    assert accounting["ordinary_segment_rows_upper_bound_including_pointer_cues"] == 3_828
    assert accounting["segment_capacity"] == 3_949
    assert accounting["head_to_root_transfer"] == {
        "projection": "HeadToRootTransfer",
        "teacher_position": 3,
        "learning_arrivals": 1,
        "ordinary_segment_rows_upper_bound": 11,
        "direction_isolated": True,
    }
    occurrences = accounting["episodes"]["t2_clamp_then_sum/004"][
        "constructor_pointer_occurrences"
    ]
    assert len(occurrences) == 11
    assert occurrences[0]["surface"] == "sum"
    assert occurrences[3] == {
        "occurrence": 3,
        "teacher_position": 3,
        "surface": "if",
        "surface_feedback_source": "t2_clamp_then_sum/004@3",
        "semantic_cue": "expression-int-head",
        "absolute_address": [0, 0, 0],
        "absolute_address_offset": 13,
        "relative_address": [],
        "relative_address_offset": 0,
        "parent_occurrence": None,
        "parent_pointer_source": None,
        "maximum_persistent_rows": 22,
    }
    assert [
        item["teacher_position"]
        for item in occurrences
        if item["surface"] == "integer"
    ] == [6, 9, 14, 18]

    definition, definition_accounting = probe._definition(
        donors, probe._target_runtime_input(target.public)
    )
    assert definition.config.segment_capacity == 3_949
    assert definition_accounting == accounting


def test_head_to_root_row_delta_is_removed_before_pointer_split() -> None:
    assert probe._infer_pointer_rows_from_teacher_delta(
        learning_step_delta=4,
        segment_row_delta=45,
        returned_exact_rows=7,
    ) == (5, 22)
    assert probe._infer_pointer_rows_from_teacher_delta(
        learning_step_delta=4,
        segment_row_delta=23,
        returned_exact_rows=7,
    ) == (5, 0)


def test_native_pins_and_spent_result_are_frozen() -> None:
    assert set(probe.EXPECTED_NATIVE_MECHANISM_SHA256) == {
        path.relative_to(probe.REPO_ROOT).as_posix()
        for path in probe.NATIVE_MECHANISM_PATHS
    }
    for path in probe.NATIVE_MECHANISM_PATHS:
        relative = path.relative_to(probe.REPO_ROOT).as_posix()
        assert (
            probe._sha256_file(path)
            == probe.EXPECTED_NATIVE_MECHANISM_SHA256[relative]
        )
    assert probe._sha256_file(probe.OUTPUT_PATH) == (
        "feb547bf49b7e4ca200bc1bdc93477fc01d2088cdf33dffcd47def8f8f461283"
    )
    assert probe._sha256_file(probe.SPEND_MARKER_PATH) == (
        "7b8b58cbdebe887801b6f5c3b351b1cfc76aa88ee9991b59ea491e40c690fa77"
    )
    payload = json.loads(probe.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert payload["outcome"] == "STOP_NATIVE_TYPED_ROOT_OPERATION_TRANSFER"
    assert payload["first_failure"]["detail"]["completed_surface_prefix"] == 23
    target = payload["visible_train_target_teacher_free"]
    assert target["generations"] == []
    assert target["program"]["exact_tokens"] is False
    assert target["sandbox"]["attempted"] is False
    assert payload["authority"]["teacher_free_target_attempts_completed"] == 1


def test_marker_immediately_precedes_one_no_argument_generation_call() -> None:
    source = inspect.getsource(probe.main)
    generation = "raw_generations = circuit.generate_typed()"
    assert source.count(".generate_typed(") == 1
    assert source.count(generation) == 1
    assert (
        "spend_marker_record = _create_spend_marker()\n"
        f"                {generation}"
    ) in source


def test_spend_marker_is_durable_exclusive_and_cleanup_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_artifact = probe.OUTPUT_PATH
    canonical_marker = probe.SPEND_MARKER_PATH
    test_dir = (
        probe.REPO_ROOT
        / ".pytest_tmp"
        / f"root_operation_transfer_spend_marker_{probe.os.getpid()}"
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
        assert marker_payload["experiment"] == (
            "r3-public-simple-swe-native-typed-root-operation-transfer"
        )
        assert record["state"] == "TARGET_ATTEMPT_SPENT"
        with pytest.raises(RuntimeError, match="spend marker exists"):
            probe._create_spend_marker()
    finally:
        marker.unlink(missing_ok=True)
        artifact.unlink(missing_ok=True)
        test_dir.rmdir()
    assert not test_dir.exists()
    assert canonical_artifact.exists()
    assert canonical_marker.exists()
