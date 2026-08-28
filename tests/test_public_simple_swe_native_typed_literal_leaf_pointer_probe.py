from __future__ import annotations

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
import public_simple_swe_native_typed_literal_leaf_pointer_probe as probe
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


def test_literal_leaf_relocation_programs_and_copy_schedules_are_frozen() -> None:
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
        (2, ("-", "8")),
        (2, ("-", "8")),
        (3, ("+", "8")),
        (3, ("+", "8")),
    )
    assert probe._expected_copy_schedule(target.public) == (
        (2, ("-", "1", "1")),
        (2, ("-", "1", "1")),
        (3, ("+", "1", "1")),
        (3, ("+", "1", "1")),
    )


def test_pointer_occurrences_and_literal_leaf_capacity_are_frozen() -> None:
    _, donors, _ = _selected()
    donor = donors[0]
    teachers = {probe.CLAMP_SUM_KEY: probe._internal_teacher_tokens(probe.CLAMP_SUM_KEY)}
    prompts = {
        probe.CLAMP_SUM_KEY: probe._tokenized_public_prompt(donor.public),
    }
    accounting = probe._learning_arrival_accounting(prompts, teachers)

    assert accounting["base_learning_arrivals"] == 337
    assert accounting["constructor_pointer_cue_arrivals"] == 11
    assert accounting["total_learning_arrivals"] == 348
    assert accounting["noninitial_learning_arrivals"] == 347
    assert accounting["base_temporal_segment_rows_upper_bound"] == 3_696
    assert accounting["ordinary_segment_rows_upper_bound_including_pointer_cues"] == 3_817
    assert accounting["segment_capacity"] == 3_938
    occurrences = accounting["episodes"]["t2_clamp_then_sum/006"][
        "constructor_pointer_occurrences"
    ]
    assert len(occurrences) == 11
    assert occurrences[0]["surface"] == "sum"
    assert occurrences[5] == {
        "occurrence": 5,
        "teacher_position": 6,
        "surface": "integer",
        "surface_feedback_source": "t2_clamp_then_sum/006@6",
        "semantic_cue": "expression-int-child-1",
        "absolute_address": [0, 0, 0, 0, 1],
        "absolute_address_offset": 122,
        "relative_address": [1],
        "relative_address_offset": 2,
        "parent_occurrence": 4,
        "parent_pointer_source": "t2_clamp_then_sum/006#4",
        "maximum_persistent_rows": 22,
    }
    assert [
        item["teacher_position"]
        for item in occurrences
        if item["surface"] == "integer"
    ] == [6, 9, 14, 18]


def test_native_hashes_and_spent_result_are_frozen() -> None:
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
        "f7acfaa56730ea47a2de98c230d81f9836118a3cbb38f11e8881959264825f99"
    )
    assert probe._sha256_file(probe.SPEND_MARKER_PATH) == (
        "f532023f1d05c2e92a6f30dfdc304d4991a065e4b0fb6d3cee0cf9f88bd68145"
    )
    payload = json.loads(probe.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert payload["outcome"] == "STOP_NATIVE_TYPED_LITERAL_LEAF_POINTER"
    assert payload["first_failure"]["detail"]["completed_surface_prefix"] == 0
    assert payload["authority"]["teacher_free_target_attempts_completed"] == 1


def test_spend_marker_is_durable_exclusive_and_cleanup_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_artifact = probe.OUTPUT_PATH
    canonical_marker = probe.SPEND_MARKER_PATH
    test_dir = (
        probe.REPO_ROOT
        / ".pytest_tmp"
        / f"literal_leaf_spend_marker_{probe.os.getpid()}"
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
            "r3-public-simple-swe-native-typed-literal-leaf-pointer"
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
