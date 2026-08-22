from __future__ import annotations

from pathlib import Path

import json

import pytest

from humanbrain.typed_curriculum import (
    EvidenceExistsError,
    ExecutionRefusedError,
    FROZEN_PUBLIC_CURRICULUM_SHA256,
    load_public_curriculum,
    run,
)


PROTOCOL = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "typed_program_public_train_protocol_v1.json"
)


def test_public_curriculum_exposes_only_the_three_fixed_train_programs() -> None:
    curriculum = load_public_curriculum(PROTOCOL)

    assert [task.task_id for task in curriculum.train_tasks] == [
        "train_int_identity",
        "train_int_add_negative_two",
        "train_list_map_add_positive_three",
    ]
    assert [task.teacher_tokens for task in curriculum.train_tasks] == [
        ("variable-0",),
        ("add", "variable-0", "integer", "-", "2", "end"),
        (
            "map",
            "bind",
            "add",
            "variable-0",
            "integer",
            "+",
            "3",
            "end",
            "variable-0",
        ),
    ]
    assert sum(len(task.semantic_cases) for task in curriculum.train_tasks) == 9
    assert sum(len(task.teacher_tokens) for task in curriculum.train_tasks) == 16


def test_public_curriculum_raw_bytes_are_preregistered() -> None:
    curriculum = load_public_curriculum(PROTOCOL)

    assert curriculum.raw_sha256 == FROZEN_PUBLIC_CURRICULUM_SHA256


def test_loaded_semantic_values_are_deeply_immutable() -> None:
    curriculum = load_public_curriculum(PROTOCOL)
    mixed_list_case = curriculum.train_tasks[2].semantic_cases[1]

    assert mixed_list_case.arguments == ((-4, 0, 8),)
    assert mixed_list_case.expected == (-1, 3, 11)
    assert isinstance(mixed_list_case.arguments[0], tuple)


def test_modified_protocol_is_rejected_even_when_its_json_is_valid(tmp_path: Path) -> None:
    modified = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    modified["train"]["tasks"][0]["prompt"] = "Return this integer unchanged."
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(modified), encoding="utf-8")

    with pytest.raises(ValueError, match="raw SHA-256 is not preregistered"):
        load_public_curriculum(path)


def _isolated_protocol(tmp_path: Path) -> Path:
    path = tmp_path / "benchmarks" / PROTOCOL.name
    path.parent.mkdir(parents=True)
    path.write_bytes(PROTOCOL.read_bytes())
    return path


def test_runner_is_a_nonwriting_train_plan_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = _isolated_protocol(tmp_path)

    assert run(("--protocol", str(protocol))) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "checkpoint_count": 1,
        "completion_attempts_per_train_task": 1,
        "development_completion_opened": False,
        "development_run_count": 0,
        "development_task_count": 2,
        "execute": False,
        "mode": "plan",
        "protocol_id": "humanbrain-public-typed-program-train-v1",
        "protocol_sha256": FROZEN_PUBLIC_CURRICULUM_SHA256,
        "train_semantic_case_count": 9,
        "train_task_count": 3,
        "train_task_ids": [
            "train_int_identity",
            "train_int_add_negative_two",
            "train_list_map_add_positive_three",
        ],
        "train_teacher_token_count": 16,
    }
    assert not (tmp_path / "artifacts").exists()


def test_execute_requires_the_exact_raw_protocol_sha_before_writing(
    tmp_path: Path,
) -> None:
    protocol = _isolated_protocol(tmp_path)

    with pytest.raises(ExecutionRefusedError, match="exact raw protocol SHA-256"):
        run(("--protocol", str(protocol), "--execute"))
    with pytest.raises(ExecutionRefusedError, match="exact raw protocol SHA-256"):
        run(
            (
                "--protocol",
                str(protocol),
                "--execute",
                "--protocol-sha256",
                "0" * 64,
            )
        )

    assert not (tmp_path / "artifacts").exists()


def test_execute_rejects_existing_checkpoint_evidence_before_training(
    tmp_path: Path,
) -> None:
    protocol = _isolated_protocol(tmp_path)
    checkpoint = (
        tmp_path / "artifacts" / "typed_program_public_train_v1" / "checkpoint.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"already exists")

    with pytest.raises(EvidenceExistsError, match="checkpoint evidence already exists"):
        run(
            (
                "--protocol",
                str(protocol),
                "--execute",
                "--protocol-sha256",
                FROZEN_PUBLIC_CURRICULUM_SHA256,
            )
        )
