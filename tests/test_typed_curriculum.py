from __future__ import annotations

import json
import tracemalloc
from hashlib import sha256
from pathlib import Path

import pytest

import humanbrain.typed_curriculum as typed_curriculum
from humanbrain.typed_curriculum import (
    CurriculumGateError,
    EvidenceExistsError,
    ExecutionRefusedError,
    FROZEN_PUBLIC_CURRICULUM_SHA256,
    load_public_curriculum,
    run,
)
from humanbrain.typed_program import Completion, Program, ProgramRaceAudit


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


def test_protocol_loader_requires_the_preregistered_task_and_run_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    modified["execution"]["gates"].remove("timing_memory_accounting")
    raw = json.dumps(modified, sort_keys=True).encode("utf-8")
    path = tmp_path / "missing-accounting-gate.json"
    path.write_bytes(raw)
    monkeypatch.setattr(
        typed_curriculum,
        "FROZEN_PUBLIC_CURRICULUM_SHA256",
        sha256(raw).hexdigest(),
    )

    with pytest.raises(ValueError, match="execution gate commitments changed"):
        load_public_curriculum(path)


def test_protocol_read_is_bounded_before_json_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = Path.open
    read_sizes: list[int] = []

    class BoundedReader:
        def __init__(self, stream: object) -> None:
            self._stream = stream

        def __enter__(self) -> BoundedReader:
            self._stream.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *args: object) -> object:
            return self._stream.__exit__(*args)  # type: ignore[attr-defined]

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self._stream.read(size)  # type: ignore[attr-defined,no-any-return]

    def tracked_open(path: Path, *args: object, **kwargs: object) -> BoundedReader:
        return BoundedReader(real_open(path, *args, **kwargs))

    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("protocol loader used an unbounded read")
        ),
    )
    monkeypatch.setattr(Path, "open", tracked_open)

    load_public_curriculum(PROTOCOL)

    assert read_sizes == [131_073]


def _relocate_runner(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    production = typed_curriculum._production_dependencies()
    monkeypatch.setattr(
        typed_curriculum,
        "_production_dependencies",
        lambda: typed_curriculum._RunDependencies(
            repository_root=root,
            protocol_path=production.protocol_path,
            grow=production.grow,
            complete=production.complete,
            restore_checkpoint=production.restore_checkpoint,
        ),
    )


def test_runner_is_a_nonwriting_train_plan_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _relocate_runner(monkeypatch, tmp_path)

    assert run(()) == 0

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _relocate_runner(monkeypatch, tmp_path)

    with pytest.raises(ExecutionRefusedError, match="exact raw protocol SHA-256"):
        run(("--execute",))
    with pytest.raises(ExecutionRefusedError, match="exact raw protocol SHA-256"):
        run(
            (
                "--execute",
                "--protocol-sha256",
                "0" * 64,
            )
        )

    assert not (tmp_path / "artifacts").exists()


def test_execute_rejects_existing_checkpoint_evidence_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _relocate_runner(monkeypatch, tmp_path)
    checkpoint = (
        tmp_path / "artifacts" / "typed_program_public_train_v1" / "checkpoint.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"already exists")

    with pytest.raises(EvidenceExistsError, match="checkpoint evidence already exists"):
        run(
            (
                "--execute",
                "--protocol-sha256",
                FROZEN_PUBLIC_CURRICULUM_SHA256,
            )
        )


def test_public_runner_has_no_protocol_relocation_option() -> None:
    with pytest.raises(SystemExit):
        run(("--protocol", str(PROTOCOL)))


class _FakeCheckpoint:
    def __init__(self, raw: bytes, *, fail_audit_read: bool = False) -> None:
        self._raw = raw
        self._fail_audit_read = fail_audit_read
        self.sha256 = sha256(raw).hexdigest()
        self.training_digest = "1" * 64
        self.training_input_sha256 = ("2" * 64, "3" * 64, "4" * 64)
        self.training_sequence_token_counts = (11, 12, 13)
        self.cell_synapse_sha256 = "5" * 64
        self.protocol_sha256 = "6" * 64

    def to_bytes(self) -> bytes:
        if self._fail_audit_read:
            raise RuntimeError("fixture audit read failed after completion")
        return self._raw


def _fake_completion(
    checkpoint: _FakeCheckpoint,
    prompt: str,
    signature: object,
) -> Completion:
    curriculum = load_public_curriculum(PROTOCOL)
    task = next(task for task in curriculum.train_tasks if task.prompt == prompt)
    program = Program(task.teacher_tokens, signature)
    races = tuple(
        ProgramRaceAudit(
            accepted_token=token,
            winner_cell_id=300 + index,
            spike_time_ms=0.25 + index,
            simulated_steps=index + 1,
            prefix_sha256=sha256(
                json.dumps(
                    list(program.tokens[:index]),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            allowed_cell_ids=(300 + index,),
            excluded_cell_ids=(400 + index,),
            output_current_sha256="7" * 64,
        )
        for index, token in enumerate(program.tokens)
    )
    return Completion(
        program=program,
        races=races,
        prompt_sha256=sha256(prompt.encode("utf-8")).hexdigest(),
        model_input_sha256="8" * 64,
        checkpoint_sha256=checkpoint.sha256,
        checkpoint_cell_synapse_before=checkpoint.cell_synapse_sha256,
        checkpoint_cell_synapse_after=checkpoint.cell_synapse_sha256,
    )


def _install_successful_fake_runner(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> None:
    raw_checkpoint = b'{"fixture":"successful-common-checkpoint"}'
    monkeypatch.setattr(
        typed_curriculum,
        "_production_dependencies",
        lambda: typed_curriculum._RunDependencies(
            repository_root=root,
            protocol_path=PROTOCOL,
            grow=lambda _protocol, _examples: _FakeCheckpoint(raw_checkpoint),
            complete=_fake_completion,
            restore_checkpoint=lambda raw: _FakeCheckpoint(raw),
        ),
    )


def test_public_execute_attempts_every_train_task_once_and_seals_full_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_checkpoint = b'{"fixture":"one-common-checkpoint"}'
    grow_calls: list[tuple[object, tuple[object, ...]]] = []
    restore_calls: list[bytes] = []
    restored: list[_FakeCheckpoint] = []
    completion_calls: list[tuple[int, str]] = []

    def fake_grow(protocol: object, examples: tuple[object, ...]) -> _FakeCheckpoint:
        grow_calls.append((protocol, examples))
        return _FakeCheckpoint(raw_checkpoint)

    def fake_restore(raw: bytes) -> _FakeCheckpoint:
        restore_calls.append(raw)
        checkpoint = _FakeCheckpoint(
            raw,
            fail_audit_read=len(restore_calls) == 2,
        )
        restored.append(checkpoint)
        return checkpoint

    def fake_complete(
        checkpoint: _FakeCheckpoint,
        prompt: str,
        signature: object,
    ) -> Completion:
        completion_calls.append((id(checkpoint), prompt))
        return _fake_completion(checkpoint, prompt, signature)

    monkeypatch.setattr(
        typed_curriculum,
        "_production_dependencies",
        lambda: typed_curriculum._RunDependencies(
            repository_root=tmp_path,
            protocol_path=PROTOCOL,
            grow=fake_grow,
            complete=fake_complete,
            restore_checkpoint=fake_restore,
        ),
    )
    arguments = (
        "--execute",
        "--protocol-sha256",
        FROZEN_PUBLIC_CURRICULUM_SHA256,
    )

    with pytest.raises(CurriculumGateError, match="TRAIN gates failed"):
        run(arguments)

    summary = json.loads(capsys.readouterr().out)
    evidence = (
        tmp_path / "artifacts" / "typed_program_public_train_v1" / "evidence"
    )
    checkpoint_path = evidence.parent / "checkpoint.json"
    evidence_files = sorted(evidence.glob("*.json"))
    second_task = json.loads(evidence_files[3].read_text(encoding="utf-8"))

    assert len(grow_calls) == 1
    assert len(grow_calls[0][1]) == 3
    assert restore_calls == [raw_checkpoint, raw_checkpoint, raw_checkpoint]
    assert [identity for identity, _prompt in completion_calls] == [
        id(checkpoint) for checkpoint in restored
    ]
    assert len(set(identity for identity, _prompt in completion_calls)) == 3
    assert [prompt for _identity, prompt in completion_calls] == [
        "Return the integer unchanged.",
        "Subtract two from the integer.",
        "Add three to every integer in the list.",
    ]
    assert checkpoint_path.read_bytes() == raw_checkpoint
    assert [path.name for path in evidence_files] == [
        "000-run-start.json",
        "001-training.json",
        "002-train_int_identity.json",
        "003-train_int_add_negative_two.json",
        "004-train_list_map_add_positive_three.json",
        "005-summary.json",
    ]
    assert not list(evidence.glob("*.tmp"))
    assert second_task["completion_returned"] is True
    assert second_task["result"]["tokens"] == [
        "add",
        "variable-0",
        "integer",
        "-",
        "2",
        "end",
    ]
    assert second_task["result"]["race_count"] == 6
    assert second_task["result"]["program_sha256"]
    assert second_task["result"]["prompt_sha256"]
    assert second_task["result"]["checkpoint_sha256"] == sha256(
        raw_checkpoint
    ).hexdigest()
    assert second_task["result"]["gates"]["timing_memory_accounting"][
        "passed"
    ] is True

    assert summary["all_train_gates_passed"] is False
    assert summary["completion_attempts"] == 3
    assert summary["measurements"]["completion_attempts"] == 3
    assert summary["measurements"]["completion_tokens"] == 16
    assert summary["measurements"]["cell_races"] == 16
    assert summary["measurements"]["training_sequences"] == 3
    assert summary["measurements"]["training_tokens"] == 36
    assert summary["measurements"]["checkpoint_bytes"] == len(raw_checkpoint)
    assert summary["measurements"]["evidence_bytes"] == sum(
        path.stat().st_size for path in evidence_files
    )
    assert summary["gates"]["timing_memory_accounting"]["passed"] is True
    assert summary["task_results"][1]["completion_token_count"] == 6
    assert summary["task_results"][1]["race_count"] == 6
    assert summary["task_results"][1]["program_sha256"]
    assert summary["task_results"][1]["prompt_sha256"]
    assert summary["task_results"][1]["checkpoint_sha256"] == sha256(
        raw_checkpoint
    ).hexdigest()

    before = {path.name: path.read_bytes() for path in evidence_files}
    with pytest.raises(EvidenceExistsError, match="checkpoint evidence already exists"):
        run(arguments)
    assert {path.name: path.read_bytes() for path in evidence_files} == before


def test_timing_memory_accounting_gate_controls_task_and_run_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_successful_fake_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        typed_curriculum.tracemalloc,
        "get_traced_memory",
        lambda: (10, 5),
    )

    with pytest.raises(CurriculumGateError, match="TRAIN gates failed"):
        run(
            (
                "--execute",
                "--protocol-sha256",
                FROZEN_PUBLIC_CURRICULUM_SHA256,
            )
        )

    summary = json.loads(capsys.readouterr().out)
    task_report = json.loads(
        (
            tmp_path
            / "artifacts"
            / "typed_program_public_train_v1"
            / "evidence"
            / "002-train_int_identity.json"
        ).read_text(encoding="utf-8")
    )
    assert task_report["result"]["gates"]["timing_memory_accounting"] == {
        "detail": (
            "all registered timing, memory, and accounting counters are present "
            "and valid"
        ),
        "passed": False,
        "status": "failed",
    }
    assert task_report["result"]["all_gates_passed"] is False
    assert summary["gates"]["timing_memory_accounting"]["passed"] is False
    assert summary["all_train_gates_passed"] is False


def test_one_completion_exception_does_not_skip_later_train_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_checkpoint = b'{"fixture":"completion-failure-checkpoint"}'
    prompts: list[str] = []

    def complete_once(
        checkpoint: _FakeCheckpoint,
        prompt: str,
        signature: object,
    ) -> Completion:
        prompts.append(prompt)
        if len(prompts) == 2:
            raise RuntimeError("fixture completion failed")
        return _fake_completion(checkpoint, prompt, signature)

    monkeypatch.setattr(
        typed_curriculum,
        "_production_dependencies",
        lambda: typed_curriculum._RunDependencies(
            repository_root=tmp_path,
            protocol_path=PROTOCOL,
            grow=lambda _protocol, _examples: _FakeCheckpoint(raw_checkpoint),
            complete=complete_once,
            restore_checkpoint=lambda raw: _FakeCheckpoint(raw),
        ),
    )

    with pytest.raises(CurriculumGateError, match="TRAIN gates failed"):
        run(
            (
                "--execute",
                "--protocol-sha256",
                FROZEN_PUBLIC_CURRICULUM_SHA256,
            )
        )

    summary = json.loads(capsys.readouterr().out)
    assert prompts == [
        "Return the integer unchanged.",
        "Subtract two from the integer.",
        "Add three to every integer in the list.",
    ]
    assert summary["completion_attempts"] == 3
    assert [task["completion_returned"] for task in summary["task_results"]] == [
        True,
        False,
        True,
    ]


def test_execute_preserves_callers_existing_tracemalloc_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_fake_runner(monkeypatch, tmp_path)
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        assert run(
            (
                "--execute",
                "--protocol-sha256",
                FROZEN_PUBLIC_CURRICULUM_SHA256,
            )
        ) == 0

        assert tracemalloc.is_tracing() is True
    finally:
        if not already_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()


def test_execute_stops_only_the_tracemalloc_session_it_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_successful_fake_runner(monkeypatch, tmp_path)
    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        tracemalloc.stop()
    try:
        assert run(
            (
                "--execute",
                "--protocol-sha256",
                FROZEN_PUBLIC_CURRICULUM_SHA256,
            )
        ) == 0

        assert tracemalloc.is_tracing() is False
    finally:
        if already_tracing and not tracemalloc.is_tracing():
            tracemalloc.start()
