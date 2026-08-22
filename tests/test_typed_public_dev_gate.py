from __future__ import annotations

import io
import json
import tracemalloc
from hashlib import sha256
from pathlib import Path

import humanbrain.typed_public_dev_gate as typed_public_dev_gate
import pytest
from humanbrain.typed_program import Completion, Program, ProgramRaceAudit
from humanbrain.typed_public_dev_gate import (
    FROZEN_PUBLIC_DEV_GATE_SHA256,
    DevGateError,
    EvidenceExistsError,
    ExecutionRefusedError,
    load_public_dev_gate,
    run,
)

PROTOCOL = Path(__file__).parents[1] / "benchmarks" / "typed_program_public_dev_protocol_v1.json"


def _fixed_size_json(value: object) -> bytes:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_bytes = PROTOCOL.stat().st_size
    assert len(raw) <= expected_bytes
    return raw + (b" " * (expected_bytes - len(raw)))


def test_public_dev_gate_exposes_only_the_two_teacher_free_definitions() -> None:
    gate = load_public_dev_gate(PROTOCOL)

    assert gate.raw_sha256 == FROZEN_PUBLIC_DEV_GATE_SHA256
    assert [task.task_id for task in gate.tasks] == [
        "dev_int_two_stage_offset",
        "dev_list_two_stage_map_offset",
    ]
    assert sum(len(task.semantic_cases) for task in gate.tasks) == 6
    assert [task.literals for task in gate.tasks] == [(7, -4), (-6, 8)]
    assert gate.tasks[1].semantic_cases[1].arguments == ((-5, 0, 7),)
    assert gate.tasks[1].semantic_cases[1].expected == (-3, 2, 9)
    assert not hasattr(gate.tasks[0], "teacher_tokens")


def test_loader_rejects_a_changed_task_gate_even_with_a_matching_raw_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    modified["execution"]["task_gates"].remove("semantics")
    raw = _fixed_size_json(modified)
    path = tmp_path / PROTOCOL.name
    path.write_bytes(raw)
    (tmp_path / "typed_program_public_train_protocol_v1.json").write_bytes(
        (PROTOCOL.parent / "typed_program_public_train_protocol_v1.json").read_bytes()
    )
    monkeypatch.setattr(
        typed_public_dev_gate,
        "FROZEN_PUBLIC_DEV_GATE_SHA256",
        sha256(raw).hexdigest(),
    )

    with pytest.raises(ValueError, match="task gate commitments changed"):
        load_public_dev_gate(path)


def test_loader_rejects_changed_train_inputs_even_with_a_matching_raw_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    modified["train_inputs"]["checkpoint"]["raw_sha256"] = "0" * 64
    raw = _fixed_size_json(modified)
    path = tmp_path / PROTOCOL.name
    path.write_bytes(raw)
    (tmp_path / "typed_program_public_train_protocol_v1.json").write_bytes(
        (PROTOCOL.parent / "typed_program_public_train_protocol_v1.json").read_bytes()
    )
    monkeypatch.setattr(
        typed_public_dev_gate,
        "FROZEN_PUBLIC_DEV_GATE_SHA256",
        sha256(raw).hexdigest(),
    )

    with pytest.raises(ValueError, match="TRAIN input commitments changed"):
        load_public_dev_gate(path)


def test_loader_rejects_changed_reporting_even_with_a_matching_raw_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    modified["reporting"]["promotion_semantic_passes_required"] = 2
    raw = _fixed_size_json(modified)
    path = tmp_path / PROTOCOL.name
    path.write_bytes(raw)
    (tmp_path / "typed_program_public_train_protocol_v1.json").write_bytes(
        (PROTOCOL.parent / "typed_program_public_train_protocol_v1.json").read_bytes()
    )
    monkeypatch.setattr(
        typed_public_dev_gate,
        "FROZEN_PUBLIC_DEV_GATE_SHA256",
        sha256(raw).hexdigest(),
    )

    with pytest.raises(ValueError, match="reporting commitments changed"):
        load_public_dev_gate(path)


def test_protocol_reads_use_exact_max_plus_one_before_json_allocation(
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

    monkeypatch.setattr(Path, "open", tracked_open)

    load_public_dev_gate(PROTOCOL)

    assert read_sizes == [9_705, 7_691, 131_073]


def test_runner_is_a_source_checked_nonwriting_plan_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    production = typed_public_dev_gate._production_dependencies()

    def unexpected_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("plan mode opened the DEV runtime")

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=unexpected_call,
            restore_checkpoint=unexpected_call,
        ),
    )

    assert run(()) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "checkpoint": {
            "cell_synapse_sha256": (
                "86b40631c49db55124dff3212645dc571555c5939fc4004a5d94c54b49e99779"
            ),
            "git_blob": "430faa1221c7721d97f43a9c311671e30bb65317",
            "path": "artifacts/typed_program_public_train_v1/checkpoint.json",
            "raw_sha256": ("0c2c4fd3e16ab978ae6e25a16beab9d1c84ae867f63bffdad3745c07b68dcb9d"),
        },
        "checkpoint_restore_attempts": 2,
        "completion_attempts": 2,
        "development_semantic_case_count": 6,
        "development_task_count": 2,
        "development_task_ids": [
            "dev_int_two_stage_offset",
            "dev_list_two_stage_map_offset",
        ],
        "evidence_directory": "artifacts/typed_program_public_dev_v1/evidence",
        "execute": False,
        "expected_execution": {
            "cold_wall_time": "up to 10 minutes",
            "process_memory": "under 512 MiB",
            "warm_wall_time": "1-3 minutes",
        },
        "mode": "plan",
        "model_calls": 0,
        "pass_at_1_range": [0, 2],
        "promotion": {
            "semantic_passes_required": 1,
            "task_count": 2,
            "requires_no_boundary_violations": True,
            "requires_syntax_and_compile": True,
        },
        "protocol_id": "humanbrain-public-typed-program-dev-v1",
        "protocol_sha256": FROZEN_PUBLIC_DEV_GATE_SHA256,
        "source": {
            "commit": "1ae5418b92e81dc48c6a2e36fb8822ceec7ce202",
            "files": [
                {
                    "git_blob": "8d8b6aed9958b70a7d5583ffab1f8b61315766c6",
                    "path": "benchmarks/typed_program_public_train.py",
                    "raw_sha256": (
                        "c45dbc15f55b7ad8a8c4dd971c46096b48f4488f935f962eb503412d1c499bc1"
                    ),
                },
                {
                    "git_blob": "c680b488a5bb2a5e970e244244af5da03d1808c0",
                    "path": "src/humanbrain/typed_curriculum.py",
                    "raw_sha256": (
                        "d171f12aca180f87295468ef282b10d267dddd5f1f8265ba353dc3a1bb4eb8e6"
                    ),
                },
                {
                    "git_blob": "93e1bb1c8d426d3dade24b7cbd5d201243998ba6",
                    "path": "src/humanbrain/typed_program.py",
                    "raw_sha256": (
                        "447570e75264bb4419cd725f6a7508f92236088299a4d8fd9742f8ee334a819b"
                    ),
                },
                {
                    "git_blob": "5c94b46ec632702bdf062c3104d4c1bb013e38c5",
                    "path": "src/humanbrain/typed_program_types.py",
                    "raw_sha256": (
                        "5f06a43bc771a8d61881d7ebf7c7decefcadbe245d77ce10640dbfff5174d099"
                    ),
                },
            ],
        },
        "train_artifact_commit": "8291fe1a0039db7141817c438854a1817a0a8300",
        "train_artifact_file_count": 7,
        "train_protocol": {
            "git_blob": "afdacef6f9d53a1e16b76155d8accc3a1ecf9e46",
            "path": "benchmarks/typed_program_public_train_protocol_v1.json",
            "raw_sha256": ("1ed77f9fa2395f6be8bed74648dc9d3de3027e367e626e427e56de912f4ed191"),
        },
        "train_summary": {
            "git_blob": "5ec6d687ed62aeb674231941c7ac52591d6f980f",
            "path": "artifacts/typed_program_public_train_v1/evidence/005-summary.json",
            "raw_sha256": ("71133044d55e48a3bbe5cb147a33ce2f543e0a0dad855ae11e2e1b140b6b02d7"),
        },
    }
    assert not (tmp_path / "artifacts").exists()


class _FakeCheckpoint:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self.sha256 = sha256(raw).hexdigest()
        self.cell_synapse_sha256 = (
            "86b40631c49db55124dff3212645dc571555c5939fc4004a5d94c54b49e99779"
        )

    def to_bytes(self) -> bytes:
        return self._raw


def _completion_for_tokens(
    checkpoint: _FakeCheckpoint,
    prompt: str,
    signature: object,
    tokens: tuple[str, ...],
) -> Completion:
    program = Program(tokens, signature)
    races = tuple(
        ProgramRaceAudit(
            accepted_token=token,
            winner_cell_id=700 + index,
            spike_time_ms=0.5 + index,
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
            allowed_cell_ids=(700 + index,),
            excluded_cell_ids=(900 + index,),
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


def _fake_dev_completion(
    checkpoint: _FakeCheckpoint,
    prompt: str,
    signature: object,
) -> Completion:
    gate = load_public_dev_gate(PROTOCOL)
    task = next(task for task in gate.tasks if task.prompt == prompt)
    tokens_by_task = {
        "dev_int_two_stage_offset": (
            "add",
            "add",
            "variable-0",
            "integer",
            "+",
            "7",
            "end",
            "integer",
            "-",
            "4",
            "end",
        ),
        "dev_list_two_stage_map_offset": (
            "map",
            "bind",
            "add",
            "add",
            "variable-0",
            "integer",
            "-",
            "6",
            "end",
            "integer",
            "+",
            "8",
            "end",
            "variable-0",
        ),
    }
    return _completion_for_tokens(
        checkpoint,
        prompt,
        signature,
        tokens_by_task[task.task_id],
    )


def test_execute_uses_one_fresh_restore_and_completion_per_task_and_seals_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    raw_checkpoint = (
        production.repository_root
        / "artifacts"
        / "typed_program_public_train_v1"
        / "checkpoint.json"
    ).read_bytes()
    restore_calls: list[bytes] = []
    restored: list[_FakeCheckpoint] = []
    completion_calls: list[tuple[int, str]] = []

    def fake_restore(raw: bytes) -> _FakeCheckpoint:
        restore_calls.append(raw)
        checkpoint = _FakeCheckpoint(raw)
        restored.append(checkpoint)
        return checkpoint

    def fake_complete(
        checkpoint: _FakeCheckpoint,
        prompt: str,
        signature: object,
    ) -> Completion:
        completion_calls.append((id(checkpoint), prompt))
        return _fake_dev_completion(checkpoint, prompt, signature)

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=fake_complete,
            restore_checkpoint=fake_restore,
        ),
    )

    assert run(("--execute", "--protocol-sha256", FROZEN_PUBLIC_DEV_GATE_SHA256)) == 0

    summary = json.loads(capsys.readouterr().out)
    evidence = tmp_path / "artifacts" / "typed_program_public_dev_v1" / "evidence"
    evidence_files = sorted(evidence.glob("*.json"))
    first_task = json.loads(evidence_files[1].read_text(encoding="utf-8"))

    assert restore_calls == [raw_checkpoint, raw_checkpoint]
    assert [identity for identity, _prompt in completion_calls] == [
        id(checkpoint) for checkpoint in restored
    ]
    assert len(set(identity for identity, _prompt in completion_calls)) == 2
    assert [path.name for path in evidence_files] == [
        "000-run-start.json",
        "001-dev_int_two_stage_offset.json",
        "002-dev_list_two_stage_map_offset.json",
        "003-summary.json",
    ]
    assert not list(evidence.glob("*.tmp"))
    assert first_task["restore_attempt_count"] == 1
    assert first_task["completion_attempt_count"] == 1
    assert first_task["result"]["boundary_violations"] == []
    assert first_task["result"]["memorization_control"] == {
        "category": "not exact public TRAIN-label replay",
        "exact_train_label_match": False,
        "matched_train_task_ids": [],
        "model_calls": 0,
        "note": "this control does not establish broad generalization",
    }
    assert summary["pass_at_1"] == 2
    assert summary["pass_at_1_range"] == [0, 2]
    assert summary["failures"] == []
    assert summary["promotion"] == {
        "achieved": True,
        "eligible_task_count": 2,
        "semantic_passes_required": 1,
        "task_count": 2,
    }
    assert summary["measurements"]["checkpoint_restore_attempts"] == 2
    assert summary["measurements"]["completion_attempts"] == 2
    assert summary["measurements"]["evidence_bytes"] == sum(
        path.stat().st_size for path in evidence_files
    )


def test_one_completion_failure_does_not_skip_the_later_dev_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    prompts: list[str] = []

    def complete_with_first_failure(
        checkpoint: _FakeCheckpoint,
        prompt: str,
        signature: object,
    ) -> Completion:
        prompts.append(prompt)
        if len(prompts) == 1:
            raise RuntimeError("fixture completion failed")
        return _fake_dev_completion(checkpoint, prompt, signature)

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=complete_with_first_failure,
            restore_checkpoint=lambda raw: _FakeCheckpoint(raw),
        ),
    )

    assert run(("--execute", "--protocol-sha256", FROZEN_PUBLIC_DEV_GATE_SHA256)) == 0

    summary = json.loads(capsys.readouterr().out)
    evidence = tmp_path / "artifacts" / "typed_program_public_dev_v1" / "evidence"
    first_task = json.loads(
        (evidence / "001-dev_int_two_stage_offset.json").read_text(encoding="utf-8")
    )
    assert prompts == [
        "Add seven to the integer, then subtract four.",
        "For every integer, subtract six and then add eight.",
    ]
    assert first_task["completion_attempt_count"] == 1
    assert first_task["result"]["attempt_error"] == ("RuntimeError: fixture completion failed")
    assert summary["pass_at_1"] == 1
    assert summary["promotion"]["achieved"] is True
    assert [failure["task_id"] for failure in summary["failures"]] == ["dev_int_two_stage_offset"]


def test_exact_train_label_replay_is_distinguished_without_an_extra_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    production = typed_public_dev_gate._production_dependencies()

    def completion_with_one_train_replay(
        checkpoint: _FakeCheckpoint,
        prompt: str,
        signature: object,
    ) -> Completion:
        if prompt == "Add seven to the integer, then subtract four.":
            return _completion_for_tokens(
                checkpoint,
                prompt,
                signature,
                ("variable-0",),
            )
        return _fake_dev_completion(checkpoint, prompt, signature)

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=completion_with_one_train_replay,
            restore_checkpoint=lambda raw: _FakeCheckpoint(raw),
        ),
    )

    assert run(("--execute", "--protocol-sha256", FROZEN_PUBLIC_DEV_GATE_SHA256)) == 0

    summary = json.loads(capsys.readouterr().out)
    task = json.loads(
        (
            tmp_path
            / "artifacts"
            / "typed_program_public_dev_v1"
            / "evidence"
            / "001-dev_int_two_stage_offset.json"
        ).read_text(encoding="utf-8")
    )
    assert task["result"]["memorization_control"] == {
        "category": "exact public TRAIN-label replay",
        "exact_train_label_match": True,
        "matched_train_task_ids": ["train_int_identity"],
        "model_calls": 0,
        "note": "this control does not establish broad generalization",
    }
    assert task["result"]["gates"]["teacher_free_label_boundary"]["passed"] is False
    assert "teacher_free_label_boundary" in task["result"]["boundary_violations"]
    assert summary["pass_at_1"] == 1
    assert summary["promotion"]["eligible_task_count"] == 1


def test_existing_dev_evidence_fails_closed_before_input_or_runtime_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    evidence_root = tmp_path / "artifacts" / "typed_program_public_dev_v1"
    evidence_root.mkdir(parents=True)

    def unexpected_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("existing evidence did not stop the runner")

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=unexpected_call,
            restore_checkpoint=unexpected_call,
            git_blob_at_commit=unexpected_call,
        ),
    )

    with pytest.raises(EvidenceExistsError, match="evidence root already exists"):
        run(())


def test_execute_requires_the_exact_raw_dev_protocol_hash_before_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = typed_public_dev_gate._production_dependencies()

    def unexpected_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("authorization failure reached the DEV preflight")

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=unexpected_call,
            restore_checkpoint=unexpected_call,
            git_blob_at_commit=unexpected_call,
        ),
    )

    with pytest.raises(ExecutionRefusedError, match="exact raw DEV protocol SHA-256"):
        run(("--execute", "--protocol-sha256", "0" * 64))
    assert not (tmp_path / "artifacts").exists()


def test_plan_rejects_a_mismatched_committed_git_blob_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = typed_public_dev_gate._production_dependencies()

    def unexpected_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Git preflight failure reached the DEV runtime")

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=unexpected_call,
            restore_checkpoint=unexpected_call,
            git_blob_at_commit=lambda _root, _commit, _path: "0" * 40,
        ),
    )

    with pytest.raises(ValueError, match="checked TRAIN Git blob changed"):
        run(())
    assert not (tmp_path / "artifacts").exists()


def test_plan_queries_each_checked_file_at_its_declared_commit_and_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    gate = load_public_dev_gate(PROTOCOL)
    expected_blobs = {
        gate.train_protocol_relative_path: gate.train_protocol_git_blob,
        **{
            relative_path: git_blob
            for relative_path, _raw_sha256, git_blob, _expected_bytes in (
                gate.source_file_identities
            )
        },
        **{
            relative_path: git_blob
            for relative_path, _raw_sha256, git_blob, _expected_bytes in (
                gate.train_artifact_file_identities
            )
        },
    }
    calls: list[tuple[str, Path]] = []

    def fake_git_blob(_root: Path, commit: str, relative_path: Path) -> str:
        calls.append((commit, relative_path))
        return expected_blobs[relative_path]

    def unexpected_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("plan mode opened the DEV runtime")

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=unexpected_call,
            restore_checkpoint=unexpected_call,
            git_blob_at_commit=fake_git_blob,
        ),
    )

    assert run(()) == 0
    capsys.readouterr()
    assert calls == [
        (gate.train_artifact_commit, gate.train_protocol_relative_path),
        *[
            (gate.source_commit, relative_path)
            for relative_path, _raw_sha256, _git_blob, _expected_bytes in (
                gate.source_file_identities
            )
        ],
        *[
            (gate.train_artifact_commit, relative_path)
            for relative_path, _raw_sha256, _git_blob, _expected_bytes in (
                gate.train_artifact_file_identities
            )
        ],
    ]
    assert not (tmp_path / "artifacts").exists()


def test_plan_fails_closed_when_the_declared_source_commit_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    gate = load_public_dev_gate(PROTOCOL)
    calls: list[tuple[str, Path]] = []

    def fake_git_blob(_root: Path, commit: str, relative_path: Path) -> str:
        calls.append((commit, relative_path))
        if commit == gate.source_commit:
            raise OSError("fixture source commit is unavailable")
        assert relative_path == gate.train_protocol_relative_path
        return gate.train_protocol_git_blob

    def unexpected_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unavailable source commit reached the DEV runtime")

    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=unexpected_call,
            restore_checkpoint=unexpected_call,
            git_blob_at_commit=fake_git_blob,
        ),
    )

    with pytest.raises(ValueError, match="checked TRAIN Git blob is unavailable"):
        run(())
    assert calls == [
        (gate.train_artifact_commit, gate.train_protocol_relative_path),
        (gate.source_commit, gate.source_file_identities[0][0]),
    ]
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("relative_path", "expected_bytes", "error"),
    [
        (
            Path("benchmarks/typed_program_public_dev_protocol_v1.json"),
            9_704,
            "public DEV protocol size changed",
        ),
        (
            Path("benchmarks/typed_program_public_train_protocol_v1.json"),
            7_690,
            "public TRAIN protocol size changed",
        ),
        (
            Path("src/humanbrain/typed_program.py"),
            49_243,
            "checked TRAIN input size changed",
        ),
        (
            Path("artifacts/typed_program_public_train_v1/checkpoint.json"),
            986_677,
            "checked TRAIN input size changed",
        ),
    ],
)
def test_execute_rejects_oversized_committed_inputs_after_one_bounded_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
    expected_bytes: int,
    error: str,
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    gate = load_public_dev_gate(PROTOCOL)
    target_path = (production.repository_root / relative_path).resolve()
    raw_target = target_path.read_bytes()
    assert len(raw_target) == expected_bytes
    read_sizes: list[int] = []
    real_open = Path.open
    expected_blobs = {
        gate.train_protocol_relative_path: gate.train_protocol_git_blob,
        **{
            path: git_blob
            for path, _raw_sha256, git_blob, _expected_bytes in (gate.source_file_identities)
        },
        **{
            path: git_blob
            for path, _raw_sha256, git_blob, _expected_bytes in (
                gate.train_artifact_file_identities
            )
        },
    }

    def bounded_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.resolve() == target_path:

            class OversizedReader(io.BytesIO):
                def read(self, size: int = -1) -> bytes:
                    read_sizes.append(size)
                    return super().read(size)

            return OversizedReader(raw_target + b"x")
        return real_open(path, *args, **kwargs)

    def unexpected_call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("oversized committed input reached the DEV runtime")

    monkeypatch.setattr(Path, "open", bounded_open)
    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=unexpected_call,
            restore_checkpoint=unexpected_call,
            git_blob_at_commit=lambda _root, _commit, path: expected_blobs[path],
        ),
    )

    with pytest.raises(ValueError, match=error):
        run(("--execute", "--protocol-sha256", FROZEN_PUBLIC_DEV_GATE_SHA256))

    assert read_sizes == [expected_bytes + 1]
    assert not (tmp_path / "artifacts").exists()


def test_execute_bounds_checkpoint_preflight_and_later_immutability_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    checkpoint_path = (
        production.repository_root
        / "artifacts"
        / "typed_program_public_train_v1"
        / "checkpoint.json"
    ).resolve()
    raw_checkpoint = checkpoint_path.read_bytes()
    read_sizes: list[int] = []
    real_open = Path.open

    class TrackedReader:
        def __init__(self, stream: object) -> None:
            self._stream = stream

        def __enter__(self) -> TrackedReader:
            self._stream.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *args: object) -> object:
            return self._stream.__exit__(*args)  # type: ignore[attr-defined]

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self._stream.read(size)  # type: ignore[attr-defined,no-any-return]

    def tracked_open(path: Path, *args: object, **kwargs: object) -> object:
        stream = real_open(path, *args, **kwargs)
        if path.resolve() == checkpoint_path:
            return TrackedReader(stream)
        return stream

    def forbidden_read_bytes(_path: Path) -> bytes:
        raise AssertionError("checkpoint validation used Path.read_bytes")

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=_fake_dev_completion,
            restore_checkpoint=lambda raw: _FakeCheckpoint(raw),
        ),
    )

    assert run(("--execute", "--protocol-sha256", FROZEN_PUBLIC_DEV_GATE_SHA256)) == 0
    assert read_sizes == [len(raw_checkpoint) + 1] * 3


@pytest.mark.parametrize(
    "arguments",
    [
        ("--protocol", str(PROTOCOL)),
        ("--checkpoint", "alternate.json"),
        ("--evidence", "alternate"),
        ("--task", "dev_int_two_stage_offset"),
        ("--seed", "7"),
    ],
)
def test_public_runner_has_no_relocation_selection_or_seed_options(
    arguments: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        run(arguments)


def test_execute_preserves_preexisting_tracemalloc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        monkeypatch.setattr(
            typed_public_dev_gate,
            "_production_dependencies",
            lambda: typed_public_dev_gate._RunDependencies(
                repository_root=production.repository_root,
                evidence_repository_root=tmp_path,
                protocol_path=production.protocol_path,
                complete=_fake_dev_completion,
                restore_checkpoint=lambda raw: _FakeCheckpoint(raw),
            ),
        )

        assert run(("--execute", "--protocol-sha256", FROZEN_PUBLIC_DEV_GATE_SHA256)) == 0
        assert tracemalloc.is_tracing()
    finally:
        if not was_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()


def test_execute_stops_runner_owned_tracemalloc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.stop()
    try:
        monkeypatch.setattr(
            typed_public_dev_gate,
            "_production_dependencies",
            lambda: typed_public_dev_gate._RunDependencies(
                repository_root=production.repository_root,
                evidence_repository_root=tmp_path,
                protocol_path=production.protocol_path,
                complete=_fake_dev_completion,
                restore_checkpoint=lambda raw: _FakeCheckpoint(raw),
            ),
        )

        assert run(("--execute", "--protocol-sha256", FROZEN_PUBLIC_DEV_GATE_SHA256)) == 0
        assert not tracemalloc.is_tracing()
    finally:
        if was_tracing and not tracemalloc.is_tracing():
            tracemalloc.start()


def test_timing_accounting_boundary_controls_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    production = typed_public_dev_gate._production_dependencies()
    monkeypatch.setattr(
        typed_public_dev_gate,
        "_production_dependencies",
        lambda: typed_public_dev_gate._RunDependencies(
            repository_root=production.repository_root,
            evidence_repository_root=tmp_path,
            protocol_path=production.protocol_path,
            complete=_fake_dev_completion,
            restore_checkpoint=lambda raw: _FakeCheckpoint(raw),
        ),
    )
    monkeypatch.setattr(
        typed_public_dev_gate.tracemalloc,
        "get_traced_memory",
        lambda: (10, 5),
    )

    with pytest.raises(DevGateError, match="promotion milestone was not met"):
        run(("--execute", "--protocol-sha256", FROZEN_PUBLIC_DEV_GATE_SHA256))

    summary = json.loads(capsys.readouterr().out)
    assert summary["pass_at_1"] == 2
    assert summary["promotion"]["eligible_task_count"] == 0
    assert summary["promotion"]["achieved"] is False
    assert all(
        "timing_memory_accounting" in task["boundary_violations"]
        for task in summary["task_results"]
    )
