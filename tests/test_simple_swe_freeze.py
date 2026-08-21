from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from humanbrain.language import CellLanguageModel
from humanbrain.simple_swe import freeze
from humanbrain.simple_swe.freeze import (
    FreezeIntegrityError,
    create_freeze_manifest,
    verify_freeze_manifest,
)

PROTOCOL = (
    Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-c", "core.excludesFile=", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def _clean_repository(
    root: Path, monkeypatch: pytest.MonkeyPatch | None = None
) -> Path:
    repository = root / "source"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "simple-swe@example.invalid")
    _git(repository, "config", "user.name", "Simple SWE Test")
    (repository / "model_source.py").write_text("CELL_ONLY = True\n", encoding="utf-8")
    installed_source = repository / "src" / "humanbrain" / "execution.py"
    installed_source.parent.mkdir(parents=True)
    installed_source.write_text("CELL_EXECUTION = True\n", encoding="utf-8")
    _git(repository, "add", "model_source.py", "src/humanbrain/execution.py")
    _git(repository, "commit", "-q", "-m", "frozen source")
    if monkeypatch is not None:
        monkeypatch.setattr(
            freeze,
            "_installed_execution_source_paths",
            lambda: (installed_source.resolve(),),
        )
    return repository


def _checkpoint(root: Path) -> Path:
    model = CellLanguageModel.grow(["def identity(value):\n    return value\n"])
    checkpoint = root / "generic-cell-checkpoint.npz"
    model.save(checkpoint)
    return checkpoint


def test_freeze_manifest_commits_clean_source_and_checkpoint_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _clean_repository(tmp_path, monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    manifest_path = tmp_path / "frozen" / "freeze.json"

    manifest = create_freeze_manifest(
        protocol_path=PROTOCOL,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=manifest_path,
    )
    verified = verify_freeze_manifest(
        manifest_path=manifest_path,
        protocol_path=PROTOCOL,
        checkpoint_path=checkpoint,
        source_root=source,
    )
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert verified == manifest
    assert persisted["checkpoint_sha256"] == manifest.checkpoint_sha256
    assert len({
        manifest.checkpoint_sha256,
        manifest.checkpoint_state_sha256,
        manifest.model_config_sha256,
        manifest.tokenizer_sha256,
        manifest.source_tree_sha256,
    }) == 5
    assert manifest.checkpoint_format == "human-cell-brain-checkpoint-v1"
    assert manifest.no_learning_after_freeze is True
    assert manifest.one_checkpoint_for_all_tasks is True
    assert manifest.evaluation_root == str(manifest_path.parent.resolve())
    assert manifest.compiler_factory_id == "humanbrain-frozen-compiler-registry-v1"
    assert manifest.canonical_runner_id == "humanbrain-simple-swe-sealed-runner-v1"
    assert manifest.sandbox_executor_id == "humanbrain-simple-swe-job-sandbox-v1"
    assert manifest.sandbox_python_flags == ("-I", "-S", "-B")
    assert manifest.sandbox_python_version.startswith("3.11.")
    assert manifest.simulation_backend == "python"
    assert manifest.simulation_artifact_sha256 is None
    assert {
        "humanbrain.simple_swe.protocol",
        "humanbrain.simple_swe.tasks",
        "humanbrain.simple_swe._worker",
        "humanbrain.simple_swe.runner",
        "humanbrain.simple_swe.reveal",
        "humanbrain.simple_swe.artifacts",
    }.issubset(freeze._EXECUTION_MODULES)


def test_freeze_manifest_cannot_move_to_an_alternate_evidence_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _clean_repository(tmp_path, monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    manifest_path = tmp_path / "canonical" / "freeze.json"
    create_freeze_manifest(
        protocol_path=PROTOCOL,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=manifest_path,
    )
    moved = tmp_path / "alternate" / "freeze.json"
    moved.parent.mkdir()
    moved.write_bytes(manifest_path.read_bytes())

    with pytest.raises(FreezeIntegrityError, match="canonical evidence root"):
        verify_freeze_manifest(
            manifest_path=moved,
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
        )


def test_verify_freeze_rejects_sandbox_runtime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _clean_repository(tmp_path, monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    manifest_path = tmp_path / "freeze.json"
    create_freeze_manifest(
        protocol_path=PROTOCOL,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=manifest_path,
    )
    frozen = freeze.python_runtime_identity

    def changed_runtime(executable, language):
        identity = frozen(executable, language)
        return {**identity, "executable_sha256": "0" * 64}

    monkeypatch.setattr(freeze, "python_runtime_identity", changed_runtime)

    with pytest.raises(FreezeIntegrityError, match="runtime commitment drift"):
        verify_freeze_manifest(
            manifest_path=manifest_path,
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
        )


def test_frozen_native_library_bytes_are_rehashed_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from humanbrain import native

    monkeypatch.setenv("HUMANBRAIN_DEVICE", "cpu")
    artifact = tmp_path / "humanbrain-kernel.dll"
    artifact.write_bytes(b"frozen native kernel")
    build = SimpleNamespace(source_digest="source-digest", library_path=artifact)
    monkeypatch.setattr(native, "loaded_build_info", lambda: build)
    identity = freeze.SimulationBackendIdentity(
        backend="cpu",
        device="fixture-cpu",
        device_policy="cpu",
        source_digest=build.source_digest,
        artifact_path=str(artifact.resolve()),
        artifact_sha256=sha256(artifact.read_bytes()).hexdigest(),
        architecture=None,
    )

    freeze._verify_simulation_backend_artifact(identity)
    artifact.write_bytes(b"tampered native kernel")

    with pytest.raises(FreezeIntegrityError, match="backend artifact drift"):
        freeze._verify_simulation_backend_artifact(identity)


def test_freeze_rejects_a_dirty_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _clean_repository(tmp_path, monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    (source / "uncommitted.txt").write_text("drift", encoding="utf-8")

    with pytest.raises(FreezeIntegrityError, match="source tree is dirty"):
        create_freeze_manifest(
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
            output_path=tmp_path / "freeze.json",
        )


def test_freeze_rejects_an_unrelated_clean_git_repository(tmp_path: Path) -> None:
    source = _clean_repository(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    with pytest.raises(
        FreezeIntegrityError, match="installed Human Brain execution source"
    ):
        create_freeze_manifest(
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
            output_path=tmp_path / "freeze.json",
        )


def test_verify_freeze_rejects_checkpoint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _clean_repository(tmp_path, monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    manifest_path = tmp_path / "freeze.json"
    create_freeze_manifest(
        protocol_path=PROTOCOL,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=manifest_path,
    )
    with checkpoint.open("ab") as stream:
        stream.write(b"drift")

    with pytest.raises(FreezeIntegrityError, match="checkpoint hash drift"):
        verify_freeze_manifest(
            manifest_path=manifest_path,
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
        )


def test_verify_freeze_rejects_source_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _clean_repository(tmp_path, monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    manifest_path = tmp_path / "freeze.json"
    create_freeze_manifest(
        protocol_path=PROTOCOL,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=manifest_path,
    )
    (source / "model_source.py").write_text("CELL_ONLY = 'changed'\n", encoding="utf-8")
    _git(source, "add", "model_source.py")
    _git(source, "commit", "-q", "-m", "changed source")

    with pytest.raises(FreezeIntegrityError, match="source commit drift"):
        verify_freeze_manifest(
            manifest_path=manifest_path,
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
        )
