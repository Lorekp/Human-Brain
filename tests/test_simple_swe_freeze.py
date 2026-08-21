from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from humanbrain.language import CellLanguageModel
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


def _clean_repository(root: Path) -> Path:
    repository = root / "source"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "simple-swe@example.invalid")
    _git(repository, "config", "user.name", "Simple SWE Test")
    (repository / "model_source.py").write_text("CELL_ONLY = True\n", encoding="utf-8")
    _git(repository, "add", "model_source.py")
    _git(repository, "commit", "-q", "-m", "frozen source")
    return repository


def _checkpoint(root: Path) -> Path:
    model = CellLanguageModel.grow(["def identity(value):\n    return value\n"])
    checkpoint = root / "generic-cell-checkpoint.npz"
    model.save(checkpoint)
    return checkpoint


def test_freeze_manifest_commits_clean_source_and_checkpoint_components(
    tmp_path: Path,
) -> None:
    source = _clean_repository(tmp_path)
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


def test_freeze_rejects_a_dirty_source_tree(tmp_path: Path) -> None:
    source = _clean_repository(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    (source / "uncommitted.txt").write_text("drift", encoding="utf-8")

    with pytest.raises(FreezeIntegrityError, match="source tree is dirty"):
        create_freeze_manifest(
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
            output_path=tmp_path / "freeze.json",
        )


def test_verify_freeze_rejects_checkpoint_drift(tmp_path: Path) -> None:
    source = _clean_repository(tmp_path)
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


def test_verify_freeze_rejects_source_commit_drift(tmp_path: Path) -> None:
    source = _clean_repository(tmp_path)
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
