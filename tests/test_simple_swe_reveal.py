from __future__ import annotations

import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from humanbrain.language import CellLanguageModel
from humanbrain.simple_swe import freeze
from humanbrain.simple_swe import reveal
from humanbrain.simple_swe.artifacts import ArtifactLedger
from humanbrain.simple_swe.freeze import (
    FreezeIntegrityError,
    _load_manifest,
    create_freeze_manifest,
)
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import (
    _generate_partition,
    canonical_json_bytes,
    canonical_sha256,
)


PROTOCOL = (
    Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"
)


def _ledger(path: Path) -> ArtifactLedger:
    return ArtifactLedger(path, protocol_sha256=load_protocol(PROTOCOL).sha256)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-c", "core.excludesFile=", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def _frozen_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    source = root / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "simple-swe@example.invalid")
    _git(source, "config", "user.name", "Simple SWE Test")
    (source / "brain.py").write_text("CELL_ONLY = True\n", encoding="utf-8")
    installed_source = source / "src" / "humanbrain" / "execution.py"
    installed_source.parent.mkdir(parents=True)
    installed_source.write_text("CELL_EXECUTION = True\n", encoding="utf-8")
    _git(source, "add", "brain.py", "src/humanbrain/execution.py")
    _git(source, "commit", "-q", "-m", "frozen source")
    monkeypatch.setattr(
        freeze,
        "_installed_execution_source_paths",
        lambda: (installed_source.resolve(),),
    )
    model = CellLanguageModel.grow(["def identity(value):\n    return value\n"])
    checkpoint = root / "checkpoint.npz"
    model.save(checkpoint)
    manifest = root / "freeze.json"
    create_freeze_manifest(
        protocol_path=PROTOCOL,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=manifest,
    )
    return source, checkpoint, manifest


def _entropy_must_not_run(_count: int) -> bytes:
    raise AssertionError("eval entropy was requested before every precondition passed")


def test_reveal_rejects_hidden_bundle_inside_model_workspace_before_entropy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(reveal.secrets, "token_bytes", _entropy_must_not_run)

    with pytest.raises(FreezeIntegrityError, match="private eval bundle"):
        reveal.reveal_evaluation(
            manifest_path=manifest,
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
            private_bundle_path=source / "hidden-tests.json",
            ledger=_ledger(tmp_path / "public-ledger"),
        )


def test_reveal_revalidates_clean_source_before_entropy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)
    (source / "uncommitted.txt").write_text("drift", encoding="utf-8")
    monkeypatch.setattr(reveal.secrets, "token_bytes", _entropy_must_not_run)

    with pytest.raises(FreezeIntegrityError, match="source tree is dirty"):
        reveal.reveal_evaluation(
            manifest_path=manifest,
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
            private_bundle_path=tmp_path / "private" / "hidden-tests.json",
            ledger=_ledger(tmp_path / "public-ledger"),
        )


def test_reveal_rejects_public_ledger_inside_model_workspace_before_entropy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(reveal.secrets, "token_bytes", _entropy_must_not_run)

    with pytest.raises(FreezeIntegrityError, match="artifact ledger"):
        reveal.reveal_evaluation(
            manifest_path=manifest,
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
            private_bundle_path=tmp_path / "private" / "hidden-tests.json",
            ledger=_ledger(source / "public-ledger"),
        )


def test_validly_rehashed_task_substitution_cannot_match_the_revealed_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _checkpoint, manifest_path = _frozen_fixture(tmp_path, monkeypatch)
    protocol = load_protocol(PROTOCOL)
    manifest = _load_manifest(manifest_path)
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    synthetic_seed = sha256(b"simple-swe-synthetic-seed-regression").digest()
    tasks = _generate_partition(
        protocol,
        "eval",
        int.from_bytes(synthetic_seed, byteorder="big", signed=False),
    )
    payload = reveal._bundle_payload(
        protocol, manifest, manifest_sha256, synthetic_seed, tasks
    )
    substituted_public = payload["tasks"][0]["public"]
    substituted_private = payload["tasks"][0]["private"]
    substituted_public["prompt"] += "\nReturn the constant zero."
    public_record = dict(substituted_public)
    public_record.pop("task_sha256")
    substituted_public["task_sha256"] = canonical_sha256(public_record)
    substituted_private["public_task_sha256"] = substituted_public["task_sha256"]
    bundle_path = tmp_path / "synthetic-substituted-bundle.json"
    bundle_path.write_bytes(canonical_json_bytes(payload))
    bundle = reveal.load_evaluation_bundle(bundle_path)

    with pytest.raises(FreezeIntegrityError, match="deterministic eval seed"):
        reveal._validate_bundle_against_freeze(
            bundle, protocol, manifest, manifest_sha256
        )


def test_private_bundle_cannot_predate_the_public_run_start_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest_path = _frozen_fixture(tmp_path, monkeypatch)
    protocol = load_protocol(PROTOCOL)
    manifest = _load_manifest(manifest_path)
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    synthetic_seed = sha256(b"simple-swe-preexisting-bundle-regression").digest()
    tasks = _generate_partition(
        protocol,
        "eval",
        int.from_bytes(synthetic_seed, byteorder="big", signed=False),
    )
    bundle_path = tmp_path / "synthetic-preexisting-bundle.json"
    bundle_path.write_bytes(
        canonical_json_bytes(
            reveal._bundle_payload(
                protocol, manifest, manifest_sha256, synthetic_seed, tasks
            )
        )
    )

    with pytest.raises(FreezeIntegrityError, match="predates.*run-start"):
        reveal.reveal_evaluation(
            manifest_path=manifest_path,
            protocol_path=PROTOCOL,
            checkpoint_path=checkpoint,
            source_root=source,
            private_bundle_path=bundle_path,
            ledger=_ledger(tmp_path / "public-ledger"),
        )
