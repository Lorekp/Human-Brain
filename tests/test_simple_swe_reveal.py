from __future__ import annotations

import inspect
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest
from humanbrain.language import CellLanguageModel
from humanbrain.simple_swe import freeze, reveal
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

PROTOCOL = Path(__file__).parents[1] / "benchmarks" / "simple_swe_protocol_v1.json"


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
    checkpoint = root / "checkpoint.npz"
    CellLanguageModel.grow(["def identity(value):\n    return value\n"]).save(
        checkpoint
    )
    manifest = root / "freeze.json"
    create_freeze_manifest(
        protocol_path=PROTOCOL,
        checkpoint_path=checkpoint,
        source_root=source,
        output_path=manifest,
    )
    return source, checkpoint, manifest


def _arguments(source: Path, checkpoint: Path, manifest: Path) -> dict[str, Path]:
    return {
        "manifest_path": manifest,
        "protocol_path": PROTOCOL,
        "checkpoint_path": checkpoint,
        "source_root": source,
    }


def _entropy_must_not_run(_count: int) -> bytes:
    raise AssertionError("eval entropy was requested before every precondition passed")


def test_reservation_revalidates_clean_source_before_entropy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)
    (source / "uncommitted.txt").write_text("drift", encoding="utf-8")
    monkeypatch.setattr(reveal.secrets, "token_bytes", _entropy_must_not_run)

    with pytest.raises(FreezeIntegrityError, match="source tree is dirty"):
        reveal.reserve_evaluation_seed(**_arguments(source, checkpoint, manifest))


def test_seed_is_persisted_and_committed_before_any_task_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)
    fixed_seed = sha256(b"fixture immutable eval seed").digest()
    monkeypatch.setattr(reveal.secrets, "token_bytes", lambda _count: fixed_seed)
    monkeypatch.setattr(
        reveal,
        "_generate_partition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tasks generated during seed reservation")
        ),
    )

    reservation = reveal.reserve_evaluation_seed(
        **_arguments(source, checkpoint, manifest)
    )
    ledger = ArtifactLedger(
        reservation.paths.ledger_root,
        protocol_sha256=load_protocol(PROTOCOL).sha256,
    )

    assert reservation.eval_seed_sha256 == sha256(fixed_seed).hexdigest()
    assert reservation.paths.seed_reservation_path.is_file()
    assert not reservation.paths.bundle_path.exists()
    assert [event.kind for event in ledger.verify()] == [
        "run-start",
        "evaluation-seed-reserved",
    ]
    assert ledger.verify()[1].body["evaluation_tasks_generated"] is False
    assert ledger.verify()[0].body["sandbox_python_flags"] == ["-I", "-S", "-B"]
    assert ledger.verify()[0].body["production_dependencies_injected"] is False


def test_crash_after_seed_reservation_resumes_the_same_seed_without_reroll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)
    arguments = _arguments(source, checkpoint, manifest)
    fixed_seed = sha256(b"crash safe eval seed").digest()
    entropy_calls = 0

    def entropy(_count: int) -> bytes:
        nonlocal entropy_calls
        entropy_calls += 1
        return fixed_seed

    monkeypatch.setattr(reveal.secrets, "token_bytes", entropy)
    reservation = reveal.reserve_evaluation_seed(**arguments)
    reveal.record_external_anchor(
        **arguments,
        public_uri="https://example.invalid/simple-swe-anchor",
        receipt_sha256="a" * 64,
    )
    original_generate = reveal._generate_partition
    monkeypatch.setattr(
        reveal,
        "_generate_partition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic crash")),
    )

    with pytest.raises(RuntimeError, match="synthetic crash"):
        reveal.reveal_evaluation(**arguments)

    assert not reservation.paths.bundle_path.exists()
    monkeypatch.setattr(reveal.secrets, "token_bytes", _entropy_must_not_run)
    monkeypatch.setattr(reveal, "_generate_partition", original_generate)
    revealed = reveal.reveal_evaluation(**arguments)

    assert entropy_calls == 1
    assert revealed.eval_seed_sha256 == reservation.eval_seed_sha256
    assert revealed.task_count == 30
    ledger = ArtifactLedger(
        revealed.ledger_root, protocol_sha256=load_protocol(PROTOCOL).sha256
    )
    assert [event.kind for event in ledger.verify()] == [
        "run-start",
        "evaluation-seed-reserved",
        "external-anchor-recorded",
        "evaluation-revealed",
    ]


def test_reveal_stops_until_the_seed_commitment_is_externally_anchored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)
    arguments = _arguments(source, checkpoint, manifest)
    reservation = reveal.reserve_evaluation_seed(**arguments)

    with pytest.raises(FreezeIntegrityError, match="externally anchor"):
        reveal.reveal_evaluation(**arguments)

    assert not reservation.paths.bundle_path.exists()


def test_reveal_rejects_a_direct_anchor_without_a_public_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)
    arguments = _arguments(source, checkpoint, manifest)
    reservation = reveal.reserve_evaluation_seed(**arguments)
    ledger = ArtifactLedger(
        reservation.paths.ledger_root,
        protocol_sha256=load_protocol(PROTOCOL).sha256,
    )
    ledger.append(
        "external-anchor-recorded",
        {"anchor_payload_sha256": reservation.anchor_payload_sha256},
    )

    with pytest.raises(FreezeIntegrityError, match="anchor fields"):
        reveal.reveal_evaluation(**arguments)

    assert not reservation.paths.bundle_path.exists()


def test_external_anchor_requires_a_nonempty_https_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest = _frozen_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="public HTTPS URI"):
        reveal.record_external_anchor(
            **_arguments(source, checkpoint, manifest),
            public_uri="https://",
            receipt_sha256="a" * 64,
        )


def test_canonical_reveal_api_exposes_no_alternate_bundle_or_ledger_path() -> None:
    reserve_fields = set(inspect.signature(reveal.reserve_evaluation_seed).parameters)
    reveal_fields = set(inspect.signature(reveal.reveal_evaluation).parameters)

    assert reserve_fields == {
        "manifest_path",
        "protocol_path",
        "checkpoint_path",
        "source_root",
    }
    assert reveal_fields == reserve_fields
    assert "private_bundle_path" not in reveal_fields
    assert "ledger" not in reveal_fields


def test_preexisting_canonical_run_registry_is_rejected_before_entropy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, checkpoint, manifest_path = _frozen_fixture(tmp_path, monkeypatch)
    manifest = _load_manifest(manifest_path)
    paths = reveal.canonical_evaluation_paths(manifest_path, manifest)
    paths.run_root.mkdir(parents=True)
    monkeypatch.setattr(reveal.secrets, "token_bytes", _entropy_must_not_run)

    with pytest.raises(FreezeIntegrityError, match="predates run-start"):
        reveal.reserve_evaluation_seed(
            **_arguments(source, checkpoint, manifest_path)
        )


def test_validly_rehashed_task_substitution_cannot_match_the_reserved_seed(
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
        reveal.validate_evaluation_bundle(
            bundle, protocol, manifest, manifest_sha256
        )


def test_typed_signature_is_bound_by_both_task_commitments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _checkpoint, manifest_path = _frozen_fixture(tmp_path, monkeypatch)
    protocol = load_protocol(PROTOCOL)
    manifest = _load_manifest(manifest_path)
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    synthetic_seed = sha256(b"simple-swe-signature-commitment-regression").digest()
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
    substituted_public["signature"]["result"]["constructor"] = "String"
    public_record = dict(substituted_public)
    public_record.pop("task_sha256")
    substituted_public["task_sha256"] = canonical_sha256(public_record)
    substituted_private["public_task_sha256"] = substituted_public["task_sha256"]
    bundle_path = tmp_path / "synthetic-signature-substitution.json"
    bundle_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(FreezeIntegrityError, match="hidden-test commitment"):
        reveal.load_evaluation_bundle(bundle_path)
