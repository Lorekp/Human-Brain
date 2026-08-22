from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from humanbrain.simple_swe import artifacts
from humanbrain.simple_swe.artifacts import (
    ArtifactIntegrityError,
    ArtifactLedger,
    atomic_create_canonical_json,
)
from humanbrain.simple_swe.tasks import canonical_json_bytes


def test_ledger_appends_hash_linked_canonical_events_without_overwrite(
    tmp_path: Path,
) -> None:
    ledger = ArtifactLedger(tmp_path / "run", protocol_sha256="a" * 64)

    first = ledger.append("run-start", {"freeze_sha256": "b" * 64})
    second = ledger.append(
        "cell-completion", {"task_id": "task-0", "source_sha256": "c" * 64}
    )
    events = ledger.verify()

    assert [event.kind for event in events] == ["run-start", "cell-completion"]
    assert first.sequence == 0
    assert second.sequence == 1
    assert second.previous_sha256 == first.file_sha256
    assert sorted(path.name for path in (tmp_path / "run").glob("*.json")) == [
        "000000-run-start.json",
        "000001-cell-completion.json",
    ]
    persisted = json.loads(second.path.read_text(encoding="utf-8"))
    assert persisted["body_sha256"] == second.body_sha256
    with pytest.raises(FileExistsError):
        ledger.append_at(1, "cell-completion", {"task_id": "replacement"})


def test_ledger_rejects_tampering_and_protocol_mixing(tmp_path: Path) -> None:
    root = tmp_path / "run"
    ledger = ArtifactLedger(root, protocol_sha256="a" * 64)
    event = ledger.append("run-start", {"freeze_sha256": "b" * 64})
    payload = json.loads(event.path.read_text(encoding="utf-8"))
    payload["body"]["freeze_sha256"] = "changed"
    event.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="canonical|hash"):
        ledger.verify()
    with pytest.raises(ArtifactIntegrityError, match="protocol"):
        ArtifactLedger(root, protocol_sha256="d" * 64).verify()


def test_ledger_requires_contiguous_sequence_and_valid_event_kind(tmp_path: Path) -> None:
    ledger = ArtifactLedger(tmp_path / "run", protocol_sha256="a" * 64)

    with pytest.raises(ValueError, match="event kind"):
        ledger.append("Cell Completion", {})
    with pytest.raises(ValueError, match="next sequence"):
        ledger.append_at(2, "cell-completion", {})


def test_canonical_json_publication_persists_a_verifiable_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.json"
    payload = {"protocol": "v2", "rows": [3, 1, 2]}

    publication = atomic_create_canonical_json(
        path,
        kind="procedural-address-space-evidence",
        payload=payload,
    )
    record = json.loads(path.read_bytes())

    assert path.read_bytes() == canonical_json_bytes(record)
    assert record == {
        "kind": "procedural-address-space-evidence",
        "payload": payload,
        "payload_sha256": publication.payload_sha256,
        "schema": "humanbrain.canonical-json-artifact.v1",
    }
    assert publication.path == path
    assert publication.file_sha256 == sha256(path.read_bytes()).hexdigest()


def test_canonical_json_publication_interruption_leaves_no_partial_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence.json"

    def interrupt_before_publish(source: object, target: object) -> None:
        assert Path(source).parent == Path(target).parent == tmp_path
        raise OSError("simulated interruption")

    monkeypatch.setattr(artifacts.os, "link", interrupt_before_publish)

    with pytest.raises(OSError, match="simulated interruption"):
        atomic_create_canonical_json(path, kind="evidence", payload={"ok": True})

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_canonical_json_publication_never_overwrites_an_existing_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.json"
    path.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="immutable artifact already exists"):
        atomic_create_canonical_json(path, kind="evidence", payload={"ok": True})

    assert path.read_bytes() == b"sentinel"
