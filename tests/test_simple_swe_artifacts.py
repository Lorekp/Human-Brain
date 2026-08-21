from __future__ import annotations

import json
from pathlib import Path

import pytest

from humanbrain.simple_swe.artifacts import ArtifactIntegrityError, ArtifactLedger


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
