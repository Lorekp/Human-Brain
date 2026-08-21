from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from humanbrain.simple_swe.tasks import canonical_json_bytes, canonical_sha256


SCHEMA = "humanbrain.simple-swe-artifact-event.v1"
GENESIS_SHA256 = "0" * 64
_KIND = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


class ArtifactIntegrityError(RuntimeError):
    """An immutable benchmark artifact chain is incomplete or altered."""


@dataclass(frozen=True)
class ArtifactEvent:
    sequence: int
    kind: str
    protocol_sha256: str
    previous_sha256: str
    body_sha256: str
    body: dict[str, Any]
    file_sha256: str
    path: Path


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be lowercase SHA-256 hex")


def atomic_create_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactLedger:
    """Append-only, hash-linked events written one atomic file at a time."""

    def __init__(self, root: str | Path, *, protocol_sha256: str) -> None:
        _validate_sha256(protocol_sha256, "protocol_sha256")
        self.root = Path(root).resolve()
        self.protocol_sha256 = protocol_sha256

    def verify(self) -> tuple[ArtifactEvent, ...]:
        if not self.root.exists():
            return ()
        paths = sorted(self.root.glob("*.json"))
        events: list[ArtifactEvent] = []
        previous = GENESIS_SHA256
        for sequence, path in enumerate(paths):
            raw = path.read_bytes()
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ArtifactIntegrityError(f"artifact is not JSON: {path.name}") from error
            if not isinstance(record, dict):
                raise ArtifactIntegrityError(f"artifact is not an object: {path.name}")
            if record.get("protocol_sha256") != self.protocol_sha256:
                raise ArtifactIntegrityError("artifact protocol commitment differs")
            if raw != canonical_json_bytes(record):
                raise ArtifactIntegrityError(f"artifact is not canonical: {path.name}")
            kind = record.get("kind")
            if not isinstance(kind, str) or not _KIND.fullmatch(kind):
                raise ArtifactIntegrityError(f"artifact event kind is invalid: {path.name}")
            expected_name = f"{sequence:06d}-{kind}.json"
            if path.name != expected_name or record.get("sequence") != sequence:
                raise ArtifactIntegrityError("artifact sequence is not contiguous")
            if record.get("schema") != SCHEMA:
                raise ArtifactIntegrityError("artifact schema differs")
            if record.get("previous_sha256") != previous:
                raise ArtifactIntegrityError("artifact previous hash differs")
            body = record.get("body")
            if not isinstance(body, dict):
                raise ArtifactIntegrityError("artifact body must be an object")
            body_sha256 = canonical_sha256(body)
            if record.get("body_sha256") != body_sha256:
                raise ArtifactIntegrityError("artifact body hash differs")
            file_sha256 = sha256(raw).hexdigest()
            event = ArtifactEvent(
                sequence=sequence,
                kind=kind,
                protocol_sha256=self.protocol_sha256,
                previous_sha256=previous,
                body_sha256=body_sha256,
                body=body,
                file_sha256=file_sha256,
                path=path,
            )
            events.append(event)
            previous = file_sha256
        return tuple(events)

    def append(self, kind: str, body: dict[str, Any]) -> ArtifactEvent:
        return self.append_at(len(self.verify()), kind, body)

    def append_at(
        self, sequence: int, kind: str, body: dict[str, Any]
    ) -> ArtifactEvent:
        if not _KIND.fullmatch(kind):
            raise ValueError("event kind must be lowercase kebab-case")
        path = self.root / f"{sequence:06d}-{kind}.json"
        if path.exists():
            raise FileExistsError(f"immutable artifact already exists: {path}")
        events = self.verify()
        if sequence != len(events):
            raise ValueError("event sequence must equal the next sequence")
        previous = events[-1].file_sha256 if events else GENESIS_SHA256
        record = {
            "schema": SCHEMA,
            "sequence": sequence,
            "kind": kind,
            "protocol_sha256": self.protocol_sha256,
            "previous_sha256": previous,
            "body_sha256": canonical_sha256(body),
            "body": body,
        }
        atomic_create_bytes(path, canonical_json_bytes(record))
        return self.verify()[-1]


__all__ = [
    "ArtifactEvent",
    "ArtifactIntegrityError",
    "ArtifactLedger",
    "atomic_create_bytes",
]
