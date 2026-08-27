"""Immutable authority snapshot for the frozen calibration-v3 plan.

The official runner remains the sole production producer of this record.
Keeping the snapshot here lets future execution modules consume it without
importing runner control flow, native code, persistence, or protocol builders.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MAX_PROTOCOL_BYTES = 1 << 20


class FrozenPlanAuthorityError(ValueError):
    """The proposed authority cannot be represented by one exact snapshot."""


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_json_tree(value: object, *, depth: int = 0) -> None:
    if depth > 128:
        raise FrozenPlanAuthorityError("authority protocol is too deeply nested")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is list:
        for child in value:
            _validate_json_tree(child, depth=depth + 1)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise FrozenPlanAuthorityError("authority protocol keys must be exact strings")
        for child in value.values():
            _validate_json_tree(child, depth=depth + 1)
        return
    raise FrozenPlanAuthorityError("authority protocol is not exact JSON data")


def _canonical_protocol_snapshot(value: object) -> bytes:
    if type(value) is not dict:
        raise FrozenPlanAuthorityError("authority protocol must be an exact object")
    try:
        _validate_json_tree(value)
        snapshot = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except FrozenPlanAuthorityError:
        raise
    except (RecursionError, RuntimeError, TypeError, ValueError) as exc:
        raise FrozenPlanAuthorityError(
            "authority protocol changed or failed canonical encoding"
        ) from exc
    if len(snapshot) > _MAX_PROTOCOL_BYTES:
        raise FrozenPlanAuthorityError("authority protocol exceeds the one-mebibyte bound")
    return snapshot


@dataclass(frozen=True, slots=True, init=False)
class FrozenPlanAuthority:
    """One immutable protocol snapshot and its external source identity."""

    repository_root: Path
    source_bundle_identity_sha256: str
    _canonical_protocol_bytes: bytes = field(repr=False)
    _protocol_sha256: str = field(repr=False)

    def __init__(
        self,
        *,
        repository_root: Path,
        protocol: dict[str, Any],
        protocol_sha256: str,
        source_bundle_identity_sha256: str,
    ) -> None:
        if not isinstance(repository_root, Path):
            raise FrozenPlanAuthorityError("authority repository root must be a pathlib.Path")
        if not _is_lower_sha256(protocol_sha256):
            raise FrozenPlanAuthorityError(
                "authority protocol digest must be one lowercase SHA-256"
            )
        if not _is_lower_sha256(source_bundle_identity_sha256):
            raise FrozenPlanAuthorityError(
                "authority source-bundle digest must be one lowercase SHA-256"
            )

        snapshot = _canonical_protocol_snapshot(protocol)
        snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
        if protocol_sha256 != snapshot_sha256:
            raise FrozenPlanAuthorityError(
                "authority protocol digest does not match its canonical snapshot"
            )

        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(
            self,
            "source_bundle_identity_sha256",
            source_bundle_identity_sha256,
        )
        object.__setattr__(self, "_canonical_protocol_bytes", snapshot)
        object.__setattr__(self, "_protocol_sha256", snapshot_sha256)

    @property
    def protocol(self) -> dict[str, Any]:
        """Return a fresh mutable transport copy of the immutable snapshot."""

        value = json.loads(self._canonical_protocol_bytes)
        if type(value) is not dict:  # pragma: no cover - constructor proves this.
            raise RuntimeError("authority protocol snapshot is not an object")
        return value

    @property
    def protocol_sha256(self) -> str:
        """Digest of the exact bytes from which ``protocol`` is decoded."""

        return self._protocol_sha256


__all__ = ["FrozenPlanAuthority", "FrozenPlanAuthorityError"]
