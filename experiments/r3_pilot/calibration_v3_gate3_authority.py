"""Exact nested Gate3 source authority for calibration-v3 consumers.

This module is pure transport and validation.  It does not import the native
runtime, construct artifact parents, or execute any scientific fixture.  The
caller supplies the two live source computations; they are invoked only after
the already-bound Gate3 document has been read and validated completely.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

_MAX_GATE3_PROTOCOL_BYTES: Final = 1 << 20
_EXACT_SOURCE_MAXIMUM_FILE_BYTES: Final = 1 << 20
_EXACT_SOURCE_MAXIMUM_AGGREGATE_BYTES: Final = 32 << 20
_EXACT_SOURCE_FRAMING: Final = (
    "humanbrain-procedural-runtime-source-v1: ordered UTF-8 path length/path "
    "plus uint64 byte length/raw bytes"
)


class Gate3AuthorityError(RuntimeError):
    """The bound Gate3 document or its live source closure is invalid."""


@dataclass(frozen=True, slots=True)
class Gate3SourceAuthority:
    """The exact live procedural source closure approved by nested Gate3."""

    source_manifest: tuple[dict[str, str | int], ...]
    source_bundle_sha256: str


def _canonical_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Gate3AuthorityError(f"{label} is not a lowercase SHA-256 identity")
    return value


def _canonical_git_commit(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Gate3AuthorityError(f"{label} is not a lowercase Git commit identity")
    return value


def _canonical_source_bytes(raw: bytes) -> bytes:
    """Apply the repository's exact checkout-byte hashing rule."""

    without_crlf = raw.replace(b"\r\n", b"\n")
    if b"\r" in without_crlf:
        raise Gate3AuthorityError("governing Gate3 canonical source has a lone CR byte")
    return without_crlf


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise Gate3AuthorityError(f"governing Gate3 has duplicate JSON key: {key}")
        value[key] = member
    return value


def _reject_json_constant(value: str) -> None:
    raise Gate3AuthorityError(f"governing Gate3 has nonstandard JSON constant: {value}")


def _path_is_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(reparse_flag and getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _read_bounded_plain_file(path: Path) -> bytes:
    def validate(metadata: os.stat_result, subject: str) -> tuple[int, int, int]:
        if stat.S_ISLNK(metadata.st_mode) or _path_is_reparse(metadata):
            raise Gate3AuthorityError(f"governing Gate3 {subject} is a symlink or reparse point")
        if not stat.S_ISREG(metadata.st_mode):
            raise Gate3AuthorityError(f"governing Gate3 {subject} is not a regular file")
        size = int(metadata.st_size)
        if not 1 <= size <= _MAX_GATE3_PROTOCOL_BYTES:
            raise Gate3AuthorityError("governing Gate3 file size is outside its bound")
        return int(metadata.st_dev), int(metadata.st_ino), size

    try:
        before = validate(os.lstat(path), "path")
        with path.open("rb") as stream:
            descriptor = validate(os.fstat(stream.fileno()), "descriptor")
            if descriptor != before:
                raise Gate3AuthorityError("governing Gate3 path changed before its bounded read")
            payload = stream.read(_MAX_GATE3_PROTOCOL_BYTES + 1)
        after = validate(os.lstat(path), "path")
    except Gate3AuthorityError:
        raise
    except OSError as exc:
        raise Gate3AuthorityError("governing Gate3 file is unavailable") from exc
    if after != before or len(payload) != before[2]:
        raise Gate3AuthorityError("governing Gate3 path changed during its bounded read")
    return payload


def _validated_manifest_records(
    label: str,
    value: object,
    *,
    container_type: type[list[Any]] | type[tuple[Any, ...]],
) -> tuple[dict[str, str | int], ...]:
    if type(value) is not container_type or not value:
        raise Gate3AuthorityError(f"{label} manifest is not a nonempty exact sequence")
    records: list[dict[str, str | int]] = []
    seen_paths: set[str] = set()
    aggregate_bytes = 0
    for index, raw_record in enumerate(value):
        if type(raw_record) is not dict or set(raw_record) != {
            "path",
            "size",
            "sha256",
        }:
            raise Gate3AuthorityError(f"{label} manifest row {index} shape is not exact")
        relative_path = raw_record["path"]
        size = raw_record["size"]
        digest = raw_record["sha256"]
        if (
            type(relative_path) is not str
            or not relative_path
            or "\\" in relative_path
            or ":" in relative_path
            or relative_path.startswith("/")
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
            or relative_path in seen_paths
        ):
            raise Gate3AuthorityError(f"{label} manifest path {index} is not canonical")
        if type(size) is not int or not 0 <= size <= _EXACT_SOURCE_MAXIMUM_FILE_BYTES:
            raise Gate3AuthorityError(f"{label} manifest row {index} size is invalid")
        canonical_digest = _canonical_sha256(f"{label} manifest row {index} digest", digest)
        seen_paths.add(relative_path)
        aggregate_bytes += size
        if aggregate_bytes > _EXACT_SOURCE_MAXIMUM_AGGREGATE_BYTES:
            raise Gate3AuthorityError(f"{label} manifest exceeds its aggregate limit")
        records.append({"path": relative_path, "size": size, "sha256": canonical_digest})
    return tuple(records)


def _bound_gate3_path(repository_root: Path, source_binding: object) -> tuple[Path, str]:
    if type(source_binding) is not dict:
        raise Gate3AuthorityError("source binding is not an exact object")
    relative_path = source_binding.get("governing_gate3_path")
    files = source_binding.get("files")
    if type(relative_path) is not str or not relative_path:
        raise Gate3AuthorityError("governing Gate3 path is missing")
    if type(files) is not dict or relative_path not in files:
        raise Gate3AuthorityError("governing Gate3 path is not source-bound")
    row = files[relative_path]
    if type(row) is not dict or set(row) != {"commit", "sha256"}:
        raise Gate3AuthorityError("governing Gate3 source-binding row shape is not exact")
    _canonical_git_commit("governing Gate3 source-binding commit", row["commit"])
    bound_digest = _canonical_sha256("governing Gate3 source-binding digest", row["sha256"])
    parts = relative_path.split("/")
    if (
        "\\" in relative_path
        or ":" in relative_path
        or relative_path.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise Gate3AuthorityError("governing Gate3 path is not canonical")
    try:
        root = repository_root.resolve()
        path = root.joinpath(*parts)
        path.resolve().relative_to(root)
    except (OSError, ValueError) as exc:
        raise Gate3AuthorityError(
            "governing Gate3 path escapes or cannot resolve within the repository"
        ) from exc
    return path, bound_digest


def _declared_source_authority(
    repository_root: Path,
    source_binding: object,
) -> tuple[tuple[dict[str, str | int], ...], str, int]:
    path, bound_digest = _bound_gate3_path(repository_root, source_binding)
    payload = _read_bounded_plain_file(path)
    observed_digest = hashlib.sha256(_canonical_source_bytes(payload)).hexdigest()
    if observed_digest != bound_digest:
        raise Gate3AuthorityError("governing Gate3 bytes do not match their outer source binding")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except Gate3AuthorityError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Gate3AuthorityError("governing Gate3 JSON is not strict UTF-8 JSON") from exc
    if type(document) is not dict:
        raise Gate3AuthorityError("governing Gate3 document is not an exact object")
    nested = document.get("source_binding")
    if type(nested) is not dict or set(nested) != {
        "framing",
        "files",
        "bundle_sha256",
        "limits",
    }:
        raise Gate3AuthorityError("governing Gate3 source_binding shape is not exact")
    if nested["framing"] != _EXACT_SOURCE_FRAMING:
        raise Gate3AuthorityError("governing Gate3 source framing is not exact")
    limits = nested["limits"]
    if type(limits) is not dict or set(limits) != {
        "maximum_file_bytes",
        "maximum_aggregate_bytes",
        "actual_aggregate_bytes",
    }:
        raise Gate3AuthorityError("governing Gate3 source limits shape is not exact")
    maximum_file_bytes = limits["maximum_file_bytes"]
    maximum_aggregate_bytes = limits["maximum_aggregate_bytes"]
    actual_aggregate_bytes = limits["actual_aggregate_bytes"]
    if (
        type(maximum_file_bytes) is not int
        or maximum_file_bytes != _EXACT_SOURCE_MAXIMUM_FILE_BYTES
        or type(maximum_aggregate_bytes) is not int
        or maximum_aggregate_bytes != _EXACT_SOURCE_MAXIMUM_AGGREGATE_BYTES
        or type(actual_aggregate_bytes) is not int
        or not 0 < actual_aggregate_bytes <= maximum_aggregate_bytes
    ):
        raise Gate3AuthorityError("governing Gate3 source limits are invalid")
    declared_manifest = _validated_manifest_records("Gate3", nested["files"], container_type=list)
    if sum(int(record["size"]) for record in declared_manifest) != actual_aggregate_bytes:
        raise Gate3AuthorityError("governing Gate3 aggregate bytes do not match its manifest")
    declared_digest = _canonical_sha256("Gate3 source-bundle identity", nested["bundle_sha256"])
    return declared_manifest, declared_digest, actual_aggregate_bytes


def authorize_gate3_source(
    repository_root: Path,
    source_binding: object,
    *,
    live_manifest: Callable[[Path], object],
    live_bundle_sha256: Callable[[Path], object],
) -> Gate3SourceAuthority:
    """Prove the exact nested manifest and digest against the live closure."""

    declared_manifest, declared_digest, actual_aggregate_bytes = _declared_source_authority(
        repository_root, source_binding
    )
    observed_manifest = _validated_manifest_records(
        "live source", live_manifest(repository_root), container_type=tuple
    )
    if observed_manifest != declared_manifest:
        raise Gate3AuthorityError("Gate3 source manifest does not match the live source manifest")
    if sum(int(record["size"]) for record in observed_manifest) != actual_aggregate_bytes:
        raise Gate3AuthorityError("live source aggregate bytes do not match the Gate3 authority")
    observed_digest = _canonical_sha256(
        "live source-bundle identity", live_bundle_sha256(repository_root)
    )
    if observed_digest != declared_digest:
        raise Gate3AuthorityError(
            "Gate3-approved identity does not match the live source-bundle identity"
        )
    return Gate3SourceAuthority(
        source_manifest=observed_manifest,
        source_bundle_sha256=observed_digest,
    )
