"""Bounded raw workspace transport for neuron-selected byte splices.

This module indexes and applies an explicit proposal. It receives no request,
teacher span, expected output, semantic roles or occurrence-selection policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath, PureWindowsPath

from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile

MAX_FILES = 4
MAX_FILE_BYTES = 128
MAX_PATH_BYTES = 31
MAX_REQUEST_BYTES = 31
MAX_REPLACEMENT_BYTES = 16


def _file_digest(files: tuple[OpaqueWorkspaceFile, ...]) -> str:
    digest = sha256(b"humanbrain.raw-splice-workspace.v1\0")
    digest.update(len(files).to_bytes(4, "big"))
    for item in files:
        for raw in (item.path.encode("utf-8"), item.content):
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _check_path(path: str) -> None:
    # OpaqueWorkspaceFile already checks canonical relative POSIX spelling.
    # These additional restrictions preserve that spelling on the Windows host.
    if (
        len(path.encode("utf-8")) > MAX_PATH_BYTES
        or any(ord(char) < 32 or char in '\\:<>"|?*' for char in path)
        or PureWindowsPath(path).is_reserved()
        or any(part.endswith((".", " ")) for part in PurePosixPath(path).parts)
    ):
        raise ValueError("workspace path exceeds its bound or portable spelling")


@dataclass(frozen=True)
class SpliceWorkspace:
    """Immutable raw observation, canonically indexed without semantic filters."""

    files: tuple[OpaqueWorkspaceFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.files, tuple) or not 1 <= len(self.files) <= MAX_FILES:
            raise ValueError("workspace must contain one to four immutable files")
        paths = set()
        for item in self.files:
            if not isinstance(item, OpaqueWorkspaceFile):
                raise TypeError("workspace entries must be OpaqueWorkspaceFile values")
            _check_path(item.path)
            if len(item.content) > MAX_FILE_BYTES:
                raise ValueError("workspace file exceeds 128 raw bytes")
            identity = item.path.casefold()
            if identity in paths:
                raise ValueError("workspace paths must be distinct on the execution host")
            paths.add(identity)
        ordered = tuple(sorted(self.files, key=lambda item: item.path.encode("utf-8")))
        object.__setattr__(self, "files", ordered)

    @property
    def sha256(self) -> str:
        return _file_digest(self.files)

    @property
    def boundaries(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (item.path, offset)
            for item in self.files
            for offset in range(len(item.content) + 1)
        )


@dataclass(frozen=True)
class SpliceProposal:
    """Actual output choices, including tentative fields when operation is STOP."""

    operation: str
    path: str
    start: int
    end: int
    replacement: bytes
    stop_reason: str
    workspace_sha256: str

    def __post_init__(self) -> None:
        if self.operation not in ("STOP", "SPLICE"):
            raise ValueError("operation must be the explicit STOP or SPLICE output")
        if (
            not isinstance(self.path, str)
            or type(self.start) is not int or type(self.end) is not int
        ):
            raise TypeError("proposal needs a path and integer byte offsets")
        if not isinstance(self.replacement, bytes) or len(self.replacement) > MAX_REPLACEMENT_BYTES:
            raise ValueError("proposal replacement must contain at most 16 raw bytes")
        if self.stop_reason not in ("eos", "limit"):
            raise ValueError("payload status must preserve its actual EOS or limit")
        if (
            not isinstance(self.workspace_sha256, str) or len(self.workspace_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.workspace_sha256)
        ):
            raise ValueError("proposal must bind an exact workspace digest")

    def to_record(self) -> dict:
        return {
            "operation": self.operation, "path": self.path,
            "start": self.start, "end": self.end,
            "replacement_hex": self.replacement.hex(), "stop_reason": self.stop_reason,
            "workspace_sha256": self.workspace_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return sha256(b"humanbrain.raw-splice-proposal.v1\0" + encoded.encode("utf-8")).hexdigest()


def apply_splice(
    workspace: SpliceWorkspace, proposal: SpliceProposal,
) -> tuple[OpaqueWorkspaceFile, ...]:
    """Apply exactly the selected bytes once, or leave an explicit STOP unchanged.

Invalid SPLICE proposals raise instead of selecting another edit. Resulting
files may grow by at most 16 bytes; a subsequent observation must independently
meet the input geometry. Payload bytes receive no Unicode or syntax repair.
"""
    if not isinstance(workspace, SpliceWorkspace) or not isinstance(proposal, SpliceProposal):
        raise TypeError("apply_splice requires the explicit workspace and proposal")
    if proposal.workspace_sha256 != workspace.sha256:
        raise ValueError("proposal refers to a stale or different workspace")
    if proposal.operation == "STOP":
        return workspace.files
    if proposal.stop_reason != "eos":
        raise ValueError("SPLICE requires an actual payload EOS")
    selected = next((item for item in workspace.files if item.path == proposal.path), None)
    if selected is None or not 0 <= proposal.start <= proposal.end <= len(selected.content):
        raise ValueError("selected file or raw span is outside the workspace")
    replacement = OpaqueWorkspaceFile(
        selected.path,
        selected.content[:proposal.start] + proposal.replacement + selected.content[proposal.end:],
    )
    return tuple(replacement if item.path == selected.path else item for item in workspace.files)
