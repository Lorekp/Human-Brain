"""Larger raw observations and exact byte-splice transport for sequence assemblies.

Observation framing preserves raw public fields. Exact actuation receives only
the workspace and proposal, without teacher spans, grades or semantic routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath, PureWindowsPath

from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile

MAX_FILES = 4
MAX_FILE_BYTES = 512
MAX_PATH_BYTES = 31
MAX_REQUEST_BYTES = 256
MAX_REPLACEMENT_BYTES = 128
MAX_STREAM_BYTES = 512
EMPTY_SYMBOL = -1
EOS_SYMBOL = 256
REQUEST_SLOTS = MAX_REQUEST_BYTES + 1
PATH_SLOTS = MAX_PATH_BYTES + 1
FILE_SLOTS = MAX_FILE_BYTES + 1
STREAM_SLOTS = MAX_STREAM_BYTES + 1
RETURNCODE_SLOTS = 8
QUERY_SYMBOLS = REQUEST_SLOTS + 2 * STREAM_SLOTS + RETURNCODE_SLOTS
SENSOR_SYMBOLS = QUERY_SYMBOLS + MAX_FILES * (PATH_SLOTS + FILE_SLOTS)
QUERY_FLAG_NAMES = (
    "feedback_present",
    "returncode_present",
    "accepted",
    "timeout",
    "capture_complete",
    "files_complete",
    "output_limit_exceeded",
)
SENSOR_FLAG_NAMES = (
    "workspace_present",
    "file_0_present",
    "file_1_present",
    "file_2_present",
    "file_3_present",
    *QUERY_FLAG_NAMES,
)


def _symbols(raw: bytes, capacity: int) -> tuple[int, ...]:
    """Raw bytes then EOS; absent padding is distinct from every input symbol."""
    return (*raw, EOS_SYMBOL, *((EMPTY_SYMBOL,) * (capacity - len(raw))))


@dataclass(frozen=True)
class SensorFrame:
    """Fixed symbol/flag fields; encoding into neuronal rails is the caller's job.

    Symbols use -1 for absent padding, 0..255 for actual bytes and 256 for EOS.
    No hash, request interpretation or semantic category enters this frame.
    """

    symbols: tuple[int, ...]
    flags: tuple[bool, ...]


@dataclass(frozen=True)
class ToolFeedback:
    """Actual public tool result, excluding grade and desired behavior."""

    returncode: int | None
    stdout: bytes
    stderr: bytes
    accepted: bool
    timeout: bool
    capture_complete: bool
    files_complete: bool
    output_limit_exceeded: bool

    def __post_init__(self) -> None:
        if self.returncode is not None and (
            type(self.returncode) is not int or not -(1 << 63) <= self.returncode < (1 << 63)
        ):
            raise ValueError("returncode must be a signed64 integer or absent")
        for raw in (self.stdout, self.stderr):
            if not isinstance(raw, bytes) or len(raw) > MAX_STREAM_BYTES:
                raise ValueError("each actual diagnostic stream must fit 512 raw bytes")
        for name in QUERY_FLAG_NAMES[2:]:
            if type(getattr(self, name)) is not bool:
                raise TypeError("tool outcome flags must be booleans")

    def to_record(self) -> dict:
        return {
            "returncode": self.returncode,
            "stdout_hex": self.stdout.hex(),
            "stderr_hex": self.stderr.hex(),
            **{name: getattr(self, name) for name in QUERY_FLAG_NAMES[2:]},
        }


@dataclass(frozen=True)
class RawObservation:
    """Whole public request, all files and the latest actual diagnostic result.

    None workspace represents reading without files, never a dropped oversized
    workspace. None feedback means no result; an empty stream is still present.
    Source/attempt custody belongs to the journal and is not neural input.
    """

    request: bytes
    workspace: SpliceWorkspace | None
    feedback: ToolFeedback | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, bytes) or len(self.request) > MAX_REQUEST_BYTES:
            raise ValueError("request must contain at most 256 raw bytes")
        if self.workspace is not None and not isinstance(self.workspace, SpliceWorkspace):
            raise TypeError("observation workspace must be an explicit bounded workspace or None")
        if self.feedback is not None and not isinstance(self.feedback, ToolFeedback):
            raise TypeError("feedback must contain only actual bounded tool fields")

    def query_frame(self) -> SensorFrame:
        """Fixed request257/stdout513/stderr513/returncode8 symbols and seven flags.

        Signed little-endian64 preserves negative and Windows unsigned32 exit
        codes. An absent returncode has eight absent symbols and a separate flag.
        """
        request = _symbols(self.request, MAX_REQUEST_BYTES)
        if self.feedback is None:
            return SensorFrame(
                request + (EMPTY_SYMBOL,) * (2 * STREAM_SLOTS + RETURNCODE_SLOTS),
                (False,) * len(QUERY_FLAG_NAMES),
            )
        feedback = self.feedback
        code = (
            tuple(feedback.returncode.to_bytes(8, "little", signed=True))
            if feedback.returncode is not None
            else (EMPTY_SYMBOL,) * RETURNCODE_SLOTS
        )
        return SensorFrame(
            request
            + _symbols(feedback.stdout, MAX_STREAM_BYTES)
            + _symbols(feedback.stderr, MAX_STREAM_BYTES)
            + code,
            (
                True,
                feedback.returncode is not None,
                *(getattr(feedback, name) for name in QUERY_FLAG_NAMES[2:]),
            ),
        )

    def sensor_frame(self) -> SensorFrame:
        """Query frame followed by four fixed path32/content513 file populations.

        Every file appears in the workspace's raw UTF-8 sorted order, including
        files that do not contain any relevant answer or edit location.
        """
        query = self.query_frame()
        items = self.workspace.files if self.workspace is not None else ()
        symbols = list(query.symbols)
        for index in range(MAX_FILES):
            if index < len(items):
                item = items[index]
                symbols.extend(_symbols(item.path.encode("utf-8"), MAX_PATH_BYTES))
                symbols.extend(_symbols(item.content, MAX_FILE_BYTES))
            else:
                symbols.extend((EMPTY_SYMBOL,) * (PATH_SLOTS + FILE_SLOTS))
        return SensorFrame(
            tuple(symbols),
            (self.workspace is not None, *(i < len(items) for i in range(MAX_FILES)), *query.flags),
        )

    def to_record(self) -> dict:
        return {
            "schema": "sequence-assembly-raw-observation-v1",
            "request_hex": self.request.hex(),
            "files": None
            if self.workspace is None
            else [
                {"path": item.path, "content_hex": item.content.hex()}
                for item in self.workspace.files
            ],
            "feedback": None if self.feedback is None else self.feedback.to_record(),
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":")).encode()
        return sha256(b"humanbrain.sequence-assembly-observation.v1\0" + raw).hexdigest()


def _file_digest(files: tuple[OpaqueWorkspaceFile, ...]) -> str:
    digest = sha256(b"humanbrain.sequence-assembly-workspace.v1\0")
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
        or any(
            PureWindowsPath(part).is_reserved() or part.endswith((".", " "))
            for part in PurePosixPath(path).parts
        )
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
                raise ValueError("workspace file exceeds 512 raw bytes")
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
            (item.path, offset) for item in self.files for offset in range(len(item.content) + 1)
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
            or type(self.start) is not int
            or type(self.end) is not int
        ):
            raise TypeError("proposal needs a path and integer byte offsets")
        if not isinstance(self.replacement, bytes) or len(self.replacement) > MAX_REPLACEMENT_BYTES:
            raise ValueError("proposal replacement must contain at most 128 raw bytes")
        if self.stop_reason not in ("eos", "limit"):
            raise ValueError("payload status must preserve its actual EOS or limit")
        if (
            not isinstance(self.workspace_sha256, str)
            or len(self.workspace_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.workspace_sha256)
        ):
            raise ValueError("proposal must bind an exact workspace digest")

    def to_record(self) -> dict:
        return {
            "operation": self.operation,
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "replacement_hex": self.replacement.hex(),
            "stop_reason": self.stop_reason,
            "workspace_sha256": self.workspace_sha256,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return sha256(
            b"humanbrain.sequence-assembly-proposal.v1\0" + encoded.encode("utf-8")
        ).hexdigest()


def apply_splice(
    workspace: SpliceWorkspace,
    proposal: SpliceProposal,
) -> tuple[OpaqueWorkspaceFile, ...]:
    """Apply exactly the selected bytes once, or leave an explicit STOP unchanged.

    Invalid SPLICE proposals raise instead of selecting another edit. Resulting
    files may grow by at most 128 bytes; a subsequent observation must independently
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
        selected.content[: proposal.start]
        + proposal.replacement
        + selected.content[proposal.end :],
    )
    return tuple(replacement if item.path == selected.path else item for item in workspace.files)
