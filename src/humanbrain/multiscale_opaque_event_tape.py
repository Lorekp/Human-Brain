"""Lossless multiscale events for opaque query and workspace byte streams.

ASCII identifier atoms stay byte-identical across the complete query and
workspace streams.  Generic frames preserve every other byte plus workspace,
file, path, and physical-line structure.  Whole path and line frames are also
the transport units used by a source-relative motor system.

This adapter derives representations only.  It does not select a file, line,
action, or answer, and it contains no language or repository ontology.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Self

__all__ = [
    "MultiscaleOpaqueEventTape",
    "OpaqueEventTapeError",
    "OpaqueWorkspaceFile",
]


_IDENTIFIER = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_LENGTH_BYTES = 8

_QUERY_START = b"\x00opaque-event-tape/query-start/v1"
_QUERY_END = b"\x00opaque-event-tape/query-end/v1"
_WORKSPACE_START = b"\x00opaque-event-tape/workspace-start/v1"
_WORKSPACE_END = b"\x00opaque-event-tape/workspace-end/v1"
_FILE_START = b"\x00opaque-event-tape/file-start/v1"
_FILE_END = b"\x00opaque-event-tape/file-end/v1"
_PATH_LEXICAL_START = b"\x00opaque-event-tape/path-lexical-start/v1"
_PATH_LEXICAL_END = b"\x00opaque-event-tape/path-lexical-end/v1"
_CONTENT_START = b"\x00opaque-event-tape/content-start/v1"
_CONTENT_END = b"\x00opaque-event-tape/content-end/v1"
_LINE_START = b"\x00opaque-event-tape/line-start/v1"
_LINE_END = b"\x00opaque-event-tape/line-end/v1"

_QUERY_LITERAL = b"\x00opaque-event-tape/query-literal/v1\x00"
_SOURCE_LITERAL = b"\x00opaque-event-tape/source-literal/v1\x00"
_PATH_MOTOR = b"\x00opaque-event-tape/path-motor/v1\x00"
_LINE_MOTOR = b"\x00opaque-event-tape/line-motor/v1\x00"
_CONSTRUCTION_TOKEN = object()


class OpaqueEventTapeError(ValueError):
    """The supplied bytes cannot be represented by this exact transport."""


@dataclass(frozen=True, slots=True)
class OpaqueWorkspaceFile:
    """One opaque byte stream at a canonical relative POSIX path."""

    path: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise OpaqueEventTapeError("path must be a non-empty string")
        normalized = PurePosixPath(self.path)
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.as_posix() != self.path
            or self.path == "."
        ):
            raise OpaqueEventTapeError("path must be canonical relative POSIX text")
        try:
            self.path.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise OpaqueEventTapeError("path must have an exact UTF-8 encoding") from exc
        if not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")


def _frame(prefix: bytes, payload: bytes) -> bytes:
    return prefix + len(payload).to_bytes(_LENGTH_BYTES, "big") + payload


def _unframe(event: bytes, prefix: bytes, name: str) -> bytes:
    if not isinstance(event, bytes) or not event.startswith(prefix):
        raise OpaqueEventTapeError(f"{name} frame changed")
    length_start = len(prefix)
    payload_start = length_start + _LENGTH_BYTES
    if len(event) < payload_start:
        raise OpaqueEventTapeError(f"{name} frame length is missing")
    length = int.from_bytes(event[length_start:payload_start], "big")
    payload = event[payload_start:]
    if len(payload) != length:
        raise OpaqueEventTapeError(f"{name} frame length changed")
    return payload


def _lexical_events(data: bytes, literal_prefix: bytes) -> tuple[bytes, ...]:
    events: list[bytes] = []
    cursor = 0
    for match in _IDENTIFIER.finditer(data):
        if cursor < match.start():
            events.append(_frame(literal_prefix, data[cursor : match.start()]))
        events.append(match.group(0))
        cursor = match.end()
    if cursor < len(data):
        events.append(_frame(literal_prefix, data[cursor:]))
    return tuple(events)


def _decode_lexical_events(events: tuple[bytes, ...], literal_prefix: bytes) -> bytes:
    parts: list[bytes] = []
    for event in events:
        if _IDENTIFIER.fullmatch(event) is not None:
            parts.append(event)
        else:
            parts.append(_unframe(event, literal_prefix, "literal"))
    return b"".join(parts)


def _physical_lines(content: bytes) -> tuple[bytes, ...]:
    lines = tuple(content.splitlines(keepends=True))
    if b"".join(lines) != content:
        raise RuntimeError("physical-line partition ceased to be lossless")
    return lines


def _require_files(files: object) -> tuple[OpaqueWorkspaceFile, ...]:
    if type(files) is not tuple or not files:
        raise OpaqueEventTapeError("files must be a non-empty tuple")
    if any(not isinstance(item, OpaqueWorkspaceFile) for item in files):
        raise TypeError("files must contain OpaqueWorkspaceFile values")
    ordered = tuple(sorted(files, key=lambda item: item.path))
    if len({item.path for item in ordered}) != len(ordered):
        raise OpaqueEventTapeError("workspace paths must be unique")
    return ordered


def _source_events(files: tuple[OpaqueWorkspaceFile, ...]) -> tuple[bytes, ...]:
    events: list[bytes] = [_WORKSPACE_START]
    for item in files:
        path = item.path.encode("utf-8")
        events.extend(
            (
                _FILE_START,
                _frame(_PATH_MOTOR, path),
                _PATH_LEXICAL_START,
                *_lexical_events(path, _SOURCE_LITERAL),
                _PATH_LEXICAL_END,
                _CONTENT_START,
            )
        )
        for line in _physical_lines(item.content):
            events.extend(
                (
                    _LINE_START,
                    _frame(_LINE_MOTOR, line),
                    *_lexical_events(line, _SOURCE_LITERAL),
                    _LINE_END,
                )
            )
        events.extend((_CONTENT_END, _FILE_END))
    events.append(_WORKSPACE_END)
    return tuple(events)


class MultiscaleOpaqueEventTape:
    """One immutable complete query/workspace observation with generic views."""

    __slots__ = ("_query_events", "_source_events")

    def __init__(
        self,
        token: object,
        *,
        query_events: tuple[bytes, ...],
        source_events: tuple[bytes, ...],
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use MultiscaleOpaqueEventTape.observe")
        self._query_events = query_events
        self._source_events = source_events

    @classmethod
    def observe(
        cls,
        query: bytes,
        files: tuple[OpaqueWorkspaceFile, ...],
    ) -> Self:
        """Derive one deterministic observation from complete byte streams."""

        if not isinstance(query, bytes) or not query:
            raise OpaqueEventTapeError("query must be non-empty bytes")
        ordered = _require_files(files)
        query_events = (
            _QUERY_START,
            *_lexical_events(query, _QUERY_LITERAL),
            _QUERY_END,
        )
        source_events = _source_events(ordered)
        observation = cls(
            _CONSTRUCTION_TOKEN,
            query_events=query_events,
            source_events=source_events,
        )
        if observation.reconstruct_query() != query:
            raise RuntimeError("query event tape ceased to be lossless")
        if observation.reconstruct_workspace() != ordered:
            raise RuntimeError("workspace event tape ceased to be lossless")
        return observation

    @property
    def query_events(self) -> tuple[bytes, ...]:
        """Complete query events; identifier atoms remain raw bytes."""

        return self._query_events

    @property
    def source_events(self) -> tuple[bytes, ...]:
        """Complete workspace events at boundary, line, and lexical scales."""

        return self._source_events

    @property
    def motor_source_tokens(self) -> tuple[bytes, ...]:
        """Current path/line units available to source-relative motor copying."""

        return tuple(
            event for event in self._source_events if event.startswith((_PATH_MOTOR, _LINE_MOTOR))
        )

    def reconstruct_query(self) -> bytes:
        """Decode the exact original query only from its complete event stream."""

        if (
            len(self._query_events) < 2
            or self._query_events[0] != _QUERY_START
            or self._query_events[-1] != _QUERY_END
        ):
            raise OpaqueEventTapeError("query boundaries changed")
        return _decode_lexical_events(self._query_events[1:-1], _QUERY_LITERAL)

    def reconstruct_workspace(self) -> tuple[OpaqueWorkspaceFile, ...]:
        """Decode every exact path and byte stream from workspace events."""

        events = self._source_events
        if len(events) < 2 or events[0] != _WORKSPACE_START or events[-1] != _WORKSPACE_END:
            raise OpaqueEventTapeError("workspace boundaries changed")

        files: list[OpaqueWorkspaceFile] = []
        cursor = 1
        while cursor < len(events) - 1:
            if events[cursor] != _FILE_START:
                raise OpaqueEventTapeError("file start changed")
            cursor += 1
            path_bytes = _unframe(events[cursor], _PATH_MOTOR, "path motor")
            cursor += 1
            if events[cursor] != _PATH_LEXICAL_START:
                raise OpaqueEventTapeError("path lexical start changed")
            cursor += 1
            path_lexical_start = cursor
            while cursor < len(events) - 1 and events[cursor] != _PATH_LEXICAL_END:
                cursor += 1
            if cursor >= len(events) - 1:
                raise OpaqueEventTapeError("path lexical end is missing")
            if (
                _decode_lexical_events(
                    events[path_lexical_start:cursor],
                    _SOURCE_LITERAL,
                )
                != path_bytes
            ):
                raise OpaqueEventTapeError("path lexical and motor views disagree")
            cursor += 1
            if events[cursor] != _CONTENT_START:
                raise OpaqueEventTapeError("content start changed")
            cursor += 1

            lines: list[bytes] = []
            while cursor < len(events) - 1 and events[cursor] != _CONTENT_END:
                if events[cursor] != _LINE_START:
                    raise OpaqueEventTapeError("line start changed")
                cursor += 1
                line = _unframe(events[cursor], _LINE_MOTOR, "line motor")
                cursor += 1
                lexical_start = cursor
                while cursor < len(events) - 1 and events[cursor] != _LINE_END:
                    cursor += 1
                if cursor >= len(events) - 1:
                    raise OpaqueEventTapeError("line end is missing")
                if _decode_lexical_events(events[lexical_start:cursor], _SOURCE_LITERAL) != line:
                    raise OpaqueEventTapeError("line lexical and motor views disagree")
                lines.append(line)
                cursor += 1
            if cursor >= len(events) - 1:
                raise OpaqueEventTapeError("content end is missing")
            cursor += 1
            if events[cursor] != _FILE_END:
                raise OpaqueEventTapeError("file end changed")
            cursor += 1
            try:
                path = path_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise OpaqueEventTapeError("path motor frame is not UTF-8") from exc
            files.append(OpaqueWorkspaceFile(path, b"".join(lines)))

        ordered = tuple(files)
        if tuple(item.path for item in ordered) != tuple(
            sorted(item.path for item in ordered)
        ) or len({item.path for item in ordered}) != len(ordered):
            raise OpaqueEventTapeError("workspace file order changed")
        return ordered

    def motor_tokens_for_replacement(
        self,
        path: str,
        content: bytes,
    ) -> tuple[bytes, ...]:
        """Project an observed complete replacement onto live motor units.

        This is a training/transport boundary, not an inference decision.  It
        fails unless every projected path/line token is an occurrence in this
        observation, including multiplicity.
        """

        replacement_file = OpaqueWorkspaceFile(path, content)
        if replacement_file.path not in {item.path for item in self.reconstruct_workspace()}:
            raise OpaqueEventTapeError("replacement path is absent from the observation")
        projected = (
            _frame(_PATH_MOTOR, replacement_file.path.encode("utf-8")),
            *(_frame(_LINE_MOTOR, line) for line in _physical_lines(replacement_file.content)),
        )
        available = Counter(self.motor_source_tokens)
        required = Counter(projected)
        if any(required[event] > available[event] for event in required):
            raise OpaqueEventTapeError(
                "replacement contains a motor unit absent from the observation"
            )
        return projected

    def materialize_motor_emission(
        self,
        emissions: tuple[bytes, ...],
    ) -> tuple[OpaqueWorkspaceFile, ...]:
        """Apply one emitted live path plus complete replacement line sequence."""

        if type(emissions) is not tuple or not emissions:
            raise OpaqueEventTapeError("emissions must be a non-empty tuple")
        if any(not isinstance(event, bytes) or not event for event in emissions):
            raise OpaqueEventTapeError("emissions must contain non-empty byte events")
        available = Counter(self.motor_source_tokens)
        emitted = Counter(emissions)
        if any(emitted[event] > available[event] for event in emitted):
            raise OpaqueEventTapeError("emission contains a non-live motor occurrence")

        path_bytes = _unframe(emissions[0], _PATH_MOTOR, "emitted path")
        try:
            path = path_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OpaqueEventTapeError("emitted path is not UTF-8") from exc
        replacement = b"".join(
            _unframe(event, _LINE_MOTOR, "emitted line") for event in emissions[1:]
        )
        files = self.reconstruct_workspace()
        if path not in {item.path for item in files}:
            raise OpaqueEventTapeError("emitted path is absent from the observation")
        return tuple(
            OpaqueWorkspaceFile(item.path, replacement if item.path == path else item.content)
            for item in files
        )
