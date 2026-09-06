"""Bounded, byte-preserving events; no vocabulary, decoding or model terminator."""

from dataclasses import dataclass

MAX_EVENT_BYTES = 16


@dataclass(frozen=True)
class ByteEvent:
    """One raw payload and its half-open offsets in the complete input stream."""

    start: int
    end: int
    payload: bytes


def _byte_class(value: int) -> int:
    if 48 <= value <= 57 or 65 <= value <= 90 or 97 <= value <= 122 or value == 95:
        return 1
    if value in b"\t\n\v\f\r ":
        return 2
    return 0


class ByteEventSegmenter:
    """Stream capped alphanumeric/underscore and whitespace runs, or singletons.

    Only an unfinished run is retained. A full 16-byte run emits immediately;
    chunk boundaries do not finish a run. ``finish`` flushes the input boundary
    once and closes the stream. Construct another instance for another string.
    Counters describe host bytes and temporary buffering, never neuronal work.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._kind = 0
        self._start = 0
        self._offset = 0
        self._max_buffered = 0
        self._finished = False

    @property
    def bytes_seen(self) -> int:
        return self._offset

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    @property
    def max_buffered_bytes(self) -> int:
        return self._max_buffered

    def _flush(self) -> ByteEvent:
        event = ByteEvent(self._start, self._offset, bytes(self._buffer))
        self._buffer.clear()
        self._kind = 0
        return event

    def feed(self, chunk: bytes) -> tuple[ByteEvent, ...]:
        """Consume raw bytes without retaining the chunk or any emitted payload."""
        if not isinstance(chunk, bytes):
            raise ValueError("chunks must be raw bytes")
        if self._finished:
            raise ValueError("cannot feed a finished byte-event stream")
        events = []
        for value in chunk:
            kind = _byte_class(value)
            if self._buffer and kind != self._kind:
                events.append(self._flush())
            if kind == 0:
                events.append(ByteEvent(self._offset, self._offset + 1, bytes([value])))
                self._offset += 1
                continue
            if not self._buffer:
                self._start = self._offset
                self._kind = kind
            self._buffer.append(value)
            self._offset += 1
            self._max_buffered = max(self._max_buffered, len(self._buffer))
            if len(self._buffer) == MAX_EVENT_BYTES:
                events.append(self._flush())
        return tuple(events)

    def finish(self) -> tuple[ByteEvent, ...]:
        """Flush any unfinished raw run; empty or already finished inputs emit nothing."""
        self._finished = True
        return (self._flush(),) if self._buffer else ()


def segment_bytes(raw: bytes) -> tuple[ByteEvent, ...]:
    """One-shot form of the identical streaming policy; no EOS is appended."""
    segmenter = ByteEventSegmenter()
    return segmenter.feed(raw) + segmenter.finish()
