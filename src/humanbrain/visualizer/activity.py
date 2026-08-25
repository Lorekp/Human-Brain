"""Disk-backed reader for live and completed Observatory activity streams."""

from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

from humanbrain.observatory import ACTIVITY_STREAM_FORMAT, activity_stream_path

from .model import ExperimentRun


ActivityEventKind = Literal["phase", "state", "zero"]


@dataclass(frozen=True, slots=True)
class ActivityHeader:
    run_id: str
    configured_neurons: int
    active_neurons: int
    sample_gids: tuple[int, ...]
    state_stride: int
    model_family: str
    token_interval_ms: float
    integration_dt_ms: float


@dataclass(frozen=True, slots=True)
class ActivityFrame:
    index: int
    phase: str
    token_id: int
    biological_time_ms: float
    spike_count: int
    population_firing_rate_hz: float
    neuron_step_activity_density: float
    voltage_relative_mv: tuple[float, ...] | None
    synaptic_current_pa: tuple[float, ...] | None
    eligibility: tuple[float, ...] | None

    @property
    def has_state(self) -> bool:
        return self.voltage_relative_mv is not None


@dataclass(frozen=True, slots=True)
class ActivityPhaseRun:
    start_position: int
    end_position: int
    phase: str


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    position: int
    kinds: tuple[ActivityEventKind, ...]
    label: str


class ActivityStreamReader:
    """Index a trace for full replay while bounding decoded frame objects.

    One 64-bit file offset and compact timeline value are retained per frame.
    Decoded frames outside the rolling live window are loaded directly from
    disk on demand. Reader memory therefore does not depend on sampled-state
    payload size and never scales with the configured neuron population.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_frames: int = 20_000,
        random_access_cache: int = 256,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        if random_access_cache < 1:
            raise ValueError("random_access_cache must be positive")
        self.path = Path(path)
        self.header: ActivityHeader | None = None
        self.frames: deque[ActivityFrame] = deque(maxlen=max_frames)
        self.complete: bool | None = None
        self.declared_frame_count: int | None = None
        self.issue: str | None = None
        self._offset = 0
        self._source_size = 0
        self._last_frame_index: int | None = None
        self._frame_offsets = array("Q")
        self._frame_indices = array("Q")
        self._firing_rates = array("f")
        self._phase_runs: list[ActivityPhaseRun] = []
        self._phase_positions = array("Q")
        self._state_positions = array("Q")
        self._zero_positions = array("Q")
        self._random_access_cache = random_access_cache
        self._cache: OrderedDict[int, ActivityFrame] = OrderedDict()

    @property
    def total_frames(self) -> int:
        return len(self._frame_offsets)

    @property
    def source_size(self) -> int:
        return self._source_size

    @property
    def indexed_bytes(self) -> int:
        return self._offset

    @property
    def caught_up(self) -> bool:
        return self._offset >= self._source_size

    @property
    def phase_runs(self) -> tuple[ActivityPhaseRun, ...]:
        return tuple(self._phase_runs)

    def poll(self, *, max_bytes: int = 16 * 1024 * 1024) -> int:
        """Index complete records appended since the prior poll.

        The byte budget prevents opening a large completed trace from freezing
        the native UI. A single JSONL record may exceed the budget and is still
        consumed atomically.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if not self.path.is_file():
            self._source_size = 0
            return 0
        self._source_size = self.path.stat().st_size
        if self._source_size < self._offset:
            self._reset()
            self._source_size = self.path.stat().st_size
        added = 0
        with self.path.open("rb") as stream:
            stream.seek(self._offset)
            poll_start = self._offset
            while stream.tell() - poll_start < max_bytes:
                line_offset = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                if not raw_line.endswith(b"\n"):
                    stream.seek(line_offset)
                    break
                self._offset = stream.tell()
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                    added += self._accept(record, line_offset=line_offset)
                except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                    self.issue = f"invalid activity record: {error}"
        return added

    def frame_at(self, position: int) -> ActivityFrame:
        """Load one exact frame by zero-based replay position."""
        if type(position) is not int or not 0 <= position < self.total_frames:
            raise IndexError("activity frame position is out of range")
        cached = self._cache.get(position)
        if cached is not None:
            self._cache.move_to_end(position)
            return cached
        with self.path.open("rb") as stream:
            stream.seek(self._frame_offsets[position])
            raw_line = stream.readline()
        if not raw_line.endswith(b"\n"):
            raise ValueError("indexed activity frame is incomplete")
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"indexed activity frame is invalid: {error}") from error
        frame = self._parse_frame(record)
        if frame.index != self._frame_indices[position]:
            raise ValueError("indexed activity frame no longer matches its trace")
        self._cache[position] = frame
        self._cache.move_to_end(position)
        while len(self._cache) > self._random_access_cache:
            self._cache.popitem(last=False)
        return frame

    def state_frame_at_or_before(self, position: int) -> ActivityFrame | None:
        """Return the most recent exact sampled-state frame at a position."""
        if self.total_frames == 0:
            return None
        position = min(max(0, position), self.total_frames - 1)
        index = bisect_right(self._state_positions, position) - 1
        if index < 0:
            return None
        return self.frame_at(int(self._state_positions[index]))

    def event_at(self, position: int) -> ActivityEvent | None:
        """Describe exact recorded events attached to one token frame."""
        if type(position) is not int or not 0 <= position < self.total_frames:
            return None
        kinds: list[ActivityEventKind] = []
        labels: list[str] = []
        if _contains(self._phase_positions, position):
            kinds.append("phase")
            labels.append(f"phase → {self.frame_at(position).phase.replace('_', ' ')}")
        if _contains(self._state_positions, position):
            kinds.append("state")
            labels.append("sampled state captured")
        if _contains(self._zero_positions, position):
            kinds.append("zero")
            labels.append("zero-spike token")
        if not kinds:
            return None
        return ActivityEvent(position=position, kinds=tuple(kinds), label="  •  ".join(labels))

    def previous_event(
        self,
        position: int,
        kind: ActivityEventKind | None = None,
    ) -> ActivityEvent | None:
        event_position = self._adjacent_event_position(position, kind, previous=True)
        return None if event_position is None else self.event_at(event_position)

    def next_event(
        self,
        position: int,
        kind: ActivityEventKind | None = None,
    ) -> ActivityEvent | None:
        event_position = self._adjacent_event_position(position, kind, previous=False)
        return None if event_position is None else self.event_at(event_position)

    def timeline_points(
        self,
        *,
        max_points: int = 6_000,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return a min/max envelope of the full indexed firing-rate trace."""
        if max_points < 2:
            raise ValueError("max_points must be at least 2")
        count = self.total_frames
        if count == 0:
            return (), ()
        if count <= max_points:
            return (
                tuple(float(value) for value in self._frame_indices),
                tuple(float(value) for value in self._firing_rates),
            )
        bucket_count = max(1, max_points // 2)
        positions: list[int] = []
        for bucket in range(bucket_count):
            start = bucket * count // bucket_count
            end = max(start + 1, (bucket + 1) * count // bucket_count)
            minimum = min(range(start, end), key=self._firing_rates.__getitem__)
            maximum = max(range(start, end), key=self._firing_rates.__getitem__)
            positions.extend(sorted({minimum, maximum}))
        return (
            tuple(float(self._frame_indices[position]) for position in positions),
            tuple(float(self._firing_rates[position]) for position in positions),
        )

    def _adjacent_event_position(
        self,
        position: int,
        kind: ActivityEventKind | None,
        *,
        previous: bool,
    ) -> int | None:
        sources = (
            (self._event_positions(kind),)
            if kind is not None
            else (self._phase_positions, self._state_positions, self._zero_positions)
        )
        candidates: list[int] = []
        for values in sources:
            index = bisect_left(values, position) - 1 if previous else bisect_right(values, position)
            if 0 <= index < len(values):
                candidates.append(int(values[index]))
        if not candidates:
            return None
        return max(candidates) if previous else min(candidates)

    def _event_positions(self, kind: ActivityEventKind) -> array:
        if kind == "phase":
            return self._phase_positions
        if kind == "state":
            return self._state_positions
        if kind == "zero":
            return self._zero_positions
        raise ValueError(f"unknown activity event kind: {kind!r}")

    def _reset(self) -> None:
        self.header = None
        self.frames.clear()
        self.complete = None
        self.declared_frame_count = None
        self.issue = None
        self._offset = 0
        self._source_size = 0
        self._last_frame_index = None
        self._frame_offsets = array("Q")
        self._frame_indices = array("Q")
        self._firing_rates = array("f")
        self._phase_runs.clear()
        self._phase_positions = array("Q")
        self._state_positions = array("Q")
        self._zero_positions = array("Q")
        self._cache.clear()

    def _accept(self, record: Any, *, line_offset: int) -> int:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        record_type = record.get("type")
        if record_type == "header":
            if record.get("format") != ACTIVITY_STREAM_FORMAT:
                raise ValueError("unsupported activity format")
            header = ActivityHeader(
                run_id=_text(record, "run_id"),
                configured_neurons=_integer(record, "configured_neurons", minimum=1),
                active_neurons=_integer(record, "active_neurons", minimum=1),
                sample_gids=_integer_tuple(record, "sample_gids"),
                state_stride=_integer(record, "state_stride", minimum=1),
                model_family=_text(record, "model_family"),
                token_interval_ms=_number(record, "token_interval_ms"),
                integration_dt_ms=_number(record, "integration_dt_ms"),
            )
            if header.active_neurons > header.configured_neurons:
                raise ValueError("active_neurons exceeds configured_neurons")
            if not header.sample_gids or len(set(header.sample_gids)) != len(header.sample_gids):
                raise ValueError("sample_gids must be non-empty and unique")
            if any(gid < 0 or gid >= header.active_neurons for gid in header.sample_gids):
                raise ValueError("sample_gids must fit active_neurons")
            if self.header is not None or self.total_frames:
                raise ValueError("activity stream contains more than one header")
            self.header = header
            return 0
        if record_type == "frame":
            frame = self._parse_frame(record)
            if self._last_frame_index is not None and frame.index <= self._last_frame_index:
                raise ValueError("frame indices must increase strictly")
            position = self.total_frames
            self._frame_offsets.append(line_offset)
            self._frame_indices.append(frame.index)
            self._firing_rates.append(frame.population_firing_rate_hz)
            if not self._phase_runs or self._phase_runs[-1].phase != frame.phase:
                self._phase_runs.append(ActivityPhaseRun(position, position + 1, frame.phase))
                self._phase_positions.append(position)
            else:
                previous = self._phase_runs[-1]
                self._phase_runs[-1] = ActivityPhaseRun(
                    previous.start_position,
                    position + 1,
                    previous.phase,
                )
            if frame.has_state:
                self._state_positions.append(position)
            if frame.spike_count == 0:
                self._zero_positions.append(position)
            self.frames.append(frame)
            self._cache[position] = frame
            self._cache.move_to_end(position)
            while len(self._cache) > self._random_access_cache:
                self._cache.popitem(last=False)
            self._last_frame_index = frame.index
            return 1
        if record_type == "footer":
            complete = record.get("complete")
            if type(complete) is not bool:
                raise ValueError("complete must be a boolean")
            declared_frame_count = _integer(record, "frame_count")
            if declared_frame_count != self.total_frames:
                raise ValueError(
                    f"footer declares {declared_frame_count} frames but "
                    f"{self.total_frames} were indexed"
                )
            self.complete = complete
            self.declared_frame_count = declared_frame_count
            return 0
        raise ValueError(f"unknown activity record type: {record_type!r}")

    def _parse_frame(self, record: Any) -> ActivityFrame:
        if self.header is None:
            raise ValueError("frame appeared before the header")
        if not isinstance(record, dict) or record.get("type") != "frame":
            raise ValueError("indexed activity record is not a frame")
        frame = ActivityFrame(
            index=_integer(record, "index"),
            phase=_text(record, "phase"),
            token_id=_integer(record, "token_id"),
            biological_time_ms=_number(record, "biological_time_ms"),
            spike_count=_integer(record, "spike_count"),
            population_firing_rate_hz=_number(record, "population_firing_rate_hz"),
            neuron_step_activity_density=_number(record, "neuron_step_activity_density"),
            voltage_relative_mv=_optional_number_tuple(record, "voltage_relative_mv"),
            synaptic_current_pa=_optional_number_tuple(record, "synaptic_current_pa"),
            eligibility=_optional_number_tuple(record, "eligibility"),
        )
        sample_count = len(self.header.sample_gids)
        channels = (
            frame.voltage_relative_mv,
            frame.synaptic_current_pa,
            frame.eligibility,
        )
        if any(channel is None for channel in channels) and any(
            channel is not None for channel in channels
        ):
            raise ValueError("sample state channels must appear together")
        if frame.has_state and not all(
            len(channel) == sample_count for channel in channels if channel is not None
        ):
            raise ValueError("sample channel lengths do not match sample_gids")
        return frame


def reader_for_run(root: str | Path, run: ExperimentRun) -> ActivityStreamReader:
    return ActivityStreamReader(activity_stream_path(root, run.experiment_id, run.arm))


def _contains(values: array, position: int) -> bool:
    index = bisect_left(values, position)
    return index < len(values) and values[index] == position


def _integer(record: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = record.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _number(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _integer_tuple(record: dict[str, Any], key: str) -> tuple[int, ...]:
    value = record.get(key)
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{key} must be an integer array")
    return tuple(value)


def _optional_number_tuple(
    record: dict[str, Any],
    key: str,
) -> tuple[float, ...] | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(type(item) not in (int, float) for item in value):
        raise ValueError(f"{key} must be a numeric array")
    numbers = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in numbers):
        raise ValueError(f"{key} must contain only finite numbers")
    return numbers
