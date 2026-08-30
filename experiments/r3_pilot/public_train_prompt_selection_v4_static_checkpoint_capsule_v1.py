"""Static, teacher-free inference capsule for the banked prompt V4 checkpoint.

Normal imports never execute the banked learner or its target-bearing protocol.
The small public interface authenticates one immutable NumPy artifact and maps
ordered physical prompt records to ordered physical relay winners, distributed
edge bindings, and an explicit terminal state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from humanbrain import cells as cell_runtime
from humanbrain import resting_glif_race
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCHEMA = "public-train-prompt-selection-v4-static-checkpoint-capsule-v1"
BANKED_COMMIT = "561a5819c9942a5be08ebabc94255ac463b3e496"
BANKED_SOURCE_SHA256 = "d59b9dc8c44fbe488e74204b07b8a254c1d6ca68d1d14f9cce11233ba5858cf0"
BANKED_TEST_SHA256 = "695ca9518dd19b3042ffd7a8c69f377f8b68fc44d6c19111ba86833853c7f5e5"
BANKED_RESULT_SHA256 = "e2249e0277a9ad49ea5eabe452843294ce7c658307e1ad467741a190b0e5fd21"
BANKED_CHECKPOINT_SHA256 = "e81336c6901edac0aada25c4ad89857686008fb276b94f705944ff9962a67d69"
BANKED_SOURCE_PATH = Path(__file__).with_name("public_train_prompt_selection_v4.py")
BANKED_TEST_PATH = (
    Path(__file__).resolve().parents[2] / "tests/test_public_train_prompt_selection_v4.py"
)

RESTING_GLIF_RACE_SOURCE_SHA256 = "8e97b75ac82005e2dd7375aa5caedf337dc35a471da4ebf1ff6300fa1b5bf82f"
CELL_RUNTIME_SOURCE_SHA256 = "5ca05a64a89f930db3ea351e3e236e3242bf21652fff127747a36d9ca488d2e6"

ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_prompt_selection_v4_static_checkpoint_capsule_v1.npz"
)
EXPECTED_ARTIFACT_SHA256 = "45d932c140f21f492d81af4fd250f89798bc715aca3fca04ad53cf28818ad96f"
EXPECTED_ARTIFACT_CONTENT_ROOT = "b6bcac0e642c2f9c7f4fd449745a501980724232ac43e206069c4d3d6b77896d"

STOP = "STOP"
COLLISION = "COLLISION"
CAP = "CAP"

_CHECKPOINT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.checkpoint.v1\0"
_REGISTRY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.registry.v1\0"
_BINDING_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.binding.v1\0"
_DECISION_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.decision.v1\0"
_ARTIFACT_CONTENT_NAMESPACE = (
    b"humanbrain.public-train-prompt-selection-v4-static-capsule-v1.content\0"
)
_INPUT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4-static-capsule-v1.input\0"
_OCCURRENCE_NAMESPACE = (
    b"humanbrain.public-train-prompt-selection-v4-static-capsule-v1.occurrence\0"
)
_ACTUAL_WINNER_NAMESPACE = (
    b"humanbrain.public-train-prompt-selection-v4-static-capsule-v1.actual-winners\0"
)
_CERTIFICATE_NAMESPACE = (
    b"humanbrain.public-train-prompt-selection-v4-static-capsule-v1.certificate\0"
)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _array_frame(name: str, array: np.ndarray) -> bytes:
    header = _canonical_json(
        {
            "name": name,
            "dtype": array.dtype.str,
            "shape": tuple(int(value) for value in array.shape),
        }
    )
    payload = np.ascontiguousarray(array).tobytes(order="C")
    return len(header).to_bytes(8, "little") + header + len(payload).to_bytes(8, "little") + payload


def _checkpoint_payload_from_header(
    header: bytes,
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> bytes:
    return (
        _CHECKPOINT_NAMESPACE
        + len(header).to_bytes(8, "little")
        + header
        + _array_frame("excitatory", excitatory)
        + _array_frame("inhibitory", inhibitory)
    )


@dataclass(frozen=True, slots=True)
class PromptPhysicalWinner:
    branch_id: str
    decision: str
    selected_cell_id: int | None
    relay_id: str | None
    first_spike_step: int
    simultaneous_spike_candidates: int
    decision_sha256: str


@dataclass(frozen=True, slots=True)
class PromptOccurrenceBinding:
    parent_cells: tuple[int, ...]
    child_cells: tuple[int, ...]
    sha256: str

    def record(self) -> dict[str, object]:
        return {
            "parent_cells": self.parent_cells,
            "child_cells": self.child_cells,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class PromptStructuralOccurrence:
    sequence_index: int
    state_sha256: str
    channel_winner: PromptPhysicalWinner
    branch_winners: tuple[PromptPhysicalWinner, ...]
    emitted_bindings: tuple[tuple[str, str | None], ...]
    occurrence_binding: PromptOccurrenceBinding | None
    sha256: str


@dataclass(frozen=True, slots=True)
class PromptStructuralCertificate:
    status: str
    checkpoint_sha256: str
    input_sha256: str
    occurrences: tuple[PromptStructuralOccurrence, ...]
    terminal_winner: PromptPhysicalWinner | None
    actual_winner_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _OutputCell:
    cell_id: int
    kind: str
    channel: str
    branch_role: str
    target_id: str

    def record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "kind": self.kind,
            "channel": self.channel,
            "branch_role": self.branch_role,
            "target_id": self.target_id,
        }


@dataclass(frozen=True, slots=True)
class _OutputBranch:
    branch_id: str
    kind: str
    channel: str
    branch_role: str
    candidate_cell_ids: tuple[int, ...]
    relation_cells: tuple[int, ...]
    role_cells: tuple[int, ...]

    def registry_record(self, stop_score: int) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "kind": self.kind,
            "channel": self.channel,
            "branch_role": self.branch_role,
            "candidate_cell_ids": self.candidate_cell_ids,
            "tonic_stop_score": stop_score,
            "selection": "repeated-masked-channel-local-first-spike-glif-wta",
        }


class _Decision(NamedTuple):
    branch_id: str
    kind: str
    channel: str
    branch_role: str
    occurrence_index: int
    decision: str
    selected_cell_id: int | None
    selected_target_id: str | None
    raw_candidate_scores: tuple[int, ...]
    masked_candidate_count: int
    first_spike_step: int
    simultaneous_spike_candidates: int
    feature_cells: tuple[int, ...]
    sha256: str

    def record(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self._fields}


class _Trace(NamedTuple):
    input_sha256: str
    physical_event_count: int
    lane_cells: tuple[int, ...]


class StaticPromptCheckpointCapsule:
    """Authenticated immutable implementation hidden behind the inference seam."""

    __slots__ = (
        "_artifact_sha256",
        "_branches",
        "_channel_branch",
        "_content_root",
        "_excitatory",
        "_inhibitory",
        "_metadata",
        "_output_cells",
    )

    def __init__(
        self,
        *,
        artifact_sha256: str,
        content_root: str,
        metadata: Mapping[str, object],
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        output_cells: tuple[_OutputCell, ...],
        channel_branch: _OutputBranch,
        branches: tuple[_OutputBranch, ...],
    ) -> None:
        self._artifact_sha256 = artifact_sha256
        self._content_root = content_root
        self._metadata = _freeze(metadata)
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._output_cells = output_cells
        self._channel_branch = channel_branch
        self._branches = branches

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def content_root(self) -> str:
        return self._content_root

    @property
    def checkpoint_sha256(self) -> str:
        return str(self._metadata["banked_checkpoint_sha256"])


def _runtime_source_sha256(module: object) -> str:
    path = Path(str(module.__file__)).resolve()
    return sha256(path.read_bytes()).hexdigest()


def _require_runtime_sources() -> None:
    if (
        _runtime_source_sha256(resting_glif_race) != RESTING_GLIF_RACE_SOURCE_SHA256
        or _runtime_source_sha256(cell_runtime) != CELL_RUNTIME_SOURCE_SHA256
    ):
        raise RuntimeError("static prompt capsule GLIF runtime source changed")


def _parse_output_cell(record: Mapping[str, object]) -> _OutputCell:
    if set(record) != {"cell_id", "kind", "channel", "branch_role", "target_id"}:
        raise RuntimeError("static prompt capsule output-cell schema changed")
    return _OutputCell(
        int(record["cell_id"]),
        str(record["kind"]),
        str(record["channel"]),
        str(record["branch_role"]),
        str(record["target_id"]),
    )


def _parse_branch(
    record: Mapping[str, object],
    features: Mapping[str, object],
    *,
    stop_score: int,
) -> _OutputBranch:
    if (
        set(record)
        != {
            "branch_id",
            "kind",
            "channel",
            "branch_role",
            "candidate_cell_ids",
            "tonic_stop_score",
            "selection",
        }
        or int(record["tonic_stop_score"]) != stop_score
        or record["selection"] != "repeated-masked-channel-local-first-spike-glif-wta"
        or set(features) != {"branch_id", "relation_cells", "role_cells"}
        or features["branch_id"] != record["branch_id"]
    ):
        raise RuntimeError("static prompt capsule branch schema changed")
    return _OutputBranch(
        branch_id=str(record["branch_id"]),
        kind=str(record["kind"]),
        channel=str(record["channel"]),
        branch_role=str(record["branch_role"]),
        candidate_cell_ids=tuple(int(value) for value in record["candidate_cell_ids"]),
        relation_cells=tuple(int(value) for value in features["relation_cells"]),
        role_cells=tuple(int(value) for value in features["role_cells"]),
    )


@lru_cache(maxsize=4)
def _load_checkpoint_capsule_cached(path: Path) -> StaticPromptCheckpointCapsule:
    _require_runtime_sources()
    raw = path.read_bytes()
    artifact_sha256 = sha256(raw).hexdigest()
    if not EXPECTED_ARTIFACT_SHA256 or artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError("static prompt checkpoint artifact bytes changed")
    with np.load(path, allow_pickle=False) as state:
        if set(state.files) != {
            "metadata",
            "checkpoint_header",
            "excitatory",
            "inhibitory",
        }:
            raise RuntimeError("static prompt checkpoint artifact keys changed")
        metadata_array = state["metadata"]
        checkpoint_header_array = state["checkpoint_header"]
        excitatory = np.array(state["excitatory"], copy=True, order="C")
        inhibitory = np.array(state["inhibitory"], copy=True, order="C")
    if metadata_array.shape != () or metadata_array.dtype.kind != "U":
        raise RuntimeError("static prompt checkpoint metadata framing changed")
    metadata_text = str(metadata_array.item())
    metadata = json.loads(metadata_text)
    if _canonical_json(metadata).decode("ascii") != metadata_text:
        raise RuntimeError("static prompt checkpoint metadata is not canonical")
    if checkpoint_header_array.ndim != 1 or checkpoint_header_array.dtype != np.uint8:
        raise RuntimeError("static prompt checkpoint header framing changed")
    checkpoint_header = np.ascontiguousarray(checkpoint_header_array).tobytes(order="C")

    required = {
        "schema",
        "banked_commit",
        "banked_source_sha256",
        "banked_test_sha256",
        "banked_result_sha256",
        "banked_checkpoint_sha256",
        "banked_dependency_sha256",
        "output_registry_sha256",
        "branch_registry_sha256",
        "checkpoint_header_sha256",
        "array_shape",
        "array_dtype",
        "excitatory_sha256",
        "inhibitory_sha256",
        "tick_cap",
        "stop_score",
        "race_steps",
        "latency_current_pa_hex",
        "latency_table_sha256",
        "prompt_lane_count",
        "prompt_cells_per_lane",
        "prompt_feature_offset",
        "input_cell_count",
        "occurrence_cap",
        "prompt_multipliers",
        "prompt_position_gains",
        "prompt_biases",
        "stop_state_base",
        "output_cells",
        "channel_branch",
        "output_branches",
        "branch_features",
        "reproduction_fixture",
        "artifact_content_root",
    }
    if set(metadata) != required:
        raise RuntimeError("static prompt checkpoint metadata keys changed")
    if (
        metadata["schema"] != SCHEMA
        or metadata["banked_commit"] != BANKED_COMMIT
        or metadata["banked_source_sha256"] != BANKED_SOURCE_SHA256
        or metadata["banked_test_sha256"] != BANKED_TEST_SHA256
        or metadata["banked_result_sha256"] != BANKED_RESULT_SHA256
        or metadata["banked_checkpoint_sha256"] != BANKED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("static prompt checkpoint banked authority changed")
    content_record = dict(metadata)
    content_root = str(content_record.pop("artifact_content_root"))
    if (
        not EXPECTED_ARTIFACT_CONTENT_ROOT
        or content_root != EXPECTED_ARTIFACT_CONTENT_ROOT
        or _namespaced_sha256(_ARTIFACT_CONTENT_NAMESPACE, content_record) != content_root
    ):
        raise RuntimeError("static prompt checkpoint content root changed")
    shape = tuple(int(value) for value in metadata["array_shape"])
    tick_cap = int(metadata["tick_cap"])
    if (
        shape != excitatory.shape
        or shape != inhibitory.shape
        or metadata["array_dtype"] != "|u1"
        or excitatory.dtype != np.uint8
        or inhibitory.dtype != np.uint8
        or not excitatory.flags.c_contiguous
        or not inhibitory.flags.c_contiguous
        or np.any(excitatory > tick_cap)
        or np.any(inhibitory > tick_cap)
        or sha256(excitatory.tobytes(order="C")).hexdigest() != metadata["excitatory_sha256"]
        or sha256(inhibitory.tobytes(order="C")).hexdigest() != metadata["inhibitory_sha256"]
        or sha256(checkpoint_header).hexdigest() != metadata["checkpoint_header_sha256"]
        or sha256(
            _checkpoint_payload_from_header(checkpoint_header, excitatory, inhibitory)
        ).hexdigest()
        != BANKED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("static prompt checkpoint array/checkpoint authentication failed")

    output_records = tuple(metadata["output_cells"])
    cells = tuple(_parse_output_cell(record) for record in output_records)
    if tuple(cell.cell_id for cell in cells) != tuple(range(len(cells))) or len(cells) != shape[0]:
        raise RuntimeError("static prompt checkpoint output-cell identity changed")
    stop_score = int(metadata["stop_score"])
    channel_record = metadata["channel_branch"]
    channel_branch = _parse_branch(
        channel_record,
        {"branch_id": channel_record["branch_id"], "relation_cells": (), "role_cells": ()},
        stop_score=stop_score,
    )
    branch_records = tuple(metadata["output_branches"])
    feature_records = tuple(metadata["branch_features"])
    if len(branch_records) != len(feature_records):
        raise RuntimeError("static prompt checkpoint branch-feature registry changed")
    branches = tuple(
        _parse_branch(record, feature, stop_score=stop_score)
        for record, feature in zip(branch_records, feature_records, strict=True)
    )
    semantic_candidates = tuple(cell for branch in branches for cell in branch.candidate_cell_ids)
    if (
        len(set(semantic_candidates)) != len(semantic_candidates)
        or set(semantic_candidates) != {cell.cell_id for cell in cells if cell.kind != "channel"}
        or any(
            len(branch.relation_cells) != int(metadata["prompt_lane_count"]) for branch in branches
        )
        or any(len(branch.role_cells) != int(metadata["prompt_lane_count"]) for branch in branches)
    ):
        raise RuntimeError("static prompt checkpoint inference registry changed")
    cell_root = _namespaced_sha256(_REGISTRY_NAMESPACE + b"cells\0", output_records)
    branch_root = _namespaced_sha256(
        _REGISTRY_NAMESPACE + b"branches\0", (channel_record, *branch_records)
    )
    if (
        cell_root != metadata["output_registry_sha256"]
        or branch_root != metadata["branch_registry_sha256"]
    ):
        raise RuntimeError("static prompt checkpoint registry roots changed")

    return StaticPromptCheckpointCapsule(
        artifact_sha256=artifact_sha256,
        content_root=content_root,
        metadata=metadata,
        excitatory=excitatory,
        inhibitory=inhibitory,
        output_cells=cells,
        channel_branch=channel_branch,
        branches=branches,
    )


def load_checkpoint_capsule(
    path: Path = ARTIFACT_PATH,
) -> StaticPromptCheckpointCapsule:
    """Load and authenticate the immutable checkpoint artifact without learning."""

    return _load_checkpoint_capsule_cached(Path(path).resolve())


def _normalize_span_records(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    required = {
        "start_token",
        "stop_token",
        "first_spike_step",
        "assembly_cells",
        "spike_cells",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("physical span record schema changed")
        assembly = tuple(int(cell) for cell in row["assembly_cells"])
        spikes = tuple(int(cell) for cell in row["spike_cells"])
        records.append(
            {
                "start_token": int(row["start_token"]),
                "stop_token": int(row["stop_token"]),
                "first_spike_step": int(row["first_spike_step"]),
                "assembly_cells": assembly,
                "spike_cells": spikes,
            }
        )
    return tuple(records)


def _normalize_literal_records(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    required = {"canonical_token_index", "typed_output_active_cells"}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("typed literal record schema changed")
        records.append(
            {
                "canonical_token_index": int(row["canonical_token_index"]),
                "typed_output_active_cells": tuple(
                    int(cell) for cell in row["typed_output_active_cells"]
                ),
            }
        )
    return tuple(records)


def _normalize_signature_role_record(row: Mapping[str, object]) -> dict[str, object]:
    required = {
        "parameters",
        "result",
        "role_binding",
        "entrypoint_spelling_present",
        "parameter_spelling_present",
    }
    if (
        not isinstance(row, Mapping)
        or set(row) != required
        or row["entrypoint_spelling_present"] is not False
        or row["parameter_spelling_present"] is not False
    ):
        raise ValueError("name-free signature role/type record changed")
    return {
        "parameters": _plain(row["parameters"]),
        "result": _plain(row["result"]),
        "role_binding": _plain(row["role_binding"]),
        "entrypoint_spelling_present": False,
        "parameter_spelling_present": False,
    }


def _physical_event_stream(
    spans: tuple[dict[str, object], ...],
    literals: tuple[dict[str, object], ...],
    signature_role: dict[str, object],
) -> tuple[int, ...]:
    events: list[int] = [10_001, len(spans)]
    for ordinal, span in enumerate(spans):
        events.extend(
            (
                10_101,
                ordinal,
                int(span["start_token"]),
                int(span["stop_token"]),
                int(span["first_spike_step"]),
            )
        )
        assembly = tuple(int(cell) for cell in span["assembly_cells"])
        spikes = tuple(int(cell) for cell in span["spike_cells"])
        events.extend((10_111, len(assembly), *assembly, 10_121, len(spikes), *spikes))
    events.extend((20_001, len(literals)))
    for ordinal, literal in enumerate(literals):
        active = tuple(int(cell) for cell in literal["typed_output_active_cells"])
        events.extend(
            (
                20_101,
                ordinal,
                int(literal["canonical_token_index"]),
                len(active),
                *active,
            )
        )
    role_bytes = _canonical_json(
        {
            "parameters": signature_role["parameters"],
            "result": signature_role["result"],
            "role_binding": signature_role["role_binding"],
        }
    )
    events.extend((30_001, len(role_bytes), *(int(value) for value in role_bytes)))
    return tuple(events)


def _encode_trace(
    capsule: StaticPromptCheckpointCapsule,
    spans: tuple[dict[str, object], ...],
    literals: tuple[dict[str, object], ...],
    signature_role: dict[str, object],
) -> _Trace:
    metadata = capsule._metadata
    lane_count = int(metadata["prompt_lane_count"])
    cell_count = int(metadata["prompt_cells_per_lane"])
    multipliers = tuple(int(value) for value in metadata["prompt_multipliers"])
    gains = tuple(int(value) for value in metadata["prompt_position_gains"])
    biases = tuple(int(value) for value in metadata["prompt_biases"])
    events = _physical_event_stream(spans, literals, signature_role)
    state = tuple((lane * 37 + 11) % cell_count for lane in range(lane_count))
    for position, event in enumerate(events, start=1):
        state = tuple(
            (state[lane] * multipliers[lane] + int(event) + position * gains[lane] + biases[lane])
            % cell_count
            for lane in range(lane_count)
        )
    input_record = {
        "span_events": spans,
        "literal_events": literals,
        "signature_role_record": signature_role,
    }
    return _Trace(
        _namespaced_sha256(_INPUT_NAMESPACE, input_record),
        len(events),
        state,
    )


def _state_from_cells(cells: tuple[int, ...]) -> tuple[tuple[int, ...], str]:
    record = {"cells": cells, "integer_handle_neural_input": False}
    return cells, _namespaced_sha256(_BINDING_NAMESPACE + b"state\0", record)


def _initial_state(
    capsule: StaticPromptCheckpointCapsule, trace: _Trace
) -> tuple[tuple[int, ...], str]:
    metadata = capsule._metadata
    cell_count = int(metadata["prompt_cells_per_lane"])
    gains = tuple(int(value) for value in metadata["prompt_position_gains"])
    biases = tuple(int(value) for value in metadata["prompt_biases"])
    cells = tuple(
        (trace.lane_cells[lane] * 313 + gains[lane] * 17 + biases[lane]) % cell_count
        for lane in range(int(metadata["prompt_lane_count"]))
    )
    return _state_from_cells(cells)


def _advance_state(
    capsule: StaticPromptCheckpointCapsule,
    state: tuple[int, ...],
    physical_winners: tuple[int, ...],
) -> tuple[tuple[int, ...], str]:
    metadata = capsule._metadata
    multipliers = tuple(int(value) for value in metadata["prompt_multipliers"])
    biases = tuple(int(value) for value in metadata["prompt_biases"])
    cell_count = int(metadata["prompt_cells_per_lane"])
    drive = sum((position + 1) * (cell + 1) for position, cell in enumerate(physical_winners))
    cells = tuple(
        (state[lane] * multipliers[lane] + drive + biases[lane]) % cell_count
        for lane in range(int(metadata["prompt_lane_count"]))
    )
    return _state_from_cells(cells)


def _occurrence_binding(
    capsule: StaticPromptCheckpointCapsule, state: tuple[int, ...]
) -> PromptOccurrenceBinding:
    metadata = capsule._metadata
    biases = tuple(int(value) for value in metadata["prompt_biases"])
    cell_count = int(metadata["prompt_cells_per_lane"])

    def side_cells(side: int) -> tuple[int, ...]:
        used: set[int] = set()
        cells: list[int] = []
        for lane, base in enumerate(state):
            candidate = (
                base * (337 if side == 0 else 347) + biases[lane] + side * 149
            ) % cell_count
            while candidate in used:
                candidate = (candidate + 1) % cell_count
            used.add(candidate)
            cells.append(candidate)
        return tuple(sorted(cells))

    parent = side_cells(0)
    child = side_cells(1)
    record = {
        "parent_cells": parent,
        "child_cells": child,
        "integer_handles_are_serialization_only": True,
    }
    return PromptOccurrenceBinding(
        parent,
        child,
        _namespaced_sha256(_BINDING_NAMESPACE + b"edge\0", record),
    )


def _prompt_feature_cells(
    capsule: StaticPromptCheckpointCapsule,
    trace: _Trace,
    state: tuple[int, ...],
    *,
    relation_cells: tuple[int, ...] = (),
    role_cells: tuple[int, ...] = (),
    binding_cells: tuple[int, ...] = (),
    kind_code: int,
) -> tuple[int, ...]:
    metadata = capsule._metadata
    lane_count = int(metadata["prompt_lane_count"])
    cell_count = int(metadata["prompt_cells_per_lane"])
    feature_offset = int(metadata["prompt_feature_offset"])
    multipliers = tuple(int(value) for value in metadata["prompt_multipliers"])
    biases = tuple(int(value) for value in metadata["prompt_biases"])
    cells: list[int] = []
    for lane in range(lane_count):
        relation = relation_cells[lane] if relation_cells else 0
        role = role_cells[lane] if role_cells else 0
        binding = binding_cells[lane] if binding_cells else state[lane]
        local = (
            trace.lane_cells[lane] * multipliers[lane]
            + state[lane] * 53
            + relation * 17
            + role * 29
            + binding * 31
            + kind_code * 43
            + biases[lane]
        ) % cell_count
        cells.append(feature_offset + lane * cell_count + local)
    return tuple(cells)


def _channel_feature(
    capsule: StaticPromptCheckpointCapsule, trace: _Trace, state: tuple[int, ...]
) -> tuple[int, ...]:
    return _prompt_feature_cells(capsule, trace, state, kind_code=1)


def _branch_feature(
    capsule: StaticPromptCheckpointCapsule,
    trace: _Trace,
    state: tuple[int, ...],
    branch: _OutputBranch,
) -> tuple[int, ...]:
    if branch.channel == "ordered-parent-child-edge":
        binding = _occurrence_binding(capsule, state)
        cell_count = int(capsule._metadata["prompt_cells_per_lane"])
        binding_cells = tuple(
            (binding.parent_cells[lane] + binding.child_cells[lane]) % cell_count
            for lane in range(int(capsule._metadata["prompt_lane_count"]))
        )
    else:
        binding_cells = state
    return _prompt_feature_cells(
        capsule,
        trace,
        state,
        relation_cells=branch.relation_cells,
        role_cells=branch.role_cells,
        binding_cells=binding_cells,
        kind_code=2 if branch.kind == "atom" else 3,
    )


@lru_cache(maxsize=4)
def _compiled_race(step_count: int) -> CompiledRestingGlifRace:
    return CompiledRestingGlifRace.compile(step_count=step_count)


def _decide(
    capsule: StaticPromptCheckpointCapsule,
    branch: _OutputBranch,
    feature: tuple[int, ...],
    *,
    occurrence_index: int,
    masked_cell_ids: set[int],
) -> _Decision:
    candidates = tuple(cell for cell in branch.candidate_cell_ids if cell not in masked_cell_ids)
    if feature and candidates:
        candidate_rows = np.asarray(candidates, dtype=np.intp)
        active = np.asarray(feature, dtype=np.intp)
        raw_array = np.sum(
            capsule._excitatory[candidate_rows][:, active].astype(np.int16)
            - capsule._inhibitory[candidate_rows][:, active].astype(np.int16),
            axis=1,
            dtype=np.int64,
        )
        raw = tuple(int(value) for value in raw_array)
    else:
        raw = tuple(0 for _cell in candidates)
    metadata = capsule._metadata
    current_hex = tuple(str(value) for value in metadata["latency_current_pa_hex"])
    current_scores = (
        *(min(max(score, 0), len(current_hex) - 1) for score in raw),
        int(metadata["stop_score"]),
    )
    currents = np.asarray(
        tuple(float.fromhex(current_hex[score]) for score in current_scores),
        dtype=np.float64,
    )
    race = _compiled_race(int(metadata["race_steps"])).race_many(
        currents[None, :], require_unique=False
    )[0]
    if race.simultaneous_spike_candidates != 1:
        decision, selected_cell, selected_target = COLLISION, None, None
    elif race.winner_cell == len(candidates):
        decision, selected_cell, selected_target = STOP, None, None
    else:
        selected_cell = candidates[race.winner_cell]
        selected_target = capsule._output_cells[selected_cell].target_id
        decision = "RELAY"
    record = {
        "branch_id": branch.branch_id,
        "kind": branch.kind,
        "channel": branch.channel,
        "branch_role": branch.branch_role,
        "occurrence_index": occurrence_index,
        "decision": decision,
        "selected_cell_id": selected_cell,
        "selected_target_id": selected_target,
        "raw_candidate_scores": raw,
        "masked_candidate_count": len(branch.candidate_cell_ids) - len(candidates),
        "first_spike_step": race.spike_step,
        "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
        "feature_cells": feature,
    }
    return _Decision(
        **record,
        sha256=_namespaced_sha256(_DECISION_NAMESPACE, record),
    )


def _physical_winner(decision: _Decision) -> PromptPhysicalWinner:
    return PromptPhysicalWinner(
        branch_id=decision.branch_id,
        decision=decision.decision,
        selected_cell_id=decision.selected_cell_id,
        relay_id=decision.selected_target_id,
        first_spike_step=decision.first_spike_step,
        simultaneous_spike_candidates=decision.simultaneous_spike_candidates,
        decision_sha256=decision.sha256,
    )


def _actual_winner_sha256(record: Mapping[str, object]) -> str:
    return _namespaced_sha256(_ACTUAL_WINNER_NAMESPACE, record)


def infer_prompt(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
) -> PromptStructuralCertificate:
    """Run teacher-free physical prompt inference against the static checkpoint."""

    capsule = load_checkpoint_capsule()
    spans = _normalize_span_records(span_records)
    literals = _normalize_literal_records(literal_records)
    signature_role = _normalize_signature_role_record(signature_role_record)
    trace = _encode_trace(capsule, spans, literals, signature_role)
    state, state_sha256 = _initial_state(capsule, trace)
    occurrences: list[PromptStructuralOccurrence] = []
    actual_occurrences: list[dict[str, object]] = []
    terminal_decision: _Decision | None = None
    status = CAP
    for sequence_index in range(int(capsule._metadata["occurrence_cap"])):
        route = _decide(
            capsule,
            capsule._channel_branch,
            _channel_feature(capsule, trace, state),
            occurrence_index=sequence_index,
            masked_cell_ids=set(),
        )
        if route.decision != "RELAY" or route.selected_target_id is None:
            terminal_decision = route
            status = route.decision
            break
        selected_channel = route.selected_target_id
        branch_decisions: list[_Decision] = []
        emitted: list[tuple[str, str | None]] = []
        physical_winners: list[int] = [int(route.selected_cell_id)]
        occurrence_mask: set[int] = set()
        for branch in capsule._branches:
            if branch.channel != selected_channel:
                continue
            decision = _decide(
                capsule,
                branch,
                _branch_feature(capsule, trace, state, branch),
                occurrence_index=sequence_index,
                masked_cell_ids=occurrence_mask,
            )
            branch_decisions.append(decision)
            emitted.append((branch.branch_id, decision.selected_target_id))
            if decision.selected_cell_id is not None:
                occurrence_mask.add(decision.selected_cell_id)
                physical_winners.append(decision.selected_cell_id)
            else:
                physical_winners.append(
                    int(capsule._metadata["stop_state_base"]) + len(branch_decisions) - 1
                )
        binding = (
            _occurrence_binding(capsule, state)
            if selected_channel == "ordered-parent-child-edge"
            else None
        )
        actual_record = {
            "sequence_index": sequence_index,
            "state_sha256": state_sha256,
            "channel_decision": route.record(),
            "selected_channel": selected_channel,
            "branch_decisions": tuple(row.record() for row in branch_decisions),
            "emitted_by_branch": tuple(emitted),
            "occurrence_binding": binding.record() if binding is not None else None,
        }
        actual_occurrences.append(actual_record)
        occurrence_record = {
            "sequence_index": sequence_index,
            "state_sha256": state_sha256,
            "channel_decision_sha256": route.sha256,
            "branch_decision_sha256s": tuple(row.sha256 for row in branch_decisions),
            "emitted_bindings": tuple(emitted),
            "occurrence_binding": binding.record() if binding is not None else None,
        }
        occurrences.append(
            PromptStructuralOccurrence(
                sequence_index=sequence_index,
                state_sha256=state_sha256,
                channel_winner=_physical_winner(route),
                branch_winners=tuple(_physical_winner(row) for row in branch_decisions),
                emitted_bindings=tuple(emitted),
                occurrence_binding=binding,
                sha256=_namespaced_sha256(_OCCURRENCE_NAMESPACE, occurrence_record),
            )
        )
        state, state_sha256 = _advance_state(capsule, state, tuple(physical_winners))
    actual_record = {
        "termination": status,
        "occurrences": tuple(actual_occurrences),
        "terminal_decision": (
            terminal_decision.record() if terminal_decision is not None else None
        ),
    }
    actual_winner_sha256 = _actual_winner_sha256(actual_record)
    terminal_winner = _physical_winner(terminal_decision) if terminal_decision is not None else None
    certificate_record = {
        "status": status,
        "checkpoint_sha256": capsule.checkpoint_sha256,
        "input_sha256": trace.input_sha256,
        "occurrences": tuple(row.sha256 for row in occurrences),
        "terminal_winner": (
            terminal_winner.decision_sha256 if terminal_winner is not None else None
        ),
        "actual_winner_sha256": actual_winner_sha256,
    }
    return PromptStructuralCertificate(
        status=status,
        checkpoint_sha256=capsule.checkpoint_sha256,
        input_sha256=trace.input_sha256,
        occurrences=tuple(occurrences),
        terminal_winner=terminal_winner,
        actual_winner_sha256=actual_winner_sha256,
        sha256=_namespaced_sha256(_CERTIFICATE_NAMESPACE, certificate_record),
    )


def _load_reproduction_fixture() -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
    str,
]:
    fixture = load_checkpoint_capsule()._metadata["reproduction_fixture"]
    return (
        tuple(fixture["span_records"]),
        tuple(fixture["literal_records"]),
        fixture["signature_role_record"],
        str(fixture["banked_actual_winner_sha256"]),
    )


__all__ = [
    "ARTIFACT_PATH",
    "BANKED_CHECKPOINT_SHA256",
    "BANKED_COMMIT",
    "BANKED_RESULT_SHA256",
    "BANKED_SOURCE_SHA256",
    "BANKED_TEST_SHA256",
    "CAP",
    "COLLISION",
    "EXPECTED_ARTIFACT_CONTENT_ROOT",
    "EXPECTED_ARTIFACT_SHA256",
    "SCHEMA",
    "STOP",
    "PromptOccurrenceBinding",
    "PromptPhysicalWinner",
    "PromptStructuralCertificate",
    "PromptStructuralOccurrence",
    "StaticPromptCheckpointCapsule",
    "infer_prompt",
    "load_checkpoint_capsule",
]
