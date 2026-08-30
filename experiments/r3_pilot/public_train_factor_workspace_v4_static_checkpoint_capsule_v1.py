"""Pure, teacher-free inference over the banked factor-workspace V4 state.

The learned arrays are recovered bytes from the already-published V4 checkpoint,
not a new experiment.  This module authenticates those bytes and reconstructs
only the frozen content x selector-role representation and factor-membership
relays needed by a later joint TRAIN successor.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_factor_workspace_v4_static_checkpoint_v1.npz"
)
BANKED_SOURCE_PATH = REPO_ROOT / "experiments/r3_pilot/public_train_factor_workspace_v4.py"
BANKED_TEST_PATH = REPO_ROOT / "tests/test_public_train_factor_workspace_v4.py"
BANKED_PROTOCOL_SOURCE_PATH = (
    REPO_ROOT / "experiments/r3_pilot/public_train_factor_workspace_v4_protocol.py"
)
BANKED_PROTOCOL_TEST_PATH = (
    REPO_ROOT / "tests/test_public_train_factor_workspace_v4_protocol.py"
)

BANKED_COMMIT = "2bee9d0120cf4d957e9aa426410490a333180139"
BANKED_PROTOCOL_COMMIT = "c9da529ea0c440ceb90a0f2ea896e3e3d379e2c0"
BANKED_SOURCE_SHA256 = "f91a51db6b2b236571647c6b5027a713167d22f68d0463ee55afdcd9fda92ecf"
BANKED_TEST_SHA256 = "23fbd16abe55d5ac808bd548ce6e5c9fe893cdbc1ff4fc5147c355761773e9f0"
BANKED_PROTOCOL_SOURCE_SHA256 = (
    "9df77077bbce8ba206fc6dcf8a1df4d282005e983199020be94b5b979db533b3"
)
BANKED_PROTOCOL_TEST_SHA256 = (
    "8980d6b8750a46fb9397dd29263dc513fa9e7a6e746f3afbf79e2a7fb896a1b4"
)
BANKED_PROTOCOL_SHA256 = (
    "83335e57fb896e9ba509a3f7f34e1bba39e7394d700c80813a702c4b34cc0f12"
)
BANKED_CHECKPOINT_SHA256 = (
    "63fd83a72f23aeb759bd72d032bb9b7c44adfcdead8557992602279b99847578"
)
BANKED_RESULT_SHA256 = (
    "96ab4c088382271e2208bbc71d8f308497cd5a560d601529e124938dda73cc4e"
)
ARTIFACT_SHA256 = "9af718cca6a4bb7b412b5067fce704df50c9b929259c2d9d6e1baab8e94e5490"

FACTOR_REGISTRY = (
    "add",
    "subtract",
    "minimum",
    "maximum",
    "absolute",
    "less-than",
    "map",
    "sum",
    "count",
    "contains",
    "filter",
    "reverse",
    "class:ascii-alphanumeric",
    "class:ascii-decimal-digits",
    "class:ascii-hex-digits",
    "class:ascii-letters",
    "class:ascii-lowercase",
    "class:ascii-octal-digits",
    "class:ascii-printable",
    "class:ascii-punctuation",
    "class:ascii-uppercase",
    "class:ascii-whitespace",
    "class:english-consonants",
    "class:english-vowels",
)
FACTOR_COUNT = len(FACTOR_REGISTRY)
LEXICAL_CELL_COUNT = 512
ASSEMBLY_ACTIVE_CELL_COUNT = 12
SELECTOR_DIRECTION_CELL_COUNT = 2
SELECTOR_CUE_ROLE_CELL_COUNT = 4
SELECTOR_SELECTED_ROLE_CELL_COUNT = 4
SELECTOR_ROLE_CELL_COUNT = 10
REPRESENTATION_CELL_COUNT = 5_632
ACTIVE_ROLE_CELL_COUNT = 3
ACTIVE_REPRESENTATION_CELL_COUNT = 48
MEMBERSHIP_SCORE_THRESHOLD = 12
EVENT_CAPACITY = 4
TICK_CAP = 4
STATE_SHAPE = (FACTOR_COUNT, REPRESENTATION_CELL_COUNT)
STATE_DTYPE = np.dtype(np.uint8)
STATE_BYTES = 270_336

_ARTIFACT_FORMAT = "public-train-factor-workspace-v4-static-checkpoint-v1"
_ARTIFACT_KEYS = ("metadata", "excitatory_ticks", "inhibitory_ticks")
_CHECKPOINT_NAMESPACE = b"public-train-factor-workspace-v4-checkpoint\0"
_OBSERVATION_NAMESPACE = b"public-train-factor-workspace-v4-static-observation-v1\0"
_REPRESENTATION_NAMESPACE = b"public-train-factor-workspace-v4-static-representation-v1\0"
_SPAN_NAMESPACE = b"public-train-factor-workspace-v4-static-span-v1\0"
_CERTIFICATE_NAMESPACE = b"public-train-factor-workspace-v4-static-certificate-v1\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _record_sha256(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _array_frame(name: str, array: np.ndarray) -> bytes:
    header = _canonical_json(
        {
            "name": name,
            "shape": array.shape,
            "dtype": array.dtype.str,
            "nbytes": array.nbytes,
        }
    )
    return b"".join(
        (
            len(header).to_bytes(8, "little"),
            header,
            memoryview(array).cast("B").tobytes(),
        )
    )


def _checkpoint_payload(
    metadata: dict[str, Any],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
) -> bytes:
    metadata_frame = _canonical_json(metadata)
    return b"".join(
        (
            _CHECKPOINT_NAMESPACE,
            len(metadata_frame).to_bytes(8, "little"),
            metadata_frame,
            _array_frame("excitatory_ticks", excitatory_ticks),
            _array_frame("inhibitory_ticks", inhibitory_ticks),
        )
    )


def _immutable_array(array: np.ndarray) -> np.ndarray:
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=STATE_DTYPE).reshape(
        STATE_SHAPE
    )
    immutable.setflags(write=False)
    return immutable


def _validate_array(
    name: str,
    array: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    if (
        array.shape != STATE_SHAPE
        or array.dtype != STATE_DTYPE
        or not array.flags.c_contiguous
        or array.nbytes != STATE_BYTES // 2
        or int(array.max(initial=0)) > TICK_CAP
    ):
        raise RuntimeError(f"static factor checkpoint {name} geometry changed")
    expected_record = metadata["arrays"][name]
    actual_record = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }
    if actual_record != expected_record:
        raise RuntimeError(f"static factor checkpoint {name} bytes changed")
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class FrozenStaticFactorCheckpoint:
    """Authenticated immutable V4 state with no fitting surface."""

    artifact_sha256: str
    checkpoint_sha256: str
    protocol_sha256: str
    result_sha256: str
    _excitatory_ticks: np.ndarray = field(repr=False)
    _inhibitory_ticks: np.ndarray = field(repr=False)

    @property
    def excitatory_ticks(self) -> np.ndarray:
        return self._excitatory_ticks

    @property
    def inhibitory_ticks(self) -> np.ndarray:
        return self._inhibitory_ticks

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> FrozenStaticFactorCheckpoint:
        if _file_sha256(path) != ARTIFACT_SHA256:
            raise RuntimeError("static factor checkpoint artifact changed")
        dependency_receipts = (
            (BANKED_SOURCE_PATH, BANKED_SOURCE_SHA256),
            (BANKED_TEST_PATH, BANKED_TEST_SHA256),
            (BANKED_PROTOCOL_SOURCE_PATH, BANKED_PROTOCOL_SOURCE_SHA256),
            (BANKED_PROTOCOL_TEST_PATH, BANKED_PROTOCOL_TEST_SHA256),
        )
        if any(_file_sha256(source) != expected for source, expected in dependency_receipts):
            raise RuntimeError("banked factor-workspace dependency receipt changed")

        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != _ARTIFACT_KEYS:
                raise RuntimeError("static factor checkpoint members changed")
            metadata_array = archive["metadata"]
            if metadata_array.shape != () or metadata_array.dtype.kind != "U":
                raise RuntimeError("static factor checkpoint metadata framing changed")
            metadata_text = str(metadata_array.item())
            metadata = json.loads(metadata_text)
            if metadata_text.encode("utf-8") != _canonical_json(metadata):
                raise RuntimeError("static factor checkpoint metadata is not canonical JSON")
            excitatory = _validate_array("excitatory_ticks", archive["excitatory_ticks"], metadata)
            inhibitory = _validate_array("inhibitory_ticks", archive["inhibitory_ticks"], metadata)

        expected_metadata = {
            "format": _ARTIFACT_FORMAT,
            "purpose": "artifact-recovery-only-not-new-scientific-evidence",
            "banked_commit": BANKED_COMMIT,
            "banked_source_sha256": BANKED_SOURCE_SHA256,
            "banked_test_sha256": BANKED_TEST_SHA256,
            "banked_protocol_commit": BANKED_PROTOCOL_COMMIT[:7],
            "banked_protocol_sha256": BANKED_PROTOCOL_SHA256,
            "banked_protocol_source_sha256": BANKED_PROTOCOL_SOURCE_SHA256,
            "banked_protocol_test_sha256": BANKED_PROTOCOL_TEST_SHA256,
            "banked_checkpoint_sha256": BANKED_CHECKPOINT_SHA256,
            "banked_result_sha256": BANKED_RESULT_SHA256,
            "array_count": 2,
            "state_bytes": STATE_BYTES,
            "normal_runtime_imports_executable_learner": False,
            "normal_runtime_imports_target_protocol": False,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError("static factor checkpoint provenance changed")
        checkpoint_metadata = metadata.get("banked_checkpoint_metadata")
        if not isinstance(checkpoint_metadata, dict):
            raise RuntimeError("banked factor checkpoint metadata is missing")
        checkpoint_sha256 = sha256(
            _checkpoint_payload(checkpoint_metadata, excitatory, inhibitory)
        ).hexdigest()
        if checkpoint_sha256 != BANKED_CHECKPOINT_SHA256:
            raise RuntimeError("static factor checkpoint identity changed")

        return cls(
            artifact_sha256=ARTIFACT_SHA256,
            checkpoint_sha256=checkpoint_sha256,
            protocol_sha256=BANKED_PROTOCOL_SHA256,
            result_sha256=BANKED_RESULT_SHA256,
            _excitatory_ticks=_immutable_array(excitatory),
            _inhibitory_ticks=_immutable_array(inhibitory),
        )


@dataclass(frozen=True, slots=True)
class RawSpanObservation:
    """One selected assembly plus selector geometry, with no semantic labels."""

    direction: str
    cue_slot: int
    selected_slot: int
    assembly_cells: tuple[int, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        cells = tuple(self.assembly_cells)
        if self.direction not in ("NEXT", "PREV"):
            raise ValueError("raw span direction must be NEXT or PREV")
        if (
            type(self.cue_slot) is not int
            or type(self.selected_slot) is not int
            or not 0 <= self.cue_slot < 4
            or not 0 <= self.selected_slot < 4
            or self.cue_slot == self.selected_slot
        ):
            raise ValueError("raw span selector slots changed")
        expected_selected = (
            (self.cue_slot + 1) % 4
            if self.direction == "NEXT"
            else (self.cue_slot - 1) % 4
        )
        if self.selected_slot != expected_selected:
            raise ValueError("raw span selector geometry is inconsistent")
        if (
            len(cells) != ASSEMBLY_ACTIVE_CELL_COUNT
            or any(type(cell) is not int or not 0 <= cell < LEXICAL_CELL_COUNT for cell in cells)
            or len(set(cells)) != len(cells)
            or tuple(sorted(cells)) != cells
        ):
            raise ValueError("raw span assembly must be 12 sorted distinct lexical cells")
        object.__setattr__(self, "assembly_cells", cells)
        object.__setattr__(
            self,
            "sha256",
            _record_sha256(_OBSERVATION_NAMESPACE, self.record()),
        )

    def record(self, *, include_sha256: bool = False) -> dict[str, object]:
        record: dict[str, object] = {
            "direction": self.direction,
            "cue_slot": self.cue_slot,
            "selected_slot": self.selected_slot,
            "assembly_cells": self.assembly_cells,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SpanFactorDecision:
    observation: RawSpanObservation
    status: str
    representation_sha256: str
    raw_scores: tuple[int, ...]
    physical_occupancy: tuple[bool, ...]
    factor_membership: tuple[str, ...]
    membership_count: int
    stop_spike: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "observation": self.observation.record(include_sha256=True),
            "status": self.status,
            "representation_sha256": self.representation_sha256,
            "raw_scores": self.raw_scores,
            "physical_occupancy": self.physical_occupancy,
            "factor_membership": self.factor_membership,
            "membership_count": self.membership_count,
            "stop_spike": self.stop_spike,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class FactorSetCertificate:
    status: str
    span_decisions: tuple[SpanFactorDecision, ...]
    event_counts: tuple[int, ...]
    physical_occupancy: tuple[bool, ...]
    factor_membership: tuple[str, ...]
    membership_count: int
    checkpoint_sha256: str
    artifact_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "span_decisions": tuple(row.record() for row in self.span_decisions),
            "event_counts": self.event_counts,
            "physical_occupancy": self.physical_occupancy,
            "factor_membership": self.factor_membership,
            "membership_count": self.membership_count,
            "checkpoint_sha256": self.checkpoint_sha256,
            "artifact_sha256": self.artifact_sha256,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


def _representation_cells(observation: RawSpanObservation) -> tuple[int, ...]:
    direction_role = 0 if observation.direction == "NEXT" else 1
    role_cells = (
        direction_role,
        SELECTOR_DIRECTION_CELL_COUNT + observation.cue_slot,
        SELECTOR_DIRECTION_CELL_COUNT
        + SELECTOR_CUE_ROLE_CELL_COUNT
        + observation.selected_slot,
    )
    conjunctions = tuple(
        LEXICAL_CELL_COUNT + role_cell * LEXICAL_CELL_COUNT + lexical_cell
        for role_cell in role_cells
        for lexical_cell in observation.assembly_cells
    )
    active_cells = tuple(sorted((*observation.assembly_cells, *conjunctions)))
    if (
        len(active_cells) != ACTIVE_REPRESENTATION_CELL_COUNT
        or len(set(active_cells)) != len(active_cells)
        or active_cells[-1] >= REPRESENTATION_CELL_COUNT
    ):
        raise RuntimeError("static factor representation geometry changed")
    return active_cells


def _membership_relays(raw_scores: tuple[int, ...]) -> tuple[tuple[bool, ...], bool]:
    if len(raw_scores) != FACTOR_COUNT:
        raise RuntimeError("static factor score geometry changed")
    nonnegative = np.maximum(np.asarray(raw_scores, dtype=np.int64), 0)
    occupancy = tuple(
        bool(value) for value in np.greater_equal(nonnegative, MEMBERSHIP_SCORE_THRESHOLD)
    )
    return occupancy, not any(occupancy)


def _infer_span(
    checkpoint: FrozenStaticFactorCheckpoint,
    observation: RawSpanObservation,
) -> SpanFactorDecision:
    active_cells = _representation_cells(observation)
    active = np.asarray(active_cells, dtype=np.intp)
    excitatory = np.sum(
        checkpoint.excitatory_ticks[:, active], axis=1, dtype=np.int64
    )
    inhibitory = np.sum(
        checkpoint.inhibitory_ticks[:, active], axis=1, dtype=np.int64
    )
    raw_scores = tuple(int(value) for value in excitatory - inhibitory)
    occupancy, stop_spike = _membership_relays(raw_scores)
    membership_count = sum(occupancy)
    if membership_count == 0:
        status = "STOP_ZERO"
    elif membership_count > EVENT_CAPACITY:
        status = "STOP_EVENT_CAPACITY_4"
    else:
        status = "READY_FACTOR_SET"
    factor_membership = tuple(
        factor
        for factor, active_factor in zip(FACTOR_REGISTRY, occupancy, strict=True)
        if active_factor
    )
    representation_sha256 = _record_sha256(
        _REPRESENTATION_NAMESPACE,
        {"active_cells": active_cells},
    )
    record = {
        "observation": observation.record(include_sha256=True),
        "status": status,
        "representation_sha256": representation_sha256,
        "raw_scores": raw_scores,
        "physical_occupancy": occupancy,
        "factor_membership": factor_membership,
        "membership_count": membership_count,
        "stop_spike": stop_spike,
    }
    return SpanFactorDecision(
        observation=observation,
        status=status,
        representation_sha256=representation_sha256,
        raw_scores=raw_scores,
        physical_occupancy=occupancy,
        factor_membership=factor_membership,
        membership_count=membership_count,
        stop_spike=stop_spike,
        sha256=_record_sha256(_SPAN_NAMESPACE, record),
    )


STATIC_CHECKPOINT = FrozenStaticFactorCheckpoint.load()


def infer_factors(
    raw_span_observations: Iterable[RawSpanObservation],
) -> FactorSetCertificate:
    """Infer and aggregate factor events without fitting or answer-bearing input."""

    observations = tuple(raw_span_observations)
    if any(not isinstance(row, RawSpanObservation) for row in observations):
        raise TypeError("factor inference accepts only RawSpanObservation rows")
    decisions = tuple(_infer_span(STATIC_CHECKPOINT, row) for row in observations)
    event_counts = tuple(
        sum(int(row.physical_occupancy[index]) for row in decisions)
        for index in range(FACTOR_COUNT)
    )
    occupancy = tuple(count > 0 for count in event_counts)
    membership_count = sum(occupancy)
    if any(count > 1 for count in event_counts):
        status = "STOP_DUPLICATE_FACTOR"
    elif membership_count > EVENT_CAPACITY:
        status = "STOP_EVENT_CAPACITY_4"
    elif membership_count == 0:
        status = "STOP_ZERO"
    else:
        status = "READY_FACTOR_SET"
    factor_membership = tuple(
        factor
        for factor, active_factor in zip(FACTOR_REGISTRY, occupancy, strict=True)
        if active_factor
    )
    record = {
        "status": status,
        "span_decisions": tuple(row.record() for row in decisions),
        "event_counts": event_counts,
        "physical_occupancy": occupancy,
        "factor_membership": factor_membership,
        "membership_count": membership_count,
        "checkpoint_sha256": STATIC_CHECKPOINT.checkpoint_sha256,
        "artifact_sha256": STATIC_CHECKPOINT.artifact_sha256,
    }
    return FactorSetCertificate(
        status=status,
        span_decisions=decisions,
        event_counts=event_counts,
        physical_occupancy=occupancy,
        factor_membership=factor_membership,
        membership_count=membership_count,
        checkpoint_sha256=STATIC_CHECKPOINT.checkpoint_sha256,
        artifact_sha256=STATIC_CHECKPOINT.artifact_sha256,
        sha256=_record_sha256(_CERTIFICATE_NAMESPACE, record),
    )


__all__ = [
    "ARTIFACT_PATH",
    "ARTIFACT_SHA256",
    "BANKED_CHECKPOINT_SHA256",
    "BANKED_COMMIT",
    "BANKED_PROTOCOL_SHA256",
    "BANKED_RESULT_SHA256",
    "FACTOR_REGISTRY",
    "STATIC_CHECKPOINT",
    "FactorSetCertificate",
    "FrozenStaticFactorCheckpoint",
    "RawSpanObservation",
    "SpanFactorDecision",
    "infer_factors",
]
