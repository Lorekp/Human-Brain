"""Fresh compositional prompt-selection V3 learner/result.

The deep module spends the frozen V3 protocol once.  It first trains a fresh
cap-four local workspace on all 138 self-posed physical relation/role/filler
lessons, freezes that checkpoint, and runs causal controls.  Only then may the
canonical 24 FIT teachers be projected into the frozen atom/transition
vocabulary.  Projection is atomic: one unsupported FIT atom prevents every
canonical teacher update and all heldout target access/replay.

The implementation has no whole-target head, host label union, host ranking,
candidate program, or proof path.  Host code authenticates teachers and scores
already-selected physical relay cells after channel-local GLIF races.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as v4spec
from experiments.r3_pilot import public_train_prompt_selection_v3_protocol as protocol_v3
from humanbrain.resting_glif_race import CompiledRestingGlifRace

PROTOCOL_SCOPE = "PUBLIC_TRAIN_PROMPT_SELECTION_V3"
PASS = "PASS_PUBLIC_TRAIN_PROMPT_SELECTION_V3"
STOP_SELF_POSED = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_SELF_POSED"
STOP_CONTROLS = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_CONTROLS"
STOP_FIT_PROJECTION = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_FIT_PROJECTION"
STOP_FIT = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_FIT"
STOP_HELDOUT = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_HELDOUT"

EXPECTED_PROTOCOL_SHA256 = "1eb2e3503790ebd721f69cc5d6f6dd8a32e0bb816d9b7f3dc8f1b4be9f15983f"
EXPECTED_PROTOCOL_SOURCE_SHA256 = "62bc379a9d0254a2976eb8af4f7dd8c2d283786ef8960301dedc52335f991bae"
EXPECTED_SELECTION_V1_SHA256 = "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47"
EXPECTED_CAPSULE_SHA256 = "484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf"
EXPECTED_V4_PROTOCOL_SHA256 = "37aa868988319bd70cea735ad715736dabaef2ac59a45ee0c70677fa89ac2124"

_PINNED_SOURCE_SHA256S = MappingProxyType(
    {
        "experiments/r3_pilot/public_train_prompt_selection_v3_protocol.py": (
            EXPECTED_PROTOCOL_SOURCE_SHA256
        ),
        "experiments/r3_pilot/public_train_prompt_selection_freeze_v1_protocol.py": (
            "a35ee1de48bea57404d101b69ec729d36470a88a55caa2aff5163dde246894ef"
        ),
        "experiments/r3_pilot/public_train_structural_annotation_capsule_v1.py": (
            "d2e977b3aff2e496307168ed64e85c2ebbc03579224bf58b0fd175f0ed096c88"
        ),
        "experiments/r3_pilot/exclusive_factor_workspace_v4_protocol.py": (
            "6dfb55ca119a9c1b46c296047b83941fa7607e4276631d0112d075ce5fd2c08b"
        ),
    }
)

TICK_CAP = 4
STOP_SCORE = 12
RACE_STEPS = v4spec.RACE_STEPS
FEATURE_CELL_COUNT = 512
EVENT_FEATURE_OFFSET = 0
TRANSITION_FEATURE_OFFSET = FEATURE_CELL_COUNT
INPUT_CELL_COUNT = FEATURE_CELL_COUNT * 2
ACTIVE_FEATURE_CELL_COUNT = 12

FIT = "FIT"
HELDOUT = "HELDOUT"

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_READ_COUNT = 0
PROGRAM_ARTIFACT_READ_COUNT = 0
EXECUTABLE_SOURCE_READ_COUNT = 0
AST_ARTIFACT_READ_COUNT = 0
PROPERTY_ROW_READ_COUNT = 0
TASK_IDENTIFIER_NEURAL_INPUT_COUNT = 0
FAMILY_IDENTIFIER_NEURAL_INPUT_COUNT = 0
ENTRYPOINT_SPELLING_NEURAL_INPUT_COUNT = 0
AUDIT_HASH_NEURAL_INPUT_COUNT = 0
PROMPT_FACTOR_INFERENCE_COUNT = 0
WHOLE_TARGET_HEAD_COUNT = 0
HOST_LABEL_ASSEMBLY_COUNT = 0
HOST_LOOKUP_SELECTION_COUNT = 0
HOST_UNION_SELECTION_COUNT = 0
HOST_RANKING_COUNT = 0
CANDIDATE_PROGRAM_COUNT = 0
PROOF_COUNT = 0
CANONICAL_FIT_TEACHER_UPDATE_COUNT = 0
HELDOUT_TARGET_READ_COUNT = 0
HELDOUT_REPLAY_IDENTITY_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]

_DEPENDENCY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.dependency.v2\0"
_REGISTRY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.registry.v2\0"
_TRACE_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.trace.v2\0"
_WORK_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.work.v2\0"
_CHECKPOINT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.checkpoint.v2\0"
_DECISION_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.decision.v2\0"
_REPLAY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.replay.v2\0"
_BATCH_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.batch.v2\0"
_CONTROL_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.control.v2\0"
_CONTROL_PANEL_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.controls.v2\0"
_PROJECTION_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.projection.v2\0"
_SURFACE_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.surface.v2\0"
_RESULT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.result.v2\0"


def _plain_record(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain_record(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_record(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain_record(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _freeze_record(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_record(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_record(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class WorkLedger:
    events: tuple[tuple[str, int], ...]
    total: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {"events": self.events, "total": self.total}
        if include_sha256:
            record["sha256"] = self.sha256
        return record


class _WorkMeter:
    def __init__(self) -> None:
        self._events: Counter[str] = Counter()

    def charge(self, event: str, count: int = 1) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("work charge must be a nonnegative integer")
        self._events[event] += count

    def merge(self, ledger: WorkLedger) -> None:
        for event, count in ledger.events:
            self.charge(event, count)

    def freeze(self) -> WorkLedger:
        events = tuple(sorted(self._events.items()))
        record = {"events": events, "total": sum(count for _event, count in events)}
        return WorkLedger(
            events=events,
            total=record["total"],
            sha256=_namespaced_sha256(_WORK_NAMESPACE, record),
        )


@dataclass(frozen=True, slots=True)
class OutputCell:
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
class OutputBranch:
    kind: str
    channel: str
    branch_role: str
    candidate_cell_ids: tuple[int, ...]

    def record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "channel": self.channel,
            "branch_role": self.branch_role,
            "candidate_cell_ids": self.candidate_cell_ids,
            "tonic_stop_score": STOP_SCORE,
            "selection": "channel-local-first-spike-glif-wta",
        }


@dataclass(frozen=True, slots=True)
class EventFeature:
    relation_cells: tuple[int, ...]
    role_cells: tuple[int, ...]
    active_cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TransitionFeature:
    relation_cells: tuple[int, ...]
    from_role_cells: tuple[int, ...]
    to_role_cells: tuple[int, ...]
    active_cells: tuple[int, ...]


class _CanonicalRelayLabel(Protocol):
    channel: str
    fields: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PhysicalInputTrace:
    lesson_index: int
    mode: str
    raw_events: tuple[protocol_v3.PhysicalAssemblyEvent, ...]
    event_features: tuple[EventFeature, ...]
    transition_features: tuple[TransitionFeature, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "lesson_index": self.lesson_index,
            "mode": self.mode,
            "raw_events": tuple(row.record() for row in self.raw_events),
            "event_features": tuple(
                {
                    "relation_cells": row.relation_cells,
                    "role_cells": row.role_cells,
                    "active_cells": row.active_cells,
                }
                for row in self.event_features
            ),
            "transition_features": tuple(
                {
                    "relation_cells": row.relation_cells,
                    "from_role_cells": row.from_role_cells,
                    "to_role_cells": row.to_role_cells,
                    "active_cells": row.active_cells,
                }
                for row in self.transition_features
            ),
            "neural_sources": (
                "physical-relation-assemblies",
                "physical-role-assemblies",
                "physical-filler-assemblies",
                "ordered-role-transition-assemblies",
            ),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class BranchDecision:
    kind: str
    channel: str
    branch_role: str
    decision: str
    selected_cell_id: int | None
    selected_target_id: str | None
    raw_candidate_scores: tuple[int, ...]
    first_spike_step: int
    simultaneous_spike_candidates: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "channel": self.channel,
            "branch_role": self.branch_role,
            "decision": self.decision,
            "selected_cell_id": self.selected_cell_id,
            "selected_target_id": self.selected_target_id,
            "raw_candidate_scores": self.raw_candidate_scores,
            "first_spike_step": self.first_spike_step,
            "simultaneous_spike_candidates": self.simultaneous_spike_candidates,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class LessonReplay:
    lesson_index: int
    channel: str
    trace_sha256: str
    decisions: tuple[BranchDecision, ...]
    emitted_target_ids: tuple[str, ...]
    expected_target_ids: tuple[str, ...]
    exact: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "lesson_index": self.lesson_index,
            "channel": self.channel,
            "trace_sha256": self.trace_sha256,
            "decisions": tuple(row.record() for row in self.decisions),
            "emitted_target_ids": self.emitted_target_ids,
            "expected_target_ids": self.expected_target_ids,
            "exact": self.exact,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ChannelOutcome:
    channel: str
    lesson_count: int
    exact_lesson_count: int
    mismatch_lesson_count: int

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    name: str
    lesson_count: int
    exact_lesson_count: int
    mismatch_lesson_count: int
    channel_outcomes: tuple[ChannelOutcome, ...]
    replays: tuple[LessonReplay, ...]
    work: WorkLedger
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "lesson_count": self.lesson_count,
            "exact_lesson_count": self.exact_lesson_count,
            "mismatch_lesson_count": self.mismatch_lesson_count,
            "channel_outcomes": tuple(row.record() for row in self.channel_outcomes),
            "replays": tuple(row.record() for row in self.replays),
            "work": self.work.record(),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    name: str
    exact_lesson_count: int
    mismatch_lesson_count: int
    checkpoint_sha256: str
    work: WorkLedger
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "exact_lesson_count": self.exact_lesson_count,
            "mismatch_lesson_count": self.mismatch_lesson_count,
            "checkpoint_sha256": self.checkpoint_sha256,
            "work": self.work.record(),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ProjectionChannelOutcome:
    channel: str
    fit_identity_count: int
    supported_identity_count: int
    unsupported_identity_count: int
    missing_atom_occurrence_count: int

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class CanonicalFitProjectionAudit:
    status: str
    fit_identity_count: int
    projectable_identity_count: int
    unsupported_identity_count: int
    channel_outcomes: tuple[ProjectionChannelOutcome, ...]
    missing_atom_ids: tuple[str, ...]
    missing_atom_occurrence_count: int
    canonical_teacher_update_count: int
    heldout_target_read_count: int
    heldout_replay_identity_count: int
    work: WorkLedger
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "fit_identity_count": self.fit_identity_count,
            "projectable_identity_count": self.projectable_identity_count,
            "unsupported_identity_count": self.unsupported_identity_count,
            "channel_outcomes": tuple(row.record() for row in self.channel_outcomes),
            "missing_atom_ids": self.missing_atom_ids,
            "missing_atom_occurrence_count": self.missing_atom_occurrence_count,
            "canonical_teacher_update_count": self.canonical_teacher_update_count,
            "heldout_target_read_count": self.heldout_target_read_count,
            "heldout_replay_identity_count": self.heldout_replay_identity_count,
            "work": self.work.record(),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    output_cell_count: int
    atom_cell_count: int
    transition_cell_count: int
    branch_count: int
    array_shape: tuple[int, int]
    array_dtype: str
    array_bytes: int
    tick_cap: int
    excitatory_nonzero_ticks: int
    inhibitory_nonzero_ticks: int
    self_posed_teacher_lesson_count: int
    canonical_fit_teacher_update_count: int
    heldout_teacher_identity_count: int
    checkpoint_sha256: str

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class PublicTrainPromptSelectionV3Result:
    scope: str
    status: str
    checkpoint: CheckpointSummary
    self_posed_result: ReplayBatch
    controls: tuple[ControlOutcome, ...]
    fit_projection_audit: CanonicalFitProjectionAudit
    heldout_channel_outcomes: tuple[ProjectionChannelOutcome, ...]
    training_work: WorkLedger
    physical_work_total: int
    dependency_sha256: str
    output_registry_sha256: str
    branch_registry_sha256: str
    checkpoint_sha256: str
    self_posed_result_sha256: str
    control_panel_sha256: str
    fit_projection_sha256: str
    surface_boundary_sha256: str
    result_sha256: str

    def root_record(self, *, include_result_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "output_registry_sha256": self.output_registry_sha256,
            "branch_registry_sha256": self.branch_registry_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "self_posed_result_sha256": self.self_posed_result_sha256,
            "control_panel_sha256": self.control_panel_sha256,
            "fit_projection_sha256": self.fit_projection_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "training_work": self.training_work.record(),
            "physical_work_total": self.physical_work_total,
            "self_posed_lesson_count": self.self_posed_result.lesson_count,
            "self_posed_exact_lesson_count": self.self_posed_result.exact_lesson_count,
            "canonical_fit_identity_count": self.fit_projection_audit.fit_identity_count,
            "canonical_fit_projectable_identity_count": (
                self.fit_projection_audit.projectable_identity_count
            ),
            "canonical_fit_teacher_update_count": (
                self.fit_projection_audit.canonical_teacher_update_count
            ),
            "heldout_target_read_count": self.fit_projection_audit.heldout_target_read_count,
            "heldout_replay_identity_count": (
                self.fit_projection_audit.heldout_replay_identity_count
            ),
            "checkpoint": self.checkpoint.record(),
        }
        if include_result_sha256:
            record["result_sha256"] = self.result_sha256
        return record


def _authenticate_dependencies() -> None:
    if (
        protocol_v3.PROTOCOL.protocol_sha256 != EXPECTED_PROTOCOL_SHA256
        or protocol_v3.PROTOCOL.status != protocol_v3.PASS_PROTOCOL
        or protocol_v3.LEARNER_GATE_RECORD["licensed_successor_scope"] != PROTOCOL_SCOPE
        or protocol_v3.LEARNER_GATE_RECORD["canonical_heldout_training_identity_count"] != 0
        or protocol_v3.LEARNER_GATE_RECORD["whole_target_class_heads_allowed"] is not False
        or v4spec.PROTOCOL.protocol_sha256 != EXPECTED_V4_PROTOCOL_SHA256
        or v4spec.TICK_CAP != TICK_CAP
        or v4spec.STOP_SCORE != STOP_SCORE
    ):
        raise RuntimeError("prompt-selection V3 frozen protocol/mechanism changed")
    for relative_path, expected in _PINNED_SOURCE_SHA256S.items():
        path = _ROOT / relative_path
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"prompt-selection V3 pinned dependency changed: {relative_path}")


_authenticate_dependencies()

DEPENDENCY_RECORD = _freeze_record(
    {
        "protocol_scope": protocol_v3.PROTOCOL.scope,
        "protocol_status": protocol_v3.PROTOCOL.status,
        "protocol_sha256": protocol_v3.PROTOCOL.protocol_sha256,
        "protocol_learner_gate_sha256": protocol_v3.PROTOCOL.learner_gate_sha256,
        "protocol_vocabulary_sha256": protocol_v3.PROTOCOL.vocabulary_sha256,
        "protocol_curriculum_sha256": protocol_v3.PROTOCOL.curriculum_sha256,
        "protocol_audit_sha256": protocol_v3.PROTOCOL.audit_sha256,
        "banked_v4_protocol_sha256": v4spec.PROTOCOL.protocol_sha256,
        "banked_v4_local_rule_sha256": v4spec.PROTOCOL.local_rule_sha256,
        "pinned_source_sha256s": tuple(_PINNED_SOURCE_SHA256S.items()),
        "teacher_schedule": ("complete-138-self-posed-pass-then-atomic-at-most-24-fit-projection"),
        "canonical_heldout_teacher_identity_count": 0,
        "v2_checkpoint_or_head_reuse": False,
        "whole_target_head_count": WHOLE_TARGET_HEAD_COUNT,
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(_DEPENDENCY_NAMESPACE, dict(DEPENDENCY_RECORD))


def _build_output_registry() -> tuple[tuple[OutputCell, ...], tuple[OutputBranch, ...], str, str]:
    cells: list[OutputCell] = []
    for atom in protocol_v3.OUTPUT_ATOMS:
        cells.append(
            OutputCell(
                cell_id=len(cells),
                kind="atom",
                channel=atom.channel,
                branch_role=atom.role,
                target_id=atom.atom_id,
            )
        )
    for transition in protocol_v3.OUTPUT_TRANSITIONS:
        cells.append(
            OutputCell(
                cell_id=len(cells),
                kind="transition",
                channel=transition.channel,
                branch_role=transition.from_role,
                target_id=transition.transition_id,
            )
        )
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for cell in cells:
        grouped[(cell.kind, cell.channel, cell.branch_role)].append(cell.cell_id)
    channel_order = {
        schema.channel: index for index, schema in enumerate(protocol_v3.RELAY_SCHEMAS)
    }
    branches = tuple(
        OutputBranch(kind, channel, role, tuple(candidate_ids))
        for (kind, channel, role), candidate_ids in sorted(
            grouped.items(),
            key=lambda row: (
                channel_order[row[0][1]],
                0 if row[0][0] == "atom" else 1,
                row[0][2],
            ),
        )
    )
    if len(cells) != 160 or len({row.target_id for row in cells}) != len(cells):
        raise RuntimeError("prompt-selection V3 output registry changed")
    return (
        tuple(cells),
        branches,
        _namespaced_sha256(_REGISTRY_NAMESPACE + b"cells\0", tuple(row.record() for row in cells)),
        _namespaced_sha256(
            _REGISTRY_NAMESPACE + b"branches\0", tuple(row.record() for row in branches)
        ),
    )


OUTPUT_CELLS, OUTPUT_BRANCHES, OUTPUT_REGISTRY_SHA256, BRANCH_REGISTRY_SHA256 = (
    _build_output_registry()
)


def _conjunction_cells(kind: str, parts: tuple[int, ...], *, offset: int) -> tuple[int, ...]:
    cells: set[int] = set()
    counter = 0
    payload = b"".join(int(value).to_bytes(2, "big") for value in parts)
    while len(cells) < ACTIVE_FEATURE_CELL_COUNT:
        digest = sha256(
            _TRACE_NAMESPACE + kind.encode("ascii") + b"\0" + payload + counter.to_bytes(4, "big")
        ).digest()
        for position in range(0, len(digest), 2):
            cells.add(
                offset + int.from_bytes(digest[position : position + 2], "big") % FEATURE_CELL_COUNT
            )
            if len(cells) == ACTIVE_FEATURE_CELL_COUNT:
                break
        counter += 1
    return tuple(sorted(cells))


def _transform_events(
    events: tuple[protocol_v3.PhysicalAssemblyEvent, ...], mode: str
) -> tuple[protocol_v3.PhysicalAssemblyEvent, ...]:
    if mode == "normal":
        return events
    if mode == "cell-off":
        return ()
    if mode == "reverse-order":
        return tuple(reversed(events))
    if len(events) < 2:
        raise RuntimeError("causal derangement needs at least two events")
    rows = list(events)
    left, right = rows[0], rows[1]
    if mode == "role-swap":
        rows[0] = protocol_v3.PhysicalAssemblyEvent(
            left.relation_cells, right.role_cells, left.filler_cells
        )
        rows[1] = protocol_v3.PhysicalAssemblyEvent(
            right.relation_cells, left.role_cells, right.filler_cells
        )
    elif mode == "filler-swap":
        rows[0] = protocol_v3.PhysicalAssemblyEvent(
            left.relation_cells, left.role_cells, right.filler_cells
        )
        rows[1] = protocol_v3.PhysicalAssemblyEvent(
            right.relation_cells, right.role_cells, left.filler_cells
        )
    else:
        raise ValueError(f"unknown trace mode: {mode}")
    return tuple(rows)


def _encode_trace(
    lesson: protocol_v3.SelfPosedRelayLesson,
    *,
    mode: str,
    meter: _WorkMeter,
) -> PhysicalInputTrace:
    events = _transform_events(lesson.causal_input.ordered_events, mode)
    event_features = tuple(
        EventFeature(
            relation_cells=row.relation_cells,
            role_cells=row.role_cells,
            active_cells=_conjunction_cells(
                "event-role-filler",
                (*row.relation_cells, *row.role_cells, *row.filler_cells),
                offset=EVENT_FEATURE_OFFSET,
            ),
        )
        for row in events
    )
    transition_features = tuple(
        TransitionFeature(
            relation_cells=left.relation_cells,
            from_role_cells=left.role_cells,
            to_role_cells=right.role_cells,
            active_cells=_conjunction_cells(
                "ordered-role-transition",
                (*left.relation_cells, *left.role_cells, *right.role_cells),
                offset=TRANSITION_FEATURE_OFFSET,
            ),
        )
        for left, right in pairwise(events)
    )
    meter.charge("physical-assembly-cell-read", len(events) * 3 * 12)
    meter.charge("fixed-conjunction-synapse-read", len(events) * 36 * 12)
    meter.charge("fixed-transition-conjunction-synapse-read", len(transition_features) * 36 * 12)
    meter.charge(
        "conjunction-cell-emission",
        (len(event_features) + len(transition_features)) * ACTIVE_FEATURE_CELL_COUNT,
    )
    record = {
        "lesson_index": lesson.lesson_index,
        "mode": mode,
        "raw_events": tuple(row.record() for row in events),
        "event_features": tuple(dataclasses.asdict(row) for row in event_features),
        "transition_features": tuple(dataclasses.asdict(row) for row in transition_features),
    }
    return PhysicalInputTrace(
        lesson_index=lesson.lesson_index,
        mode=mode,
        raw_events=events,
        event_features=event_features,
        transition_features=transition_features,
        sha256=_namespaced_sha256(_TRACE_NAMESPACE, record),
    )


def _array_frame(name: str, array: np.ndarray) -> bytes:
    header = _canonical_json(
        {"name": name, "dtype": array.dtype.str, "shape": tuple(int(v) for v in array.shape)}
    )
    payload = np.ascontiguousarray(array).tobytes(order="C")
    return len(header).to_bytes(8, "little") + header + len(payload).to_bytes(8, "little") + payload


def _checkpoint_payload(
    metadata: Mapping[str, object], excitatory: np.ndarray, inhibitory: np.ndarray
) -> bytes:
    header = _canonical_json(metadata)
    return (
        _CHECKPOINT_NAMESPACE
        + len(header).to_bytes(8, "little")
        + header
        + _array_frame("excitatory", excitatory)
        + _array_frame("inhibitory", inhibitory)
    )


class FrozenCompositionalWorkspace:
    __slots__ = (
        "_branches",
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_metadata",
        "_output_cells",
    )

    def __init__(
        self,
        *,
        output_cells: tuple[OutputCell, ...],
        branches: tuple[OutputBranch, ...],
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        metadata: Mapping[str, object],
    ) -> None:
        expected_shape = (len(output_cells), INPUT_CELL_COUNT)
        if (
            excitatory.shape != expected_shape
            or inhibitory.shape != expected_shape
            or excitatory.dtype != np.uint8
            or inhibitory.dtype != np.uint8
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("prompt-selection V3 arrays violate the cap-four domain")
        self._output_cells = tuple(output_cells)
        self._branches = tuple(branches)
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._metadata = _freeze_record(metadata)
        self._checkpoint_sha256 = sha256(
            _checkpoint_payload(self._metadata, self._excitatory, self._inhibitory)
        ).hexdigest()
        self.authenticate()

    @property
    def output_cells(self) -> tuple[OutputCell, ...]:
        return self._output_cells

    @property
    def branches(self) -> tuple[OutputBranch, ...]:
        return self._branches

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def excitatory_ticks(self) -> np.ndarray:
        value = self._excitatory.copy(order="C")
        value.setflags(write=False)
        return value

    @property
    def inhibitory_ticks(self) -> np.ndarray:
        value = self._inhibitory.copy(order="C")
        value.setflags(write=False)
        return value

    def authenticate(self) -> None:
        if (
            self._metadata["dependency_sha256"] != DEPENDENCY_SHA256
            or self._metadata["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256
            or self._metadata["output_registry_sha256"] != OUTPUT_REGISTRY_SHA256
            or self._metadata["branch_registry_sha256"] != BRANCH_REGISTRY_SHA256
            or self._metadata["canonical_fit_teacher_update_count"] != 0
            or self._metadata["heldout_teacher_identity_count"] != 0
            or tuple(row.record() for row in self._output_cells) != self._metadata["output_cells"]
            or tuple(row.record() for row in self._branches) != self._metadata["branches"]
            or sha256(
                _checkpoint_payload(self._metadata, self._excitatory, self._inhibitory)
            ).hexdigest()
            != self._checkpoint_sha256
        ):
            raise RuntimeError("prompt-selection V3 checkpoint authentication failed")


def _selected_channel(trace: PhysicalInputTrace, meter: _WorkMeter) -> str | None:
    if not trace.raw_events:
        meter.charge("relation-router-stop")
        return None
    observed = set(trace.raw_events[0].relation_cells)
    matches: list[str] = []
    for schema in protocol_v3.RELAY_SCHEMAS:
        candidate = set(protocol_v3._assembly_cells("relation", schema.channel))
        meter.charge("relation-router-cell-read", 24)
        if len(observed & candidate) == ACTIVE_FEATURE_CELL_COUNT:
            matches.append(schema.channel)
    meter.charge("relation-router-decision")
    return matches[0] if len(matches) == 1 else None


def _branch_feature(
    branch: OutputBranch, trace: PhysicalInputTrace, meter: _WorkMeter
) -> tuple[int, ...]:
    expected_relation = protocol_v3._assembly_cells("relation", branch.channel)
    expected_role = protocol_v3._assembly_cells("role", f"{branch.channel}|{branch.branch_role}")
    if branch.kind == "atom":
        matches = tuple(
            row.active_cells
            for row in trace.event_features
            if row.relation_cells == expected_relation and row.role_cells == expected_role
        )
        meter.charge("role-router-cell-read", len(trace.event_features) * 48)
    else:
        matches = tuple(
            row.active_cells
            for row in trace.transition_features
            if row.relation_cells == expected_relation and row.from_role_cells == expected_role
        )
        meter.charge("transition-router-cell-read", len(trace.transition_features) * 48)
    meter.charge("branch-feature-router-decision")
    return matches[0] if len(matches) == 1 else ()


def _increment_ticks(
    values: np.ndarray,
    row: int,
    active_cells: tuple[int, ...],
    *,
    arrivals_per_cell: int,
) -> int:
    committed = 0
    for cell in active_cells:
        previous = int(values[row, cell])
        updated = min(TICK_CAP, previous + arrivals_per_cell)
        values[row, cell] = updated
        committed += updated - previous
    return committed


def _base_metadata(*, teacher_enabled: bool, training_work: WorkLedger | None) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "public-train-prompt-selection-v3-compositional-checkpoint",
        "dependency_sha256": DEPENDENCY_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "output_registry_sha256": OUTPUT_REGISTRY_SHA256,
        "branch_registry_sha256": BRANCH_REGISTRY_SHA256,
        "output_cells": tuple(row.record() for row in OUTPUT_CELLS),
        "branches": tuple(row.record() for row in OUTPUT_BRANCHES),
        "array_shape": (len(OUTPUT_CELLS), INPUT_CELL_COUNT),
        "array_dtype": np.dtype(np.uint8).str,
        "tick_cap": TICK_CAP,
        "self_posed_teacher_lesson_count": (
            len(protocol_v3.PROTOCOL.lessons) if teacher_enabled else 0
        ),
        "canonical_fit_teacher_update_count": 0,
        "heldout_teacher_identity_count": 0,
        "training_schedule_sha256s": (
            tuple(row.sha256 for row in protocol_v3.PROTOCOL.lessons) if teacher_enabled else ()
        ),
        "schedule_hashes_are_authentication_only": True,
        "whole_target_head_count": WHOLE_TARGET_HEAD_COUNT,
    }
    if training_work is not None:
        record["training_work"] = training_work.record()
    return record


def _train_workspace(
    *,
    teachers: tuple[protocol_v3.SelfPosedRelayLesson, ...],
    teacher_enabled: bool,
    require_teacher_route: bool = True,
) -> tuple[FrozenCompositionalWorkspace, WorkLedger]:
    lessons = protocol_v3.PROTOCOL.lessons
    if len(lessons) != 138 or len(teachers) != len(lessons):
        raise RuntimeError("V3 training requires the complete 138-lesson curriculum")
    excitatory = np.zeros((len(OUTPUT_CELLS), INPUT_CELL_COUNT), dtype=np.uint8)
    inhibitory = np.zeros_like(excitatory)
    meter = _WorkMeter()
    for lesson, teacher in zip(lessons, teachers, strict=True):
        trace = _encode_trace(lesson, mode="normal", meter=meter)
        channel = _selected_channel(trace, meter)
        if channel != lesson.audit_channel or teacher.audit_channel != channel:
            raise RuntimeError("physical relation router and teacher channel diverged")
        teacher_targets = set((*teacher.target_atom_ids, *teacher.target_transition_ids))
        for branch in (row for row in OUTPUT_BRANCHES if row.channel == channel):
            feature = _branch_feature(branch, trace, meter)
            target_cells = tuple(
                cell_id
                for cell_id in branch.candidate_cell_ids
                if OUTPUT_CELLS[cell_id].target_id in teacher_targets
            )
            if len(target_cells) > 1 or (
                require_teacher_route and bool(feature) != bool(target_cells)
            ):
                raise RuntimeError("teacher target left the routed channel-local branch")
            if not teacher_enabled:
                continue
            meter.charge("teacher-cell-spike")
            if target_cells and feature:
                target_cell = target_cells[0]
                meter.charge(
                    "excitatory-tick-commit",
                    _increment_ticks(
                        excitatory,
                        target_cell,
                        feature,
                        arrivals_per_cell=2,
                    ),
                )
                for rival in branch.candidate_cell_ids:
                    if rival == target_cell:
                        continue
                    meter.charge(
                        "inhibitory-tick-commit",
                        _increment_ticks(
                            inhibitory,
                            rival,
                            feature,
                            arrivals_per_cell=1,
                        ),
                    )
            elif feature:
                for rival in branch.candidate_cell_ids:
                    meter.charge(
                        "inhibitory-tick-commit",
                        _increment_ticks(
                            inhibitory,
                            rival,
                            feature,
                            arrivals_per_cell=1,
                        ),
                    )
    work = meter.freeze()
    workspace = FrozenCompositionalWorkspace(
        output_cells=OUTPUT_CELLS,
        branches=OUTPUT_BRANCHES,
        excitatory=excitatory,
        inhibitory=inhibitory,
        metadata=_base_metadata(teacher_enabled=teacher_enabled, training_work=work),
    )
    return workspace, work


_RACE = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)


def _decide_branch(
    workspace: FrozenCompositionalWorkspace,
    branch: OutputBranch,
    feature: tuple[int, ...],
    meter: _WorkMeter,
) -> BranchDecision:
    if feature:
        candidates = np.asarray(branch.candidate_cell_ids, dtype=np.intp)
        active = np.asarray(feature, dtype=np.intp)
        raw_array = np.sum(
            workspace._excitatory[candidates][:, active].astype(np.int16)
            - workspace._inhibitory[candidates][:, active].astype(np.int16),
            axis=1,
            dtype=np.int64,
        )
        meter.charge("synaptic-tick-read", len(branch.candidate_cell_ids) * len(feature) * 2)
        raw = tuple(int(value) for value in raw_array)
    else:
        raw = tuple(0 for _cell in branch.candidate_cell_ids)
    current_scores = (
        *(min(max(score, 0), len(v4spec.PROTOCOL.latency_table) - 1) for score in raw),
        STOP_SCORE,
    )
    currents = np.asarray(
        tuple(v4spec.PROTOCOL.latency_table[score].current_pa for score in current_scores),
        dtype=np.float64,
    )
    race = _RACE.race_many(currents[None, :], require_unique=False)[0]
    meter.charge("principal-current-write", len(current_scores))
    meter.charge("glif-principal-cell-step", len(current_scores) * RACE_STEPS)
    meter.charge("masked-wta-decision")
    if race.simultaneous_spike_candidates != 1:
        decision, selected_cell, selected_target = "COLLISION", None, None
    elif race.winner_cell == len(branch.candidate_cell_ids):
        decision, selected_cell, selected_target = "STOP", None, None
    else:
        selected_cell = branch.candidate_cell_ids[race.winner_cell]
        selected_target = workspace.output_cells[selected_cell].target_id
        decision = "RELAY"
    record = {
        "kind": branch.kind,
        "channel": branch.channel,
        "branch_role": branch.branch_role,
        "decision": decision,
        "selected_cell_id": selected_cell,
        "selected_target_id": selected_target,
        "raw_candidate_scores": raw,
        "first_spike_step": race.spike_step,
        "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
    }
    return BranchDecision(
        **record,
        sha256=_namespaced_sha256(_DECISION_NAMESPACE, record),
    )


def _replay_batch(
    workspace: FrozenCompositionalWorkspace,
    *,
    name: str,
    mode: str,
) -> ReplayBatch:
    workspace.authenticate()
    meter = _WorkMeter()
    rows: list[LessonReplay] = []
    for lesson in protocol_v3.PROTOCOL.lessons:
        trace = _encode_trace(lesson, mode=mode, meter=meter)
        channel = _selected_channel(trace, meter)
        decisions: tuple[BranchDecision, ...]
        if channel is None:
            decisions = ()
        else:
            decisions = tuple(
                _decide_branch(
                    workspace,
                    branch,
                    _branch_feature(branch, trace, meter),
                    meter,
                )
                for branch in OUTPUT_BRANCHES
                if branch.channel == channel
            )
        emitted = tuple(
            sorted(
                row.selected_target_id for row in decisions if row.selected_target_id is not None
            )
        )
        expected = tuple(sorted((*lesson.target_atom_ids, *lesson.target_transition_ids)))
        record = {
            "lesson_index": lesson.lesson_index,
            "channel": lesson.audit_channel,
            "trace_sha256": trace.sha256,
            "decisions": tuple(row.record() for row in decisions),
            "emitted_target_ids": emitted,
            "expected_target_ids": expected,
            "exact": emitted == expected,
        }
        rows.append(
            LessonReplay(
                lesson_index=lesson.lesson_index,
                channel=lesson.audit_channel,
                trace_sha256=trace.sha256,
                decisions=decisions,
                emitted_target_ids=emitted,
                expected_target_ids=expected,
                exact=emitted == expected,
                sha256=_namespaced_sha256(_REPLAY_NAMESPACE, record),
            )
        )
    workspace.authenticate()
    channel_rows = tuple(
        ChannelOutcome(
            channel=schema.channel,
            lesson_count=sum(row.channel == schema.channel for row in rows),
            exact_lesson_count=sum(row.channel == schema.channel and row.exact for row in rows),
            mismatch_lesson_count=sum(
                row.channel == schema.channel and not row.exact for row in rows
            ),
        )
        for schema in protocol_v3.RELAY_SCHEMAS
    )
    work = meter.freeze()
    exact = sum(row.exact for row in rows)
    record = {
        "name": name,
        "lesson_count": len(rows),
        "exact_lesson_count": exact,
        "mismatch_lesson_count": len(rows) - exact,
        "channel_outcomes": tuple(row.record() for row in channel_rows),
        "replays": tuple(row.record() for row in rows),
        "work": work.record(),
    }
    return ReplayBatch(
        name=name,
        lesson_count=len(rows),
        exact_lesson_count=exact,
        mismatch_lesson_count=len(rows) - exact,
        channel_outcomes=channel_rows,
        replays=tuple(rows),
        work=work,
        sha256=_namespaced_sha256(_BATCH_NAMESPACE, record),
    )


def _control_outcome(
    name: str,
    workspace: FrozenCompositionalWorkspace,
    replay: ReplayBatch,
    extra_work: WorkLedger | None = None,
) -> ControlOutcome:
    meter = _WorkMeter()
    meter.merge(replay.work)
    if extra_work is not None:
        meter.merge(extra_work)
    work = meter.freeze()
    record = {
        "name": name,
        "exact_lesson_count": replay.exact_lesson_count,
        "mismatch_lesson_count": replay.mismatch_lesson_count,
        "checkpoint_sha256": workspace.checkpoint_sha256,
        "work": work.record(),
    }
    return ControlOutcome(
        name=name,
        exact_lesson_count=replay.exact_lesson_count,
        mismatch_lesson_count=replay.mismatch_lesson_count,
        checkpoint_sha256=workspace.checkpoint_sha256,
        work=work,
        sha256=_namespaced_sha256(_CONTROL_NAMESPACE, record),
    )


def _production_for_slot(slot: object, root_production: str) -> str:
    value = str(slot)
    if value == "root":
        return root_production
    rules = (
        (
            ("parameter-arg0", "collection-arg0", "string-arg0", "bound-item0", "bound-char0"),
            "parameter",
        ),
        (("literal", "implicit-zero"), "literal"),
        (("value-binder",), "value-binder"),
        (("predicate-binder",), "predicate-binder"),
        (("bound-add",), "add"),
        (("bound-absolute",), "absolute"),
        (("greater-than-as-less-than",), "less-than"),
        (("contains-class",), "contains"),
        (("mapped-clamp",), "map"),
        (("filtered-string",), "filter"),
        (("lower-extremum",), "maximum"),
        (("upper-extremum",), "minimum"),
    )
    for fragments, production in rules:
        if any(fragment in value for fragment in fragments):
            return production
    return f"UNSUPPORTED:{value}"


def _source_role(value: object) -> str:
    text = str(value)
    if "signature-parameter" in text:
        return "input-parameter"
    if "bound-item" in text:
        return "bound-item"
    if "bound-char" in text:
        return "bound-char"
    if "predicate-binder" in text:
        return "predicate"
    if "compiler-ascii-class" in text:
        return "class"
    if "nested-" in text:
        return "nested-result"
    if "literal" in text or "constant" in text:
        return "literal"
    if "value-binder" in text:
        return "collection"
    return f"UNSUPPORTED:{text}"


def _parameter_slot(value: object) -> str:
    return {
        "parameter-arg0": "input-parameter",
        "collection-arg0": "collection-parameter",
        "string-arg0": "text-parameter",
    }.get(str(value), f"UNSUPPORTED:{value}")


def _literal_slot(value: object) -> str:
    text = str(value)
    if "lower" in text or text == "implicit-zero":
        return "lower-bound"
    if "upper" in text:
        return "upper-bound"
    if "offset" in text:
        return "offset"
    if "threshold" in text:
        return "threshold"
    if "ascii-class" in text:
        return "ascii-class"
    return f"UNSUPPORTED:{text}"


def _copy_role(value: object) -> str:
    text = str(value)
    return "emit-ascii-class" if "ascii-class" in text else "emit-int"


_ATOM_IDS = {row.atom_id for row in protocol_v3.OUTPUT_ATOMS}
_TRANSITION_IDS = {row.transition_id for row in protocol_v3.OUTPUT_TRANSITIONS}


def _project_label(
    label: _CanonicalRelayLabel, root_production: str
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    channel = str(label.channel)
    fields = label.fields
    if not isinstance(fields, Mapping):
        raise RuntimeError("canonical FIT relay fields are not immutable records")
    if channel == "root-production-family":
        bindings = {
            "root-production": str(fields["root_production"]),
            "root-family": str(fields["root_family"]),
            "input-type": str(fields["input_type"]),
            "output-type": str(fields["output_type"]),
        }
    elif channel == "ordered-parent-child-edge":
        bindings = {
            "parent-production": _production_for_slot(fields["parent_slot"], root_production),
            "child-production": _production_for_slot(fields["child_slot"], root_production),
            "child-ordinal": str(fields["child_ordinal"]),
        }
    elif channel == "argument-operand-orientation":
        bindings = {
            "operator-production": _production_for_slot(fields["operator_slot"], root_production),
            "operand-ordinal": str(fields["operand_ordinal"]),
            "source-role": _source_role(fields["source_role"]),
        }
    elif channel == "nesting-relation":
        bindings = {
            "outer-production": _production_for_slot(fields["outer_slot"], root_production),
            "inner-production": _production_for_slot(fields["inner_slot"], root_production),
            "depth": str(fields["nesting_depth"]),
        }
    elif channel == "parameter-role-binding":
        bindings = {
            "semantic-slot": _parameter_slot(fields["semantic_argument_slot"]),
            "source-role": "function-argument",
        }
    elif channel == "literal-slot-binding":
        bindings = {
            "literal-source-role": (
                "generic-literal"
                if type(fields["literal_source_ordinal"]) is int
                else "compiler-constant"
            ),
            "semantic-slot": _literal_slot(fields["semantic_literal_slot"]),
        }
    elif channel == "literal-value-copy":
        bindings = {
            "semantic-slot": _literal_slot(fields["semantic_literal_slot"]),
            "copy-role": _copy_role(fields["typed_output_copy_role"]),
        }
    elif channel == "terminal-root-eclass-agreement":
        bindings = {
            "root-authority": "selected-root",
            "eclass-authority": "retained-eclass",
            "agreement-state": "exact-object-identity",
        }
    else:
        raise RuntimeError("canonical FIT relay left the eight-channel vocabulary")
    schema = next(row for row in protocol_v3.RELAY_SCHEMAS if row.channel == channel)
    atom_ids = tuple(
        f"{channel}|{slot.role}|{slot.filler_kind}|{bindings[slot.role]}" for slot in schema.slots
    )
    transition_ids = tuple(
        f"{channel}|{left.role}->{right.role}" for left, right in pairwise(schema.slots)
    )
    missing = tuple(
        target
        for target in (*atom_ids, *transition_ids)
        if target not in (_ATOM_IDS if target in atom_ids else _TRANSITION_IDS)
    )
    return channel, tuple(sorted(set((*atom_ids, *transition_ids)))), missing


def _build_fit_projection_audit() -> CanonicalFitProjectionAudit:
    # Lazy imports guarantee the self-posed checkpoint exists before any
    # structural capsule object, including heldout annotations, is materialized.
    from experiments.r3_pilot import (
        public_train_prompt_selection_freeze_v1_protocol as selection_v1,
    )
    from experiments.r3_pilot import (
        public_train_structural_annotation_capsule_v1 as annotations_v1,
    )

    if (
        selection_v1.PROTOCOL.protocol_sha256 != EXPECTED_SELECTION_V1_SHA256
        or annotations_v1.CAPSULE.capsule_sha256 != EXPECTED_CAPSULE_SHA256
    ):
        raise RuntimeError("canonical FIT projection dependency changed")
    fit_keys = tuple(
        row.causal_key_sha256 for row in selection_v1.PROTOCOL.canonical_inputs if row.split == FIT
    )
    if len(fit_keys) != 24:
        raise RuntimeError("canonical FIT identity inventory changed")
    annotation_by_key = {row.causal_key_sha256: row for row in annotations_v1.CAPSULE.annotations}
    meter = _WorkMeter()
    identity_missing: list[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]] = []
    channel_missing_identities: Counter[str] = Counter()
    channel_missing_occurrences: Counter[str] = Counter()
    all_missing: list[str] = []
    for causal_key in fit_keys:
        annotation = annotation_by_key[causal_key]
        root_label = next(
            row for row in annotation.relay_labels if row.channel == "root-production-family"
        )
        root_production = str(root_label.fields["root_production"])
        missing_by_channel: dict[str, list[str]] = defaultdict(list)
        for label in annotation.relay_labels:
            channel, _targets, missing = _project_label(label, root_production)
            meter.charge("canonical-fit-relay-field-read", len(label.fields))
            meter.charge("canonical-fit-vocabulary-lookup", len(_targets))
            if missing:
                missing_by_channel[channel].extend(missing)
                all_missing.extend(missing)
        frozen_missing = tuple(
            (channel, tuple(sorted(values)))
            for channel, values in sorted(missing_by_channel.items())
        )
        identity_missing.append((causal_key, frozen_missing))
        for channel, values in frozen_missing:
            channel_missing_identities[channel] += 1
            channel_missing_occurrences[channel] += len(values)
    channel_outcomes = tuple(
        ProjectionChannelOutcome(
            channel=schema.channel,
            fit_identity_count=24,
            supported_identity_count=24 - channel_missing_identities[schema.channel],
            unsupported_identity_count=channel_missing_identities[schema.channel],
            missing_atom_occurrence_count=channel_missing_occurrences[schema.channel],
        )
        for schema in protocol_v3.RELAY_SCHEMAS
    )
    unsupported = sum(bool(missing) for _key, missing in identity_missing)
    status = STOP_FIT_PROJECTION if unsupported else PASS
    work = meter.freeze()
    record = {
        "status": status,
        "fit_identity_count": len(fit_keys),
        "projectable_identity_count": len(fit_keys) - unsupported,
        "unsupported_identity_count": unsupported,
        "channel_outcomes": tuple(row.record() for row in channel_outcomes),
        "missing_atom_ids": tuple(sorted(set(all_missing))),
        "missing_atom_occurrence_count": len(all_missing),
        "canonical_teacher_update_count": CANONICAL_FIT_TEACHER_UPDATE_COUNT,
        "heldout_target_read_count": HELDOUT_TARGET_READ_COUNT,
        "heldout_replay_identity_count": HELDOUT_REPLAY_IDENTITY_COUNT,
        "work": work.record(),
    }
    return CanonicalFitProjectionAudit(
        status=status,
        fit_identity_count=len(fit_keys),
        projectable_identity_count=len(fit_keys) - unsupported,
        unsupported_identity_count=unsupported,
        channel_outcomes=channel_outcomes,
        missing_atom_ids=tuple(sorted(set(all_missing))),
        missing_atom_occurrence_count=len(all_missing),
        canonical_teacher_update_count=CANONICAL_FIT_TEACHER_UPDATE_COUNT,
        heldout_target_read_count=HELDOUT_TARGET_READ_COUNT,
        heldout_replay_identity_count=HELDOUT_REPLAY_IDENTITY_COUNT,
        work=work,
        sha256=_namespaced_sha256(_PROJECTION_NAMESPACE, record),
    )


def _build_result() -> tuple[
    PublicTrainPromptSelectionV3Result,
    FrozenCompositionalWorkspace,
    Mapping[str, object],
    str,
    str,
]:
    lessons = protocol_v3.PROTOCOL.lessons
    untrained_workspace, teacher_off_training = _train_workspace(
        teachers=lessons, teacher_enabled=False
    )
    teacher_off_replay = _replay_batch(untrained_workspace, name="teacher-off", mode="normal")
    treatment_workspace, training_work = _train_workspace(teachers=lessons, teacher_enabled=True)
    self_posed = _replay_batch(treatment_workspace, name="self-posed-treatment", mode="normal")
    cell_off = _replay_batch(treatment_workspace, name="cell-off", mode="cell-off")
    order_control = _replay_batch(treatment_workspace, name="reverse-order", mode="reverse-order")
    role_control = _replay_batch(treatment_workspace, name="role-swap", mode="role-swap")
    filler_control = _replay_batch(treatment_workspace, name="filler-swap", mode="filler-swap")
    lessons_by_channel: dict[str, list[protocol_v3.SelfPosedRelayLesson]] = defaultdict(list)
    for lesson in lessons:
        lessons_by_channel[lesson.audit_channel].append(lesson)
    deranged_by_index: dict[int, protocol_v3.SelfPosedRelayLesson] = {}
    for channel_lessons in lessons_by_channel.values():
        for index, lesson in enumerate(channel_lessons):
            deranged_by_index[lesson.lesson_index] = channel_lessons[
                (index + 1) % len(channel_lessons)
            ]
    deranged_teachers = tuple(deranged_by_index[index] for index in range(len(lessons)))
    deranged_workspace, deranged_training = _train_workspace(
        teachers=deranged_teachers,
        teacher_enabled=True,
        require_teacher_route=False,
    )
    deranged_replay = _replay_batch(deranged_workspace, name="teacher-derangement", mode="normal")
    controls = (
        _control_outcome(
            "teacher-off",
            untrained_workspace,
            teacher_off_replay,
            teacher_off_training,
        ),
        _control_outcome("cell-off", treatment_workspace, cell_off),
        _control_outcome("reverse-order", treatment_workspace, order_control),
        _control_outcome("role-swap", treatment_workspace, role_control),
        _control_outcome("filler-swap", treatment_workspace, filler_control),
        _control_outcome(
            "teacher-derangement",
            deranged_workspace,
            deranged_replay,
            deranged_training,
        ),
    )
    control_pass = all(row.exact_lesson_count < len(lessons) for row in controls)
    if self_posed.exact_lesson_count != len(lessons):
        status = STOP_SELF_POSED
        fit_projection = CanonicalFitProjectionAudit(
            status=STOP_SELF_POSED,
            fit_identity_count=0,
            projectable_identity_count=0,
            unsupported_identity_count=0,
            channel_outcomes=(),
            missing_atom_ids=(),
            missing_atom_occurrence_count=0,
            canonical_teacher_update_count=0,
            heldout_target_read_count=0,
            heldout_replay_identity_count=0,
            work=_WorkMeter().freeze(),
            sha256=_namespaced_sha256(_PROJECTION_NAMESPACE, {"status": STOP_SELF_POSED}),
        )
    elif not control_pass:
        status = STOP_CONTROLS
        fit_projection = CanonicalFitProjectionAudit(
            status=STOP_CONTROLS,
            fit_identity_count=0,
            projectable_identity_count=0,
            unsupported_identity_count=0,
            channel_outcomes=(),
            missing_atom_ids=(),
            missing_atom_occurrence_count=0,
            canonical_teacher_update_count=0,
            heldout_target_read_count=0,
            heldout_replay_identity_count=0,
            work=_WorkMeter().freeze(),
            sha256=_namespaced_sha256(_PROJECTION_NAMESPACE, {"status": STOP_CONTROLS}),
        )
    else:
        # The treatment checkpoint is already immutable at this point.
        frozen_before_projection = treatment_workspace.checkpoint_sha256
        fit_projection = _build_fit_projection_audit()
        treatment_workspace.authenticate()
        if treatment_workspace.checkpoint_sha256 != frozen_before_projection:
            raise RuntimeError("checkpoint changed during canonical FIT projection")
        status = fit_projection.status

    checkpoint = CheckpointSummary(
        output_cell_count=len(OUTPUT_CELLS),
        atom_cell_count=len(protocol_v3.OUTPUT_ATOMS),
        transition_cell_count=len(protocol_v3.OUTPUT_TRANSITIONS),
        branch_count=len(OUTPUT_BRANCHES),
        array_shape=treatment_workspace._excitatory.shape,
        array_dtype=treatment_workspace._excitatory.dtype.str,
        array_bytes=(
            treatment_workspace._excitatory.nbytes + treatment_workspace._inhibitory.nbytes
        ),
        tick_cap=TICK_CAP,
        excitatory_nonzero_ticks=int(np.count_nonzero(treatment_workspace._excitatory)),
        inhibitory_nonzero_ticks=int(np.count_nonzero(treatment_workspace._inhibitory)),
        self_posed_teacher_lesson_count=len(lessons),
        canonical_fit_teacher_update_count=0,
        heldout_teacher_identity_count=0,
        checkpoint_sha256=treatment_workspace.checkpoint_sha256,
    )
    control_root = _namespaced_sha256(
        _CONTROL_PANEL_NAMESPACE, tuple(row.record() for row in controls)
    )
    surface_record = {
        "neural_input_sources": (
            "physical-relation-assemblies",
            "physical-role-assemblies",
            "physical-filler-assemblies",
            "ordered-role-transition-assemblies",
        ),
        "neural_output": "channel-local compositional atoms and ordered transitions",
        "self_posed_teacher_lesson_count": len(lessons),
        "canonical_fit_teacher_update_count": 0,
        "heldout_teacher_identity_count": 0,
        "heldout_target_read_count": 0,
        "heldout_replay_identity_count": 0,
        "learner_array_count": 2,
        "learner_array_bytes": checkpoint.array_bytes,
        "learning_pass_count": 1,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_output_read_count": DESIRED_OUTPUT_READ_COUNT,
        "program_artifact_read_count": PROGRAM_ARTIFACT_READ_COUNT,
        "executable_source_read_count": EXECUTABLE_SOURCE_READ_COUNT,
        "ast_artifact_read_count": AST_ARTIFACT_READ_COUNT,
        "property_row_read_count": PROPERTY_ROW_READ_COUNT,
        "task_identifier_neural_input_count": TASK_IDENTIFIER_NEURAL_INPUT_COUNT,
        "family_identifier_neural_input_count": FAMILY_IDENTIFIER_NEURAL_INPUT_COUNT,
        "entrypoint_spelling_neural_input_count": ENTRYPOINT_SPELLING_NEURAL_INPUT_COUNT,
        "audit_hash_neural_input_count": AUDIT_HASH_NEURAL_INPUT_COUNT,
        "prompt_factor_inference_count": PROMPT_FACTOR_INFERENCE_COUNT,
        "whole_target_head_count": WHOLE_TARGET_HEAD_COUNT,
        "host_label_assembly_count": HOST_LABEL_ASSEMBLY_COUNT,
        "host_lookup_selection_count": HOST_LOOKUP_SELECTION_COUNT,
        "host_union_selection_count": HOST_UNION_SELECTION_COUNT,
        "host_ranking_count": HOST_RANKING_COUNT,
        "candidate_program_count": CANDIDATE_PROGRAM_COUNT,
        "proof_count": PROOF_COUNT,
        "checkpoint_frozen_before_canonical_fit_projection": True,
        "fit_projection_atomic_before_teacher_updates": True,
        "heldout_surface_opened": False,
    }
    surface_sha = _namespaced_sha256(_SURFACE_NAMESPACE, surface_record)
    total_meter = _WorkMeter()
    total_meter.merge(training_work)
    total_meter.merge(self_posed.work)
    total_meter.merge(fit_projection.work)
    for control in controls:
        total_meter.merge(control.work)
    physical_work_total = total_meter.freeze().total
    root_without_result = {
        "scope": PROTOCOL_SCOPE,
        "status": status,
        "dependency_sha256": DEPENDENCY_SHA256,
        "output_registry_sha256": OUTPUT_REGISTRY_SHA256,
        "branch_registry_sha256": BRANCH_REGISTRY_SHA256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "self_posed_result_sha256": self_posed.sha256,
        "control_panel_sha256": control_root,
        "fit_projection_sha256": fit_projection.sha256,
        "surface_boundary_sha256": surface_sha,
        "training_work": training_work.record(),
        "physical_work_total": physical_work_total,
        "self_posed_lesson_count": self_posed.lesson_count,
        "self_posed_exact_lesson_count": self_posed.exact_lesson_count,
        "canonical_fit_identity_count": fit_projection.fit_identity_count,
        "canonical_fit_projectable_identity_count": fit_projection.projectable_identity_count,
        "canonical_fit_teacher_update_count": fit_projection.canonical_teacher_update_count,
        "heldout_target_read_count": fit_projection.heldout_target_read_count,
        "heldout_replay_identity_count": fit_projection.heldout_replay_identity_count,
        "checkpoint": checkpoint.record(),
    }
    result = PublicTrainPromptSelectionV3Result(
        scope=PROTOCOL_SCOPE,
        status=status,
        checkpoint=checkpoint,
        self_posed_result=self_posed,
        controls=controls,
        fit_projection_audit=fit_projection,
        heldout_channel_outcomes=(),
        training_work=training_work,
        physical_work_total=physical_work_total,
        dependency_sha256=DEPENDENCY_SHA256,
        output_registry_sha256=OUTPUT_REGISTRY_SHA256,
        branch_registry_sha256=BRANCH_REGISTRY_SHA256,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        self_posed_result_sha256=self_posed.sha256,
        control_panel_sha256=control_root,
        fit_projection_sha256=fit_projection.sha256,
        surface_boundary_sha256=surface_sha,
        result_sha256=_namespaced_sha256(_RESULT_NAMESPACE, root_without_result),
    )
    return (
        result,
        treatment_workspace,
        _freeze_record(surface_record),
        surface_sha,
        control_root,
    )


(
    RESULT,
    WORKSPACE,
    SURFACE_BOUNDARY_RECORD,
    SURFACE_BOUNDARY_SHA256,
    CONTROL_PANEL_SHA256,
) = _build_result()
STATUS = RESULT.status
CHECKPOINT = RESULT.checkpoint
SELF_POSED_RESULT = RESULT.self_posed_result
CONTROLS = RESULT.controls
FIT_PROJECTION_AUDIT = RESULT.fit_projection_audit


__all__ = [
    "BRANCH_REGISTRY_SHA256",
    "CHECKPOINT",
    "CONTROLS",
    "CONTROL_PANEL_SHA256",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "FIT_PROJECTION_AUDIT",
    "OUTPUT_BRANCHES",
    "OUTPUT_CELLS",
    "OUTPUT_REGISTRY_SHA256",
    "PASS",
    "PROTOCOL_SCOPE",
    "RESULT",
    "SELF_POSED_RESULT",
    "STATUS",
    "STOP_CONTROLS",
    "STOP_FIT",
    "STOP_FIT_PROJECTION",
    "STOP_HELDOUT",
    "STOP_SELF_POSED",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "WORKSPACE",
    "CanonicalFitProjectionAudit",
    "CheckpointSummary",
    "FrozenCompositionalWorkspace",
    "PublicTrainPromptSelectionV3Result",
]
