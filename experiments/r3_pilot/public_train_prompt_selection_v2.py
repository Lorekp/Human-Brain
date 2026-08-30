"""Public-TRAIN prompt selection V2 learner/result.

The module spends the frozen 24/18 identity split once.  Only ordered physical
span events, typed-literal cell events, and name-free signature role/type events
drive a fixed twelve-lane recurrent trace.  FIT teachers update channel-local,
masked relay workspaces with the banked V5 uint8 cap-four excitatory/inhibitory
rule.  Every retained branch is selected by a physical first-spike GLIF race;
host code only authenticates and admits the already-selected relay-cell set.

V1 and the structural annotation capsule remain immutable dependencies.  This
module reads no desired output, executable program, AST/source artifact,
property row, DEV/protected input, task/family identity, entrypoint spelling,
or audit identity as neural evidence.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
from experiments.r3_pilot import exclusive_factor_workspace_v4 as v4impl
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as v4spec
from experiments.r3_pilot import public_train_prompt_factor_manifest_v1 as prompt_manifest
from experiments.r3_pilot import public_train_prompt_selection_freeze_v1_protocol as selection_v1
from experiments.r3_pilot import public_train_structural_annotation_capsule_v1 as annotations_v1
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as v5spec
from humanbrain.resting_glif_race import CompiledRestingGlifRace

PROTOCOL_SCOPE = "PUBLIC_TRAIN_PROMPT_SELECTION_V2"
PASS = "PASS_PUBLIC_TRAIN_PROMPT_SELECTION_V2"
STOP_FIT = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V2_FIT"
STOP_HELDOUT = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V2_HELDOUT"
STOP_CONTROLS = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V2_CAUSAL_CONTROLS"

FIT = "FIT"
HELDOUT = "HELDOUT"
TRACE_LANE_COUNT = 12
TRACE_CELLS_PER_LANE = 512
TRACE_CELL_COUNT = TRACE_LANE_COUNT * TRACE_CELLS_PER_LANE
TICK_CAP = 4
STOP_SCORE = 12
RACE_STEPS = v4spec.RACE_STEPS

EXPECTED_SELECTION_V1_SHA256 = "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47"
EXPECTED_ANNOTATION_CAPSULE_SHA256 = (
    "484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf"
)
EXPECTED_V5_PROTOCOL_SHA256 = "587bd68a293ca2116570cf4e5ee7a860bbdd9928138982ae43748c4f87a252c1"

_PINNED_SOURCE_SHA256S = MappingProxyType(
    {
        "experiments/r3_pilot/public_train_prompt_selection_freeze_v1_protocol.py": (
            "a35ee1de48bea57404d101b69ec729d36470a88a55caa2aff5163dde246894ef"
        ),
        "experiments/r3_pilot/public_train_structural_annotation_capsule_v1.py": (
            "d2e977b3aff2e496307168ed64e85c2ebbc03579224bf58b0fd175f0ed096c88"
        ),
        "experiments/r3_pilot/teacher_gated_factor_workspace_v5.py": (
            "c76292164b43586b9400f8277f1c2aa71b2a136f113f36bc3f89712f0ca6895d"
        ),
    }
)

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
HOST_LOOKUP_SELECTION_COUNT = 0
HOST_UNION_SELECTION_COUNT = 0
HOST_RANKING_COUNT = 0
CANDIDATE_PROGRAM_COUNT = 0
PROOF_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]
_CHANNELS = tuple(channel.channel for channel in selection_v1.RELAY_TARGET_VOCABULARY)
_CHANNEL_ORDER = {channel: index for index, channel in enumerate(_CHANNELS)}
_TRACE_MULTIPLIERS = np.asarray(
    (257, 263, 269, 271, 277, 281, 283, 293, 299, 301, 303, 307),
    dtype=np.int64,
)
_TRACE_POSITION_GAINS = np.asarray(
    (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41),
    dtype=np.int64,
)
_TRACE_BIASES = np.asarray(
    (17, 43, 71, 101, 137, 173, 211, 251, 293, 337, 383, 431),
    dtype=np.int64,
)

_DEPENDENCY_NAMESPACE = b"public-train-prompt-selection-v2-dependency\0"
_TRACE_NAMESPACE = b"public-train-prompt-selection-v2-recurrent-trace\0"
_LABEL_REGISTRY_NAMESPACE = b"public-train-prompt-selection-v2-label-registry\0"
_BRANCH_REGISTRY_NAMESPACE = b"public-train-prompt-selection-v2-branch-registry\0"
_WORK_NAMESPACE = b"public-train-prompt-selection-v2-work\0"
_CHECKPOINT_NAMESPACE = b"public-train-prompt-selection-v2-checkpoint\0"
_DECISION_NAMESPACE = b"public-train-prompt-selection-v2-decision\0"
_IDENTITY_REPLAY_NAMESPACE = b"public-train-prompt-selection-v2-identity-replay\0"
_BATCH_NAMESPACE = b"public-train-prompt-selection-v2-batch\0"
_CONTROL_NAMESPACE = b"public-train-prompt-selection-v2-control\0"
_CONTROL_PANEL_NAMESPACE = b"public-train-prompt-selection-v2-control-panel\0"
_SURFACE_NAMESPACE = b"public-train-prompt-selection-v2-surface\0"
_RESULT_NAMESPACE = b"public-train-prompt-selection-v2-result\0"


def _plain_record(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain_record(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain_record(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _freeze_record(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_record(item) for key, item in value.items()})
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

    def freeze(self) -> WorkLedger:
        events = tuple(sorted(self._events.items()))
        total = sum(count for _event, count in events)
        record = {"events": events, "total": total}
        return WorkLedger(
            events=events,
            total=total,
            sha256=_namespaced_sha256(_WORK_NAMESPACE, record),
        )


@dataclass(frozen=True, slots=True)
class PhysicalTrace:
    causal_key_sha256: str
    split: str
    physical_event_count: int
    active_cells: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_key_sha256": self.causal_key_sha256,
            "split": self.split,
            "physical_event_count": self.physical_event_count,
            "active_cells": self.active_cells,
            "causal_key_is_authentication_only": True,
            "neural_sources": (
                "ordered-physical-span-events",
                "ordered-typed-literal-cell-events",
                "ordered-name-free-signature-role-type-events",
            ),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RelayLabelCell:
    cell_id: int
    channel: str
    label: annotations_v1.AbstractRelayLabel

    def record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "channel": self.channel,
            "label": self.label.record(),
        }


@dataclass(frozen=True, slots=True)
class RelayBranch:
    channel: str
    branch_key: tuple[object, ...]
    candidate_cell_ids: tuple[int, ...]

    def record(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "branch_key": self.branch_key,
            "candidate_cell_ids": self.candidate_cell_ids,
            "tonic_stop_score": STOP_SCORE,
            "selection": "channel-local-masked-first-spike-wta",
        }


@dataclass(frozen=True, slots=True)
class BranchDecision:
    channel: str
    branch_key: tuple[object, ...]
    decision: str
    selected_cell_id: int | None
    selected_label: annotations_v1.AbstractRelayLabel | None
    raw_candidate_scores: tuple[int, ...]
    first_spike_step: int
    simultaneous_spike_candidates: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "channel": self.channel,
            "branch_key": self.branch_key,
            "decision": self.decision,
            "selected_cell_id": self.selected_cell_id,
            "selected_label": (
                self.selected_label.record() if self.selected_label is not None else None
            ),
            "raw_candidate_scores": self.raw_candidate_scores,
            "first_spike_step": self.first_spike_step,
            "simultaneous_spike_candidates": self.simultaneous_spike_candidates,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class IdentityReplay:
    causal_key_sha256: str
    split: str
    trace_sha256: str
    decisions: tuple[BranchDecision, ...]
    emitted_by_channel: tuple[tuple[str, tuple[str, ...]], ...]
    exact_by_channel: tuple[tuple[str, bool], ...]
    exact_target_set: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_key_sha256": self.causal_key_sha256,
            "split": self.split,
            "trace_sha256": self.trace_sha256,
            "decisions": tuple(row.record() for row in self.decisions),
            "emitted_by_channel": self.emitted_by_channel,
            "exact_by_channel": self.exact_by_channel,
            "exact_target_set": self.exact_target_set,
            "causal_key_is_admission_metadata_only": True,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ChannelAdmission:
    channel: str
    exact_identity_count: int
    mismatch_identity_count: int
    expected_label_count: int
    emitted_label_count: int

    def record(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "exact_identity_count": self.exact_identity_count,
            "mismatch_identity_count": self.mismatch_identity_count,
            "expected_label_count": self.expected_label_count,
            "emitted_label_count": self.emitted_label_count,
        }


@dataclass(frozen=True, slots=True)
class ReplayBatchResult:
    split: str
    identity_count: int
    exact_identity_count: int
    mismatch_identity_count: int
    channel_admissions: tuple[ChannelAdmission, ...]
    replays: tuple[IdentityReplay, ...]
    checkpoint_authentication_count: int
    work: WorkLedger
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "split": self.split,
            "identity_count": self.identity_count,
            "exact_identity_count": self.exact_identity_count,
            "mismatch_identity_count": self.mismatch_identity_count,
            "channel_admissions": tuple(row.record() for row in self.channel_admissions),
            "replays": tuple(row.record() for row in self.replays),
            "checkpoint_authentication_count": self.checkpoint_authentication_count,
            "work": self.work.record(),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class ControlResult:
    name: str
    exact_identity_count: int
    mismatch_identity_count: int
    stop_decision_count: int
    collision_decision_count: int
    checkpoint_sha256: str
    work: WorkLedger
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "exact_identity_count": self.exact_identity_count,
            "mismatch_identity_count": self.mismatch_identity_count,
            "stop_decision_count": self.stop_decision_count,
            "collision_decision_count": self.collision_decision_count,
            "checkpoint_sha256": self.checkpoint_sha256,
            "work": self.work.record(),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    candidate_label_cell_count: int
    branch_count: int
    array_shape: tuple[int, int]
    array_dtype: str
    array_bytes: int
    tick_cap: int
    excitatory_nonzero_ticks: int
    inhibitory_nonzero_ticks: int
    fit_teacher_identity_count: int
    heldout_teacher_identity_count: int
    checkpoint_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "candidate_label_cell_count": self.candidate_label_cell_count,
            "branch_count": self.branch_count,
            "array_shape": self.array_shape,
            "array_dtype": self.array_dtype,
            "array_bytes": self.array_bytes,
            "tick_cap": self.tick_cap,
            "excitatory_nonzero_ticks": self.excitatory_nonzero_ticks,
            "inhibitory_nonzero_ticks": self.inhibitory_nonzero_ticks,
            "fit_teacher_identity_count": self.fit_teacher_identity_count,
            "heldout_teacher_identity_count": self.heldout_teacher_identity_count,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True, slots=True)
class PublicTrainPromptSelectionV2Result:
    scope: str
    status: str
    checkpoint: CheckpointSummary
    fit_result: ReplayBatchResult
    heldout_result: ReplayBatchResult
    controls: tuple[ControlResult, ...]
    trace_work: WorkLedger
    training_work: WorkLedger
    physical_work_total: int
    dependency_sha256: str
    trace_registry_sha256: str
    label_registry_sha256: str
    branch_registry_sha256: str
    checkpoint_sha256: str
    fit_result_sha256: str
    heldout_result_sha256: str
    control_panel_sha256: str
    surface_boundary_sha256: str
    result_sha256: str

    def root_record(self, *, include_result_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "trace_registry_sha256": self.trace_registry_sha256,
            "label_registry_sha256": self.label_registry_sha256,
            "branch_registry_sha256": self.branch_registry_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "fit_result_sha256": self.fit_result_sha256,
            "heldout_result_sha256": self.heldout_result_sha256,
            "control_panel_sha256": self.control_panel_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "trace_work": self.trace_work.record(),
            "training_work": self.training_work.record(),
            "physical_work_total": self.physical_work_total,
            "fit_identity_count": self.fit_result.identity_count,
            "fit_exact_identity_count": self.fit_result.exact_identity_count,
            "heldout_identity_count": self.heldout_result.identity_count,
            "heldout_exact_identity_count": self.heldout_result.exact_identity_count,
            "checkpoint": self.checkpoint.record(),
        }
        if include_result_sha256:
            record["result_sha256"] = self.result_sha256
        return record


@dataclass(frozen=True, slots=True)
class _PhysicalInput:
    causal_key_sha256: str
    split: str
    span_events: tuple[Mapping[str, object], ...]
    literal_events: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]


def _branch_key(label: annotations_v1.AbstractRelayLabel) -> tuple[object, ...]:
    fields = label.fields
    if not isinstance(fields, Mapping):
        raise RuntimeError("relay label fields are not frozen records")
    if label.channel == "root-production-family":
        return ("root",)
    if label.channel == "ordered-parent-child-edge":
        return (fields["parent_slot"], fields["child_ordinal"])
    if label.channel == "argument-operand-orientation":
        return (fields["operator_slot"], fields["operand_ordinal"])
    if label.channel == "nesting-relation":
        return (fields["outer_slot"], fields["nesting_depth"])
    if label.channel == "parameter-role-binding":
        return (fields["signature_parameter_role"],)
    if label.channel == "literal-slot-binding":
        return (fields["literal_source_ordinal"],)
    if label.channel == "literal-value-copy":
        return (fields["semantic_literal_slot"],)
    if label.channel == "terminal-root-eclass-agreement":
        return ("agreement",)
    raise RuntimeError("relay label left the frozen eight-channel vocabulary")


def _authenticate_dependencies() -> None:
    if selection_v1.PROTOCOL.protocol_sha256 != EXPECTED_SELECTION_V1_SHA256:
        raise RuntimeError("spent prompt-selection V1 root changed")
    if annotations_v1.CAPSULE.capsule_sha256 != EXPECTED_ANNOTATION_CAPSULE_SHA256:
        raise RuntimeError("structural annotation capsule root changed")
    if annotations_v1.CAPSULE.status != annotations_v1.PASS_IDENTIFIABLE:
        raise RuntimeError("structural annotation capsule is not identifiable")
    if v5spec.PROTOCOL.protocol_sha256 != EXPECTED_V5_PROTOCOL_SHA256:
        raise RuntimeError("banked teacher-gated V5 protocol root changed")
    if (
        v5spec.TICK_CAP != TICK_CAP
        or v5spec.ASSEMBLY_ACTIVE_CELL_COUNT != TRACE_LANE_COUNT
        or v4spec.STOP_SCORE != STOP_SCORE
    ):
        raise RuntimeError("banked V5 local-rule physical constants changed")
    for relative_path, expected_sha in _PINNED_SOURCE_SHA256S.items():
        if sha256((_ROOT / relative_path).read_bytes()).hexdigest() != expected_sha:
            raise RuntimeError(f"pinned learner dependency source changed: {relative_path}")


_authenticate_dependencies()

DEPENDENCY_RECORD = _freeze_record(
    {
        "prompt_selection_v1_scope": selection_v1.PROTOCOL.scope,
        "prompt_selection_v1_status": selection_v1.PROTOCOL.status,
        "prompt_selection_v1_sha256": selection_v1.PROTOCOL.protocol_sha256,
        "prompt_selection_v1_signature_bindings_sha256": (
            selection_v1.PROTOCOL.signature_bindings_sha256
        ),
        "prompt_selection_v1_canonical_split_sha256": (
            selection_v1.PROTOCOL.canonical_split_sha256
        ),
        "annotation_capsule_scope": annotations_v1.CAPSULE.scope,
        "annotation_capsule_status": annotations_v1.CAPSULE.status,
        "annotation_capsule_sha256": annotations_v1.CAPSULE.capsule_sha256,
        "annotation_projection_sha256": annotations_v1.CAPSULE.teacher_projection_sha256,
        "banked_v5_protocol_sha256": v5spec.PROTOCOL.protocol_sha256,
        "banked_v5_local_rule_sha256": v5spec.PROTOCOL.local_rule_sha256,
        "pinned_source_sha256s": tuple(_PINNED_SOURCE_SHA256S.items()),
        "teacher_schedule": "exactly-one-ordered-pass-over-24-fit-causal-identities",
        "heldout_teacher_identity_count": 0,
        "causal_key_sha256_used_as_neural_input": False,
        "prompt_input_sha256_used_as_neural_input": False,
        "full_signature_used_as_neural_input": False,
        "split_used_as_neural_input": False,
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(_DEPENDENCY_NAMESPACE, dict(DEPENDENCY_RECORD))


def _load_physical_inputs() -> tuple[_PhysicalInput, ...]:
    observed = prompt_manifest.load_public_train_prompt_inputs()
    bindings = selection_v1.PROTOCOL.signature_bindings
    if len(observed) != len(bindings) or len(observed) != 90:
        raise RuntimeError("public-TRAIN physical prompt appearance count changed")
    grouped: dict[str, _PhysicalInput] = {}
    for prompt_input, binding in zip(observed, bindings, strict=True):
        if prompt_input.sha256 != binding.prompt_input_sha256:
            raise RuntimeError("physical prompt observation order changed")
        spans = selection_v1._physical_span_event_record(prompt_input)
        literals = selection_v1._typed_literal_event_record(prompt_input)
        span_root = selection_v1._namespaced_sha256(selection_v1._SPAN_EVENTS_NAMESPACE, spans)
        literal_root = selection_v1._namespaced_sha256(
            selection_v1._LITERAL_EVENTS_NAMESPACE,
            literals,
        )
        causal = binding.causal_key
        if (
            span_root != causal.physical_span_events_sha256
            or literal_root != causal.typed_literal_events_sha256
            or binding.signature_role_sha256 != causal.signature_role_events_sha256
        ):
            raise RuntimeError("recomputed physical causal events changed")
        row = _PhysicalInput(
            causal_key_sha256=causal.sha256,
            split="",
            span_events=tuple(_freeze_record(record) for record in spans),
            literal_events=tuple(_freeze_record(record) for record in literals),
            signature_role_record=binding.signature_role_record,
        )
        incumbent = grouped.setdefault(binding.prompt_input_sha256, row)
        if incumbent != row:
            raise RuntimeError("identical prompt input has divergent physical events")

    rows: list[_PhysicalInput] = []
    for canonical in selection_v1.PROTOCOL.canonical_inputs:
        row = grouped.get(canonical.prompt_input_sha256)
        if row is None or row.causal_key_sha256 != canonical.causal_key_sha256:
            raise RuntimeError("canonical identity lacks authenticated physical events")
        rows.append(
            _PhysicalInput(
                causal_key_sha256=row.causal_key_sha256,
                split=canonical.split,
                span_events=row.span_events,
                literal_events=row.literal_events,
                signature_role_record=row.signature_role_record,
            )
        )
    if (
        len(rows) != 42
        or sum(row.split == FIT for row in rows) != 24
        or sum(row.split == HELDOUT for row in rows) != 18
    ):
        raise RuntimeError("canonical 24/18 physical input inventory changed")
    return tuple(rows)


def _physical_event_stream(row: _PhysicalInput, *, reverse_span_order: bool) -> tuple[int, ...]:
    events: list[int] = []
    span_events = tuple(reversed(row.span_events)) if reverse_span_order else row.span_events
    events.extend((10_001, len(span_events)))
    for ordinal, span in enumerate(span_events):
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
    events.extend((20_001, len(row.literal_events)))
    for ordinal, literal in enumerate(row.literal_events):
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
    role = row.signature_role_record
    if (
        set(role)
        != {
            "parameters",
            "result",
            "role_binding",
            "entrypoint_spelling_present",
            "parameter_spelling_present",
        }
        or role["entrypoint_spelling_present"] is not False
        or role["parameter_spelling_present"] is not False
    ):
        raise RuntimeError("name-free signature role/type surface changed")
    role_bytes = _canonical_json(
        {
            "parameters": role["parameters"],
            "result": role["result"],
            "role_binding": role["role_binding"],
        }
    )
    events.extend((30_001, len(role_bytes), *(int(value) for value in role_bytes)))
    return tuple(events)


def _encode_trace(
    row: _PhysicalInput,
    *,
    reverse_span_order: bool,
    meter: _WorkMeter,
) -> PhysicalTrace:
    events = _physical_event_stream(row, reverse_span_order=reverse_span_order)
    state = np.asarray(
        tuple((lane * 37 + 11) % TRACE_CELLS_PER_LANE for lane in range(TRACE_LANE_COUNT)),
        dtype=np.int64,
    )
    for position, event in enumerate(events, start=1):
        state = (
            state * _TRACE_MULTIPLIERS
            + int(event)
            + position * _TRACE_POSITION_GAINS
            + _TRACE_BIASES
        ) % TRACE_CELLS_PER_LANE
    active = tuple(
        lane * TRACE_CELLS_PER_LANE + int(state[lane]) for lane in range(TRACE_LANE_COUNT)
    )
    if len(set(active)) != TRACE_LANE_COUNT:
        raise RuntimeError("twelve-lane recurrent trace lost occurrence identity")
    meter.charge("physical-causal-event", len(events))
    meter.charge("recurrent-cell-transition", len(events) * TRACE_LANE_COUNT)
    meter.charge("trace-cell-emission", TRACE_LANE_COUNT)
    record = {
        "causal_key_sha256": row.causal_key_sha256,
        "split": row.split,
        "physical_event_count": len(events),
        "active_cells": active,
        "causal_key_is_authentication_only": True,
        "neural_sources": (
            "ordered-physical-span-events",
            "ordered-typed-literal-cell-events",
            "ordered-name-free-signature-role-type-events",
        ),
        "reverse_span_order_control": reverse_span_order,
    }
    return PhysicalTrace(
        causal_key_sha256=row.causal_key_sha256,
        split=row.split,
        physical_event_count=len(events),
        active_cells=active,
        sha256=_namespaced_sha256(_TRACE_NAMESPACE, record),
    )


def _build_relay_registries(
    fit_annotations: tuple[annotations_v1.StructuralAnnotation, ...],
) -> tuple[tuple[RelayLabelCell, ...], tuple[RelayBranch, ...], str, str]:
    labels_by_sha: dict[str, annotations_v1.AbstractRelayLabel] = {}
    grouped: dict[
        tuple[str, tuple[object, ...]],
        dict[str, annotations_v1.AbstractRelayLabel],
    ] = defaultdict(dict)
    for annotation in fit_annotations:
        seen_branches: set[tuple[str, tuple[object, ...]]] = set()
        for label in annotation.relay_labels:
            branch = (label.channel, _branch_key(label))
            if branch in seen_branches:
                raise RuntimeError("FIT target has two labels in one masked branch")
            seen_branches.add(branch)
            incumbent = labels_by_sha.setdefault(label.sha256, label)
            if incumbent != label:
                raise RuntimeError("generic relay label content address aliases")
            grouped[branch][label.sha256] = label

    ordered_labels = tuple(
        sorted(
            labels_by_sha.values(),
            key=lambda label: (_CHANNEL_ORDER[label.channel], label.sha256),
        )
    )
    cells = tuple(
        RelayLabelCell(cell_id=index, channel=label.channel, label=label)
        for index, label in enumerate(ordered_labels)
    )
    cell_by_sha = {cell.label.sha256: cell.cell_id for cell in cells}
    branches = tuple(
        RelayBranch(
            channel=channel,
            branch_key=branch_key,
            candidate_cell_ids=tuple(
                cell_by_sha[label_sha] for label_sha in sorted(labels_by_sha256)
            ),
        )
        for (channel, branch_key), labels_by_sha256 in sorted(
            grouped.items(),
            key=lambda item: (
                _CHANNEL_ORDER[item[0][0]],
                _canonical_json(item[0][1]),
            ),
        )
    )
    if len(cells) != 71 or len(branches) != 41:
        raise RuntimeError("FIT-only relay cell/branch registry changed")
    label_root = _namespaced_sha256(
        _LABEL_REGISTRY_NAMESPACE,
        tuple(cell.record() for cell in cells),
    )
    branch_root = _namespaced_sha256(
        _BRANCH_REGISTRY_NAMESPACE,
        tuple(branch.record() for branch in branches),
    )
    return cells, branches, label_root, branch_root


def _target_cells_by_branch(
    annotation: annotations_v1.StructuralAnnotation,
    cell_by_label_sha256: Mapping[str, int],
) -> dict[tuple[str, tuple[object, ...]], int]:
    targets: dict[tuple[str, tuple[object, ...]], int] = {}
    for label in annotation.relay_labels:
        cell = cell_by_label_sha256.get(label.sha256)
        if cell is None:
            raise RuntimeError("FIT teacher label is absent from the FIT relay registry")
        branch = (label.channel, _branch_key(label))
        incumbent = targets.setdefault(branch, cell)
        if incumbent != cell:
            raise RuntimeError("FIT teacher emitted two cells in one branch")
    return targets


def _array_frame(name: str, array: np.ndarray) -> bytes:
    header = _canonical_json(
        {"name": name, "dtype": array.dtype.str, "shape": tuple(int(v) for v in array.shape)}
    )
    payload = np.ascontiguousarray(array).tobytes(order="C")
    return len(header).to_bytes(8, "little") + header + len(payload).to_bytes(8, "little") + payload


def _checkpoint_payload(
    metadata: Mapping[str, object],
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> bytes:
    header = _canonical_json(metadata)
    return (
        _CHECKPOINT_NAMESPACE
        + len(header).to_bytes(8, "little")
        + header
        + _array_frame("excitatory", excitatory)
        + _array_frame("inhibitory", inhibitory)
    )


class FrozenPromptSelectionWorkspace:
    __slots__ = (
        "_branches",
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_label_cells",
        "_metadata",
    )

    def __init__(
        self,
        *,
        label_cells: tuple[RelayLabelCell, ...],
        branches: tuple[RelayBranch, ...],
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        metadata: Mapping[str, object],
    ) -> None:
        expected_shape = (len(label_cells), TRACE_CELL_COUNT)
        if (
            excitatory.shape != expected_shape
            or inhibitory.shape != expected_shape
            or excitatory.dtype != np.uint8
            or inhibitory.dtype != np.uint8
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("prompt-selection V2 arrays violate the frozen cap-four domain")
        self._label_cells = tuple(label_cells)
        self._branches = tuple(branches)
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._metadata = _freeze_record(metadata)
        self._checkpoint_sha256 = sha256(
            _checkpoint_payload(
                self._metadata,
                self._excitatory,
                self._inhibitory,
            )
        ).hexdigest()
        self.authenticate()

    @property
    def label_cells(self) -> tuple[RelayLabelCell, ...]:
        return self._label_cells

    @property
    def branches(self) -> tuple[RelayBranch, ...]:
        return self._branches

    @property
    def excitatory_ticks(self) -> np.ndarray:
        values = self._excitatory.copy(order="C")
        values.setflags(write=False)
        return values

    @property
    def inhibitory_ticks(self) -> np.ndarray:
        values = self._inhibitory.copy(order="C")
        values.setflags(write=False)
        return values

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    def authenticate(self) -> None:
        if (
            self._metadata["dependency_sha256"] != DEPENDENCY_SHA256
            or self._metadata["label_registry_sha256"] != LABEL_REGISTRY_SHA256
            or self._metadata["branch_registry_sha256"] != BRANCH_REGISTRY_SHA256
            or self._metadata["fit_teacher_identity_count"] not in (0, 24)
            or self._metadata["heldout_teacher_identity_count"] != 0
            or tuple(cell.record() for cell in self._label_cells) != self._metadata["label_cells"]
            or tuple(branch.record() for branch in self._branches) != self._metadata["branches"]
            or sha256(
                _checkpoint_payload(
                    self._metadata,
                    self._excitatory,
                    self._inhibitory,
                )
            ).hexdigest()
            != self._checkpoint_sha256
        ):
            raise RuntimeError("prompt-selection V2 checkpoint authentication failed")


def _raw_scores(
    workspace: FrozenPromptSelectionWorkspace,
    branch: RelayBranch,
    trace: PhysicalTrace,
    meter: _WorkMeter,
) -> tuple[int, ...]:
    candidates = np.asarray(branch.candidate_cell_ids, dtype=np.intp)
    active = np.asarray(trace.active_cells, dtype=np.intp)
    raw = np.sum(
        workspace._excitatory[candidates][:, active].astype(np.int16)
        - workspace._inhibitory[candidates][:, active].astype(np.int16),
        axis=1,
        dtype=np.int64,
    )
    if np.any(raw < -TRACE_LANE_COUNT * TICK_CAP) or np.any(raw > TRACE_LANE_COUNT * TICK_CAP):
        raise RuntimeError("branch-local raw score left the frozen cap-four domain")
    meter.charge(
        "synaptic-tick-read",
        len(branch.candidate_cell_ids) * TRACE_LANE_COUNT * 2,
    )
    return tuple(int(value) for value in raw)


_RACE = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)


def _decide_branch(
    workspace: FrozenPromptSelectionWorkspace,
    branch: RelayBranch,
    trace: PhysicalTrace,
    meter: _WorkMeter,
) -> BranchDecision:
    raw = _raw_scores(workspace, branch, trace, meter)
    current_scores = (*(max(score, 0) for score in raw), STOP_SCORE)
    currents = np.asarray(
        tuple(v4spec.PROTOCOL.latency_table[score].current_pa for score in current_scores),
        dtype=np.float64,
    )
    race = _RACE.race_many(currents[None, :], require_unique=False)[0]
    meter.charge("principal-current-write", len(current_scores))
    meter.charge("glif-principal-cell-step", len(current_scores) * RACE_STEPS)
    meter.charge("masked-wta-decision")
    if race.simultaneous_spike_candidates != 1:
        decision = "COLLISION"
        selected_cell_id = None
        selected_label = None
    elif race.winner_cell == len(branch.candidate_cell_ids):
        decision = "STOP"
        selected_cell_id = None
        selected_label = None
    else:
        decision = "RELAY"
        selected_cell_id = branch.candidate_cell_ids[race.winner_cell]
        selected_label = workspace.label_cells[selected_cell_id].label
    without_sha = {
        "channel": branch.channel,
        "branch_key": branch.branch_key,
        "decision": decision,
        "selected_cell_id": selected_cell_id,
        "selected_label": selected_label.record() if selected_label is not None else None,
        "raw_candidate_scores": raw,
        "first_spike_step": race.spike_step,
        "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
    }
    return BranchDecision(
        channel=branch.channel,
        branch_key=branch.branch_key,
        decision=decision,
        selected_cell_id=selected_cell_id,
        selected_label=selected_label,
        raw_candidate_scores=raw,
        first_spike_step=race.spike_step,
        simultaneous_spike_candidates=race.simultaneous_spike_candidates,
        sha256=_namespaced_sha256(_DECISION_NAMESPACE, without_sha),
    )


def _increment_ticks(
    values: np.ndarray,
    row: int,
    active_cells: tuple[int, ...],
    *,
    first_cell_bonus: bool,
) -> int:
    committed = 0
    for ordinal, cell in enumerate(active_cells):
        arrivals = 2 if first_cell_bonus and ordinal == 0 else 1
        previous = int(values[row, cell])
        updated = min(TICK_CAP, previous + arrivals)
        values[row, cell] = updated
        committed += updated - previous
    return committed


def _train_workspace(
    *,
    traces: tuple[PhysicalTrace, ...],
    teachers: tuple[annotations_v1.StructuralAnnotation, ...],
    label_cells: tuple[RelayLabelCell, ...],
    branches: tuple[RelayBranch, ...],
    teacher_enabled: bool,
) -> tuple[FrozenPromptSelectionWorkspace, WorkLedger]:
    if len(traces) != len(teachers) or len(traces) != 24:
        raise RuntimeError("training requires exactly the 24 frozen FIT identities")
    excitatory = np.zeros((len(label_cells), TRACE_CELL_COUNT), dtype=np.uint8)
    inhibitory = np.zeros_like(excitatory)
    cell_by_label = {cell.label.sha256: cell.cell_id for cell in label_cells}
    meter = _WorkMeter()
    metadata = {
        "schema": "public-train-prompt-selection-v2-checkpoint",
        "dependency_sha256": DEPENDENCY_SHA256,
        "label_registry_sha256": LABEL_REGISTRY_SHA256,
        "branch_registry_sha256": BRANCH_REGISTRY_SHA256,
        "label_cells": tuple(cell.record() for cell in label_cells),
        "branches": tuple(branch.record() for branch in branches),
        "array_shape": (len(label_cells), TRACE_CELL_COUNT),
        "array_dtype": np.dtype(np.uint8).str,
        "tick_cap": TICK_CAP,
        "trace_lane_count": TRACE_LANE_COUNT,
        "fit_teacher_identity_count": 24 if teacher_enabled else 0,
        "heldout_teacher_identity_count": 0,
        "fit_schedule_causal_key_sha256s": tuple(trace.causal_key_sha256 for trace in traces),
        "causal_key_schedule_is_authentication_only": True,
    }
    teacher_identity_count = 24 if teacher_enabled else 0
    scratch = FrozenPromptSelectionWorkspace(
        label_cells=label_cells,
        branches=branches,
        excitatory=excitatory,
        inhibitory=inhibitory,
        metadata={**metadata, "fit_teacher_identity_count": teacher_identity_count},
    )
    for trace, teacher in zip(traces, teachers, strict=True):
        targets = _target_cells_by_branch(teacher, cell_by_label)
        for branch in branches:
            _decide_branch(scratch, branch, trace, meter)
            if not teacher_enabled:
                continue
            meter.charge("teacher-cell-spike")
            branch_id = (branch.channel, branch.branch_key)
            target_cell = targets.get(branch_id)
            if target_cell is not None:
                committed = _increment_ticks(
                    excitatory,
                    target_cell,
                    trace.active_cells,
                    first_cell_bonus=True,
                )
                meter.charge("excitatory-tick-commit", committed)
            for rival_cell in branch.candidate_cell_ids:
                if rival_cell == target_cell:
                    continue
                committed = _increment_ticks(
                    inhibitory,
                    rival_cell,
                    trace.active_cells,
                    first_cell_bonus=False,
                )
                meter.charge("inhibitory-tick-commit", committed)
        scratch = FrozenPromptSelectionWorkspace(
            label_cells=label_cells,
            branches=branches,
            excitatory=excitatory,
            inhibitory=inhibitory,
            metadata={**metadata, "fit_teacher_identity_count": teacher_identity_count},
        )
    final_metadata = {
        **metadata,
        "fit_teacher_identity_count": teacher_identity_count,
        "training_work": meter.freeze().record(),
    }
    workspace = FrozenPromptSelectionWorkspace(
        label_cells=label_cells,
        branches=branches,
        excitatory=excitatory,
        inhibitory=inhibitory,
        metadata=final_metadata,
    )
    return workspace, meter.freeze()


def _expected_by_channel(
    annotation: annotations_v1.StructuralAnnotation,
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {channel: [] for channel in _CHANNELS}
    for label in annotation.relay_labels:
        grouped[label.channel].append(label.sha256)
    return {channel: tuple(sorted(grouped[channel])) for channel in _CHANNELS}


def _select_identity(
    workspace: FrozenPromptSelectionWorkspace,
    trace: PhysicalTrace,
    expected: annotations_v1.StructuralAnnotation,
    meter: _WorkMeter,
) -> IdentityReplay:
    decisions = tuple(
        _decide_branch(workspace, branch, trace, meter) for branch in workspace.branches
    )
    selected_by_channel: dict[str, list[int]] = {channel: [] for channel in _CHANNELS}
    for decision in decisions:
        if decision.selected_cell_id is not None:
            selected_by_channel[decision.channel].append(decision.selected_cell_id)

    emitted: list[tuple[str, tuple[str, ...]]] = []
    for channel in _CHANNELS:
        channel_cells = tuple(cell for cell in workspace.label_cells if cell.channel == channel)
        local_by_global = {cell.cell_id: local for local, cell in enumerate(channel_cells)}
        currents = np.zeros(len(channel_cells), dtype=np.float64)
        for global_cell in selected_by_channel[channel]:
            currents[local_by_global[global_cell]] += float(v4spec.RELAY_COPY_CURRENT_PA)
            meter.charge("output-relay-current-write")
        relay = v4impl._DiscreteNeuronPopulation(
            len(channel_cells),
            threshold_pa=float(v4spec.RELAY_COPY_CURRENT_PA),
        ).step(currents)
        meter.charge("output-relay-cell-step", len(channel_cells))
        label_roots = tuple(
            sorted(channel_cells[int(local)].label.sha256 for local in np.flatnonzero(relay))
        )
        emitted.append((channel, label_roots))

    expected_channels = _expected_by_channel(expected)
    emitted_map = dict(emitted)
    exact_by_channel = tuple(
        (channel, emitted_map[channel] == expected_channels[channel]) for channel in _CHANNELS
    )
    meter.charge(
        "target-admission-label-read",
        sum(len(value) for value in emitted_map.values())
        + sum(len(value) for value in expected_channels.values()),
    )
    without_sha = {
        "causal_key_sha256": trace.causal_key_sha256,
        "split": trace.split,
        "trace_sha256": trace.sha256,
        "decisions": tuple(row.record() for row in decisions),
        "emitted_by_channel": tuple(emitted),
        "exact_by_channel": exact_by_channel,
        "exact_target_set": all(exact for _channel, exact in exact_by_channel),
        "causal_key_is_admission_metadata_only": True,
    }
    return IdentityReplay(
        causal_key_sha256=trace.causal_key_sha256,
        split=trace.split,
        trace_sha256=trace.sha256,
        decisions=decisions,
        emitted_by_channel=tuple(emitted),
        exact_by_channel=exact_by_channel,
        exact_target_set=without_sha["exact_target_set"],
        sha256=_namespaced_sha256(_IDENTITY_REPLAY_NAMESPACE, without_sha),
    )


def _replay_batch(
    workspace: FrozenPromptSelectionWorkspace,
    *,
    split: str,
    traces: tuple[PhysicalTrace, ...],
    expected: tuple[annotations_v1.StructuralAnnotation, ...],
) -> ReplayBatchResult:
    if len(traces) != len(expected) or any(trace.split != split for trace in traces):
        raise RuntimeError("replay batch split/target shape changed")
    workspace.authenticate()
    meter = _WorkMeter()
    replays = tuple(
        _select_identity(workspace, trace, target, meter)
        for trace, target in zip(traces, expected, strict=True)
    )
    workspace.authenticate()
    channel_rows: list[ChannelAdmission] = []
    for channel in _CHANNELS:
        exact_count = sum(dict(row.exact_by_channel)[channel] for row in replays)
        expected_count = sum(len(_expected_by_channel(target)[channel]) for target in expected)
        emitted_count = sum(len(dict(row.emitted_by_channel)[channel]) for row in replays)
        channel_rows.append(
            ChannelAdmission(
                channel=channel,
                exact_identity_count=exact_count,
                mismatch_identity_count=len(replays) - exact_count,
                expected_label_count=expected_count,
                emitted_label_count=emitted_count,
            )
        )
    work = meter.freeze()
    exact = sum(row.exact_target_set for row in replays)
    without_sha = {
        "split": split,
        "identity_count": len(replays),
        "exact_identity_count": exact,
        "mismatch_identity_count": len(replays) - exact,
        "channel_admissions": tuple(row.record() for row in channel_rows),
        "replays": tuple(row.record() for row in replays),
        "checkpoint_authentication_count": 2,
        "work": work.record(),
    }
    return ReplayBatchResult(
        split=split,
        identity_count=len(replays),
        exact_identity_count=exact,
        mismatch_identity_count=len(replays) - exact,
        channel_admissions=tuple(channel_rows),
        replays=replays,
        checkpoint_authentication_count=2,
        work=work,
        sha256=_namespaced_sha256(_BATCH_NAMESPACE, without_sha),
    )


def _control_result(
    name: str,
    workspace: FrozenPromptSelectionWorkspace,
    batch: ReplayBatchResult,
    training_work: WorkLedger | None,
) -> ControlResult:
    events = Counter(dict(batch.work.events))
    if training_work is not None:
        events.update(dict(training_work.events))
    merged_meter = _WorkMeter()
    for event, count in events.items():
        merged_meter.charge(event, count)
    work = merged_meter.freeze()
    stop_count = sum(
        decision.decision == "STOP" for replay in batch.replays for decision in replay.decisions
    )
    collision_count = sum(
        decision.decision == "COLLISION"
        for replay in batch.replays
        for decision in replay.decisions
    )
    without_sha = {
        "name": name,
        "exact_identity_count": batch.exact_identity_count,
        "mismatch_identity_count": batch.mismatch_identity_count,
        "stop_decision_count": stop_count,
        "collision_decision_count": collision_count,
        "checkpoint_sha256": workspace.checkpoint_sha256,
        "work": work.record(),
    }
    return ControlResult(
        name=name,
        exact_identity_count=batch.exact_identity_count,
        mismatch_identity_count=batch.mismatch_identity_count,
        stop_decision_count=stop_count,
        collision_decision_count=collision_count,
        checkpoint_sha256=workspace.checkpoint_sha256,
        work=work,
        sha256=_namespaced_sha256(_CONTROL_NAMESPACE, without_sha),
    )


PHYSICAL_INPUTS = _load_physical_inputs()
_ANNOTATION_BY_CAUSAL_KEY = {
    row.causal_key_sha256: row for row in annotations_v1.CAPSULE.annotations
}
_FIT_INPUTS = tuple(row for row in PHYSICAL_INPUTS if row.split == FIT)
_HELDOUT_INPUTS = tuple(row for row in PHYSICAL_INPUTS if row.split == HELDOUT)
_FIT_ANNOTATIONS = tuple(_ANNOTATION_BY_CAUSAL_KEY[row.causal_key_sha256] for row in _FIT_INPUTS)
_HELDOUT_ANNOTATIONS = tuple(
    _ANNOTATION_BY_CAUSAL_KEY[row.causal_key_sha256] for row in _HELDOUT_INPUTS
)

_FIT_TRACE_METER = _WorkMeter()
FIT_TRACES = tuple(
    _encode_trace(row, reverse_span_order=False, meter=_FIT_TRACE_METER) for row in _FIT_INPUTS
)

LABEL_CELLS, BRANCHES, LABEL_REGISTRY_SHA256, BRANCH_REGISTRY_SHA256 = _build_relay_registries(
    _FIT_ANNOTATIONS
)

# Controls are constructed before the treatment checkpoint.
_UNTRAINED_ARRAY = np.zeros((len(LABEL_CELLS), TRACE_CELL_COUNT), dtype=np.uint8)
_BASE_METADATA = {
    "schema": "public-train-prompt-selection-v2-checkpoint",
    "dependency_sha256": DEPENDENCY_SHA256,
    "label_registry_sha256": LABEL_REGISTRY_SHA256,
    "branch_registry_sha256": BRANCH_REGISTRY_SHA256,
    "label_cells": tuple(cell.record() for cell in LABEL_CELLS),
    "branches": tuple(branch.record() for branch in BRANCHES),
    "array_shape": (len(LABEL_CELLS), TRACE_CELL_COUNT),
    "array_dtype": np.dtype(np.uint8).str,
    "tick_cap": TICK_CAP,
    "trace_lane_count": TRACE_LANE_COUNT,
    "fit_teacher_identity_count": 0,
    "heldout_teacher_identity_count": 0,
    "fit_schedule_causal_key_sha256s": tuple(trace.causal_key_sha256 for trace in FIT_TRACES),
    "causal_key_schedule_is_authentication_only": True,
}
_UNTRAINED_WORKSPACE = FrozenPromptSelectionWorkspace(
    label_cells=LABEL_CELLS,
    branches=BRANCHES,
    excitatory=_UNTRAINED_ARRAY,
    inhibitory=_UNTRAINED_ARRAY,
    metadata=_BASE_METADATA,
)
_UNTRAINED_BATCH = _replay_batch(
    _UNTRAINED_WORKSPACE,
    split=FIT,
    traces=FIT_TRACES,
    expected=_FIT_ANNOTATIONS,
)
_UNTRAINED_CONTROL = _control_result(
    "untrained",
    _UNTRAINED_WORKSPACE,
    _UNTRAINED_BATCH,
    None,
)

_LABEL_OFF_WORKSPACE, _LABEL_OFF_TRAINING_WORK = _train_workspace(
    traces=FIT_TRACES,
    teachers=_FIT_ANNOTATIONS,
    label_cells=LABEL_CELLS,
    branches=BRANCHES,
    teacher_enabled=False,
)
_LABEL_OFF_BATCH = _replay_batch(
    _LABEL_OFF_WORKSPACE,
    split=FIT,
    traces=FIT_TRACES,
    expected=_FIT_ANNOTATIONS,
)
_LABEL_OFF_CONTROL = _control_result(
    "label-off",
    _LABEL_OFF_WORKSPACE,
    _LABEL_OFF_BATCH,
    _LABEL_OFF_TRAINING_WORK,
)

_SHUFFLE_OFFSET = next(
    offset
    for offset in range(1, len(_FIT_ANNOTATIONS))
    if all(
        row.target_set_sha256
        != _FIT_ANNOTATIONS[(index + offset) % len(_FIT_ANNOTATIONS)].target_set_sha256
        for index, row in enumerate(_FIT_ANNOTATIONS)
    )
)
_SHUFFLED_TEACHERS = tuple(
    _FIT_ANNOTATIONS[(index + _SHUFFLE_OFFSET) % len(_FIT_ANNOTATIONS)]
    for index in range(len(_FIT_ANNOTATIONS))
)
_SHUFFLED_WORKSPACE, _SHUFFLED_TRAINING_WORK = _train_workspace(
    traces=FIT_TRACES,
    teachers=_SHUFFLED_TEACHERS,
    label_cells=LABEL_CELLS,
    branches=BRANCHES,
    teacher_enabled=True,
)
_SHUFFLED_BATCH = _replay_batch(
    _SHUFFLED_WORKSPACE,
    split=FIT,
    traces=FIT_TRACES,
    expected=_FIT_ANNOTATIONS,
)
_SHUFFLED_CONTROL = _control_result(
    "shuffled-target-set",
    _SHUFFLED_WORKSPACE,
    _SHUFFLED_BATCH,
    _SHUFFLED_TRAINING_WORK,
)

WORKSPACE, TRAINING_WORK = _train_workspace(
    traces=FIT_TRACES,
    teachers=_FIT_ANNOTATIONS,
    label_cells=LABEL_CELLS,
    branches=BRANCHES,
    teacher_enabled=True,
)
FIT_RESULT = _replay_batch(
    WORKSPACE,
    split=FIT,
    traces=FIT_TRACES,
    expected=_FIT_ANNOTATIONS,
)

# The checkpoint is frozen and FIT has been replayed before heldout events are
# converted into recurrent traces.  These traces never enter a teacher path.
_HELDOUT_TRACE_METER = _WorkMeter()
HELDOUT_TRACES = tuple(
    _encode_trace(row, reverse_span_order=False, meter=_HELDOUT_TRACE_METER)
    for row in _HELDOUT_INPUTS
)
_TRACE_WORK_METER = _WorkMeter()
for _event, _count in (
    *_FIT_TRACE_METER.freeze().events,
    *_HELDOUT_TRACE_METER.freeze().events,
):
    _TRACE_WORK_METER.charge(_event, _count)
TRACE_WORK = _TRACE_WORK_METER.freeze()
TRACE_REGISTRY_SHA256 = _namespaced_sha256(
    _TRACE_NAMESPACE + b"registry\0",
    {
        "fit": tuple(trace.record() for trace in FIT_TRACES),
        "heldout": tuple(trace.record() for trace in HELDOUT_TRACES),
        "heldout_constructed_after_checkpoint_sha256": WORKSPACE.checkpoint_sha256,
        "work": TRACE_WORK.record(),
    },
)
HELDOUT_RESULT = _replay_batch(
    WORKSPACE,
    split=HELDOUT,
    traces=HELDOUT_TRACES,
    expected=_HELDOUT_ANNOTATIONS,
)

_DERANGED_TRACE_METER = _WorkMeter()
_DERANGED_TRACES = tuple(
    _encode_trace(row, reverse_span_order=True, meter=_DERANGED_TRACE_METER) for row in _FIT_INPUTS
)
_DERANGED_BATCH = _replay_batch(
    WORKSPACE,
    split=FIT,
    traces=_DERANGED_TRACES,
    expected=_FIT_ANNOTATIONS,
)
_DERANGED_CONTROL = _control_result(
    "reversed-span-event-order",
    WORKSPACE,
    _DERANGED_BATCH,
    _DERANGED_TRACE_METER.freeze(),
)

CONTROLS = (
    _UNTRAINED_CONTROL,
    _LABEL_OFF_CONTROL,
    _SHUFFLED_CONTROL,
    _DERANGED_CONTROL,
)
CONTROL_PANEL_SHA256 = _namespaced_sha256(
    _CONTROL_PANEL_NAMESPACE,
    {
        "shuffle_offset": _SHUFFLE_OFFSET,
        "controls": tuple(control.record() for control in CONTROLS),
    },
)

_CONTROL_PASS = (
    _UNTRAINED_CONTROL.exact_identity_count == 0
    and _LABEL_OFF_CONTROL.exact_identity_count == 0
    and _SHUFFLED_CONTROL.exact_identity_count == 0
    and _DERANGED_CONTROL.exact_identity_count < len(_FIT_INPUTS)
)
if FIT_RESULT.exact_identity_count != len(_FIT_INPUTS):
    STATUS = STOP_FIT
elif HELDOUT_RESULT.exact_identity_count != len(_HELDOUT_INPUTS):
    STATUS = STOP_HELDOUT
elif not _CONTROL_PASS:
    STATUS = STOP_CONTROLS
else:
    STATUS = PASS

CHECKPOINT = CheckpointSummary(
    candidate_label_cell_count=len(LABEL_CELLS),
    branch_count=len(BRANCHES),
    array_shape=WORKSPACE._excitatory.shape,
    array_dtype=WORKSPACE._excitatory.dtype.str,
    array_bytes=WORKSPACE._excitatory.nbytes + WORKSPACE._inhibitory.nbytes,
    tick_cap=TICK_CAP,
    excitatory_nonzero_ticks=int(np.count_nonzero(WORKSPACE._excitatory)),
    inhibitory_nonzero_ticks=int(np.count_nonzero(WORKSPACE._inhibitory)),
    fit_teacher_identity_count=24,
    heldout_teacher_identity_count=0,
    checkpoint_sha256=WORKSPACE.checkpoint_sha256,
)

SURFACE_BOUNDARY_RECORD = _freeze_record(
    {
        "neural_input_sources": (
            "ordered-physical-span-events",
            "ordered-typed-literal-cell-events",
            "ordered-name-free-signature-role-type-events",
        ),
        "neural_output_channels": _CHANNELS,
        "fit_teacher_identity_count": 24,
        "heldout_teacher_identity_count": 0,
        "heldout_read_only_replay_identity_count": 18,
        "learner_array_count": 2,
        "learner_array_bytes": CHECKPOINT.array_bytes,
        "learning_pass_count": 1,
        "physical_fit_replay_identity_count": FIT_RESULT.identity_count,
        "physical_heldout_replay_identity_count": HELDOUT_RESULT.identity_count,
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
        "host_lookup_selection_count": HOST_LOOKUP_SELECTION_COUNT,
        "host_union_selection_count": HOST_UNION_SELECTION_COUNT,
        "host_ranking_count": HOST_RANKING_COUNT,
        "candidate_program_count": CANDIDATE_PROGRAM_COUNT,
        "proof_count": PROOF_COUNT,
        "full_signature_authentication_only": True,
        "causal_key_sha256_authentication_and_admission_only": True,
        "split_authentication_and_reporting_only": True,
    }
)
SURFACE_BOUNDARY_SHA256 = _namespaced_sha256(
    _SURFACE_NAMESPACE,
    dict(SURFACE_BOUNDARY_RECORD),
)

_ROOT_WITHOUT_RESULT = {
    "scope": PROTOCOL_SCOPE,
    "status": STATUS,
    "dependency_sha256": DEPENDENCY_SHA256,
    "trace_registry_sha256": TRACE_REGISTRY_SHA256,
    "label_registry_sha256": LABEL_REGISTRY_SHA256,
    "branch_registry_sha256": BRANCH_REGISTRY_SHA256,
    "checkpoint_sha256": CHECKPOINT.checkpoint_sha256,
    "fit_result_sha256": FIT_RESULT.sha256,
    "heldout_result_sha256": HELDOUT_RESULT.sha256,
    "control_panel_sha256": CONTROL_PANEL_SHA256,
    "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
    "trace_work": TRACE_WORK.record(),
    "training_work": TRAINING_WORK.record(),
    "physical_work_total": (
        TRACE_WORK.total
        + TRAINING_WORK.total
        + FIT_RESULT.work.total
        + HELDOUT_RESULT.work.total
        + sum(control.work.total for control in CONTROLS)
    ),
    "fit_identity_count": FIT_RESULT.identity_count,
    "fit_exact_identity_count": FIT_RESULT.exact_identity_count,
    "heldout_identity_count": HELDOUT_RESULT.identity_count,
    "heldout_exact_identity_count": HELDOUT_RESULT.exact_identity_count,
    "checkpoint": CHECKPOINT.record(),
}
RESULT = PublicTrainPromptSelectionV2Result(
    scope=PROTOCOL_SCOPE,
    status=STATUS,
    checkpoint=CHECKPOINT,
    fit_result=FIT_RESULT,
    heldout_result=HELDOUT_RESULT,
    controls=CONTROLS,
    trace_work=TRACE_WORK,
    training_work=TRAINING_WORK,
    physical_work_total=_ROOT_WITHOUT_RESULT["physical_work_total"],
    dependency_sha256=DEPENDENCY_SHA256,
    trace_registry_sha256=TRACE_REGISTRY_SHA256,
    label_registry_sha256=LABEL_REGISTRY_SHA256,
    branch_registry_sha256=BRANCH_REGISTRY_SHA256,
    checkpoint_sha256=CHECKPOINT.checkpoint_sha256,
    fit_result_sha256=FIT_RESULT.sha256,
    heldout_result_sha256=HELDOUT_RESULT.sha256,
    control_panel_sha256=CONTROL_PANEL_SHA256,
    surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
    result_sha256=_namespaced_sha256(_RESULT_NAMESPACE, _ROOT_WITHOUT_RESULT),
)


__all__ = [
    "BRANCHES",
    "BRANCH_REGISTRY_SHA256",
    "CHECKPOINT",
    "CONTROLS",
    "CONTROL_PANEL_SHA256",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "FIT_RESULT",
    "FIT_TRACES",
    "HELDOUT_RESULT",
    "HELDOUT_TRACES",
    "LABEL_CELLS",
    "LABEL_REGISTRY_SHA256",
    "PASS",
    "PHYSICAL_INPUTS",
    "PROTOCOL_SCOPE",
    "RESULT",
    "STATUS",
    "STOP_CONTROLS",
    "STOP_FIT",
    "STOP_HELDOUT",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "TRACE_REGISTRY_SHA256",
    "TRAINING_WORK",
    "WORKSPACE",
    "FrozenPromptSelectionWorkspace",
    "PublicTrainPromptSelectionV2Result",
]
