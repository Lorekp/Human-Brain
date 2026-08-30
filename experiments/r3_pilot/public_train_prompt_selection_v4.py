"""One-spend learner/result for the frozen prompt-selection V4 protocol.

The learner first fits the complete 110-lesson self-posed curriculum with a
fresh cap-four teacher-gated branch-local workspace.  After causal controls
pass, that checkpoint is frozen.  Only then are the 24 authorized canonical
FIT teachers opened atomically for one prompt-derived teacher pass and one
read-only replay.  The 18 heldout targets are never read or replayed.

All semantic publications come from physical GLIF races.  A learned channel
race selects each prompt-ordered occurrence; channel-local atom/transition
branches then run repeated masked races with tonic STOP.  Immediate edges use
fresh distributed parent/child occurrence bindings.  Host code authenticates
teachers and scores the already-published ordered relay stream only.
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

import numpy as np
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as race_spec
from experiments.r3_pilot import public_train_prompt_selection_v4_protocol as protocol_v4
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as v5spec
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCOPE = "PUBLIC_TRAIN_PROMPT_SELECTION_V4"
PASS = "PASS_PUBLIC_TRAIN_PROMPT_SELECTION_V4"
STOP_SELF_POSED = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_SELF_POSED"
STOP_CONTROLS = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_CAUSAL_CONTROLS"
STOP_FIT_PREFLIGHT = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_FIT_PREFLIGHT"
STOP_FIT = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_FIT"

V4_PROTOCOL_COMMIT = "4471534d7cc562a14dd5d8101a9594ab892a2372"
V4_PROTOCOL_ROOT = "97f8ea272aac11188a0c5f31355085fddcf69eed602b0140eabc2f90d4920918"
V4_PROTOCOL_SOURCE_PATH = (
    "experiments/r3_pilot/public_train_prompt_selection_v4_protocol.py"
)
V4_PROTOCOL_TEST_PATH = "tests/test_public_train_prompt_selection_v4_protocol.py"
V4_PROTOCOL_SOURCE_SHA256 = (
    "3473d2b46eed8c974dfe3b30b3d075614ac4966a5b4e25f135202d2ad986b1f6"
)
V4_PROTOCOL_TEST_SHA256 = (
    "11d2ae6c2d01a2e9965fa42fd23701dc3b45e90a709055cee2c33586dc2d0e51"
)

V3_LEARNER_COMMIT = "582de79f7b75aceea8f4ed64871310a15c96830a"
V3_LEARNER_SOURCE_PATH = "experiments/r3_pilot/public_train_prompt_selection_v3.py"
V3_LEARNER_TEST_PATH = "tests/test_public_train_prompt_selection_v3.py"
V3_LEARNER_SOURCE_SHA256 = (
    "7300b87a67807a6e2f095ba70177f3bc034bf54bf9f40b538b0bd53be6fdd681"
)
V3_LEARNER_TEST_SHA256 = (
    "59570a1c541cc4dbde9faf7f7d42fc5c8a5deee2062b1876039de921cf386e60"
)
V3_LEARNER_RESULT_SHA256_AUDIT_ONLY = (
    "e73446e71164a5435c89031f01afed0171b494f54772df57c34c9db86816ad84"
)

V5_PROTOCOL_ROOT = "587bd68a293ca2116570cf4e5ee7a860bbdd9928138982ae43748c4f87a252c1"
TICK_CAP = 4
STOP_SCORE = 12
RACE_STEPS = race_spec.RACE_STEPS
SELF_FEATURE_CELL_COUNT = 512
SELF_EVENT_FEATURE_OFFSET = 0
SELF_TRANSITION_FEATURE_OFFSET = SELF_FEATURE_CELL_COUNT
SELF_INPUT_CELL_COUNT = SELF_FEATURE_CELL_COUNT * 2
PROMPT_LANE_COUNT = protocol_v4.ASSEMBLY_ACTIVE_CELL_COUNT
PROMPT_CELLS_PER_LANE = protocol_v4.PHYSICAL_CELL_COUNT
PROMPT_FEATURE_OFFSET = SELF_INPUT_CELL_COUNT
PROMPT_INPUT_CELL_COUNT = PROMPT_LANE_COUNT * PROMPT_CELLS_PER_LANE
INPUT_CELL_COUNT = SELF_INPUT_CELL_COUNT + PROMPT_INPUT_CELL_COUNT
ACTIVE_FEATURE_CELL_COUNT = protocol_v4.ASSEMBLY_ACTIVE_CELL_COUNT
OCCURRENCE_CAP = 32
FIT = "FIT"

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
PATH_HASH_NEURAL_INPUT_COUNT = 0
ABSOLUTE_DEPTH_NEURAL_INPUT_COUNT = 0
WHOLE_TARGET_HEAD_COUNT = 0
HOST_UNION_COUNT = 0
HOST_DEDUPLICATION_COUNT = 0
HOST_FILTERING_COUNT = 0
HOST_REORDERING_COUNT = 0
HOST_RANKING_COUNT = 0
CANDIDATE_PROGRAM_COUNT = 0
PROOF_COUNT = 0
HELDOUT_TARGET_READ_COUNT = 0
HELDOUT_INPUT_TRACE_COUNT = 0
HELDOUT_REPLAY_IDENTITY_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]
_PINNED_SOURCE_SHA256S = MappingProxyType(
    {
        V4_PROTOCOL_SOURCE_PATH: V4_PROTOCOL_SOURCE_SHA256,
        V4_PROTOCOL_TEST_PATH: V4_PROTOCOL_TEST_SHA256,
        V3_LEARNER_SOURCE_PATH: V3_LEARNER_SOURCE_SHA256,
        V3_LEARNER_TEST_PATH: V3_LEARNER_TEST_SHA256,
        "experiments/r3_pilot/public_train_prompt_factor_manifest_v1.py": (
            "95aa4af55c9cd9d6fa841e203beccf2dce8a67fe41bf3e15d1fe1dfeacb67c78"
        ),
        "experiments/r3_pilot/public_train_prompt_selection_freeze_v1_protocol.py": (
            "a35ee1de48bea57404d101b69ec729d36470a88a55caa2aff5163dde246894ef"
        ),
        "experiments/r3_pilot/public_train_structural_annotation_capsule_v1.py": (
            "d2e977b3aff2e496307168ed64e85c2ebbc03579224bf58b0fd175f0ed096c88"
        ),
        "experiments/r3_pilot/teacher_gated_factor_workspace_v5.py": (
            "c76292164b43586b9400f8277f1c2aa71b2a136f113f36bc3f89712f0ca6895d"
        ),
        "experiments/r3_pilot/teacher_gated_factor_workspace_v5_protocol.py": (
            "d62fe76444ef48f0b58723f0cede1b6ca76e86c78623eb43c07f38be8e8c8746"
        ),
        "experiments/r3_pilot/exclusive_factor_workspace_v4_protocol.py": (
            "6dfb55ca119a9c1b46c296047b83941fa7607e4276631d0112d075ce5fd2c08b"
        ),
    }
)

_DEPENDENCY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.dependency.v1\0"
_REGISTRY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.registry.v1\0"
_TRACE_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.trace.v1\0"
_BINDING_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.binding.v1\0"
_WORK_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.work.v1\0"
_CHECKPOINT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.checkpoint.v1\0"
_DECISION_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.decision.v1\0"
_SELF_REPLAY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.self-replay.v1\0"
_SELF_BATCH_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.self-batch.v1\0"
_CONTROL_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.control.v1\0"
_CONTROL_PANEL_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.controls.v1\0"
_FIT_PREFLIGHT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.fit-preflight.v1\0"
_FIT_REPLAY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.fit-replay.v1\0"
_FIT_BATCH_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.fit-batch.v1\0"
_SURFACE_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.surface.v1\0"
_RESULT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.result.v1\0"


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
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


def _file_sha256(relative_path: str) -> str:
    return sha256((_ROOT / relative_path).read_bytes()).hexdigest()


def _authenticate_dependencies() -> None:
    provenance = protocol_v4.DEPENDENCY_RECORD
    if (
        protocol_v4.PROTOCOL.status != protocol_v4.PASS_PROTOCOL
        or protocol_v4.PROTOCOL.protocol_sha256 != V4_PROTOCOL_ROOT
        or protocol_v4.LEARNER_GATE_RECORD["fresh_v4_learner_required"] is not True
        or protocol_v4.LEARNER_GATE_RECORD["learner_execution_authorized"] is not False
        or protocol_v4.LEARNER_GATE_RECORD["required_teacher_schedule"]
        != "110-self-posed-then-24-fit-only"
        or protocol_v4.FIT_PROJECTION_AUDIT.projectable_identity_count != 24
        or protocol_v4.FIT_PROJECTION_AUDIT.heldout_target_read_count != 0
        or provenance["v3_learner_source_commit"] != V3_LEARNER_COMMIT
        or provenance["v3_learner_source_sha256"] != V3_LEARNER_SOURCE_SHA256
        or provenance["v3_learner_test_sha256"] != V3_LEARNER_TEST_SHA256
        or provenance["v3_learner_result_sha256_audit_only"]
        != V3_LEARNER_RESULT_SHA256_AUDIT_ONLY
        or provenance["v3_learner_imported"] is not False
        or provenance["v3_learner_source_executed"] is not False
        or provenance["v3_learner_result_is_dependency_authority"] is not False
        or v5spec.PROTOCOL.protocol_sha256 != V5_PROTOCOL_ROOT
        or v5spec.TICK_CAP != TICK_CAP
        or race_spec.STOP_SCORE != STOP_SCORE
        or protocol_v4.ASSEMBLY_ACTIVE_CELL_COUNT != ACTIVE_FEATURE_CELL_COUNT
    ):
        raise RuntimeError("prompt-selection V4 authenticated dependency changed")
    for relative_path, expected in _PINNED_SOURCE_SHA256S.items():
        if _file_sha256(relative_path) != expected:
            raise RuntimeError(f"prompt-selection V4 pinned source changed: {relative_path}")


_authenticate_dependencies()

DEPENDENCY_RECORD = _freeze(
    {
        "v4_protocol_commit": V4_PROTOCOL_COMMIT,
        "v4_protocol_scope": protocol_v4.PROTOCOL.scope,
        "v4_protocol_status": protocol_v4.PROTOCOL.status,
        "v4_protocol_root": V4_PROTOCOL_ROOT,
        "v4_protocol_source_sha256": V4_PROTOCOL_SOURCE_SHA256,
        "v4_protocol_test_sha256": V4_PROTOCOL_TEST_SHA256,
        "v4_protocol_dependency_sha256": protocol_v4.DEPENDENCY_SHA256,
        "v4_protocol_vocabulary_sha256": protocol_v4.VOCABULARY_SHA256,
        "v4_protocol_curriculum_sha256": protocol_v4.CURRICULUM_SHA256,
        "v4_protocol_topology_sha256": protocol_v4.TOPOLOGY_SHA256,
        "v4_protocol_fit_projection_sha256": protocol_v4.FIT_PROJECTION_AUDIT.sha256,
        "v3_learner_commit": V3_LEARNER_COMMIT,
        "v3_learner_source_sha256": V3_LEARNER_SOURCE_SHA256,
        "v3_learner_test_sha256": V3_LEARNER_TEST_SHA256,
        "v3_learner_result_sha256_audit_only": V3_LEARNER_RESULT_SHA256_AUDIT_ONLY,
        "v3_learner_module_imported_or_executed": False,
        "v3_checkpoint_weight_or_output_reuse": False,
        "banked_v5_protocol_sha256": v5spec.PROTOCOL.protocol_sha256,
        "banked_v5_local_rule_sha256": v5spec.PROTOCOL.local_rule_sha256,
        "pinned_source_sha256s": tuple(_PINNED_SOURCE_SHA256S.items()),
        "teacher_schedule": "one-110-self-posed-pass-then-at-most-one-24-fit-pass",
        "heldout_target_read_or_replay_count": 0,
        "dependency_and_audit_hashes_are_not_neural_evidence": True,
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(_DEPENDENCY_NAMESPACE, DEPENDENCY_RECORD)


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
            raise ValueError("physical work must be a nonnegative integer")
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
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class OutputBranch:
    branch_id: str
    kind: str
    channel: str
    branch_role: str
    candidate_cell_ids: tuple[int, ...]

    def record(self) -> dict[str, object]:
        return {
            **dataclasses.asdict(self),
            "tonic_stop_score": STOP_SCORE,
            "selection": "repeated-masked-channel-local-first-spike-glif-wta",
        }


def _build_output_registry() -> tuple[
    tuple[OutputCell, ...], OutputBranch, tuple[OutputBranch, ...], str, str
]:
    cells: list[OutputCell] = []
    for schema in protocol_v4.RELAY_SCHEMAS:
        cells.append(
            OutputCell(len(cells), "channel", "occurrence-channel", "router", schema.channel)
        )
    channel_branch = OutputBranch(
        branch_id="channel|occurrence-channel|router",
        kind="channel",
        channel="occurrence-channel",
        branch_role="router",
        candidate_cell_ids=tuple(range(len(protocol_v4.RELAY_SCHEMAS))),
    )
    for atom in protocol_v4.OUTPUT_ATOMS:
        cells.append(OutputCell(len(cells), "atom", atom.channel, atom.role, atom.atom_id))
    for transition in protocol_v4.OUTPUT_TRANSITIONS:
        cells.append(
            OutputCell(
                len(cells),
                "transition",
                transition.channel,
                transition.from_role,
                transition.transition_id,
            )
        )
    branches: list[OutputBranch] = []
    for schema in protocol_v4.RELAY_SCHEMAS:
        for slot in schema.slots:
            candidates = tuple(
                cell.cell_id
                for cell in cells
                if cell.kind == "atom"
                and cell.channel == schema.channel
                and cell.branch_role == slot.role
            )
            branches.append(
                OutputBranch(
                    f"atom|{schema.channel}|{slot.role}",
                    "atom",
                    schema.channel,
                    slot.role,
                    candidates,
                )
            )
        transition_roles: list[str] = []
        for transition in protocol_v4.OUTPUT_TRANSITIONS:
            if (
                transition.channel == schema.channel
                and transition.from_role not in transition_roles
            ):
                transition_roles.append(transition.from_role)
        for role in transition_roles:
            candidates = tuple(
                cell.cell_id
                for cell in cells
                if cell.kind == "transition"
                and cell.channel == schema.channel
                and cell.branch_role == role
            )
            branches.append(
                OutputBranch(
                    f"transition|{schema.channel}|{role}",
                    "transition",
                    schema.channel,
                    role,
                    candidates,
                )
            )
    semantic_cells = tuple(cell for cell in cells if cell.kind != "channel")
    branch_candidates = tuple(
        candidate for branch in branches for candidate in branch.candidate_cell_ids
    )
    if (
        len(protocol_v4.RELAY_SCHEMAS) != 7
        or len(protocol_v4.OUTPUT_ATOMS) != 103
        or len(protocol_v4.OUTPUT_TRANSITIONS) != 24
        or len(cells) != 134
        or len(set(branch_candidates)) != len(semantic_cells)
        or set(branch_candidates) != {cell.cell_id for cell in semantic_cells}
        or any(not branch.candidate_cell_ids for branch in branches)
    ):
        raise RuntimeError("prompt-selection V4 output registry changed")
    cell_root = _namespaced_sha256(
        _REGISTRY_NAMESPACE + b"cells\0", tuple(cell.record() for cell in cells)
    )
    branch_root = _namespaced_sha256(
        _REGISTRY_NAMESPACE + b"branches\0",
        (channel_branch.record(), *(branch.record() for branch in branches)),
    )
    return tuple(cells), channel_branch, tuple(branches), cell_root, branch_root


(
    OUTPUT_CELLS,
    CHANNEL_BRANCH,
    OUTPUT_BRANCHES,
    OUTPUT_REGISTRY_SHA256,
    BRANCH_REGISTRY_SHA256,
) = _build_output_registry()
_CELL_BY_TARGET = {cell.target_id: cell for cell in OUTPUT_CELLS}
_BRANCH_BY_ID = {branch.branch_id: branch for branch in OUTPUT_BRANCHES}
_BRANCH_BY_CELL = {
    cell_id: branch
    for branch in OUTPUT_BRANCHES
    for cell_id in branch.candidate_cell_ids
}
_CHANNEL_CELL_BY_TARGET = {
    cell.target_id: cell.cell_id for cell in OUTPUT_CELLS if cell.kind == "channel"
}


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


class FrozenPromptSelectionV4Workspace:
    __slots__ = (
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_metadata",
    )

    def __init__(
        self,
        *,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        metadata: Mapping[str, object],
    ) -> None:
        expected_shape = (len(OUTPUT_CELLS), INPUT_CELL_COUNT)
        if (
            excitatory.shape != expected_shape
            or inhibitory.shape != expected_shape
            or excitatory.dtype != np.uint8
            or inhibitory.dtype != np.uint8
            or np.any(excitatory > TICK_CAP)
            or np.any(inhibitory > TICK_CAP)
        ):
            raise ValueError("prompt-selection V4 arrays violate the cap-four domain")
        self._excitatory = np.array(excitatory, copy=True, order="C")
        self._inhibitory = np.array(inhibitory, copy=True, order="C")
        self._excitatory.setflags(write=False)
        self._inhibitory.setflags(write=False)
        self._metadata = _freeze(metadata)
        self._checkpoint_sha256 = sha256(
            _checkpoint_payload(self._metadata, self._excitatory, self._inhibitory)
        ).hexdigest()
        self.authenticate()

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
            or self._metadata["protocol_sha256"] != V4_PROTOCOL_ROOT
            or self._metadata["output_registry_sha256"] != OUTPUT_REGISTRY_SHA256
            or self._metadata["branch_registry_sha256"] != BRANCH_REGISTRY_SHA256
            or self._metadata["heldout_teacher_identity_count"] != 0
            or self._metadata["heldout_target_read_count"] != 0
            or self._metadata["whole_target_head_count"] != 0
            or sha256(
                _checkpoint_payload(self._metadata, self._excitatory, self._inhibitory)
            ).hexdigest()
            != self._checkpoint_sha256
        ):
            raise RuntimeError("prompt-selection V4 checkpoint authentication failed")


@dataclass(frozen=True, slots=True)
class SelfEventFeature:
    relation_cells: tuple[int, ...]
    role_cells: tuple[int, ...]
    active_cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelfTransitionFeature:
    relation_cells: tuple[int, ...]
    from_role_cells: tuple[int, ...]
    to_role_cells: tuple[int, ...]
    active_cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelfTrace:
    lesson_index: int
    mode: str
    raw_events: tuple[protocol_v4.PhysicalAssemblyEvent, ...]
    event_features: tuple[SelfEventFeature, ...]
    transition_features: tuple[SelfTransitionFeature, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "lesson_index": self.lesson_index,
            "mode": self.mode,
            "raw_events": tuple(row.record() for row in self.raw_events),
            "event_features": tuple(dataclasses.asdict(row) for row in self.event_features),
            "transition_features": tuple(
                dataclasses.asdict(row) for row in self.transition_features
            ),
            "neural_sources": (
                "physical-relation-assemblies",
                "physical-role-assemblies",
                "physical-filler-assemblies",
                "ordered-immediate-role-transition-assemblies",
            ),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class BranchDecision:
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

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class SelfLessonReplay:
    lesson_index: int
    channel: str
    trace_sha256: str
    decisions: tuple[BranchDecision, ...]
    emitted_channel: str | None
    emitted_by_branch: tuple[tuple[str, str | None], ...]
    expected_by_branch: tuple[tuple[str, str | None], ...]
    exact: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "sha256"
        }
        record["decisions"] = tuple(row.record() for row in self.decisions)
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
class SelfReplayBatch:
    name: str
    mode: str
    lesson_count: int
    exact_lesson_count: int
    mismatch_lesson_count: int
    channel_outcomes: tuple[ChannelOutcome, ...]
    replays: tuple[SelfLessonReplay, ...]
    work: WorkLedger
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "mode": self.mode,
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


def _conjunction_cells(kind: str, parts: tuple[int, ...], *, offset: int) -> tuple[int, ...]:
    cells: set[int] = set()
    counter = 0
    payload = b"".join(int(value).to_bytes(2, "big") for value in parts)
    while len(cells) < ACTIVE_FEATURE_CELL_COUNT:
        digest = sha256(
            _TRACE_NAMESPACE
            + kind.encode("ascii")
            + b"\0"
            + payload
            + counter.to_bytes(4, "big")
        ).digest()
        for position in range(0, len(digest), 2):
            cells.add(
                offset
                + int.from_bytes(digest[position : position + 2], "big")
                % SELF_FEATURE_CELL_COUNT
            )
            if len(cells) == ACTIVE_FEATURE_CELL_COUNT:
                break
        counter += 1
    return tuple(sorted(cells))


def _transform_self_events(
    events: tuple[protocol_v4.PhysicalAssemblyEvent, ...], mode: str
) -> tuple[protocol_v4.PhysicalAssemblyEvent, ...]:
    if mode == "normal":
        return events
    if mode == "cell-off":
        return ()
    if mode == "reverse-order":
        return tuple(reversed(events))
    if len(events) < 2:
        raise RuntimeError("causal derangement requires at least two physical events")
    rows = list(events)
    left, right = rows[0], rows[1]
    if mode == "role-swap":
        rows[0] = protocol_v4.PhysicalAssemblyEvent(
            left.relation_cells, right.role_cells, left.filler_cells
        )
        rows[1] = protocol_v4.PhysicalAssemblyEvent(
            right.relation_cells, left.role_cells, right.filler_cells
        )
    elif mode == "filler-swap":
        rows[0] = protocol_v4.PhysicalAssemblyEvent(
            left.relation_cells, left.role_cells, right.filler_cells
        )
        rows[1] = protocol_v4.PhysicalAssemblyEvent(
            right.relation_cells, right.role_cells, left.filler_cells
        )
    else:
        raise ValueError(f"unknown self-posed trace mode: {mode}")
    return tuple(rows)


def _encode_self_trace(
    lesson: protocol_v4.SelfPosedRelayLesson,
    *,
    mode: str,
    meter: _WorkMeter,
) -> SelfTrace:
    events = _transform_self_events(lesson.causal_input.ordered_events, mode)
    event_features = tuple(
        SelfEventFeature(
            relation_cells=row.relation_cells,
            role_cells=row.role_cells,
            active_cells=_conjunction_cells(
                "self-event-role-filler",
                (*row.relation_cells, *row.role_cells, *row.filler_cells),
                offset=SELF_EVENT_FEATURE_OFFSET,
            ),
        )
        for row in events
    )
    transition_features = tuple(
        SelfTransitionFeature(
            relation_cells=left.relation_cells,
            from_role_cells=left.role_cells,
            to_role_cells=right.role_cells,
            active_cells=_conjunction_cells(
                "self-ordered-immediate-transition",
                (*left.relation_cells, *left.role_cells, *right.role_cells),
                offset=SELF_TRANSITION_FEATURE_OFFSET,
            ),
        )
        for left, right in pairwise(events)
    )
    meter.charge("self-physical-assembly-cell-read", len(events) * 3 * 12)
    meter.charge("self-fixed-conjunction-synapse-read", len(events) * 36 * 12)
    meter.charge(
        "self-fixed-transition-conjunction-synapse-read",
        len(transition_features) * 36 * 12,
    )
    meter.charge(
        "self-conjunction-cell-emission",
        (len(event_features) + len(transition_features)) * ACTIVE_FEATURE_CELL_COUNT,
    )
    record = {
        "lesson_index": lesson.lesson_index,
        "mode": mode,
        "raw_events": tuple(row.record() for row in events),
        "event_features": tuple(dataclasses.asdict(row) for row in event_features),
        "transition_features": tuple(dataclasses.asdict(row) for row in transition_features),
    }
    return SelfTrace(
        lesson_index=lesson.lesson_index,
        mode=mode,
        raw_events=events,
        event_features=event_features,
        transition_features=transition_features,
        sha256=_namespaced_sha256(_TRACE_NAMESPACE + b"self\0", record),
    )


def _self_channel_feature(trace: SelfTrace, meter: _WorkMeter) -> tuple[int, ...]:
    if not trace.raw_events:
        meter.charge("self-channel-router-stop")
        return ()
    relation = trace.raw_events[0].relation_cells
    if any(row.relation_cells != relation for row in trace.raw_events):
        meter.charge("self-channel-router-poison")
        return ()
    meter.charge("self-channel-router-cell-read", len(trace.raw_events) * 12)
    return _conjunction_cells("self-channel-router", relation, offset=SELF_EVENT_FEATURE_OFFSET)


def _self_branch_feature(
    branch: OutputBranch, trace: SelfTrace, meter: _WorkMeter
) -> tuple[int, ...]:
    expected_relation = protocol_v4._assembly_cells("relation", branch.channel)
    expected_role = protocol_v4._assembly_cells(
        "role", f"{branch.channel}|{branch.branch_role}"
    )
    if branch.kind == "atom":
        matches = tuple(
            row.active_cells
            for row in trace.event_features
            if row.relation_cells == expected_relation and row.role_cells == expected_role
        )
        meter.charge("self-role-router-cell-read", len(trace.event_features) * 48)
    else:
        matches = tuple(
            row.active_cells
            for row in trace.transition_features
            if row.relation_cells == expected_relation
            and row.from_role_cells == expected_role
        )
        meter.charge(
            "self-transition-router-cell-read", len(trace.transition_features) * 48
        )
    meter.charge("self-branch-feature-router-decision")
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


def _update_branch(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    branch: OutputBranch,
    feature: tuple[int, ...],
    target_cell: int | None,
    meter: _WorkMeter,
) -> None:
    if not feature:
        if target_cell is not None:
            raise RuntimeError("teacher target has no routed physical feature")
        return
    meter.charge("teacher-cell-spike")
    if target_cell is not None:
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


def _workspace_metadata(
    *,
    phase: str,
    self_teacher_count: int,
    fit_teacher_count: int,
    base_checkpoint_sha256: str | None,
    training_work: WorkLedger,
    schedule_sha256s: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema": "public-train-prompt-selection-v4-checkpoint",
        "phase": phase,
        "dependency_sha256": DEPENDENCY_SHA256,
        "protocol_sha256": V4_PROTOCOL_ROOT,
        "output_registry_sha256": OUTPUT_REGISTRY_SHA256,
        "branch_registry_sha256": BRANCH_REGISTRY_SHA256,
        "output_cells": tuple(row.record() for row in OUTPUT_CELLS),
        "channel_branch": CHANNEL_BRANCH.record(),
        "output_branches": tuple(row.record() for row in OUTPUT_BRANCHES),
        "array_shape": (len(OUTPUT_CELLS), INPUT_CELL_COUNT),
        "array_dtype": np.dtype(np.uint8).str,
        "tick_cap": TICK_CAP,
        "self_posed_teacher_lesson_count": self_teacher_count,
        "canonical_fit_teacher_identity_count": fit_teacher_count,
        "heldout_teacher_identity_count": 0,
        "heldout_target_read_count": 0,
        "whole_target_head_count": 0,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "training_schedule_sha256s": schedule_sha256s,
        "schedule_sha256s_are_authentication_only": True,
        "training_work": training_work.record(),
    }


_RACE = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)


def _decide_branch(
    workspace: FrozenPromptSelectionV4Workspace,
    branch: OutputBranch,
    feature: tuple[int, ...],
    meter: _WorkMeter,
    *,
    occurrence_index: int,
    masked_cell_ids: set[int],
) -> BranchDecision:
    candidates = tuple(
        cell for cell in branch.candidate_cell_ids if cell not in masked_cell_ids
    )
    masked_count = len(branch.candidate_cell_ids) - len(candidates)
    if feature and candidates:
        candidate_rows = np.asarray(candidates, dtype=np.intp)
        active = np.asarray(feature, dtype=np.intp)
        raw_array = np.sum(
            workspace._excitatory[candidate_rows][:, active].astype(np.int16)
            - workspace._inhibitory[candidate_rows][:, active].astype(np.int16),
            axis=1,
            dtype=np.int64,
        )
        meter.charge("synaptic-tick-read", len(candidates) * len(feature) * 2)
        raw = tuple(int(value) for value in raw_array)
    else:
        raw = tuple(0 for _cell in candidates)
    current_scores = (
        *(min(max(score, 0), len(race_spec.PROTOCOL.latency_table) - 1) for score in raw),
        STOP_SCORE,
    )
    currents = np.asarray(
        tuple(race_spec.PROTOCOL.latency_table[score].current_pa for score in current_scores),
        dtype=np.float64,
    )
    race = _RACE.race_many(currents[None, :], require_unique=False)[0]
    meter.charge("principal-current-write", len(current_scores))
    meter.charge("glif-principal-cell-step", len(current_scores) * RACE_STEPS)
    meter.charge("repeated-masked-wta-decision")
    if race.simultaneous_spike_candidates != 1:
        decision, selected_cell, selected_target = "COLLISION", None, None
    elif race.winner_cell == len(candidates):
        decision, selected_cell, selected_target = "STOP", None, None
    else:
        selected_cell = candidates[race.winner_cell]
        selected_target = OUTPUT_CELLS[selected_cell].target_id
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
        "masked_candidate_count": masked_count,
        "first_spike_step": race.spike_step,
        "simultaneous_spike_candidates": race.simultaneous_spike_candidates,
        "feature_cells": feature,
    }
    return BranchDecision(
        **record,
        sha256=_namespaced_sha256(_DECISION_NAMESPACE, record),
    )


def _expected_self_by_branch(
    lesson: protocol_v4.SelfPosedRelayLesson,
) -> tuple[tuple[str, str | None], ...]:
    targets = set((*lesson.target_atom_ids, *lesson.target_transition_ids))
    rows: list[tuple[str, str | None]] = []
    for branch in OUTPUT_BRANCHES:
        if branch.channel != lesson.audit_channel:
            continue
        matches = tuple(
            cell_id
            for cell_id in branch.candidate_cell_ids
            if OUTPUT_CELLS[cell_id].target_id in targets
        )
        if len(matches) > 1:
            raise RuntimeError("self-posed teacher has two targets in one branch")
        rows.append(
            (
                branch.branch_id,
                OUTPUT_CELLS[matches[0]].target_id if matches else None,
            )
        )
    if {target for _branch, target in rows if target is not None} != targets:
        raise RuntimeError("self-posed teacher left the branch-local registry")
    return tuple(rows)


def _train_self_workspace(
    *,
    teachers: tuple[protocol_v4.SelfPosedRelayLesson, ...],
    teacher_enabled: bool,
    require_teacher_route: bool = True,
) -> tuple[FrozenPromptSelectionV4Workspace, WorkLedger]:
    lessons = protocol_v4.LESSONS
    if len(lessons) != 110 or len(teachers) != len(lessons):
        raise RuntimeError("V4 self-posed training requires exactly 110 lessons")
    excitatory = np.zeros((len(OUTPUT_CELLS), INPUT_CELL_COUNT), dtype=np.uint8)
    inhibitory = np.zeros_like(excitatory)
    meter = _WorkMeter()
    for lesson, teacher in zip(lessons, teachers, strict=True):
        trace = _encode_self_trace(lesson, mode="normal", meter=meter)
        if teacher.audit_channel != lesson.audit_channel:
            raise RuntimeError("self-posed derangement crossed a channel boundary")
        if not teacher_enabled:
            continue
        channel_target = _CHANNEL_CELL_BY_TARGET[teacher.audit_channel]
        _update_branch(
            excitatory,
            inhibitory,
            CHANNEL_BRANCH,
            _self_channel_feature(trace, meter),
            channel_target,
            meter,
        )
        expected = dict(_expected_self_by_branch(teacher))
        for branch in OUTPUT_BRANCHES:
            if branch.channel != teacher.audit_channel:
                continue
            target_id = expected[branch.branch_id]
            target_cell = _CELL_BY_TARGET[target_id].cell_id if target_id is not None else None
            feature = _self_branch_feature(branch, trace, meter)
            if target_cell is not None and not feature:
                if require_teacher_route:
                    raise RuntimeError("self-posed teacher target left its physical branch")
                continue
            _update_branch(
                excitatory,
                inhibitory,
                branch,
                feature,
                target_cell,
                meter,
            )
    if np.count_nonzero(excitatory[:, PROMPT_FEATURE_OFFSET:]) or np.count_nonzero(
        inhibitory[:, PROMPT_FEATURE_OFFSET:]
    ):
        raise RuntimeError("self-posed pass wrote the canonical prompt feature bank")
    work = meter.freeze()
    workspace = FrozenPromptSelectionV4Workspace(
        excitatory=excitatory,
        inhibitory=inhibitory,
        metadata=_workspace_metadata(
            phase="self-posed",
            self_teacher_count=len(lessons) if teacher_enabled else 0,
            fit_teacher_count=0,
            base_checkpoint_sha256=None,
            training_work=work,
            schedule_sha256s=(
                tuple(row.sha256 for row in lessons) if teacher_enabled else ()
            ),
        ),
    )
    return workspace, work


def _replay_self_workspace(
    workspace: FrozenPromptSelectionV4Workspace,
    *,
    name: str,
    mode: str,
) -> SelfReplayBatch:
    workspace.authenticate()
    meter = _WorkMeter()
    rows: list[SelfLessonReplay] = []
    for lesson in protocol_v4.LESSONS:
        trace = _encode_self_trace(lesson, mode=mode, meter=meter)
        route = _decide_branch(
            workspace,
            CHANNEL_BRANCH,
            _self_channel_feature(trace, meter),
            meter,
            occurrence_index=0,
            masked_cell_ids=set(),
        )
        emitted_channel = route.selected_target_id
        branch_decisions: list[BranchDecision] = []
        emitted: list[tuple[str, str | None]] = []
        occurrence_mask: set[int] = set()
        if emitted_channel is not None:
            for branch in OUTPUT_BRANCHES:
                if branch.channel != emitted_channel:
                    continue
                decision = _decide_branch(
                    workspace,
                    branch,
                    _self_branch_feature(branch, trace, meter),
                    meter,
                    occurrence_index=0,
                    masked_cell_ids=occurrence_mask,
                )
                branch_decisions.append(decision)
                emitted.append((branch.branch_id, decision.selected_target_id))
                if decision.selected_cell_id is not None:
                    occurrence_mask.add(decision.selected_cell_id)
        expected = _expected_self_by_branch(lesson)
        exact = emitted_channel == lesson.audit_channel and tuple(emitted) == expected
        decisions = (route, *branch_decisions)
        record = {
            "lesson_index": lesson.lesson_index,
            "channel": lesson.audit_channel,
            "trace_sha256": trace.sha256,
            "decisions": tuple(row.record() for row in decisions),
            "emitted_channel": emitted_channel,
            "emitted_by_branch": tuple(emitted),
            "expected_by_branch": expected,
            "exact": exact,
        }
        rows.append(
            SelfLessonReplay(
                lesson_index=lesson.lesson_index,
                channel=lesson.audit_channel,
                trace_sha256=trace.sha256,
                decisions=tuple(decisions),
                emitted_channel=emitted_channel,
                emitted_by_branch=tuple(emitted),
                expected_by_branch=expected,
                exact=exact,
                sha256=_namespaced_sha256(_SELF_REPLAY_NAMESPACE, record),
            )
        )
    workspace.authenticate()
    channel_rows = tuple(
        ChannelOutcome(
            schema.channel,
            sum(row.channel == schema.channel for row in rows),
            sum(row.channel == schema.channel and row.exact for row in rows),
            sum(row.channel == schema.channel and not row.exact for row in rows),
        )
        for schema in protocol_v4.RELAY_SCHEMAS
    )
    work = meter.freeze()
    exact_count = sum(row.exact for row in rows)
    record = {
        "name": name,
        "mode": mode,
        "lesson_count": len(rows),
        "exact_lesson_count": exact_count,
        "mismatch_lesson_count": len(rows) - exact_count,
        "channel_outcomes": tuple(row.record() for row in channel_rows),
        "replays": tuple(row.record() for row in rows),
        "work": work.record(),
    }
    return SelfReplayBatch(
        name=name,
        mode=mode,
        lesson_count=len(rows),
        exact_lesson_count=exact_count,
        mismatch_lesson_count=len(rows) - exact_count,
        channel_outcomes=channel_rows,
        replays=tuple(rows),
        work=work,
        sha256=_namespaced_sha256(_SELF_BATCH_NAMESPACE, record),
    )


def _control_outcome(
    name: str,
    workspace: FrozenPromptSelectionV4Workspace,
    replay: SelfReplayBatch,
    training_work: WorkLedger | None = None,
) -> ControlOutcome:
    meter = _WorkMeter()
    meter.merge(replay.work)
    if training_work is not None:
        meter.merge(training_work)
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


def _deranged_self_teachers() -> tuple[protocol_v4.SelfPosedRelayLesson, ...]:
    by_channel: dict[str, list[protocol_v4.SelfPosedRelayLesson]] = defaultdict(list)
    for lesson in protocol_v4.LESSONS:
        by_channel[lesson.audit_channel].append(lesson)
    deranged: dict[int, protocol_v4.SelfPosedRelayLesson] = {}
    for rows in by_channel.values():
        if len(rows) < 2:
            raise RuntimeError("self-posed teacher derangement lacks a channel peer")
        for index, lesson in enumerate(rows):
            deranged[lesson.lesson_index] = rows[(index + 1) % len(rows)]
    return tuple(deranged[index] for index in range(len(protocol_v4.LESSONS)))


@dataclass(frozen=True, slots=True)
class _PhysicalFitInput:
    causal_key_sha256: str
    span_events: tuple[Mapping[str, object], ...]
    literal_events: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PromptTrace:
    causal_key_sha256: str
    physical_event_count: int
    lane_cells: tuple[int, ...]
    active_cells: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_key_sha256": self.causal_key_sha256,
            "physical_event_count": self.physical_event_count,
            "lane_cells": self.lane_cells,
            "active_cells": self.active_cells,
            "causal_key_sha256_is_authentication_only": True,
            "neural_sources": (
                "ordered-physical-span-events",
                "ordered-typed-literal-cell-events",
                "ordered-name-free-signature-role-type-events",
            ),
            "task_family_entrypoint_or_audit_hash_neural_input": False,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class OccurrenceState:
    cells: tuple[int, ...]
    sha256: str

    def record(self) -> dict[str, object]:
        return {"cells": self.cells, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class OccurrenceBinding:
    parent_cells: tuple[int, ...]
    child_cells: tuple[int, ...]
    sha256: str

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class CanonicalTeacherOccurrence:
    sequence_index: int
    channel: str
    expected_by_branch: tuple[tuple[str, str | None], ...]
    edge_serialization: object | None

    def record(self) -> dict[str, object]:
        return {
            "sequence_index": self.sequence_index,
            "channel": self.channel,
            "expected_by_branch": self.expected_by_branch,
            "edge_serialization": self.edge_serialization,
            "occurrence_integer_handles_are_teacher_serialization_only": True,
        }


@dataclass(frozen=True, slots=True)
class CanonicalFitItem:
    causal_key_sha256: str
    trace: PromptTrace
    occurrences: tuple[CanonicalTeacherOccurrence, ...]
    projection_receipt_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "causal_key_sha256": self.causal_key_sha256,
            "trace": self.trace.record(),
            "occurrences": tuple(row.record() for row in self.occurrences),
            "projection_receipt_sha256": self.projection_receipt_sha256,
            "causal_key_and_receipt_hashes_are_authentication_only": True,
        }


@dataclass(frozen=True, slots=True)
class FitPreflight:
    status: str
    fit_identity_count: int
    prompt_trace_collision_count: int
    route_feature_target_conflict_count: int
    branch_feature_target_conflict_count: int
    occurrence_overflow_count: int
    binding_self_alias_count: int
    projection_receipt_mismatch_count: int
    maximum_occurrence_count: int
    target_occurrence_count: int
    edge_occurrence_count: int
    heldout_target_read_count: int
    heldout_replay_identity_count: int
    work: WorkLedger
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in {"work", "sha256"}
        }
        record["work"] = self.work.record()
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CanonicalOccurrenceReplay:
    sequence_index: int
    state_sha256: str
    channel_decision: BranchDecision
    selected_channel: str | None
    branch_decisions: tuple[BranchDecision, ...]
    emitted_by_branch: tuple[tuple[str, str | None], ...]
    expected_channel: str | None
    expected_by_branch: tuple[tuple[str, str | None], ...]
    occurrence_binding: OccurrenceBinding | None
    exact: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "sequence_index": self.sequence_index,
            "state_sha256": self.state_sha256,
            "channel_decision": self.channel_decision.record(),
            "selected_channel": self.selected_channel,
            "branch_decisions": tuple(row.record() for row in self.branch_decisions),
            "emitted_by_branch": self.emitted_by_branch,
            "expected_channel": self.expected_channel,
            "expected_by_branch": self.expected_by_branch,
            "occurrence_binding": (
                self.occurrence_binding.record() if self.occurrence_binding is not None else None
            ),
            "exact": self.exact,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CanonicalIdentityReplay:
    causal_key_sha256: str
    trace_sha256: str
    occurrence_replays: tuple[CanonicalOccurrenceReplay, ...]
    emitted_occurrence_count: int
    expected_occurrence_count: int
    termination: str
    exact: bool
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_key_sha256": self.causal_key_sha256,
            "trace_sha256": self.trace_sha256,
            "occurrence_replays": tuple(row.record() for row in self.occurrence_replays),
            "emitted_occurrence_count": self.emitted_occurrence_count,
            "expected_occurrence_count": self.expected_occurrence_count,
            "termination": self.termination,
            "exact": self.exact,
            "causal_key_sha256_is_admission_metadata_only": True,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class CanonicalFitReplayBatch:
    identity_count: int
    exact_identity_count: int
    mismatch_identity_count: int
    emitted_occurrence_count: int
    expected_occurrence_count: int
    channel_outcomes: tuple[ChannelOutcome, ...]
    replays: tuple[CanonicalIdentityReplay, ...]
    checkpoint_authentication_count: int
    work: WorkLedger
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "identity_count": self.identity_count,
            "exact_identity_count": self.exact_identity_count,
            "mismatch_identity_count": self.mismatch_identity_count,
            "emitted_occurrence_count": self.emitted_occurrence_count,
            "expected_occurrence_count": self.expected_occurrence_count,
            "channel_outcomes": tuple(row.record() for row in self.channel_outcomes),
            "replays": tuple(row.record() for row in self.replays),
            "checkpoint_authentication_count": self.checkpoint_authentication_count,
            "work": self.work.record(),
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


_PROMPT_MULTIPLIERS = (
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    299,
    301,
    303,
    307,
)
_PROMPT_POSITION_GAINS = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_PROMPT_BIASES = (17, 43, 71, 101, 137, 173, 211, 251, 293, 337, 383, 431)
_STOP_STATE_BASE = len(OUTPUT_CELLS) + len(OUTPUT_BRANCHES) + 1


def _load_fit_physical_inputs() -> tuple[_PhysicalFitInput, ...]:
    from experiments.r3_pilot import (
        public_train_prompt_factor_manifest_v1 as prompt_manifest,
    )
    from experiments.r3_pilot import (
        public_train_prompt_selection_freeze_v1_protocol as selection_v1,
    )

    if selection_v1.PROTOCOL.protocol_sha256 != (
        "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47"
    ):
        raise RuntimeError(STOP_FIT_PREFLIGHT)
    observed = prompt_manifest.load_public_train_prompt_inputs()
    bindings = selection_v1.PROTOCOL.signature_bindings
    if len(observed) != len(bindings) or len(observed) != 90:
        raise RuntimeError(STOP_FIT_PREFLIGHT)
    grouped: dict[str, _PhysicalFitInput] = {}
    for prompt_input, binding in zip(observed, bindings, strict=True):
        if prompt_input.sha256 != binding.prompt_input_sha256:
            raise RuntimeError(STOP_FIT_PREFLIGHT)
        spans = selection_v1._physical_span_event_record(prompt_input)
        literals = selection_v1._typed_literal_event_record(prompt_input)
        span_root = selection_v1._namespaced_sha256(selection_v1._SPAN_EVENTS_NAMESPACE, spans)
        literal_root = selection_v1._namespaced_sha256(
            selection_v1._LITERAL_EVENTS_NAMESPACE, literals
        )
        causal = binding.causal_key
        if (
            span_root != causal.physical_span_events_sha256
            or literal_root != causal.typed_literal_events_sha256
            or binding.signature_role_sha256 != causal.signature_role_events_sha256
        ):
            raise RuntimeError(STOP_FIT_PREFLIGHT)
        row = _PhysicalFitInput(
            causal_key_sha256=causal.sha256,
            span_events=tuple(_freeze(record) for record in spans),
            literal_events=tuple(_freeze(record) for record in literals),
            signature_role_record=binding.signature_role_record,
        )
        incumbent = grouped.setdefault(binding.prompt_input_sha256, row)
        if incumbent != row:
            raise RuntimeError(STOP_FIT_PREFLIGHT)
    fit_rows: list[_PhysicalFitInput] = []
    split_counts = Counter(row.split for row in selection_v1.PROTOCOL.canonical_inputs)
    for canonical in selection_v1.PROTOCOL.canonical_inputs:
        if canonical.split != FIT:
            continue
        physical = grouped.get(canonical.prompt_input_sha256)
        if physical is None or physical.causal_key_sha256 != canonical.causal_key_sha256:
            raise RuntimeError(STOP_FIT_PREFLIGHT)
        fit_rows.append(physical)
    if len(fit_rows) != 24 or split_counts != {"FIT": 24, "HELDOUT": 18}:
        raise RuntimeError(STOP_FIT_PREFLIGHT)
    return tuple(fit_rows)


def _physical_prompt_event_stream(row: _PhysicalFitInput) -> tuple[int, ...]:
    events: list[int] = [10_001, len(row.span_events)]
    for ordinal, span in enumerate(row.span_events):
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


def _encode_prompt_trace(row: _PhysicalFitInput, meter: _WorkMeter) -> PromptTrace:
    events = _physical_prompt_event_stream(row)
    state = tuple((lane * 37 + 11) % PROMPT_CELLS_PER_LANE for lane in range(PROMPT_LANE_COUNT))
    for position, event in enumerate(events, start=1):
        state = tuple(
            (
                state[lane] * _PROMPT_MULTIPLIERS[lane]
                + int(event)
                + position * _PROMPT_POSITION_GAINS[lane]
                + _PROMPT_BIASES[lane]
            )
            % PROMPT_CELLS_PER_LANE
            for lane in range(PROMPT_LANE_COUNT)
        )
    active = tuple(
        PROMPT_FEATURE_OFFSET + lane * PROMPT_CELLS_PER_LANE + state[lane]
        for lane in range(PROMPT_LANE_COUNT)
    )
    meter.charge("canonical-physical-causal-event", len(events))
    meter.charge("canonical-recurrent-cell-transition", len(events) * PROMPT_LANE_COUNT)
    meter.charge("canonical-trace-cell-emission", PROMPT_LANE_COUNT)
    record = {
        "causal_key_sha256": row.causal_key_sha256,
        "physical_event_count": len(events),
        "lane_cells": state,
        "active_cells": active,
        "causal_key_sha256_is_authentication_only": True,
    }
    return PromptTrace(
        causal_key_sha256=row.causal_key_sha256,
        physical_event_count=len(events),
        lane_cells=state,
        active_cells=active,
        sha256=_namespaced_sha256(_TRACE_NAMESPACE + b"canonical\0", record),
    )


def _expected_targets_by_branch(
    channel: str, targets: tuple[str, ...]
) -> tuple[tuple[str, str | None], ...]:
    rows: list[tuple[str, str | None]] = []
    seen: list[str] = []
    for branch in OUTPUT_BRANCHES:
        if branch.channel != channel:
            continue
        matches = tuple(
            cell_id
            for cell_id in branch.candidate_cell_ids
            if OUTPUT_CELLS[cell_id].target_id in targets
        )
        if len(matches) > 1:
            raise RuntimeError("canonical teacher has two targets in one physical branch")
        target = OUTPUT_CELLS[matches[0]].target_id if matches else None
        if target is not None:
            seen.append(target)
        rows.append((branch.branch_id, target))
    if len(seen) != len(targets) or any(target not in seen for target in targets):
        raise RuntimeError("canonical teacher left the frozen output registry")
    return tuple(rows)


def _attach_fit_teachers(traces: tuple[PromptTrace, ...]) -> tuple[CanonicalFitItem, ...]:
    from experiments.r3_pilot import (
        public_train_structural_annotation_capsule_v1 as annotations_v1,
    )

    if annotations_v1.CAPSULE.capsule_sha256 != (
        "484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf"
    ):
        raise RuntimeError(STOP_FIT_PREFLIGHT)
    receipt_by_key = {
        row.causal_key_sha256: row for row in protocol_v4.FIT_PROJECTION_RECEIPTS
    }
    items: list[CanonicalFitItem] = []
    for trace in traces:
        annotation = next(
            row
            for row in annotations_v1.CAPSULE.annotations
            if row.causal_key_sha256 == trace.causal_key_sha256
        )
        target_record, _removed, _edges, _repeated = protocol_v4._fit_target_record(
            annotation
        )
        reduced_sha = protocol_v4._namespaced_sha256(
            protocol_v4._FIT_TARGET_NAMESPACE, target_record
        )
        receipt = receipt_by_key.get(trace.causal_key_sha256)
        if receipt is None or receipt.reduced_target_sha256 != reduced_sha:
            raise RuntimeError(STOP_FIT_PREFLIGHT)
        edge_by_sequence = {
            int(row["sequence_index"]): _freeze(row)
            for row in target_record["immediate_edge_occurrences"]
        }
        occurrences: list[CanonicalTeacherOccurrence] = []
        for row in target_record["semantic_occurrences"]:
            sequence_index = int(row["sequence_index"])
            channel = str(row["channel"])
            targets = tuple(str(target) for target in row["targets"])
            occurrences.append(
                CanonicalTeacherOccurrence(
                    sequence_index=sequence_index,
                    channel=channel,
                    expected_by_branch=_expected_targets_by_branch(channel, targets),
                    edge_serialization=edge_by_sequence.get(sequence_index),
                )
            )
        if tuple(row.sequence_index for row in occurrences) != tuple(range(len(occurrences))):
            raise RuntimeError(STOP_FIT_PREFLIGHT)
        items.append(
            CanonicalFitItem(
                causal_key_sha256=trace.causal_key_sha256,
                trace=trace,
                occurrences=tuple(occurrences),
                projection_receipt_sha256=receipt.sha256,
            )
        )
    if (
        len(items) != 24
        or {row.causal_key_sha256 for row in items} != set(receipt_by_key)
        or len({row.causal_key_sha256 for row in items}) != 24
    ):
        raise RuntimeError(STOP_FIT_PREFLIGHT)
    return tuple(items)


def _state_from_cells(cells: tuple[int, ...]) -> OccurrenceState:
    if (
        len(cells) != PROMPT_LANE_COUNT
        or any(not 0 <= cell < PROMPT_CELLS_PER_LANE for cell in cells)
    ):
        raise RuntimeError("distributed occurrence state changed")
    record = {"cells": cells, "integer_handle_neural_input": False}
    return OccurrenceState(cells, _namespaced_sha256(_BINDING_NAMESPACE + b"state\0", record))


def _initial_occurrence_state(trace: PromptTrace) -> OccurrenceState:
    cells = tuple(
        (
            trace.lane_cells[lane] * 313
            + _PROMPT_POSITION_GAINS[lane] * 17
            + _PROMPT_BIASES[lane]
        )
        % PROMPT_CELLS_PER_LANE
        for lane in range(PROMPT_LANE_COUNT)
    )
    return _state_from_cells(cells)


def _advance_occurrence_state(
    state: OccurrenceState, physical_winner_cells: tuple[int, ...]
) -> OccurrenceState:
    winner_drive = sum(
        (position + 1) * (cell + 1)
        for position, cell in enumerate(physical_winner_cells)
    )
    cells = tuple(
        (
            state.cells[lane] * _PROMPT_MULTIPLIERS[lane]
            + winner_drive
            + _PROMPT_BIASES[lane]
        )
        % PROMPT_CELLS_PER_LANE
        for lane in range(PROMPT_LANE_COUNT)
    )
    return _state_from_cells(cells)


def _occurrence_binding(state: OccurrenceState) -> OccurrenceBinding:
    def side_cells(side: int) -> tuple[int, ...]:
        used: set[int] = set()
        cells: list[int] = []
        for lane, base in enumerate(state.cells):
            candidate = (
                base * (337 if side == 0 else 347)
                + _PROMPT_BIASES[lane]
                + side * 149
            ) % protocol_v4.PHYSICAL_CELL_COUNT
            while candidate in used:
                candidate = (candidate + 1) % protocol_v4.PHYSICAL_CELL_COUNT
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
    return OccurrenceBinding(
        parent,
        child,
        _namespaced_sha256(_BINDING_NAMESPACE + b"edge\0", record),
    )


def _prompt_feature_cells(
    trace: PromptTrace,
    state: OccurrenceState,
    *,
    relation_cells: tuple[int, ...] = (),
    role_cells: tuple[int, ...] = (),
    binding_cells: tuple[int, ...] = (),
    kind_code: int,
) -> tuple[int, ...]:
    cells: list[int] = []
    for lane in range(PROMPT_LANE_COUNT):
        relation = relation_cells[lane] if relation_cells else 0
        role = role_cells[lane] if role_cells else 0
        binding = binding_cells[lane] if binding_cells else state.cells[lane]
        local = (
            trace.lane_cells[lane] * _PROMPT_MULTIPLIERS[lane]
            + state.cells[lane] * 53
            + relation * 17
            + role * 29
            + binding * 31
            + kind_code * 43
            + _PROMPT_BIASES[lane]
        ) % PROMPT_CELLS_PER_LANE
        cells.append(PROMPT_FEATURE_OFFSET + lane * PROMPT_CELLS_PER_LANE + local)
    return tuple(cells)


def _canonical_channel_feature(
    trace: PromptTrace, state: OccurrenceState, meter: _WorkMeter
) -> tuple[int, ...]:
    meter.charge("canonical-channel-conjunction-synapse-read", PROMPT_LANE_COUNT * 3)
    meter.charge("canonical-channel-feature-emission", PROMPT_LANE_COUNT)
    return _prompt_feature_cells(trace, state, kind_code=1)


def _canonical_branch_feature(
    trace: PromptTrace,
    state: OccurrenceState,
    branch: OutputBranch,
    meter: _WorkMeter,
) -> tuple[int, ...]:
    relation = protocol_v4._assembly_cells("relation", branch.channel)
    role = protocol_v4._assembly_cells("role", f"{branch.channel}|{branch.branch_role}")
    if branch.channel == "ordered-parent-child-edge":
        binding = _occurrence_binding(state)
        binding_cells = tuple(
            (binding.parent_cells[lane] + binding.child_cells[lane])
            % protocol_v4.PHYSICAL_CELL_COUNT
            for lane in range(PROMPT_LANE_COUNT)
        )
    else:
        binding_cells = state.cells
    meter.charge("canonical-branch-conjunction-synapse-read", PROMPT_LANE_COUNT * 6)
    meter.charge("canonical-branch-feature-emission", PROMPT_LANE_COUNT)
    return _prompt_feature_cells(
        trace,
        state,
        relation_cells=relation,
        role_cells=role,
        binding_cells=binding_cells,
        kind_code=2 if branch.kind == "atom" else 3,
    )


def _winner_state_cells(
    channel: str, expected_by_branch: tuple[tuple[str, str | None], ...]
) -> tuple[int, ...]:
    winners = [_CHANNEL_CELL_BY_TARGET[channel]]
    for branch_index, (branch_id, target_id) in enumerate(expected_by_branch):
        if target_id is None:
            winners.append(_STOP_STATE_BASE + branch_index)
        else:
            winners.append(_CELL_BY_TARGET[target_id].cell_id)
        if branch_id not in _BRANCH_BY_ID:
            raise RuntimeError(STOP_FIT_PREFLIGHT)
    return tuple(winners)


def _audit_fit_preflight(
    items: tuple[CanonicalFitItem, ...], trace_work: WorkLedger
) -> FitPreflight:
    meter = _WorkMeter()
    meter.merge(trace_work)
    trace_counts = Counter(item.trace.lane_cells for item in items)
    trace_collisions = sum(count - 1 for count in trace_counts.values() if count > 1)
    route_targets: dict[tuple[int, ...], str] = {}
    branch_targets: dict[tuple[str, tuple[int, ...]], str] = {}
    route_conflicts = 0
    branch_conflicts = 0
    overflow = 0
    binding_alias = 0
    receipt_mismatch = 0
    edge_count = 0
    max_occurrences = 0
    for item in items:
        max_occurrences = max(max_occurrences, len(item.occurrences))
        overflow += len(item.occurrences) >= OCCURRENCE_CAP
        state = _initial_occurrence_state(item.trace)
        for occurrence in item.occurrences:
            route_feature = _canonical_channel_feature(item.trace, state, meter)
            incumbent = route_targets.setdefault(route_feature, occurrence.channel)
            route_conflicts += incumbent != occurrence.channel
            for branch_id, target_id in occurrence.expected_by_branch:
                branch = _BRANCH_BY_ID[branch_id]
                feature = _canonical_branch_feature(item.trace, state, branch, meter)
                target = target_id if target_id is not None else "<STOP>"
                incumbent_target = branch_targets.setdefault((branch_id, feature), target)
                branch_conflicts += incumbent_target != target
            if occurrence.channel == "ordered-parent-child-edge":
                binding = _occurrence_binding(state)
                binding_alias += binding.parent_cells == binding.child_cells
                edge_count += 1
            state = _advance_occurrence_state(
                state,
                _winner_state_cells(occurrence.channel, occurrence.expected_by_branch),
            )
        terminal_feature = _canonical_channel_feature(item.trace, state, meter)
        incumbent_terminal = route_targets.setdefault(terminal_feature, "<STOP>")
        route_conflicts += incumbent_terminal != "<STOP>"
        receipt = next(
            row
            for row in protocol_v4.FIT_PROJECTION_RECEIPTS
            if row.causal_key_sha256 == item.causal_key_sha256
        )
        receipt_mismatch += receipt.sha256 != item.projection_receipt_sha256
    status = (
        protocol_v4.PASS_PROTOCOL
        if (
            len(items) == 24
            and trace_collisions == 0
            and route_conflicts == 0
            and branch_conflicts == 0
            and overflow == 0
            and binding_alias == 0
            and receipt_mismatch == 0
        )
        else STOP_FIT_PREFLIGHT
    )
    work = meter.freeze()
    record = {
        "status": status,
        "fit_identity_count": len(items),
        "prompt_trace_collision_count": trace_collisions,
        "route_feature_target_conflict_count": route_conflicts,
        "branch_feature_target_conflict_count": branch_conflicts,
        "occurrence_overflow_count": overflow,
        "binding_self_alias_count": binding_alias,
        "projection_receipt_mismatch_count": receipt_mismatch,
        "maximum_occurrence_count": max_occurrences,
        "target_occurrence_count": sum(len(item.occurrences) for item in items),
        "edge_occurrence_count": edge_count,
        "heldout_target_read_count": 0,
        "heldout_replay_identity_count": 0,
        "work": work.record(),
    }
    return FitPreflight(
        status=status,
        fit_identity_count=len(items),
        prompt_trace_collision_count=trace_collisions,
        route_feature_target_conflict_count=route_conflicts,
        branch_feature_target_conflict_count=branch_conflicts,
        occurrence_overflow_count=overflow,
        binding_self_alias_count=binding_alias,
        projection_receipt_mismatch_count=receipt_mismatch,
        maximum_occurrence_count=max_occurrences,
        target_occurrence_count=sum(len(item.occurrences) for item in items),
        edge_occurrence_count=edge_count,
        heldout_target_read_count=0,
        heldout_replay_identity_count=0,
        work=work,
        sha256=_namespaced_sha256(_FIT_PREFLIGHT_NAMESPACE, record),
    )


def _train_canonical_fit_once(
    base: FrozenPromptSelectionV4Workspace,
    items: tuple[CanonicalFitItem, ...],
) -> tuple[FrozenPromptSelectionV4Workspace, WorkLedger]:
    if len(items) != 24:
        raise RuntimeError(STOP_FIT_PREFLIGHT)
    base.authenticate()
    excitatory = base.excitatory_ticks.copy(order="C")
    inhibitory = base.inhibitory_ticks.copy(order="C")
    excitatory.setflags(write=True)
    inhibitory.setflags(write=True)
    meter = _WorkMeter()
    for item in items:
        state = _initial_occurrence_state(item.trace)
        for occurrence in item.occurrences:
            _update_branch(
                excitatory,
                inhibitory,
                CHANNEL_BRANCH,
                _canonical_channel_feature(item.trace, state, meter),
                _CHANNEL_CELL_BY_TARGET[occurrence.channel],
                meter,
            )
            for branch_id, target_id in occurrence.expected_by_branch:
                branch = _BRANCH_BY_ID[branch_id]
                target_cell = (
                    _CELL_BY_TARGET[target_id].cell_id if target_id is not None else None
                )
                _update_branch(
                    excitatory,
                    inhibitory,
                    branch,
                    _canonical_branch_feature(item.trace, state, branch, meter),
                    target_cell,
                    meter,
                )
            state = _advance_occurrence_state(
                state,
                _winner_state_cells(occurrence.channel, occurrence.expected_by_branch),
            )
        _update_branch(
            excitatory,
            inhibitory,
            CHANNEL_BRANCH,
            _canonical_channel_feature(item.trace, state, meter),
            None,
            meter,
        )
    work = meter.freeze()
    workspace = FrozenPromptSelectionV4Workspace(
        excitatory=excitatory,
        inhibitory=inhibitory,
        metadata=_workspace_metadata(
            phase="canonical-fit",
            self_teacher_count=110,
            fit_teacher_count=24,
            base_checkpoint_sha256=base.checkpoint_sha256,
            training_work=work,
            schedule_sha256s=tuple(item.projection_receipt_sha256 for item in items),
        ),
    )
    base.authenticate()
    return workspace, work


def _replay_canonical_fit_once(
    workspace: FrozenPromptSelectionV4Workspace,
    items: tuple[CanonicalFitItem, ...],
) -> CanonicalFitReplayBatch:
    workspace.authenticate()
    checkpoint_before = workspace.checkpoint_sha256
    meter = _WorkMeter()
    identity_rows: list[CanonicalIdentityReplay] = []
    channel_exact: Counter[str] = Counter()
    channel_total: Counter[str] = Counter()
    for item in items:
        state = _initial_occurrence_state(item.trace)
        occurrence_rows: list[CanonicalOccurrenceReplay] = []
        termination = "CAP"
        for sequence_index in range(OCCURRENCE_CAP):
            route = _decide_branch(
                workspace,
                CHANNEL_BRANCH,
                _canonical_channel_feature(item.trace, state, meter),
                meter,
                occurrence_index=sequence_index,
                masked_cell_ids=set(),
            )
            if route.decision == "STOP":
                termination = "STOP"
                break
            if route.decision != "RELAY" or route.selected_target_id is None:
                termination = route.decision
                break
            selected_channel = route.selected_target_id
            branch_rows: list[BranchDecision] = []
            emitted: list[tuple[str, str | None]] = []
            occurrence_mask: set[int] = set()
            physical_winners: list[int] = [route.selected_cell_id]
            for branch in OUTPUT_BRANCHES:
                if branch.channel != selected_channel:
                    continue
                decision = _decide_branch(
                    workspace,
                    branch,
                    _canonical_branch_feature(item.trace, state, branch, meter),
                    meter,
                    occurrence_index=sequence_index,
                    masked_cell_ids=occurrence_mask,
                )
                branch_rows.append(decision)
                emitted.append((branch.branch_id, decision.selected_target_id))
                if decision.selected_cell_id is not None:
                    occurrence_mask.add(decision.selected_cell_id)
                    physical_winners.append(decision.selected_cell_id)
                else:
                    physical_winners.append(_STOP_STATE_BASE + len(branch_rows) - 1)
            expected = (
                item.occurrences[sequence_index]
                if sequence_index < len(item.occurrences)
                else None
            )
            expected_channel = expected.channel if expected is not None else None
            expected_by_branch = expected.expected_by_branch if expected is not None else ()
            exact = (
                expected is not None
                and selected_channel == expected_channel
                and tuple(emitted) == expected_by_branch
            )
            if expected is not None:
                channel_total[expected.channel] += 1
                channel_exact[expected.channel] += exact
            binding = (
                _occurrence_binding(state)
                if selected_channel == "ordered-parent-child-edge"
                else None
            )
            record = {
                "sequence_index": sequence_index,
                "state_sha256": state.sha256,
                "channel_decision": route.record(),
                "selected_channel": selected_channel,
                "branch_decisions": tuple(row.record() for row in branch_rows),
                "emitted_by_branch": tuple(emitted),
                "expected_channel": expected_channel,
                "expected_by_branch": expected_by_branch,
                "occurrence_binding": binding.record() if binding is not None else None,
                "exact": exact,
            }
            occurrence_rows.append(
                CanonicalOccurrenceReplay(
                    sequence_index=sequence_index,
                    state_sha256=state.sha256,
                    channel_decision=route,
                    selected_channel=selected_channel,
                    branch_decisions=tuple(branch_rows),
                    emitted_by_branch=tuple(emitted),
                    expected_channel=expected_channel,
                    expected_by_branch=expected_by_branch,
                    occurrence_binding=binding,
                    exact=exact,
                    sha256=_namespaced_sha256(_FIT_REPLAY_NAMESPACE, record),
                )
            )
            state = _advance_occurrence_state(state, tuple(physical_winners))
        identity_exact = (
            termination == "STOP"
            and len(occurrence_rows) == len(item.occurrences)
            and all(row.exact for row in occurrence_rows)
        )
        identity_record = {
            "causal_key_sha256": item.causal_key_sha256,
            "trace_sha256": item.trace.sha256,
            "occurrence_replays": tuple(row.record() for row in occurrence_rows),
            "emitted_occurrence_count": len(occurrence_rows),
            "expected_occurrence_count": len(item.occurrences),
            "termination": termination,
            "exact": identity_exact,
            "causal_key_sha256_is_admission_metadata_only": True,
        }
        identity_rows.append(
            CanonicalIdentityReplay(
                causal_key_sha256=item.causal_key_sha256,
                trace_sha256=item.trace.sha256,
                occurrence_replays=tuple(occurrence_rows),
                emitted_occurrence_count=len(occurrence_rows),
                expected_occurrence_count=len(item.occurrences),
                termination=termination,
                exact=identity_exact,
                sha256=_namespaced_sha256(_FIT_REPLAY_NAMESPACE + b"identity\0", identity_record),
            )
        )
    workspace.authenticate()
    if workspace.checkpoint_sha256 != checkpoint_before:
        raise RuntimeError("canonical FIT checkpoint mutated during read-only replay")
    channel_rows = tuple(
        ChannelOutcome(
            schema.channel,
            channel_total[schema.channel],
            channel_exact[schema.channel],
            channel_total[schema.channel] - channel_exact[schema.channel],
        )
        for schema in protocol_v4.RELAY_SCHEMAS
    )
    exact_count = sum(row.exact for row in identity_rows)
    work = meter.freeze()
    record = {
        "identity_count": len(identity_rows),
        "exact_identity_count": exact_count,
        "mismatch_identity_count": len(identity_rows) - exact_count,
        "emitted_occurrence_count": sum(
            row.emitted_occurrence_count for row in identity_rows
        ),
        "expected_occurrence_count": sum(
            row.expected_occurrence_count for row in identity_rows
        ),
        "channel_outcomes": tuple(row.record() for row in channel_rows),
        "replays": tuple(row.record() for row in identity_rows),
        "checkpoint_authentication_count": 2,
        "work": work.record(),
    }
    return CanonicalFitReplayBatch(
        identity_count=len(identity_rows),
        exact_identity_count=exact_count,
        mismatch_identity_count=len(identity_rows) - exact_count,
        emitted_occurrence_count=record["emitted_occurrence_count"],
        expected_occurrence_count=record["expected_occurrence_count"],
        channel_outcomes=channel_rows,
        replays=tuple(identity_rows),
        checkpoint_authentication_count=2,
        work=work,
        sha256=_namespaced_sha256(_FIT_BATCH_NAMESPACE, record),
    )


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    phase: str
    output_cell_count: int
    channel_router_cell_count: int
    atom_cell_count: int
    transition_cell_count: int
    branch_count: int
    array_shape: tuple[int, int]
    array_dtype: str
    array_bytes: int
    tick_cap: int
    excitatory_nonzero_ticks: int
    inhibitory_nonzero_ticks: int
    excitatory_max_tick: int
    inhibitory_max_tick: int
    self_posed_teacher_lesson_count: int
    canonical_fit_teacher_identity_count: int
    heldout_teacher_identity_count: int
    base_checkpoint_sha256: str | None
    checkpoint_sha256: str

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class PublicTrainPromptSelectionV4Result:
    scope: str
    status: str
    base_checkpoint: CheckpointSummary
    checkpoint: CheckpointSummary
    self_posed_result: SelfReplayBatch
    controls: tuple[ControlOutcome, ...]
    fit_preflight: FitPreflight
    fit_result: CanonicalFitReplayBatch | None
    self_training_work: WorkLedger
    fit_training_work: WorkLedger
    total_work: WorkLedger
    physical_work_total: int
    dependency_sha256: str
    output_registry_sha256: str
    branch_registry_sha256: str
    base_checkpoint_sha256: str
    checkpoint_sha256: str
    self_posed_result_sha256: str
    control_panel_sha256: str
    fit_preflight_sha256: str
    fit_result_sha256: str | None
    surface_boundary_sha256: str
    result_sha256: str

    def root_record(self, *, include_result_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "output_registry_sha256": self.output_registry_sha256,
            "branch_registry_sha256": self.branch_registry_sha256,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "self_posed_result_sha256": self.self_posed_result_sha256,
            "control_panel_sha256": self.control_panel_sha256,
            "fit_preflight_sha256": self.fit_preflight_sha256,
            "fit_result_sha256": self.fit_result_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "self_training_work": self.self_training_work.record(),
            "fit_training_work": self.fit_training_work.record(),
            "total_work": self.total_work.record(),
            "physical_work_total": self.physical_work_total,
            "self_posed_lesson_count": self.self_posed_result.lesson_count,
            "self_posed_exact_lesson_count": self.self_posed_result.exact_lesson_count,
            "canonical_fit_identity_count": (
                self.fit_result.identity_count if self.fit_result is not None else 0
            ),
            "canonical_fit_exact_identity_count": (
                self.fit_result.exact_identity_count if self.fit_result is not None else 0
            ),
            "canonical_fit_teacher_identity_count": (
                self.checkpoint.canonical_fit_teacher_identity_count
            ),
            "heldout_target_read_count": 0,
            "heldout_replay_identity_count": 0,
            "base_checkpoint": self.base_checkpoint.record(),
            "checkpoint": self.checkpoint.record(),
        }
        if include_result_sha256:
            record["result_sha256"] = self.result_sha256
        return record


def _checkpoint_summary(
    workspace: FrozenPromptSelectionV4Workspace,
    *,
    phase: str,
    self_teacher_count: int,
    fit_teacher_count: int,
    base_checkpoint_sha256: str | None,
) -> CheckpointSummary:
    workspace.authenticate()
    return CheckpointSummary(
        phase=phase,
        output_cell_count=len(OUTPUT_CELLS),
        channel_router_cell_count=len(protocol_v4.RELAY_SCHEMAS),
        atom_cell_count=len(protocol_v4.OUTPUT_ATOMS),
        transition_cell_count=len(protocol_v4.OUTPUT_TRANSITIONS),
        branch_count=1 + len(OUTPUT_BRANCHES),
        array_shape=workspace._excitatory.shape,
        array_dtype=workspace._excitatory.dtype.str,
        array_bytes=workspace._excitatory.nbytes + workspace._inhibitory.nbytes,
        tick_cap=TICK_CAP,
        excitatory_nonzero_ticks=int(np.count_nonzero(workspace._excitatory)),
        inhibitory_nonzero_ticks=int(np.count_nonzero(workspace._inhibitory)),
        excitatory_max_tick=int(np.max(workspace._excitatory)),
        inhibitory_max_tick=int(np.max(workspace._inhibitory)),
        self_posed_teacher_lesson_count=self_teacher_count,
        canonical_fit_teacher_identity_count=fit_teacher_count,
        heldout_teacher_identity_count=0,
        base_checkpoint_sha256=base_checkpoint_sha256,
        checkpoint_sha256=workspace.checkpoint_sha256,
    )


def _empty_fit_preflight(status: str) -> FitPreflight:
    work = _WorkMeter().freeze()
    record = {
        "status": status,
        "fit_identity_count": 0,
        "prompt_trace_collision_count": 0,
        "route_feature_target_conflict_count": 0,
        "branch_feature_target_conflict_count": 0,
        "occurrence_overflow_count": 0,
        "binding_self_alias_count": 0,
        "projection_receipt_mismatch_count": 0,
        "maximum_occurrence_count": 0,
        "target_occurrence_count": 0,
        "edge_occurrence_count": 0,
        "heldout_target_read_count": 0,
        "heldout_replay_identity_count": 0,
        "work": work.record(),
    }
    return FitPreflight(
        status=status,
        fit_identity_count=0,
        prompt_trace_collision_count=0,
        route_feature_target_conflict_count=0,
        branch_feature_target_conflict_count=0,
        occurrence_overflow_count=0,
        binding_self_alias_count=0,
        projection_receipt_mismatch_count=0,
        maximum_occurrence_count=0,
        target_occurrence_count=0,
        edge_occurrence_count=0,
        heldout_target_read_count=0,
        heldout_replay_identity_count=0,
        work=work,
        sha256=_namespaced_sha256(_FIT_PREFLIGHT_NAMESPACE, record),
    )


def _build_surface_record(
    *,
    status: str,
    checkpoint: CheckpointSummary,
    fit_preflight: FitPreflight,
    fit_result: CanonicalFitReplayBatch | None,
) -> Mapping[str, object]:
    fit_attempted = checkpoint.canonical_fit_teacher_identity_count == 24
    record = {
        "status": status,
        "neural_input_sources": (
            "ordered-self-posed-relation-role-filler-assemblies",
            "ordered-public-train-physical-span-events",
            "ordered-typed-literal-cell-events",
            "ordered-name-free-signature-role-type-events",
            "recurrent-distributed-occurrence-state",
        ),
        "neural_output": (
            "physical-occurrence-channel-winner-then-channel-local-atom-transition-winners"
        ),
        "relay_schema_count": len(protocol_v4.RELAY_SCHEMAS),
        "atom_cell_count": len(protocol_v4.OUTPUT_ATOMS),
        "transition_cell_count": len(protocol_v4.OUTPUT_TRANSITIONS),
        "channel_router_cell_count": len(protocol_v4.RELAY_SCHEMAS),
        "treatment_learner_array_count": 2,
        "treatment_learner_array_bytes": checkpoint.array_bytes,
        "self_posed_learning_pass_count": 1,
        "canonical_fit_learning_pass_count": int(fit_attempted),
        "canonical_fit_teacher_identity_count": (
            checkpoint.canonical_fit_teacher_identity_count
        ),
        "canonical_fit_target_read_count": fit_preflight.fit_identity_count,
        "canonical_fit_replay_attempt_count": int(fit_result is not None),
        "canonical_fit_replay_identity_count": (
            fit_result.identity_count if fit_result is not None else 0
        ),
        "heldout_teacher_identity_count": 0,
        "heldout_target_read_count": HELDOUT_TARGET_READ_COUNT,
        "heldout_input_trace_count": HELDOUT_INPUT_TRACE_COUNT,
        "heldout_replay_identity_count": HELDOUT_REPLAY_IDENTITY_COUNT,
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
        "path_hash_neural_input_count": PATH_HASH_NEURAL_INPUT_COUNT,
        "absolute_depth_neural_input_count": ABSOLUTE_DEPTH_NEURAL_INPUT_COUNT,
        "whole_target_head_count": WHOLE_TARGET_HEAD_COUNT,
        "host_union_count": HOST_UNION_COUNT,
        "host_deduplication_count": HOST_DEDUPLICATION_COUNT,
        "host_filtering_count": HOST_FILTERING_COUNT,
        "host_reordering_count": HOST_REORDERING_COUNT,
        "host_ranking_count": HOST_RANKING_COUNT,
        "candidate_program_count": CANDIDATE_PROGRAM_COUNT,
        "proof_count": PROOF_COUNT,
        "occurrence_integer_handles_are_serialization_only": True,
        "ephemeral_parent_child_bindings_are_distributed_assemblies": True,
        "one_repeated_masked_wta_per_occurrence_branch_plus_stop": True,
        "same_target_may_recur_under_distinct_occurrence_bindings": True,
        "ordered_publication_preserved": True,
        "checkpoint_frozen_before_and_after_read_only_replay": True,
        "heldout_surface_opened": False,
    }
    return _freeze(record)


def _execute_once() -> tuple[
    PublicTrainPromptSelectionV4Result,
    FrozenPromptSelectionV4Workspace,
    FrozenPromptSelectionV4Workspace,
    Mapping[str, object],
    str,
]:
    lessons = protocol_v4.LESSONS
    untrained_workspace, teacher_off_training = _train_self_workspace(
        teachers=lessons,
        teacher_enabled=False,
    )
    teacher_off_replay = _replay_self_workspace(
        untrained_workspace,
        name="teacher-off",
        mode="normal",
    )
    base_workspace, self_training = _train_self_workspace(
        teachers=lessons,
        teacher_enabled=True,
    )
    self_result = _replay_self_workspace(
        base_workspace,
        name="self-posed-treatment",
        mode="normal",
    )
    cell_off = _replay_self_workspace(base_workspace, name="cell-off", mode="cell-off")
    reverse_order = _replay_self_workspace(
        base_workspace, name="reverse-order", mode="reverse-order"
    )
    role_swap = _replay_self_workspace(base_workspace, name="role-swap", mode="role-swap")
    filler_swap = _replay_self_workspace(
        base_workspace, name="filler-swap", mode="filler-swap"
    )
    deranged_workspace, deranged_training = _train_self_workspace(
        teachers=_deranged_self_teachers(),
        teacher_enabled=True,
        require_teacher_route=False,
    )
    deranged_replay = _replay_self_workspace(
        deranged_workspace,
        name="teacher-derangement",
        mode="normal",
    )
    controls = (
        _control_outcome(
            "teacher-off",
            untrained_workspace,
            teacher_off_replay,
            teacher_off_training,
        ),
        _control_outcome("cell-off", base_workspace, cell_off),
        _control_outcome("reverse-order", base_workspace, reverse_order),
        _control_outcome("role-swap", base_workspace, role_swap),
        _control_outcome("filler-swap", base_workspace, filler_swap),
        _control_outcome(
            "teacher-derangement",
            deranged_workspace,
            deranged_replay,
            deranged_training,
        ),
    )
    control_root = _namespaced_sha256(
        _CONTROL_PANEL_NAMESPACE, tuple(row.record() for row in controls)
    )
    control_pass = (
        controls[0].exact_lesson_count == 0
        and controls[1].exact_lesson_count == 0
        and all(row.exact_lesson_count < len(lessons) for row in controls[2:])
    )
    base_checkpoint_sha = base_workspace.checkpoint_sha256
    base_summary = _checkpoint_summary(
        base_workspace,
        phase="self-posed",
        self_teacher_count=110,
        fit_teacher_count=0,
        base_checkpoint_sha256=None,
    )
    fit_training = _WorkMeter().freeze()
    fit_result: CanonicalFitReplayBatch | None = None
    final_workspace = base_workspace
    if self_result.exact_lesson_count != len(lessons):
        status = STOP_SELF_POSED
        fit_preflight = _empty_fit_preflight(status)
    elif not control_pass:
        status = STOP_CONTROLS
        fit_preflight = _empty_fit_preflight(status)
    else:
        base_workspace.authenticate()
        fit_trace_meter = _WorkMeter()
        fit_inputs = _load_fit_physical_inputs()
        fit_traces = tuple(_encode_prompt_trace(row, fit_trace_meter) for row in fit_inputs)
        fit_items = _attach_fit_teachers(fit_traces)
        fit_preflight = _audit_fit_preflight(fit_items, fit_trace_meter.freeze())
        base_workspace.authenticate()
        if base_workspace.checkpoint_sha256 != base_checkpoint_sha:
            raise RuntimeError("base checkpoint changed during atomic FIT preflight")
        if fit_preflight.status != protocol_v4.PASS_PROTOCOL:
            status = STOP_FIT_PREFLIGHT
        else:
            final_workspace, fit_training = _train_canonical_fit_once(
                base_workspace, fit_items
            )
            fit_result = _replay_canonical_fit_once(final_workspace, fit_items)
            status = PASS if fit_result.exact_identity_count == 24 else STOP_FIT
    final_summary = _checkpoint_summary(
        final_workspace,
        phase=("canonical-fit" if fit_result is not None else "self-posed"),
        self_teacher_count=110,
        fit_teacher_count=(24 if fit_result is not None else 0),
        base_checkpoint_sha256=(base_checkpoint_sha if fit_result is not None else None),
    )
    surface_record = _build_surface_record(
        status=status,
        checkpoint=final_summary,
        fit_preflight=fit_preflight,
        fit_result=fit_result,
    )
    surface_sha = _namespaced_sha256(_SURFACE_NAMESPACE, surface_record)
    total_meter = _WorkMeter()
    total_meter.merge(self_training)
    total_meter.merge(self_result.work)
    for control in controls:
        total_meter.merge(control.work)
    total_meter.merge(fit_preflight.work)
    total_meter.merge(fit_training)
    if fit_result is not None:
        total_meter.merge(fit_result.work)
    total_work = total_meter.freeze()
    root_without_result = {
        "scope": SCOPE,
        "status": status,
        "dependency_sha256": DEPENDENCY_SHA256,
        "output_registry_sha256": OUTPUT_REGISTRY_SHA256,
        "branch_registry_sha256": BRANCH_REGISTRY_SHA256,
        "base_checkpoint_sha256": base_summary.checkpoint_sha256,
        "checkpoint_sha256": final_summary.checkpoint_sha256,
        "self_posed_result_sha256": self_result.sha256,
        "control_panel_sha256": control_root,
        "fit_preflight_sha256": fit_preflight.sha256,
        "fit_result_sha256": fit_result.sha256 if fit_result is not None else None,
        "surface_boundary_sha256": surface_sha,
        "self_training_work": self_training.record(),
        "fit_training_work": fit_training.record(),
        "total_work": total_work.record(),
        "physical_work_total": total_work.total,
        "self_posed_lesson_count": self_result.lesson_count,
        "self_posed_exact_lesson_count": self_result.exact_lesson_count,
        "canonical_fit_identity_count": fit_result.identity_count if fit_result else 0,
        "canonical_fit_exact_identity_count": (
            fit_result.exact_identity_count if fit_result else 0
        ),
        "canonical_fit_teacher_identity_count": final_summary.canonical_fit_teacher_identity_count,
        "heldout_target_read_count": 0,
        "heldout_replay_identity_count": 0,
        "base_checkpoint": base_summary.record(),
        "checkpoint": final_summary.record(),
    }
    result = PublicTrainPromptSelectionV4Result(
        scope=SCOPE,
        status=status,
        base_checkpoint=base_summary,
        checkpoint=final_summary,
        self_posed_result=self_result,
        controls=controls,
        fit_preflight=fit_preflight,
        fit_result=fit_result,
        self_training_work=self_training,
        fit_training_work=fit_training,
        total_work=total_work,
        physical_work_total=total_work.total,
        dependency_sha256=DEPENDENCY_SHA256,
        output_registry_sha256=OUTPUT_REGISTRY_SHA256,
        branch_registry_sha256=BRANCH_REGISTRY_SHA256,
        base_checkpoint_sha256=base_summary.checkpoint_sha256,
        checkpoint_sha256=final_summary.checkpoint_sha256,
        self_posed_result_sha256=self_result.sha256,
        control_panel_sha256=control_root,
        fit_preflight_sha256=fit_preflight.sha256,
        fit_result_sha256=fit_result.sha256 if fit_result is not None else None,
        surface_boundary_sha256=surface_sha,
        result_sha256=_namespaced_sha256(_RESULT_NAMESPACE, root_without_result),
    )
    return result, base_workspace, final_workspace, surface_record, control_root


(
    RESULT,
    BASE_WORKSPACE,
    WORKSPACE,
    SURFACE_BOUNDARY_RECORD,
    CONTROL_PANEL_SHA256,
) = _execute_once()
STATUS = RESULT.status
BASE_CHECKPOINT = RESULT.base_checkpoint
CHECKPOINT = RESULT.checkpoint
SELF_POSED_RESULT = RESULT.self_posed_result
CONTROLS = RESULT.controls
FIT_PREFLIGHT = RESULT.fit_preflight
FIT_RESULT = RESULT.fit_result
SURFACE_BOUNDARY_SHA256 = RESULT.surface_boundary_sha256

__all__ = [
    "BASE_CHECKPOINT",
    "BASE_WORKSPACE",
    "BRANCH_REGISTRY_SHA256",
    "CHECKPOINT",
    "CONTROLS",
    "CONTROL_PANEL_SHA256",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "FIT_PREFLIGHT",
    "FIT_RESULT",
    "OUTPUT_BRANCHES",
    "OUTPUT_CELLS",
    "OUTPUT_REGISTRY_SHA256",
    "PASS",
    "RESULT",
    "SCOPE",
    "SELF_POSED_RESULT",
    "STATUS",
    "STOP_CONTROLS",
    "STOP_FIT",
    "STOP_FIT_PREFLIGHT",
    "STOP_SELF_POSED",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "WORKSPACE",
    "CanonicalFitReplayBatch",
    "CheckpointSummary",
    "FitPreflight",
    "FrozenPromptSelectionV4Workspace",
    "PublicTrainPromptSelectionV4Result",
]
