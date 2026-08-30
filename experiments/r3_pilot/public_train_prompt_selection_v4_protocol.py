"""Fresh data-only prompt topology protocol after the terminal V3 FIT STOP.

V4 removes the absolute nesting/depth channel.  Ordered immediate edges retain
their multiplicity and bind ephemeral parent/child node occurrences; no path,
depth, task, family, program, or audit identity is learner-visible.  This file
contains no learner arrays or execution.
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
from typing import Final

from experiments.r3_pilot import public_train_prompt_selection_v3_protocol as v3

PROTOCOL_SCOPE = "PUBLIC_TRAIN_PROMPT_SELECTION_V4_PROTOCOL"
PASS_PROTOCOL = "PASS_PUBLIC_TRAIN_PROMPT_SELECTION_V4_PROTOCOL"
STOP_DEPENDENCY = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_DEPENDENCY"
STOP_VOCABULARY = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_VOCABULARY"
STOP_CURRICULUM = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_CURRICULUM"
STOP_TOPOLOGY = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_TOPOLOGY"
STOP_FIT_PROJECTION = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_FIT_PROJECTION"
STOP_LEARNER_EXECUTION = (
    "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V4_DATA_ONLY_NO_LEARNER_EXECUTION"
)

SELF_POSED_TRAIN = "SELF_POSED_TRAIN"
REMOVED_CHANNEL = "nesting-relation"
FIT = "FIT"

V3_PROTOCOL_COMMIT = "8248e3164fa6c1a52433787a2d6a6758c2d64fc2"
V3_PROTOCOL_SOURCE_PATH = (
    "experiments/r3_pilot/public_train_prompt_selection_v3_protocol.py"
)
V3_PROTOCOL_TEST_PATH = "tests/test_public_train_prompt_selection_v3_protocol.py"
V3_PROTOCOL_SOURCE_SHA256 = (
    "62bc379a9d0254a2976eb8af4f7dd8c2d283786ef8960301dedc52335f991bae"
)
V3_PROTOCOL_TEST_SHA256 = (
    "7ac9af168841972af31acf974ebb9445527abed9423eb8b3ded2c68c4c401aa5"
)
V3_PROTOCOL_SHA256 = "1eb2e3503790ebd721f69cc5d6f6dd8a32e0bb816d9b7f3dc8f1b4be9f15983f"

# Audit provenance only.  It is deliberately not an executable dependency.
V3_LEARNER_RESULT_SHA256_AUDIT_ONLY = (
    "e73446e71164a5435c89031f01afed0171b494f54772df57c34c9db86816ad84"
)
V3_LEARNER_SOURCE_COMMIT = "582de79f7b75aceea8f4ed64871310a15c96830a"
V3_LEARNER_SOURCE_PATH = "experiments/r3_pilot/public_train_prompt_selection_v3.py"
V3_LEARNER_TEST_PATH = "tests/test_public_train_prompt_selection_v3.py"
V3_LEARNER_SOURCE_SHA256 = (
    "7300b87a67807a6e2f095ba70177f3bc034bf54bf9f40b538b0bd53be6fdd681"
)
V3_LEARNER_TEST_SHA256 = (
    "59570a1c541cc4dbde9faf7f7d42fc5c8a5deee2062b1876039de921cf386e60"
)
V3_LEARNER_DEPENDENCY_BINDING_COUNT = 1

PHYSICAL_CELL_COUNT = v3.PHYSICAL_CELL_COUNT
ASSEMBLY_ACTIVE_CELL_COUNT = v3.ASSEMBLY_ACTIVE_CELL_COUNT
MIN_ATOM_SUPPORT = 2
MIN_TRANSITION_SUPPORT = 2

PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT = 90
CANONICAL_CAUSAL_IDENTITY_COUNT = 42
CANONICAL_FIT_IDENTITY_COUNT = 24
CANONICAL_HELDOUT_IDENTITY_COUNT = 18

LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
CANONICAL_FIT_TEACHER_UPDATE_COUNT = 0
HELDOUT_TARGET_READ_COUNT = 0
HELDOUT_REPLAY_IDENTITY_COUNT = 0
DESIRED_OUTPUT_READ_COUNT = 0
PROGRAM_ARTIFACT_READ_COUNT = 0
EXECUTABLE_SOURCE_READ_COUNT = 0
AST_ARTIFACT_READ_COUNT = 0
PROPERTY_ROW_READ_COUNT = 0
TASK_IDENTIFIER_READ_COUNT = 0
FAMILY_IDENTIFIER_READ_COUNT = 0
DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]
_DEPENDENCY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.dependency.v1\0"
_VOCABULARY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.vocabulary.v1\0"
_ATOM_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.atom.v1\0"
_TRANSITION_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.transition.v1\0"
_ASSEMBLY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.assembly.v1\0"
_LESSON_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.lesson.v1\0"
_PAIR_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.pair.v1\0"
_CURRICULUM_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.curriculum.v1\0"
_AUDIT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.audit.v1\0"
_TOPOLOGY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.topology.v1\0"
_FIT_TARGET_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.fit-target.v1\0"
_FIT_RECEIPT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.fit-receipt.v1\0"
_FIT_AUDIT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.fit-audit.v1\0"
_SURFACE_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.surface.v1\0"
_PROTOCOL_NAMESPACE = b"humanbrain.public-train-prompt-selection-v4.protocol.v1\0"


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


def _verify_dependencies() -> None:
    if (
        _file_sha256(V3_PROTOCOL_SOURCE_PATH) != V3_PROTOCOL_SOURCE_SHA256
        or _file_sha256(V3_PROTOCOL_TEST_PATH) != V3_PROTOCOL_TEST_SHA256
        or v3.PROTOCOL.protocol_sha256 != V3_PROTOCOL_SHA256
        or v3.PROTOCOL.status != v3.PASS_PROTOCOL
        or _file_sha256(V3_LEARNER_SOURCE_PATH) != V3_LEARNER_SOURCE_SHA256
        or _file_sha256(V3_LEARNER_TEST_PATH) != V3_LEARNER_TEST_SHA256
        or V3_LEARNER_DEPENDENCY_BINDING_COUNT != 1
    ):
        raise RuntimeError(STOP_DEPENDENCY)


_verify_dependencies()

DEPENDENCY_RECORD = _freeze(
    {
        "v3_protocol_commit": V3_PROTOCOL_COMMIT,
        "v3_protocol_source_path": V3_PROTOCOL_SOURCE_PATH,
        "v3_protocol_source_sha256": V3_PROTOCOL_SOURCE_SHA256,
        "v3_protocol_test_path": V3_PROTOCOL_TEST_PATH,
        "v3_protocol_test_sha256": V3_PROTOCOL_TEST_SHA256,
        "v3_protocol_sha256": V3_PROTOCOL_SHA256,
        "v3_learner_result_sha256_audit_only": V3_LEARNER_RESULT_SHA256_AUDIT_ONLY,
        "v3_learner_source_commit": V3_LEARNER_SOURCE_COMMIT,
        "v3_learner_source_path": V3_LEARNER_SOURCE_PATH,
        "v3_learner_source_sha256": V3_LEARNER_SOURCE_SHA256,
        "v3_learner_test_path": V3_LEARNER_TEST_PATH,
        "v3_learner_test_sha256": V3_LEARNER_TEST_SHA256,
        "v3_learner_dependency_binding_count": V3_LEARNER_DEPENDENCY_BINDING_COUNT,
        "v3_learner_source_and_test_bytes_authenticated": True,
        "v3_learner_imported": False,
        "v3_learner_source_executed": False,
        "v3_learner_result_is_dependency_authority": False,
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(_DEPENDENCY_NAMESPACE, DEPENDENCY_RECORD)


@dataclass(frozen=True, slots=True)
class RelaySlotSpec:
    role: str
    filler_kind: str
    fillers: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class RelayRelationSchema:
    channel: str
    slots: tuple[RelaySlotSpec, ...]

    def record(self) -> dict[str, object]:
        return {"channel": self.channel, "slots": tuple(row.record() for row in self.slots)}


RELAY_SCHEMAS = tuple(
    RelayRelationSchema(
        row.channel,
        tuple(RelaySlotSpec(slot.role, slot.filler_kind, slot.fillers) for slot in row.slots),
    )
    for row in v3.RELAY_SCHEMAS
    if row.channel != REMOVED_CHANNEL
)


@dataclass(frozen=True, slots=True)
class RelayOutputAtom:
    atom_index: int
    channel: str
    role: str
    filler_kind: str
    filler: str
    atom_id: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            "atom_index": self.atom_index,
            "channel": self.channel,
            "role": self.role,
            "filler_kind": self.filler_kind,
            "filler": self.filler,
            "atom_id": self.atom_id,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class RelayOutputTransition:
    transition_index: int
    channel: str
    from_role: str
    to_role: str
    transition_id: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            "transition_index": self.transition_index,
            "channel": self.channel,
            "from_role": self.from_role,
            "to_role": self.to_role,
            "transition_id": self.transition_id,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


def _build_vocabulary() -> tuple[tuple[RelayOutputAtom, ...], tuple[RelayOutputTransition, ...]]:
    atoms: list[RelayOutputAtom] = []
    transitions: list[RelayOutputTransition] = []
    for schema in RELAY_SCHEMAS:
        for slot in schema.slots:
            for filler in slot.fillers:
                atom_id = f"{schema.channel}|{slot.role}|{slot.filler_kind}|{filler}"
                record = {
                    "atom_index": len(atoms),
                    "channel": schema.channel,
                    "role": slot.role,
                    "filler_kind": slot.filler_kind,
                    "filler": filler,
                    "atom_id": atom_id,
                }
                atoms.append(
                    RelayOutputAtom(
                        **record,
                        sha256=_namespaced_sha256(_ATOM_NAMESPACE, record),
                    )
                )
        roles = tuple(slot.role for slot in schema.slots)
        adjacent = tuple(pairwise(roles))
        for from_role, to_role in adjacent + tuple(
            (right, left) for left, right in adjacent
        ):
            transition_id = f"{schema.channel}|{from_role}->{to_role}"
            record = {
                "transition_index": len(transitions),
                "channel": schema.channel,
                "from_role": from_role,
                "to_role": to_role,
                "transition_id": transition_id,
            }
            transitions.append(
                RelayOutputTransition(
                    **record,
                    sha256=_namespaced_sha256(_TRANSITION_NAMESPACE, record),
                )
            )
    if (
        len(RELAY_SCHEMAS) != 7
        or any(row.channel == REMOVED_CHANNEL for row in RELAY_SCHEMAS)
        or len(atoms) != 103
        or len(transitions) != 24
        or len({row.atom_id for row in atoms}) != len(atoms)
        or len({row.transition_id for row in transitions}) != len(transitions)
    ):
        raise RuntimeError(STOP_VOCABULARY)
    return tuple(atoms), tuple(transitions)


OUTPUT_ATOMS, OUTPUT_TRANSITIONS = _build_vocabulary()
VOCABULARY_RECORD = _freeze(
    {
        "schemas": tuple(row.record() for row in RELAY_SCHEMAS),
        "atoms": tuple(row.record() for row in OUTPUT_ATOMS),
        "transitions": tuple(row.record() for row in OUTPUT_TRANSITIONS),
        "removed_channel": REMOVED_CHANNEL,
        "absolute_depth_atom_count": 0,
    }
)
VOCABULARY_SHA256 = _namespaced_sha256(_VOCABULARY_NAMESPACE, VOCABULARY_RECORD)


def _assembly_cells(domain: str, token: str) -> tuple[int, ...]:
    cells: set[int] = set()
    counter = 0
    while len(cells) < ASSEMBLY_ACTIVE_CELL_COUNT:
        digest = sha256(
            _ASSEMBLY_NAMESPACE
            + domain.encode("ascii")
            + b"\0"
            + token.encode("ascii")
            + b"\0"
            + counter.to_bytes(4, "big")
        ).digest()
        for offset in range(0, len(digest), 2):
            cells.add(int.from_bytes(digest[offset : offset + 2], "big") % PHYSICAL_CELL_COUNT)
            if len(cells) == ASSEMBLY_ACTIVE_CELL_COUNT:
                break
        counter += 1
    return tuple(sorted(cells))


@dataclass(frozen=True, slots=True)
class PhysicalAssemblyEvent:
    relation_cells: tuple[int, ...]
    role_cells: tuple[int, ...]
    filler_cells: tuple[int, ...]

    def __post_init__(self) -> None:
        for cells in (self.relation_cells, self.role_cells, self.filler_cells):
            if (
                len(cells) != ASSEMBLY_ACTIVE_CELL_COUNT
                or len(set(cells)) != ASSEMBLY_ACTIVE_CELL_COUNT
                or tuple(sorted(cells)) != cells
                or any(
                    type(cell) is not int or not 0 <= cell < PHYSICAL_CELL_COUNT
                    for cell in cells
                )
            ):
                raise ValueError("V4 physical assembly changed")

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class PhysicalCausalAssemblyKey:
    ordered_events: tuple[PhysicalAssemblyEvent, ...]

    def record(self) -> dict[str, object]:
        return {"ordered_events": tuple(row.record() for row in self.ordered_events)}

    def cell_multiset(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                cell
                for event in self.ordered_events
                for cells in (event.relation_cells, event.role_cells, event.filler_cells)
                for cell in cells
            )
        )


@dataclass(frozen=True, slots=True)
class SelfPosedRelayLesson:
    lesson_index: int
    split: str
    causal_input: PhysicalCausalAssemblyKey
    target_atom_ids: tuple[str, ...]
    target_transition_ids: tuple[str, ...]
    audit_channel: str
    audit_order: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {
            "causal_input": self.causal_input.record(),
            "teacher_target": {
                "atom_ids": self.target_atom_ids,
                "transition_ids": self.target_transition_ids,
                "exact_compositional_target": True,
            },
            "audit_only": {
                "lesson_index": self.lesson_index,
                "split": self.split,
                "channel": self.audit_channel,
                "order": self.audit_order,
            },
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class MatchedCounterfactualPair:
    pair_index: int
    kind: str
    left_lesson_index: int
    right_lesson_index: int
    same_physical_cell_multiset: bool
    atom_targets_differ: bool
    transition_targets_differ: bool
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


_ATOM_BY_ID = {row.atom_id: row for row in OUTPUT_ATOMS}


def _ordered_atom_ids(lesson: object) -> tuple[str, ...]:
    atoms_by_role = {
        atom_id.split("|", 3)[1]: atom_id for atom_id in lesson.target_atom_ids
    }
    roles: list[str] = []
    for transition_index, transition_id in enumerate(lesson.target_transition_ids):
        arrow = transition_id.split("|", 1)[1]
        left, right = arrow.split("->", 1)
        if transition_index == 0:
            roles.append(left)
        roles.append(right)
    if set(roles) != set(atoms_by_role) or len(roles) != len(atoms_by_role):
        raise RuntimeError(STOP_CURRICULUM)
    return tuple(atoms_by_role[role] for role in roles)


def _event_for_atom(atom_id: str) -> PhysicalAssemblyEvent:
    atom = _ATOM_BY_ID[atom_id]
    return PhysicalAssemblyEvent(
        relation_cells=_assembly_cells("relation", atom.channel),
        role_cells=_assembly_cells("role", f"{atom.channel}|{atom.role}"),
        filler_cells=_assembly_cells("filler", f"{atom.filler_kind}|{atom.filler}"),
    )


def _build_curriculum() -> tuple[
    tuple[SelfPosedRelayLesson, ...], tuple[MatchedCounterfactualPair, ...], int
]:
    lessons: list[SelfPosedRelayLesson] = []
    old_to_new: dict[int, int] = {}
    physical_key_reuse_count = 0
    for old in v3.PROTOCOL.lessons:
        if old.audit_channel == REMOVED_CHANNEL:
            continue
        ordered_atom_ids = _ordered_atom_ids(old)
        causal_input = PhysicalCausalAssemblyKey(
            tuple(_event_for_atom(atom_id) for atom_id in ordered_atom_ids)
        )
        physical_key_reuse_count += causal_input.record() == old.causal_input_record()
        record = {
            "causal_input": causal_input.record(),
            "teacher_target": {
                "atom_ids": old.target_atom_ids,
                "transition_ids": old.target_transition_ids,
                "exact_compositional_target": True,
            },
            "audit_only": {
                "lesson_index": len(lessons),
                "split": SELF_POSED_TRAIN,
                "channel": old.audit_channel,
                "order": old.audit_order,
            },
        }
        lesson = SelfPosedRelayLesson(
            lesson_index=len(lessons),
            split=SELF_POSED_TRAIN,
            causal_input=causal_input,
            target_atom_ids=old.target_atom_ids,
            target_transition_ids=old.target_transition_ids,
            audit_channel=old.audit_channel,
            audit_order=old.audit_order,
            sha256=_namespaced_sha256(_LESSON_NAMESPACE, record),
        )
        old_to_new[old.lesson_index] = lesson.lesson_index
        lessons.append(lesson)

    pairs: list[MatchedCounterfactualPair] = []
    for old in v3.PROTOCOL.counterfactual_pairs:
        if old.left_lesson_index not in old_to_new or old.right_lesson_index not in old_to_new:
            continue
        left_index = old_to_new[old.left_lesson_index]
        right_index = old_to_new[old.right_lesson_index]
        left = lessons[left_index]
        right = lessons[right_index]
        record = {
            "pair_index": len(pairs),
            "kind": old.kind,
            "left_lesson_index": left_index,
            "right_lesson_index": right_index,
            "same_physical_cell_multiset": (
                left.causal_input.cell_multiset() == right.causal_input.cell_multiset()
            ),
            "atom_targets_differ": left.target_atom_ids != right.target_atom_ids,
            "transition_targets_differ": (
                left.target_transition_ids != right.target_transition_ids
            ),
        }
        pairs.append(
            MatchedCounterfactualPair(
                **record,
                sha256=_namespaced_sha256(_PAIR_NAMESPACE, record),
            )
        )
    return tuple(lessons), tuple(pairs), physical_key_reuse_count


LESSONS, COUNTERFACTUAL_PAIRS, V3_PHYSICAL_KEY_REUSE_COUNT = _build_curriculum()


@dataclass(frozen=True, slots=True)
class PrelearningAudit:
    status: str
    lesson_count: int
    pair_count: int
    unique_causal_key_count: int
    causal_key_target_collision_count: int
    output_atom_count: int
    output_transition_count: int
    minimum_atom_support: int
    minimum_transition_support: int
    unsupported_atom_count: int
    unsupported_transition_count: int
    order_pair_count: int
    role_pair_count: int
    counterfactual_failure_count: int
    v3_physical_key_reuse_count: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        if not include_sha256:
            record.pop("sha256")
        return record


def _audit_curriculum() -> PrelearningAudit:
    targets_by_key: dict[PhysicalCausalAssemblyKey, set[tuple[object, ...]]] = defaultdict(set)
    for row in LESSONS:
        targets_by_key[row.causal_input].add((row.target_atom_ids, row.target_transition_ids))
    collision_count = sum(len(targets) != 1 for targets in targets_by_key.values())
    atom_support = Counter(atom for row in LESSONS for atom in row.target_atom_ids)
    transition_support = Counter(
        target for row in LESSONS for target in row.target_transition_ids
    )
    unsupported_atoms = sum(
        atom_support[row.atom_id] < MIN_ATOM_SUPPORT for row in OUTPUT_ATOMS
    )
    unsupported_transitions = sum(
        transition_support[row.transition_id] < MIN_TRANSITION_SUPPORT
        for row in OUTPUT_TRANSITIONS
    )
    order_pairs = tuple(row for row in COUNTERFACTUAL_PAIRS if row.kind == "event-order-reversal")
    role_pairs = tuple(row for row in COUNTERFACTUAL_PAIRS if row.kind == "role-filler-swap")
    failures = sum(
        not row.same_physical_cell_multiset
        or (
            row.kind == "event-order-reversal"
            and (row.atom_targets_differ or not row.transition_targets_differ)
        )
        or (
            row.kind == "role-filler-swap"
            and (not row.atom_targets_differ or row.transition_targets_differ)
        )
        for row in COUNTERFACTUAL_PAIRS
    )
    record = {
        "status": PASS_PROTOCOL,
        "lesson_count": len(LESSONS),
        "pair_count": len(COUNTERFACTUAL_PAIRS),
        "unique_causal_key_count": len(targets_by_key),
        "causal_key_target_collision_count": collision_count,
        "output_atom_count": len(OUTPUT_ATOMS),
        "output_transition_count": len(OUTPUT_TRANSITIONS),
        "minimum_atom_support": min(atom_support.values(), default=0),
        "minimum_transition_support": min(transition_support.values(), default=0),
        "unsupported_atom_count": unsupported_atoms,
        "unsupported_transition_count": unsupported_transitions,
        "order_pair_count": len(order_pairs),
        "role_pair_count": len(role_pairs),
        "counterfactual_failure_count": failures,
        "v3_physical_key_reuse_count": V3_PHYSICAL_KEY_REUSE_COUNT,
    }
    if record != {
        "status": PASS_PROTOCOL,
        "lesson_count": 110,
        "pair_count": 57,
        "unique_causal_key_count": 110,
        "causal_key_target_collision_count": 0,
        "output_atom_count": 103,
        "output_transition_count": 24,
        "minimum_atom_support": 2,
        "minimum_transition_support": 2,
        "unsupported_atom_count": 0,
        "unsupported_transition_count": 0,
        "order_pair_count": 55,
        "role_pair_count": 2,
        "counterfactual_failure_count": 0,
        "v3_physical_key_reuse_count": 0,
    }:
        raise RuntimeError(STOP_CURRICULUM)
    return PrelearningAudit(
        **record,
        sha256=_namespaced_sha256(_AUDIT_NAMESPACE, record),
    )


PRELEARNING_AUDIT = _audit_curriculum()
CURRICULUM_SHA256 = _namespaced_sha256(
    _CURRICULUM_NAMESPACE,
    {
        "lessons": tuple(row.record() for row in LESSONS),
        "pairs": tuple(row.record() for row in COUNTERFACTUAL_PAIRS),
        "audit_sha256": PRELEARNING_AUDIT.sha256,
    },
)


@dataclass(frozen=True, slots=True)
class EdgeOccurrenceTarget:
    sequence_index: int
    parent_occurrence: int
    child_occurrence: int
    child_ordinal: int
    parent_production: str
    child_production: str

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def edge_type(self) -> tuple[str, str, int]:
        return (self.parent_production, self.child_production, self.child_ordinal)


def _edge(
    sequence_index: int,
    parent_occurrence: int,
    child_occurrence: int,
    child_ordinal: int,
    parent_production: str,
    child_production: str,
) -> EdgeOccurrenceTarget:
    return EdgeOccurrenceTarget(
        sequence_index,
        parent_occurrence,
        child_occurrence,
        child_ordinal,
        parent_production,
        child_production,
    )


RECURSIVE_ADD_SHORT = (
    _edge(0, 0, 1, 0, "add", "add"),
    _edge(1, 0, 4, 1, "add", "literal"),
    _edge(2, 1, 2, 0, "add", "parameter"),
    _edge(3, 1, 3, 1, "add", "literal"),
)
RECURSIVE_ADD_LONG = (
    _edge(0, 0, 1, 0, "add", "add"),
    _edge(1, 0, 6, 1, "add", "literal"),
    _edge(2, 1, 2, 0, "add", "add"),
    _edge(3, 1, 5, 1, "add", "literal"),
    _edge(4, 2, 3, 0, "add", "parameter"),
    _edge(5, 2, 4, 1, "add", "literal"),
)


def _edge_type_set_sha256(edges: tuple[EdgeOccurrenceTarget, ...]) -> str:
    return _namespaced_sha256(
        _TOPOLOGY_NAMESPACE + b"edge-type-set\0",
        tuple(sorted(set(row.edge_type() for row in edges))),
    )


def _occurrence_graph_sha256(edges: tuple[EdgeOccurrenceTarget, ...]) -> str:
    return _namespaced_sha256(
        _TOPOLOGY_NAMESPACE + b"occurrence-graph\0",
        tuple(row.record() for row in edges),
    )


RECURSIVE_ADD_SHORT_TYPE_SET_SHA256 = _edge_type_set_sha256(RECURSIVE_ADD_SHORT)
RECURSIVE_ADD_LONG_TYPE_SET_SHA256 = _edge_type_set_sha256(RECURSIVE_ADD_LONG)
RECURSIVE_ADD_SHORT_TARGET_SHA256 = _occurrence_graph_sha256(RECURSIVE_ADD_SHORT)
RECURSIVE_ADD_LONG_TARGET_SHA256 = _occurrence_graph_sha256(RECURSIVE_ADD_LONG)
if (
    RECURSIVE_ADD_SHORT_TYPE_SET_SHA256 != RECURSIVE_ADD_LONG_TYPE_SET_SHA256
    or RECURSIVE_ADD_SHORT_TARGET_SHA256 == RECURSIVE_ADD_LONG_TARGET_SHA256
):
    raise RuntimeError(STOP_TOPOLOGY)

TOPOLOGY_CONTRACT = _freeze(
    {
        "rooted_ordered_immediate_edges_only": True,
        "absolute_depth_channel_count": 0,
        "ephemeral_parent_child_occurrence_bindings": "distributed-physical-assemblies",
        "occurrence_integer_handles_are_serialization_only": True,
        "one_repeated_masked_wta_per_edge_occurrence_then_stop": True,
        "multiplicity_preserved": True,
        "same_edge_type_may_win_more_than_once_for_distinct_occurrences": True,
        "host_union_forbidden": True,
        "host_deduplication_forbidden": True,
        "host_filtering_forbidden": True,
        "host_reordering_forbidden": True,
        "path_hash_neural_evidence_forbidden": True,
        "absolute_depth_neural_evidence_forbidden": True,
        "audit_hash_neural_evidence_forbidden": True,
        "recursive_add_poison": {
            "short_edge_count": len(RECURSIVE_ADD_SHORT),
            "long_edge_count": len(RECURSIVE_ADD_LONG),
            "edge_type_set_sha256_equal": True,
            "short_target_sha256": RECURSIVE_ADD_SHORT_TARGET_SHA256,
            "long_target_sha256": RECURSIVE_ADD_LONG_TARGET_SHA256,
            "targets_must_differ": True,
        },
    }
)
TOPOLOGY_SHA256 = _namespaced_sha256(_TOPOLOGY_NAMESPACE, TOPOLOGY_CONTRACT)


def _production_for_slot(slot: object, root_production: str) -> str:
    value = str(slot)
    if value == "root":
        return root_production
    rules = (
        (
            (
                "parameter-arg0",
                "collection-arg0",
                "string-arg0",
                "bound-item0",
                "bound-char0",
            ),
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
    for fragment, role in (
        ("signature-parameter", "input-parameter"),
        ("bound-item", "bound-item"),
        ("bound-char", "bound-char"),
        ("predicate-binder", "predicate"),
        ("compiler-ascii-class", "class"),
        ("nested-", "nested-result"),
        ("literal", "literal"),
        ("constant", "literal"),
        ("value-binder", "collection"),
    ):
        if fragment in text:
            return role
    return f"UNSUPPORTED:{text}"


def _parameter_slot(value: object) -> str:
    return {
        "parameter-arg0": "input-parameter",
        "collection-arg0": "collection-parameter",
        "string-arg0": "text-parameter",
    }.get(str(value), f"UNSUPPORTED:{value}")


def _literal_slot(value: object) -> str:
    text = str(value)
    for fragment, slot in (
        ("lower", "lower-bound"),
        ("implicit-zero", "lower-bound"),
        ("upper", "upper-bound"),
        ("offset", "offset"),
        ("threshold", "threshold"),
        ("ascii-class", "ascii-class"),
    ):
        if fragment in text:
            return slot
    return f"UNSUPPORTED:{text}"


def _project_label(label: object, root_production: str) -> tuple[str, tuple[str, ...]]:
    channel = str(label.channel)
    fields = label.fields
    if channel == REMOVED_CHANNEL:
        raise RuntimeError(STOP_FIT_PROJECTION)
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
            "copy-role": (
                "emit-ascii-class"
                if "ascii-class" in str(fields["typed_output_copy_role"])
                else "emit-int"
            ),
        }
    elif channel == "terminal-root-eclass-agreement":
        bindings = {
            "root-authority": "selected-root",
            "eclass-authority": "retained-eclass",
            "agreement-state": "exact-object-identity",
        }
    else:
        raise RuntimeError(STOP_FIT_PROJECTION)
    schema = next(row for row in RELAY_SCHEMAS if row.channel == channel)
    atom_ids = tuple(
        f"{channel}|{slot.role}|{slot.filler_kind}|{bindings[slot.role]}"
        for slot in schema.slots
    )
    transition_ids = tuple(
        f"{channel}|{left.role}->{right.role}" for left, right in pairwise(schema.slots)
    )
    targets = tuple(sorted((*atom_ids, *transition_ids)))
    atom_registry = {row.atom_id for row in OUTPUT_ATOMS}
    transition_registry = {row.transition_id for row in OUTPUT_TRANSITIONS}
    if any(
        target not in (atom_registry if target in atom_ids else transition_registry)
        for target in targets
    ):
        raise RuntimeError(STOP_FIT_PROJECTION)
    return channel, targets


@dataclass(frozen=True, slots=True)
class CanonicalFitProjectionReceipt:
    causal_key_sha256: str
    original_target_set_sha256: str
    reduced_target_sha256: str
    retained_label_count: int
    removed_label_count: int
    edge_occurrence_count: int
    repeated_edge_type_count: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        if not include_sha256:
            record.pop("sha256")
        return record


@dataclass(frozen=True, slots=True)
class CanonicalFitProjectionAudit:
    status: str
    fit_identity_count: int
    projectable_identity_count: int
    unsupported_identity_count: int
    original_target_class_count: int
    reduced_target_class_count: int
    target_class_merge_count: int
    target_class_split_count: int
    distinct_target_collision_count: int
    causal_key_collision_count: int
    retained_label_count: int
    removed_label_count: int
    edge_occurrence_count: int
    repeated_edge_identity_count: int
    class_sizes: tuple[int, ...]
    heldout_target_read_count: int
    heldout_replay_identity_count: int
    receipts_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        if not include_sha256:
            record.pop("sha256")
        return record


def _fit_target_record(annotation: object) -> tuple[dict[str, object], int, int, int]:
    root_label = next(
        row for row in annotation.relay_labels if row.channel == "root-production-family"
    )
    root_production = str(root_label.fields["root_production"])
    semantic_occurrences: list[dict[str, object]] = []
    edges: list[EdgeOccurrenceTarget] = []
    node_occurrences: dict[str, int] = {}
    removed_count = 0
    retained_index = 0
    for label in annotation.relay_labels:
        if label.channel == REMOVED_CHANNEL:
            removed_count += 1
            continue
        channel, targets = _project_label(label, root_production)
        semantic_occurrences.append(
            {
                "sequence_index": retained_index,
                "channel": channel,
                "targets": targets,
            }
        )
        if channel == "ordered-parent-child-edge":
            parent_slot = str(label.fields["parent_slot"])
            child_slot = str(label.fields["child_slot"])
            for slot in (parent_slot, child_slot):
                if slot not in node_occurrences:
                    node_occurrences[slot] = len(node_occurrences)
            edges.append(
                EdgeOccurrenceTarget(
                    sequence_index=retained_index,
                    parent_occurrence=node_occurrences[parent_slot],
                    child_occurrence=node_occurrences[child_slot],
                    child_ordinal=int(label.fields["child_ordinal"]),
                    parent_production=_production_for_slot(parent_slot, root_production),
                    child_production=_production_for_slot(child_slot, root_production),
                )
            )
        retained_index += 1
    repeated_edge_types = sum(
        count > 1 for count in Counter(row.edge_type() for row in edges).values()
    )
    target_record = {
        "semantic_occurrences": tuple(semantic_occurrences),
        "immediate_edge_occurrences": tuple(row.record() for row in edges),
        "absolute_depth_target_count": 0,
        "host_reordered": False,
        "host_deduplicated": False,
    }
    return target_record, removed_count, len(edges), repeated_edge_types


def _build_fit_projection() -> tuple[
    tuple[CanonicalFitProjectionReceipt, ...], CanonicalFitProjectionAudit
]:
    from experiments.r3_pilot import (
        public_train_prompt_selection_freeze_v1_protocol as selection_v1,
    )
    from experiments.r3_pilot import (
        public_train_structural_annotation_capsule_v1 as annotations_v1,
    )

    if (
        selection_v1.PROTOCOL.protocol_sha256 != v3.SELECTION_V1_PROTOCOL_SHA256
        or annotations_v1.CAPSULE.capsule_sha256 != v3.CAPSULE_SHA256
    ):
        raise RuntimeError(STOP_DEPENDENCY)
    fit_keys = tuple(
        row.causal_key_sha256
        for row in selection_v1.PROTOCOL.canonical_inputs
        if row.split == FIT
    )
    if len(fit_keys) != CANONICAL_FIT_IDENTITY_COUNT or len(set(fit_keys)) != len(fit_keys):
        raise RuntimeError(STOP_FIT_PROJECTION)

    receipts: list[CanonicalFitProjectionReceipt] = []
    old_to_new: dict[str, set[str]] = defaultdict(set)
    new_to_old: dict[str, set[str]] = defaultdict(set)
    retained_total = 0
    removed_total = 0
    edge_total = 0
    repeated_identity_count = 0
    for causal_key in fit_keys:
        annotation = next(
            row
            for row in annotations_v1.CAPSULE.annotations
            if row.causal_key_sha256 == causal_key
        )
        target_record, removed_count, edge_count, repeated_count = _fit_target_record(annotation)
        reduced_sha = _namespaced_sha256(_FIT_TARGET_NAMESPACE, target_record)
        receipt_record = {
            "causal_key_sha256": causal_key,
            "original_target_set_sha256": annotation.target_set_sha256,
            "reduced_target_sha256": reduced_sha,
            "retained_label_count": len(annotation.relay_labels) - removed_count,
            "removed_label_count": removed_count,
            "edge_occurrence_count": edge_count,
            "repeated_edge_type_count": repeated_count,
        }
        receipts.append(
            CanonicalFitProjectionReceipt(
                **receipt_record,
                sha256=_namespaced_sha256(_FIT_RECEIPT_NAMESPACE, receipt_record),
            )
        )
        old_to_new[annotation.target_set_sha256].add(reduced_sha)
        new_to_old[reduced_sha].add(annotation.target_set_sha256)
        retained_total += receipt_record["retained_label_count"]
        removed_total += removed_count
        edge_total += edge_count
        repeated_identity_count += bool(repeated_count)

    merges = sum(len(targets) > 1 for targets in new_to_old.values())
    splits = sum(len(targets) > 1 for targets in old_to_new.values())
    class_sizes = tuple(
        sorted(
            sum(receipt.reduced_target_sha256 == target for receipt in receipts)
            for target in new_to_old
        )
    )
    receipts_sha = _namespaced_sha256(
        _FIT_RECEIPT_NAMESPACE + b"all\0",
        tuple(row.record() for row in receipts),
    )
    record = {
        "status": PASS_PROTOCOL,
        "fit_identity_count": len(receipts),
        "projectable_identity_count": len(receipts),
        "unsupported_identity_count": 0,
        "original_target_class_count": len(old_to_new),
        "reduced_target_class_count": len(new_to_old),
        "target_class_merge_count": merges,
        "target_class_split_count": splits,
        "distinct_target_collision_count": merges,
        "causal_key_collision_count": len(receipts)
        - len({row.causal_key_sha256 for row in receipts}),
        "retained_label_count": retained_total,
        "removed_label_count": removed_total,
        "edge_occurrence_count": edge_total,
        "repeated_edge_identity_count": repeated_identity_count,
        "class_sizes": class_sizes,
        "heldout_target_read_count": HELDOUT_TARGET_READ_COUNT,
        "heldout_replay_identity_count": HELDOUT_REPLAY_IDENTITY_COUNT,
        "receipts_sha256": receipts_sha,
    }
    expected = {
        "status": PASS_PROTOCOL,
        "fit_identity_count": 24,
        "projectable_identity_count": 24,
        "unsupported_identity_count": 0,
        "original_target_class_count": 6,
        "reduced_target_class_count": 6,
        "target_class_merge_count": 0,
        "target_class_split_count": 0,
        "distinct_target_collision_count": 0,
        "causal_key_collision_count": 0,
        "retained_label_count": 357,
        "removed_label_count": 44,
        "edge_occurrence_count": 117,
        "repeated_edge_identity_count": 0,
        "class_sizes": (1, 2, 4, 5, 5, 7),
        "heldout_target_read_count": 0,
        "heldout_replay_identity_count": 0,
        "receipts_sha256": receipts_sha,
    }
    if record != expected:
        raise RuntimeError(STOP_FIT_PROJECTION)
    return tuple(receipts), CanonicalFitProjectionAudit(
        **record,
        sha256=_namespaced_sha256(_FIT_AUDIT_NAMESPACE, record),
    )


FIT_PROJECTION_RECEIPTS, FIT_PROJECTION_AUDIT = _build_fit_projection()

LEARNER_GATE_RECORD = _freeze(
    {
        "fresh_v4_learner_required": True,
        "learner_execution_authorized": False,
        "atomic_stop": STOP_LEARNER_EXECUTION,
        "blocking_dependency": (
            "separately-banked-v4-protocol-and-separately-named-v4-learner"
        ),
        "v3_learner_source_and_test_bytes_authenticated": True,
        "v3_learner_result_root_is_audit_provenance_only": True,
        "v3_checkpoint_or_weight_reuse_allowed": False,
        "required_teacher_schedule": "110-self-posed-then-24-fit-only",
        "required_edge_output": (
            "repeated-masked-wta-per-occurrence-plus-stop-with-ephemeral-bindings"
        ),
        "heldout_target_or_replay_allowed": False,
        "host_union_dedup_filter_or_reorder_allowed": False,
        "absolute_depth_or_path_hash_allowed": False,
    }
)

SURFACE_BOUNDARY_RECORD = _freeze(
    {
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "canonical_fit_teacher_update_count": CANONICAL_FIT_TEACHER_UPDATE_COUNT,
        "heldout_target_read_count": HELDOUT_TARGET_READ_COUNT,
        "heldout_replay_identity_count": HELDOUT_REPLAY_IDENTITY_COUNT,
        "desired_output_read_count": DESIRED_OUTPUT_READ_COUNT,
        "program_artifact_read_count": PROGRAM_ARTIFACT_READ_COUNT,
        "executable_source_read_count": EXECUTABLE_SOURCE_READ_COUNT,
        "ast_artifact_read_count": AST_ARTIFACT_READ_COUNT,
        "property_row_read_count": PROPERTY_ROW_READ_COUNT,
        "task_identifier_read_count": TASK_IDENTIFIER_READ_COUNT,
        "family_identifier_read_count": FAMILY_IDENTIFIER_READ_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "data_only": True,
        "heldout_surface_opened": False,
    }
)
SURFACE_BOUNDARY_SHA256 = _namespaced_sha256(
    _SURFACE_NAMESPACE, SURFACE_BOUNDARY_RECORD
)


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainPromptSelectionV4Protocol:
    scope: str
    status: str
    dependency_sha256: str
    vocabulary_sha256: str
    curriculum_sha256: str
    prelearning_audit: PrelearningAudit
    topology_sha256: str
    fit_projection_receipts: tuple[CanonicalFitProjectionReceipt, ...]
    fit_projection_audit: CanonicalFitProjectionAudit
    learner_gate: object
    surface_boundary: object
    surface_boundary_sha256: str
    protocol_sha256: str

    def root_record(self, *, include_protocol_sha256: bool = True) -> dict[str, object]:
        record = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "vocabulary_sha256": self.vocabulary_sha256,
            "curriculum_sha256": self.curriculum_sha256,
            "prelearning_audit": self.prelearning_audit.record(),
            "topology_sha256": self.topology_sha256,
            "fit_projection_receipts": tuple(
                row.record() for row in self.fit_projection_receipts
            ),
            "fit_projection_audit": self.fit_projection_audit.record(),
            "learner_gate": self.learner_gate,
            "surface_boundary": self.surface_boundary,
            "surface_boundary_sha256": self.surface_boundary_sha256,
        }
        if include_protocol_sha256:
            record["protocol_sha256"] = self.protocol_sha256
        return record


def _build_protocol() -> FrozenPublicTrainPromptSelectionV4Protocol:
    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": PASS_PROTOCOL,
        "dependency_sha256": DEPENDENCY_SHA256,
        "vocabulary_sha256": VOCABULARY_SHA256,
        "curriculum_sha256": CURRICULUM_SHA256,
        "prelearning_audit": PRELEARNING_AUDIT.record(),
        "topology_sha256": TOPOLOGY_SHA256,
        "fit_projection_receipts": tuple(row.record() for row in FIT_PROJECTION_RECEIPTS),
        "fit_projection_audit": FIT_PROJECTION_AUDIT.record(),
        "learner_gate": LEARNER_GATE_RECORD,
        "surface_boundary": SURFACE_BOUNDARY_RECORD,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
    }
    return FrozenPublicTrainPromptSelectionV4Protocol(
        scope=PROTOCOL_SCOPE,
        status=PASS_PROTOCOL,
        dependency_sha256=DEPENDENCY_SHA256,
        vocabulary_sha256=VOCABULARY_SHA256,
        curriculum_sha256=CURRICULUM_SHA256,
        prelearning_audit=PRELEARNING_AUDIT,
        topology_sha256=TOPOLOGY_SHA256,
        fit_projection_receipts=FIT_PROJECTION_RECEIPTS,
        fit_projection_audit=FIT_PROJECTION_AUDIT,
        learner_gate=LEARNER_GATE_RECORD,
        surface_boundary=SURFACE_BOUNDARY_RECORD,
        surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
        protocol_sha256=_namespaced_sha256(_PROTOCOL_NAMESPACE, root_without_protocol),
    )


PROTOCOL: Final = _build_protocol()

__all__ = [
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "CANONICAL_CAUSAL_IDENTITY_COUNT",
    "CANONICAL_FIT_IDENTITY_COUNT",
    "CANONICAL_HELDOUT_IDENTITY_COUNT",
    "COUNTERFACTUAL_PAIRS",
    "CURRICULUM_SHA256",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "FIT_PROJECTION_AUDIT",
    "FIT_PROJECTION_RECEIPTS",
    "LEARNER_ARRAY_COUNT",
    "LEARNER_GATE_RECORD",
    "LEARNING_RUN_COUNT",
    "LESSONS",
    "MIN_ATOM_SUPPORT",
    "MIN_TRANSITION_SUPPORT",
    "OUTPUT_ATOMS",
    "OUTPUT_TRANSITIONS",
    "PASS_PROTOCOL",
    "PHYSICAL_CELL_COUNT",
    "PRELEARNING_AUDIT",
    "PROGRAM_ARTIFACT_READ_COUNT",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT",
    "RECURSIVE_ADD_LONG",
    "RECURSIVE_ADD_LONG_TARGET_SHA256",
    "RECURSIVE_ADD_LONG_TYPE_SET_SHA256",
    "RECURSIVE_ADD_SHORT",
    "RECURSIVE_ADD_SHORT_TARGET_SHA256",
    "RECURSIVE_ADD_SHORT_TYPE_SET_SHA256",
    "RELAY_SCHEMAS",
    "REMOVED_CHANNEL",
    "STOP_LEARNER_EXECUTION",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "TOPOLOGY_CONTRACT",
    "TOPOLOGY_SHA256",
    "V3_LEARNER_RESULT_SHA256_AUDIT_ONLY",
    "V3_LEARNER_SOURCE_COMMIT",
    "V3_LEARNER_SOURCE_PATH",
    "V3_LEARNER_SOURCE_SHA256",
    "V3_LEARNER_TEST_PATH",
    "V3_LEARNER_TEST_SHA256",
    "V3_PROTOCOL_COMMIT",
    "V3_PROTOCOL_SHA256",
    "V3_PROTOCOL_SOURCE_SHA256",
    "V3_PROTOCOL_TEST_SHA256",
    "VOCABULARY_RECORD",
    "VOCABULARY_SHA256",
    "CanonicalFitProjectionAudit",
    "CanonicalFitProjectionReceipt",
    "EdgeOccurrenceTarget",
    "FrozenPublicTrainPromptSelectionV4Protocol",
    "MatchedCounterfactualPair",
    "PhysicalAssemblyEvent",
    "PhysicalCausalAssemblyKey",
    "PrelearningAudit",
    "RelayOutputAtom",
    "RelayOutputTransition",
    "RelayRelationSchema",
    "RelaySlotSpec",
    "SelfPosedRelayLesson",
]
