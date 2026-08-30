"""Data-only compositional protocol after the terminal prompt-selection V2 STOP.

This module freezes a compiler-grammar-owned vocabulary of small relay atoms and
ordered atom transitions.  A fresh self-posed TRAIN cohort binds those outputs
to sparse physical relation/role/filler assemblies.  Canonical prompt labels are
not imported, read, or used to construct the vocabulary or curriculum.

There is deliberately no learner, array, electrical execution, candidate, or
proof interface here.  A separately named V3 learner is licensed only when the
prelearning total-function, support, and matched-counterfactual audits pass.
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

PROTOCOL_SCOPE = "PUBLIC_TRAIN_PROMPT_SELECTION_V3_PROTOCOL"
LICENSED_LEARNER_SCOPE = "PUBLIC_TRAIN_PROMPT_SELECTION_V3"
PASS_PROTOCOL = "PASS_PUBLIC_TRAIN_PROMPT_SELECTION_V3_PROTOCOL"
STOP_VOCABULARY = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_VOCABULARY"
STOP_TOTAL_FUNCTION = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_TOTAL_FUNCTION"
STOP_ATOM_SUPPORT = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_ATOM_SUPPORT"
STOP_TRANSITION_SUPPORT = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_TRANSITION_SUPPORT"
STOP_COUNTERFACTUAL = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V3_COUNTERFACTUAL"

SELF_POSED_TRAIN = "SELF_POSED_TRAIN"
FORWARD = "FORWARD"
REVERSED = "REVERSED"

PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT = 90
CANONICAL_CAUSAL_IDENTITY_COUNT = 42
CANONICAL_FIT_IDENTITY_COUNT = 24
CANONICAL_HELDOUT_IDENTITY_COUNT = 18
V2_FIT_EXACT_IDENTITY_COUNT = 16
V2_HELDOUT_EXACT_IDENTITY_COUNT = 0

V2_COMMIT = "b479688"
V2_SOURCE_PATH = "experiments/r3_pilot/public_train_prompt_selection_v2.py"
V2_SOURCE_SHA256 = "b5a388c57b864c39bd9008d96a00ae03f2007ab2b28582a920aacdf8dd4608aa"
V2_STATUS = "STOP_PUBLIC_TRAIN_PROMPT_SELECTION_V2_FIT"
V2_RESULT_SHA256 = "72228a1177bfcaf162533f5e8d91a46ec73e0bc180d5b8ed869ff49162a74c2b"

CAPSULE_COMMIT = "5c34341"
CAPSULE_SOURCE_PATH = "experiments/r3_pilot/public_train_structural_annotation_capsule_v1.py"
CAPSULE_SOURCE_SHA256 = "d2e977b3aff2e496307168ed64e85c2ebbc03579224bf58b0fd175f0ed096c88"
CAPSULE_STATUS = "PASS_PUBLIC_TRAIN_STRUCTURAL_ANNOTATION_CAPSULE_V1_IDENTIFIABLE"
CAPSULE_SHA256 = "484521f34a98b53f4e21a1d279510c07165fc65f8de72b54845c06a831762caf"
SELECTION_V1_PROTOCOL_SHA256 = "dc6cb9afd1ce92e15cfeca3545f3b0c26ec041a71c040dae686f76acfcf3ff47"

COMPILER_GRAMMAR_SOURCE_COMMIT = "0121fd279b3dd9dda59776a31b9fe7a12f774217"
COMPILER_GRAMMAR_SOURCE_PATH = "src/humanbrain/simple_swe/tasks.py"
COMPILER_GRAMMAR_SOURCE_SHA256 = "b0edfc2eedc25335bcb2318a7b13026f6ab7affc29643af76e16f4c6b57240c3"

PHYSICAL_CELL_COUNT = 4096
ASSEMBLY_ACTIVE_CELL_COUNT = 12
MIN_ATOM_SUPPORT = 2
MIN_TRANSITION_SUPPORT = 1

DEV_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_OUTPUT_READ_COUNT = 0
PROGRAM_ARTIFACT_READ_COUNT = 0
EXECUTABLE_SOURCE_READ_COUNT = 0
AST_ARTIFACT_READ_COUNT = 0
PROPERTY_ROW_READ_COUNT = 0
TASK_IDENTIFIER_READ_COUNT = 0
FAMILY_IDENTIFIER_READ_COUNT = 0
ENTRYPOINT_SPELLING_NEURAL_EVIDENCE_COUNT = 0
AUDIT_HASH_NEURAL_EVIDENCE_COUNT = 0
CANONICAL_TARGET_LABEL_READ_COUNT = 0
CANONICAL_ANNOTATION_IMPORT_COUNT = 0
CANONICAL_IDENTITY_PAYLOAD_READ_COUNT = 0
LEARNER_ARRAY_COUNT = 0
LEARNING_RUN_COUNT = 0
ELECTRICAL_EXECUTION_COUNT = 0
CANDIDATE_COUNT = 0
PROOF_COUNT = 0
WHOLE_TARGET_CLASS_HEAD_COUNT = 0

_ROOT = Path(__file__).resolve().parents[2]

_DEPENDENCY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.dependency.v1\0"
_GRAMMAR_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.grammar.v1\0"
_VOCABULARY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.vocabulary.v1\0"
_ATOM_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.atom.v1\0"
_TRANSITION_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.transition.v1\0"
_ASSEMBLY_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.assembly.v1\0"
_LESSON_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.lesson.v1\0"
_PAIR_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.pair.v1\0"
_CURRICULUM_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.curriculum.v1\0"
_AUDIT_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.audit.v1\0"
_CONTROL_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.control.v1\0"
_LEARNER_GATE_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.learner-gate.v1\0"
_SURFACE_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.surface.v1\0"
_PROTOCOL_NAMESPACE = b"humanbrain.public-train-prompt-selection-v3.protocol.v1\0"


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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class GrammarProduction:
    """One compiler-owned abstract production, independent of prompt targets."""

    production: str
    family: str
    input_types: tuple[str, ...]
    output_type: str
    child_roles: tuple[str, ...]
    root_eligible: bool

    def record(self) -> dict[str, object]:
        return {
            "production": self.production,
            "family": self.family,
            "input_types": self.input_types,
            "output_type": self.output_type,
            "child_roles": self.child_roles,
            "root_eligible": self.root_eligible,
        }


# This declaration is the sole source of production/family/type fillers.  It is
# intentionally defined before any canonical dependency record and does not use
# the V1 capsule or V2 result at construction time.
GRAMMAR_PRODUCTIONS = (
    GrammarProduction("parameter", "terminal-parameter", (), "Generic", (), False),
    GrammarProduction("literal", "terminal-literal", (), "Generic", (), False),
    GrammarProduction("add", "scalar-additive", ("Int", "Int"), "Int", ("left", "right"), True),
    GrammarProduction("absolute", "scalar-unary", ("Int",), "Int", ("operand",), False),
    GrammarProduction("maximum", "scalar-extremum", ("Int", "Int"), "Int", ("left", "right"), True),
    GrammarProduction("minimum", "scalar-extremum", ("Int", "Int"), "Int", ("left", "right"), True),
    GrammarProduction(
        "map", "list-map", ("List[Int]", "Int->Int"), "List[Int]", ("collection", "binder"), True
    ),
    GrammarProduction(
        "value-binder", "binder", ("List[Int]",), "Int", ("bound-item", "body"), False
    ),
    GrammarProduction("sum", "list-fold", ("List[Int]",), "Int", ("collection",), True),
    GrammarProduction(
        "less-than", "scalar-predicate", ("Int", "Int"), "Bool", ("left", "right"), False
    ),
    GrammarProduction(
        "predicate-binder", "binder", ("List[Int]",), "Bool", ("bound-item", "predicate"), False
    ),
    GrammarProduction(
        "count", "list-count", ("List[Int]", "Int->Bool"), "Int", ("collection", "binder"), True
    ),
    GrammarProduction(
        "contains",
        "string-predicate",
        ("AsciiClass", "Char"),
        "Bool",
        ("class", "bound-char"),
        False,
    ),
    GrammarProduction(
        "filter", "string-filter", ("String", "Char->Bool"), "String", ("text", "binder"), True
    ),
    GrammarProduction("reverse", "string-transform", ("String",), "String", ("text",), True),
)

PRODUCTION_FILLERS = tuple(row.production for row in GRAMMAR_PRODUCTIONS)
COMPOSITE_PRODUCTION_FILLERS = tuple(
    row.production for row in GRAMMAR_PRODUCTIONS if row.child_roles
)
ROOT_PRODUCTION_FILLERS = tuple(row.production for row in GRAMMAR_PRODUCTIONS if row.root_eligible)
ROOT_FAMILY_FILLERS = tuple(
    dict.fromkeys(row.family for row in GRAMMAR_PRODUCTIONS if row.root_eligible)
)
ROOT_INPUT_TYPE_FILLERS = tuple(
    dict.fromkeys(row.input_types[0] for row in GRAMMAR_PRODUCTIONS if row.root_eligible)
)
ROOT_OUTPUT_TYPE_FILLERS = tuple(
    dict.fromkeys(row.output_type for row in GRAMMAR_PRODUCTIONS if row.root_eligible)
)


@dataclass(frozen=True, slots=True)
class RelaySlotSpec:
    role: str
    filler_kind: str
    fillers: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "filler_kind": self.filler_kind,
            "fillers": self.fillers,
        }


@dataclass(frozen=True, slots=True)
class RelayRelationSchema:
    channel: str
    slots: tuple[RelaySlotSpec, ...]

    def record(self) -> dict[str, object]:
        return {"channel": self.channel, "slots": tuple(row.record() for row in self.slots)}


RELAY_SCHEMAS = (
    RelayRelationSchema(
        "root-production-family",
        (
            RelaySlotSpec("root-production", "production", ROOT_PRODUCTION_FILLERS),
            RelaySlotSpec("root-family", "family", ROOT_FAMILY_FILLERS),
            RelaySlotSpec("input-type", "type", ROOT_INPUT_TYPE_FILLERS),
            RelaySlotSpec("output-type", "type", ROOT_OUTPUT_TYPE_FILLERS),
        ),
    ),
    RelayRelationSchema(
        "ordered-parent-child-edge",
        (
            RelaySlotSpec("parent-production", "production", COMPOSITE_PRODUCTION_FILLERS),
            RelaySlotSpec("child-production", "production", PRODUCTION_FILLERS),
            RelaySlotSpec("child-ordinal", "ordinal", ("0", "1", "2")),
        ),
    ),
    RelayRelationSchema(
        "argument-operand-orientation",
        (
            RelaySlotSpec("operator-production", "production", COMPOSITE_PRODUCTION_FILLERS),
            RelaySlotSpec("operand-ordinal", "ordinal", ("0", "1", "2")),
            RelaySlotSpec(
                "source-role",
                "semantic-role",
                (
                    "input-parameter",
                    "literal",
                    "nested-result",
                    "collection",
                    "bound-item",
                    "predicate",
                    "text",
                    "class",
                    "bound-char",
                ),
            ),
        ),
    ),
    RelayRelationSchema(
        "nesting-relation",
        (
            RelaySlotSpec("outer-production", "production", COMPOSITE_PRODUCTION_FILLERS),
            RelaySlotSpec("inner-production", "production", COMPOSITE_PRODUCTION_FILLERS),
            RelaySlotSpec("depth", "depth", ("1", "2", "3")),
        ),
    ),
    RelayRelationSchema(
        "parameter-role-binding",
        (
            RelaySlotSpec(
                "semantic-slot",
                "semantic-slot",
                ("input-parameter", "collection-parameter", "text-parameter"),
            ),
            RelaySlotSpec(
                "source-role",
                "semantic-role",
                ("function-argument", "bound-item", "bound-char"),
            ),
        ),
    ),
    RelayRelationSchema(
        "literal-slot-binding",
        (
            RelaySlotSpec(
                "literal-source-role",
                "semantic-role",
                ("generic-literal", "compiler-constant"),
            ),
            RelaySlotSpec(
                "semantic-slot",
                "semantic-slot",
                ("lower-bound", "upper-bound", "offset", "threshold", "ascii-class"),
            ),
        ),
    ),
    RelayRelationSchema(
        "literal-value-copy",
        (
            RelaySlotSpec(
                "semantic-slot",
                "semantic-slot",
                ("lower-bound", "upper-bound", "offset", "threshold", "ascii-class"),
            ),
            RelaySlotSpec("copy-role", "copy-role", ("emit-int", "emit-ascii-class")),
        ),
    ),
    RelayRelationSchema(
        "terminal-root-eclass-agreement",
        (
            RelaySlotSpec("root-authority", "authority", ("retained-root", "selected-root")),
            RelaySlotSpec("eclass-authority", "authority", ("retained-eclass", "selected-eclass")),
            RelaySlotSpec(
                "agreement-state",
                "agreement",
                ("exact-object-identity", "content-address-match"),
            ),
        ),
    ),
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
        record: dict[str, object] = {
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
        record: dict[str, object] = {
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
                        atom_index=len(atoms),
                        channel=schema.channel,
                        role=slot.role,
                        filler_kind=slot.filler_kind,
                        filler=filler,
                        atom_id=atom_id,
                        sha256=_namespaced_sha256(_ATOM_NAMESPACE, record),
                    )
                )

    transitions: list[RelayOutputTransition] = []
    for schema in RELAY_SCHEMAS:
        roles = tuple(slot.role for slot in schema.slots)
        pairs = tuple(pairwise(roles))
        for from_role, to_role in pairs + tuple((right, left) for left, right in pairs):
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
                    transition_index=len(transitions),
                    channel=schema.channel,
                    from_role=from_role,
                    to_role=to_role,
                    transition_id=transition_id,
                    sha256=_namespaced_sha256(_TRANSITION_NAMESPACE, record),
                )
            )
    if (
        len({row.atom_id for row in atoms}) != len(atoms)
        or len({row.transition_id for row in transitions}) != len(transitions)
        or {row.channel for row in atoms} != {row.channel for row in RELAY_SCHEMAS}
    ):
        raise RuntimeError(STOP_VOCABULARY)
    return tuple(atoms), tuple(transitions)


OUTPUT_ATOMS, OUTPUT_TRANSITIONS = _build_vocabulary()


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
                type(cells) is not tuple
                or len(cells) != ASSEMBLY_ACTIVE_CELL_COUNT
                or len(set(cells)) != ASSEMBLY_ACTIVE_CELL_COUNT
                or tuple(sorted(cells)) != cells
                or any(
                    type(cell) is not int or not 0 <= cell < PHYSICAL_CELL_COUNT for cell in cells
                )
            ):
                raise ValueError("physical relation/role/filler assembly changed")

    def record(self) -> dict[str, object]:
        return {
            "relation_cells": self.relation_cells,
            "role_cells": self.role_cells,
            "filler_cells": self.filler_cells,
        }


@dataclass(frozen=True, slots=True)
class PhysicalCausalAssemblyKey:
    ordered_events: tuple[PhysicalAssemblyEvent, ...]

    def __post_init__(self) -> None:
        if type(self.ordered_events) is not tuple or len(self.ordered_events) < 2:
            raise ValueError("V3 causal assembly needs at least two ordered events")

    def record(self) -> dict[str, object]:
        return {"ordered_events": tuple(event.record() for event in self.ordered_events)}

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

    def causal_input_record(self) -> dict[str, object]:
        return self.causal_input.record()

    def teacher_target_record(self) -> dict[str, object]:
        return {
            "atom_ids": self.target_atom_ids,
            "transition_ids": self.target_transition_ids,
            "exact_compositional_target": True,
        }

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "causal_input": self.causal_input_record(),
            "teacher_target": self.teacher_target_record(),
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
        record: dict[str, object] = {
            "pair_index": self.pair_index,
            "kind": self.kind,
            "left_lesson_index": self.left_lesson_index,
            "right_lesson_index": self.right_lesson_index,
            "same_physical_cell_multiset": self.same_physical_cell_multiset,
            "atom_targets_differ": self.atom_targets_differ,
            "transition_targets_differ": self.transition_targets_differ,
        }
        if include_sha256:
            record["sha256"] = self.sha256
        return record


@dataclass(frozen=True, slots=True)
class PrelearningAudit:
    status: str
    lesson_count: int
    unique_causal_key_count: int
    causal_key_target_collision_count: int
    output_atom_count: int
    output_transition_count: int
    unsupported_atom_count: int
    unsupported_transition_count: int
    minimum_atom_support: int
    minimum_transition_support: int
    order_counterfactual_pair_count: int
    role_counterfactual_pair_count: int
    counterfactual_cell_multiset_mismatch_count: int
    order_target_failure_count: int
    role_target_failure_count: int
    causal_key_teacher_field_count: int
    canonical_target_label_read_count: int
    whole_target_class_head_count: int
    total_function: bool
    full_support: bool
    counterfactual_coverage: bool
    causal_binding_sha256: str
    atom_support_sha256: str
    transition_support_sha256: str
    counterfactual_sha256: str
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        record = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        if not include_sha256:
            record.pop("sha256")
        return record


def _event_for_atom(atom: RelayOutputAtom) -> PhysicalAssemblyEvent:
    return PhysicalAssemblyEvent(
        relation_cells=_assembly_cells("relation", atom.channel),
        role_cells=_assembly_cells("role", f"{atom.channel}|{atom.role}"),
        filler_cells=_assembly_cells("filler", f"{atom.filler_kind}|{atom.filler}"),
    )


def _build_curriculum() -> tuple[
    tuple[SelfPosedRelayLesson, ...], tuple[MatchedCounterfactualPair, ...]
]:
    atoms_by_binding = {(row.channel, row.role, row.filler): row for row in OUTPUT_ATOMS}
    transitions_by_roles = {
        (row.channel, row.from_role, row.to_role): row for row in OUTPUT_TRANSITIONS
    }
    lessons: list[SelfPosedRelayLesson] = []
    lesson_by_key: dict[PhysicalCausalAssemblyKey, int] = {}
    pairs: list[MatchedCounterfactualPair] = []

    def add_lesson(
        schema: RelayRelationSchema,
        assignment: tuple[RelayOutputAtom, ...],
        order: str,
    ) -> int:
        ordered_atoms = assignment if order == FORWARD else tuple(reversed(assignment))
        causal_input = PhysicalCausalAssemblyKey(
            tuple(_event_for_atom(row) for row in ordered_atoms)
        )
        atom_ids = tuple(sorted(row.atom_id for row in assignment))
        transition_ids = tuple(
            transitions_by_roles[(schema.channel, left.role, right.role)].transition_id
            for left, right in pairwise(ordered_atoms)
        )
        existing = lesson_by_key.get(causal_input)
        if existing is not None:
            prior = lessons[existing]
            if (prior.target_atom_ids, prior.target_transition_ids) != (
                atom_ids,
                transition_ids,
            ):
                raise RuntimeError(STOP_TOTAL_FUNCTION)
            return existing
        record = {
            "causal_input": causal_input.record(),
            "teacher_target": {
                "atom_ids": atom_ids,
                "transition_ids": transition_ids,
                "exact_compositional_target": True,
            },
            "audit_only": {
                "lesson_index": len(lessons),
                "split": SELF_POSED_TRAIN,
                "channel": schema.channel,
                "order": order,
            },
        }
        lesson = SelfPosedRelayLesson(
            lesson_index=len(lessons),
            split=SELF_POSED_TRAIN,
            causal_input=causal_input,
            target_atom_ids=atom_ids,
            target_transition_ids=transition_ids,
            audit_channel=schema.channel,
            audit_order=order,
            sha256=_namespaced_sha256(_LESSON_NAMESPACE, record),
        )
        lesson_by_key[causal_input] = lesson.lesson_index
        lessons.append(lesson)
        return lesson.lesson_index

    def add_pair(kind: str, left_index: int, right_index: int) -> None:
        key = (kind, left_index, right_index)
        if any((row.kind, row.left_lesson_index, row.right_lesson_index) == key for row in pairs):
            return
        left = lessons[left_index]
        right = lessons[right_index]
        record = {
            "pair_index": len(pairs),
            "kind": kind,
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

    for schema in RELAY_SCHEMAS:
        width = max(len(slot.fillers) for slot in schema.slots)
        assignments: list[tuple[RelayOutputAtom, ...]] = []
        for variant in range(width):
            assignment = tuple(
                atoms_by_binding[
                    (
                        schema.channel,
                        slot.role,
                        slot.fillers[(variant + slot_index) % len(slot.fillers)],
                    )
                ]
                for slot_index, slot in enumerate(schema.slots)
            )
            if assignment not in assignments:
                assignments.append(assignment)

        role_swaps: list[tuple[tuple[RelayOutputAtom, ...], tuple[RelayOutputAtom, ...]]] = []
        for left_slot_index, left_slot in enumerate(schema.slots):
            for right_slot_index in range(left_slot_index + 1, len(schema.slots)):
                right_slot = schema.slots[right_slot_index]
                common = tuple(
                    filler for filler in left_slot.fillers if filler in set(right_slot.fillers)
                )
                if left_slot.filler_kind != right_slot.filler_kind or len(common) < 2:
                    continue
                base = list(assignments[0])
                base[left_slot_index] = atoms_by_binding[
                    (schema.channel, left_slot.role, common[0])
                ]
                base[right_slot_index] = atoms_by_binding[
                    (schema.channel, right_slot.role, common[1])
                ]
                swapped = list(base)
                swapped[left_slot_index] = atoms_by_binding[
                    (schema.channel, left_slot.role, common[1])
                ]
                swapped[right_slot_index] = atoms_by_binding[
                    (schema.channel, right_slot.role, common[0])
                ]
                left_assignment = tuple(base)
                right_assignment = tuple(swapped)
                role_swaps.append((left_assignment, right_assignment))
                for assignment in (left_assignment, right_assignment):
                    if assignment not in assignments:
                        assignments.append(assignment)

        for assignment in assignments:
            forward_index = add_lesson(schema, assignment, FORWARD)
            reverse_index = add_lesson(schema, assignment, REVERSED)
            add_pair("event-order-reversal", forward_index, reverse_index)
        for left_assignment, right_assignment in role_swaps:
            add_pair(
                "role-filler-swap",
                add_lesson(schema, left_assignment, FORWARD),
                add_lesson(schema, right_assignment, FORWARD),
            )
    return tuple(lessons), tuple(pairs)


def _audit_curriculum(
    lessons: tuple[SelfPosedRelayLesson, ...],
    pairs: tuple[MatchedCounterfactualPair, ...],
) -> PrelearningAudit:
    targets_by_key: dict[
        PhysicalCausalAssemblyKey, set[tuple[tuple[str, ...], tuple[str, ...]]]
    ] = defaultdict(set)
    for row in lessons:
        targets_by_key[row.causal_input].add((row.target_atom_ids, row.target_transition_ids))
    collisions = sum(len(targets) != 1 for targets in targets_by_key.values())

    atom_support = Counter(atom for row in lessons for atom in row.target_atom_ids)
    transition_support = Counter(
        transition for row in lessons for transition in row.target_transition_ids
    )
    unsupported_atoms = tuple(
        row.atom_id for row in OUTPUT_ATOMS if atom_support[row.atom_id] < MIN_ATOM_SUPPORT
    )
    unsupported_transitions = tuple(
        row.transition_id
        for row in OUTPUT_TRANSITIONS
        if transition_support[row.transition_id] < MIN_TRANSITION_SUPPORT
    )
    order_pairs = tuple(row for row in pairs if row.kind == "event-order-reversal")
    role_pairs = tuple(row for row in pairs if row.kind == "role-filler-swap")
    multiset_mismatches = sum(not row.same_physical_cell_multiset for row in pairs)
    order_failures = sum(
        row.atom_targets_differ or not row.transition_targets_differ for row in order_pairs
    )
    role_failures = sum(
        not row.atom_targets_differ or row.transition_targets_differ for row in role_pairs
    )
    if collisions:
        status = STOP_TOTAL_FUNCTION
    elif unsupported_atoms:
        status = STOP_ATOM_SUPPORT
    elif unsupported_transitions:
        status = STOP_TRANSITION_SUPPORT
    elif (
        not order_pairs or not role_pairs or multiset_mismatches or order_failures or role_failures
    ):
        status = STOP_COUNTERFACTUAL
    else:
        status = PASS_PROTOCOL

    causal_binding_sha = _namespaced_sha256(
        _AUDIT_NAMESPACE + b"causal-bindings\0",
        tuple(
            {
                "causal_input": row.causal_input_record(),
                "teacher_target": row.teacher_target_record(),
            }
            for row in lessons
        ),
    )
    atom_support_record = tuple((row.atom_id, atom_support[row.atom_id]) for row in OUTPUT_ATOMS)
    transition_support_record = tuple(
        (row.transition_id, transition_support[row.transition_id]) for row in OUTPUT_TRANSITIONS
    )
    counterfactual_record = tuple(row.record() for row in pairs)
    record = {
        "status": status,
        "lesson_count": len(lessons),
        "unique_causal_key_count": len(targets_by_key),
        "causal_key_target_collision_count": collisions,
        "output_atom_count": len(OUTPUT_ATOMS),
        "output_transition_count": len(OUTPUT_TRANSITIONS),
        "unsupported_atom_count": len(unsupported_atoms),
        "unsupported_transition_count": len(unsupported_transitions),
        "minimum_atom_support": min(atom_support.values(), default=0),
        "minimum_transition_support": min(transition_support.values(), default=0),
        "order_counterfactual_pair_count": len(order_pairs),
        "role_counterfactual_pair_count": len(role_pairs),
        "counterfactual_cell_multiset_mismatch_count": multiset_mismatches,
        "order_target_failure_count": order_failures,
        "role_target_failure_count": role_failures,
        "causal_key_teacher_field_count": 0,
        "canonical_target_label_read_count": CANONICAL_TARGET_LABEL_READ_COUNT,
        "whole_target_class_head_count": WHOLE_TARGET_CLASS_HEAD_COUNT,
        "total_function": collisions == 0,
        "full_support": not unsupported_atoms and not unsupported_transitions,
        "counterfactual_coverage": bool(order_pairs and role_pairs)
        and not (multiset_mismatches or order_failures or role_failures),
        "causal_binding_sha256": causal_binding_sha,
        "atom_support_sha256": _namespaced_sha256(
            _AUDIT_NAMESPACE + b"atom-support\0", atom_support_record
        ),
        "transition_support_sha256": _namespaced_sha256(
            _AUDIT_NAMESPACE + b"transition-support\0", transition_support_record
        ),
        "counterfactual_sha256": _namespaced_sha256(
            _AUDIT_NAMESPACE + b"counterfactuals\0", counterfactual_record
        ),
    }
    return PrelearningAudit(
        **record,
        sha256=_namespaced_sha256(_AUDIT_NAMESPACE, record),
    )


@dataclass(frozen=True, slots=True)
class FrozenPublicTrainPromptSelectionV3Protocol:
    scope: str
    status: str
    grammar_productions: tuple[GrammarProduction, ...]
    relay_schemas: tuple[RelayRelationSchema, ...]
    output_atoms: tuple[RelayOutputAtom, ...]
    output_transitions: tuple[RelayOutputTransition, ...]
    lessons: tuple[SelfPosedRelayLesson, ...]
    counterfactual_pairs: tuple[MatchedCounterfactualPair, ...]
    prelearning_audit: PrelearningAudit
    dependency_sha256: str
    grammar_sha256: str
    vocabulary_sha256: str
    curriculum_sha256: str
    audit_sha256: str
    controls_sha256: str
    learner_gate_sha256: str
    surface_boundary_sha256: str
    protocol_sha256: str

    def root_record(self, *, include_protocol_sha256: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "scope": self.scope,
            "status": self.status,
            "dependency_sha256": self.dependency_sha256,
            "grammar_sha256": self.grammar_sha256,
            "vocabulary_sha256": self.vocabulary_sha256,
            "curriculum_sha256": self.curriculum_sha256,
            "audit_sha256": self.audit_sha256,
            "controls_sha256": self.controls_sha256,
            "learner_gate_sha256": self.learner_gate_sha256,
            "surface_boundary_sha256": self.surface_boundary_sha256,
            "canonical_prompt_appearance_count": PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT,
            "canonical_causal_identity_count": CANONICAL_CAUSAL_IDENTITY_COUNT,
            "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
            "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
            "v2_fit_exact_identity_count": V2_FIT_EXACT_IDENTITY_COUNT,
            "v2_heldout_exact_identity_count": V2_HELDOUT_EXACT_IDENTITY_COUNT,
            "production_count": len(self.grammar_productions),
            "relay_channel_count": len(self.relay_schemas),
            "output_atom_count": len(self.output_atoms),
            "output_transition_count": len(self.output_transitions),
            "self_posed_train_lesson_count": len(self.lessons),
            "order_counterfactual_pair_count": sum(
                row.kind == "event-order-reversal" for row in self.counterfactual_pairs
            ),
            "role_counterfactual_pair_count": sum(
                row.kind == "role-filler-swap" for row in self.counterfactual_pairs
            ),
            "learner_array_count": LEARNER_ARRAY_COUNT,
            "learning_run_count": LEARNING_RUN_COUNT,
            "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        }
        if include_protocol_sha256:
            record["protocol_sha256"] = self.protocol_sha256
        return record


def _authenticate_dependencies() -> None:
    checks = (
        (V2_SOURCE_PATH, V2_SOURCE_SHA256),
        (CAPSULE_SOURCE_PATH, CAPSULE_SOURCE_SHA256),
        (COMPILER_GRAMMAR_SOURCE_PATH, COMPILER_GRAMMAR_SOURCE_SHA256),
    )
    if any(
        not _is_sha256(expected)
        or not (_ROOT / relative_path).is_file()
        or sha256((_ROOT / relative_path).read_bytes()).hexdigest() != expected
        for relative_path, expected in checks
    ):
        raise RuntimeError("prompt-selection V3 frozen dependency changed")


GRAMMAR_RECORD = _freeze_record(
    {
        "compiler_grammar_source_commit": COMPILER_GRAMMAR_SOURCE_COMMIT,
        "compiler_grammar_source_path": COMPILER_GRAMMAR_SOURCE_PATH,
        "compiler_grammar_source_sha256": COMPILER_GRAMMAR_SOURCE_SHA256,
        "productions": tuple(row.record() for row in GRAMMAR_PRODUCTIONS),
        "relay_schemas": tuple(row.record() for row in RELAY_SCHEMAS),
        "declaration_order": "grammar-before-canonical-dependency-binding",
        "vocabulary_from_canonical_target_union": False,
        "canonical_target_label_read_count": CANONICAL_TARGET_LABEL_READ_COUNT,
    }
)
GRAMMAR_SHA256 = _namespaced_sha256(_GRAMMAR_NAMESPACE, dict(GRAMMAR_RECORD))

DEPENDENCY_RECORD = _freeze_record(
    {
        "v2_commit": V2_COMMIT,
        "v2_source_path": V2_SOURCE_PATH,
        "v2_source_sha256": V2_SOURCE_SHA256,
        "v2_status": V2_STATUS,
        "v2_result_sha256": V2_RESULT_SHA256,
        "v2_fit_exact_identity_count": V2_FIT_EXACT_IDENTITY_COUNT,
        "v2_heldout_exact_identity_count": V2_HELDOUT_EXACT_IDENTITY_COUNT,
        "capsule_commit": CAPSULE_COMMIT,
        "capsule_source_path": CAPSULE_SOURCE_PATH,
        "capsule_source_sha256": CAPSULE_SOURCE_SHA256,
        "capsule_status": CAPSULE_STATUS,
        "capsule_sha256": CAPSULE_SHA256,
        "selection_v1_protocol_sha256": SELECTION_V1_PROTOCOL_SHA256,
        "canonical_prompt_appearance_count": PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT,
        "canonical_causal_identity_count": CANONICAL_CAUSAL_IDENTITY_COUNT,
        "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
        "canonical_identity_payload_read_count": CANONICAL_IDENTITY_PAYLOAD_READ_COUNT,
        "canonical_annotation_module_imported": False,
        "v2_learner_module_imported_or_executed": False,
        "dependency_hashes_are_not_neural_evidence": True,
    }
)
DEPENDENCY_SHA256 = _namespaced_sha256(_DEPENDENCY_NAMESPACE, dict(DEPENDENCY_RECORD))

VOCABULARY_RECORD = _freeze_record(
    {
        "atoms": tuple(row.record() for row in OUTPUT_ATOMS),
        "transitions": tuple(row.record() for row in OUTPUT_TRANSITIONS),
        "atom_semantics": "channel-local relation/role/filler binding",
        "transition_semantics": "ordered adjacent role binding inside one relay channel",
        "whole_target_class_head_count": WHOLE_TARGET_CLASS_HEAD_COUNT,
        "canonical_target_derived_atom_count": 0,
    }
)
VOCABULARY_SHA256 = _namespaced_sha256(_VOCABULARY_NAMESPACE, dict(VOCABULARY_RECORD))

CONTROL_RECORD = _freeze_record(
    {
        "target_label_off": "causal assembly key remains unchanged; no target is admissible",
        "event_order_derangement": (
            "same relation/role/filler cell multiset; reversed ordered transitions required"
        ),
        "role_filler_derangement": (
            "same relation/role/filler cell multiset; different atom bindings required"
        ),
        "whole_target_label_head": "forbidden",
        "canonical_heldout_target_access": "forbidden",
        "canonical_heldout_is_fresh_adjudicative_evidence": False,
        "future_learner_admission": (
            "channel-local masked WTA for exact atoms and ordered transitions only"
        ),
        "future_learner_may_allocate_whole_target_classes": False,
        "future_learner_may_use_audit_or_dependency_hashes": False,
        "future_learner_may_use_task_family_or_entrypoint_identity": False,
    }
)
CONTROLS_SHA256 = _namespaced_sha256(_CONTROL_NAMESPACE, dict(CONTROL_RECORD))

LEARNER_GATE_RECORD = _freeze_record(
    {
        "licensed_successor_scope": LICENSED_LEARNER_SCOPE,
        "prerequisite_protocol_status": PASS_PROTOCOL,
        "required_self_posed_train_curriculum": True,
        "learner_visible_self_posed_input": (
            "ordered sparse relation/role/filler assembly events only"
        ),
        "learner_output": ("channel-local exact relay atoms plus ordered transition atoms"),
        "whole_target_class_heads_allowed": False,
        "canonical_fit_training_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_fit_projection": (
            "compiler-grammar normalization into the frozen atom/transition vocabulary"
        ),
        "canonical_heldout_training_identity_count": 0,
        "canonical_heldout_target_read_count": 0,
        "canonical_heldout_replay_allowed": False,
        "v2_checkpoint_or_weight_reuse_allowed": False,
        "fresh_checkpoint_required": True,
        "task_family_entrypoint_or_audit_identity_allowed": False,
        "desired_program_property_dev_or_protected_evidence_allowed": False,
        "minimum_admission": (
            "exact target-set admission per channel and exact ordered transitions"
        ),
        "claim_if_passed": "self-posed mechanism plus canonical FIT admission only",
        "joint_train_integration_licensed": False,
        "fresh_evaluation_required_before_joint_train": True,
    }
)
LEARNER_GATE_SHA256 = _namespaced_sha256(_LEARNER_GATE_NAMESPACE, dict(LEARNER_GATE_RECORD))

SURFACE_BOUNDARY_RECORD = _freeze_record(
    {
        "dev_input_count": DEV_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_output_read_count": DESIRED_OUTPUT_READ_COUNT,
        "program_artifact_read_count": PROGRAM_ARTIFACT_READ_COUNT,
        "executable_source_read_count": EXECUTABLE_SOURCE_READ_COUNT,
        "ast_artifact_read_count": AST_ARTIFACT_READ_COUNT,
        "property_row_read_count": PROPERTY_ROW_READ_COUNT,
        "task_identifier_read_count": TASK_IDENTIFIER_READ_COUNT,
        "family_identifier_read_count": FAMILY_IDENTIFIER_READ_COUNT,
        "entrypoint_spelling_neural_evidence_count": (ENTRYPOINT_SPELLING_NEURAL_EVIDENCE_COUNT),
        "audit_hash_neural_evidence_count": AUDIT_HASH_NEURAL_EVIDENCE_COUNT,
        "canonical_target_label_read_count": CANONICAL_TARGET_LABEL_READ_COUNT,
        "canonical_annotation_import_count": CANONICAL_ANNOTATION_IMPORT_COUNT,
        "canonical_identity_payload_read_count": CANONICAL_IDENTITY_PAYLOAD_READ_COUNT,
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "proof_count": PROOF_COUNT,
        "whole_target_class_head_count": WHOLE_TARGET_CLASS_HEAD_COUNT,
        "data_only": True,
        "learner_or_execution_interface_present": False,
        "canonical_labels_used_to_construct_vocabulary_or_curriculum": False,
    }
)
SURFACE_BOUNDARY_SHA256 = _namespaced_sha256(_SURFACE_NAMESPACE, dict(SURFACE_BOUNDARY_RECORD))


def _build_protocol() -> FrozenPublicTrainPromptSelectionV3Protocol:
    _authenticate_dependencies()
    lessons, pairs = _build_curriculum()
    audit = _audit_curriculum(lessons, pairs)
    if audit.status != PASS_PROTOCOL:
        raise RuntimeError(audit.status)
    curriculum_record = {
        "split": SELF_POSED_TRAIN,
        "lessons": tuple(row.record() for row in lessons),
        "counterfactual_pairs": tuple(row.record() for row in pairs),
        "canonical_target_label_read_count": CANONICAL_TARGET_LABEL_READ_COUNT,
        "canonical_identity_payload_read_count": CANONICAL_IDENTITY_PAYLOAD_READ_COUNT,
        "fresh_from_v2_lesson_count": len(lessons),
    }
    curriculum_sha = _namespaced_sha256(_CURRICULUM_NAMESPACE, curriculum_record)
    root_without_protocol = {
        "scope": PROTOCOL_SCOPE,
        "status": audit.status,
        "dependency_sha256": DEPENDENCY_SHA256,
        "grammar_sha256": GRAMMAR_SHA256,
        "vocabulary_sha256": VOCABULARY_SHA256,
        "curriculum_sha256": curriculum_sha,
        "audit_sha256": audit.sha256,
        "controls_sha256": CONTROLS_SHA256,
        "learner_gate_sha256": LEARNER_GATE_SHA256,
        "surface_boundary_sha256": SURFACE_BOUNDARY_SHA256,
        "canonical_prompt_appearance_count": PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT,
        "canonical_causal_identity_count": CANONICAL_CAUSAL_IDENTITY_COUNT,
        "canonical_fit_identity_count": CANONICAL_FIT_IDENTITY_COUNT,
        "canonical_heldout_identity_count": CANONICAL_HELDOUT_IDENTITY_COUNT,
        "v2_fit_exact_identity_count": V2_FIT_EXACT_IDENTITY_COUNT,
        "v2_heldout_exact_identity_count": V2_HELDOUT_EXACT_IDENTITY_COUNT,
        "production_count": len(GRAMMAR_PRODUCTIONS),
        "relay_channel_count": len(RELAY_SCHEMAS),
        "output_atom_count": len(OUTPUT_ATOMS),
        "output_transition_count": len(OUTPUT_TRANSITIONS),
        "self_posed_train_lesson_count": len(lessons),
        "order_counterfactual_pair_count": sum(row.kind == "event-order-reversal" for row in pairs),
        "role_counterfactual_pair_count": sum(row.kind == "role-filler-swap" for row in pairs),
        "learner_array_count": LEARNER_ARRAY_COUNT,
        "learning_run_count": LEARNING_RUN_COUNT,
        "electrical_execution_count": ELECTRICAL_EXECUTION_COUNT,
    }
    return FrozenPublicTrainPromptSelectionV3Protocol(
        scope=PROTOCOL_SCOPE,
        status=audit.status,
        grammar_productions=GRAMMAR_PRODUCTIONS,
        relay_schemas=RELAY_SCHEMAS,
        output_atoms=OUTPUT_ATOMS,
        output_transitions=OUTPUT_TRANSITIONS,
        lessons=lessons,
        counterfactual_pairs=pairs,
        prelearning_audit=audit,
        dependency_sha256=DEPENDENCY_SHA256,
        grammar_sha256=GRAMMAR_SHA256,
        vocabulary_sha256=VOCABULARY_SHA256,
        curriculum_sha256=curriculum_sha,
        audit_sha256=audit.sha256,
        controls_sha256=CONTROLS_SHA256,
        learner_gate_sha256=LEARNER_GATE_SHA256,
        surface_boundary_sha256=SURFACE_BOUNDARY_SHA256,
        protocol_sha256=_namespaced_sha256(_PROTOCOL_NAMESPACE, root_without_protocol),
    )


PROTOCOL: Final = _build_protocol()


__all__ = [
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "CANONICAL_CAUSAL_IDENTITY_COUNT",
    "CANONICAL_FIT_IDENTITY_COUNT",
    "CANONICAL_HELDOUT_IDENTITY_COUNT",
    "CAPSULE_COMMIT",
    "CAPSULE_SHA256",
    "CAPSULE_SOURCE_SHA256",
    "COMPILER_GRAMMAR_SOURCE_SHA256",
    "CONTROLS_SHA256",
    "CONTROL_RECORD",
    "DEPENDENCY_RECORD",
    "DEPENDENCY_SHA256",
    "GRAMMAR_PRODUCTIONS",
    "GRAMMAR_RECORD",
    "GRAMMAR_SHA256",
    "LEARNER_ARRAY_COUNT",
    "LEARNER_GATE_RECORD",
    "LEARNER_GATE_SHA256",
    "LEARNING_RUN_COUNT",
    "LICENSED_LEARNER_SCOPE",
    "MIN_ATOM_SUPPORT",
    "MIN_TRANSITION_SUPPORT",
    "OUTPUT_ATOMS",
    "OUTPUT_TRANSITIONS",
    "PASS_PROTOCOL",
    "PHYSICAL_CELL_COUNT",
    "PROGRAM_ARTIFACT_READ_COUNT",
    "PROTOCOL",
    "PROTOCOL_SCOPE",
    "PUBLIC_TRAIN_PROMPT_APPEARANCE_COUNT",
    "RELAY_SCHEMAS",
    "SELF_POSED_TRAIN",
    "SURFACE_BOUNDARY_RECORD",
    "SURFACE_BOUNDARY_SHA256",
    "V2_COMMIT",
    "V2_FIT_EXACT_IDENTITY_COUNT",
    "V2_HELDOUT_EXACT_IDENTITY_COUNT",
    "V2_RESULT_SHA256",
    "V2_SOURCE_SHA256",
    "VOCABULARY_RECORD",
    "VOCABULARY_SHA256",
    "WHOLE_TARGET_CLASS_HEAD_COUNT",
    "FrozenPublicTrainPromptSelectionV3Protocol",
    "GrammarProduction",
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
