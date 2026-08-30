"""Packed-primary scalar semantic frontier, before concrete proof witnesses.

The frontier grows immutable operator families from literal/role domains and
physical semantic e-classes.  Concrete expressions do not exist during build:
one selected proof, or the first two ambiguous proofs, is materialized only
after a frozen desired-independent frontier enters the physical vector race.

This is deliberately the smallest scalar vertical slice.  It contains role
and literal terminals, absolute, additive multiplicity, and one canonical
alternating extrema chain.  It is not the full typed semantic V2 grammar.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import product

from experiments.r3_pilot import public_typed_operator_successor as operator_successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from humanbrain.typed_program import FactorAction

FRONTIER_READY = "PACKED_SCALAR_FRONTIER_READY"
STOP_WORK_BOUND = "STOP_PACKED_SCALAR_FRONTIER_WORK_BOUND"
SELECTED = "SELECTED"
STOP_NO_EXACT = "STOP_NO_EXACT_SEMANTIC_VECTOR"
STOP_AMBIGUOUS = "STOP_AMBIGUOUS_VISIBLE_SPEC"

MAX_RECIPE_RELATION_WORK = 30_000
MAX_PHYSICAL_PREDICTIONS = 30_000
MAX_PHYSICAL_RACES = 80_000

_ROLE_SLOT = packed.TypedSlot("role", "Int", "argument", 0)
_LITERAL_SLOT = packed.TypedSlot("literal", "Int", "literal", 0)
_NAMESPACE = b"typed-semantic-packed-frontier-v1\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("utf-8"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


def _slot_record(slot: packed.TypedSlot) -> tuple[object, ...]:
    return slot.to_record()


def _vector_record(
    values: Sequence[output_cells.TypedOutputEncoding],
) -> tuple[str, ...]:
    return tuple(value.sha256 for value in values)


def _vector_sha256(values: Sequence[output_cells.TypedOutputEncoding]) -> str:
    return _digest("physical-vector", _vector_record(values))


def _eclass_id(values: Sequence[output_cells.TypedOutputEncoding]) -> str:
    return _digest("e-class-id", _vector_record(values))


def _family_identity(production: str, child_family_ids: Sequence[str]) -> str:
    return _digest(
        "family-id",
        {"production": production, "children": tuple(child_family_ids)},
    )


class _FrontierSession(packed.PackedRelationSession):
    """One enclosing relation session with frontier events in the same cap."""

    __slots__ = ()

    def charge_frontier(self, event: str) -> None:
        if type(event) is not str or not event:
            raise ValueError("frontier event must be a nonempty string")
        self._charge(f"frontier:{event}")


@dataclass(frozen=True, slots=True)
class PackedFamily:
    """One symbolic proof family whose relation precedes semantic expansion."""

    family_id: str
    production: str
    inside_cost: int
    child_family_ids: tuple[str, ...]
    child_slot_indices: tuple[tuple[int, ...], ...]
    relation: packed.PackedBindingRelation
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedENode:
    """One family partition keyed only by child physical semantic e-classes."""

    enode_id: str
    family_id: str
    child_semantic_refs: tuple[tuple[str, ...], ...]
    proof_slice: packed.PackedFamilySlice
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    physical_output_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedEClass:
    """One physical semantic vector with authenticated proof-family slices."""

    eclass_id: str
    minimum_inside_cost: int
    enode_ids: tuple[str, ...]
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    proof_slices: tuple[packed.PackedFamilySlice, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedWitness:
    family_id: str
    binding: tuple[int, ...]
    source: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedFrontierWorkAudit:
    relation: packed.PackedRelationWorkAudit
    physical_predictions: int
    prediction_cache_hits: int
    physical_races: int
    concrete_proof_witnesses: int
    materialized_witnesses: int
    bound_name: str | None
    bound_attempted: int | None
    agenda_trace: tuple[tuple[str, str], ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedFrontierState:
    status: str
    case_arguments: tuple[int, ...]
    literals: tuple[int, ...]
    requested_max_ast_cost: int
    maximum_completed_ast_cost: int
    operator_checkpoint_sha256: str
    families: tuple[PackedFamily, ...]
    enodes: tuple[PackedENode, ...]
    eclasses: tuple[PackedEClass, ...]
    root_eclass_ids: tuple[str, ...]
    build_work_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PackedFrontierSelection:
    status: str
    frontier_sha256: str
    selected_eclass_id: str | None
    exact_proof_count: int
    witnesses: tuple[PackedWitness, ...]
    selected_proof_forest_sha256: str | None
    race: semantic_race.TypedSemanticVectorRaceAudit | None
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    work: PackedFrontierWorkAudit
    sha256: str


class _PhysicalWorkBoundExceeded(RuntimeError):
    def __init__(self, name: str, attempted: int) -> None:
        super().__init__(f"{name} limit exceeded at {attempted}")
        self.name = name
        self.attempted = attempted


@dataclass(slots=True)
class _FamilyParts:
    family: PackedFamily
    enodes: list[PackedENode]


def _family_payload(family: PackedFamily) -> dict[str, object]:
    return {
        "family_id": family.family_id,
        "production": family.production,
        "inside_cost": family.inside_cost,
        "child_family_ids": family.child_family_ids,
        "child_slot_indices": family.child_slot_indices,
        "relation_sha256": family.relation.sha256,
        "slots": tuple(_slot_record(slot) for slot in family.relation.slots),
    }


def _enode_payload(enode: PackedENode) -> dict[str, object]:
    return {
        "enode_id": enode.enode_id,
        "family_id": enode.family_id,
        "child_semantic_refs": enode.child_semantic_refs,
        "proof_slice_sha256": enode.proof_slice.sha256,
        "physical_output_sha256": enode.physical_output_sha256,
        "physical_outputs": _vector_record(enode.physical_outputs),
    }


def _eclass_payload(eclass: PackedEClass) -> dict[str, object]:
    return {
        "eclass_id": eclass.eclass_id,
        "minimum_inside_cost": eclass.minimum_inside_cost,
        "enode_ids": eclass.enode_ids,
        "physical_outputs": _vector_record(eclass.physical_outputs),
        "proof_slices": tuple(
            (item.family_id, item.relation.sha256, item.sha256) for item in eclass.proof_slices
        ),
    }


class _Builder:
    def __init__(
        self,
        supplied_case_arguments: Iterable[int],
        supplied_literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
    ) -> None:
        self.case_arguments: tuple[int, ...] = ()
        self.supplied_case_arguments = supplied_case_arguments
        self.literals: tuple[int, ...] = ()
        self.supplied_literals = supplied_literals
        self.max_ast_cost = max_ast_cost
        self.operator = operator
        self.session = _FrontierSession(work_limit=MAX_RECIPE_RELATION_WORK)
        self.families: dict[str, _FamilyParts] = {}
        self.enodes: dict[str, PackedENode] = {}
        self._eclass_enodes: dict[str, list[PackedENode]] = {}
        self._eclass_outputs: dict[str, tuple[output_cells.TypedOutputEncoding, ...]] = {}
        self._prediction_cache: dict[
            tuple[str, str, str | None],
            output_cells.TypedOutputEncoding,
        ] = {}
        self.physical_predictions = 0
        self.prediction_cache_hits = 0
        self.physical_races = 0
        self.bound_name: str | None = None
        self.bound_attempted: int | None = None
        self.agenda_trace: list[tuple[str, str]] = []
        self.maximum_completed_ast_cost = 0
        self.normalization_count = 0

    def _charge(self, event: str) -> None:
        self.session.charge_frontier(event)

    def _record(self, kind: str, identity: str) -> None:
        self.agenda_trace.append((kind, identity))

    def _family_id(
        self,
        production: str,
        children: tuple[PackedFamily, ...],
    ) -> str:
        return _family_identity(production, tuple(child.family_id for child in children))

    @staticmethod
    def _new_slot(
        source: packed.TypedSlot,
        occurrences: dict[tuple[str, str, str], int],
    ) -> packed.TypedSlot:
        key = (source.kind, source.value_type, source.name)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        return packed.TypedSlot(*key, occurrence)

    def _compose_relation(
        self,
        children: tuple[PackedFamily, ...],
        relations: tuple[packed.PackedBindingRelation, ...],
    ) -> tuple[
        packed.PackedBindingRelation,
        tuple[tuple[int, ...], ...],
        tuple[tuple[packed.TypedSlot, ...], ...],
    ]:
        if len(children) != len(relations) or not children:
            raise ValueError("family composition needs aligned children")
        occurrences: dict[tuple[str, str, str], int] = {}
        renamed: list[packed.PackedBindingRelation] = []
        target_slots_by_child: list[tuple[packed.TypedSlot, ...]] = []
        for child, relation in zip(children, relations, strict=True):
            if relation.slots != child.relation.slots:
                raise RuntimeError("child semantic slice schema changed")
            targets = tuple(self._new_slot(slot, occurrences) for slot in relation.slots)
            target_slots_by_child.append(targets)
            renamed.append(self.session.rename(relation, targets))
        combined = renamed[0]
        for relation in renamed[1:]:
            combined = self.session.cartesian_product(combined, relation)
        positions = {slot: index for index, slot in enumerate(combined.slots)}
        child_indices = tuple(
            tuple(positions[slot] for slot in slots) for slots in target_slots_by_child
        )
        return combined, child_indices, tuple(target_slots_by_child)

    def _store_family(
        self,
        production: str,
        inside_cost: int,
        children: tuple[PackedFamily, ...],
        relation: packed.PackedBindingRelation,
        child_slot_indices: tuple[tuple[int, ...], ...],
    ) -> PackedFamily:
        family_id = self._family_id(production, children)
        payload = {
            "family_id": family_id,
            "production": production,
            "inside_cost": inside_cost,
            "child_family_ids": tuple(child.family_id for child in children),
            "child_slot_indices": child_slot_indices,
            "relation_sha256": relation.sha256,
            "slots": tuple(_slot_record(slot) for slot in relation.slots),
        }
        family = PackedFamily(
            family_id=family_id,
            production=production,
            inside_cost=inside_cost,
            child_family_ids=tuple(child.family_id for child in children),
            child_slot_indices=child_slot_indices,
            relation=relation,
            sha256=_digest("family", payload),
        )
        if family_id in self.families:
            raise RuntimeError("duplicate packed family")
        self._charge("family")
        self._record("family-primary", family_id)
        self.families[family_id] = _FamilyParts(family, [])
        return family

    def _store_enode(
        self,
        family: PackedFamily,
        child_refs: Sequence[tuple[str, ...]],
        relation: packed.PackedBindingRelation,
        outputs: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> PackedENode:
        slice_ = self.session.family_slice(family.family_id, relation)
        output_sha = _vector_sha256(outputs)
        refs = tuple(sorted(set(tuple(item) for item in child_refs)))
        enode_id = _digest(
            "e-node-id",
            {
                "family_id": family.family_id,
                "child_semantic_refs": refs,
                "relation_sha256": relation.sha256,
                "physical_output_sha256": output_sha,
            },
        )
        payload = {
            "enode_id": enode_id,
            "family_id": family.family_id,
            "child_semantic_refs": refs,
            "proof_slice_sha256": slice_.sha256,
            "physical_output_sha256": output_sha,
            "physical_outputs": _vector_record(outputs),
        }
        enode = PackedENode(
            enode_id=enode_id,
            family_id=family.family_id,
            child_semantic_refs=refs,
            proof_slice=slice_,
            physical_outputs=outputs,
            physical_output_sha256=output_sha,
            sha256=_digest("e-node", payload),
        )
        incumbent = self.enodes.setdefault(enode_id, enode)
        if incumbent != enode:
            raise RuntimeError("packed e-node identity collision")
        self._charge("e-node")
        self.families[family.family_id].enodes.append(enode)
        eclass_id = _eclass_id(outputs)
        self._eclass_outputs.setdefault(eclass_id, outputs)
        self._eclass_enodes.setdefault(eclass_id, []).append(enode)
        return enode

    def _predict_case(
        self,
        production: str,
        left: output_cells.TypedOutputEncoding,
        right: output_cells.TypedOutputEncoding | None = None,
    ) -> output_cells.TypedOutputEncoding:
        key = (production, left.sha256, None if right is None else right.sha256)
        cached = self._prediction_cache.get(key)
        if cached is not None:
            output_cells.TypedOutputCellBank.validate(cached)
            self.prediction_cache_hits += 1
            return cached
        attempted_prediction = self.physical_predictions + 1
        if attempted_prediction > MAX_PHYSICAL_PREDICTIONS:
            raise _PhysicalWorkBoundExceeded("physical-predictions", attempted_prediction)
        reserved_races = 1 if production == "absolute" else (2 if production == "add" else 4)
        attempted_race = self.physical_races + reserved_races
        if attempted_race > MAX_PHYSICAL_RACES:
            raise _PhysicalWorkBoundExceeded("physical-races", attempted_race)
        self.physical_predictions = attempted_prediction
        self._record("prediction", _digest("prediction-key", key))
        if right is None:
            action = FactorAction((), "APPLY1", production, "SELECTED", "NONE", None)
            prediction = self.operator.predict(left, action)
        else:
            action = FactorAction((), "APPLY2", production, "SELECTED", "SCOPE0", None)
            prediction = self.operator.predict(left, action, scopes=(right,))
        actual_races = len(prediction.races)
        if actual_races != reserved_races:
            raise RuntimeError("physical operator race reservation drifted")
        self.physical_races += actual_races
        output = output_cells.TypedOutputCellBank.validate(prediction.output)
        self._prediction_cache[key] = output
        return output

    def _predict_vector(
        self,
        production: str,
        children: tuple[PackedENode, ...],
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if len(children) == 1:
            return tuple(
                self._predict_case(production, value) for value in children[0].physical_outputs
            )
        if len(children) == 2:
            return tuple(
                self._predict_case(production, left, right)
                for left, right in zip(
                    children[0].physical_outputs,
                    children[1].physical_outputs,
                    strict=True,
                )
            )
        raise ValueError("only unary and binary packed families are supported")

    @staticmethod
    def _semantic_ref(enode: PackedENode) -> str:
        return f"{enode.family_id}:{_eclass_id(enode.physical_outputs)}"

    def _partition_family(
        self,
        family: PackedFamily,
        children: tuple[PackedFamily, ...],
        child_slots: tuple[tuple[packed.TypedSlot, ...], ...],
        constraints: tuple[packed.SlotConstraint, ...] = (),
    ) -> None:
        self._charge("family-semantic-expansion")
        self._record("semantic-expand", family.family_id)
        child_enode_rows = tuple(tuple(self.families[child.family_id].enodes) for child in children)
        grouped: dict[
            str,
            tuple[
                tuple[output_cells.TypedOutputEncoding, ...],
                packed.PackedBindingRelation,
                list[tuple[str, ...]],
            ],
        ] = {}
        for child_nodes in product(*child_enode_rows):
            relation, _indices, _targets = self._compose_relation(
                children,
                tuple(node.proof_slice.relation for node in child_nodes),
            )
            if relation.slots != family.relation.slots:
                raise RuntimeError("family partition slot rebasing drifted")
            if constraints:
                relation = self.session.select(relation, constraints)
                if self.session.saturating_count(relation, 1) == 0:
                    continue
            outputs = self._predict_vector(family.production, child_nodes)
            output_sha = _vector_sha256(outputs)
            refs = tuple(self._semantic_ref(node) for node in child_nodes)
            incumbent = grouped.get(output_sha)
            if incumbent is None:
                grouped[output_sha] = (outputs, relation, [refs])
            else:
                incumbent_outputs, incumbent_relation, incumbent_refs = incumbent
                merged = self.session.union(incumbent_relation, relation)
                grouped[output_sha] = (
                    incumbent_outputs,
                    merged,
                    [*incumbent_refs, refs],
                )
        if not grouped:
            if self.session.saturating_count(family.relation, 1) != 0:
                raise RuntimeError("nonempty family has no semantic partition")
            return
        coverage: packed.PackedBindingRelation | None = None
        for output_sha in sorted(grouped):
            outputs, relation, refs = grouped[output_sha]
            coverage = relation if coverage is None else self.session.union(coverage, relation)
            self._store_enode(family, refs, relation, outputs)
        if coverage is None or coverage.sha256 != family.relation.sha256:
            raise RuntimeError("semantic e-node partitions do not cover family")

    def _terminal_families(self) -> tuple[PackedFamily, PackedFamily]:
        arguments: list[int] = []
        for value in self.supplied_case_arguments:
            self._charge("argument-value-read")
            if type(value) is not int:
                raise ValueError("case_arguments must contain exact integers")
            arguments.append(value)
        if not arguments:
            raise ValueError("case_arguments must contain exact integers")
        self.case_arguments = tuple(arguments)
        role_relation = self.session.domain(_ROLE_SLOT, (0,))
        literal_relation = self.session.domain(_LITERAL_SLOT, self.supplied_literals)
        literal_count = self.session.exact_count(literal_relation)
        if literal_count == 0:
            raise ValueError("literals must contain at least one exact integer")
        self.literals = tuple(
            binding[0] for binding in self.session.first_bindings(literal_relation, literal_count)
        )
        role_family = self._store_family("role", 1, (), role_relation, ())
        role_output_list: list[output_cells.TypedOutputEncoding] = []
        for value in self.case_arguments:
            self._charge("terminal-output")
            role_output_list.append(
                output_cells.TypedOutputCellBank.encode(
                    value, declared_type=output_cells.OutputType.INT
                )
            )
        role_outputs = tuple(role_output_list)
        self._store_enode(role_family, (), role_relation, role_outputs)

        literal_family = self._store_family("literal", 1, (), literal_relation, ())
        coverage: packed.PackedBindingRelation | None = None
        for literal in self.literals:
            relation = self.session.select(
                literal_relation,
                (packed.SlotConstraint.value(_LITERAL_SLOT, "eq", literal),),
            )
            literal_outputs: list[output_cells.TypedOutputEncoding] = []
            for _case in self.case_arguments:
                self._charge("terminal-output")
                literal_outputs.append(
                    output_cells.TypedOutputCellBank.encode(
                        literal, declared_type=output_cells.OutputType.INT
                    )
                )
            outputs = tuple(literal_outputs)
            self._store_enode(literal_family, (), relation, outputs)
            coverage = relation if coverage is None else self.session.union(coverage, relation)
        if coverage is None or coverage.sha256 != literal_relation.sha256:
            raise RuntimeError("literal semantic partitions do not cover domain")
        self.maximum_completed_ast_cost = 1
        return role_family, literal_family

    def _unary_family(self, child: PackedFamily) -> PackedFamily:
        relation, indices, child_slots = self._compose_relation((child,), (child.relation,))
        family = self._store_family("absolute", child.inside_cost + 1, (child,), relation, indices)
        self._partition_family(family, (child,), child_slots)
        self.maximum_completed_ast_cost = max(self.maximum_completed_ast_cost, family.inside_cost)
        return family

    def _binary_family(
        self,
        production: str,
        left: PackedFamily,
        right: PackedFamily,
        *,
        constraint_factory: object | None = None,
    ) -> PackedFamily:
        relation, indices, child_slots = self._compose_relation(
            (left, right), (left.relation, right.relation)
        )
        constraints: tuple[packed.SlotConstraint, ...] = ()
        if constraint_factory == "nonzero-right" and 0 in self.literals:
            right_literal = child_slots[1][0]
            raw_slice = self.session.family_slice(
                self._family_id(production, (left, right)), relation
            )
            normalized = self.session.normalize(
                raw_slice,
                packed.ProofNormalizationProjection(
                    normalization_id="add-right-zero-v1",
                    source_family_id=raw_slice.family_id,
                    target_family_id=left.family_id,
                    retained_source_slots=child_slots[0],
                    target_slots=left.relation.slots,
                    required_constants=((right_literal, 0),),
                ),
            )
            self.session.authenticate_slice(normalized)
            if normalized.relation.sha256 != left.relation.sha256:
                raise RuntimeError("x + 0 normalization did not recover its child")
            self.normalization_count += 1
            constraints = (packed.SlotConstraint.value(right_literal, "ne", 0),)
            relation = self.session.select(relation, constraints)
        elif constraint_factory == "ordered-bounds":
            lower = child_slots[0][0]
            upper = child_slots[1][0]
            constraints = (packed.SlotConstraint.slots(lower, "le", upper),)
            relation = self.session.select(relation, constraints)
        family = self._store_family(
            production,
            left.inside_cost + right.inside_cost + 1,
            (left, right),
            relation,
            indices,
        )
        self._partition_family(family, (left, right), child_slots, constraints)
        self.maximum_completed_ast_cost = max(self.maximum_completed_ast_cost, family.inside_cost)
        return family

    def _finalize_eclasses(self) -> tuple[PackedEClass, ...]:
        eclasses: list[PackedEClass] = []
        for eclass_id in sorted(self._eclass_enodes):
            enodes = tuple(sorted(self._eclass_enodes[eclass_id], key=lambda item: item.enode_id))
            outputs = self._eclass_outputs[eclass_id]
            proof_slices = tuple(
                sorted(
                    (enode.proof_slice for enode in enodes),
                    key=lambda item: item.family_id,
                )
            )
            payload = {
                "eclass_id": eclass_id,
                "minimum_inside_cost": min(
                    self.families[enode.family_id].family.inside_cost for enode in enodes
                ),
                "enode_ids": tuple(enode.enode_id for enode in enodes),
                "physical_outputs": _vector_record(outputs),
                "proof_slices": tuple(
                    (item.family_id, item.relation.sha256, item.sha256) for item in proof_slices
                ),
            }
            eclass = PackedEClass(
                eclass_id=eclass_id,
                minimum_inside_cost=payload["minimum_inside_cost"],
                enode_ids=payload["enode_ids"],
                physical_outputs=outputs,
                proof_slices=proof_slices,
                sha256=_digest("e-class", payload),
            )
            self._charge("e-class")
            eclasses.append(eclass)
        return tuple(eclasses)

    def build(self) -> tuple[PackedFrontierState, _FrontierSession]:
        try:
            role, literal = self._terminal_families()
            if self.max_ast_cost >= 2:
                self._unary_family(role)
            if self.max_ast_cost >= 3:
                self._binary_family("add", role, role)
                self._binary_family("add", role, literal, constraint_factory="nonzero-right")
                extremum = self._binary_family("maximum", role, literal)
            else:
                extremum = None
            if self.max_ast_cost >= 5 and extremum is not None:
                extremum = self._binary_family(
                    "minimum",
                    extremum,
                    literal,
                    constraint_factory="ordered-bounds",
                )
            for cost, production in ((7, "maximum"), (9, "minimum")):
                if self.max_ast_cost >= cost and extremum is not None:
                    extremum = self._binary_family(production, extremum, literal)
            eclasses = self._finalize_eclasses()
            families = tuple(
                sorted(
                    (item.family for item in self.families.values()),
                    key=lambda item: (item.inside_cost, item.family_id),
                )
            )
            enodes = tuple(sorted(self.enodes.values(), key=lambda item: item.enode_id))
            _authenticate_components(
                self.session,
                self.operator,
                families,
                enodes,
                eclasses,
            )
            self._charge("frontier-seal")
            build_work = self.work_audit()
            payload = {
                "status": FRONTIER_READY,
                "case_arguments": self.case_arguments,
                "literals": self.literals,
                "requested_max_ast_cost": self.max_ast_cost,
                "maximum_completed_ast_cost": self.maximum_completed_ast_cost,
                "operator_checkpoint_sha256": self.operator.checkpoint_sha256,
                "families": tuple(item.sha256 for item in families),
                "enodes": tuple(item.sha256 for item in enodes),
                "eclasses": tuple(item.sha256 for item in eclasses),
                "root_eclass_ids": tuple(item.eclass_id for item in eclasses),
                "build_work_sha256": build_work.sha256,
            }
            state = PackedFrontierState(
                status=FRONTIER_READY,
                case_arguments=self.case_arguments,
                literals=self.literals,
                requested_max_ast_cost=self.max_ast_cost,
                maximum_completed_ast_cost=self.maximum_completed_ast_cost,
                operator_checkpoint_sha256=self.operator.checkpoint_sha256,
                families=families,
                enodes=enodes,
                eclasses=eclasses,
                root_eclass_ids=tuple(item.eclass_id for item in eclasses),
                build_work_sha256=build_work.sha256,
                sha256=_digest("frontier", payload),
            )
            return state, self.session
        except packed.PackedRelationWorkLimitExceeded as error:
            self.bound_name = "recipe-relation-work"
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.session
        except _PhysicalWorkBoundExceeded as error:
            self.bound_name = error.name
            self.bound_attempted = error.attempted
            return self._stopped_state(), self.session

    def _stopped_state(self) -> PackedFrontierState:
        work = self.work_audit()
        payload = {
            "status": STOP_WORK_BOUND,
            "case_arguments": self.case_arguments,
            "literals": self.literals,
            "requested_max_ast_cost": self.max_ast_cost,
            "maximum_completed_ast_cost": self.maximum_completed_ast_cost,
            "operator_checkpoint_sha256": self.operator.checkpoint_sha256,
            "families": (),
            "enodes": (),
            "eclasses": (),
            "root_eclass_ids": (),
            "build_work_sha256": work.sha256,
        }
        return PackedFrontierState(
            status=STOP_WORK_BOUND,
            case_arguments=self.case_arguments,
            literals=self.literals,
            requested_max_ast_cost=self.max_ast_cost,
            maximum_completed_ast_cost=self.maximum_completed_ast_cost,
            operator_checkpoint_sha256=self.operator.checkpoint_sha256,
            families=(),
            enodes=(),
            eclasses=(),
            root_eclass_ids=(),
            build_work_sha256=work.sha256,
            sha256=_digest("frontier", payload),
        )

    def work_audit(self) -> PackedFrontierWorkAudit:
        return _work_audit(
            self.session.audit,
            self.physical_predictions,
            self.prediction_cache_hits,
            self.physical_races,
            0,
            self.bound_name,
            self.bound_attempted,
            tuple(self.agenda_trace),
        )


def _work_audit(
    relation: packed.PackedRelationWorkAudit,
    physical_predictions: int,
    prediction_cache_hits: int,
    physical_races: int,
    concrete_proof_witnesses: int,
    bound_name: str | None,
    bound_attempted: int | None,
    agenda_trace: tuple[tuple[str, str], ...],
) -> PackedFrontierWorkAudit:
    event_counts = dict(relation.event_counts)
    materialized = event_counts.get("frontier:materialize-witness", 0)
    payload = {
        "relation_event_sha256": relation.event_sha256,
        "relation_completed": relation.completed_units,
        "relation_attempted": relation.attempted_units,
        "physical_predictions": physical_predictions,
        "prediction_cache_hits": prediction_cache_hits,
        "physical_races": physical_races,
        "concrete_proof_witnesses": concrete_proof_witnesses,
        "materialized_witnesses": materialized,
        "bound_name": bound_name,
        "bound_attempted": bound_attempted,
        "agenda_trace": agenda_trace,
    }
    return PackedFrontierWorkAudit(
        relation=relation,
        physical_predictions=physical_predictions,
        prediction_cache_hits=prediction_cache_hits,
        physical_races=physical_races,
        concrete_proof_witnesses=concrete_proof_witnesses,
        materialized_witnesses=materialized,
        bound_name=bound_name,
        bound_attempted=bound_attempted,
        agenda_trace=agenda_trace,
        sha256=_digest("work", payload),
    )


def _authenticate_components(
    session: _FrontierSession,
    operator: operator_successor.FrozenTypedOperatorSuccessor,
    families: tuple[PackedFamily, ...],
    enodes: tuple[PackedENode, ...],
    eclasses: tuple[PackedEClass, ...],
) -> None:
    if operator.live_checkpoint_sha256 != operator.checkpoint_sha256:
        raise RuntimeError("packed frontier operator checkpoint changed")
    family_by_id: dict[str, PackedFamily] = {}
    for family in families:
        session.authenticate_relation(family.relation)
        if family.sha256 != _digest("family", _family_payload(family)):
            raise RuntimeError("packed family authentication failed")
        if family.family_id != _family_identity(family.production, family.child_family_ids):
            raise RuntimeError("packed family structural identity changed")
        if family.family_id in family_by_id:
            raise RuntimeError("packed frontier contains duplicate families")
        family_by_id[family.family_id] = family
    for family in families:
        try:
            children = tuple(family_by_id[child_id] for child_id in family.child_family_ids)
        except KeyError as error:
            raise RuntimeError("packed family references an unknown child") from error
        if len(children) != len(family.child_slot_indices):
            raise RuntimeError("packed family child slot mapping changed")
        mapped_indices: list[int] = []
        for child, indices in zip(children, family.child_slot_indices, strict=True):
            if len(indices) != len(child.relation.slots) or any(
                type(index) is not int or not 0 <= index < len(family.relation.slots)
                for index in indices
            ):
                raise RuntimeError("packed family child slot mapping changed")
            mapped_indices.extend(indices)
        if children and tuple(sorted(mapped_indices)) != tuple(range(len(family.relation.slots))):
            raise RuntimeError("packed family slot coverage changed")
        if family.production in {"role", "literal"}:
            expected_cost = 1
            if children:
                raise RuntimeError("packed terminal family gained a child")
        elif family.production == "absolute":
            if len(children) != 1:
                raise RuntimeError("packed unary family arity changed")
            expected_cost = children[0].inside_cost + 1
        elif family.production in {"add", "minimum", "maximum"}:
            if len(children) != 2:
                raise RuntimeError("packed binary family arity changed")
            expected_cost = children[0].inside_cost + children[1].inside_cost + 1
        else:
            raise RuntimeError("packed family production changed")
        if family.inside_cost != expected_cost:
            raise RuntimeError("packed family cost changed")
    enode_by_id: dict[str, PackedENode] = {}
    for enode in enodes:
        session.authenticate_slice(enode.proof_slice)
        if enode.family_id not in family_by_id:
            raise RuntimeError("packed e-node references an unknown family")
        if enode.proof_slice.family_id != enode.family_id:
            raise RuntimeError("packed e-node proof family changed")
        if enode.proof_slice.relation.slots != family_by_id[enode.family_id].relation.slots:
            raise RuntimeError("packed e-node proof schema changed")
        for value in enode.physical_outputs:
            output_cells.TypedOutputCellBank.validate(value)
        if enode.physical_output_sha256 != _vector_sha256(enode.physical_outputs):
            raise RuntimeError("packed e-node physical outputs changed")
        expected_enode_id = _digest(
            "e-node-id",
            {
                "family_id": enode.family_id,
                "child_semantic_refs": enode.child_semantic_refs,
                "relation_sha256": enode.proof_slice.relation.sha256,
                "physical_output_sha256": enode.physical_output_sha256,
            },
        )
        if enode.enode_id != expected_enode_id:
            raise RuntimeError("packed e-node identity changed")
        if enode.sha256 != _digest("e-node", _enode_payload(enode)):
            raise RuntimeError("packed e-node authentication failed")
        if enode.enode_id in enode_by_id:
            raise RuntimeError("packed frontier contains duplicate e-nodes")
        enode_by_id[enode.enode_id] = enode
    semantic_pairs = {(enode.family_id, _eclass_id(enode.physical_outputs)) for enode in enodes}
    for enode in enodes:
        child_ids = family_by_id[enode.family_id].child_family_ids
        if not child_ids and enode.child_semantic_refs:
            raise RuntimeError("packed terminal e-node gained child semantics")
        if child_ids and not enode.child_semantic_refs:
            raise RuntimeError("packed operator e-node lost child semantics")
        for references in enode.child_semantic_refs:
            if len(references) != len(child_ids):
                raise RuntimeError("packed e-node semantic arity changed")
            for reference, child_id in zip(references, child_ids, strict=True):
                prefix = f"{child_id}:"
                if not reference.startswith(prefix):
                    raise RuntimeError("packed e-node child family reference changed")
                if (child_id, reference.removeprefix(prefix)) not in semantic_pairs:
                    raise RuntimeError("packed e-node child e-class reference changed")
    for eclass in eclasses:
        if type(eclass.proof_slices) is not tuple:
            raise RuntimeError("packed e-class proof slices changed")
        for proof_slice in eclass.proof_slices:
            session.authenticate_slice(proof_slice)
        for value in eclass.physical_outputs:
            output_cells.TypedOutputCellBank.validate(value)
        if eclass.eclass_id != _eclass_id(eclass.physical_outputs):
            raise RuntimeError("packed e-class physical identity changed")
        selected = tuple(enode_by_id[item] for item in eclass.enode_ids)
        if any(
            enode.physical_output_sha256 != _vector_sha256(eclass.physical_outputs)
            for enode in selected
        ):
            raise RuntimeError("packed e-class contains a different semantic e-node")
        expected_slices = tuple(
            sorted(
                (
                    enode.proof_slice.family_id,
                    enode.proof_slice.relation.sha256,
                    enode.proof_slice.sha256,
                )
                for enode in selected
            )
        )
        actual_slices = tuple(
            (item.family_id, item.relation.sha256, item.sha256) for item in eclass.proof_slices
        )
        if expected_slices != actual_slices:
            raise RuntimeError("packed e-class proof mapping changed")
        if eclass.sha256 != _digest("e-class", _eclass_payload(eclass)):
            raise RuntimeError("packed e-class authentication failed")
        expected_minimum = min(family_by_id[enode.family_id].inside_cost for enode in selected)
        if eclass.minimum_inside_cost != expected_minimum:
            raise RuntimeError("packed e-class minimum cost changed")


class PackedScalarFrontier:
    """One sealed frontier plus its sole cumulative packed relation session."""

    __slots__ = (
        "_agenda_trace",
        "_bound_attempted",
        "_bound_name",
        "_concrete_proof_witnesses",
        "_operator",
        "_physical_predictions",
        "_physical_races",
        "_prediction_cache_hits",
        "_session",
        "_state",
    )

    def __init__(
        self,
        state: PackedFrontierState,
        session: _FrontierSession,
        operator: operator_successor.FrozenTypedOperatorSuccessor,
        builder: _Builder,
    ) -> None:
        self._state = state
        self._session = session
        self._operator = operator
        self._physical_predictions = builder.physical_predictions
        self._prediction_cache_hits = builder.prediction_cache_hits
        self._physical_races = builder.physical_races
        self._concrete_proof_witnesses = 0
        self._bound_name = builder.bound_name
        self._bound_attempted = builder.bound_attempted
        self._agenda_trace = tuple(builder.agenda_trace)

    @classmethod
    def build(
        cls,
        *,
        case_arguments: Iterable[int],
        literals: Iterable[int],
        max_ast_cost: int,
        operator: operator_successor.FrozenTypedOperatorSuccessor | None = None,
    ) -> PackedScalarFrontier:
        if isinstance(case_arguments, (str, bytes)):
            raise TypeError("case_arguments must be an iterable of exact integers")
        if isinstance(literals, (str, bytes)):
            raise TypeError("literals must be an iterable of exact integers")
        if type(max_ast_cost) is not int or not 1 <= max_ast_cost <= 9:
            raise ValueError("max_ast_cost must be in [1, 9]")
        frozen_operator = (
            operator_successor.train_public_typed_operator_successor()[0]
            if operator is None
            else operator
        )
        if not isinstance(frozen_operator, operator_successor.FrozenTypedOperatorSuccessor):
            raise TypeError("operator must be a FrozenTypedOperatorSuccessor")
        frozen_operator.clear_effect_memo()
        before = frozen_operator.live_checkpoint_sha256
        if before != frozen_operator.checkpoint_sha256:
            raise RuntimeError("typed operator checkpoint changed before frontier build")
        builder = _Builder(
            case_arguments,
            literals,
            max_ast_cost,
            frozen_operator,
        )
        state, session = builder.build()
        if frozen_operator.live_checkpoint_sha256 != before:
            raise RuntimeError("typed operator checkpoint changed during frontier build")
        return cls(state, session, frozen_operator, builder)

    @property
    def state(self) -> PackedFrontierState:
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def families(self) -> tuple[PackedFamily, ...]:
        return self._state.families

    @property
    def enodes(self) -> tuple[PackedENode, ...]:
        return self._state.enodes

    @property
    def eclasses(self) -> tuple[PackedEClass, ...]:
        return self._state.eclasses

    @property
    def work(self) -> PackedFrontierWorkAudit:
        return _work_audit(
            self._session.audit,
            self._physical_predictions,
            self._prediction_cache_hits,
            self._physical_races,
            self._concrete_proof_witnesses,
            self._bound_name,
            self._bound_attempted,
            self._agenda_trace,
        )

    def _state_payload(self) -> dict[str, object]:
        return {
            "status": self._state.status,
            "case_arguments": self._state.case_arguments,
            "literals": self._state.literals,
            "requested_max_ast_cost": self._state.requested_max_ast_cost,
            "maximum_completed_ast_cost": self._state.maximum_completed_ast_cost,
            "operator_checkpoint_sha256": self._state.operator_checkpoint_sha256,
            "families": tuple(item.sha256 for item in self._state.families),
            "enodes": tuple(item.sha256 for item in self._state.enodes),
            "eclasses": tuple(item.sha256 for item in self._state.eclasses),
            "root_eclass_ids": self._state.root_eclass_ids,
            "build_work_sha256": self._state.build_work_sha256,
        }

    def _authenticate(self) -> None:
        if self._state.status != FRONTIER_READY:
            raise RuntimeError("only a complete packed frontier can be authenticated")
        if self._operator.live_checkpoint_sha256 != self._state.operator_checkpoint_sha256:
            raise RuntimeError("packed frontier operator checkpoint changed")
        _authenticate_components(
            self._session,
            self._operator,
            self._state.families,
            self._state.enodes,
            self._state.eclasses,
        )
        if self._state.root_eclass_ids != tuple(item.eclass_id for item in self._state.eclasses):
            raise RuntimeError("packed frontier root e-class mapping changed")
        if self._state.sha256 != _digest("frontier", self._state_payload()):
            raise RuntimeError("packed frontier authentication failed")

    def _authenticate_stopped(self) -> None:
        if self._state.status != STOP_WORK_BOUND:
            raise RuntimeError("packed frontier has an unknown terminal status")
        if any(
            (
                self._state.families,
                self._state.enodes,
                self._state.eclasses,
                self._state.root_eclass_ids,
            )
        ):
            raise RuntimeError("stopped packed frontier exposed partial topology")
        if self._operator.live_checkpoint_sha256 != self._state.operator_checkpoint_sha256:
            raise RuntimeError("stopped packed frontier operator checkpoint changed")
        if self._state.build_work_sha256 != self.work.sha256:
            raise RuntimeError("stopped packed frontier work prefix changed")
        if self._state.sha256 != _digest("frontier", self._state_payload()):
            raise RuntimeError("stopped packed frontier authentication failed")

    def _materialize(
        self,
        witness: packed.ProofWitness,
    ) -> PackedWitness:
        self._session.charge_frontier("materialize-witness")
        families = {item.family_id: item for item in self._state.families}

        def render(family_id: str, binding: tuple[int, ...]) -> str:
            family = families[family_id]
            if len(binding) != len(family.relation.slots):
                raise RuntimeError("proof witness binding arity changed")
            if family.production == "role":
                return "x"
            if family.production == "literal":
                return str(binding[0])
            children = []
            for child_id, indices in zip(
                family.child_family_ids,
                family.child_slot_indices,
                strict=True,
            ):
                child_binding = tuple(binding[index] for index in indices)
                children.append(render(child_id, child_binding))
            if family.production == "absolute":
                return f"abs({children[0]})"
            symbols = {
                "add": "+",
                "minimum": "min",
                "maximum": "max",
            }
            if family.production == "add":
                return f"({children[0]} {symbols['add']} {children[1]})"
            return f"{symbols[family.production]}({children[0]}, {children[1]})"

        source = render(witness.family_id, witness.binding)
        payload = {
            "family_id": witness.family_id,
            "binding": witness.binding,
            "source": source,
        }
        return PackedWitness(
            family_id=witness.family_id,
            binding=witness.binding,
            source=source,
            sha256=_digest("witness", payload),
        )

    def _selection(
        self,
        status: str,
        *,
        selected_eclass_id: str | None = None,
        exact_proof_count: int = 0,
        witnesses: tuple[PackedWitness, ...] = (),
        selected_proof_forest_sha256: str | None = None,
        race: semantic_race.TypedSemanticVectorRaceAudit | None = None,
        checkpoint_before: str | None = None,
    ) -> PackedFrontierSelection:
        before = (
            self._operator.live_checkpoint_sha256
            if checkpoint_before is None
            else checkpoint_before
        )
        after = self._operator.live_checkpoint_sha256
        work = self.work
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "selected_eclass_id": selected_eclass_id,
            "exact_proof_count": exact_proof_count,
            "witnesses": tuple(item.sha256 for item in witnesses),
            "selected_proof_forest_sha256": selected_proof_forest_sha256,
            "race_sha256": None if race is None else race.sha256,
            "checkpoint_before_sha256": before,
            "checkpoint_after_sha256": after,
            "work_sha256": work.sha256,
        }
        return PackedFrontierSelection(
            status=status,
            frontier_sha256=self._state.sha256,
            selected_eclass_id=selected_eclass_id,
            exact_proof_count=exact_proof_count,
            witnesses=witnesses,
            selected_proof_forest_sha256=selected_proof_forest_sha256,
            race=race,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
            work=work,
            sha256=_digest("selection", payload),
        )

    def select(self, desired_outputs: Sequence[int]) -> PackedFrontierSelection:
        if self._state.status != FRONTIER_READY:
            self._authenticate_stopped()
            return self._selection(STOP_WORK_BOUND)
        try:
            self._authenticate()
            if isinstance(desired_outputs, (str, bytes)):
                raise TypeError("desired_outputs must be exact integers")
            desired_values = tuple(desired_outputs)
            if len(desired_values) != len(self._state.case_arguments) or any(
                type(value) is not int for value in desired_values
            ):
                raise ValueError("desired_outputs must align with frontier cases")
            desired = tuple(
                output_cells.TypedOutputCellBank.encode(
                    value, declared_type=output_cells.OutputType.INT
                )
                for value in desired_values
            )
            before = self._operator.live_checkpoint_sha256
            attempted_races = self._physical_races + 1
            if attempted_races > MAX_PHYSICAL_RACES:
                raise _PhysicalWorkBoundExceeded("physical-races", attempted_races)
            self._physical_races = attempted_races
            race = semantic_race.select_exact_semantic_vector(
                self._state.root_eclass_ids,
                tuple(item.physical_outputs for item in self._state.eclasses),
                desired,
            )
            if self._operator.live_checkpoint_sha256 != before:
                raise RuntimeError("operator checkpoint changed during selection")
            if race.status == semantic_race.STOP_NO_EXACT:
                return self._selection(STOP_NO_EXACT, race=race, checkpoint_before=before)
            if race.status != semantic_race.SELECTED or race.winner_key is None:
                raise RuntimeError("unique e-class bank returned a physical tie")
            eclass = next(
                item for item in self._state.eclasses if item.eclass_id == race.winner_key
            )
            selected_forest = self._session.proof_forest(eclass.proof_slices, saturation_limit=2)
            self._session.authenticate_forest(selected_forest)
            self._concrete_proof_witnesses = len(selected_forest.first_two_witnesses)
            proof_count = self._session.forest_exact_count(selected_forest)
            proof_witnesses = self._session.forest_first_witnesses(selected_forest)
            if proof_count == 1:
                witnesses = (self._materialize(proof_witnesses[0]),)
                return self._selection(
                    SELECTED,
                    selected_eclass_id=eclass.eclass_id,
                    exact_proof_count=proof_count,
                    witnesses=witnesses,
                    selected_proof_forest_sha256=selected_forest.sha256,
                    race=race,
                    checkpoint_before=before,
                )
            witnesses = tuple(self._materialize(item) for item in proof_witnesses[:2])
            return self._selection(
                STOP_AMBIGUOUS,
                selected_eclass_id=eclass.eclass_id,
                exact_proof_count=proof_count,
                witnesses=witnesses,
                selected_proof_forest_sha256=selected_forest.sha256,
                race=race,
                checkpoint_before=before,
            )
        except packed.PackedRelationWorkLimitExceeded as error:
            self._bound_name = "recipe-relation-work"
            self._bound_attempted = error.attempted
            return self._selection(STOP_WORK_BOUND)
        except _PhysicalWorkBoundExceeded as error:
            self._bound_name = error.name
            self._bound_attempted = error.attempted
            return self._selection(STOP_WORK_BOUND)


__all__ = [
    "FRONTIER_READY",
    "MAX_PHYSICAL_PREDICTIONS",
    "MAX_PHYSICAL_RACES",
    "MAX_RECIPE_RELATION_WORK",
    "SELECTED",
    "STOP_AMBIGUOUS",
    "STOP_NO_EXACT",
    "STOP_WORK_BOUND",
    "PackedEClass",
    "PackedENode",
    "PackedFamily",
    "PackedFrontierSelection",
    "PackedFrontierState",
    "PackedFrontierWorkAudit",
    "PackedScalarFrontier",
    "PackedWitness",
]
