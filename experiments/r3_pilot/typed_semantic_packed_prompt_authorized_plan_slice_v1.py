"""Prompt-authorized selected proof-plan slicing for generic packed breadth.

The banked generic V4 frontier remains the desired-free and proof-free builder.
This fresh successor adds one in-process seam between construction and
selection: an immutable, operator-name-free structural certificate is matched
against the authenticated retained proof plans.  A private authority issues one
``PlanSliceReceipt`` only when exactly one e-class/group/recipe-occurrence tree
matches.  The public receipt exposes hashes, generic factor topology, constrained
domains, and frozen work/root receipts -- never a program, source, or witness.

``select(desired)`` still runs the ordinary physical visible-I/O root race.  It
materializes the authorized occurrence tree only when that winner is the
receipt's e-class.  Zero or multiple structural matches stop before selection;
multiple concrete bindings remain explicit ambiguity.  Every authorization and
slice-publication operation uses the unchanged V4 relation-work ledger.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from itertools import product
from typing import NamedTuple, cast

from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from experiments.r3_pilot import typed_semantic_packed_bindings_v1 as packed
from experiments.r3_pilot import typed_semantic_packed_frontier_full_typed_v1 as full_typed
from experiments.r3_pilot import typed_semantic_packed_frontier_v2 as packed_v2
from experiments.r3_pilot import typed_semantic_packed_frontier_v3 as packed_v3
from experiments.r3_pilot import typed_semantic_packed_frontier_v4 as packed_v4
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v4 as breadth_v4
from humanbrain.typed_program_types import ProgramType, TypedSignature

FRONTIER_READY = "PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_READY"
STOP_WORK_BOUND = "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_WORK_BOUND"
STOP_NO_AUTHORIZED_SLICE = "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_NO_AUTHORIZED_SLICE"
STOP_AMBIGUOUS_AUTHORIZED_SLICE = (
    "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_AMBIGUOUS_AUTHORIZED_SLICE"
)
SELECTED = "PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_SELECTED"
STOP_NO_EXACT = "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_NO_EXACT"
STOP_AUTHORIZED_ECLASS_MISMATCH = (
    "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_AUTHORIZED_ECLASS_MISMATCH"
)
STOP_AMBIGUOUS = "STOP_PACKED_GENERIC_PROMPT_PLAN_SLICE_V1_AMBIGUOUS"

MAX_RECIPE_RELATION_WORK = breadth_v4.MAX_RECIPE_RELATION_WORK
MAX_PHYSICAL_PREDICTIONS = breadth_v4.MAX_PHYSICAL_PREDICTIONS
MAX_PHYSICAL_RACES = breadth_v4.MAX_PHYSICAL_RACES

_NAMESPACE = b"typed-semantic-packed-prompt-authorized-plan-slice-v1\0"
_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_CONSTRAINT_OPERATORS = frozenset(("eq", "ne", "lt", "le", "gt", "ge"))


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


class StructuralRelationConstraint(NamedTuple):
    """One generic relation constraint over local factor-slot ordinals."""

    left_slot: int
    operator: str
    right_slot: int | None = None
    right_value: int | None = None


class StructuralOccurrence(NamedTuple):
    """One occurrence in an ordered, operator-name-free factor tree."""

    parent_occurrence: int | None
    child_position: int | None
    factors: tuple[tuple[str, str], ...]
    parent_slot_occupancy: tuple[int, ...]
    relation_constraints: tuple[StructuralRelationConstraint, ...] = ()


class StructuralDomainConstraint(NamedTuple):
    """An allowed value/class domain for one occurrence-local factor slot."""

    occurrence: int
    slot: int
    allowed_values: tuple[int, ...]


class StructuralFactorCertificate(NamedTuple):
    """Frozen prompt-side structure with no family or executable identity."""

    occurrences: tuple[StructuralOccurrence, ...]
    domains: tuple[StructuralDomainConstraint, ...]
    literal_slot_copy_groups: tuple[tuple[tuple[int, int], ...], ...]
    sha256: str


def _constraint_record(value: StructuralRelationConstraint) -> tuple[object, ...]:
    return (
        value.left_slot,
        value.operator,
        value.right_slot,
        value.right_value,
    )


def _occurrence_record(value: StructuralOccurrence) -> dict[str, object]:
    return {
        "parent_occurrence": value.parent_occurrence,
        "child_position": value.child_position,
        "factors": value.factors,
        "parent_slot_occupancy": value.parent_slot_occupancy,
        "relation_constraints": tuple(
            _constraint_record(item) for item in value.relation_constraints
        ),
    }


def _certificate_payload(value: StructuralFactorCertificate) -> dict[str, object]:
    return {
        "occurrences": tuple(_occurrence_record(item) for item in value.occurrences),
        "domains": tuple(
            (item.occurrence, item.slot, item.allowed_values) for item in value.domains
        ),
        "literal_slot_copy_groups": value.literal_slot_copy_groups,
    }


def freeze_structural_factor_certificate(
    occurrences: Iterable[StructuralOccurrence],
    domains: Iterable[StructuralDomainConstraint] = (),
    literal_slot_copy_groups: Iterable[Iterable[tuple[int, int]]] = (),
) -> StructuralFactorCertificate:
    """Validate and freeze a generic occurrence/factor certificate."""

    frozen_occurrences = tuple(occurrences)
    if not frozen_occurrences or any(
        type(item) is not StructuralOccurrence for item in frozen_occurrences
    ):
        raise ValueError("a structural certificate needs exact occurrence records")
    if (
        frozen_occurrences[0].parent_occurrence is not None
        or frozen_occurrences[0].child_position is not None
        or frozen_occurrences[0].parent_slot_occupancy
    ):
        raise ValueError("occurrence zero must be the root")

    children: dict[int, list[tuple[int, int]]] = {
        index: [] for index in range(len(frozen_occurrences))
    }
    for index, occurrence in enumerate(frozen_occurrences):
        if not occurrence.factors or any(
            type(factor) is not tuple
            or len(factor) != 2
            or factor[0] not in {"literal", "role"}
            or type(factor[1]) is not str
            or not factor[1]
            for factor in occurrence.factors
        ):
            raise ValueError("occurrence factors must be generic kind/type pairs")
        if index:
            parent = occurrence.parent_occurrence
            position = occurrence.child_position
            if (
                type(parent) is not int
                or not 0 <= parent < index
                or type(position) is not int
                or position < 0
                or len(occurrence.parent_slot_occupancy) != len(occurrence.factors)
                or len(set(occurrence.parent_slot_occupancy))
                != len(occurrence.parent_slot_occupancy)
                or any(
                    type(slot) is not int or not 0 <= slot < len(frozen_occurrences[parent].factors)
                    for slot in occurrence.parent_slot_occupancy
                )
            ):
                raise ValueError("occurrence parent/occupancy topology is invalid")
            children[parent].append((position, index))
        for constraint in occurrence.relation_constraints:
            if (
                type(constraint) is not StructuralRelationConstraint
                or type(constraint.left_slot) is not int
                or not 0 <= constraint.left_slot < len(occurrence.factors)
                or constraint.operator not in _CONSTRAINT_OPERATORS
                or (constraint.right_slot is None) == (constraint.right_value is None)
                or (
                    constraint.right_slot is not None
                    and (
                        type(constraint.right_slot) is not int
                        or not 0 <= constraint.right_slot < len(occurrence.factors)
                    )
                )
                or (constraint.right_value is not None and type(constraint.right_value) is not int)
            ):
                raise ValueError("occurrence relation constraint is invalid")
    for entries in children.values():
        if entries and tuple(position for position, _index in sorted(entries)) != tuple(
            range(len(entries))
        ):
            raise ValueError("occurrence child positions must be contiguous")

    frozen_domains = tuple(domains)
    domain_keys: set[tuple[int, int]] = set()
    normalized_domains: list[StructuralDomainConstraint] = []
    for domain in frozen_domains:
        if (
            type(domain) is not StructuralDomainConstraint
            or type(domain.occurrence) is not int
            or not 0 <= domain.occurrence < len(frozen_occurrences)
            or type(domain.slot) is not int
            or not 0 <= domain.slot < len(frozen_occurrences[domain.occurrence].factors)
            or not domain.allowed_values
            or any(type(value) is not int for value in domain.allowed_values)
        ):
            raise ValueError("structural domain constraint is invalid")
        key = (domain.occurrence, domain.slot)
        if key in domain_keys:
            raise ValueError("structural domain slot repeated")
        domain_keys.add(key)
        normalized_domains.append(
            StructuralDomainConstraint(
                domain.occurrence,
                domain.slot,
                tuple(sorted(set(domain.allowed_values))),
            )
        )
    normalized_domains.sort(key=lambda item: (item.occurrence, item.slot))

    literal_refs = {
        (index, slot)
        for index, occurrence in enumerate(frozen_occurrences)
        if not children[index]
        for slot, factor in enumerate(occurrence.factors)
        if factor[0] == "literal"
    }
    normalized_groups: list[tuple[tuple[int, int], ...]] = []
    seen_refs: set[tuple[int, int]] = set()
    for supplied_group in literal_slot_copy_groups:
        group = tuple(sorted(tuple(item) for item in supplied_group))
        if (
            not group
            or len(set(group)) != len(group)
            or any(ref not in literal_refs for ref in group)
        ):
            raise ValueError("literal copy group is invalid")
        if seen_refs & set(group):
            raise ValueError("literal copy reference repeated")
        seen_refs.update(group)
        normalized_groups.append(group)
    if seen_refs != literal_refs:
        raise ValueError("literal copy groups must cover every terminal literal slot")
    normalized_copy_groups = tuple(sorted(normalized_groups))
    partial = StructuralFactorCertificate(
        occurrences=frozen_occurrences,
        domains=tuple(normalized_domains),
        literal_slot_copy_groups=normalized_copy_groups,
        sha256="",
    )
    return partial._replace(
        sha256=_digest("structural-factor-certificate", _certificate_payload(partial))
    )


def _authenticate_certificate(value: StructuralFactorCertificate) -> None:
    if type(value) is not StructuralFactorCertificate or value.sha256 != _digest(
        "structural-factor-certificate", _certificate_payload(value)
    ):
        raise RuntimeError("structural factor certificate authentication failed")


class AuthenticatedPlanSlice(NamedTuple):
    """Public non-executable description of one authorized retained slice."""

    family_topology_sha256: str
    recipe_topology_sha256: str
    constrained_domains: tuple[tuple[int, str, str, tuple[int, ...]], ...]
    root_receipt_sha256: str
    sha256: str


class PlanSliceReceipt(NamedTuple):
    """Identity-issued authority to publish one retained recipe slice."""

    authenticated_eclass_id: str
    plan_slices: tuple[AuthenticatedPlanSlice, ...]
    structural_certificate_sha256: str
    frontier_root_receipt_sha256: str
    build_work_receipt_sha256: str
    authorization_work_receipt_sha256: str
    sha256: str


class PackedPromptPlanSliceState(NamedTuple):
    status: str
    packed_frontier_sha256: str
    structural_certificate_sha256: str
    authorized_slice_count: int
    plan_slice_receipt_sha256: str | None
    frontier_root_receipt_sha256: str
    build_work_receipt_sha256: str
    authorization_work_receipt_sha256: str
    sha256: str


class PackedPromptPlanSliceSelection(NamedTuple):
    status: str
    frontier_sha256: str
    plan_slice_receipt_sha256: str | None
    selected_eclass_id: str | None
    exact_proof_count: int
    witnesses: tuple[breadth_v3.PackedGenericTypedWitness, ...]
    selected_proof_forest_sha256: str | None
    race: semantic_race.TypedSemanticVectorRaceAudit | None
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    work: full_typed.PackedFullTypedWorkAudit
    sha256: str


class _SliceNodeBinding(NamedTuple):
    occurrence: int
    root_id: str
    schema: tuple[packed.TypedSlot, ...]
    recipe: packed_v4._TupleProofRecipe | None
    children: tuple[_SliceNodeBinding, ...]
    topology_sha256: str


class _SliceCandidate(NamedTuple):
    eclass: packed_v3.PackedEClass
    group: packed_v3.PackedSemanticGroupRef
    family: packed_v3.PackedFamily
    node: _SliceNodeBinding
    constrained_domains: tuple[tuple[int, tuple[int, ...]], ...]
    topology_sha256: str


class _IssuedPlanBinding(NamedTuple):
    frontier: breadth_v4.PackedGenericTypedTrainBreadthV4
    route: breadth_v4._PackedListIntFrontier
    owner_token: object
    frozen: packed_v4.PackedFrozenWorkset
    authenticated: packed_v4._AuthenticatedFrozen
    component: packed_v3._AuthenticatedComponents
    candidate: _SliceCandidate
    certificate: StructuralFactorCertificate
    receipt: PlanSliceReceipt


def _plan_slice_payload(value: AuthenticatedPlanSlice) -> dict[str, object]:
    return {
        "family_topology_sha256": value.family_topology_sha256,
        "recipe_topology_sha256": value.recipe_topology_sha256,
        "constrained_domains": value.constrained_domains,
        "root_receipt_sha256": value.root_receipt_sha256,
    }


def _receipt_payload(value: PlanSliceReceipt) -> dict[str, object]:
    return {
        "authenticated_eclass_id": value.authenticated_eclass_id,
        "plan_slices": tuple(item.sha256 for item in value.plan_slices),
        "structural_certificate_sha256": value.structural_certificate_sha256,
        "frontier_root_receipt_sha256": value.frontier_root_receipt_sha256,
        "build_work_receipt_sha256": value.build_work_receipt_sha256,
        "authorization_work_receipt_sha256": value.authorization_work_receipt_sha256,
    }


def _make_plan_slice_authority() -> tuple[object, object]:
    issued: dict[int, _IssuedPlanBinding] = {}

    def issue(
        *,
        frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
        route: breadth_v4._PackedListIntFrontier,
        frozen: packed_v4.PackedFrozenWorkset,
        authenticated: packed_v4._AuthenticatedFrozen,
        component: packed_v3._AuthenticatedComponents,
        candidate: _SliceCandidate,
        certificate: StructuralFactorCertificate,
        root_receipt_sha256: str,
        build_work_receipt_sha256: str,
    ) -> PlanSliceReceipt:
        route._workset.session.charge_workset("authorized-plan-slice-receipt-issue")
        authorization_work = route.work.sha256
        root_schema = candidate.group.schema
        public_domains = tuple(
            (
                ordinal,
                root_schema[ordinal].kind,
                root_schema[ordinal].value_type,
                values,
            )
            for ordinal, values in candidate.constrained_domains
        )
        partial_slice = AuthenticatedPlanSlice(
            family_topology_sha256=_digest(
                "authenticated-family-topology",
                packed_v3._family_payload(candidate.family),
            ),
            recipe_topology_sha256=candidate.node.topology_sha256,
            constrained_domains=public_domains,
            root_receipt_sha256=_digest(
                "authorized-slice-root",
                {
                    "group_sha256": candidate.group.sha256,
                    "root_id": candidate.group.root_id,
                    "topology_sha256": candidate.node.topology_sha256,
                },
            ),
            sha256="",
        )
        plan_slice = partial_slice._replace(
            sha256=_digest("authenticated-plan-slice", _plan_slice_payload(partial_slice))
        )
        partial = PlanSliceReceipt(
            authenticated_eclass_id=candidate.eclass.eclass_id,
            plan_slices=(plan_slice,),
            structural_certificate_sha256=certificate.sha256,
            frontier_root_receipt_sha256=root_receipt_sha256,
            build_work_receipt_sha256=build_work_receipt_sha256,
            authorization_work_receipt_sha256=authorization_work,
            sha256="",
        )
        receipt = partial._replace(sha256=_digest("plan-slice-receipt", _receipt_payload(partial)))
        issued[id(receipt)] = _IssuedPlanBinding(
            frontier=frontier,
            route=route,
            owner_token=route._workset._owner_token,
            frozen=frozen,
            authenticated=authenticated,
            component=component,
            candidate=candidate,
            certificate=certificate,
            receipt=receipt,
        )
        return receipt

    def authenticate(
        frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
        receipt: PlanSliceReceipt,
    ) -> _IssuedPlanBinding:
        binding = issued.get(id(receipt))
        if (
            type(receipt) is not PlanSliceReceipt
            or type(binding) is not _IssuedPlanBinding
            or binding.receipt is not receipt
            or binding.frontier is not frontier
            or binding.route is not frontier._route
            or binding.owner_token is not binding.route._workset._owner_token
            or binding.frozen is not binding.route.state.frozen_workset
            or binding.authenticated is not binding.route._workset._authenticated
            or binding.component is not binding.route._authenticated_components
            or binding.certificate.sha256 != receipt.structural_certificate_sha256
            or receipt.sha256 != _digest("plan-slice-receipt", _receipt_payload(receipt))
        ):
            raise RuntimeError("plan slice receipt authentication failed")
        return binding

    return issue, authenticate


_issue_plan_slice_receipt, _authenticate_plan_slice_receipt = _make_plan_slice_authority()


def _children_by_occurrence(
    certificate: StructuralFactorCertificate,
) -> dict[int, tuple[int, ...]]:
    mutable: dict[int, list[tuple[int, int]]] = {
        index: [] for index in range(len(certificate.occurrences))
    }
    for index, occurrence in enumerate(certificate.occurrences[1:], start=1):
        mutable[cast(int, occurrence.parent_occurrence)].append(
            (cast(int, occurrence.child_position), index)
        )
    return {
        parent: tuple(index for _position, index in sorted(entries))
        for parent, entries in mutable.items()
    }


def _domains_by_occurrence(
    certificate: StructuralFactorCertificate,
) -> dict[int, tuple[StructuralDomainConstraint, ...]]:
    mutable: dict[int, list[StructuralDomainConstraint]] = {}
    for domain in certificate.domains:
        mutable.setdefault(domain.occurrence, []).append(domain)
    return {key: tuple(value) for key, value in mutable.items()}


def _factor_record(schema: Sequence[packed.TypedSlot]) -> tuple[tuple[str, str], ...]:
    return tuple((slot.kind, slot.value_type) for slot in schema)


def _normalized_recipe_constraints(
    recipe: packed_v4._TupleProofRecipe,
) -> tuple[StructuralRelationConstraint, ...]:
    schema = recipe.schema
    return tuple(
        StructuralRelationConstraint(
            left_slot=schema.index(item.left),
            operator=item.operator,
            right_slot=(None if item.right_slot is None else schema.index(item.right_slot)),
            right_value=item.right_value,
        )
        for item in recipe.constraints
    )


def _terminal_domains(
    workset: breadth_v4._FactorizedSemanticPartitionWorkset,
    authenticated: packed_v4._AuthenticatedFrozen,
    root_id: str,
    schema: tuple[packed.TypedSlot, ...],
) -> tuple[tuple[int, ...], ...]:
    domains = [set[int]() for _slot in schema]
    seen: set[tuple[str, int]] = set()

    def visit(node_id: str, level: int) -> None:
        if node_id == packed._TERMINAL_ID:
            if level != len(schema):
                raise RuntimeError("terminal plan root ended before its schema")
            return
        if not 0 <= level < len(schema):
            raise RuntimeError("terminal plan root exceeded its schema")
        marker = (node_id, level)
        if marker in seen:
            return
        seen.add(marker)
        workset.session.charge_workset("authorized-slice-terminal-node")
        node = authenticated.arena.node(node_id)
        if (
            node.node_id != node_id
            or node.node_id != packed._node_id(node.slot, node.edges)
            or node.slot != schema[level]
        ):
            raise RuntimeError("authorized terminal topology changed")
        for value, child in node.edges:
            workset.session.charge_workset("authorized-slice-terminal-edge")
            domains[level].add(value)
            visit(child, level + 1)

    visit(root_id, 0)
    if any(not values for values in domains):
        raise RuntimeError("authorized terminal topology has an empty domain")
    return tuple(tuple(sorted(values)) for values in domains)


def _node_payload(value: _SliceNodeBinding) -> dict[str, object]:
    return {
        "occurrence": value.occurrence,
        "root_id": value.root_id,
        "schema": tuple(slot.to_record() for slot in value.schema),
        "recipe": (None if value.recipe is None else packed_v4._recipe_payload(value.recipe)),
        "children": tuple(item.topology_sha256 for item in value.children),
    }


def _match_node(
    *,
    workset: breadth_v4._FactorizedSemanticPartitionWorkset,
    authenticated: packed_v4._AuthenticatedFrozen,
    certificate: StructuralFactorCertificate,
    children_index: Mapping[int, tuple[int, ...]],
    domain_index: Mapping[int, tuple[StructuralDomainConstraint, ...]],
    root_id: str,
    schema: tuple[packed.TypedSlot, ...],
    occurrence_index: int,
    memo: dict[tuple[str, tuple[tuple[object, ...], ...], int], tuple[_SliceNodeBinding, ...]],
) -> tuple[_SliceNodeBinding, ...]:
    key = (root_id, tuple(slot.to_record() for slot in schema), occurrence_index)
    workset.session.charge_workset("authorized-slice-match-lookup")
    cached = memo.get(key)
    if cached is not None:
        workset.session.charge_workset("authorized-slice-match-cache-hit")
        return cached
    occurrence = certificate.occurrences[occurrence_index]
    if _factor_record(schema) != occurrence.factors:
        memo[key] = ()
        return ()
    node = authenticated.plans.get(root_id)
    child_occurrences = children_index[occurrence_index]
    if node is None:
        if child_occurrences or occurrence.relation_constraints:
            memo[key] = ()
            return ()
        candidate_domains = _terminal_domains(workset, authenticated, root_id, schema)
        if any(
            not set(candidate_domains[item.slot]) & set(item.allowed_values)
            for item in domain_index.get(occurrence_index, ())
        ):
            memo[key] = ()
            return ()
        partial = _SliceNodeBinding(
            occurrence=occurrence_index,
            root_id=root_id,
            schema=schema,
            recipe=None,
            children=(),
            topology_sha256="",
        )
        result = partial._replace(
            topology_sha256=_digest("authorized-slice-node", _node_payload(partial))
        )
        memo[key] = (result,)
        return (result,)

    workset.session.charge_workset("authorized-slice-plan-node")
    workset._validate_plan(node, root_id)
    if (
        node.kind != "semantic-group"
        or node.schema != schema
        or not child_occurrences
        or node.child_plan_ids
        or not node.recipes
        or any(
            not set(node.domains[item.slot]) & set(item.allowed_values)
            for item in domain_index.get(occurrence_index, ())
        )
    ):
        memo[key] = ()
        return ()

    matches: dict[str, _SliceNodeBinding] = {}
    for recipe in node.recipes:
        workset.session.charge_workset("authorized-slice-recipe-candidate")
        if (
            recipe.schema != schema
            or len(recipe.child_root_ids) != len(child_occurrences)
            or len(recipe.child_schemas) != len(child_occurrences)
            or len(recipe.child_target_slots) != len(child_occurrences)
            or _normalized_recipe_constraints(recipe) != occurrence.relation_constraints
        ):
            continue
        child_options: list[tuple[_SliceNodeBinding, ...]] = []
        valid = True
        for child_position, child_occurrence in enumerate(child_occurrences):
            workset.session.charge_workset("authorized-slice-recipe-edge")
            child_schema = recipe.child_schemas[child_position]
            targets = recipe.child_target_slots[child_position]
            occupancy = tuple(schema.index(slot) for slot in targets)
            if occupancy != certificate.occurrences[child_occurrence].parent_slot_occupancy:
                valid = False
                break
            options = _match_node(
                workset=workset,
                authenticated=authenticated,
                certificate=certificate,
                children_index=children_index,
                domain_index=domain_index,
                root_id=recipe.child_root_ids[child_position],
                schema=child_schema,
                occurrence_index=child_occurrence,
                memo=memo,
            )
            if not options:
                valid = False
                break
            child_options.append(options)
        if not valid:
            continue
        for selected_children in product(*child_options):
            partial = _SliceNodeBinding(
                occurrence=occurrence_index,
                root_id=root_id,
                schema=schema,
                recipe=recipe,
                children=tuple(selected_children),
                topology_sha256="",
            )
            binding = partial._replace(
                topology_sha256=_digest("authorized-slice-node", _node_payload(partial))
            )
            matches.setdefault(binding.topology_sha256, binding)
            if len(matches) >= 2:
                result = tuple(matches.values())
                memo[key] = result
                return result
    result = tuple(matches.values())
    memo[key] = result
    return result


def _root_slot_maps(
    root: _SliceNodeBinding,
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}

    def visit(node: _SliceNodeBinding, local_to_root: tuple[int, ...]) -> None:
        for slot, root_slot in enumerate(local_to_root):
            result[(node.occurrence, slot)] = root_slot
        if node.recipe is None:
            return
        for child, targets in zip(
            node.children,
            node.recipe.child_target_slots,
            strict=True,
        ):
            occupancy = tuple(node.schema.index(slot) for slot in targets)
            visit(child, tuple(local_to_root[index] for index in occupancy))

    visit(root, tuple(range(len(root.schema))))
    return result


def _literal_copy_groups(
    certificate: StructuralFactorCertificate,
    root_slot_maps: Mapping[tuple[int, int], int],
    children_index: Mapping[int, tuple[int, ...]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for occurrence_index, occurrence in enumerate(certificate.occurrences):
        if children_index[occurrence_index]:
            continue
        for slot, factor in enumerate(occurrence.factors):
            if factor[0] == "literal":
                groups.setdefault(root_slot_maps[(occurrence_index, slot)], []).append(
                    (occurrence_index, slot)
                )
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def _constrained_root_domains(
    certificate: StructuralFactorCertificate,
    node: _SliceNodeBinding,
    root_slot_maps: Mapping[tuple[int, int], int],
    workset: breadth_v4._FactorizedSemanticPartitionWorkset,
    authenticated: packed_v4._AuthenticatedFrozen,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    root_plan = authenticated.plans.get(node.root_id)
    root_domains = (
        _terminal_domains(workset, authenticated, node.root_id, node.schema)
        if root_plan is None
        else root_plan.domains
    )
    allowed: dict[int, set[int]] = {}
    for constraint in certificate.domains:
        root_slot = root_slot_maps[(constraint.occurrence, constraint.slot)]
        values = set(constraint.allowed_values) & set(root_domains[root_slot])
        if root_slot in allowed:
            values &= allowed[root_slot]
        if not values:
            raise RuntimeError("authorized domain constraints became contradictory")
        allowed[root_slot] = values
    return tuple((slot, tuple(sorted(values))) for slot, values in sorted(allowed.items()))


def _bind_certificate(
    frontier: breadth_v4.PackedGenericTypedTrainBreadthV4,
    certificate: StructuralFactorCertificate,
) -> tuple[str, int, PlanSliceReceipt | None, str, str, str]:
    _authenticate_certificate(certificate)
    frontier._authenticate()
    route = frontier._route
    if type(route) is not breadth_v4._PackedListIntFrontier:
        raise RuntimeError("plan slicing requires the exact generic V4 cross-type route")
    frozen = route.state.frozen_workset
    authenticated = route._workset._authenticated
    component = route._authenticated_components
    typed_capsule = route._typed_capsule
    if (
        type(frozen) is not packed_v4.PackedFrozenWorkset
        or type(authenticated) is not packed_v4._AuthenticatedFrozen
        or type(component) is not packed_v3._AuthenticatedComponents
        or type(typed_capsule) is not full_typed._AuthenticatedTypedCapsule
        or authenticated is not route._workset._authenticated
        or component is not route._workset._component_capsule
    ):
        raise RuntimeError("generic V4 retained capsules are unavailable")
    workset = cast(breadth_v4._FactorizedSemanticPartitionWorkset, route._workset)
    session = workset.session
    session.charge_workset("authorized-slice-certificate")
    for _occurrence in certificate.occurrences:
        session.charge_workset("authorized-slice-certificate-occurrence")
    for _domain in certificate.domains:
        session.charge_workset("authorized-slice-certificate-domain")
    for group in certificate.literal_slot_copy_groups:
        for _reference in group:
            session.charge_workset("authorized-slice-certificate-copy")

    children_index = _children_by_occurrence(certificate)
    domain_index = _domains_by_occurrence(certificate)
    memo: dict[
        tuple[str, tuple[tuple[object, ...], ...], int],
        tuple[_SliceNodeBinding, ...],
    ] = {}
    candidates: dict[str, _SliceCandidate] = {}
    for eclass in typed_capsule.root_eclasses:
        session.charge_workset("authorized-slice-eclass-candidate")
        full_eclass = component.eclass_index.get(eclass.eclass_id)
        if (
            type(eclass) is not packed_v3.PackedEClass
            or type(full_eclass) is not packed_v3.PackedEClass
            or full_eclass.physical_outputs != eclass.physical_outputs
            or not set(eclass.group_refs) <= set(full_eclass.group_refs)
        ):
            raise RuntimeError("authorized root e-class identity changed")
        for group in eclass.group_refs:
            session.charge_workset("authorized-slice-group-candidate")
            if authenticated.group_index.get(group.group_id) is not group:
                raise RuntimeError("authorized semantic group identity changed")
            family = component.family_index.get(group.family_id)
            if type(family) is not packed_v3.PackedFamily:
                raise RuntimeError("authorized family topology changed")
            matches = _match_node(
                workset=workset,
                authenticated=authenticated,
                certificate=certificate,
                children_index=children_index,
                domain_index=domain_index,
                root_id=group.root_id,
                schema=group.schema,
                occurrence_index=0,
                memo=memo,
            )
            for node in matches:
                root_maps = _root_slot_maps(node)
                if (
                    _literal_copy_groups(certificate, root_maps, children_index)
                    != certificate.literal_slot_copy_groups
                ):
                    continue
                constrained = _constrained_root_domains(
                    certificate,
                    node,
                    root_maps,
                    workset,
                    authenticated,
                )
                topology_sha256 = _digest(
                    "authorized-candidate-topology",
                    {
                        "eclass_id": eclass.eclass_id,
                        "group_sha256": group.sha256,
                        "family_sha256": family.sha256,
                        "node_sha256": node.topology_sha256,
                        "constrained_domains": constrained,
                    },
                )
                candidates.setdefault(
                    topology_sha256,
                    _SliceCandidate(
                        eclass=eclass,
                        group=group,
                        family=family,
                        node=node,
                        constrained_domains=constrained,
                        topology_sha256=topology_sha256,
                    ),
                )
                if len(candidates) >= 2:
                    break
            if len(candidates) >= 2:
                break
        if len(candidates) >= 2:
            break

    root_receipt = _digest(
        "frontier-root-receipt",
        {
            "frontier_sha256": frontier.sha256,
            "frozen_workset_sha256": frozen.sha256,
            "root_eclass_ids": route.state.root_eclass_ids,
        },
    )
    build_work = route.state.build_work_sha256
    if not candidates:
        return (
            STOP_NO_AUTHORIZED_SLICE,
            0,
            None,
            root_receipt,
            build_work,
            route.work.sha256,
        )
    if len(candidates) > 1:
        return (
            STOP_AMBIGUOUS_AUTHORIZED_SLICE,
            2,
            None,
            root_receipt,
            build_work,
            route.work.sha256,
        )
    candidate = next(iter(candidates.values()))
    receipt = _issue_plan_slice_receipt(
        frontier=frontier,
        route=route,
        frozen=frozen,
        authenticated=authenticated,
        component=component,
        candidate=candidate,
        certificate=certificate,
        root_receipt_sha256=root_receipt,
        build_work_receipt_sha256=build_work,
    )
    return (
        FRONTIER_READY,
        1,
        receipt,
        root_receipt,
        build_work,
        receipt.authorization_work_receipt_sha256,
    )


def _materialize_authorized_forest(
    binding: _IssuedPlanBinding,
) -> packed.PackedProofForest:
    route = binding.route
    workset = cast(breadth_v4._FactorizedSemanticPartitionWorkset, route._workset)
    session = workset.session
    session.activate_selected_compaction()
    workset._operation("materialize-authorized-plan-slice")
    if (
        binding.authenticated is not workset._authenticated
        or binding.authenticated.owner_token is not workset._owner_token
        or binding.frozen is not route.state.frozen_workset
        or binding.candidate.group not in binding.candidate.eclass.group_refs
        or binding.authenticated.group_index.get(binding.candidate.group.group_id)
        is not binding.candidate.group
    ):
        raise RuntimeError("authorized plan slice crossed its retained frontier")
    memo: dict[str, packed.PackedBindingRelation] = {}

    def materialize(node: _SliceNodeBinding) -> packed.PackedBindingRelation:
        cached = memo.get(node.topology_sha256)
        if cached is not None:
            session.charge_workset("authorized-slice-materialize-cache-hit")
            return cached
        if node.topology_sha256 != _digest("authorized-slice-node", _node_payload(node)):
            raise RuntimeError("authorized plan slice topology changed")
        retained = binding.authenticated.plans.get(node.root_id)
        if node.recipe is None:
            if retained is not None or node.children:
                raise RuntimeError("authorized terminal slice changed")
            relation, _exact, _bindings = workset._materialize_root_once(
                node.schema,
                node.root_id,
                binding.authenticated.arena,
            )
            memo[node.topology_sha256] = relation
            return relation
        session.charge_workset("authorized-slice-materialize-node")
        workset._validate_plan(cast(packed_v4._AggregateProofPlan, retained), node.root_id)
        if (
            retained is None
            or retained.kind != "semantic-group"
            or retained.schema != node.schema
            or not any(item is node.recipe for item in retained.recipes)
            or len(node.children) != len(node.recipe.child_root_ids)
        ):
            raise RuntimeError("authorized retained recipe changed")
        renamed: list[packed.PackedBindingRelation] = []
        for child, expected_root, expected_schema, targets in zip(
            node.children,
            node.recipe.child_root_ids,
            node.recipe.child_schemas,
            node.recipe.child_target_slots,
            strict=True,
        ):
            session.charge_workset("authorized-slice-materialize-edge")
            if child.root_id != expected_root or child.schema != expected_schema:
                raise RuntimeError("authorized recipe child changed")
            renamed.append(session.rename(materialize(child), targets))
        if not renamed:
            raise RuntimeError("authorized recipe lost its children")
        relation = renamed[0]
        for child_relation in renamed[1:]:
            relation = session.cartesian_product(relation, child_relation)
        if node.recipe.constraints:
            relation = session.select(relation, node.recipe.constraints)
        if relation.slots != node.schema or relation.root_id is None:
            raise RuntimeError("authorized recipe materialized incorrectly")
        memo[node.topology_sha256] = relation
        return relation

    relation = materialize(binding.candidate.node)
    if binding.candidate.constrained_domains:
        slots = tuple(
            binding.candidate.group.schema[ordinal]
            for ordinal, _values in binding.candidate.constrained_domains
        )
        rows = product(*(values for _ordinal, values in binding.candidate.constrained_domains))
        domain_relation = session.from_bindings(slots, rows)
        relation = session.join(relation, domain_relation)
    if relation.root_id is None:
        raise RuntimeError("authorized domain slice materialized empty")
    family_slice = session.family_slice(binding.candidate.group.family_id, relation)
    return session.proof_forest((family_slice,), saturation_limit=2)


def _state_payload(value: PackedPromptPlanSliceState) -> dict[str, object]:
    return {
        "status": value.status,
        "packed_frontier_sha256": value.packed_frontier_sha256,
        "structural_certificate_sha256": value.structural_certificate_sha256,
        "authorized_slice_count": value.authorized_slice_count,
        "plan_slice_receipt_sha256": value.plan_slice_receipt_sha256,
        "frontier_root_receipt_sha256": value.frontier_root_receipt_sha256,
        "build_work_receipt_sha256": value.build_work_receipt_sha256,
        "authorization_work_receipt_sha256": value.authorization_work_receipt_sha256,
    }


class PackedGenericTypedPromptPlanSliceV1:
    """Deep build(certificate) -> physically-raced select(desired) module."""

    __slots__ = ("_frontier", "_plan_slice_receipt", "_state")

    def __init__(self) -> None:
        raise TypeError("use PackedGenericTypedPromptPlanSliceV1.build")

    @classmethod
    def build(
        cls,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        structural_certificate: StructuralFactorCertificate,
    ) -> PackedGenericTypedPromptPlanSliceV1:
        if not isinstance(signature, TypedSignature) or not (
            signature.result == _INT
            and len(signature.parameters) == 1
            and signature.parameters[0].type == _LIST_INT
        ):
            raise ValueError("prompt plan slicing requires List[Int] -> Int")
        _authenticate_certificate(structural_certificate)
        frontier = breadth_v4.PackedGenericTypedTrainBreadthV4.build(
            signature,
            cases,
            literals,
            max_cost,
        )
        root_receipt = _digest(
            "stopped-root-receipt",
            {
                "frontier_sha256": frontier.sha256,
                "root_eclass_ids": frontier.root_eclass_ids,
            },
        )
        build_work = frontier.state.route_build_work_sha256
        authorization_work = frontier.work.sha256
        status = STOP_WORK_BOUND
        authorized_count = 0
        receipt: PlanSliceReceipt | None = None
        if frontier.status == breadth_v4.FRONTIER_READY:
            try:
                (
                    status,
                    authorized_count,
                    receipt,
                    root_receipt,
                    build_work,
                    authorization_work,
                ) = _bind_certificate(frontier, structural_certificate)
            except packed.PackedRelationWorkLimitExceeded as error:
                route = cast(breadth_v4._PackedListIntFrontier, frontier._route)
                route._bound_name = "recipe-relation-work"
                route._bound_attempted = error.attempted
                status = STOP_WORK_BOUND
                authorization_work = route.work.sha256
        partial = PackedPromptPlanSliceState(
            status=status,
            packed_frontier_sha256=frontier.sha256,
            structural_certificate_sha256=structural_certificate.sha256,
            authorized_slice_count=authorized_count,
            plan_slice_receipt_sha256=None if receipt is None else receipt.sha256,
            frontier_root_receipt_sha256=root_receipt,
            build_work_receipt_sha256=build_work,
            authorization_work_receipt_sha256=authorization_work,
            sha256="",
        )
        state = partial._replace(sha256=_digest("prompt-plan-slice-state", _state_payload(partial)))
        instance = object.__new__(cls)
        instance._frontier = frontier
        instance._plan_slice_receipt = receipt
        instance._state = state
        return instance

    @property
    def state(self) -> PackedPromptPlanSliceState:
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def root_eclass_ids(self) -> tuple[str, ...]:
        return self._frontier.root_eclass_ids

    @property
    def plan_slice_receipt(self) -> PlanSliceReceipt | None:
        return self._plan_slice_receipt

    @property
    def work(self) -> full_typed.PackedFullTypedWorkAudit:
        return self._frontier.work

    def _selection(
        self,
        status: str,
        *,
        selected_eclass_id: str | None = None,
        exact_proof_count: int = 0,
        witnesses: tuple[breadth_v3.PackedGenericTypedWitness, ...] = (),
        selected_proof_forest_sha256: str | None = None,
        race: semantic_race.TypedSemanticVectorRaceAudit | None = None,
        checkpoint_before: str | None = None,
    ) -> PackedPromptPlanSliceSelection:
        route = cast(breadth_v4._PackedListIntFrontier, self._frontier._route)
        before = (
            route._operator.live_checkpoint_sha256
            if checkpoint_before is None
            else checkpoint_before
        )
        after = route._operator.live_checkpoint_sha256
        work = self.work
        receipt_sha = None if self._plan_slice_receipt is None else self._plan_slice_receipt.sha256
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "plan_slice_receipt_sha256": receipt_sha,
            "selected_eclass_id": selected_eclass_id,
            "exact_proof_count": exact_proof_count,
            "witnesses": tuple(item.sha256 for item in witnesses),
            "selected_proof_forest_sha256": selected_proof_forest_sha256,
            "race_sha256": None if race is None else race.sha256,
            "checkpoint_before_sha256": before,
            "checkpoint_after_sha256": after,
            "work_sha256": work.sha256,
        }
        return PackedPromptPlanSliceSelection(
            status=status,
            frontier_sha256=self._state.sha256,
            plan_slice_receipt_sha256=receipt_sha,
            selected_eclass_id=selected_eclass_id,
            exact_proof_count=exact_proof_count,
            witnesses=witnesses,
            selected_proof_forest_sha256=selected_proof_forest_sha256,
            race=race,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
            work=work,
            sha256=_digest("prompt-plan-slice-selection", payload),
        )

    def _authenticate_state(self) -> None:
        if (
            self._state.packed_frontier_sha256 != self._frontier.sha256
            or self._state.sha256 != _digest("prompt-plan-slice-state", _state_payload(self._state))
            or (
                self._plan_slice_receipt is None
                and self._state.plan_slice_receipt_sha256 is not None
            )
            or (
                self._plan_slice_receipt is not None
                and self._state.plan_slice_receipt_sha256 != self._plan_slice_receipt.sha256
            )
        ):
            raise RuntimeError("prompt plan slice state authentication failed")

    def select(self, desired: Sequence[object]) -> PackedPromptPlanSliceSelection:
        self._authenticate_state()
        if self._state.status != FRONTIER_READY:
            return self._selection(self._state.status)
        receipt = self._plan_slice_receipt
        if receipt is None:
            raise RuntimeError("ready plan slice lost its receipt")
        self._frontier._authenticate()
        binding = _authenticate_plan_slice_receipt(self._frontier, receipt)
        route = binding.route
        try:
            route._workset.session.charge_workset("authorized-plan-slice-receipt-authenticate")
            if isinstance(desired, (str, bytes, bytearray)):
                raise TypeError("desired outputs must be exact integers")
            desired_values: list[int] = []
            for value in desired:
                route._workset.session.charge_frontier("desired-value-read")
                if type(value) is not int:
                    raise ValueError("desired outputs must align with typed cases")
                desired_values.append(value)
            if len(desired_values) != len(route.state.case_arguments):
                raise ValueError("desired outputs must align with typed cases")
            encoded = tuple(
                output_cells.TypedOutputCellBank.encode(
                    value,
                    declared_type=output_cells.OutputType.INT,
                )
                for value in desired_values
            )
            before = route._operator.live_checkpoint_sha256
            attempted_races = route._physical_races + 1
            if attempted_races > MAX_PHYSICAL_RACES:
                raise packed_v2._PhysicalWorkBoundExceeded(
                    "physical-races",
                    attempted_races,
                )
            route._physical_races = attempted_races
            typed_capsule = cast(full_typed._AuthenticatedTypedCapsule, route._typed_capsule)
            race = semantic_race.select_exact_semantic_vector(
                typed_capsule.root_eclass_ids,
                tuple(item.physical_outputs for item in typed_capsule.root_eclasses),
                encoded,
            )
            if route._operator.live_checkpoint_sha256 != before:
                raise RuntimeError("typed operator changed during authorized selection")
            if race.status == semantic_race.STOP_NO_EXACT:
                return self._selection(STOP_NO_EXACT, race=race, checkpoint_before=before)
            if race.status != semantic_race.SELECTED or race.winner_key is None:
                raise RuntimeError("authorized root e-class bank returned a physical tie")
            if race.winner_key != receipt.authenticated_eclass_id:
                route._concrete_proof_witnesses = 0
                return self._selection(
                    STOP_AUTHORIZED_ECLASS_MISMATCH,
                    selected_eclass_id=race.winner_key,
                    race=race,
                    checkpoint_before=before,
                )
            route._workset.session.charge_frontier("winner-eclass-index-lookup")
            ordinal = cast(Mapping[str, int], typed_capsule.root_ordinal_index).get(race.winner_key)
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < len(typed_capsule.root_eclasses)
                or typed_capsule.root_eclasses[ordinal] is not binding.candidate.eclass
            ):
                raise RuntimeError("authorized physical winner identity changed")
            forest = _materialize_authorized_forest(binding)
            proof_count = forest.exact_logical_recipe_count
            if proof_count < 1 or not forest.first_two_witnesses:
                raise RuntimeError("authorized plan slice published no proof")
            route_witnesses = tuple(
                route._render_witness(item, binding.component)
                for item in forest.first_two_witnesses[:2]
            )
            witnesses = tuple(self._frontier._compile_witness(item) for item in route_witnesses)
            route._concrete_proof_witnesses = len(forest.first_two_witnesses)
            status = SELECTED if proof_count == 1 else STOP_AMBIGUOUS
            return self._selection(
                status,
                selected_eclass_id=race.winner_key,
                exact_proof_count=proof_count,
                witnesses=witnesses,
                selected_proof_forest_sha256=forest.sha256,
                race=race,
                checkpoint_before=before,
            )
        except packed.PackedRelationWorkLimitExceeded as error:
            route._bound_name = "recipe-relation-work"
            route._bound_attempted = error.attempted
            route._concrete_proof_witnesses = 0
            return self._selection(STOP_WORK_BOUND)
        except packed_v2._PhysicalWorkBoundExceeded as error:
            route._bound_name = error.name
            route._bound_attempted = error.attempted
            route._concrete_proof_witnesses = 0
            return self._selection(STOP_WORK_BOUND)


__all__ = [
    "FRONTIER_READY",
    "MAX_PHYSICAL_PREDICTIONS",
    "MAX_PHYSICAL_RACES",
    "MAX_RECIPE_RELATION_WORK",
    "SELECTED",
    "STOP_AMBIGUOUS",
    "STOP_AMBIGUOUS_AUTHORIZED_SLICE",
    "STOP_AUTHORIZED_ECLASS_MISMATCH",
    "STOP_NO_AUTHORIZED_SLICE",
    "STOP_NO_EXACT",
    "STOP_WORK_BOUND",
    "AuthenticatedPlanSlice",
    "PackedGenericTypedPromptPlanSliceV1",
    "PackedPromptPlanSliceSelection",
    "PackedPromptPlanSliceState",
    "PlanSliceReceipt",
    "StructuralDomainConstraint",
    "StructuralFactorCertificate",
    "StructuralOccurrence",
    "StructuralRelationConstraint",
    "freeze_structural_factor_certificate",
]
