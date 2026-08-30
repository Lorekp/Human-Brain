"""Private V6 occurrence-relay to packed plan-slice integration.

The public build seam accepts one opaque V6 relay certificate plus ordinary
generic packed inputs.  It authenticates the issued exact object, reconstructs
occurrence order only by walking root/child-position carrier relays, translates
physical factor/type/relation/domain/copy relays, and freezes one private
operator-name-free structural certificate.  Only then may the banked packed
plan-slice builder run.  Desired values remain exclusively in ``select``.

This is a self-posed integration mechanism.  It does not consume canonical or
public prompts and does not establish language authority or transfer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from typing import NamedTuple

from experiments.r3_pilot import prompt_occurrence_factor_relays_v6 as relays
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "PACKED_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1"
READY = plan_slice.FRONTIER_READY
SELECTED = plan_slice.SELECTED
STOP_SOURCE_CERTIFICATE = (
    "STOP_PACKED_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_SOURCE_CERTIFICATE"
)
STOP_SIGNATURE_MISMATCH = (
    "STOP_PACKED_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_SIGNATURE_MISMATCH"
)
STOP_RELAY_ZERO = "STOP_PACKED_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_RELAY_ZERO"
STOP_RELAY_AMBIGUOUS = "STOP_PACKED_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_RELAY_AMBIGUOUS"
STOP_RELAY_UNSUPPORTED = "STOP_PACKED_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_RELAY_UNSUPPORTED"

PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_INPUT_COUNT = 0
DESIRED_BUILD_INPUT_COUNT = 0
PROGRAM_INPUT_COUNT = 0
TASK_INPUT_COUNT = 0
FAMILY_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_NAMESPACE = b"packed-occurrence-factor-authorized-plan-slice-v1\0"
_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


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


class ConverterWorkAudit(NamedTuple):
    occurrence_carrier_reads: int
    source_relay_reads: int
    edge_relay_reads: int
    factor_slot_relay_reads: int
    relation_relay_reads: int
    domain_relay_reads: int
    copy_group_relay_reads: int
    copy_member_relay_reads: int
    signature_relay_reads: int
    physical_relay_reads: int
    structural_freeze_attempts: int
    sha256: str


class ConverterReceiptSummary(NamedTuple):
    source_certificate_sha256: str
    source_view_sha256: str
    source_checkpoint_sha256: str
    source_protocol_sha256: str
    physical_work_receipt_sha256: str
    structural_certificate_sha256: str
    work: ConverterWorkAudit
    sha256: str


class OccurrenceAuthorizedBuildState(NamedTuple):
    status: str
    stop_reason: str | None
    source_certificate_sha256: str | None
    source_view_sha256: str | None
    source_checkpoint_sha256: str | None
    source_protocol_sha256: str | None
    physical_work_receipt_sha256: str | None
    converter_receipt: ConverterReceiptSummary | None
    structural_certificate_sha256: str | None
    structural_freeze_attempts: int
    plan_build_attempts: int
    plan_slice_state_sha256: str | None
    desired_reads_before_select: int
    proof_witnesses_before_select: int
    sha256: str


class OccurrenceAuthorizedSelection(NamedTuple):
    status: str
    build_state_sha256: str
    source_certificate_sha256: str | None
    converter_receipt_sha256: str | None
    plan_slice_selection_sha256: str | None
    selected_eclass_id: str | None
    exact_proof_count: int
    witnesses: tuple[object, ...]
    selected_proof_forest_sha256: str | None
    work: object | None
    sha256: str


class _ConversionStop(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class _IssuedConversion(NamedTuple):
    authority: object
    source_certificate: relays.OccurrenceFactorRelayCertificate
    source_view: relays.OccurrenceFactorRelayView
    structural_certificate: plan_slice.StructuralFactorCertificate
    summary: ConverterReceiptSummary


_CONVERTER_AUTHORITY = object()
_ISSUED_CONVERSIONS: dict[int, _IssuedConversion] = {}


def _work_payload(work: ConverterWorkAudit) -> dict[str, object]:
    return {
        "occurrence_carrier_reads": work.occurrence_carrier_reads,
        "source_relay_reads": work.source_relay_reads,
        "edge_relay_reads": work.edge_relay_reads,
        "factor_slot_relay_reads": work.factor_slot_relay_reads,
        "relation_relay_reads": work.relation_relay_reads,
        "domain_relay_reads": work.domain_relay_reads,
        "copy_group_relay_reads": work.copy_group_relay_reads,
        "copy_member_relay_reads": work.copy_member_relay_reads,
        "signature_relay_reads": work.signature_relay_reads,
        "physical_relay_reads": work.physical_relay_reads,
        "structural_freeze_attempts": work.structural_freeze_attempts,
    }


def _summary_payload(summary: ConverterReceiptSummary) -> dict[str, object]:
    return {
        "source_certificate_sha256": summary.source_certificate_sha256,
        "source_view_sha256": summary.source_view_sha256,
        "source_checkpoint_sha256": summary.source_checkpoint_sha256,
        "source_protocol_sha256": summary.source_protocol_sha256,
        "physical_work_receipt_sha256": summary.physical_work_receipt_sha256,
        "structural_certificate_sha256": summary.structural_certificate_sha256,
        "work_sha256": summary.work.sha256,
    }


def _state_payload(state: OccurrenceAuthorizedBuildState) -> dict[str, object]:
    return {
        "status": state.status,
        "stop_reason": state.stop_reason,
        "source_certificate_sha256": state.source_certificate_sha256,
        "source_view_sha256": state.source_view_sha256,
        "source_checkpoint_sha256": state.source_checkpoint_sha256,
        "source_protocol_sha256": state.source_protocol_sha256,
        "physical_work_receipt_sha256": state.physical_work_receipt_sha256,
        "converter_receipt_sha256": (
            state.converter_receipt.sha256 if state.converter_receipt is not None else None
        ),
        "structural_certificate_sha256": state.structural_certificate_sha256,
        "structural_freeze_attempts": state.structural_freeze_attempts,
        "plan_build_attempts": state.plan_build_attempts,
        "plan_slice_state_sha256": state.plan_slice_state_sha256,
        "desired_reads_before_select": state.desired_reads_before_select,
        "proof_witnesses_before_select": state.proof_witnesses_before_select,
    }


def _new_state(**values: object) -> OccurrenceAuthorizedBuildState:
    partial = OccurrenceAuthorizedBuildState(sha256="", **values)
    return partial._replace(
        sha256=_digest("occurrence-authorized-build-state", _state_payload(partial))
    )


def _physical_type(
    assembly: relays.PhysicalTypeAssembly,
) -> str:
    if type(assembly) is not relays.PhysicalTypeAssembly:
        raise _ConversionStop(STOP_RELAY_AMBIGUOUS, "factor type relay is not exact")
    matches = tuple(
        item
        for item in relays.TYPE_ASSEMBLIES
        if assembly is item
        and assembly.active_cells == item.active_cells
        and assembly.sha256 == item.sha256
    )
    if len(matches) != 1:
        raise _ConversionStop(
            STOP_RELAY_AMBIGUOUS,
            "factor type cells do not identify one authenticated physical type",
        )
    return matches[0].value_type


def _decode_int_value(event: relays.TypedValueEvent) -> int:
    if type(event) is not relays.TypedValueEvent:
        raise _ConversionStop(STOP_RELAY_AMBIGUOUS, "typed value relay is not exact")
    cells = event.typed_output_active_cells
    if (
        len(cells) != 2
        or cells[0] != output_cells.TYPE_OFFSET
        or not output_cells.INT_OFFSET
        <= cells[1]
        < output_cells.INT_OFFSET + output_cells.INT_KNOWN_CELL_COUNT
    ):
        raise _ConversionStop(
            STOP_RELAY_AMBIGUOUS,
            "typed value relay does not contain one physical Int value",
        )
    return output_cells.INT_MIN + cells[1] - output_cells.INT_OFFSET


def _ordered_occurrence_carriers(
    view: relays.OccurrenceFactorRelayView,
) -> tuple[
    tuple[tuple[int, ...], ...],
    Mapping[tuple[int, ...], tuple[tuple[int, tuple[int, ...]], ...]],
    Mapping[tuple[int, ...], tuple[tuple[int, ...], int, tuple[int, ...]]],
]:
    source_carriers = tuple(item.carrier_cells for item in view.source_relays)
    if (
        len(source_carriers) != 8
        or len(set(source_carriers)) != len(source_carriers)
        or view.root_carrier_cells not in source_carriers
        or set(source_carriers) != set(view.occurrence_carriers)
    ):
        raise _ConversionStop(
            STOP_RELAY_AMBIGUOUS,
            "source relays do not bind eight unique occurrence carriers",
        )
    mutable_children: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    parent_by_child: dict[tuple[int, ...], tuple[tuple[int, ...], int, tuple[int, ...]]] = {}
    edge_keys: set[tuple[tuple[int, ...], int]] = set()
    if len(view.edges) != 7:
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "the retained integration requires seven physical edge relays",
        )
    for edge in view.edges:
        if type(edge) is not relays.CarrierEdgeRelay:
            raise _ConversionStop(STOP_RELAY_AMBIGUOUS, "edge relay is not exact")
        key = (edge.parent_carrier_cells, edge.child_position)
        if (
            key in edge_keys
            or edge.child_carrier_cells in parent_by_child
            or edge.parent_carrier_cells not in source_carriers
            or edge.child_carrier_cells not in source_carriers
            or edge.parent_carrier_cells == edge.child_carrier_cells
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "edge relays contain a duplicate parent, position, or child",
            )
        edge_keys.add(key)
        mutable_children[edge.parent_carrier_cells].append(
            (edge.child_position, edge.child_carrier_cells)
        )
        parent_by_child[edge.child_carrier_cells] = (
            edge.parent_carrier_cells,
            edge.child_position,
            edge.parent_slot_occupancy,
        )
    if view.root_carrier_cells in parent_by_child or any(
        carrier not in parent_by_child
        for carrier in source_carriers
        if carrier != view.root_carrier_cells
    ):
        raise _ConversionStop(
            STOP_RELAY_AMBIGUOUS,
            "physical carriers do not form one rooted occurrence tree",
        )
    children: dict[tuple[int, ...], tuple[tuple[int, tuple[int, ...]], ...]] = {}
    for parent, entries in mutable_children.items():
        ordered_entries = tuple(sorted(entries))
        if tuple(position for position, _child in ordered_entries) != tuple(
            range(len(ordered_entries))
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "physical child-position relays are not contiguous",
            )
        children[parent] = ordered_entries

    ordered: list[tuple[int, ...]] = []
    visiting: set[tuple[int, ...]] = set()

    def visit(carrier: tuple[int, ...]) -> None:
        if carrier in visiting or carrier in ordered:
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "physical occurrence carrier traversal contains a cycle",
            )
        visiting.add(carrier)
        ordered.append(carrier)
        for _position, child in children.get(carrier, ()):
            visit(child)
        visiting.remove(carrier)

    visit(view.root_carrier_cells)
    if set(ordered) != set(source_carriers):
        raise _ConversionStop(
            STOP_RELAY_AMBIGUOUS,
            "physical occurrence carrier traversal is disconnected",
        )
    return tuple(ordered), children, parent_by_child


def _convert_ready_view(
    source_certificate: relays.OccurrenceFactorRelayCertificate,
    view: relays.OccurrenceFactorRelayView,
    signature: TypedSignature,
) -> _IssuedConversion:
    if (
        source_certificate.status != relays.READY
        or source_certificate.checkpoint_sha256 != view.checkpoint_sha256
        or view.checkpoint_sha256 != relays.STATIC_CONSUMER.checkpoint_sha256
        or view.protocol_sha256 != relays.STATIC_CONSUMER.protocol_sha256
        or view.protocol_sha256 != relays.PROTOCOL.protocol_sha256
    ):
        raise _ConversionStop(
            STOP_SOURCE_CERTIFICATE,
            "V6 source checkpoint or protocol identity changed",
        )
    ordered, _children, parent_by_child = _ordered_occurrence_carriers(view)
    occurrence_index = {carrier: index for index, carrier in enumerate(ordered)}

    source_keys = {item.carrier_cells for item in view.source_relays}
    if len(source_keys) != len(view.source_relays) or source_keys != set(ordered):
        raise _ConversionStop(
            STOP_RELAY_AMBIGUOUS,
            "carrier source relays are zero or multiple",
        )

    factor_rows: dict[tuple[int, ...], list[relays.FactorSlotRelay]] = defaultdict(list)
    if len(view.factor_slots) != 16:
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "the retained integration requires sixteen factor-slot relays",
        )
    for factor in view.factor_slots:
        if (
            type(factor) is not relays.FactorSlotRelay
            or factor.carrier_cells not in occurrence_index
            or not 0 <= factor.factor_kind_cell < len(relays.FACTOR_KINDS)
            or factor.factor_kind != relays.FACTOR_KINDS[factor.factor_kind_cell]
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "factor slot is not bound to one physical kind/type relay",
            )
        _physical_type(factor.type_assembly)
        factor_rows[factor.carrier_cells].append(factor)

    frozen_factors: dict[tuple[int, ...], tuple[tuple[str, str], ...]] = {}
    for carrier in ordered:
        rows = tuple(sorted(factor_rows.get(carrier, ()), key=lambda item: item.local_slot))
        if not rows or tuple(item.local_slot for item in rows) != tuple(range(len(rows))):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "factor slots are not one complete ordered carrier-keyed tuple",
            )
        frozen_factors[carrier] = tuple(
            (relays.FACTOR_KINDS[item.factor_kind_cell], _physical_type(item.type_assembly))
            for item in rows
        )

    relation_rows: dict[tuple[int, ...], list[plan_slice.StructuralRelationConstraint]] = (
        defaultdict(list)
    )
    if len(view.relations) != 1:
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "the retained integration requires one generic relation relay",
        )
    for relation in view.relations:
        if (
            type(relation) is not relays.RelationConstraintRelay
            or relation.carrier_cells not in occurrence_index
            or not 0 <= relation.operator_cell < len(relays.RELATION_OPERATORS)
            or relation.operator != relays.RELATION_OPERATORS[relation.operator_cell]
            or not 0 <= relation.left_slot < len(frozen_factors[relation.carrier_cells])
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "relation relay has no exact carrier/slot/operator identity",
            )
        if relation.right_slot is not None:
            if (
                relation.right_value_event is not None
                or relation.value_relay_cells
                or not 0 <= relation.right_slot < len(frozen_factors[relation.carrier_cells])
            ):
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "slot relation has conflicting right-hand physical evidence",
                )
            right_value = None
        else:
            if relation.right_value_event is None or not relation.value_relay_cells:
                raise _ConversionStop(
                    STOP_RELAY_ZERO,
                    "value relation is missing its typed value relay",
                )
            right_value = _decode_int_value(relation.right_value_event)
        relation_rows[relation.carrier_cells].append(
            plan_slice.StructuralRelationConstraint(
                left_slot=relation.left_slot,
                operator=relays.RELATION_OPERATORS[relation.operator_cell],
                right_slot=relation.right_slot,
                right_value=right_value,
            )
        )

    domain_values: dict[tuple[tuple[int, ...], int], set[int]] = defaultdict(set)
    if len(view.domains) != 2:
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "the retained integration requires two finite domain relays",
        )
    for domain in view.domains:
        if (
            type(domain) is not relays.DomainRelay
            or domain.carrier_cells not in occurrence_index
            or domain.domain_kind_cell != relays.DOMAIN_LITERAL
            or domain.type_assembly is not relays.TYPE_BY_NAME["Int"]
            or domain.class_cells != relays.EXACT_INT_CLASS_CELLS
            or not domain.value_relay_cells
            or not 0 <= domain.local_slot < len(frozen_factors[domain.carrier_cells])
            or frozen_factors[domain.carrier_cells][domain.local_slot][0] != "literal"
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "domain relay is not one complete finite literal/class binding",
            )
        domain_values[(domain.carrier_cells, domain.local_slot)].add(
            _decode_int_value(domain.typed_value_event)
        )

    copy_groups: list[tuple[tuple[int, int], ...]] = []
    seen_copy_members: set[tuple[int, int]] = set()
    if len(view.literal_copy_groups) != 2:
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "the retained integration requires two literal-copy groups",
        )
    copy_member_reads = 0
    group_cells: set[tuple[int, ...]] = set()
    for group in view.literal_copy_groups:
        if (
            type(group) is not relays.LiteralCopyGroupRelay
            or group.group_cells in group_cells
            or not group.members
            or len(group.members) != len(group.event_receipt_sha256s)
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "literal-copy group relay is zero, repeated, or unauthenticated",
            )
        group_cells.add(group.group_cells)
        converted: list[tuple[int, int]] = []
        for carrier, local_slot in group.members:
            copy_member_reads += 1
            if (
                carrier not in occurrence_index
                or not 0 <= local_slot < len(frozen_factors[carrier])
                or frozen_factors[carrier][local_slot][0] != "literal"
            ):
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "literal-copy member lacks one carrier-keyed literal slot",
                )
            member = (occurrence_index[carrier], local_slot)
            if member in seen_copy_members:
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "literal-copy member appears in multiple physical groups",
                )
            seen_copy_members.add(member)
            converted.append(member)
        copy_groups.append(tuple(sorted(converted)))

    expected_signature: set[tuple[int, int, str]] = set()
    for item in view.signature_relays:
        if (
            type(item) is not relays.SignatureTypeRelay
            or item.root_carrier_cells != view.root_carrier_cells
            or item.signature_role_cell
            not in {
                relays.SIGNATURE_PARAMETER,
                relays.SIGNATURE_RETURN,
            }
            or item.signature_role
            != ("parameter" if item.signature_role_cell == relays.SIGNATURE_PARAMETER else "return")
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "signature relay is not one physical root-keyed type",
            )
        expected_signature.add(
            (
                item.signature_role_cell,
                item.position,
                _physical_type(item.type_assembly),
            )
        )
    if expected_signature != {
        (relays.SIGNATURE_PARAMETER, 0, "List[Int]"),
        (relays.SIGNATURE_RETURN, 0, "Int"),
    }:
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "physical typed-signature relays are incomplete",
        )
    if not (
        type(signature) is TypedSignature
        and signature.result == _INT
        and len(signature.parameters) == 1
        and signature.parameters[0].type == _LIST_INT
    ):
        raise _ConversionStop(
            STOP_SIGNATURE_MISMATCH,
            "ordinary packed signature does not match the physical signature relays",
        )

    occurrences: list[plan_slice.StructuralOccurrence] = []
    for carrier in ordered:
        if carrier == view.root_carrier_cells:
            parent_occurrence = child_position = None
            occupancy: tuple[int, ...] = ()
        else:
            parent_carrier, child_position, occupancy = parent_by_child[carrier]
            parent_occurrence = occurrence_index[parent_carrier]
            if (
                len(occupancy) != len(frozen_factors[carrier])
                or len(set(occupancy)) != len(occupancy)
                or any(not 0 <= slot < len(frozen_factors[parent_carrier]) for slot in occupancy)
            ):
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "edge occupancy does not bind the complete child factor tuple",
                )
        constraints = tuple(
            sorted(
                relation_rows.get(carrier, ()),
                key=lambda item: (
                    item.left_slot,
                    relays.RELATION_OPERATORS.index(item.operator),
                    -1 if item.right_slot is None else item.right_slot,
                    -1 if item.right_value is None else item.right_value,
                ),
            )
        )
        occurrences.append(
            plan_slice.StructuralOccurrence(
                parent_occurrence=parent_occurrence,
                child_position=child_position,
                factors=frozen_factors[carrier],
                parent_slot_occupancy=occupancy,
                relation_constraints=constraints,
            )
        )

    domains = tuple(
        plan_slice.StructuralDomainConstraint(
            occurrence=occurrence_index[carrier],
            slot=slot,
            allowed_values=tuple(sorted(values)),
        )
        for (carrier, slot), values in sorted(
            domain_values.items(),
            key=lambda item: (occurrence_index[item[0][0]], item[0][1]),
        )
    )
    if (
        len(occurrences) != 8
        or len(domains) != 2
        or len(copy_groups) != 2
        or any(not values for values in domain_values.values())
    ):
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "translated physical fields do not form one complete retained slice",
        )

    structural = plan_slice.freeze_structural_factor_certificate(
        occurrences,
        domains,
        copy_groups,
    )
    physical_reads = (
        len(ordered)
        + len(view.source_relays)
        + len(view.edges)
        + len(view.factor_slots)
        + len(view.relations)
        + len(view.domains)
        + len(view.literal_copy_groups)
        + copy_member_reads
        + len(view.signature_relays)
    )
    work_partial = ConverterWorkAudit(
        occurrence_carrier_reads=len(ordered),
        source_relay_reads=len(view.source_relays),
        edge_relay_reads=len(view.edges),
        factor_slot_relay_reads=len(view.factor_slots),
        relation_relay_reads=len(view.relations),
        domain_relay_reads=len(view.domains),
        copy_group_relay_reads=len(view.literal_copy_groups),
        copy_member_relay_reads=copy_member_reads,
        signature_relay_reads=len(view.signature_relays),
        physical_relay_reads=physical_reads,
        structural_freeze_attempts=1,
        sha256="",
    )
    work = work_partial._replace(sha256=_digest("converter-work", _work_payload(work_partial)))
    summary_partial = ConverterReceiptSummary(
        source_certificate_sha256=source_certificate.sha256,
        source_view_sha256=view.sha256,
        source_checkpoint_sha256=view.checkpoint_sha256,
        source_protocol_sha256=view.protocol_sha256,
        physical_work_receipt_sha256=view.physical_work_receipt_sha256,
        structural_certificate_sha256=structural.sha256,
        work=work,
        sha256="",
    )
    summary = summary_partial._replace(
        sha256=_digest("converter-receipt", _summary_payload(summary_partial))
    )
    issued = _IssuedConversion(
        _CONVERTER_AUTHORITY,
        source_certificate,
        view,
        structural,
        summary,
    )
    _ISSUED_CONVERSIONS[id(issued)] = issued
    return issued


def _authenticate_conversion(issued: _IssuedConversion) -> None:
    if (
        type(issued) is not _IssuedConversion
        or _ISSUED_CONVERSIONS.get(id(issued)) is not issued
        or issued.authority is not _CONVERTER_AUTHORITY
        or issued.source_view
        is not relays.require_occurrence_factor_relays_v6(issued.source_certificate)
        or issued.summary.source_certificate_sha256 != issued.source_certificate.sha256
        or issued.summary.source_view_sha256 != issued.source_view.sha256
        or issued.summary.source_checkpoint_sha256 != issued.source_view.checkpoint_sha256
        or issued.summary.source_protocol_sha256 != issued.source_view.protocol_sha256
        or issued.summary.physical_work_receipt_sha256
        != issued.source_view.physical_work_receipt_sha256
        or issued.summary.structural_certificate_sha256 != issued.structural_certificate.sha256
        or issued.summary.work.sha256
        != _digest("converter-work", _work_payload(issued.summary.work))
        or issued.summary.sha256 != _digest("converter-receipt", _summary_payload(issued.summary))
    ):
        raise RuntimeError("private occurrence-factor conversion changed")


class PackedOccurrenceFactorAuthorizedPlanSliceV1:
    """Deep opaque-certificate build -> desired-only select integration."""

    __slots__ = ("_conversion", "_frontier", "_source_certificate", "_state")

    def __init__(self) -> None:
        raise TypeError("use PackedOccurrenceFactorAuthorizedPlanSliceV1.build")

    @classmethod
    def _stopped(
        cls,
        status: str,
        reason: str,
        *,
        source_certificate: object | None = None,
        source_view: relays.OccurrenceFactorRelayView | None = None,
    ) -> PackedOccurrenceFactorAuthorizedPlanSliceV1:
        source_sha = (
            source_certificate.sha256
            if type(source_certificate) is relays.OccurrenceFactorRelayCertificate
            else None
        )
        state = _new_state(
            status=status,
            stop_reason=reason,
            source_certificate_sha256=source_sha,
            source_view_sha256=None if source_view is None else source_view.sha256,
            source_checkpoint_sha256=(
                None if source_view is None else source_view.checkpoint_sha256
            ),
            source_protocol_sha256=(None if source_view is None else source_view.protocol_sha256),
            physical_work_receipt_sha256=(
                None if source_view is None else source_view.physical_work_receipt_sha256
            ),
            converter_receipt=None,
            structural_certificate_sha256=None,
            structural_freeze_attempts=0,
            plan_build_attempts=0,
            plan_slice_state_sha256=None,
            desired_reads_before_select=0,
            proof_witnesses_before_select=0,
        )
        instance = object.__new__(cls)
        instance._source_certificate = source_certificate
        instance._conversion = None
        instance._frontier = None
        instance._state = state
        return instance

    @classmethod
    def build(
        cls,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        source_certificate: relays.OccurrenceFactorRelayCertificate,
    ) -> PackedOccurrenceFactorAuthorizedPlanSliceV1:
        try:
            view = relays.require_occurrence_factor_relays_v6(source_certificate)
        except (TypeError, RuntimeError) as error:
            return cls._stopped(
                STOP_SOURCE_CERTIFICATE,
                str(error),
                source_certificate=source_certificate,
            )
        try:
            conversion = _convert_ready_view(source_certificate, view, signature)
        except _ConversionStop as error:
            return cls._stopped(
                error.status,
                error.reason,
                source_certificate=source_certificate,
                source_view=view,
            )
        _authenticate_conversion(conversion)
        frontier = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
            signature,
            cases,
            literals,
            max_cost,
            conversion.structural_certificate,
        )
        events = dict(frontier.work.relation.event_counts)
        desired_reads = events.get("frontier:desired-value-read", 0)
        proof_witnesses = frontier.work.concrete_proof_witnesses
        if desired_reads != 0 or proof_witnesses != 0:
            raise RuntimeError("packed plan slice consumed desired/proof evidence during build")
        state = _new_state(
            status=frontier.status,
            stop_reason=None,
            source_certificate_sha256=source_certificate.sha256,
            source_view_sha256=view.sha256,
            source_checkpoint_sha256=view.checkpoint_sha256,
            source_protocol_sha256=view.protocol_sha256,
            physical_work_receipt_sha256=view.physical_work_receipt_sha256,
            converter_receipt=conversion.summary,
            structural_certificate_sha256=conversion.structural_certificate.sha256,
            structural_freeze_attempts=1,
            plan_build_attempts=1,
            plan_slice_state_sha256=frontier.sha256,
            desired_reads_before_select=desired_reads,
            proof_witnesses_before_select=proof_witnesses,
        )
        instance = object.__new__(cls)
        instance._source_certificate = source_certificate
        instance._conversion = conversion
        instance._frontier = frontier
        instance._state = state
        return instance

    @property
    def state(self) -> OccurrenceAuthorizedBuildState:
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def sha256(self) -> str:
        return self._state.sha256

    @property
    def plan_slice_receipt(self) -> plan_slice.PlanSliceReceipt | None:
        return None if self._frontier is None else self._frontier.plan_slice_receipt

    @property
    def work(self) -> object | None:
        return None if self._frontier is None else self._frontier.work

    def _authenticate(self) -> None:
        if self._state.sha256 != _digest(
            "occurrence-authorized-build-state", _state_payload(self._state)
        ):
            raise RuntimeError("occurrence-authorized integration state changed")
        if self._conversion is None:
            if self._frontier is not None or self._state.plan_build_attempts != 0:
                raise RuntimeError("stopped integration retained a plan frontier")
            return
        _authenticate_conversion(self._conversion)
        if (
            self._frontier is None
            or self._state.source_certificate_sha256 != self._conversion.source_certificate.sha256
            or self._state.converter_receipt is not self._conversion.summary
            or self._state.structural_certificate_sha256
            != self._conversion.structural_certificate.sha256
            or self._state.plan_slice_state_sha256 != self._frontier.sha256
        ):
            raise RuntimeError("occurrence-authorized integration binding changed")

    def _selection(
        self,
        status: str,
        delegated: plan_slice.PackedPromptPlanSliceSelection | None = None,
    ) -> OccurrenceAuthorizedSelection:
        witnesses: tuple[object, ...] = () if delegated is None else delegated.witnesses
        payload = {
            "status": status,
            "build_state_sha256": self._state.sha256,
            "source_certificate_sha256": self._state.source_certificate_sha256,
            "converter_receipt_sha256": (
                None
                if self._state.converter_receipt is None
                else self._state.converter_receipt.sha256
            ),
            "plan_slice_selection_sha256": (None if delegated is None else delegated.sha256),
            "selected_eclass_id": (None if delegated is None else delegated.selected_eclass_id),
            "exact_proof_count": 0 if delegated is None else delegated.exact_proof_count,
            "witness_sha256s": tuple(getattr(item, "sha256", None) for item in witnesses),
            "selected_proof_forest_sha256": (
                None if delegated is None else delegated.selected_proof_forest_sha256
            ),
            "work_sha256": (None if delegated is None else delegated.work.sha256),
        }
        return OccurrenceAuthorizedSelection(
            status=status,
            build_state_sha256=self._state.sha256,
            source_certificate_sha256=self._state.source_certificate_sha256,
            converter_receipt_sha256=(
                None
                if self._state.converter_receipt is None
                else self._state.converter_receipt.sha256
            ),
            plan_slice_selection_sha256=(None if delegated is None else delegated.sha256),
            selected_eclass_id=(None if delegated is None else delegated.selected_eclass_id),
            exact_proof_count=0 if delegated is None else delegated.exact_proof_count,
            witnesses=witnesses,
            selected_proof_forest_sha256=(
                None if delegated is None else delegated.selected_proof_forest_sha256
            ),
            work=None if delegated is None else delegated.work,
            sha256=_digest("occurrence-authorized-selection", payload),
        )

    def select(self, desired: Sequence[object]) -> OccurrenceAuthorizedSelection:
        self._authenticate()
        if self._frontier is None:
            return self._selection(self._state.status)
        delegated = self._frontier.select(desired)
        self._authenticate()
        return self._selection(delegated.status, delegated)


__all__ = [
    "CANONICAL_INPUT_COUNT",
    "DESIRED_BUILD_INPUT_COUNT",
    "DEV_INPUT_COUNT",
    "HELDOUT_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "READY",
    "SCHEMA",
    "SELECTED",
    "STOP_RELAY_AMBIGUOUS",
    "STOP_RELAY_UNSUPPORTED",
    "STOP_RELAY_ZERO",
    "STOP_SIGNATURE_MISMATCH",
    "STOP_SOURCE_CERTIFICATE",
    "ConverterReceiptSummary",
    "ConverterWorkAudit",
    "OccurrenceAuthorizedBuildState",
    "OccurrenceAuthorizedSelection",
    "PackedOccurrenceFactorAuthorizedPlanSliceV1",
]
