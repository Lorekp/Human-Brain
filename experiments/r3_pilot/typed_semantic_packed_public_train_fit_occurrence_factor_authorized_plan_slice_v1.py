"""Target-free static occurrence relays to private packed plan authority.

The public build seam accepts an exact-object certificate issued by the static
public-TRAIN FIT occurrence-factor checkpoint plus ordinary packed inputs.  It
never imports the FIT learner, its protocol/fixtures, producer V2, or the
self-posed V6 mechanism.  The learned relay view is converted privately into
one structural factor certificate and then delegated to the banked
desired-free packed build -> desired-only select seam.

This is same-FIT mechanism integration only.  It is not a raw-prompt adapter,
does not spend the canonical panel, and provides no heldout or transfer claim.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as source,
)
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "PACKED_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1"
READY = plan_slice.FRONTIER_READY
SELECTED = plan_slice.SELECTED
STOP_SOURCE_CERTIFICATE = (
    "STOP_PACKED_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_"
    "SOURCE_CERTIFICATE"
)
STOP_SIGNATURE_MISMATCH = (
    "STOP_PACKED_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_"
    "SIGNATURE_MISMATCH"
)
STOP_RELAY_ZERO = (
    "STOP_PACKED_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_"
    "RELAY_ZERO"
)
STOP_RELAY_AMBIGUOUS = (
    "STOP_PACKED_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_"
    "RELAY_AMBIGUOUS"
)
STOP_RELAY_UNSUPPORTED = (
    "STOP_PACKED_PUBLIC_TRAIN_FIT_OCCURRENCE_FACTOR_AUTHORIZED_PLAN_SLICE_V1_"
    "RELAY_UNSUPPORTED"
)

EXPECTED_SOURCE_PROTOCOL_SHA256 = (
    "792b517f364f075c46fef51671ea26b8117eb0f78d37d3b1d0cdee6867c26e61"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "aee5d7da81f11a2d71bd1a6cb37383f8364a5e5427fe5055ef15c11790dba290"
)
STATIC_CAPSULE_BANK_COMMIT = "d2376c7dc75314ae285ffb98edeae3e704b63d49"
EXPECTED_STATIC_CAPSULE_SOURCE_SHA256 = (
    "0af68609e46afd65e8dc50c656f7ad22919a08329045d3068f8a457813870c89"
)
EXPECTED_STATIC_CAPSULE_ARTIFACT_SHA256 = (
    "42ca5251b7a045ca98da4e2b008a82d8c89d42f229f6b3eb670ade7a30919711"
)
EXPECTED_STATIC_CAPSULE_CONTENT_ROOT = (
    "9dcd2022ad8550bf597ee3614f3052699216e2fe63c3a57e67e3bafb0b1eeba8"
)
EXPECTED_PLAN_SLICE_SOURCE_SHA256 = (
    "012cd2a52794a74d966223d50c1e7c645e5aeb150f4326bde531c8cf81f29576"
)
WORK_CAPS = (30_000, 30_000, 80_000)

PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
DESIRED_BUILD_INPUT_COUNT = 0
PROGRAM_INPUT_COUNT = 0
TASK_INPUT_COUNT = 0
FAMILY_INPUT_COUNT = 0
HASH_EVIDENCE_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
V2_TARGET_READ_COUNT = 0
LEARNER_PROTOCOL_READ_COUNT = 0
LEARNER_FIXTURE_READ_COUNT = 0

_NAMESPACE = b"packed-public-train-fit-occurrence-factor-plan-slice-v1\0"
_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_FACTOR_KINDS = ("literal", "role")
_RELATION_OPERATORS = ("eq", "ne", "lt", "le", "gt", "ge")
_RHS_SLOT = 0
_RHS_VALUE = 1


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


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


DEPENDENCY_SHA256 = _digest(
    "dependency",
    {
        "source_protocol_sha256": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "source_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
        "static_capsule_bank_commit": STATIC_CAPSULE_BANK_COMMIT,
        "static_capsule_source_sha256": EXPECTED_STATIC_CAPSULE_SOURCE_SHA256,
        "static_capsule_artifact_sha256": EXPECTED_STATIC_CAPSULE_ARTIFACT_SHA256,
        "static_capsule_content_root": EXPECTED_STATIC_CAPSULE_CONTENT_ROOT,
        "plan_slice_source_sha256": EXPECTED_PLAN_SLICE_SOURCE_SHA256,
        "work_caps": WORK_CAPS,
        "teacher_runtime_imports": (),
    },
)


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
    dependency_sha256: str
    source_certificate_sha256: str
    source_view_sha256: str
    source_checkpoint_sha256: str
    source_protocol_sha256: str
    physical_work_receipt_sha256: str
    structural_certificate_sha256: str
    work: ConverterWorkAudit
    sha256: str


class PublicTrainFitOccurrenceAuthorizedBuildState(NamedTuple):
    status: str
    stop_reason: str | None
    dependency_sha256: str
    source_certificate_sha256: str | None
    source_view_sha256: str | None
    source_checkpoint_sha256: str | None
    source_protocol_sha256: str | None
    physical_work_receipt_sha256: str | None
    converter_receipt: ConverterReceiptSummary | None
    structural_certificate_sha256: str | None
    occurrence_count: int
    structural_freeze_attempts: int
    plan_build_attempts: int
    plan_slice_state_sha256: str | None
    desired_reads_before_select: int
    proof_witnesses_before_select: int
    sha256: str


class PublicTrainFitOccurrenceAuthorizedSelection(NamedTuple):
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
    source_certificate: source.OccurrenceFactorStaticCertificate
    source_view: source.OccurrenceFactorRelayView
    structural_certificate: plan_slice.StructuralFactorCertificate
    summary: ConverterReceiptSummary


_CONVERTER_AUTHORITY = object()
_ISSUED_CONVERSIONS: dict[int, _IssuedConversion] = {}


def _work_payload(work: ConverterWorkAudit) -> dict[str, object]:
    return {
        field: getattr(work, field)
        for field in work._fields
        if field != "sha256"
    }


def _summary_payload(summary: ConverterReceiptSummary) -> dict[str, object]:
    return {
        "dependency_sha256": summary.dependency_sha256,
        "source_certificate_sha256": summary.source_certificate_sha256,
        "source_view_sha256": summary.source_view_sha256,
        "source_checkpoint_sha256": summary.source_checkpoint_sha256,
        "source_protocol_sha256": summary.source_protocol_sha256,
        "physical_work_receipt_sha256": summary.physical_work_receipt_sha256,
        "structural_certificate_sha256": summary.structural_certificate_sha256,
        "work_sha256": summary.work.sha256,
    }


def _state_payload(
    state: PublicTrainFitOccurrenceAuthorizedBuildState,
) -> dict[str, object]:
    return {
        "status": state.status,
        "stop_reason": state.stop_reason,
        "dependency_sha256": state.dependency_sha256,
        "source_certificate_sha256": state.source_certificate_sha256,
        "source_view_sha256": state.source_view_sha256,
        "source_checkpoint_sha256": state.source_checkpoint_sha256,
        "source_protocol_sha256": state.source_protocol_sha256,
        "physical_work_receipt_sha256": state.physical_work_receipt_sha256,
        "converter_receipt_sha256": (
            None if state.converter_receipt is None else state.converter_receipt.sha256
        ),
        "structural_certificate_sha256": state.structural_certificate_sha256,
        "occurrence_count": state.occurrence_count,
        "structural_freeze_attempts": state.structural_freeze_attempts,
        "plan_build_attempts": state.plan_build_attempts,
        "plan_slice_state_sha256": state.plan_slice_state_sha256,
        "desired_reads_before_select": state.desired_reads_before_select,
        "proof_witnesses_before_select": state.proof_witnesses_before_select,
    }


def _new_state(**values: object) -> PublicTrainFitOccurrenceAuthorizedBuildState:
    partial = PublicTrainFitOccurrenceAuthorizedBuildState(sha256="", **values)
    return partial._replace(
        sha256=_digest("authorized-build-state", _state_payload(partial))
    )


def _require_dependencies(view: source.OccurrenceFactorRelayView) -> None:
    caps = (
        plan_slice.MAX_RECIPE_RELATION_WORK,
        plan_slice.MAX_PHYSICAL_PREDICTIONS,
        plan_slice.MAX_PHYSICAL_RACES,
    )
    if (
        view.protocol_sha256 != EXPECTED_SOURCE_PROTOCOL_SHA256
        or view.checkpoint_sha256 != EXPECTED_SOURCE_CHECKPOINT_SHA256
        or source.ARTIFACT_SHA256 != EXPECTED_STATIC_CAPSULE_ARTIFACT_SHA256
        or source.ARTIFACT_CONTENT_ROOT != EXPECTED_STATIC_CAPSULE_CONTENT_ROOT
        or _file_sha256(Path(source.__file__).resolve())
        != EXPECTED_STATIC_CAPSULE_SOURCE_SHA256
        or caps != WORK_CAPS
        or _file_sha256(Path(plan_slice.__file__).resolve())
        != EXPECTED_PLAN_SLICE_SOURCE_SHA256
    ):
        raise _ConversionStop(
            STOP_SOURCE_CERTIFICATE,
            "static source, packed plan-slice, or work-cap dependency changed",
        )


def _decode_int_value(cells: tuple[int, ...]) -> int:
    if (
        type(cells) is not tuple
        or len(cells) != 2
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


def _signature_type_map(
    view: source.OccurrenceFactorRelayView,
    signature: TypedSignature,
) -> Mapping[tuple[int, ...], str]:
    if not (
        type(signature) is TypedSignature
        and signature.result == _INT
        and len(signature.parameters) == 1
        and signature.parameters[0].type == _LIST_INT
    ):
        raise _ConversionStop(
            STOP_SIGNATURE_MISMATCH,
            "ordinary packed signature is not List[Int] -> Int",
        )
    rows: dict[str, tuple[int, ...]] = {}
    for relay in view.signature_relays:
        if (
            relay.root_carrier_cells != view.root_carrier_cells
            or relay.signature_position != 0
            or relay.signature_role not in {"parameter", "return"}
            or relay.signature_role in rows
            or not relay.physical_type_cells
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "signature relays are not two unique root-keyed physical types",
            )
        expected_role_cell = 0 if relay.signature_role == "parameter" else 1
        if relay.signature_role_cell != expected_role_cell:
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "signature role text disagrees with its physical role cell",
            )
        rows[relay.signature_role] = relay.physical_type_cells
    if set(rows) != {"parameter", "return"} or rows["parameter"] == rows["return"]:
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "learned List[Int]/Int signature type relays are incomplete",
        )
    return {rows["parameter"]: "List[Int]", rows["return"]: "Int"}


def _ordered_carriers(
    view: source.OccurrenceFactorRelayView,
) -> tuple[
    tuple[tuple[int, ...], ...],
    Mapping[tuple[int, ...], tuple[tuple[int, tuple[int, ...]], ...]],
    Mapping[tuple[int, ...], tuple[tuple[int, ...], int, tuple[int, ...]]],
]:
    carriers = tuple(relay.carrier_cells for relay in view.source_relays)
    carrier_set = set(carriers)
    if (
        len(carriers) not in (8, 9)
        or len(carrier_set) != len(carriers)
        or carrier_set != set(view.occurrence_carriers)
        or view.root_carrier_cells not in carrier_set
    ):
        raise _ConversionStop(
            STOP_RELAY_UNSUPPORTED,
            "source relays are not one retained 8/9-node physical shape",
        )

    mutable_children: dict[
        tuple[int, ...], list[tuple[int, tuple[int, ...]]]
    ] = defaultdict(list)
    parent_by_child: dict[
        tuple[int, ...], tuple[tuple[int, ...], int, tuple[int, ...]]
    ] = {}
    edge_keys: set[tuple[tuple[int, ...], int]] = set()
    for edge in view.edges:
        key = (edge.parent_carrier_cells, edge.child_position)
        if (
            key in edge_keys
            or edge.child_carrier_cells in parent_by_child
            or edge.parent_carrier_cells not in carrier_set
            or edge.child_carrier_cells not in carrier_set
            or edge.parent_carrier_cells == edge.child_carrier_cells
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "edge relays repeat a parent position or child carrier",
            )
        edge_keys.add(key)
        mutable_children[edge.parent_carrier_cells].append(
            (edge.child_position, edge.child_carrier_cells)
        )
        parent_by_child[edge.child_carrier_cells] = (
            edge.parent_carrier_cells,
            edge.child_position,
            edge.parent_factor_slot_occupancy,
        )
    if len(view.edges) != len(carriers) - 1:
        raise _ConversionStop(STOP_RELAY_UNSUPPORTED, "edge relay count is incomplete")
    if view.root_carrier_cells in parent_by_child or any(
        carrier not in parent_by_child
        for carrier in carriers
        if carrier != view.root_carrier_cells
    ):
        raise _ConversionStop(
            STOP_RELAY_AMBIGUOUS,
            "physical carrier relays do not form one rooted tree",
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
                "physical occurrence traversal contains a cycle",
            )
        visiting.add(carrier)
        ordered.append(carrier)
        for _position, child in children.get(carrier, ()):
            visit(child)
        visiting.remove(carrier)

    visit(view.root_carrier_cells)
    if set(ordered) != carrier_set:
        raise _ConversionStop(
            STOP_RELAY_AMBIGUOUS,
            "physical occurrence traversal is disconnected",
        )
    return tuple(ordered), children, parent_by_child


def _convert_ready_view(
    certificate: source.OccurrenceFactorStaticCertificate,
    view: source.OccurrenceFactorRelayView,
    signature: TypedSignature,
) -> _IssuedConversion:
    _require_dependencies(view)
    type_names = _signature_type_map(view, signature)
    ordered, _children, parent_by_child = _ordered_carriers(view)
    occurrence_index = {carrier: index for index, carrier in enumerate(ordered)}

    factor_rows: dict[tuple[int, ...], list[object]] = defaultdict(list)
    for factor in view.factor_slots:
        if (
            factor.carrier_cells not in occurrence_index
            or not 0 <= factor.factor_kind_cell < len(_FACTOR_KINDS)
            or factor.factor_kind != _FACTOR_KINDS[factor.factor_kind_cell]
            or factor.physical_type_cells not in type_names
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "factor slot lacks one learned carrier/kind/type identity",
            )
        factor_rows[factor.carrier_cells].append(factor)

    frozen_factors: dict[tuple[int, ...], tuple[tuple[str, str], ...]] = {}
    for carrier in ordered:
        rows = tuple(sorted(factor_rows.get(carrier, ()), key=lambda row: row.local_slot))
        if not rows or tuple(row.local_slot for row in rows) != tuple(range(len(rows))):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "factor slots are not one complete ordered carrier tuple",
            )
        frozen_factors[carrier] = tuple(
            (_FACTOR_KINDS[row.factor_kind_cell], type_names[row.physical_type_cells])
            for row in rows
        )

    relation_rows: dict[
        tuple[int, ...], list[plan_slice.StructuralRelationConstraint]
    ] = defaultdict(list)
    relation_keys: set[tuple[tuple[int, ...], int]] = set()
    for relation in view.relations:
        carrier = relation.carrier_cells
        key = (carrier, relation.left_slot)
        local_factors = frozen_factors.get(carrier, ())
        if (
            key in relation_keys
            or not 0 <= relation.left_slot < len(local_factors)
            or not 0 <= relation.operator_cell < len(_RELATION_OPERATORS)
            or relation.operator != _RELATION_OPERATORS[relation.operator_cell]
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "relation lacks one carrier/slot/operator identity",
            )
        relation_keys.add(key)
        if relation.rhs_kind_cell == _RHS_SLOT:
            if (
                type(relation.right_slot) is not int
                or not 0 <= relation.right_slot < len(local_factors)
                or relation.right_value_event_cells
                or relation.right_value_type_cells
            ):
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "slot relation contains conflicting right-hand evidence",
                )
            right_slot = relation.right_slot
            right_value = None
        elif relation.rhs_kind_cell == _RHS_VALUE:
            if (
                relation.right_slot is not None
                or type_names.get(relation.right_value_type_cells) != "Int"
            ):
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "value relation lacks its learned Int type relay",
                )
            right_slot = None
            right_value = _decode_int_value(relation.right_value_event_cells)
        else:
            raise _ConversionStop(STOP_RELAY_UNSUPPORTED, "relation RHS kind is unknown")
        relation_rows[carrier].append(
            plan_slice.StructuralRelationConstraint(
                relation.left_slot,
                _RELATION_OPERATORS[relation.operator_cell],
                right_slot,
                right_value,
            )
        )

    domain_keys: set[tuple[tuple[int, ...], int]] = set()
    domains: list[plan_slice.StructuralDomainConstraint] = []
    for domain in view.domains:
        carrier = domain.carrier_cells
        key = (carrier, domain.local_slot)
        local_factors = frozen_factors.get(carrier, ())
        if (
            key in domain_keys
            or not 0 <= domain.local_slot < len(local_factors)
            or local_factors[domain.local_slot] != ("literal", "Int")
            or type_names.get(domain.physical_type_cells) != "Int"
            or not domain.complete_allowed_value_event_cells
        ):
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "literal domain is not one complete carrier-keyed Int slot",
            )
        domain_keys.add(key)
        domains.append(
            plan_slice.StructuralDomainConstraint(
                occurrence_index[carrier],
                domain.local_slot,
                tuple(
                    sorted(
                        {
                            _decode_int_value(event_cells)
                            for event_cells in domain.complete_allowed_value_event_cells
                        }
                    )
                ),
            )
        )

    copy_groups: list[tuple[tuple[int, int], ...]] = []
    seen_group_cells: set[tuple[int, ...]] = set()
    seen_members: set[tuple[int, int]] = set()
    copy_member_reads = 0
    for group in view.literal_copy_groups:
        if not group.copy_group_cells or group.copy_group_cells in seen_group_cells:
            raise _ConversionStop(
                STOP_RELAY_AMBIGUOUS,
                "literal-copy group identity is empty or repeated",
            )
        seen_group_cells.add(group.copy_group_cells)
        converted: list[tuple[int, int]] = []
        for carrier, local_slot in group.members:
            copy_member_reads += 1
            local_factors = frozen_factors.get(carrier, ())
            if (
                carrier not in occurrence_index
                or not 0 <= local_slot < len(local_factors)
                or local_factors[local_slot] != ("literal", "Int")
            ):
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "literal-copy member lacks one carrier-keyed literal slot",
                )
            member = (occurrence_index[carrier], local_slot)
            if member in seen_members:
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "literal-copy member appears in multiple physical groups",
                )
            seen_members.add(member)
            converted.append(member)
        if not converted:
            raise _ConversionStop(STOP_RELAY_ZERO, "literal-copy group is empty")
        copy_groups.append(tuple(sorted(converted)))

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
                or any(
                    not 0 <= slot < len(frozen_factors[parent_carrier])
                    for slot in occupancy
                )
            ):
                raise _ConversionStop(
                    STOP_RELAY_AMBIGUOUS,
                    "edge occupancy does not bind the complete child factor tuple",
                )
        constraints = tuple(
            sorted(
                relation_rows.get(carrier, ()),
                key=lambda row: (
                    row.left_slot,
                    _RELATION_OPERATORS.index(row.operator),
                    -1 if row.right_slot is None else row.right_slot,
                    -1 if row.right_value is None else row.right_value,
                ),
            )
        )
        occurrences.append(
            plan_slice.StructuralOccurrence(
                parent_occurrence,
                child_position,
                frozen_factors[carrier],
                occupancy,
                constraints,
            )
        )

    try:
        structural = plan_slice.freeze_structural_factor_certificate(
            occurrences,
            domains,
            copy_groups,
        )
    except ValueError as error:
        raise _ConversionStop(STOP_RELAY_AMBIGUOUS, str(error)) from error

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
        len(ordered),
        len(view.source_relays),
        len(view.edges),
        len(view.factor_slots),
        len(view.relations),
        len(view.domains),
        len(view.literal_copy_groups),
        copy_member_reads,
        len(view.signature_relays),
        physical_reads,
        1,
        "",
    )
    work = work_partial._replace(
        sha256=_digest("converter-work", _work_payload(work_partial))
    )
    summary_partial = ConverterReceiptSummary(
        DEPENDENCY_SHA256,
        certificate.sha256,
        view.sha256,
        view.checkpoint_sha256,
        view.protocol_sha256,
        view.physical_work_receipt_sha256,
        structural.sha256,
        work,
        "",
    )
    summary = summary_partial._replace(
        sha256=_digest("converter-receipt", _summary_payload(summary_partial))
    )
    issued = _IssuedConversion(
        _CONVERTER_AUTHORITY,
        certificate,
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
        or source.require_occurrence_factors(issued.source_certificate)
        is not issued.source_view
        or issued.summary.dependency_sha256 != DEPENDENCY_SHA256
        or issued.summary.source_certificate_sha256 != issued.source_certificate.sha256
        or issued.summary.source_view_sha256 != issued.source_view.sha256
        or issued.summary.source_checkpoint_sha256 != issued.source_view.checkpoint_sha256
        or issued.summary.source_protocol_sha256 != issued.source_view.protocol_sha256
        or issued.summary.physical_work_receipt_sha256
        != issued.source_view.physical_work_receipt_sha256
        or issued.summary.structural_certificate_sha256
        != issued.structural_certificate.sha256
        or issued.summary.work.sha256
        != _digest("converter-work", _work_payload(issued.summary.work))
        or issued.summary.sha256
        != _digest("converter-receipt", _summary_payload(issued.summary))
    ):
        raise RuntimeError("private static occurrence-factor conversion changed")


class PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1:
    """Deep static-certificate build -> desired-only select integration."""

    __slots__ = ("_conversion", "_frontier", "_source_certificate", "_state")

    def __init__(self) -> None:
        raise TypeError(
            "use PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build"
        )

    @classmethod
    def _stopped(
        cls,
        status: str,
        reason: str,
        *,
        source_certificate: object | None = None,
        source_view: source.OccurrenceFactorRelayView | None = None,
    ) -> PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1:
        source_sha = (
            source_certificate.sha256
            if type(source_certificate) is source.OccurrenceFactorStaticCertificate
            else None
        )
        state = _new_state(
            status=status,
            stop_reason=reason,
            dependency_sha256=DEPENDENCY_SHA256,
            source_certificate_sha256=source_sha,
            source_view_sha256=None if source_view is None else source_view.sha256,
            source_checkpoint_sha256=(
                None if source_view is None else source_view.checkpoint_sha256
            ),
            source_protocol_sha256=(
                None if source_view is None else source_view.protocol_sha256
            ),
            physical_work_receipt_sha256=(
                None if source_view is None else source_view.physical_work_receipt_sha256
            ),
            converter_receipt=None,
            structural_certificate_sha256=None,
            occurrence_count=0,
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
        source_certificate: source.OccurrenceFactorStaticCertificate,
    ) -> PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1:
        try:
            view = source.require_occurrence_factors(source_certificate)
        except (TypeError, RuntimeError) as error:
            status = STOP_SOURCE_CERTIFICATE
            if type(source_certificate) is source.OccurrenceFactorStaticCertificate:
                status = {
                    source.STOP_ZERO: STOP_RELAY_ZERO,
                    source.STOP_AMBIGUOUS: STOP_RELAY_AMBIGUOUS,
                    source.STOP_UNSUPPORTED: STOP_RELAY_UNSUPPORTED,
                }.get(source_certificate.status, STOP_SOURCE_CERTIFICATE)
            return cls._stopped(
                status,
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
            dependency_sha256=DEPENDENCY_SHA256,
            source_certificate_sha256=source_certificate.sha256,
            source_view_sha256=view.sha256,
            source_checkpoint_sha256=view.checkpoint_sha256,
            source_protocol_sha256=view.protocol_sha256,
            physical_work_receipt_sha256=view.physical_work_receipt_sha256,
            converter_receipt=conversion.summary,
            structural_certificate_sha256=conversion.structural_certificate.sha256,
            occurrence_count=len(view.occurrence_carriers),
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
    def state(self) -> PublicTrainFitOccurrenceAuthorizedBuildState:
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
            "authorized-build-state", _state_payload(self._state)
        ):
            raise RuntimeError("static occurrence-authorized integration state changed")
        if self._conversion is None:
            if self._frontier is not None or self._state.plan_build_attempts != 0:
                raise RuntimeError("stopped integration retained a plan frontier")
            return
        _authenticate_conversion(self._conversion)
        if (
            self._frontier is None
            or self._state.source_certificate_sha256
            != self._conversion.source_certificate.sha256
            or self._state.converter_receipt is not self._conversion.summary
            or self._state.structural_certificate_sha256
            != self._conversion.structural_certificate.sha256
            or self._state.plan_slice_state_sha256 != self._frontier.sha256
        ):
            raise RuntimeError("static occurrence-authorized binding changed")

    def _selection(
        self,
        status: str,
        delegated: plan_slice.PackedPromptPlanSliceSelection | None = None,
    ) -> PublicTrainFitOccurrenceAuthorizedSelection:
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
            "plan_slice_selection_sha256": None if delegated is None else delegated.sha256,
            "selected_eclass_id": None if delegated is None else delegated.selected_eclass_id,
            "exact_proof_count": 0 if delegated is None else delegated.exact_proof_count,
            "witness_sha256s": tuple(getattr(item, "sha256", None) for item in witnesses),
            "selected_proof_forest_sha256": (
                None if delegated is None else delegated.selected_proof_forest_sha256
            ),
            "work_sha256": None if delegated is None else delegated.work.sha256,
        }
        return PublicTrainFitOccurrenceAuthorizedSelection(
            status,
            self._state.sha256,
            self._state.source_certificate_sha256,
            (
                None
                if self._state.converter_receipt is None
                else self._state.converter_receipt.sha256
            ),
            None if delegated is None else delegated.sha256,
            None if delegated is None else delegated.selected_eclass_id,
            0 if delegated is None else delegated.exact_proof_count,
            witnesses,
            None if delegated is None else delegated.selected_proof_forest_sha256,
            None if delegated is None else delegated.work,
            _digest("authorized-selection", payload),
        )

    def select(
        self,
        desired: Sequence[object],
    ) -> PublicTrainFitOccurrenceAuthorizedSelection:
        self._authenticate()
        if self._frontier is None:
            return self._selection(self._state.status)
        delegated = self._frontier.select(desired)
        self._authenticate()
        return self._selection(delegated.status, delegated)


CLAIM_BOUNDARY = (
    "public-TRAIN same-FIT static learned occurrence/factor relay to private packed "
    "plan authorization only; raw-prompt adaptation, canonical panel selection, "
    "heldout, DEV, M0, C1, Path-2, G0, coding benchmarks, and broad capability "
    "remain unproven"
)


__all__ = [
    "CANONICAL_PANEL_SPEND_COUNT",
    "CLAIM_BOUNDARY",
    "DEPENDENCY_SHA256",
    "DESIRED_BUILD_INPUT_COUNT",
    "DEV_INPUT_COUNT",
    "EXPECTED_STATIC_CAPSULE_ARTIFACT_SHA256",
    "EXPECTED_STATIC_CAPSULE_CONTENT_ROOT",
    "EXPECTED_STATIC_CAPSULE_SOURCE_SHA256",
    "G0_INPUT_COUNT",
    "HELDOUT_INPUT_COUNT",
    "LEARNER_FIXTURE_READ_COUNT",
    "LEARNER_PROTOCOL_READ_COUNT",
    "M0_INPUT_COUNT",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "READY",
    "SCHEMA",
    "SELECTED",
    "STATIC_CAPSULE_BANK_COMMIT",
    "STOP_RELAY_AMBIGUOUS",
    "STOP_RELAY_UNSUPPORTED",
    "STOP_RELAY_ZERO",
    "STOP_SIGNATURE_MISMATCH",
    "STOP_SOURCE_CERTIFICATE",
    "V2_TARGET_READ_COUNT",
    "WORK_CAPS",
    "ConverterReceiptSummary",
    "ConverterWorkAudit",
    "PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1",
    "PublicTrainFitOccurrenceAuthorizedBuildState",
    "PublicTrainFitOccurrenceAuthorizedSelection",
]
