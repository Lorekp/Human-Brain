"""Private canonical prompt/factor attention to plan-slice authority.

The public build seam accepts only raw prompt observations and the ordinary
generic packed build inputs.  It executes and authenticates the static
prompt/factor attention handoff, privately recovers the exact prompt
certificate for the same input, and refuses to manufacture the forgeable
``StructuralFactorCertificate`` accepted by the self-posed plan-slice layer.

The present prompt checkpoint does not publish an occurrence-local carrier
join.  Its edge bindings are edge-local, its role/slot/copy winners have no
shared node-occurrence identity, the attention handoff retains only global
factor occupancy, and typed literal events are not bound to literal
occurrences.  Therefore this first canonical authority stops before any
structural-certificate freeze or packed build.  The opaque handoff already
owns the later ``select(desired)`` seam; only a future READY receipt can
delegate to the ordinary visible-I/O race.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

from experiments.r3_pilot import canonical_train_prompt_factor_attention_v1 as attention
from experiments.r3_pilot import (
    public_train_factor_workspace_v4_static_checkpoint_capsule_v1 as factor_checkpoint,
)
from experiments.r3_pilot import (
    public_train_prompt_selection_v4_static_checkpoint_capsule_v1 as prompt_checkpoint,
)
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "canonical-train-prompt-factor-plan-slice-authority-v1"
READY = "CANONICAL_TRAIN_PROMPT_FACTOR_PLAN_SLICE_AUTHORITY_V1_READY"
STOP_ATTENTION = (
    "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_PLAN_SLICE_AUTHORITY_V1_ATTENTION"
)
STOP_IDENTITY = (
    "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_PLAN_SLICE_AUTHORITY_V1_IDENTITY"
)
STOP_SIGNATURE_ROLE = (
    "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_PLAN_SLICE_AUTHORITY_V1_SIGNATURE_ROLE"
)
STOP_OCCURRENCE_LOCAL_CARRIER_JOIN = (
    "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_PLAN_SLICE_AUTHORITY_V1_"
    "OCCURRENCE_LOCAL_CARRIER_JOIN"
)

MISSING_OCCURRENCE_RELAYS = (
    "stable-node-occurrence-identity-across-edge-role-slot-copy",
    "occurrence-local-selected-factor-span-selector-geometry",
    "occurrence-bound-typed-literal-event-identity",
    "occurrence-local-packed-factor-slot-schema-occupancy-and-relations",
)

_NAMESPACE = b"canonical-train-prompt-factor-plan-slice-authority-v1\0"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
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


@dataclass(frozen=True, slots=True)
class OccurrenceLocalCarrierJoinAudit:
    ordered_edge_binding_count: int
    unique_edge_endpoint_count: int
    reused_edge_endpoint_count: int
    non_edge_role_slot_copy_event_count: int
    bound_non_edge_role_slot_copy_event_count: int
    typed_literal_event_count: int
    occurrence_bound_typed_literal_event_count: int
    ready_factor_observation_count: int
    retained_factor_observation_identity_count: int
    packed_factor_slot_schema_count: int
    missing_relays: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalTrainPlanSliceAuthorityState:
    status: str
    stop_reason: str | None
    raw_prompt_input_root_sha256: str
    packed_build_input_root_sha256: str
    joint_input_root_sha256: str
    attention_receipt_sha256: str
    attention_status: str
    prompt_checkpoint_sha256: str
    prompt_artifact_sha256: str
    prompt_artifact_content_root_sha256: str
    prompt_output_registry_sha256: str
    prompt_factor_connector_sha256: str
    prompt_input_sha256: str
    prompt_actual_winner_sha256: str
    prompt_certificate_sha256: str
    factor_checkpoint_sha256: str
    factor_artifact_sha256: str
    factor_membership: tuple[str, ...]
    factor_scan_sha256: str
    signature_role_agreement: bool
    prompt_factor_agreement: bool
    occurrence_join: OccurrenceLocalCarrierJoinAudit
    structural_conversion_candidate_count: int
    structural_certificate_freeze_attempt_count: int
    plan_slice_build_attempt_count: int
    plan_slice_receipt_sha256: str | None
    authenticated_eclass_id: str | None
    eclass_receipt_identity_verified: bool
    attention_inference_attempt_count: int
    prompt_recovery_attempt_count: int
    factor_inference_attempt_count: int
    desired_input_count_during_build: int
    desired_value_read_count_during_build: int
    caller_factor_observation_count: int
    caller_structural_certificate_count: int
    target_input_count: int
    expected_input_count: int
    program_input_count: int
    task_input_count: int
    family_input_count: int
    heldout_input_count: int
    protected_input_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalTrainPlanSliceSelection:
    status: str
    stop_reason: str | None
    authority_handoff_sha256: str
    packed_selection: plan_slice.PackedPromptPlanSliceSelection | None
    desired_value_read_count: int
    sha256: str


class _AuthorityReceipt(NamedTuple):
    authority: object
    state: CanonicalTrainPlanSliceAuthorityState
    attention_handoff: attention.PromptFactorAttentionHandoff
    prompt_certificate: prompt_checkpoint.PromptStructuralCertificate
    frontier: plan_slice.PackedGenericTypedPromptPlanSliceV1 | None
    plan_receipt: plan_slice.PlanSliceReceipt | None
    sha256: str


def _join_payload(value: OccurrenceLocalCarrierJoinAudit) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value.__dataclass_fields__
        if field != "sha256"
    }


def _state_payload(value: CanonicalTrainPlanSliceAuthorityState) -> dict[str, object]:
    return {
        field: (
            value.occurrence_join.sha256
            if field == "occurrence_join"
            else getattr(value, field)
        )
        for field in value.__dataclass_fields__
        if field != "sha256"
    }


def _receipt_payload(value: _AuthorityReceipt) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "state_sha256": value.state.sha256,
        "attention_receipt_sha256": value.attention_handoff.sha256,
        "prompt_certificate_sha256": value.prompt_certificate.sha256,
        "plan_slice_receipt_sha256": (
            None if value.plan_receipt is None else value.plan_receipt.sha256
        ),
    }


def _make_authority() -> tuple[
    Callable[..., _AuthorityReceipt],
    Callable[[_AuthorityReceipt], bool],
]:
    authority = object()
    issued: dict[int, _AuthorityReceipt] = {}

    def issue(
        *,
        state: CanonicalTrainPlanSliceAuthorityState,
        attention_handoff: attention.PromptFactorAttentionHandoff,
        prompt_certificate: prompt_checkpoint.PromptStructuralCertificate,
        frontier: plan_slice.PackedGenericTypedPromptPlanSliceV1 | None,
        plan_receipt: plan_slice.PlanSliceReceipt | None,
    ) -> _AuthorityReceipt:
        partial = _AuthorityReceipt(
            authority,
            state,
            attention_handoff,
            prompt_certificate,
            frontier,
            plan_receipt,
            "",
        )
        receipt = partial._replace(
            sha256=_digest("private-authority-receipt", _receipt_payload(partial))
        )
        issued[id(receipt)] = receipt
        return receipt

    def authenticate(value: _AuthorityReceipt) -> bool:
        if (
            type(value) is not _AuthorityReceipt
            or issued.get(id(value)) is not value
            or value.authority is not authority
            or value.state.occurrence_join.sha256
            != _digest("occurrence-local-carrier-join", _join_payload(value.state.occurrence_join))
            or value.state.sha256 != _digest("authority-state", _state_payload(value.state))
            or value.sha256
            != _digest("private-authority-receipt", _receipt_payload(value))
            or not attention._authenticate_attention_receipt(
                value.attention_handoff._receipt
            )
            or value.attention_handoff._receipt.state is not value.attention_handoff.state
            or value.prompt_certificate.sha256
            != value.state.prompt_certificate_sha256
            or value.prompt_certificate.input_sha256 != value.state.prompt_input_sha256
            or value.prompt_certificate.actual_winner_sha256
            != value.state.prompt_actual_winner_sha256
        ):
            return False
        if value.state.status == READY:
            return bool(
                value.frontier is not None
                and value.plan_receipt is not None
                and value.frontier.status == plan_slice.FRONTIER_READY
                and value.frontier.plan_slice_receipt is value.plan_receipt
                and value.state.plan_slice_receipt_sha256 == value.plan_receipt.sha256
                and value.state.authenticated_eclass_id
                == value.plan_receipt.authenticated_eclass_id
                and value.state.eclass_receipt_identity_verified
            )
        return bool(value.frontier is None and value.plan_receipt is None)

    return issue, authenticate


_issue_authority_receipt, _authenticate_authority_receipt = _make_authority()


class CanonicalTrainPromptFactorPlanSliceHandoff:
    """Opaque build authority; desired values enter only through ``select``."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _AuthorityReceipt) -> None:
        if not _authenticate_authority_receipt(receipt):
            raise TypeError("canonical plan-slice handoff requires a private receipt")
        self._receipt = receipt

    @property
    def status(self) -> str:
        return self._receipt.state.status

    @property
    def state(self) -> CanonicalTrainPlanSliceAuthorityState:
        return self._receipt.state

    @property
    def sha256(self) -> str:
        return self._receipt.sha256

    def select(self, desired: Sequence[object]) -> CanonicalTrainPlanSliceSelection:
        """Delegate only a READY identity-bound receipt to the visible-I/O race."""

        if not _authenticate_authority_receipt(self._receipt):
            raise RuntimeError("canonical plan-slice authority authentication failed")
        if self._receipt.state.status != READY:
            partial = CanonicalTrainPlanSliceSelection(
                status=self._receipt.state.status,
                stop_reason=self._receipt.state.stop_reason,
                authority_handoff_sha256=self.sha256,
                packed_selection=None,
                desired_value_read_count=0,
                sha256="",
            )
            return CanonicalTrainPlanSliceSelection(
                **{
                    field: getattr(partial, field)
                    for field in partial.__dataclass_fields__
                    if field != "sha256"
                },
                sha256=_digest(
                    "selection",
                    {
                        "status": partial.status,
                        "stop_reason": partial.stop_reason,
                        "authority_handoff_sha256": partial.authority_handoff_sha256,
                        "packed_selection_sha256": None,
                        "desired_value_read_count": 0,
                    },
                ),
            )
        frontier = self._receipt.frontier
        plan_receipt = self._receipt.plan_receipt
        if frontier is None or plan_receipt is None:
            raise RuntimeError("READY authority lost its exact plan-slice objects")
        packed_selection = frontier.select(desired)
        if (
            packed_selection.plan_slice_receipt_sha256 != plan_receipt.sha256
            or (
                packed_selection.status == plan_slice.SELECTED
                and packed_selection.selected_eclass_id
                != plan_receipt.authenticated_eclass_id
            )
        ):
            raise RuntimeError("visible-I/O winner disagreed with plan-slice authority")
        partial = CanonicalTrainPlanSliceSelection(
            status=packed_selection.status,
            stop_reason=None,
            authority_handoff_sha256=self.sha256,
            packed_selection=packed_selection,
            desired_value_read_count=1,
            sha256="",
        )
        return CanonicalTrainPlanSliceSelection(
            status=partial.status,
            stop_reason=None,
            authority_handoff_sha256=partial.authority_handoff_sha256,
            packed_selection=packed_selection,
            desired_value_read_count=1,
            sha256=_digest(
                "selection",
                {
                    "status": partial.status,
                    "stop_reason": None,
                    "authority_handoff_sha256": partial.authority_handoff_sha256,
                    "packed_selection_sha256": packed_selection.sha256,
                    "desired_value_read_count": 1,
                },
            ),
        )


def _signature_role_agrees(
    signature_role_record: Mapping[str, object],
    signature: TypedSignature,
) -> bool:
    record = _plain(signature_role_record)
    if not isinstance(record, dict):
        return False
    parameters = record.get("parameters")
    if not isinstance(parameters, list) or len(parameters) != len(signature.parameters):
        return False
    if record.get("entrypoint_spelling_present") is not False:
        return False
    if record.get("parameter_spelling_present") is not False:
        return False
    if record.get("role_binding") != "reverse-declaration-depth":
        return False
    for index, (supplied, parameter) in enumerate(
        zip(parameters, signature.parameters, strict=True)
    ):
        if not isinstance(supplied, dict) or supplied != {
            "role": f"ARG{len(signature.parameters) - index - 1}",
            "type": _plain(parameter.type.to_record()),
        }:
            return False
    return record.get("result") == _plain(signature.result.to_record())


def _prompt_factor_agrees(state: attention.PromptFactorAttentionState) -> bool:
    coincidence = tuple(
        prompt_support and factor_support
        for prompt_support, factor_support in zip(
            state.prompt_physical_support,
            state.scan.physical_membership,
            strict=True,
        )
    )
    named = tuple(
        factor
        for factor, occupied in zip(
            factor_checkpoint.FACTOR_REGISTRY,
            coincidence,
            strict=True,
        )
        if occupied
    )
    return bool(
        state.projected_factor_membership == state.scan.physical_membership
        and state.physical_coincidence_spikes == coincidence
        and state.factor_membership == named
        and state.membership_count == len(named)
    )


def _occurrence_join_audit(
    certificate: prompt_checkpoint.PromptStructuralCertificate,
    typed_literal_event_count: int,
    attention_state: attention.PromptFactorAttentionState,
) -> OccurrenceLocalCarrierJoinAudit:
    edge_bindings = tuple(
        occurrence.occurrence_binding
        for occurrence in certificate.occurrences
        if occurrence.occurrence_binding is not None
    )
    endpoints = tuple(
        cells
        for binding in edge_bindings
        for cells in (binding.parent_cells, binding.child_cells)
    )
    endpoint_counts = Counter(endpoints)
    role_channels = {
        "argument-operand-orientation",
        "parameter-role-binding",
        "literal-slot-binding",
        "literal-value-copy",
    }
    role_occurrences = tuple(
        occurrence
        for occurrence in certificate.occurrences
        if any(
            key.startswith("atom|") and key.split("|", 2)[1] in role_channels
            for key, _value in occurrence.emitted_bindings
        )
    )
    partial = OccurrenceLocalCarrierJoinAudit(
        ordered_edge_binding_count=len(edge_bindings),
        unique_edge_endpoint_count=len(endpoint_counts),
        reused_edge_endpoint_count=sum(count > 1 for count in endpoint_counts.values()),
        non_edge_role_slot_copy_event_count=len(role_occurrences),
        bound_non_edge_role_slot_copy_event_count=sum(
            occurrence.occurrence_binding is not None
            for occurrence in role_occurrences
        ),
        typed_literal_event_count=typed_literal_event_count,
        occurrence_bound_typed_literal_event_count=0,
        ready_factor_observation_count=attention_state.scan.ready_observation_count,
        retained_factor_observation_identity_count=0,
        packed_factor_slot_schema_count=0,
        missing_relays=MISSING_OCCURRENCE_RELAYS,
        sha256="",
    )
    return OccurrenceLocalCarrierJoinAudit(
        **{
            field: getattr(partial, field)
            for field in partial.__dataclass_fields__
            if field != "sha256"
        },
        sha256=_digest("occurrence-local-carrier-join", _join_payload(partial)),
    )


def issue_canonical_train_plan_slice_authority(
    span_records: Sequence[Mapping[str, object]],
    prompt_literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    signature: TypedSignature,
    cases: Sequence[Sequence[object]],
    literals: Sequence[int],
    max_cost: int,
) -> CanonicalTrainPromptFactorPlanSliceHandoff:
    """Issue an authenticated authority or the exact occurrence-join STOP."""

    if type(signature) is not TypedSignature:
        raise TypeError("generic packed build requires an exact TypedSignature")
    if type(max_cost) is not int or max_cost < 0:
        raise ValueError("generic packed max_cost must be a nonnegative integer")
    if any(type(value) is not int for value in literals):
        raise TypeError("generic packed literals must be integers")

    spans = attention._normalize_spans(span_records)
    prompt_literals = attention._normalize_literals(prompt_literal_records)
    signature_role = _plain(signature_role_record)
    raw_root = _digest(
        "raw-prompt-input",
        {
            "span_records": spans,
            "literal_records": prompt_literals,
            "signature_role_record": signature_role,
        },
    )
    build_root = _digest(
        "generic-packed-build-input",
        {
            "signature": signature.to_record(),
            "cases": cases,
            "literals": tuple(literals),
            "max_cost": max_cost,
        },
    )
    joint_root = _digest(
        "joint-input",
        {"raw_prompt_root": raw_root, "packed_build_root": build_root},
    )

    attention_handoff = attention.infer_prompt_factor_attention(
        spans,
        prompt_literals,
        signature_role,
    )
    if not attention._authenticate_attention_receipt(attention_handoff._receipt):
        raise RuntimeError("private attention receipt authentication failed")
    attention_state = attention_handoff.state
    prompt_certificate = prompt_checkpoint.infer_prompt(
        spans,
        prompt_literals,
        signature_role,
    )
    connector = attention._prompt_factor_connector()
    connector_ok = attention._authenticate_prompt_factor_connector(connector)
    prompt_identity_ok = bool(
        prompt_certificate.checkpoint_sha256
        == attention_state.prompt_checkpoint_sha256
        == prompt_checkpoint.BANKED_CHECKPOINT_SHA256
        and attention_state.prompt_artifact_sha256
        == prompt_checkpoint.EXPECTED_ARTIFACT_SHA256
        and prompt_certificate.input_sha256 == attention_state.prompt_input_sha256
        and prompt_certificate.actual_winner_sha256
        == attention_state.prompt_actual_winner_sha256
        and prompt_certificate.sha256 == attention_state.prompt_certificate_sha256
        and connector_ok
        and connector.checkpoint_sha256 == prompt_certificate.checkpoint_sha256
        and connector.artifact_sha256 == attention_state.prompt_artifact_sha256
        and connector.content_root_sha256
        == prompt_checkpoint.EXPECTED_ARTIFACT_CONTENT_ROOT
        and connector.output_registry_sha256
        == attention_state.prompt_output_registry_sha256
        and connector.sha256 == attention_state.prompt_factor_connector_sha256
    )
    factor_identity_ok = bool(
        attention_state.factor_checkpoint_sha256
        == factor_checkpoint.BANKED_CHECKPOINT_SHA256
        and attention_state.factor_artifact_sha256
        == factor_checkpoint.ARTIFACT_SHA256
    )
    signature_agreement = _signature_role_agrees(signature_role, signature)
    factor_agreement = _prompt_factor_agrees(attention_state)
    join = _occurrence_join_audit(
        prompt_certificate,
        len(prompt_literals),
        attention_state,
    )

    if attention_state.status != attention.ATTENTION_READY:
        status = STOP_ATTENTION
        reason = attention_state.stop_reason or "prompt/factor attention did not become READY"
    elif not prompt_identity_ok or not factor_identity_ok or not factor_agreement:
        status = STOP_IDENTITY
        reason = "prompt, factor, attention, or artifact identity roots disagreed"
    elif not signature_agreement:
        status = STOP_SIGNATURE_ROLE
        reason = "name-free prompt signature roles disagree with the packed signature types"
    else:
        status = STOP_OCCURRENCE_LOCAL_CARRIER_JOIN
        reason = (
            "missing learned occurrence-local carrier join: stable node-occurrence "
            "identity across edge/role/slot/copy events, exact selected factor "
            "span+selector geometry, typed-literal event identity, and packed "
            "factor-slot occupancy/relations"
        )

    partial = CanonicalTrainPlanSliceAuthorityState(
        status=status,
        stop_reason=reason,
        raw_prompt_input_root_sha256=raw_root,
        packed_build_input_root_sha256=build_root,
        joint_input_root_sha256=joint_root,
        attention_receipt_sha256=attention_handoff.sha256,
        attention_status=attention_state.status,
        prompt_checkpoint_sha256=prompt_certificate.checkpoint_sha256,
        prompt_artifact_sha256=attention_state.prompt_artifact_sha256,
        prompt_artifact_content_root_sha256=(
            prompt_checkpoint.EXPECTED_ARTIFACT_CONTENT_ROOT
        ),
        prompt_output_registry_sha256=attention_state.prompt_output_registry_sha256,
        prompt_factor_connector_sha256=(
            attention_state.prompt_factor_connector_sha256
        ),
        prompt_input_sha256=prompt_certificate.input_sha256,
        prompt_actual_winner_sha256=prompt_certificate.actual_winner_sha256,
        prompt_certificate_sha256=prompt_certificate.sha256,
        factor_checkpoint_sha256=attention_state.factor_checkpoint_sha256,
        factor_artifact_sha256=attention_state.factor_artifact_sha256,
        factor_membership=attention_state.factor_membership,
        factor_scan_sha256=attention_state.scan.sha256,
        signature_role_agreement=signature_agreement,
        prompt_factor_agreement=factor_agreement,
        occurrence_join=join,
        structural_conversion_candidate_count=0,
        structural_certificate_freeze_attempt_count=0,
        plan_slice_build_attempt_count=0,
        plan_slice_receipt_sha256=None,
        authenticated_eclass_id=None,
        eclass_receipt_identity_verified=False,
        attention_inference_attempt_count=1,
        prompt_recovery_attempt_count=1,
        factor_inference_attempt_count=(
            attention_state.scan.factor_inference_attempts
        ),
        desired_input_count_during_build=0,
        desired_value_read_count_during_build=0,
        caller_factor_observation_count=0,
        caller_structural_certificate_count=0,
        target_input_count=0,
        expected_input_count=0,
        program_input_count=0,
        task_input_count=0,
        family_input_count=0,
        heldout_input_count=0,
        protected_input_count=0,
        sha256="",
    )
    state = CanonicalTrainPlanSliceAuthorityState(
        **{
            field: getattr(partial, field)
            for field in partial.__dataclass_fields__
            if field != "sha256"
        },
        sha256=_digest("authority-state", _state_payload(partial)),
    )
    receipt = _issue_authority_receipt(
        state=state,
        attention_handoff=attention_handoff,
        prompt_certificate=prompt_certificate,
        frontier=None,
        plan_receipt=None,
    )
    return CanonicalTrainPromptFactorPlanSliceHandoff(receipt)


__all__ = [
    "MISSING_OCCURRENCE_RELAYS",
    "READY",
    "SCHEMA",
    "STOP_ATTENTION",
    "STOP_IDENTITY",
    "STOP_OCCURRENCE_LOCAL_CARRIER_JOIN",
    "STOP_SIGNATURE_ROLE",
    "CanonicalTrainPlanSliceAuthorityState",
    "CanonicalTrainPlanSliceSelection",
    "CanonicalTrainPromptFactorPlanSliceHandoff",
    "OccurrenceLocalCarrierJoinAudit",
    "issue_canonical_train_plan_slice_authority",
]
