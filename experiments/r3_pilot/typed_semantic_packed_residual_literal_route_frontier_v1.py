"""Apply one learned residual-literal route to a desired-free packed build."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple

from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as learner
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "TYPED_SEMANTIC_PACKED_RESIDUAL_LITERAL_ROUTE_FRONTIER_V1"
FRONTIER_READY = plan_slice.FRONTIER_READY
SELECTED = plan_slice.SELECTED

CLAIM_BOUNDARY = (
    "self-posed target-free-at-build composition of a learned residual literal "
    "route with banked repetition and operand-order states, producing one exact "
    "packed witness for unseen typed-Int payloads; destination domain "
    "(occurrence 6, slot 0) and the count/comparison/addition topology remain "
    "explicit scaffolding, and no "
    "operator discovery, parsing, public transfer, or coding capability is claimed"
)

PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
OLD_OCCURRENCE_CERTIFICATE_IMPORT_COUNT = 0
DESIRED_READS_DURING_BUILD = 0

_NAMESPACE = b"typed-semantic-packed-residual-literal-route-frontier-v1\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(label: bytes, value: object) -> str:
    return sha256(_NAMESPACE + label + _canonical_json(value)).hexdigest()


class ResidualLiteralRouteFrontierState(NamedTuple):
    status: str
    checkpoint_sha256: str
    ordered_checkpoint_sha256: str
    binding_certificate_sha256: str
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    residual_payload_cells: tuple[int, ...]
    residual_token_index: int | None
    relation: int | None
    operand_action: int | None
    selected_route: int | None
    repeated_value: int | None
    residual_value: int | None
    effective_additive_literal: int | None
    additive_literal_occurrence: int
    additive_literal_slot: int
    structural_scaffold_sha256: str | None
    delegated_frontier_sha256: str | None
    packed_build_count: int
    desired_reads_during_build: int
    sha256: str


class ResidualLiteralRouteSelection(NamedTuple):
    status: str
    frontier_sha256: str
    binding_certificate_sha256: str
    selected_route: int | None
    repeated_value: int | None
    residual_value: int | None
    effective_additive_literal: int | None
    delegated: plan_slice.PackedPromptPlanSliceSelection | None
    sha256: str


def _state_payload(value: ResidualLiteralRouteFrontierState) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field != "sha256"
    }


class PackedResidualLiteralRouteFrontierV1:
    """Deep desired-free ``build`` followed by one desired-bearing ``select``."""

    __slots__ = ("_binding_certificate", "_checkpoint", "_delegate", "_state")

    def __init__(self) -> None:
        raise TypeError("use PackedResidualLiteralRouteFrontierV1.build")

    @classmethod
    def build(
        cls,
        checkpoint: learner.FrozenCausalResidualLiteralRouteV1,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        equality_gate: bool = True,
        literal_role_gate: bool = True,
        order_gate: bool = True,
        upstream_cell_gate: bool = True,
        residual_gate: bool = True,
        cell_gate: bool = True,
    ) -> PackedResidualLiteralRouteFrontierV1:
        if type(checkpoint) is not learner.FrozenCausalResidualLiteralRouteV1:
            raise TypeError("residual route frontier requires the exact checkpoint type")
        checkpoint.authenticate()
        certificate = checkpoint.bind(
            span_records,
            literal_records,
            signature_role_record,
            equality_gate=equality_gate,
            literal_role_gate=literal_role_gate,
            order_gate=order_gate,
            upstream_cell_gate=upstream_cell_gate,
            residual_gate=residual_gate,
            cell_gate=cell_gate,
        )
        learner._authenticate_binding(certificate)

        repeated_payload: tuple[int, ...] = ()
        repeated_positions: tuple[int, ...] = ()
        residual_payload: tuple[int, ...] = ()
        residual_position: int | None = None
        relation: int | None = None
        operand_action: int | None = None
        selected_route: int | None = None
        repeated_value: int | None = None
        residual_value: int | None = None
        effective_literal: int | None = None
        scaffold_sha: str | None = None
        delegate: plan_slice.PackedGenericTypedPromptPlanSliceV1 | None = None
        status = certificate.status
        if status == learner.READY:
            binding = learner.require_residual_literal_route_binding(certificate)
            repeated_payload = binding.repeated_payload_cells
            repeated_positions = binding.repeated_token_indices
            residual_payload = binding.residual_payload_cells
            residual_position = binding.residual_token_index
            relation = binding.relation
            operand_action = binding.operand_action
            selected_route = binding.selected_route
            repeated_value = literal_role.decode_int_payload(repeated_payload)
            residual_value = literal_role.decode_int_payload(residual_payload)
            effective_literal = (
                residual_value
                if selected_route == learner.RESIDUAL_TO_LITERAL_HOLE
                else learner.DEFAULT_LITERAL
            )
            scaffold = learner.freeze_residual_literal_route_scaffold(
                repeated_value,
                residual_value,
                operand_action,
                selected_route,
            )
            scaffold_sha = scaffold.sha256
            delegate = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
                signature,
                cases,
                literals,
                max_cost,
                scaffold,
            )
            status = delegate.status

        partial = ResidualLiteralRouteFrontierState(
            status,
            checkpoint.checkpoint_sha256,
            checkpoint.ordered_checkpoint_sha256,
            certificate.sha256,
            repeated_payload,
            repeated_positions,
            residual_payload,
            residual_position,
            relation,
            operand_action,
            selected_route,
            repeated_value,
            residual_value,
            effective_literal,
            learner.ADDITIVE_LITERAL_OCCURRENCE,
            learner.ADDITIVE_LITERAL_SLOT,
            scaffold_sha,
            None if delegate is None else delegate.sha256,
            0 if delegate is None else 1,
            DESIRED_READS_DURING_BUILD,
            "",
        )
        state = partial._replace(sha256=_digest(b"state\0", _state_payload(partial)))
        instance = object.__new__(cls)
        instance._checkpoint = checkpoint
        instance._binding_certificate = certificate
        instance._delegate = delegate
        instance._state = state
        instance._authenticate()
        return instance

    @property
    def state(self) -> ResidualLiteralRouteFrontierState:
        self._authenticate()
        return self._state

    @property
    def status(self) -> str:
        self._authenticate()
        return self._state.status

    @property
    def sha256(self) -> str:
        self._authenticate()
        return self._state.sha256

    @property
    def delegated_frontier(
        self,
    ) -> plan_slice.PackedGenericTypedPromptPlanSliceV1 | None:
        self._authenticate()
        return self._delegate

    def _authenticate(self) -> None:
        self._checkpoint.authenticate()
        learner._authenticate_binding(self._binding_certificate)
        if (
            self._state.checkpoint_sha256 != self._checkpoint.checkpoint_sha256
            or self._state.ordered_checkpoint_sha256
            != self._checkpoint.ordered_checkpoint_sha256
            or self._state.binding_certificate_sha256
            != self._binding_certificate.sha256
            or self._state.delegated_frontier_sha256
            != (None if self._delegate is None else self._delegate.sha256)
            or self._state.packed_build_count != (0 if self._delegate is None else 1)
            or self._state.desired_reads_during_build != 0
            or self._state.sha256 != _digest(b"state\0", _state_payload(self._state))
        ):
            raise RuntimeError("residual literal route frontier authentication failed")

    def select(self, desired: Sequence[object]) -> ResidualLiteralRouteSelection:
        self._authenticate()
        delegated = None if self._delegate is None else self._delegate.select(desired)
        status = self._state.status if delegated is None else delegated.status
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "binding_certificate_sha256": self._binding_certificate.sha256,
            "selected_route": self._state.selected_route,
            "repeated_value": self._state.repeated_value,
            "residual_value": self._state.residual_value,
            "effective_additive_literal": self._state.effective_additive_literal,
            "delegated_sha256": None if delegated is None else delegated.sha256,
        }
        return ResidualLiteralRouteSelection(
            status,
            self._state.sha256,
            self._binding_certificate.sha256,
            self._state.selected_route,
            self._state.repeated_value,
            self._state.residual_value,
            self._state.effective_additive_literal,
            delegated,
            _digest(b"selection\0", payload),
        )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "residual_route_config_sha256": learner.CONFIG_SHA256,
        "desired_reads_during_build": DESIRED_READS_DURING_BUILD,
        "public_train_input_count": PUBLIC_TRAIN_INPUT_COUNT,
        "canonical_panel_spend_count": CANONICAL_PANEL_SPEND_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "c1_input_count": C1_INPUT_COUNT,
        "path2_input_count": PATH2_INPUT_COUNT,
        "g0_input_count": G0_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "old_occurrence_certificate_import_count": (
            OLD_OCCURRENCE_CERTIFICATE_IMPORT_COUNT
        ),
    }
)
CONFIG_SHA256 = _digest(b"config\0", dict(CONFIG_RECORD))


__all__ = [
    "C1_INPUT_COUNT",
    "CANONICAL_PANEL_SPEND_COUNT",
    "CLAIM_BOUNDARY",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DESIRED_READS_DURING_BUILD",
    "DEV_INPUT_COUNT",
    "FRONTIER_READY",
    "G0_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "OLD_OCCURRENCE_CERTIFICATE_IMPORT_COUNT",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "SCHEMA",
    "SELECTED",
    "PackedResidualLiteralRouteFrontierV1",
    "ResidualLiteralRouteFrontierState",
    "ResidualLiteralRouteSelection",
]
