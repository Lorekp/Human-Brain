"""Bind one learned relative ordered-operand action into a packed frontier.

The learner owns physical order observation, consequence learning, and its
exact-object binding authority.  This module only adapts an authenticated
binding to one of the two explicit ``List[Int] -> Int`` comparison scaffolds.
Desired outputs are absent from ``build`` and appear only in ``select``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple

from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as literal_role
from experiments.r3_pilot import causal_relative_ordered_operand_learner_v1 as learner
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "TYPED_SEMANTIC_PACKED_RELATIVE_ORDERED_OPERAND_FRONTIER_V1"
FRONTIER_READY = plan_slice.FRONTIER_READY
SELECTED = plan_slice.SELECTED

CLAIM_BOUNDARY = (
    "self-posed target-free-at-build transfer of one consequence-learned "
    "anonymous R,R,S versus S,R,R relation into one ordered operand choice "
    "for an unseen typed-Int payload and new physical prompt assembly; the "
    "two action scaffolds, comparison/count/add operators, typed Int decoder, "
    "and prompt token clock remain explicit scaffolding, so parsing, operator "
    "learning, arbitrary composition, and coding capability remain unproven"
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

_NAMESPACE = b"typed-semantic-packed-relative-ordered-operand-frontier-v1\0"


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


class RelativeOrderedOperandFrontierState(NamedTuple):
    status: str
    checkpoint_sha256: str
    literal_role_checkpoint_sha256: str
    binding_certificate_sha256: str
    repeated_payload_cells: tuple[int, ...]
    repeated_token_indices: tuple[int, ...]
    singleton_token_index: int | None
    relation: int | None
    selected_action: int | None
    literal_hole_value: int | None
    structural_scaffold_sha256: str | None
    delegated_frontier_sha256: str | None
    packed_build_count: int
    desired_reads_during_build: int
    sha256: str


class RelativeOrderedOperandSelection(NamedTuple):
    status: str
    frontier_sha256: str
    binding_certificate_sha256: str
    selected_action: int | None
    literal_hole_value: int | None
    delegated: plan_slice.PackedPromptPlanSliceSelection | None
    sha256: str


def _state_payload(value: RelativeOrderedOperandFrontierState) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field != "sha256"
    }


class PackedRelativeOrderedOperandFrontierV1:
    """Target-free ``build`` followed by one desired-bearing ``select``."""

    __slots__ = ("_binding_certificate", "_checkpoint", "_delegate", "_state")

    def __init__(self) -> None:
        raise TypeError("use PackedRelativeOrderedOperandFrontierV1.build")

    @classmethod
    def build(
        cls,
        checkpoint: learner.FrozenCausalRelativeOrderedOperandV1,
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
        cell_gate: bool = True,
    ) -> PackedRelativeOrderedOperandFrontierV1:
        if type(checkpoint) is not learner.FrozenCausalRelativeOrderedOperandV1:
            raise TypeError("ordered operand frontier requires the exact checkpoint type")
        checkpoint.authenticate()
        certificate = checkpoint.bind(
            span_records,
            literal_records,
            signature_role_record,
            equality_gate=equality_gate,
            literal_role_gate=literal_role_gate,
            order_gate=order_gate,
            cell_gate=cell_gate,
        )
        learner._authenticate_binding(certificate)

        repeated_payload: tuple[int, ...] = ()
        repeated_positions: tuple[int, ...] = ()
        singleton_position: int | None = None
        relation: int | None = None
        action: int | None = None
        hole_value: int | None = None
        scaffold_sha: str | None = None
        delegate: plan_slice.PackedGenericTypedPromptPlanSliceV1 | None = None
        status = certificate.status
        if status == learner.READY:
            binding = learner.require_relative_ordered_operand_binding(certificate)
            repeated_payload = binding.repeated_payload_cells
            repeated_positions = binding.repeated_token_indices
            singleton_position = binding.singleton_token_index
            relation = binding.relation
            action = binding.selected_action
            hole_value = literal_role.decode_int_payload(repeated_payload)
            scaffold = learner.freeze_ordered_operand_scaffold(hole_value, action)
            scaffold_sha = scaffold.sha256
            delegate = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
                signature,
                cases,
                literals,
                max_cost,
                scaffold,
            )
            status = delegate.status

        partial = RelativeOrderedOperandFrontierState(
            status,
            checkpoint.checkpoint_sha256,
            checkpoint.literal_role_checkpoint_sha256,
            certificate.sha256,
            repeated_payload,
            repeated_positions,
            singleton_position,
            relation,
            action,
            hole_value,
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
    def state(self) -> RelativeOrderedOperandFrontierState:
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
    def literal_hole_value(self) -> int | None:
        self._authenticate()
        return self._state.literal_hole_value

    @property
    def selected_action(self) -> int | None:
        self._authenticate()
        return self._state.selected_action

    @property
    def binding_certificate(self) -> learner.RelativeOrderedOperandCertificate:
        self._authenticate()
        return self._binding_certificate

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
            or self._state.literal_role_checkpoint_sha256
            != self._checkpoint.literal_role_checkpoint_sha256
            or self._state.binding_certificate_sha256
            != self._binding_certificate.sha256
            or self._state.delegated_frontier_sha256
            != (None if self._delegate is None else self._delegate.sha256)
            or self._state.packed_build_count != (0 if self._delegate is None else 1)
            or self._state.desired_reads_during_build != 0
            or self._state.sha256 != _digest(b"state\0", _state_payload(self._state))
        ):
            raise RuntimeError("relative ordered operand frontier authentication failed")

    def select(self, desired: Sequence[object]) -> RelativeOrderedOperandSelection:
        self._authenticate()
        delegated = None if self._delegate is None else self._delegate.select(desired)
        status = self._state.status if delegated is None else delegated.status
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "binding_certificate_sha256": self._binding_certificate.sha256,
            "selected_action": self._state.selected_action,
            "literal_hole_value": self._state.literal_hole_value,
            "delegated_sha256": None if delegated is None else delegated.sha256,
        }
        return RelativeOrderedOperandSelection(
            status,
            self._state.sha256,
            self._binding_certificate.sha256,
            self._state.selected_action,
            self._state.literal_hole_value,
            delegated,
            _digest(b"selection\0", payload),
        )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "ordered_operand_config_sha256": learner.CONFIG_SHA256,
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
    "PackedRelativeOrderedOperandFrontierV1",
    "RelativeOrderedOperandFrontierState",
    "RelativeOrderedOperandSelection",
]
