"""Bind one learned relative literal role into one packed semantic hole.

The learner in :mod:`causal_relative_literal_role_learner_v1` owns prompt
equality, relation learning, and its exact-object binding authority.  This
module owns only the explicit experimental adapter from that authenticated
binding to the fixed ``List[Int] -> Int`` scaffold and the existing packed
``build``/``select`` frontier.  Desired outputs are absent from ``build``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple

from experiments.r3_pilot import causal_relative_literal_role_learner_v1 as learner
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "TYPED_SEMANTIC_PACKED_RELATIVE_LITERAL_HOLE_FRONTIER_V1"
FRONTIER_READY = plan_slice.FRONTIER_READY
SELECTED = plan_slice.SELECTED

CLAIM_BOUNDARY = (
    "self-posed target-free-at-build transfer of one learned anonymous "
    "repetition role and one unseen typed-Int payload into one fixed "
    "List[Int]-to-Int packed literal hole; the fixed structural certificate "
    "is explicit scaffolding, and raw language understanding, template "
    "learning, public-panel transfer, protected evaluation, and coding "
    "capability remain unproven"
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

_NAMESPACE = b"typed-semantic-packed-relative-literal-hole-frontier-v1\0"


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


class RelativeLiteralHoleFrontierState(NamedTuple):
    status: str
    checkpoint_sha256: str
    binding_certificate_sha256: str
    selected_payload_cells: tuple[int, ...]
    selected_member_token_indices: tuple[int, ...]
    literal_hole_value: int | None
    structural_scaffold_sha256: str | None
    delegated_frontier_sha256: str | None
    packed_build_count: int
    desired_reads_during_build: int
    sha256: str


class RelativeLiteralHoleSelection(NamedTuple):
    status: str
    frontier_sha256: str
    binding_certificate_sha256: str
    literal_hole_value: int | None
    delegated: plan_slice.PackedPromptPlanSliceSelection | None
    sha256: str


def _state_payload(value: RelativeLiteralHoleFrontierState) -> dict[str, object]:
    return {
        field: getattr(value, field)
        for field in value._fields
        if field != "sha256"
    }


class PackedRelativeLiteralHoleFrontierV1:
    """Deep target-free ``build`` followed by desired-bearing ``select``."""

    __slots__ = (
        "_binding_certificate",
        "_checkpoint",
        "_delegate",
        "_state",
    )

    def __init__(self) -> None:
        raise TypeError("use PackedRelativeLiteralHoleFrontierV1.build")

    @classmethod
    def build(
        cls,
        checkpoint: learner.FrozenCausalRelativeLiteralRoleV1,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        equality_gate: bool = True,
        cell_gate: bool = True,
    ) -> PackedRelativeLiteralHoleFrontierV1:
        if type(checkpoint) is not learner.FrozenCausalRelativeLiteralRoleV1:
            raise TypeError("relative literal frontier requires the exact checkpoint type")
        checkpoint.authenticate()
        binding_certificate = checkpoint.bind(
            span_records,
            literal_records,
            signature_role_record,
            equality_gate=equality_gate,
            cell_gate=cell_gate,
        )
        learner._authenticate_binding(binding_certificate)

        selected_payload: tuple[int, ...] = ()
        selected_positions: tuple[int, ...] = ()
        hole_value: int | None = None
        scaffold_sha: str | None = None
        delegate: plan_slice.PackedGenericTypedPromptPlanSliceV1 | None = None
        status = binding_certificate.status
        if status == learner.READY:
            binding = learner.require_relative_literal_binding(binding_certificate)
            selected_payload = binding.payload_cells
            selected_positions = binding.member_token_indices
            hole_value = learner.decode_int_payload(selected_payload)
            scaffold = learner.freeze_literal_hole_scaffold(hole_value)
            scaffold_sha = scaffold.sha256
            delegate = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
                signature,
                cases,
                literals,
                max_cost,
                scaffold,
            )
            status = delegate.status

        partial = RelativeLiteralHoleFrontierState(
            status,
            checkpoint.checkpoint_sha256,
            binding_certificate.sha256,
            selected_payload,
            selected_positions,
            hole_value,
            scaffold_sha,
            None if delegate is None else delegate.sha256,
            0 if delegate is None else 1,
            DESIRED_READS_DURING_BUILD,
            "",
        )
        state = partial._replace(
            sha256=_digest(b"state\0", _state_payload(partial))
        )
        instance = object.__new__(cls)
        instance._checkpoint = checkpoint
        instance._binding_certificate = binding_certificate
        instance._delegate = delegate
        instance._state = state
        instance._authenticate()
        return instance

    @property
    def state(self) -> RelativeLiteralHoleFrontierState:
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
    def binding_certificate(self) -> learner.RelativeLiteralBindingCertificate:
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
            or self._state.binding_certificate_sha256
            != self._binding_certificate.sha256
            or self._state.delegated_frontier_sha256
            != (None if self._delegate is None else self._delegate.sha256)
            or self._state.packed_build_count != (0 if self._delegate is None else 1)
            or self._state.desired_reads_during_build != 0
            or self._state.sha256
            != _digest(b"state\0", _state_payload(self._state))
        ):
            raise RuntimeError("relative literal hole frontier authentication failed")

    def select(self, desired: Sequence[object]) -> RelativeLiteralHoleSelection:
        self._authenticate()
        delegated = None if self._delegate is None else self._delegate.select(desired)
        status = self._state.status if delegated is None else delegated.status
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "binding_certificate_sha256": self._binding_certificate.sha256,
            "literal_hole_value": self._state.literal_hole_value,
            "delegated_sha256": None if delegated is None else delegated.sha256,
        }
        return RelativeLiteralHoleSelection(
            status,
            self._state.sha256,
            self._binding_certificate.sha256,
            self._state.literal_hole_value,
            delegated,
            _digest(b"selection\0", payload),
        )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
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
    "PackedRelativeLiteralHoleFrontierV1",
    "RelativeLiteralHoleFrontierState",
    "RelativeLiteralHoleSelection",
]
