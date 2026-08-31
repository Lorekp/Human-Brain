"""Thin desired-free packed adapter for the append-effect reducer policy."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from types import MappingProxyType
from typing import NamedTuple

from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as learner
from experiments.r3_pilot import causal_residual_literal_route_learner_v1 as residual
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "TYPED_SEMANTIC_PACKED_APPEND_EFFECT_REDUCER_FRONTIER_V1"
FRONTIER_READY = plan_slice.FRONTIER_READY
SELECTED = plan_slice.SELECTED

CLAIM_BOUNDARY = learner.CLAIM_BOUNDARY
PUBLIC_TRAIN_INPUT_COUNT = 0
CANONICAL_PANEL_SPEND_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
DESIRED_READS_DURING_BUILD = 0

_NAMESPACE = b"typed-semantic-packed-append-effect-reducer-frontier-v1\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(label: bytes, value: object) -> str:
    return sha256(_NAMESPACE + label + _canonical_json(value)).hexdigest()


class AppendEffectReducerFrontierState(NamedTuple):
    status: str
    checkpoint_sha256: str
    residual_checkpoint_sha256: str
    binding_certificate_sha256: str
    effect_relation: int | None
    selected_action: int | None
    probe_sha256: str
    scaffold_sha256: str | None
    delegated_frontier_sha256: str | None
    packed_build_count: int
    desired_reads_during_build: int
    sha256: str


class AppendEffectReducerSelection(NamedTuple):
    status: str
    frontier_sha256: str
    binding_certificate_sha256: str
    effect_relation: int | None
    selected_action: int | None
    delegated: plan_slice.PackedPromptPlanSliceSelection | None
    sha256: str


def _state_payload(value: AppendEffectReducerFrontierState) -> dict[str, object]:
    return {
        field: getattr(value, field) for field in value._fields if field != "sha256"
    }


def _normalize_final_cases(
    values: Iterable[tuple[object, ...]],
) -> tuple[tuple[object, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("final cases must be an iterable of argument tuples")
    frozen = tuple(tuple(row) for row in values)
    if not frozen:
        raise ValueError("final cases cannot be empty")
    return frozen


def _list_case_key(row: tuple[object, ...]) -> tuple[int, ...] | None:
    if len(row) != 1 or type(row[0]) is not list:
        return None
    values = row[0]
    if any(type(value) is not int for value in values):
        return None
    return tuple(values)


class PackedAppendEffectReducerFrontierV1:
    """Deep learned ``build`` followed by one desired-bearing ``select``."""

    __slots__ = ("_binding_certificate", "_checkpoint", "_delegate", "_state")

    def __init__(self) -> None:
        raise TypeError("use PackedAppendEffectReducerFrontierV1.build")

    @classmethod
    def build(
        cls,
        checkpoint: learner.FrozenCausalAppendEffectReducerV1,
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        probe_cases: tuple[tuple[list[int]], ...],
        probe_outputs: Sequence[object],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        equality_gate: bool = True,
        predicate_gate: bool = True,
        effect_gate: bool = True,
        completeness_gate: bool = True,
        cell_gate: bool = True,
    ) -> PackedAppendEffectReducerFrontierV1:
        if type(checkpoint) is not learner.FrozenCausalAppendEffectReducerV1:
            raise TypeError("append-effect frontier requires the exact checkpoint")
        checkpoint.authenticate()
        certificate = checkpoint.bind(
            span_records,
            literal_records,
            signature_role_record,
            probe_cases,
            probe_outputs,  # type: ignore[arg-type]
            equality_gate=equality_gate,
            predicate_gate=predicate_gate,
            effect_gate=effect_gate,
            completeness_gate=completeness_gate,
            cell_gate=cell_gate,
        )
        receipt = learner._authenticate_binding(certificate)
        if receipt.checkpoint_sha256 != checkpoint.checkpoint_sha256:
            raise RuntimeError("append-effect certificate belongs to another checkpoint")

        effect_relation: int | None = None
        selected_action: int | None = None
        probe_sha = receipt.probe_sha256
        scaffold_sha: str | None = None
        delegate: plan_slice.PackedGenericTypedPromptPlanSliceV1 | None = None
        status = certificate.status
        if status == learner.READY:
            binding = learner.require_append_effect_reducer_binding(certificate)
            effect_relation = binding.effect_relation
            selected_action = binding.selected_action
            frozen_cases = _normalize_final_cases(cases)
            probe_keys = {row[0] for row in binding.probe_cases}
            final_keys = tuple(_list_case_key(row) for row in frozen_cases)
            if any(key is not None and key in probe_keys for key in final_keys):
                status = learner.STOP_PROBE_FINAL_OVERLAP
            else:
                upstream = residual.require_residual_literal_route_binding(
                    receipt.residual_certificate
                )
                scaffold = learner._reducer_scaffold(upstream, selected_action)
                scaffold_sha = scaffold.sha256
                delegate = plan_slice.PackedGenericTypedPromptPlanSliceV1.build(
                    signature,
                    frozen_cases,
                    tuple(literals),
                    max_cost,
                    scaffold,
                )
                status = delegate.status

        partial = AppendEffectReducerFrontierState(
            status,
            checkpoint.checkpoint_sha256,
            checkpoint.residual_checkpoint_sha256,
            certificate.sha256,
            effect_relation,
            selected_action,
            probe_sha,
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
    def state(self) -> AppendEffectReducerFrontierState:
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

    def _authenticate(self) -> None:
        self._checkpoint.authenticate()
        receipt = learner._authenticate_binding(self._binding_certificate)
        if (
            receipt.checkpoint_sha256 != self._checkpoint.checkpoint_sha256
            or self._state.binding_certificate_sha256
            != self._binding_certificate.sha256
            or self._state.delegated_frontier_sha256
            != (None if self._delegate is None else self._delegate.sha256)
            or self._state.packed_build_count != (0 if self._delegate is None else 1)
            or self._state.desired_reads_during_build != 0
            or self._state.sha256 != _digest(b"state\0", _state_payload(self._state))
        ):
            raise RuntimeError("append-effect frontier authentication failed")

    def select(self, desired: Sequence[object]) -> AppendEffectReducerSelection:
        self._authenticate()
        delegated = None if self._delegate is None else self._delegate.select(desired)
        status = self._state.status if delegated is None else delegated.status
        payload = {
            "status": status,
            "frontier_sha256": self._state.sha256,
            "binding_certificate_sha256": self._binding_certificate.sha256,
            "effect_relation": self._state.effect_relation,
            "selected_action": self._state.selected_action,
            "delegated_sha256": None if delegated is None else delegated.sha256,
        }
        return AppendEffectReducerSelection(
            status,
            self._state.sha256,
            self._binding_certificate.sha256,
            self._state.effect_relation,
            self._state.selected_action,
            delegated,
            _digest(b"selection\0", payload),
        )


CONFIG_RECORD = MappingProxyType(
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "learner_config_sha256": learner.CONFIG_SHA256,
        "public_train_input_count": PUBLIC_TRAIN_INPUT_COUNT,
        "canonical_panel_spend_count": CANONICAL_PANEL_SPEND_COUNT,
        "dev_input_count": DEV_INPUT_COUNT,
        "m0_input_count": M0_INPUT_COUNT,
        "c1_input_count": C1_INPUT_COUNT,
        "g0_input_count": G0_INPUT_COUNT,
        "protected_input_count": PROTECTED_INPUT_COUNT,
        "desired_reads_during_build": DESIRED_READS_DURING_BUILD,
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
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "SCHEMA",
    "SELECTED",
    "AppendEffectReducerFrontierState",
    "AppendEffectReducerSelection",
    "PackedAppendEffectReducerFrontierV1",
]
