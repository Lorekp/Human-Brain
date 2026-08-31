"""Continuous neutral occurrence streams to ordered executable witnesses.

The learned boundary gate is the only grouping authority.  Every exact ordered
occurrence certificate it issues is continued through the bridge with one
identical desired-free prompt payload; this layer supplies no boundary, group
length, reducer action, desired program, or operator identity.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from typing import NamedTuple

from experiments.r3_pilot import causal_anonymous_occurrence_bundle_schema_v1 as occurrence
from experiments.r3_pilot import causal_learned_occurrence_boundary_gate_v1 as boundary_gate
from experiments.r3_pilot import causal_occurrence_executable_reducer_bridge_v1 as bridge
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_CONTINUOUS_OCCURRENCE_EXECUTABLE_STREAM_V1"
READY = "CAUSAL_CONTINUOUS_OCCURRENCE_EXECUTABLE_STREAM_V1_READY"

CLAIM_BOUNDARY = (
    "self-posed continuous neutral pure-chain/pure-fanout streams only: learned "
    "physical completeness-by-recurrence grouping issues ordered occurrence "
    "certificates that select count/raw-sum executable witnesses under one shared "
    "desired-free prompt payload; stream start/end, graph consequence projection, "
    "prompt literal binding, reducer implementations, typed enumeration, and "
    "compilation remain explicit, so mixed graphs, parsing, arbitrary algorithms, "
    "public coding transfer, and SWE capability remain unproven"
)

# The composite owns no new functional population or learned state.  Both banked
# dependencies were originally numbered immediately above the same segmented
# base, so the composite link namespace keeps the bridge branch first and rebases
# the gate's four already-learned feature cells after it.
NEW_CELL_COUNT = 0
NEW_TRAINABLE_STATE_BYTES = 0
NEW_PERSISTENT_STATE_BYTES = 0
REBASED_GATE_FEATURE_CELL_OFFSET = bridge.TOTAL_CELL_COUNT
REBASED_GATE_FEATURE_CELL_STOP = (
    REBASED_GATE_FEATURE_CELL_OFFSET + boundary_gate.NEW_CELL_COUNT
)
TOTAL_CELL_COUNT = REBASED_GATE_FEATURE_CELL_STOP
COMPOSED_PERSISTENT_STATE_BYTES = (
    bridge.COMPOSED_PERSISTENT_STATE_BYTES + boundary_gate.PERSISTENT_STATE_BYTES
)
RUNTIME_STATE_BYTES = boundary_gate.RUNTIME_STATE_BYTES

BOUNDARY_BOOLEAN_INPUT_COUNT = 0
GROUP_LENGTH_INPUT_COUNT = 0
SELECTED_ACTION_INPUT_COUNT = 0
DESIRED_PROGRAM_INPUT_COUNT = 0
OPERATOR_NAME_INPUT_COUNT = 0
PUBLIC_TRAIN_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0

_NAMESPACE = b"causal-continuous-occurrence-executable-stream-v1\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_WORK_NAMESPACE = _NAMESPACE + b"work\0"
_OUTCOME_NAMESPACE = _NAMESPACE + b"outcome\0"
_CONSTRUCTION_TOKEN = object()


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


class ContinuousExecutableWork(NamedTuple):
    raw_event_count: int
    occurrence_certificate_count: int
    bridge_build_count: int
    exact_proof_count: int
    grouping_physical_cell_steps: int
    certificate_provenance_occurrence_cell_steps: int
    residual_binding_cell_steps: int
    bridge_physical_cell_steps: int
    total_executed_physical_cell_steps: int
    packed_frontier_build_count: int
    authorized_slice_count: int
    proof_materialization_count: int
    occurrence_certificate_sha256s: tuple[str, ...]
    witness_certificate_sha256s: tuple[str, ...]
    packed_work_sha256s: tuple[str, ...]
    sha256: str


class ContinuousOccurrenceExecutableOutcome(NamedTuple):
    status: str
    stop_reason: str | None
    grouping: boundary_gate.LearnedBoundaryStreamOutcome
    occurrence_certificates: tuple[occurrence.OccurrenceGraphSchemaCertificate, ...]
    witness_certificates: tuple[bridge.OccurrenceExecutableWitnessCertificate, ...]
    work: ContinuousExecutableWork
    checkpoint_sha256: str
    sha256: str


def _checkpoint_sha256(
    gate_checkpoint_sha256: str,
    bridge_checkpoint_sha256: str,
    occurrence_checkpoint_sha256: str,
    segmented_checkpoint_sha256: str,
) -> str:
    return sha256(
        _CHECKPOINT_NAMESPACE
        + gate_checkpoint_sha256.encode("ascii")
        + bridge_checkpoint_sha256.encode("ascii")
        + occurrence_checkpoint_sha256.encode("ascii")
        + segmented_checkpoint_sha256.encode("ascii")
        + _canonical_json(
            {
                "total_cell_count": TOTAL_CELL_COUNT,
                "persistent_state_bytes": COMPOSED_PERSISTENT_STATE_BYTES,
            }
        )
    ).hexdigest()


def _freeze_work(
    grouping: boundary_gate.LearnedBoundaryStreamOutcome,
    witnesses: Sequence[bridge.OccurrenceExecutableWitnessCertificate],
) -> ContinuousExecutableWork:
    occurrence_steps = sum(
        value.work.occurrence_physical_cell_steps for value in witnesses
    )
    residual_steps = sum(value.work.residual_binding_cell_steps for value in witnesses)
    bridge_steps = sum(value.work.bridge_physical_cell_steps for value in witnesses)
    packed_roots = tuple(
        value.work.packed_work_sha256
        for value in witnesses
        if value.work.packed_work_sha256 is not None
    )
    partial = ContinuousExecutableWork(
        grouping.work.input_raw_event_count,
        len(grouping.certificates),
        len(witnesses),
        sum(value.exact_proof_count for value in witnesses),
        grouping.work.total_physical_cell_steps,
        occurrence_steps,
        residual_steps,
        bridge_steps,
        grouping.work.total_physical_cell_steps + residual_steps + bridge_steps,
        sum(value.work.packed_frontier_build_count for value in witnesses),
        sum(value.work.authorized_slice_count for value in witnesses),
        sum(value.work.proof_materialization_count for value in witnesses),
        tuple(value.sha256 for value in grouping.certificates),
        tuple(value.sha256 for value in witnesses),
        packed_roots,
        "",
    )
    payload = {
        field: getattr(partial, field)
        for field in partial._fields
        if field != "sha256"
    }
    return partial._replace(sha256=_digest(_WORK_NAMESPACE, payload))


def _outcome(
    checkpoint: FrozenContinuousOccurrenceExecutableStreamV1,
    grouping: boundary_gate.LearnedBoundaryStreamOutcome,
    witnesses: Sequence[bridge.OccurrenceExecutableWitnessCertificate],
    *,
    status: str,
    stop_reason: str | None,
) -> ContinuousOccurrenceExecutableOutcome:
    frozen_witnesses = tuple(witnesses)
    frozen_occurrences = tuple(grouping.certificates)
    work = _freeze_work(grouping, frozen_witnesses)
    payload = {
        "status": status,
        "stop_reason": stop_reason,
        "grouping_sha256": grouping.sha256,
        "occurrences": tuple(value.sha256 for value in frozen_occurrences),
        "witnesses": tuple(value.sha256 for value in frozen_witnesses),
        "work_sha256": work.sha256,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
    }
    return ContinuousOccurrenceExecutableOutcome(
        status,
        stop_reason,
        grouping,
        frozen_occurrences,
        frozen_witnesses,
        work,
        checkpoint.checkpoint_sha256,
        _digest(_OUTCOME_NAMESPACE, payload),
    )


class FrozenContinuousOccurrenceExecutableStreamV1:
    """Immutable ordered composition of the learned gate and exact bridge."""

    __slots__ = ("_bridge", "_checkpoint_sha256", "_gate")

    def __init__(
        self,
        gate_checkpoint: boundary_gate.FrozenLearnedOccurrenceBoundaryGateV1,
        bridge_checkpoint: bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1,
        *,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("continuous executable checkpoints are builder-owned")
        if type(gate_checkpoint) is not boundary_gate.FrozenLearnedOccurrenceBoundaryGateV1:
            raise TypeError("exact learned boundary checkpoint required")
        if type(bridge_checkpoint) is not bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1:
            raise TypeError("exact occurrence executable bridge required")
        gate_checkpoint.authenticate()
        bridge_checkpoint.authenticate()
        self._gate = gate_checkpoint
        self._bridge = bridge_checkpoint
        self._require_shared_authority()
        self._checkpoint_sha256 = _checkpoint_sha256(
            gate_checkpoint.checkpoint_sha256,
            bridge_checkpoint.checkpoint_sha256,
            gate_checkpoint._occurrence.checkpoint_sha256,
            gate_checkpoint._segmented.checkpoint_sha256,
        )
        self.authenticate()

    def _require_shared_authority(self) -> None:
        bridge_segmented = (
            self._bridge._content_checkpoint._segmented_checkpoint
        )
        if (
            self._gate._occurrence.checkpoint_sha256
            != self._bridge._occurrence_checkpoint.checkpoint_sha256
        ):
            raise ValueError("gate and bridge occurrence authorities differ")
        if (
            self._gate._segmented.checkpoint_sha256
            != bridge_segmented.checkpoint_sha256
        ):
            raise ValueError("gate and bridge segmented router authorities differ")

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    def authenticate(self) -> None:
        self._gate.authenticate()
        self._bridge.authenticate()
        self._require_shared_authority()
        expected = _checkpoint_sha256(
            self._gate.checkpoint_sha256,
            self._bridge.checkpoint_sha256,
            self._gate._occurrence.checkpoint_sha256,
            self._gate._segmented.checkpoint_sha256,
        )
        if self._checkpoint_sha256 != expected:
            raise RuntimeError("continuous executable checkpoint changed")

    def build(
        self,
        raw_events: Sequence[boundary_gate.RawNeutralOccurrenceEvent],
        span_records: Sequence[Mapping[str, object]],
        literal_records: Sequence[Mapping[str, object]],
        signature_role_record: Mapping[str, object],
        *,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        literals: Iterable[int],
        max_cost: int,
        grouping_cell_gate: bool = True,
        bridge_relay_cell_gate: bool = True,
    ) -> ContinuousOccurrenceExecutableOutcome:
        self.authenticate()
        grouping = self._gate.group(raw_events, gate_cell_gate=grouping_cell_gate)
        if grouping.status != boundary_gate.READY:
            return _outcome(
                self,
                grouping,
                (),
                status=grouping.status,
                stop_reason=grouping.stop_reason,
            )

        frozen_spans = tuple(span_records)
        frozen_literals_records = tuple(literal_records)
        frozen_cases = tuple(cases)
        frozen_literals = tuple(literals)
        witnesses: list[bridge.OccurrenceExecutableWitnessCertificate] = []
        for certificate in grouping.certificates:
            witness = self._bridge.build_from_occurrence_certificate(
                certificate,
                frozen_spans,
                frozen_literals_records,
                signature_role_record,
                signature=signature,
                cases=frozen_cases,
                literals=frozen_literals,
                max_cost=max_cost,
                relay_cell_gate=bridge_relay_cell_gate,
            )
            witnesses.append(witness)
            if witness.status != bridge.READY:
                return _outcome(
                    self,
                    grouping,
                    witnesses,
                    status=witness.status,
                    stop_reason=witness.stop_reason,
                )

        return _outcome(
            self,
            grouping,
            witnesses,
            status=READY,
            stop_reason=None,
        )


def compose_causal_continuous_occurrence_executable_stream_v1(
    gate_checkpoint: boundary_gate.FrozenLearnedOccurrenceBoundaryGateV1,
    bridge_checkpoint: bridge.FrozenCausalOccurrenceExecutableReducerBridgeV1,
) -> FrozenContinuousOccurrenceExecutableStreamV1:
    return FrozenContinuousOccurrenceExecutableStreamV1(
        gate_checkpoint,
        bridge_checkpoint,
        construction_token=_CONSTRUCTION_TOKEN,
    )


CONFIG_RECORD = {
    "schema": SCHEMA,
    "new_cell_count": NEW_CELL_COUNT,
    "total_cell_count": TOTAL_CELL_COUNT,
    "new_trainable_state_bytes": NEW_TRAINABLE_STATE_BYTES,
    "new_persistent_state_bytes": NEW_PERSISTENT_STATE_BYTES,
    "composed_persistent_state_bytes": COMPOSED_PERSISTENT_STATE_BYTES,
    "runtime_state_bytes": RUNTIME_STATE_BYTES,
    "claim_boundary": CLAIM_BOUNDARY,
}
CONFIG_SHA256 = _digest(_NAMESPACE + b"config\0", CONFIG_RECORD)


__all__ = [
    "BOUNDARY_BOOLEAN_INPUT_COUNT",
    "C1_INPUT_COUNT",
    "CLAIM_BOUNDARY",
    "COMPOSED_PERSISTENT_STATE_BYTES",
    "CONFIG_RECORD",
    "CONFIG_SHA256",
    "DESIRED_PROGRAM_INPUT_COUNT",
    "DEV_INPUT_COUNT",
    "G0_INPUT_COUNT",
    "GROUP_LENGTH_INPUT_COUNT",
    "M0_INPUT_COUNT",
    "NEW_CELL_COUNT",
    "NEW_PERSISTENT_STATE_BYTES",
    "NEW_TRAINABLE_STATE_BYTES",
    "OPERATOR_NAME_INPUT_COUNT",
    "PATH2_INPUT_COUNT",
    "PROTECTED_INPUT_COUNT",
    "PUBLIC_TRAIN_INPUT_COUNT",
    "READY",
    "RUNTIME_STATE_BYTES",
    "SCHEMA",
    "SELECTED_ACTION_INPUT_COUNT",
    "TOTAL_CELL_COUNT",
    "ContinuousExecutableWork",
    "ContinuousOccurrenceExecutableOutcome",
    "FrozenContinuousOccurrenceExecutableStreamV1",
    "compose_causal_continuous_occurrence_executable_stream_v1",
]
