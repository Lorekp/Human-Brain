"""Authorize one microrepository repair from continuous raw task context.

The frozen raw-chunk bridge is the only cognitive entry point.  It receives one
authenticated continuous byte stream and the visible typed cases, recovers two
admitted chunk traces, and must reactivate both distinct recurrent relation
cells before the repository adapter may proceed.  The already frozen recurrent
frontier is then handed to the V1 repository loop; it is never rebuilt here.

Repository paths, source attestation, expected-output loading, patch I/O, test
execution, and rollback remain environment adapters.  Desired outputs enter
only inside the unchanged one-shot V1 ``repair`` method after this preparation
state is frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

from experiments.r3_pilot import causal_microrepository_repair_loop_v1 as repository
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import common_causal_predictive_byte_chunker_v1 as chunker
from experiments.r3_pilot import common_raw_chunk_recurrent_relation_bridge_v1 as bridge

SCHEMA = "CAUSAL_RAW_CONTEXT_MICROREPOSITORY_REPAIR_LOOP_V2"
PREPARED = f"{SCHEMA}_PREPARED"
STOP_CHECKPOINT = f"STOP_{SCHEMA}_CHECKPOINT"
STOP_RAW_CONTEXT_INVARIANT = f"STOP_{SCHEMA}_RAW_CONTEXT_INVARIANT"

ADDED_CELL_COUNT = 0
ADDED_TRAINABLE_BYTES = 0
ADDED_CHECKPOINT_BYTES = 0
RAW_CONTEXT_RECALLS = 1

PUBLIC_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0

CLAIM_BOUNDARY = (
    "one deterministic authenticated raw motif stream authorizes both learned "
    "relation cells and one self-posed List[Int] -> Int microrepository repair; "
    "task segmentation, typed visible cases, source attestation, desired-output "
    "loading, filesystem patching, process testing, compilation, and rollback "
    "remain explicit host scaffolding"
)

_PREPARED_NAMESPACE = b"causal-raw-context-microrepository-prepared-v2"
_RESULT_NAMESPACE = b"causal-raw-context-microrepository-result-v2"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


class PreparedRawContextMicroRepositoryStateV2(NamedTuple):
    status: str
    bridge_checkpoint_sha256: str
    policy_checkpoint_sha256: str
    task_stream_sha256: str
    raw_rollout_sha256: str | None
    inner_prepared_sha256: str
    case_count: int
    raw_context_recalls: int
    cut_count: int
    activated_relation_cell_count: int
    distinct_relation_cell_count: int
    activated_relation_receipt_count: int
    distinct_relation_receipt_count: int
    event_assembly_reads: int
    predictor_equality_reads: int
    chunker_cell_steps: int
    admission_equality_reads: int
    bridge_trace_equality_reads: int
    bridge_relation_cell_steps: int
    selected_policy_actuations: int
    committed_edits: int
    revisions: int
    unselected_splice_executions: int
    desired_reads_during_prepare: int
    source_materializations_during_prepare: int
    patch_writes_during_prepare: int
    repository_tests_during_prepare: int
    added_cell_count: int
    added_trainable_bytes: int
    added_checkpoint_bytes: int
    caller_chunk_span_input: bool
    caller_trace_id_input: bool
    caller_relation_id_or_order_input: bool
    caller_edit_path_input: bool
    desired_or_program_label_input: bool
    raw_stream_retention_count: int
    sha256: str


@dataclass(frozen=True)
class RawContextMicroRepositoryRepairResultV2:
    status: str
    prepared_sha256: str
    inner_result: repository.MicroRepositoryRepairResultV1
    sha256: str


@dataclass(slots=True)
class PreparedRawContextMicroRepositoryRepairV2:
    """Thin audit wrapper around the unchanged V1 one-shot repair."""

    _inner: repository.PreparedMicroRepositoryRepairV1
    state: PreparedRawContextMicroRepositoryStateV2
    _consumed: bool = False

    def repair(self) -> RawContextMicroRepositoryRepairResultV2:
        if self._consumed:
            raise RuntimeError("prepared raw-context repair is one-shot")
        self._consumed = True
        inner_result = self._inner.repair()
        status = inner_result.status if self.state.status == PREPARED else self.state.status
        result_sha256 = _digest(
            _RESULT_NAMESPACE,
            {
                "status": status,
                "prepared": self.state.sha256,
                "inner_status": inner_result.status,
                "inner_result": inner_result.sha256,
            },
        )
        return RawContextMicroRepositoryRepairResultV2(
            status=status,
            prepared_sha256=self.state.sha256,
            inner_result=inner_result,
            sha256=result_sha256,
        )


def _prepared_state(
    *,
    status: str,
    fitted_bridge: bridge.FrozenCommonRawChunkRecurrentRelationBridgeV1,
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    task_stream: chunker.PhysicalContinuousByteStream,
    cases: tuple[tuple[object, ...], ...],
    rollout: bridge.RawChunkRelationRollout | None,
    inner: repository.PreparedMicroRepositoryRepairV1,
) -> PreparedRawContextMicroRepositoryStateV2:
    if rollout is None:
        raw_sha256 = None
        cuts = cells = receipts = ()
        work = None
    else:
        raw_sha256 = rollout.sha256
        cuts = rollout.cut_offsets
        cells = rollout.activated_relation_cells
        receipts = rollout.activated_relation_receipt_sha256s
        work = rollout.work
    payload = {
        "status": status,
        "bridge": fitted_bridge.checkpoint_sha256,
        "policy": policy.checkpoint_sha256,
        "stream": task_stream.sha256,
        "raw_rollout": raw_sha256,
        "inner": inner.state.sha256,
        "cases": len(cases),
        "raw_recalls": int(rollout is not None),
        "cuts": len(cuts),
        "activated_cells": len(cells),
        "distinct_cells": len(set(cells)),
        "activated_receipts": len(receipts),
        "distinct_receipts": len(set(receipts)),
        "event_reads": 0 if work is None else work.event_assembly_reads,
        "predictor_reads": 0 if work is None else work.predictor_equality_reads,
        "chunker_steps": 0 if work is None else work.chunker_cell_steps,
        "admission_reads": 0 if work is None else work.admission_equality_reads,
        "bridge_reads": 0 if work is None else work.bridge_trace_equality_reads,
        "bridge_steps": 0 if work is None else work.bridge_relation_cell_steps,
        "selected_actuations": 0 if work is None else work.selected_actuations,
        "committed_edits": 0 if work is None else work.committed_edits,
        "revisions": 0
        if work is None
        else rollout.frontier.state.work.revisions
        if rollout.frontier is not None
        else 0,
        "unselected_splices": 0 if work is None else work.unselected_splice_executions,
        "desired_reads": 0,
        "source_materializations": 0,
        "patch_writes": 0,
        "repository_tests": 0,
        "added_cells": ADDED_CELL_COUNT,
        "added_trainable_bytes": ADDED_TRAINABLE_BYTES,
        "added_checkpoint_bytes": ADDED_CHECKPOINT_BYTES,
        "caller_chunk_span": False,
        "caller_trace_id": False,
        "caller_relation_id_or_order": False,
        "caller_edit_path": False,
        "desired_or_program_label": False,
        "raw_stream_retention": 0,
    }
    return PreparedRawContextMicroRepositoryStateV2(
        status=status,
        bridge_checkpoint_sha256=fitted_bridge.checkpoint_sha256,
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        task_stream_sha256=task_stream.sha256,
        raw_rollout_sha256=raw_sha256,
        inner_prepared_sha256=inner.state.sha256,
        case_count=len(cases),
        raw_context_recalls=int(rollout is not None),
        cut_count=len(cuts),
        activated_relation_cell_count=len(cells),
        distinct_relation_cell_count=len(set(cells)),
        activated_relation_receipt_count=len(receipts),
        distinct_relation_receipt_count=len(set(receipts)),
        event_assembly_reads=0 if work is None else work.event_assembly_reads,
        predictor_equality_reads=0 if work is None else work.predictor_equality_reads,
        chunker_cell_steps=0 if work is None else work.chunker_cell_steps,
        admission_equality_reads=0 if work is None else work.admission_equality_reads,
        bridge_trace_equality_reads=0 if work is None else work.bridge_trace_equality_reads,
        bridge_relation_cell_steps=0 if work is None else work.bridge_relation_cell_steps,
        selected_policy_actuations=0 if work is None else work.selected_actuations,
        committed_edits=0 if work is None else work.committed_edits,
        revisions=payload["revisions"],
        unselected_splice_executions=0 if work is None else work.unselected_splice_executions,
        desired_reads_during_prepare=0,
        source_materializations_during_prepare=0,
        patch_writes_during_prepare=0,
        repository_tests_during_prepare=0,
        added_cell_count=ADDED_CELL_COUNT,
        added_trainable_bytes=ADDED_TRAINABLE_BYTES,
        added_checkpoint_bytes=ADDED_CHECKPOINT_BYTES,
        caller_chunk_span_input=False,
        caller_trace_id_input=False,
        caller_relation_id_or_order_input=False,
        caller_edit_path_input=False,
        desired_or_program_label_input=False,
        raw_stream_retention_count=0,
        sha256=_digest(_PREPARED_NAMESPACE, payload),
    )


def prepare_raw_context_microrepository_repair_v2(
    fitted_bridge: bridge.FrozenCommonRawChunkRecurrentRelationBridgeV1,
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    task_stream: chunker.PhysicalContinuousByteStream,
    spec: repository.MicroRepositoryRepairSpecV1,
    *,
    predictor_gate: bool = True,
    comparator_gate: bool = True,
    hypothesis_action_gate: bool = True,
    reset_gate: bool = True,
    replay_gate: bool = True,
    admission_gate: bool = True,
    bridge_cell_gate: bool = True,
    relation_cell_gate: bool = True,
    policy_cell_gate: bool = True,
    effect_cell_gate: bool = True,
    actuation_gate: bool = True,
    relation_replay_cell_gate: bool = True,
) -> PreparedRawContextMicroRepositoryRepairV2:
    """Freeze raw authorization, then delegate one frozen frontier to V1."""

    if type(fitted_bridge) is not bridge.FrozenCommonRawChunkRecurrentRelationBridgeV1:
        raise TypeError("exact frozen raw chunk relation bridge required")
    if type(policy) is not recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
        raise TypeError("exact frozen recurrent relation policy required")
    if type(task_stream) is not chunker.PhysicalContinuousByteStream:
        raise TypeError("exact authenticated continuous task stream required")
    if type(spec) is not repository.MicroRepositoryRepairSpecV1:
        raise TypeError("exact microrepository repair spec required")
    gates = (
        predictor_gate,
        comparator_gate,
        hypothesis_action_gate,
        reset_gate,
        replay_gate,
        admission_gate,
        bridge_cell_gate,
        relation_cell_gate,
        policy_cell_gate,
        effect_cell_gate,
        actuation_gate,
        relation_replay_cell_gate,
    )
    if any(type(gate) is not bool for gate in gates):
        raise TypeError("all raw repository gates must be exact booleans")
    fitted_bridge.authenticate()
    policy.authenticate()
    task_stream.authenticate()
    if fitted_bridge.state.policy_checkpoint_sha256 != policy.checkpoint_sha256:
        raise RuntimeError(STOP_CHECKPOINT)

    snapshot = repository._snapshot_microrepository_repair_v1(policy, spec)
    cases = ()
    rollout = None
    frontier = None
    case_manifest_reads = 0
    raw_status = snapshot.attestation.status
    if raw_status == repository.PREPARED:
        case_manifest_reads = 1
        try:
            cases = repository._read_case_manifest(snapshot.case_path)
        except OSError, UnicodeError, json.JSONDecodeError, ValueError:
            raw_status = repository.STOP_ADAPTER_INPUT
    if raw_status == repository.PREPARED and cases:
        rollout = fitted_bridge.recall(
            task_stream,
            cases,
            predictor_gate=predictor_gate,
            comparator_gate=comparator_gate,
            hypothesis_action_gate=hypothesis_action_gate,
            reset_gate=reset_gate,
            replay_gate=replay_gate,
            admission_gate=admission_gate,
            bridge_cell_gate=bridge_cell_gate,
            relation_cell_gate=relation_cell_gate,
            policy_cell_gate=policy_cell_gate,
            effect_cell_gate=effect_cell_gate,
            actuation_gate=actuation_gate,
            relation_replay_cell_gate=relation_replay_cell_gate,
        )
        raw_status = rollout.status
        if raw_status == recurrent.READY:
            frontier = rollout.frontier
            valid = (
                len(rollout.activated_relation_cells) == repository.EXPECTED_RELATION_COUNT
                and len(set(rollout.activated_relation_cells)) == repository.EXPECTED_RELATION_COUNT
                and len(rollout.activated_relation_receipt_sha256s)
                == repository.EXPECTED_RELATION_COUNT
                and len(set(rollout.activated_relation_receipt_sha256s))
                == repository.EXPECTED_RELATION_COUNT
                and set(rollout.activated_relation_receipt_sha256s)
                == set(policy.state.relation_receipt_sha256s)
                and frontier is not None
                and frontier.state.desired_reads_during_build == 0
                and frontier.state.source_materializations_during_build == 0
                and frontier.state.work.selected_actuations == repository.EXPECTED_POLICY_ACTUATIONS
                and frontier.state.work.committed_edits == repository.EXPECTED_POLICY_ACTUATIONS
                and frontier.state.work.revisions == 0
                and frontier.state.work.unselected_splice_executions == 0
            )
            if not valid:
                raw_status = STOP_RAW_CONTEXT_INVARIANT
                frontier = None

    inner = repository._finalize_microrepository_repair_from_snapshot_v1(
        policy,
        spec,
        cases,
        frontier,
        snapshot,
        source_reads=1,
        case_manifest_reads=case_manifest_reads,
    )
    if raw_status == recurrent.READY:
        status = PREPARED if inner.state.status == repository.PREPARED else inner.state.status
    elif inner.state.status in {
        repository.STOP_SOURCE_DRIFT,
        repository.STOP_ADAPTER_INPUT,
    }:
        status = inner.state.status
    else:
        status = raw_status
    state = _prepared_state(
        status=status,
        fitted_bridge=fitted_bridge,
        policy=policy,
        task_stream=task_stream,
        cases=cases,
        rollout=rollout,
        inner=inner,
    )
    return PreparedRawContextMicroRepositoryRepairV2(inner, state)


__all__ = [
    "ADDED_CELL_COUNT",
    "ADDED_CHECKPOINT_BYTES",
    "ADDED_TRAINABLE_BYTES",
    "CLAIM_BOUNDARY",
    "PREPARED",
    "SCHEMA",
    "STOP_CHECKPOINT",
    "STOP_RAW_CONTEXT_INVARIANT",
    "PreparedRawContextMicroRepositoryRepairV2",
    "PreparedRawContextMicroRepositoryStateV2",
    "RawContextMicroRepositoryRepairResultV2",
    "prepare_raw_context_microrepository_repair_v2",
]
