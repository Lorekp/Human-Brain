"""Bridge raw predictive chunks to a recurrent two-relation coding policy.

Training observes only authenticated continuous physical streams, visible typed
cases, and the effects produced by causal relation-cell interventions.  Each
admitted chunk trace recruits a small E/H race over the two already learned
relation cells.  A binding becomes usable only after two independent physical
provenances agree.  Duplicate provenance is a non-vote and unknown or tied
currents stop.

Recall accepts one raw stream and visible cases.  Chunk order is not relation
order: both distinct relation cells must win before the frozen recurrent policy
may perform its own affordance, actuation, effect, and commit loop.  Desired
outputs remain outside this bridge and enter only through the returned frozen
frontier's ``select`` method.

This is a narrow deterministic-motif composition experiment.  It is not a
language parser, an open-vocabulary tokenizer, or repository-level coding.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import common_causal_predictive_byte_chunker_v1 as chunker
from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from humanbrain import typed_program

SCHEMA = "COMMON_RAW_CHUNK_RECURRENT_RELATION_BRIDGE_V1"
READY = f"{SCHEMA}_READY"
STOP_TRAINING = f"STOP_{SCHEMA}_TRAINING"
STOP_BRIDGE_GATE = f"STOP_{SCHEMA}_BRIDGE_GATE"
STOP_BRIDGE_UNKNOWN = f"STOP_{SCHEMA}_UNKNOWN"
STOP_BRIDGE_AMBIGUOUS = f"STOP_{SCHEMA}_AMBIGUOUS"
STOP_RELATION_SET = f"STOP_{SCHEMA}_RELATION_SET"

RELATION_COUNT = recurrent.RELATION_COUNT
MIN_INDEPENDENT_PROVENANCES = 2
TICK_CAP = 255
RELATION_CELL_OFFSET = 9_100_000
RELATION_CELLS = tuple(
    RELATION_CELL_OFFSET + index for index in range(RELATION_COUNT)
)

PUBLIC_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
HELDOUT_INPUT_COUNT = 0

CLAIM_BOUNDARY = (
    "authenticated deterministic raw motifs activate two consequence-learned "
    "relation cells and gate one self-posed two-edit List[Int]->Int rollout; "
    "fixed predictive chunking, typed visible cases, execution, token actuation, "
    "and compilation remain scaffolding"
)

_LESSON_NAMESPACE = b"common-raw-chunk-recurrent-relation-lesson-v1"
_PROVENANCE_NAMESPACE = b"common-raw-chunk-recurrent-relation-provenance-v1"
_CHECKPOINT_NAMESPACE = b"common-raw-chunk-recurrent-relation-checkpoint-v1"
_RECALL_NAMESPACE = b"common-raw-chunk-recurrent-relation-recall-v1"


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\x00" + _json(value)).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return sha256(
        array.dtype.str.encode("ascii")
        + _json(array.shape)
        + array.tobytes(order="C")
    ).hexdigest()


def _freeze(array: np.ndarray, dtype: np.dtype[object] | type[np.generic]) -> np.ndarray:
    frozen = np.array(array, dtype=dtype, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


def _case_payload(cases: tuple[tuple[object, ...], ...]) -> list[list[object]]:
    return [[value for value in case] for case in cases]


def _lesson_sha256(
    stream: chunker.PhysicalContinuousByteStream,
    cases: tuple[tuple[object, ...], ...],
    observed_outputs: tuple[int, ...],
) -> str:
    return _digest(
        _LESSON_NAMESPACE,
        {
            "stream": stream.sha256,
            "cases": _case_payload(cases),
            "observed_outputs": observed_outputs,
        },
    )


@dataclass(frozen=True)
class PhysicalChunkRelationConsequenceLesson:
    """One raw stream paired with an observed causal output consequence."""

    stream: chunker.PhysicalContinuousByteStream
    cases: tuple[tuple[object, ...], ...]
    observed_outputs: tuple[int, ...]
    sha256: str

    def authenticate(self) -> None:
        if type(self.stream) is not chunker.PhysicalContinuousByteStream:
            raise TypeError("lesson stream must be an exact physical byte stream")
        self.stream.authenticate()
        if type(self.cases) is not tuple or not self.cases:
            raise ValueError("consequence lesson requires visible typed cases")
        if any(type(case) is not tuple for case in self.cases):
            raise TypeError("consequence cases must be tuples")
        if (
            type(self.observed_outputs) is not tuple
            or len(self.observed_outputs) != len(self.cases)
            or any(type(value) is not int for value in self.observed_outputs)
        ):
            raise ValueError("one exact integer consequence is required per case")
        if self.sha256 != _lesson_sha256(
            self.stream, self.cases, self.observed_outputs
        ):
            raise ValueError("physical chunk consequence lesson changed")


def observe_physical_chunk_relation_consequence(
    stream: chunker.PhysicalContinuousByteStream,
    cases: Iterable[tuple[object, ...]],
    observed_outputs: Iterable[int],
) -> PhysicalChunkRelationConsequenceLesson:
    """Seal an environmental consequence without a relation label or edit path."""

    frozen_cases = tuple(cases)
    frozen_outputs = tuple(observed_outputs)
    lesson = PhysicalChunkRelationConsequenceLesson(
        stream=stream,
        cases=frozen_cases,
        observed_outputs=frozen_outputs,
        sha256=_lesson_sha256(stream, frozen_cases, frozen_outputs),
    )
    lesson.authenticate()
    return lesson


class BridgeTrainingWork(NamedTuple):
    raw_lesson_count: int
    accepted_evidence_count: int
    duplicate_provenance_nonvotes: int
    unresolved_consequence_races: int
    causal_relation_races: int
    candidate_relation_executions: int
    candidate_case_executions: int
    chunker_cell_steps: int
    trace_equality_reads: int
    excitatory_ticks: int
    inhibitory_ticks: int
    learned_trace_cells: int
    learned_relation_cells: int
    checkpoint_array_bytes: int
    raw_stream_retention_count: int
    caller_relation_id_retention_count: int


class BridgeTrainingState(NamedTuple):
    status: str
    chunker_checkpoint_sha256: str
    policy_checkpoint_sha256: str
    evidence_count: int
    trace_count: int
    ready_trace_count: int
    ready_relation_count: int
    evidence_traces_sha256: str
    evidence_provenances_sha256: str
    evidence_winners_sha256: str
    traces_sha256: str
    excitatory_sha256: str
    inhibitory_sha256: str
    work: BridgeTrainingWork
    checkpoint_sha256: str


class BridgeRecallWork(NamedTuple):
    event_assembly_reads: int
    predictor_equality_reads: int
    chunker_cell_steps: int
    admission_equality_reads: int
    bridge_trace_equality_reads: int
    bridge_relation_cell_steps: int
    activated_relation_cells: int
    selected_actuations: int
    committed_edits: int
    unselected_splice_executions: int


@dataclass(frozen=True)
class RawChunkRelationRollout:
    status: str
    cut_offsets: tuple[int, ...]
    admitted_chunk_traces: tuple[tuple[int, ...], ...]
    activated_relation_cells: tuple[int, ...]
    activated_relation_receipt_sha256s: tuple[str, ...]
    relation_raw_currents: tuple[tuple[int, ...], ...]
    frontier: recurrent.RecurrentFragmentRelationFrontierV2 | None
    work: BridgeRecallWork
    checkpoint_sha256: str
    sha256: str

    def record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "cut_offsets": self.cut_offsets,
            "admitted_chunk_traces": self.admitted_chunk_traces,
            "activated_relation_cells": self.activated_relation_cells,
            "activated_relation_receipt_sha256s": (
                self.activated_relation_receipt_sha256s
            ),
            "relation_raw_currents": self.relation_raw_currents,
            "frontier_state_sha256": (
                None if self.frontier is None else self.frontier.state.sha256
            ),
            "work": self.work._asdict(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "caller_chunk_span_input": False,
            "caller_factor_tuple_input": False,
            "caller_route_input": False,
            "caller_relation_id_or_order_input": False,
            "caller_edit_path_input": False,
            "desired_outputs_during_rollout": False,
            "program_label_input": False,
            "raw_stream_retention": False,
            "sha256": self.sha256,
        }


class _ChunkExtraction(NamedTuple):
    status: str
    cuts: tuple[int, ...]
    traces: tuple[tuple[int, ...], ...]
    event_reads: int
    predictor_reads: int
    chunker_steps: int
    admission_reads: int


def _extract_chunks(
    fitted: chunker.FrozenCommonCausalPredictiveByteChunkerV1,
    stream: chunker.PhysicalContinuousByteStream,
    *,
    predictor_gate: bool,
    comparator_gate: bool,
    hypothesis_action_gate: bool,
    reset_gate: bool,
    replay_gate: bool,
    admission_gate: bool,
) -> _ChunkExtraction:
    segment = chunker._segment_recall(
        stream,
        fitted._predictor_sources,
        fitted._predictor_targets,
        fitted._predictor_support,
        fitted._boundary_excitatory,
        fitted._boundary_inhibitory,
        predictor_gate=predictor_gate,
        comparator_gate=comparator_gate,
        hypothesis_action_gate=hypothesis_action_gate,
        reset_gate=reset_gate,
        replay_gate=replay_gate,
    )
    status = segment.status
    admitted: tuple[tuple[int, ...], ...] = ()
    admission_reads = 0
    if status == chunker.RECALLED:
        status, admitted, admission_reads = fitted._match_admissions(
            segment.traces,
            segment.chunk_roots,
            admission_gate=admission_gate,
        )
    cell_steps = (
        segment.comparator_steps
        + segment.hypothesis_steps
        + segment.replay_steps
        + segment.trace_steps
    )
    return _ChunkExtraction(
        status=status,
        cuts=segment.cuts,
        traces=admitted,
        event_reads=len(stream.events) * chunker.TRACE_LANES,
        predictor_reads=segment.predictor_reads,
        chunker_steps=cell_steps,
        admission_reads=admission_reads,
    )


def _provenance_root(stream: chunker.PhysicalContinuousByteStream) -> bytes:
    return bytes.fromhex(
        _digest(_PROVENANCE_NAMESPACE, {"cells": stream.provenance_cells})
    )


def _relation_program(
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    record: recurrent.FragmentRelationReceipt,
) -> typed_program.Program:
    base = typed_program.Program(record.base_program_tokens, policy._library._signature)
    sites = tuple(
        site
        for site in typed_program.program_edit_sites(base)
        if site.path and recurrent._role_descriptor(base, site)[0] == record.precondition_sha256
    )
    if len(sites) != 1:
        raise RuntimeError("causal relation intervention lost its unique affordance")
    program = typed_program.Program(
        fragments._splice_tokens(base, sites[0].path, record.donor_tokens),
        policy._library._signature,
    )
    if program.sha256 != record.result_program_sha256:
        raise RuntimeError("causal relation intervention changed")
    return program


def _causal_relation_race(
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    lesson: PhysicalChunkRelationConsequenceLesson,
) -> tuple[semantic_race.TypedSemanticVectorRaceAudit, int]:
    vectors = []
    for record in policy.relation_receipts:
        program = _relation_program(policy, record)
        values = fragments._execute_without_rendering(
            program.tokens,
            policy._library._signature,
            lesson.cases,
        )
        encoded = fragments._encode_outputs(values)
        if encoded is None:
            raise RuntimeError("candidate relation consequence was not encodable")
        vectors.append(encoded)
    observed = fragments._encode_outputs(lesson.observed_outputs)
    if observed is None:
        raise RuntimeError("observed relation consequence was not encodable")
    race = semantic_race.select_exact_semantic_vector(
        tuple(f"bridge-relation-cell:{index}" for index in range(RELATION_COUNT)),
        tuple(vectors),
        observed,
    )
    return race, RELATION_COUNT * len(lesson.cases)


def _row_winner(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    row: int,
) -> tuple[str, int | None, tuple[int, ...]]:
    excitation = excitatory[row].astype(np.int16)
    currents = excitation - inhibitory[row].astype(np.int16)
    eligible = np.flatnonzero(excitation >= MIN_INDEPENDENT_PROVENANCES)
    if eligible.size == 0:
        return STOP_BRIDGE_UNKNOWN, None, tuple(int(value) for value in currents)
    best = int(np.max(currents[eligible]))
    winners = tuple(int(index) for index in eligible if int(currents[index]) == best)
    if len(winners) != 1:
        return STOP_BRIDGE_AMBIGUOUS, None, tuple(int(value) for value in currents)
    if best <= 0:
        return STOP_BRIDGE_UNKNOWN, None, tuple(int(value) for value in currents)
    return READY, winners[0], tuple(int(value) for value in currents)


def _checkpoint_sha256(
    chunker_checkpoint: str,
    policy_checkpoint: str,
    state: BridgeTrainingState,
    arrays: tuple[np.ndarray, ...],
) -> str:
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "chunker": chunker_checkpoint,
            "policy": policy_checkpoint,
            "state": state._asdict() | {"checkpoint_sha256": ""},
            "arrays": tuple(_array_sha256(array) for array in arrays),
            "claim": CLAIM_BOUNDARY,
        },
    )


class FrozenCommonRawChunkRecurrentRelationBridgeV1:
    """Frozen trace-to-relation E/H state over banked child mechanisms."""

    def __init__(
        self,
        *,
        fitted_chunker: chunker.FrozenCommonCausalPredictiveByteChunkerV1,
        policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
        arrays: tuple[np.ndarray, ...],
        state: BridgeTrainingState,
    ) -> None:
        self._chunker = fitted_chunker
        self._policy = policy
        (
            evidence_traces,
            evidence_provenances,
            evidence_winners,
            traces,
            excitatory,
            inhibitory,
        ) = arrays
        self._evidence_traces = _freeze(evidence_traces, np.int32)
        self._evidence_provenances = _freeze(evidence_provenances, np.uint8)
        self._evidence_winners = _freeze(evidence_winners, np.uint8)
        self._traces = _freeze(traces, np.int32)
        self._excitatory = _freeze(excitatory, np.uint8)
        self._inhibitory = _freeze(inhibitory, np.uint8)
        self._state = state
        self.authenticate()

    @property
    def state(self) -> BridgeTrainingState:
        return self._state

    @property
    def checkpoint_sha256(self) -> str:
        return self._state.checkpoint_sha256

    @property
    def excitatory(self) -> np.ndarray:
        return self._excitatory

    @property
    def inhibitory(self) -> np.ndarray:
        return self._inhibitory

    def _arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self._evidence_traces,
            self._evidence_provenances,
            self._evidence_winners,
            self._traces,
            self._excitatory,
            self._inhibitory,
        )

    def authenticate(self) -> None:
        if type(self._chunker) is not chunker.FrozenCommonCausalPredictiveByteChunkerV1:
            raise TypeError("exact frozen predictive chunker required")
        if type(self._policy) is not recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
            raise TypeError("exact frozen recurrent relation policy required")
        self._chunker.authenticate()
        self._policy.authenticate()
        evidence_count = self._evidence_winners.shape[0]
        trace_count = self._traces.shape[0]
        arrays = self._arrays()
        if (
            self._evidence_traces.shape != (evidence_count, chunker.TRACE_LANES)
            or self._evidence_provenances.shape != (evidence_count, 32)
            or self._traces.shape != (trace_count, chunker.TRACE_LANES)
            or self._excitatory.shape != (trace_count, RELATION_COUNT)
            or self._inhibitory.shape != self._excitatory.shape
            or self._evidence_traces.dtype != np.dtype(np.int32)
            or self._evidence_provenances.dtype != np.dtype(np.uint8)
            or self._evidence_winners.dtype != np.dtype(np.uint8)
            or self._traces.dtype != np.dtype(np.int32)
            or self._excitatory.dtype != np.dtype(np.uint8)
            or self._inhibitory.dtype != np.dtype(np.uint8)
            or any(array.flags.writeable for array in arrays)
        ):
            raise RuntimeError("raw chunk relation bridge geometry changed")
        ready = tuple(
            winner
            for row in range(trace_count)
            for status, winner, _ in (
                (_row_winner(self._excitatory, self._inhibitory, row),)
            )
            if status == READY and winner is not None
        )
        state = self._state
        expected = _checkpoint_sha256(
            self._chunker.checkpoint_sha256,
            self._policy.checkpoint_sha256,
            state,
            arrays,
        )
        if (
            state.status != READY
            or state.chunker_checkpoint_sha256 != self._chunker.checkpoint_sha256
            or state.policy_checkpoint_sha256 != self._policy.checkpoint_sha256
            or state.evidence_count != evidence_count
            or state.trace_count != trace_count
            or state.ready_trace_count != len(ready)
            or state.ready_relation_count != len(set(ready))
            or state.evidence_traces_sha256 != _array_sha256(self._evidence_traces)
            or state.evidence_provenances_sha256
            != _array_sha256(self._evidence_provenances)
            or state.evidence_winners_sha256
            != _array_sha256(self._evidence_winners)
            or state.traces_sha256 != _array_sha256(self._traces)
            or state.excitatory_sha256 != _array_sha256(self._excitatory)
            or state.inhibitory_sha256 != _array_sha256(self._inhibitory)
            or state.work.checkpoint_array_bytes != sum(array.nbytes for array in arrays)
            or state.work.raw_stream_retention_count != 0
            or state.work.caller_relation_id_retention_count != 0
            or state.checkpoint_sha256 != expected
        ):
            raise RuntimeError("raw chunk relation bridge checkpoint changed")

    def _result(
        self,
        *,
        status: str,
        extraction: _ChunkExtraction,
        cells: tuple[int, ...] = (),
        receipt_sha256s: tuple[str, ...] = (),
        currents: tuple[tuple[int, ...], ...] = (),
        frontier: recurrent.RecurrentFragmentRelationFrontierV2 | None = None,
        bridge_reads: int = 0,
        bridge_steps: int = 0,
    ) -> RawChunkRelationRollout:
        selected_actuations = 0 if frontier is None else frontier.state.work.selected_actuations
        committed = 0 if frontier is None else frontier.state.work.committed_edits
        unselected = (
            0 if frontier is None else frontier.state.work.unselected_splice_executions
        )
        work = BridgeRecallWork(
            event_assembly_reads=extraction.event_reads,
            predictor_equality_reads=extraction.predictor_reads,
            chunker_cell_steps=extraction.chunker_steps,
            admission_equality_reads=extraction.admission_reads,
            bridge_trace_equality_reads=bridge_reads,
            bridge_relation_cell_steps=bridge_steps,
            activated_relation_cells=len(cells),
            selected_actuations=selected_actuations,
            committed_edits=committed,
            unselected_splice_executions=unselected,
        )
        record = {
            "status": status,
            "cuts": extraction.cuts,
            "traces": extraction.traces,
            "cells": cells,
            "receipts": receipt_sha256s,
            "currents": currents,
            "frontier": None if frontier is None else frontier.state.sha256,
            "work": work._asdict(),
            "checkpoint": self.checkpoint_sha256,
        }
        return RawChunkRelationRollout(
            status=status,
            cut_offsets=extraction.cuts,
            admitted_chunk_traces=extraction.traces,
            activated_relation_cells=cells,
            activated_relation_receipt_sha256s=receipt_sha256s,
            relation_raw_currents=currents,
            frontier=frontier,
            work=work,
            checkpoint_sha256=self.checkpoint_sha256,
            sha256=_digest(_RECALL_NAMESPACE, record),
        )

    def recall(
        self,
        stream: chunker.PhysicalContinuousByteStream,
        cases: Iterable[tuple[object, ...]],
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
    ) -> RawChunkRelationRollout:
        """Segment, reactivate a relation set, then run desired-free construction."""

        self.authenticate()
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
            raise TypeError("all raw chunk relation gates must be exact booleans")
        frozen_cases = tuple(cases)
        extraction = _extract_chunks(
            self._chunker,
            stream,
            predictor_gate=predictor_gate,
            comparator_gate=comparator_gate,
            hypothesis_action_gate=hypothesis_action_gate,
            reset_gate=reset_gate,
            replay_gate=replay_gate,
            admission_gate=admission_gate,
        )
        if extraction.status != chunker.RECALLED:
            return self._result(status=extraction.status, extraction=extraction)
        if not bridge_cell_gate:
            return self._result(status=STOP_BRIDGE_GATE, extraction=extraction)

        indices: list[int] = []
        cells: list[int] = []
        receipts: list[str] = []
        currents: list[tuple[int, ...]] = []
        reads = 0
        steps = 0
        for trace in extraction.traces:
            reads += self._traces.shape[0] * chunker.TRACE_LANES
            rows = np.flatnonzero(
                np.all(
                    self._traces == np.asarray(trace, dtype=np.int32),
                    axis=1,
                )
            )
            if rows.size == 0:
                return self._result(
                    status=STOP_BRIDGE_UNKNOWN,
                    extraction=extraction,
                    currents=tuple(currents),
                    bridge_reads=reads,
                    bridge_steps=steps,
                )
            if rows.size != 1:
                return self._result(
                    status=STOP_BRIDGE_AMBIGUOUS,
                    extraction=extraction,
                    currents=tuple(currents),
                    bridge_reads=reads,
                    bridge_steps=steps,
                )
            row = int(rows[0])
            row_status, winner, row_currents = _row_winner(
                self._excitatory, self._inhibitory, row
            )
            currents.append(row_currents)
            steps += RELATION_COUNT + 1
            if row_status != READY or winner is None:
                return self._result(
                    status=row_status,
                    extraction=extraction,
                    currents=tuple(currents),
                    bridge_reads=reads,
                    bridge_steps=steps,
                )
            indices.append(winner)
            cells.append(RELATION_CELLS[winner])
            receipts.append(self._policy.relation_receipts[winner].receipt_sha256)

        if len(indices) != RELATION_COUNT or len(set(indices)) != RELATION_COUNT:
            return self._result(
                status=STOP_RELATION_SET,
                extraction=extraction,
                cells=tuple(cells),
                receipt_sha256s=tuple(receipts),
                currents=tuple(currents),
                bridge_reads=reads,
                bridge_steps=steps,
            )
        frontier = self._policy.build(
            self._policy._library._signature,
            frozen_cases,
            relation_cell_gate=relation_cell_gate,
            policy_cell_gate=policy_cell_gate,
            effect_cell_gate=effect_cell_gate,
            actuation_gate=actuation_gate,
            replay_cell_gate=relation_replay_cell_gate,
        )
        return self._result(
            status=frontier.status,
            extraction=extraction,
            cells=tuple(cells),
            receipt_sha256s=tuple(receipts),
            currents=tuple(currents),
            frontier=frontier,
            bridge_reads=reads,
            bridge_steps=steps,
        )


def fit_common_raw_chunk_recurrent_relation_bridge_v1(
    fitted_chunker: chunker.FrozenCommonCausalPredictiveByteChunkerV1,
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    lessons: Sequence[PhysicalChunkRelationConsequenceLesson],
    *,
    bridge_plasticity_gate: bool = True,
) -> FrozenCommonRawChunkRecurrentRelationBridgeV1:
    """Learn trace-to-relation E/H bindings from causal consequence lessons."""

    if type(fitted_chunker) is not chunker.FrozenCommonCausalPredictiveByteChunkerV1:
        raise TypeError("exact frozen predictive chunker required")
    if type(policy) is not recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
        raise TypeError("exact frozen recurrent relation policy required")
    if type(bridge_plasticity_gate) is not bool:
        raise TypeError("bridge plasticity gate must be an exact boolean")
    if not bridge_plasticity_gate:
        raise RuntimeError(STOP_BRIDGE_GATE)
    fitted_chunker.authenticate()
    policy.authenticate()
    frozen_lessons = tuple(lessons)
    if not frozen_lessons or any(
        type(lesson) is not PhysicalChunkRelationConsequenceLesson
        for lesson in frozen_lessons
    ):
        raise ValueError("physical consequence lessons are required")

    evidence: list[tuple[tuple[int, ...], bytes, int]] = []
    seen: set[tuple[tuple[int, ...], bytes]] = set()
    duplicate_nonvotes = unresolved = relation_races = 0
    candidate_programs = candidate_cases = chunker_steps = admission_reads = 0
    for lesson in frozen_lessons:
        lesson.authenticate()
        extraction = _extract_chunks(
            fitted_chunker,
            lesson.stream,
            predictor_gate=True,
            comparator_gate=True,
            hypothesis_action_gate=True,
            reset_gate=True,
            replay_gate=True,
            admission_gate=True,
        )
        chunker_steps += extraction.chunker_steps
        admission_reads += extraction.admission_reads
        if extraction.status != chunker.RECALLED or len(extraction.traces) != 1:
            raise RuntimeError(
                f"{STOP_TRAINING}: consequence lesson did not recover one admitted chunk"
            )
        trace = extraction.traces[0]
        provenance = _provenance_root(lesson.stream)
        key = (trace, provenance)
        if key in seen:
            duplicate_nonvotes += 1
            continue
        seen.add(key)
        race, case_executions = _causal_relation_race(policy, lesson)
        relation_races += 1
        candidate_programs += RELATION_COUNT
        candidate_cases += case_executions
        if race.status != semantic_race.SELECTED or race.winner_key is None:
            unresolved += 1
            continue
        winner = int(race.winner_key.rsplit(":", 1)[1])
        evidence.append((trace, provenance, winner))

    trace_rows = tuple(dict.fromkeys(item[0] for item in evidence))
    row_by_trace = {trace: row for row, trace in enumerate(trace_rows)}
    excitatory = np.zeros((len(trace_rows), RELATION_COUNT), dtype=np.uint8)
    inhibitory = np.zeros_like(excitatory)
    for trace, _, winner in evidence:
        row = row_by_trace[trace]
        excitatory[row, winner] = min(TICK_CAP, int(excitatory[row, winner]) + 1)
        for other in range(RELATION_COUNT):
            if other != winner:
                inhibitory[row, other] = min(
                    TICK_CAP, int(inhibitory[row, other]) + 1
                )
    ready = tuple(
        winner
        for row in range(len(trace_rows))
        for status, winner, _ in ((_row_winner(excitatory, inhibitory, row),))
        if status == READY and winner is not None
    )
    if len(set(ready)) != RELATION_COUNT:
        raise RuntimeError(
            f"{STOP_TRAINING}: two independently supported relation bindings required"
        )

    evidence_traces = np.asarray(
        [item[0] for item in evidence], dtype=np.int32
    ).reshape(-1, chunker.TRACE_LANES)
    evidence_provenances = np.asarray(
        [list(item[1]) for item in evidence], dtype=np.uint8
    ).reshape(-1, 32)
    evidence_winners = np.asarray(
        [item[2] for item in evidence], dtype=np.uint8
    )
    traces = np.asarray(trace_rows, dtype=np.int32).reshape(
        -1, chunker.TRACE_LANES
    )
    arrays = tuple(
        _freeze(array, array.dtype)
        for array in (
            evidence_traces,
            evidence_provenances,
            evidence_winners,
            traces,
            excitatory,
            inhibitory,
        )
    )
    work = BridgeTrainingWork(
        raw_lesson_count=len(frozen_lessons),
        accepted_evidence_count=len(evidence),
        duplicate_provenance_nonvotes=duplicate_nonvotes,
        unresolved_consequence_races=unresolved,
        causal_relation_races=relation_races,
        candidate_relation_executions=candidate_programs,
        candidate_case_executions=candidate_cases,
        chunker_cell_steps=chunker_steps,
        trace_equality_reads=admission_reads,
        excitatory_ticks=int(excitatory.sum()),
        inhibitory_ticks=int(inhibitory.sum()),
        learned_trace_cells=len(trace_rows),
        learned_relation_cells=RELATION_COUNT,
        checkpoint_array_bytes=sum(array.nbytes for array in arrays),
        raw_stream_retention_count=0,
        caller_relation_id_retention_count=0,
    )
    partial = BridgeTrainingState(
        status=READY,
        chunker_checkpoint_sha256=fitted_chunker.checkpoint_sha256,
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        evidence_count=len(evidence),
        trace_count=len(trace_rows),
        ready_trace_count=len(ready),
        ready_relation_count=len(set(ready)),
        evidence_traces_sha256=_array_sha256(arrays[0]),
        evidence_provenances_sha256=_array_sha256(arrays[1]),
        evidence_winners_sha256=_array_sha256(arrays[2]),
        traces_sha256=_array_sha256(arrays[3]),
        excitatory_sha256=_array_sha256(arrays[4]),
        inhibitory_sha256=_array_sha256(arrays[5]),
        work=work,
        checkpoint_sha256="",
    )
    state = partial._replace(
        checkpoint_sha256=_checkpoint_sha256(
            fitted_chunker.checkpoint_sha256,
            policy.checkpoint_sha256,
            partial,
            arrays,
        )
    )
    return FrozenCommonRawChunkRecurrentRelationBridgeV1(
        fitted_chunker=fitted_chunker,
        policy=policy,
        arrays=arrays,
        state=state,
    )
