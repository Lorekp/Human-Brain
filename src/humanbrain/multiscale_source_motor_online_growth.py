"""Replay-free append-only growth for complete query/workspace motor learning.

Each accepted evaluator-approved transition grows one new neuronal
``BoundaryMultiscaleSourceMotorController`` shard from only the current
complete query and lossless multi-file workspace.  Older raw experiences are
never replayed.  Stored controllers are frozen and copied for every inference,
then an outer fitted-GLIF race either selects one emitting shard or physically
selects STOP when support is absent or collides.

The implementation deliberately preserves the claim boundary of the wrapped
controller: it grows exact learned neuronal structure, not language semantics
or arbitrary edit programs.  Learned event bytes remain synaptic content even
though complete raw training sequences are discarded.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from typing import Protocol

import numpy as np

from humanbrain.boundary_multiscale_source_motor_controller import (
    BoundaryMultiscaleSourceMotorAblation,
    BoundaryMultiscaleSourceMotorController,
    SuccessfulBoundaryMultiscaleSourceMotorExperience,
)
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.source_relative_action_tape import (
    OccurrencePreservingActionTape,
    TapeValence,
)

__all__ = [
    "AppendOnlyMultiscaleSourceMotorStack",
    "BoundaryMultiscaleSourceMotorAblation",
    "MultiscaleSourceMotorExecutionAuthority",
    "MultiscaleSourceMotorExecutionReceipt",
    "MultiscaleSourceMotorSuccessEvidence",
    "OnlineMultiscaleGrowthAblation",
    "OnlineMultiscaleSourceMotorDecision",
    "OnlineMultiscaleSourceMotorGrowthReceipt",
    "OnlineMultiscaleSourceMotorGrowthTransition",
    "OnlineMultiscaleSourceMotorSummary",
    "OnlineMultiscaleStructuralRaceTrace",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_SCORE_CAP = 2
_TONIC_STOP_SCORE = 1
_EMIT_SCORE = 2

_OBSERVATION_NAMESPACE = b"humanbrain.boundary-multiscale-online.observation-v1\0"
_ACTION_NAMESPACE = b"humanbrain.boundary-multiscale-online.action-v1\0"
_EXECUTION_RECEIPT_NAMESPACE = b"humanbrain.boundary-multiscale-online.execution-receipt-v1\0"
_EVIDENCE_NAMESPACE = b"humanbrain.boundary-multiscale-online.evidence-v1\0"
_TRAJECTORY_NAMESPACE = b"humanbrain.boundary-multiscale-online.trajectory-v1\0"
_SEGMENT_NAMESPACE = b"humanbrain.boundary-multiscale-online.segment-v1\0"
_PROVENANCE_NAMESPACE = b"humanbrain.boundary-multiscale-online.provenance-v1\0"
_CHECKPOINT_NAMESPACE = b"humanbrain.boundary-multiscale-online.checkpoint-v1\0"
_QUERY_NAMESPACE = b"humanbrain.boundary-multiscale-online.query-v1\0"
_RACE_NAMESPACE = b"humanbrain.boundary-multiscale-online.race-v1\0"
_DECISION_NAMESPACE = b"humanbrain.boundary-multiscale-online.decision-v1\0"
_GROWTH_NAMESPACE = b"humanbrain.boundary-multiscale-online.growth-v1\0"
_EXECUTION_RECEIPT_CONSTRUCTION_TOKEN = object()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _canonical_observation(
    query: object,
    files: object,
) -> tuple[bytes, tuple[OpaqueWorkspaceFile, ...]]:
    if not isinstance(query, bytes) or not query:
        raise ValueError("query must be non-empty bytes")
    if type(files) is not tuple:
        raise TypeError("files must be a tuple")
    observation = MultiscaleOpaqueEventTape.observe(query, files)
    return observation.reconstruct_query(), observation.reconstruct_workspace()


def _workspace_record(files: tuple[OpaqueWorkspaceFile, ...]) -> tuple[dict[str, str], ...]:
    return tuple({"content_hex": item.content.hex(), "path": item.path} for item in files)


def _observation_sha256(
    query: bytes,
    before_files: tuple[OpaqueWorkspaceFile, ...],
) -> str:
    return _digest(
        _OBSERVATION_NAMESPACE,
        {
            "before_files": _workspace_record(before_files),
            "query_hex": query.hex(),
        },
    )


def _action_sha256(after_files: tuple[OpaqueWorkspaceFile, ...]) -> str:
    return _digest(
        _ACTION_NAMESPACE,
        {"after_files": _workspace_record(after_files)},
    )


def _expected_emissions(
    query: bytes,
    before_files: tuple[OpaqueWorkspaceFile, ...],
    after_files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[bytes, ...]:
    before = MultiscaleOpaqueEventTape.observe(query, before_files)
    after = MultiscaleOpaqueEventTape.observe(query, after_files)
    canonical_before = before.reconstruct_workspace()
    canonical_after = after.reconstruct_workspace()
    before_paths = tuple(item.path for item in canonical_before)
    after_paths = tuple(item.path for item in canonical_after)
    if before_paths != after_paths:
        raise ValueError("successful evidence must retain every workspace path")
    changed = tuple(
        (before_item, after_item)
        for before_item, after_item in zip(canonical_before, canonical_after, strict=True)
        if before_item.content != after_item.content
    )
    if len(changed) != 1:
        raise ValueError("successful evidence must change exactly one file")
    changed_before, changed_after = changed[0]
    return before.motor_tokens_for_replacement(
        changed_before.path,
        changed_after.content,
    )


def _build_score_currents() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(_RACE_STEPS)
    crossing_current = population.parameters.threshold_inf_relative_mv / (voltage - threshold)
    currents: list[float] = []
    for score in range(_SCORE_CAP + 1):
        first_step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("multiscale online fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(row.spike_step != _ZERO_SCORE_SPIKE_STEP - score for score, row in enumerate(controls)):
        raise RuntimeError("multiscale online fitted-GLIF calibration changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class OnlineMultiscaleGrowthAblation(StrEnum):
    NONE = "none"
    NO_UPDATE = "no_update"


@dataclass(frozen=True, slots=True, init=False)
class MultiscaleSourceMotorExecutionReceipt:
    """Authority-issued PASS for one learner, full observation, and workspace action."""

    learner_checkpoint_sha256: str
    observation_sha256: str
    action_sha256: str
    evaluator_sha256: str
    outcome: str
    sha256: str

    def __init__(
        self,
        construction_token: object,
        *,
        learner_checkpoint_sha256: str,
        observation_sha256: str,
        action_sha256: str,
        evaluator_sha256: str,
        outcome: str,
        sha256: str,
    ) -> None:
        if construction_token is not _EXECUTION_RECEIPT_CONSTRUCTION_TOKEN:
            raise TypeError("execution receipts are issued only by execution authority")
        object.__setattr__(self, "learner_checkpoint_sha256", learner_checkpoint_sha256)
        object.__setattr__(self, "observation_sha256", observation_sha256)
        object.__setattr__(self, "action_sha256", action_sha256)
        object.__setattr__(self, "evaluator_sha256", evaluator_sha256)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "sha256", sha256)

    def authenticate(self) -> None:
        learner = _require_sha256(
            self.learner_checkpoint_sha256,
            "learner_checkpoint_sha256",
        )
        observation = _require_sha256(self.observation_sha256, "observation_sha256")
        action = _require_sha256(self.action_sha256, "action_sha256")
        evaluator = _require_sha256(self.evaluator_sha256, "evaluator_sha256")
        if self.outcome != "PASS":
            raise ValueError("successful multiscale evidence requires a PASS receipt")
        record = {
            "action_sha256": action,
            "evaluator_sha256": evaluator,
            "learner_checkpoint_sha256": learner,
            "observation_sha256": observation,
            "outcome": self.outcome,
        }
        if self.sha256 != _digest(_EXECUTION_RECEIPT_NAMESPACE, record):
            raise ValueError("multiscale execution receipt root changed")


class MultiscaleSourceMotorExecutionAuthority:
    """Invoke one caller-pinned evaluator before issuing an in-process PASS receipt."""

    __slots__ = ("_evaluator", "_evaluator_sha256")

    def __init__(
        self,
        *,
        evaluator: Callable[..., object],
        evaluator_sha256: str,
    ) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self._evaluator = evaluator
        self._evaluator_sha256 = _require_sha256(evaluator_sha256, "evaluator_sha256")

    def execute_and_issue(
        self,
        *,
        learner_checkpoint_sha256: str,
        query: bytes,
        before_files: tuple[OpaqueWorkspaceFile, ...],
        after_files: tuple[OpaqueWorkspaceFile, ...],
    ) -> MultiscaleSourceMotorExecutionReceipt:
        learner = _require_sha256(
            learner_checkpoint_sha256,
            "learner_checkpoint_sha256",
        )
        canonical_query, canonical_before = _canonical_observation(query, before_files)
        _, canonical_after = _canonical_observation(query, after_files)
        _expected_emissions(canonical_query, canonical_before, canonical_after)
        passed = self._evaluator(
            query=canonical_query,
            before_files=canonical_before,
            after_files=canonical_after,
        )
        if passed is not True:
            raise ValueError("frozen evaluator did not PASS the multiscale action")
        observation = _observation_sha256(canonical_query, canonical_before)
        action = _action_sha256(canonical_after)
        record = {
            "action_sha256": action,
            "evaluator_sha256": self._evaluator_sha256,
            "learner_checkpoint_sha256": learner,
            "observation_sha256": observation,
            "outcome": "PASS",
        }
        receipt = MultiscaleSourceMotorExecutionReceipt(
            _EXECUTION_RECEIPT_CONSTRUCTION_TOKEN,
            learner_checkpoint_sha256=learner,
            observation_sha256=observation,
            action_sha256=action,
            evaluator_sha256=self._evaluator_sha256,
            outcome="PASS",
            sha256=_digest(_EXECUTION_RECEIPT_NAMESPACE, record),
        )
        receipt.authenticate()
        return receipt


@dataclass(frozen=True, slots=True)
class MultiscaleSourceMotorSuccessEvidence:
    """One complete successful transition bound to evaluator and learner custody."""

    query: bytes
    before_files: tuple[OpaqueWorkspaceFile, ...]
    after_files: tuple[OpaqueWorkspaceFile, ...]
    observation_sha256: str
    action_sha256: str
    execution_receipt: MultiscaleSourceMotorExecutionReceipt
    provenance_id: str
    sha256: str

    @classmethod
    def seal(
        cls,
        *,
        query: bytes,
        before_files: tuple[OpaqueWorkspaceFile, ...],
        after_files: tuple[OpaqueWorkspaceFile, ...],
        execution_receipt: MultiscaleSourceMotorExecutionReceipt,
        provenance_id: str,
    ) -> MultiscaleSourceMotorSuccessEvidence:
        canonical_query, canonical_before = _canonical_observation(query, before_files)
        _, canonical_after = _canonical_observation(query, after_files)
        _expected_emissions(canonical_query, canonical_before, canonical_after)
        if not isinstance(execution_receipt, MultiscaleSourceMotorExecutionReceipt):
            raise TypeError("execution_receipt must be MultiscaleSourceMotorExecutionReceipt")
        execution_receipt.authenticate()
        if not isinstance(provenance_id, str) or not provenance_id:
            raise ValueError("provenance_id must be non-empty text")
        observation = _observation_sha256(canonical_query, canonical_before)
        action = _action_sha256(canonical_after)
        if execution_receipt.observation_sha256 != observation:
            raise ValueError("execution receipt does not bind the complete observation")
        if execution_receipt.action_sha256 != action:
            raise ValueError("execution receipt does not bind the complete action")
        record = {
            "action_sha256": action,
            "execution_receipt_sha256": execution_receipt.sha256,
            "observation_sha256": observation,
            "provenance_id": provenance_id,
        }
        return cls(
            query=canonical_query,
            before_files=canonical_before,
            after_files=canonical_after,
            observation_sha256=observation,
            action_sha256=action,
            execution_receipt=execution_receipt,
            provenance_id=provenance_id,
            sha256=_digest(_EVIDENCE_NAMESPACE, record),
        )

    def authenticate(self) -> None:
        query, before = _canonical_observation(self.query, self.before_files)
        _, after = _canonical_observation(self.query, self.after_files)
        _expected_emissions(query, before, after)
        if not isinstance(self.execution_receipt, MultiscaleSourceMotorExecutionReceipt):
            raise TypeError("execution_receipt must be MultiscaleSourceMotorExecutionReceipt")
        self.execution_receipt.authenticate()
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be non-empty text")
        observation = _observation_sha256(query, before)
        action = _action_sha256(after)
        if self.observation_sha256 != observation:
            raise ValueError("successful evidence observation custody changed")
        if self.action_sha256 != action:
            raise ValueError("successful evidence action custody changed")
        if self.execution_receipt.observation_sha256 != observation:
            raise ValueError("execution receipt does not bind the complete observation")
        if self.execution_receipt.action_sha256 != action:
            raise ValueError("execution receipt does not bind the complete action")
        record = {
            "action_sha256": action,
            "execution_receipt_sha256": self.execution_receipt.sha256,
            "observation_sha256": observation,
            "provenance_id": self.provenance_id,
        }
        if self.sha256 != _digest(_EVIDENCE_NAMESPACE, record):
            raise ValueError("successful multiscale evidence root changed")


@dataclass(frozen=True, slots=True)
class OnlineMultiscaleStructuralRaceTrace:
    population: str
    candidate_cell_ids: tuple[int, ...]
    candidate_scores: tuple[int, ...]
    tonic_stop_cell_id: int
    winner_cell_id: int
    spike_step: int
    simultaneous_spike_candidates: int
    stopped: bool
    stop_reason: str | None
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class OnlineMultiscaleSourceMotorSummary:
    segment_count: int
    reserved_cell_capacity_estimate: int
    reserved_synapse_capacity_estimate: int
    learned_event_bytes_retained_estimate: int
    excitatory_support_tick_estimate: int
    next_reserved_cell_address: int
    next_reserved_synapse_address: int
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class OnlineMultiscaleSourceMotorDecision:
    decision: str
    emissions: tuple[bytes, ...]
    winning_segment_ordinal: int | None
    winning_selector_cell_id: int
    stop_reason: str | None
    component_decision_sha256s: tuple[str, ...]
    component_receipt_sha256: str | None
    race: OnlineMultiscaleStructuralRaceTrace
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class OnlineMultiscaleSourceMotorGrowthReceipt:
    status: str
    stop_reason: str | None
    parent_checkpoint_sha256: str
    successor_checkpoint_sha256: str
    evidence_sha256: str
    observation_sha256: str
    action_sha256: str
    execution_receipt_sha256: str
    compiled_trajectory_sha256: str
    prior_provenance_sha256: str
    successor_provenance_sha256: str
    reserved_cell_address_start: int | None
    reserved_cell_capacity_estimate_added: int
    reserved_synapse_address_start: int | None
    reserved_synapse_capacity_estimate_added: int
    learned_event_bytes_retained_estimate_added: int
    excitatory_support_tick_estimate_added: int
    coactivation_support_tick_estimate_added: int
    positive_success_support_tick_estimate_added: int
    inhibitory_support_tick_estimate_added: int
    source_experience_count: int
    current_experience_training_rows: int
    committed_evidence_tick_phases: int
    parent_experience_replay_count: int
    raw_training_sequences_retained: int
    admission_race: OnlineMultiscaleStructuralRaceTrace | None
    sha256: str


@dataclass(frozen=True, slots=True)
class OnlineMultiscaleSourceMotorGrowthTransition:
    successor: AppendOnlyMultiscaleSourceMotorStack
    receipt: OnlineMultiscaleSourceMotorGrowthReceipt


@dataclass(frozen=True, slots=True)
class _ComponentEmission:
    emitted: bool
    emissions: tuple[bytes, ...]
    stop_reason: str | None
    decision_sha256: str
    receipt_sha256: str | None
    physical_cell_steps: int


class _FrozenController(Protocol):
    @property
    def checkpoint_sha256(self) -> str: ...

    def respond_observation(
        self,
        observation: MultiscaleOpaqueEventTape,
        *,
        ablation: BoundaryMultiscaleSourceMotorAblation,
    ) -> _ComponentEmission: ...


def _copy_action_tape(
    tape: OccurrencePreservingActionTape,
) -> OccurrencePreservingActionTape:
    if tape._pending is not None:
        raise RuntimeError("cannot freeze a multiscale action tape with a pending decision")
    cloned = copy(tape)
    cloned._adjustments = dict(tape._adjustments)
    cloned._pending = None
    return cloned


def _copy_controller(
    controller: BoundaryMultiscaleSourceMotorController,
) -> BoundaryMultiscaleSourceMotorController:
    if controller._action_tape._pending is not None:
        raise RuntimeError("cannot freeze a multiscale controller with a pending decision")
    cloned = copy(controller)
    cloned._binders = tuple(copy(binder) for binder in controller._binders)
    cloned._action_tape = _copy_action_tape(controller._action_tape)
    return cloned


@dataclass(frozen=True, slots=True)
class _ControllerAdapter:
    controller: BoundaryMultiscaleSourceMotorController
    expected_checkpoint_sha256: str

    @property
    def checkpoint_sha256(self) -> str:
        if self.controller._action_tape._pending is not None:
            raise RuntimeError("frozen multiscale controller gained pending state")
        if self.controller.checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise RuntimeError("frozen multiscale controller checkpoint changed")
        return self.expected_checkpoint_sha256

    def respond_observation(
        self,
        observation: MultiscaleOpaqueEventTape,
        *,
        ablation: BoundaryMultiscaleSourceMotorAblation,
    ) -> _ComponentEmission:
        _ = self.checkpoint_sha256
        ephemeral = _copy_controller(self.controller)
        decision = ephemeral._respond_observation(observation, ablation=ablation)
        receipt_sha256: str | None = None
        if decision.decision == "EMIT":
            if decision.action is None or decision.action.eligibility is None:
                raise RuntimeError("multiscale EMIT lost its exact eligibility receipt")
            receipt_sha256 = decision.action.eligibility.sha256
            consequence = ephemeral.resolve(TapeValence.NEUTRAL)
            if consequence.receipt_sha256 != receipt_sha256:
                raise RuntimeError("multiscale neutral resolution lost receipt custody")
        if ephemeral.checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise RuntimeError("ephemeral multiscale response changed learned state")
        _ = self.checkpoint_sha256
        stop_reason = None
        if decision.decision == "STOP":
            stop_reason = f"{decision.stop_stage}:{decision.stop_reason}"
        return _ComponentEmission(
            emitted=decision.decision == "EMIT",
            emissions=decision.emissions,
            stop_reason=stop_reason,
            decision_sha256=decision.sha256,
            receipt_sha256=receipt_sha256,
            physical_cell_steps=decision.physical_cell_steps,
        )


@dataclass(frozen=True, slots=True)
class _ControllerEstimates:
    reserved_cell_capacity: int
    reserved_synapse_capacity: int
    learned_event_bytes: int
    excitatory_support_ticks: int


def _controller_estimates(
    controller: BoundaryMultiscaleSourceMotorController,
) -> _ControllerEstimates:
    action_summary = controller._action_tape.training_summary
    binder_cells = 0
    binder_synapses = 0
    binder_ticks = 0
    learned_event_bytes = sum(len(event) for event in controller._action_tape._contexts)
    for binder in controller._binders:
        summary = binder.training_summary
        learned_synapses = summary.propagation_hop_count + summary.boundary_synapse_count
        boundary_walk_cells = summary.left_boundary_hop_bound + summary.right_boundary_hop_bound
        binder_cells += 2 + summary.propagation_hop_count + boundary_walk_cells
        binder_synapses += learned_synapses
        binder_ticks += learned_synapses * summary.demonstration_count
        learned_event_bytes += sum(
            0 if synapse.expected_event is None else len(synapse.expected_event)
            for synapse in binder._schema.propagation
        )
        learned_event_bytes += sum(len(synapse.event) for synapse in binder._schema.boundaries)
    return _ControllerEstimates(
        # One outer selector plus the wrapped neuronal populations.
        reserved_cell_capacity=1 + binder_cells + action_summary.total_cells,
        reserved_synapse_capacity=(1 + binder_synapses + action_summary.learned_synapses),
        learned_event_bytes=learned_event_bytes,
        excitatory_support_ticks=binder_ticks + action_summary.excitatory_ticks,
    )


@dataclass(frozen=True, slots=True)
class _FrozenSegment:
    ordinal: int
    selector_cell_id: int
    reserved_cell_capacity_estimate: int
    reserved_synapse_address_start: int
    reserved_synapse_capacity_estimate: int
    learned_event_bytes_retained_estimate: int
    excitatory_support_tick_estimate: int
    observation_sha256: str | None
    action_sha256: str | None
    trajectory_sha256: str | None
    evidence_sha256: str | None
    adapter: _FrozenController

    @property
    def reserved_cell_end(self) -> int:
        return self.selector_cell_id + self.reserved_cell_capacity_estimate

    @property
    def reserved_synapse_end(self) -> int:
        return self.reserved_synapse_address_start + self.reserved_synapse_capacity_estimate

    def record(self) -> dict[str, object]:
        return {
            "action_sha256": self.action_sha256,
            "component_checkpoint_sha256": self.adapter.checkpoint_sha256,
            "evidence_sha256": self.evidence_sha256,
            "excitatory_support_tick_estimate": self.excitatory_support_tick_estimate,
            "learned_event_bytes_retained_estimate": (self.learned_event_bytes_retained_estimate),
            "observation_sha256": self.observation_sha256,
            "ordinal": self.ordinal,
            "reserved_cell_capacity_estimate": self.reserved_cell_capacity_estimate,
            "reserved_synapse_address_start": self.reserved_synapse_address_start,
            "reserved_synapse_capacity_estimate": (self.reserved_synapse_capacity_estimate),
            "selector_cell_id": self.selector_cell_id,
            "trajectory_sha256": self.trajectory_sha256,
        }


def _physical_race(
    *,
    population: str,
    candidate_cell_ids: tuple[int, ...],
    candidate_scores: tuple[int, ...],
    stop_cell_id: int,
) -> OnlineMultiscaleStructuralRaceTrace:
    if len(candidate_cell_ids) != len(candidate_scores):
        raise RuntimeError("multiscale online race candidate custody differs")
    if any(score not in range(_SCORE_CAP + 1) for score in candidate_scores):
        raise ValueError("multiscale online race score is out of range")
    scores = (*candidate_scores, _TONIC_STOP_SCORE)
    cells = (*candidate_cell_ids, stop_cell_id)
    currents = _SCORE_CURRENTS[np.asarray(scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(cells) * _RACE_STEPS
    stopped = False
    stop_reason: str | None = None
    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_TONIC_STOP_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1:
            raise RuntimeError("multiscale online collision STOP relay changed")
        physical_cell_steps += _RACE_STEPS
        winner_cell_id = stop_cell_id
        stopped = True
        stop_reason = "collision"
    elif result.winner_cell == len(candidate_cell_ids):
        winner_cell_id = stop_cell_id
        stopped = True
        stop_reason = "tonic_stop"
    else:
        winner_cell_id = candidate_cell_ids[result.winner_cell]
    record = {
        "candidate_cell_ids": candidate_cell_ids,
        "candidate_scores": candidate_scores,
        "physical_cell_steps": physical_cell_steps,
        "population": population,
        "simultaneous_spike_candidates": result.simultaneous_spike_candidates,
        "spike_step": result.spike_step,
        "stop_reason": stop_reason,
        "stopped": stopped,
        "tonic_stop_cell_id": stop_cell_id,
        "winner_cell_id": winner_cell_id,
    }
    return OnlineMultiscaleStructuralRaceTrace(
        **record,
        sha256=_digest(_RACE_NAMESPACE, record),
    )


def _trajectory_sha256(
    *,
    controller: BoundaryMultiscaleSourceMotorController,
    observation_sha256: str,
    action_sha256: str,
    expected_emissions: tuple[bytes, ...],
) -> str:
    return _digest(
        _TRAJECTORY_NAMESPACE,
        {
            "action_sha256": action_sha256,
            "component_checkpoint_sha256": controller.checkpoint_sha256,
            "expected_emissions_hex": tuple(event.hex() for event in expected_emissions),
            "observation_sha256": observation_sha256,
        },
    )


def _checkpoint_sha256(
    segments: tuple[_FrozenSegment, ...],
    provenance_sha256: str,
) -> str:
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "provenance_sha256": provenance_sha256,
            "segments": tuple(segment.record() for segment in segments),
            "tonic_stop_cell_id": 0,
        },
    )


class AppendOnlyMultiscaleSourceMotorStack:
    """One immutable learner over a frozen base and replay-free appended shards."""

    __slots__ = ("_checkpoint_sha256", "_provenance_sha256", "_segments", "_summary")

    def __init__(
        self,
        *,
        segments: tuple[_FrozenSegment, ...],
        provenance_sha256: str,
    ) -> None:
        if not segments or segments[0].ordinal != 0:
            raise ValueError("online multiscale stack requires one base segment")
        if any(segment.ordinal != ordinal for ordinal, segment in enumerate(segments)):
            raise ValueError("online multiscale segment ordinals must be contiguous")
        if segments[0].selector_cell_id != 1:
            raise ValueError("base selector must follow the global tonic STOP cell")
        for left, right in pairwise(segments):
            if left.reserved_cell_end != right.selector_cell_id:
                raise ValueError("online multiscale cell reservations must be append-only")
            if left.reserved_synapse_end != right.reserved_synapse_address_start:
                raise ValueError("online multiscale synapse reservations must be append-only")
        self._segments = segments
        self._provenance_sha256 = _require_sha256(
            provenance_sha256,
            "provenance_sha256",
        )
        self._checkpoint_sha256 = _checkpoint_sha256(segments, provenance_sha256)
        self._summary = OnlineMultiscaleSourceMotorSummary(
            segment_count=len(segments),
            reserved_cell_capacity_estimate=segments[-1].reserved_cell_end,
            reserved_synapse_capacity_estimate=segments[-1].reserved_synapse_end,
            learned_event_bytes_retained_estimate=sum(
                segment.learned_event_bytes_retained_estimate for segment in segments
            ),
            excitatory_support_tick_estimate=sum(
                segment.excitatory_support_tick_estimate for segment in segments
            ),
            next_reserved_cell_address=segments[-1].reserved_cell_end,
            next_reserved_synapse_address=segments[-1].reserved_synapse_end,
            provenance_sha256=provenance_sha256,
        )

    @classmethod
    def seed(
        cls,
        controller: BoundaryMultiscaleSourceMotorController,
    ) -> AppendOnlyMultiscaleSourceMotorStack:
        if not isinstance(controller, BoundaryMultiscaleSourceMotorController):
            raise TypeError("controller must be a BoundaryMultiscaleSourceMotorController")
        frozen = _copy_controller(controller)
        estimates = _controller_estimates(frozen)
        adapter = _ControllerAdapter(frozen, frozen.checkpoint_sha256)
        segment = _FrozenSegment(
            ordinal=0,
            selector_cell_id=1,
            reserved_cell_capacity_estimate=estimates.reserved_cell_capacity,
            reserved_synapse_address_start=0,
            reserved_synapse_capacity_estimate=estimates.reserved_synapse_capacity,
            learned_event_bytes_retained_estimate=estimates.learned_event_bytes,
            excitatory_support_tick_estimate=estimates.excitatory_support_ticks,
            observation_sha256=None,
            action_sha256=None,
            trajectory_sha256=None,
            evidence_sha256=None,
            adapter=adapter,
        )
        provenance = _digest(
            _PROVENANCE_NAMESPACE,
            {
                "base_checkpoint_sha256": frozen.checkpoint_sha256,
                "base_training_experience_count": frozen.training_summary.experience_count,
            },
        )
        return cls(segments=(segment,), provenance_sha256=provenance)

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> OnlineMultiscaleSourceMotorSummary:
        return self._summary

    def respond(
        self,
        query: bytes,
        files: tuple[OpaqueWorkspaceFile, ...],
        *,
        ablation: BoundaryMultiscaleSourceMotorAblation = (
            BoundaryMultiscaleSourceMotorAblation.NONE
        ),
    ) -> OnlineMultiscaleSourceMotorDecision:
        """Run all frozen neuronal shards and physically arbitrate once."""

        if not isinstance(ablation, BoundaryMultiscaleSourceMotorAblation):
            raise TypeError("ablation must be a BoundaryMultiscaleSourceMotorAblation")
        observation = MultiscaleOpaqueEventTape.observe(query, files)
        canonical_query = observation.reconstruct_query()
        canonical_files = observation.reconstruct_workspace()
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "checkpoint_sha256": self._checkpoint_sha256,
                "files": _workspace_record(canonical_files),
                "query_hex": canonical_query.hex(),
            },
        )
        component_decisions = tuple(
            segment.adapter.respond_observation(
                observation,
                ablation=ablation,
            )
            for segment in self._segments
        )
        race = _physical_race(
            population="online_multiscale_segment",
            candidate_cell_ids=tuple(segment.selector_cell_id for segment in self._segments),
            candidate_scores=tuple(
                _EMIT_SCORE if decision.emitted else 0 for decision in component_decisions
            ),
            stop_cell_id=0,
        )
        physical_cell_steps = race.physical_cell_steps + sum(
            decision.physical_cell_steps for decision in component_decisions
        )
        winning_ordinal: int | None = None
        emissions: tuple[bytes, ...] = ()
        receipt_sha256: str | None = None
        if not race.stopped:
            winning_ordinal = next(
                segment.ordinal
                for segment in self._segments
                if segment.selector_cell_id == race.winner_cell_id
            )
            selected = component_decisions[winning_ordinal]
            if not selected.emitted or selected.receipt_sha256 is None:
                raise RuntimeError("physical multiscale winner lost its EMIT receipt")
            emissions = selected.emissions
            receipt_sha256 = selected.receipt_sha256
        record = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "component_decision_sha256s": tuple(
                decision.decision_sha256 for decision in component_decisions
            ),
            "component_receipt_sha256": receipt_sha256,
            "decision": "STOP" if race.stopped else "EMIT",
            "emissions_hex": tuple(token.hex() for token in emissions),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256": race.sha256,
            "stop_reason": race.stop_reason,
            "winning_segment_ordinal": winning_ordinal,
            "winning_selector_cell_id": race.winner_cell_id,
        }
        return OnlineMultiscaleSourceMotorDecision(
            decision="STOP" if race.stopped else "EMIT",
            emissions=emissions,
            winning_segment_ordinal=winning_ordinal,
            winning_selector_cell_id=race.winner_cell_id,
            stop_reason=race.stop_reason,
            component_decision_sha256s=tuple(
                decision.decision_sha256 for decision in component_decisions
            ),
            component_receipt_sha256=receipt_sha256,
            race=race,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )

    def extend_success(
        self,
        evidence: MultiscaleSourceMotorSuccessEvidence,
        *,
        ablation: OnlineMultiscaleGrowthAblation = OnlineMultiscaleGrowthAblation.NONE,
    ) -> OnlineMultiscaleSourceMotorGrowthTransition:
        """Append one frozen shard using only the current approved experience."""

        if not isinstance(evidence, MultiscaleSourceMotorSuccessEvidence):
            raise TypeError("evidence must be MultiscaleSourceMotorSuccessEvidence")
        if not isinstance(ablation, OnlineMultiscaleGrowthAblation):
            raise TypeError("ablation must be an OnlineMultiscaleGrowthAblation")
        evidence.authenticate()
        parent_checkpoint = self._checkpoint_sha256
        if evidence.execution_receipt.learner_checkpoint_sha256 != parent_checkpoint:
            raise ValueError("execution receipt does not bind this parent learner")
        for segment in self._segments:
            _ = segment.adapter.checkpoint_sha256

        expected = _expected_emissions(
            evidence.query,
            evidence.before_files,
            evidence.after_files,
        )
        experience = SuccessfulBoundaryMultiscaleSourceMotorExperience(
            query=evidence.query,
            before_files=evidence.before_files,
            after_files=evidence.after_files,
            provenance_id=evidence.provenance_id,
        )
        # Two identical rows encode two learning phases from this one current
        # approved experience: structural coactivation and positive success.
        # They do not replay any parent experience, and neither row is retained.
        candidate = BoundaryMultiscaleSourceMotorController.grow((experience, experience))
        frozen_candidate = _copy_controller(candidate)
        candidate_adapter = _ControllerAdapter(
            frozen_candidate,
            frozen_candidate.checkpoint_sha256,
        )
        trajectory = _trajectory_sha256(
            controller=frozen_candidate,
            observation_sha256=evidence.observation_sha256,
            action_sha256=evidence.action_sha256,
            expected_emissions=expected,
        )
        exact_existing = next(
            (
                segment
                for segment in self._segments[1:]
                if segment.observation_sha256 == evidence.observation_sha256
            ),
            None,
        )
        if exact_existing is not None:
            if exact_existing.action_sha256 == evidence.action_sha256:
                return self._zero_growth(
                    status="DUPLICATE",
                    stop_reason=None,
                    evidence=evidence,
                    trajectory_sha256=trajectory,
                    admission_race=None,
                )
            return self._conflict_growth(
                evidence=evidence,
                trajectory_sha256=trajectory,
                existing_cell_id=exact_existing.selector_cell_id,
                stop_reason="incompatible_observation_action",
            )

        parent_probe = self.respond(evidence.query, evidence.before_files)
        if parent_probe.decision == "EMIT":
            if parent_probe.emissions == expected:
                return self._zero_growth(
                    status="DUPLICATE",
                    stop_reason=None,
                    evidence=evidence,
                    trajectory_sha256=trajectory,
                    admission_race=None,
                )
            return self._conflict_growth(
                evidence=evidence,
                trajectory_sha256=trajectory,
                existing_cell_id=parent_probe.winning_selector_cell_id,
                stop_reason="supported_observation_action_conflict",
            )
        if parent_probe.stop_reason == "collision":
            return self._conflict_growth(
                evidence=evidence,
                trajectory_sha256=trajectory,
                existing_cell_id=self._segments[0].selector_cell_id,
                stop_reason="parent_observation_collision",
            )

        candidate_observation = MultiscaleOpaqueEventTape.observe(
            evidence.query,
            evidence.before_files,
        )
        candidate_probe = candidate_adapter.respond_observation(
            candidate_observation,
            ablation=BoundaryMultiscaleSourceMotorAblation.NONE,
        )
        if not candidate_probe.emitted or candidate_probe.emissions != expected:
            raise ValueError("current successful experience did not recruit an exact motor shard")

        if ablation is OnlineMultiscaleGrowthAblation.NO_UPDATE:
            return self._zero_growth(
                status="NO_UPDATE",
                stop_reason=None,
                evidence=evidence,
                trajectory_sha256=trajectory,
                admission_race=None,
            )

        estimates = _controller_estimates(frozen_candidate)
        if estimates.excitatory_support_ticks % 2:
            raise RuntimeError("two-phase multiscale support accounting changed")
        phase_ticks = estimates.excitatory_support_ticks // 2
        segment = _FrozenSegment(
            ordinal=len(self._segments),
            selector_cell_id=self._summary.next_reserved_cell_address,
            reserved_cell_capacity_estimate=estimates.reserved_cell_capacity,
            reserved_synapse_address_start=self._summary.next_reserved_synapse_address,
            reserved_synapse_capacity_estimate=estimates.reserved_synapse_capacity,
            learned_event_bytes_retained_estimate=estimates.learned_event_bytes,
            excitatory_support_tick_estimate=estimates.excitatory_support_ticks,
            observation_sha256=evidence.observation_sha256,
            action_sha256=evidence.action_sha256,
            trajectory_sha256=trajectory,
            evidence_sha256=evidence.sha256,
            adapter=candidate_adapter,
        )
        provenance = _digest(
            _PROVENANCE_NAMESPACE,
            {
                "compiled_trajectory_sha256": trajectory,
                "evidence_sha256": evidence.sha256,
                "parent_provenance_sha256": self._provenance_sha256,
                "segment": segment.record(),
            },
        )
        successor = AppendOnlyMultiscaleSourceMotorStack(
            segments=(*self._segments, segment),
            provenance_sha256=provenance,
        )
        for retained in self._segments:
            _ = retained.adapter.checkpoint_sha256
        if self._checkpoint_sha256 != parent_checkpoint:
            raise RuntimeError("multiscale structural growth mutated its parent")
        receipt = self._growth_receipt(
            status="APPENDED",
            stop_reason=None,
            successor=successor,
            evidence=evidence,
            trajectory_sha256=trajectory,
            reserved_cell_address_start=segment.selector_cell_id,
            reserved_cell_capacity_estimate_added=segment.reserved_cell_capacity_estimate,
            reserved_synapse_address_start=segment.reserved_synapse_address_start,
            reserved_synapse_capacity_estimate_added=(segment.reserved_synapse_capacity_estimate),
            learned_event_bytes_retained_estimate_added=(
                segment.learned_event_bytes_retained_estimate
            ),
            excitatory_support_tick_estimate_added=(segment.excitatory_support_tick_estimate),
            coactivation_support_tick_estimate_added=phase_ticks,
            positive_success_support_tick_estimate_added=phase_ticks,
            committed_evidence_tick_phases=2,
            admission_race=None,
        )
        return OnlineMultiscaleSourceMotorGrowthTransition(successor, receipt)

    def _conflict_growth(
        self,
        *,
        evidence: MultiscaleSourceMotorSuccessEvidence,
        trajectory_sha256: str,
        existing_cell_id: int,
        stop_reason: str,
    ) -> OnlineMultiscaleSourceMotorGrowthTransition:
        candidate_cell_id = self._summary.next_reserved_cell_address
        race = _physical_race(
            population="online_multiscale_growth_conflict",
            candidate_cell_ids=(existing_cell_id, candidate_cell_id),
            candidate_scores=(_EMIT_SCORE, _EMIT_SCORE),
            stop_cell_id=0,
        )
        if not race.stopped or race.stop_reason != "collision":
            raise RuntimeError("incompatible multiscale action did not collision-STOP")
        return self._zero_growth(
            status="CONFLICT_STOP",
            stop_reason=stop_reason,
            evidence=evidence,
            trajectory_sha256=trajectory_sha256,
            admission_race=race,
        )

    def _zero_growth(
        self,
        *,
        status: str,
        stop_reason: str | None,
        evidence: MultiscaleSourceMotorSuccessEvidence,
        trajectory_sha256: str,
        admission_race: OnlineMultiscaleStructuralRaceTrace | None,
    ) -> OnlineMultiscaleSourceMotorGrowthTransition:
        receipt = self._growth_receipt(
            status=status,
            stop_reason=stop_reason,
            successor=self,
            evidence=evidence,
            trajectory_sha256=trajectory_sha256,
            reserved_cell_address_start=None,
            reserved_cell_capacity_estimate_added=0,
            reserved_synapse_address_start=None,
            reserved_synapse_capacity_estimate_added=0,
            learned_event_bytes_retained_estimate_added=0,
            excitatory_support_tick_estimate_added=0,
            coactivation_support_tick_estimate_added=0,
            positive_success_support_tick_estimate_added=0,
            committed_evidence_tick_phases=0,
            admission_race=admission_race,
        )
        return OnlineMultiscaleSourceMotorGrowthTransition(self, receipt)

    def _growth_receipt(
        self,
        *,
        status: str,
        stop_reason: str | None,
        successor: AppendOnlyMultiscaleSourceMotorStack,
        evidence: MultiscaleSourceMotorSuccessEvidence,
        trajectory_sha256: str,
        reserved_cell_address_start: int | None,
        reserved_cell_capacity_estimate_added: int,
        reserved_synapse_address_start: int | None,
        reserved_synapse_capacity_estimate_added: int,
        learned_event_bytes_retained_estimate_added: int,
        excitatory_support_tick_estimate_added: int,
        coactivation_support_tick_estimate_added: int,
        positive_success_support_tick_estimate_added: int,
        committed_evidence_tick_phases: int,
        admission_race: OnlineMultiscaleStructuralRaceTrace | None,
    ) -> OnlineMultiscaleSourceMotorGrowthReceipt:
        record = {
            "action_sha256": evidence.action_sha256,
            "admission_race_sha256": (None if admission_race is None else admission_race.sha256),
            "coactivation_support_tick_estimate_added": (coactivation_support_tick_estimate_added),
            "committed_evidence_tick_phases": committed_evidence_tick_phases,
            "compiled_trajectory_sha256": trajectory_sha256,
            "current_experience_training_rows": 2,
            "evidence_sha256": evidence.sha256,
            "execution_receipt_sha256": evidence.execution_receipt.sha256,
            "excitatory_support_tick_estimate_added": (excitatory_support_tick_estimate_added),
            "inhibitory_support_tick_estimate_added": 0,
            "learned_event_bytes_retained_estimate_added": (
                learned_event_bytes_retained_estimate_added
            ),
            "observation_sha256": evidence.observation_sha256,
            "parent_checkpoint_sha256": self._checkpoint_sha256,
            "parent_experience_replay_count": 0,
            "positive_success_support_tick_estimate_added": (
                positive_success_support_tick_estimate_added
            ),
            "prior_provenance_sha256": self._provenance_sha256,
            "raw_training_sequences_retained": 0,
            "reserved_cell_address_start": reserved_cell_address_start,
            "reserved_cell_capacity_estimate_added": (reserved_cell_capacity_estimate_added),
            "reserved_synapse_address_start": reserved_synapse_address_start,
            "reserved_synapse_capacity_estimate_added": (reserved_synapse_capacity_estimate_added),
            "source_experience_count": 1,
            "status": status,
            "stop_reason": stop_reason,
            "successor_checkpoint_sha256": successor.checkpoint_sha256,
            "successor_provenance_sha256": successor.training_summary.provenance_sha256,
        }
        return OnlineMultiscaleSourceMotorGrowthReceipt(
            status=status,
            stop_reason=stop_reason,
            parent_checkpoint_sha256=self._checkpoint_sha256,
            successor_checkpoint_sha256=successor.checkpoint_sha256,
            evidence_sha256=evidence.sha256,
            observation_sha256=evidence.observation_sha256,
            action_sha256=evidence.action_sha256,
            execution_receipt_sha256=evidence.execution_receipt.sha256,
            compiled_trajectory_sha256=trajectory_sha256,
            prior_provenance_sha256=self._provenance_sha256,
            successor_provenance_sha256=successor.training_summary.provenance_sha256,
            reserved_cell_address_start=reserved_cell_address_start,
            reserved_cell_capacity_estimate_added=(reserved_cell_capacity_estimate_added),
            reserved_synapse_address_start=reserved_synapse_address_start,
            reserved_synapse_capacity_estimate_added=(reserved_synapse_capacity_estimate_added),
            learned_event_bytes_retained_estimate_added=(
                learned_event_bytes_retained_estimate_added
            ),
            excitatory_support_tick_estimate_added=(excitatory_support_tick_estimate_added),
            coactivation_support_tick_estimate_added=(coactivation_support_tick_estimate_added),
            positive_success_support_tick_estimate_added=(
                positive_success_support_tick_estimate_added
            ),
            inhibitory_support_tick_estimate_added=0,
            source_experience_count=1,
            current_experience_training_rows=2,
            committed_evidence_tick_phases=committed_evidence_tick_phases,
            parent_experience_replay_count=0,
            raw_training_sequences_retained=0,
            admission_race=admission_race,
            sha256=_digest(_GROWTH_NAMESPACE, record),
        )
