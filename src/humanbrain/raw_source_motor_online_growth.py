"""Replay-free online structural growth for the raw-source motor stack.

One authenticated successful before/after experience is compiled once into the
existing cue-free binder and source-relative motor tape.  The resulting frozen
neuronal shard is appended behind monotonic capacity reservations; no prior raw
training sequence is replayed or rebuilt.  Learned context and neighborhood
event bytes remain synaptic content.  A fitted-GLIF ensemble race gives one
emitting shard control or physically selects STOP on absence/collision.

This is a deliberately narrow continual-learning slice.  Exact context events,
opaque token activation, and the training-only trajectory compiler remain
scaffolding.  It is not context abstraction or motor-program induction.
"""

from __future__ import annotations

import json
from copy import copy
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from typing import Protocol

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.raw_source_motor_controller import (
    RawSourceMotorController,
)
from humanbrain.raw_tape_role_binder import (
    NeuronalRawTapeRoleBinder,
    RawTapeRoleDemonstration,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.source_relative_action_tape import (
    OccurrencePreservingActionTape,
    SourceRelativeMotorDemonstration,
    TapeValence,
)
from humanbrain.source_relative_training_compiler import (
    compile_successful_token_tapes,
)

__all__ = [
    "AppendOnlyRawSourceMotorStack",
    "OnlineGrowthAblation",
    "OnlineRawSourceMotorDecision",
    "OnlineRawSourceMotorGrowthReceipt",
    "OnlineRawSourceMotorGrowthTransition",
    "OnlineRawSourceMotorSummary",
    "SourceMotorExecutionAuthority",
    "SourceMotorExecutionReceipt",
    "SourceMotorSuccessEvidence",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_SCORE_CAP = 2
_TONIC_STOP_SCORE = 1
_EMIT_SCORE = 2

_OBSERVATION_NAMESPACE = b"humanbrain.raw-source-online.observation-v1\0"
_ACTION_NAMESPACE = b"humanbrain.raw-source-online.action-v1\0"
_EXECUTION_RECEIPT_NAMESPACE = b"humanbrain.raw-source-online.execution-receipt-v1\0"
_EVIDENCE_NAMESPACE = b"humanbrain.raw-source-online.evidence-v1\0"
_TRAJECTORY_NAMESPACE = b"humanbrain.raw-source-online.trajectory-v1\0"
_SEGMENT_NAMESPACE = b"humanbrain.raw-source-online.segment-v1\0"
_PROVENANCE_NAMESPACE = b"humanbrain.raw-source-online.provenance-v1\0"
_CHECKPOINT_NAMESPACE = b"humanbrain.raw-source-online.checkpoint-v1\0"
_QUERY_NAMESPACE = b"humanbrain.raw-source-online.query-v1\0"
_RACE_NAMESPACE = b"humanbrain.raw-source-online.race-v1\0"
_DECISION_NAMESPACE = b"humanbrain.raw-source-online.decision-v1\0"
_GROWTH_NAMESPACE = b"humanbrain.raw-source-online.growth-v1\0"
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


def _require_event(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{name} must be non-empty bytes")
    return value


def _require_events(value: object, name: str) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    events = value
    if not events:
        raise ValueError(f"{name} must not be empty")
    for ordinal, event in enumerate(events):
        _require_event(event, f"{name}[{ordinal}]")
    return events


def _observation_sha256(context: bytes, before: tuple[bytes, ...]) -> str:
    return _digest(
        _OBSERVATION_NAMESPACE,
        {
            "before_hex": tuple(token.hex() for token in before),
            "context_hex": context.hex(),
        },
    )


def _action_sha256(after: tuple[bytes, ...]) -> str:
    return _digest(
        _ACTION_NAMESPACE,
        {"emitted_after_hex": tuple(token.hex() for token in after)},
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
            raise RuntimeError("online-growth fitted-GLIF latency interval changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(row.spike_step != _ZERO_SCORE_SPIKE_STEP - score for score, row in enumerate(controls)):
        raise RuntimeError("online-growth fitted-GLIF latency calibration changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class OnlineGrowthAblation(StrEnum):
    NONE = "none"
    NO_UPDATE = "no_update"


@dataclass(frozen=True, slots=True, init=False)
class SourceMotorExecutionReceipt:
    """Authority-issued PASS for one learner, observation, and action."""

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
            raise ValueError("successful motor evidence requires a PASS receipt")
        record = {
            "action_sha256": action,
            "evaluator_sha256": evaluator,
            "learner_checkpoint_sha256": learner,
            "observation_sha256": observation,
            "outcome": self.outcome,
        }
        if self.sha256 != _digest(_EXECUTION_RECEIPT_NAMESPACE, record):
            raise ValueError("execution receipt root changed")


class SourceMotorExecutionAuthority:
    """Issue PASS only after invoking one caller-pinned frozen evaluator.

    This is an in-process custody boundary, not cryptographic authority.  The
    runner remains responsible for authenticating the evaluator commitment.
    """

    __slots__ = ("_evaluator", "_evaluator_sha256")

    def __init__(self, *, evaluator: object, evaluator_sha256: str) -> None:
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self._evaluator = evaluator
        self._evaluator_sha256 = _require_sha256(
            evaluator_sha256,
            "evaluator_sha256",
        )

    def execute_and_issue(
        self,
        *,
        learner_checkpoint_sha256: str,
        context_event: bytes,
        before_tokens: tuple[bytes, ...],
        emitted_after_tokens: tuple[bytes, ...],
    ) -> SourceMotorExecutionReceipt:
        learner = _require_sha256(
            learner_checkpoint_sha256,
            "learner_checkpoint_sha256",
        )
        context = _require_event(context_event, "context_event")
        before = _require_events(before_tokens, "before_tokens")
        after = _require_events(emitted_after_tokens, "emitted_after_tokens")
        passed = self._evaluator(
            context_event=context,
            before_tokens=before,
            emitted_after_tokens=after,
        )
        if passed is not True:
            raise ValueError("frozen evaluator did not PASS the emitted action")
        observation = _observation_sha256(context, before)
        action = _action_sha256(after)
        record = {
            "action_sha256": action,
            "evaluator_sha256": self._evaluator_sha256,
            "learner_checkpoint_sha256": learner,
            "observation_sha256": observation,
            "outcome": "PASS",
        }
        receipt = SourceMotorExecutionReceipt(
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
class SourceMotorSuccessEvidence:
    """One successful public experience bound to an external receipt root."""

    context_event: bytes
    before_tokens: tuple[bytes, ...]
    emitted_after_tokens: tuple[bytes, ...]
    observation_sha256: str
    action_sha256: str
    execution_receipt: SourceMotorExecutionReceipt
    provenance_id: str
    sha256: str

    @classmethod
    def seal(
        cls,
        *,
        context_event: bytes,
        before_tokens: tuple[bytes, ...],
        emitted_after_tokens: tuple[bytes, ...],
        execution_receipt: SourceMotorExecutionReceipt,
        provenance_id: str,
    ) -> SourceMotorSuccessEvidence:
        context = _require_event(context_event, "context_event")
        before = _require_events(before_tokens, "before_tokens")
        after = _require_events(emitted_after_tokens, "emitted_after_tokens")
        if not isinstance(execution_receipt, SourceMotorExecutionReceipt):
            raise TypeError("execution_receipt must be SourceMotorExecutionReceipt")
        execution_receipt.authenticate()
        if not isinstance(provenance_id, str) or not provenance_id:
            raise ValueError("provenance_id must be a non-empty string")
        observation = _observation_sha256(context, before)
        action = _action_sha256(after)
        if execution_receipt.observation_sha256 != observation:
            raise ValueError("execution receipt does not bind the evidence observation")
        if execution_receipt.action_sha256 != action:
            raise ValueError("execution receipt does not bind the evidence action")
        record = {
            "action_sha256": action,
            "execution_receipt_sha256": execution_receipt.sha256,
            "observation_sha256": observation,
            "provenance_id": provenance_id,
        }
        return cls(
            context_event=context,
            before_tokens=before,
            emitted_after_tokens=after,
            observation_sha256=observation,
            action_sha256=action,
            execution_receipt=execution_receipt,
            provenance_id=provenance_id,
            sha256=_digest(_EVIDENCE_NAMESPACE, record),
        )

    def authenticate(self) -> None:
        context = _require_event(self.context_event, "context_event")
        before = _require_events(self.before_tokens, "before_tokens")
        after = _require_events(self.emitted_after_tokens, "emitted_after_tokens")
        if not isinstance(self.execution_receipt, SourceMotorExecutionReceipt):
            raise TypeError("execution_receipt must be SourceMotorExecutionReceipt")
        self.execution_receipt.authenticate()
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be a non-empty string")
        observation = _observation_sha256(context, before)
        action = _action_sha256(after)
        if self.observation_sha256 != observation:
            raise ValueError("successful evidence observation custody changed")
        if self.action_sha256 != action:
            raise ValueError("successful evidence action custody changed")
        if self.execution_receipt.observation_sha256 != observation:
            raise ValueError("execution receipt does not bind the evidence observation")
        if self.execution_receipt.action_sha256 != action:
            raise ValueError("execution receipt does not bind the evidence action")
        record = {
            "action_sha256": action,
            "execution_receipt_sha256": self.execution_receipt.sha256,
            "observation_sha256": observation,
            "provenance_id": self.provenance_id,
        }
        if self.sha256 != _digest(_EVIDENCE_NAMESPACE, record):
            raise ValueError("successful evidence root changed")


@dataclass(frozen=True, slots=True)
class OnlineStructuralRaceTrace:
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
class OnlineRawSourceMotorSummary:
    segment_count: int
    reserved_cell_capacity: int
    reserved_synapse_capacity_estimate: int
    next_reserved_cell_address: int
    next_reserved_synapse_address: int
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class OnlineRawSourceMotorDecision:
    decision: str
    emissions: tuple[bytes, ...]
    winning_segment_ordinal: int | None
    winning_selector_cell_id: int
    stop_reason: str | None
    component_decision_sha256s: tuple[str, ...]
    component_receipt_sha256: str | None
    race: OnlineStructuralRaceTrace
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class OnlineRawSourceMotorGrowthReceipt:
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
    reserved_cell_capacity_added: int
    reserved_synapse_address_start: int | None
    reserved_synapse_capacity_estimate_added: int
    excitatory_support_tick_estimate_added: int
    coactivation_support_tick_estimate_added: int
    positive_success_support_tick_estimate_added: int
    inhibitory_support_tick_estimate_added: int
    source_experience_count: int
    committed_evidence_tick_phases: int
    parent_experience_replay_count: int
    raw_training_sequences_retained: int
    admission_race: OnlineStructuralRaceTrace | None
    sha256: str


@dataclass(frozen=True, slots=True)
class OnlineRawSourceMotorGrowthTransition:
    successor: AppendOnlyRawSourceMotorStack
    receipt: OnlineRawSourceMotorGrowthReceipt


@dataclass(frozen=True, slots=True)
class _SegmentEmission:
    emitted: bool
    emissions: tuple[bytes, ...]
    stop_reason: str | None
    decision_sha256: str
    receipt_sha256: str | None
    physical_cell_steps: int


class _FrozenMotorAdapter(Protocol):
    @property
    def checkpoint_sha256(self) -> str: ...

    def respond(
        self,
        context_events: tuple[bytes, ...],
        source_tokens: tuple[bytes, ...],
    ) -> _SegmentEmission: ...


def _copy_action_tape(
    tape: OccurrencePreservingActionTape,
) -> OccurrencePreservingActionTape:
    """Copy every mutable inference field while sharing frozen learned rows."""

    if tape._pending is not None:
        raise RuntimeError("cannot freeze an action tape with a pending decision")
    cloned = copy(tape)
    cloned._adjustments = dict(tape._adjustments)
    cloned._pending = None
    return cloned


def _copy_controller(controller: RawSourceMotorController) -> RawSourceMotorController:
    """Isolate mutable pending/consequence state from callers and siblings."""

    if controller._pending is not None:
        raise RuntimeError("cannot freeze a controller with a pending decision")
    cloned = copy(controller)
    cloned._tape = _copy_action_tape(controller._tape)
    cloned._pending = None
    return cloned


@dataclass(frozen=True, slots=True)
class _BaseControllerAdapter:
    controller: RawSourceMotorController
    expected_checkpoint_sha256: str

    @property
    def checkpoint_sha256(self) -> str:
        if self.controller._pending is not None or self.controller._tape._pending is not None:
            raise RuntimeError("frozen base-controller gained pending state")
        if self.controller.checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise RuntimeError("frozen base-controller checkpoint changed")
        return self.expected_checkpoint_sha256

    def respond(
        self,
        context_events: tuple[bytes, ...],
        source_tokens: tuple[bytes, ...],
    ) -> _SegmentEmission:
        if self.controller.checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise RuntimeError("frozen base-controller checkpoint changed")
        ephemeral = _copy_controller(self.controller)
        decision = ephemeral.respond(context_events, source_tokens)
        receipt_sha256 = None
        if decision.decision == "EMIT":
            if decision.receipt is None:
                raise RuntimeError("base EMIT lost its exact receipt")
            receipt_sha256 = decision.receipt.sha256
            update = ephemeral.resolve(TapeValence.NEUTRAL)
            if update.receipt_sha256 != receipt_sha256:
                raise RuntimeError("base neutral resolution lost receipt custody")
        if ephemeral.checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise RuntimeError("ephemeral base response changed learned state")
        if self.controller.checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise RuntimeError("stored base response changed learned state")
        return _SegmentEmission(
            emitted=decision.decision == "EMIT",
            emissions=decision.emissions,
            stop_reason=decision.stop_reason,
            decision_sha256=decision.sha256,
            receipt_sha256=receipt_sha256,
            physical_cell_steps=decision.physical_cell_steps,
        )


@dataclass(frozen=True, slots=True)
class _BinderTapeAdapter:
    binder: NeuronalRawTapeRoleBinder
    tape: OccurrencePreservingActionTape
    expected_checkpoint_sha256: str

    @property
    def checkpoint_sha256(self) -> str:
        if self.tape._pending is not None:
            raise RuntimeError("frozen online shard gained pending state")
        current = _digest(
            _SEGMENT_NAMESPACE,
            {
                "binder_checkpoint_sha256": self.binder.checkpoint_sha256,
                "tape_checkpoint_sha256": self.tape.checkpoint_sha256,
            },
        )
        if current != self.expected_checkpoint_sha256:
            raise RuntimeError("frozen online shard checkpoint changed")
        return current

    def respond(
        self,
        context_events: tuple[bytes, ...],
        source_tokens: tuple[bytes, ...],
    ) -> _SegmentEmission:
        _ = self.checkpoint_sha256
        binder = copy(self.binder)
        tape = _copy_action_tape(self.tape)
        binding = binder.respond(context_events, source_tokens)
        if binding.decision == "STOP":
            return _SegmentEmission(
                emitted=False,
                emissions=(),
                stop_reason=f"binder:{binding.stop_reason}",
                decision_sha256=binding.sha256,
                receipt_sha256=None,
                physical_cell_steps=binding.physical_cell_steps,
            )
        tape_decision = tape.respond(
            context_events,
            binding.cue_inventory,
            source_tokens,
        )
        physical_cell_steps = binding.physical_cell_steps + tape_decision.physical_cell_steps
        decision_sha256 = _digest(
            _DECISION_NAMESPACE,
            {
                "binder_decision_sha256": binding.sha256,
                "tape_decision_sha256": tape_decision.sha256,
            },
        )
        receipt_sha256 = None
        if tape_decision.decision == "EMIT":
            receipt = tape_decision.eligibility
            if receipt is None:
                raise RuntimeError("online shard EMIT lost its exact receipt")
            receipt_sha256 = receipt.sha256
            update = tape.resolve(TapeValence.NEUTRAL)
            if update.receipt_sha256 != receipt_sha256:
                raise RuntimeError("online neutral resolution lost receipt custody")
        if tape.checkpoint_sha256 != self.tape.checkpoint_sha256:
            raise RuntimeError("ephemeral online response changed learned state")
        _ = self.checkpoint_sha256
        return _SegmentEmission(
            emitted=tape_decision.decision == "EMIT",
            emissions=tape_decision.emissions,
            stop_reason=(
                None if tape_decision.stop_reason is None else f"tape:{tape_decision.stop_reason}"
            ),
            decision_sha256=decision_sha256,
            receipt_sha256=receipt_sha256,
            physical_cell_steps=physical_cell_steps,
        )


@dataclass(frozen=True, slots=True)
class _FrozenSegment:
    ordinal: int
    selector_cell_id: int
    reserved_cell_capacity: int
    reserved_synapse_address_start: int
    reserved_synapse_capacity_estimate: int
    excitatory_support_tick_estimate: int
    inhibitory_support_tick_estimate: int
    context_sha256: str | None
    trajectory_sha256: str | None
    evidence_sha256: str | None
    adapter: _FrozenMotorAdapter

    @property
    def reserved_cell_end(self) -> int:
        return self.selector_cell_id + self.reserved_cell_capacity

    @property
    def reserved_synapse_end(self) -> int:
        return self.reserved_synapse_address_start + self.reserved_synapse_capacity_estimate

    def record(self) -> dict[str, object]:
        return {
            "reserved_cell_capacity": self.reserved_cell_capacity,
            "component_checkpoint_sha256": self.adapter.checkpoint_sha256,
            "context_sha256": self.context_sha256,
            "evidence_sha256": self.evidence_sha256,
            "excitatory_support_tick_estimate": self.excitatory_support_tick_estimate,
            "inhibitory_support_tick_estimate": self.inhibitory_support_tick_estimate,
            "ordinal": self.ordinal,
            "selector_cell_id": self.selector_cell_id,
            "reserved_synapse_capacity_estimate": self.reserved_synapse_capacity_estimate,
            "reserved_synapse_address_start": self.reserved_synapse_address_start,
            "trajectory_sha256": self.trajectory_sha256,
        }


def _physical_race(
    *,
    population: str,
    candidate_cell_ids: tuple[int, ...],
    candidate_scores: tuple[int, ...],
    stop_cell_id: int,
) -> OnlineStructuralRaceTrace:
    if len(candidate_cell_ids) != len(candidate_scores):
        raise RuntimeError("online-growth race candidate custody differs")
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
            raise RuntimeError("online-growth collision STOP relay changed")
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
    return OnlineStructuralRaceTrace(
        **record,
        sha256=_digest(_RACE_NAMESPACE, record),
    )


def _trajectory_sha256(
    demonstration: SourceRelativeMotorDemonstration,
    binder: NeuronalRawTapeRoleBinder,
) -> str:
    return _digest(
        _TRAJECTORY_NAMESPACE,
        {
            "binder_checkpoint_sha256": binder.checkpoint_sha256,
            "motor_steps": tuple(step.record() for step in demonstration.motor_steps),
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


class AppendOnlyRawSourceMotorStack:
    """One deep interface over a frozen base and append-only neuronal shards."""

    __slots__ = ("_checkpoint_sha256", "_provenance_sha256", "_segments", "_summary")

    def __init__(
        self,
        *,
        segments: tuple[_FrozenSegment, ...],
        provenance_sha256: str,
    ) -> None:
        if not segments or segments[0].ordinal != 0:
            raise ValueError("online motor stack requires one base segment")
        if any(segment.ordinal != ordinal for ordinal, segment in enumerate(segments)):
            raise ValueError("online motor segment ordinals must be contiguous")
        if segments[0].selector_cell_id != 1:
            raise ValueError("base selector must follow the global tonic STOP cell")
        for left, right in pairwise(segments):
            if left.reserved_cell_end != right.selector_cell_id:
                raise ValueError("online motor cell reservations must be append-only")
            if left.reserved_synapse_end != right.reserved_synapse_address_start:
                raise ValueError("online motor synapse reservations must be append-only")
        self._segments = segments
        self._provenance_sha256 = _require_sha256(
            provenance_sha256,
            "provenance_sha256",
        )
        self._checkpoint_sha256 = _checkpoint_sha256(segments, provenance_sha256)
        self._summary = OnlineRawSourceMotorSummary(
            segment_count=len(segments),
            reserved_cell_capacity=segments[-1].reserved_cell_end,
            reserved_synapse_capacity_estimate=segments[-1].reserved_synapse_end,
            next_reserved_cell_address=segments[-1].reserved_cell_end,
            next_reserved_synapse_address=segments[-1].reserved_synapse_end,
            provenance_sha256=provenance_sha256,
        )

    @classmethod
    def seed(cls, controller: RawSourceMotorController) -> AppendOnlyRawSourceMotorStack:
        if not isinstance(controller, RawSourceMotorController):
            raise TypeError("controller must be a RawSourceMotorController")
        frozen_controller = _copy_controller(controller)
        summary = frozen_controller.training_summary
        reserved_cell_capacity = (
            1
            + summary.binder.context_count
            + summary.binder.role_slot_count
            + summary.tape.total_cells
        )
        reserved_synapse_capacity_estimate = (
            1 + summary.binder.neighborhood_synapse_count + summary.tape.learned_synapses
        )
        adapter = _BaseControllerAdapter(
            frozen_controller,
            frozen_controller.checkpoint_sha256,
        )
        segment = _FrozenSegment(
            ordinal=0,
            selector_cell_id=1,
            reserved_cell_capacity=reserved_cell_capacity,
            reserved_synapse_address_start=0,
            reserved_synapse_capacity_estimate=reserved_synapse_capacity_estimate,
            excitatory_support_tick_estimate=0,
            inhibitory_support_tick_estimate=0,
            context_sha256=None,
            trajectory_sha256=None,
            evidence_sha256=None,
            adapter=adapter,
        )
        provenance = _digest(
            _PROVENANCE_NAMESPACE,
            {
                "base_checkpoint_sha256": frozen_controller.checkpoint_sha256,
                "base_training_custody_sha256": summary.training_custody_sha256,
            },
        )
        return cls(segments=(segment,), provenance_sha256=provenance)

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> OnlineRawSourceMotorSummary:
        return self._summary

    def respond(
        self,
        context_events: tuple[bytes, ...],
        source_tokens: tuple[bytes, ...],
    ) -> OnlineRawSourceMotorDecision:
        """Run every frozen neuronal shard, then physically arbitrate once."""

        contexts = _require_events(context_events, "context_events")
        source = _require_events(source_tokens, "source_tokens")
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "checkpoint_sha256": self._checkpoint_sha256,
                "context_events_hex": tuple(event.hex() for event in contexts),
                "source_tokens_hex": tuple(token.hex() for token in source),
            },
        )
        component_decisions = tuple(
            segment.adapter.respond(contexts, source) for segment in self._segments
        )
        race = _physical_race(
            population="online_segment",
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
                raise RuntimeError("physical segment winner lost its EMIT receipt")
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
        return OnlineRawSourceMotorDecision(
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
        evidence: SourceMotorSuccessEvidence,
        *,
        ablation: OnlineGrowthAblation = OnlineGrowthAblation.NONE,
    ) -> OnlineRawSourceMotorGrowthTransition:
        """Append one new frozen neuronal shard without replaying old experience."""

        if not isinstance(evidence, SourceMotorSuccessEvidence):
            raise TypeError("evidence must be SourceMotorSuccessEvidence")
        if not isinstance(ablation, OnlineGrowthAblation):
            raise TypeError("ablation must be OnlineGrowthAblation")
        evidence.authenticate()
        parent_checkpoint = self._checkpoint_sha256
        if evidence.execution_receipt.learner_checkpoint_sha256 != parent_checkpoint:
            raise ValueError("execution receipt does not bind this parent learner")
        for segment in self._segments:
            _ = segment.adapter.checkpoint_sha256

        demonstration = compile_successful_token_tapes(
            context_event=evidence.context_event,
            before_tokens=evidence.before_tokens,
            after_tokens=evidence.emitted_after_tokens,
        )
        if not demonstration.cue_inventory:
            raise ValueError("successful online motor experience must compile at least one cue")
        binder_demo = RawTapeRoleDemonstration(
            context_event=demonstration.context_event,
            source_tokens=demonstration.source_tokens,
            demonstrated_roles=demonstration.cue_inventory,
        )
        binder = NeuronalRawTapeRoleBinder.grow((binder_demo,))
        # The tape API represents support as demonstration rows. These two
        # identical rows are two ticks from this one authenticated experience:
        # structural coactivation, then positive-success support. They are not
        # two experiences and do not replay any parent training sequence.
        current_success_phase_rows = (demonstration, demonstration)
        tape = OccurrencePreservingActionTape.grow(current_success_phase_rows)
        trajectory_sha256 = _trajectory_sha256(demonstration, binder)
        context_sha256 = sha256(evidence.context_event).hexdigest()

        existing = next(
            (segment for segment in self._segments[1:] if segment.context_sha256 == context_sha256),
            None,
        )
        if existing is not None:
            if existing.trajectory_sha256 == trajectory_sha256:
                return self._zero_growth(
                    status="DUPLICATE",
                    stop_reason=None,
                    evidence=evidence,
                    trajectory_sha256=trajectory_sha256,
                    admission_race=None,
                )
            race = _physical_race(
                population="online_growth_conflict",
                candidate_cell_ids=(
                    existing.selector_cell_id,
                    self._summary.next_reserved_cell_address,
                ),
                candidate_scores=(_EMIT_SCORE, _EMIT_SCORE),
                stop_cell_id=0,
            )
            if not race.stopped or race.stop_reason != "collision":
                raise RuntimeError("incompatible context programs did not collision-STOP")
            return self._zero_growth(
                status="CONFLICT_STOP",
                stop_reason="incompatible_context_program",
                evidence=evidence,
                trajectory_sha256=trajectory_sha256,
                admission_race=race,
            )

        base_probe = self._segments[0].adapter.respond(
            (evidence.context_event,),
            evidence.before_tokens,
        )
        if base_probe.emitted:
            if base_probe.emissions == evidence.emitted_after_tokens:
                return self._zero_growth(
                    status="DUPLICATE",
                    stop_reason=None,
                    evidence=evidence,
                    trajectory_sha256=trajectory_sha256,
                    admission_race=None,
                )
            race = _physical_race(
                population="online_growth_conflict",
                candidate_cell_ids=(1, self._summary.next_reserved_cell_address),
                candidate_scores=(_EMIT_SCORE, _EMIT_SCORE),
                stop_cell_id=0,
            )
            return self._zero_growth(
                status="CONFLICT_STOP",
                stop_reason="base_context_program_conflict",
                evidence=evidence,
                trajectory_sha256=trajectory_sha256,
                admission_race=race,
            )
        if base_probe.stop_reason != "binder:context_schema:tonic_stop":
            race = _physical_race(
                population="online_growth_conflict",
                candidate_cell_ids=(1, self._summary.next_reserved_cell_address),
                candidate_scores=(_EMIT_SCORE, _EMIT_SCORE),
                stop_cell_id=0,
            )
            return self._zero_growth(
                status="CONFLICT_STOP",
                stop_reason="base_context_already_occupied",
                evidence=evidence,
                trajectory_sha256=trajectory_sha256,
                admission_race=race,
            )
        if not base_probe.decision_sha256:
            raise RuntimeError("base admission probe lost decision custody")

        if ablation is OnlineGrowthAblation.NO_UPDATE:
            return self._zero_growth(
                status="NO_UPDATE",
                stop_reason=None,
                evidence=evidence,
                trajectory_sha256=trajectory_sha256,
                admission_race=None,
            )

        adapter_checkpoint = _digest(
            _SEGMENT_NAMESPACE,
            {
                "binder_checkpoint_sha256": binder.checkpoint_sha256,
                "tape_checkpoint_sha256": tape.checkpoint_sha256,
            },
        )
        adapter = _BinderTapeAdapter(
            copy(binder),
            _copy_action_tape(tape),
            adapter_checkpoint,
        )
        binder_summary = binder.training_summary
        tape_summary = tape.training_summary
        reserved_cell_capacity = (
            1
            + binder_summary.context_count
            + binder_summary.role_slot_count
            + tape_summary.total_cells
        )
        reserved_synapse_capacity_estimate = (
            1 + binder_summary.neighborhood_synapse_count + tape_summary.learned_synapses
        )
        excitatory_support_tick_estimate = (
            2 + binder_summary.neighborhood_synapse_count + tape_summary.excitatory_ticks
        )
        coactivation_support_tick_estimate = reserved_synapse_capacity_estimate
        positive_success_support_tick_estimate = 1 + tape_summary.learned_synapses
        if (
            coactivation_support_tick_estimate + positive_success_support_tick_estimate
            != excitatory_support_tick_estimate
        ):
            raise RuntimeError("online growth tick phase accounting changed")
        segment = _FrozenSegment(
            ordinal=len(self._segments),
            selector_cell_id=self._summary.next_reserved_cell_address,
            reserved_cell_capacity=reserved_cell_capacity,
            reserved_synapse_address_start=self._summary.next_reserved_synapse_address,
            reserved_synapse_capacity_estimate=reserved_synapse_capacity_estimate,
            excitatory_support_tick_estimate=excitatory_support_tick_estimate,
            inhibitory_support_tick_estimate=0,
            context_sha256=context_sha256,
            trajectory_sha256=trajectory_sha256,
            evidence_sha256=evidence.sha256,
            adapter=adapter,
        )
        provenance = _digest(
            _PROVENANCE_NAMESPACE,
            {
                "compiled_trajectory_sha256": trajectory_sha256,
                "evidence_sha256": evidence.sha256,
                "parent_provenance_sha256": self._provenance_sha256,
                "segment": segment.record(),
            },
        )
        successor = AppendOnlyRawSourceMotorStack(
            segments=(*self._segments, segment),
            provenance_sha256=provenance,
        )
        for retained in self._segments:
            _ = retained.adapter.checkpoint_sha256
        if self._checkpoint_sha256 != parent_checkpoint:
            raise RuntimeError("online structural growth mutated its parent")
        receipt = self._growth_receipt(
            status="APPENDED",
            stop_reason=None,
            successor=successor,
            evidence=evidence,
            trajectory_sha256=trajectory_sha256,
            reserved_cell_address_start=segment.selector_cell_id,
            reserved_cell_capacity_added=segment.reserved_cell_capacity,
            reserved_synapse_address_start=segment.reserved_synapse_address_start,
            reserved_synapse_capacity_estimate_added=(segment.reserved_synapse_capacity_estimate),
            excitatory_support_tick_estimate_added=(segment.excitatory_support_tick_estimate),
            coactivation_support_tick_estimate_added=(coactivation_support_tick_estimate),
            positive_success_support_tick_estimate_added=(positive_success_support_tick_estimate),
            inhibitory_support_tick_estimate_added=0,
            committed_evidence_tick_phases=2,
            admission_race=None,
        )
        return OnlineRawSourceMotorGrowthTransition(successor, receipt)

    def _zero_growth(
        self,
        *,
        status: str,
        stop_reason: str | None,
        evidence: SourceMotorSuccessEvidence,
        trajectory_sha256: str,
        admission_race: OnlineStructuralRaceTrace | None,
    ) -> OnlineRawSourceMotorGrowthTransition:
        receipt = self._growth_receipt(
            status=status,
            stop_reason=stop_reason,
            successor=self,
            evidence=evidence,
            trajectory_sha256=trajectory_sha256,
            reserved_cell_address_start=None,
            reserved_cell_capacity_added=0,
            reserved_synapse_address_start=None,
            reserved_synapse_capacity_estimate_added=0,
            excitatory_support_tick_estimate_added=0,
            coactivation_support_tick_estimate_added=0,
            positive_success_support_tick_estimate_added=0,
            inhibitory_support_tick_estimate_added=0,
            committed_evidence_tick_phases=0,
            admission_race=admission_race,
        )
        return OnlineRawSourceMotorGrowthTransition(self, receipt)

    def _growth_receipt(
        self,
        *,
        status: str,
        stop_reason: str | None,
        successor: AppendOnlyRawSourceMotorStack,
        evidence: SourceMotorSuccessEvidence,
        trajectory_sha256: str,
        reserved_cell_address_start: int | None,
        reserved_cell_capacity_added: int,
        reserved_synapse_address_start: int | None,
        reserved_synapse_capacity_estimate_added: int,
        excitatory_support_tick_estimate_added: int,
        coactivation_support_tick_estimate_added: int,
        positive_success_support_tick_estimate_added: int,
        inhibitory_support_tick_estimate_added: int,
        committed_evidence_tick_phases: int,
        admission_race: OnlineStructuralRaceTrace | None,
    ) -> OnlineRawSourceMotorGrowthReceipt:
        record = {
            "action_sha256": evidence.action_sha256,
            "admission_race_sha256": (None if admission_race is None else admission_race.sha256),
            "compiled_trajectory_sha256": trajectory_sha256,
            "coactivation_support_tick_estimate_added": (coactivation_support_tick_estimate_added),
            "committed_evidence_tick_phases": committed_evidence_tick_phases,
            "evidence_sha256": evidence.sha256,
            "execution_receipt_sha256": evidence.execution_receipt.sha256,
            "excitatory_support_tick_estimate_added": (excitatory_support_tick_estimate_added),
            "inhibitory_support_tick_estimate_added": (inhibitory_support_tick_estimate_added),
            "observation_sha256": evidence.observation_sha256,
            "parent_checkpoint_sha256": self._checkpoint_sha256,
            "parent_experience_replay_count": 0,
            "positive_success_support_tick_estimate_added": (
                positive_success_support_tick_estimate_added
            ),
            "prior_provenance_sha256": self._provenance_sha256,
            "raw_training_sequences_retained": 0,
            "reserved_cell_address_start": reserved_cell_address_start,
            "reserved_cell_capacity_added": reserved_cell_capacity_added,
            "reserved_synapse_address_start": reserved_synapse_address_start,
            "reserved_synapse_capacity_estimate_added": (reserved_synapse_capacity_estimate_added),
            "source_experience_count": 1,
            "status": status,
            "stop_reason": stop_reason,
            "successor_checkpoint_sha256": successor.checkpoint_sha256,
            "successor_provenance_sha256": successor.training_summary.provenance_sha256,
        }
        return OnlineRawSourceMotorGrowthReceipt(
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
            reserved_cell_capacity_added=reserved_cell_capacity_added,
            reserved_synapse_address_start=reserved_synapse_address_start,
            reserved_synapse_capacity_estimate_added=(reserved_synapse_capacity_estimate_added),
            excitatory_support_tick_estimate_added=(excitatory_support_tick_estimate_added),
            coactivation_support_tick_estimate_added=(coactivation_support_tick_estimate_added),
            positive_success_support_tick_estimate_added=(
                positive_success_support_tick_estimate_added
            ),
            inhibitory_support_tick_estimate_added=(inhibitory_support_tick_estimate_added),
            source_experience_count=1,
            committed_evidence_tick_phases=committed_evidence_tick_phases,
            parent_experience_replay_count=0,
            raw_training_sequences_retained=0,
            admission_race=admission_race,
            sha256=_digest(_GROWTH_NAMESPACE, record),
        )
