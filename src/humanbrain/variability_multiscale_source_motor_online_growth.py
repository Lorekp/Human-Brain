"""Replay-free online growth for variability-aware opaque motor controllers.

One already-generalized controller seeds the immutable stack.  A new neuronal
shard can be appended only from a current cohort of at least two independent,
evaluator-authenticated complete transitions.  The cohort is compiled once and
discarded; neither parent experiences nor raw current rows are retained.

Frozen inference accepts only a raw query, raw workspace files, and a causal
ablation.  Every shard runs on that same observation, after which one fitted
GLIF5 population selects the sole emitting shard or physically selects tonic
or collision STOP.  This is a narrow opaque continual-structure mechanism, not
general language understanding or arbitrary software repair.
"""

from __future__ import annotations

import json
from copy import copy
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.multiscale_source_motor_online_growth import (
    MultiscaleSourceMotorExecutionAuthority,
    MultiscaleSourceMotorExecutionReceipt,
    MultiscaleSourceMotorSuccessEvidence,
    OnlineMultiscaleGrowthAblation,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.source_relative_action_tape import (
    OccurrencePreservingActionTape,
    TapeValence,
)
from humanbrain.variability_multiscale_source_motor_controller import (
    SuccessfulVariabilityMultiscaleSourceMotorExperience,
    VariabilityMultiscaleSourceMotorAblation,
    VariabilityMultiscaleSourceMotorController,
)

__all__ = [
    "AppendOnlyVariabilityMultiscaleSourceMotorStack",
    "MultiscaleSourceMotorExecutionAuthority",
    "MultiscaleSourceMotorExecutionReceipt",
    "MultiscaleSourceMotorSuccessEvidence",
    "OnlineMultiscaleGrowthAblation",
    "VariabilityMultiscaleOnlineDecision",
    "VariabilityMultiscaleOnlineGrowthReceipt",
    "VariabilityMultiscaleOnlineGrowthTransition",
    "VariabilityMultiscaleOnlineSummary",
    "VariabilityMultiscaleShardRaceTrace",
    "VariabilityMultiscaleSourceMotorAblation",
]


_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_SCORE_CAP = 2
_TONIC_STOP_SCORE = 1
_EMIT_SCORE = 2
_MAX_CURRENT_SUCCESSES = 64

_CHECKPOINT_NAMESPACE = b"humanbrain.variability-multiscale-online.checkpoint.v1\0"
_PROVENANCE_NAMESPACE = b"humanbrain.variability-multiscale-online.provenance.v1\0"
_QUERY_NAMESPACE = b"humanbrain.variability-multiscale-online.query.v1\0"
_RACE_NAMESPACE = b"humanbrain.variability-multiscale-online.race.v1\0"
_DECISION_NAMESPACE = b"humanbrain.variability-multiscale-online.decision.v1\0"
_GROWTH_NAMESPACE = b"humanbrain.variability-multiscale-online.growth.v1\0"
_TRAJECTORY_NAMESPACE = b"humanbrain.variability-multiscale-online.trajectory.v1\0"
_CONSTRUCTION_TOKEN = object()


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


def _workspace_record(
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {"content_hex": item.content.hex(), "path": item.path} for item in files
    )


def _canonical_observation(
    query: bytes,
    files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[bytes, tuple[OpaqueWorkspaceFile, ...]]:
    observation = MultiscaleOpaqueEventTape.observe(query, files)
    return observation.reconstruct_query(), observation.reconstruct_workspace()


def _expected_emissions(
    query: bytes,
    before_files: tuple[OpaqueWorkspaceFile, ...],
    after_files: tuple[OpaqueWorkspaceFile, ...],
) -> tuple[bytes, ...]:
    before = MultiscaleOpaqueEventTape.observe(query, before_files)
    after = MultiscaleOpaqueEventTape.observe(query, after_files)
    canonical_before = before.reconstruct_workspace()
    canonical_after = after.reconstruct_workspace()
    if tuple(item.path for item in canonical_before) != tuple(
        item.path for item in canonical_after
    ):
        raise ValueError("successful evidence must retain every workspace path")
    changed = tuple(
        (before_item, after_item)
        for before_item, after_item in zip(
            canonical_before,
            canonical_after,
            strict=True,
        )
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
    crossing_current = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    currents: list[float] = []
    for score in range(_SCORE_CAP + 1):
        first_step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("variability-online fitted-GLIF calibration changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        result.spike_step != _ZERO_SCORE_SPIKE_STEP - score
        for score, result in enumerate(controls)
    ):
        raise RuntimeError("variability-online fitted-GLIF latency changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


@dataclass(frozen=True, slots=True)
class VariabilityMultiscaleShardRaceTrace:
    population: str
    shard_cell_ids: tuple[int, ...]
    shard_scores: tuple[int, ...]
    tonic_stop_cell_id: int
    winner_cell_id: int
    spike_step: int
    simultaneous_spike_cells: int
    stopped: bool
    stop_reason: str | None
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VariabilityMultiscaleOnlineSummary:
    shard_count: int
    appended_structure_count: int
    seed_experience_count: int
    committed_current_success_count: int
    selector_cell_count: int
    parent_experience_replay_count: int
    raw_training_sequences_retained: int
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class VariabilityMultiscaleOnlineDecision:
    decision: str
    emissions: tuple[bytes, ...]
    winning_shard_ordinal: int | None
    winning_selector_cell_id: int
    stop_reason: str | None
    component_decision_sha256s: tuple[str, ...]
    component_stop_reasons: tuple[str | None, ...]
    component_receipt_sha256: str | None
    race: VariabilityMultiscaleShardRaceTrace
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class VariabilityMultiscaleOnlineGrowthReceipt:
    status: str
    stop_reason: str | None
    parent_checkpoint_sha256: str
    successor_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    evidence_sha256s: tuple[str, ...]
    execution_receipt_sha256s: tuple[str, ...]
    compiled_trajectory_sha256: str
    prior_provenance_sha256: str
    successor_provenance_sha256: str
    appended_shard_ordinal: int | None
    current_success_count: int
    current_experience_training_rows: int
    parent_experience_replay_count: int
    raw_training_sequences_retained: int
    parent_probe_decision_sha256s: tuple[str, ...]
    candidate_probe_decision_sha256s: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class VariabilityMultiscaleOnlineGrowthTransition:
    successor: AppendOnlyVariabilityMultiscaleSourceMotorStack
    receipt: VariabilityMultiscaleOnlineGrowthReceipt


@dataclass(frozen=True, slots=True)
class _ComponentEmission:
    emitted: bool
    emissions: tuple[bytes, ...]
    stop_reason: str | None
    decision_sha256: str
    receipt_sha256: str | None
    physical_cell_steps: int


def _copy_action_tape(
    tape: OccurrencePreservingActionTape,
) -> OccurrencePreservingActionTape:
    if tape._pending is not None:
        raise RuntimeError("cannot freeze a variability action tape with pending state")
    cloned = copy(tape)
    cloned._adjustments = dict(tape._adjustments)
    cloned._pending = None
    return cloned


def _copy_controller(
    controller: VariabilityMultiscaleSourceMotorController,
) -> VariabilityMultiscaleSourceMotorController:
    if controller._action_tape._pending is not None:
        raise RuntimeError("cannot freeze a variability controller with pending state")
    cloned = copy(controller)
    cloned._binders = tuple(copy(binder) for binder in controller._binders)
    cloned._action_tape = _copy_action_tape(controller._action_tape)
    return cloned


@dataclass(frozen=True, slots=True)
class _ControllerAdapter:
    controller: VariabilityMultiscaleSourceMotorController
    expected_checkpoint_sha256: str

    @property
    def checkpoint_sha256(self) -> str:
        if self.controller._action_tape._pending is not None:
            raise RuntimeError("frozen variability controller gained pending state")
        if self.controller.checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise RuntimeError("frozen variability controller checkpoint changed")
        return self.expected_checkpoint_sha256

    def respond(
        self,
        query: bytes,
        files: tuple[OpaqueWorkspaceFile, ...],
        *,
        ablation: VariabilityMultiscaleSourceMotorAblation,
    ) -> _ComponentEmission:
        _ = self.checkpoint_sha256
        ephemeral = _copy_controller(self.controller)
        decision = ephemeral.respond(query, files, ablation=ablation)
        receipt_sha256: str | None = None
        if decision.decision == "EMIT":
            if decision.action is None or decision.action.eligibility is None:
                raise RuntimeError("variability EMIT lost its eligibility receipt")
            receipt_sha256 = decision.action.eligibility.sha256
            consequence = ephemeral.resolve(TapeValence.NEUTRAL)
            if consequence.receipt_sha256 != receipt_sha256:
                raise RuntimeError("variability neutral resolution lost receipt custody")
        if ephemeral.checkpoint_sha256 != self.expected_checkpoint_sha256:
            raise RuntimeError("ephemeral variability response changed learned state")
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
class _FrozenShard:
    ordinal: int
    selector_cell_id: int
    source_success_count: int
    evidence_sha256s: tuple[str, ...]
    observation_sha256s: tuple[str, ...]
    action_sha256s: tuple[str, ...]
    trajectory_sha256: str | None
    adapter: _ControllerAdapter

    def record(self) -> dict[str, object]:
        return {
            "action_sha256s": self.action_sha256s,
            "controller_checkpoint_sha256": self.adapter.checkpoint_sha256,
            "evidence_sha256s": self.evidence_sha256s,
            "observation_sha256s": self.observation_sha256s,
            "ordinal": self.ordinal,
            "selector_cell_id": self.selector_cell_id,
            "source_success_count": self.source_success_count,
            "trajectory_sha256": self.trajectory_sha256,
        }


def _physical_race(
    *,
    shard_cell_ids: tuple[int, ...],
    shard_scores: tuple[int, ...],
) -> VariabilityMultiscaleShardRaceTrace:
    if len(shard_cell_ids) != len(shard_scores) or not shard_cell_ids:
        raise RuntimeError("variability shard race custody differs")
    if any(score not in range(_SCORE_CAP + 1) for score in shard_scores):
        raise ValueError("variability shard score is out of range")
    stop_cell_id = 0
    scores = (*shard_scores, _TONIC_STOP_SCORE)
    currents = _SCORE_CURRENTS[np.asarray(scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(scores) * _RACE_STEPS
    stop_reason: str | None = None
    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_TONIC_STOP_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1 or relay.winner_cell != 0:
            raise RuntimeError("variability shard collision STOP relay changed")
        physical_cell_steps += _RACE_STEPS
        winner_cell_id = stop_cell_id
        stop_reason = "collision"
    elif result.winner_cell == len(shard_cell_ids):
        winner_cell_id = stop_cell_id
        stop_reason = "tonic_stop"
    else:
        winner_cell_id = shard_cell_ids[result.winner_cell]
    record = {
        "physical_cell_steps": physical_cell_steps,
        "population": "variability_online_shard",
        "shard_cell_ids": shard_cell_ids,
        "shard_scores": shard_scores,
        "simultaneous_spike_cells": result.simultaneous_spike_candidates,
        "spike_step": result.spike_step,
        "stop_reason": stop_reason,
        "stopped": stop_reason is not None,
        "tonic_stop_cell_id": stop_cell_id,
        "winner_cell_id": winner_cell_id,
    }
    return VariabilityMultiscaleShardRaceTrace(
        **record,
        sha256=_digest(_RACE_NAMESPACE, record),
    )


def _trajectory_sha256(
    *,
    controller_checkpoint_sha256: str,
    successes: tuple[MultiscaleSourceMotorSuccessEvidence, ...],
    expected_emissions: tuple[tuple[bytes, ...], ...],
) -> str:
    return _digest(
        _TRAJECTORY_NAMESPACE,
        {
            "action_sha256s": tuple(item.action_sha256 for item in successes),
            "controller_checkpoint_sha256": controller_checkpoint_sha256,
            "evidence_sha256s": tuple(item.sha256 for item in successes),
            "expected_emissions_hex": tuple(
                tuple(event.hex() for event in row) for row in expected_emissions
            ),
            "observation_sha256s": tuple(
                item.observation_sha256 for item in successes
            ),
        },
    )


def _checkpoint_sha256(
    shards: tuple[_FrozenShard, ...],
    provenance_sha256: str,
) -> str:
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "provenance_sha256": provenance_sha256,
            "shards": tuple(shard.record() for shard in shards),
        },
    )


@dataclass(frozen=True, slots=True, init=False)
class AppendOnlyVariabilityMultiscaleSourceMotorStack:
    """One immutable generalized seed plus replay-free learned shards."""

    _shards: tuple[_FrozenShard, ...]
    _provenance_sha256: str
    _checkpoint_sha256: str
    _summary: VariabilityMultiscaleOnlineSummary

    def __init__(
        self,
        token: object,
        *,
        shards: tuple[_FrozenShard, ...],
        provenance_sha256: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use AppendOnlyVariabilityMultiscaleSourceMotorStack.seed")
        if not shards:
            raise ValueError("online stack requires one generalized seed shard")
        if tuple(shard.ordinal for shard in shards) != tuple(range(len(shards))):
            raise ValueError("shard ordinals must be contiguous")
        if tuple(shard.selector_cell_id for shard in shards) != tuple(
            range(1, len(shards) + 1)
        ):
            raise ValueError("selector cell addresses must be contiguous")
        for shard in shards:
            _ = shard.adapter.checkpoint_sha256
        checkpoint = _checkpoint_sha256(shards, provenance_sha256)
        seed_experiences = shards[0].source_success_count
        committed = sum(shard.source_success_count for shard in shards[1:])
        summary = VariabilityMultiscaleOnlineSummary(
            shard_count=len(shards),
            appended_structure_count=len(shards) - 1,
            seed_experience_count=seed_experiences,
            committed_current_success_count=committed,
            selector_cell_count=len(shards) + 1,
            parent_experience_replay_count=0,
            raw_training_sequences_retained=0,
            provenance_sha256=provenance_sha256,
        )
        object.__setattr__(self, "_shards", shards)
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)
        object.__setattr__(self, "_checkpoint_sha256", checkpoint)
        object.__setattr__(self, "_summary", summary)

    @classmethod
    def seed(
        cls,
        controller: VariabilityMultiscaleSourceMotorController,
    ) -> AppendOnlyVariabilityMultiscaleSourceMotorStack:
        if not isinstance(controller, VariabilityMultiscaleSourceMotorController):
            raise TypeError("controller must be VariabilityMultiscaleSourceMotorController")
        if controller.training_summary.experience_count < 2:
            raise ValueError("seed controller must already be generalized")
        frozen = _copy_controller(controller)
        adapter = _ControllerAdapter(frozen, frozen.checkpoint_sha256)
        shard = _FrozenShard(
            ordinal=0,
            selector_cell_id=1,
            source_success_count=frozen.training_summary.experience_count,
            evidence_sha256s=(),
            observation_sha256s=(),
            action_sha256s=(),
            trajectory_sha256=None,
            adapter=adapter,
        )
        provenance = _digest(
            _PROVENANCE_NAMESPACE,
            {
                "seed_checkpoint_sha256": frozen.checkpoint_sha256,
                "seed_experience_count": frozen.training_summary.experience_count,
            },
        )
        return cls(
            _CONSTRUCTION_TOKEN,
            shards=(shard,),
            provenance_sha256=provenance,
        )

    @property
    def checkpoint_sha256(self) -> str:
        for shard in self._shards:
            _ = shard.adapter.checkpoint_sha256
        current = _checkpoint_sha256(self._shards, self._provenance_sha256)
        if current != self._checkpoint_sha256:
            raise RuntimeError("immutable variability stack checkpoint changed")
        return current

    @property
    def training_summary(self) -> VariabilityMultiscaleOnlineSummary:
        return self._summary

    def respond(
        self,
        query: bytes,
        files: tuple[OpaqueWorkspaceFile, ...],
        *,
        ablation: VariabilityMultiscaleSourceMotorAblation = (
            VariabilityMultiscaleSourceMotorAblation.NONE
        ),
    ) -> VariabilityMultiscaleOnlineDecision:
        """Run every frozen shard and physically arbitrate one public response."""

        if not isinstance(ablation, VariabilityMultiscaleSourceMotorAblation):
            raise TypeError(
                "ablation must be a VariabilityMultiscaleSourceMotorAblation"
            )
        canonical_query, canonical_files = _canonical_observation(query, files)
        checkpoint = self.checkpoint_sha256
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "checkpoint_sha256": checkpoint,
                "files": _workspace_record(canonical_files),
                "query_hex": canonical_query.hex(),
            },
        )
        components = tuple(
            shard.adapter.respond(
                canonical_query,
                canonical_files,
                ablation=ablation,
            )
            for shard in self._shards
        )
        race = _physical_race(
            shard_cell_ids=tuple(shard.selector_cell_id for shard in self._shards),
            shard_scores=tuple(
                _EMIT_SCORE if component.emitted else 0 for component in components
            ),
        )
        physical_cell_steps = race.physical_cell_steps + sum(
            component.physical_cell_steps for component in components
        )
        winning_ordinal: int | None = None
        emissions: tuple[bytes, ...] = ()
        receipt_sha256: str | None = None
        if not race.stopped:
            selector_to_ordinal = {
                shard.selector_cell_id: shard.ordinal for shard in self._shards
            }
            winning_ordinal = selector_to_ordinal[race.winner_cell_id]
            selected = components[winning_ordinal]
            if not selected.emitted or selected.receipt_sha256 is None:
                raise RuntimeError("physical variability winner lost its EMIT receipt")
            emissions = selected.emissions
            receipt_sha256 = selected.receipt_sha256
        component_hashes = tuple(component.decision_sha256 for component in components)
        component_stops = tuple(component.stop_reason for component in components)
        record = {
            "checkpoint_sha256": checkpoint,
            "component_decision_sha256s": component_hashes,
            "component_receipt_sha256": receipt_sha256,
            "component_stop_reasons": component_stops,
            "decision": "STOP" if race.stopped else "EMIT",
            "emissions_hex": tuple(event.hex() for event in emissions),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "race_sha256": race.sha256,
            "stop_reason": race.stop_reason,
            "winning_selector_cell_id": race.winner_cell_id,
            "winning_shard_ordinal": winning_ordinal,
        }
        decision = VariabilityMultiscaleOnlineDecision(
            decision="STOP" if race.stopped else "EMIT",
            emissions=emissions,
            winning_shard_ordinal=winning_ordinal,
            winning_selector_cell_id=race.winner_cell_id,
            stop_reason=race.stop_reason,
            component_decision_sha256s=component_hashes,
            component_stop_reasons=component_stops,
            component_receipt_sha256=receipt_sha256,
            race=race,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=checkpoint,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )
        if self.checkpoint_sha256 != checkpoint:
            raise RuntimeError("variability inference mutated its frozen checkpoint")
        return decision

    def extend_successes(
        self,
        successes: tuple[MultiscaleSourceMotorSuccessEvidence, ...],
        *,
        ablation: OnlineMultiscaleGrowthAblation = OnlineMultiscaleGrowthAblation.NONE,
    ) -> VariabilityMultiscaleOnlineGrowthTransition:
        """Append one structure from only an independent current success cohort."""

        if type(successes) is not tuple or not 2 <= len(successes) <= _MAX_CURRENT_SUCCESSES:
            raise ValueError("successes must contain at least two bounded current rows")
        if any(
            not isinstance(item, MultiscaleSourceMotorSuccessEvidence)
            for item in successes
        ):
            raise TypeError("successes must contain MultiscaleSourceMotorSuccessEvidence")
        if not isinstance(ablation, OnlineMultiscaleGrowthAblation):
            raise TypeError("ablation must be OnlineMultiscaleGrowthAblation")
        parent_checkpoint = self.checkpoint_sha256
        for item in successes:
            item.authenticate()
            if item.execution_receipt.learner_checkpoint_sha256 != parent_checkpoint:
                raise ValueError("every execution receipt must bind this current learner")
        if len({item.sha256 for item in successes}) != len(successes):
            raise ValueError("current successes must be independent, not duplicate evidence")
        if len({item.observation_sha256 for item in successes}) != len(successes):
            raise ValueError("current successes need distinct complete observations")
        if len({item.provenance_id for item in successes}) != len(successes):
            raise ValueError("current successes need distinct provenance")

        expected = tuple(
            _expected_emissions(item.query, item.before_files, item.after_files)
            for item in successes
        )
        experiences = tuple(
            SuccessfulVariabilityMultiscaleSourceMotorExperience(
                query=item.query,
                before_files=item.before_files,
                after_files=item.after_files,
                provenance_id=item.provenance_id,
            )
            for item in successes
        )
        candidate = VariabilityMultiscaleSourceMotorController.grow(experiences)
        frozen_candidate = _copy_controller(candidate)
        candidate_adapter = _ControllerAdapter(
            frozen_candidate,
            frozen_candidate.checkpoint_sha256,
        )
        candidate_probes = tuple(
            candidate_adapter.respond(
                item.query,
                item.before_files,
                ablation=VariabilityMultiscaleSourceMotorAblation.NONE,
            )
            for item in successes
        )
        if any(
            not probe.emitted or probe.emissions != expected_row
            for probe, expected_row in zip(candidate_probes, expected, strict=True)
        ):
            raise ValueError("current successes did not recruit one reusable shard")
        trajectory = _trajectory_sha256(
            controller_checkpoint_sha256=frozen_candidate.checkpoint_sha256,
            successes=successes,
            expected_emissions=expected,
        )

        parent_probes = tuple(
            self.respond(item.query, item.before_files) for item in successes
        )
        parent_emits = tuple(probe.decision == "EMIT" for probe in parent_probes)
        parent_matches = tuple(
            probe.decision == "EMIT" and probe.emissions == expected_row
            for probe, expected_row in zip(parent_probes, expected, strict=True)
        )
        if all(parent_matches):
            return self._zero_growth(
                status="DUPLICATE",
                stop_reason=None,
                successes=successes,
                candidate_checkpoint_sha256=frozen_candidate.checkpoint_sha256,
                trajectory_sha256=trajectory,
                parent_probes=parent_probes,
                candidate_probes=candidate_probes,
            )
        if any(parent_emits) or any(
            probe.stop_reason == "collision" for probe in parent_probes
        ):
            return self._zero_growth(
                status="CONFLICT_STOP",
                stop_reason="parent_structure_overlap",
                successes=successes,
                candidate_checkpoint_sha256=frozen_candidate.checkpoint_sha256,
                trajectory_sha256=trajectory,
                parent_probes=parent_probes,
                candidate_probes=candidate_probes,
            )
        if ablation is OnlineMultiscaleGrowthAblation.NO_UPDATE:
            return self._zero_growth(
                status="NO_UPDATE",
                stop_reason=None,
                successes=successes,
                candidate_checkpoint_sha256=frozen_candidate.checkpoint_sha256,
                trajectory_sha256=trajectory,
                parent_probes=parent_probes,
                candidate_probes=candidate_probes,
            )

        ordinal = len(self._shards)
        shard = _FrozenShard(
            ordinal=ordinal,
            selector_cell_id=ordinal + 1,
            source_success_count=len(successes),
            evidence_sha256s=tuple(item.sha256 for item in successes),
            observation_sha256s=tuple(item.observation_sha256 for item in successes),
            action_sha256s=tuple(item.action_sha256 for item in successes),
            trajectory_sha256=trajectory,
            adapter=candidate_adapter,
        )
        provenance = _digest(
            _PROVENANCE_NAMESPACE,
            {
                "current_evidence_sha256s": shard.evidence_sha256s,
                "parent_provenance_sha256": self._provenance_sha256,
                "shard": shard.record(),
            },
        )
        successor = AppendOnlyVariabilityMultiscaleSourceMotorStack(
            _CONSTRUCTION_TOKEN,
            shards=(*self._shards, shard),
            provenance_sha256=provenance,
        )
        if self.checkpoint_sha256 != parent_checkpoint:
            raise RuntimeError("variability growth mutated its parent checkpoint")
        receipt = self._growth_receipt(
            status="APPENDED",
            stop_reason=None,
            successor=successor,
            successes=successes,
            candidate_checkpoint_sha256=frozen_candidate.checkpoint_sha256,
            trajectory_sha256=trajectory,
            appended_shard_ordinal=ordinal,
            parent_probes=parent_probes,
            candidate_probes=candidate_probes,
        )
        return VariabilityMultiscaleOnlineGrowthTransition(successor, receipt)

    def _zero_growth(
        self,
        *,
        status: str,
        stop_reason: str | None,
        successes: tuple[MultiscaleSourceMotorSuccessEvidence, ...],
        candidate_checkpoint_sha256: str,
        trajectory_sha256: str,
        parent_probes: tuple[VariabilityMultiscaleOnlineDecision, ...],
        candidate_probes: tuple[_ComponentEmission, ...],
    ) -> VariabilityMultiscaleOnlineGrowthTransition:
        receipt = self._growth_receipt(
            status=status,
            stop_reason=stop_reason,
            successor=self,
            successes=successes,
            candidate_checkpoint_sha256=candidate_checkpoint_sha256,
            trajectory_sha256=trajectory_sha256,
            appended_shard_ordinal=None,
            parent_probes=parent_probes,
            candidate_probes=candidate_probes,
        )
        return VariabilityMultiscaleOnlineGrowthTransition(self, receipt)

    def _growth_receipt(
        self,
        *,
        status: str,
        stop_reason: str | None,
        successor: AppendOnlyVariabilityMultiscaleSourceMotorStack,
        successes: tuple[MultiscaleSourceMotorSuccessEvidence, ...],
        candidate_checkpoint_sha256: str,
        trajectory_sha256: str,
        appended_shard_ordinal: int | None,
        parent_probes: tuple[VariabilityMultiscaleOnlineDecision, ...],
        candidate_probes: tuple[_ComponentEmission, ...],
    ) -> VariabilityMultiscaleOnlineGrowthReceipt:
        evidence_sha256s = tuple(item.sha256 for item in successes)
        execution_sha256s = tuple(
            item.execution_receipt.sha256 for item in successes
        )
        parent_probe_sha256s = tuple(item.sha256 for item in parent_probes)
        candidate_probe_sha256s = tuple(
            item.decision_sha256 for item in candidate_probes
        )
        record = {
            "appended_shard_ordinal": appended_shard_ordinal,
            "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
            "candidate_probe_decision_sha256s": candidate_probe_sha256s,
            "compiled_trajectory_sha256": trajectory_sha256,
            "current_experience_training_rows": len(successes),
            "current_success_count": len(successes),
            "evidence_sha256s": evidence_sha256s,
            "execution_receipt_sha256s": execution_sha256s,
            "parent_checkpoint_sha256": self._checkpoint_sha256,
            "parent_experience_replay_count": 0,
            "parent_probe_decision_sha256s": parent_probe_sha256s,
            "prior_provenance_sha256": self._provenance_sha256,
            "raw_training_sequences_retained": 0,
            "status": status,
            "stop_reason": stop_reason,
            "successor_checkpoint_sha256": successor.checkpoint_sha256,
            "successor_provenance_sha256": successor.training_summary.provenance_sha256,
        }
        return VariabilityMultiscaleOnlineGrowthReceipt(
            status=status,
            stop_reason=stop_reason,
            parent_checkpoint_sha256=self._checkpoint_sha256,
            successor_checkpoint_sha256=successor.checkpoint_sha256,
            candidate_checkpoint_sha256=candidate_checkpoint_sha256,
            evidence_sha256s=evidence_sha256s,
            execution_receipt_sha256s=execution_sha256s,
            compiled_trajectory_sha256=trajectory_sha256,
            prior_provenance_sha256=self._provenance_sha256,
            successor_provenance_sha256=successor.training_summary.provenance_sha256,
            appended_shard_ordinal=appended_shard_ordinal,
            current_success_count=len(successes),
            current_experience_training_rows=len(successes),
            parent_experience_replay_count=0,
            raw_training_sequences_retained=0,
            parent_probe_decision_sha256s=parent_probe_sha256s,
            candidate_probe_decision_sha256s=candidate_probe_sha256s,
            sha256=_digest(_GROWTH_NAMESPACE, record),
        )
