"""Compose variability-aware opaque grounding with a neuronal motor tape.

The learner receives independent successful full-stream transitions only while
growing.  It keeps a shared source-relative action and neuronal binders whose
stable and variable event hops were inferred across those transitions.  Public
inference accepts one complete query and workspace; no target, path, span,
candidate, repaired source, delimiter identity, or task ontology is accepted.

The event relation is deliberately opaque and can be reused outside source
code.  This controller is one software-facing composition test, not a claim of
language understanding or arbitrary editing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.source_relative_action_tape import (
    ActionTapeConsequenceUpdate,
    ActionTapeDecision,
    OccurrencePreservingActionTape,
    TapeValence,
)
from humanbrain.source_relative_training_compiler import (
    compile_successful_token_tapes,
)
from humanbrain.variability_aware_segment_binder import (
    VariabilityAwareSegmentBinder,
    VariabilityBinderAblation,
    VariabilityBindingDecision,
    VariabilitySegmentDemonstration,
)

_CHECKPOINT_NAMESPACE = (
    b"humanbrain.variability-multiscale-source-motor.checkpoint.v1\x00"
)
_QUERY_NAMESPACE = b"humanbrain.variability-multiscale-source-motor.query.v1\x00"
_DECISION_NAMESPACE = b"humanbrain.variability-multiscale-source-motor.decision.v1\x00"
_INVENTORY_RACE_NAMESPACE = (
    b"humanbrain.variability-multiscale-source-motor.inventory-race.v1\x00"
)
_CONSTRUCTION_TOKEN = object()

_RACE_STEPS = 600
_ZERO_SCORE_SPIKE_STEP = 568
_DRIVE_SCORE = 2
_TONIC_SCORE = 1


def _digest(namespace: bytes, record: object) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(namespace + payload).hexdigest()


def _build_score_currents() -> tuple[np.ndarray, CompiledRestingGlifRace]:
    population = AdaptiveGlifPopulation(1)
    voltage, threshold = population.compile_resting_response(_RACE_STEPS)
    crossing_current = population.parameters.threshold_inf_relative_mv / (
        voltage - threshold
    )
    currents: list[float] = []
    for score in range(_DRIVE_SCORE + 1):
        first_step = _ZERO_SCORE_SPIKE_STEP - score
        lower = float(crossing_current[first_step])
        upper = float(crossing_current[first_step - 1])
        if not lower < upper:
            raise RuntimeError("inventory fitted-GLIF calibration changed")
        currents.append((lower + upper) / 2.0)
    table = np.ascontiguousarray(currents, dtype=np.float64)
    table.setflags(write=False)
    race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
    controls = race.race_many(table[:, None])
    if any(
        result.spike_step != _ZERO_SCORE_SPIKE_STEP - score
        for score, result in enumerate(controls)
    ):
        raise RuntimeError("inventory fitted-GLIF latency changed")
    return table, race


_SCORE_CURRENTS, _CELL_RACE = _build_score_currents()


class VariabilityMultiscaleSourceMotorAblation(StrEnum):
    NONE = "none"
    EQUALITY_OFF = "equality_off"
    PROPAGATION_OFF = "propagation_off"
    BOUNDARY_OFF = "boundary_off"


@dataclass(frozen=True, slots=True)
class VariabilityInventoryRaceTrace:
    population: str
    cell_count: int
    cell_scores_sha256: str
    winner_cell: int | None
    spike_step: int
    simultaneous_spike_cells: int
    stopped: bool
    stop_reason: str | None
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _InventoryRaceDecision:
    column: int | None
    trace: VariabilityInventoryRaceTrace


def _physical_identity_choice(
    *,
    population: str,
    expected_event: bytes,
    candidate_events: tuple[bytes, ...],
) -> _InventoryRaceDecision:
    """Race matching event cells against one tonic STOP cell."""

    if not candidate_events:
        raise ValueError("inventory race requires candidate events")
    scores = tuple(
        _DRIVE_SCORE if event == expected_event else 0 for event in candidate_events
    )
    all_scores = (*scores, _TONIC_SCORE)
    currents = _SCORE_CURRENTS[np.asarray(all_scores, dtype=np.intp)]
    result = _CELL_RACE.race_many(currents[None, :], require_unique=False)[0]
    physical_cell_steps = len(all_scores) * _RACE_STEPS
    stopped = False
    stop_reason: str | None = None
    winner_cell: int | None = result.winner_cell
    if result.simultaneous_spike_candidates != 1:
        relay = _CELL_RACE.race_many(
            np.asarray([[_SCORE_CURRENTS[_TONIC_SCORE]]], dtype=np.float64)
        )[0]
        if relay.simultaneous_spike_candidates != 1:
            raise RuntimeError("inventory collision STOP relay changed")
        physical_cell_steps += _RACE_STEPS
        stopped = True
        stop_reason = "collision"
        winner_cell = None
    elif result.winner_cell == len(candidate_events):
        stopped = True
        stop_reason = "tonic_stop"
        winner_cell = None

    score_root = _digest(_INVENTORY_RACE_NAMESPACE, {"cell_scores": all_scores})
    record = {
        "cell_count": len(all_scores),
        "cell_scores_sha256": score_root,
        "physical_cell_steps": physical_cell_steps,
        "population": population,
        "simultaneous_spike_cells": result.simultaneous_spike_candidates,
        "spike_step": result.spike_step,
        "stop_reason": stop_reason,
        "stopped": stopped,
        "winner_cell": winner_cell,
    }
    trace = VariabilityInventoryRaceTrace(
        **record,
        sha256=_digest(_INVENTORY_RACE_NAMESPACE, record),
    )
    return _InventoryRaceDecision(column=winner_cell, trace=trace)


@dataclass(frozen=True, slots=True)
class SuccessfulVariabilityMultiscaleSourceMotorExperience:
    """One independent successful complete query/workspace transition."""

    query: bytes
    before_files: tuple[OpaqueWorkspaceFile, ...]
    after_files: tuple[OpaqueWorkspaceFile, ...]
    provenance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.query, bytes) or not self.query:
            raise ValueError("query must be non-empty bytes")
        if type(self.before_files) is not tuple or not self.before_files:
            raise ValueError("before_files must be a non-empty tuple")
        if type(self.after_files) is not tuple or not self.after_files:
            raise ValueError("after_files must be a non-empty tuple")
        if any(not isinstance(item, OpaqueWorkspaceFile) for item in self.before_files):
            raise TypeError("before_files must contain OpaqueWorkspaceFile values")
        if any(not isinstance(item, OpaqueWorkspaceFile) for item in self.after_files):
            raise TypeError("after_files must contain OpaqueWorkspaceFile values")
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be non-empty text")


@dataclass(frozen=True, slots=True)
class VariabilityMultiscaleSourceMotorTrainingSummary:
    experience_count: int
    role_binder_count: int
    role_offsets: tuple[int, ...]
    propagation_hop_count: int
    stable_intermediate_synapse_count: int
    wildcard_intermediate_hop_count: int
    wildcard_support_ticks: int
    boundary_synapse_count: int
    action_program_cells: int
    action_recurrent_state_cells: int
    action_learned_synapses: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class VariabilityMultiscaleSourceMotorDecision:
    decision: str
    emissions: tuple[bytes, ...]
    stop_stage: str | None
    stop_reason: str | None
    bindings: tuple[VariabilityBindingDecision, ...]
    inventory_races: tuple[VariabilityInventoryRaceTrace, ...]
    action: ActionTapeDecision | None
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


class VariabilityMultiscaleSourceMotorController:
    """Ground variable boundary-relative roles, then run one motor chain."""

    __slots__ = ("_action_tape", "_binders", "_summary")

    def __init__(
        self,
        token: object,
        *,
        binders: tuple[VariabilityAwareSegmentBinder, ...],
        action_tape: OccurrencePreservingActionTape,
        summary: VariabilityMultiscaleSourceMotorTrainingSummary,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use VariabilityMultiscaleSourceMotorController.grow")
        self._binders = binders
        self._action_tape = action_tape
        self._summary = summary

    @classmethod
    def grow(
        cls,
        experiences: Iterable[SuccessfulVariabilityMultiscaleSourceMotorExperience],
    ) -> VariabilityMultiscaleSourceMotorController:
        """Learn one shared action and variable relations from full transitions."""

        rows = tuple(experiences)
        if len(rows) < 2:
            raise ValueError(
                "at least two independent successful experiences are required"
            )
        if any(
            not isinstance(row, SuccessfulVariabilityMultiscaleSourceMotorExperience)
            for row in rows
        ):
            raise TypeError(
                "experiences must contain "
                "SuccessfulVariabilityMultiscaleSourceMotorExperience"
            )
        transition_keys = tuple(
            (row.query, row.before_files, row.after_files) for row in rows
        )
        if len(set(transition_keys)) != len(rows):
            raise ValueError(
                "successful experiences must be independent, not duplicate replays "
                "with alternate provenance"
            )
        if len({row.provenance_id for row in rows}) != len(rows):
            raise ValueError(
                "independent successful experiences need distinct provenance"
            )
        if len({row.query for row in rows}) != len(rows):
            raise ValueError("independent successful experiences need distinct queries")
        if len({row.before_files for row in rows}) != len(rows):
            raise ValueError(
                "independent successful experiences need distinct workspaces"
            )

        motor_demonstrations = []
        observations: list[MultiscaleOpaqueEventTape] = []
        for row in rows:
            before = MultiscaleOpaqueEventTape.observe(row.query, row.before_files)
            after = MultiscaleOpaqueEventTape.observe(row.query, row.after_files)
            before_files = before.reconstruct_workspace()
            after_files = after.reconstruct_workspace()
            before_paths = tuple(item.path for item in before_files)
            after_paths = tuple(item.path for item in after_files)
            if before_paths != after_paths:
                raise ValueError(
                    "successful experience must retain every workspace path"
                )
            changed = tuple(
                (before_item, after_item)
                for before_item, after_item in zip(
                    before_files, after_files, strict=True
                )
                if before_item.content != after_item.content
            )
            if len(changed) != 1:
                raise ValueError("successful experience must change exactly one file")
            changed_before, changed_after = changed[0]
            after_motor_tokens = before.motor_tokens_for_replacement(
                changed_before.path,
                changed_after.content,
            )
            demonstration = compile_successful_token_tapes(
                context_event=before.query_events[0],
                before_tokens=before.motor_source_tokens,
                after_tokens=after_motor_tokens,
            )
            if not demonstration.cue_inventory:
                raise ValueError(
                    "successful experience produced no grounded motor role"
                )
            observations.append(before)
            motor_demonstrations.append(demonstration)

        shared_program = motor_demonstrations[0].motor_steps
        if any(row.motor_steps != shared_program for row in motor_demonstrations[1:]):
            raise ValueError("successful experiences do not share one action shape")
        cue_count = len(motor_demonstrations[0].cue_inventory)
        if any(len(row.cue_inventory) != cue_count for row in motor_demonstrations[1:]):
            raise ValueError("successful experiences do not share one motor-role shape")

        binders = tuple(
            VariabilityAwareSegmentBinder.grow(
                VariabilitySegmentDemonstration(
                    query_events=observation.query_events,
                    source_events=observation.source_events,
                    demonstrated_role=demonstration.cue_inventory[cue_ordinal],
                )
                for observation, demonstration in zip(
                    observations,
                    motor_demonstrations,
                    strict=True,
                )
            )
            for cue_ordinal in range(cue_count)
        )
        if any(
            binder.training_summary.wildcard_intermediate_hop_count == 0
            for binder in binders
        ):
            raise ValueError(
                "independent successful experiences must vary intermediate identities"
            )

        action_tape = OccurrencePreservingActionTape.grow(motor_demonstrations)
        checkpoint = _digest(
            _CHECKPOINT_NAMESPACE,
            {
                "action_checkpoint_sha256": action_tape.checkpoint_sha256,
                "binder_checkpoint_sha256s": tuple(
                    binder.checkpoint_sha256 for binder in binders
                ),
            },
        )
        binder_summaries = tuple(binder.training_summary for binder in binders)
        action_summary = action_tape.training_summary
        summary = VariabilityMultiscaleSourceMotorTrainingSummary(
            experience_count=len(rows),
            role_binder_count=len(binders),
            role_offsets=tuple(item.role_offset for item in binder_summaries),
            propagation_hop_count=sum(
                item.propagation_hop_count for item in binder_summaries
            ),
            stable_intermediate_synapse_count=sum(
                item.stable_intermediate_synapse_count for item in binder_summaries
            ),
            wildcard_intermediate_hop_count=sum(
                item.wildcard_intermediate_hop_count for item in binder_summaries
            ),
            wildcard_support_ticks=sum(
                item.wildcard_support_ticks for item in binder_summaries
            ),
            boundary_synapse_count=sum(
                item.boundary_synapse_count for item in binder_summaries
            ),
            action_program_cells=action_summary.program_cells,
            action_recurrent_state_cells=action_summary.recurrent_state_cells,
            action_learned_synapses=action_summary.learned_synapses,
            checkpoint_sha256=checkpoint,
        )
        return cls(
            _CONSTRUCTION_TOKEN,
            binders=binders,
            action_tape=action_tape,
            summary=summary,
        )

    @property
    def checkpoint_sha256(self) -> str:
        return _digest(
            _CHECKPOINT_NAMESPACE,
            {
                "action_checkpoint_sha256": self._action_tape.checkpoint_sha256,
                "binder_checkpoint_sha256s": tuple(
                    binder.checkpoint_sha256 for binder in self._binders
                ),
            },
        )

    @property
    def training_summary(self) -> VariabilityMultiscaleSourceMotorTrainingSummary:
        return self._summary

    def respond(
        self,
        query: bytes,
        files: tuple[OpaqueWorkspaceFile, ...],
        *,
        ablation: VariabilityMultiscaleSourceMotorAblation = (
            VariabilityMultiscaleSourceMotorAblation.NONE
        ),
    ) -> VariabilityMultiscaleSourceMotorDecision:
        """Process one complete public query/workspace observation exactly once."""

        if not isinstance(ablation, VariabilityMultiscaleSourceMotorAblation):
            raise TypeError(
                "ablation must be a VariabilityMultiscaleSourceMotorAblation"
            )
        observation = MultiscaleOpaqueEventTape.observe(query, files)
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "query_events_hex": tuple(
                    event.hex() for event in observation.query_events
                ),
                "source_events_hex": tuple(
                    event.hex() for event in observation.source_events
                ),
            },
        )
        binder_ablation = {
            VariabilityMultiscaleSourceMotorAblation.NONE: VariabilityBinderAblation.NONE,
            VariabilityMultiscaleSourceMotorAblation.EQUALITY_OFF: (
                VariabilityBinderAblation.EQUALITY_OFF
            ),
            VariabilityMultiscaleSourceMotorAblation.PROPAGATION_OFF: (
                VariabilityBinderAblation.PROPAGATION_OFF
            ),
            VariabilityMultiscaleSourceMotorAblation.BOUNDARY_OFF: (
                VariabilityBinderAblation.BOUNDARY_OFF
            ),
        }[ablation]
        bindings: list[VariabilityBindingDecision] = []
        inventory_races: list[VariabilityInventoryRaceTrace] = []
        roles: list[bytes] = []
        for ordinal, binder in enumerate(self._binders):
            binding = binder.respond(
                observation.query_events,
                observation.source_events,
                ablation=binder_ablation,
            )
            bindings.append(binding)
            if binding.decision != "BIND" or binding.role_identity is None:
                return self._decision(
                    decision="STOP",
                    emissions=(),
                    stop_stage=f"role_binder_{ordinal}",
                    stop_reason=binding.stop_reason,
                    bindings=tuple(bindings),
                    inventory_races=tuple(inventory_races),
                    action=None,
                    query_sha256=query_sha256,
                )
            roles.append(binding.role_identity)

        role_inventory = tuple(roles)
        for ordinal, role in enumerate(role_inventory):
            race = _physical_identity_choice(
                population=f"role_inventory_identity_{ordinal}",
                expected_event=role,
                candidate_events=role_inventory,
            )
            inventory_races.append(race.trace)
            if race.column is None:
                return self._decision(
                    decision="STOP",
                    emissions=(),
                    stop_stage="role_inventory",
                    stop_reason="role_inventory:identity_collision",
                    bindings=tuple(bindings),
                    inventory_races=tuple(inventory_races),
                    action=None,
                    query_sha256=query_sha256,
                )

        live_motor_tokens = observation.motor_source_tokens
        for ordinal, role in enumerate(role_inventory):
            race = _physical_identity_choice(
                population=f"live_motor_role_occurrence_{ordinal}",
                expected_event=role,
                candidate_events=live_motor_tokens,
            )
            inventory_races.append(race.trace)
            if race.column is None:
                return self._decision(
                    decision="STOP",
                    emissions=(),
                    stop_stage="role_inventory",
                    stop_reason="role_inventory:non_unique_live_motor_role",
                    bindings=tuple(bindings),
                    inventory_races=tuple(inventory_races),
                    action=None,
                    query_sha256=query_sha256,
                )

        action = self._action_tape.respond(
            observation.query_events,
            role_inventory,
            live_motor_tokens,
        )
        if action.decision != "EMIT":
            return self._decision(
                decision="STOP",
                emissions=(),
                stop_stage="action_tape",
                stop_reason=action.stop_reason,
                bindings=tuple(bindings),
                inventory_races=tuple(inventory_races),
                action=action,
                query_sha256=query_sha256,
            )
        return self._decision(
            decision="EMIT",
            emissions=action.emissions,
            stop_stage=None,
            stop_reason=None,
            bindings=tuple(bindings),
            inventory_races=tuple(inventory_races),
            action=action,
            query_sha256=query_sha256,
        )

    def resolve(self, valence: TapeValence) -> ActionTapeConsequenceUpdate:
        """Route a consequence only to the exact pending neuronal motor chain."""

        return self._action_tape.resolve(valence)

    def _decision(
        self,
        *,
        decision: str,
        emissions: tuple[bytes, ...],
        stop_stage: str | None,
        stop_reason: str | None,
        bindings: tuple[VariabilityBindingDecision, ...],
        inventory_races: tuple[VariabilityInventoryRaceTrace, ...],
        action: ActionTapeDecision | None,
        query_sha256: str,
    ) -> VariabilityMultiscaleSourceMotorDecision:
        physical_cell_steps = sum(item.physical_cell_steps for item in bindings)
        physical_cell_steps += sum(item.physical_cell_steps for item in inventory_races)
        if action is not None:
            physical_cell_steps += action.physical_cell_steps
        checkpoint_sha256 = self.checkpoint_sha256
        record = {
            "action_sha256": None if action is None else action.sha256,
            "binding_sha256s": tuple(item.sha256 for item in bindings),
            "checkpoint_sha256": checkpoint_sha256,
            "decision": decision,
            "emissions_hex": tuple(event.hex() for event in emissions),
            "inventory_race_sha256s": tuple(item.sha256 for item in inventory_races),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "stop_reason": stop_reason,
            "stop_stage": stop_stage,
        }
        return VariabilityMultiscaleSourceMotorDecision(
            decision=decision,
            emissions=emissions,
            stop_stage=stop_stage,
            stop_reason=stop_reason,
            bindings=bindings,
            inventory_races=inventory_races,
            action=action,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )


__all__ = [
    "SuccessfulVariabilityMultiscaleSourceMotorExperience",
    "VariabilityInventoryRaceTrace",
    "VariabilityMultiscaleSourceMotorAblation",
    "VariabilityMultiscaleSourceMotorController",
    "VariabilityMultiscaleSourceMotorDecision",
    "VariabilityMultiscaleSourceMotorTrainingSummary",
]
