"""Compose boundary-aware opaque grounding with a neuronal motor tape.

The inference boundary contains only one complete query byte stream, one
complete tuple of opaque workspace files, and an optional causal ablation.
Successful before/after transitions are visible only while growing the frozen
binders and motor assembly.  No changed path, span, candidate, repaired source,
delimiter identity, or task ontology enters inference.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from humanbrain.boundary_aware_segment_binder import (
    BoundaryAwareSegmentBinder,
    BoundaryBinderAblation,
    BoundaryBindingDecision,
    BoundarySegmentDemonstration,
)
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.source_relative_action_tape import (
    ActionTapeConsequenceUpdate,
    ActionTapeDecision,
    OccurrencePreservingActionTape,
    TapeValence,
)
from humanbrain.source_relative_training_compiler import (
    compile_successful_token_tapes,
)

_CHECKPOINT_NAMESPACE = b"humanbrain.boundary-multiscale-source-motor.checkpoint.v1\x00"
_QUERY_NAMESPACE = b"humanbrain.boundary-multiscale-source-motor.query.v1\x00"
_DECISION_NAMESPACE = b"humanbrain.boundary-multiscale-source-motor.decision.v1\x00"
_CONSTRUCTION_TOKEN = object()


def _digest(namespace: bytes, record: object) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(namespace + payload).hexdigest()


class BoundaryMultiscaleSourceMotorAblation(StrEnum):
    NONE = "none"
    EQUALITY_OFF = "equality_off"
    PROPAGATION_OFF = "propagation_off"
    BOUNDARY_OFF = "boundary_off"


@dataclass(frozen=True, slots=True)
class SuccessfulBoundaryMultiscaleSourceMotorExperience:
    """One public successful query/workspace transition used during growth."""

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
class BoundaryMultiscaleSourceMotorTrainingSummary:
    experience_count: int
    role_binder_count: int
    boundary_synapse_count: int
    action_program_cells: int
    action_recurrent_state_cells: int
    action_learned_synapses: int
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class BoundaryMultiscaleSourceMotorDecision:
    decision: str
    emissions: tuple[bytes, ...]
    stop_stage: str | None
    stop_reason: str | None
    bindings: tuple[BoundaryBindingDecision, ...]
    action: ActionTapeDecision | None
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


class BoundaryMultiscaleSourceMotorController:
    """Ground boundary-relative live roles, then run one neuronal motor chain."""

    __slots__ = ("_action_tape", "_binders", "_summary")

    def __init__(
        self,
        token: object,
        *,
        binders: tuple[BoundaryAwareSegmentBinder, ...],
        action_tape: OccurrencePreservingActionTape,
        summary: BoundaryMultiscaleSourceMotorTrainingSummary,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use BoundaryMultiscaleSourceMotorController.grow")
        self._binders = binders
        self._action_tape = action_tape
        self._summary = summary

    @classmethod
    def grow(
        cls,
        experiences: Iterable[SuccessfulBoundaryMultiscaleSourceMotorExperience],
    ) -> BoundaryMultiscaleSourceMotorController:
        """Grow boundary binders and a motor assembly from public successes."""

        rows = tuple(experiences)
        if len(rows) < 2:
            raise ValueError("at least two successful experiences are required")
        if any(
            not isinstance(row, SuccessfulBoundaryMultiscaleSourceMotorExperience) for row in rows
        ):
            raise TypeError(
                "experiences must contain SuccessfulBoundaryMultiscaleSourceMotorExperience"
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
                raise ValueError("successful experience must retain every workspace path")
            changed = tuple(
                (before_item, after_item)
                for before_item, after_item in zip(before_files, after_files, strict=True)
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
                raise ValueError("successful experience produced no grounded motor role")
            observations.append(before)
            motor_demonstrations.append(demonstration)

        cue_count = len(motor_demonstrations[0].cue_inventory)
        if any(len(row.cue_inventory) != cue_count for row in motor_demonstrations):
            raise ValueError("successful experiences do not share one motor-role shape")

        binders = tuple(
            BoundaryAwareSegmentBinder.grow(
                BoundarySegmentDemonstration(
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
        action_tape = OccurrencePreservingActionTape.grow(motor_demonstrations)
        checkpoint = _digest(
            _CHECKPOINT_NAMESPACE,
            {
                "action_checkpoint_sha256": action_tape.checkpoint_sha256,
                "binder_checkpoint_sha256s": tuple(binder.checkpoint_sha256 for binder in binders),
            },
        )
        action_summary = action_tape.training_summary
        summary = BoundaryMultiscaleSourceMotorTrainingSummary(
            experience_count=len(rows),
            role_binder_count=len(binders),
            boundary_synapse_count=sum(
                binder.training_summary.boundary_synapse_count for binder in binders
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
    def training_summary(self) -> BoundaryMultiscaleSourceMotorTrainingSummary:
        return self._summary

    def respond(
        self,
        query: bytes,
        files: tuple[OpaqueWorkspaceFile, ...],
        *,
        ablation: BoundaryMultiscaleSourceMotorAblation = (
            BoundaryMultiscaleSourceMotorAblation.NONE
        ),
    ) -> BoundaryMultiscaleSourceMotorDecision:
        """Process one complete public query/workspace observation exactly once."""

        if not isinstance(ablation, BoundaryMultiscaleSourceMotorAblation):
            raise TypeError("ablation must be a BoundaryMultiscaleSourceMotorAblation")
        observation = MultiscaleOpaqueEventTape.observe(query, files)
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "query_events_hex": tuple(event.hex() for event in observation.query_events),
                "source_events_hex": tuple(event.hex() for event in observation.source_events),
            },
        )
        binder_ablation = {
            BoundaryMultiscaleSourceMotorAblation.NONE: BoundaryBinderAblation.NONE,
            BoundaryMultiscaleSourceMotorAblation.EQUALITY_OFF: (
                BoundaryBinderAblation.EQUALITY_OFF
            ),
            BoundaryMultiscaleSourceMotorAblation.PROPAGATION_OFF: (
                BoundaryBinderAblation.PROPAGATION_OFF
            ),
            BoundaryMultiscaleSourceMotorAblation.BOUNDARY_OFF: (
                BoundaryBinderAblation.BOUNDARY_OFF
            ),
        }[ablation]
        bindings: list[BoundaryBindingDecision] = []
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
                    action=None,
                    query_sha256=query_sha256,
                )
            roles.append(binding.role_identity)

        if len(set(roles)) != len(roles):
            return self._decision(
                decision="STOP",
                emissions=(),
                stop_stage="role_inventory",
                stop_reason="role_inventory:identity_collision",
                bindings=tuple(bindings),
                action=None,
                query_sha256=query_sha256,
            )
        live_motor_tokens = observation.motor_source_tokens
        if any(live_motor_tokens.count(role) != 1 for role in roles):
            return self._decision(
                decision="STOP",
                emissions=(),
                stop_stage="role_inventory",
                stop_reason="role_inventory:non_unique_live_motor_role",
                bindings=tuple(bindings),
                action=None,
                query_sha256=query_sha256,
            )

        action = self._action_tape.respond(
            observation.query_events,
            tuple(roles),
            live_motor_tokens,
        )
        if action.decision != "EMIT":
            return self._decision(
                decision="STOP",
                emissions=(),
                stop_stage="action_tape",
                stop_reason=action.stop_reason,
                bindings=tuple(bindings),
                action=action,
                query_sha256=query_sha256,
            )
        return self._decision(
            decision="EMIT",
            emissions=action.emissions,
            stop_stage=None,
            stop_reason=None,
            bindings=tuple(bindings),
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
        bindings: tuple[BoundaryBindingDecision, ...],
        action: ActionTapeDecision | None,
        query_sha256: str,
    ) -> BoundaryMultiscaleSourceMotorDecision:
        physical_cell_steps = sum(item.physical_cell_steps for item in bindings)
        if action is not None:
            physical_cell_steps += action.physical_cell_steps
        checkpoint_sha256 = self.checkpoint_sha256
        record = {
            "action_sha256": None if action is None else action.sha256,
            "binding_sha256s": tuple(item.sha256 for item in bindings),
            "checkpoint_sha256": checkpoint_sha256,
            "decision": decision,
            "emissions_hex": tuple(event.hex() for event in emissions),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "stop_reason": stop_reason,
            "stop_stage": stop_stage,
        }
        return BoundaryMultiscaleSourceMotorDecision(
            decision=decision,
            emissions=emissions,
            stop_stage=stop_stage,
            stop_reason=stop_reason,
            bindings=bindings,
            action=action,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )


__all__ = [
    "BoundaryMultiscaleSourceMotorAblation",
    "BoundaryMultiscaleSourceMotorController",
    "BoundaryMultiscaleSourceMotorDecision",
    "BoundaryMultiscaleSourceMotorTrainingSummary",
    "SuccessfulBoundaryMultiscaleSourceMotorExperience",
]
