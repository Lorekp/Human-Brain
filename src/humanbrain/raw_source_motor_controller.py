"""End-to-end neuronal motor control from a complete opaque source tape.

Training may inspect successful context/before/after token experiences.  A
training-only compiler converts each into generic motor activity, and those
same compiled cues train the raw-tape role binder.  The retained controller
contains only the binder, motor tape, and authenticated training custody.

Public inference accepts only context events and the complete raw source tape.
The binder neuronally selects ordered cue identities, then the occurrence tape
neuronally selects and executes REBIND/COPY/ADVANCE/EMIT/END.  No cue inventory,
motor trace, after tape, alignment, target span, candidate, or source position
can be supplied at inference.

Exact context events and compiler-produced training trajectories remain
micro-gate scaffolding.  This module is not autonomous context discovery or
program induction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256

from humanbrain.raw_tape_role_binder import (
    NeuronalRawTapeRoleBinder,
    RawTapeRoleDemonstration,
    RoleBinderAblation,
    RoleBinderConfig,
    RoleBinderTrainingSummary,
    RoleBindingDecision,
)
from humanbrain.source_relative_action_tape import (
    ActionTapeAblation,
    ActionTapeConfig,
    ActionTapeConsequenceUpdate,
    ActionTapeDecision,
    ActionTapeTrainingSummary,
    OccurrencePreservingActionTape,
    TapeValence,
)
from humanbrain.source_relative_training_compiler import compile_successful_token_tapes

__all__ = [
    "RawSourceMotorAblation",
    "RawSourceMotorConfig",
    "RawSourceMotorConsequenceUpdate",
    "RawSourceMotorController",
    "RawSourceMotorDecision",
    "RawSourceMotorPendingError",
    "RawSourceMotorReceipt",
    "RawSourceMotorTrainingSummary",
    "SuccessfulSourceMotorExperience",
]


_TRAINING_NAMESPACE = b"humanbrain.raw-source-motor-controller.training-v1\0"
_CHECKPOINT_NAMESPACE = b"humanbrain.raw-source-motor-controller.checkpoint-v1\0"
_QUERY_NAMESPACE = b"humanbrain.raw-source-motor-controller.query-v1\0"
_RECEIPT_NAMESPACE = b"humanbrain.raw-source-motor-controller.receipt-v1\0"
_DECISION_NAMESPACE = b"humanbrain.raw-source-motor-controller.decision-v1\0"
_UPDATE_NAMESPACE = b"humanbrain.raw-source-motor-controller.update-v1\0"
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


def _require_event(value: object, name: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{name} must be non-empty bytes")
    return value


def _require_events(value: object, name: str, *, allow_empty: bool) -> tuple[bytes, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    events = value
    if not allow_empty and not events:
        raise ValueError(f"{name} must not be empty")
    for index, event in enumerate(events):
        _require_event(event, f"{name}[{index}]")
    return events


class RawSourceMotorAblation(StrEnum):
    NONE = "none"
    BINDER_OFF = "binder_off"
    TAPE_OFF = "tape_off"
    ASSEMBLY_OFF = "assembly_off"


class RawSourceMotorPendingError(RuntimeError):
    """Raised when a controller decision is not resolved exactly once."""


@dataclass(frozen=True, slots=True)
class SuccessfulSourceMotorExperience:
    """One public successful training experience; never an inference input."""

    context_event: bytes
    before_tokens: tuple[bytes, ...]
    after_tokens: tuple[bytes, ...]
    provenance_id: str

    def __post_init__(self) -> None:
        _require_event(self.context_event, "context_event")
        _require_events(self.before_tokens, "before_tokens", allow_empty=False)
        _require_events(self.after_tokens, "after_tokens", allow_empty=False)
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RawSourceMotorConfig:
    binder: RoleBinderConfig = field(default_factory=RoleBinderConfig)
    tape: ActionTapeConfig = field(default_factory=ActionTapeConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.binder, RoleBinderConfig):
            raise TypeError("binder must be a RoleBinderConfig")
        if not isinstance(self.tape, ActionTapeConfig):
            raise TypeError("tape must be an ActionTapeConfig")


@dataclass(frozen=True, slots=True)
class RawSourceMotorTrainingSummary:
    experience_count: int
    context_count: int
    training_custody_sha256: str
    binder: RoleBinderTrainingSummary
    tape: ActionTapeTrainingSummary


@dataclass(frozen=True, slots=True)
class RawSourceMotorReceipt:
    checkpoint_sha256: str
    training_custody_sha256: str
    query_sha256: str
    binder_decision_sha256: str
    tape_decision_sha256: str
    tape_receipt_sha256: str
    output_sha256: str
    physical_cell_steps: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RawSourceMotorDecision:
    decision: str
    emissions: tuple[bytes, ...]
    stop_stage: str | None
    stop_reason: str | None
    binder_decision: RoleBindingDecision
    tape_decision: ActionTapeDecision | None
    receipt: RawSourceMotorReceipt | None
    physical_cell_steps: int
    checkpoint_sha256: str
    query_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RawSourceMotorConsequenceUpdate:
    valence: TapeValence
    receipt_sha256: str
    tape_update: ActionTapeConsequenceUpdate
    prior_training_sha256: str
    training_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    receipt: RawSourceMotorReceipt


def _experience_record(experience: SuccessfulSourceMotorExperience) -> dict[str, object]:
    return {
        "after_tokens_hex": tuple(token.hex() for token in experience.after_tokens),
        "before_tokens_hex": tuple(token.hex() for token in experience.before_tokens),
        "context_event_hex": experience.context_event.hex(),
        "provenance_id": experience.provenance_id,
    }


class RawSourceMotorController:
    """Compose learned raw-tape role binding with learned motor execution."""

    __slots__ = (
        "_binder",
        "_checkpoint_sha256",
        "_config",
        "_pending",
        "_summary",
        "_tape",
        "_training_custody_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        binder: NeuronalRawTapeRoleBinder,
        tape: OccurrencePreservingActionTape,
        config: RawSourceMotorConfig,
        summary: RawSourceMotorTrainingSummary,
        training_custody_sha256: str,
        checkpoint_sha256: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use RawSourceMotorController.grow")
        self._binder = binder
        self._tape = tape
        self._config = config
        self._summary = summary
        self._training_custody_sha256 = training_custody_sha256
        self._checkpoint_sha256 = checkpoint_sha256
        self._pending: _PendingDecision | None = None

    @classmethod
    def grow(
        cls,
        experiences: Iterable[SuccessfulSourceMotorExperience],
        config: RawSourceMotorConfig | None = None,
    ) -> RawSourceMotorController:
        """Compile successful experiences and grow both neuronal components."""

        rows = tuple(experiences)
        if not rows:
            raise ValueError("experiences must not be empty")
        if any(not isinstance(row, SuccessfulSourceMotorExperience) for row in rows):
            raise TypeError("experiences must contain SuccessfulSourceMotorExperience")
        provenance_ids = tuple(row.provenance_id for row in rows)
        if len(set(provenance_ids)) != len(provenance_ids):
            raise ValueError("experience provenance identifiers must be unique")
        settings = config or RawSourceMotorConfig()
        if not isinstance(settings, RawSourceMotorConfig):
            raise TypeError("config must be a RawSourceMotorConfig")

        motor_demonstrations = tuple(
            compile_successful_token_tapes(
                context_event=row.context_event,
                before_tokens=row.before_tokens,
                after_tokens=row.after_tokens,
            )
            for row in rows
        )
        if any(not demonstration.cue_inventory for demonstration in motor_demonstrations):
            raise ValueError("successful controller experiences must compile at least one cue")
        binder_demonstrations = tuple(
            RawTapeRoleDemonstration(
                context_event=demonstration.context_event,
                source_tokens=demonstration.source_tokens,
                demonstrated_roles=demonstration.cue_inventory,
            )
            for demonstration in motor_demonstrations
        )
        binder = NeuronalRawTapeRoleBinder.grow(
            binder_demonstrations,
            settings.binder,
        )
        tape = OccurrencePreservingActionTape.grow(
            motor_demonstrations,
            settings.tape,
        )
        custody = _digest(
            _TRAINING_NAMESPACE,
            {
                "compiled": tuple(
                    {
                        "cue_inventory_hex": tuple(
                            cue.hex() for cue in demonstration.cue_inventory
                        ),
                        "motor_steps": tuple(step.record() for step in demonstration.motor_steps),
                    }
                    for demonstration in motor_demonstrations
                ),
                "experiences": tuple(_experience_record(row) for row in rows),
            },
        )
        checkpoint = _digest(
            _CHECKPOINT_NAMESPACE,
            {
                "binder_checkpoint_sha256": binder.checkpoint_sha256,
                "config": asdict(settings),
                "tape_checkpoint_sha256": tape.checkpoint_sha256,
                "training_custody_sha256": custody,
            },
        )
        summary = RawSourceMotorTrainingSummary(
            experience_count=len(rows),
            context_count=len({row.context_event for row in rows}),
            training_custody_sha256=custody,
            binder=binder.training_summary,
            tape=tape.training_summary,
        )
        return cls(
            _CONSTRUCTION_TOKEN,
            binder=binder,
            tape=tape,
            config=settings,
            summary=summary,
            training_custody_sha256=custody,
            checkpoint_sha256=checkpoint,
        )

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def training_summary(self) -> RawSourceMotorTrainingSummary:
        return self._summary

    def respond(
        self,
        context_events: tuple[bytes, ...],
        source_tokens: tuple[bytes, ...],
        *,
        ablation: RawSourceMotorAblation = RawSourceMotorAblation.NONE,
    ) -> RawSourceMotorDecision:
        """Bind and execute from raw source only; an EMIT remains pending."""

        if self._pending is not None:
            raise RawSourceMotorPendingError("resolve the pending EMIT decision before responding")
        contexts = _require_events(context_events, "context_events", allow_empty=False)
        source = _require_events(source_tokens, "source_tokens", allow_empty=False)
        if not isinstance(ablation, RawSourceMotorAblation):
            raise TypeError("ablation must be a RawSourceMotorAblation")
        query_sha256 = _digest(
            _QUERY_NAMESPACE,
            {
                "ablation": ablation.value,
                "checkpoint_sha256": self._checkpoint_sha256,
                "context_events_hex": tuple(event.hex() for event in contexts),
                "source_tokens_hex": tuple(token.hex() for token in source),
            },
        )

        binder_ablation = (
            RoleBinderAblation.BINDER_OFF
            if ablation is RawSourceMotorAblation.BINDER_OFF
            else RoleBinderAblation.NONE
        )
        binding = self._binder.respond(
            contexts,
            source,
            ablation=binder_ablation,
        )
        if binding.decision == "STOP":
            return self._stop(
                query_sha256=query_sha256,
                stage="binder",
                reason=binding.stop_reason,
                binding=binding,
                tape_decision=None,
            )
        if binding.decision != "BIND" or not binding.cue_inventory:
            raise RuntimeError("raw role binder returned an invalid successful decision")

        if ablation is RawSourceMotorAblation.TAPE_OFF:
            tape_ablation = ActionTapeAblation.RELATIONAL_TAPE_OFF
        elif ablation is RawSourceMotorAblation.ASSEMBLY_OFF:
            tape_ablation = ActionTapeAblation.ASSEMBLY_OFF
        else:
            tape_ablation = ActionTapeAblation.NONE
        tape_decision = self._tape.respond(
            contexts,
            binding.cue_inventory,
            source,
            ablation=tape_ablation,
        )
        physical_cell_steps = binding.physical_cell_steps + tape_decision.physical_cell_steps
        if tape_decision.decision == "STOP":
            return self._stop(
                query_sha256=query_sha256,
                stage="tape",
                reason=tape_decision.stop_reason,
                binding=binding,
                tape_decision=tape_decision,
            )
        tape_receipt = tape_decision.eligibility
        if tape_decision.decision != "EMIT" or tape_receipt is None:
            raise RuntimeError("motor tape returned an invalid successful decision")

        output_sha256 = _digest(
            _DECISION_NAMESPACE,
            {"emissions_hex": tuple(event.hex() for event in tape_decision.emissions)},
        )
        receipt_record = {
            "binder_decision_sha256": binding.sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
            "output_sha256": output_sha256,
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "tape_decision_sha256": tape_decision.sha256,
            "tape_receipt_sha256": tape_receipt.sha256,
            "training_custody_sha256": self._training_custody_sha256,
        }
        receipt = RawSourceMotorReceipt(
            checkpoint_sha256=self._checkpoint_sha256,
            training_custody_sha256=self._training_custody_sha256,
            query_sha256=query_sha256,
            binder_decision_sha256=binding.sha256,
            tape_decision_sha256=tape_decision.sha256,
            tape_receipt_sha256=tape_receipt.sha256,
            output_sha256=output_sha256,
            physical_cell_steps=physical_cell_steps,
            sha256=_digest(_RECEIPT_NAMESPACE, receipt_record),
        )
        result = self._decision(
            decision="EMIT",
            emissions=tape_decision.emissions,
            stop_stage=None,
            stop_reason=None,
            binding=binding,
            tape_decision=tape_decision,
            receipt=receipt,
            physical_cell_steps=physical_cell_steps,
            query_sha256=query_sha256,
        )
        self._pending = _PendingDecision(receipt)
        return result

    def resolve(self, valence: TapeValence) -> RawSourceMotorConsequenceUpdate:
        """Resolve one exact pending motor decision and propagate its consequence."""

        if not isinstance(valence, TapeValence):
            raise TypeError("valence must be a TapeValence")
        pending = self._pending
        if pending is None:
            raise RawSourceMotorPendingError("no pending EMIT decision to resolve")
        prior = self._checkpoint_sha256
        tape_update = self._tape.resolve(valence)
        if tape_update.receipt_sha256 != pending.receipt.tape_receipt_sha256:
            raise RuntimeError("motor tape consequence receipt custody changed")
        self._pending = None
        self._checkpoint_sha256 = _digest(
            _CHECKPOINT_NAMESPACE,
            {
                "binder_checkpoint_sha256": self._binder.checkpoint_sha256,
                "config": asdict(self._config),
                "tape_checkpoint_sha256": self._tape.checkpoint_sha256,
                "training_custody_sha256": self._training_custody_sha256,
            },
        )
        record = {
            "prior_training_sha256": prior,
            "receipt_sha256": pending.receipt.sha256,
            "tape_update_sha256": tape_update.sha256,
            "training_sha256": self._checkpoint_sha256,
            "valence": valence.value,
        }
        return RawSourceMotorConsequenceUpdate(
            valence=valence,
            receipt_sha256=pending.receipt.sha256,
            tape_update=tape_update,
            prior_training_sha256=prior,
            training_sha256=self._checkpoint_sha256,
            sha256=_digest(_UPDATE_NAMESPACE, record),
        )

    def _stop(
        self,
        *,
        query_sha256: str,
        stage: str,
        reason: str | None,
        binding: RoleBindingDecision,
        tape_decision: ActionTapeDecision | None,
    ) -> RawSourceMotorDecision:
        if reason is None:
            raise RuntimeError("physical STOP lost its component reason")
        physical_cell_steps = binding.physical_cell_steps + (
            0 if tape_decision is None else tape_decision.physical_cell_steps
        )
        return self._decision(
            decision="STOP",
            emissions=(),
            stop_stage=stage,
            stop_reason=f"{stage}:{reason}",
            binding=binding,
            tape_decision=tape_decision,
            receipt=None,
            physical_cell_steps=physical_cell_steps,
            query_sha256=query_sha256,
        )

    def _decision(
        self,
        *,
        decision: str,
        emissions: tuple[bytes, ...],
        stop_stage: str | None,
        stop_reason: str | None,
        binding: RoleBindingDecision,
        tape_decision: ActionTapeDecision | None,
        receipt: RawSourceMotorReceipt | None,
        physical_cell_steps: int,
        query_sha256: str,
    ) -> RawSourceMotorDecision:
        record = {
            "binder_decision_sha256": binding.sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
            "decision": decision,
            "emissions_hex": tuple(event.hex() for event in emissions),
            "physical_cell_steps": physical_cell_steps,
            "query_sha256": query_sha256,
            "receipt_sha256": None if receipt is None else receipt.sha256,
            "stop_reason": stop_reason,
            "stop_stage": stop_stage,
            "tape_decision_sha256": (None if tape_decision is None else tape_decision.sha256),
        }
        return RawSourceMotorDecision(
            decision=decision,
            emissions=emissions,
            stop_stage=stop_stage,
            stop_reason=stop_reason,
            binder_decision=binding,
            tape_decision=tape_decision,
            receipt=receipt,
            physical_cell_steps=physical_cell_steps,
            checkpoint_sha256=self._checkpoint_sha256,
            query_sha256=query_sha256,
            sha256=_digest(_DECISION_NAMESPACE, record),
        )
