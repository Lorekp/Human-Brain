"""Replay-free continual outcome memory for an opaque fixed action catalogue.

The module owns a small physical action race.  Callers provide only rooted action
identities, a rooted selection context, and nonnegative base activation.
Caller-attested PASS/FAIL outcomes add excitatory/inhibitory ticks to the
selected action.  The result is deliberately domain-neutral: proposed sources,
executions, and consequences remain opaque SHA-256 identities, so an outer
adapter must authenticate what the outcome means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from types import MappingProxyType

import numpy as np
from experiments.g0 import common_event_cells_v1 as common_events

SCHEMA = "COMMON_CONTINUAL_ACTION_OUTCOME_MEMORY_V1"
CLAIM_BOUNDARY = (
    "replay-free-caller-attested-binary-outcome-revision-of-a-fixed-opaque-action-catalogue-"
    "through-a-physical-neuronal-race-not-action-discovery-source-synthesis-"
    "public-outcome-authentication-semantics-debugging-execution-swe-or-intelligence"
)

READY = "READY"
EXPLORATION_READY = "EXPLORATION_READY"
STOP_CELL_OFF = "STOP_CELL_OFF"
STOP_NO_CANDIDATE = "STOP_NO_CANDIDATE"
STOP_PHYSICAL_RACE = "STOP_PHYSICAL_RACE"
STOP_INSUFFICIENT_OUTCOME_EVIDENCE = "STOP_INSUFFICIENT_OUTCOME_EVIDENCE"

MAX_ACTIONS = common_events.OUTPUT_TOKEN_COUNT
MAX_AUXILIARY_EVENT_ROOTS = 16
MAX_PROVENANCE_ROWS = 1_024
MAX_OUTPUT_SCORE = common_events.MAX_OUTPUT_SCORE - common_events.STOP_SCORE
TONIC_SCORE = 2
TICK_CAP = 255
TICK_DTYPE = np.dtype(np.uint8)

_CELLS = common_events.CommonEventCells()
_CONSTRUCTION_TOKEN = object()
_SHA256_HEX = frozenset("0123456789abcdef")

_ACTIVATION_NAMESPACE = b"common-continual-action-outcome-activation-v1"
_CATALOGUE_NAMESPACE = b"common-continual-action-outcome-catalogue-v1"
_DECISION_NAMESPACE = b"common-continual-action-outcome-decision-v1"
_OUTCOME_NAMESPACE = b"common-continual-action-outcome-selected-outcome-v1"
_PROVENANCE_NAMESPACE = b"common-continual-action-outcome-provenance-v1"
_CHECKPOINT_NAMESPACE = b"common-continual-action-outcome-checkpoint-v1"


class ContinualActionOutcomeMemoryError(ValueError):
    """An activation, decision, outcome, or checkpoint failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _validate_roots(
    values: object,
    *,
    label: str,
    nonempty: bool,
    canonical: bool,
) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be one exact tuple")
    roots = values
    if (
        (nonempty and not roots)
        or len(set(roots)) != len(roots)
        or any(not _valid_sha256(value) for value in roots)
        or (canonical and roots != tuple(sorted(roots)))
    ):
        raise ContinualActionOutcomeMemoryError(f"{label} is not canonical")
    return roots


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    frame = (
        array.dtype.str.encode("ascii")
        + b"\0"
        + _canonical_json(array.shape)
        + b"\0"
        + array.tobytes(order="C")
    )
    return sha256(frame).hexdigest()


def _freeze_ticks(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=TICK_DTYPE).copy()
    result.setflags(write=False)
    return result


def _catalogue_sha256(action_sha256s: tuple[str, ...]) -> str:
    return _digest(_CATALOGUE_NAMESPACE, action_sha256s)


def _provenance_event_roots(
    row: ActionOutcomeProvenanceV1,
) -> tuple[str, ...]:
    return (
        row.sha256,
        row.outcome_sha256,
        row.decision_sha256,
        row.activation_sha256,
        row.selection_context_sha256,
        row.proposed_source_sha256,
        row.execution_sha256,
        row.consequence_sha256,
        *row.auxiliary_event_sha256s,
    )


def _physical_output_record(raw_scores: tuple[tuple[int, ...], ...]) -> dict[str, object]:
    return dict(_CELLS.race_output(raw_scores).record())


def _logical_to_physical(score: int, *, exploration_gate: bool) -> int:
    if exploration_gate:
        shifted = common_events.STOP_SCORE + score
    else:
        shifted = common_events.STOP_SCORE + score - TONIC_SCORE
    return max(
        common_events.MIN_OUTPUT_SCORE,
        min(common_events.MAX_OUTPUT_SCORE, shifted),
    )


def _physical_scores(
    *,
    action_indices: tuple[int, ...],
    effective_scores: tuple[int, ...],
    cell_gate: bool,
    exploration_gate: bool,
) -> tuple[tuple[int, ...], ...]:
    raw = np.zeros(
        (common_events.OUTPUT_DIGIT_LANES, common_events.OUTPUT_DIGIT_RADIX),
        dtype=np.int64,
    )
    if not cell_gate:
        return tuple(tuple(int(value) for value in row) for row in raw)
    for action_index, effective_score in zip(action_indices, effective_scores, strict=True):
        physical_score = _logical_to_physical(
            effective_score,
            exploration_gate=exploration_gate,
        )
        remaining = action_index
        reversed_digits: list[int] = []
        for _ in range(common_events.OUTPUT_DIGIT_LANES):
            reversed_digits.append(remaining % common_events.OUTPUT_DIGIT_RADIX)
            remaining //= common_events.OUTPUT_DIGIT_RADIX
        if remaining:
            raise ContinualActionOutcomeMemoryError(
                "action index exceeds the physical output token space"
            )
        digits = tuple(reversed(reversed_digits))
        for lane, digit in enumerate(digits):
            raw[lane, digit] = max(int(raw[lane, digit]), physical_score)
    return tuple(tuple(int(value) for value in row) for row in raw)


def _decision_result(
    *,
    active_action_sha256s: tuple[str, ...],
    action_indices: tuple[int, ...],
    effective_scores: tuple[int, ...],
    cell_gate: bool,
    exploration_gate: bool,
    physical_output_record: dict[str, object],
) -> tuple[str, str | None, int | None]:
    if not cell_gate:
        return STOP_CELL_OFF, None, None
    if not action_indices:
        return STOP_NO_CANDIDATE, None, None
    threshold = 0 if exploration_gate else TONIC_SCORE
    maximum = max(effective_scores)
    if maximum <= threshold:
        return STOP_INSUFFICIENT_OUTCOME_EVIDENCE, None, None
    if effective_scores.count(maximum) > 1:
        return STOP_PHYSICAL_RACE, None, None
    winner_position = effective_scores.index(maximum)
    expected_index = action_indices[winner_position]
    if (
        physical_output_record["decision"] != "TOKEN"
        or physical_output_record["token"] != expected_index
    ):
        return STOP_PHYSICAL_RACE, None, None
    return (
        EXPLORATION_READY if exploration_gate else READY,
        active_action_sha256s[winner_position],
        expected_index,
    )


@dataclass(frozen=True, slots=True)
class ActionActivationV1:
    """One rooted selection context with a frozen set of base action scores."""

    selection_context_sha256: str
    active_action_sha256s: tuple[str, ...]
    base_scores: tuple[int, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "selection_context_sha256": self.selection_context_sha256,
            "active_action_sha256s": self.active_action_sha256s,
            "base_scores": self.base_scores,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        _validate_roots(
            self.active_action_sha256s,
            label="active action roots",
            nonempty=False,
            canonical=True,
        )
        if (
            not _valid_sha256(self.selection_context_sha256)
            or type(self.base_scores) is not tuple
            or len(self.base_scores) != len(self.active_action_sha256s)
            or any(
                type(score) is not int or not 0 <= score <= MAX_OUTPUT_SCORE
                for score in self.base_scores
            )
            or self.sha256 != _digest(_ACTIVATION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualActionOutcomeMemoryError("action activation authentication failed")


def seal_action_activation_v1(
    *,
    selection_context_sha256: str,
    active_action_sha256s: tuple[str, ...],
    base_scores: tuple[int, ...],
) -> ActionActivationV1:
    if type(active_action_sha256s) is not tuple or type(base_scores) is not tuple:
        raise TypeError("activation actions and scores must be exact tuples")
    if len(active_action_sha256s) != len(base_scores):
        raise ContinualActionOutcomeMemoryError("activation actions and scores differ")
    if len(set(active_action_sha256s)) != len(active_action_sha256s):
        raise ContinualActionOutcomeMemoryError("activation action replay is forbidden")
    if any(not _valid_sha256(value) for value in active_action_sha256s):
        raise ContinualActionOutcomeMemoryError("activation action root is invalid")
    if any(type(score) is not int or not 0 <= score <= MAX_OUTPUT_SCORE for score in base_scores):
        raise ContinualActionOutcomeMemoryError("activation base score is invalid")
    ordered = tuple(sorted(zip(active_action_sha256s, base_scores, strict=True)))
    payload = {
        "selection_context_sha256": selection_context_sha256,
        "active_action_sha256s": tuple(item[0] for item in ordered),
        "base_scores": tuple(item[1] for item in ordered),
    }
    result = ActionActivationV1(
        **payload,
        sha256=_digest(_ACTIVATION_NAMESPACE, payload),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class ActionDecisionV1:
    status: str
    checkpoint_sha256: str
    catalogue_sha256: str
    activation: ActionActivationV1
    action_indices: tuple[int, ...]
    excitatory_ticks: tuple[int, ...]
    inhibitory_ticks: tuple[int, ...]
    effective_scores: tuple[int, ...]
    tonic_score: int
    selected_action_sha256: str | None
    selected_action_index: int | None
    cell_gate: bool
    outcome_gate: bool
    exploration_gate: bool
    physical_raw_scores: tuple[tuple[int, ...], ...]
    physical_output_record: MappingProxyType
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "status": self.status,
            "checkpoint_sha256": self.checkpoint_sha256,
            "catalogue_sha256": self.catalogue_sha256,
            "activation": self.activation.record(),
            "action_indices": self.action_indices,
            "excitatory_ticks": self.excitatory_ticks,
            "inhibitory_ticks": self.inhibitory_ticks,
            "effective_scores": self.effective_scores,
            "tonic_score": self.tonic_score,
            "selected_action_sha256": self.selected_action_sha256,
            "selected_action_index": self.selected_action_index,
            "cell_gate": self.cell_gate,
            "outcome_gate": self.outcome_gate,
            "exploration_gate": self.exploration_gate,
            "physical_raw_scores": self.physical_raw_scores,
            "physical_output_record": dict(self.physical_output_record),
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.activation) is not ActionActivationV1:
            raise TypeError("decision activation must be exact")
        self.activation.authenticate()
        count = len(self.activation.active_action_sha256s)
        if (
            self.status
            not in (
                READY,
                EXPLORATION_READY,
                STOP_CELL_OFF,
                STOP_NO_CANDIDATE,
                STOP_PHYSICAL_RACE,
                STOP_INSUFFICIENT_OUTCOME_EVIDENCE,
            )
            or not _valid_sha256(self.checkpoint_sha256)
            or not _valid_sha256(self.catalogue_sha256)
            or type(self.action_indices) is not tuple
            or len(self.action_indices) != count
            or len(set(self.action_indices)) != len(self.action_indices)
            or any(
                type(index) is not int or not 0 <= index < MAX_ACTIONS
                for index in self.action_indices
            )
            or any(
                type(values) is not tuple
                or len(values) != count
                or any(type(value) is not int or not 0 <= value <= TICK_CAP for value in values)
                for values in (self.excitatory_ticks, self.inhibitory_ticks)
            )
            or type(self.effective_scores) is not tuple
            or len(self.effective_scores) != count
            or any(
                type(score) is not int or not 0 <= score <= MAX_OUTPUT_SCORE
                for score in self.effective_scores
            )
            or type(self.cell_gate) is not bool
            or type(self.outcome_gate) is not bool
            or type(self.exploration_gate) is not bool
            or self.tonic_score != (0 if self.exploration_gate else TONIC_SCORE)
            or type(self.physical_output_record) is not MappingProxyType
        ):
            raise ContinualActionOutcomeMemoryError("action decision authentication failed")
        expected_excitatory = self.excitatory_ticks if self.outcome_gate else (0,) * count
        expected_inhibitory = self.inhibitory_ticks if self.outcome_gate else (0,) * count
        expected_effective = tuple(
            max(
                0,
                min(
                    MAX_OUTPUT_SCORE,
                    base + 2 * (positive - negative),
                ),
            )
            for base, positive, negative in zip(
                self.activation.base_scores,
                expected_excitatory,
                expected_inhibitory,
                strict=True,
            )
        )
        expected_raw = _physical_scores(
            action_indices=self.action_indices,
            effective_scores=expected_effective,
            cell_gate=self.cell_gate,
            exploration_gate=self.exploration_gate,
        )
        expected_output = _physical_output_record(expected_raw)
        expected_status, expected_root, expected_index = _decision_result(
            active_action_sha256s=self.activation.active_action_sha256s,
            action_indices=self.action_indices,
            effective_scores=expected_effective,
            cell_gate=self.cell_gate,
            exploration_gate=self.exploration_gate,
            physical_output_record=expected_output,
        )
        if (
            (not self.outcome_gate and any(self.excitatory_ticks + self.inhibitory_ticks))
            or self.effective_scores != expected_effective
            or self.physical_raw_scores != expected_raw
            or dict(self.physical_output_record) != expected_output
            or self.status != expected_status
            or self.selected_action_sha256 != expected_root
            or self.selected_action_index != expected_index
            or self.sha256 != _digest(_DECISION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualActionOutcomeMemoryError("action decision authentication failed")


@dataclass(frozen=True, slots=True)
class SelectedActionOutcomeV1:
    decision: ActionDecisionV1
    selected_action_sha256: str
    proposed_source_sha256: str
    execution_sha256: str
    consequence_sha256: str
    auxiliary_event_sha256s: tuple[str, ...]
    passed_cell: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "decision": self.decision.record(),
            "selected_action_sha256": self.selected_action_sha256,
            "proposed_source_sha256": self.proposed_source_sha256,
            "execution_sha256": self.execution_sha256,
            "consequence_sha256": self.consequence_sha256,
            "auxiliary_event_sha256s": self.auxiliary_event_sha256s,
            "passed_cell": self.passed_cell,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.decision) is not ActionDecisionV1:
            raise TypeError("selected outcome decision must be exact")
        self.decision.authenticate()
        auxiliary_roots = _validate_roots(
            self.auxiliary_event_sha256s,
            label="selected outcome auxiliary event roots",
            nonempty=False,
            canonical=True,
        )
        rooted_fields = (
            self.selected_action_sha256,
            self.proposed_source_sha256,
            self.execution_sha256,
            self.consequence_sha256,
            *auxiliary_roots,
        )
        event_roots = (
            self.sha256,
            self.decision.sha256,
            self.decision.activation.sha256,
            self.decision.activation.selection_context_sha256,
            self.proposed_source_sha256,
            self.execution_sha256,
            self.consequence_sha256,
            *auxiliary_roots,
        )
        reserved_roots = (
            self.selected_action_sha256,
            self.decision.checkpoint_sha256,
            self.decision.catalogue_sha256,
        )
        if (
            self.decision.status not in (READY, EXPLORATION_READY)
            or self.selected_action_sha256 != self.decision.selected_action_sha256
            or any(not _valid_sha256(value) for value in rooted_fields)
            or len(set(rooted_fields)) != len(rooted_fields)
            or len(set(event_roots)) != len(event_roots)
            or set(event_roots).intersection(reserved_roots)
            or len(auxiliary_roots) > MAX_AUXILIARY_EVENT_ROOTS
            or type(self.passed_cell) is not int
            or self.passed_cell not in (0, 1)
            or self.sha256 != _digest(_OUTCOME_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualActionOutcomeMemoryError("selected action outcome authentication failed")


def seal_selected_action_outcome_v1(
    *,
    decision: ActionDecisionV1,
    proposed_source_sha256: str,
    execution_sha256: str,
    consequence_sha256: str,
    auxiliary_event_sha256s: tuple[str, ...],
    passed_cell: int,
) -> SelectedActionOutcomeV1:
    if type(decision) is not ActionDecisionV1:
        raise TypeError("selected outcome requires one exact action decision")
    decision.authenticate()
    if decision.selected_action_sha256 is None:
        raise ContinualActionOutcomeMemoryError("a STOP decision has no selected outcome")
    payload = {
        "decision": decision,
        "selected_action_sha256": decision.selected_action_sha256,
        "proposed_source_sha256": proposed_source_sha256,
        "execution_sha256": execution_sha256,
        "consequence_sha256": consequence_sha256,
        "auxiliary_event_sha256s": auxiliary_event_sha256s,
        "passed_cell": passed_cell,
    }
    result = SelectedActionOutcomeV1(
        **payload,
        sha256=_digest(
            _OUTCOME_NAMESPACE,
            {
                "decision": decision.record(),
                "selected_action_sha256": decision.selected_action_sha256,
                "proposed_source_sha256": proposed_source_sha256,
                "execution_sha256": execution_sha256,
                "consequence_sha256": consequence_sha256,
                "auxiliary_event_sha256s": auxiliary_event_sha256s,
                "passed_cell": passed_cell,
            },
        ),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class ActionOutcomeProvenanceV1:
    outcome_sha256: str
    decision_sha256: str
    parent_checkpoint_sha256: str
    activation_sha256: str
    selection_context_sha256: str
    action_sha256: str
    proposed_source_sha256: str
    execution_sha256: str
    consequence_sha256: str
    auxiliary_event_sha256s: tuple[str, ...]
    passed_cell: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "outcome_sha256": self.outcome_sha256,
            "decision_sha256": self.decision_sha256,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "activation_sha256": self.activation_sha256,
            "selection_context_sha256": self.selection_context_sha256,
            "action_sha256": self.action_sha256,
            "proposed_source_sha256": self.proposed_source_sha256,
            "execution_sha256": self.execution_sha256,
            "consequence_sha256": self.consequence_sha256,
            "auxiliary_event_sha256s": self.auxiliary_event_sha256s,
            "passed_cell": self.passed_cell,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        auxiliary_roots = _validate_roots(
            self.auxiliary_event_sha256s,
            label="action outcome provenance auxiliary event roots",
            nonempty=False,
            canonical=True,
        )
        outcome_roots = (
            self.action_sha256,
            self.proposed_source_sha256,
            self.execution_sha256,
            self.consequence_sha256,
            *auxiliary_roots,
        )
        event_roots = _provenance_event_roots(self)
        reserved_roots = (self.action_sha256, self.parent_checkpoint_sha256)
        if (
            any(
                not _valid_sha256(value)
                for value in (
                    self.outcome_sha256,
                    self.decision_sha256,
                    self.parent_checkpoint_sha256,
                    self.activation_sha256,
                    self.selection_context_sha256,
                    self.action_sha256,
                    self.proposed_source_sha256,
                    self.execution_sha256,
                    self.consequence_sha256,
                )
            )
            or len(set(outcome_roots)) != len(outcome_roots)
            or len(set(event_roots)) != len(event_roots)
            or set(event_roots).intersection(reserved_roots)
            or len(auxiliary_roots) > MAX_AUXILIARY_EVENT_ROOTS
            or type(self.passed_cell) is not int
            or self.passed_cell not in (0, 1)
            or self.sha256 != _digest(_PROVENANCE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise ContinualActionOutcomeMemoryError(
                "action outcome provenance authentication failed"
            )


def _provenance_from_outcome(outcome: SelectedActionOutcomeV1) -> ActionOutcomeProvenanceV1:
    payload = {
        "outcome_sha256": outcome.sha256,
        "decision_sha256": outcome.decision.sha256,
        "parent_checkpoint_sha256": outcome.decision.checkpoint_sha256,
        "activation_sha256": outcome.decision.activation.sha256,
        "selection_context_sha256": outcome.decision.activation.selection_context_sha256,
        "action_sha256": outcome.selected_action_sha256,
        "proposed_source_sha256": outcome.proposed_source_sha256,
        "execution_sha256": outcome.execution_sha256,
        "consequence_sha256": outcome.consequence_sha256,
        "auxiliary_event_sha256s": outcome.auxiliary_event_sha256s,
        "passed_cell": outcome.passed_cell,
    }
    result = ActionOutcomeProvenanceV1(
        **payload,
        sha256=_digest(_PROVENANCE_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _checkpoint_sha256(
    *,
    action_sha256s: tuple[str, ...],
    provenance: tuple[ActionOutcomeProvenanceV1, ...],
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> str:
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "action_sha256s": action_sha256s,
            "provenance": tuple(row.record() for row in provenance),
            "excitatory_sha256": _array_sha256(excitatory),
            "inhibitory_sha256": _array_sha256(inhibitory),
        },
    )


class FrozenContinualActionOutcomeMemoryV1:
    """One immutable fixed-catalogue checkpoint with replayed E/H outcome state."""

    __slots__ = (
        "_action_sha256s",
        "_checkpoint_sha256",
        "_excitatory",
        "_inhibitory",
        "_provenance",
    )

    def __init__(
        self,
        *,
        action_sha256s: tuple[str, ...],
        provenance: tuple[ActionOutcomeProvenanceV1, ...],
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
        checkpoint_sha256: str,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("continual action outcome memories are factory-owned")
        self._action_sha256s = action_sha256s
        self._provenance = provenance
        self._excitatory = _freeze_ticks(excitatory)
        self._inhibitory = _freeze_ticks(inhibitory)
        self._checkpoint_sha256 = checkpoint_sha256
        self.authenticate()

    @property
    def action_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        return self._action_sha256s

    @property
    def catalogue_sha256(self) -> str:
        self.authenticate()
        return _catalogue_sha256(self._action_sha256s)

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    @property
    def provenance(self) -> tuple[ActionOutcomeProvenanceV1, ...]:
        self.authenticate()
        return self._provenance

    @property
    def checkpoint_array_sha256s(self) -> tuple[str, str]:
        self.authenticate()
        return _array_sha256(self._excitatory), _array_sha256(self._inhibitory)

    @property
    def excitatory_ticks(self) -> tuple[int, ...]:
        self.authenticate()
        return tuple(int(value) for value in self._excitatory)

    @property
    def inhibitory_ticks(self) -> tuple[int, ...]:
        self.authenticate()
        return tuple(int(value) for value in self._inhibitory)

    def authenticate(self) -> None:
        _validate_roots(
            self._action_sha256s,
            label="checkpoint action catalogue",
            nonempty=True,
            canonical=True,
        )
        if not 1 <= len(self._action_sha256s) <= MAX_ACTIONS:
            raise ContinualActionOutcomeMemoryError("action catalogue capacity changed")
        if type(self._provenance) is not tuple or len(self._provenance) > MAX_PROVENANCE_ROWS:
            raise ContinualActionOutcomeMemoryError("checkpoint provenance changed")
        for row in self._provenance:
            if type(row) is not ActionOutcomeProvenanceV1:
                raise TypeError("checkpoint provenance rows must be exact")
            row.authenticate()
        if (
            len({row.outcome_sha256 for row in self._provenance}) != len(self._provenance)
            or len({row.decision_sha256 for row in self._provenance}) != len(self._provenance)
            or len({row.activation_sha256 for row in self._provenance}) != len(self._provenance)
            or len({row.selection_context_sha256 for row in self._provenance})
            != len(self._provenance)
            or len({row.proposed_source_sha256 for row in self._provenance})
            != len(self._provenance)
            or len({row.execution_sha256 for row in self._provenance}) != len(self._provenance)
            or len({row.consequence_sha256 for row in self._provenance}) != len(self._provenance)
            or any(row.action_sha256 not in self._action_sha256s for row in self._provenance)
        ):
            raise ContinualActionOutcomeMemoryError("checkpoint provenance replay or splice")
        action_rows = {value: index for index, value in enumerate(self._action_sha256s)}
        catalogue_root = _catalogue_sha256(self._action_sha256s)
        if (
            catalogue_root in action_rows
            or self._checkpoint_sha256 in action_rows
            or self._checkpoint_sha256 == catalogue_root
        ):
            raise ContinualActionOutcomeMemoryError(
                "checkpoint or catalogue root aliases fixed catalogue custody"
            )
        expected_excitatory = np.zeros(len(self._action_sha256s), dtype=TICK_DTYPE)
        expected_inhibitory = np.zeros_like(expected_excitatory)
        prefix: list[ActionOutcomeProvenanceV1] = []
        custody_roots: set[str] = set()
        prefix_checkpoint_roots: set[str] = set()
        for row in self._provenance:
            event_roots = _provenance_event_roots(row)
            expected_parent = _checkpoint_sha256(
                action_sha256s=self._action_sha256s,
                provenance=tuple(prefix),
                excitatory=expected_excitatory,
                inhibitory=expected_inhibitory,
            )
            reserved_roots = {
                *self._action_sha256s,
                catalogue_root,
                *prefix_checkpoint_roots,
                expected_parent,
            }
            if custody_roots.intersection(event_roots) or reserved_roots.intersection(event_roots):
                raise ContinualActionOutcomeMemoryError(
                    "checkpoint event-root custody replay or cross-role splice"
                )
            if row.parent_checkpoint_sha256 != expected_parent:
                raise ContinualActionOutcomeMemoryError(
                    "checkpoint provenance parent chain changed"
                )
            target = expected_excitatory if row.passed_cell == 1 else expected_inhibitory
            index = action_rows[row.action_sha256]
            target[index] = np.uint8(min(TICK_CAP, int(target[index]) + 1))
            prefix.append(row)
            custody_roots.update(event_roots)
            prefix_checkpoint_roots.add(expected_parent)
        arrays_valid = (
            self._excitatory.dtype == TICK_DTYPE
            and self._excitatory.shape == (len(self._action_sha256s),)
            and np.array_equal(self._excitatory, expected_excitatory)
            and self._inhibitory.dtype == TICK_DTYPE
            and self._inhibitory.shape == (len(self._action_sha256s),)
            and np.array_equal(self._inhibitory, expected_inhibitory)
            and not self._excitatory.flags.writeable
            and not self._inhibitory.flags.writeable
        )
        expected_checkpoint = _checkpoint_sha256(
            action_sha256s=self._action_sha256s,
            provenance=self._provenance,
            excitatory=expected_excitatory,
            inhibitory=expected_inhibitory,
        )
        if (
            not arrays_valid
            or self._checkpoint_sha256 != expected_checkpoint
            or expected_checkpoint in custody_roots
            or expected_checkpoint in action_rows
            or expected_checkpoint == catalogue_root
        ):
            raise ContinualActionOutcomeMemoryError("checkpoint authentication failed")

    def decide(
        self,
        activation: ActionActivationV1,
        *,
        cell_gate: bool = True,
        outcome_gate: bool = True,
        exploration_gate: bool = False,
    ) -> ActionDecisionV1:
        self.authenticate()
        if type(activation) is not ActionActivationV1:
            raise TypeError("decision requires one exact action activation")
        if any(type(gate) is not bool for gate in (cell_gate, outcome_gate, exploration_gate)):
            raise TypeError("decision gates must be exact booleans")
        activation.authenticate()
        catalogue_rows = {value: index for index, value in enumerate(self._action_sha256s)}
        reserved_context_roots = {
            *self._action_sha256s,
            _catalogue_sha256(self._action_sha256s),
            self._checkpoint_sha256,
        }
        if activation.selection_context_sha256 in reserved_context_roots:
            raise ContinualActionOutcomeMemoryError(
                "selection context aliases checkpoint or catalogue custody"
            )
        if any(value not in catalogue_rows for value in activation.active_action_sha256s):
            raise ContinualActionOutcomeMemoryError("activation contains an unknown action")
        action_indices = tuple(catalogue_rows[value] for value in activation.active_action_sha256s)
        if outcome_gate:
            excitatory = tuple(int(self._excitatory[index]) for index in action_indices)
            inhibitory = tuple(int(self._inhibitory[index]) for index in action_indices)
        else:
            excitatory = (0,) * len(action_indices)
            inhibitory = (0,) * len(action_indices)
        effective = tuple(
            max(
                0,
                min(
                    MAX_OUTPUT_SCORE,
                    base + 2 * (positive - negative),
                ),
            )
            for base, positive, negative in zip(
                activation.base_scores,
                excitatory,
                inhibitory,
                strict=True,
            )
        )
        raw = _physical_scores(
            action_indices=action_indices,
            effective_scores=effective,
            cell_gate=cell_gate,
            exploration_gate=exploration_gate,
        )
        output = _physical_output_record(raw)
        status, selected_root, selected_index = _decision_result(
            active_action_sha256s=activation.active_action_sha256s,
            action_indices=action_indices,
            effective_scores=effective,
            cell_gate=cell_gate,
            exploration_gate=exploration_gate,
            physical_output_record=output,
        )
        partial = ActionDecisionV1(
            status=status,
            checkpoint_sha256=self._checkpoint_sha256,
            catalogue_sha256=_catalogue_sha256(self._action_sha256s),
            activation=activation,
            action_indices=action_indices,
            excitatory_ticks=excitatory,
            inhibitory_ticks=inhibitory,
            effective_scores=effective,
            tonic_score=0 if exploration_gate else TONIC_SCORE,
            selected_action_sha256=selected_root,
            selected_action_index=selected_index,
            cell_gate=cell_gate,
            outcome_gate=outcome_gate,
            exploration_gate=exploration_gate,
            physical_raw_scores=raw,
            physical_output_record=MappingProxyType(output),
            sha256="",
        )
        decision = replace(
            partial,
            sha256=_digest(_DECISION_NAMESPACE, partial.record(include_sha256=False)),
        )
        decision.authenticate()
        self.authenticate()
        return decision


def _freeze_memory(
    *,
    action_sha256s: tuple[str, ...],
    provenance: tuple[ActionOutcomeProvenanceV1, ...],
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> FrozenContinualActionOutcomeMemoryV1:
    excitatory = _freeze_ticks(excitatory)
    inhibitory = _freeze_ticks(inhibitory)
    checkpoint = _checkpoint_sha256(
        action_sha256s=action_sha256s,
        provenance=provenance,
        excitatory=excitatory,
        inhibitory=inhibitory,
    )
    return FrozenContinualActionOutcomeMemoryV1(
        action_sha256s=action_sha256s,
        provenance=provenance,
        excitatory=excitatory,
        inhibitory=inhibitory,
        checkpoint_sha256=checkpoint,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def start_continual_action_outcome_memory_v1(
    action_sha256s: tuple[str, ...],
) -> FrozenContinualActionOutcomeMemoryV1:
    roots = _validate_roots(
        tuple(sorted(action_sha256s)) if type(action_sha256s) is tuple else action_sha256s,
        label="action catalogue",
        nonempty=True,
        canonical=True,
    )
    if len(roots) > MAX_ACTIONS:
        raise ContinualActionOutcomeMemoryError("action catalogue capacity exceeded")
    zeros = np.zeros(len(roots), dtype=TICK_DTYPE)
    return _freeze_memory(
        action_sha256s=roots,
        provenance=(),
        excitatory=zeros,
        inhibitory=zeros,
    )


def advance_continual_action_outcome_memory_v1(
    parent: FrozenContinualActionOutcomeMemoryV1,
    outcome: SelectedActionOutcomeV1,
    *,
    plasticity_gate: bool = True,
    consequence_gate: bool = True,
) -> FrozenContinualActionOutcomeMemoryV1:
    if type(parent) is not FrozenContinualActionOutcomeMemoryV1:
        raise TypeError("advance requires one exact continual action outcome memory")
    if type(outcome) is not SelectedActionOutcomeV1:
        raise TypeError("advance requires one exact selected action outcome")
    if type(plasticity_gate) is not bool or type(consequence_gate) is not bool:
        raise TypeError("advance gates must be exact booleans")
    parent.authenticate()
    outcome.authenticate()
    expected_decision = parent.decide(
        outcome.decision.activation,
        cell_gate=outcome.decision.cell_gate,
        outcome_gate=outcome.decision.outcome_gate,
        exploration_gate=outcome.decision.exploration_gate,
    )
    if expected_decision != outcome.decision:
        raise ContinualActionOutcomeMemoryError("outcome decision is not from this parent")
    parent_root = parent.checkpoint_sha256
    parent_arrays = parent.checkpoint_array_sha256s
    if not plasticity_gate or not consequence_gate:
        return parent
    provenance = parent._provenance
    if any(row.outcome_sha256 == outcome.sha256 for row in provenance):
        raise ContinualActionOutcomeMemoryError("selected outcome replay is forbidden")
    if any(row.decision_sha256 == outcome.decision.sha256 for row in provenance):
        raise ContinualActionOutcomeMemoryError("action decision replay is forbidden")
    if any(row.activation_sha256 == outcome.decision.activation.sha256 for row in provenance):
        raise ContinualActionOutcomeMemoryError("action activation replay is forbidden")
    if any(
        row.selection_context_sha256 == outcome.decision.activation.selection_context_sha256
        for row in provenance
    ):
        raise ContinualActionOutcomeMemoryError("selection context replay is forbidden")
    if any(row.proposed_source_sha256 == outcome.proposed_source_sha256 for row in provenance):
        raise ContinualActionOutcomeMemoryError("proposed source replay is forbidden")
    if any(row.execution_sha256 == outcome.execution_sha256 for row in provenance):
        raise ContinualActionOutcomeMemoryError("action execution replay is forbidden")
    if any(row.consequence_sha256 == outcome.consequence_sha256 for row in provenance):
        raise ContinualActionOutcomeMemoryError("action consequence replay is forbidden")
    if len(provenance) >= MAX_PROVENANCE_ROWS:
        raise ContinualActionOutcomeMemoryError("action outcome provenance capacity exceeded")
    new_provenance = _provenance_from_outcome(outcome)
    historical_event_roots = {value for row in provenance for value in _provenance_event_roots(row)}
    reserved_roots = {
        *parent._action_sha256s,
        parent.catalogue_sha256,
        parent.checkpoint_sha256,
    }
    new_event_roots = _provenance_event_roots(new_provenance)
    if historical_event_roots.intersection(new_event_roots) or reserved_roots.intersection(
        new_event_roots
    ):
        raise ContinualActionOutcomeMemoryError("event-root custody replay or cross-role splice")
    action_index = parent._action_sha256s.index(outcome.selected_action_sha256)
    excitatory = np.asarray(parent._excitatory).copy()
    inhibitory = np.asarray(parent._inhibitory).copy()
    target = excitatory if outcome.passed_cell == 1 else inhibitory
    target[action_index] = np.uint8(min(TICK_CAP, int(target[action_index]) + 1))
    successor = _freeze_memory(
        action_sha256s=parent._action_sha256s,
        provenance=(*provenance, new_provenance),
        excitatory=excitatory,
        inhibitory=inhibitory,
    )
    parent.authenticate()
    if parent.checkpoint_sha256 != parent_root or parent.checkpoint_array_sha256s != parent_arrays:
        raise RuntimeError("continual action outcome advance mutated its parent")
    return successor


__all__ = [
    "CLAIM_BOUNDARY",
    "EXPLORATION_READY",
    "MAX_ACTIONS",
    "MAX_AUXILIARY_EVENT_ROOTS",
    "MAX_OUTPUT_SCORE",
    "READY",
    "SCHEMA",
    "STOP_CELL_OFF",
    "STOP_INSUFFICIENT_OUTCOME_EVIDENCE",
    "STOP_NO_CANDIDATE",
    "STOP_PHYSICAL_RACE",
    "TICK_CAP",
    "TONIC_SCORE",
    "ActionActivationV1",
    "ActionDecisionV1",
    "ActionOutcomeProvenanceV1",
    "ContinualActionOutcomeMemoryError",
    "FrozenContinualActionOutcomeMemoryV1",
    "SelectedActionOutcomeV1",
    "advance_continual_action_outcome_memory_v1",
    "seal_action_activation_v1",
    "seal_selected_action_outcome_v1",
    "start_continual_action_outcome_memory_v1",
]
