"""Factorized feedback memory for a fixed two-axis hypothesis bank.

The learner stores only cue-to-component excitatory/inhibitory ticks on two
independent axes.  Full hypotheses are reconstructed from those component
scores and handed, as a complete four-candidate bank, to the established
physical action race.  Environment adapters remain responsible for producing
rooted cues, proposals, executions, consequences, and feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import product

from experiments.r3_pilot.common_continual_action_outcome_memory_v1 import (
    MAX_OUTPUT_SCORE,
    TONIC_SCORE,
    ActionDecisionV1,
    seal_action_activation_v1,
    start_continual_action_outcome_memory_v1,
)

SCHEMA = "COMMON_FACTORIZED_FEEDBACK_HYPOTHESIS_MEMORY_V1"
CLAIM_BOUNDARY = (
    "caller-attested-two-axis-feedback-conditioned-component-evidence-and-"
    "compositional-fixed-bank-hypothesis-selection-through-a-physical-neuronal-race-"
    "not-cue-discovery-feedback-semantics-diagnosis-correctness-source-synthesis-"
    "repository-debugging-transfer-swe-or-intelligence"
)

TICK_CAP = 255
MAX_CUES_PER_AXIS = 16
MAX_CUE_ROWS_PER_AXIS = 1_024
MAX_PROVENANCE_ROWS = 1_024

_CONSTRUCTION_TOKEN = object()
_SHA256_HEX = frozenset("0123456789abcdef")

_HYPOTHESIS_NAMESPACE = b"common-factorized-feedback-hypothesis-v1"
_OBSERVATION_NAMESPACE = b"common-factorized-feedback-observation-v1"
_SCORE_NAMESPACE = b"common-factorized-feedback-score-v1"
_RACE_CONTEXT_NAMESPACE = b"common-factorized-feedback-race-context-v1"
_DECISION_NAMESPACE = b"common-factorized-feedback-decision-v1"
_OUTCOME_NAMESPACE = b"common-factorized-feedback-outcome-v1"
_CHECKPOINT_NAMESPACE = b"common-factorized-feedback-checkpoint-v1"


class FactorizedFeedbackHypothesisMemoryError(ValueError):
    """A hypothesis, observation, outcome, or checkpoint failed custody checks."""


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


def _canonical_roots(
    values: object,
    *,
    label: str,
    nonempty: bool,
    maximum: int | None = None,
) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be one exact tuple")
    roots = values
    if (
        (nonempty and not roots)
        or (maximum is not None and len(roots) > maximum)
        or len(set(roots)) != len(roots)
        or any(not _valid_sha256(value) for value in roots)
    ):
        raise FactorizedFeedbackHypothesisMemoryError(f"{label} is invalid")
    return tuple(sorted(roots))


@dataclass(frozen=True, slots=True)
class FactorizedHypothesisV1:
    """One full hypothesis composed from exactly one component on each axis."""

    component_sha256s: tuple[str, str]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {"component_sha256s": self.component_sha256s}
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if (
            type(self.component_sha256s) is not tuple
            or len(self.component_sha256s) != 2
            or len(set(self.component_sha256s)) != 2
            or any(not _valid_sha256(value) for value in self.component_sha256s)
            or self.sha256 != _digest(_HYPOTHESIS_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized hypothesis authentication failed"
            )


def seal_factorized_hypothesis_v1(
    component_sha256s: tuple[str, str],
) -> FactorizedHypothesisV1:
    if type(component_sha256s) is not tuple or len(component_sha256s) != 2:
        raise TypeError("hypothesis components must be one exact two-root tuple")
    payload = {"component_sha256s": component_sha256s}
    result = FactorizedHypothesisV1(
        component_sha256s=component_sha256s,
        sha256=_digest(_HYPOTHESIS_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _canonical_bank(
    hypotheses: object,
) -> tuple[FactorizedHypothesisV1, ...]:
    if type(hypotheses) is not tuple:
        raise TypeError("hypothesis bank must be one exact tuple")
    if len(hypotheses) != 4 or any(
        type(value) is not FactorizedHypothesisV1 for value in hypotheses
    ):
        raise FactorizedFeedbackHypothesisMemoryError(
            "hypothesis bank must contain four exact hypotheses"
        )
    for hypothesis in hypotheses:
        hypothesis.authenticate()
    bank = tuple(sorted(hypotheses, key=lambda value: value.component_sha256s))
    if len({value.sha256 for value in bank}) != 4:
        raise FactorizedFeedbackHypothesisMemoryError("hypothesis replay is forbidden")
    axis_components = tuple(
        tuple(sorted({value.component_sha256s[axis] for value in bank})) for axis in range(2)
    )
    if (
        len(axis_components[0]) != 2
        or len(axis_components[1]) != 2
        or set(axis_components[0]).intersection(axis_components[1])
        or {value.component_sha256s for value in bank}
        != set(product(axis_components[0], axis_components[1]))
        or set(value.sha256 for value in bank).intersection(
            {*axis_components[0], *axis_components[1]}
        )
    ):
        raise FactorizedFeedbackHypothesisMemoryError(
            "hypothesis bank is not one disjoint full 2x2 Cartesian product"
        )
    return bank


def _axis_components(
    bank: tuple[FactorizedHypothesisV1, ...],
) -> tuple[tuple[str, str], tuple[str, str]]:
    return tuple(  # type: ignore[return-value]
        tuple(sorted({value.component_sha256s[axis] for value in bank})) for axis in range(2)
    )


@dataclass(frozen=True, slots=True)
class FactorizedFeedbackObservationV1:
    """One rooted selection context and two independently rooted cue sets."""

    selection_context_sha256: str
    axis_cue_sha256s: tuple[tuple[str, ...], tuple[str, ...]]
    active_hypothesis_sha256s: tuple[str, ...]
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "selection_context_sha256": self.selection_context_sha256,
            "axis_cue_sha256s": self.axis_cue_sha256s,
            "active_hypothesis_sha256s": self.active_hypothesis_sha256s,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.axis_cue_sha256s) is not tuple or len(self.axis_cue_sha256s) != 2:
            raise FactorizedFeedbackHypothesisMemoryError(
                "observation must contain exactly two cue axes"
            )
        axes = tuple(
            _canonical_roots(
                values,
                label=f"axis {axis} cues",
                nonempty=True,
                maximum=MAX_CUES_PER_AXIS,
            )
            for axis, values in enumerate(self.axis_cue_sha256s)
        )
        active = _canonical_roots(
            self.active_hypothesis_sha256s,
            label="active hypothesis roots",
            nonempty=True,
        )
        cue_roots = {*axes[0], *axes[1]}
        if (
            not _valid_sha256(self.selection_context_sha256)
            or axes != self.axis_cue_sha256s
            or active != self.active_hypothesis_sha256s
            or set(axes[0]).intersection(axes[1])
            or self.selection_context_sha256 in cue_roots
            or self.selection_context_sha256 in active
            or cue_roots.intersection(active)
            or self.sha256 != _digest(_OBSERVATION_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized feedback observation authentication failed"
            )


def seal_factorized_feedback_observation_v1(
    *,
    selection_context_sha256: str,
    axis_cue_sha256s: tuple[tuple[str, ...], tuple[str, ...]],
    active_hypothesis_sha256s: tuple[str, ...],
) -> FactorizedFeedbackObservationV1:
    if type(axis_cue_sha256s) is not tuple or len(axis_cue_sha256s) != 2:
        raise TypeError("observation cues must be one exact two-axis tuple")
    axes = tuple(
        _canonical_roots(
            values,
            label=f"axis {axis} cues",
            nonempty=True,
            maximum=MAX_CUES_PER_AXIS,
        )
        for axis, values in enumerate(axis_cue_sha256s)
    )
    active = _canonical_roots(
        active_hypothesis_sha256s,
        label="active hypothesis roots",
        nonempty=True,
    )
    payload = {
        "selection_context_sha256": selection_context_sha256,
        "axis_cue_sha256s": axes,
        "active_hypothesis_sha256s": active,
    }
    result = FactorizedFeedbackObservationV1(
        **payload,
        sha256=_digest(_OBSERVATION_NAMESPACE, payload),
    )
    result.authenticate()
    return result


@dataclass(frozen=True, slots=True)
class FactorizedHypothesisScoreV1:
    """Auditable per-axis evidence retained for one full hypothesis."""

    hypothesis: FactorizedHypothesisV1
    axis_excitatory_ticks: tuple[tuple[int, ...], tuple[int, ...]]
    axis_inhibitory_ticks: tuple[tuple[int, ...], tuple[int, ...]]
    axis_net_scores: tuple[int, int]
    total_score: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "hypothesis": self.hypothesis.record(),
            "axis_excitatory_ticks": self.axis_excitatory_ticks,
            "axis_inhibitory_ticks": self.axis_inhibitory_ticks,
            "axis_net_scores": self.axis_net_scores,
            "total_score": self.total_score,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.hypothesis) is not FactorizedHypothesisV1:
            raise TypeError("hypothesis score requires one exact hypothesis")
        self.hypothesis.authenticate()
        tick_axes = (self.axis_excitatory_ticks, self.axis_inhibitory_ticks)
        expected_net = tuple(
            min(
                MAX_OUTPUT_SCORE,
                TONIC_SCORE
                * sum(
                    max(0, positive - negative)
                    for positive, negative in zip(
                        self.axis_excitatory_ticks[axis],
                        self.axis_inhibitory_ticks[axis],
                        strict=True,
                    )
                ),
            )
            for axis in range(2)
        )
        if (
            any(type(values) is not tuple or len(values) != 2 for values in tick_axes)
            or any(type(axis_values) is not tuple for values in tick_axes for axis_values in values)
            or any(
                type(value) is not int or not 0 <= value <= TICK_CAP
                for values in tick_axes
                for axis_values in values
                for value in axis_values
            )
            or any(
                len(self.axis_excitatory_ticks[axis]) != len(self.axis_inhibitory_ticks[axis])
                for axis in range(2)
            )
            or type(self.axis_net_scores) is not tuple
            or len(self.axis_net_scores) != 2
            or self.axis_net_scores != expected_net
            or self.total_score != sum(expected_net)
            or self.sha256 != _digest(_SCORE_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized hypothesis score authentication failed"
            )


def _seal_score(
    *,
    hypothesis: FactorizedHypothesisV1,
    excitatory: tuple[tuple[int, ...], tuple[int, ...]],
    inhibitory: tuple[tuple[int, ...], tuple[int, ...]],
) -> FactorizedHypothesisScoreV1:
    net = tuple(
        min(
            MAX_OUTPUT_SCORE,
            TONIC_SCORE
            * sum(
                max(0, positive - negative)
                for positive, negative in zip(excitatory[axis], inhibitory[axis], strict=True)
            ),
        )
        for axis in range(2)
    )
    payload = {
        "hypothesis": hypothesis,
        "axis_excitatory_ticks": excitatory,
        "axis_inhibitory_ticks": inhibitory,
        "axis_net_scores": net,
        "total_score": sum(net),
    }
    result = FactorizedHypothesisScoreV1(
        **payload,
        sha256=_digest(
            _SCORE_NAMESPACE,
            {
                **payload,
                "hypothesis": hypothesis.record(),
            },
        ),
    )
    result.authenticate()
    return result


def _physical_base_scores(
    scores: tuple[FactorizedHypothesisScoreV1, ...],
) -> tuple[int, ...]:
    """Fit raw two-axis totals into one physical race without changing the maximum."""

    raw = tuple(score.total_score for score in scores)
    overflow = max(0, max(raw, default=0) - MAX_OUTPUT_SCORE)
    return tuple(max(0, score - overflow) for score in raw)


def _race_context_sha256(
    *,
    checkpoint_sha256: str,
    observation: FactorizedFeedbackObservationV1,
    cell_gate: bool,
    outcome_gate: bool,
    axis_gates: tuple[bool, bool],
    exploration_hypothesis_sha256: str | None,
) -> str:
    return _digest(
        _RACE_CONTEXT_NAMESPACE,
        {
            "checkpoint_sha256": checkpoint_sha256,
            "observation_sha256": observation.sha256,
            "cell_gate": cell_gate,
            "outcome_gate": outcome_gate,
            "axis_gates": axis_gates,
            "exploration_hypothesis_sha256": exploration_hypothesis_sha256,
        },
    )


@dataclass(frozen=True, slots=True)
class FactorizedFeedbackDecisionV1:
    """Factor evidence plus the sole physical full-hypothesis race decision."""

    checkpoint_sha256: str
    observation: FactorizedFeedbackObservationV1
    hypothesis_scores: tuple[FactorizedHypothesisScoreV1, ...]
    action_decision: ActionDecisionV1
    cell_gate: bool
    outcome_gate: bool
    axis_gates: tuple[bool, bool]
    exploration_hypothesis_sha256: str | None
    sha256: str

    @property
    def status(self) -> str:
        return self.action_decision.status

    @property
    def selected_hypothesis_sha256(self) -> str | None:
        return self.action_decision.selected_action_sha256

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "observation": self.observation.record(),
            "hypothesis_scores": tuple(value.record() for value in self.hypothesis_scores),
            "action_decision": self.action_decision.record(),
            "cell_gate": self.cell_gate,
            "outcome_gate": self.outcome_gate,
            "axis_gates": self.axis_gates,
            "exploration_hypothesis_sha256": self.exploration_hypothesis_sha256,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.observation) is not FactorizedFeedbackObservationV1:
            raise TypeError("factorized decision observation must be exact")
        if type(self.action_decision) is not ActionDecisionV1:
            raise TypeError("factorized decision physical action decision must be exact")
        self.observation.authenticate()
        self.action_decision.authenticate()
        if type(self.hypothesis_scores) is not tuple or len(self.hypothesis_scores) != 4:
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized decision must retain four score rows"
            )
        for score in self.hypothesis_scores:
            if type(score) is not FactorizedHypothesisScoreV1:
                raise TypeError("factorized decision score rows must be exact")
            score.authenticate()
        bank = _canonical_bank(tuple(value.hypothesis for value in self.hypothesis_scores))
        score_roots = tuple(value.hypothesis.sha256 for value in self.hypothesis_scores)
        active_roots = self.observation.active_hypothesis_sha256s
        if any(type(value) is not bool for value in (self.cell_gate, self.outcome_gate)):
            raise TypeError("factorized decision gates must be exact booleans")
        if (
            type(self.axis_gates) is not tuple
            or len(self.axis_gates) != 2
            or any(type(value) is not bool for value in self.axis_gates)
            or not _valid_sha256(self.checkpoint_sha256)
            or score_roots != active_roots
            or {value.sha256 for value in bank} != set(active_roots)
            or any(
                len(score.axis_excitatory_ticks[axis])
                != len(self.observation.axis_cue_sha256s[axis])
                for score in self.hypothesis_scores
                for axis in range(2)
            )
            or (
                self.exploration_hypothesis_sha256 is not None
                and self.exploration_hypothesis_sha256 not in active_roots
            )
        ):
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized decision bank or gates changed"
            )
        if self.exploration_hypothesis_sha256 is None:
            expected_base = _physical_base_scores(self.hypothesis_scores)
        else:
            expected_base = tuple(
                TONIC_SCORE if root == self.exploration_hypothesis_sha256 else 0
                for root in active_roots
            )
        expected_activation = seal_action_activation_v1(
            selection_context_sha256=_race_context_sha256(
                checkpoint_sha256=self.checkpoint_sha256,
                observation=self.observation,
                cell_gate=self.cell_gate,
                outcome_gate=self.outcome_gate,
                axis_gates=self.axis_gates,
                exploration_hypothesis_sha256=self.exploration_hypothesis_sha256,
            ),
            active_action_sha256s=active_roots,
            base_scores=expected_base,
        )
        expected_action = start_continual_action_outcome_memory_v1(active_roots).decide(
            expected_activation,
            cell_gate=self.cell_gate,
            outcome_gate=self.outcome_gate,
            exploration_gate=self.exploration_hypothesis_sha256 is not None,
        )
        if self.action_decision != expected_action or self.sha256 != _digest(
            _DECISION_NAMESPACE, self.record(include_sha256=False)
        ):
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized feedback decision authentication failed"
            )


@dataclass(frozen=True, slots=True)
class FactorizedFeedbackOutcomeV1:
    """Caller-attested selected proposal execution and structured feedback."""

    decision: FactorizedFeedbackDecisionV1
    selected_hypothesis_sha256: str
    proposal_sha256: str
    execution_sha256: str
    consequence_sha256: str
    feedback_sha256: str
    passed_cell: int
    sha256: str

    def record(self, *, include_sha256: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "decision": self.decision.record(),
            "selected_hypothesis_sha256": self.selected_hypothesis_sha256,
            "proposal_sha256": self.proposal_sha256,
            "execution_sha256": self.execution_sha256,
            "consequence_sha256": self.consequence_sha256,
            "feedback_sha256": self.feedback_sha256,
            "passed_cell": self.passed_cell,
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def authenticate(self) -> None:
        if type(self.decision) is not FactorizedFeedbackDecisionV1:
            raise TypeError("factorized feedback outcome decision must be exact")
        self.decision.authenticate()
        rooted_fields = (
            self.selected_hypothesis_sha256,
            self.proposal_sha256,
            self.execution_sha256,
            self.consequence_sha256,
            self.feedback_sha256,
        )
        event_roots = _outcome_event_roots(self)
        cue_roots = _outcome_cue_roots(self)
        bank_roots = {value.hypothesis.sha256 for value in self.decision.hypothesis_scores}
        component_roots = {
            component
            for value in self.decision.hypothesis_scores
            for component in value.hypothesis.component_sha256s
        }
        if (
            self.decision.selected_hypothesis_sha256 is None
            or self.selected_hypothesis_sha256 != self.decision.selected_hypothesis_sha256
            or any(not _valid_sha256(value) for value in rooted_fields)
            or len(set(rooted_fields)) != len(rooted_fields)
            or len(set(event_roots)) != len(event_roots)
            or set(cue_roots).intersection(event_roots)
            or {
                self.proposal_sha256,
                self.execution_sha256,
                self.consequence_sha256,
                self.feedback_sha256,
            }.intersection(bank_roots | component_roots)
            or type(self.passed_cell) is not int
            or self.passed_cell not in (0, 1)
            or self.sha256 != _digest(_OUTCOME_NAMESPACE, self.record(include_sha256=False))
        ):
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized feedback outcome authentication failed"
            )


def seal_factorized_feedback_outcome_v1(
    *,
    decision: FactorizedFeedbackDecisionV1,
    proposal_sha256: str,
    execution_sha256: str,
    consequence_sha256: str,
    feedback_sha256: str,
    passed_cell: int,
) -> FactorizedFeedbackOutcomeV1:
    if type(decision) is not FactorizedFeedbackDecisionV1:
        raise TypeError("feedback outcome requires one exact factorized decision")
    decision.authenticate()
    if decision.selected_hypothesis_sha256 is None:
        raise FactorizedFeedbackHypothesisMemoryError("a STOP decision has no outcome")
    payload = {
        "decision": decision,
        "selected_hypothesis_sha256": decision.selected_hypothesis_sha256,
        "proposal_sha256": proposal_sha256,
        "execution_sha256": execution_sha256,
        "consequence_sha256": consequence_sha256,
        "feedback_sha256": feedback_sha256,
        "passed_cell": passed_cell,
    }
    result = FactorizedFeedbackOutcomeV1(
        **payload,
        sha256=_digest(
            _OUTCOME_NAMESPACE,
            {
                **payload,
                "decision": decision.record(),
            },
        ),
    )
    result.authenticate()
    return result


def _outcome_event_roots(outcome: FactorizedFeedbackOutcomeV1) -> tuple[str, ...]:
    return (
        outcome.sha256,
        outcome.decision.sha256,
        outcome.decision.observation.sha256,
        outcome.decision.observation.selection_context_sha256,
        outcome.decision.action_decision.sha256,
        outcome.decision.action_decision.activation.sha256,
        outcome.decision.action_decision.activation.selection_context_sha256,
        outcome.proposal_sha256,
        outcome.execution_sha256,
        outcome.consequence_sha256,
        outcome.feedback_sha256,
    )


def _outcome_cue_roots(outcome: FactorizedFeedbackOutcomeV1) -> tuple[str, ...]:
    return (
        *outcome.decision.observation.axis_cue_sha256s[0],
        *outcome.decision.observation.axis_cue_sha256s[1],
    )


AxisTickRows = tuple[tuple[int, int], ...]
FactorTickRows = tuple[AxisTickRows, AxisTickRows]
AxisCueRows = tuple[tuple[str, ...], tuple[str, ...]]


def _validate_state(
    *,
    axis_cue_sha256s: object,
    excitatory_ticks: object,
    inhibitory_ticks: object,
) -> tuple[AxisCueRows, FactorTickRows, FactorTickRows]:
    if (
        type(axis_cue_sha256s) is not tuple
        or len(axis_cue_sha256s) != 2
        or type(excitatory_ticks) is not tuple
        or len(excitatory_ticks) != 2
        or type(inhibitory_ticks) is not tuple
        or len(inhibitory_ticks) != 2
    ):
        raise FactorizedFeedbackHypothesisMemoryError(
            "factorized checkpoint state must contain two axes"
        )
    for axis in range(2):
        cues = axis_cue_sha256s[axis]
        positive_rows = excitatory_ticks[axis]
        negative_rows = inhibitory_ticks[axis]
        if (
            type(cues) is not tuple
            or len(cues) > MAX_CUE_ROWS_PER_AXIS
            or len(set(cues)) != len(cues)
            or any(not _valid_sha256(value) for value in cues)
            or type(positive_rows) is not tuple
            or type(negative_rows) is not tuple
            or len(positive_rows) != len(cues)
            or len(negative_rows) != len(cues)
        ):
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized checkpoint cue catalogue changed"
            )
        for rows in (positive_rows, negative_rows):
            if any(
                type(row) is not tuple
                or len(row) != 2
                or any(type(value) is not int or not 0 <= value <= TICK_CAP for value in row)
                for row in rows
            ):
                raise FactorizedFeedbackHypothesisMemoryError(
                    "factorized checkpoint tick rows changed"
                )
    if set(axis_cue_sha256s[0]).intersection(axis_cue_sha256s[1]):
        raise FactorizedFeedbackHypothesisMemoryError("cue identities crossed axes")
    return axis_cue_sha256s, excitatory_ticks, inhibitory_ticks


def _checkpoint_sha256(
    *,
    hypotheses: tuple[FactorizedHypothesisV1, ...],
    provenance: tuple[FactorizedFeedbackOutcomeV1, ...],
    axis_cue_sha256s: AxisCueRows,
    excitatory_ticks: FactorTickRows,
    inhibitory_ticks: FactorTickRows,
) -> str:
    return _digest(
        _CHECKPOINT_NAMESPACE,
        {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "hypotheses": tuple(value.record() for value in hypotheses),
            "provenance": tuple(value.record() for value in provenance),
            "axis_cue_sha256s": axis_cue_sha256s,
            "excitatory_ticks": excitatory_ticks,
            "inhibitory_ticks": inhibitory_ticks,
        },
    )


def _route_state(
    *,
    hypotheses: tuple[FactorizedHypothesisV1, ...],
    checkpoint_sha256: str,
    axis_cue_sha256s: AxisCueRows,
    excitatory_ticks: FactorTickRows,
    inhibitory_ticks: FactorTickRows,
    observation: FactorizedFeedbackObservationV1,
    cell_gate: bool,
    outcome_gate: bool,
    axis_gates: tuple[bool, bool],
    exploration_hypothesis_sha256: str | None,
) -> FactorizedFeedbackDecisionV1:
    if type(observation) is not FactorizedFeedbackObservationV1:
        raise TypeError("route requires one exact factorized feedback observation")
    observation.authenticate()
    if any(type(value) is not bool for value in (cell_gate, outcome_gate)):
        raise TypeError("route gates must be exact booleans")
    if (
        type(axis_gates) is not tuple
        or len(axis_gates) != 2
        or any(type(value) is not bool for value in axis_gates)
    ):
        raise TypeError("axis gates must be one exact two-boolean tuple")
    bank_by_root = {value.sha256: value for value in hypotheses}
    expected_active = tuple(sorted(bank_by_root))
    reserved_roots = {
        checkpoint_sha256,
        *expected_active,
        *(component for value in hypotheses for component in value.component_sha256s),
    }
    observation_roots = {
        observation.selection_context_sha256,
        *observation.axis_cue_sha256s[0],
        *observation.axis_cue_sha256s[1],
    }
    if observation.active_hypothesis_sha256s != expected_active:
        raise FactorizedFeedbackHypothesisMemoryError(
            "route requires the complete frozen hypothesis bank"
        )
    if reserved_roots.intersection(observation_roots):
        raise FactorizedFeedbackHypothesisMemoryError(
            "observation aliases checkpoint or hypothesis custody"
        )
    if (
        exploration_hypothesis_sha256 is not None
        and exploration_hypothesis_sha256 not in bank_by_root
    ):
        raise FactorizedFeedbackHypothesisMemoryError(
            "exploration target is outside the frozen hypothesis bank"
        )
    components = _axis_components(hypotheses)
    cue_rows = tuple(
        {value: index for index, value in enumerate(axis_cue_sha256s[axis])} for axis in range(2)
    )
    scores: list[FactorizedHypothesisScoreV1] = []
    for root in expected_active:
        hypothesis = bank_by_root[root]
        positive_by_cue: list[tuple[int, ...]] = []
        negative_by_cue: list[tuple[int, ...]] = []
        for axis in range(2):
            if not outcome_gate or not axis_gates[axis]:
                zeros = (0,) * len(observation.axis_cue_sha256s[axis])
                positive_by_cue.append(zeros)
                negative_by_cue.append(zeros)
                continue
            component_index = components[axis].index(hypothesis.component_sha256s[axis])
            positive_by_cue.append(
                tuple(
                    excitatory_ticks[axis][cue_rows[axis][cue]][component_index]
                    if cue in cue_rows[axis]
                    else 0
                    for cue in observation.axis_cue_sha256s[axis]
                )
            )
            negative_by_cue.append(
                tuple(
                    inhibitory_ticks[axis][cue_rows[axis][cue]][component_index]
                    if cue in cue_rows[axis]
                    else 0
                    for cue in observation.axis_cue_sha256s[axis]
                )
            )
        scores.append(
            _seal_score(
                hypothesis=hypothesis,
                excitatory=tuple(positive_by_cue),  # type: ignore[arg-type]
                inhibitory=tuple(negative_by_cue),  # type: ignore[arg-type]
            )
        )
    score_rows = tuple(scores)
    if exploration_hypothesis_sha256 is None:
        base_scores = _physical_base_scores(score_rows)
    else:
        base_scores = tuple(
            TONIC_SCORE if root == exploration_hypothesis_sha256 else 0 for root in expected_active
        )
    activation = seal_action_activation_v1(
        selection_context_sha256=_race_context_sha256(
            checkpoint_sha256=checkpoint_sha256,
            observation=observation,
            cell_gate=cell_gate,
            outcome_gate=outcome_gate,
            axis_gates=axis_gates,
            exploration_hypothesis_sha256=exploration_hypothesis_sha256,
        ),
        active_action_sha256s=expected_active,
        base_scores=base_scores,
    )
    action_decision = start_continual_action_outcome_memory_v1(expected_active).decide(
        activation,
        cell_gate=cell_gate,
        outcome_gate=outcome_gate,
        exploration_gate=exploration_hypothesis_sha256 is not None,
    )
    payload = {
        "checkpoint_sha256": checkpoint_sha256,
        "observation": observation,
        "hypothesis_scores": score_rows,
        "action_decision": action_decision,
        "cell_gate": cell_gate,
        "outcome_gate": outcome_gate,
        "axis_gates": axis_gates,
        "exploration_hypothesis_sha256": exploration_hypothesis_sha256,
    }
    result = FactorizedFeedbackDecisionV1(
        **payload,
        sha256=_digest(
            _DECISION_NAMESPACE,
            {
                **payload,
                "observation": observation.record(),
                "hypothesis_scores": tuple(value.record() for value in score_rows),
                "action_decision": action_decision.record(),
            },
        ),
    )
    result.authenticate()
    return result


def _apply_outcome(
    *,
    hypotheses: tuple[FactorizedHypothesisV1, ...],
    axis_cue_sha256s: AxisCueRows,
    excitatory_ticks: FactorTickRows,
    inhibitory_ticks: FactorTickRows,
    outcome: FactorizedFeedbackOutcomeV1,
) -> tuple[AxisCueRows, FactorTickRows, FactorTickRows]:
    selected = next(
        value for value in hypotheses if value.sha256 == outcome.selected_hypothesis_sha256
    )
    components = _axis_components(hypotheses)
    new_cues = [list(axis_cue_sha256s[0]), list(axis_cue_sha256s[1])]
    new_positive = [
        [list(row) for row in excitatory_ticks[0]],
        [list(row) for row in excitatory_ticks[1]],
    ]
    new_negative = [
        [list(row) for row in inhibitory_ticks[0]],
        [list(row) for row in inhibitory_ticks[1]],
    ]
    target = new_positive if outcome.passed_cell == 1 else new_negative
    for axis in range(2):
        for cue in outcome.decision.observation.axis_cue_sha256s[axis]:
            if cue not in new_cues[axis]:
                if len(new_cues[axis]) >= MAX_CUE_ROWS_PER_AXIS:
                    raise FactorizedFeedbackHypothesisMemoryError(
                        "factorized cue catalogue capacity exceeded"
                    )
                new_cues[axis].append(cue)
                new_positive[axis].append([0, 0])
                new_negative[axis].append([0, 0])
            row = new_cues[axis].index(cue)
            component = components[axis].index(selected.component_sha256s[axis])
            target[axis][row][component] = min(
                TICK_CAP,
                target[axis][row][component] + 1,
            )
    cue_state: AxisCueRows = tuple(tuple(values) for values in new_cues)  # type: ignore[assignment]
    positive_state: FactorTickRows = tuple(  # type: ignore[assignment]
        tuple(tuple(row) for row in rows) for rows in new_positive
    )
    negative_state: FactorTickRows = tuple(  # type: ignore[assignment]
        tuple(tuple(row) for row in rows) for rows in new_negative
    )
    return cue_state, positive_state, negative_state


class FrozenFactorizedFeedbackHypothesisMemoryV1:
    """One immutable checkpoint reconstructed from ordered feedback provenance."""

    __slots__ = (
        "_axis_cue_sha256s",
        "_checkpoint_sha256",
        "_excitatory_ticks",
        "_hypotheses",
        "_inhibitory_ticks",
        "_provenance",
    )

    def __init__(
        self,
        *,
        hypotheses: tuple[FactorizedHypothesisV1, ...],
        provenance: tuple[FactorizedFeedbackOutcomeV1, ...],
        axis_cue_sha256s: AxisCueRows,
        excitatory_ticks: FactorTickRows,
        inhibitory_ticks: FactorTickRows,
        checkpoint_sha256: str,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("factorized feedback memories are factory-owned")
        self._hypotheses = hypotheses
        self._provenance = provenance
        self._axis_cue_sha256s = axis_cue_sha256s
        self._excitatory_ticks = excitatory_ticks
        self._inhibitory_ticks = inhibitory_ticks
        self._checkpoint_sha256 = checkpoint_sha256
        self.authenticate()

    @property
    def hypotheses(self) -> tuple[FactorizedHypothesisV1, ...]:
        self.authenticate()
        return self._hypotheses

    @property
    def hypothesis_sha256s(self) -> tuple[str, ...]:
        self.authenticate()
        return tuple(sorted(value.sha256 for value in self._hypotheses))

    @property
    def axis_component_sha256s(self) -> tuple[tuple[str, str], tuple[str, str]]:
        self.authenticate()
        return _axis_components(self._hypotheses)

    @property
    def axis_cue_sha256s(self) -> AxisCueRows:
        self.authenticate()
        return self._axis_cue_sha256s

    @property
    def excitatory_ticks(self) -> FactorTickRows:
        self.authenticate()
        return self._excitatory_ticks

    @property
    def inhibitory_ticks(self) -> FactorTickRows:
        self.authenticate()
        return self._inhibitory_ticks

    @property
    def provenance(self) -> tuple[FactorizedFeedbackOutcomeV1, ...]:
        self.authenticate()
        return self._provenance

    @property
    def checkpoint_sha256(self) -> str:
        self.authenticate()
        return self._checkpoint_sha256

    def authenticate(self) -> None:
        bank = _canonical_bank(self._hypotheses)
        if bank != self._hypotheses:
            raise FactorizedFeedbackHypothesisMemoryError(
                "checkpoint hypothesis bank is not canonical"
            )
        _validate_state(
            axis_cue_sha256s=self._axis_cue_sha256s,
            excitatory_ticks=self._excitatory_ticks,
            inhibitory_ticks=self._inhibitory_ticks,
        )
        if (
            type(self._provenance) is not tuple
            or len(self._provenance) > MAX_PROVENANCE_ROWS
            or not _valid_sha256(self._checkpoint_sha256)
        ):
            raise FactorizedFeedbackHypothesisMemoryError("checkpoint provenance or root changed")
        cue_state: AxisCueRows = ((), ())
        positive_state: FactorTickRows = ((), ())
        negative_state: FactorTickRows = ((), ())
        prefix: tuple[FactorizedFeedbackOutcomeV1, ...] = ()
        historical_cue_roots: set[str] = set()
        historical_event_roots: set[str] = set()
        prefix_checkpoint_roots: set[str] = set()
        reserved_bank_roots = {value.sha256 for value in bank} | {
            component for value in bank for component in value.component_sha256s
        }
        for outcome in self._provenance:
            if type(outcome) is not FactorizedFeedbackOutcomeV1:
                raise TypeError("checkpoint provenance outcomes must be exact")
            outcome.authenticate()
            expected_parent = _checkpoint_sha256(
                hypotheses=bank,
                provenance=prefix,
                axis_cue_sha256s=cue_state,
                excitatory_ticks=positive_state,
                inhibitory_ticks=negative_state,
            )
            expected_decision = _route_state(
                hypotheses=bank,
                checkpoint_sha256=expected_parent,
                axis_cue_sha256s=cue_state,
                excitatory_ticks=positive_state,
                inhibitory_ticks=negative_state,
                observation=outcome.decision.observation,
                cell_gate=outcome.decision.cell_gate,
                outcome_gate=outcome.decision.outcome_gate,
                axis_gates=outcome.decision.axis_gates,
                exploration_hypothesis_sha256=(outcome.decision.exploration_hypothesis_sha256),
            )
            event_roots = _outcome_event_roots(outcome)
            cue_roots = _outcome_cue_roots(outcome)
            if (
                outcome.decision.checkpoint_sha256 != expected_parent
                or outcome.decision != expected_decision
                or historical_event_roots.intersection(event_roots)
                or historical_event_roots.intersection(cue_roots)
                or historical_cue_roots.intersection(event_roots)
                or (reserved_bank_roots | prefix_checkpoint_roots | {expected_parent}).intersection(
                    event_roots
                )
                or (reserved_bank_roots | prefix_checkpoint_roots | {expected_parent}).intersection(
                    cue_roots
                )
            ):
                raise FactorizedFeedbackHypothesisMemoryError(
                    "checkpoint provenance replay, splice, or parent mismatch"
                )
            cue_state, positive_state, negative_state = _apply_outcome(
                hypotheses=bank,
                axis_cue_sha256s=cue_state,
                excitatory_ticks=positive_state,
                inhibitory_ticks=negative_state,
                outcome=outcome,
            )
            prefix = (*prefix, outcome)
            historical_cue_roots.update(cue_roots)
            historical_event_roots.update(event_roots)
            prefix_checkpoint_roots.add(expected_parent)
        expected_checkpoint = _checkpoint_sha256(
            hypotheses=bank,
            provenance=prefix,
            axis_cue_sha256s=cue_state,
            excitatory_ticks=positive_state,
            inhibitory_ticks=negative_state,
        )
        if (
            cue_state != self._axis_cue_sha256s
            or positive_state != self._excitatory_ticks
            or negative_state != self._inhibitory_ticks
            or expected_checkpoint != self._checkpoint_sha256
            or expected_checkpoint in historical_cue_roots
            or expected_checkpoint in historical_event_roots
            or expected_checkpoint in reserved_bank_roots
        ):
            raise FactorizedFeedbackHypothesisMemoryError(
                "factorized feedback checkpoint authentication failed"
            )

    def route(
        self,
        observation: FactorizedFeedbackObservationV1,
        *,
        cell_gate: bool = True,
        outcome_gate: bool = True,
        axis_gates: tuple[bool, bool] = (True, True),
        exploration_hypothesis_sha256: str | None = None,
    ) -> FactorizedFeedbackDecisionV1:
        self.authenticate()
        result = _route_state(
            hypotheses=self._hypotheses,
            checkpoint_sha256=self._checkpoint_sha256,
            axis_cue_sha256s=self._axis_cue_sha256s,
            excitatory_ticks=self._excitatory_ticks,
            inhibitory_ticks=self._inhibitory_ticks,
            observation=observation,
            cell_gate=cell_gate,
            outcome_gate=outcome_gate,
            axis_gates=axis_gates,
            exploration_hypothesis_sha256=exploration_hypothesis_sha256,
        )
        self.authenticate()
        return result


def _freeze_memory(
    *,
    hypotheses: tuple[FactorizedHypothesisV1, ...],
    provenance: tuple[FactorizedFeedbackOutcomeV1, ...],
    axis_cue_sha256s: AxisCueRows,
    excitatory_ticks: FactorTickRows,
    inhibitory_ticks: FactorTickRows,
) -> FrozenFactorizedFeedbackHypothesisMemoryV1:
    checkpoint = _checkpoint_sha256(
        hypotheses=hypotheses,
        provenance=provenance,
        axis_cue_sha256s=axis_cue_sha256s,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
    )
    return FrozenFactorizedFeedbackHypothesisMemoryV1(
        hypotheses=hypotheses,
        provenance=provenance,
        axis_cue_sha256s=axis_cue_sha256s,
        excitatory_ticks=excitatory_ticks,
        inhibitory_ticks=inhibitory_ticks,
        checkpoint_sha256=checkpoint,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def start_factorized_feedback_hypothesis_memory_v1(
    hypotheses: tuple[FactorizedHypothesisV1, ...],
) -> FrozenFactorizedFeedbackHypothesisMemoryV1:
    bank = _canonical_bank(hypotheses)
    return _freeze_memory(
        hypotheses=bank,
        provenance=(),
        axis_cue_sha256s=((), ()),
        excitatory_ticks=((), ()),
        inhibitory_ticks=((), ()),
    )


def advance_factorized_feedback_hypothesis_memory_v1(
    parent: FrozenFactorizedFeedbackHypothesisMemoryV1,
    outcome: FactorizedFeedbackOutcomeV1,
    *,
    plasticity_gate: bool = True,
    consequence_gate: bool = True,
) -> FrozenFactorizedFeedbackHypothesisMemoryV1:
    if type(parent) is not FrozenFactorizedFeedbackHypothesisMemoryV1:
        raise TypeError("advance requires one exact factorized feedback memory")
    if type(outcome) is not FactorizedFeedbackOutcomeV1:
        raise TypeError("advance requires one exact factorized feedback outcome")
    if type(plasticity_gate) is not bool or type(consequence_gate) is not bool:
        raise TypeError("advance gates must be exact booleans")
    parent.authenticate()
    outcome.authenticate()
    expected_decision = parent.route(
        outcome.decision.observation,
        cell_gate=outcome.decision.cell_gate,
        outcome_gate=outcome.decision.outcome_gate,
        axis_gates=outcome.decision.axis_gates,
        exploration_hypothesis_sha256=(outcome.decision.exploration_hypothesis_sha256),
    )
    if expected_decision != outcome.decision:
        raise FactorizedFeedbackHypothesisMemoryError(
            "feedback outcome decision is not from this parent"
        )
    if not plasticity_gate or not consequence_gate:
        return parent
    if len(parent._provenance) >= MAX_PROVENANCE_ROWS:
        raise FactorizedFeedbackHypothesisMemoryError(
            "factorized feedback provenance capacity exceeded"
        )
    historical_event_roots = {
        root for value in parent._provenance for root in _outcome_event_roots(value)
    }
    historical_cue_roots = {
        root for value in parent._provenance for root in _outcome_cue_roots(value)
    }
    new_event_roots = _outcome_event_roots(outcome)
    new_cue_roots = _outcome_cue_roots(outcome)
    reserved_roots = {
        parent._checkpoint_sha256,
        *(value.sha256 for value in parent._hypotheses),
        *(component for value in parent._hypotheses for component in value.component_sha256s),
    }
    if (
        historical_event_roots.intersection(new_event_roots)
        or historical_event_roots.intersection(new_cue_roots)
        or historical_cue_roots.intersection(new_event_roots)
        or reserved_roots.intersection(new_event_roots)
        or reserved_roots.intersection(new_cue_roots)
    ):
        raise FactorizedFeedbackHypothesisMemoryError(
            "feedback outcome event-root replay or cross-role splice"
        )
    parent_snapshot = (
        parent._checkpoint_sha256,
        parent._axis_cue_sha256s,
        parent._excitatory_ticks,
        parent._inhibitory_ticks,
        parent._provenance,
    )
    cues, positive, negative = _apply_outcome(
        hypotheses=parent._hypotheses,
        axis_cue_sha256s=parent._axis_cue_sha256s,
        excitatory_ticks=parent._excitatory_ticks,
        inhibitory_ticks=parent._inhibitory_ticks,
        outcome=outcome,
    )
    successor = _freeze_memory(
        hypotheses=parent._hypotheses,
        provenance=(*parent._provenance, outcome),
        axis_cue_sha256s=cues,
        excitatory_ticks=positive,
        inhibitory_ticks=negative,
    )
    parent.authenticate()
    if parent_snapshot != (
        parent._checkpoint_sha256,
        parent._axis_cue_sha256s,
        parent._excitatory_ticks,
        parent._inhibitory_ticks,
        parent._provenance,
    ):
        raise RuntimeError("factorized feedback advance mutated its parent")
    return successor


__all__ = [
    "CLAIM_BOUNDARY",
    "MAX_CUES_PER_AXIS",
    "MAX_CUE_ROWS_PER_AXIS",
    "MAX_PROVENANCE_ROWS",
    "SCHEMA",
    "TICK_CAP",
    "FactorizedFeedbackDecisionV1",
    "FactorizedFeedbackHypothesisMemoryError",
    "FactorizedFeedbackObservationV1",
    "FactorizedFeedbackOutcomeV1",
    "FactorizedHypothesisScoreV1",
    "FactorizedHypothesisV1",
    "FrozenFactorizedFeedbackHypothesisMemoryV1",
    "advance_factorized_feedback_hypothesis_memory_v1",
    "seal_factorized_feedback_observation_v1",
    "seal_factorized_feedback_outcome_v1",
    "seal_factorized_hypothesis_v1",
    "start_factorized_feedback_hypothesis_memory_v1",
]
