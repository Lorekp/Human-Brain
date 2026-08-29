"""Cell-only horizon-2 planning through a learned successor circuit.

Host execution is confined to target-free public motor training and the one
physically selected real action.  Target-free pre-value x motor-efference x
observed-post-value events grow ten local successor banks.  During planning,
those frozen synapses predict raw next values for both plies; per-case physical
coincidence cells and fixed value synapses drive a fresh 100-cell ordered-pair
race, whose spike reaches the ordinary motor bank through a fixed projection.
No candidate program is constructed or executed during inference.

Only the visible mixed-order development matrix is declared here.  A withheld
k=3 boundary belongs in a separate runner after a persisted visible PASS.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import (  # noqa: E402
    public_typed_reward_modulated_edit_actor as banked,
)
from experiments.r3_pilot.public_typed_sensorimotor_edit_controller import (  # noqa: E402
    _RAW_INPUT_STEPS,
    _drive_spike_mask,
    _thermometer_current,
)
from humanbrain.cells import AdaptiveGlifPopulation  # noqa: E402

SEED = 17
VALUE_MIN = -32
VALUE_MAX = 32
VALUE_COUNT = VALUE_MAX - VALUE_MIN + 1
VISIBLE_GRIDS = (
    (-12, -7, -1, 2, 6, 11),
    (-10, -5, 0, 3, 7, 12),
)

_ACTION_BY_LABEL = {action.label: action for action in banked.ACTIONS}
PAIR_BANK = tuple(
    (first, second) for first in banked.ACTIONS for second in banked.ACTIONS
)
PAIR_LABELS = tuple(
    f"{first.label}->{second.label}" for first, second in PAIR_BANK
)

_VALUE_THRESHOLDS = tuple(range(VALUE_MIN + 1, VALUE_MAX + 1))
_VALUE_PROJECTION_SCALE_PA = 10_000.0
_VALUE_PROJECTION_LIMIT_PA = 5_000.0
_VALUE_PROJECTION_STEPS = 30
_SUCCESSOR_TRAINING_VALUES = tuple(range(-20, 21))
_SUCCESSOR_LEARNING_RATE_PA = 1_200.0
_SUCCESSOR_MAXIMUM_WEIGHT_PA = 1_200.0
_SUCCESSOR_BASE_CURRENT_PA = 600.0
_RACE_STEPS = 600

_COINCIDENCE_SCALE_PA = 10_000.0
_COINCIDENCE_LIMIT_PA = 5_000.0
_PAIR_BASE_CURRENT_PA = 600.0
# Low-current lexicographic value scale.  One additional terminal match is
# worth more than all six intermediate matches plus both edit costs and the
# delayed-edit cost (70 > 30 + 20 + 5), while adjacent semantic values remain
# separated by the fitted cell's latency bins around 600-1,050 pA.
_TERMINAL_MATCH_PA = 70.0
_INTERMEDIATE_MATCH_PA = 5.0
_EDIT_EVENT_COST_PA = 10.0
_DELAYED_EDIT_COST_PA = 5.0
_MOTOR_CURRENT_PA = 5_000.0
_VISIBLE_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_horizon2_successor_visible_probe.json"
)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    target_actions: tuple[str, str]
    expected_actions: tuple[str, str, str]


VISIBLE_MATRIX = tuple(
    spec
    for k in (1, 2, 4)
    for spec in (
        TaskSpec(
            f"abs(x)+{k}",
            ("WRAP_ABS", f"WRAP_ADD[{k}]"),
            ("WRAP_ABS", f"WRAP_ADD[{k}]", "DONE"),
        ),
        TaskSpec(
            f"abs(x+{k})",
            (f"WRAP_ADD[{k}]", "WRAP_ABS"),
            (f"WRAP_ADD[{k}]", "WRAP_ABS", "DONE"),
        ),
        TaskSpec(
            f"abs(x)-{k}",
            ("WRAP_ABS", f"WRAP_SUB[{k}]"),
            ("WRAP_ABS", f"WRAP_SUB[{k}]", "DONE"),
        ),
        TaskSpec(
            f"abs(x-{k})",
            (f"WRAP_SUB[{k}]", "WRAP_ABS"),
            (f"WRAP_SUB[{k}]", "WRAP_ABS", "DONE"),
        ),
    )
)

@dataclass(frozen=True, slots=True)
class SpentRegression:
    name: str
    verdict: str
    actions: tuple[str, ...]
    final_outputs: tuple[int, ...]
    unique_races: bool
    frozen_checkpoint_immutable: bool
    disposition: str


SPENT_ABS_PLUS_THREE = SpentRegression(
    name="abs(x)+3",
    verdict="STOP_G_RMA1",
    actions=("WRAP_ADD[3]", "WRAP_ABS", "WRAP_ABS"),
    final_outputs=(10, 3, 4, 13),
    unique_races=True,
    frozen_checkpoint_immutable=True,
    disposition="Banked; never rerun or tune against this target.",
)


@dataclass(frozen=True, slots=True)
class TransitionObservation:
    current_value: int
    action: str
    next_value: int


@dataclass(frozen=True, slots=True)
class CellRace:
    bank: str
    candidates: tuple[str, ...]
    currents_pa: tuple[float, ...]
    winner_cell: int
    winner: str
    spike_step: int
    simultaneous_spike_candidates: int


class CellRaceFailure(RuntimeError):
    """A successor or pair population had no unique physical first spiker."""


def _physical_race(
    currents_pa: np.ndarray,
    labels: Sequence[str],
    *,
    bank: str,
) -> CellRace:
    currents = np.ascontiguousarray(currents_pa, dtype=np.float64)
    candidates = tuple(labels)
    if currents.shape != (len(candidates),) or not candidates:
        raise ValueError("physical race needs one current per candidate")
    population = AdaptiveGlifPopulation(len(candidates))
    for step in range(_RACE_STEPS):
        fired = np.flatnonzero(population.step(currents))
        if fired.size:
            if fired.size != 1:
                raise CellRaceFailure(
                    f"{bank} produced {fired.size} simultaneous first spikes"
                )
            winner = int(fired[0])
            return CellRace(
                bank=bank,
                candidates=candidates,
                currents_pa=tuple(float(value) for value in currents),
                winner_cell=winner,
                winner=candidates[winner],
                spike_step=step,
                simultaneous_spike_candidates=1,
            )
    raise CellRaceFailure(f"{bank} produced no spike within its fixed bound")


class ExactValueCellEncoder:
    """Fixed thermometer/opponent circuit producing one raw-value cell."""

    @staticmethod
    def projection() -> tuple[np.ndarray, np.ndarray]:
        weights = np.zeros((VALUE_COUNT, len(_VALUE_THRESHOLDS)), dtype=np.float64)
        bias = np.zeros(VALUE_COUNT, dtype=np.float64)
        # -32 is the absence of the first threshold.  Interior values are one
        # adjacent-threshold opponent pair; +32 is the final threshold.
        weights[0, 0] = -1.0
        bias[0] = 1.0
        for row in range(1, VALUE_COUNT - 1):
            weights[row, row - 1] = 1.0
            weights[row, row] = -1.0
        weights[-1, -1] = 1.0
        return weights, bias

    @classmethod
    def projection_sha256(cls) -> str:
        weights, bias = cls.projection()
        return sha256(weights.tobytes() + bias.tobytes()).hexdigest()

    @classmethod
    def encode(cls, value: int) -> np.ndarray:
        if type(value) is not int or not VALUE_MIN <= value <= VALUE_MAX:
            raise ValueError(f"raw value must be an integer in [{VALUE_MIN}, {VALUE_MAX}]")
        thermometer = _drive_spike_mask(
            AdaptiveGlifPopulation(len(_VALUE_THRESHOLDS)),
            _thermometer_current(value),
            steps=_RAW_INPUT_STEPS,
        ).astype(np.float64)
        weights, bias = cls.projection()
        net = weights @ thermometer + bias
        currents = np.ascontiguousarray(
            np.clip(
                net * _VALUE_PROJECTION_SCALE_PA,
                -_VALUE_PROJECTION_LIMIT_PA,
                _VALUE_PROJECTION_LIMIT_PA,
            )
        )
        exact_cells = _drive_spike_mask(
            AdaptiveGlifPopulation(VALUE_COUNT),
            currents,
            steps=_VALUE_PROJECTION_STEPS,
        ).astype(np.float64)
        if np.count_nonzero(exact_cells) != 1:
            raise CellRaceFailure("exact-value projection did not produce one cell")
        return np.ascontiguousarray(exact_cells)


def _successor_checkpoint_sha256(
    weights_pa: np.ndarray,
    training_digest: str,
) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(weights_pa, dtype=np.float64).tobytes())
    digest.update(training_digest.encode("ascii"))
    digest.update(ExactValueCellEncoder.projection_sha256().encode("ascii"))
    digest.update(str(_SUCCESSOR_LEARNING_RATE_PA).encode("ascii"))
    digest.update(str(_SUCCESSOR_MAXIMUM_WEIGHT_PA).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenSuccessorModel:
    weights_pa: np.ndarray
    training_digest: str
    training_observations: int
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _successor_checkpoint_sha256(self.weights_pa, self.training_digest)

    def predict(self, value: int, action: str) -> tuple[int, CellRace, str]:
        if action not in _ACTION_BY_LABEL:
            raise ValueError("successor action is outside the fixed bank")
        features = ExactValueCellEncoder.encode(value)
        feature_sha = sha256(features.tobytes()).hexdigest()
        before_sha = self.live_checkpoint_sha256
        action_cell = next(
            index
            for index, candidate in enumerate(banked.ACTIONS)
            if candidate.label == action
        )
        currents = np.ascontiguousarray(
            _SUCCESSOR_BASE_CURRENT_PA
            + self.weights_pa[action_cell] @ features
        )
        labels = tuple(str(value) for value in range(VALUE_MIN, VALUE_MAX + 1))
        race = _physical_race(currents, labels, bank="learned-successor-value")
        predicted = VALUE_MIN + race.winner_cell
        after_sha = self.live_checkpoint_sha256
        if before_sha != after_sha or after_sha != self.checkpoint_sha256:
            raise RuntimeError("frozen successor checkpoint changed during prediction")
        return predicted, race, feature_sha

    def predict_many(
        self,
        values: Sequence[int],
        action: str,
    ) -> tuple[tuple[int, ...], tuple[CellRace, ...], tuple[str, ...]]:
        predictions: list[int] = []
        races: list[CellRace] = []
        feature_hashes: list[str] = []
        for value in values:
            predicted, race, feature_sha = self.predict(value, action)
            predictions.append(predicted)
            races.append(race)
            feature_hashes.append(feature_sha)
        return tuple(predictions), tuple(races), tuple(feature_hashes)


@dataclass(frozen=True, slots=True)
class SuccessorTrainingSummary:
    seed: int
    value_min: int
    value_max: int
    values: tuple[int, ...]
    action_labels: tuple[str, ...]
    observations: int
    local_updates: int
    learned_synapses: int
    learning_rate_pa: float
    maximum_weight_pa: float
    exact_value_projection_sha256: str
    training_digest: str
    checkpoint_sha256: str
    host_execution_scope: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _collect_public_motor_observations() -> tuple[TransitionObservation, ...]:
    """Execute only target-free public motor transitions for fitting."""
    identity = banked._identity_program()
    base_outputs = banked._execute(identity, _SUCCESSOR_TRAINING_VALUES)
    observations: list[TransitionObservation] = []
    for action in banked.ACTIONS:
        next_program = banked._apply_action(identity, action)
        next_outputs = banked._execute(next_program, _SUCCESSOR_TRAINING_VALUES)
        observations.extend(
            TransitionObservation(current, action.label, next_value)
            for current, next_value in zip(base_outputs, next_outputs, strict=True)
        )
    return tuple(observations)


def train_public_successor(
) -> tuple[FrozenSuccessorModel, SuccessorTrainingSummary]:
    """Locally bind pre x efference x post before any composite target exists."""
    observations = _collect_public_motor_observations()
    weights = np.zeros(
        (len(banked.ACTIONS), VALUE_COUNT, VALUE_COUNT), dtype=np.float64
    )
    local_updates = 0
    for item in observations:
        if not VALUE_MIN <= item.next_value <= VALUE_MAX:
            raise ValueError("public successor target left the fitted raw range")
        pre = ExactValueCellEncoder.encode(item.current_value)
        post = ExactValueCellEncoder.encode(item.next_value)
        action_cell = next(
            index
            for index, candidate in enumerate(banked.ACTIONS)
            if candidate.label == item.action
        )
        # One local three-factor event: presynaptic value cell, motor
        # efference bank, and observed postsynaptic value cell.
        weights[action_cell] = np.minimum(
            weights[action_cell]
            + _SUCCESSOR_LEARNING_RATE_PA * np.outer(post, pre),
            _SUCCESSOR_MAXIMUM_WEIGHT_PA,
        )
        local_updates += 1
    weights = np.ascontiguousarray(weights)
    training_record = tuple(asdict(item) for item in observations)
    training_digest = sha256(_canonical_json(training_record)).hexdigest()
    checkpoint_sha = _successor_checkpoint_sha256(weights, training_digest)
    weights.setflags(write=False)
    frozen = FrozenSuccessorModel(
        weights_pa=weights,
        training_digest=training_digest,
        training_observations=len(observations),
        checkpoint_sha256=checkpoint_sha,
    )
    return frozen, SuccessorTrainingSummary(
        seed=SEED,
        value_min=VALUE_MIN,
        value_max=VALUE_MAX,
        values=_SUCCESSOR_TRAINING_VALUES,
        action_labels=tuple(action.label for action in banked.ACTIONS),
        observations=len(observations),
        local_updates=local_updates,
        learned_synapses=int(np.count_nonzero(weights)),
        learning_rate_pa=_SUCCESSOR_LEARNING_RATE_PA,
        maximum_weight_pa=_SUCCESSOR_MAXIMUM_WEIGHT_PA,
        exact_value_projection_sha256=ExactValueCellEncoder.projection_sha256(),
        training_digest=training_digest,
        checkpoint_sha256=checkpoint_sha,
        host_execution_scope=(
            "Target-free primitive motor fitting only; candidate inference performs "
            "no host candidate edit, program execution, loss, ranking, or arg-extremum; "
            "the simulator accumulates committed fixed coincidence synapses."
        ),
    )


@dataclass(frozen=True, slots=True)
class PairCandidate:
    pair: str
    first_action: str
    second_action: str
    predicted_intermediate_outputs: tuple[int, ...]
    predicted_terminal_outputs: tuple[int, ...]
    intermediate_active_synapses: int
    terminal_active_synapses: int
    intermediate_coincidence_sha256: str
    terminal_coincidence_sha256: str
    successor_trace_sha256: str
    scoring_current_pa: float


def _coincidence_features(
    predicted: Sequence[int],
    desired: Sequence[int],
) -> tuple[np.ndarray, str]:
    """Pool one physical exact-value AND per case after prediction."""
    predicted_values = tuple(predicted)
    desired_values = tuple(desired)
    if not predicted_values or len(predicted_values) != len(desired_values):
        raise ValueError("coincidence needs paired predicted and desired values")
    traces: list[np.ndarray] = []
    for predicted_value, desired_value in zip(
        predicted_values, desired_values, strict=True
    ):
        predicted_cells = ExactValueCellEncoder.encode(predicted_value)
        desired_cells = ExactValueCellEncoder.encode(desired_value)
        and_net = predicted_cells + desired_cells - 1.5
        currents = np.ascontiguousarray(
            np.clip(
                and_net * _COINCIDENCE_SCALE_PA,
                -_COINCIDENCE_LIMIT_PA,
                _COINCIDENCE_LIMIT_PA,
            )
        )
        coincidence = _drive_spike_mask(
            AdaptiveGlifPopulation(VALUE_COUNT),
            currents,
            steps=_VALUE_PROJECTION_STEPS,
        ).astype(np.float64)
        active = int(np.count_nonzero(coincidence))
        if active > 1:
            raise CellRaceFailure("value coincidence produced multiple cells")
        traces.append(np.ascontiguousarray(coincidence))
    features = np.ascontiguousarray(np.stack(traces))
    return features, sha256(features.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class FixedPairValueProjection:
    """Materialized target-independent synapses into the 100 pair cells."""

    weights_pa: np.ndarray
    bias_pa: np.ndarray
    sha256: str

    @classmethod
    def create(cls, case_count: int) -> "FixedPairValueProjection":
        if case_count <= 0:
            raise ValueError("pair value projection needs at least one case")
        # Layout: pair x {intermediate, terminal} x case x exact-value cell.
        weights = np.empty(
            (len(PAIR_BANK), 2, case_count, VALUE_COUNT), dtype=np.float64
        )
        weights[:, 0, :, :] = _INTERMEDIATE_MATCH_PA
        weights[:, 1, :, :] = _TERMINAL_MATCH_PA
        bias = np.empty(len(PAIR_BANK), dtype=np.float64)
        for pair_cell, (first, second) in enumerate(PAIR_BANK):
            edit_events = int(first.kind != "done") + int(second.kind != "done")
            delayed_edit = int(second.kind != "done")
            bias[pair_cell] = (
                _PAIR_BASE_CURRENT_PA
                - _EDIT_EVENT_COST_PA * edit_events
                - _DELAYED_EDIT_COST_PA * delayed_edit
            )
        digest = sha256()
        digest.update(weights.tobytes())
        digest.update(bias.tobytes())
        digest.update(_canonical_json(PAIR_LABELS))
        frozen_weights = np.ascontiguousarray(weights)
        frozen_bias = np.ascontiguousarray(bias)
        frozen_weights.setflags(write=False)
        frozen_bias.setflags(write=False)
        return cls(frozen_weights, frozen_bias, digest.hexdigest())

    def currents(self, coincidence_spikes: np.ndarray) -> np.ndarray:
        features = np.ascontiguousarray(coincidence_spikes, dtype=np.float64)
        if features.shape != self.weights_pa.shape:
            raise ValueError("pair coincidence features do not match fixed synapses")
        # This einsum is the simulator's fixed synaptic current accumulation,
        # not a score, rank, max, or target-conditioned host reduction.
        return np.ascontiguousarray(
            self.bias_pa + np.einsum("ptcv,ptcv->p", self.weights_pa, features)
        )


def _pair_to_motor_projection() -> np.ndarray:
    projection = np.zeros((len(banked.ACTIONS), len(PAIR_BANK)), dtype=np.float64)
    for pair_cell, (first, _) in enumerate(PAIR_BANK):
        action_cell = next(
            index
            for index, action in enumerate(banked.ACTIONS)
            if action.label == first.label
        )
        projection[action_cell, pair_cell] = 1.0
    return projection


def _motor_race_from_pair(pair_race: CellRace) -> CellRace:
    pair_spikes = np.zeros(len(PAIR_BANK), dtype=np.float64)
    pair_spikes[pair_race.winner_cell] = 1.0
    currents = np.ascontiguousarray(
        _MOTOR_CURRENT_PA * (_pair_to_motor_projection() @ pair_spikes)
    )
    return _physical_race(
        currents,
        tuple(action.label for action in banked.ACTIONS),
        bank="pair-to-motor",
    )


@dataclass(frozen=True, slots=True)
class Horizon2Decision:
    selected_pair: str
    first_action: str
    second_action: str
    predicted_intermediate_outputs: tuple[int, ...]
    predicted_terminal_outputs: tuple[int, ...]
    pair_race: CellRace
    motor_race: CellRace
    value_projection_sha256: str
    candidate_count: int
    successor_cell_races: int
    candidate_digest: str
    successor_checkpoint_before_sha256: str
    successor_checkpoint_after_sha256: str


def plan_horizon2(
    successor: FrozenSuccessorModel,
    current_outputs: Sequence[int],
    desired_outputs: Sequence[int],
) -> Horizon2Decision:
    """Predict both plies in cells and physically race the 100 fixed pairs."""
    current = tuple(current_outputs)
    desired = tuple(desired_outputs)
    if not current or len(current) != len(desired):
        raise ValueError("horizon-2 planning needs paired current and desired outputs")
    checkpoint_before = successor.live_checkpoint_sha256
    if checkpoint_before != successor.checkpoint_sha256:
        raise RuntimeError("successor checkpoint was mutated before planning")

    first_predictions: dict[str, tuple[int, ...]] = {}
    first_trace_hashes: dict[str, tuple[str, ...]] = {}
    successor_races = 0
    for action in banked.ACTIONS:
        predicted, races, feature_hashes = successor.predict_many(current, action.label)
        first_predictions[action.label] = predicted
        first_trace_hashes[action.label] = feature_hashes
        successor_races += len(races)

    candidate_records: list[dict[str, object]] = []
    coincidence_spikes = np.zeros(
        (len(PAIR_BANK), 2, len(current), VALUE_COUNT), dtype=np.float64
    )
    for pair_cell, (first, second) in enumerate(PAIR_BANK):
        intermediate = first_predictions[first.label]
        terminal, races, terminal_trace_hashes = successor.predict_many(
            intermediate, second.label
        )
        successor_races += len(races)
        intermediate_features, intermediate_sha = _coincidence_features(
            intermediate, desired
        )
        terminal_features, terminal_sha = _coincidence_features(terminal, desired)
        coincidence_spikes[pair_cell, 0] = intermediate_features
        coincidence_spikes[pair_cell, 1] = terminal_features
        trace_sha = sha256(
            _canonical_json(
                {
                    "first": first_trace_hashes[first.label],
                    "second": terminal_trace_hashes,
                }
            )
        ).hexdigest()
        candidate_records.append(
            dict(
                pair=f"{first.label}->{second.label}",
                first_action=first.label,
                second_action=second.label,
                predicted_intermediate_outputs=intermediate,
                predicted_terminal_outputs=terminal,
                intermediate_active_synapses=int(
                    np.count_nonzero(intermediate_features)
                ),
                terminal_active_synapses=int(np.count_nonzero(terminal_features)),
                intermediate_coincidence_sha256=intermediate_sha,
                terminal_coincidence_sha256=terminal_sha,
                successor_trace_sha256=trace_sha,
            )
        )

    value_projection = FixedPairValueProjection.create(len(current))
    currents = value_projection.currents(coincidence_spikes)
    candidates = [
        PairCandidate(
            **record,
            scoring_current_pa=float(currents[pair_cell]),
        )
        for pair_cell, record in enumerate(candidate_records)
    ]
    pair_race = _physical_race(
        currents,
        PAIR_LABELS,
        bank=f"horizon-2-pair:{value_projection.sha256}",
    )
    winner = candidates[pair_race.winner_cell]
    motor_race = _motor_race_from_pair(pair_race)
    if motor_race.winner != winner.first_action:
        raise RuntimeError("fixed pair-to-motor projection changed action identity")
    candidate_digest = sha256(
        _canonical_json(tuple(asdict(candidate) for candidate in candidates))
    ).hexdigest()
    checkpoint_after = successor.live_checkpoint_sha256
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("frozen successor changed during horizon-2 planning")
    return Horizon2Decision(
        selected_pair=winner.pair,
        first_action=winner.first_action,
        second_action=winner.second_action,
        predicted_intermediate_outputs=winner.predicted_intermediate_outputs,
        predicted_terminal_outputs=winner.predicted_terminal_outputs,
        pair_race=pair_race,
        motor_race=motor_race,
        value_projection_sha256=value_projection.sha256,
        candidate_count=len(candidates),
        successor_cell_races=successor_races,
        candidate_digest=candidate_digest,
        successor_checkpoint_before_sha256=checkpoint_before,
        successor_checkpoint_after_sha256=checkpoint_after,
    )


@dataclass(frozen=True, slots=True)
class SelectedStep:
    step: int
    before_outputs: tuple[int, ...]
    selected_action: str
    after_outputs: tuple[int, ...]
    program_tokens: tuple[str, ...]
    decision: Horizon2Decision


@dataclass(frozen=True, slots=True)
class TaskResult:
    name: str
    arguments: tuple[int, ...]
    target_tokens: tuple[str, ...]
    desired_outputs: tuple[int, ...]
    expected_actions: tuple[str, ...]
    selected_actions: tuple[str, ...]
    steps: tuple[SelectedStep, ...]
    final_tokens: tuple[str, ...]
    final_outputs: tuple[int, ...]
    exact_actions: bool
    exact_tokens: bool
    exact_outputs: bool
    unique_races: bool
    successor_checkpoint_immutable: bool
    cell_pair_predictions: int
    successor_cell_races: int
    selected_real_executions: int
    failure_stage: str | None
    error: str | None
    passed: bool


def _target_program(spec: TaskSpec) -> banked.tp.Program:
    program = banked._identity_program()
    for label in spec.target_actions:
        program = banked._apply_action(program, _ACTION_BY_LABEL[label])
    return program


def _run_task(
    successor: FrozenSuccessorModel,
    spec: TaskSpec,
    arguments: tuple[int, ...],
) -> TaskResult:
    target = _target_program(spec)
    desired = banked._execute(target, arguments)
    program = banked._identity_program()
    current = banked._execute(program, arguments)
    steps: list[SelectedStep] = []
    cell_pair_predictions = 0
    successor_races = 0
    failure_stage: str | None = None
    error_message: str | None = None
    for step_number in range(1, 4):
        try:
            decision = plan_horizon2(successor, current, desired)
        except CellRaceFailure as error:
            failure_stage = f"planning_{step_number}"
            error_message = str(error)
            break
        cell_pair_predictions += decision.candidate_count
        successor_races += decision.successor_cell_races
        # The sole inference-time environment transition: one physical winner.
        program = banked._apply_action(
            program, _ACTION_BY_LABEL[decision.motor_race.winner]
        )
        after = banked._execute(program, arguments)
        steps.append(
            SelectedStep(
                step=step_number,
                before_outputs=current,
                selected_action=decision.motor_race.winner,
                after_outputs=after,
                program_tokens=program.tokens,
                decision=decision,
            )
        )
        current = after

    selected_actions = tuple(step.selected_action for step in steps)
    exact_actions = selected_actions == spec.expected_actions
    exact_tokens = program.tokens == target.tokens
    exact_outputs = current == desired
    unique_races = failure_stage is None and all(
        step.decision.pair_race.simultaneous_spike_candidates == 1
        and step.decision.motor_race.simultaneous_spike_candidates == 1
        for step in steps
    )
    immutable = all(
        step.decision.successor_checkpoint_before_sha256
        == step.decision.successor_checkpoint_after_sha256
        == successor.checkpoint_sha256
        for step in steps
    )
    passed = (
        len(steps) == 3
        and exact_actions
        and exact_tokens
        and exact_outputs
        and unique_races
        and immutable
    )
    return TaskResult(
        name=spec.name,
        arguments=arguments,
        target_tokens=target.tokens,
        desired_outputs=desired,
        expected_actions=spec.expected_actions,
        selected_actions=selected_actions,
        steps=tuple(steps),
        final_tokens=program.tokens,
        final_outputs=current,
        exact_actions=exact_actions,
        exact_tokens=exact_tokens,
        exact_outputs=exact_outputs,
        unique_races=unique_races,
        successor_checkpoint_immutable=immutable,
        cell_pair_predictions=cell_pair_predictions,
        successor_cell_races=successor_races,
        selected_real_executions=len(steps),
        failure_stage=failure_stage,
        error=error_message,
        passed=passed,
    )


@dataclass(frozen=True, slots=True)
class MatrixResult:
    schema: str
    successor_training: SuccessorTrainingSummary
    successor_checkpoint_sha256: str
    pair_bank_sha256: str
    scoring_sha256: str
    tasks: tuple[TaskResult, ...]
    passed_tasks: int
    total_tasks: int
    cell_pair_predictions: int
    successor_cell_races: int
    selected_real_executions: int
    spent_regression: SpentRegression
    verdict: str
    retry_policy: str
    claim_boundary: str


def _pair_bank_sha256() -> str:
    return sha256(_canonical_json(PAIR_LABELS)).hexdigest()


def _scoring_sha256(case_count: int) -> str:
    return sha256(
        _canonical_json(
            {
                "base_current_pa": _PAIR_BASE_CURRENT_PA,
                "terminal_match_pa": _TERMINAL_MATCH_PA,
                "intermediate_match_pa": _INTERMEDIATE_MATCH_PA,
                "edit_event_cost_pa": _EDIT_EVENT_COST_PA,
                "delayed_edit_cost_pa": _DELAYED_EDIT_COST_PA,
                "coincidence_scale_pa": _COINCIDENCE_SCALE_PA,
                "fixed_pair_value_projection_sha256": (
                    FixedPairValueProjection.create(case_count).sha256
                ),
                "pair_to_motor_projection_sha256": sha256(
                    _pair_to_motor_projection().tobytes()
                ).hexdigest(),
            }
        )
    ).hexdigest()


def _run_matrix(
    successor: FrozenSuccessorModel,
    training: SuccessorTrainingSummary,
    specs: Sequence[TaskSpec],
    arguments: tuple[int, ...],
    *,
    schema: str,
    pass_verdict: str,
    stop_verdict: str,
) -> MatrixResult:
    tasks = tuple(_run_task(successor, spec, arguments) for spec in specs)
    passed_tasks = sum(task.passed for task in tasks)
    return MatrixResult(
        schema=schema,
        successor_training=training,
        successor_checkpoint_sha256=successor.checkpoint_sha256,
        pair_bank_sha256=_pair_bank_sha256(),
        scoring_sha256=_scoring_sha256(len(arguments)),
        tasks=tasks,
        passed_tasks=passed_tasks,
        total_tasks=len(tasks),
        cell_pair_predictions=sum(task.cell_pair_predictions for task in tasks),
        successor_cell_races=sum(task.successor_cell_races for task in tasks),
        selected_real_executions=sum(
            task.selected_real_executions for task in tasks
        ),
        spent_regression=SPENT_ABS_PLUS_THREE,
        verdict=pass_verdict if passed_tasks == len(tasks) else stop_verdict,
        retry_policy=(
            "Deterministic public development only; no seed, feature, score, "
            "horizon, grid, action-bank, or expected-order sweep."
        ),
        claim_boundary=(
            "Learned cell-successor horizon-2 public scalar planning only. Host execution "
            "is restricted to target-free motor fitting and selected real actions; not "
            "Simple-SWE, M0, held-out, C1, or broad coding capability evidence."
        ),
    )


def _verify_visible_readiness(
    successor: FrozenSuccessorModel,
    training: SuccessorTrainingSummary,
) -> tuple[str, str]:
    expected_observations = _collect_public_motor_observations()
    expected_digest = sha256(
        _canonical_json(tuple(asdict(item) for item in expected_observations))
    ).hexdigest()
    expected_count = len(_SUCCESSOR_TRAINING_VALUES) * len(banked.ACTIONS)
    action_labels = tuple(action.label for action in banked.ACTIONS)
    pair_sha = _pair_bank_sha256()
    scoring_sha = _scoring_sha256(len(VISIBLE_GRIDS[0]))
    weights = successor.weights_pa
    conditions = (
        training.seed == SEED,
        training.value_min == VALUE_MIN,
        training.value_max == VALUE_MAX,
        training.values == _SUCCESSOR_TRAINING_VALUES,
        training.action_labels == action_labels,
        training.observations == expected_count,
        training.local_updates == expected_count,
        training.learned_synapses == expected_count,
        training.learning_rate_pa == _SUCCESSOR_LEARNING_RATE_PA,
        training.maximum_weight_pa == _SUCCESSOR_MAXIMUM_WEIGHT_PA,
        training.exact_value_projection_sha256
        == ExactValueCellEncoder.projection_sha256(),
        training.training_digest == expected_digest,
        successor.training_digest == expected_digest,
        successor.training_observations == expected_count,
        training.checkpoint_sha256 == successor.checkpoint_sha256,
        successor.live_checkpoint_sha256 == successor.checkpoint_sha256,
        weights.shape == (len(banked.ACTIONS), VALUE_COUNT, VALUE_COUNT),
        weights.dtype == np.dtype(np.float64),
        weights.flags.c_contiguous,
        not weights.flags.writeable,
        np.isfinite(weights).all(),
        np.all(weights >= 0.0),
        np.all(weights <= _SUCCESSOR_MAXIMUM_WEIGHT_PA),
        int(np.count_nonzero(weights)) == expected_count,
        len(VISIBLE_MATRIX) == 12,
        len(VISIBLE_GRIDS) == 2,
        all(len(grid) == len(VISIBLE_GRIDS[0]) == 6 for grid in VISIBLE_GRIDS),
        len(pair_sha) == len(scoring_sha) == 64,
    )
    if not all(conditions):
        raise RuntimeError("visible matrix readiness authentication failed")
    return pair_sha, scoring_sha


def run_visible_matrix(
    successor: FrozenSuccessorModel,
    training: SuccessorTrainingSummary,
) -> MatrixResult:
    """Run the re-runnable k={1,2,4} public development matrix."""
    pair_sha, scoring_sha = _verify_visible_readiness(successor, training)
    # Each task is evaluated on both predeclared visible grids.
    expanded_specs = tuple(
        TaskSpec(
            f"{spec.name}@G{grid_index}",
            spec.target_actions,
            spec.expected_actions,
        )
        for grid_index in range(len(VISIBLE_GRIDS))
        for spec in VISIBLE_MATRIX
    )
    expanded_arguments = tuple(
        VISIBLE_GRIDS[index // len(VISIBLE_MATRIX)]
        for index in range(len(expanded_specs))
    )
    tasks = tuple(
        _run_task(successor, spec, arguments)
        for spec, arguments in zip(
            expanded_specs, expanded_arguments, strict=True
        )
    )
    passed_tasks = sum(task.passed for task in tasks)
    return MatrixResult(
        schema="public-typed-horizon2-successor-visible-v1",
        successor_training=training,
        successor_checkpoint_sha256=successor.checkpoint_sha256,
        pair_bank_sha256=pair_sha,
        scoring_sha256=scoring_sha,
        tasks=tasks,
        passed_tasks=passed_tasks,
        total_tasks=len(tasks),
        cell_pair_predictions=sum(task.cell_pair_predictions for task in tasks),
        successor_cell_races=sum(task.successor_cell_races for task in tasks),
        selected_real_executions=sum(
            task.selected_real_executions for task in tasks
        ),
        spent_regression=SPENT_ABS_PLUS_THREE,
        verdict=(
            "PASS_HORIZON2_SUCCESSOR_VISIBLE"
            if passed_tasks == len(tasks)
            else "STOP_HORIZON2_SUCCESSOR_VISIBLE"
        ),
        retry_policy=(
            "Re-runnable public development; no seed, feature, score, horizon, "
            "grid, action-bank, or expected-order sweep after first result."
        ),
        claim_boundary=(
            "Learned cell-successor horizon-2 visible scalar development only. Host "
            "execution is restricted to target-free motor fitting and selected real "
            "actions; not the fresh k=3 boundary or coding capability evidence."
        ),
    )


def main() -> int:
    if _VISIBLE_ARTIFACT_PATH.exists():
        prior = json.loads(_VISIBLE_ARTIFACT_PATH.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "artifact": str(_VISIBLE_ARTIFACT_PATH),
                    "verdict": prior.get("verdict"),
                    "passed_tasks": prior.get("passed_tasks"),
                    "total_tasks": prior.get("total_tasks"),
                    "guard": "existing artifact; matrix not rerun",
                },
                sort_keys=True,
            )
        )
        return 0
    successor, training = train_public_successor()
    result = run_visible_matrix(successor, training)
    _VISIBLE_ARTIFACT_PATH.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(_VISIBLE_ARTIFACT_PATH),
                "verdict": result.verdict,
                "passed_tasks": result.passed_tasks,
                "total_tasks": result.total_tasks,
                "successor_checkpoint_sha256": result.successor_checkpoint_sha256,
                "scoring_sha256": result.scoring_sha256,
            },
            sort_keys=True,
        )
    )
    return 0 if result.verdict == "PASS_HORIZON2_SUCCESSOR_VISIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
