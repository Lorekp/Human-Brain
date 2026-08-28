"""Target-free sensorimotor learning for generic scalar typed edits.

The controller first motor-babbles a fixed, target-independent edit vocabulary.
Its own physical motor winner is an efference-copy teaching signal: raw program
outputs before and after that action pass through fitted-cell thermometer and
opponent populations, and the resulting transition-cell activity is bound to
the action by ``SparseCellReadout``'s local synapses.  At inference, current
outputs and desired public outputs enter the same raw sensory path and the
learned inverse map selects exactly one generic typed edit.

The host never computes a residual for the policy, maps an observation to an
action, enumerates sibling programs, keeps a best candidate, or stores policy
currents/use counts.  It only tokenizes raw integers, executes one selected
program, and applies one cell-selected generic edit through the public typed
program-edit seam.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.cells import AdaptiveGlifPopulation  # noqa: E402
from humanbrain.readout import SparseCellReadout  # noqa: E402
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

SEED = 17
MOTOR_EPISODES = 100
VALUE_MIN = -32
VALUE_MAX = 32
VALUE_THRESHOLDS = tuple(range(VALUE_MIN + 1, VALUE_MAX + 1))
MAGNITUDE_COUNT = 32

_RAW_VALUE_CELLS = len(VALUE_THRESHOLDS)
_RAW_INPUT_CURRENT_PA = 5_000.0
_RAW_INPUT_STEPS = 30
_RELATION_CURRENT_SCALE_PA = 10_000.0
_RELATION_CURRENT_LIMIT_PA = 5_000.0
_RELATION_CELLS = 256
_EQUALITY_CELLS = 4
_FEATURE_CELLS = _RELATION_CELLS + _EQUALITY_CELLS

_OUTPUT_BASE_CURRENT_PA = 600.0
_OUTPUT_RACE_STEPS = 600
_LEARNING_RATE_PA = 32.0
_MAXIMUM_WEIGHT_PA = 1_600.0
_READOUT_FAN_IN = _FEATURE_CELLS // SparseCellReadout.CANDIDATE_FACTOR

_MOTOR_ARGUMENT_GRIDS = (
    (-12, -7, -3, 0, 4, 9),
    (-10, -5, -1, 2, 6, 11),
)
_GATE_ARGUMENTS = (-13, -6, 1, 10)
_GATE_PLUS_EXPECTED = (-10, -3, 4, 13)
_GATE_MINUS_EXPECTED = (-16, -9, -2, 7)
_ARTIFACT_PATH = Path(__file__).with_name(
    "public_typed_sensorimotor_edit_controller_probe.json"
)


@dataclass(frozen=True, slots=True)
class EditAction:
    label: str
    kind: str
    offset: int | None = None


_ACTIONS = (
    EditAction("KEEP", "keep"),
    EditAction("ABS_ROOT", "absolute"),
    *tuple(
        EditAction(f"OFFSET_ROOT[{value:+d}]", "offset", value)
        for value in (*range(-9, 0), *range(1, 10))
    ),
)
_ACTION_BY_LABEL = {action.label: action for action in _ACTIONS}


@dataclass(frozen=True, slots=True)
class CellRace:
    bank: str
    candidates: tuple[str, ...]
    currents_pa: tuple[float, ...]
    winner_cell: int
    winner: str
    spike_step: int
    simultaneous_spike_candidates: int


@dataclass(frozen=True, slots=True)
class MotorEpisode:
    episode: int
    argument_grid: int
    motor_race: CellRace
    inverse_race: CellRace
    teaching_target_cell: int
    transition_sha256: str
    edited_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MotorCurriculumSummary:
    seed: int
    episodes: int
    coverage: tuple[tuple[str, int], ...]
    local_updates: int
    structural_recruitments: int
    initial_checkpoint_sha256: str
    frozen_checkpoint_sha256: str
    records: tuple[MotorEpisode, ...]


@dataclass(frozen=True, slots=True)
class GateArm:
    name: str
    arguments: tuple[int, ...]
    before_outputs: tuple[int, ...]
    desired_outputs: tuple[int, ...]
    selected_action: str
    edited_tokens: tuple[str, ...]
    final_outputs: tuple[int, ...]
    passed: bool
    executed_programs: int
    transition_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str
    inverse_race: CellRace


@dataclass(frozen=True, slots=True)
class FixedGateResult:
    schema: str
    seed: int
    motor_episodes: int
    curriculum: MotorCurriculumSummary
    plus_arm: GateArm | None
    minus_arm: GateArm | None
    verdict: str
    retry_policy: str
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class _FrozenArray:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


def _freeze_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[_FrozenArray, ...]:
    frozen: list[_FrozenArray] = []
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        frozen.append(
            _FrozenArray(name, value.dtype.str, tuple(value.shape), value.tobytes())
        )
    return tuple(frozen)


def _thaw_arrays(arrays: Sequence[_FrozenArray]) -> dict[str, np.ndarray]:
    return {
        value.name: np.frombuffer(value.data, dtype=np.dtype(value.dtype))
        .reshape(value.shape)
        .copy()
        for value in arrays
    }


def _checkpoint_sha256(
    arrays: Sequence[_FrozenArray],
    counters: Mapping[str, int],
    output_heterogeneity_pa: np.ndarray,
    pacemaker_synapses_pa: np.ndarray,
) -> str:
    digest = sha256()
    for value in arrays:
        digest.update(value.name.encode("utf-8"))
        digest.update(value.dtype.encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.data)
    digest.update(
        json.dumps(dict(sorted(counters.items())), separators=(",", ":")).encode(
            "ascii"
        )
    )
    digest.update(np.ascontiguousarray(output_heterogeneity_pa).tobytes())
    digest.update(np.ascontiguousarray(pacemaker_synapses_pa).tobytes())
    return digest.hexdigest()


def _drive_spike_mask(
    population: AdaptiveGlifPopulation,
    currents_pa: np.ndarray,
    *,
    steps: int = _RAW_INPUT_STEPS,
) -> np.ndarray:
    population.reset()
    seen = np.zeros(population.cell_count, dtype=bool)
    for _ in range(steps):
        fired = np.flatnonzero(population.step(currents_pa))
        if fired.size:
            seen[fired] = True
    return seen


def _thermometer_current(value: int) -> np.ndarray:
    if type(value) is not int or not VALUE_MIN <= value <= VALUE_MAX:
        raise ValueError(
            f"raw scalar observation must be an int in [{VALUE_MIN}, {VALUE_MAX}]"
        )
    return np.asarray(
        [
            _RAW_INPUT_CURRENT_PA if value >= threshold else 0.0
            for threshold in VALUE_THRESHOLDS
        ],
        dtype=np.float64,
    )


def _sign_current(value: int) -> np.ndarray:
    current = np.zeros(3, dtype=np.float64)
    current[0 if value < 0 else 1 if value == 0 else 2] = _RAW_INPUT_CURRENT_PA
    return current


class _TransitionCellEncoder:
    """Fixed fitted-cell relation circuit over raw before/after integers."""

    def __init__(self) -> None:
        self._before = AdaptiveGlifPopulation(_RAW_VALUE_CELLS)
        self._after = AdaptiveGlifPopulation(_RAW_VALUE_CELLS)
        self._before_sign = AdaptiveGlifPopulation(3)
        self._relation = AdaptiveGlifPopulation(_RELATION_CELLS)
        self._equality = AdaptiveGlifPopulation(_EQUALITY_CELLS)
        self._relation_weights, self._relation_bias = self._relation_projection()
        self._equality_weights, self._equality_bias = self._equality_projection()

    @staticmethod
    def _relation_projection() -> tuple[np.ndarray, np.ndarray]:
        # Input layout: BEFORE thermometer, AFTER thermometer, BEFORE sign.
        weights = np.zeros((_RELATION_CELLS, 2 * _RAW_VALUE_CELLS + 3))
        bias = np.zeros(_RELATION_CELLS)

        def direction_row(row: int, direction: int, magnitude: int) -> None:
            if direction == 0:  # after is above before
                weights[row, :_RAW_VALUE_CELLS] = -1.0
                weights[row, _RAW_VALUE_CELLS : 2 * _RAW_VALUE_CELLS] = 1.0
            else:  # before is above after
                weights[row, :_RAW_VALUE_CELLS] = 1.0
                weights[row, _RAW_VALUE_CELLS : 2 * _RAW_VALUE_CELLS] = -1.0
            bias[row] = -(float(magnitude) - 0.5)

        # Case-pooled, translation-invariant signed magnitude thresholds.
        for direction in range(2):
            for magnitude in range(1, MAGNITUDE_COUNT + 1):
                row = direction * MAGNITUDE_COUNT + magnitude - 1
                direction_row(row, direction, magnitude)

        # Preserve before-sign conjunctions so nonlinear ABS effects remain
        # distinct from a uniform signed translation.
        sign_gate = float(_RAW_VALUE_CELLS)
        gated_base = 2 * MAGNITUDE_COUNT
        for sign in range(3):
            for direction in range(2):
                for magnitude in range(1, MAGNITUDE_COUNT + 1):
                    row = (
                        gated_base
                        + sign * 2 * MAGNITUDE_COUNT
                        + direction * MAGNITUDE_COUNT
                        + magnitude
                        - 1
                    )
                    direction_row(row, direction, magnitude)
                    weights[row, 2 * _RAW_VALUE_CELLS + sign] = sign_gate
                    bias[row] -= sign_gate
        return weights, bias

    @staticmethod
    def _equality_projection() -> tuple[np.ndarray, np.ndarray]:
        # Input layout: 256 relation spikes followed by the three raw sign
        # cells.  UP magnitude-1 is relation 0; DOWN magnitude-1 is 32.
        weights = np.zeros((_EQUALITY_CELLS, _RELATION_CELLS + 3))
        bias = np.full(_EQUALITY_CELLS, 0.5)
        weights[:, 0] = -1.0
        weights[:, MAGNITUDE_COUNT] = -1.0
        sign_gate = float(_RAW_VALUE_CELLS)
        for sign in range(3):
            row = sign + 1
            weights[row, _RELATION_CELLS + sign] = sign_gate
            bias[row] -= sign_gate
        return weights, bias

    @staticmethod
    def _synaptic_current(net: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            np.clip(
                net * _RELATION_CURRENT_SCALE_PA,
                -_RELATION_CURRENT_LIMIT_PA,
                _RELATION_CURRENT_LIMIT_PA,
            ),
            dtype=np.float64,
        )

    def encode(
        self,
        before_values: Sequence[int],
        after_values: Sequence[int],
    ) -> np.ndarray:
        before = tuple(before_values)
        after = tuple(after_values)
        if not before or len(before) != len(after):
            raise ValueError("transition encoding needs paired raw observations")

        accumulated = np.zeros(_FEATURE_CELLS, dtype=np.float64)
        for raw_before, raw_after in zip(before, after, strict=True):
            before_spikes = _drive_spike_mask(
                self._before, _thermometer_current(raw_before)
            )
            after_spikes = _drive_spike_mask(
                self._after, _thermometer_current(raw_after)
            )
            sign_spikes = _drive_spike_mask(
                self._before_sign, _sign_current(raw_before)
            )
            sensory = np.concatenate(
                (
                    before_spikes.astype(np.float64),
                    after_spikes.astype(np.float64),
                    sign_spikes.astype(np.float64),
                )
            )
            relation_net = self._relation_weights @ sensory + self._relation_bias
            relation_spikes = _drive_spike_mask(
                self._relation, self._synaptic_current(relation_net)
            )
            equality_input = np.concatenate(
                (relation_spikes.astype(np.float64), sign_spikes.astype(np.float64))
            )
            equality_net = self._equality_weights @ equality_input + self._equality_bias
            equality_spikes = _drive_spike_mask(
                self._equality, self._synaptic_current(equality_net)
            )
            accumulated += np.concatenate(
                (
                    relation_spikes.astype(np.float64),
                    equality_spikes.astype(np.float64),
                )
            )
        return np.ascontiguousarray(accumulated / float(len(before)))


def _race_population(
    population: AdaptiveGlifPopulation,
    currents_pa: np.ndarray,
    *,
    bank: str,
) -> CellRace:
    if currents_pa.shape != (len(_ACTIONS),):
        raise ValueError("action race current has the wrong shape")
    labels = tuple(action.label for action in _ACTIONS)
    for step in range(_OUTPUT_RACE_STEPS):
        fired = np.flatnonzero(population.step(currents_pa))
        if fired.size:
            if fired.size != 1:
                raise RuntimeError(
                    f"{bank} fitted-cell race produced {fired.size} "
                    "simultaneous spikes"
                )
            winner = int(fired[0])
            return CellRace(
                bank=bank,
                candidates=labels,
                currents_pa=tuple(float(value) for value in currents_pa),
                winner_cell=winner,
                winner=labels[winner],
                spike_step=step,
                simultaneous_spike_candidates=int(fired.size),
            )
    raise RuntimeError(f"{bank} fitted-cell race did not spike within its bound")


def _fresh_action_race(currents_pa: np.ndarray, *, bank: str) -> CellRace:
    return _race_population(
        AdaptiveGlifPopulation(len(_ACTIONS)), currents_pa, bank=bank
    )


def _literal_tokens(value: int) -> tuple[str, ...]:
    sign = "-" if value < 0 else "+"
    return ("integer", sign, *tuple(str(abs(value))), "end")


def _apply_action(program: tp.Program, action: EditAction) -> tp.Program:
    root = next(site for site in tp.program_edit_sites(program) if not site.path)
    if action.kind == "keep":
        edited = program
    elif action.kind == "absolute":
        edited = tp.replace_subtree(program, (), ("absolute", *root.tokens))
    elif action.kind == "offset":
        if action.offset is None or action.offset == 0:
            raise ValueError("offset action needs a nonzero signed literal")
        edited = tp.replace_subtree(
            program,
            (),
            ("add", *root.tokens, *_literal_tokens(action.offset)),
        )
    else:
        raise AssertionError("unknown generic edit action")
    tp.compile(edited)
    return edited


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("value", integer),),
        result=integer,
    )


def _identity_program() -> tp.Program:
    program = tp.Program(("variable-0",), _signature())
    tp.compile(program)
    return program


def _execute(program: tp.Program, arguments: Sequence[int]) -> tuple[int, ...]:
    compiled = tp.compile(program)
    namespace: dict[str, object] = {}
    exec(
        compiled.source,
        {"__builtins__": {"abs": abs, "max": max, "min": min, "sum": sum}},
        namespace,
    )
    function = namespace[program.signature.entry_point]
    outputs: list[int] = []
    for value in arguments:
        result = function(value)  # type: ignore[operator]
        if type(result) is not int:
            raise TypeError("scalar motor observation must be an int")
        outputs.append(result)
    return tuple(outputs)


@dataclass(frozen=True, slots=True)
class _FrozenInverseController:
    arrays: tuple[_FrozenArray, ...]
    counters: dict[str, int]
    output_heterogeneity_pa: np.ndarray
    pacemaker_synapses_pa: np.ndarray
    checkpoint_sha256: str

    def _readout(self, eligibility: np.ndarray) -> SparseCellReadout:
        return SparseCellReadout.from_checkpoint(
            output_cells=len(_ACTIONS),
            reservoir_cells=_FEATURE_CELLS,
            fan_in=_READOUT_FAN_IN,
            reservoir_eligibility=eligibility,
            output_heterogeneity_pa=np.array(
                self.output_heterogeneity_pa, copy=True
            ),
            pacemaker_synapses_pa=np.array(self.pacemaker_synapses_pa, copy=True),
            arrays=_thaw_arrays(self.arrays),
            counters=dict(self.counters),
        )

    def race(
        self,
        before_outputs: Sequence[int],
        desired_outputs: Sequence[int],
    ) -> tuple[EditAction, CellRace, str, str, str]:
        eligibility = _TransitionCellEncoder().encode(
            before_outputs, desired_outputs
        )
        transition_sha = sha256(eligibility.tobytes()).hexdigest()
        readout = self._readout(eligibility)
        before_sha = _checkpoint_sha256(
            _freeze_arrays(readout.checkpoint_arrays()),
            readout.checkpoint_counters(),
            self.output_heterogeneity_pa,
            self.pacemaker_synapses_pa,
        )
        race = _fresh_action_race(
            readout.current(_OUTPUT_BASE_CURRENT_PA), bank="inverse-action"
        )
        after_sha = _checkpoint_sha256(
            _freeze_arrays(readout.checkpoint_arrays()),
            readout.checkpoint_counters(),
            self.output_heterogeneity_pa,
            self.pacemaker_synapses_pa,
        )
        return _ACTION_BY_LABEL[race.winner], race, transition_sha, before_sha, after_sha


def train_target_free_motor_inverse(
) -> tuple[_FrozenInverseController, MotorCurriculumSummary]:
    """Run the single fixed target-free motor curriculum and freeze synapses."""
    rng = np.random.default_rng(SEED)
    eligibility = np.zeros(_FEATURE_CELLS, dtype=np.float64)
    output_heterogeneity_pa = rng.normal(0.0, 4.0, len(_ACTIONS))
    pacemaker_synapses_pa = np.zeros(len(_ACTIONS), dtype=np.float64)
    readout = SparseCellReadout.create(
        output_cells=len(_ACTIONS),
        reservoir_cells=_FEATURE_CELLS,
        fan_in=_READOUT_FAN_IN,
        rng=rng,
        reservoir_eligibility=eligibility,
        output_heterogeneity_pa=output_heterogeneity_pa,
        pacemaker_synapses_pa=pacemaker_synapses_pa,
    )
    initial_arrays = _freeze_arrays(readout.checkpoint_arrays())
    initial_sha = _checkpoint_sha256(
        initial_arrays,
        readout.checkpoint_counters(),
        output_heterogeneity_pa,
        pacemaker_synapses_pa,
    )

    motor_population = AdaptiveGlifPopulation(len(_ACTIONS))
    # This fixed heterogeneous drive is the only exploration input.  The
    # population remains live across episodes, so fitted refractory/adaptation
    # dynamics—not a host use counter—move the winner around the action bank.
    motor_currents = np.ascontiguousarray(
        _OUTPUT_BASE_CURRENT_PA + rng.normal(0.0, 4.0, len(_ACTIONS))
    )
    encoder = _TransitionCellEncoder()
    identity = _identity_program()
    trace_decays = np.zeros(_FEATURE_CELLS, dtype=np.float64)
    records: list[MotorEpisode] = []

    for episode in range(MOTOR_EPISODES):
        grid_index = episode % len(_MOTOR_ARGUMENT_GRIDS)
        arguments = _MOTOR_ARGUMENT_GRIDS[grid_index]
        motor_race = _race_population(
            motor_population, motor_currents, bank="motor-babble"
        )
        motor_action = _ACTIONS[motor_race.winner_cell]
        edited = _apply_action(identity, motor_action)
        before_outputs = _execute(identity, arguments)
        after_outputs = _execute(edited, arguments)
        eligibility[:] = encoder.encode(before_outputs, after_outputs)
        transition_sha = sha256(eligibility.tobytes()).hexdigest()

        inverse_race = _fresh_action_race(
            readout.current(_OUTPUT_BASE_CURRENT_PA), bank="inverse-learning"
        )
        # The only teaching identity is the physical motor winner above: an
        # efference copy of a self-generated action, never a task target.
        readout.teach(
            target=motor_race.winner_cell,
            winner=inverse_race.winner_cell,
            learning_rate_pa=_LEARNING_RATE_PA,
            maximum_weight_pa=_MAXIMUM_WEIGHT_PA,
            trace_decays=trace_decays,
            target_post_gate=1.0,
            winner_post_gate=1.0,
            activity_decay=0.99,
        )
        presentation = readout.advance_target(motor_race.winner_cell)
        readout.turnover(
            output=motor_race.winner_cell,
            structural_event=presentation,
            random_seed=SEED,
            minimum_age=2,
            turnover_probability=0.0,
            learning_rate_pa=_LEARNING_RATE_PA,
            maximum_weight_pa=_MAXIMUM_WEIGHT_PA,
            homeostatic_rate=0.0,
        )
        records.append(
            MotorEpisode(
                episode=episode + 1,
                argument_grid=grid_index,
                motor_race=motor_race,
                inverse_race=inverse_race,
                teaching_target_cell=motor_race.winner_cell,
                transition_sha256=transition_sha,
                edited_tokens=edited.tokens,
            )
        )

    frozen_arrays = _freeze_arrays(readout.checkpoint_arrays())
    counters = readout.checkpoint_counters()
    frozen_sha = _checkpoint_sha256(
        frozen_arrays, counters, output_heterogeneity_pa, pacemaker_synapses_pa
    )
    frozen = _FrozenInverseController(
        arrays=frozen_arrays,
        counters=dict(counters),
        output_heterogeneity_pa=np.array(output_heterogeneity_pa, copy=True),
        pacemaker_synapses_pa=np.array(pacemaker_synapses_pa, copy=True),
        checkpoint_sha256=frozen_sha,
    )
    coverage = tuple(
        (
            action.label,
            sum(record.motor_race.winner == action.label for record in records),
        )
        for action in _ACTIONS
    )
    summary = MotorCurriculumSummary(
        seed=SEED,
        episodes=MOTOR_EPISODES,
        coverage=coverage,
        local_updates=readout.applied_updates,
        structural_recruitments=readout.structural_recruitments,
        initial_checkpoint_sha256=initial_sha,
        frozen_checkpoint_sha256=frozen_sha,
        records=tuple(records),
    )
    return frozen, summary


def _run_gate_arm(
    controller: _FrozenInverseController,
    *,
    name: str,
    desired_outputs: tuple[int, ...],
) -> GateArm:
    identity = _identity_program()
    before_outputs = _execute(identity, _GATE_ARGUMENTS)
    action, race, transition_sha, before_sha, after_sha = controller.race(
        before_outputs, desired_outputs
    )
    edited = _apply_action(identity, action)
    final_outputs = _execute(edited, _GATE_ARGUMENTS)
    return GateArm(
        name=name,
        arguments=_GATE_ARGUMENTS,
        before_outputs=before_outputs,
        desired_outputs=desired_outputs,
        selected_action=action.label,
        edited_tokens=edited.tokens,
        final_outputs=final_outputs,
        passed=final_outputs == desired_outputs,
        executed_programs=1,
        transition_sha256=transition_sha,
        checkpoint_before_sha256=before_sha,
        checkpoint_after_sha256=after_sha,
        inverse_race=race,
    )


def run_fixed_gate() -> FixedGateResult:
    """Spend the fixed seed-17 curriculum and public +/-3 A/B gate once."""
    controller, curriculum = train_target_free_motor_inverse()
    if any(count < 2 for _, count in curriculum.coverage):
        return FixedGateResult(
            schema="public-typed-sensorimotor-edit-controller-v1",
            seed=SEED,
            motor_episodes=MOTOR_EPISODES,
            curriculum=curriculum,
            plus_arm=None,
            minus_arm=None,
            verdict="STOP_MOTOR_COVERAGE",
            retry_policy="No seed, width, learning-rate, curriculum-order, or action-bank sweep.",
            claim_boundary="Target-free public mechanism probe; no protected data or coding benchmark evidence.",
        )

    plus_arm = _run_gate_arm(
        controller, name="fresh-plus-three", desired_outputs=_GATE_PLUS_EXPECTED
    )
    minus_arm = _run_gate_arm(
        controller, name="fresh-minus-three", desired_outputs=_GATE_MINUS_EXPECTED
    )
    passed = (
        plus_arm.selected_action == "OFFSET_ROOT[+3]"
        and minus_arm.selected_action == "OFFSET_ROOT[-3]"
        and plus_arm.passed
        and minus_arm.passed
        and plus_arm.checkpoint_before_sha256
        == plus_arm.checkpoint_after_sha256
        == curriculum.frozen_checkpoint_sha256
        and minus_arm.checkpoint_before_sha256
        == minus_arm.checkpoint_after_sha256
        == curriculum.frozen_checkpoint_sha256
    )
    return FixedGateResult(
        schema="public-typed-sensorimotor-edit-controller-v1",
        seed=SEED,
        motor_episodes=MOTOR_EPISODES,
        curriculum=curriculum,
        plus_arm=plus_arm,
        minus_arm=minus_arm,
        verdict=(
            "PASS_SENSORIMOTOR_INVERSE_EDIT_V1"
            if passed
            else "STOP_SENSORIMOTOR_INVERSE_EDIT_V1"
        ),
        retry_policy="No seed, width, learning-rate, curriculum-order, or action-bank sweep.",
        claim_boundary=(
            "Target-free motor-effect binding and two fresh public scalar root edits only; "
            "not composition, Simple-SWE, M0, held-out, C1, or broad coding evidence."
        ),
    )


def main() -> int:
    if _ARTIFACT_PATH.exists():
        stored = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "artifact": _ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix(),
                    "status": "already-spent",
                    "verdict": stored["verdict"],
                },
                sort_keys=True,
            )
        )
        return 0
    result = run_fixed_gate()
    _ARTIFACT_PATH.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": _ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix(),
                "verdict": result.verdict,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
