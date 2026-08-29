"""Total, cell-raced extension of the banked horizon-2 successor.

The banked successor learns target-free primitive transitions for raw values
``-20..20``.  Its 65 output cells tie when a pre/action row has no learned
synapse.  This module adds one explicit physical ``UNKNOWN`` cell.  A fixed
tonic current makes UNKNOWN the unique winner for unsupported rows; learned
known transitions still outrun it, and fixed UNKNOWN self-synapses propagate
the state through later plies.

No composite target, gate, artifact writer, or host-side fallback lives here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import (  # noqa: E402
    public_typed_horizon2_successor_planner as banked,
)
from experiments.r3_pilot.public_typed_sensorimotor_edit_controller import (  # noqa: E402
    _drive_spike_mask,
)
from humanbrain.cells import AdaptiveGlifPopulation  # noqa: E402


class UnknownSuccessor(Enum):
    """The single explicit state outside the learned raw transition support."""

    UNKNOWN = "UNKNOWN"


UNKNOWN = UnknownSuccessor.UNKNOWN
SuccessorState: TypeAlias = int | UnknownSuccessor

KNOWN_VALUE_COUNT = banked.VALUE_COUNT
UNKNOWN_CELL = KNOWN_VALUE_COUNT
TOTAL_STATE_COUNT = KNOWN_VALUE_COUNT + 1
STATE_LABELS = tuple(
    str(value) for value in range(banked.VALUE_MIN, banked.VALUE_MAX + 1)
) + (UNKNOWN.value,)

_ACTION_INDEX = {
    action.label: index for index, action in enumerate(banked.banked.ACTIONS)
}
_KNOWN_BASE_CURRENT_PA = banked._SUCCESSOR_BASE_CURRENT_PA
_UNKNOWN_TONIC_CURRENT_PA = 1_000.0
_UNKNOWN_PROPAGATION_WEIGHT_PA = banked._SUCCESSOR_MAXIMUM_WEIGHT_PA
_UNKNOWN_INPUT_CURRENT_PA = 5_000.0
_COINCIDENCE_SCALE_PA = banked._COINCIDENCE_SCALE_PA
_COINCIDENCE_LIMIT_PA = banked._COINCIDENCE_LIMIT_PA


class TotalValueCellEncoder:
    """Extend the fixed exact-value population with one physical UNKNOWN cell."""

    @classmethod
    def projection_sha256(cls) -> str:
        digest = sha256()
        digest.update(banked.ExactValueCellEncoder.projection_sha256().encode("ascii"))
        digest.update(str(TOTAL_STATE_COUNT).encode("ascii"))
        digest.update(str(_UNKNOWN_INPUT_CURRENT_PA).encode("ascii"))
        return digest.hexdigest()

    @classmethod
    def encode(cls, state: SuccessorState) -> np.ndarray:
        if state is UNKNOWN:
            currents = np.zeros(TOTAL_STATE_COUNT, dtype=np.float64)
            currents[UNKNOWN_CELL] = _UNKNOWN_INPUT_CURRENT_PA
            spikes = _drive_spike_mask(
                AdaptiveGlifPopulation(TOTAL_STATE_COUNT),
                currents,
                steps=banked._VALUE_PROJECTION_STEPS,
            ).astype(np.float64)
        elif type(state) is int:
            known = banked.ExactValueCellEncoder.encode(state)
            spikes = np.concatenate((known, np.zeros(1, dtype=np.float64)))
        else:
            raise TypeError("successor state must be a raw integer or UNKNOWN")
        spikes = np.ascontiguousarray(spikes, dtype=np.float64)
        if np.count_nonzero(spikes) != 1:
            raise banked.CellRaceFailure(
                "total exact-value projection did not produce one state cell"
            )
        return spikes


def _checkpoint_sha256(
    weights_pa: np.ndarray,
    bias_pa: np.ndarray,
    base_checkpoint_sha256: str,
) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(weights_pa, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(bias_pa, dtype=np.float64).tobytes())
    digest.update(base_checkpoint_sha256.encode("ascii"))
    digest.update(TotalValueCellEncoder.projection_sha256().encode("ascii"))
    digest.update("\n".join(STATE_LABELS).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenTotalSuccessor:
    """Read-only local transition banks with a total physical output race."""

    weights_pa: np.ndarray
    bias_pa: np.ndarray
    base_checkpoint_sha256: str
    training_digest: str
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _checkpoint_sha256(
            self.weights_pa,
            self.bias_pa,
            self.base_checkpoint_sha256,
        )

    def predict(
        self,
        state: SuccessorState,
        action: str,
    ) -> tuple[SuccessorState, banked.CellRace, str]:
        if action not in _ACTION_INDEX:
            raise ValueError("successor action is outside the fixed motor bank")
        before_sha = self.live_checkpoint_sha256
        if before_sha != self.checkpoint_sha256:
            raise RuntimeError("total successor checkpoint was mutated before prediction")
        features = TotalValueCellEncoder.encode(state)
        feature_sha = sha256(features.tobytes()).hexdigest()
        currents = np.ascontiguousarray(
            self.bias_pa + self.weights_pa[_ACTION_INDEX[action]] @ features
        )
        race = banked._physical_race(
            currents,
            STATE_LABELS,
            bank="total-successor-state",
        )
        predicted: SuccessorState = (
            UNKNOWN
            if race.winner_cell == UNKNOWN_CELL
            else banked.VALUE_MIN + race.winner_cell
        )
        after_sha = self.live_checkpoint_sha256
        if after_sha != before_sha:
            raise RuntimeError("frozen total successor changed during prediction")
        return predicted, race, feature_sha

    def predict_many(
        self,
        states: Sequence[SuccessorState],
        action: str,
    ) -> tuple[tuple[SuccessorState, ...], tuple[banked.CellRace, ...]]:
        predictions: list[SuccessorState] = []
        races: list[banked.CellRace] = []
        for state in states:
            predicted, race, _ = self.predict(state, action)
            predictions.append(predicted)
            races.append(race)
        return tuple(predictions), tuple(races)


@dataclass(frozen=True, slots=True)
class TotalSuccessorTrainingSummary:
    base_checkpoint_sha256: str
    checkpoint_sha256: str
    observations: int
    learned_known_synapses: int
    fixed_unknown_synapses: int
    known_pre_min: int
    known_pre_max: int
    state_cells: int
    unknown_cell: int
    unknown_tonic_current_pa: float
    unknown_propagation_weight_pa: float
    exact_value_projection_sha256: str
    claim_boundary: str


def train_public_total_successor(
) -> tuple[FrozenTotalSuccessor, TotalSuccessorTrainingSummary]:
    """Freeze the banked target-free local learner, then add UNKNOWN anatomy."""
    base, base_summary = banked.train_public_successor()
    weights = np.zeros(
        (len(banked.banked.ACTIONS), TOTAL_STATE_COUNT, TOTAL_STATE_COUNT),
        dtype=np.float64,
    )
    weights[:, :KNOWN_VALUE_COUNT, :KNOWN_VALUE_COUNT] = base.weights_pa
    # Fixed action-independent UNKNOWN propagation.  These are anatomy, not
    # observations or inferred host fallbacks.
    weights[:, UNKNOWN_CELL, UNKNOWN_CELL] = _UNKNOWN_PROPAGATION_WEIGHT_PA
    bias = np.full(TOTAL_STATE_COUNT, _KNOWN_BASE_CURRENT_PA, dtype=np.float64)
    bias[UNKNOWN_CELL] = _UNKNOWN_TONIC_CURRENT_PA

    weights = np.ascontiguousarray(weights)
    bias = np.ascontiguousarray(bias)
    checkpoint_sha = _checkpoint_sha256(
        weights,
        bias,
        base.checkpoint_sha256,
    )
    weights.setflags(write=False)
    bias.setflags(write=False)
    frozen = FrozenTotalSuccessor(
        weights_pa=weights,
        bias_pa=bias,
        base_checkpoint_sha256=base.checkpoint_sha256,
        training_digest=base.training_digest,
        checkpoint_sha256=checkpoint_sha,
    )
    return frozen, TotalSuccessorTrainingSummary(
        base_checkpoint_sha256=base.checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha,
        observations=base_summary.observations,
        learned_known_synapses=base_summary.learned_synapses,
        fixed_unknown_synapses=len(banked.banked.ACTIONS),
        known_pre_min=min(base_summary.values),
        known_pre_max=max(base_summary.values),
        state_cells=TOTAL_STATE_COUNT,
        unknown_cell=UNKNOWN_CELL,
        unknown_tonic_current_pa=_UNKNOWN_TONIC_CURRENT_PA,
        unknown_propagation_weight_pa=_UNKNOWN_PROPAGATION_WEIGHT_PA,
        exact_value_projection_sha256=TotalValueCellEncoder.projection_sha256(),
        claim_boundary=(
            "Target-free primitive support totalization only; no composite target, "
            "planner result, fresh boundary, or coding-capability evidence."
        ),
    )


def coincidence_features(
    predicted: Sequence[SuccessorState],
    desired: Sequence[int],
) -> tuple[np.ndarray, str]:
    """Return fixed physical equality cells; UNKNOWN never matches a raw target."""
    predicted_states = tuple(predicted)
    desired_values = tuple(desired)
    if not predicted_states or len(predicted_states) != len(desired_values):
        raise ValueError("coincidence needs paired predicted and desired states")
    traces: list[np.ndarray] = []
    for predicted_state, desired_value in zip(
        predicted_states,
        desired_values,
        strict=True,
    ):
        predicted_cells = TotalValueCellEncoder.encode(predicted_state)
        desired_cells = TotalValueCellEncoder.encode(desired_value)
        currents = np.ascontiguousarray(
            np.clip(
                (predicted_cells + desired_cells - 1.5) * _COINCIDENCE_SCALE_PA,
                -_COINCIDENCE_LIMIT_PA,
                _COINCIDENCE_LIMIT_PA,
            )
        )
        coincidence = _drive_spike_mask(
            AdaptiveGlifPopulation(TOTAL_STATE_COUNT),
            currents,
            steps=banked._VALUE_PROJECTION_STEPS,
        ).astype(np.float64)
        if np.count_nonzero(coincidence) > 1:
            raise banked.CellRaceFailure("total coincidence produced multiple cells")
        traces.append(np.ascontiguousarray(coincidence))
    features = np.ascontiguousarray(np.stack(traces))
    return features, sha256(features.tobytes()).hexdigest()
