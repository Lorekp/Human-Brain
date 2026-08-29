"""Fixed physical value scoring for complete factor-action candidates.

Each candidate receives only its own predicted typed-output cells and exact
target-coincidence spikes.  Fixed synapses reward supported typed states and
exact matches while strongly inhibiting any physical UNKNOWN factor.  Raw
currents can feed either a strict unique race or tie-inclusive tier planning.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.resting_glif_race import CompiledRestingGlifRace
from humanbrain.typed_program import FactorAction


_BASE_CURRENT_PA = 600.0
_SUPPORTED_CASE_WEIGHT_PA = 100.0
_EXACT_MATCH_WEIGHT_PA = 1_200.0
_UNKNOWN_INHIBITION_PA = 10_000.0
_STOP_CURRENT_PA = 650.0
_RACE_STEPS = 600

_FEATURE_COUNT = output_cells.TOTAL_CELL_COUNT + 1
_COINCIDENCE_FEATURE = output_cells.TOTAL_CELL_COUNT
_RACE = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _freeze(array: np.ndarray) -> np.ndarray:
    frozen = np.ascontiguousarray(array, dtype=np.float64)
    frozen.setflags(write=False)
    return frozen


def _action_record(action: FactorAction) -> dict[str, object]:
    return asdict(action)


def _unknown_cell_indices() -> tuple[int, ...]:
    return (
        output_cells.INT_OFFSET + output_cells.INT_UNKNOWN_CELL,
        output_cells.BOOL_OFFSET + output_cells.BOOL_UNKNOWN_CELL,
        output_cells.LIST_LENGTH_OFFSET + output_cells.LIST_LENGTH_UNKNOWN_CELL,
        *tuple(
            output_cells.LIST_ITEM_OFFSET
            + position * output_cells.INT_CELL_COUNT
            + output_cells.INT_UNKNOWN_CELL
            for position in range(output_cells.LIST_CAPACITY)
        ),
    )


@dataclass(frozen=True, slots=True)
class TypedFactorValueFeatures:
    cells: np.ndarray
    sha256: str
    coincidence_spikes: int


@dataclass(frozen=True, slots=True)
class TypedFactorValueDecision:
    status: str
    selected_candidate_cell: int | None
    selected_action: FactorAction | None
    currents_pa: tuple[float, ...]
    winner_cell: int
    spike_step: int
    simultaneous_spike_candidates: int
    feature_sha256: str
    projection_sha256: str


@dataclass(frozen=True, slots=True)
class FixedTypedFactorValueProjection:
    candidate_bank: tuple[FactorAction, ...]
    desired: tuple[output_cells.TypedOutputEncoding, ...]
    weights_pa: np.ndarray
    bias_pa: np.ndarray
    stop_current_pa: float
    sha256: str

    @classmethod
    def build(
        cls,
        candidate_bank: Sequence[FactorAction],
        desired: Sequence[output_cells.TypedOutputEncoding],
    ) -> FixedTypedFactorValueProjection:
        candidates = tuple(candidate_bank)
        targets = tuple(desired)
        if not candidates or not all(type(action) is FactorAction for action in candidates):
            raise TypeError("value projection needs a nonempty FactorAction bank")
        if len(set(candidates)) != len(candidates):
            raise ValueError("value projection candidate bank must be unique")
        if not targets or not all(
            isinstance(target, output_cells.TypedOutputEncoding)
            and target.supported
            for target in targets
        ):
            raise ValueError("value projection needs supported physical targets")

        weights = np.zeros(
            (len(candidates), len(targets), _FEATURE_COUNT),
            dtype=np.float64,
        )
        weights[:, :, output_cells.TYPE_SLICE] = _SUPPORTED_CASE_WEIGHT_PA
        weights[:, :, _COINCIDENCE_FEATURE] = _EXACT_MATCH_WEIGHT_PA
        weights[:, :, _unknown_cell_indices()] = -_UNKNOWN_INHIBITION_PA
        bias = np.full(len(candidates), _BASE_CURRENT_PA, dtype=np.float64)
        frozen_weights = _freeze(weights)
        frozen_bias = _freeze(bias)
        digest = sha256()
        digest.update(b"public-typed-factor-value-projection-v1\0")
        digest.update(_canonical_json(tuple(_action_record(item) for item in candidates)))
        digest.update(_canonical_json(tuple(target.sha256 for target in targets)))
        digest.update(frozen_weights.tobytes())
        digest.update(frozen_bias.tobytes())
        digest.update(np.asarray((_STOP_CURRENT_PA,), dtype=np.float64).tobytes())
        return cls(
            candidate_bank=candidates,
            desired=targets,
            weights_pa=frozen_weights,
            bias_pa=frozen_bias,
            stop_current_pa=_STOP_CURRENT_PA,
            sha256=digest.hexdigest(),
        )

    def features(
        self,
        predictions: Sequence[Sequence[output_cells.TypedOutputEncoding]],
    ) -> TypedFactorValueFeatures:
        rows = tuple(tuple(row) for row in predictions)
        expected = (len(self.candidate_bank), len(self.desired))
        if len(rows) != expected[0] or any(len(row) != expected[1] for row in rows):
            raise ValueError("predicted outputs must align with candidates and cases")
        if not all(
            isinstance(value, output_cells.TypedOutputEncoding)
            for row in rows
            for value in row
        ):
            raise TypeError("predictions must be physical typed-output encodings")

        cells = np.zeros((*expected, _FEATURE_COUNT), dtype=np.float64)
        coincidence_count = 0
        detectors = tuple(
            output_cells.FixedTypedCoincidenceDetector.for_target(target)
            for target in self.desired
        )
        for candidate_cell, row in enumerate(rows):
            for case, (predicted, detector) in enumerate(
                zip(row, detectors, strict=True)
            ):
                cells[candidate_cell, case, : output_cells.TOTAL_CELL_COUNT] = (
                    predicted.cells
                )
                coincidence = detector.detect(predicted)
                if coincidence.spiked:
                    cells[candidate_cell, case, _COINCIDENCE_FEATURE] = 1.0
                    coincidence_count += 1
        frozen = _freeze(cells)
        return TypedFactorValueFeatures(
            cells=frozen,
            sha256=sha256(frozen.tobytes()).hexdigest(),
            coincidence_spikes=coincidence_count,
        )

    def tier_currents(
        self,
        features: TypedFactorValueFeatures,
    ) -> np.ndarray:
        expected = (
            len(self.candidate_bank),
            len(self.desired),
            _FEATURE_COUNT,
        )
        if features.cells.shape != expected:
            raise ValueError("typed value features do not match this projection")
        candidate_currents = self.bias_pa + np.einsum(
            "ckf,ckf->c",
            self.weights_pa,
            features.cells,
        )
        return np.ascontiguousarray(
            np.concatenate(
                (candidate_currents, np.asarray((self.stop_current_pa,)))
            )
        )

    def select_unique(
        self,
        predictions: Sequence[Sequence[output_cells.TypedOutputEncoding]],
    ) -> TypedFactorValueDecision:
        features = self.features(predictions)
        currents = self.tier_currents(features)
        result = _RACE.race_many(currents[None, :])[0]
        stop_cell = len(self.candidate_bank)
        selected = None if result.winner_cell == stop_cell else result.winner_cell
        return TypedFactorValueDecision(
            status="STOP_ALL_UNKNOWN" if selected is None else "SELECTED",
            selected_candidate_cell=selected,
            selected_action=None if selected is None else self.candidate_bank[selected],
            currents_pa=tuple(float(value) for value in currents),
            winner_cell=result.winner_cell,
            spike_step=result.spike_step,
            simultaneous_spike_candidates=result.simultaneous_spike_candidates,
            feature_sha256=features.sha256,
            projection_sha256=self.sha256,
        )


__all__ = [
    "FixedTypedFactorValueProjection",
    "TypedFactorValueDecision",
    "TypedFactorValueFeatures",
]
