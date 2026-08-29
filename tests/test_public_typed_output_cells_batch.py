from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race


def _result_key(result: output.TypedCoincidenceResult) -> tuple[object, ...]:
    return (
        result.spiked,
        result.spike_mask.dtype.str,
        result.spike_mask.shape,
        result.spike_mask.tobytes(),
        result.current_pa,
        result.matched_factor_cells,
        result.active_target_factors,
        result.detector_sha256,
        result.predicted_sha256,
    )


def _regime_detector(
    predictions: tuple[output.TypedOutputEncoding, ...],
) -> output.FixedTypedCoincidenceDetector:
    # These four exact currents cover inhibition, a positive subthreshold row,
    # the first coarse threshold bin used by this fixture, and saturation.
    currents_pa = (-5_000.0, 2_500.0, 4_375.0, 5_000.0)
    weights = np.zeros(output.TOTAL_CELL_COUNT, dtype=np.float64)
    for encoded, current_pa in zip(predictions, currents_pa, strict=True):
        cell = int(np.flatnonzero(encoded.int_cells)[0])
        weights[output.INT_OFFSET + cell] = current_pa / 10_000.0
    weights.setflags(write=False)
    return output.FixedTypedCoincidenceDetector(
        weights=weights,
        bias=0.0,
        active_target_factors=1,
        enabled=True,
        target_sha256="physical-regime-fixture",
        sha256="physical-regime-detector",
    )


def test_detect_many_is_scalar_exact_across_physical_current_regimes(
    monkeypatch,
) -> None:
    predictions = tuple(
        output.TypedOutputCellBank.encode(value) for value in (-3, -2, -1, 0)
    )
    detector = _regime_detector(predictions)
    scalar = tuple(detector.detect(predicted) for predicted in predictions)

    population_widths: list[int] = []
    original_population = output.AdaptiveGlifPopulation

    class TrackingPopulation(original_population):
        def __init__(self, cell_count: int, *args, **kwargs) -> None:
            population_widths.append(cell_count)
            super().__init__(cell_count, *args, **kwargs)

    monkeypatch.setattr(output, "AdaptiveGlifPopulation", TrackingPopulation)
    batched = detector.detect_many(predictions)

    assert population_widths == [4]
    assert tuple(result.current_pa for result in batched) == (
        -5_000.0,
        2_500.0,
        4_375.0,
        5_000.0,
    )
    assert tuple(result.spiked for result in batched) == (False, False, True, True)
    assert tuple(map(_result_key, batched)) == tuple(map(_result_key, scalar))
    assert detector.detect_many(()) == ()


def test_detect_many_preserves_repeated_exact_and_mixed_mismatch_rows() -> None:
    detector = output.FixedTypedCoincidenceDetector.for_target([1, -2])
    predictions = ([1, -2], [1, -1], [1, -2], [1, output.INT_MAX + 1], True)

    scalar = tuple(detector.detect(predicted) for predicted in predictions)
    batched = detector.detect_many(predictions)

    assert tuple(result.spiked for result in batched) == (
        True,
        False,
        True,
        False,
        False,
    )
    assert tuple(result.matched_factor_cells for result in batched) == (4, 3, 4, 3, 0)
    assert tuple(map(_result_key, batched)) == tuple(map(_result_key, scalar))


def test_semantic_race_batch_is_exactly_the_scalar_audit(monkeypatch) -> None:
    encode = output.TypedOutputCellBank.encode
    desired = (encode(2), encode(-3))
    keys = ("wrong-right", "exact", "wrong-left")
    candidates = (
        (encode(2), encode(-2)),
        (encode(2), encode(-3)),
        (encode(1), encode(-3)),
    )

    batched = semantic_race.select_exact_semantic_vector(keys, candidates, desired)
    original_detector = output.FixedTypedCoincidenceDetector

    class ScalarDetector:
        def __init__(self, detector: output.FixedTypedCoincidenceDetector) -> None:
            self.detector = detector
            self.sha256 = detector.sha256

        @classmethod
        def for_target(cls, target):
            return cls(original_detector.for_target(target))

        def detect_many(self, predictions):
            return tuple(self.detector.detect(predicted) for predicted in predictions)

    monkeypatch.setattr(
        semantic_race.output_cells,
        "FixedTypedCoincidenceDetector",
        ScalarDetector,
    )
    scalar = semantic_race.select_exact_semantic_vector(keys, candidates, desired)

    assert asdict(batched) == asdict(scalar)
    assert batched.sha256 == "26914d87230dc316fb17316f2fb5c20857d3c7565ea7d3714c614789a057d3d7"
    assert (
        output.TypedOutputCellBank.projection_sha256()
        == "f3ebca86516460b7a03ab5588cf26de2c6df07d0f5a109e8a6d5646eed015f44"
    )
    assert (
        original_detector.for_target(desired[0]).sha256
        == "d1db5ed9ec00c86e94cdcd3de4955b1f789ef91bd78a46c715048b9a1c9879e7"
    )


def test_batched_physical_phase_microbenchmark() -> None:
    currents = np.tile(
        np.asarray((-5_000.0, 2_500.0, 4_375.0, 5_000.0)),
        16,
    )

    def scalar_run() -> np.ndarray:
        return np.asarray(
            [
                output._drive_spike_mask(np.asarray((current,)))[0]
                for current in currents
            ]
        )

    def batch_run() -> np.ndarray:
        return output._drive_spike_mask(currents)

    scalar_run()
    batch_run()
    scalar_start = perf_counter()
    scalar = scalar_run()
    scalar_seconds = perf_counter() - scalar_start
    batch_start = perf_counter()
    batched = batch_run()
    batch_seconds = perf_counter() - batch_start

    assert np.array_equal(batched, scalar)
    assert batch_seconds < scalar_seconds
    speedup = scalar_seconds / batch_seconds
    print(
        "typed coincidence physical phase: "
        f"rows={len(currents)} scalar={scalar_seconds:.6f}s "
        f"batch={batch_seconds:.6f}s speedup={speedup:.2f}x"
    )
