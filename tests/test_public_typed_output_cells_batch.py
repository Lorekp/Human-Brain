from __future__ import annotations

from dataclasses import asdict, fields
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


def test_four_current_physical_kernel_is_scalar_exact(monkeypatch) -> None:
    currents = np.asarray((-5_000.0, 2_500.0, 4_375.0, 5_000.0))
    scalar = tuple(
        bool(output._drive_spike_mask(np.asarray((current,)))[0])
        for current in currents
    )

    driven_shapes: list[tuple[int, ...]] = []
    original_kernel = output._OUTPUT_SPIKE_KERNEL

    class TrackingKernel:
        def accumulated_spike_mask(self, driven: np.ndarray) -> np.ndarray:
            driven_shapes.append(driven.shape)
            return original_kernel.accumulated_spike_mask(driven)

    monkeypatch.setattr(output, "_OUTPUT_SPIKE_KERNEL", TrackingKernel())
    batched = output._drive_spike_mask(currents)

    assert driven_shapes == [(4,)]
    assert tuple(bool(value) for value in batched) == (
        False,
        False,
        True,
        True,
    )
    assert tuple(bool(value) for value in batched) == scalar


def test_detector_has_exact_compact_storage_and_no_public_constructor() -> None:
    detector = output.FixedTypedCoincidenceDetector.for_target([1, -2])

    assert [field.name for field in fields(type(detector))] == [
        "target",
        "physical_digest",
    ]
    assert type(detector.target) is output.TypedOutputEncoding
    assert len(detector.physical_digest) == 32
    assert not hasattr(detector, "weights")
    try:
        output.FixedTypedCoincidenceDetector()
    except TypeError:
        pass
    else:  # pragma: no cover - documents the constructor prohibition.
        raise AssertionError("detector constructor must remain bank-owned")
    try:
        output.FixedTypedCoincidenceDetector(  # type: ignore[call-arg]
            detector.target, detector.physical_digest
        )
    except TypeError:
        pass
    else:  # pragma: no cover - documents the constructor prohibition.
        raise AssertionError("detector constructor must remain bank-owned")


def test_detect_many_preserves_scalar_and_trusted_batch_results(monkeypatch) -> None:
    detector = output.FixedTypedCoincidenceDetector.for_target([1, -2])
    raw_predictions = ([1, -2], [1, -1], [1, -2], [1, output.INT_MAX + 1], True)
    encoded = tuple(
        output.TypedOutputCellBank.encode(prediction)
        for prediction in raw_predictions
    )

    scalar = tuple(detector.detect(prediction) for prediction in encoded)
    batched = detector.detect_many(encoded)

    assert tuple(result.spiked for result in batched) == (
        True,
        False,
        True,
        False,
        False,
    )
    assert tuple(result.matched_factor_cells for result in batched) == (4, 3, 4, 3, 0)
    assert tuple(map(_result_key, batched)) == tuple(map(_result_key, scalar))

    def forbidden_validate(*args, **kwargs):
        raise AssertionError("trusted batch repeated external validation")

    monkeypatch.setattr(output.TypedOutputCellBank, "validate", forbidden_validate)
    trusted = detector._detect_many_trusted(encoded)
    assert tuple(map(_result_key, trusted)) == tuple(map(_result_key, scalar))
    assert detector._detect_many_trusted(()) == ()


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
            self.target = detector.target
            self.sha256 = detector.sha256

        @classmethod
        def for_target(cls, target):
            return cls(original_detector.for_target(target))

        def detect_many(self, predictions):
            return tuple(self.detector.detect(predicted) for predicted in predictions)

        def _detect_many_trusted(self, predictions):
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
