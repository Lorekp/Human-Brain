import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output


def test_supported_encodings_are_unique_and_bool_precedes_int() -> None:
    integer_values = tuple(
        output.TypedOutputCellBank.encode(value)
        for value in (output.INT_MIN, 0, output.INT_MAX)
    )
    false_value = output.TypedOutputCellBank.encode(False)
    true_value = output.TypedOutputCellBank.encode(True)
    integer_one = output.TypedOutputCellBank.encode(1)

    assert len({encoding.sha256 for encoding in integer_values}) == 3
    assert all(encoding.output_type is output.OutputType.INT for encoding in integer_values)
    assert all(np.count_nonzero(encoding.cells) == 2 for encoding in integer_values)
    assert false_value.output_type is true_value.output_type is output.OutputType.BOOL
    assert integer_one.output_type is output.OutputType.INT
    assert true_value.sha256 != integer_one.sha256
    assert np.count_nonzero(true_value.bool_cells) == 1
    assert np.count_nonzero(true_value.int_cells) == 0
    assert np.count_nonzero(integer_one.int_cells) == 1
    assert np.count_nonzero(integer_one.bool_cells) == 0


def test_lists_use_factor_banks_and_physically_mask_inactive_positions() -> None:
    encoded = output.TypedOutputCellBank.encode([1, -2])
    changed = output.TypedOutputCellBank.encode([1, 3])
    empty = output.TypedOutputCellBank.encode([])

    assert output.TOTAL_CELL_COUNT == (
        output.TYPE_CELL_COUNT
        + output.INT_CELL_COUNT
        + output.BOOL_CELL_COUNT
        + output.LIST_LENGTH_CELL_COUNT
        + output.LIST_CAPACITY * output.INT_CELL_COUNT
    )
    assert encoded.output_type is output.OutputType.LIST_INT
    assert encoded.supported
    assert encoded.active_factors == (
        "type",
        "list-length",
        "list-item-0",
        "list-item-1",
    )
    assert tuple(np.count_nonzero(row) for row in encoded.list_item_cells) == (
        1,
        1,
        0,
        0,
    )
    assert not np.any(encoded.list_item_cells[2:])
    assert not np.any(empty.list_item_cells)
    assert empty.active_factors == ("type", "list-length")
    difference = np.flatnonzero(encoded.cells != changed.cells)
    item_one_start = output.LIST_ITEM_OFFSET + output.INT_CELL_COUNT
    assert np.all(
        (difference >= item_one_start)
        & (difference < item_one_start + output.INT_CELL_COUNT)
    )


def test_invalid_values_activate_physical_unknown_without_aliasing_bool() -> None:
    unsupported_int = output.INT_MAX + 1
    high_int = output.TypedOutputCellBank.encode(unsupported_int)
    bool_as_int = output.TypedOutputCellBank.encode(
        True, declared_type=output.OutputType.INT
    )
    bad_item = output.TypedOutputCellBank.encode([1, unsupported_int])
    overlong = output.TypedOutputCellBank.encode([0, 1, 2, 3, 4])
    unsupported = output.TypedOutputCellBank.encode("not-a-typed-output")

    assert not high_int.supported and high_int.unknown_active
    assert high_int.int_cells[output.INT_UNKNOWN_CELL] == 1.0
    assert not bool_as_int.supported and bool_as_int.unknown_active
    assert bool_as_int.output_type is output.OutputType.INT
    assert bool_as_int.int_cells[output.INT_UNKNOWN_CELL] == 1.0
    assert not bad_item.supported and bad_item.unknown_active
    assert bad_item.list_item_cells[1, output.INT_UNKNOWN_CELL] == 1.0
    assert not overlong.supported and overlong.unknown_active
    assert overlong.list_length_cells[output.LIST_LENGTH_UNKNOWN_CELL] == 1.0
    assert not np.any(overlong.list_item_cells)
    assert not unsupported.supported and unsupported.unknown_active
    assert unsupported.output_type is None
    assert not np.any(unsupported.type_cells)


def test_fixed_physical_coincidence_distinguishes_exact_mismatch_and_unknown() -> None:
    unsupported_int = output.INT_MAX + 1
    detector = output.FixedTypedCoincidenceDetector.for_target([1, -2])

    exact = detector.detect([1, -2])
    mismatch = detector.detect([1, -1])
    unknown = detector.detect([1, unsupported_int])
    wrong_type = detector.detect(True)

    assert exact.spiked
    assert exact.matched_factor_cells == exact.active_target_factors == 4
    assert exact.current_pa == 5_000.0
    assert not mismatch.spiked
    assert mismatch.matched_factor_cells == 3
    assert mismatch.current_pa == -5_000.0
    assert not unknown.spiked
    assert not wrong_type.spiked

    invalid_target = output.FixedTypedCoincidenceDetector.for_target(unsupported_int)
    assert not invalid_target.enabled
    assert not invalid_target.detect(unsupported_int).spiked
    assert not output.exact_typed_output_coincidence(1, True).spiked
    assert output.exact_typed_output_coincidence([], []).spiked


def test_projection_encoding_and_detector_hashes_are_deterministic() -> None:
    bank_hash_a = output.TypedOutputCellBank.projection_sha256()
    bank_hash_b = output.TypedOutputCellBank.projection_sha256()
    encoded_a = output.TypedOutputCellBank.encode([2, -3])
    encoded_b = output.TypedOutputCellBank.encode([2, -3])
    detector_a = output.FixedTypedCoincidenceDetector.for_target(encoded_a)
    detector_b = output.FixedTypedCoincidenceDetector.for_target(encoded_b)
    different = output.FixedTypedCoincidenceDetector.for_target([-3, 2])

    assert len(bank_hash_a) == 64
    assert bank_hash_a == bank_hash_b
    assert encoded_a.sha256 == encoded_b.sha256
    assert detector_a.sha256 == detector_b.sha256
    assert detector_a.sha256 != different.sha256
    result = detector_a.detect(encoded_b)
    assert result.spiked
    assert result.detector_sha256 == detector_a.sha256
    assert result.predicted_sha256 == encoded_b.sha256
