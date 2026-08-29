from dataclasses import fields

import numpy as np
import pytest

from experiments.r3_pilot import public_typed_output_cells as output


def test_compact_encodings_have_exact_storage_and_derived_scalar_state() -> None:
    integer_values = tuple(
        output.TypedOutputCellBank.encode(value)
        for value in (output.INT_MIN, 0, output.INT_MAX)
    )
    false_value = output.TypedOutputCellBank.encode(False)
    true_value = output.TypedOutputCellBank.encode(True)
    integer_one = output.TypedOutputCellBank.encode(1)

    assert [field.name for field in fields(output.TypedOutputEncoding)] == [
        "factor_winners_le",
        "physical_digest",
    ]
    assert len(integer_one.factor_winners_le) == 72
    assert len(integer_one.physical_digest) == 32
    assert not hasattr(integer_one, "cells")
    assert not hasattr(integer_one, "active_factor_mask")
    with pytest.raises(TypeError):
        output.TypedOutputEncoding()
    with pytest.raises(TypeError):
        output.TypedOutputEncoding(b"", b"")  # type: ignore[call-arg]

    assert len({encoding.sha256 for encoding in integer_values}) == 3
    assert all(
        encoding.output_type is output.OutputType.INT
        for encoding in integer_values
    )
    assert all(encoding.active_factors == ("type", "int") for encoding in integer_values)
    assert false_value.output_type is true_value.output_type is output.OutputType.BOOL
    assert integer_one.output_type is output.OutputType.INT
    assert true_value.sha256 != integer_one.sha256
    assert false_value.factor_winner(output.BOOL_FACTOR_SLOT) == output.BOOL_FALSE_CELL
    assert true_value.factor_winner(output.BOOL_FACTOR_SLOT) == output.BOOL_TRUE_CELL
    assert integer_one.factor_winner(output.INT_FACTOR_SLOT) == 1 - output.INT_MIN
    assert integer_one.factor_winner(output.BOOL_FACTOR_SLOT) is None


def test_lists_use_compact_factor_winners_and_local_physical_drives() -> None:
    encoded = output.TypedOutputCellBank.encode([1, -2])
    changed = output.TypedOutputCellBank.encode([1, 3])
    empty = output.TypedOutputCellBank.encode([])

    assert encoded.output_type is output.OutputType.LIST_INT
    assert encoded.supported
    assert encoded.active_factors == (
        "type",
        "list-length",
        "list-item-0",
        "list-item-1",
    )
    assert encoded.factor_winner(output.LIST_LENGTH_FACTOR_SLOT) == 2
    assert tuple(
        encoded.factor_winner(output.LIST_ITEM_FACTOR_START + position)
        for position in range(output.LIST_CAPACITY)
    ) == (65, 62, None, None)
    assert empty.active_factors == ("type", "list-length")

    indices = encoded.active_global_cell_indices()
    audit = output.TypedOutputCellBank.materialize_for_audit(encoded)
    changed_audit = output.TypedOutputCellBank.materialize_for_audit(changed)
    assert indices == tuple(int(index) for index in np.flatnonzero(audit.cells))
    assert indices == tuple(sorted(indices))
    difference = np.flatnonzero(audit.cells != changed_audit.cells)
    item_one_start = output.LIST_ITEM_OFFSET + output.INT_CELL_COUNT
    assert np.all(
        (difference >= item_one_start)
        & (difference < item_one_start + output.INT_CELL_COUNT)
    )

    active_drive = output.TypedOutputCellBank._drive_factor(
        encoded, output.LIST_ITEM_FACTOR_START, 2_500.0
    )
    inactive_drive = output.TypedOutputCellBank._drive_factor(
        encoded, output.BOOL_FACTOR_SLOT, 2_500.0
    )
    assert active_drive.dtype == np.dtype(np.float64)
    assert active_drive.flags.c_contiguous
    assert np.flatnonzero(active_drive).tolist() == [65]
    assert active_drive[65] == 2_500.0
    assert inactive_drive.shape == (output.BOOL_CELL_COUNT,)
    assert not np.any(inactive_drive)


def test_invalid_values_activate_only_canonical_unknown_winners() -> None:
    unsupported_int = output.INT_MAX + 1
    high_int = output.TypedOutputCellBank.encode(unsupported_int)
    bool_as_int = output.TypedOutputCellBank.encode(
        True, declared_type=output.OutputType.INT
    )
    bad_item = output.TypedOutputCellBank.encode([1, unsupported_int])
    overlong = output.TypedOutputCellBank.encode([0, 1, 2, 3, 4])
    unsupported = output.TypedOutputCellBank.encode(object())

    assert not high_int.supported and high_int.unknown_active
    assert high_int.factor_winner(output.INT_FACTOR_SLOT) == output.INT_UNKNOWN_CELL
    assert not bool_as_int.supported and bool_as_int.unknown_active
    assert bool_as_int.output_type is output.OutputType.INT
    assert bool_as_int.factor_winner(output.INT_FACTOR_SLOT) == output.INT_UNKNOWN_CELL
    assert not bad_item.supported and bad_item.unknown_active
    assert (
        bad_item.factor_winner(output.LIST_ITEM_FACTOR_START + 1)
        == output.INT_UNKNOWN_CELL
    )
    assert not overlong.supported and overlong.unknown_active
    assert (
        overlong.factor_winner(output.LIST_LENGTH_FACTOR_SLOT)
        == output.LIST_LENGTH_UNKNOWN_CELL
    )
    assert all(
        overlong.factor_winner(output.LIST_ITEM_FACTOR_START + position) is None
        for position in range(output.LIST_CAPACITY)
    )
    assert not unsupported.supported and unsupported.unknown_active
    assert unsupported.output_type is None
    assert unsupported.factor_winner(output.TYPE_FACTOR_SLOT) is None
    assert unsupported.active_factors == output.FACTOR_LABELS[1:]


def test_physical_winner_construction_is_canonical_and_type_checked() -> None:
    original = output.TypedOutputCellBank.encode("brain")
    winners = tuple(
        original.factor_winner(slot) for slot in range(output.FACTOR_COUNT)
    )
    rebuilt = output.TypedOutputCellBank._from_physical_winners(
        output.OutputType.STRING, winners
    )

    assert rebuilt == original
    assert output.TypedOutputCellBank.decode(rebuilt) == "brain"
    with pytest.raises(ValueError, match="output type"):
        output.TypedOutputCellBank._from_physical_winners(
            output.OutputType.INT, winners
        )
    bad_winners = list(winners)
    bad_winners[output.STRING_ITEM_FACTOR_START] = True  # type: ignore[list-item]
    with pytest.raises(TypeError, match="int or None"):
        output.TypedOutputCellBank._from_physical_winners(
            output.OutputType.STRING, bad_winners
        )


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
    invalid_result = invalid_target.detect(unsupported_int)
    assert not invalid_target.enabled
    assert invalid_result.matched_factor_cells == 0
    assert invalid_result.current_pa == -5_000.0
    assert not invalid_result.spiked
    assert not output.exact_typed_output_coincidence(1, True).spiked
    assert output.exact_typed_output_coincidence([], []).spiked


def test_projection_encoding_and_detector_hashes_match_dense_goldens() -> None:
    expected_encodings = (
        (output.INT_MIN, None, "17618d99834d3b46d00d9e491788cd1a23e4154cd7f2863d44f9513b2ec84cc0"),
        (0, None, "c015d83b1e7a767ab3ab458ef3e9a6b852f82a0880200008d85097632492e8ae"),
        (output.INT_MAX, None, "1d14624ece5c74546845ece621326356841769d366e037a3bc17a70a6e3ec51b"),
        (output.INT_MAX + 1, None, "5548fb0569248e668f9acf4832c408e30446c619b11c72776fa26e4f7cdb58a1"),
        (False, None, "87b9de08e030062130fc6e43c7aa0d53512a808da6cbc87afc79c6b80b7a361b"),
        (True, None, "0bf102c4a47c99c161fd9837f06f4063b0d308594a022c49e1d4f896c5fed9af"),
        ([2, -3], None, "f363ed350368f2f1467febcc785fec66c7b65a6f895041f983c5597b29090315"),
        ("~", output.OutputType.CHAR, "64307479b9a21f759f664f973502351a3781a189e37d4ce71effc1580d36a1dc"),
        ("brain", None, "f346c6d6e670b914d739f51a9f5bfe7c18e144b8ed328caf29b765b1957fd8b6"),
    )

    assert (
        output.TypedOutputCellBank.projection_sha256()
        == "f3ebca86516460b7a03ab5588cf26de2c6df07d0f5a109e8a6d5646eed015f44"
    )
    for value, declared_type, expected in expected_encodings:
        assert (
            output.TypedOutputCellBank.encode(
                value, declared_type=declared_type
            ).sha256
            == expected
        )
    assert (
        output.FixedTypedCoincidenceDetector.for_target([2, -3]).sha256
        == "a02f948039c66858e46f9d17f4a3b669b9d2186c0e097dcd618892f1c1b41dd9"
    )
    assert (
        output.FixedTypedCoincidenceDetector.for_target(2).sha256
        == "d1db5ed9ec00c86e94cdcd3de4955b1f789ef91bd78a46c715048b9a1c9879e7"
    )
