from dataclasses import replace

import numpy as np
import pytest

from experiments.r3_pilot import public_typed_output_cells as output


@pytest.mark.parametrize(
    ("value", "declared_type"),
    (
        (output.INT_MIN, output.OutputType.INT),
        (True, output.OutputType.BOOL),
        ([1, -2], output.OutputType.LIST_INT),
        ("A", output.OutputType.CHAR),
        ("brain", output.OutputType.STRING),
    ),
)
def test_strict_decoder_round_trips_every_supported_output_type(
    value: object, declared_type: output.OutputType
) -> None:
    encoded = output.TypedOutputCellBank.encode(value, declared_type=declared_type)

    assert output.TypedOutputCellBank.decode(encoded) == value


def test_char_catalog_is_fixed_printable_ascii_and_char_is_declaration_only() -> None:
    assert output.PRINTABLE_ASCII == tuple(chr(code) for code in range(0x20, 0x7F))
    assert output.CHAR_UNKNOWN_CELL == 95
    assert output.CHAR_CELL_COUNT == 96

    char = output.TypedOutputCellBank.encode(
        "~", declared_type=output.OutputType.CHAR
    )
    inferred = output.TypedOutputCellBank.encode("~")

    assert char.output_type is output.OutputType.CHAR
    assert char.active_factors == ("type", "char")
    assert np.flatnonzero(char.char_cells).tolist() == [ord("~") - 0x20]
    assert output.TypedOutputCellBank.decode(char) == "~"
    assert inferred.output_type is output.OutputType.STRING
    assert inferred.active_factors == ("type", "string-length", "string-char-0")
    assert output.TypedOutputCellBank.decode(inferred) == "~"
    assert not char.cells.flags.writeable
    assert not char.active_factor_mask.flags.writeable
    assert not char.char_cells.flags.writeable


def test_repeated_physical_categories_reuse_bounded_readonly_cache_entries() -> None:
    output._physical_one_hot.cache_clear()
    first = output._physical_one_hot(output.CHAR_CELL_COUNT, ord("A") - 0x20)
    second = output._physical_one_hot(output.CHAR_CELL_COUNT, ord("A") - 0x20)
    before_string = output._physical_one_hot.cache_info()
    repeated = output.TypedOutputCellBank.encode("A" * output.STRING_CAPACITY)
    after_string = output._physical_one_hot.cache_info()

    assert first is second
    assert not first.flags.writeable
    assert output._physical_one_hot.cache_parameters() == {
        "maxsize": output._PHYSICAL_ONE_HOT_CACHE_SIZE,
        "typed": False,
    }
    assert after_string.hits - before_string.hits == output.STRING_CAPACITY
    assert output.TypedOutputCellBank.decode(repeated) == "A" * output.STRING_CAPACITY

    projection_a = output._int_projection()
    projection_b = output._int_projection()
    assert projection_a[0] is projection_b[0]
    assert projection_a[1] is projection_b[1]
    assert not projection_a[0].flags.writeable
    assert not projection_a[1].flags.writeable


def test_string_uses_length_and_twenty_six_independent_position_factors() -> None:
    text = "".join(output.PRINTABLE_ASCII[: output.STRING_CAPACITY])
    encoded = output.TypedOutputCellBank.encode(text)
    changed = output.TypedOutputCellBank.encode(text[:13] + "Z" + text[14:])
    empty = output.TypedOutputCellBank.encode("")

    assert output.STRING_CAPACITY == 26
    assert output.TOTAL_CELL_COUNT == (
        output.TYPE_CELL_COUNT
        + output.INT_CELL_COUNT
        + output.BOOL_CELL_COUNT
        + output.LIST_LENGTH_CELL_COUNT
        + output.LIST_CAPACITY * output.INT_CELL_COUNT
        + output.CHAR_CELL_COUNT
        + output.STRING_LENGTH_CELL_COUNT
        + output.STRING_CAPACITY * output.CHAR_CELL_COUNT
    )
    assert encoded.supported
    assert encoded.string_length_cells[output.STRING_CAPACITY] == 1.0
    assert tuple(np.count_nonzero(row) for row in encoded.string_item_cells) == (
        1,
    ) * output.STRING_CAPACITY
    assert encoded.active_factors == (
        "type",
        "string-length",
        *(f"string-char-{position}" for position in range(output.STRING_CAPACITY)),
    )
    assert output.TypedOutputCellBank.decode(encoded) == text

    difference = np.flatnonzero(encoded.cells != changed.cells)
    changed_start = output.STRING_ITEM_OFFSET + 13 * output.CHAR_CELL_COUNT
    assert np.all(
        (difference >= changed_start)
        & (difference < changed_start + output.CHAR_CELL_COUNT)
    )
    assert empty.active_factors == ("type", "string-length")
    assert not np.any(empty.string_item_cells)
    assert output.TypedOutputCellBank.decode(empty) == ""


def test_non_printable_and_overlong_values_use_physical_unknown_cells() -> None:
    bad_char = output.TypedOutputCellBank.encode(
        "\n", declared_type=output.OutputType.CHAR
    )
    multiple_chars = output.TypedOutputCellBank.encode(
        "AB", declared_type=output.OutputType.CHAR
    )
    bad_string = output.TypedOutputCellBank.encode("A\n")
    unicode_string = output.TypedOutputCellBank.encode("café")
    overlong = output.TypedOutputCellBank.encode("A" * (output.STRING_CAPACITY + 1))
    unsupported = output.TypedOutputCellBank.encode(object())

    assert not bad_char.supported and bad_char.unknown_active
    assert bad_char.char_cells[output.CHAR_UNKNOWN_CELL] == 1.0
    assert not multiple_chars.supported and multiple_chars.unknown_active
    assert not bad_string.supported and bad_string.unknown_active
    assert bad_string.string_item_cells[1, output.CHAR_UNKNOWN_CELL] == 1.0
    assert not unicode_string.supported and unicode_string.unknown_active
    assert unicode_string.string_item_cells[3, output.CHAR_UNKNOWN_CELL] == 1.0
    assert not overlong.supported and overlong.unknown_active
    assert overlong.string_length_cells[output.STRING_LENGTH_UNKNOWN_CELL] == 1.0
    assert not np.any(overlong.string_item_cells)
    assert not unsupported.supported and unsupported.unknown_active
    assert unsupported.char_cells[output.CHAR_UNKNOWN_CELL] == 1.0
    assert unsupported.string_length_cells[output.STRING_LENGTH_UNKNOWN_CELL] == 1.0
    assert np.all(
        unsupported.string_item_cells[:, output.CHAR_UNKNOWN_CELL] == 1.0
    )
    with pytest.raises(ValueError, match="UNKNOWN"):
        output.TypedOutputCellBank.decode(bad_string)


def test_char_and_string_exact_coincidence_stays_typed_and_positional() -> None:
    char_detector = output.FixedTypedCoincidenceDetector.for_target(
        "A", declared_type=output.OutputType.CHAR
    )
    exact_char = char_detector.detect("A", declared_type=output.OutputType.CHAR)
    wrong_char = char_detector.detect("a", declared_type=output.OutputType.CHAR)
    inferred_string = char_detector.detect("A")

    assert exact_char.spiked
    assert exact_char.matched_factor_cells == exact_char.active_target_factors == 2
    assert not wrong_char.spiked
    assert wrong_char.matched_factor_cells == 1
    assert not inferred_string.spiked

    detector = output.FixedTypedCoincidenceDetector.for_target("cat")
    exact = detector.detect("cat")
    positional_mismatch = detector.detect("car")
    length_mismatch = detector.detect("cats")

    assert exact.spiked
    assert exact.matched_factor_cells == exact.active_target_factors == 5
    assert not positional_mismatch.spiked
    assert positional_mismatch.matched_factor_cells == 4
    assert not length_mismatch.spiked
    assert length_mismatch.matched_factor_cells == 4


def test_decoder_rejects_noncanonical_encoding_and_projection_identity_is_fixed() -> None:
    encoded = output.TypedOutputCellBank.encode("brain")
    mutable = replace(encoded, cells=encoded.cells.copy())
    cells = encoded.cells.copy()
    inactive_position = 12
    cells[
        output.STRING_ITEM_OFFSET
        + inactive_position * output.CHAR_CELL_COUNT
        + (ord("X") - output.PRINTABLE_ASCII_MIN)
    ] = 1.0
    cells.setflags(write=False)
    forged = replace(encoded, cells=cells)

    with pytest.raises(ValueError, match="immutable"):
        output.TypedOutputCellBank.decode(mutable)
    with pytest.raises(ValueError, match="not canonical"):
        output.TypedOutputCellBank.decode(forged)
    with pytest.raises(TypeError, match="TypedOutputEncoding"):
        output.TypedOutputCellBank.decode("brain")  # type: ignore[arg-type]

    projection = output.TypedOutputCellBank.projection_sha256()
    assert projection == output.TypedOutputCellBank.projection_sha256()
    assert projection == "f3ebca86516460b7a03ab5588cf26de2c6df07d0f5a109e8a6d5646eed015f44"
