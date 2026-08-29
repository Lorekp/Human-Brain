from struct import pack_into

import numpy as np
import pytest

from experiments.r3_pilot import public_typed_output_cells as output


def _forge_encoding(
    factor_winners_le: object,
    physical_digest: object,
) -> output.TypedOutputEncoding:
    forged = object.__new__(output.TypedOutputEncoding)
    object.__setattr__(forged, "factor_winners_le", factor_winners_le)
    object.__setattr__(forged, "physical_digest", physical_digest)
    return forged


def _replace_winner(payload: bytes, slot: int, winner: int) -> bytes:
    changed = bytearray(payload)
    pack_into("<H", changed, slot * 2, winner)
    return bytes(changed)


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

    assert output.TypedOutputCellBank.validate(encoded) is encoded
    assert output.TypedOutputCellBank.decode(encoded) == value
    assert output.TypedOutputCellBank._decode_trusted(encoded) == value


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
    assert char.factor_winner(output.CHAR_FACTOR_SLOT) == ord("~") - 0x20
    assert output.TypedOutputCellBank.decode(char) == "~"
    assert inferred.output_type is output.OutputType.STRING
    assert inferred.active_factors == ("type", "string-length", "string-char-0")
    assert output.TypedOutputCellBank.decode(inferred) == "~"
    assert type(char.factor_winners_le) is bytes
    assert type(char.physical_digest) is bytes


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
    assert encoded.supported
    assert (
        encoded.factor_winner(output.STRING_LENGTH_FACTOR_SLOT)
        == output.STRING_CAPACITY
    )
    assert all(
        encoded.factor_winner(output.STRING_ITEM_FACTOR_START + position)
        == position
        for position in range(output.STRING_CAPACITY)
    )
    assert encoded.active_factors == (
        "type",
        "string-length",
        *(f"string-char-{position}" for position in range(output.STRING_CAPACITY)),
    )
    assert output.TypedOutputCellBank.decode(encoded) == text

    audit = output.TypedOutputCellBank.materialize_for_audit(encoded)
    changed_audit = output.TypedOutputCellBank.materialize_for_audit(changed)
    difference = np.flatnonzero(audit.cells != changed_audit.cells)
    changed_start = output.STRING_ITEM_OFFSET + 13 * output.CHAR_CELL_COUNT
    assert np.all(
        (difference >= changed_start)
        & (difference < changed_start + output.CHAR_CELL_COUNT)
    )
    assert audit.cells.shape == (output.TOTAL_CELL_COUNT,)
    assert audit.cells.dtype == np.dtype(np.float64)
    assert audit.cells.flags.c_contiguous and not audit.cells.flags.writeable
    assert audit.active_factor_mask.shape == (output.FACTOR_COUNT,)
    assert audit.active_factor_mask.dtype == np.dtype(np.bool_)
    assert (
        audit.active_factor_mask.flags.c_contiguous
        and not audit.active_factor_mask.flags.writeable
    )
    assert encoded.active_global_cell_indices() == tuple(
        int(cell) for cell in np.flatnonzero(audit.cells)
    )
    assert empty.active_factors == ("type", "string-length")
    for slot in range(output.FACTOR_COUNT):
        if empty.factor_winner(slot) is None:
            assert not np.any(
                output.TypedOutputCellBank._drive_factor(empty, slot, 5_000.0)
            )
    assert output.TypedOutputCellBank.decode(empty) == ""


def test_non_printable_and_overlong_values_use_physical_unknown_winners() -> None:
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
    assert bad_char.factor_winner(output.CHAR_FACTOR_SLOT) == output.CHAR_UNKNOWN_CELL
    assert not multiple_chars.supported and multiple_chars.unknown_active
    assert not bad_string.supported and bad_string.unknown_active
    assert (
        bad_string.factor_winner(output.STRING_ITEM_FACTOR_START + 1)
        == output.CHAR_UNKNOWN_CELL
    )
    assert not unicode_string.supported and unicode_string.unknown_active
    assert (
        unicode_string.factor_winner(output.STRING_ITEM_FACTOR_START + 3)
        == output.CHAR_UNKNOWN_CELL
    )
    assert not overlong.supported and overlong.unknown_active
    assert (
        overlong.factor_winner(output.STRING_LENGTH_FACTOR_SLOT)
        == output.STRING_LENGTH_UNKNOWN_CELL
    )
    assert all(
        overlong.factor_winner(output.STRING_ITEM_FACTOR_START + position) is None
        for position in range(output.STRING_CAPACITY)
    )
    assert not unsupported.supported and unsupported.unknown_active
    assert (
        unsupported.factor_winner(output.STRING_LENGTH_FACTOR_SLOT)
        == output.STRING_LENGTH_UNKNOWN_CELL
    )
    assert all(
        unsupported.factor_winner(output.STRING_ITEM_FACTOR_START + position)
        == output.CHAR_UNKNOWN_CELL
        for position in range(output.STRING_CAPACITY)
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


def test_validation_rejects_noncanonical_payloads_and_digest_forgery() -> None:
    encoded = output.TypedOutputCellBank.encode("brain")
    different = output.TypedOutputCellBank.encode("braim")
    inactive = 0xFFFF
    corruptions = (
        _forge_encoding(bytearray(encoded.factor_winners_le), encoded.physical_digest),
        _forge_encoding(encoded.factor_winners_le[:-2], encoded.physical_digest),
        _forge_encoding(encoded.factor_winners_le, encoded.physical_digest[:-1]),
        _forge_encoding(
            _replace_winner(
                encoded.factor_winners_le, output.INT_FACTOR_SLOT, output.INT_CELL_COUNT
            ),
            encoded.physical_digest,
        ),
        _forge_encoding(
            _replace_winner(encoded.factor_winners_le, output.TYPE_FACTOR_SLOT, inactive),
            encoded.physical_digest,
        ),
        _forge_encoding(
            _replace_winner(
                encoded.factor_winners_le, output.STRING_LENGTH_FACTOR_SLOT, inactive
            ),
            encoded.physical_digest,
        ),
        _forge_encoding(
            _replace_winner(
                encoded.factor_winners_le,
                output.STRING_ITEM_FACTOR_START + 12,
                0,
            ),
            encoded.physical_digest,
        ),
        _forge_encoding(
            _replace_winner(encoded.factor_winners_le, output.TYPE_FACTOR_SLOT, 0),
            encoded.physical_digest,
        ),
        _forge_encoding(
            encoded.factor_winners_le,
            bytes((encoded.physical_digest[0] ^ 1,)) + encoded.physical_digest[1:],
        ),
        _forge_encoding(different.factor_winners_le, encoded.physical_digest),
    )

    for forged in corruptions:
        with pytest.raises((TypeError, ValueError)):
            output.TypedOutputCellBank.validate(forged)
    with pytest.raises(TypeError, match="TypedOutputEncoding"):
        output.TypedOutputCellBank.decode("brain")  # type: ignore[arg-type]


def test_all_external_encoding_ingress_rejects_before_physical_work(monkeypatch) -> None:
    encoded = output.TypedOutputCellBank.encode("brain")
    forged = _forge_encoding(
        encoded.factor_winners_le,
        bytes((encoded.physical_digest[0] ^ 1,)) + encoded.physical_digest[1:],
    )
    detector = output.FixedTypedCoincidenceDetector.for_target(encoded)

    def forbidden_drive(*args, **kwargs):
        raise AssertionError("physical work ran before compact validation")

    monkeypatch.setattr(output, "_drive_spike_mask", forbidden_drive)
    with pytest.raises(ValueError, match="digest"):
        output.TypedOutputCellBank.decode(forged)
    with pytest.raises(ValueError, match="digest"):
        output.FixedTypedCoincidenceDetector.for_target(forged)
    with pytest.raises(ValueError, match="digest"):
        detector.detect(forged)


def test_production_core_paths_never_call_audit_materialization(monkeypatch) -> None:
    encoded = output.TypedOutputCellBank.encode("brain")
    detector = output.FixedTypedCoincidenceDetector.for_target(encoded)

    def forbidden_materializer(*args, **kwargs):
        raise AssertionError("production path materialized the dense projection")

    monkeypatch.setattr(
        output.TypedOutputCellBank,
        "materialize_for_audit",
        forbidden_materializer,
    )
    assert output.TypedOutputCellBank.decode(encoded) == "brain"
    assert detector.detect(encoded).spiked
