from __future__ import annotations

from functools import cache

import pytest

from experiments.r3_pilot import public_typed_operator_successor as successor
from experiments.r3_pilot import public_typed_output_cells as output
from humanbrain.typed_program import FactorAction


_DONE = FactorAction((), "DONE", "NONE", "NONE", "NONE", None)
_STRING_LENGTH_FACTOR_SLOT = output.FACTOR_LABELS.index("string-length")
_STRING_ITEM_FACTOR_SLOTS = tuple(
    output.FACTOR_LABELS.index(f"string-char-{position}")
    for position in range(output.STRING_CAPACITY)
)


@cache
def _model():
    return successor.train_public_typed_operator_successor()[0]


@pytest.mark.parametrize(
    ("value", "declared_type", "banks"),
    (
        ("A", output.OutputType.CHAR, ("copy-type", "copy-char")),
        (
            "brain",
            output.OutputType.STRING,
            (
                "copy-type",
                "copy-string-length",
                "copy-string-char-0",
                "copy-string-char-1",
                "copy-string-char-2",
                "copy-string-char-3",
                "copy-string-char-4",
            ),
        ),
    ),
)
def test_done_copies_text_through_type_length_and_positional_races(
    value: str,
    declared_type: output.OutputType,
    banks: tuple[str, ...],
) -> None:
    model = _model()
    encoded = output.TypedOutputCellBank.encode(value, declared_type=declared_type)
    before = model.live_checkpoint_sha256

    prediction = model.predict(encoded, _DONE)

    assert prediction.output.sha256 == encoded.sha256
    assert output.TypedOutputCellBank.decode(prediction.output) == value
    assert tuple(race.bank for race in prediction.races) == banks
    assert all(race.simultaneous_spike_candidates == 1 for race in prediction.races)
    assert prediction.checkpoint_before_sha256 == before
    assert prediction.checkpoint_after_sha256 == before


def test_copy_preserves_a_string_with_a_physical_unknown_character() -> None:
    model = _model()
    encoded = output.TypedOutputCellBank.encode(
        "A\nB",
        declared_type=output.OutputType.STRING,
    )

    prediction = model.predict(encoded, _DONE)

    assert prediction.output.sha256 == encoded.sha256
    assert prediction.output.supported is False
    assert prediction.output.factor_winner(_STRING_LENGTH_FACTOR_SLOT) == 3
    assert (
        prediction.output.factor_winner(_STRING_ITEM_FACTOR_SLOTS[1])
        == output.CHAR_UNKNOWN_CELL
    )
    assert tuple(race.bank for race in prediction.races) == (
        "copy-type",
        "copy-string-length",
        "copy-string-char-0",
        "copy-string-char-1",
        "copy-string-char-2",
    )
