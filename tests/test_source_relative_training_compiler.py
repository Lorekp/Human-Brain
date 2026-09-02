from __future__ import annotations

import pytest
from humanbrain.source_relative_action_tape import (
    OccurrencePreservingActionTape,
    TapeOperation,
    TapeValence,
)
from humanbrain.source_relative_training_compiler import (
    SourceRelativeTrainingCompilerError,
    compile_successful_token_tapes,
)


def _tokens(prefix: int) -> tuple[bytes, bytes, bytes, bytes]:
    return tuple(bytes((prefix, suffix)) for suffix in (0x11, 0x22, 0x33, 0x44))


def test_one_distinct_reorder_pair_transfers_to_fresh_renamed_tokens() -> None:
    context = bytes.fromhex("c710")
    before = _tokens(0xD7)
    after = (before[2], before[3], before[0], before[1])

    demonstration = compile_successful_token_tapes(
        context_event=context,
        before_tokens=before,
        after_tokens=after,
    )

    assert demonstration.cue_inventory == (before[2], before[0])
    assert tuple(step.operation for step in demonstration.motor_steps) == (
        TapeOperation.REBIND,
        TapeOperation.COPY,
        TapeOperation.EMIT,
        TapeOperation.ADVANCE,
        TapeOperation.COPY,
        TapeOperation.EMIT,
        TapeOperation.REBIND,
        TapeOperation.COPY,
        TapeOperation.EMIT,
        TapeOperation.ADVANCE,
        TapeOperation.COPY,
        TapeOperation.EMIT,
        TapeOperation.END,
    )
    assert tuple(
        step.cue_ordinal
        for step in demonstration.motor_steps
        if step.operation is TapeOperation.REBIND
    ) == (0, 1)

    # Two presentations of the same single distinct successful pair provide
    # enough recurrent support to outrun the tape's tonic STOP cell.
    model = OccurrencePreservingActionTape.grow((demonstration, demonstration))
    replay = model.respond(
        (context,),
        demonstration.cue_inventory,
        before,
    )
    assert replay.emissions == after
    model.resolve(TapeValence.NEUTRAL)

    fresh_tokens = _tokens(0xE7)
    fresh_source = (
        bytes.fromhex("e7aa"),
        fresh_tokens[2],
        fresh_tokens[3],
        bytes.fromhex("e7bb"),
        fresh_tokens[0],
        fresh_tokens[1],
    )
    fresh_cues = (fresh_tokens[2], fresh_tokens[0])
    transfer = model.respond((context,), fresh_cues, fresh_source)

    assert transfer.emissions == (
        fresh_tokens[2],
        fresh_tokens[3],
        fresh_tokens[0],
        fresh_tokens[1],
    )
    assert set(transfer.emissions).isdisjoint(before)
    assert transfer.focus_occurrence_cells != replay.focus_occurrence_cells
    model.resolve(TapeValence.NEUTRAL)


@pytest.mark.parametrize(
    ("before", "after", "message"),
    (
        (
            (bytes.fromhex("a101"), bytes.fromhex("a102"), bytes.fromhex("a101")),
            (bytes.fromhex("a101"),),
            "ambiguous source anchor",
        ),
        (
            (bytes.fromhex("b101"), bytes.fromhex("b102")),
            (bytes.fromhex("b101"), bytes.fromhex("b1ff")),
            "absent from the source tape",
        ),
    ),
)
def test_ambiguous_or_unexpressible_pairs_fail_closed(
    before: tuple[bytes, ...],
    after: tuple[bytes, ...],
    message: str,
) -> None:
    with pytest.raises(SourceRelativeTrainingCompilerError, match=message):
        compile_successful_token_tapes(
            context_event=bytes.fromhex("cf01"),
            before_tokens=before,
            after_tokens=after,
        )
