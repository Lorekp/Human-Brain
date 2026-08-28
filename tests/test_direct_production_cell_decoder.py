import numpy as np

from humanbrain import typed_program as tp
from humanbrain.concept_tokens import ConceptTokenizer
from humanbrain.sdr_union import SDRProjector
from humanbrain.tokenizer import BytePhraseTokenizer
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

from experiments.r3_pilot.direct_production_cell_decoder import (
    DirectProductionCellDecoder,
    _ASSEMBLY_CELLS,
    _ASSEMBLY_ONES,
    _RANDOM_SEED,
    _project_contextual_prompt,
)


def test_adjacent_pair_cells_distinguish_reversed_prompt_token_order() -> None:
    tokenizer = ConceptTokenizer(BytePhraseTokenizer())
    projector = SDRProjector(
        n_bits=_ASSEMBLY_CELLS,
        w_ones=_ASSEMBLY_ONES,
        seed=_RANDOM_SEED,
    )
    forward = "ab"
    reverse = "ba"

    assert sorted(tokenizer.encode(forward)) == sorted(tokenizer.encode(reverse))
    forward_cells, forward_unigrams, _ = _project_contextual_prompt(
        tokenizer,
        projector,
        forward,
        '{"signature":"same"}',
        '{"role":"same"}',
    )
    reverse_cells, reverse_unigrams, _ = _project_contextual_prompt(
        tokenizer,
        projector,
        reverse,
        '{"signature":"same"}',
        '{"role":"same"}',
    )

    assert np.array_equal(forward_unigrams, reverse_unigrams)
    assert not np.array_equal(forward_cells, reverse_cells)


def test_direct_assembly_drives_a_local_physical_production_race() -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="solve",
        parameters=(
            ProgramParameter("first", integer),
            ProgramParameter("second", integer),
        ),
        result=integer,
    )
    examples = (
        tp.Example(
            prompt="Given integers first and second, return first minus second.",
            signature=signature,
            teacher_tokens=("subtract", "variable-1", "variable-0"),
        ),
        tp.Example(
            prompt=(
                "Given integers first and second, return the absolute value of "
                "first; second is unused."
            ),
            signature=signature,
            teacher_tokens=("absolute", "variable-1"),
        ),
        tp.Example(
            prompt=(
                "Given integers first and second, return second unchanged; "
                "first is unused."
            ),
            signature=signature,
            teacher_tokens=("variable-0",),
        ),
        tp.Example(
            prompt=(
                "Given integers first and second, return first unchanged; "
                "second is unused."
            ),
            signature=signature,
            teacher_tokens=("variable-1",),
        ),
    )

    decoder = DirectProductionCellDecoder.fit(examples, epochs=8)
    completions = tuple(
        decoder.complete(example.prompt, signature) for example in examples
    )

    assert tuple(completion.program.tokens for completion in completions) == tuple(
        example.teacher_tokens for example in examples
    )
    assert decoder.local_updates > 0
    assert decoder.structural_recruitments > 0
    assert all(
        completion.readout_before_sha256 == completion.readout_after_sha256
        for completion in completions
    )
    assert all(
        race.accepted_token in race.allowed_tokens
        and race.contextual_unigram_cells > 0
        and race.contextual_pair_cells > 0
        and race.active_assembly_cells > 0
        and race.spike_time_ms > 0.0
        for completion in completions
        for race in completion.races
    )
