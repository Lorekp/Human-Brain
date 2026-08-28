from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

from experiments.r3_pilot.direct_production_cell_decoder import (
    DirectProductionCellDecoder,
)


def test_direct_assembly_drives_a_local_physical_production_race() -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("arg0", integer),),
        result=integer,
    )
    examples = (
        tp.Example(
            prompt="Return the supplied integer unchanged.",
            signature=signature,
            teacher_tokens=("variable-0",),
        ),
        tp.Example(
            prompt="Return the absolute value of the supplied integer.",
            signature=signature,
            teacher_tokens=("absolute", "variable-0"),
        ),
    )

    decoder = DirectProductionCellDecoder.fit(examples, epochs=8)
    completion = decoder.complete(examples[1].prompt, signature)

    assert completion.program.tokens == examples[1].teacher_tokens
    assert completion.source == "def solve(arg0):\n    return abs(arg0)\n"
    assert decoder.local_updates > 0
    assert decoder.structural_recruitments > 0
    assert completion.readout_before_sha256 == completion.readout_after_sha256
    assert all(
        race.accepted_token in race.allowed_tokens
        and race.prompt_cells > 0
        and race.signature_cells > 0
        and race.role_cells > 0
        and race.active_assembly_cells > 0
        and race.spike_time_ms > 0.0
        for race in completion.races
    )
