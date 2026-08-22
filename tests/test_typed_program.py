from __future__ import annotations

from hashlib import sha256

from humanbrain.typed_program import (
    Example,
    Layer1Protocol,
    ProgramParameter,
    ProgramType,
    TypedSignature,
    compile,
    complete,
    grow,
)


def test_layer1_int_program_grows_completes_and_compiles_through_cell_races() -> None:
    signature = TypedSignature(
        entry_point="increment",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    teacher = ("add", "variable-0", "integer", "+", "1", "end")

    checkpoint = grow(
        Layer1Protocol(),
        (Example(prompt="Increment the integer.", signature=signature, teacher_tokens=teacher),),
    )
    completion = complete(checkpoint, "Increment the integer.", signature)

    assert completion.program.tokens == teacher
    assert completion.race_count == len(teacher)
    assert completion.prompt_sha256 == sha256(b"Increment the integer.").hexdigest()
    assert compile(completion.program).source == (
        "def increment(value):\n"
        "    return value + 1\n"
    )
