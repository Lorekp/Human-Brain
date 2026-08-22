from __future__ import annotations

from dataclasses import asdict

from humanbrain.typed_program import ProgramParameter, ProgramType, TypedSignature


def test_typed_signature_strictly_round_trips_its_canonical_record() -> None:
    signature = TypedSignature(
        entry_point="group",
        parameters=(
            ProgramParameter(
                "items",
                ProgramType.List(ProgramType.Pair(ProgramType.Char(), ProgramType.Int())),
            ),
            ProgramParameter("enabled", ProgramType.Bool()),
        ),
        result=ProgramType.String(),
    )

    assert TypedSignature.from_record(asdict(signature)) == signature
    assert TypedSignature.from_record(signature.to_record()) == signature
