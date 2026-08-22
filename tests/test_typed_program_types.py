from __future__ import annotations

from dataclasses import asdict

import pytest
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


def test_program_type_rejects_unknown_constructors_and_wrong_arity() -> None:
    with pytest.raises(ValueError, match="unknown program type constructor"):
        ProgramType("Number")
    with pytest.raises(ValueError, match="List expects 1 type arguments"):
        ProgramType("List")


def test_typed_signature_rejects_duplicate_parameter_names() -> None:
    with pytest.raises(ValueError, match="parameter names must be unique"):
        TypedSignature(
            entry_point="same",
            parameters=(
                ProgramParameter("value", ProgramType.Int()),
                ProgramParameter("value", ProgramType.Bool()),
            ),
            result=ProgramType.Int(),
        )


def test_typed_signature_record_rejects_missing_or_extra_keys() -> None:
    record = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    ).to_record()
    record["task_id"] = "forbidden"

    with pytest.raises(ValueError, match="must contain exactly"):
        TypedSignature.from_record(record)


@pytest.mark.parametrize("name", ("", "not-valid", "class"))
def test_typed_signature_rejects_invalid_python_identifiers(name: str) -> None:
    with pytest.raises(ValueError, match="Python identifier"):
        ProgramParameter(name, ProgramType.Int())
    with pytest.raises(ValueError, match="Python identifier"):
        TypedSignature(entry_point=name, parameters=(), result=ProgramType.Int())
