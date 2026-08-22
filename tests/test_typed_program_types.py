from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from humanbrain.typed_program import ProgramParameter, ProgramType, TypedSignature


ROOT = Path(__file__).resolve().parents[1]


def test_lightweight_type_import_excludes_simulator_dependencies_and_preserves_facade() -> None:
    probe = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(ROOT / 'src')!r})",
            "from humanbrain import typed_program_types",
            "forbidden = sorted(name for name in sys.modules if "
            "name == 'numpy' or name.startswith('numpy.') or "
            "name == 'humanbrain.recurrent' or name.startswith('humanbrain.recurrent.') or "
            "name == 'humanbrain.native' or name.startswith('humanbrain.native.') or "
            "name == 'humanbrain.cuda' or name.startswith('humanbrain.cuda.') or "
            "name.startswith('humanbrain._native'))",
            "if forbidden: raise AssertionError(forbidden)",
            "from humanbrain import typed_program",
            "assert typed_program.ProgramType is typed_program_types.ProgramType",
            "assert typed_program.ProgramParameter is typed_program_types.ProgramParameter",
            "assert typed_program.TypedSignature is typed_program_types.TypedSignature",
        )
    )

    subprocess.run(
        [sys.executable, "-I", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )


def test_simple_swe_task_signatures_import_without_simulator_dependencies() -> None:
    probe = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(ROOT / 'src')!r})",
            "from humanbrain import typed_program_types",
            "from humanbrain.simple_swe import tasks",
            "assert tasks.ProgramType is typed_program_types.ProgramType",
            "assert tasks.ProgramParameter is typed_program_types.ProgramParameter",
            "assert tasks.TypedSignature is typed_program_types.TypedSignature",
            "forbidden = sorted(name for name in sys.modules if "
            "name == 'numpy' or name.startswith('numpy.') or "
            "name == 'humanbrain.recurrent' or name.startswith('humanbrain.recurrent.') or "
            "name == 'humanbrain.native' or name.startswith('humanbrain.native.') or "
            "name == 'humanbrain.cuda' or name.startswith('humanbrain.cuda.') or "
            "name.startswith('humanbrain._native'))",
            "if forbidden: raise AssertionError(forbidden)",
        )
    )

    subprocess.run(
        [sys.executable, "-I", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )


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


def test_program_type_constructor_arities_cannot_be_mutated() -> None:
    with pytest.raises(TypeError):
        ProgramType._ARITY["Int"] = 1  # type: ignore[index]


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
