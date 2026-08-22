from __future__ import annotations

import copy
import json
from hashlib import sha256

import humanbrain.typed_program as typed_program
import pytest
from humanbrain.typed_program import (
    Example,
    FrozenCheckpoint,
    Layer1Protocol,
    Program,
    ProgramFormatError,
    ProgramParameter,
    ProgramType,
    TypedSignature,
    compile,
    complete,
    grow,
)


def _canonical_checkpoint_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _replace_nested(record: object, path: tuple[str | int, ...], value: object) -> None:
    cursor = record
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]


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
    assert compile(completion.program).source == ("def increment(value):\n    return value + 1\n")


def test_generic_map_composes_a_typed_binder_and_list_argument() -> None:
    signature = TypedSignature(
        entry_point="increment_all",
        parameters=(ProgramParameter("items", ProgramType.List(ProgramType.Int())),),
        result=ProgramType.List(ProgramType.Int()),
    )
    teacher = (
        "map",
        "bind",
        "add",
        "variable-0",
        "integer",
        "+",
        "1",
        "end",
        "variable-0",
    )

    checkpoint = grow(
        Layer1Protocol(),
        (Example(prompt="Increment every integer.", signature=signature, teacher_tokens=teacher),),
    )
    completion = complete(checkpoint, "Increment every integer.", signature)

    assert completion.program.tokens == teacher
    assert completion.race_count == len(teacher)
    assert compile(completion.program).source == (
        "def increment_all(items):\n    return [item_0 + 1 for item_0 in items]\n"
    )


def test_checkpoint_corruption_is_rejected_before_array_or_circuit_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = TypedSignature(
        entry_point="increment",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    teacher = ("add", "variable-0", "integer", "+", "1", "end")
    checkpoint = grow(
        Layer1Protocol(),
        (Example(prompt="Increment.", signature=signature, teacher_tokens=teacher),),
    )
    valid_record = json.loads(checkpoint.to_bytes().decode("utf-8"))

    array_name = valid_record["circuit"]["arrays"][0]["name"]
    mutations = (
        (
            "object dtype",
            ("circuit", "arrays", 0, "dtype"),
            "|O",
            "dtype does not match the protocol",
        ),
        (
            "oversized shape",
            ("circuit", "arrays", 0, "shape"),
            [1_000_000_000],
            "shape does not match the protocol",
        ),
        (
            "negative shape",
            ("circuit", "arrays", 0, "shape"),
            [-1],
            "shape does not match the protocol",
        ),
        (
            "wrong byte length",
            ("circuit", "arrays", 0, "data"),
            "",
            "byte length does not match the protocol",
        ),
        (
            "wrong circuit config",
            ("circuit", "metadata", "circuit_config", "reservoir_cells"),
            129,
            "metadata does not match the protocol",
        ),
        (
            "wrong token count",
            ("circuit", "metadata", "token_cells"),
            valid_record["circuit"]["metadata"]["token_cells"] + 1,
            "metadata does not match the protocol",
        ),
        (
            "wrong Allen identity",
            ("circuit", "metadata", "allen_glif5_config_sha256"),
            "0" * 64,
            "metadata does not match the protocol",
        ),
        (
            "wrong fitted cell parameters",
            ("circuit", "metadata", "cell_parameters", "rest_mv"),
            0.0,
            "metadata does not match the protocol",
        ),
        (
            "extra circuit metadata",
            ("circuit", "metadata", "task_id"),
            "forbidden",
            "must contain exactly",
        ),
        (
            "wrong array set",
            ("circuit", "arrays", 0, "name"),
            f"unexpected_{array_name}",
            "array names do not match the protocol",
        ),
        (
            "empty training provenance",
            ("training_input_sha256",),
            [],
            "at least one training input digest",
        ),
    )

    def forbidden_restore(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("corrupt checkpoint reached array or circuit restoration")

    monkeypatch.setattr(typed_program.np, "frombuffer", forbidden_restore)
    monkeypatch.setattr(
        typed_program.RecurrentCellCircuit,
        "restore_cell_synapse_state",
        forbidden_restore,
    )

    for _label, path, replacement, message in mutations:
        corrupted = copy.deepcopy(valid_record)
        _replace_nested(corrupted, path, replacement)
        with pytest.raises(ValueError, match=message):
            FrozenCheckpoint.from_bytes(_canonical_checkpoint_bytes(corrupted))


def test_checkpoint_round_trip_preserves_framed_prompt_signature_audit() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    signature_text = _canonical_checkpoint_bytes(signature.to_record()).decode("utf-8")
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt=signature_text,
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    restored = FrozenCheckpoint.from_bytes(checkpoint.to_bytes())
    record = json.loads(restored.to_bytes().decode("utf-8"))
    token_order = tuple(item["token"] for item in record["protocol"]["primitive_table"])
    prompt_signature_delimiter = 256 + token_order.index("prompt-signature-delimiter")
    input_delimiter = 256 + token_order.index("input-delimiter")
    signature_bytes = _canonical_checkpoint_bytes(signature.to_record())
    framed_ids = (
        *signature_text.encode("utf-8"),
        prompt_signature_delimiter,
        *signature_bytes,
        input_delimiter,
    )
    framed_bytes = b"".join(token_id.to_bytes(2, "big") for token_id in framed_ids)
    unframed_ids = (
        *signature_text.encode("utf-8"),
        *signature_bytes,
        input_delimiter,
    )
    unframed_bytes = b"".join(token_id.to_bytes(2, "big") for token_id in unframed_ids)

    assert restored == checkpoint
    assert restored.to_bytes() == checkpoint.to_bytes()
    assert restored.training_input_sha256 == (sha256(framed_bytes).hexdigest(),)
    assert restored.training_input_sha256 != (sha256(unframed_bytes).hexdigest(),)
    completion = complete(restored, signature_text, signature)
    assert completion.model_input_sha256 == restored.training_input_sha256[0]


def test_signed_literals_are_bounded_canonical_and_type_checked() -> None:
    signature = TypedSignature(
        entry_point="subtract_twelve",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    teacher = ("add", "variable-0", "integer", "-", "1", "2", "end")
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Subtract twelve.",
                signature=signature,
                teacher_tokens=teacher,
            ),
        ),
    )

    completion = complete(checkpoint, "Subtract twelve.", signature)

    assert completion.program.tokens == teacher
    assert completion.race_count == len(teacher)
    assert compile(completion.program).source == (
        "def subtract_twelve(value):\n    return value + -12\n"
    )

    constant_signature = TypedSignature(
        entry_point="constant",
        parameters=(),
        result=ProgramType.Int(),
    )
    invalid_literals = (
        ("integer", "-", "0", "end"),
        ("integer", "+", "0", "1", "end"),
        ("integer", "+", "1", "2", "3", "4", "5", "end"),
    )
    for tokens in invalid_literals:
        with pytest.raises(ProgramFormatError):
            compile(Program(tokens=tokens, signature=constant_signature))

    bool_signature = TypedSignature(
        entry_point="not_an_integer",
        parameters=(),
        result=ProgramType.Bool(),
    )
    with pytest.raises(ProgramFormatError):
        compile(
            Program(
                tokens=("integer", "+", "1", "end"),
                signature=bool_signature,
            )
        )
