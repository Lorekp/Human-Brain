from __future__ import annotations

from hashlib import sha256

import pytest

from humanbrain.role_chunk_episode import (
    RoleChunkEpisodeEncoder,
    RolePayloadKind,
)
from humanbrain.simple_swe.tasks import PublicTask
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


def _public_task(
    *,
    task_id: str,
    entry_point: str,
    parameter: str,
    parameter_type: ProgramType = ProgramType.Int(),
    result_type: ProgramType = ProgramType.Int(),
    description: str,
    partition: str = "train",
) -> PublicTask:
    signature = TypedSignature(
        entry_point=entry_point,
        parameters=(ProgramParameter(parameter, parameter_type),),
        result=result_type,
    )
    prompt = (
        "Return exactly one complete Python function.\n"
        f"def {entry_point}({parameter}):\n"
        f'    """{description}"""'
    )
    digest = sha256(prompt.encode("utf-8")).hexdigest()
    return PublicTask(
        task_id=task_id,
        partition=partition,
        tier=0,
        family="fixture_family",
        ordinal=0,
        signature=signature,
        prompt=prompt,
        hidden_test_sha256="1" * 64,
        task_sha256=digest,
    )


def _donors() -> tuple[PublicTask, PublicTask]:
    return (
        _public_task(
            task_id="train/a",
            entry_point="solve_a91f",
            parameter="n",
            description="Return n plus the fixed offset -11 and cap it at 4.",
        ),
        _public_task(
            task_id="train/b",
            entry_point="solve_b72e",
            parameter="value",
            description="Return value plus the fixed offset +37 and cap it at 9.",
        ),
    )


def test_encoder_normalizes_roles_and_restores_exact_prompt() -> None:
    donors = _donors()
    encoder = RoleChunkEpisodeEncoder.fit_public_train(donors)
    first = encoder.encode(donors[0])
    second = encoder.encode(donors[1])

    assert first.structural_text == second.structural_text
    assert first.token_ids == second.token_ids
    assert first.restore_prompt() == donors[0].prompt
    assert second.restore_prompt() == donors[1].prompt
    assert encoder.tokenizer.decode(first.token_ids) == first.structural_text
    assert encoder.tokenizer.concepts
    assert first.structural_sha256 == second.structural_sha256

    assert tuple(
        payload.value
        for payload in first.payloads_for(RolePayloadKind.ENTRY_POINT)
    ) == ("solve_a91f",)
    assert tuple(
        payload.value for payload in first.payloads_for(RolePayloadKind.PARAMETER)
    ) == ("n",)
    assert tuple(
        payload.value
        for payload in first.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    ) == ("-11", "4")


def test_parameter_replacement_respects_identifier_boundaries_and_types() -> None:
    scalar = _public_task(
        task_id="train/scalar",
        entry_point="solve_123abc",
        parameter="n",
        description="Return n when it is nonnegative; keep token n2 and value2 at -3.",
    )
    vector = _public_task(
        task_id="train/vector",
        entry_point="solve_456def",
        parameter="values",
        parameter_type=ProgramType.List(ProgramType.Int()),
        result_type=ProgramType.List(ProgramType.Int()),
        description="Return values with each item shifted by -8.",
    )
    encoder = RoleChunkEpisodeEncoder.fit_public_train((scalar, vector))

    scalar_episode = encoder.encode(scalar)
    vector_episode = encoder.encode(vector)

    assert "nonnegative" in scalar_episode.structural_text
    assert "n2" in scalar_episode.structural_text
    assert "value2" in scalar_episode.structural_text
    assert "PARAMETER:0:INT" in scalar_episode.structural_text
    assert "PARAMETER:0:LIST_INT" in vector_episode.structural_text
    assert "ENTRY_POINT:RET_LIST_INT" in vector_episode.structural_text
    assert scalar_episode.restore_prompt() == scalar.prompt
    assert vector_episode.restore_prompt() == vector.prompt


def test_integer_and_float_literals_keep_exact_spelling() -> None:
    task = _public_task(
        task_id="train/numeric",
        entry_point="solve_numbers",
        parameter="x",
        description="Map x from +4 through -0.50 and 1e-3; leave item2 alone.",
    )
    encoder = RoleChunkEpisodeEncoder.fit_public_train((task,))
    episode = encoder.encode(task)

    assert tuple(
        payload.value
        for payload in episode.payloads_for(RolePayloadKind.INTEGER_LITERAL)
    ) == ("+4",)
    assert tuple(
        payload.value
        for payload in episode.payloads_for(RolePayloadKind.FLOAT_LITERAL)
    ) == ("-0.50", "1e-3")
    assert "item2" in episode.structural_text
    assert episode.restore_prompt() == task.prompt


def test_fit_is_deterministic_under_donor_order() -> None:
    donors = _donors()
    forward = RoleChunkEpisodeEncoder.fit_public_train(donors)
    reverse = RoleChunkEpisodeEncoder.fit_public_train(tuple(reversed(donors)))

    assert forward.tokenizer.base_tokenizer.phrases == reverse.tokenizer.base_tokenizer.phrases
    assert forward.tokenizer.concepts == reverse.tokenizer.concepts
    assert forward.encode(donors[0]).token_ids == reverse.encode(donors[0]).token_ids


def test_encoder_rejects_non_public_or_non_train_inputs() -> None:
    train = _donors()[0]
    dev = _public_task(
        task_id="dev/a",
        entry_point="solve_dev",
        parameter="n",
        description="Return n plus 1.",
        partition="dev",
    )

    with pytest.raises(ValueError, match="public TRAIN"):
        RoleChunkEpisodeEncoder.fit_public_train((dev,))
    encoder = RoleChunkEpisodeEncoder.fit_public_train((train,))
    with pytest.raises(ValueError, match="public TRAIN"):
        encoder.encode(dev)
    with pytest.raises(TypeError, match="exact PublicTask"):
        encoder.encode(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique task IDs"):
        RoleChunkEpisodeEncoder.fit_public_train((train, train))


def test_marker_namespace_collision_is_rejected() -> None:
    task = _public_task(
        task_id="train/collision",
        entry_point="solve_collision",
        parameter="n",
        description="Return n and preserve [[HB:FOREIGN]].",
    )

    with pytest.raises(ValueError, match="marker namespace"):
        RoleChunkEpisodeEncoder.fit_public_train((task,))
