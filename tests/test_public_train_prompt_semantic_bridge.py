from __future__ import annotations

import dataclasses
import inspect

from experiments.r3_pilot import public_train_prompt_semantic_bridge as bridge
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


def _signature(entry_point: str, *parameter_names: str) -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point=entry_point,
        parameters=tuple(
            ProgramParameter(name, integer) for name in parameter_names
        ),
        result=integer,
    )


def _prompt(signature: TypedSignature, suffix: str) -> str:
    parameters = ", ".join(
        parameter.name for parameter in signature.parameters
    )
    return (
        bridge.FROZEN_PROMPT_PREFIX
        + f"def {signature.entry_point}({parameters}):\n"
        + f'    """{suffix}"""'
    )


def test_declared_renaming_is_de_bruijn_invariant() -> None:
    first = _signature("difference", "alpha", "beta")
    renamed = _signature("compute", "left", "right")
    first_input = bridge.observe_public_train_prompt(
        _prompt(first, "Return alpha minus beta plus -7."),
        first,
    )
    renamed_input = bridge.observe_public_train_prompt(
        _prompt(renamed, "Return left minus right plus -7."),
        renamed,
    )

    assert first_input.audit_prompt_sha256 != renamed_input.audit_prompt_sha256
    assert first_input.tokens == renamed_input.tokens == (
        "def",
        "<ENTRY>",
        "(",
        "<ARG1>",
        ",",
        "<ARG0>",
        ")",
        ":",
        "return",
        "<ARG1>",
        "minus",
        "<ARG0>",
        "plus",
        "<INT>",
        ".",
    )
    assert first_input.lexical_sha256 == renamed_input.lexical_sha256
    assert first_input.literal_channel_sha256 == renamed_input.literal_channel_sha256
    assert first_input.sha256 == renamed_input.sha256
    assert tuple(item.semantic_sha256 for item in first_input.signed_literals) == (
        tuple(item.semantic_sha256 for item in renamed_input.signed_literals)
    )


def test_ordered_spans_preserve_operand_order() -> None:
    signature = _signature("difference", "alpha", "beta")
    forward = bridge.observe_public_train_prompt(
        _prompt(signature, "Return alpha minus beta."),
        signature,
    )
    reverse = bridge.observe_public_train_prompt(
        _prompt(signature, "Return beta minus alpha."),
        signature,
    )

    assert sorted(forward.tokens) == sorted(reverse.tokens)
    assert forward.tokens != reverse.tokens
    assert forward.lexical_sha256 != reverse.lexical_sha256
    assert forward.sha256 != reverse.sha256
    forward_bigrams = tuple(span.tokens for span in forward.spans if len(span.tokens) == 2)
    reverse_bigrams = tuple(span.tokens for span in reverse.spans if len(span.tokens) == 2)
    assert ("<ARG1>", "minus") in forward_bigrams
    assert ("<ARG0>", "minus") in reverse_bigrams
    assert forward_bigrams != reverse_bigrams


def test_signed_literals_leave_one_lexical_token_but_keep_copy_records() -> None:
    signature = _signature("offset", "value")
    suffix = "Return value plus -3, then value plus +5, then 0."
    observed = bridge.observe_public_train_prompt(_prompt(signature, suffix), signature)

    assert observed.tokens.count("<INT>") == 3
    assert not ({"-3", "+5", "0"} & set(observed.tokens))
    assert tuple(item.lexeme for item in observed.signed_literals) == ("-3", "+5", "0")
    assert tuple(item.value for item in observed.signed_literals) == (-3, 5, 0)
    for item in observed.signed_literals:
        assert suffix[item.suffix_character_start : item.suffix_character_end] == item.lexeme
        assert observed.tokens[item.canonical_token_index] == "<INT>"
        assert item.typed_output.supported is True
        assert item.typed_output_active_cells
    assert len(
        {item.typed_output.sha256 for item in observed.signed_literals}
    ) == 3
    assert len({item.semantic_sha256 for item in observed.signed_literals}) == 3
    assert tuple(
        item.typed_output_active_cells for item in observed.signed_literals
    ) == ((0, 66), (0, 74), (0, 69))
    assert tuple(item.semantic_sha256 for item in observed.signed_literals) == (
        "b8d23df40f87fd622cf6e4e21c84f3970039160839a73483e72cdf2abe0f5cd4",
        "6cbe5625560a78c56410891279cb906efc4a7d81fcd27555129ed0a85dc66a99",
        "c667a364810afdcc05868e35f95f32b11e41a3cfb8feb41f842edf5070809e6b",
    )
    assert tuple(item.audit_sha256 for item in observed.signed_literals) == (
        "133f3548f6eb9837476fbdc13a29fa0badf377565397ed23d095fe1b9b529f12",
        "d1c8ad1181dca4e4ca97d16ad1241c105f7009e7958b1ccaf9dc80c80c2b19f4",
        "8248ff9a42d0ef9ad9b1f2b102326ddda1f9d55f915cda060d587663a9b7b983",
    )
    assert observed.literal_channel_sha256 == (
        "81c778aecea452c21170b6ce1f4f3a4e29b87bcc44859e041b15f88d09690326"
    )
    assert observed.sha256 == (
        "737aaa0c3a9a95c316373aec82c5acdb7a6c2503060de71841065e1aa84f2fa0"
    )

    integer_unigrams = tuple(
        span
        for span in observed.spans
        if span.tokens == ("<INT>",)
    )
    assert len(integer_unigrams) == 3
    assert len({span.identity_sha256 for span in integer_unigrams}) == 1
    assert len({span.assembly_cells for span in integer_unigrams}) == 1

    negative = bridge.observe_public_train_prompt(
        _prompt(signature, "Return value plus -3."),
        signature,
    )
    positive = bridge.observe_public_train_prompt(
        _prompt(signature, "Return value plus +5."),
        signature,
    )
    assert negative.lexical_sha256 == positive.lexical_sha256
    assert negative.literal_channel_sha256 != positive.literal_channel_sha256
    assert negative.signed_literals[0].value == -3
    assert positive.signed_literals[0].value == 5
    assert (
        negative.signed_literals[0].typed_output.sha256
        != positive.signed_literals[0].typed_output.sha256
    )


def test_sparse_assemblies_are_deterministic_distributed_and_physically_driven() -> None:
    signature = _signature("transform", "values")
    prompt = _prompt(
        signature,
        "Return every integer in values after adding -2.",
    )
    first = bridge.observe_public_train_prompt(prompt, signature)
    second = bridge.observe_public_train_prompt(prompt, signature)

    assert first.sha256 == second.sha256
    assert tuple(span.sha256 for span in first.spans) == tuple(
        span.sha256 for span in second.spans
    )
    assert all(
        len(span.assembly_cells) == bridge.ASSEMBLY_ACTIVE_CELL_COUNT
        for span in first.spans
    )
    assert bridge.ASSEMBLY_ACTIVE_CELL_COUNT > 1
    assert bridge.ASSEMBLY_ACTIVE_CELL_COUNT < bridge.ASSEMBLY_CELL_COUNT // 10
    assert all(
        len(set(span.assembly_cells)) == bridge.ASSEMBLY_ACTIVE_CELL_COUNT
        and span.spike_cells == span.assembly_cells
        and span.first_spike_step >= 0
        for span in first.spans
    )
    distinct = {span.identity_sha256: span.assembly_cells for span in first.spans}
    assert len(distinct) > 4
    assert len(set(distinct.values())) == len(distinct)
    assemblies = tuple(map(set, distinct.values()))
    assert any(
        assemblies[left] & assemblies[right]
        for left in range(len(assemblies))
        for right in range(left + 1, len(assemblies))
    )


def test_configuration_hashes_and_checkpoint_free_scope_are_pinned() -> None:
    assert bridge.TOKENIZER_SHA256 == (
        "73777898bca9c1b009a3fb652de1993464f2bc6b0e14dafca5d706a0519fe62e"
    )
    assert bridge.ASSEMBLY_PROJECTION_SHA256 == (
        "05181981735d4dbe6f7e609e26b26bc105bc289d8e77a9a108bbd50f17fcc89c"
    )

    signature = _signature("offset", "value")
    observed = bridge.observe_public_train_prompt(
        _prompt(signature, "Return value plus -3."),
        signature,
    )
    assert observed.lexical_sha256 == (
        "5845f8637d4e032411513c166a7cbd3cc1a5ad197359cf3578e5f117a7978be0"
    )
    assert observed.scope == bridge.PUBLIC_TRAIN_INPUT_SCOPE
    assert observed.trainable_state_bytes == bridge.TRAINABLE_STATE_BYTES == 0
    assert tuple(inspect.signature(bridge.observe_public_train_prompt).parameters) == (
        "prompt",
        "signature",
    )
    forbidden_field_fragments = (
        "candidate",
        "checkpoint",
        "desired",
        "family",
        "ordinal",
        "output_case",
        "task_id",
        "teacher",
    )
    field_names = tuple(field.name for field in dataclasses.fields(bridge.PromptSpanInput))
    assert not any(
        fragment in field_name
        for field_name in field_names
        for fragment in forbidden_field_fragments
    )


def test_only_the_exact_frozen_callable_frame_is_accepted() -> None:
    signature = _signature("offset", "value")
    prompt = _prompt(signature, "Return value plus 2.")
    bridge.observe_public_train_prompt(prompt, signature)

    changed_directive = prompt.replace("Implement this exact task:", "Write this task:")
    changed_signature = prompt.replace("def offset(value):", "def offset(other):")
    for changed in (changed_directive, changed_signature, prompt + "\n"):
        try:
            bridge.observe_public_train_prompt(changed, signature)
        except ValueError as error:
            assert "frozen callable framing" in str(error)
        else:
            raise AssertionError("non-frozen prompt framing was accepted")
