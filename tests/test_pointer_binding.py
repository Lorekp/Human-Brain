import pytest
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import (
    PointerBoundCellModel,
    PointerTrainingExample,
    identifier_inventory,
    materialize,
    pointerize,
    python_interface_inventory,
)


def test_pointer_binding_completes_and_learns_through_cell_model() -> None:
    observed = '''"""docstring decoy"""
def beta(alpha):
    # comment phantom
    raise ValueError("error-string ghost")
'''
    renamed_observed = '''"""docstring decoy"""
def delta(gamma):
    # comment phantom
    raise ValueError("error-string ghost")
'''
    inventory = identifier_inventory(observed)
    renamed_inventory = identifier_inventory(renamed_observed)
    assert inventory.symbols == ("beta", "alpha", "ValueError")
    assert renamed_inventory.symbols == ("delta", "gamma", "ValueError")
    assert inventory.sha256 == identifier_inventory(observed).sha256
    with pytest.raises(TypeError, match="source must be a string"):
        identifier_inventory(7)  # type: ignore[arg-type]

    interface = python_interface_inventory(observed, "beta")
    renamed_interface = python_interface_inventory(renamed_observed, "delta")
    assert interface.symbols == ("beta", "alpha")
    assert renamed_interface.symbols == ("delta", "gamma")
    assert pointerize(
        "def beta(alpha):\n    result = alpha.lower()\n    return result\n",
        interface,
    ) == (
        "def <@0>(<@1>):\n"
        "    result = <@1>.lower()\n"
        "    return result\n"
    )
    with pytest.raises(ValueError, match="matching entry_point"):
        python_interface_inventory("answer = 1\n", "answer")
    with pytest.raises(ValueError, match="matching entry_point"):
        python_interface_inventory(observed, "missing")

    surface = "alpha(beta, alphabet, beta2, _alpha)!"
    pointed = "<@1>(<@0>, alphabet, beta2, _alpha)!"
    assert pointerize(surface, inventory) == pointed
    assert materialize(pointed, inventory) == surface

    prompt = "copy beta"
    renamed_prompt = "copy delta"
    assert pointerize(prompt, inventory) == pointerize(
        renamed_prompt, renamed_inventory
    )
    examples = [
        PointerTrainingExample(prompt, "alpha!", inventory),
        PointerTrainingExample(renamed_prompt, "gamma!", renamed_inventory),
    ]
    model = PointerBoundCellModel.grow(
        examples * 2,
        ModelConfig(max_order=8, max_phrases=0),
    )
    digest_before = model.training_digest
    branches_before = model.grown_branches

    raw = model.raw_complete(
        prompt,
        inventory,
        cue_text="Copy the requested identifier.",
        max_new_tokens=5,
        stop_text="!",
    )
    renamed_raw = model.raw_complete(
        renamed_prompt,
        renamed_inventory,
        cue_text="Copy the requested identifier.",
        max_new_tokens=5,
        stop_text="!",
    )
    assert raw == renamed_raw == "<@1>!"
    assert (
        model.complete(
            prompt,
            inventory,
            cue_text="Copy the requested identifier.",
            max_new_tokens=5,
            stop_text="!",
        )
        == "alpha!"
    )
    assert (
        model.complete(
            renamed_prompt,
            renamed_inventory,
            cue_text="Copy the requested identifier.",
            max_new_tokens=5,
            stop_text="!",
        )
        == "gamma!"
    )
    assert model.training_digest == digest_before
    assert model.grown_branches == branches_before

    update = model.learn(
        "copy gamma",
        "delta!",
        renamed_inventory,
        cue_text="Copy the requested identifier.",
        strength=2.0,
    )
    assert update.prior_training_digest == digest_before
    assert update.training_digest == model.training_digest
    assert update.training_digest != digest_before
    assert model.grown_branches == branches_before + update.added_branches
    assert update.grown_transitions > 0


def test_python_interface_inventory_selects_only_the_public_callable() -> None:
    source = (
        "def helper_before(decoy):\n"
        "    return decoy\n\n"
        "async def public_entry(first, /, second=0, *rest, flag=True, **options):\n"
        "    return first\n\n"
        "def helper_after(other):\n"
        "    return other\n"
    )

    assert python_interface_inventory(source, "public_entry").symbols == (
        "public_entry",
        "first",
        "second",
        "rest",
        "flag",
        "options",
    )
    with pytest.raises(ValueError, match="matching entry_point"):
        python_interface_inventory(source, "missing")
    with pytest.raises(ValueError, match="matching entry_point"):
        python_interface_inventory(
            "def repeated(first):\n    return first\n"
            "def repeated(second):\n    return second\n",
            "repeated",
        )
    with pytest.raises(TypeError, match="entry_point must be a string"):
        python_interface_inventory(source, 7)  # type: ignore[arg-type]
