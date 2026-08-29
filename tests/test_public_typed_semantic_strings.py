from __future__ import annotations

from dataclasses import fields

import pytest

from experiments.r3_pilot import public_typed_operator_successor as operator
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain import typed_program
from humanbrain import typed_semantic_synthesis as synthesis
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


_EXPECTED_CLASSES = (
    "ascii-alphanumeric",
    "ascii-decimal-digits",
    "ascii-hex-digits",
    "ascii-letters",
    "ascii-lowercase",
    "ascii-octal-digits",
    "ascii-printable",
    "ascii-punctuation",
    "ascii-uppercase",
    "ascii-whitespace",
    "english-consonants",
    "english-vowels",
)


def _signature() -> TypedSignature:
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("text", ProgramType.String()),),
        result=ProgramType.String(),
    )


@pytest.fixture(scope="module")
def model() -> semantic.FrozenTypedSemanticSynthesizer:
    return semantic.build_public_typed_semantic_synthesizer(
        literals=(0,),
        max_ast_cost=7,
    )[0]


def test_ascii_class_is_private_but_the_frozen_catalog_is_broad() -> None:
    assert typed_program.ASCII_CLASS_LABELS == _EXPECTED_CLASSES
    assert tuple(typed_program._ASCII_CLASS_CHARACTERS) == _EXPECTED_CLASSES
    assert typed_program._ASCII_CLASS_CHARACTERS["ascii-whitespace"] == " \t\n\v\f\r"
    assert len(typed_program._ASCII_CLASS_CHARACTERS["ascii-printable"]) == 95
    assert "A" in typed_program._ASCII_CLASS_CHARACTERS["english-vowels"]
    assert "B" in typed_program._ASCII_CLASS_CHARACTERS["english-consonants"]
    assert not hasattr(ProgramType, "AsciiClass")
    with pytest.raises(ValueError, match="unknown program type constructor"):
        ProgramType("AsciiClass")

    whitespace_program = typed_program.Program(
        (
            "filter",
            "bind",
            "contains",
            "ascii-whitespace",
            "variable-0",
            "variable-0",
        ),
        _signature(),
    )
    namespace: dict[str, object] = {}
    exec(typed_program.compile(whitespace_program).source, namespace)  # noqa: S102
    solve = namespace["solve"]
    assert callable(solve)
    assert solve(" \t\n\v\f\rA!0") == " \t\n\v\f\r"


def test_compiler_and_enumerator_pin_the_generic_string_costs() -> None:
    dag = synthesis.enumerate_typed_candidate_dag(
        _signature(),
        literals=(0,),
        max_ast_cost=7,
    )
    by_tokens = {
        node.tokens: node
        for node in dag.nodes
    }
    class_tokens = ("english-consonants",)
    contains_tokens = ("contains", *class_tokens, "variable-0")
    bind_tokens = ("bind", *contains_tokens)
    filter_tokens = ("filter", *bind_tokens, "variable-0")
    reverse_filter_tokens = ("reverse", *filter_tokens)

    assert by_tokens[("variable-0",)].ast_cost == 1
    assert by_tokens[("reverse", "variable-0")].ast_cost == 2
    assert by_tokens[class_tokens].result_type_label == "AsciiClass"
    assert by_tokens[contains_tokens].ast_cost == 3
    assert by_tokens[bind_tokens].ast_cost == 4
    assert by_tokens[filter_tokens].ast_cost == 6
    assert by_tokens[reverse_filter_tokens].ast_cost == 7

    compiled = typed_program.compile(
        typed_program.Program(reverse_filter_tokens, _signature())
    )
    assert compiled.source == (
        "def solve(text):\n"
        "    return ''.join((char_0 for char_0 in text if char_0 in "
        "'BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz'))[::-1]\n"
    )


def test_physical_class_bank_and_contains_cover_the_full_catalog() -> None:
    frozen, summary = operator.train_public_typed_operator_successor()
    assert operator._ASCII_CLASS_CELL_COUNT == 13
    assert frozen.class_pair_class_contacts_pa.shape == (1_141, 13)
    assert frozen.class_pair_char_contacts_pa.shape == (1_141, 96)
    assert frozen.contains_weights_pa.shape == (3, 1_141)
    assert summary.primitive_observations == 51_192

    representatives = {
        "ascii-alphanumeric": ("A", "!"),
        "ascii-decimal-digits": ("9", "A"),
        "ascii-hex-digits": ("F", "G"),
        "ascii-letters": ("z", "7"),
        "ascii-lowercase": ("z", "Z"),
        "ascii-octal-digits": ("7", "8"),
        "ascii-printable": (" ", None),
        "ascii-punctuation": ("!", "A"),
        "ascii-uppercase": ("Z", "z"),
        "ascii-whitespace": (" ", "\t"),
        "english-consonants": ("B", "A"),
        "english-vowels": ("A", "B"),
    }
    for label, (member, nonmember) in representatives.items():
        class_value, class_races = frozen._ascii_class(label, enabled=True)
        assert class_value.supported
        assert class_value.label == label
        assert len(class_races) == 1
        assert class_races[0].bank == "ascii-class"
        member_value = output_cells.TypedOutputCellBank.encode(
            member,
            declared_type=output_cells.OutputType.CHAR,
        )
        contained, contains_races = frozen._contains(class_value, member_value)
        assert output_cells.TypedOutputCellBank.decode(contained) is True
        assert tuple(race.bank for race in contains_races) == (
            "operator-ascii-class-char-pair",
            "operator-contains-output",
        )
        if nonmember is not None:
            nonmember_value = output_cells.TypedOutputCellBank.encode(
                nonmember,
                declared_type=output_cells.OutputType.CHAR,
            )
            excluded, _ = frozen._contains(class_value, nonmember_value)
            if nonmember == "\t":
                assert not excluded.supported
            else:
                assert output_cells.TypedOutputCellBank.decode(excluded) is False


def test_direct_reverse_synthesizes_at_cost_two_with_physical_copy_telemetry(
    model: semantic.FrozenTypedSemanticSynthesizer,
) -> None:
    before = model.live_checkpoint_sha256
    decision = model.synthesize(
        _signature(),
        (("ab",), ("A1!",), ("",)),
        ("ba", "!1A", ""),
    )

    assert decision.status == semantic.STATUS_DONE
    assert decision.program is not None
    assert decision.program.tokens == ("reverse", "variable-0")
    assert decision.tier_attempts[-1].ast_cost == 2
    assert decision.operator_predictions == 63
    assert decision.operator_races == 74
    assert decision.operator_races_sha256 == (
        "4ed8ab1c38ec4350a25314a0581c2bd5b5cdde2a1b8503a67b5154fe83a0dace"
    )
    assert decision.checkpoint_before_sha256 == before
    assert decision.checkpoint_after_sha256 == before


@pytest.mark.parametrize(
    ("arguments", "desired", "tokens", "predictions", "races"),
    (
        (
            (("aB3!",), ("xyzE",), ("IOu",), ("",)),
            ("a", "E", "IOu", ""),
            ("filter", "bind", "contains", "english-vowels", "variable-0", "variable-0"),
            364,
            894,
        ),
        (
            (("a08F9!",), ("765x",), ("G2?",), ("",)),
            ("089", "765", "2", ""),
            (
                "filter",
                "bind",
                "contains",
                "ascii-decimal-digits",
                "variable-0",
                "variable-0",
            ),
            412,
            996,
        ),
        (
            (("aB3cE!",), ("xyzOU",), ("A1t",), ("",)),
            ("cB", "zyx", "t", ""),
            (
                "reverse",
                "filter",
                "bind",
                "contains",
                "english-consonants",
                "variable-0",
                "variable-0",
            ),
            484,
            1_274,
        ),
    ),
)
def test_self_posed_string_mechanisms_are_exact_and_physical(
    model: semantic.FrozenTypedSemanticSynthesizer,
    arguments: tuple[tuple[str], ...],
    desired: tuple[str, ...],
    tokens: tuple[str, ...],
    predictions: int,
    races: int,
) -> None:
    before = model.live_checkpoint_sha256
    decision = model.synthesize(_signature(), arguments, desired)

    assert decision.status == semantic.STATUS_DONE
    assert decision.program is not None
    assert decision.program.tokens == tokens
    assert decision.tier_attempts[-1].ast_cost == (
        7 if tokens[0] == "reverse" else 6
    )
    assert decision.operator_predictions == predictions
    assert decision.operator_races == races
    assert decision.checkpoint_before_sha256 == before
    assert decision.checkpoint_after_sha256 == before
    assert model.live_checkpoint_sha256 == before


def test_retention_is_desired_independent_over_the_same_string_context(
    model: semantic.FrozenTypedSemanticSynthesizer,
) -> None:
    arguments = (("aB3cE!",), ("xyzOU",), ("A1t",), ("",))
    vowels = model.synthesize(_signature(), arguments, ("aE", "OU", "A", ""))
    consonants = model.synthesize(
        _signature(),
        arguments,
        ("Bc", "xyz", "t", ""),
    )

    assert vowels.status == consonants.status == semantic.STATUS_DONE
    assert vowels.operator_predictions == consonants.operator_predictions
    assert vowels.operator_races == consonants.operator_races
    assert vowels.operator_races_sha256 == consonants.operator_races_sha256
    assert tuple(
        attempt.representative_candidate_ids for attempt in vowels.tier_attempts
    ) == tuple(
        attempt.representative_candidate_ids for attempt in consonants.tier_attempts
    )


def test_catalog_off_is_checkpointed_unknown_and_licenses_no_filter() -> None:
    on = semantic.build_public_typed_semantic_synthesizer(
        literals=(0,),
        max_ast_cost=7,
    )[0]
    off = semantic.build_public_typed_semantic_synthesizer(
        literals=(0,),
        max_ast_cost=7,
        _ascii_catalog_enabled=False,
    )[0]
    assert on.operator.checkpoint_sha256 == off.operator.checkpoint_sha256
    assert on.checkpoint_sha256 != off.checkpoint_sha256

    class_value, class_races = off.operator._ascii_class(
        "english-vowels",
        enabled=off._ascii_catalog_enabled,
    )
    assert not class_value.supported
    assert class_value.label is None
    assert len(class_races) == 1

    decision = off.synthesize(
        _signature(),
        (("aB3!",), ("xyzE",), ("IOu",), ("",)),
        ("a", "E", "IOu", ""),
    )
    assert decision.status == semantic.STATUS_STOP
    assert decision.program is None
    assert decision.tier_attempts[0].evaluated_candidates == 1
    on_control = on.synthesize(
        _signature(),
        (("aB3!",), ("xyzE",), ("IOu",), ("",)),
        ("a!", "xE", "Iu", ""),
    )
    assert on_control.status == semantic.STATUS_STOP
    assert tuple(
        (attempt.ast_cost, attempt.evaluated_candidates)
        for attempt in decision.tier_attempts
    ) == tuple(
        (attempt.ast_cost, attempt.evaluated_candidates)
        for attempt in on_control.tier_attempts
    )
    assert tuple(
        (attempt.ast_cost, attempt.evaluated_candidates)
        for attempt in decision.tier_attempts
    ) == ((1, 1), (2, 1), (3, 0), (4, 0), (5, 0), (6, 12), (7, 12))
    assert not decision.tier_attempts[5].representative_candidate_ids
    assert not decision.tier_attempts[6].representative_candidate_ids

    exhaustive = synthesis.enumerate_typed_candidate_dag(
        _signature(),
        literals=(0,),
        max_ast_cost=7,
    )
    assert len(
        [node for node in exhaustive.nodes if node.provenance.production == "contains"]
    ) == 12
    assert len(
        [node for node in exhaustive.nodes if node.provenance.production == "filter"]
    ) == 12
    assert len(
        [
            node
            for node in exhaustive.nodes
            if node.provenance.production == "reverse" and node.ast_cost == 7
        ]
    ) == 12

    off_budget = semantic._SemanticFrontierBudget()
    enabled_budget = semantic._SemanticFrontierBudget()
    work_context = semantic._WorkContext(
        ast_cost=1,
        environment_type_labels=("Char", "String"),
        result_type_label="AsciiClass",
        context_shape=("bound", ((0, 0, "String", 1),)),
    )
    off_budget.set_context(work_context)
    enabled_budget.set_context(work_context)
    evaluator = semantic._CaseEvaluator(
        off.operator,
        exhaustive,
        ("A",),
        ascii_catalog_enabled=off._ascii_catalog_enabled,
        budget=off_budget,
    )
    enabled_evaluator = semantic._CaseEvaluator(
        on.operator,
        exhaustive,
        ("A",),
        ascii_catalog_enabled=on._ascii_catalog_enabled,
        budget=enabled_budget,
    )
    class_nodes = tuple(
        node
        for node in exhaustive.nodes
        if node.result_type_label == "AsciiClass"
    )
    contains_nodes = tuple(
        node
        for node in exhaustive.nodes
        if node.provenance.production == "contains"
    )
    assert tuple(node.provenance.production for node in class_nodes) == _EXPECTED_CLASSES
    outputs = tuple(
        evaluator.evaluate(node.node_id, bound=("A",)) for node in class_nodes
    )
    enabled_outputs = tuple(
        enabled_evaluator.evaluate(node.node_id, bound=("A",))
        for node in class_nodes
    )
    off_predicates = tuple(
        evaluator.evaluate(node.node_id, bound=("A",))
        for node in contains_nodes
    )
    enabled_predicates = tuple(
        enabled_evaluator.evaluate(node.node_id, bound=("A",))
        for node in contains_nodes
    )
    assert all(not output.supported for output in outputs)
    assert all(not output.supported for output in off_predicates)
    assert all(output.supported for output in enabled_predicates)
    assert tuple(output.label for output in enabled_outputs) == _EXPECTED_CLASSES
    assert evaluator.predictions == enabled_evaluator.predictions == 24
    assert tuple(race.bank for race in evaluator.races) == tuple(
        race.bank for race in enabled_evaluator.races
    )
    assert tuple(race.bank for race in evaluator.races) == (
        ("ascii-class",) * 12
        + (
            "operator-ascii-class-char-pair",
            "operator-contains-output",
        )
        * 12
    )
    assert tuple(race.winner_cell for race in evaluator.races[:12]) == (
        operator._ASCII_CLASS_UNKNOWN_CELL,
    ) * 12


@pytest.mark.parametrize("value", ("x" * 27, "a\nb"))
def test_unsupported_string_contexts_stop_without_retention(
    model: semantic.FrozenTypedSemanticSynthesizer,
    value: str,
) -> None:
    before = model.live_checkpoint_sha256
    decision = model.synthesize(_signature(), ((value,),), ("",))
    assert decision.status == semantic.STATUS_STOP
    assert decision.candidate_count == 0
    assert decision.operator_predictions == 0
    assert decision.operator_races == 0
    assert decision.checkpoint_after_sha256 == before


def test_no_forbidden_task_or_target_state_enters_the_frozen_model(
    model: semantic.FrozenTypedSemanticSynthesizer,
) -> None:
    names = {field.name for field in fields(model)}
    forbidden = {
        "prompt",
        "task_id",
        "family_id",
        "desired",
        "property_rows",
        "canonical_outputs",
        "dev",
        "heldout",
        "c1",
    }
    assert names.isdisjoint(forbidden)
    assert len(typed_program.ASCII_CLASS_LABELS) == 12
