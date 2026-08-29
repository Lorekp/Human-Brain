import ast
from pathlib import Path

import numpy as np
import pytest

from experiments.r3_pilot import public_typed_edit_site_observation as observation
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


def _integer_literal(value: int) -> tuple[str, ...]:
    sign = "+" if value >= 0 else "-"
    return ("integer", sign, *tuple(str(abs(value))), "end")


def _raw_outputs(
    result: observation.CurrentEditSiteObservation,
) -> tuple[object, ...]:
    return tuple(context.raw_output for context in result.contexts)


def test_observes_scalar_root_and_child_from_the_current_program() -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("x", integer),),
        result=integer,
    )
    program = tp.Program(
        ("add", "absolute", "variable-0", *_integer_literal(2)),
        signature,
    )
    contexts = ((-3,), (0,), (5,))

    root = observation.observe_current_edit_site(program, (), contexts)
    child = observation.observe_current_edit_site(program, (0,), contexts)

    assert _raw_outputs(root) == (5, 2, 7)
    assert _raw_outputs(child) == (3, 0, 5)
    assert root.path == () and child.path == (0,)
    assert root.result_type_label == child.result_type_label == "Int"
    assert root.scope_type_labels == child.scope_type_labels == ("Int",)
    assert root.all_supported and child.all_supported
    assert root.output_sha256 != child.output_sha256
    assert root.physical_output_cells.shape == (
        len(contexts),
        observation.output_cells.TOTAL_CELL_COUNT,
    )


def test_two_parameter_contexts_follow_de_bruijn_binding_order() -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="difference",
        parameters=(
            ProgramParameter("a", integer),
            ProgramParameter("b", integer),
        ),
        result=integer,
    )
    a_minus_b = tp.Program(
        ("subtract", "variable-1", "variable-0"),
        signature,
    )
    b_minus_a = tp.Program(
        ("subtract", "variable-0", "variable-1"),
        signature,
    )
    # Local contexts are (variable-0=b, variable-1=a), not signature order.
    contexts = ((2, 7), (4, -3), (5, 5))

    forward = observation.observe_current_edit_site(a_minus_b, (), contexts)
    reverse = observation.observe_current_edit_site(b_minus_a, (), contexts)

    assert forward.scope_type_labels == reverse.scope_type_labels == ("Int", "Int")
    assert _raw_outputs(forward) == (5, -7, 0)
    assert _raw_outputs(reverse) == (-5, 7, 0)


def test_map_body_and_collection_sites_receive_immutable_local_contexts() -> None:
    integer = ProgramType.Int()
    list_integer = ProgramType.List(integer)
    signature = TypedSignature(
        entry_point="map_values",
        parameters=(ProgramParameter("values", list_integer),),
        result=list_integer,
    )
    program = tp.Program(
        ("map", "bind", "absolute", "variable-0", "variable-0"),
        signature,
    )
    whole_list = [-3, 0, 4]

    body = observation.observe_current_edit_site(
        program,
        (0, 0),
        ((-3, whole_list), (4, whole_list)),
    )
    collection = observation.observe_current_edit_site(
        program,
        (1,),
        ((whole_list,),),
    )
    whole_list[0] = 99

    assert body.scope_type_labels == ("Int", "List[Int]")
    assert tuple(context.scope_values for context in body.contexts) == (
        (-3, (-3, 0, 4)),
        (4, (-3, 0, 4)),
    )
    assert _raw_outputs(body) == (3, 4)
    assert collection.scope_type_labels == ("List[Int]",)
    assert collection.contexts[0].scope_values == ((-3, 0, 4),)
    assert collection.contexts[0].raw_output == (-3, 0, 4)
    assert collection.contexts[0].output_encoding.output_type is (
        observation.output_cells.OutputType.LIST_INT
    )
    assert collection.contexts[0].output_encoding.supported


def test_bool_scopes_and_results_remain_bool_typed() -> None:
    boolean = ProgramType.Bool()
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("flag", boolean),),
        result=boolean,
    )
    program = tp.Program(("variable-0",), signature)

    result = observation.observe_current_edit_site(
        program,
        (),
        ((False,), (True,)),
    )

    assert _raw_outputs(result) == (False, True)
    assert result.scope_type_labels == ("Bool",)
    assert all(
        context.output_encoding.output_type is observation.output_cells.OutputType.BOOL
        and np.count_nonzero(context.output_encoding.bool_cells) == 1
        and np.count_nonzero(context.output_encoding.int_cells) == 0
        for context in result.contexts
    )


def test_output_failures_and_unsupported_results_become_physical_unknown() -> None:
    integer = ProgramType.Int()
    integer_signature = TypedSignature(
        entry_point="offset",
        parameters=(ProgramParameter("x", integer),),
        result=integer,
    )
    out_of_domain = tp.Program(
        ("add", "variable-0", *_integer_literal(20)),
        integer_signature,
    )
    too_large = observation.observe_current_edit_site(out_of_domain, (), ((1,),))

    list_integer = ProgramType.List(integer)
    list_boolean = ProgramType.List(ProgramType.Bool())
    unsupported_signature = TypedSignature(
        entry_point="flags",
        parameters=(ProgramParameter("values", list_integer),),
        result=list_boolean,
    )
    unsupported_program = tp.Program(
        (
            "map",
            "bind",
            "less-than",
            "variable-0",
            *_integer_literal(0),
            "variable-0",
        ),
        unsupported_signature,
    )
    unsupported = observation.observe_current_edit_site(
        unsupported_program,
        (),
        (([-1, 0, 2],),),
    )

    assert too_large.contexts[0].raw_output == 21
    assert too_large.contexts[0].status == observation.STATUS_OUTPUT_UNKNOWN
    assert too_large.contexts[0].output_encoding.unknown_active
    assert unsupported.contexts[0].raw_output == (True, False, False)
    assert (
        unsupported.contexts[0].status
        == observation.STATUS_UNSUPPORTED_RESULT_TYPE
    )
    assert unsupported.contexts[0].output_encoding.unknown_active


def test_runtime_failures_become_declared_physical_unknown(monkeypatch) -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("x", integer),),
        result=integer,
    )
    program = tp.Program(("variable-0",), signature)
    real_compile = observation.tp.compile_edit_site

    def failing_compile(current: tp.Program, path: tuple[int, ...]) -> tp.CompiledEditSite:
        compiled = real_compile(current, path)
        return tp.CompiledEditSite(
            path=compiled.path,
            result_type_label=compiled.result_type_label,
            scope_type_labels=compiled.scope_type_labels,
            source=(
                "def edit_site(scope_0):\n"
                "    raise RuntimeError('synthetic trusted failure')\n"
            ),
        )

    monkeypatch.setattr(observation.tp, "compile_edit_site", failing_compile)
    result = observation.observe_current_edit_site(program, (), ((3,),))

    assert result.contexts[0].raw_output is None
    assert result.contexts[0].status == observation.STATUS_RUNTIME_UNKNOWN
    assert result.contexts[0].output_encoding.output_type is (
        observation.output_cells.OutputType.INT
    )
    assert result.contexts[0].output_encoding.unknown_active


def test_hashes_are_deterministic_and_only_selected_site_is_compiled(monkeypatch) -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="absolute_value",
        parameters=(ProgramParameter("x", integer),),
        result=integer,
    )
    program = tp.Program(("absolute", "variable-0"), signature)
    contexts = ((-4,), (0,), (3,))
    real_compile = observation.tp.compile_edit_site
    compiled_calls: list[tuple[tp.Program, tuple[int, ...]]] = []

    def recording_compile(
        current: tp.Program,
        path: tuple[int, ...],
    ) -> tp.CompiledEditSite:
        compiled_calls.append((current, path))
        return real_compile(current, path)

    monkeypatch.setattr(observation.tp, "compile_edit_site", recording_compile)
    first = observation.observe_current_edit_site(program, (), contexts)
    second = observation.observe_current_edit_site(program, (), contexts)

    assert compiled_calls == [(program, ()), (program, ())]
    assert first.program_sha256 == second.program_sha256 == program.sha256
    assert first.compiled_sha256 == second.compiled_sha256
    assert first.contexts_sha256 == second.contexts_sha256
    assert first.output_sha256 == second.output_sha256
    assert first.sha256 == second.sha256
    reordered = observation.observe_current_edit_site(
        program,
        (),
        tuple(reversed(contexts)),
    )
    assert reordered.contexts_sha256 != first.contexts_sha256
    assert reordered.output_sha256 != first.output_sha256


def test_module_has_no_edit_candidate_or_parser_private_seam() -> None:
    source = Path(observation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            call_names.append(node.func.attr)

    assert {
        "replace_subtree",
        "apply_factor_action",
        "apply_selected_wrap",
        "program_edit_sites",
        "Program",
        "compile",
        "_parse_expression",
        "_validate_program",
    }.isdisjoint(call_names)
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tp"
        and node.func.attr == "compile_edit_site"
        for node in ast.walk(tree)
    ) == 1
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "tp"
        and node.attr.startswith("_")
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    ("contexts", "error"),
    (
        (((1, 2),), ValueError),
        (((True,),), TypeError),
    ),
)
def test_contexts_must_match_the_compiled_local_scope(
    contexts: tuple[tuple[object, ...], ...],
    error: type[Exception],
) -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("x", integer),),
        result=integer,
    )
    program = tp.Program(("variable-0",), signature)

    with pytest.raises(error):
        observation.observe_current_edit_site(program, (), contexts)
