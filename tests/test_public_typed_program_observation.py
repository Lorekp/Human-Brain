import ast
from pathlib import Path

import numpy as np

from experiments.r3_pilot import public_typed_program_observation as observation
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


def _integer_literal(value: int) -> tuple[str, ...]:
    sign = "+" if value >= 0 else "-"
    return ("integer", sign, *tuple(str(abs(value))), "end")


def _one_int_signature(result: ProgramType | None = None) -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("x", integer),),
        result=integer if result is None else result,
    )


def _two_int_signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        entry_point="solve",
        parameters=(
            ProgramParameter("a", integer),
            ProgramParameter("b", integer),
        ),
        result=integer,
    )


def _list_signature(result: ProgramType | None = None) -> TypedSignature:
    integer = ProgramType.Int()
    list_int = ProgramType.List(integer)
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("xs", list_int),),
        result=list_int if result is None else result,
    )


def _raw_outputs(
    result: observation.CurrentProgramRootObservation,
) -> tuple[object, ...]:
    return tuple(case.raw_output for case in result.cases)


def test_observes_both_noncommutative_int_orders_through_current_compiler() -> None:
    signature = _one_int_signature()
    outer_add = tp.Program(
        ("add", "absolute", "variable-0", *_integer_literal(2)),
        signature,
    )
    inner_subtract = tp.Program(
        ("absolute", "subtract", "variable-0", *_integer_literal(2)),
        signature,
    )
    cases = ((-3,), (0,), (5,))

    outer = observation.observe_current_program_root(outer_add, cases)
    inner = observation.observe_current_program_root(inner_subtract, cases)

    assert _raw_outputs(outer) == (5, 2, 7)
    assert _raw_outputs(inner) == (5, 2, 3)
    assert outer.all_supported and inner.all_supported
    assert outer.program_sha256 == outer_add.sha256
    assert inner.program_sha256 == inner_subtract.sha256
    assert outer.cases_sha256 == inner.cases_sha256
    assert outer.output_sha256 != inner.output_sha256
    assert outer.physical_output_cells.shape == (
        len(cases),
        observation.output_cells.TOTAL_CELL_COUNT,
    )


def test_observes_both_binary_operand_bindings() -> None:
    signature = _two_int_signature()
    a_minus_b = tp.Program(
        ("subtract", "variable-1", "variable-0"),
        signature,
    )
    b_minus_a = tp.Program(
        ("subtract", "variable-0", "variable-1"),
        signature,
    )
    cases = ((7, 2), (-3, 4), (5, 5))

    forward = observation.observe_current_program_root(a_minus_b, cases)
    reverse = observation.observe_current_program_root(b_minus_a, cases)

    assert _raw_outputs(forward) == (5, -7, 0)
    assert _raw_outputs(reverse) == (-5, 7, 0)
    assert all(
        case.output_encoding.output_type is observation.output_cells.OutputType.INT
        for case in (*forward.cases, *reverse.cases)
    )


def test_observes_both_bool_operand_orders_as_bool_cells() -> None:
    signature = _one_int_signature(ProgramType.Bool())
    x_less_two = tp.Program(
        ("less-than", "variable-0", *_integer_literal(2)),
        signature,
    )
    two_less_x = tp.Program(
        ("less-than", *_integer_literal(2), "variable-0"),
        signature,
    )
    cases = ((-1,), (2,), (4,))

    forward = observation.observe_current_program_root(x_less_two, cases)
    reverse = observation.observe_current_program_root(two_less_x, cases)

    assert _raw_outputs(forward) == (True, False, False)
    assert _raw_outputs(reverse) == (False, False, True)
    assert all(
        case.output_encoding.output_type is observation.output_cells.OutputType.BOOL
        and np.count_nonzero(case.output_encoding.bool_cells) == 1
        and np.count_nonzero(case.output_encoding.int_cells) == 0
        for case in (*forward.cases, *reverse.cases)
    )


def test_observes_list_absolute_and_list_add_without_whole_list_ids() -> None:
    signature = _list_signature()
    map_absolute = tp.Program(
        ("map", "bind", "absolute", "variable-0", "variable-0"),
        signature,
    )
    map_add_two = tp.Program(
        (
            "map",
            "bind",
            "add",
            "variable-0",
            *_integer_literal(2),
            "variable-0",
        ),
        signature,
    )
    cases = (([-3, 0, 4],), ([1, -2],))

    absolute = observation.observe_current_program_root(map_absolute, cases)
    shifted = observation.observe_current_program_root(map_add_two, cases)

    assert _raw_outputs(absolute) == ((3, 0, 4), (1, 2))
    assert _raw_outputs(shifted) == ((-1, 2, 6), (3, 0))
    assert tuple(absolute.cases[0].arguments[0]) == (-3, 0, 4)
    assert tuple(
        np.count_nonzero(row)
        for row in absolute.cases[1].output_encoding.list_item_cells
    ) == (1, 1, 0, 0)


def test_output_domain_and_result_type_failures_become_physical_unknown() -> None:
    integer_signature = _one_int_signature()
    out_of_domain = tp.Program(
        (
            "add",
            "variable-0",
            *_integer_literal(observation.output_cells.INT_MAX),
        ),
        integer_signature,
    )
    too_large = observation.observe_current_program_root(out_of_domain, ((1,),))

    list_int = ProgramType.List(ProgramType.Int())
    list_bool = ProgramType.List(ProgramType.Bool())
    list_bool_signature = _list_signature(list_bool)
    unsupported_result = tp.Program(
        (
            "map",
            "bind",
            "less-than",
            "variable-0",
            *_integer_literal(0),
            "variable-0",
        ),
        list_bool_signature,
    )
    unsupported = observation.observe_current_program_root(
        unsupported_result,
        (([-1, 0, 2],),),
    )

    identity_list = tp.Program(("variable-0",), _list_signature(list_int))
    overlong = observation.observe_current_program_root(
        identity_list,
        (([0, 1, 2, 3, 4],),),
    )

    assert too_large.cases[0].raw_output == observation.output_cells.INT_MAX + 1
    assert too_large.cases[0].status == observation.STATUS_OUTPUT_UNKNOWN
    assert too_large.cases[0].output_encoding.unknown_active
    assert unsupported.cases[0].raw_output == (True, False, False)
    assert (
        unsupported.cases[0].status
        == observation.STATUS_UNSUPPORTED_RESULT_TYPE
    )
    assert unsupported.cases[0].output_encoding.unknown_active
    assert overlong.cases[0].status == observation.STATUS_OUTPUT_UNKNOWN
    assert overlong.cases[0].output_encoding.unknown_active


def test_hashes_are_deterministic_and_only_current_program_is_compiled(monkeypatch) -> None:
    signature = _one_int_signature()
    program = tp.Program(("absolute", "variable-0"), signature)
    cases = ((-4,), (0,), (3,))
    real_compile = observation.tp.compile
    compiled_programs: list[tp.Program] = []

    def recording_compile(current: tp.Program) -> tp.CompiledPython:
        compiled_programs.append(current)
        return real_compile(current)

    monkeypatch.setattr(observation.tp, "compile", recording_compile)
    first = observation.observe_current_program_root(program, cases)
    second = observation.observe_current_program_root(program, cases)

    assert compiled_programs == [program, program]
    assert first.program_sha256 == second.program_sha256 == program.sha256
    assert first.compiled_sha256 == second.compiled_sha256
    assert first.cases_sha256 == second.cases_sha256
    assert first.output_sha256 == second.output_sha256
    assert first.sha256 == second.sha256
    reordered = observation.observe_current_program_root(program, tuple(reversed(cases)))
    assert reordered.cases_sha256 != first.cases_sha256
    assert reordered.output_sha256 != first.output_sha256


def test_observation_module_has_no_candidate_or_parser_private_seam() -> None:
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
        "_parse_expression",
        "_validate_program",
    }.isdisjoint(call_names)
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tp"
        and node.func.attr == "compile"
        for node in ast.walk(tree)
    ) == 1
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "tp"
        and node.attr.startswith("_")
        for node in ast.walk(tree)
    )
