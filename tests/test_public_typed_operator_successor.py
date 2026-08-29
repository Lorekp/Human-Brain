from __future__ import annotations

import ast
import inspect
from functools import cache
from pathlib import Path

from experiments.r3_pilot import public_typed_operator_successor as successor
from experiments.r3_pilot import public_typed_output_cells as outputs
from humanbrain.typed_program import FactorAction


@cache
def _trained():
    return successor.train_public_typed_operator_successor()


def _encode(value, declared_type=None):
    return outputs.TypedOutputCellBank.encode(value, declared_type=declared_type)


def _active_int(encoded) -> int | None:
    cell = int(encoded.int_cells.argmax())
    return None if cell == outputs.INT_UNKNOWN_CELL else outputs.INT_MIN + cell


def _active_bool(encoded) -> bool | None:
    cell = int(encoded.bool_cells.argmax())
    if cell == outputs.BOOL_UNKNOWN_CELL:
        return None
    return cell == outputs.BOOL_TRUE_CELL


def test_target_free_integer_operator_transitions_and_unknown_are_physical() -> None:
    model, summary = _trained()
    selected = _encode(-7, outputs.OutputType.INT)

    absolute = model.predict(
        selected,
        FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None),
    )
    add = model.predict(
        selected,
        FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 2),
    )
    overflow = model.predict(
        _encode(20, outputs.OutputType.INT),
        FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 2),
    )

    assert _active_int(absolute.output) == 7
    assert _active_int(add.output) == -5
    assert overflow.output.supported is False
    assert overflow.output.unknown_active
    assert all(race.simultaneous_spike_candidates == 1 for race in absolute.races + add.races + overflow.races)
    assert model.live_checkpoint_sha256 == summary.checkpoint_sha256
    assert summary.primitive_observations == 41 + 3 * 41 * 41


def test_scope_binding_and_operand_orientation_share_one_successor() -> None:
    model, _ = _trained()
    selected = _encode(2, outputs.OutputType.INT)
    scopes = (
        _encode(2, outputs.OutputType.INT),
        _encode(-7, outputs.OutputType.INT),
    )

    a_minus_b = model.predict(
        selected,
        FactorAction((), "APPLY2", "subtract", "SCOPE1", "SCOPE0", None),
        scopes=scopes,
    )
    b_minus_a = model.predict(
        selected,
        FactorAction((), "APPLY2", "subtract", "SCOPE0", "SCOPE1", None),
        scopes=scopes,
    )

    assert _active_int(a_minus_b.output) == -9
    assert _active_int(b_minus_a.output) == 9


def test_less_than_orientation_and_bound_body_effects() -> None:
    model, _ = _trained()
    selected = _encode(-3, outputs.OutputType.INT)
    scopes = (
        _encode(-3, outputs.OutputType.INT),
        _encode([-3, 0, 2], outputs.OutputType.LIST_INT),
    )

    variable_first = model.predict(
        selected,
        FactorAction((), "APPLY2", "less-than", "BOUND0", "LITERAL", 2),
        scopes=scopes,
    )
    literal_first = model.predict(
        selected,
        FactorAction((), "APPLY2", "less-than", "LITERAL", "BOUND0", 2),
        scopes=scopes,
    )
    body_add = model.predict(
        selected,
        FactorAction((), "APPLY2", "add", "BOUND0", "LITERAL", 2),
        scopes=scopes,
    )

    assert _active_bool(variable_first.output) is True
    assert _active_bool(literal_first.output) is False
    assert _active_int(body_add.output) == -1


def test_done_and_map_bind_copy_factorized_outputs_through_cell_races() -> None:
    model, _ = _trained()
    values = _encode([-3, 0, 2], outputs.OutputType.LIST_INT)

    done = model.predict(
        values,
        FactorAction((), "DONE", "NONE", "NONE", "NONE", None),
    )
    mapped = model.predict(
        values,
        FactorAction((), "MAP_BIND", "map", "BOUND0", "SELECTED", None),
    )

    assert done.output.sha256 == values.sha256
    assert mapped.output.sha256 == values.sha256
    assert len(done.races) == len(mapped.races) == 5
    assert all(race.simultaneous_spike_candidates == 1 for race in done.races + mapped.races)


def test_predict_many_is_readonly_and_inference_has_no_candidate_program_path() -> None:
    model, _ = _trained()
    action = FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None)
    rows = tuple(_encode(value, outputs.OutputType.INT) for value in (-4, 0, 5))
    before = model.live_checkpoint_sha256
    predictions = model.predict_many(rows, action)

    assert tuple(_active_int(item.output) for item in predictions) == (4, 0, 5)
    assert model.live_checkpoint_sha256 == before

    source = Path(successor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    inference_names = {"predict", "predict_many", "_unary", "_binary", "_source"}
    inference_source = "\n".join(
        ast.get_source_segment(source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in inference_names
    )
    for forbidden in (
        "apply_factor_action",
        "replace_subtree",
        "tp.compile",
        "exec(",
        "sorted(",
        "argmax",
    ):
        assert forbidden not in inference_source
    assert "abs(" not in inspect.getsource(successor.FrozenTypedOperatorSuccessor.predict)
