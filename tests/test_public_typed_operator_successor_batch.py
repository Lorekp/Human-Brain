from __future__ import annotations

import hashlib
import json
from time import perf_counter

import numpy as np

from experiments.r3_pilot import public_typed_factor_h1_root as planner
from experiments.r3_pilot import public_typed_operator_successor as successor
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_program_observation as observation
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


# 81d4e94's original goldens predated c66ed7f expanding Int cells from
# [-20, 20] to [-64, 64].  The checkpoint authenticates that projection.
_CHECKPOINT_SHA256 = "62100b5296fe37eb13b3a7a1d9f92f032de494af12c95e20c3d46c4409cca8d0"
_ROOT_LEDGER_SHA256 = "7519a7f5887f39047379914b37e7551ef2ea4121264661abbf4c31473314ff6e"


def _encode_int(value: int):
    return output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=output_cells.OutputType.INT,
    )


def _dense_binary_effect(model, operator, left, right):
    pair_currents = np.ascontiguousarray(
        model.pair_bias_pa
        + model.pair_left_contacts_pa @ left.int_cells
        + model.pair_right_contacts_pa @ right.int_cells
    )
    pair_race = successor._physical_race(
        pair_currents,
        bank="operator-int-pair",
    )
    pair_spike = np.zeros(successor._PAIR_COUNT, dtype=np.float64)
    pair_spike[pair_race.winner_cell] = 1.0
    if operator == "less-than":
        output_currents = np.ascontiguousarray(
            model.bool_output_bias_pa + model.less_than_weights_pa @ pair_spike
        )
        output_race = successor._physical_race(
            output_currents,
            bank="operator-less-than-output",
        )
        output = successor._bool_from_winner(output_race.winner_cell)
    else:
        weights = (
            model.add_weights_pa
            if operator == "add"
            else model.subtract_weights_pa
        )
        output_currents = np.ascontiguousarray(
            model.int_output_bias_pa + weights @ pair_spike
        )
        output_race = successor._physical_race(
            output_currents,
            bank=f"operator-{operator}-output",
        )
        output = successor._int_from_winner(output_race.winner_cell)
    return output, (pair_race, output_race)


def _dense_unary_effect(model, operand):
    currents = np.ascontiguousarray(
        model.unary_bias_pa + model.unary_absolute_weights_pa @ operand.int_cells
    )
    race = successor._physical_race(
        currents,
        bank="operator-absolute-output",
    )
    return successor._int_from_winner(race.winner_cell), (race,)


def _two_int_incumbent() -> tp.Program:
    integer = ProgramType.Int()
    signature = TypedSignature(
        "solve",
        (
            ProgramParameter("a", integer),
            ProgramParameter("b", integer),
        ),
        integer,
    )
    return tp.Program(("absolute", "variable-0"), signature)


def _root_workload():
    incumbent = _two_int_incumbent()
    model = planner.build_public_typed_factor_h1_root()
    current = observation.observe_current_program_root(
        incumbent,
        planner.PUBLIC_TWO_INT_CASES,
    )
    selected = tuple(case.output_encoding for case in current.cases)
    scopes = planner._scope_rows(incumbent, current)
    bank = tp.factor_action_candidates(
        incumbent,
        literals=planner.PUBLIC_LITERAL_BANK,
    )
    root_actions = tuple(action for action in bank.actions if action.path == ())
    return incumbent, model, selected, scopes, bank, root_actions


def _assert_prediction_parity(left, right) -> None:
    assert left.output.sha256 == right.output.sha256
    assert left.races == right.races
    assert left.supported == right.supported
    assert left.action == right.action
    assert left.checkpoint_before_sha256 == right.checkpoint_before_sha256
    assert left.checkpoint_after_sha256 == right.checkpoint_after_sha256


def _prediction_ledger_sha256(bank, root_actions, batched) -> str:
    predictions_by_action = dict(zip(root_actions, batched, strict=True))
    rows = [
        [
            [
                list(action.path),
                action.schema,
                action.operator,
                action.left_source,
                action.right_source,
                action.literal,
            ],
            [
                [
                    prediction.output.sha256,
                    [
                        [
                            race.bank,
                            race.candidate_count,
                            race.winner_cell,
                            race.spike_step,
                            race.simultaneous_spike_candidates,
                            race.currents_sha256,
                        ]
                        for race in prediction.races
                    ],
                ]
                for prediction in predictions_by_action.get(action, ())
            ],
        ]
        for action in bank.actions
    ]
    payload = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_action_batch_matches_scalar_races_and_reduces_checkpoint_hashing(
    monkeypatch,
) -> None:
    _, model, selected, scopes, bank, root_actions = _root_workload()
    operator = model.operator_successor
    original_hash = successor._array_sha256
    hash_calls = 0

    def counted_hash(*arrays):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(*arrays)

    monkeypatch.setattr(successor, "_array_sha256", counted_hash)
    scalar = tuple(
        operator.predict_many(selected, action, scope_rows=scopes)
        for action in root_actions
    )
    scalar_hash_calls = hash_calls
    hash_calls = 0
    batched = operator.predict_actions_many(
        selected,
        root_actions,
        scope_rows=scopes,
    )

    assert len(bank.actions) == 212
    assert len(root_actions) == 130
    assert successor._unknown.cache_parameters()["maxsize"] == 4
    assert (
        successor._int_from_winner.cache_parameters()["maxsize"]
        == successor.output_cells.INT_CELL_COUNT
    )
    assert (
        successor._literal_int.cache_parameters()["maxsize"]
        == successor.output_cells.INT_CELL_COUNT
    )
    assert scalar_hash_calls == 2 * len(root_actions)
    assert hash_calls == 2
    assert len(batched) == len(scalar)
    for scalar_rows, batched_rows in zip(scalar, batched, strict=True):
        assert len(batched_rows) == len(scalar_rows) == 3
        for scalar_prediction, batched_prediction in zip(
            scalar_rows,
            batched_rows,
            strict=True,
        ):
            _assert_prediction_parity(scalar_prediction, batched_prediction)
            assert not batched_prediction.output.cells.flags.writeable
            assert not batched_prediction.output.active_factor_mask.flags.writeable
    assert operator.checkpoint_sha256 == _CHECKPOINT_SHA256
    assert operator.live_checkpoint_sha256 == _CHECKPOINT_SHA256


def test_target_free_root_candidate_ledger_is_bit_exact_after_batching() -> None:
    _, model, selected, scopes, bank, root_actions = _root_workload()
    batched = model.operator_successor.predict_actions_many(
        selected,
        root_actions,
        scope_rows=scopes,
    )

    assert len(bank.actions) == 212
    assert _prediction_ledger_sha256(bank, root_actions, batched) == _ROOT_LEDGER_SHA256
    assert model.operator_successor.checkpoint_sha256 == _CHECKPOINT_SHA256
    assert model.operator_successor.live_checkpoint_sha256 == _CHECKPOINT_SHA256


def test_event_sparse_batches_preserve_dense_full_current_ledgers() -> None:
    model, summary = successor.train_public_typed_operator_successor()
    invalid = output_cells.TypedOutputCellBank.encode(
        [1],
        declared_type=output_cells.OutputType.LIST_INT,
    )
    rows = (
        (_encode_int(output_cells.INT_MIN), _encode_int(output_cells.INT_MIN)),
        (_encode_int(-7), _encode_int(2)),
        (_encode_int(50), _encode_int(9)),
        (_encode_int(output_cells.INT_MAX), _encode_int(2)),
        (invalid, _encode_int(1)),
    )

    for operator in ("add", "subtract", "less-than"):
        dense = tuple(
            _dense_binary_effect(model, operator, left, right)
            for left, right in rows
        )
        sparse = model._binary_many(
            operator,
            tuple(left for left, _ in rows),
            tuple(right for _, right in rows),
        )
        assert tuple((output.sha256, races) for output, races in sparse) == tuple(
            (output.sha256, races) for output, races in dense
        )

    for operand in (
        _encode_int(output_cells.INT_MIN),
        _encode_int(0),
        _encode_int(output_cells.INT_MAX),
        invalid,
    ):
        assert model._unary(operand) == _dense_unary_effect(model, operand)

    assert model.checkpoint_sha256 == summary.checkpoint_sha256 == _CHECKPOINT_SHA256
    assert model.live_checkpoint_sha256 == _CHECKPOINT_SHA256


def test_event_sparse_binary_batch_has_bounded_memory_and_speedup() -> None:
    model, _ = successor.train_public_typed_operator_successor()
    pairs = tuple(
        (_encode_int(left), _encode_int(right))
        for left in range(-8, 0)
        for right in range(4)
    )

    dense_started = perf_counter()
    dense = tuple(
        _dense_binary_effect(model, "add", left, right)
        for left, right in pairs
    )
    dense_seconds = perf_counter() - dense_started
    sparse_started = perf_counter()
    sparse = model._binary_many(
        "add",
        tuple(left for left, _ in pairs),
        tuple(right for _, right in pairs),
    )
    sparse_seconds = perf_counter() - sparse_started

    assert tuple((output.sha256, races) for output, races in sparse) == tuple(
        (output.sha256, races) for output, races in dense
    )
    assert successor._EVENT_BATCH_MAX_ROWS == 256
    assert (
        successor._EVENT_BATCH_MAX_ROWS
        * successor._PAIR_COUNT
        * np.dtype(np.float64).itemsize
        < 35_000_000
    )
    assert sparse_seconds * 2 < dense_seconds


def test_event_sparse_batch_only_races_lru_misses_and_accepts_empty_rows(
    monkeypatch,
) -> None:
    memo = successor._OperatorEffectMemo(max_entries=2)
    effect = (_encode_int(0), ())
    keys = tuple(("checkpoint", "binary", "add", str(index)) for index in range(3))
    memo.resolve(keys[0], lambda: effect)
    memo.resolve(keys[1], lambda: effect)
    before_plan = memo.info()
    assert memo.miss_mask((keys[0], keys[2], keys[1], keys[0])) == (
        False,
        True,
        True,
        True,
    )
    assert memo.info() == before_plan

    model, summary = successor.train_public_typed_operator_successor()
    action = tp.FactorAction(
        (),
        "APPLY2",
        "add",
        "SELECTED",
        "LITERAL",
        1,
    )
    original_binary_many = successor.FrozenTypedOperatorSuccessor._binary_many
    raced_rows: list[int] = []

    def counted_binary_many(self, operator, left_rows, right_rows):
        raced_rows.append(len(left_rows))
        return original_binary_many(self, operator, left_rows, right_rows)

    monkeypatch.setattr(
        successor.FrozenTypedOperatorSuccessor,
        "_binary_many",
        counted_binary_many,
    )
    model.clear_effect_memo()
    model.predict_actions_many((_encode_int(0), _encode_int(1)), (action,))
    cold = model.effect_memo_info()
    mixed_rows = (_encode_int(0), _encode_int(2), _encode_int(1))
    mixed = model.predict_actions_many(mixed_rows, (action,))[0]
    mixed_info = model.effect_memo_info()
    warm = model.predict_actions_many(mixed_rows, (action,))[0]
    warm_info = model.effect_memo_info()

    assert raced_rows == [2, 1]
    assert cold == successor.OperatorEffectMemoInfo(4_096, 2, 0, 2, 0)
    assert mixed_info == successor.OperatorEffectMemoInfo(4_096, 3, 2, 3, 0)
    assert warm_info == successor.OperatorEffectMemoInfo(4_096, 3, 5, 3, 0)
    assert tuple((row.output.sha256, row.races) for row in warm) == tuple(
        (row.output.sha256, row.races) for row in mixed
    )
    assert model.predict_actions_many((), (action,)) == ((),)
    assert model.live_checkpoint_sha256 == summary.checkpoint_sha256
