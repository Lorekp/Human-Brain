from __future__ import annotations

import hashlib
import json

from experiments.r3_pilot import public_typed_factor_h1_root as planner
from experiments.r3_pilot import public_typed_operator_successor as successor
from experiments.r3_pilot import public_typed_program_observation as observation
from humanbrain import typed_program as tp
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature


_CHECKPOINT_SHA256 = "a15bfe3ec52bc4a9b385da25f7831fc1645df94f9ac5ef16dcc23d194ff4c217"
_ROOT_LEDGER_SHA256 = "207e8e8185b3046d2dd44b914d7dd1eb76a9d767280182e42527d38955f39233"


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
