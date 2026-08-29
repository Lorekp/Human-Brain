from __future__ import annotations

from time import perf_counter

from experiments.r3_pilot import public_typed_operator_successor as successor
from experiments.r3_pilot import public_typed_output_cells as outputs
from humanbrain.typed_program import FactorAction


def _encode(value: int) -> outputs.TypedOutputEncoding:
    return outputs.TypedOutputCellBank.encode(
        value,
        declared_type=outputs.OutputType.INT,
    )


def test_normalized_effect_memo_preserves_fresh_actions_and_checkpoint() -> None:
    model, summary = successor.train_public_typed_operator_successor()
    rows = (_encode(outputs.INT_MIN), _encode(outputs.INT_MAX - 3))
    cold_action = FactorAction(
        (), "APPLY2", "add", "SELECTED", "LITERAL", 3
    )
    warm_action = FactorAction(
        (0,), "APPLY2", "add", "SELECTED", "LITERAL", 3
    )
    model.clear_effect_memo()

    cold_started = perf_counter()
    cold = model.predict_actions_many(rows, (cold_action,))[0]
    cold_seconds = perf_counter() - cold_started
    cold_info = model.effect_memo_info()

    warm_started = perf_counter()
    warm = model.predict_actions_many(rows, (warm_action,))[0]
    warm_seconds = perf_counter() - warm_started
    warm_info = model.effect_memo_info()

    print(
        f"operator memo A/B: cold={cold_seconds:.6f}s "
        f"warm={warm_seconds:.6f}s hits={warm_info.hits}"
    )
    assert cold_info.max_entries == 4_096
    assert cold_info.max_entries < outputs.INT_KNOWN_CELL_COUNT**2
    assert cold_info.entries == cold_info.misses == len(rows)
    assert cold_info.hits == cold_info.evictions == 0
    assert warm_info.misses == cold_info.misses
    assert warm_info.hits - cold_info.hits == len(rows)
    assert tuple((item.output.sha256, item.races) for item in warm) == tuple(
        (item.output.sha256, item.races) for item in cold
    )
    assert all(item.action is warm_action for item in warm)
    assert all(warm_item is not cold_item for warm_item, cold_item in zip(warm, cold))
    assert model.live_checkpoint_sha256 == summary.checkpoint_sha256

    model.clear_effect_memo()
    assert model.effect_memo_info() == successor.OperatorEffectMemoInfo(
        max_entries=4_096,
        entries=0,
        hits=0,
        misses=0,
        evictions=0,
    )
    assert model.live_checkpoint_sha256 == summary.checkpoint_sha256


def test_effect_memo_eviction_is_bounded_and_deterministic() -> None:
    memo = successor._OperatorEffectMemo(max_entries=2)
    effect = (_encode(0), ())
    key_1 = ("checkpoint", "binary", "add", "one")
    key_2 = ("checkpoint", "binary", "add", "two")
    key_3 = ("checkpoint", "binary", "add", "three")

    assert memo.resolve(key_1, lambda: effect) is effect
    assert memo.resolve(key_2, lambda: effect) is effect
    assert memo.resolve(key_1, lambda: effect) is effect
    assert memo.resolve(key_3, lambda: effect) is effect
    assert memo.info() == successor.OperatorEffectMemoInfo(2, 2, 1, 3, 1)

    recomputed: list[str] = []
    memo.resolve(key_2, lambda: (recomputed.append("key-2"), effect)[1])
    assert recomputed == ["key-2"]
    assert memo.info() == successor.OperatorEffectMemoInfo(2, 2, 1, 4, 2)
