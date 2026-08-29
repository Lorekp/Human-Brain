from __future__ import annotations

from time import perf_counter

from experiments.r3_pilot import public_typed_operator_successor as successor
from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


_FIXTURE_NAME = "public-TRAIN-derived/scalar-offset/four-visible-rows"


def test_enclosing_checkpoint_authentication_preserves_train_fixture_ledger(
    monkeypatch,
) -> None:
    """Compare the optimization with the exact pre-change evaluator path.

    The named fixture follows the public-TRAIN scalar-offset visibility shape.
    Only four deterministic regression-style rows enter synthesis; this is
    target-free engineering coverage, not a capability, property-case,
    held-out, or transfer result.
    """

    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("value", integer),),
        result=integer,
    )
    arguments = ((-4,), (0,), (3,), (8,))
    desired = tuple(row[0] + 2 for row in arguments)
    model, _summary = semantic.build_public_typed_semantic_synthesizer(
        literals=(-3, 0, 2),
    )

    optimized_predict = semantic._CaseEvaluator._predict
    original_hash = successor._array_sha256
    hash_calls = 0

    def counted_hash(*arrays):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(*arrays)

    monkeypatch.setattr(successor, "_array_sha256", counted_hash)

    reference_ledger = []

    def reference_predict(self, selected, action, scopes=()):
        prediction = self.model.predict(selected, action, scopes=scopes)
        self.predictions += 1
        self.races.extend(prediction.races)
        reference_ledger.append((prediction.output.sha256, prediction.races))
        return prediction.output

    model.operator.clear_effect_memo()
    monkeypatch.setattr(semantic._CaseEvaluator, "_predict", reference_predict)
    started = perf_counter()
    reference = model.synthesize(signature, arguments, desired)
    reference_seconds = perf_counter() - started
    reference_hash_calls = hash_calls

    optimized_ledger = []

    def recording_optimized_predict(self, selected, action, scopes=()):
        races_before = len(self.races)
        output = optimized_predict(self, selected, action, scopes)
        optimized_ledger.append(
            (output.sha256, tuple(self.races[races_before:]))
        )
        return output

    model.operator.clear_effect_memo()
    hash_calls = 0
    monkeypatch.setattr(
        semantic._CaseEvaluator,
        "_predict",
        recording_optimized_predict,
    )
    started = perf_counter()
    optimized = model.synthesize(signature, arguments, desired)
    optimized_seconds = perf_counter() - started
    optimized_hash_calls = hash_calls

    assert optimized == reference
    assert optimized_ledger == reference_ledger
    assert reference_hash_calls == 2 * reference.operator_predictions + 2
    assert optimized_hash_calls == 2
    assert optimized.checkpoint_before_sha256 == model.checkpoint_sha256
    assert optimized.checkpoint_after_sha256 == model.checkpoint_sha256
    assert model.operator.live_checkpoint_sha256 == model.operator.checkpoint_sha256
    print(
        {
            "fixture": _FIXTURE_NAME,
            "reference_seconds": reference_seconds,
            "optimized_seconds": optimized_seconds,
            "reference_hash_calls": reference_hash_calls,
            "optimized_hash_calls": optimized_hash_calls,
            "operator_predictions": optimized.operator_predictions,
            "operator_races": optimized.operator_races,
            "operator_races_sha256": optimized.operator_races_sha256,
            "checkpoint_sha256": optimized.checkpoint_before_sha256,
        }
    )
