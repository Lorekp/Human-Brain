from __future__ import annotations

from dataclasses import asdict
from functools import cache

import pytest

from experiments.r3_pilot import public_typed_operator_successor as operator
from experiments.r3_pilot import public_typed_semantic_synthesizer as semantic
from humanbrain.typed_program_types import (
    ProgramParameter,
    ProgramType,
    TypedSignature,
)


_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


def _signature() -> TypedSignature:
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("values", _LIST_INT),),
        result=_INT,
    )


@cache
def _sentinel_a():
    model = semantic.build_public_typed_semantic_synthesizer(
        literals=tuple(range(-12, 13)),
        max_ast_cost=9,
    )[0]
    decision = model.synthesize(
        _signature(),
        (([0, 0, 0, 0],), ([0, 0, 0, 0],)),
        (0, 1),
    )
    return model, decision


def _sentinel_b(desired: tuple[int, ...]):
    model = semantic.build_public_typed_semantic_synthesizer(
        literals=tuple(range(-12, 13)),
        max_ast_cost=9,
    )[0]
    decision = model.synthesize(
        _signature(),
        (
            ([-12, -4, 5, 12],),
            ([12, 5, -4, -12],),
            ([-4, 12, -12, 5],),
            ([5, -12, 12, -4],),
        ),
        desired,
    )
    return model, decision


def test_frozen_budget_schema_and_checkpoint_are_private_and_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert semantic._MAX_RETAINED_REPRESENTATIVES_PER_BUCKET == 256
    assert semantic._MAX_DECLARED_ROOT_CANDIDATES == 6_000
    assert semantic._MAX_OPERATOR_PREDICTIONS == 30_000
    assert semantic._MAX_PHYSICAL_RACES == 80_000
    assert semantic._WORK_BOUND_CHECK_ORDER == (
        "declared-root-candidates",
        "operator-predictions",
        "physical-races",
        "retained-representatives-per-bucket",
    )
    assert semantic._PREDICTION_SCHEMA_MAX_RACES == (
        ("absolute", 1),
        ("add", 2),
        ("subtract", 2),
        ("less-than", 2),
        ("minimum", 4),
        ("maximum", 4),
        ("list-copy", 6),
        ("ascii-class", 1),
        ("contains", 2),
        ("string-copy", 29),
    )

    model = semantic.build_public_typed_semantic_synthesizer(
        literals=(0,), max_ast_cost=2
    )[0]
    checkpoint = model.live_checkpoint_sha256
    assert not hasattr(semantic, "_PREDICTION_SCHEMA_MAX_RACES_BY_NAME")
    revised_schema = (("absolute", 2), *semantic._PREDICTION_SCHEMA_MAX_RACES[1:])
    monkeypatch.setattr(semantic, "_PREDICTION_SCHEMA_MAX_RACES", revised_schema)
    assert model.live_checkpoint_sha256 != checkpoint
    budget = semantic._SemanticFrontierBudget()
    budget.set_context(
        semantic._WorkContext(
            ast_cost=1,
            environment_type_labels=("Int",),
            result_type_label="Int",
            context_shape=("root", 1),
        )
    )
    assert budget.reserve_prediction("absolute") == 2
    assert model.operator.live_checkpoint_sha256 == model.operator.checkpoint_sha256


def test_sentinel_a_remains_complete_and_exhaustive_inside_every_bound() -> None:
    model, decision = _sentinel_a()
    assert decision.status == semantic.STATUS_STOP
    assert decision.work_bound is None
    assert decision.candidate_count == 4_757
    assert decision.observed_semantic_classes == 87
    assert decision.operator_predictions == 23_642
    assert decision.operator_races == 63_456
    assert tuple(
        (
            attempt.ast_cost,
            attempt.evaluated_candidates,
            len(attempt.representative_candidate_ids),
        )
        for attempt in decision.tier_attempts
    ) == (
        (1, 25, 25),
        (2, 26, 0),
        (3, 1_600, 24),
        (4, 24, 0),
        (5, 3_024, 30),
        (6, 2, 0),
        (7, 24, 8),
        (8, 0, 0),
        (9, 72, 0),
    )
    assert decision.checkpoint_before_sha256 == model.checkpoint_sha256
    assert decision.checkpoint_after_sha256 == model.checkpoint_sha256


def test_sentinel_b_stops_before_prediction_30001_and_is_desired_independent() -> None:
    model, decision = _sentinel_b((0, 1, 2, 3))
    other_model, alternate = _sentinel_b((3, 2, 1, 0))

    assert decision.status == semantic.STATUS_WORK_BOUND_STOP
    assert decision.work_bound is not None
    audit = decision.work_bound
    assert audit.bound_name == "operator-predictions"
    assert audit.limit == 30_000
    assert audit.completed == 30_000
    assert audit.attempted == 30_001
    assert audit.ast_cost == 3
    assert audit.completed_declared_candidates == 1_651
    assert audit.completed_supported_root_candidates == 1_441
    assert audit.completed_operator_predictions == 30_000
    assert audit.completed_physical_races == 77_964
    assert audit.completed_retained_representatives == 174
    assert audit.completed_operator_races_sha256 == (
        "506b110fac0558063293d08527c8c0ae6ad89849e5d97b24859d6bd82123970c"
    )
    assert decision.operator_predictions == 30_000
    assert decision.operator_races == 77_964
    assert decision.operator_races_sha256 == audit.completed_operator_races_sha256
    assert tuple(attempt.ast_cost for attempt in decision.tier_attempts) == (1, 2)
    assert decision.checkpoint_before_sha256 == model.checkpoint_sha256
    assert decision.checkpoint_after_sha256 == model.checkpoint_sha256

    assert alternate.status == semantic.STATUS_WORK_BOUND_STOP
    assert alternate.work_bound is not None
    assert asdict(alternate.work_bound) == asdict(audit)
    assert other_model.checkpoint_sha256 == model.checkpoint_sha256


def test_all_four_bounds_are_exactly_inclusive_and_unknown_schema_errors() -> None:
    context = semantic._WorkContext(
        ast_cost=3,
        environment_type_labels=("Int",),
        result_type_label="Int",
        context_shape=("root", 1),
    )

    declared = semantic._SemanticFrontierBudget()
    declared.admit_tier(1, 6_000)
    assert declared.completed_declared_candidates == 6_000
    with pytest.raises(semantic._WorkBoundDenied) as declared_denial:
        declared.admit_tier(2, 1)
    assert declared_denial.value.bound_name == "declared-root-candidates"
    assert declared_denial.value.completed == 6_000
    assert declared_denial.value.attempted == 6_001
    assert declared_denial.value.context is None

    predictions = semantic._SemanticFrontierBudget()
    predictions.set_context(context)
    predictions.completed_operator_predictions = 29_999
    reserved = predictions.reserve_prediction("absolute")
    predictions.complete_prediction((), reserved)
    assert predictions.completed_operator_predictions == 30_000
    with pytest.raises(semantic._WorkBoundDenied) as prediction_denial:
        predictions.reserve_prediction("absolute")
    assert prediction_denial.value.bound_name == "operator-predictions"
    assert prediction_denial.value.attempted == 30_001

    races = semantic._SemanticFrontierBudget()
    races.set_context(context)
    races.completed_physical_races = 79_999
    races.reserve_prediction("absolute")
    races.completed_physical_races = 80_000
    with pytest.raises(semantic._WorkBoundDenied) as race_denial:
        races.reserve_prediction("absolute")
    assert race_denial.value.bound_name == "physical-races"
    assert race_denial.value.completed == 80_000
    assert race_denial.value.attempted == 80_001
    with pytest.raises(RuntimeError, match="unknown physical prediction schema"):
        races.reserve_prediction("unregistered")

    synthetic_race = operator.OperatorRace(
        bank="synthetic",
        candidate_count=1,
        winner_cell=0,
        spike_step=1,
        simultaneous_spike_candidates=1,
        currents_sha256="0" * 64,
    )
    stale_schema = semantic._SemanticFrontierBudget()
    with pytest.raises(RuntimeError, match="conservative race reservation"):
        stale_schema.complete_prediction((synthetic_race, synthetic_race), 1)
    assert stale_schema.completed_operator_predictions == 0
    assert stale_schema.completed_physical_races == 0
    stale_schema.completed_physical_races = 79_999
    with pytest.raises(RuntimeError, match="hard race limit"):
        stale_schema.complete_prediction((synthetic_race, synthetic_race), 2)

    representatives = semantic._SemanticFrontierBudget()
    seen: dict[tuple[object, ...], str] = {}
    retained: set[str] = set()
    bucket = (("Int",), "Int", ("root", 1))
    for index in range(256):
        semantic_key = (*bucket, (str(index).encode("ascii"),))
        assert representatives.retain(
            semantic_key=semantic_key,
            bucket_key=bucket,
            node_id=f"node-{index}",
            seen_semantics=seen,
            retained=retained,
            context=context,
        )
    duplicate = (*bucket, (b"0",))
    assert not representatives.retain(
        semantic_key=duplicate,
        bucket_key=bucket,
        node_id="duplicate",
        seen_semantics=seen,
        retained=retained,
        context=context,
    )
    with pytest.raises(semantic._WorkBoundDenied) as representative_denial:
        representatives.retain(
            semantic_key=(*bucket, (b"overflow",)),
            bucket_key=bucket,
            node_id="overflow",
            seen_semantics=seen,
            retained=retained,
            context=context,
        )
    assert representative_denial.value.bound_name == (
        "retained-representatives-per-bucket"
    )
    assert representative_denial.value.completed == 256
    assert representative_denial.value.attempted == 257
    assert len(seen) == len(retained) == 256


def test_binder_retention_uses_predicate_semantics_and_bound_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_functions: list[tuple[object, ...]] = []
    original = semantic._SemanticFrontierBudget.retain

    def recording_retain(self, **kwargs):
        bucket_key = kwargs["bucket_key"]
        if bucket_key[1] == "Function[Int,Bool]":
            retained_functions.append(kwargs["semantic_key"])
        return original(self, **kwargs)

    monkeypatch.setattr(semantic._SemanticFrontierBudget, "retain", recording_retain)
    model = semantic.build_public_typed_semantic_synthesizer(
        literals=(3,), max_ast_cost=6
    )[0]
    decision = model.synthesize(
        _signature(),
        (([-1, 4],), ([2],), ([],)),
        (1, 0, 0),
    )

    assert decision.status == semantic.STATUS_DONE
    assert retained_functions
    for semantic_key in retained_functions:
        assert semantic_key[:3] == (
            ("List[Int]",),
            "Function[Int,Bool]",
            (
                "bound",
                (
                    (0, 0, "List[Int]", 2),
                    (1, 0, "List[Int]", 1),
                    (2, 0, "List[Int]", 0),
                ),
            ),
        )
        assert semantic_key[3][0] == "predicate"
        predicate_semantic_key = semantic_key[3][1]
        assert predicate_semantic_key[1] == "Bool"
        assert predicate_semantic_key[2] == semantic_key[2]
