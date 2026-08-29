from __future__ import annotations

from types import SimpleNamespace

import pytest
from experiments.r3_pilot import public_factorized_source_edit_canary as v1
from experiments.r3_pilot import public_factorized_source_edit_canary_v2 as v2
from experiments.r3_pilot import public_typed_factor_horizon2_int as h2
from humanbrain import typed_program as tp


def _decision(first: tp.FactorAction) -> SimpleNamespace:
    return SimpleNamespace(
        status=h2.STATUS_SELECTED,
        selected_sequence=h2.FactorH2Sequence(first, first),
    )


def test_v2_freezes_v1_protocol_and_exact_v3_prerequisite() -> None:
    assert v2.PUBLIC_TASKS is v1.PUBLIC_TASKS
    assert v2.PUBLIC_LITERAL_BANK is v1.PUBLIC_LITERAL_BANK
    assert v2.PUBLIC_EXECUTION_LIMITS is v1.PUBLIC_EXECUTION_LIMITS
    assert v1._assert_public_protocol() == {
        "abs-value-minus-three": True,
        "three-minus-abs-value": True,
        "list-map-absolute": False,
        "list-map-subtract-two": False,
    }
    assert v1._assert_created_child_prerequisite()["artifact_sha256"] == (
        v1._REQUIRED_CREATED_CHILD_ARTIFACT_SHA256
    )


def test_applied_actions_excludes_selected_but_unactuated_replan() -> None:
    first, second = v1.PUBLIC_TASKS[0].expected_actions
    result = SimpleNamespace(
        checkpoint_audit=SimpleNamespace(
            steps=(
                SimpleNamespace(
                    decision_index=0, live_after_actuation_sha256="frozen"
                ),
                SimpleNamespace(decision_index=1, live_after_actuation_sha256=None),
            )
        ),
        decisions=(_decision(first), _decision(second)),
        actuations=(SimpleNamespace(),),
        edit_count=1,
    )
    assert v2._applied_actions(result) == (first,)


def test_applied_actions_fails_closed_on_inconsistent_ledger() -> None:
    first = v1.PUBLIC_TASKS[0].expected_actions[0]
    result = SimpleNamespace(
        checkpoint_audit=SimpleNamespace(
            steps=(
                SimpleNamespace(
                    decision_index=0, live_after_actuation_sha256="frozen"
                ),
            )
        ),
        decisions=(_decision(first),),
        actuations=(),
        edit_count=0,
    )
    with pytest.raises(RuntimeError, match="actuation ledger is inconsistent"):
        v2._applied_actions(result)
