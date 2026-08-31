from __future__ import annotations

import inspect
from collections.abc import Callable

from experiments.r3_pilot import typed_semantic_packed_absolute_map_fold_v1 as absolute
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)


def _load_aggregate(source: str) -> Callable[[list[int]], int]:
    namespace: dict[str, object] = {}
    exec(source, namespace)
    aggregate = namespace["aggregate"]
    assert callable(aggregate)
    return aggregate


def test_absolute_map_fold_publishes_its_full_range_topology_only() -> None:
    signature = absolute.list_int_to_int_signature()
    boundary_cases = (([64],), ([-2],), ([4],), ([5],))

    build_source = inspect.getsource(absolute._AbsoluteListIntBuilder.build)
    assert "desired" not in build_source
    assert "threshold" not in build_source
    assert "_predicate_count" not in build_source
    assert "less-than" not in build_source
    assert '"add"' not in build_source
    assert '"maximum"' not in build_source
    assert '"minimum"' not in build_source
    assert absolute.FIXED_DUMMY_LITERALS == (0,)

    publication = absolute.PackedPhysicalAbsoluteMapFoldV1.build(
        signature,
        boundary_cases,
    )
    receipt = publication.plan_slice_receipt
    assert publication.status == absolute.READY
    assert publication.state.authorized_slice_count == 1
    assert publication.state.desired_reads_during_build == 0
    assert publication.state.operator_name_inputs_during_build == 0
    assert publication._frontier.state.literals == (0,)
    assert receipt is not None
    plan_slice._authenticate_plan_slice_receipt(publication._frontier, receipt)
    assert publication.work.concrete_proof_witnesses == 0
    assert publication.work.materialized_witnesses == 0
    assert publication._materialization is None

    result = publication.materialize()
    assert result.status == absolute.READY
    assert result.exact_proof_count == 1
    assert result.witness is not None
    aggregate = _load_aggregate(result.witness.source)
    assert [aggregate(row[0]) for row in boundary_cases] == [64, 2, 4, 5]

    ordinary_cases = (([-7, 2, 0],), ([3, -4],))
    ordinary = absolute.PackedPhysicalAbsoluteMapFoldV1.build(
        signature,
        ordinary_cases,
    ).materialize()
    assert ordinary.status == absolute.READY
    assert ordinary.exact_proof_count == 1
    assert ordinary.witness is not None
    ordinary_aggregate = _load_aggregate(ordinary.witness.source)
    assert [ordinary_aggregate(row[0]) for row in ordinary_cases] == [9, 7]

    stopped = absolute.PackedPhysicalAbsoluteMapFoldV1.build(
        signature,
        boundary_cases,
        absolute_cell_gate=False,
    )
    assert stopped.status == absolute.STOP_PLAN_SLICE
    assert stopped.state.authorized_slice_count == 0
    assert stopped.plan_slice_receipt is None
    assert stopped.work.concrete_proof_witnesses == 0
    assert stopped.work.materialized_witnesses == 0
    stopped_result = stopped.materialize()
    assert stopped_result.exact_proof_count == 0
    assert stopped_result.witness is None
