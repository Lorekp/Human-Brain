from __future__ import annotations

import inspect

import pytest
from experiments.r3_pilot import causal_append_effect_reducer_learner_v1 as append
from experiments.r3_pilot import causal_occurrence_absolute_map_fold_v1 as map_fold
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import (
    typed_semantic_packed_prompt_authorized_plan_slice_v1 as plan_slice,
)


def test_raw_sum_binding_is_narrow_full_range_and_desired_free() -> None:
    signature = map_fold.absolute.list_int_to_int_signature()
    boundary_cases = (([64],), ([-2],), ([4],), ([5],))
    boundary_desired = (64, -2, 4, 5)

    build_source = inspect.getsource(map_fold._RawSumListIntBuilder.build)
    assert "desired" not in build_source
    assert "threshold" not in build_source
    assert "_predicate_count" not in build_source
    assert "less-than" not in build_source
    assert '"add"' not in build_source
    assert '"maximum"' not in build_source
    assert '"minimum"' not in build_source
    assert map_fold.RAW_SUM_DUMMY_LITERALS == (0,)

    delegate, binding = map_fold._raw_sum_binding(signature, boundary_cases)
    receipt = delegate.plan_slice_receipt
    assert delegate.status == plan_slice.FRONTIER_READY
    assert delegate.state.authorized_slice_count == 1
    assert delegate._frontier.state.literals == (0,)
    assert receipt is not None
    assert (
        plan_slice._authenticate_plan_slice_receipt(delegate._frontier, receipt)
        is binding
    )
    scaffold = append.freeze_raw_sum_scaffold()
    plan_slice._authenticate_certificate(scaffold)
    assert delegate.state.structural_certificate_sha256 == scaffold.sha256
    assert tuple(
        output_cells.TypedOutputCellBank.decode(item)
        for item in binding.candidate.eclass.physical_outputs
    ) == boundary_desired
    assert delegate.work.concrete_proof_witnesses == 0
    assert delegate.work.materialized_witnesses == 0

    selected = delegate.select(boundary_desired)
    assert selected.status == plan_slice.SELECTED
    assert selected.exact_proof_count == 1
    assert len(selected.witnesses) == 1
    assert selected.witnesses[0].source == (
        "def aggregate(numbers):\n    return sum(numbers)\n"
    )

    ordinary_cases = (([3, -1],), ([-4, 2],))
    ordinary_delegate, _ordinary_binding = map_fold._raw_sum_binding(
        signature, ordinary_cases
    )
    ordinary_selected = ordinary_delegate.select((2, -2))
    assert ordinary_selected.status == plan_slice.SELECTED
    assert ordinary_selected.exact_proof_count == 1
    assert ordinary_selected.witnesses[0].source == selected.witnesses[0].source

    with pytest.raises(map_fold.OccurrenceAbsoluteMapFoldFitStop) as source_stop:
        map_fold._raw_sum_binding(
            signature,
            boundary_cases,
            source_cell_gate=False,
        )
    assert source_stop.value.status == map_fold.STOP_PLAN_SLICE

    with pytest.raises(map_fold.OccurrenceAbsoluteMapFoldFitStop) as scaffold_stop:
        map_fold._raw_sum_binding(
            signature,
            boundary_cases,
            scaffold_cell_gate=False,
        )
    assert scaffold_stop.value.status == map_fold.STOP_PLAN_SLICE
