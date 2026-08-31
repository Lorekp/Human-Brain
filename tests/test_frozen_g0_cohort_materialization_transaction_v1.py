from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from experiments.g0 import frozen_g0_authorized_generator_entrypoints_v2 as plans
from experiments.g0 import frozen_g0_cohort_materialization_transaction_v1 as transaction


def test_exact_plan_transaction_stops_before_impossible_cohort_spend() -> None:
    regulatory = plans.REGULATORY_PLAN
    spatial = plans.SPATIAL_PLAN
    equal_regulatory = dataclasses.replace(regulatory)
    copied_regulatory = copy.copy(regulatory)
    field_forged_regulatory = plans.FrozenGenerationPlan(
        **{**regulatory.record(), "domain": "field-forged-regulatory"}
    )
    same_type_uninitialized = object.__new__(plans.FrozenGenerationPlan)

    assert equal_regulatory == regulatory and equal_regulatory is not regulatory
    assert copied_regulatory == regulatory and copied_regulatory is not regulatory
    for forged in (
        object(),
        equal_regulatory,
        copied_regulatory,
        field_forged_regulatory,
        same_type_uninitialized,
    ):
        with pytest.raises(
            transaction.G0CohortMaterializationStop,
            match="EXACT_REGULATORY_PLAN_REQUIRED",
        ):
            transaction.materialize_frozen_g0_cohorts(forged, spatial)
    with pytest.raises(
        transaction.G0CohortMaterializationStop,
        match="EXACT_SPATIAL_PLAN_REQUIRED",
    ):
        transaction.materialize_frozen_g0_cohorts(regulatory, dataclasses.replace(spatial))

    with pytest.raises(
        transaction.G0CohortMaterializationStop,
        match="required=192:maximum=120",
    ):
        transaction.materialize_frozen_g0_cohorts(regulatory, spatial)

    assert transaction.MEMORY_S11_REQUIRED_LAYOUT_COUNT == 192
    assert transaction.MEMORY_S11_MAXIMUM_DISTINCT_FULL_STATES == 120
    assert transaction.MATERIALIZATION_STOP.generator_seed_execution_count == 0
    assert transaction.MATERIALIZATION_STOP.artifact_publication_count == 0
    assert transaction.materialization_execution_counts() == transaction.ZERO_EXECUTION_COUNTS
    print(
        json.dumps(
            {
                "source_sha256": transaction.SOURCE_SHA256,
                "config_sha256": transaction.CONFIG_SHA256,
                "transaction_root_sha256": transaction.TRANSACTION_ROOT_SHA256,
                "stop_sha256": transaction.MATERIALIZATION_STOP.sha256,
                "required_layouts": transaction.MEMORY_S11_REQUIRED_LAYOUT_COUNT,
                "maximum_distinct_layouts": (transaction.MEMORY_S11_MAXIMUM_DISTINCT_FULL_STATES),
                "execution_counts": transaction.materialization_execution_counts(),
            },
            sort_keys=True,
        )
    )
