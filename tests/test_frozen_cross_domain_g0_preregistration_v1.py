from __future__ import annotations

import dataclasses
import inspect

import pytest
from experiments.g0 import frozen_cross_domain_g0_preregistration_v1 as g0

EXPECTED_ROOTS = {
    "adapter_sha256": "29a6dcec69d89c74c373e7f4bf8709e5e57d7cc06d3aa15f8ad3a2691db1fd99",
    "regulatory_sha256": "303843e486994137aeba2ae7d63b11c16542bc82071f360d1d672911808a874d",
    "spatial_sha256": "8705597cd610e00390a73bcd926fc8b3bcbfbd8be07b08646214972ad8402100",
    "statistics_sha256": "1a834201548c865c4b3ed59defbcff8adc07e33a0e86c22562951ed685caf445",
    "surface_boundary_sha256": "9fe2a478a4ac941c6d9c2f570d8b78dee15084e9f64a03285fe2f69f185a6890",
    "protocol_sha256": "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468",
}


def test_exact_seeds_splits_schedules_and_stratum_arithmetic_are_frozen() -> None:
    assert g0.LEARNER_SEEDS == (17, 29, 43, 71, 101)
    assert g0.BOOTSTRAP_RESAMPLES == 50_000
    assert g0.REGULATORY_SPLIT.record() == {
        "smoke": {"seed": 470000, "count": 64},
        "learning": {"seed": 470001, "count": 512},
        "adjudicative": {"seed": 470002, "count": 128},
        "rename": {"seed": 470003, "scope": "learning+adjudicative"},
        "whole_system_or_layout_holdout": True,
        "canonical_identity_overlap_allowed": False,
    }
    assert g0.SPATIAL_SPLIT.record()["smoke"] == {"seed": 470100, "count": 64}
    assert g0.SPATIAL_SPLIT.record()["learning"] == {"seed": 470101, "count": 512}
    assert g0.SPATIAL_SPLIT.record()["adjudicative"] == {"seed": 470102, "count": 128}
    assert g0.SPATIAL_SPLIT.rename_seed == 470103

    assert sum(g0.REGULATORY_SCHEDULE["query_strata_per_system"].values()) == 48
    assert g0.REGULATORY_SCHEDULE["queries_per_learner_seed"] == 128 * 48 == 6_144
    assert sum(g0.SPATIAL_SPECIFICATION["learning_layout_partition"].values()) == 512
    assert sum(g0.SPATIAL_SPECIFICATION["adjudicative_layout_partition"].values()) == 128
    assert g0.SPATIAL_SCHEDULE["learning_episodes_per_learner_seed"] == 512 * 32
    assert g0.SPATIAL_SCHEDULE["adjudicative_episodes_all_learner_seeds"] == 128 * 4 * 5


def test_thresholds_rename_baselines_and_four_causal_ablations_are_exact() -> None:
    regulatory_thresholds = tuple(
        (row.metric, row.point_estimate_minimum, row.lower_confidence_bound_minimum)
        for row in g0.REGULATORY_THRESHOLDS
    )
    assert regulatory_thresholds == (
        ("all-heldout-queries", 0.82, 0.78),
        ("next-state-prediction", 0.85, 0.80),
        ("single-intervention", 0.80, 0.75),
        ("paired-counterfactual", 0.75, 0.70),
        ("relation-composition", 0.75, 0.70),
        ("delayed-memory", 0.75, 0.70),
    )
    spatial_thresholds = tuple(
        (row.metric, row.point_estimate_minimum, row.lower_confidence_bound_minimum)
        for row in g0.SPATIAL_THRESHOLDS
    )
    assert spatial_thresholds == (
        ("all-heldout-layout-success", 0.70, 0.65),
        ("fresh-memory-layouts", 0.75, 0.68),
        ("fresh-key-door-layouts", 0.70, 0.63),
        ("heldout-compositions", 0.60, 0.55),
        ("delayed-cue-subset", 0.70, 0.63),
        ("multi-step-intervention-subset", 0.60, 0.55),
    )
    assert g0.REGULATORY.rename_requirements["mapped-query-agreement-minimum"] == 0.90
    assert g0.SPATIAL.rename_requirements["paired-success-failure-agreement-minimum"] == 0.85
    assert len(g0.REGULATORY.baselines) == 5
    assert len(g0.SPATIAL.baselines) == 4
    assert set(g0.FOCUSED_ABLATIONS) == {
        "learning-off",
        "cell-or-output-population-off",
        "working-and-persistent-memory-reset-each-observation",
        "fixed-representation-consequence-action-derangement",
        "post_outcome_ablation_expansion_forbidden",
    }


def test_common_adapter_is_data_only_and_generation_remains_fail_closed() -> None:
    assert tuple(g0.EVENT_ENVELOPE) == (
        "observation",
        "action",
        "outcome",
        "boundary",
        "query",
        "shape_and_padding_are_serialization_only",
    )
    assert g0.ADAPTER_CONTRACT["domain_specific_computational_primitive_delta_required"] == 0
    assert "adjacency" in g0.ADAPTER_CONTRACT["regulatory_forbidden"]
    assert "mission" in g0.ADAPTER_CONTRACT["spatial_forbidden"]
    assert "partial-image-7x7x3" in g0.ADAPTER_CONTRACT["spatial_allowed"]
    assert len(g0.COMMON_ARCHITECTURE_FREEZE_REQUIREMENTS) == 9

    surface = g0.SURFACE_BOUNDARY_RECORD
    assert surface["generation_authorized"] is False
    assert surface["common_architecture_binding_required_before_generation"] is True
    assert surface["data_only"] is True
    assert all(
        surface[key] == 0
        for key in surface
        if key.endswith("_count")
    )

    source = inspect.getsource(g0)
    assert "import random" not in source
    assert "import numpy" not in source
    assert "import minigrid" not in source
    assert "def generate" not in source
    assert "def evaluate" not in source


def test_protocol_roots_rebuild_and_all_records_are_immutable() -> None:
    protocol = g0.PROTOCOL
    assert protocol.status == "FROZEN_DESIGN_AWAITING_COMMON_ARCHITECTURE_BINDING"
    assert protocol == g0._build_protocol()
    actual_roots = {
        field.name: getattr(protocol, field.name)
        for field in dataclasses.fields(protocol)
        if field.name.endswith("sha256")
    }
    assert actual_roots == EXPECTED_ROOTS
    assert protocol.protocol_sha256 == g0._namespaced_sha256(
        g0._PROTOCOL_NAMESPACE,
        protocol.root_record(include_protocol_sha256=False),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "PASS"  # type: ignore[misc]
    with pytest.raises(TypeError):
        protocol.adapter_contract["forged"] = True  # type: ignore[index]
