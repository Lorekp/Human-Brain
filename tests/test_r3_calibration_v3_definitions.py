import numpy as np
import pytest
from experiments.r3_pilot import calibration_v3_definitions as definitions
from experiments.r3_pilot import calibration_v3_protocol as protocol
from humanbrain.cortex import DendriticCortex, seeded_threshold_offsets_mv
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
)


def test_native_definition_contract_freezes_every_identity_input() -> None:
    frozen = protocol.build_protocol()

    assert frozen["schema_version"] == 7
    assert frozen["execution_status"] == "NOT_RUNNABLE_PRE_OUTCOME"
    assert frozen["source_binding"]["runnable"] is False
    assert frozen["native_definition_contract"] == {
        "config": {
            "base_cells_per_column": 8,
            "column_count": 4096,
            "connected_permanence_ticks": 250,
            "construction_seed_scope": [17],
            "context_policy": "BOUND_WINNERS",
            "initial_permanence_ticks": 255,
            "max_pyramidal_cells": 131072,
            "max_segments_per_cell": 255,
            "negative_decrement_ticks": 2,
            "novel_race_cap": 8,
            "permanence_increment_ticks": 50,
            "plasticity_policy": "PLASTIC",
            "prediction_quorum": 11,
            "segment_capacity": 23286406,
            "theta_arm_scope": [[10, 13], [16, 18], [18, 20], [20, 20]],
        },
        "threshold_offsets_mv": {
            "dtype": "float64",
            "generator": "numpy.random.default_rng(seed).normal",
            "mean_microvolts": 0,
            "ordering": "ascending_physical_gid",
            "physical_cell_count_rule": ("2 * max_pyramidal_cells + 3 * column_count"),
            "seed_source": "construction_seed",
            "standard_deviation_microvolts": 25,
        },
        "token_codebook": {
            "active_columns_per_token": 11,
            "builder": "production_codebook_preimage",
            "canonical_json_sha256": (
                "388177db5f1d8cba9f0606d5e065148498418b10d27d7abf6117d09b44238a9c"
            ),
            "projection_seed": 17,
            "token_count": 768,
        },
    }


def test_python_oracle_reuses_the_exact_seeded_threshold_generator() -> None:
    expected = np.random.default_rng(17).normal(0.0, 0.025, 32)
    generated = seeded_threshold_offsets_mv(32, 17)
    cortex = DendriticCortex(cell_count=32, seed=17)

    assert generated.dtype == np.dtype(np.float64)
    assert np.array_equal(generated, expected)
    assert np.array_equal(cortex._threshold_offset, expected)


def test_factory_builds_the_exact_physical_native_definition() -> None:
    frozen = protocol.build_protocol()
    definition = definitions.build_r3_f0_definition(frozen, (18, 20), 17)

    assert type(definition) is NativeTemporalCircuitDefinition
    assert definition.config == NativeTemporalCircuitConfig(
        column_count=4096,
        max_pyramidal_cells=131072,
        segment_capacity=23286406,
        construction_seed=17,
        connected_permanence_ticks=250,
        initial_permanence_ticks=255,
        permanence_increment_ticks=50,
        negative_decrement_ticks=2,
        base_cells_per_column=8,
        novel_race_cap=8,
        prediction_quorum=11,
        max_segments_per_cell=255,
        theta_match=18,
        theta_active=20,
        context_policy=TemporalContextPolicy.BOUND_WINNERS,
        plasticity_policy=TemporalPlasticityPolicy.PLASTIC,
    )
    assert definition.token_codes == protocol.production_codebook_preimage()
    expected_thresholds = seeded_threshold_offsets_mv(274432, 17)
    observed_thresholds = np.asarray(definition.threshold_offsets_mv, dtype=np.float64)
    assert observed_thresholds.shape == (274432,)
    assert np.array_equal(observed_thresholds, expected_thresholds)


def test_frozen_arm_identity_table_matches_the_native_definition_query() -> None:
    frozen = protocol.build_protocol()
    table = frozen["screens"]["r3_f0_train"][
        "native_circuit_identity_sha256_by_arm_and_seed"
    ]

    for theta_match, theta_active in protocol.THRESHOLD_ARMS:
        arm = (theta_match, theta_active)
        slug = f"theta-m-{theta_match}_theta-a-{theta_active}"
        expected = table[slug]["construction-seed-0000000017"]
        assert definitions.build_r3_f0_definition(
            frozen, arm, 17
        ).circuit_identity_sha256() == expected


def test_q4_factory_is_bounded_and_matches_its_frozen_native_identity() -> None:
    frozen = protocol.build_protocol()
    screen = frozen["screens"]["compressed_reserve_q4"]
    identity_table = screen[
        "native_circuit_identity_sha256_by_arm_and_seed"
    ]

    for theta_match, theta_active in protocol.THRESHOLD_ARMS:
        arm = (theta_match, theta_active)
        arm_slug = f"theta-m-{theta_match}_theta-a-{theta_active}"
        for seed in protocol.DEVELOPMENT_SEEDS:
            definition = definitions.build_compressed_reserve_q4_definition(
                frozen,
                arm,
                seed,
            )
            seed_slug = f"construction-seed-{seed:010d}"
            assert definition.config.segment_capacity == 1_419
            assert definition.config.max_segments_per_cell == 4
            assert tuple(token for token, _columns in definition.token_codes) == (
                *protocol.COMPRESSED_RESERVE_CONTEXT_IDS,
                protocol.DISJOINT_TARGET_TOKEN,
            )
            assert (
                definition.circuit_identity_sha256()
                == identity_table[arm_slug][seed_slug]
            )

    with pytest.raises(protocol.ProtocolError, match="frozen seed scope"):
        definitions.build_compressed_reserve_q4_definition(
            frozen,
            protocol.THRESHOLD_ARMS[0],
            protocol.CONFIRMATION_SEEDS[0],
        )
    with pytest.raises(protocol.ProtocolError, match="frozen seed scope"):
        definitions.build_r3_f0_definition(
            frozen,
            protocol.THRESHOLD_ARMS[0],
            protocol.DEVELOPMENT_SEEDS[0],
        )


def test_confirmation_factories_accept_only_the_frozen_confirmation_seeds() -> None:
    frozen = protocol.build_protocol()
    arm = protocol.THRESHOLD_ARMS[0]

    for seed in protocol.CONFIRMATION_SEEDS:
        q4 = definitions.build_compressed_reserve_q4_confirmation_definition(
            frozen,
            arm,
            seed,
        )
        production = (
            definitions.build_full_cohort_q255_confirmation_production_definition(
                frozen,
                arm,
                seed,
            )
        )
        disjoint = (
            definitions.build_full_cohort_q255_confirmation_disjoint_definition(
                frozen,
                arm,
                seed,
            )
        )

        assert q4.config.construction_seed == seed
        assert q4.config.segment_capacity == 1_419
        assert production.config.construction_seed == seed
        assert production.config.segment_capacity == 22_451
        assert disjoint.config.construction_seed == seed
        assert disjoint.config.segment_capacity == 22_451
        assert production.circuit_identity_sha256() != disjoint.circuit_identity_sha256()

    for factory in (
        definitions.build_compressed_reserve_q4_confirmation_definition,
        definitions.build_full_cohort_q255_confirmation_production_definition,
        definitions.build_full_cohort_q255_confirmation_disjoint_definition,
    ):
        with pytest.raises(protocol.ProtocolError, match="frozen seed scope"):
            factory(frozen, arm, protocol.DEVELOPMENT_SEEDS[0])
