"""Canonical native circuit definitions for the frozen calibration-v3 R3-F0 arm."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from experiments.r3_pilot import calibration_v3_protocol as protocol_authority
from humanbrain.cortex import seeded_threshold_offsets_mv
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuitConfig,
    NativeTemporalCircuitDefinition,
    TemporalContextPolicy,
    TemporalPlasticityPolicy,
)
from humanbrain.sdr_union import token_sdr_indices


def build_r3_f0_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Resolve one exact physical native definition from frozen R3-F0 authority."""

    protocol_authority.validate_protocol(protocol, require_runnable=False)
    contract = protocol["native_definition_contract"]
    config_authority = contract["config"]
    seed_scope = tuple(config_authority["construction_seed_scope"])
    arm_scope = tuple(tuple(value) for value in config_authority["theta_arm_scope"])
    if type(seed) is not int or seed not in seed_scope:
        raise protocol_authority.ProtocolError(
            "seed must be an exact integer in "
            "native_definition_contract.config.construction_seed_scope"
        )
    if (
        type(arm) is not tuple
        or len(arm) != 2
        or any(type(value) is not int for value in arm)
        or arm not in arm_scope
    ):
        raise protocol_authority.ProtocolError(
            "arm must be an exact integer pair in native_definition_contract.config.theta_arm_scope"
        )

    config = NativeTemporalCircuitConfig(
        column_count=config_authority["column_count"],
        max_pyramidal_cells=config_authority["max_pyramidal_cells"],
        segment_capacity=config_authority["segment_capacity"],
        construction_seed=seed,
        connected_permanence_ticks=config_authority["connected_permanence_ticks"],
        initial_permanence_ticks=config_authority["initial_permanence_ticks"],
        permanence_increment_ticks=config_authority["permanence_increment_ticks"],
        negative_decrement_ticks=config_authority["negative_decrement_ticks"],
        base_cells_per_column=config_authority["base_cells_per_column"],
        novel_race_cap=config_authority["novel_race_cap"],
        prediction_quorum=config_authority["prediction_quorum"],
        max_segments_per_cell=config_authority["max_segments_per_cell"],
        theta_match=arm[0],
        theta_active=arm[1],
        context_policy=TemporalContextPolicy[config_authority["context_policy"]],
        plasticity_policy=TemporalPlasticityPolicy[config_authority["plasticity_policy"]],
    )
    physical_cell_count = 2 * config.max_pyramidal_cells + 3 * config.column_count
    return NativeTemporalCircuitDefinition(
        config=config,
        token_codes=protocol_authority.production_codebook_preimage(),
        threshold_offsets_mv=seeded_threshold_offsets_mv(physical_cell_count, seed).tolist(),
    )


def build_compressed_reserve_q4_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Build the bounded native q=4 definition from frozen screen authority."""

    base = build_r3_f0_definition(protocol, arm, seed)
    screen = protocol["screens"]["compressed_reserve_q4"]
    config = replace(
        base.config,
        max_segments_per_cell=screen["max_segments_per_cell"],
        segment_capacity=screen["expected_target_segments"],
    )
    token_ids = (*screen["context_ids"], screen["target_token"])
    token_codes = tuple(
        (
            token_id,
            tuple(
                sorted(
                    map(
                        int,
                        token_sdr_indices(
                            token_id,
                            seed=protocol_authority.PROJECTION_SEED,
                            n_bits=config.column_count,
                            w_ones=11,
                        ),
                    )
                )
            ),
        )
        for token_id in token_ids
    )
    return NativeTemporalCircuitDefinition(
        config=config,
        token_codes=token_codes,
        threshold_offsets_mv=base.threshold_offsets_mv,
    )
