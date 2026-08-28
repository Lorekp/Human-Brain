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


def _build_definition(
    protocol: dict[str, Any],
    arm: tuple[int, int],
    seed: int,
    *,
    seed_scope: tuple[int, ...],
) -> NativeTemporalCircuitDefinition:
    protocol_authority.validate_protocol(protocol, require_runnable=False)
    contract = protocol["native_definition_contract"]
    config_authority = contract["config"]
    arm_scope = tuple(tuple(value) for value in config_authority["theta_arm_scope"])
    if type(seed) is not int or seed not in seed_scope:
        raise protocol_authority.ProtocolError(
            "seed must be an exact integer in the supplied frozen seed scope"
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


def build_r3_f0_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Resolve one exact physical native definition from frozen R3-F0 authority."""

    seed_scope = tuple(
        protocol["native_definition_contract"]["config"]["construction_seed_scope"]
    )
    return _build_definition(protocol, arm, seed, seed_scope=seed_scope)


def _phase_seed_scope(
    protocol: dict[str, Any], *, independent_confirmation: bool
) -> tuple[int, ...]:
    scope_name = (
        "electrical_q4_q255_independent_confirmation"
        if independent_confirmation
        else "electrical_q4_q255_development"
    )
    return tuple(protocol["threshold_selection"]["seed_scope"][scope_name])


def _build_q255_definition(
    protocol: dict[str, Any],
    arm: tuple[int, int],
    seed: int,
    *,
    independent_confirmation: bool,
) -> NativeTemporalCircuitDefinition:
    return _build_definition(
        protocol,
        arm,
        seed,
        seed_scope=_phase_seed_scope(
            protocol,
            independent_confirmation=independent_confirmation,
        ),
    )


def _build_compressed_reserve_q4_definition(
    protocol: dict[str, Any],
    arm: tuple[int, int],
    seed: int,
    *,
    independent_confirmation: bool,
) -> NativeTemporalCircuitDefinition:
    base = _build_definition(
        protocol,
        arm,
        seed,
        seed_scope=_phase_seed_scope(
            protocol,
            independent_confirmation=independent_confirmation,
        ),
    )
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


def build_compressed_reserve_q4_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Build the bounded native q=4 definition from frozen screen authority."""

    return _build_compressed_reserve_q4_definition(
        protocol,
        arm,
        seed,
        independent_confirmation=False,
    )


def build_compressed_reserve_q4_confirmation_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Build q=4 for one frozen independent-confirmation coordinate."""

    return _build_compressed_reserve_q4_definition(
        protocol,
        arm,
        seed,
        independent_confirmation=True,
    )


def build_full_cohort_q255_production_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Build the bounded q=255 production-pair circuit."""

    base = _build_q255_definition(
        protocol,
        arm,
        seed,
        independent_confirmation=False,
    )
    screen = protocol["screens"]["full_cohort_q255"]
    return NativeTemporalCircuitDefinition(
        config=replace(
            base.config,
            segment_capacity=screen["expected_target_segments"],
        ),
        token_codes=base.token_codes,
        threshold_offsets_mv=base.threshold_offsets_mv,
    )


def build_full_cohort_q255_confirmation_production_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Build q=255 production overlap for independent confirmation."""

    base = _build_q255_definition(
        protocol,
        arm,
        seed,
        independent_confirmation=True,
    )
    screen = protocol["screens"]["full_cohort_q255"]
    return NativeTemporalCircuitDefinition(
        config=replace(
            base.config,
            segment_capacity=screen["expected_target_segments"],
        ),
        token_codes=base.token_codes,
        threshold_offsets_mv=base.threshold_offsets_mv,
    )


def build_full_cohort_q255_disjoint_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Build the bounded q=255 target-disjoint control circuit."""

    base = _build_q255_definition(
        protocol,
        arm,
        seed,
        independent_confirmation=False,
    )
    screen = protocol["screens"]["full_cohort_q255"]
    learned, novel = protocol_authority.target_disjoint_context_ids()
    token_ids = (*learned, *novel, protocol_authority.DISJOINT_TARGET_TOKEN)
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
                            n_bits=base.config.column_count,
                            w_ones=11,
                        ),
                    )
                )
            ),
        )
        for token_id in token_ids
    )
    return NativeTemporalCircuitDefinition(
        config=replace(
            base.config,
            segment_capacity=screen["expected_target_segments"],
        ),
        token_codes=token_codes,
        threshold_offsets_mv=base.threshold_offsets_mv,
    )


def build_full_cohort_q255_confirmation_disjoint_definition(
    protocol: dict[str, Any], arm: tuple[int, int], seed: int
) -> NativeTemporalCircuitDefinition:
    """Build q=255 target-disjoint control for independent confirmation."""

    base = _build_q255_definition(
        protocol,
        arm,
        seed,
        independent_confirmation=True,
    )
    screen = protocol["screens"]["full_cohort_q255"]
    learned, novel = protocol_authority.target_disjoint_context_ids()
    token_ids = (*learned, *novel, protocol_authority.DISJOINT_TARGET_TOKEN)
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
                            n_bits=base.config.column_count,
                            w_ones=11,
                        ),
                    )
                )
            ),
        )
        for token_id in token_ids
    )
    return NativeTemporalCircuitDefinition(
        config=replace(
            base.config,
            segment_capacity=screen["expected_target_segments"],
        ),
        token_codes=token_codes,
        threshold_offsets_mv=base.threshold_offsets_mv,
    )
