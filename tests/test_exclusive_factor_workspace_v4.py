from __future__ import annotations

from dataclasses import replace
from typing import get_type_hints

import numpy as np
import pytest
from experiments.r3_pilot.exclusive_factor_workspace_v4 import (
    AuthenticatedSpanBatch,
    ExclusiveFactorResolution,
    FrozenExclusiveFactorBridge,
    build_frozen_exclusive_factor_bridge,
)
from experiments.r3_pilot.exclusive_factor_workspace_v4_protocol import PROTOCOL


@pytest.fixture(scope="module")
def bridge() -> FrozenExclusiveFactorBridge:
    return build_frozen_exclusive_factor_bridge(PROTOCOL)


def test_build_executes_all_electrical_controls_before_freezing_uint8_state(
    bridge: FrozenExclusiveFactorBridge,
) -> None:
    assert isinstance(bridge, FrozenExclusiveFactorBridge)
    assert bridge.build_summary.electrical_control_count == 5
    assert bridge.build_summary.electrical_pass_count == 5
    assert bridge.build_summary.electrical_preflight_passed is True
    assert bridge.build_summary.electrical_circuit_off_decision_count == 0
    assert bridge.build_summary.electrical_tamper_rejected is True
    assert bridge.build_summary.electrical_preflight_sha256 == (
        "62bc36373323a169acd7e9397ef621d9d2aa921ed2213dea374d8f540ebbf696"
    )
    assert bridge.build_summary.projection_panel_sha256 == (
        "68c949b62e4ba9a13076dd17795bb4cc201f11f93521bf6ba176b4e110b55ab1"
    )
    assert bridge.build_summary.factor_registry_sha256 == (
        "e5c5019e020c691edeed05b4068190fcd2b2c9ed6a2b3384949231fa1510368f"
    )
    assert [row.name for row in bridge.electrical_replays] == [
        row.name for row in PROTOCOL.wta_electrical_controls
    ]
    assert [row.principal_first_spike_tier for row in bridge.electrical_replays] == [
        row.principal_first_spike_tier for row in PROTOCOL.wta_electrical_controls
    ]
    assert [row.decision for row in bridge.electrical_replays] == [
        "STOP",
        "STOP",
        "STOP",
        "STOP",
        "FACTOR",
    ]
    assert bridge.electrical_replays[-1].factor == "add"
    assert tuple(row.sha256 for row in bridge.electrical_replays) == (
        "cbdf17a92172041fc31c00f8ab5fc14e69cdaf8aefd9a1102a25e207c6571573",
        "b9666a022c85112d63be10a0b99038d0d35da921bfd4e6a3e4682b1d0f916b75",
        "b51ba9571d6aa31e7fdd714e3111c9831d462defb9246a814e91dfba5020bd83",
        "a7332c94b7da5b8f23413f40f6f3d502e3dff23cde63f76f6996f148640cae8e",
        "e9546f06f15176d0da062fa02c4363289b0f48ac9e0a2bf6343e2c98b4aaccca",
    )
    assert bridge.excitatory_ticks.shape == (24, 1536)
    assert bridge.inhibitory_ticks.shape == (24, 1536)
    assert bridge.excitatory_ticks.dtype == np.dtype("uint8")
    assert bridge.inhibitory_ticks.dtype == np.dtype("uint8")
    assert bridge.learned_state_bytes == 73_728
    assert bridge.checkpoint_sha256 == (
        "c1cd4557a1c72a0389b4a5953de38dd795b63e52359ee72102cbc24906ce8384"
    )


def test_frozen_schedule_stops_on_exact_atomic_readiness_evidence(
    bridge: FrozenExclusiveFactorBridge,
) -> None:
    summary = bridge.build_summary

    assert summary.epochs == 4
    assert summary.fit_lesson_count == 36
    assert summary.heldout_lesson_count == 18
    assert summary.atomic_lesson_count == 54
    assert summary.untrained_atomic_pass_count == 0
    assert summary.cell_off_atomic_pass_count == 0
    assert summary.atomic_pass_count == 44
    assert summary.atomic_mismatch_count == 10
    assert summary.fit_atomic_pass_count == 35
    assert summary.heldout_atomic_pass_count == 9
    assert summary.atomic_checkpoint_authentications == 2
    assert summary.composition_open is False
    assert summary.composition_attempt_count == 0
    assert summary.duplicate_control_attempt_count == 0
    assert summary.overlap_control_attempt_count == 0
    assert summary.capacity_control_attempt_count == 0
    assert summary.status == (
        "STOP_SELF_POSED_EXCLUSIVE_FACTOR_WORKSPACE_V4_ATOMIC_READINESS"
    )
    assert summary.excitatory_head_update_events == 72
    assert summary.inhibitory_head_update_events == 69
    assert summary.committed_cell_ticks == 14_857
    assert summary.excitatory_max_ticks == 4
    assert summary.inhibitory_max_ticks == 4
    assert summary.atomic_result_sha256 == (
        "9d5d37940716066cfde6e8057370aad7a30680f56180b55612a67af5788affc0"
    )
    assert len(bridge.atomic_replay) == 54
    assert all(row.decision == "FACTOR" for row in bridge.atomic_replay)
    mismatches = tuple(
        row
        for row in bridge.atomic_replay
        if row.factor != row.expected_factor
    )
    assert tuple((row.expected_factor, row.factor) for row in mismatches) == (
        ("maximum", "absolute"),
        ("add", "maximum"),
        ("maximum", "absolute"),
        ("less-than", "subtract"),
        ("sum", "less-than"),
        ("less-than", "subtract"),
        ("minimum", "count"),
        ("map", "minimum"),
        ("map", "minimum"),
        ("count", "map"),
    )
    assert all(row.simultaneous_spike_candidates == 1 for row in mismatches)
    assert all(row.unique_winner_cell is not None for row in mismatches)
    assert summary.public_input_count == PROTOCOL.public_input_count == 0
    assert summary.canonical_input_count == PROTOCOL.canonical_input_count == 0
    assert summary.protected_input_count == PROTOCOL.protected_input_count == 0


def test_terminal_resolve_has_frozen_signature_and_fails_before_batch_inspection(
    bridge: FrozenExclusiveFactorBridge,
) -> None:
    hints = get_type_hints(FrozenExclusiveFactorBridge.resolve)
    assert hints["batch"] is AuthenticatedSpanBatch
    assert hints["return"] is ExclusiveFactorResolution

    class ExplodingBatch:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"terminal resolve inspected {name}")

    with pytest.raises(RuntimeError, match="stopped before atomic readiness"):
        bridge.resolve(ExplodingBatch())  # type: ignore[arg-type]


def test_protocol_checkpoint_and_snapshot_authentication_are_fail_closed(
    bridge: FrozenExclusiveFactorBridge,
) -> None:
    with pytest.raises(RuntimeError, match="protocol authentication failed"):
        build_frozen_exclusive_factor_bridge(
            replace(PROTOCOL, protocol_sha256="0" * 64)
        )

    excitatory = bridge.excitatory_ticks
    inhibitory = bridge.inhibitory_ticks
    assert excitatory.flags.writeable is False
    assert inhibitory.flags.writeable is False
    with pytest.raises(ValueError):
        excitatory[0, 0] = np.uint8(4)
    excitatory.setflags(write=True)
    excitatory[0, 0] = np.uint8(4)
    assert bridge.excitatory_ticks[0, 0] != excitatory[0, 0]

    tampered = build_frozen_exclusive_factor_bridge(PROTOCOL)
    tampered._excitatory_ticks.setflags(write=True)
    tampered._excitatory_ticks[0, 0] ^= np.uint8(1)
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        _ = tampered.checkpoint_sha256
