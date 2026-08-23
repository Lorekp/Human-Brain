"""Event-proportional stepping parity tests (L3 spike).

Contract under test: the ``event_proportional`` stepping mode reproduces the
shipped full-touch semantics bit-exactly for every dynamics-bearing
observable -- spikes, delivered/generated events, capacity fail-stop timing,
and the state of every cell the stepper touches. Dormancy mechanics
(retirement of eligibility-tail cells and bit-exact lazy catch-up) are proven
at the kernel level in ``procedural.rs`` unit tests with exact-arithmetic
fixtures; these tests pin the public Python seam with the production GLIF
constants, whose slow sub-normal tails mean natural dormancy horizons are
far longer than CI scales (see experiments/procedural_adapter_spike).
"""

from __future__ import annotations

import math

import pytest

from humanbrain.procedural_runtime import (
    NativeProceduralRuntime,
    ProceduralInputEvent,
    ProceduralRuntimeConfig,
    ProceduralRuntimeError,
)
from humanbrain.procedural_topology import TopologyManifest

_SPIKE_DRIVE_PA = 150_000.0


def _config(
    cell_count: int,
    stepping_mode: str,
    *,
    resident_capacity: int | None = None,
) -> ProceduralRuntimeConfig:
    return ProceduralRuntimeConfig(
        manifest=TopologyManifest(
            cell_count=cell_count,
            fanout=4,
            topology="uniform_no_self",
            seed=71,
        ),
        resident_capacity=resident_capacity or cell_count,
        event_capacity=max(8 * cell_count, 512),
        spike_capacity=max(8 * cell_count, 512),
        interval_distinct_capacity=max(8 * cell_count, 512),
        recurrent_weight_pa=325.0,
        recurrent_delay_steps=1,
        stepping_mode=stepping_mode,
    )


def _drive(target: int, due_step: int, current_pa: float = _SPIKE_DRIVE_PA):
    return ProceduralInputEvent(
        due_step=due_step,
        target_gid=target,
        source_gid=target,
        edge_slot=0,
        current_pa=current_pa,
    )


def test_full_touch_is_the_default_and_never_reports_dormancy():
    config = _config(64, "full_touch")
    assert config.stepping_mode == "full_touch"
    runtime = NativeProceduralRuntime(config)
    try:
        for step in range(1, 60):
            batch = [_drive(3, step)] if step % 20 == 1 else []
            result = runtime.advance(1, batch)
            assert runtime.dormant_count == 0
            assert runtime.dormant_cells() == ()
            assert result.accounting.work.analytic_catchups == 0
        assert len(runtime.snapshot().residents) > 0
    finally:
        runtime.close()


def test_invalid_stepping_mode_is_rejected_before_ffi():
    with pytest.raises(ValueError):
        _config(64, "quantum")


def test_all_active_traffic_is_bit_exact_across_modes():
    """Keep-alive epsilon currents hold every cell in non-canonical state, so
    neither mode ever dormanizes and both must agree bit-for-bit."""

    def run(mode: str):
        cells = 48
        runtime = NativeProceduralRuntime(_config(cells, mode))
        try:
            trace = []
            last_result = None
            for step in range(120):
                batch = [_drive(gid, step, 1.0e-30) for gid in range(cells)]
                if step % 17 == 5:
                    batch.extend(
                        ProceduralInputEvent(
                            due_step=step,
                            target_gid=gid,
                            source_gid=(gid + 7) % cells,
                            edge_slot=1,
                            current_pa=_SPIKE_DRIVE_PA,
                        )
                        for gid in range(0, cells, 3)
                    )
                result = runtime.advance(1, batch)
                for spike in result.spikes:
                    trace.append((spike.step, spike.gid))
                last_result = result
            return (
                tuple(trace),
                last_result.accounting.work.cell_step_evaluations,
                last_result.accounting.events.delivered,
                last_result.accounting.edges.generated,
                runtime.snapshot().residents,
                runtime.dormant_count,
            )
        finally:
            runtime.close()

    ft_trace, ft_work, ft_delivered, ft_generated, ft_residents, ft_dormant = run(
        "full_touch"
    )
    ep_trace, ep_work, ep_delivered, ep_generated, ep_residents, ep_dormant = run(
        "event_proportional"
    )

    assert ft_trace, "fixture must produce spikes"
    assert ep_dormant == 0, "keep-alive traffic must prevent dormancy"
    assert ft_trace == ep_trace
    assert ft_work == ep_work, "all-active work must be identical"
    assert ft_delivered == ep_delivered
    assert ft_generated == ep_generated
    assert len(ft_residents) == len(ep_residents)
    for ft_record, ep_record in zip(ft_residents, ep_residents):
        assert ft_record.gid == ep_record.gid
        for field in (
            "voltage_relative_mv",
            "threshold_spike_mv",
            "threshold_voltage_mv",
            "synaptic_current_pa",
            "eligibility",
        ):
            assert getattr(ft_record, field).hex() == getattr(ep_record, field).hex(), (
                field
            )
        for left, right in zip(
            ft_record.after_spike_currents_pa, ep_record.after_spike_currents_pa
        ):
            assert left.hex() == right.hex()
        assert ft_record.refractory_steps == ep_record.refractory_steps


def test_capacity_fail_stop_timing_matches_across_modes():
    """Dormant cells occupy resident-capacity slots exactly as full-touch
    residents do, so both modes must fail-stop on the same call."""
    cells = 8

    def run(mode: str):
        config = _config(cells, mode, resident_capacity=4)
        runtime = NativeProceduralRuntime(config)
        try:
            for step in range(1, 24):
                try:
                    runtime.advance(1, [_drive(gid, step) for gid in range(6)])
                except ProceduralRuntimeError as error:
                    return step, error.status
            return None, None
        finally:
            runtime.close()

    ft_step, ft_status = run("full_touch")
    ep_step, ep_status = run("event_proportional")
    assert ft_status is not None, "overflow must fail-stop"
    assert (ft_step, ft_status) == (ep_step, ep_status)


def test_event_proportional_mode_tracks_work_and_stays_queryable():
    """The public seam exposes dormant accounting and preserves the
    ScaleAccounting schema in the new mode."""
    runtime = NativeProceduralRuntime(_config(32, "event_proportional"))
    try:
        result = runtime.advance(100, [_drive(3, 50)])
        assert result.accounting.work.cell_step_evaluations > 0
        assert result.accounting.cells.configured == 32
        assert isinstance(runtime.dormant_count, int)
        assert runtime.dormant_cells() == ()
        assert runtime.current_step == 100
    finally:
        runtime.close()


def test_identical_schedules_are_reproducible_within_each_mode():
    """Run-to-run determinism inside the new mode (fail-stop/determinism
    contract)."""

    def run(mode: str):
        runtime = NativeProceduralRuntime(_config(64, mode))
        try:
            trace = []
            for step in range(200):
                batch = [_drive(gid, step) for gid in (3, 17, 41)] if step % 50 == 7 else []
                result = runtime.advance(1, batch)
                for spike in result.spikes:
                    trace.append((spike.step, spike.gid))
            return tuple(trace)
        finally:
            runtime.close()

    assert run("event_proportional") == run("event_proportional")
    assert run("full_touch") == run("full_touch")
