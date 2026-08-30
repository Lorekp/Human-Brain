from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
from experiments.r3_pilot import teacher_gated_factor_workspace_v5 as v5
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as protocol_v5

EXPECTED_ROOTS = {
    "electrical_preflight_sha256": (
        "c93cafed8773069b039f921c283f8c50023d3fcba72d6ad39d004b603556211c"
    ),
    "projection_panel_sha256": ("8e6e184413db7035733267ce58a21894d0cc521f5f211ccb63551f72e09893c0"),
    "factor_registry_sha256": ("34ba29c17861af6822702fd73455a3923ec16fd722d2b579e459022b3ec4a939"),
    "checkpoint_sha256": ("96e29d03e5b2d9f699281d62fa159bb047add7f0024cfe44ce24c11413024c0e"),
    "atomic_result_sha256": ("9b75e031caf1c40729d223342a641eeb712a63457bc55ba3eef819e1239aeae8"),
    "composition_result_sha256": (
        "f6d96fa61256ed1a0658ed67f432d0cb7aaa9b2556e77ff6f8376a307795abc7"
    ),
    "compositor_control_result_sha256": (
        "c2290b393bad73c32b5247a3a8722a5930c05f93561191ccd2e3dba472735d29"
    ),
    "teacher_off": "f8a0b530c9103633a190afed4b7e003476e09f1554e330ad704d342af006897f",
    "cell_off": "7aeb39bfdd44920e8ccf0eae259656235433e41522d7a203c9233a0101cdaf6d",
    "random_label": "e1069da6774a5276b3e39439b88cbf59d48f511cf17ac363287c63f59b481ea4",
}


@pytest.fixture(scope="module")
def bridge() -> v5.FrozenTeacherGatedFactorBridge:
    return v5.build_frozen_teacher_gated_factor_bridge(protocol_v5.PROTOCOL)


def test_public_surface_and_import_are_data_only() -> None:
    public_functions = tuple(
        name
        for name, value in vars(v5).items()
        if not name.startswith("_")
        and inspect.isfunction(value)
        and value.__module__ == v5.__name__
    )
    public_bridge_methods = tuple(
        name
        for name, value in vars(v5.FrozenTeacherGatedFactorBridge).items()
        if not name.startswith("_") and inspect.isfunction(value)
    )
    assert public_functions == ("build_frozen_teacher_gated_factor_bridge",)
    assert public_bridge_methods == ("resolve",)
    assert not any(isinstance(value, np.ndarray) for value in vars(v5).values())
    assert "CompositionCaseReplay" in v5.__all__
    assert (
        protocol_v5.PROTOCOL.public_input_count,
        protocol_v5.PROTOCOL.canonical_input_count,
        protocol_v5.PROTOCOL.protected_input_count,
    ) == (0, 0, 0)


def test_preflight_failure_precedes_every_learner_array_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    treatment_allocations: list[tuple[tuple[int, int], np.dtype[np.uint8]]] = []
    original_zeros = v5.np.zeros

    def tracked_zeros(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        dtype = np.dtype(kwargs.get("dtype", np.float64))
        if shape == protocol_v5.FUTURE_STATE_SHAPE and dtype == np.dtype(np.uint8):
            treatment_allocations.append((shape, dtype))
        return original_zeros(shape, *args, **kwargs)

    def fail_preflight(_protocol: object) -> object:
        raise RuntimeError("injected electrical preflight STOP")

    monkeypatch.setattr(v5.np, "zeros", tracked_zeros)
    monkeypatch.setattr(v5, "_execute_electrical_preflight", fail_preflight)
    with pytest.raises(RuntimeError, match="injected electrical preflight STOP"):
        v5.build_frozen_teacher_gated_factor_bridge(protocol_v5.PROTOCOL)
    assert treatment_allocations == []


def test_one_fit_and_exact_terminal_batches_are_frozen(
    bridge: v5.FrozenTeacherGatedFactorBridge,
) -> None:
    summary = bridge.build_summary
    assert summary.status == protocol_v5.COMPOSITION_EXECUTION_RECORD["pass_status"]
    assert summary.electrical_preflight_passed
    assert (summary.electrical_pass_count, summary.electrical_control_count) == (52, 52)
    assert (
        summary.fit_pass_count,
        summary.excitatory_update_events,
        summary.inhibitory_update_events,
        summary.committed_excitatory_ticks,
        summary.committed_inhibitory_ticks,
        summary.excitatory_max_ticks,
        summary.inhibitory_max_ticks,
    ) == (1, 18, 0, 216, 0, 2, 0)
    assert (
        summary.atomic_pass_count,
        summary.atomic_mismatch_count,
        summary.composition_attempt_count,
        summary.composition_exact_count,
        summary.duplicate_control_attempt_count,
        summary.overlap_control_attempt_count,
        summary.capacity_control_attempt_count,
        summary.control_exact_count,
    ) == (54, 0, 18, 18, 12, 6, 3, 21)
    assert (
        summary.teacher_off_stop_count,
        summary.cell_off_stop_count,
        summary.random_label_permuted_exact_count,
        summary.random_label_true_exact_count,
    ) == (54, 54, 54, 0)
    for field in (
        "electrical_preflight_sha256",
        "projection_panel_sha256",
        "factor_registry_sha256",
        "checkpoint_sha256",
        "atomic_result_sha256",
        "composition_result_sha256",
        "compositor_control_result_sha256",
    ):
        assert getattr(summary, field) == EXPECTED_ROOTS[field]
    assert dict(bridge._causal_control_roots) == {
        name: EXPECTED_ROOTS[name] for name in ("teacher_off", "cell_off", "random_label")
    }


def test_frozen_state_is_exact_cap4_uint8_and_tamper_detected(
    bridge: v5.FrozenTeacherGatedFactorBridge,
) -> None:
    excitatory = bridge.excitatory_ticks
    inhibitory = bridge.inhibitory_ticks
    assert excitatory.shape == inhibitory.shape == (24, 512)
    assert excitatory.dtype == inhibitory.dtype == np.dtype(np.uint8)
    assert bridge.learned_state_bytes == protocol_v5.FUTURE_STATE_BYTES == 24_576
    assert not excitatory.flags.writeable and not inhibitory.flags.writeable
    assert int(excitatory.max()) == 2 and int(inhibitory.max()) == 0
    assert bridge.build_summary.silent_row_nonzero_bytes == 0

    external = bridge.excitatory_ticks
    external.setflags(write=True)
    external[0, 0] = 4
    assert bridge.checkpoint_sha256 == EXPECTED_ROOTS["checkpoint_sha256"]
    tampered = bridge.excitatory_ticks.copy()
    tampered[0, 0] ^= np.uint8(1)
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        v5._authenticate_state(
            checkpoint_sha256=bridge.checkpoint_sha256,
            metadata=bridge._metadata,
            excitatory_ticks=tampered,
            inhibitory_ticks=bridge.inhibitory_ticks,
        )


def test_fit_replays_are_teacher_clamped_branch_local_and_reset(
    bridge: v5.FrozenTeacherGatedFactorBridge,
) -> None:
    assert len(bridge.fit_replays) == protocol_v5.FIT_COUNT == 36
    assert tuple(row.lesson_sha256 for row in bridge.fit_replays) == tuple(
        protocol_v5.LOCAL_RULE_RECORD["fit_schedule"]["lesson_sha256s"]
    )
    assert all(len(row.selected_trace_cells) == 12 for row in bridge.fit_replays)
    assert all(len(row.teacher_spike_cells) == 1 for row in bridge.fit_replays)
    assert all(not row.inhibitory_update_spike_cells for row in bridge.fit_replays)
    assert sum(bool(row.excitatory_update_spike_cells) for row in bridge.fit_replays) == 18
    assert sum(row.committed_excitatory_ticks for row in bridge.fit_replays) == 216
    assert sum(row.committed_inhibitory_ticks for row in bridge.fit_replays) == 0


def test_local_commit_is_simultaneous_uint16_before_saturation() -> None:
    excitatory = np.zeros((24, 512), dtype=np.uint8)
    inhibitory = np.zeros((24, 512), dtype=np.uint8)
    selected = np.zeros(512, dtype=np.bool_)
    selected[:12] = True
    excitatory[0, :12] = 4
    inhibitory[1, :12] = 3
    e_updates = np.zeros(24, dtype=np.bool_)
    h_updates = np.zeros(24, dtype=np.bool_)
    e_updates[0] = True
    h_updates[1] = True

    delta_e, delta_h = v5._commit_local_plasticity(
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        selected_trace_spikes=selected,
        excitatory_update_spikes=e_updates,
        inhibitory_update_spikes=h_updates,
    )
    assert (delta_e, delta_h) == (0, 12)
    assert np.all(excitatory[0, :12] == 4)
    assert np.all(inhibitory[1, :12] == 4)


def test_selector_and_teacher_preflights_bind_physical_interventions() -> None:
    selector_rows = v5._execute_selector_preflight(protocol_v5.PROTOCOL)
    observed = {row["name"]: row["observed"] for row in selector_rows}
    direct = observed["incompatible-two-cue-collision-stop"]
    repeated = observed["same-address-double-cue-collision-stop"]
    assert direct["direct_collision_spike"] and not direct["stored_repeat_collision_spike"]
    assert repeated["stored_repeat_collision_spike"] and not repeated["direct_collision_spike"]
    assert max(repeated["address_read_counts"]) == 600
    assert observed["serial-residual-current-rejection"] == {
        "cue_only_spike_count": 0,
        "phase_only_spike_count": 0,
        "post_reset_active_count": 0,
    }
    assert observed["lexical_gate_off"]["selected_trace_cells"] == ()

    teacher = {row["name"]: row["observed"] for row in v5._execute_teacher_preflight()}
    assert teacher["score-12-target-stop-tie"]["excitatory_update_spikes"]
    assert teacher["score-24-correct-unique"]["target_match_spikes"]
    assert len(teacher["two-rival-scratch"]["inhibitory_update_spikes"]) == 2
    assert teacher["teacher-off"]["teacher_spikes"] == ()
    assert teacher["cell-off"]["selected_trace_read_spike_count"] == 0
    assert all(row["post_reset_active_count"] == 0 for row in teacher.values())


def test_composition_and_control_results_bind_every_physical_path(
    bridge: v5.FrozenTeacherGatedFactorBridge,
) -> None:
    composition = bridge.composition_result
    controls = bridge.compositor_control_result
    assert composition is not None and controls is not None
    assert v5._atomic_gate_passes(protocol_v5.PROTOCOL, bridge.atomic_result)
    assert v5._composition_gate_passes(protocol_v5.PROTOCOL, composition)
    assert v5._control_gate_passes(protocol_v5.PROTOCOL, controls)
    assert tuple(case.case_sha256 for case in composition.cases) == tuple(
        case.sha256 for case in protocol_v5.PROTOCOL.composition_cases
    )
    for observed, expected in zip(
        composition.cases,
        protocol_v5.PROTOCOL.composition_cases,
        strict=True,
    ):
        assert len(observed.selector_replays) == len(observed.span_decisions) == 3
        assert observed.expected_factors == expected.expected_factors
        assert len(observed.emitted_events) == 2
        assert all(
            len(decision.factor_relay_spike_cells) <= 1 for decision in observed.span_decisions
        )
        assert observed.compositor.status == "TWO_EVENTS"
        assert sum(observed.compositor.eos_status_spikes) == 1
        assert observed.compositor.post_reset_active_count == 0
        assert (
            observed.compositor.case_reset_relay_tick,
            observed.compositor.case_reset_arrival_tick,
            observed.compositor.next_case_ready_initialization_tick,
        ) == (3, 4, 5)
    assert len(controls.controls) == 21
    assert all(sum(row.eos_status_spikes) == 1 for row in controls.controls)
    assert all(row.post_reset_active_count == 0 for row in controls.controls)
    assert any(row.overlap_poison for row in controls.controls)
    assert any(row.capacity_poison for row in controls.controls)


def test_resolve_authenticates_all_gates_before_requested_batch(
    bridge: v5.FrozenTeacherGatedFactorBridge,
) -> None:
    case = protocol_v5.PROTOCOL.composition_cases[0]
    batch = v5.AuthenticatedSpanBatch(
        spans=case.spans,
        protocol_sha256=protocol_v5.PROTOCOL.protocol_sha256,
    )
    resolution = bridge.resolve(batch)
    assert resolution.status == "TWO_EVENTS"
    assert resolution.compositor.factors == case.expected_factors
    assert resolution.checkpoint_authentications == 2
    assert len(resolution.span_decisions) == 3

    wrong_protocol = v5.AuthenticatedSpanBatch(
        spans=case.spans,
        protocol_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="batch authentication failed"):
        bridge.resolve(wrong_protocol)


class _ExplodingBatch:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"requested batch was accessed through {name}")


def _reroot_atomic(
    bridge: v5.FrozenTeacherGatedFactorBridge,
    result: v5.AtomicBatchResult,
) -> v5.AtomicBatchResult:
    return replace(
        result,
        sha256=v5._namespaced_sha256(
            v5._ATOMIC_RESULT_NAMESPACE,
            v5._atomic_result_record(
                protocol_sha256=protocol_v5.PROTOCOL.protocol_sha256,
                checkpoint_sha256=bridge.checkpoint_sha256,
                electrical_preflight_sha256=bridge.electrical_preflight_sha256,
                projection_panel_sha256=bridge.build_summary.projection_panel_sha256,
                factor_registry_sha256=bridge.build_summary.factor_registry_sha256,
                result=result,
                telemetry=dict(bridge._metadata["update_telemetry"]),
                causal_control_roots=dict(bridge._causal_control_roots),
            ),
        ),
    )


def _reroot_composition(
    bridge: v5.FrozenTeacherGatedFactorBridge,
    result: v5.CompositionBatchResult,
) -> v5.CompositionBatchResult:
    return replace(
        result,
        sha256=v5._namespaced_sha256(
            v5._COMPOSITION_RESULT_NAMESPACE,
            v5._composition_result_record(
                protocol_sha256=protocol_v5.PROTOCOL.protocol_sha256,
                checkpoint_sha256=bridge.checkpoint_sha256,
                atomic_result_sha256=bridge.atomic_result.sha256,
                result=result,
            ),
        ),
    )


def _reroot_control(
    bridge: v5.FrozenTeacherGatedFactorBridge,
    result: v5.CompositorControlBatchResult,
) -> v5.CompositorControlBatchResult:
    assert bridge.composition_result is not None
    return replace(
        result,
        sha256=v5._namespaced_sha256(
            v5._CONTROL_RESULT_NAMESPACE,
            v5._control_result_record(
                protocol_sha256=protocol_v5.PROTOCOL.protocol_sha256,
                checkpoint_sha256=bridge.checkpoint_sha256,
                composition_result_sha256=bridge.composition_result.sha256,
                result=result,
            ),
        ),
    )


def _clone_bridge(
    bridge: v5.FrozenTeacherGatedFactorBridge,
    *,
    summary: v5.BridgeBuildSummary,
    atomic: v5.AtomicBatchResult,
    composition: v5.CompositionBatchResult | None,
    control: v5.CompositorControlBatchResult | None,
) -> v5.FrozenTeacherGatedFactorBridge:
    return v5.FrozenTeacherGatedFactorBridge(
        excitatory_ticks=bridge.excitatory_ticks,
        inhibitory_ticks=bridge.inhibitory_ticks,
        metadata=bridge._metadata,
        build_summary=summary,
        fit_replays=bridge.fit_replays,
        atomic_result=atomic,
        composition_result=composition,
        compositor_control_result=control,
        causal_control_roots=dict(bridge._causal_control_roots),
    )


def test_rerooted_composition_cannot_move_factor_relay_to_neutral_span(
    bridge: v5.FrozenTeacherGatedFactorBridge,
) -> None:
    assert bridge.composition_result is not None
    registered = protocol_v5.PROTOCOL.composition_cases[0]
    observed = bridge.composition_result.cases[0]
    left, neutral, right = observed.span_decisions
    neutral_span = registered.spans[1]
    right_span = registered.spans[2]
    moved_left_event = protocol_v5._event(
        registered.left_factor,
        neutral_span.start_token,
        neutral_span.stop_token,
    )
    right_event = protocol_v5._event(
        registered.right_factor,
        right_span.start_token,
        right_span.stop_token,
    )
    tampered_case = replace(
        observed,
        span_decisions=(
            replace(
                left,
                decision="STOP",
                factor=None,
                unique_winner_cell=None,
                factor_relay_spike_cells=(),
            ),
            replace(
                neutral,
                decision="FACTOR",
                factor=registered.left_factor,
                unique_winner_cell=v5._FACTOR_INDEX[registered.left_factor],
                factor_relay_spike_cells=(v5._FACTOR_INDEX[registered.left_factor],),
            ),
            right,
        ),
        emitted_events=(moved_left_event, right_event),
        compositor=replace(
            observed.compositor,
            input_event_sha256s=(
                moved_left_event.event_sha256,
                right_event.event_sha256,
            ),
        ),
    )
    tampered = _reroot_composition(
        bridge,
        replace(
            bridge.composition_result,
            cases=(tampered_case, *bridge.composition_result.cases[1:]),
            sha256="",
        ),
    )
    with pytest.raises(RuntimeError, match="protocol-bound span/event"):
        v5._composition_gate_passes(protocol_v5.PROTOCOL, tampered)


def test_rerooted_control_requires_complete_physical_replay(
    bridge: v5.FrozenTeacherGatedFactorBridge,
) -> None:
    assert bridge.compositor_control_result is not None
    first = bridge.compositor_control_result.controls[0]
    tampered_replay = replace(
        first,
        attempted_event_count=0,
        accepted_event_count=0,
        duplicate_count=0,
        eos_status_spikes=(False, False, False, False, False),
        post_reset_active_count=99,
        event_stage_spike_counts=(),
        event_stage_evaluated_counts=(),
        case_reset_relay_tick=-1,
        case_reset_arrival_tick=-1,
        next_case_ready_initialization_tick=-1,
    )
    tampered = _reroot_control(
        bridge,
        replace(
            bridge.compositor_control_result,
            controls=(tampered_replay, *bridge.compositor_control_result.controls[1:]),
            sha256="",
        ),
    )
    with pytest.raises(RuntimeError, match="physical publication"):
        v5._control_gate_passes(protocol_v5.PROTOCOL, tampered)


def test_all_three_authenticated_stop_gates_never_touch_requested_batch(
    bridge: v5.FrozenTeacherGatedFactorBridge,
) -> None:
    atomic_stop = _reroot_atomic(
        bridge,
        replace(
            bridge.atomic_result,
            status=str(protocol_v5.COMPOSITION_EXECUTION_RECORD["atomic_failure_status"]),
            pass_count=53,
            mismatch_count=1,
            sha256="",
        ),
    )
    atomic_bridge = _clone_bridge(
        bridge,
        summary=replace(
            bridge.build_summary,
            status=atomic_stop.status,
            atomic_pass_count=53,
            atomic_mismatch_count=1,
            composition_attempt_count=0,
            composition_exact_count=0,
            duplicate_control_attempt_count=0,
            overlap_control_attempt_count=0,
            capacity_control_attempt_count=0,
            control_exact_count=0,
            atomic_result_sha256=atomic_stop.sha256,
            composition_result_sha256=None,
            compositor_control_result_sha256=None,
        ),
        atomic=atomic_stop,
        composition=None,
        control=None,
    )
    with pytest.raises(RuntimeError, match=atomic_stop.status):
        atomic_bridge.resolve(_ExplodingBatch())

    assert bridge.composition_result is not None
    composition_stop = _reroot_composition(
        bridge,
        replace(
            bridge.composition_result,
            status=str(protocol_v5.COMPOSITION_EXECUTION_RECORD["composition_failure_status"]),
            exact_count=17,
            mismatch_count=1,
            sha256="",
        ),
    )
    composition_bridge = _clone_bridge(
        bridge,
        summary=replace(
            bridge.build_summary,
            status=composition_stop.status,
            composition_exact_count=17,
            duplicate_control_attempt_count=0,
            overlap_control_attempt_count=0,
            capacity_control_attempt_count=0,
            control_exact_count=0,
            composition_result_sha256=composition_stop.sha256,
            compositor_control_result_sha256=None,
        ),
        atomic=bridge.atomic_result,
        composition=composition_stop,
        control=None,
    )
    with pytest.raises(RuntimeError, match=composition_stop.status):
        composition_bridge.resolve(_ExplodingBatch())

    assert bridge.compositor_control_result is not None
    control_stop = _reroot_control(
        bridge,
        replace(
            bridge.compositor_control_result,
            status=str(protocol_v5.COMPOSITION_EXECUTION_RECORD["control_failure_status"]),
            exact_count=20,
            mismatch_count=1,
            sha256="",
        ),
    )
    control_bridge = _clone_bridge(
        bridge,
        summary=replace(
            bridge.build_summary,
            status=control_stop.status,
            control_exact_count=20,
            compositor_control_result_sha256=control_stop.sha256,
        ),
        atomic=bridge.atomic_result,
        composition=bridge.composition_result,
        control=control_stop,
    )
    with pytest.raises(RuntimeError, match=control_stop.status):
        control_bridge.resolve(_ExplodingBatch())
