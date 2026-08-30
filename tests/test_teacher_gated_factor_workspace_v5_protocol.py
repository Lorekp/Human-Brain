from __future__ import annotations

import dataclasses
import inspect
from collections import Counter
from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pytest
from experiments.r3_pilot import exclusive_factor_workspace_v4_protocol as v4
from experiments.r3_pilot import prompt_factor_multispan_event_bridge_v3 as v3
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as v5
from humanbrain.resting_glif_race import CompiledRestingGlifRace

EXPECTED_ROOTS = {
    "atomic_input_sha256": "8b3b9144322956f2cade03c15f6bb9c45ce50bf6c6cb71a6492ceabeac2ed042",
    "curriculum_sha256": "607f92f1cc8f8f81bf26d8dc8d86e6ee2dcfefbc56c0085141d92c76ffb5bbf8",
    "split_sha256": "cd9df23c85e52bb8c01e46c121179cb9f4bb49a06d3ded1717812ae878b143dc",
    "spent_input_reference_sha256": (
        "887b22e596ca6a1c272044f6975230f2fa375e756bf6eff830793b9bc9fb282d"
    ),
    "projection_sha256": "fd7cac83f632faad0ebc2ff51cf52c7d8f046c5c8ceb33ea7ebbfea094403ad1",
    "direct_readout_channel_sha256": (
        "dbd0ca3f5aeb7103c897dc5ffa279ef3fe602ab66ca133eeb9c3317ae188601b"
    ),
    "persistent_latch_primitive_sha256": (
        "bdec7d4c02c89da8399659684751c5d94ee27388503478fad987f887f085dcaf"
    ),
    "selector_topology_sha256": "c3191703b5e7c7a1f4d58c967527fad7ec0a30c87cb34abb150925a37b5ddbcd",
    "selector_electrical_controls_sha256": (
        "6f678323241aa480704a02cbabdb9b8dbd3e150c186237a08f533d8f0c7baa0f"
    ),
    "selector_circuit_sha256": "2ba642dd4d850ec425c41804273ff61528e869677fa491d1f696a2355ad88ff3",
    "selector_full_preflight_sha256": (
        "b4d6cceae873edcd6e886ddb7f41a01e7ed659f6929029dcf6f77828c33ae073"
    ),
    "v4_wta_reference_sha256": "fdd313d438e26a64769e7e8889260673d7703d9d870006cf62b326af3d741404",
    "local_rule_sha256": "1267ce675991535826d7157d6bc4706fb4904682d2f599af0b6c1774708bbfc8",
    "control_panel_sha256": "04d6106bf300a86a503d0985f102b8b41b0e2e6df72af4f8d2fe53f9ff3cf401",
    "atomic_gate_sha256": "bc53029efff7fe50a86e695f7daf683a8ba0c6bd1db7605e4e3546c8bc7a46a1",
    "teacher_scout_circuit_sha256": (
        "94711addfea3c24279105db21cabe2a0c8b77191f85103fd9843150506492fb6"
    ),
    "electrical_preflight_contract_sha256": (
        "971f9f6bd2a887fc0dedaf05b110d2c4f0d74b533d7314f10a5a0575b6de5296"
    ),
    "neutral_sha256": "669842a2a8f8afa711db8d65e88c4b07914ca334316e72331c005ed6e147870c",
    "composition_sha256": "d8a403a8d12fa7be1e71dcb2152782138b7b46ee69472d74350532038016fe43",
    "duplicate_controls_sha256": "677e0823e98e30251140e7ffa636fbd14046861707a5abbc4903b868be02f94a",
    "overlap_controls_sha256": "4e586a091ee187b2c5de46e958b3aeb26d50f407bc311a0bf92f1b5679b9b244",
    "capacity_controls_sha256": "196c96f74d4b004cee8a62d73f0598b73e9b2efdad08ef9966c6412aeb441e5c",
    "compositor_ingress_buffer_sha256": (
        "d8a787a63a4dece82a07c72645bd24cf24ff713cd56263649706bf759bef90b3"
    ),
    "compositor_admission_sha256": (
        "bae23fae4aedff113c30ae69de484e4ea6d61b42601f1b3965659699926d6f67"
    ),
    "compositor_eos_status_sha256": (
        "fbda21ef7328ec54bf4030e80a9ce8ff25da0c1f535c9e09a302ccfcd63c9c6b"
    ),
    "compositor_circuit_sha256": (
        "56636b8acb5a386999bcc26471adde467f809329f459f9f7b26775677f745dc8"
    ),
    "composition_execution_sha256": (
        "50dbae9c1d85cd6598a26493205e274b4f434499534d5eac8a6886f330a1545a"
    ),
    "future_authentication_sha256": (
        "958c173df96b4df257df93f8ffbcc2aeddcc950bc11e87750848f901675ea986"
    ),
    "protocol_sha256": "587bd68a293ca2116570cf4e5ee7a860bbdd9928138982ae43748c4f87a252c1",
}


def _walk(value: object):
    yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)


def test_no_leak_storage_readout_primitive_persists_reads_and_resets() -> None:
    population = v5._NoLeakStorageReadoutPopulation(2)
    zeros = np.zeros(2, dtype=np.float64)

    assert not np.any(
        population.step(
            set_currents_pa=np.asarray([5_000.0, 0.0]),
            read_gate_currents_pa=zeros,
            direct_readout_currents_pa=zeros,
            reset_currents_pa=zeros,
        )
    )
    assert population.storage_charge_pa == (5_000.0, 0.0)

    for _ in range(1_024):
        assert not np.any(
            population.step(
                set_currents_pa=zeros,
                read_gate_currents_pa=zeros,
                direct_readout_currents_pa=zeros,
                reset_currents_pa=zeros,
            )
        )
    assert population.storage_charge_pa == (5_000.0, 0.0)

    for _ in range(2):
        assert tuple(
            population.step(
                set_currents_pa=zeros,
                read_gate_currents_pa=np.asarray([5_000.0, 5_000.0]),
                direct_readout_currents_pa=zeros,
                reset_currents_pa=zeros,
            )
        ) == (True, False)
        assert population.storage_charge_pa == (5_000.0, 0.0)

    for _ in range(v4.RACE_STEPS):
        assert tuple(
            population.step(
                set_currents_pa=zeros,
                read_gate_currents_pa=np.asarray([5_000.0, 5_000.0]),
                direct_readout_currents_pa=zeros,
                reset_currents_pa=zeros,
            )
        ) == (True, False)
    assert population.storage_charge_pa == (5_000.0, 0.0)

    storage_before_invalid = population.storage_charge_pa
    invalid_current_rows = (
        (np.asarray([2_500.0, 0.0]), zeros, zeros),
        (zeros, np.asarray([10_000.0, 0.0]), zeros),
        (zeros, zeros, np.asarray([15_000.0, 0.0])),
    )
    for set_currents, read_currents, direct_currents in invalid_current_rows:
        with pytest.raises(ValueError, match="persistent latch current domain changed"):
            population.step(
                set_currents_pa=set_currents,
                read_gate_currents_pa=read_currents,
                direct_readout_currents_pa=direct_currents,
                reset_currents_pa=zeros,
            )
        assert population.storage_charge_pa == storage_before_invalid

    direct = v5._NoLeakStorageReadoutPopulation(2)
    assert tuple(
        direct.step(
            set_currents_pa=zeros,
            read_gate_currents_pa=zeros,
            direct_readout_currents_pa=np.asarray([10_000.0, 0.0]),
            reset_currents_pa=zeros,
        )
    ) == (True, False)
    assert direct.storage_charge_pa == (0.0, 0.0)

    assert not np.any(
        population.step(
            set_currents_pa=np.asarray([5_000.0, 5_000.0]),
            read_gate_currents_pa=np.asarray([5_000.0, 5_000.0]),
            direct_readout_currents_pa=np.asarray([10_000.0, 10_000.0]),
            reset_currents_pa=np.asarray([-5_000.0, -5_000.0]),
        )
    )
    assert population.storage_charge_pa == (0.0, 0.0)
    assert v5.PERSISTENT_LATCH_PRIMITIVE_RECORD["operation_precedence"] == (
        "reset inhibition and zero clamp dominate the addressed storage/readout",
        "read the pre-set storage snapshot through a separate ephemeral compartment",
        "add only an explicit 0-or-10000 pA direct multi-principal readout input",
        "clear every readout membrane after observation",
        "apply set current to storage and clamp at the finite charge cap",
        "retain storage charge exactly across an idle tick",
    )
    assert v5.PERSISTENT_LATCH_PRIMITIVE_RECORD["mandatory_preflight_controls"] == {
        "idle_persistence": {
            "set_tick": 0,
            "idle_tick_count": 1024,
            "read_tick": 1025,
            "expected_storage_pa_hex": (5_000.0).hex(),
            "expected_readout_spikes": 1,
        },
        "nondestructive_repeated_read": {
            "read_ticks": (0, 1),
            "expected_readout_spikes": (1, 1),
            "expected_final_storage_pa_hex": (5_000.0).hex(),
        },
        "reset_dominance": {
            "simultaneous_set_read_reset": True,
            "expected_readout_spikes": 0,
            "expected_final_storage_pa_hex": (0.0).hex(),
        },
        "current_domain_rejection": {
            "invalid_set_current_pa_hex": (2_500.0).hex(),
            "invalid_read_gate_current_pa_hex": (10_000.0).hex(),
            "invalid_direct_readout_current_pa_hex": (15_000.0).hex(),
            "expected_mutation_before_error": False,
        },
        "direct_two_principal_collision": {
            "direct_readout_current_pa_hex": (10_000.0).hex(),
            "expected_storage_pa_hex": (0.0).hex(),
            "expected_readout_spikes": 1,
        },
    }
    assert v5.PERSISTENT_LATCH_PRIMITIVE_RECORD["set_current_domain_pa_hex"] == (
        (0.0).hex(),
        (5_000.0).hex(),
    )
    assert v5.PERSISTENT_LATCH_PRIMITIVE_RECORD["read_gate_current_domain_pa_hex"] == (
        (0.0).hex(),
        (5_000.0).hex(),
    )
    assert v5.PERSISTENT_LATCH_PRIMITIVE_RECORD["direct_readout_current_domain_pa_hex"] == (
        (0.0).hex(),
        (10_000.0).hex(),
    )
    assert v5.PERSISTENT_LATCH_PRIMITIVE_RECORD["direct_readout_channel_sha256"] == (
        v5.DIRECT_READOUT_CHANNEL_SHA256
    )
    assert {
        "purpose": "two-principal collision coincidence without aliasing a storage read gate",
        "direct_synaptic_input_current_pa_hex": (10_000.0).hex(),
        "input_current_domain_pa_hex": ((0.0).hex(), (10_000.0).hex()),
        "readout_threshold_pa_hex": (10_000.0).hex(),
        "principal_input_count": 2,
        "current_per_principal_pa_hex": (5_000.0).hex(),
        "storage_coupling": False,
        "read_gate_alias": False,
        "invalid_or_overdrive_rejected_before_state_mutation": True,
    } == v5.DIRECT_READOUT_CHANNEL_RECORD


def test_same_address_repeat_physically_precharges_later_collision() -> None:
    zeros = np.zeros(1, dtype=np.float64)
    full = np.asarray([5_000.0])
    counter = v5._NoLeakStorageReadoutPopulation(1)
    collision = v5._NoLeakStorageReadoutPopulation(1)

    first_repeat_spike = counter.step(
        set_currents_pa=full,
        read_gate_currents_pa=full,
        direct_readout_currents_pa=zeros,
        reset_currents_pa=zeros,
    )
    assert tuple(first_repeat_spike) == (False,)
    second_repeat_spike = counter.step(
        set_currents_pa=full,
        read_gate_currents_pa=full,
        direct_readout_currents_pa=zeros,
        reset_currents_pa=zeros,
    )
    assert tuple(second_repeat_spike) == (True,)
    collision.step(
        set_currents_pa=np.where(second_repeat_spike, full, zeros),
        read_gate_currents_pa=zeros,
        direct_readout_currents_pa=zeros,
        reset_currents_pa=zeros,
    )

    for _ in range(559):
        assert not np.any(
            collision.step(
                set_currents_pa=zeros,
                read_gate_currents_pa=zeros,
                direct_readout_currents_pa=zeros,
                reset_currents_pa=zeros,
            )
        )
    assert collision.storage_charge_pa == (5_000.0,)
    assert tuple(
        collision.step(
            set_currents_pa=zeros,
            read_gate_currents_pa=full,
            direct_readout_currents_pa=zeros,
            reset_currents_pa=zeros,
        )
    ) == (True,)

    circuit = v5.SELECTOR_CIRCUIT_RECORD
    assert circuit["event_counter_operation_order"] == (
        "read pre-event seen-address storage",
        "emit repeat spike only when an earlier address event was stored",
        "set/saturate seen-address storage for the current event",
        "copy a repeat spike into collision storage",
    )
    assert circuit["same_address_collision_rule"] == (
        "the second same-address event sets 5000 pA no-leak collision storage; "
        "one later leading principal supplies the 5000 pA read gate at the frozen "
        "collision tick, reaching the 10000 pA readout threshold"
    )


def test_selector_slot_replay_fixtures_are_physical_and_label_independent() -> None:
    assembly_by_token = {assembly.token: assembly for assembly in v5.PROTOCOL.lexical_assemblies}
    rows = v5.SELECTOR_FULL_PREFLIGHT_RECORD["controls"][:8]
    for row in rows:
        slot_tokens = row["slot_tokens"]
        assert len(slot_tokens) == 4
        assert row["selected_token"] == slot_tokens[row["selected_slot"]]
        assert row["selected_token"] not in v5.TAUGHT_FACTORS
        assert tuple(event["slot"] for event in row["content_replay_events"]) == (
            0,
            1,
            2,
            3,
        )
        for event in row["content_replay_events"]:
            assembly = assembly_by_token[event["token"]]
            assert event["token"] == slot_tokens[event["slot"]]
            assert event["assembly_sha256"] == assembly.sha256
            assert event["assembly_cells"] == assembly.cells
            assert event["relay_spike_cells"] == (
                assembly.cells if event["slot"] == row["selected_slot"] else ()
            )

    base = rows[0]
    selected_slot = base["selected_slot"]
    swap_slot = next(
        slot for slot in range(4) if slot not in (selected_slot, v5.SELECTOR_ADDRESSES[0].cue_slot)
    )
    permuted = list(base["slot_tokens"])
    permuted[selected_slot], permuted[swap_slot] = (
        permuted[swap_slot],
        permuted[selected_slot],
    )
    trace = v5._selector_content_trace(
        slot_tokens=tuple(permuted),
        selected_slot=selected_slot,
        decision_tick=base["decision_tick"],
        assembly_by_token=assembly_by_token,
    )
    assert trace["selected_token"] == permuted[selected_slot]
    assert trace["selected_token"] != base["selected_token"]
    assert trace["selected_lexical_relay_cells"] == assembly_by_token[permuted[selected_slot]].cells


def test_atomic_curriculum_and_lexical_projection_are_fresh_and_balanced() -> None:
    protocol = v5.PROTOCOL
    lessons = protocol.lessons
    fit_ids = set(protocol.fit_lesson_sha256s)
    heldout_ids = set(protocol.heldout_lesson_sha256s)

    assert v5.TRIADS == (
        ("add", "less-than", "count"),
        ("absolute", "sum", "minimum"),
        ("maximum", "subtract", "map"),
    )
    assert len(protocol.atomic_groups) == v5.GROUP_COUNT == 18
    assert len(lessons) == v5.LESSON_COUNT == 54
    assert len(fit_ids) == v5.FIT_COUNT == 36
    assert len(heldout_ids) == v5.HELDOUT_COUNT == 18
    assert fit_ids.isdisjoint(heldout_ids)

    for group in protocol.atomic_groups:
        assert len(group.lessons) == 3
        assert len({tuple(sorted(lesson.tokens)) for lesson in group.lessons}) == 1
        assert {lesson.factor for lesson in group.lessons} == set(v5.TRIADS[group.triad_index])
        for lesson in group.lessons:
            assert lesson.tokens[lesson.cue_slot] == lesson.cue_token
            assert lesson.tokens[lesson.selected_slot] == lesson.factor
            assert lesson.tokens.count(lesson.cue_token) == 1
            assert set(lesson.tokens) == set(v5.TRIADS[group.triad_index]) | {lesson.cue_token}

    selected_fit = Counter(
        (lesson.factor, lesson.selected_slot) for lesson in lessons if lesson.sha256 in fit_ids
    )
    unselected_fit = Counter(
        (factor, slot)
        for lesson in lessons
        if lesson.sha256 in fit_ids
        for slot, factor in enumerate(lesson.tokens)
        if factor in v5.TAUGHT_FACTORS and slot != lesson.selected_slot
    )
    assert selected_fit == Counter(
        {(factor, slot): 1 for factor in v5.TAUGHT_FACTORS for slot in range(4)}
    )
    assert unselected_fit == Counter(
        {(factor, slot): 2 for factor in v5.TAUGHT_FACTORS for slot in range(4)}
    )

    fit_components = {
        "cue_tokens": {lesson.cue_token for lesson in lessons if lesson.sha256 in fit_ids},
        "cue_slots": {lesson.cue_slot for lesson in lessons if lesson.sha256 in fit_ids},
        "selected_slots": {lesson.selected_slot for lesson in lessons if lesson.sha256 in fit_ids},
        "factors": {lesson.factor for lesson in lessons if lesson.sha256 in fit_ids},
    }
    for lesson in lessons:
        if lesson.sha256 in heldout_ids:
            assert lesson.cue_token in fit_components["cue_tokens"]
            assert lesson.cue_slot in fit_components["cue_slots"]
            assert lesson.selected_slot in fit_components["selected_slots"]
            assert lesson.factor in fit_components["factors"]

    v3_groups, _ = v3.self_posed_local_span_curriculum()
    spent_inputs = {
        lesson.teacher_span.canonical_tokens for group in v3_groups for lesson in group.lessons
    } | {lesson.tokens for lesson in v4.PROTOCOL.lessons}
    fit_inputs = {lesson.tokens for lesson in lessons if lesson.sha256 in fit_ids}
    heldout_inputs = {lesson.tokens for lesson in lessons if lesson.sha256 in heldout_ids}
    assert fit_inputs.isdisjoint(heldout_inputs)
    assert (fit_inputs | heldout_inputs).isdisjoint(spent_inputs)

    assemblies = protocol.lexical_assemblies
    assert len({assembly.token for assembly in assemblies}) == len(assemblies)
    assert all(len(assembly.cells) == 12 for assembly in assemblies)
    assert all(
        tuple(sorted(set(assembly.cells))) == assembly.cells
        and all(0 <= cell < 512 for cell in assembly.cells)
        for assembly in assemblies
    )
    assert (
        max(len(set(left.cells) & set(right.cells)) for left, right in combinations(assemblies, 2))
        <= 3
    )


def test_selector_topology_and_all_address_controls_are_physically_frozen() -> None:
    topology = v5.SELECTOR_TOPOLOGY_RECORD
    controls = v5.PROTOCOL.selector_electrical_controls

    assert topology["direct_readout_channel_sha256"] == v5.DIRECT_READOUT_CHANNEL_SHA256
    assert topology["phase_cells"] == 4
    assert topology["coincidence_address_cells"] == 8
    assert topology["selected_slot_latch_cells"] == 4
    assert topology["lexical_replay_relay_cells"] == 512
    assert topology["transition_bank_cells"] == 0
    assert topology["principal_address_cells"] == 8
    assert topology["principal_stop_cells"] == 1
    assert topology["basket_inhibitory_cells"] == 1
    assert topology["collision_cells"] == 1
    assert topology["delayed_address_relays"] == 8
    assert topology["address_candidate_count"] == 9
    assert topology["stop_candidate_index"] == 8
    assert topology["host_selected_token_materialization"] is False
    assert topology["zero_address_events"] == "STOP"
    assert topology["multiple_address_events"] == "COLLISION_STOP"
    assert topology["full_selected_content_circuit_deferred"] is True
    assert topology["full_selector_circuit_deferred"] is True
    assert topology["basket_delay_ticks"] == 1
    assert topology["address_relay_delay_ticks"] == 3
    assert topology["collision_stop_delay_ticks"] == 2

    assert len(v5.SELECTOR_ADDRESSES) == 8
    assert {(row.direction, row.cue_slot, row.selected_slot) for row in v5.SELECTOR_ADDRESSES} == {
        (direction, cue_slot, (cue_slot + (1 if direction == "NEXT" else -1)) % 4)
        for direction in ("NEXT", "PREV")
        for cue_slot in range(4)
    }
    assert len(controls) == v5.SELECTOR_ELECTRICAL_CONTROL_COUNT == 10
    assert tuple(control.name for control in controls[-2:]) == (
        "no-cue-stop",
        "incompatible-two-cue-collision-stop",
    )
    assert tuple(control.decision for control in controls[:8]) == ("SELECTED_SLOT",) * 8
    assert tuple(control.selected_slot for control in controls[:8]) == tuple(
        row.selected_slot for row in v5.SELECTOR_ADDRESSES
    )
    assert controls[-2].decision == "STOP"
    assert controls[-2].principal_first_spike_tier == (8,)
    assert controls[-1].decision == "COLLISION_STOP"
    assert controls[-1].principal_first_spike_tier == (0, 4)
    assert all(control.full_circuit_execution_deferred for control in controls)

    current_rows = np.asarray(
        [control.principal_currents_pa for control in controls],
        dtype=np.float64,
    )
    assert current_rows.shape == (10, 9)
    races = CompiledRestingGlifRace.compile(step_count=v4.RACE_STEPS).race_many(
        current_rows,
        require_unique=False,
    )
    for control, race in zip(controls, races, strict=True):
        assert race.spike_step == control.principal_first_spike_step
        assert race.simultaneous_spike_candidates == control.simultaneous_spike_candidates


def test_full_selector_circuit_and_prelearner_preflight_are_exactly_frozen() -> None:
    circuit = v5.SELECTOR_CIRCUIT_RECORD
    preflight = v5.SELECTOR_FULL_PREFLIGHT_RECORD
    electrical = v5.ELECTRICAL_PREFLIGHT_CONTRACT_RECORD

    assert circuit["direct_readout_channel_sha256"] == v5.DIRECT_READOUT_CHANNEL_SHA256
    assert circuit["persistent_latch_primitive_sha256"] == (v5.PERSISTENT_LATCH_PRIMITIVE_SHA256)
    assert circuit["cell_counts"] == {
        "phase": 4,
        "address_coincidence": 8,
        "address_current_latch": 8,
        "address_current_storage_compartment": 8,
        "address_current_readout_compartment": 8,
        "address_event_counter": 1,
        "address_event_counter_storage_compartment": 1,
        "address_event_counter_readout_compartment": 1,
        "address_principal": 8,
        "stop_principal": 1,
        "basket": 1,
        "collision": 1,
        "collision_storage_compartment": 1,
        "collision_readout_compartment": 1,
        "address_relay": 8,
        "stop_relay": 1,
        "selected_slot_latch": 4,
        "selected_slot_storage_compartment": 4,
        "selected_slot_readout_compartment": 4,
        "lexical_replay_relay": 512,
        "reset": 1,
        "transition_bank": 0,
    }
    assert circuit["current_pa_hex"] == {
        "full": (5_000.0).hex(),
        "reset_inhibition": (-5_000.0).hex(),
        "address_coincidence_threshold": (65_000.0).hex(),
        "address_event_counter_threshold": (10_000.0).hex(),
        "persistent_storage_set": (5_000.0).hex(),
        "persistent_read_gate": (5_000.0).hex(),
        "persistent_readout_threshold": (10_000.0).hex(),
        "direct_two_principal_collision_readout": (10_000.0).hex(),
        "selected_slot_latch_threshold": (10_000.0).hex(),
        "lexical_replay_threshold": (10_000.0).hex(),
        "inactive_address_principal": v4.PROTOCOL.latency_table[0].current_pa.hex(),
        "active_address_principal": v4.PROTOCOL.latency_table[13].current_pa.hex(),
        "tonic_stop_principal": v4.PROTOCOL.latency_table[12].current_pa.hex(),
    }
    assert circuit["within_tick_precedence"] == (
        "reset inhibition and scheduled external assembly spikes",
        "phase plus cue-assembly address coincidence for each cue event",
        "address event-count/current latches then per-event address hard reset",
        "principal, basket, collision, and delayed-relay threshold evaluation",
        "selected-slot latch and lexical replay relay threshold evaluation",
    )
    assert circuit["scan_ticks"] == (1, 2, 3, 4)
    assert circuit["address_scan_close_tick"] == 5
    assert circuit["principal_start_tick"] == 6
    assert circuit["unique_address_principal_spike_tick"] == 561
    assert circuit["unique_address_relay_tick"] == 564
    assert circuit["address_current_principal_read_window"] == (
        6,
        605,
        "inclusive",
    )
    assert circuit["address_current_principal_read_count"] == 600
    assert circuit["content_phase_offsets"] == (1, 3, 5, 7)
    assert circuit["content_gate_offsets"] == (2, 4, 6, 8)
    assert circuit["content_close_reset_offset"] == 9
    assert circuit["selected_latches_reset_between_scan_and_relay"] is True
    assert circuit["phase_to_latch_axons_enabled_during_scan"] is False
    assert circuit["phase_to_latch_axons_enabled_during_replay"] is True
    assert circuit["host_selected_token_materialization"] is False
    assert circuit["full_circuit_execution_deferred"] is True

    assert preflight["control_count"] == 11
    assert preflight["required_pass_count"] == 11
    assert preflight["execution_deferred"] is True
    rows = preflight["controls"]
    assert tuple(row["name"] for row in rows[:10]) == tuple(
        control.name for control in v5.PROTOCOL.selector_electrical_controls
    )
    assert rows[-1]["name"] == "same-address-double-cue-collision-stop"
    assert all(row["phase_spikes"] == ((1, 0), (2, 1), (3, 2), (4, 3)) for row in rows)
    assert all(row["post_reset_active_cell_count"] == 0 for row in rows)
    for row in rows[:8]:
        assert len(row["selected_lexical_relay_cells"]) == 12
        assert row["latch_set_tick"] == row["decision_tick"]
        assert row["selected_lexical_relay_tick"] == (
            row["decision_tick"] + 2 + 2 * row["selected_slot"]
        )
        assert row["replay_close_reset_tick"] == row["decision_tick"] + 9
    assert rows[8]["decision"] == "STOP"
    assert rows[8]["selected_lexical_relay_cells"] == ()
    assert rows[9]["decision"] == "COLLISION_STOP"
    assert rows[9]["collision_tick"] == rows[9]["principal_first_spike_tick"] + 1
    assert rows[9]["selected_lexical_relay_cells"] == ()
    assert rows[10]["address_event_count"] == 2
    assert rows[10]["principal_first_spike_tier"] == (0,)
    assert rows[10]["decision"] == "COLLISION_STOP"
    assert rows[10]["selected_lexical_relay_cells"] == ()
    repeated_cues = rows[10]["cue_assembly_events"]
    assert len(repeated_cues) == 2
    assert repeated_cues[0]["token"] == repeated_cues[1]["token"] == "cue-next-v5"
    assert repeated_cues[0]["assembly_sha256"] == repeated_cues[1]["assembly_sha256"]
    assert repeated_cues[0]["assembly_cells"] == repeated_cues[1]["assembly_cells"]
    assert len(repeated_cues[0]["assembly_cells"]) == 12
    assert tuple(
        event["repeat_readout_spike"] for event in rows[10]["event_counter_read_then_set_events"]
    ) == (False, True)
    assert rows[10]["collision_storage_set_tick"] == 1
    assert rows[10]["collision_read_gate_tick"] == rows[10]["collision_tick"]
    assert preflight["causal_controls"] == {
        "circuit_off": "zero decision, latch, and lexical-relay spikes",
        "phase_leg_off": "cue assemblies spike but no address cell may spike",
        "cue_leg_off": "all phase cells spike but no address cell may spike",
        "latch_persistent_charge_off": (
            "selected latch may receive its set input but cannot retain the "
            "subthreshold charge until replay"
        ),
        "lexical_gate_off": "selector decision survives but lexical-relay spike count is zero",
    }

    assert electrical["direct_readout_channel_sha256"] == v5.DIRECT_READOUT_CHANNEL_SHA256
    assert electrical["must_pass_before_state_allocation"] is True
    assert electrical["execution_order"] == (
        "authenticate complete V5 protocol",
        "execute persistent-latch idle, repeated-read, reset, exact-domain, and "
        "direct-collision controls",
        "execute all 11 full selector controls and causal interventions",
        "execute all 5 complete passed-V4 factor WTA controls",
        "execute teacher/scout scratch, teacher-off, and cell-off controls",
        "authenticate the complete observed electrical trace root",
        "allocate uint8 E/H learner arrays",
    )
    assert electrical["persistent_latch_controls_already_executed"] == 0
    assert electrical["selector_principal_controls_already_executed"] == 10
    assert electrical["factor_principal_controls_already_executed"] == 5
    assert electrical["full_circuit_execution_deferred"] is True


def test_v4_factor_stage_local_teacher_rule_and_falsifier_controls_are_frozen() -> None:
    protocol = v5.PROTOCOL
    reference = v5.V4_WTA_REFERENCE_RECORD
    rule = v5.LOCAL_RULE_RECORD
    controls = v5.CONTROL_PANEL_RECORD
    gate = v5.ATOMIC_GATE_RECORD

    assert reference["latency_table_sha256"] == v4.PROTOCOL.latency_table_sha256
    assert reference["wta_topology_sha256"] == v4.PROTOCOL.wta_topology_sha256
    assert reference["wta_electrical_controls_sha256"] == (
        v4.PROTOCOL.wta_electrical_controls_sha256
    )
    assert reference["factor_cells"] == 24
    assert reference["stop_candidate_index"] == 24
    assert reference["raw_score_domain"] == (-48, 48)
    assert reference["raw_to_current_score_rule"] == "current_score=max(raw_score,0)"
    assert reference["stop_score"] == 12
    assert reference["reused_v4_learner_state"] is False
    assert reference["full_factor_circuit_deferred"] is True
    assert protocol.factor_wta_electrical_controls == (v4.PROTOCOL.wta_electrical_controls)
    current_rows = np.asarray(
        [control.principal_currents_pa for control in protocol.factor_wta_electrical_controls],
        dtype=np.float64,
    )
    races = CompiledRestingGlifRace.compile(step_count=v4.RACE_STEPS).race_many(
        current_rows,
        require_unique=False,
    )
    assert tuple(race.spike_step for race in races) == tuple(
        control.principal_first_spike_step for control in protocol.factor_wta_electrical_controls
    )
    assert tuple(race.simultaneous_spike_candidates for race in races) == tuple(
        control.simultaneous_spike_candidates for control in protocol.factor_wta_electrical_controls
    )

    assert v5.FUTURE_STATE_RECORD["tick_dtype"] == "uint8"
    assert v5.FUTURE_STATE_RECORD["tick_cap"] == 4
    assert v5.FUTURE_STATE_RECORD["state_shape"] == (24, 512)
    assert v5.FUTURE_STATE_RECORD["state_arrays"] == (
        "excitatory_ticks",
        "inhibitory_ticks",
    )
    assert v5.FUTURE_STATE_RECORD["total_state_bytes"] == 24_576
    assert rule["teacher_population_cells"] == 24
    assert rule["scout_population_cells"] == 24
    assert rule["scout_basket_suppressed"] is False
    assert rule["teacher_after_autonomous_prediction"] is True
    assert rule["selected_trace_cell_count"] == 12
    assert "every non-target scout" in rule["non_target_scout"]
    assert rule["correct_unique_target"] == "suppress the complete atomic E/H commit"
    assert rule["fit_pass_count"] == 1
    assert rule["epochs"] is None
    assert rule["parameter_sweep"] is False
    schedule = rule["fit_schedule"]
    assert len(schedule["lesson_sha256s"]) == 36
    assert set(schedule["lesson_sha256s"]) == set(protocol.fit_lesson_sha256s)
    assert schedule["shuffle"] is False
    assert schedule["repeat"] is False
    assert schedule["heldout_scheduled"] is False

    assert controls["two_rival_scratch"]["expected_target_e_ticks"] == 12
    assert controls["two_rival_scratch"]["expected_rival_h_ticks"] == 24
    assert controls["two_rival_scratch"]["expected_inhibitory_update_events"] == 2
    assert controls["nonselected_distractor_scratch"]["expected_rival_events"] == 0
    assert controls["teacher_off"]["expected_checkpoint_nonzero_bytes"] == 0
    assert controls["teacher_off"]["expected_stop_count"] == 54
    assert controls["cell_off"]["expected_factor_event_count"] == 0
    assert controls["cell_off"]["expected_stop_count"] == 54
    permutation = dict(controls["random_label_permutation"]["mapping"])
    assert set(permutation) == set(v5.TAUGHT_FACTORS)
    assert set(permutation.values()) == set(v5.TAUGHT_FACTORS)
    assert all(source != target for source, target in permutation.items())
    assert controls["random_label_permutation"]["expected_permuted_exact"] == 54
    assert controls["random_label_permutation"]["expected_true_exact"] == 0
    assert controls["silent_rows"]["factor_count"] == 15
    assert set(controls["silent_rows"]["factors"]) == set(v5.FACTOR_REGISTRY) - set(
        v5.TAUGHT_FACTORS
    )
    assert controls["treatment_inference"]["current_matrix_shape"] == (54, 25)
    assert controls["treatment_inference"]["race_many_call_count"] == 1
    assert controls["treatment_inference"]["checkpoint_authentications"] == 2

    assert gate["status"] == "PREREGISTERED_EXPECTATION_FALSIFIER_NOT_RUN"
    assert gate["required_atomic_unique_exact"] == 54
    assert gate["expected_treatment_telemetry"] == {
        "excitatory_update_events": 18,
        "inhibitory_update_events": 0,
        "committed_excitatory_ticks": 216,
        "excitatory_max_ticks": 2,
        "inhibitory_max_ticks": 0,
    }
    assert gate["composition_attempts_on_any_failure"] == 0


def test_teacher_scout_anatomy_timing_and_atomic_commit_are_exactly_frozen() -> None:
    circuit = v5.TEACHER_SCOUT_CIRCUIT_RECORD
    schedule = circuit["fit_lesson_schedule"]

    assert circuit["persistent_latch_primitive_sha256"] == (v5.PERSISTENT_LATCH_PRIMITIVE_SHA256)
    assert circuit["cell_counts"] == {
        "selected_trace_latch": 512,
        "selected_trace_storage_compartment": 512,
        "selected_trace_readout_compartment": 512,
        "factor_principal": 24,
        "stop_reference": 1,
        "autonomous_factor_result_latch": 24,
        "autonomous_result_storage_compartment": 24,
        "autonomous_result_readout_compartment": 24,
        "scout": 24,
        "scout_eligibility_latch": 24,
        "scout_eligibility_storage_compartment": 24,
        "scout_eligibility_readout_compartment": 24,
        "teacher": 24,
        "target_match": 24,
        "target_e_update_relay": 24,
        "rival_h_update_relay": 24,
        "reset": 1,
    }
    assert circuit["factor_principal_start_tick"] == 574
    assert circuit["scout_stop_close_tick"] == 1130
    assert circuit["autonomous_relay_capture_close_tick"] == 1133
    assert circuit["teacher_spike_tick"] == 1134
    assert circuit["autonomous_and_scout_read_tick"] == 1135
    assert circuit["target_match_tick"] == 1136
    assert circuit["update_relay_tick"] == 1137
    assert circuit["selected_trace_read_and_atomic_commit_tick"] == 1138
    assert circuit["atomic_delta_commit_tick"] == 1138
    assert circuit["lesson_reset_tick"] == 1139
    assert circuit["scout_window"] == (574, 1130, "inclusive")
    assert circuit["selected_trace_valid_through_tick"] == 1138
    assert circuit["latch_read_is_nondestructive"] is True
    assert circuit["teacher_off_disables_plastic_commit"] is True
    assert circuit["scout_basket_suppressed"] is False
    assert circuit["winner_cell_on_nonunique_tier"] == "ignored"
    assert circuit["target_e_rule"] == (
        "the target-E relay receives 5000 pA from its delayed teacher spike "
        "and -5000 pA from a matching target-match spike; if it fires, its "
        "delayed output increments the 12 selected-trace uint8 E ticks once, "
        "saturating at 4"
    )
    assert circuit["rival_h_rule"] == (
        "each rival-H relay receives 2500 pA from the delayed teacher-open "
        "broadcast plus 2500 pA from its scout readout, -5000 pA from the "
        "same-index target teacher, and -5000 pA from target match; every relay "
        "that fires increments its 12 selected-trace uint8 H ticks once, "
        "saturating at 4"
    )
    assert circuit["correct_unique_no_update_condition"] == (
        "matching autonomous unique target latch suppresses the complete atomic "
        "E/H commit, irrespective of later nonleading scout spikes"
    )
    assert circuit["atomic_commit_precedence"] == (
        "freeze autonomous factor/STOP relay and scout traces",
        "spike one-hot teacher target",
        "read autonomous-result and scout storage nondestructively",
        "evaluate target match before update relays",
        "evaluate target-E and rival-H relays from the frozen snapshot",
        "read selected-trace storage and commit all E/H deltas simultaneously",
        "reset every lesson-local population and eligibility latch",
    )
    assert len(schedule) == 36
    assert tuple(row["lesson_sha256"] for row in schedule) == tuple(
        v5.LOCAL_RULE_RECORD["fit_schedule"]["lesson_sha256s"]
    )
    lesson_by_sha = {lesson.sha256: lesson for lesson in v5.PROTOCOL.lessons}
    for row in schedule:
        lesson = lesson_by_sha[row["lesson_sha256"]]
        assert row["factor"] == lesson.factor
        assert row["selected_slot"] == lesson.selected_slot
        assert row["selected_trace_set_tick"] == 566 + 2 * lesson.selected_slot
        assert row["selected_trace_valid_through_tick"] == 1138
        assert row["teacher_spike_tick"] == 1134
        assert row["autonomous_and_scout_read_tick"] == 1135
        assert row["target_match_tick"] == 1136
        assert row["update_relay_tick"] == 1137
        assert row["atomic_delta_commit_tick"] == 1138
        assert row["lesson_reset_tick"] == 1139
    assert circuit["full_circuit_execution_deferred"] is True


def test_composition_and_compositor_controls_are_preregistered_but_unexecuted() -> None:
    protocol = v5.PROTOCOL
    heldout_ids = set(protocol.heldout_lesson_sha256s)
    neutral_ids = {span.sha256 for span in protocol.neutral_spans}

    assert len(protocol.neutral_spans) == v5.NEUTRAL_SPAN_COUNT == 6
    assert len(neutral_ids) == 6
    assert len(protocol.composition_cases) == v5.COMPOSITION_CASE_COUNT == 18
    for triad_index, triad in enumerate(v5.TRIADS):
        cases = [case for case in protocol.composition_cases if case.triad_index == triad_index]
        assert {(case.left_factor, case.right_factor) for case in cases} == {
            (left, right) for left in triad for right in triad if left != right
        }
        for case in cases:
            assert case.expected_factors == (case.left_factor, case.right_factor)
            assert tuple((span.start_token, span.stop_token, span.kind) for span in case.spans) == (
                (0, 4, "factor"),
                (4, 8, "neutral"),
                (8, 12, "factor"),
            )
            assert case.spans[0].source_sha256 in heldout_ids
            assert case.spans[1].source_sha256 in neutral_ids
            assert case.spans[2].source_sha256 in heldout_ids

    assert len(protocol.duplicate_controls) == v5.DUPLICATE_CONTROL_COUNT == 12
    assert len(protocol.overlap_controls) == v5.OVERLAP_CONTROL_COUNT == 6
    assert len(protocol.capacity_controls) == v5.CAPACITY_CONTROL_COUNT == 3
    for control in protocol.duplicate_controls[:9]:
        assert control.kind == "exact-duplicate"
        assert control.events[0] == control.events[1]
        assert control.expected_status == "ONE_EVENT"
        assert control.expected_eos_status_latch is None
        assert control.expected_factors == (control.events[0].factor,)
    for control in protocol.duplicate_controls[9:]:
        assert control.kind == "exact-duplicate-after-full"
        assert len(control.events) == 3
        assert control.events[0] == control.events[2]
        assert control.events[0] != control.events[1]
        assert control.events[0].stop_token <= control.events[1].start_token
        assert control.expected_status == "TWO_EVENTS"
        assert control.expected_eos_status_latch is None
        assert control.expected_factors == tuple(event.factor for event in control.events[:2])
    for control in protocol.overlap_controls[:3]:
        left, right = control.events
        assert left.factor != right.factor
        assert max(left.start_token, right.start_token) < min(left.stop_token, right.stop_token)
        assert control.expected_status == "STOP_OVERLAPPING_FACTOR_EVENTS"
        assert control.expected_eos_status_latch == "overlap"
        assert control.expected_factors == ()
    for control in protocol.overlap_controls[3:]:
        assert control.kind == "distinct-overlap-after-full"
        assert len(control.events) == 3
        first, second, ingress = control.events
        assert first.stop_token <= second.start_token
        assert any(
            max(stored.start_token, ingress.start_token)
            < min(stored.stop_token, ingress.stop_token)
            for stored in (first, second)
        )
        assert control.expected_status == "STOP_OVERLAPPING_FACTOR_EVENTS"
        assert control.expected_eos_status_latch == "overlap"
        assert control.expected_factors == ()
    for control in protocol.capacity_controls:
        assert len(control.events) == 3 > v5.EVENT_CAPACITY
        assert control.expected_status == "STOP_EVENT_CAPACITY_2"
        assert control.expected_eos_status_latch == "capacity"
        assert control.expected_factors == ()


def test_compositor_execution_gates_and_future_authentication_are_exactly_frozen() -> None:
    circuit = v5.COMPOSITOR_CIRCUIT_RECORD
    execution = v5.COMPOSITION_EXECUTION_RECORD
    authentication = v5.FUTURE_AUTHENTICATION_RECORD

    assert circuit["persistent_latch_primitive_sha256"] == (v5.PERSISTENT_LATCH_PRIMITIVE_SHA256)
    assert circuit["compositor_ingress_buffer_sha256"] == (v5.COMPOSITOR_INGRESS_BUFFER_SHA256)
    assert circuit["compositor_admission_sha256"] == v5.COMPOSITOR_ADMISSION_SHA256
    assert circuit["compositor_eos_status_sha256"] == v5.COMPOSITOR_EOS_STATUS_SHA256
    assert circuit["event_capacity"] == 2
    assert circuit["factor_cells_per_slot"] == 24
    assert circuit["event_slot_count"] == 2
    assert circuit["token_position_cells"] == 12
    assert circuit["boundary_cells"] == 13
    assert circuit["duplicate_key_cells"] == 2 * 24 * 13 * 13
    assert circuit["output_factor_relays"] == 2 * 24
    assert circuit["case_persistent_compartments"] == {
        "slot_factor_storage": 2 * 24,
        "slot_factor_readout": 2 * 24,
        "slot_start_storage": 2 * 13,
        "slot_start_readout": 2 * 13,
        "slot_stop_storage": 2 * 13,
        "slot_stop_readout": 2 * 13,
        "slot_footprint_storage": 2 * 12,
        "slot_footprint_readout": 2 * 12,
        "slot_occupied_storage": 2,
        "slot_occupied_readout": 2,
        "overlap_poison_storage": 1,
        "overlap_poison_readout": 1,
        "capacity_poison_storage": 1,
        "capacity_poison_readout": 1,
        "case_poison_halt_storage": 1,
        "case_poison_halt_readout": 1,
        "event_ready_storage": 1,
        "event_ready_readout": 1,
    }
    assert circuit["event_ephemeral_no_leak_compartments"] == {
        "ingress_identity_storage": 24 + 13 + 13 + 12,
        "ingress_identity_readout": 24 + 13 + 13 + 12,
        "pre_event_occupancy_snapshot_storage": 2,
        "pre_event_occupancy_snapshot_readout": 2,
        "admission_block_storage": 1,
        "admission_block_readout": 1,
    }
    assert circuit["transient_population_counts"] == {
        "ingress_gate_cells": 24 + 13 + 13 + 12,
        "duplicate_key_coincidence_cells": 2 * 24 * 13 * 13,
        "overlap_coincidence_cells": 2 * 12,
        "duplicate_or_relay": 1,
        "overlap_or_relay": 1,
        "duplicate_decision_relay": 1,
        "distinct_overlap_decision_relay": 1,
        "capacity_decision_relay": 1,
        "poison_slot_reset_relay": 1,
        "slot_zero_admission_cell": 1,
        "slot_one_admission_cell": 1,
        "admitted_slot_relays": 2,
        "slot_copy_coincidence_cells": 2 * (24 + 13 + 13 + 12 + 1),
        "ephemeral_reset_relay": 1,
        "eos_status_candidate_cells": 5,
        "eos_status_output_relays": 5,
        "case_reset_relay": 1,
        "ingress_or_eos_phase_request_cells": 2,
    }
    assert circuit["eos_status_relays"] == {
        "zero": 1,
        "one": 1,
        "two": 1,
        "overlap": 1,
        "capacity": 1,
    }
    assert circuit["event_microticks"] == {
        "ingress_attempt_and_case_poison_halt_read": -1,
        "ingress_gate_and_slot_nondestructive_read": 0,
        "ingress_capture_equality_intersection_and_occupancy_snapshot_set": 1,
        "duplicate_and_overlap_or_relays": 2,
        "duplicate_and_distinct_overlap_decisions": 3,
        "capacity_decision": 4,
        "poison_block_set_and_poison_slot_reset_relay": 5,
        "poison_slot_reset_arrival_and_slot_zero_admission": 6,
        "slot_one_admission": 7,
        "admitted_slot_relays_and_ingress_copy_read": 8,
        "copy_coincidence_and_ephemeral_reset_relay": 9,
        "slot_storage_set_ephemeral_reset_and_ready_rearm": 10,
    }
    assert circuit["duplicate_precedes_overlap_and_capacity"] is True
    assert circuit["same_microtick_spike_influence"] is False
    assert circuit["state_timing"] == {
        "stored_slot_read_tick": 0,
        "stored_slot_read_is_nondestructive": True,
        "case_poison_halt_read_tick": -1,
        "event_ready_read_tick": -1,
        "ingress_gate_tick": 0,
        "ingress_capture_tick": 1,
        "event_ready_clear_tick": 1,
        "occupancy_snapshot_set_tick": 1,
        "poison_storage_set_ticks": {"overlap": 5, "capacity": 5},
        "case_poison_halt_set_tick": 5,
        "admission_block_set_tick": 5,
        "poison_slot_reset_relay_tick": 5,
        "poison_slot_reset_tick": 6,
        "admission_block_read_tick": 6,
        "ingress_copy_read_tick": 8,
        "copy_coincidence_tick": 9,
        "ephemeral_reset_relay_tick": 9,
        "slot_storage_set_tick": 10,
        "ephemeral_reset_tick": 10,
        "event_ready_rearm_tick": 10,
        "next_ingress_or_eos_request_tick": 11,
        "selected_phase_gate_tick": 12,
        "event_gate_stride_ticks": 12,
    }
    assert circuit["causal_delivery_edges"] == (
        "-1 ingress attempts plus ready/halt readouts -> +0 ingress gate cells",
        "+0 permitted ingress and slot readout spikes -> +1 capture/equality/intersection",
        "+0 accepted factor gate spike -> +1 event-ready storage clear",
        "+0 occupied-slot readout spikes -> +1 occupancy snapshot storage",
        "+1 equality/intersection spikes -> +2 duplicate/overlap OR relays",
        "+2 OR spikes -> +3 duplicate/distinct-overlap decision relays",
        "+3 decisions and +3 occupancy snapshot readouts -> +4 capacity cell",
        "+3 duplicate/overlap or +4 capacity spikes -> +5 status/block storage",
        "+3/+4 poison decisions -> +5 poison-slot reset relay",
        "+5 poison-slot reset spike -> +6 accepted-slot storage reset",
        "+3/+4 classification spikes -> +6 slot-zero admission inhibition",
        "+5 occupancy snapshot readout -> +6 slot-zero admission cell",
        "+6 occupancy and admission-block readouts -> +7 slot-one admission cell",
        "+6/+7 admission spikes -> +8 selected-slot relays",
        "+8 selected-slot and ingress readout spikes -> +9 copy-coincidence cells",
        "+8 completion phase -> +9 ephemeral-reset relay",
        "+9 copy spikes -> +10 admitted slot storage",
        "+9 ephemeral-reset spike -> +10 ephemeral reset and event-ready rearm",
        "+10 ready state -> +11 one-hot ingress-or-EOS request -> +12 selected gate",
    )
    assert circuit["eos_microticks"] == {
        "nondestructive_slot_factor_occupied_and_poison_read": 0,
        "ordinary_count_and_poison_status_candidates": 1,
        "one_hot_status_and_poison_gated_factor_outputs": 2,
        "case_reset_relay": 3,
        "complete_case_reset": 4,
        "next_case_event_ready_initialization": 5,
    }
    assert circuit["case_phase_serialization"] == {
        "post_event_completion_tick": 10,
        "ready_gated_one_hot_request_tick": 11,
        "selected_next_ingress_or_eos_gate_tick": 12,
        "ingress_and_eos_requests_mutually_exclusive": True,
        "request_domain": "exactly one of next-ingress or EOS",
        "invalid_request_rule": (
            "zero or two request spikes fail closed before either selected gate and "
            "before any persistent or ephemeral state mutation"
        ),
        "request_gate_rule": (
            "the exact one-hot request and event-ready readout emitted at +11 drive "
            "only the corresponding +12 ingress or EOS gate; without ready both remain silent"
        ),
        "zero_event_path": (
            "case setup initializes ready; with no ingress accepted, one EOS request "
            "selects the EOS read gate over empty occupied/poison storage"
        ),
        "early_eos_rule": (
            "before ready rearm, EOS request gates remain silent and cannot inspect "
            "ingress, snapshot, block, poison-reset, admission, or copy transients"
        ),
    }
    assert circuit["exact_duplicate_rule"] == (
        "same factor, start, and stop key emits the +3 duplicate decision, sets only "
        "the +5 admission block, preserves accepted slots, and creates no copy spike"
    )
    assert circuit["duplicate_precedence_current_rule"] == (
        "the +2 duplicate OR supplies +5000 pA to the +3 duplicate decision and "
        "-5000 pA to the +3 overlap decision; the +3 duplicate decision supplies "
        "-5000 pA to +4 capacity and, through rooted delays, blocks both admissions"
    )
    assert circuit["overlap_rule"] == (
        "at +3 the overlap OR drive survives only when duplicate inhibition is absent; "
        "that decision sets only overlap poison plus admission block at +5 and resets "
        "all accepted slots at +6"
    )
    assert circuit["capacity_rule"] == (
        "a third nonduplicate, nonoverlapping event sets only capacity poison plus "
        "admission block at +5 and resets every accepted slot at +6"
    )
    assert circuit["poison_status_mutual_exclusion_rule"] == (
        "duplicate inhibition prevents the distinct-overlap decision and both prior "
        "decisions inhibit capacity; therefore overlap and capacity set axons are "
        "physically one-hot before either persistent status storage is addressed"
    )
    assert circuit["eos_status_rule"] == (
        "the rooted five-way EOS circuit reads occupied and poison storage at +0, "
        "forms mutually exclusive ordinary-count/poison candidates at +1, emits "
        "exactly one status relay at +2, and permits factor outputs only for ONE/TWO"
    )
    assert circuit["admission_current_rule"] == (
        "+6 slot zero receives +5000 pA base drive minus its 5000 pA occupancy and "
        "classification inhibition; +7 slot one receives +5000 pA only from slot-zero "
        "pre-event occupancy, minus slot-one occupancy and the +6 block readout"
    )
    assert circuit["host_ingress_replay_or_occupancy_state"] is False
    assert circuit["ordering_rule"] == (
        "factor relays arrive in physical left-to-right span scan order; accepted "
        "slot order is output order"
    )
    assert circuit["host_factor_selection_union_filter_sort"] is False
    assert circuit["host_equality_overlap_capacity_or_sort"] is False
    assert circuit["full_circuit_execution_deferred"] is True

    ingress_record = v5.COMPOSITOR_INGRESS_BUFFER_RECORD
    assert ingress_record["identity_widths"] == {
        "factor": 24,
        "start": 13,
        "stop": 13,
        "footprint": 12,
    }
    assert ingress_record["ingress_identity_storage_compartments"] == 62
    assert ingress_record["pre_event_occupancy_snapshot_storage_compartments"] == 2
    assert ingress_record["external_ingress_attempt_tick"] == -1
    assert ingress_record["case_poison_halt_read_tick"] == -1
    assert ingress_record["ingress_gate_tick"] == 0
    assert ingress_record["capture_tick"] == 1
    assert ingress_record["ingress_copy_read_tick"] == 8
    assert ingress_record["copy_coincidence_tick"] == 9
    assert ingress_record["admitted_slot_storage_set_tick"] == 10
    assert ingress_record["ephemeral_reset_tick"] == 10
    assert ingress_record["event_ready_rearm_tick"] == 10
    assert ingress_record["next_ingress_or_eos_request_tick"] == 11
    assert ingress_record["selected_phase_gate_tick"] == 12
    assert ingress_record["event_gate_stride_ticks"] == 12
    assert ingress_record["ingress_gate_current_legs_pa_hex"] == {
        "identity": (2_500.0).hex(),
        "event_ready": (2_500.0).hex(),
        "case_poison_halt": (-5_000.0).hex(),
    }
    assert ingress_record["same_microtick_spike_influence"] is False

    phase_request = v5._CompositorPhaseRequestPopulation()
    assert tuple(
        phase_request.evaluate(
            request_spikes=np.asarray([True, False], dtype=np.bool_),
            event_ready_storage_pa=np.asarray([5_000.0]),
        )
    ) == (True, False)
    assert tuple(
        phase_request.evaluate(
            request_spikes=np.asarray([False, True], dtype=np.bool_),
            event_ready_storage_pa=np.asarray([5_000.0]),
        )
    ) == (False, True)
    assert tuple(
        phase_request.evaluate(
            request_spikes=np.asarray([False, True], dtype=np.bool_),
            event_ready_storage_pa=np.asarray([0.0]),
        )
    ) == (False, False)
    for invalid_requests in (
        np.asarray([False, False], dtype=np.bool_),
        np.asarray([True, True], dtype=np.bool_),
    ):
        with pytest.raises(ValueError, match="phase-request domain changed"):
            phase_request.evaluate(
                request_spikes=invalid_requests,
                event_ready_storage_pa=np.asarray([5_000.0]),
            )

    ready = v5._NoLeakStorageReadoutPopulation(1)
    ready_zero = np.zeros(1, dtype=np.float64)
    ready.step(
        set_currents_pa=np.asarray([5_000.0]),
        read_gate_currents_pa=ready_zero,
        direct_readout_currents_pa=ready_zero,
        reset_currents_pa=ready_zero,
    )
    first_ready = ready.step(
        set_currents_pa=ready_zero,
        read_gate_currents_pa=np.asarray([5_000.0]),
        direct_readout_currents_pa=ready_zero,
        reset_currents_pa=ready_zero,
    )
    assert tuple(first_ready) == (True,)
    ready.step(
        set_currents_pa=ready_zero,
        read_gate_currents_pa=ready_zero,
        direct_readout_currents_pa=ready_zero,
        reset_currents_pa=np.asarray([-5_000.0]),
    )
    early_storage_before = ready.storage_charge_pa
    early_ready = ready.step(
        set_currents_pa=ready_zero,
        read_gate_currents_pa=np.asarray([5_000.0]),
        direct_readout_currents_pa=ready_zero,
        reset_currents_pa=ready_zero,
    )
    assert tuple(early_ready) == (False,)
    assert ready.storage_charge_pa == early_storage_before == (0.0,)
    early_ingress = v5._NoLeakStorageReadoutPopulation(62)
    early_zeros = np.zeros(62, dtype=np.float64)
    early_ingress.step(
        set_currents_pa=early_zeros,
        read_gate_currents_pa=early_zeros,
        direct_readout_currents_pa=early_zeros,
        reset_currents_pa=early_zeros,
    )
    assert early_ingress.storage_charge_pa == (0.0,) * 62
    ready.step(
        set_currents_pa=np.asarray([5_000.0]),
        read_gate_currents_pa=ready_zero,
        direct_readout_currents_pa=ready_zero,
        reset_currents_pa=ready_zero,
    )
    second_ready = ready.step(
        set_currents_pa=ready_zero,
        read_gate_currents_pa=np.asarray([5_000.0]),
        direct_readout_currents_pa=ready_zero,
        reset_currents_pa=ready_zero,
    )
    assert tuple(second_ready) == (True,)
    assert v5.COMPOSITOR_ADMISSION_RECORD["compositor_ingress_buffer_sha256"] == (
        v5.COMPOSITOR_INGRESS_BUFFER_SHA256
    )
    assert v5.COMPOSITOR_ADMISSION_RECORD["slot_zero_evaluation_tick"] == 6
    assert v5.COMPOSITOR_ADMISSION_RECORD["canonical_occupancy_snapshots_pa_hex"] == (
        ((0.0).hex(), (0.0).hex()),
        ((5_000.0).hex(), (0.0).hex()),
        ((5_000.0).hex(), (5_000.0).hex()),
    )
    assert v5.COMPOSITOR_ADMISSION_RECORD["slot_one_evaluation_tick"] == 7
    assert v5.COMPOSITOR_ADMISSION_RECORD["admitted_slot_relay_tick"] == 8
    assert v5.COMPOSITOR_ADMISSION_RECORD["copy_coincidence_tick"] == 9
    assert v5.COMPOSITOR_ADMISSION_RECORD["admitted_slot_storage_set_tick"] == 10

    admission = v5._SequentialFirstFreeSlotAdmissionPopulation()
    assert tuple(
        admission.evaluate(
            occupied_storage_pa=np.asarray([0.0, 0.0]),
            admission_block_storage_pa=np.asarray([0.0]),
        )
    ) == (True, False)
    assert tuple(
        admission.evaluate(
            occupied_storage_pa=np.asarray([5_000.0, 0.0]),
            admission_block_storage_pa=np.asarray([0.0]),
        )
    ) == (False, True)
    assert tuple(
        admission.evaluate(
            occupied_storage_pa=np.asarray([5_000.0, 5_000.0]),
            admission_block_storage_pa=np.asarray([0.0]),
        )
    ) == (False, False)
    with pytest.raises(ValueError, match="compositor admission storage domain changed"):
        admission.evaluate(
            occupied_storage_pa=np.asarray([0.0, 5_000.0]),
            admission_block_storage_pa=np.asarray([0.0]),
        )

    ingress_pattern = np.zeros(62, dtype=np.bool_)
    ingress_pattern[[3, 24, 24 + 13 + 4, 24 + 13 + 13]] = True
    ingress_pattern[24 + 13 + 13 : 24 + 13 + 13 + 4] = True
    ingress_buffer = v5._NoLeakStorageReadoutPopulation(62)
    ingress_zeros = np.zeros(62, dtype=np.float64)
    ingress_buffer.step(
        set_currents_pa=ingress_pattern.astype(np.float64) * 5_000.0,
        read_gate_currents_pa=ingress_zeros,
        direct_readout_currents_pa=ingress_zeros,
        reset_currents_pa=ingress_zeros,
    )
    for _ in range(8):
        assert not np.any(
            ingress_buffer.step(
                set_currents_pa=ingress_zeros,
                read_gate_currents_pa=ingress_zeros,
                direct_readout_currents_pa=ingress_zeros,
                reset_currents_pa=ingress_zeros,
            )
        )
    buffered_identity = ingress_buffer.step(
        set_currents_pa=ingress_zeros,
        read_gate_currents_pa=np.full(62, 5_000.0),
        direct_readout_currents_pa=ingress_zeros,
        reset_currents_pa=ingress_zeros,
    )
    assert np.array_equal(buffered_identity, ingress_pattern)

    duplicate_admission = admission.evaluate(
        occupied_storage_pa=np.asarray([5_000.0, 0.0]),
        admission_block_storage_pa=np.asarray([5_000.0]),
    )
    assert tuple(duplicate_admission) == (False, False)
    copy_population = v5._BufferedIngressSlotCopyPopulation()
    duplicate_copy = copy_population.evaluate(
        ingress_readout_spikes=buffered_identity,
        admitted_slot_spikes=duplicate_admission,
    )
    assert duplicate_copy.shape == (2, 63)
    assert not np.any(duplicate_copy)
    duplicate_slots = v5._NoLeakStorageReadoutPopulation(2 * 63)
    duplicate_slot_zeros = np.zeros(2 * 63, dtype=np.float64)
    duplicate_slots.step(
        set_currents_pa=duplicate_copy.reshape(-1).astype(np.float64) * 5_000.0,
        read_gate_currents_pa=duplicate_slot_zeros,
        direct_readout_currents_pa=duplicate_slot_zeros,
        reset_currents_pa=duplicate_slot_zeros,
    )
    assert duplicate_slots.storage_charge_pa == (0.0,) * (2 * 63)

    admitted_copy = copy_population.evaluate(
        ingress_readout_spikes=buffered_identity,
        admitted_slot_spikes=np.asarray([True, False], dtype=np.bool_),
    )
    assert np.array_equal(admitted_copy[0, :62], ingress_pattern)
    assert bool(admitted_copy[0, 62]) is True
    assert not np.any(admitted_copy[1])
    with pytest.raises(ValueError, match="admitted empty ingress"):
        copy_population.evaluate(
            ingress_readout_spikes=np.zeros(62, dtype=np.bool_),
            admitted_slot_spikes=np.asarray([True, False], dtype=np.bool_),
        )
    assert not np.any(
        copy_population.evaluate(
            ingress_readout_spikes=np.zeros(62, dtype=np.bool_),
            admitted_slot_spikes=np.asarray([False, False], dtype=np.bool_),
        )
    )
    assert not np.any(
        copy_population.evaluate(
            ingress_readout_spikes=buffered_identity,
            admitted_slot_spikes=np.asarray([False, False], dtype=np.bool_),
        )
    )
    with pytest.raises(ValueError, match="buffered-copy spike domain changed"):
        copy_population.evaluate(
            ingress_readout_spikes=buffered_identity,
            admitted_slot_spikes=np.asarray([True, True], dtype=np.bool_),
        )
    malformed_factor = buffered_identity.copy()
    malformed_factor[4] = True
    with pytest.raises(ValueError, match="event one-hot domain changed"):
        copy_population.evaluate(
            ingress_readout_spikes=malformed_factor,
            admitted_slot_spikes=np.asarray([False, False], dtype=np.bool_),
        )
    malformed_footprint = buffered_identity.copy()
    malformed_footprint[-1] = True
    with pytest.raises(ValueError, match="footprint domain changed"):
        copy_population.evaluate(
            ingress_readout_spikes=malformed_footprint,
            admitted_slot_spikes=np.asarray([True, False], dtype=np.bool_),
        )

    assert circuit["mandatory_absorbing_poison_controls"] == {
        "overlap_first_then_capacity_pressure": {
            "event_start_stop_pairs": ((0, 4), (2, 6), (0, 4), (4, 8), (8, 12)),
            "first_poison_event_index": 1,
            "expected_status_latches": (True, False),
            "expected_eos_status": "STOP_OVERLAPPING_FACTOR_EVENTS",
            "expected_factors": (),
        },
        "capacity_first_then_overlap_pressure": {
            "event_start_stop_pairs": ((0, 4), (4, 8), (8, 12), (0, 4), (2, 6)),
            "first_poison_event_index": 2,
            "expected_status_latches": (False, True),
            "expected_eos_status": "STOP_EVENT_CAPACITY_2",
            "expected_factors": (),
        },
    }
    overlap_first = v5._AbsorbingCompositorPoisonPopulation()
    overlap_trace = (
        overlap_first.step(overlap_decision=False, capacity_decision=False),
        overlap_first.step(overlap_decision=True, capacity_decision=False),
        overlap_first.step(overlap_decision=False, capacity_decision=True),
        overlap_first.step(overlap_decision=False, capacity_decision=True),
        overlap_first.step(overlap_decision=True, capacity_decision=False),
    )
    assert overlap_trace[0] == ((False, False), False)
    assert all(step == ((True, False), True) for step in overlap_trace[1:])

    capacity_first = v5._AbsorbingCompositorPoisonPopulation()
    capacity_trace = (
        capacity_first.step(overlap_decision=False, capacity_decision=False),
        capacity_first.step(overlap_decision=False, capacity_decision=False),
        capacity_first.step(overlap_decision=False, capacity_decision=True),
        capacity_first.step(overlap_decision=True, capacity_decision=False),
        capacity_first.step(overlap_decision=True, capacity_decision=False),
    )
    assert capacity_trace[:2] == (((False, False), False),) * 2
    assert all(step == ((False, True), True) for step in capacity_trace[2:])

    eos = v5._CompositorEosStatusPopulation()
    assert v5.COMPOSITOR_EOS_STATUS_RECORD["microticks"] == {
        "nondestructive_slot_factor_occupied_and_poison_read": 0,
        "ordinary_count_and_poison_status_candidates": 1,
        "one_hot_status_and_poison_gated_factor_outputs": 2,
        "case_reset_relay": 3,
        "complete_case_reset": 4,
        "next_case_event_ready_initialization": 5,
    }
    assert (
        "event-ready storage/readout" in (v5.COMPOSITOR_EOS_STATUS_RECORD["complete_reset_scope"])
    )
    expected_factor_counts = {"zero": 0, "one": 1, "two": 2, "overlap": 0, "capacity": 0}
    for outcome, (occupied_pa, poison_pa, expected_status) in v5.COMPOSITOR_EOS_STATUS_RECORD[
        "mandatory_outcomes"
    ].items():
        observed_status = eos.evaluate_status(
            occupied_storage_pa=np.asarray(occupied_pa),
            poison_storage_pa=np.asarray(poison_pa),
        )
        assert tuple(observed_status) == expected_status
        assert np.count_nonzero(observed_status) == 1
        factor_readouts = np.zeros((2, 24), dtype=np.bool_)
        if outcome in {"one", "two"}:
            factor_readouts[0, 3] = True
        if outcome == "two":
            factor_readouts[1, 7] = True
        observed_factors = eos.evaluate_factor_outputs(
            factor_readout_spikes=factor_readouts,
            status_spikes=observed_status,
        )
        assert np.count_nonzero(observed_factors) == expected_factor_counts[outcome]

    one_status = np.asarray([False, True, False, False, False], dtype=np.bool_)
    malformed_one = np.zeros((2, 24), dtype=np.bool_)
    malformed_one[0, :2] = True
    with pytest.raises(ValueError, match="EOS factor cardinality changed"):
        eos.evaluate_factor_outputs(
            factor_readout_spikes=malformed_one,
            status_spikes=one_status,
        )
    malformed_one[:] = False
    malformed_one[0, 0] = True
    malformed_one[1, 1] = True
    with pytest.raises(ValueError, match="EOS factor cardinality changed"):
        eos.evaluate_factor_outputs(
            factor_readout_spikes=malformed_one,
            status_spikes=one_status,
        )
    zero_status = np.asarray([True, False, False, False, False], dtype=np.bool_)
    with pytest.raises(ValueError, match="EOS factor cardinality changed"):
        eos.evaluate_factor_outputs(
            factor_readout_spikes=malformed_one,
            status_spikes=zero_status,
        )
    two_status = np.asarray([False, False, True, False, False], dtype=np.bool_)
    missing_second = np.zeros((2, 24), dtype=np.bool_)
    missing_second[0, 0] = True
    with pytest.raises(ValueError, match="EOS factor cardinality changed"):
        eos.evaluate_factor_outputs(
            factor_readout_spikes=missing_second,
            status_spikes=two_status,
        )

    overlap_first.step(
        overlap_decision=False,
        capacity_decision=False,
        case_reset=True,
    )
    assert overlap_first.status_charge_pa == (0.0, 0.0)
    assert overlap_first.ingress_allowed is True
    assert tuple(
        eos.evaluate_status(
            occupied_storage_pa=np.asarray([5_000.0, 0.0]),
            poison_storage_pa=np.asarray(overlap_first.status_charge_pa),
        )
    ) == (False, True, False, False, False)

    capacity_first.step(
        overlap_decision=False,
        capacity_decision=False,
        case_reset=True,
    )
    assert tuple(
        eos.evaluate_status(
            occupied_storage_pa=np.asarray([0.0, 0.0]),
            poison_storage_pa=np.asarray(capacity_first.status_charge_pa),
        )
    ) == (True, False, False, False, False)

    two_slot_reset = v5._NoLeakStorageReadoutPopulation(2)
    two_zeros = np.zeros(2, dtype=np.float64)
    two_slot_reset.step(
        set_currents_pa=np.asarray([5_000.0, 5_000.0]),
        read_gate_currents_pa=two_zeros,
        direct_readout_currents_pa=two_zeros,
        reset_currents_pa=two_zeros,
    )
    assert tuple(
        eos.evaluate_status(
            occupied_storage_pa=np.asarray(two_slot_reset.storage_charge_pa),
            poison_storage_pa=np.asarray([0.0, 0.0]),
        )
    ) == (False, False, True, False, False)
    two_slot_reset.step(
        set_currents_pa=two_zeros,
        read_gate_currents_pa=two_zeros,
        direct_readout_currents_pa=two_zeros,
        reset_currents_pa=np.asarray([-5_000.0, -5_000.0]),
    )
    assert tuple(
        eos.evaluate_status(
            occupied_storage_pa=np.asarray(two_slot_reset.storage_charge_pa),
            poison_storage_pa=np.asarray([0.0, 0.0]),
        )
    ) == (True, False, False, False, False)

    slot = v5._NoLeakStorageReadoutPopulation(2)
    zeros = np.zeros(2, dtype=np.float64)
    full_first = np.asarray([5_000.0, 0.0])
    slot.step(
        set_currents_pa=full_first,
        read_gate_currents_pa=zeros,
        direct_readout_currents_pa=zeros,
        reset_currents_pa=zeros,
    )
    for _ in range(2):
        assert tuple(
            slot.step(
                set_currents_pa=zeros,
                read_gate_currents_pa=np.asarray([5_000.0, 5_000.0]),
                direct_readout_currents_pa=zeros,
                reset_currents_pa=zeros,
            )
        ) == (True, False)
    assert slot.storage_charge_pa == (5_000.0, 0.0)
    slot.step(
        set_currents_pa=zeros,
        read_gate_currents_pa=zeros,
        direct_readout_currents_pa=zeros,
        reset_currents_pa=np.asarray([-5_000.0, -5_000.0]),
    )
    assert slot.storage_charge_pa == (0.0, 0.0)

    status = v5._NoLeakStorageReadoutPopulation(2)
    for status_index, expected in enumerate(((True, False), (False, True))):
        one_hot_set = np.zeros(2, dtype=np.float64)
        one_hot_set[status_index] = 5_000.0
        status.step(
            set_currents_pa=one_hot_set,
            read_gate_currents_pa=zeros,
            direct_readout_currents_pa=zeros,
            reset_currents_pa=zeros,
        )
        for _ in range(2):
            assert (
                tuple(
                    status.step(
                        set_currents_pa=zeros,
                        read_gate_currents_pa=np.asarray([5_000.0, 5_000.0]),
                        direct_readout_currents_pa=zeros,
                        reset_currents_pa=zeros,
                    )
                )
                == expected
            )
        assert status.storage_charge_pa == tuple(
            5_000.0 if index == status_index else 0.0 for index in range(2)
        )
        status.step(
            set_currents_pa=zeros,
            read_gate_currents_pa=zeros,
            direct_readout_currents_pa=zeros,
            reset_currents_pa=np.asarray([-5_000.0, -5_000.0]),
        )
        assert status.storage_charge_pa == (0.0, 0.0)

    assert execution["atomic_prerequisite"] == "54/54 unique exact"
    assert execution["atomic_failure_attempt_counts"] == {
        "composition": 0,
        "duplicate": 0,
        "overlap": 0,
        "capacity": 0,
    }
    assert execution["composition_input_inspection_on_atomic_failure"] is False
    assert execution["composition_case_sha256s"] == tuple(
        case.sha256 for case in v5.PROTOCOL.composition_cases
    )
    assert execution["required_composition_exact"] == 18
    assert execution["composition_failure_control_attempt_count"] == 0
    assert execution["control_execution_order"] == (
        *(control.sha256 for control in v5.PROTOCOL.duplicate_controls),
        *(control.sha256 for control in v5.PROTOCOL.overlap_controls),
        *(control.sha256 for control in v5.PROTOCOL.capacity_controls),
    )
    assert execution["required_control_exact"] == {
        "duplicate": 12,
        "overlap": 6,
        "capacity": 3,
    }
    assert execution["retry"] is False
    assert execution["parameter_tuning"] is False
    assert execution["execution_deferred"] is True

    assert authentication["hash_algorithm"] == "SHA-256"
    assert (
        dict(authentication["checkpoint_root_dependencies"])["direct_readout_channel_sha256"]
        == v5.DIRECT_READOUT_CHANNEL_SHA256
    )
    assert authentication["checkpoint_arrays"] == (
        ("excitatory_ticks", "uint8", (24, 512)),
        ("inhibitory_ticks", "uint8", (24, 512)),
    )
    assert authentication["atomic_batch"] == {
        "current_shape": (54, 25),
        "race_many_calls": 1,
        "checkpoint_authentications": 2,
        "per_span_hashing": False,
    }
    assert authentication["composition_batch_checkpoint_authentications"] == 2
    assert (
        authentication["control_batch_control_count"]
        == execution["required_control_attempt_count"]
        == v5.DUPLICATE_CONTROL_COUNT + v5.OVERLAP_CONTROL_COUNT + v5.CAPACITY_CONTROL_COUNT
        == 21
    )
    assert authentication["control_batch_checkpoint_authentications"] == 2
    assert authentication["authentication_sequence"][-2] == (
        "authenticate checkpoint immediately before and after the 21-control batch"
    )
    assert authentication["result_hash_order"] == (
        "checkpoint",
        "atomic-result",
        "composition-result",
        "compositor-control-result",
    )
    assert authentication["terminal_resolve_sequence"] == (
        "authenticate protocol root",
        "authenticate frozen checkpoint",
        "authenticate atomic result",
        "inspect the 54-case atomic gate and exact downstream attempt counts",
        "on atomic STOP prove composition/control results absent and all downstream "
        "attempts zero, then raise before reading any requested batch field",
        "on atomic PASS authenticate the composition result",
        "inspect the exact 18-case composition gate and compositor-control attempt count",
        "on composition STOP prove the control result absent and control attempts "
        "zero, then raise before reading any requested batch field",
        "on composition PASS authenticate the compositor-control result",
        "inspect the exact 21-control compositor gate",
        "on compositor-control STOP raise before reading any requested batch field",
        "only after all three gates PASS authenticate and inspect the requested batch",
    )
    assert authentication["terminal_gate_cardinalities"] == {
        "atomic": 54,
        "composition": 18,
        "compositor_control": execution["required_control_attempt_count"],
    }
    assert authentication["terminal_result_absence_rules"] == {
        "atomic_stop": (
            "composition and compositor-control results must both be absent; "
            "composition and all compositor-control attempt counts must be zero"
        ),
        "composition_stop": (
            "the compositor-control result must be absent and every compositor-"
            "control attempt count must be zero"
        ),
        "all_other_missing_results": "fail closed before requested-batch access",
    }
    assert authentication["terminal_batch_access_barrier"] == (
        "before all three authenticated gates PASS, do not read requested batch "
        "type, length, hash, iterator, or contents"
    )
    assert authentication["future_hashes_are_schema_bound_not_run_results"] is True
    assert authentication["execution_deferred"] is True


def test_protocol_is_immutable_data_only_and_every_execution_surface_is_deferred() -> None:
    protocol = v5.PROTOCOL
    assert (
        protocol.public_input_count,
        protocol.canonical_input_count,
        protocol.protected_input_count,
        protocol.learning_run_count,
        protocol.composition_result_count,
    ) == (0, 0, 0, 0, 0)
    root = protocol.root_record()
    assert root["learner_state_constructed"] is False
    assert root["learning_executed"] is False
    assert root["composition_executed"] is False
    assert root["execution_surface_present"] is False
    assert root["private_mechanism_primitive_present"] is True
    assert root["private_mechanism_primitive_instantiated_at_import"] is False
    assert root["learner_or_full_circuit_execution_surface_present"] is False
    assert root["selector_principal_controls_executed"] is True
    assert root["factor_principal_controls_executed"] is True
    assert root["full_selected_content_circuit_executed"] is False
    assert root["full_factor_circuit_executed"] is False
    assert root["teacher_scout_circuit_executed"] is False
    assert root["compositor_circuit_executed"] is False
    assert root["composition_attempt_count"] == 0
    assert not any(isinstance(value, np.ndarray) for value in _walk(protocol))

    forbidden_surfaces = {
        "build_frozen_teacher_gated_bridge",
        "FrozenTeacherGatedBridge",
        "resolve",
        "fit",
        "compose",
        "ExclusiveFactorResolution",
    }
    assert forbidden_surfaces.isdisjoint(vars(v5))
    public_functions = {
        name
        for name, value in inspect.getmembers(v5, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == v5.__name__
    }
    assert public_functions == set()

    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "RUN"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.atomic_groups[0].lessons[0].factor = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        v5.SELECTOR_TOPOLOGY_RECORD["phase_cells"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        v5.LOCAL_RULE_RECORD["fit_pass_count"] = 2  # type: ignore[index]


def test_template_schedule_and_all_protocol_roots_are_exact() -> None:
    protocol = v5.PROTOCOL
    assert tuple(
        (
            group.template_index,
            group.split,
            group.direction,
            group.cue_slot,
            group.selected_slot,
        )
        for group in protocol.atomic_groups[:6]
    ) == tuple(
        (template_index, *template) for template_index, template in enumerate(v5.TEMPLATE_SPECS)
    )
    assert all(
        tuple(
            (
                group.template_index,
                group.split,
                group.direction,
                group.cue_slot,
                group.selected_slot,
            )
            for group in protocol.atomic_groups[triad_index * 6 : (triad_index + 1) * 6]
        )
        == tuple(
            (template_index, *template) for template_index, template in enumerate(v5.TEMPLATE_SPECS)
        )
        for triad_index in range(3)
    )
    fit_ids = set(protocol.fit_lesson_sha256s)
    ordered_fit = tuple(
        sorted(
            (lesson for lesson in protocol.lessons if lesson.sha256 in fit_ids),
            key=lambda lesson: (lesson.tokens, lesson.factor, lesson.sha256),
        )
    )
    assert v5.LOCAL_RULE_RECORD["fit_schedule"]["lesson_sha256s"] == tuple(
        lesson.sha256 for lesson in ordered_fit
    )

    actual_roots = {
        field.name: getattr(protocol, field.name)
        for field in dataclasses.fields(protocol)
        if field.name.endswith("sha256")
    }
    assert actual_roots == EXPECTED_ROOTS
    assert protocol.protocol_sha256 == v5._namespaced_sha256(
        v5._PROTOCOL_NAMESPACE,
        protocol.root_record(include_protocol_sha256=False),
    )
    assert all(
        len(root) == 64 and root == root.lower() and int(root, 16) >= 0
        for root in actual_roots.values()
    )
