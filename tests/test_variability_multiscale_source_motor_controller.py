from __future__ import annotations

from dataclasses import fields, replace
from inspect import signature

import pytest

from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.source_relative_action_tape import TapeValence
from humanbrain.variability_aware_segment_binder import VariabilityBindingDecision
from humanbrain.variability_multiscale_source_motor_controller import (
    SuccessfulVariabilityMultiscaleSourceMotorExperience,
    VariabilityMultiscaleSourceMotorAblation,
    VariabilityMultiscaleSourceMotorController,
)


def _transition(
    target_directory: str,
    target_stem: str,
    paths: tuple[str, str, str],
) -> tuple[
    bytes,
    tuple[OpaqueWorkspaceFile, ...],
    tuple[OpaqueWorkspaceFile, ...],
]:
    target_path = f"{target_directory}/{target_stem}.py"
    if target_path not in paths:
        raise ValueError("target path is absent")
    retained = f"{target_stem}_retained\n".encode("ascii")
    before = tuple(
        OpaqueWorkspaceFile(
            path,
            (
                retained + f"{target_stem}_discarded\n".encode("ascii")
                if path == target_path
                else f"opaque_{path.replace('/', '_').replace('.', '_')}\n".encode(
                    "ascii"
                )
            ),
        )
        for path in paths
    )
    after = tuple(
        OpaqueWorkspaceFile(
            item.path, retained if item.path == target_path else item.content
        )
        for item in before
    )
    query = b"retain " + target_stem.encode("ascii") + b" from the complete stream"
    return query, before, after


def _experience(
    provenance_id: str,
    target_directory: str,
    target_stem: str,
    paths: tuple[str, str, str],
) -> SuccessfulVariabilityMultiscaleSourceMotorExperience:
    query, before, after = _transition(target_directory, target_stem, paths)
    return SuccessfulVariabilityMultiscaleSourceMotorExperience(
        query=query,
        before_files=before,
        after_files=after,
        provenance_id=provenance_id,
    )


def _controller() -> VariabilityMultiscaleSourceMotorController:
    return VariabilityMultiscaleSourceMotorController.grow(
        (
            _experience(
                "opaque-train-cedar",
                "amber",
                "cedar_entry",
                (
                    "amber/cedar_entry.py",
                    "moss/brass_slot.py",
                    "quartz/umber_slot.py",
                ),
            ),
            _experience(
                "opaque-train-lilac",
                "harbor",
                "lilac_entry",
                (
                    "brass/ochre_slot.py",
                    "harbor/lilac_entry.py",
                    "umber/violet_slot.py",
                ),
            ),
        )
    )


def test_variable_controller_emits_fresh_live_transitions_at_varied_positions() -> None:
    controller = _controller()
    fresh = (
        _transition(
            "acorn",
            "maple_entry",
            (
                "acorn/maple_entry.py",
                "cedar/sable_slot.py",
                "zinc/topaz_slot.py",
            ),
        ),
        _transition(
            "onyx",
            "vireo_entry",
            (
                "birch/cobalt_slot.py",
                "onyx/vireo_entry.py",
                "willow/yarrow_slot.py",
            ),
        ),
        _transition(
            "zircon",
            "lumen_entry",
            (
                "citrine/ivory_slot.py",
                "sienna/russet_slot.py",
                "zircon/lumen_entry.py",
            ),
        ),
    )

    for query, before, expected_after in fresh:
        decision = controller.respond(query, before)
        observation = MultiscaleOpaqueEventTape.observe(query, before)

        assert decision.decision == "EMIT"
        assert len(decision.emissions) == 2
        assert all(binding.boundary_match_count == 2 for binding in decision.bindings)
        assert len(decision.inventory_races) == 2
        assert all(not race.stopped for race in decision.inventory_races)
        assert observation.materialize_motor_emission(decision.emissions) == tuple(
            sorted(expected_after, key=lambda item: item.path)
        )
        controller.resolve(TapeValence.NEUTRAL)

    summary = controller.training_summary
    assert summary.experience_count == 2
    assert summary.role_binder_count == 1
    assert summary.role_offsets == (-4,)
    assert summary.propagation_hop_count == 4
    assert summary.stable_intermediate_synapse_count == 2
    assert summary.wildcard_intermediate_hop_count == 1
    assert summary.wildcard_support_ticks == 2
    assert summary.boundary_synapse_count == 2
    assert summary.action_program_cells == 1


def test_unsupported_depth_ambiguity_and_causal_controls_physically_stop() -> None:
    controller = _controller()
    query, before, _ = _transition(
        "hazel",
        "juniper_entry",
        (
            "hazel/juniper_entry.py",
            "spruce/teal_slot.py",
            "yarrow/umber_slot.py",
        ),
    )
    expected_stages = {
        VariabilityMultiscaleSourceMotorAblation.EQUALITY_OFF: "equality",
        VariabilityMultiscaleSourceMotorAblation.PROPAGATION_OFF: "propagation",
        VariabilityMultiscaleSourceMotorAblation.BOUNDARY_OFF: "boundary",
    }

    for ablation, expected_stage in expected_stages.items():
        decision = controller.respond(query, before, ablation=ablation)

        assert decision.decision == "STOP"
        assert decision.emissions == ()
        assert decision.stop_stage == "role_binder_0"
        assert decision.bindings[-1].stop_stage == expected_stage
        assert decision.bindings[-1].races[-1].stopped

    deep_query, deep_before, _ = _transition(
        "outer/inner",
        "prism_entry",
        (
            "brass/copper_slot.py",
            "outer/inner/prism_entry.py",
            "sable/tawny_slot.py",
        ),
    )
    unsupported = controller.respond(deep_query, deep_before)

    assert unsupported.decision == "STOP"
    assert unsupported.emissions == ()
    assert unsupported.stop_stage == "role_binder_0"
    assert unsupported.bindings[-1].stop_stage == "propagation"
    assert unsupported.bindings[-1].races[-1].stopped

    ambiguous = controller.respond(
        b"retain prism_entry or quartz_entry from the complete stream",
        (
            OpaqueWorkspaceFile(
                "prism/prism_entry.py",
                b"prism_entry_retained\nprism_entry_discarded\n",
            ),
            OpaqueWorkspaceFile(
                "quartz/quartz_entry.py",
                b"quartz_entry_retained\nquartz_entry_discarded\n",
            ),
            OpaqueWorkspaceFile("sienna/tau_slot.py", b"opaque_tau\n"),
        ),
    )

    assert ambiguous.decision == "STOP"
    assert ambiguous.emissions == ()
    assert ambiguous.stop_stage == "role_binder_0"
    assert ambiguous.bindings[-1].stop_stage == "equality"
    assert ambiguous.bindings[-1].stop_reason == "equality:collision"
    assert ambiguous.bindings[-1].races[-1].simultaneous_spike_cells == 2


def test_role_inventory_collision_and_live_multiplicity_use_physical_races() -> None:
    duplicate_binder_controller = _controller()
    duplicate_binder_controller._binders = (
        duplicate_binder_controller._binders[0],
        duplicate_binder_controller._binders[0],
    )
    query, before, _ = _transition(
        "onyx",
        "vireo_entry",
        (
            "birch/cobalt_slot.py",
            "onyx/vireo_entry.py",
            "willow/yarrow_slot.py",
        ),
    )

    role_collision = duplicate_binder_controller.respond(query, before)

    assert role_collision.decision == "STOP"
    assert role_collision.stop_stage == "role_inventory"
    assert role_collision.stop_reason == "role_inventory:identity_collision"
    assert len(role_collision.inventory_races) == 1
    assert role_collision.inventory_races[-1].population == "role_inventory_identity_0"
    assert role_collision.inventory_races[-1].stopped
    assert role_collision.inventory_races[-1].stop_reason == "collision"
    assert role_collision.inventory_races[-1].simultaneous_spike_cells == 2

    class _FixedRoleBinder:
        checkpoint_sha256 = "f" * 64

        def __init__(self, role: bytes) -> None:
            self._role = role

        def respond(
            self, *_args: object, **_kwargs: object
        ) -> VariabilityBindingDecision:
            return VariabilityBindingDecision(
                decision="BIND",
                role_identity=self._role,
                stop_stage=None,
                stop_reason=None,
                boundary_match_count=2,
                races=(),
                physical_cell_steps=0,
                checkpoint_sha256=self.checkpoint_sha256,
                query_sha256="e" * 64,
                sha256="d" * 64,
            )

    duplicate_live_files = (
        OpaqueWorkspaceFile("amber/acorn.py", b"shared_event\n"),
        OpaqueWorkspaceFile("violet/vireo.py", b"shared_event\n"),
    )
    observation = MultiscaleOpaqueEventTape.observe(
        b"opaque complete observation",
        duplicate_live_files,
    )
    duplicate_role = observation.motor_source_tokens[1]
    assert duplicate_role == observation.motor_source_tokens[3]

    live_collision_controller = _controller()
    live_collision_controller._binders = (_FixedRoleBinder(duplicate_role),)
    live_collision = live_collision_controller.respond(
        b"opaque complete observation",
        duplicate_live_files,
    )

    assert live_collision.decision == "STOP"
    assert live_collision.stop_stage == "role_inventory"
    assert live_collision.stop_reason == "role_inventory:non_unique_live_motor_role"
    assert len(live_collision.inventory_races) == 2
    assert (
        live_collision.inventory_races[-1].population == "live_motor_role_occurrence_0"
    )
    assert live_collision.inventory_races[-1].stopped
    assert live_collision.inventory_races[-1].stop_reason == "collision"
    assert live_collision.inventory_races[-1].simultaneous_spike_cells == 2


def test_growth_requires_independent_variation_and_public_api_has_no_route() -> None:
    first = _experience(
        "opaque-train-cedar",
        "amber",
        "cedar_entry",
        (
            "amber/cedar_entry.py",
            "moss/brass_slot.py",
            "quartz/umber_slot.py",
        ),
    )

    with pytest.raises(ValueError, match="at least two independent"):
        VariabilityMultiscaleSourceMotorController.grow((first,))
    with pytest.raises(ValueError, match="duplicate replays"):
        VariabilityMultiscaleSourceMotorController.grow(
            (first, replace(first, provenance_id="alternate-provenance"))
        )

    same_directory = _experience(
        "opaque-train-lilac",
        "amber",
        "lilac_entry",
        (
            "amber/lilac_entry.py",
            "moss/ochre_slot.py",
            "quartz/violet_slot.py",
        ),
    )
    with pytest.raises(ValueError, match="vary intermediate identities"):
        VariabilityMultiscaleSourceMotorController.grow((first, same_directory))

    assert tuple(
        field.name
        for field in fields(SuccessfulVariabilityMultiscaleSourceMotorExperience)
    ) == ("query", "before_files", "after_files", "provenance_id")
    assert tuple(
        signature(VariabilityMultiscaleSourceMotorController.respond).parameters
    ) == (
        "self",
        "query",
        "files",
        "ablation",
    )
    assert tuple(VariabilityMultiscaleSourceMotorAblation) == (
        VariabilityMultiscaleSourceMotorAblation.NONE,
        VariabilityMultiscaleSourceMotorAblation.EQUALITY_OFF,
        VariabilityMultiscaleSourceMotorAblation.PROPAGATION_OFF,
        VariabilityMultiscaleSourceMotorAblation.BOUNDARY_OFF,
    )
