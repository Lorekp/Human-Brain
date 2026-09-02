from __future__ import annotations

from dataclasses import fields
from inspect import signature

from humanbrain.boundary_multiscale_source_motor_controller import (
    BoundaryMultiscaleSourceMotorAblation,
    BoundaryMultiscaleSourceMotorController,
    SuccessfulBoundaryMultiscaleSourceMotorExperience,
)
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueWorkspaceFile,
)
from humanbrain.source_relative_action_tape import TapeValence


def _transition(
    target: str,
    paths: tuple[str, str, str],
) -> tuple[
    bytes,
    tuple[OpaqueWorkspaceFile, ...],
    tuple[OpaqueWorkspaceFile, ...],
]:
    target_path = f"{target}.blob"
    if target_path not in paths:
        raise ValueError("target path is absent")
    retained = f"{target}_retained\n".encode("ascii")
    before = tuple(
        OpaqueWorkspaceFile(
            path,
            (
                retained + f"{target}_discarded\n".encode("ascii")
                if path == target_path
                else f"opaque_{path.replace('.', '_')}\n".encode("ascii")
            ),
        )
        for path in paths
    )
    after = tuple(
        OpaqueWorkspaceFile(item.path, retained if item.path == target_path else item.content)
        for item in before
    )
    return b"trim " + target.encode("ascii") + b" from the complete stream", before, after


def _experience(
    provenance_id: str,
    target: str,
    paths: tuple[str, str, str],
) -> SuccessfulBoundaryMultiscaleSourceMotorExperience:
    query, before, after = _transition(target, paths)
    return SuccessfulBoundaryMultiscaleSourceMotorExperience(
        query=query,
        before_files=before,
        after_files=after,
        provenance_id=provenance_id,
    )


def _controller() -> BoundaryMultiscaleSourceMotorController:
    return BoundaryMultiscaleSourceMotorController.grow(
        (
            _experience(
                "opaque-train-acorn",
                "acorn_entry",
                ("acorn_entry.blob", "moss_slot.blob", "quartz_slot.blob"),
            ),
            _experience(
                "opaque-train-lilac",
                "lilac_entry",
                ("brass_slot.blob", "lilac_entry.blob", "umber_slot.blob"),
            ),
        )
    )


def test_boundary_controller_composes_fresh_first_middle_and_last_segments() -> None:
    controller = _controller()
    fresh = (
        _transition(
            "cedar_entry",
            ("cedar_entry.blob", "maple_slot.blob", "zinc_slot.blob"),
        ),
        _transition(
            "onyx_entry",
            ("birch_slot.blob", "onyx_entry.blob", "willow_slot.blob"),
        ),
        _transition(
            "zircon_entry",
            ("cobalt_slot.blob", "sable_slot.blob", "zircon_entry.blob"),
        ),
    )

    for query, before, expected_after in fresh:
        decision = controller.respond(query, before)
        observation = MultiscaleOpaqueEventTape.observe(query, before)

        assert decision.decision == "EMIT"
        assert len(decision.emissions) == 2
        assert all(binding.boundary_match_count == 2 for binding in decision.bindings)
        assert observation.materialize_motor_emission(decision.emissions) == tuple(
            sorted(expected_after, key=lambda item: item.path)
        )
        controller.resolve(TapeValence.NEUTRAL)

    summary = controller.training_summary
    assert summary.experience_count == 2
    assert summary.role_binder_count == 1
    assert summary.boundary_synapse_count == 2
    assert summary.action_program_cells == 1


def test_boundary_controls_and_ambiguous_query_physically_stop() -> None:
    controller = _controller()
    query, before, _ = _transition(
        "hazel_entry",
        ("hazel_entry.blob", "spruce_slot.blob", "yarrow_slot.blob"),
    )
    expected_stages = {
        BoundaryMultiscaleSourceMotorAblation.EQUALITY_OFF: "equality",
        BoundaryMultiscaleSourceMotorAblation.PROPAGATION_OFF: "propagation",
        BoundaryMultiscaleSourceMotorAblation.BOUNDARY_OFF: "boundary",
    }

    for ablation, expected_stage in expected_stages.items():
        decision = controller.respond(query, before, ablation=ablation)

        assert decision.decision == "STOP"
        assert decision.emissions == ()
        assert decision.stop_stage == "role_binder_0"
        assert decision.bindings[-1].stop_stage == expected_stage
        assert decision.bindings[-1].races[-1].stopped

    left = OpaqueWorkspaceFile("prism_entry.blob", b"prism_keep\nprism_drop\n")
    right = OpaqueWorkspaceFile("quartz_entry.blob", b"quartz_keep\nquartz_drop\n")
    decoy = OpaqueWorkspaceFile("sienna_slot.blob", b"sienna_opaque\n")
    ambiguous = controller.respond(
        b"trim prism_entry or quartz_entry from the complete stream",
        (left, right, decoy),
    )

    assert ambiguous.decision == "STOP"
    assert ambiguous.stop_stage == "role_binder_0"
    assert ambiguous.bindings[-1].stop_stage == "equality"
    assert ambiguous.bindings[-1].stop_reason == "equality:collision"
    assert ambiguous.bindings[-1].races[-1].simultaneous_spike_cells == 2


def test_prepared_observation_preserves_frozen_decision_bytes() -> None:
    controller = _controller()
    query, before, _ = _transition(
        "onyx_entry",
        ("birch_slot.blob", "onyx_entry.blob", "willow_slot.blob"),
    )
    observation = MultiscaleOpaqueEventTape.observe(query, before)

    prepared = controller._respond_observation(observation)

    assert controller.checkpoint_sha256 == (
        "f61790e41aa05e948de4982aab25b7f9cbb09117fd0c9105e9d34abb8d57224e"
    )
    assert prepared.query_sha256 == (
        "53d4b6aac9a3f0835d7c4255bf779ee650f100591dd99b4f8d66662e1ad2616a"
    )
    assert prepared.bindings[0].sha256 == (
        "b70e0f0bb7e5eccb4f7c6108fc26f3acea9f85c856285fe0bcbce135c8c6cfa7"
    )
    assert prepared.action is not None
    assert prepared.action.sha256 == (
        "dcb1216462c1bb32f54b68d09ca827505aadaea242a712326ce512064f69bd8d"
    )
    assert prepared.action.eligibility is not None
    assert prepared.action.eligibility.sha256 == (
        "e1f3c2d264cc09e9b2fcdcabb51afba21cd5a4f6b963d8133f8309836a3f6cf0"
    )
    assert prepared.physical_cell_steps == 151_200
    assert prepared.sha256 == ("90c4cfca018483ddf549cda75ad0f83483727d9cb1a16310c27b167dfec50e2a")
    controller.resolve(TapeValence.NEUTRAL)

    public = controller.respond(query, before)
    assert public == prepared
    controller.resolve(TapeValence.NEUTRAL)


def test_public_boundary_contains_no_target_or_delimiter_route() -> None:
    assert tuple(
        field.name for field in fields(SuccessfulBoundaryMultiscaleSourceMotorExperience)
    ) == ("query", "before_files", "after_files", "provenance_id")
    assert tuple(signature(BoundaryMultiscaleSourceMotorController.respond).parameters) == (
        "self",
        "query",
        "files",
        "ablation",
    )
    assert tuple(BoundaryMultiscaleSourceMotorAblation) == (
        BoundaryMultiscaleSourceMotorAblation.NONE,
        BoundaryMultiscaleSourceMotorAblation.EQUALITY_OFF,
        BoundaryMultiscaleSourceMotorAblation.PROPAGATION_OFF,
        BoundaryMultiscaleSourceMotorAblation.BOUNDARY_OFF,
    )
