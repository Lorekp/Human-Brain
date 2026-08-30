from __future__ import annotations

import dataclasses
import inspect
from hashlib import sha256

import numpy as np
import pytest
from experiments.r3_pilot import public_train_prompt_selection_v3 as v3
from experiments.r3_pilot import public_train_prompt_selection_v3_protocol as protocol_v3

EXPECTED_ROOTS = {
    "dependency_sha256": "42b859bbdc377b660f75be0485b20c5696e832693019fbdc3733792756af504e",
    "output_registry_sha256": "4b3a68ab28a0ada94cca5bb82461f8a454be37c545ce513ecb7a3e8195d646b4",
    "branch_registry_sha256": "5d7756c186accde372705e34990569bd096eace45eecc1a8cde7a6824d9efa6d",
    "checkpoint_sha256": "7cb126fa194a96a26072f59d326cdb1a7c32cffa06bd5d07e27e28df8101720a",
    "self_posed_result_sha256": "5ca47594bd93cc10a151e03a77bf70f046b16e45b8afdc3965131af38d398e61",
    "control_panel_sha256": "d11a4ee57fb0ec1706f92132a3208696f41b289d0f3180fbc9f88abddc1cf374",
    "fit_projection_sha256": "c70f7a883eb8a94f8dd424c2a10c37ef02a69c7c4f7ebe0f8684d782910c1d11",
    "surface_boundary_sha256": "0f9d9aa81dd0ee6bd00b7a216574981d3e03cfe09dfab5ef77733d93cfdf171d",
    "result_sha256": "e73446e71164a5435c89031f01afed0171b494f54772df57c34c9db86816ad84",
}


def test_fresh_compositional_registry_and_prompt_physical_boundary() -> None:
    assert protocol_v3.PROTOCOL.protocol_sha256 == v3.EXPECTED_PROTOCOL_SHA256
    for relative_path, expected_sha256 in v3._PINNED_SOURCE_SHA256S.items():
        assert sha256((v3._ROOT / relative_path).read_bytes()).hexdigest() == (expected_sha256)
    assert v3.DEPENDENCY_RECORD["teacher_schedule"] == (
        "complete-138-self-posed-pass-then-atomic-at-most-24-fit-projection"
    )
    assert v3.DEPENDENCY_RECORD["v2_checkpoint_or_head_reuse"] is False
    assert v3.DEPENDENCY_RECORD["whole_target_head_count"] == 0

    assert len(v3.OUTPUT_CELLS) == 160
    assert len(v3.OUTPUT_BRANCHES) == 44
    assert sum(row.kind == "atom" for row in v3.OUTPUT_CELLS) == 132
    assert sum(row.kind == "transition" for row in v3.OUTPUT_CELLS) == 28
    assert {row.target_id for row in v3.OUTPUT_CELLS} == {
        *(row.atom_id for row in protocol_v3.OUTPUT_ATOMS),
        *(row.transition_id for row in protocol_v3.OUTPUT_TRANSITIONS),
    }
    assert all(
        {v3.OUTPUT_CELLS[cell_id].channel for cell_id in branch.candidate_cell_ids}
        == {branch.channel}
        for branch in v3.OUTPUT_BRANCHES
    )

    surface = dict(v3.SURFACE_BOUNDARY_RECORD)
    assert surface["neural_input_sources"] == (
        "physical-relation-assemblies",
        "physical-role-assemblies",
        "physical-filler-assemblies",
        "ordered-role-transition-assemblies",
    )
    assert surface["neural_output"] == ("channel-local compositional atoms and ordered transitions")
    assert (surface["learner_array_count"], surface["learner_array_bytes"]) == (
        2,
        327_680,
    )
    zero_fields = (
        "dev_input_count",
        "protected_input_count",
        "desired_output_read_count",
        "program_artifact_read_count",
        "executable_source_read_count",
        "ast_artifact_read_count",
        "property_row_read_count",
        "task_identifier_neural_input_count",
        "family_identifier_neural_input_count",
        "entrypoint_spelling_neural_input_count",
        "audit_hash_neural_input_count",
        "prompt_factor_inference_count",
        "whole_target_head_count",
        "host_label_assembly_count",
        "host_lookup_selection_count",
        "host_union_selection_count",
        "host_ranking_count",
        "candidate_program_count",
        "proof_count",
    )
    assert all(surface[field] == 0 for field in zero_fields)
    source = inspect.getsource(v3)
    assert "public_train_prompt_selection_v2" not in source
    assert "whole-target" in source
    assert "rank_candidates" not in source


def test_complete_self_posed_curriculum_freezes_one_cap_four_checkpoint() -> None:
    treatment = v3.SELF_POSED_RESULT
    assert (
        treatment.lesson_count,
        treatment.exact_lesson_count,
        treatment.mismatch_lesson_count,
    ) == (138, 138, 0)
    assert tuple(
        (
            row.channel,
            row.lesson_count,
            row.exact_lesson_count,
            row.mismatch_lesson_count,
        )
        for row in treatment.channel_outcomes
    ) == (
        ("root-production-family", 20, 20, 0),
        ("ordered-parent-child-edge", 34, 34, 0),
        ("argument-operand-orientation", 26, 26, 0),
        ("nesting-relation", 28, 28, 0),
        ("parameter-role-binding", 6, 6, 0),
        ("literal-slot-binding", 10, 10, 0),
        ("literal-value-copy", 10, 10, 0),
        ("terminal-root-eclass-agreement", 4, 4, 0),
    )
    assert all(row.exact for row in treatment.replays)
    assert all(row.emitted_target_ids == row.expected_target_ids for row in treatment.replays)
    assert v3.CHECKPOINT.record() == {
        "output_cell_count": 160,
        "atom_cell_count": 132,
        "transition_cell_count": 28,
        "branch_count": 44,
        "array_shape": (160, 1024),
        "array_dtype": "|u1",
        "array_bytes": 327_680,
        "tick_cap": 4,
        "excitatory_nonzero_ticks": 1920,
        "inhibitory_nonzero_ticks": 11_953,
        "self_posed_teacher_lesson_count": 138,
        "canonical_fit_teacher_update_count": 0,
        "heldout_teacher_identity_count": 0,
        "checkpoint_sha256": EXPECTED_ROOTS["checkpoint_sha256"],
    }
    assert v3.RESULT.training_work.events == (
        ("branch-feature-router-decision", 816),
        ("conjunction-cell-emission", 8136),
        ("excitatory-tick-commit", 7680),
        ("fixed-conjunction-synapse-read", 176_256),
        ("fixed-transition-conjunction-synapse-read", 116_640),
        ("inhibitory-tick-commit", 29_934),
        ("physical-assembly-cell-read", 14_688),
        ("relation-router-cell-read", 26_496),
        ("relation-router-decision", 138),
        ("role-router-cell-read", 60_096),
        ("teacher-cell-spike", 816),
        ("transition-router-cell-read", 40_512),
    )
    assert v3.RESULT.training_work.total == 482_208

    before = v3.WORKSPACE.checkpoint_sha256
    excitatory = v3.WORKSPACE.excitatory_ticks
    inhibitory = v3.WORKSPACE.inhibitory_ticks
    assert not excitatory.flags.writeable and not inhibitory.flags.writeable
    with pytest.raises(ValueError):
        excitatory[0, 0] = 4
    assert v3.WORKSPACE.checkpoint_sha256 == before

    clone = v3.FrozenCompositionalWorkspace(
        output_cells=v3.WORKSPACE.output_cells,
        branches=v3.WORKSPACE.branches,
        excitatory=v3.WORKSPACE.excitatory_ticks,
        inhibitory=v3.WORKSPACE.inhibitory_ticks,
        metadata=v3.WORKSPACE._metadata,
    )
    forged = clone.excitatory_ticks.copy()
    forged[0, 0] = np.uint8((int(forged[0, 0]) + 1) % 5)
    forged.setflags(write=False)
    object.__setattr__(clone, "_excitatory", forged)
    with pytest.raises(RuntimeError, match="checkpoint authentication failed"):
        clone.authenticate()


def test_all_six_causal_controls_falsify_without_checkpoint_reuse() -> None:
    assert tuple(
        (
            row.name,
            row.exact_lesson_count,
            row.mismatch_lesson_count,
            row.work.total,
        )
        for row in v3.CONTROLS
    ) == (
        ("teacher-off", 0, 138, 3_672_976),
        ("cell-off", 0, 138, 138),
        ("reverse-order", 0, 138, 3_229_198),
        ("role-swap", 0, 138, 3_229_198),
        ("filler-swap", 0, 138, 3_229_198),
        ("teacher-derangement", 0, 138, 3_713_511),
    )
    controls = {row.name: row for row in v3.CONTROLS}
    assert "teacher-cell-spike" not in dict(controls["teacher-off"].work.events)
    assert {event: count for event, count in controls["cell-off"].work.events if count} == {
        "relation-router-stop": 138
    }
    assert dict(controls["teacher-derangement"].work.events)["teacher-cell-spike"] == 816
    assert controls["cell-off"].checkpoint_sha256 == v3.CHECKPOINT.checkpoint_sha256
    assert controls["reverse-order"].checkpoint_sha256 == v3.CHECKPOINT.checkpoint_sha256
    assert controls["role-swap"].checkpoint_sha256 == v3.CHECKPOINT.checkpoint_sha256
    assert controls["filler-swap"].checkpoint_sha256 == v3.CHECKPOINT.checkpoint_sha256
    assert controls["teacher-off"].checkpoint_sha256 != v3.CHECKPOINT.checkpoint_sha256
    assert controls["teacher-derangement"].checkpoint_sha256 != (v3.CHECKPOINT.checkpoint_sha256)


def test_atomic_fit_projection_stops_on_depth_four_before_teacher_or_heldout() -> None:
    audit = v3.FIT_PROJECTION_AUDIT
    assert v3.STATUS == v3.RESULT.status == v3.STOP_FIT_PROJECTION
    assert (
        audit.fit_identity_count,
        audit.projectable_identity_count,
        audit.unsupported_identity_count,
    ) == (24, 19, 5)
    assert tuple(row.record() for row in audit.channel_outcomes) == tuple(
        {
            "channel": channel,
            "fit_identity_count": 24,
            "supported_identity_count": 19 if channel == "nesting-relation" else 24,
            "unsupported_identity_count": 5 if channel == "nesting-relation" else 0,
            "missing_atom_occurrence_count": 5 if channel == "nesting-relation" else 0,
        }
        for channel in (
            "root-production-family",
            "ordered-parent-child-edge",
            "argument-operand-orientation",
            "nesting-relation",
            "parameter-role-binding",
            "literal-slot-binding",
            "literal-value-copy",
            "terminal-root-eclass-agreement",
        )
    )
    assert audit.missing_atom_ids == ("nesting-relation|depth|depth|4",)
    assert audit.missing_atom_occurrence_count == 5
    assert audit.work.events == (
        ("canonical-fit-relay-field-read", 1133),
        ("canonical-fit-vocabulary-lookup", 1865),
    )
    assert audit.work.total == 2998
    assert (
        audit.canonical_teacher_update_count,
        audit.heldout_target_read_count,
        audit.heldout_replay_identity_count,
    ) == (0, 0, 0)
    assert v3.RESULT.heldout_channel_outcomes == ()
    assert v3.SURFACE_BOUNDARY_RECORD["checkpoint_frozen_before_canonical_fit_projection"]
    assert v3.SURFACE_BOUNDARY_RECORD["fit_projection_atomic_before_teacher_updates"]
    assert v3.SURFACE_BOUNDARY_RECORD["heldout_surface_opened"] is False

    actual_roots = {
        field.name: getattr(v3.RESULT, field.name)
        for field in dataclasses.fields(v3.RESULT)
        if field.name.endswith("sha256")
    }
    assert actual_roots == EXPECTED_ROOTS
    assert v3.RESULT.physical_work_total == 20_788_623
    assert v3.RESULT.result_sha256 == v3._namespaced_sha256(
        v3._RESULT_NAMESPACE,
        v3.RESULT.root_record(include_result_sha256=False),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v3.RESULT.status = v3.PASS  # type: ignore[misc]
