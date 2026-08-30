from __future__ import annotations

import ast
import dataclasses
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from experiments.r3_pilot import public_train_prompt_selection_v4 as v4
from experiments.r3_pilot import public_train_prompt_selection_v4_protocol as protocol_v4

EXPECTED_RESULT_SHA256 = (
    "e2249e0277a9ad49ea5eabe452843294ce7c658307e1ad467741a190b0e5fd21"
)


def test_exact_banked_protocol_and_v3_provenance_are_authenticated_without_reuse() -> None:
    assert v4.V4_PROTOCOL_COMMIT == "4471534d7cc562a14dd5d8101a9594ab892a2372"
    assert protocol_v4.PROTOCOL.protocol_sha256 == v4.V4_PROTOCOL_ROOT
    assert sha256(Path(v4.V4_PROTOCOL_SOURCE_PATH).read_bytes()).hexdigest() == (
        v4.V4_PROTOCOL_SOURCE_SHA256
    )
    assert sha256(Path(v4.V4_PROTOCOL_TEST_PATH).read_bytes()).hexdigest() == (
        v4.V4_PROTOCOL_TEST_SHA256
    )
    assert v4.DEPENDENCY_RECORD["v3_learner_commit"] == (
        "582de79f7b75aceea8f4ed64871310a15c96830a"
    )
    assert sha256(Path(v4.V3_LEARNER_SOURCE_PATH).read_bytes()).hexdigest() == (
        v4.V3_LEARNER_SOURCE_SHA256
    )
    assert sha256(Path(v4.V3_LEARNER_TEST_PATH).read_bytes()).hexdigest() == (
        v4.V3_LEARNER_TEST_SHA256
    )
    assert v4.DEPENDENCY_RECORD["v3_learner_result_sha256_audit_only"] == (
        v4.V3_LEARNER_RESULT_SHA256_AUDIT_ONLY
    )
    assert v4.DEPENDENCY_RECORD["v3_learner_module_imported_or_executed"] is False
    assert v4.DEPENDENCY_RECORD["v3_checkpoint_weight_or_output_reuse"] is False
    assert "experiments.r3_pilot.public_train_prompt_selection_v3" not in sys.modules

    tree = ast.parse(Path(v4.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "experiments.r3_pilot.public_train_prompt_selection_v3" not in imported_modules
    assert "experiments.r3_pilot.public_train_prompt_selection_v2" not in imported_modules


def test_self_posed_pass_controls_and_base_checkpoint_are_exact_and_immutable() -> None:
    assert v4.STATUS == v4.PASS
    assert v4.SELF_POSED_RESULT.lesson_count == 110
    assert v4.SELF_POSED_RESULT.exact_lesson_count == 110
    assert v4.SELF_POSED_RESULT.mismatch_lesson_count == 0
    assert v4.SELF_POSED_RESULT.sha256 == (
        "e1b045c147eb94e98466f1afa537973f2dc2e59d7b709da7f4df9692b297a0af"
    )
    assert tuple((row.name, row.exact_lesson_count) for row in v4.CONTROLS) == (
        ("teacher-off", 0),
        ("cell-off", 0),
        ("reverse-order", 0),
        ("role-swap", 0),
        ("filler-swap", 0),
        ("teacher-derangement", 0),
    )
    assert all(row.mismatch_lesson_count == 110 for row in v4.CONTROLS)
    assert v4.CONTROL_PANEL_SHA256 == (
        "20549ae7967b5617a19155f9b98a21b8b5c2ee38cb2fb394e1d07e255dfa025c"
    )
    expected_channel_counts = Counter(row.audit_channel for row in protocol_v4.LESSONS)
    assert {
        row.channel: row.exact_lesson_count
        for row in v4.SELF_POSED_RESULT.channel_outcomes
    } == expected_channel_counts

    base = v4.BASE_CHECKPOINT
    assert base.phase == "self-posed"
    assert base.checkpoint_sha256 == (
        "73284e167237e0105f6262e4232b42ee3f1bbe264d653b3fe53cf33857edca52"
    )
    assert base.output_cell_count == 134
    assert base.channel_router_cell_count == 7
    assert base.atom_cell_count == 103
    assert base.transition_cell_count == 24
    assert base.branch_count == 39
    assert base.array_shape == (134, 50_176)
    assert base.array_dtype == "|u1"
    assert base.array_bytes == 13_447_168
    assert base.excitatory_nonzero_ticks == 1_608
    assert base.inhibitory_nonzero_ticks == 8_958
    assert base.excitatory_max_tick == base.inhibitory_max_tick == 4
    assert base.self_posed_teacher_lesson_count == 110
    assert base.canonical_fit_teacher_identity_count == 0
    assert base.heldout_teacher_identity_count == 0
    assert not np.count_nonzero(
        v4.BASE_WORKSPACE._excitatory[:, v4.PROMPT_FEATURE_OFFSET :]
    )
    assert not np.count_nonzero(
        v4.BASE_WORKSPACE._inhibitory[:, v4.PROMPT_FEATURE_OFFSET :]
    )
    checkpoint_before = v4.BASE_WORKSPACE.checkpoint_sha256
    with pytest.raises(ValueError):
        v4.BASE_WORKSPACE.excitatory_ticks[0, 0] = 1
    assert v4.BASE_WORKSPACE.checkpoint_sha256 == checkpoint_before


def test_atomic_fit_preflight_and_single_replay_publish_all_ordered_occurrences() -> None:
    preflight = v4.FIT_PREFLIGHT
    assert preflight.status == protocol_v4.PASS_PROTOCOL
    assert preflight.fit_identity_count == 24
    assert preflight.prompt_trace_collision_count == 0
    assert preflight.route_feature_target_conflict_count == 0
    assert preflight.branch_feature_target_conflict_count == 0
    assert preflight.occurrence_overflow_count == 0
    assert preflight.binding_self_alias_count == 0
    assert preflight.projection_receipt_mismatch_count == 0
    assert preflight.maximum_occurrence_count == 21
    assert preflight.target_occurrence_count == 357
    assert preflight.edge_occurrence_count == 117
    assert preflight.heldout_target_read_count == 0
    assert preflight.heldout_replay_identity_count == 0
    assert preflight.sha256 == (
        "4ffa676443b1c68886f1bec7381e12f4286cf4fa705d3699031170b6a289ecc2"
    )

    fit = v4.FIT_RESULT
    assert fit is not None
    assert fit.identity_count == fit.exact_identity_count == 24
    assert fit.mismatch_identity_count == 0
    assert fit.emitted_occurrence_count == fit.expected_occurrence_count == 357
    assert fit.checkpoint_authentication_count == 2
    assert fit.sha256 == (
        "95e242c186d3a27bc511e3c905a2103313e56dcd973d6c4ba5b5222699af8620"
    )
    assert all(row.exact and row.termination == "STOP" for row in fit.replays)
    assert all(
        occurrence.exact
        and occurrence.channel_decision.decision == "RELAY"
        and occurrence.selected_channel == occurrence.expected_channel
        and occurrence.emitted_by_branch == occurrence.expected_by_branch
        for replay in fit.replays
        for occurrence in replay.occurrence_replays
    )
    edge_bindings = tuple(
        occurrence.occurrence_binding
        for replay in fit.replays
        for occurrence in replay.occurrence_replays
        if occurrence.occurrence_binding is not None
    )
    assert len(edge_bindings) == 117
    assert all(
        len(binding.parent_cells) == len(binding.child_cells) == 12
        and binding.parent_cells != binding.child_cells
        for binding in edge_bindings
    )
    assert all(
        len(
            {
                occurrence.occurrence_binding.sha256
                for occurrence in replay.occurrence_replays
                if occurrence.occurrence_binding is not None
            }
        )
        == sum(
            occurrence.occurrence_binding is not None
            for occurrence in replay.occurrence_replays
        )
        for replay in fit.replays
    )

    final = v4.CHECKPOINT
    assert final.phase == "canonical-fit"
    assert final.checkpoint_sha256 == (
        "e81336c6901edac0aada25c4ad89857686008fb276b94f705944ff9962a67d69"
    )
    assert final.base_checkpoint_sha256 == v4.BASE_CHECKPOINT.checkpoint_sha256
    assert final.canonical_fit_teacher_identity_count == 24
    assert final.heldout_teacher_identity_count == 0
    assert final.excitatory_nonzero_ticks == 25_382
    assert final.inhibitory_nonzero_ticks == 117_607
    assert final.excitatory_max_tick == final.inhibitory_max_tick == 4
    assert np.array_equal(
        v4.BASE_WORKSPACE._excitatory[:, : v4.SELF_INPUT_CELL_COUNT],
        v4.WORKSPACE._excitatory[:, : v4.SELF_INPUT_CELL_COUNT],
    )
    assert np.array_equal(
        v4.BASE_WORKSPACE._inhibitory[:, : v4.SELF_INPUT_CELL_COUNT],
        v4.WORKSPACE._inhibitory[:, : v4.SELF_INPUT_CELL_COUNT],
    )
    checkpoint_before = v4.WORKSPACE.checkpoint_sha256
    assert v4.RESULT.root_record() == {
        **v4.RESULT.root_record(include_result_sha256=False),
        "result_sha256": EXPECTED_RESULT_SHA256,
    }
    assert v4.WORKSPACE.checkpoint_sha256 == checkpoint_before


def test_claim_boundary_work_ledgers_and_forbidden_surfaces_are_exact() -> None:
    surface = v4.SURFACE_BOUNDARY_RECORD
    assert surface["relay_schema_count"] == 7
    assert surface["atom_cell_count"] == 103
    assert surface["transition_cell_count"] == 24
    assert surface["self_posed_learning_pass_count"] == 1
    assert surface["canonical_fit_learning_pass_count"] == 1
    assert surface["canonical_fit_teacher_identity_count"] == 24
    assert surface["canonical_fit_target_read_count"] == 24
    assert surface["canonical_fit_replay_attempt_count"] == 1
    assert surface["canonical_fit_replay_identity_count"] == 24
    assert surface["occurrence_integer_handles_are_serialization_only"] is True
    assert surface["ephemeral_parent_child_bindings_are_distributed_assemblies"] is True
    assert surface["one_repeated_masked_wta_per_occurrence_branch_plus_stop"] is True
    assert surface["ordered_publication_preserved"] is True
    assert surface["checkpoint_frozen_before_and_after_read_only_replay"] is True
    assert surface["heldout_surface_opened"] is False
    forbidden = (
        "heldout_teacher_identity_count",
        "heldout_target_read_count",
        "heldout_input_trace_count",
        "heldout_replay_identity_count",
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
        "path_hash_neural_input_count",
        "absolute_depth_neural_input_count",
        "whole_target_head_count",
        "host_union_count",
        "host_deduplication_count",
        "host_filtering_count",
        "host_reordering_count",
        "host_ranking_count",
        "candidate_program_count",
        "proof_count",
    )
    assert all(surface[key] == 0 for key in forbidden)
    with pytest.raises(TypeError):
        surface["heldout_target_read_count"] = 1  # type: ignore[index]

    assert v4.RESULT.self_training_work.total == 366_102
    assert v4.RESULT.fit_training_work.total == 348_615
    assert v4.RESULT.total_work.total == v4.RESULT.physical_work_total == 27_210_804
    assert v4.RESULT.total_work.sha256 == (
        "9cfad5b66ac2b8ff34004a518930c443f81b2023f7c6659321dbdc57d7391915"
    )
    assert v4.SURFACE_BOUNDARY_SHA256 == (
        "ff5e224a736940f9b2c93e93d6339b3802c39607c29ea25519b80c827458fddd"
    )
    assert v4.RESULT.result_sha256 == EXPECTED_RESULT_SHA256
    assert v4.RESULT.result_sha256 == v4._namespaced_sha256(
        v4._RESULT_NAMESPACE,
        v4.RESULT.root_record(include_result_sha256=False),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v4.RESULT.status = v4.STOP_FIT  # type: ignore[misc]

    topology = protocol_v4.TOPOLOGY_CONTRACT
    assert topology["one_repeated_masked_wta_per_edge_occurrence_then_stop"] is True
    assert topology["same_edge_type_may_win_more_than_once_for_distinct_occurrences"] is True
    assert topology["recursive_add_poison"]["targets_must_differ"] is True
