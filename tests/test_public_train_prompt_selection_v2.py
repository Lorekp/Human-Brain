from __future__ import annotations

import dataclasses
import inspect
from hashlib import sha256

import numpy as np
import pytest
from experiments.r3_pilot import public_train_prompt_selection_freeze_v1_protocol as selection_v1
from experiments.r3_pilot import public_train_prompt_selection_v2 as v2
from experiments.r3_pilot import public_train_structural_annotation_capsule_v1 as annotations_v1

EXPECTED_ROOTS = {
    "dependency_sha256": "b4ef7099d2fe9418d0ab0d850a805ff61600917647c88a59c9db643bd63dd3f9",
    "trace_registry_sha256": ("417cb58925efad9971f92ab9b613808a3bdba180698919563ff0b8c465930e29"),
    "label_registry_sha256": ("cc9895467f38c07b851e5ac0f4c48d862e5356d21bb223d73dc0ff30a3ee1b6b"),
    "branch_registry_sha256": ("1d063d96983dbce45cf1bdc25c8bbcf601b38643d8a5d40b016ef633c310442b"),
    "checkpoint_sha256": ("09bed7cad0b95735ea64ee0a84b9b4adfe6b586cc32b3a5b51f00a81edafa51f"),
    "fit_result_sha256": "960bc8dcfdad8d2d9e05c979cfc567d673b87016920e21edaee55d4ba26d5d06",
    "heldout_result_sha256": ("52f8afeee83476068644e8a846cb0e37916c931c823c06270482b23e140b2253"),
    "control_panel_sha256": ("9400102455a223714a46d3d1ed0c78805a8d2428860eed34ba53a62f5899c314"),
    "surface_boundary_sha256": ("a8c6ec6ddcbd1bd4d6ff57523adf9b442fb5553a1072818a182800d627d4d928"),
    "result_sha256": "72228a1177bfcaf162533f5e8d91a46ec73e0bc180d5b8ed869ff49162a74c2b",
}


def test_only_authenticated_physical_prompt_events_drive_fit_only_label_cells() -> None:
    assert selection_v1.PROTOCOL.protocol_sha256 == v2.EXPECTED_SELECTION_V1_SHA256
    assert annotations_v1.CAPSULE.capsule_sha256 == (v2.EXPECTED_ANNOTATION_CAPSULE_SHA256)
    for relative_path, expected_sha in v2._PINNED_SOURCE_SHA256S.items():
        assert sha256((v2._ROOT / relative_path).read_bytes()).hexdigest() == expected_sha
    assert v2.DEPENDENCY_RECORD["teacher_schedule"] == (
        "exactly-one-ordered-pass-over-24-fit-causal-identities"
    )
    assert v2.DEPENDENCY_RECORD["heldout_teacher_identity_count"] == 0
    assert v2.DEPENDENCY_RECORD["causal_key_sha256_used_as_neural_input"] is False
    assert v2.DEPENDENCY_RECORD["prompt_input_sha256_used_as_neural_input"] is False
    assert v2.DEPENDENCY_RECORD["full_signature_used_as_neural_input"] is False
    assert v2.DEPENDENCY_RECORD["split_used_as_neural_input"] is False

    fit_keys = {
        row.causal_key_sha256
        for row in selection_v1.PROTOCOL.canonical_inputs
        if row.split == v2.FIT
    }
    fit_label_roots = {
        label.sha256
        for annotation in annotations_v1.CAPSULE.annotations
        if annotation.causal_key_sha256 in fit_keys
        for label in annotation.relay_labels
    }
    assert len(fit_keys) == 24
    assert len(fit_label_roots) == len(v2.LABEL_CELLS) == 71
    assert {cell.label.sha256 for cell in v2.LABEL_CELLS} == fit_label_roots
    assert len(v2.BRANCHES) == 41
    assert all(trace.split == v2.FIT for trace in v2.FIT_TRACES)
    assert all(trace.split == v2.HELDOUT for trace in v2.HELDOUT_TRACES)
    assert all(
        len(trace.active_cells) == len(set(trace.active_cells)) == 12 for trace in v2.FIT_TRACES
    )
    assert v2.CHECKPOINT.fit_teacher_identity_count == 24
    assert v2.CHECKPOINT.heldout_teacher_identity_count == 0

    surface = dict(v2.SURFACE_BOUNDARY_RECORD)
    assert surface["neural_input_sources"] == (
        "ordered-physical-span-events",
        "ordered-typed-literal-cell-events",
        "ordered-name-free-signature-role-type-events",
    )
    assert surface["neural_output_channels"] == tuple(
        channel.channel for channel in selection_v1.RELAY_TARGET_VOCABULARY
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
        "host_lookup_selection_count",
        "host_union_selection_count",
        "host_ranking_count",
        "candidate_program_count",
        "proof_count",
    )
    assert all(surface[field] == 0 for field in zero_fields)


def test_fit_stop_and_all_heldout_replays_have_exact_channel_accounting() -> None:
    result = v2.RESULT
    assert result.status == v2.STATUS == v2.STOP_FIT
    assert (
        result.fit_result.identity_count,
        result.fit_result.exact_identity_count,
        result.fit_result.mismatch_identity_count,
    ) == (24, 16, 8)
    assert (
        result.heldout_result.identity_count,
        result.heldout_result.exact_identity_count,
        result.heldout_result.mismatch_identity_count,
    ) == (18, 0, 18)
    assert result.fit_result.checkpoint_authentication_count == 2
    assert result.heldout_result.checkpoint_authentication_count == 2
    assert all(replay.split == v2.HELDOUT for replay in result.heldout_result.replays)
    assert all(
        replay.causal_key_sha256 == trace.causal_key_sha256
        for replay, trace in zip(
            result.heldout_result.replays,
            v2.HELDOUT_TRACES,
            strict=True,
        )
    )

    assert tuple(row.record() for row in result.fit_result.channel_admissions) == (
        {
            "channel": "root-production-family",
            "exact_identity_count": 16,
            "mismatch_identity_count": 8,
            "expected_label_count": 24,
            "emitted_label_count": 16,
        },
        {
            "channel": "ordered-parent-child-edge",
            "exact_identity_count": 16,
            "mismatch_identity_count": 8,
            "expected_label_count": 117,
            "emitted_label_count": 79,
        },
        {
            "channel": "argument-operand-orientation",
            "exact_identity_count": 16,
            "mismatch_identity_count": 8,
            "expected_label_count": 98,
            "emitted_label_count": 66,
        },
        {
            "channel": "nesting-relation",
            "exact_identity_count": 19,
            "mismatch_identity_count": 5,
            "expected_label_count": 44,
            "emitted_label_count": 30,
        },
        {
            "channel": "parameter-role-binding",
            "exact_identity_count": 19,
            "mismatch_identity_count": 5,
            "expected_label_count": 24,
            "emitted_label_count": 19,
        },
        {
            "channel": "literal-slot-binding",
            "exact_identity_count": 20,
            "mismatch_identity_count": 4,
            "expected_label_count": 35,
            "emitted_label_count": 28,
        },
        {
            "channel": "literal-value-copy",
            "exact_identity_count": 20,
            "mismatch_identity_count": 4,
            "expected_label_count": 35,
            "emitted_label_count": 28,
        },
        {
            "channel": "terminal-root-eclass-agreement",
            "exact_identity_count": 24,
            "mismatch_identity_count": 0,
            "expected_label_count": 24,
            "emitted_label_count": 24,
        },
    )
    assert tuple(row.record() for row in result.heldout_result.channel_admissions) == (
        {
            "channel": "root-production-family",
            "exact_identity_count": 0,
            "mismatch_identity_count": 18,
            "expected_label_count": 18,
            "emitted_label_count": 0,
        },
        {
            "channel": "ordered-parent-child-edge",
            "exact_identity_count": 0,
            "mismatch_identity_count": 18,
            "expected_label_count": 97,
            "emitted_label_count": 0,
        },
        {
            "channel": "argument-operand-orientation",
            "exact_identity_count": 0,
            "mismatch_identity_count": 18,
            "expected_label_count": 78,
            "emitted_label_count": 0,
        },
        {
            "channel": "nesting-relation",
            "exact_identity_count": 2,
            "mismatch_identity_count": 16,
            "expected_label_count": 40,
            "emitted_label_count": 0,
        },
        {
            "channel": "parameter-role-binding",
            "exact_identity_count": 0,
            "mismatch_identity_count": 18,
            "expected_label_count": 18,
            "emitted_label_count": 0,
        },
        {
            "channel": "literal-slot-binding",
            "exact_identity_count": 1,
            "mismatch_identity_count": 17,
            "expected_label_count": 26,
            "emitted_label_count": 0,
        },
        {
            "channel": "literal-value-copy",
            "exact_identity_count": 1,
            "mismatch_identity_count": 17,
            "expected_label_count": 26,
            "emitted_label_count": 0,
        },
        {
            "channel": "terminal-root-eclass-agreement",
            "exact_identity_count": 0,
            "mismatch_identity_count": 18,
            "expected_label_count": 18,
            "emitted_label_count": 0,
        },
    )


def test_checkpoint_is_immutable_and_all_four_causal_controls_falsify() -> None:
    checkpoint = v2.CHECKPOINT
    assert checkpoint.record() == {
        "candidate_label_cell_count": 71,
        "branch_count": 41,
        "array_shape": (71, 6144),
        "array_dtype": "|u1",
        "array_bytes": 872448,
        "tick_cap": 4,
        "excitatory_nonzero_ticks": 4684,
        "inhibitory_nonzero_ticks": 15173,
        "fit_teacher_identity_count": 24,
        "heldout_teacher_identity_count": 0,
        "checkpoint_sha256": EXPECTED_ROOTS["checkpoint_sha256"],
    }
    before = v2.WORKSPACE.checkpoint_sha256
    excitatory = v2.WORKSPACE.excitatory_ticks
    inhibitory = v2.WORKSPACE.inhibitory_ticks
    assert excitatory.flags.writeable is False
    assert inhibitory.flags.writeable is False
    with pytest.raises(ValueError):
        excitatory[0, 0] = 4
    assert v2.WORKSPACE.checkpoint_sha256 == before

    clone = v2.FrozenPromptSelectionWorkspace(
        label_cells=v2.WORKSPACE.label_cells,
        branches=v2.WORKSPACE.branches,
        excitatory=v2.WORKSPACE.excitatory_ticks,
        inhibitory=v2.WORKSPACE.inhibitory_ticks,
        metadata=v2.WORKSPACE._metadata,
    )
    forged = clone.excitatory_ticks.copy()
    forged[0, 0] = np.uint8((int(forged[0, 0]) + 1) % 5)
    forged.setflags(write=False)
    object.__setattr__(clone, "_excitatory", forged)
    with pytest.raises(RuntimeError, match="checkpoint authentication failed"):
        clone.authenticate()

    assert tuple(
        (
            control.name,
            control.exact_identity_count,
            control.mismatch_identity_count,
            control.stop_decision_count,
            control.collision_decision_count,
        )
        for control in v2.CONTROLS
    ) == (
        ("untrained", 0, 24, 984, 0),
        ("label-off", 0, 24, 984, 0),
        ("shuffled-target-set", 0, 24, 669, 46),
        ("reversed-span-event-order", 0, 24, 984, 0),
    )
    assert dict(v2.CONTROLS[1].work.events).get("teacher-cell-spike", 0) == 0
    assert dict(v2.CONTROLS[2].work.events)["teacher-cell-spike"] == 984


def test_exact_physical_work_roots_and_immutable_result_surface() -> None:
    result = v2.RESULT
    assert result.trace_work.events == (
        ("physical-causal-event", 101469),
        ("recurrent-cell-transition", 1217628),
        ("trace-cell-emission", 504),
    )
    assert result.training_work.events == (
        ("excitatory-tick-commit", 5061),
        ("glif-principal-cell-step", 1612800),
        ("inhibitory-tick-commit", 15552),
        ("masked-wta-decision", 984),
        ("principal-current-write", 2688),
        ("synaptic-tick-read", 40896),
        ("teacher-cell-spike", 984),
    )
    assert (
        result.trace_work.total,
        result.training_work.total,
        result.fit_result.work.total,
        result.heldout_result.work.total,
        tuple(control.work.total for control in result.controls),
        result.physical_work_total,
    ) == (
        1_319_601,
        1_678_965,
        1_660_053,
        1_244_625,
        (1_659_473, 3_316_841, 3_339_142, 2_375_268),
        16_593_968,
    )
    actual_roots = {
        field.name: getattr(result, field.name)
        for field in dataclasses.fields(result)
        if field.name.endswith("sha256")
    }
    assert actual_roots == EXPECTED_ROOTS
    assert result.dependency_sha256 == v2._namespaced_sha256(
        v2._DEPENDENCY_NAMESPACE,
        dict(v2.DEPENDENCY_RECORD),
    )
    assert result.surface_boundary_sha256 == v2._namespaced_sha256(
        v2._SURFACE_NAMESPACE,
        dict(v2.SURFACE_BOUNDARY_RECORD),
    )
    assert result.result_sha256 == v2._namespaced_sha256(
        v2._RESULT_NAMESPACE,
        result.root_record(include_result_sha256=False),
    )
    public_functions = {
        name
        for name, value in inspect.getmembers(v2, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == v2.__name__
    }
    assert public_functions == set()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = v2.PASS  # type: ignore[misc]
