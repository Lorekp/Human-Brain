from __future__ import annotations

import dataclasses
import inspect
import json
from hashlib import sha256
from pathlib import Path

import gymnasium
import minigrid
import pytest
from experiments.g0 import frozen_spatial_environment_generator_v1 as generator
from experiments.g0 import spatial_candidate_adapter_v1 as adapter

EXPECTED_DEFINITION_ROOTS = {
    "learning-memory-s11-random": (
        "4263820b550ee50291c05fa9f2340108021ce0026ce2db0798892b6eec85cec5"
    ),
    "learning-door-key-8x8": ("6eb13a04521e67c2b4eb28b0d62dfd2657eba05a16b694a8d983063c5488f074"),
    "learning-key-corridor-s3r2": (
        "4f45a4ca788ca85546b601fff38290136e8adefb725440ef8baea93d30129be3"
    ),
    "adjudicative-memory-s13-random": (
        "38ce9df50cb82561e3d5504489931244874b0c15773759446421defb0a4efd6d"
    ),
    "adjudicative-door-key-8x8": (
        "2afe1f40160574d6f7174890bf77de652982a69b77879cb7bc4bb1e9a1ea6cd2"
    ),
    "adjudicative-key-corridor-s3r2": (
        "be6250319d329f9d696a072dcbce3e6be11c1129dbd1b357b8e5fdf3f48a480b"
    ),
    "adjudicative-cue-key-door-composition": (
        "a325bc64ec01a36eeb994cc6a8b76ca2db2bee61457ab6f6ae68fbbeb91e2031"
    ),
}

EXPECTED_TRANSCRIPT_ROOTS = {
    "learning-memory-s11-random": (
        "9e56e4f58d80fa40eca3fc01f73a19fb583d497c290ffd4d2612576f212d3aee"
    ),
    "learning-door-key-8x8": ("1dc10aff2248bd5323c578ba3e7f8a7e7b4b7be16400491407ac8aff60bc7c88"),
    "learning-key-corridor-s3r2": (
        "4c323ef848fc6ce7297c1946c8e4d4c63a599caabddf63ec256941da8ee1ae81"
    ),
    "adjudicative-memory-s13-random": (
        "b1bcef0fc4dcb576fc7169c259581d4ed36ec8394960b6099c145a591e22ec91"
    ),
    "adjudicative-door-key-8x8": (
        "f6c2d893d3344a3a84f024ce74242a02576d7ca0c47eef48c28908c0b8368f9c"
    ),
    "adjudicative-key-corridor-s3r2": (
        "2fa862924d2b6f6b411f68b55c6a649a66139460896844adcf037371b9652b02"
    ),
    "adjudicative-cue-key-door-composition": (
        "59ee3d38b1278b3ae89e076097884947e1fbe79987e4cf17cc7baae32e653882"
    ),
}

EXPECTED_LAYOUT_ROOTS = {
    "learning-memory-s11-random": (
        "9fdcaff10ff94735b9a49ff398cc4ead73c4313b3e9ed434eeab50737a29bee6"
    ),
    "learning-door-key-8x8": ("f6aa755d531d2046f85b41e9c723a2caf4445de284f5f316275b49fd303e625f"),
    "learning-key-corridor-s3r2": (
        "a0990311fab5c2cf29cf40ee682df77502cb28301c5c2571329ecfc05ac90067"
    ),
    "adjudicative-memory-s13-random": (
        "9c162b1906232fdc0c2efa2fabdac2685995028e6277cca432e3bd55cb65a6d2"
    ),
    "adjudicative-door-key-8x8": (
        "0a472f450863870931cdda60689eb8c425dc96337b4fca93ef8e59eb00c5a05e"
    ),
    "adjudicative-key-corridor-s3r2": (
        "4453658bd19b9f3204024f5f1af368b3c22833aa91d36f8e2180ce3ca795cd4e"
    ),
    "adjudicative-cue-key-door-composition": (
        "43890e12e3b72c1dbc5b7fa21fbcbaca5895173d8bc811943b244fcc94590ce5"
    ),
}


def test_frozen_definitions_bind_minigrid_preregistration_and_sources() -> None:
    assert minigrid.__version__ == generator.MINIGRID_VERSION == "3.1.0"
    assert gymnasium.__version__ == generator.GYMNASIUM_VERSION == "1.3.0"
    assert generator.EXPECTED_PREREGISTRATION_SHA256 == (
        "955828b48591f198047d5ca4966885e253763d6ea353351585edb14390320468"
    )
    assert generator.EXPECTED_SPATIAL_CONFIGURATION_SHA256 == (
        "8705597cd610e00390a73bcd926fc8b3bcbfbd8be07b08646214972ad8402100"
    )
    for relative_path, expected in generator._PINNED_PROJECT_SOURCES.items():
        assert sha256((generator._ROOT / relative_path).read_bytes()).hexdigest() == (expected)
    for name, module in generator._MINIGRID_MODULES.items():
        assert (
            sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
            == (generator.MINIGRID_SOURCE_SHA256S[name])
        )

    assert tuple(row.definition_id for row in generator.LEARNING_DEFINITIONS) == (
        "learning-memory-s11-random",
        "learning-door-key-8x8",
        "learning-key-corridor-s3r2",
    )
    assert tuple(row.definition_id for row in generator.ADJUDICATION_DEFINITIONS) == (
        "adjudicative-memory-s13-random",
        "adjudicative-door-key-8x8",
        "adjudicative-key-corridor-s3r2",
        "adjudicative-cue-key-door-composition",
    )
    assert sum(row.preregistered_layout_count for row in generator.LEARNING_DEFINITIONS) == 512
    assert sum(row.preregistered_layout_count for row in generator.ADJUDICATION_DEFINITIONS) == 128
    assert {
        row.definition_id: row.definition_sha256 for row in generator.ALL_DEFINITIONS
    } == EXPECTED_DEFINITION_ROOTS
    assert tuple((row.factory, row.max_steps) for row in generator.ALL_DEFINITIONS) == (
        ("MemoryEnv", 484),
        ("DoorKeyEnv", 256),
        ("KeyCorridorEnv", 196),
        ("MemoryEnv", 676),
        ("DoorKeyEnv", 256),
        ("KeyCorridorEnv", 196),
        ("CueKeyDoorEnv", 2916),
    )
    assert dict(generator.LEARNING_DEFINITIONS[0].configuration) == {
        "size": 11,
        "random_length": True,
        "max_steps": 484,
    }
    assert dict(generator.ADJUDICATION_DEFINITIONS[0].configuration) == {
        "size": 13,
        "random_length": True,
        "max_steps": 676,
    }
    assert dict(generator.LEARNING_DEFINITIONS[2].configuration) == {
        "room_size": 3,
        "num_rows": 2,
        "max_steps": 196,
    }
    assert generator.DEPENDENCY_SHA256 == (
        "644bb2e9296b3f79618ae786eed55a98aa9cf1a700d8ba0528a599d6aa56414e"
    )
    assert generator.SOURCE_CONFIGURATION_SHA256 == (
        "3425f82374ff39ac23fbf72b7fee93d63926b2be84d07ae44dc1416ee8048b28"
    )
    assert generator.GENERATOR_ROOT_SHA256 == (
        "7099bba537b1f08a04d0eb90ec5f8b3e8edcd3acde7ff07fc4d018348eb377e2"
    )
    assert generator.CUSTOM_LAYOUT_COUNT == 64 == 2**generator.CUSTOM_LAYOUT_CODE_BITS
    assert len(generator.CUSTOM_LAYOUT_MARKER_POSITIONS) == 6
    assert set(generator.CUSTOM_LAYOUT_MARKER_COLORS).isdisjoint({"red", "blue"})
    assert len(set(generator.CUSTOM_LAYOUT_MARKER_PATTERNS)) == 64
    assert generator.CUSTOM_LAYOUT_FAMILY_RECORD["cue_answer_encoded"] is False
    assert generator.CUSTOM_LAYOUT_FAMILY_RECORD["layout_index_candidate_field_count"] == 0
    assert generator.CUSTOM_LAYOUT_FAMILY_SHA256 == (
        "b38e92a0324ebbf3ed9b7ad6d36b53bb0254fcc04fa983ee005d34493e1b4619"
    )


def test_standard_quarantined_transcripts_are_deterministic_raw_and_distinct() -> None:
    validations = {}
    for definition in generator.ALL_DEFINITIONS:
        seed = generator.QUARANTINED_VALIDATION_SEEDS[definition.definition_id]
        first = generator.validate_quarantined_environment(definition.definition_id, seed)
        second = generator.validate_quarantined_environment(definition.definition_id, seed)
        assert first == second
        assert first.definition_sha256 == definition.definition_sha256
        assert first.fixture_seed == seed
        assert first.step_count == len(generator.DEFAULT_VALIDATION_ACTIONS) == 10
        assert first.raw_action_ids == generator.DEFAULT_VALIDATION_ACTIONS
        assert first.candidate_transcript_sha256 == sha256(first.candidate_transcript).hexdigest()
        assert (
            first.candidate_transcript_sha256 == EXPECTED_TRANSCRIPT_ROOTS[definition.definition_id]
        )
        assert first.layout_identity_sha256 == EXPECTED_LAYOUT_ROOTS[definition.definition_id]
        assert first.renamed_layout_identity_sha256 == first.layout_identity_sha256
        assert first.total_reward == 0.0
        assert not first.terminated and not first.truncated
        decoded = adapter.decode_transcript(first.candidate_transcript)
        assert tuple(step.action_id for step in decoded) == generator.DEFAULT_VALIDATION_ACTIONS
        assert all(tuple(step.record()) == adapter.CANDIDATE_FIELDS for step in decoded)
        assert all(len(step.image) == adapter.IMAGE_BYTE_COUNT for step in decoded)
        assert all(step.direction in adapter.DIRECTION_IDS for step in decoded)
        validations[definition.definition_id] = first

    assert len({row.layout_identity_sha256 for row in validations.values()}) == 7
    assert len({row.candidate_transcript_sha256 for row in validations.values()}) == 7
    assert b"mission" not in next(iter(validations.values())).candidate_transcript
    assert all(
        field.encode("ascii") not in next(iter(validations.values())).candidate_transcript
        for field in adapter.SPATIAL_FORBIDDEN_FIELDS
    )

    definition_id = "learning-door-key-8x8"
    seed = generator.QUARANTINED_VALIDATION_SEEDS[definition_id]
    ordinary = validations[definition_id]
    renamed = generator.validate_quarantined_environment(
        definition_id,
        seed,
        rename_name="cyclic-object-color",
    )
    repeated = generator.validate_quarantined_environment(
        definition_id,
        seed,
        rename_name="cyclic-object-color",
    )
    assert renamed == repeated
    assert renamed.candidate_transcript_sha256 == (
        "e586a8988ed2cdbd99d0d831c69a1211abdbff594b7407a8320186af8aad11c6"
    )
    assert renamed.renamed_layout_identity_sha256 == (
        "3be42a2c2562de42f6220d9c0ea73bb4f66d595421b6d122729b2cac34cabdb4"
    )
    assert renamed.candidate_transcript_sha256 != ordinary.candidate_transcript_sha256
    assert renamed.layout_identity_sha256 == ordinary.layout_identity_sha256
    assert renamed.renamed_layout_identity_sha256 != ordinary.layout_identity_sha256
    ordinary_steps = adapter.decode_transcript(ordinary.candidate_transcript)
    renamed_steps = adapter.decode_transcript(renamed.candidate_transcript)
    assert tuple(
        (row.direction, row.action_id, row.reward, row.terminated, row.truncated)
        for row in ordinary_steps
    ) == tuple(
        (row.direction, row.action_id, row.reward, row.terminated, row.truncated)
        for row in renamed_steps
    )
    assert any(
        left.image != right.image for left, right in zip(ordinary_steps, renamed_steps, strict=True)
    )
    assert "object_permutation" not in renamed.root_record()
    assert "color_permutation" not in renamed.root_record()
    assert generator.CYCLIC_TOKEN_RENAME.record()["inverse_exposed_to_candidate"] is False


def test_custom_cue_key_door_has_private_delay_choice_and_gate_audit() -> None:
    definition_id = "adjudicative-cue-key-door-composition"
    seed = generator.QUARANTINED_VALIDATION_SEEDS[definition_id]
    audit = generator.validate_quarantined_cue_key_door(seed)
    assert audit == generator.validate_quarantined_cue_key_door(seed)
    assert audit.record() == {
        "fixture_seed": 9_270_109,
        "initial_cue_visible": True,
        "cue_selects_one_of_two_keys": True,
        "cue_removed_step": 1,
        "irreversible_choice_step": 22,
        "cue_absent_actions_before_choice": 21,
        "minimum_cue_absence_required": 16,
        "unchosen_key_removed": True,
        "matching_key_unlocked_door": True,
        "nonmatching_key_left_door_locked": True,
        "locked_door_gates_goal": True,
        "matching_rollout_terminated_at_goal": True,
        "nonmatching_rollout_terminated": False,
        "matching_candidate_transcript_sha256": (
            "8adb41159c49e85483776df38f31ea71a09a021a12e13af6f42b0c401d9a6d11"
        ),
        "nonmatching_candidate_transcript_sha256": (
            "52b64f03202f49d8c710f79f33fb91d0b9f9d4d75092833c22a3f6a43f6788de"
        ),
        "candidate_field_audit_sha256": (
            "c060084b0ebed61e173fe12693aff20e8c81f2f1685f5eb1c9e10df06e914c48"
        ),
        "audit_sha256": ("402be18eccf2be0b626ecd896a067e935545db6975d35e4c9be22af7690d97f8"),
    }
    assert audit.cue_absent_actions_before_choice >= (audit.minimum_cue_absence_required)
    audit_payload = json.dumps(audit.record(), sort_keys=True)
    assert '"red"' not in audit_payload and '"blue"' not in audit_payload
    assert "red" not in definition_id and "blue" not in definition_id
    assert generator.SURFACE_BOUNDARY_RECORD["cue_or_door_color_candidate_field_count"] == 0
    assert generator.SURFACE_BOUNDARY_RECORD["cue_or_door_color_in_mission"] is False
    assert generator.SURFACE_BOUNDARY_RECORD["cue_or_door_color_in_definition_id"] is False
    assert generator.SURFACE_BOUNDARY_RECORD["cue_or_door_color_in_audit_record"] is False
    assert generator.SURFACE_BOUNDARY_RECORD["candidate_payload"] == (
        "adapter-transcript-bytes-only"
    )


def test_preregistered_surfaces_and_seeds_stop_before_instantiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert generator.FROZEN_COHORT_SEEDS == (470_100, 470_101, 470_102, 470_103)
    assert set(generator.QUARANTINED_VALIDATION_SEEDS.values()).isdisjoint(
        generator.FROZEN_COHORT_SEEDS
    )
    assert dict(generator.SURFACE_BOUNDARY_RECORD) == {
        "common_architecture_binding_count": 0,
        "frozen_cohort_execution_count": 0,
        "frozen_layout_generation_count": 0,
        "learner_execution_count": 0,
        "evaluator_execution_count": 0,
        "outcome_count": 0,
        "dev_input_count": 0,
        "m0_input_count": 0,
        "protected_input_count": 0,
        "preregistered_seed_execution_count": 0,
        "quarantined_source_validation_authorized": True,
        "frozen_cohort_generation_authorized": False,
        "nine_artifact_binding_required": True,
        "candidate_fields": adapter.CANDIDATE_FIELDS,
        "mission_discarded": True,
        "full_grid_candidate_exposure_count": 0,
        "position_candidate_exposure_count": 0,
        "object_identity_candidate_exposure_count": 0,
        "path_or_solver_candidate_exposure_count": 0,
        "codebook_inverse_candidate_exposure_count": 0,
        "layout_identity_candidate_exposure_count": 0,
        "cue_or_door_color_candidate_field_count": 0,
        "cue_or_door_color_in_mission": False,
        "cue_or_door_color_in_definition_id": False,
        "cue_or_door_color_in_audit_record": False,
        "rename_forward_mapping_candidate_exposure_count": 0,
        "rename_inverse_candidate_exposure_count": 0,
        "custom_layout_index_candidate_exposure_count": 0,
        "candidate_payload": "adapter-transcript-bytes-only",
    }

    class Explosive:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("frozen cohort arguments were inspected")

    for surface in generator.FROZEN_COHORT_SURFACES:
        with pytest.raises(
            generator.G0CommonArchitectureBindingRequired,
            match=r"^STOP_G0_COMMON_ARCHITECTURE_BINDING_REQUIRED$",
        ):
            generator.generate_frozen_spatial_cohort(
                Explosive(),
                surface=surface,
                architecture_binding=Explosive(),
            )

    monkeypatch.setattr(
        generator,
        "_make_environment",
        lambda _definition: (_ for _ in ()).throw(
            AssertionError("reserved seed reached MiniGrid instantiation")
        ),
    )
    definition_id = generator.ALL_DEFINITIONS[0].definition_id
    for seed in generator.FROZEN_COHORT_SEEDS:
        with pytest.raises(
            generator.QuarantinedSpatialValidationError,
            match=r"^STOP_G0_SPATIAL_VALIDATION_NOT_DECLARED$",
        ):
            generator.validate_quarantined_environment(definition_id, seed)

    adapter_source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "frozen_spatial_environment_generator_v1" not in adapter_source
    assert "spatial_fixture_generator_v1" not in adapter_source
    with pytest.raises(dataclasses.FrozenInstanceError):
        generator.ALL_DEFINITIONS[0].max_steps = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        generator.SURFACE_BOUNDARY_RECORD["outcome_count"] = 1  # type: ignore[index]
