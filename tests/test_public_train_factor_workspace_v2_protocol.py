from __future__ import annotations

import dataclasses
import inspect
from collections import Counter, defaultdict
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

import pytest
from experiments.r3_pilot import public_train_factor_workspace_v2_protocol as v2
from experiments.r3_pilot import public_train_prompt_factor_manifest_v1 as manifest_v1

EXPECTED_ROOTS = {
    "dependency_sha256": "0891d4af98e4faf3e0742686fb6c7f2e528c31a66913d8ed73940271e188e50d",
    "canonical_inventory_sha256": (
        "3c6419c11db94942db8e711d9b6332e3214c168746ea7bdb184c208de17e036b"
    ),
    "canonical_split_sha256": "774d5ca350a45561c833218729bb690134da45e6e370028d454102c52d7c3ddf",
    "active_factor_coverage_sha256": (
        "42dc2be890d2e7882efc6d88836e6f10f6f981dee6a513ee91286c56629d9d32"
    ),
    "span_feature_profiles_sha256": (
        "996f398d131a8bbeaa0a96547327ff5b06133a7ab1034816a13e46eba192b21e"
    ),
    "counterfactual_curriculum_sha256": (
        "9c0525268b787f207bf0450c1e9d8d7b47a78c85ea134b9a0a934ed840213f5a"
    ),
    "rare_factor_support_sha256": (
        "1072c6a45dca28777c1d714e02eb66eb42259bffc5e54d139842270681cd4ca0"
    ),
    "masked_wta_topology_sha256": (
        "8c27c1f968fb771bc22cc6b12dd180512e944a27a37c3b8348434093a5417e93"
    ),
    "masked_wta_controls_sha256": (
        "a1fc8de9d09d20620c65392d135295d3f97f7b401e55b8e4ca5fee1c4d5da295"
    ),
    "compositor_topology_sha256": (
        "248b6f22c48a45bdef2beb119a8e02bcf54de44c4a309d2ce99564d863364b8e"
    ),
    "compositor_controls_sha256": (
        "af17b538453d72327c9c36c4f091b48c7495cc907095322aa28fe01b6a9084f1"
    ),
    "electrical_preflight_contract_sha256": (
        "d2ad0d0a9eea6adec3b114e28207a7364f6c19eb098adc4f7aa5d6c4393ae87f"
    ),
    "surface_boundary_sha256": "04cf59c060292f7cd18adca4b77b4e6f40ad8e644b4aee569f502ceb62599e24",
    "protocol_sha256": "54aca3e9f9f7ae0bf1b1743510938522400305c64530db9b3f1467b241ee3171",
}


EXPECTED_HELDOUT = (
    "0eb123c8a8a5d799a359dd1a5cd8a0f36f8a30cffbccaf2b1c9fa676035a3a1c",
    "17bffc18f60edf862f6ac6b141fbdbc8522b30d3b78a699fe229f6d13fafca48",
    "2b98f16e11011af3a4a43e467fd032b38191eed195ee4dc397e462848b12686c",
    "449c4e8e502b9e0eb9454c4f85d6618cdcd768ad03976d3d28486727b05cb596",
    "86a085f40de0125e2e9246fc1f0e50ca1dfcc15cabbaa2cf05dbf2eb2f91348b",
    "88a0aab410711043014c050ec9a4f1fb31628d5d7838a81c5169dbb2bf307bbd",
    "915f0c59431bced6f285bd3e276776652d8fa375f6ed6c3b5fd30a4a6753b6ff",
    "c11291cc43f2cd38d2491b6ec4a4da3f45fe97f8b792fe01029d15835002b0e8",
    "cf798e16f3e6e7ef6dc6466c3e915cb6587e4c1c04c695800050a7ac3d8dffed",
    "e673f87f51571c9b3b11d4e81aac52a993e11a6ca9d7155e57e43b129c41c1f5",
    "fccd2b5c2b1538c0f6374c4d26faff3f75722773fd9ad9ebd7922a5f3381e0dd",
    "0b1b35530cf1cd008dbf1f06c9b12b5bdb4f7f21116800eb41f145565e7218db",
    "1ea63f6a9e51f0ba7c53332f6bb6afa257c1e0c9e6ac8d0d7a2c6c3105a0350a",
    "3c6d723ca3674df3a00a2bb3209ac91c532d6556154a302d133dd790d58b3417",
    "44d4833bf08a64132d580a3b4177eb7a519a5e332437276c4cfcaa90acdaa6c6",
    "53b970f9f2bc8cb840b6d1d13e71977c62b9cb203888374cc9fb170bac05a796",
    "78e892eb3ea2d16eca25293096ab61793a051cbbc8d30a72c250e5b2b753f0dc",
    "819f05c7981087f36c02d8fb77aaa262f814bbf3f3f054ffed470053fa081613",
)


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


def test_exact_42_identity_inventory_multiplicity_coverage_and_24_18_split() -> None:
    protocol = v2.PROTOCOL
    rows = protocol.canonical_identities
    assert len(rows) == v2.CANONICAL_SEMANTIC_IDENTITY_COUNT == 42
    assert len({row.prompt_input_sha256 for row in rows}) == 42
    assert sum(row.appearance_count for row in rows) == 90

    prompt_inputs = manifest_v1.load_public_train_prompt_inputs()
    manifest = manifest_v1.parse_lesson_manifest(
        manifest_v1.build_lesson_manifest_record(prompt_inputs)
    )
    manifest_v1.authenticate_lesson_targets(manifest, prompt_inputs)
    live_multiplicity = Counter(row.sha256 for row in prompt_inputs)
    live_targets: dict[str, tuple[str, ...]] = {}
    for lesson in manifest.lessons:
        factors = tuple(target.factor for target in lesson.targets)
        previous = live_targets.setdefault(lesson.prompt_input_sha256, factors)
        assert previous == factors

    frozen_by_identity = {row.prompt_input_sha256: row for row in rows}
    assert set(frozen_by_identity) == set(live_multiplicity) == set(live_targets)
    assert {identity: row.appearance_count for identity, row in frozen_by_identity.items()} == dict(
        live_multiplicity
    )
    assert {
        identity: row.target_factors for identity, row in frozen_by_identity.items()
    } == live_targets

    assert protocol.canonical_heldout_prompt_input_sha256s == EXPECTED_HELDOUT
    assert len(protocol.canonical_fit_prompt_input_sha256s) == 24
    assert set(protocol.canonical_fit_prompt_input_sha256s).isdisjoint(EXPECTED_HELDOUT)
    assert set(protocol.canonical_fit_prompt_input_sha256s) | set(EXPECTED_HELDOUT) == set(
        frozen_by_identity
    )

    heldout_coverage = Counter(
        factor
        for identity in EXPECTED_HELDOUT
        for factor in set(frozen_by_identity[identity].target_factors)
    )
    assert heldout_coverage == Counter(
        {
            "absolute": 1,
            "add": 6,
            "class:ascii-decimal-digits": 2,
            "class:ascii-lowercase": 2,
            "class:english-vowels": 2,
            "count": 4,
            "filter": 6,
            "less-than": 4,
            "map": 7,
            "maximum": 5,
            "minimum": 5,
            "reverse": 3,
            "sum": 2,
        }
    )
    assert v2.CANONICAL_SPLIT_RECORD["identity_not_appearance_split"] is True
    assert v2.CANONICAL_SPLIT_RECORD["multiplicity_is_audit_only"] is True
    assert v2.CANONICAL_SPLIT_RECORD["final_refit_identity_order"] == tuple(
        row.prompt_input_sha256 for row in rows
    )


def test_78_single_target_counterfactuals_are_matched_balanced_and_supported() -> None:
    protocol = v2.PROTOCOL
    templates = protocol.rooted_templates
    groups = protocol.counterfactual_groups
    lessons = protocol.counterfactual_lessons
    fit = tuple(lesson for lesson in lessons if lesson.split == v2.FIT)
    heldout = tuple(lesson for lesson in lessons if lesson.split == v2.HELDOUT)

    assert len(templates) == v2.ROOTED_TEMPLATE_COUNT == 6
    assert len(groups) == v2.COUNTERFACTUAL_GROUP_COUNT == 30
    assert len(lessons) == v2.COUNTERFACTUAL_LESSON_COUNT == 78
    assert len(fit) == v2.COUNTERFACTUAL_FIT_COUNT == 52
    assert len(heldout) == v2.COUNTERFACTUAL_HELDOUT_COUNT == 26
    assert len({lesson.input_sha256 for lesson in lessons}) == 78
    assert {lesson.input_sha256 for lesson in lessons}.isdisjoint(
        row.prompt_input_sha256 for row in protocol.canonical_identities
    )
    assert Counter(lesson.factor for lesson in fit) == Counter(
        {factor: 4 for factor in v2.ACTIVE_FACTORS}
    )
    assert Counter(lesson.factor for lesson in heldout) == Counter(
        {factor: 2 for factor in v2.ACTIVE_FACTORS}
    )
    assert all(lesson.record()["teacher_target_count"] == 1 for lesson in lessons)

    for template in templates:
        assert tuple(len(group) for group in template.factor_groups) == v2.GROUP_SIZES
        assert Counter(factor for group in template.factor_groups for factor in group) == Counter(
            v2.ACTIVE_FACTORS
        )
        template_groups = tuple(
            group for group in groups if group.template_index == template.template_index
        )
        assert len(template_groups) == 5
        for group in template_groups:
            assert len(group.lessons) in (2, 3)
            assert {lesson.factor for lesson in group.lessons} == set(group.factors)
            assert len({tuple(sorted(lesson.input_slots)) for lesson in group.lessons}) == 1
            assert all(
                lesson.input_slots[lesson.cue_slot] == f"cue:{lesson.direction.lower()}:v2"
                for lesson in group.lessons
            )
            assert all(
                lesson.input_slots[lesson.selected_slot]
                == f"span:{lesson.target_span_profile_sha256}"
                for lesson in group.lessons
            )

    fit_profiles: dict[str, set[str]] = defaultdict(set)
    fit_cells: dict[str, set[int]] = defaultdict(set)
    for lesson in fit:
        fit_profiles[lesson.factor].add(lesson.target_span_identity_sha256)
        fit_cells[lesson.factor].update(lesson.target_assembly_cells)
    for lesson in heldout:
        assert lesson.target_span_identity_sha256 in fit_profiles[lesson.factor]
        assert set(lesson.target_assembly_cells) <= fit_cells[lesson.factor]

    rare_support = v2.RARE_FACTOR_SUPPORT_RECORD["rare_factor_fit_span_identities"]
    assert set(rare_support) == set(v2.RARE_FIT_FACTORS)
    assert all(len(identities) == 1 for identities in rare_support.values())
    assert all(
        sum(
            lesson.factor == factor and lesson.target_span_identity_sha256 == identities[0]
            for lesson in fit
        )
        == 4
        for factor, identities in rare_support.items()
    )
    assert v2.RARE_FACTOR_SUPPORT_RECORD["heldout_profile_identity_in_fit"] is True
    assert v2.RARE_FACTOR_SUPPORT_RECORD["heldout_active_cells_subset_of_fit"] is True


def test_capacity_four_repeated_masked_wta_and_factor_set_controls_are_frozen() -> None:
    masked = v2.MASKED_WTA_TOPOLOGY_RECORD
    assert masked["principal_factor_cells"] == 24
    assert masked["principal_stop_cells"] == 1
    assert masked["principal_count"] == 25
    assert masked["mask_storage_cells"] == 24
    assert masked["mask_readout_cells"] == 24
    assert masked["same_span_multi_label_supported"] is True
    assert masked["ordered_publication_semantics"] == "physical-audit-only"
    assert masked["full_circuit_execution_deferred"] is True

    masked_by_name = {control.name: control for control in v2.PROTOCOL.masked_wta_controls}
    assert tuple(masked_by_name) == (
        "empty-stop",
        "one-then-stop",
        "same-span-two-labels",
        "same-span-reversed-physical-rank",
        "principal-collision-stop",
        "exact-four-then-stop",
        "fifth-relay-handed-to-capacity-poison",
        "reset-clears-mask",
    )
    assert masked_by_name["same-span-two-labels"].expected_factor_publication == (
        "minimum",
        "maximum",
    )
    assert masked_by_name["exact-four-then-stop"].expected_masked_factors == (
        "minimum",
        "maximum",
        "map",
        "sum",
    )
    assert masked_by_name["principal-collision-stop"].expected_factor_publication == ()

    compositor = v2.COMPOSITOR_TOPOLOGY_RECORD
    assert compositor["event_capacity"] == 4
    assert compositor["occupancy_storage_cells"] == 24
    assert compositor["cardinality_storage_cells"] == 5
    assert compositor["ordered_publication_slots"] == 4
    assert compositor["overlap_detector_cells"] == 0
    assert compositor["overlap_poison_cells"] == 0
    assert compositor["different_factor_overlap_and_nesting_allowed"] is True
    assert compositor["factor_set_semantics_ignore_ingress_order"] is True
    assert compositor["host_order_sort"] is False
    assert compositor["host_set_union"] is False
    assert compositor["host_span_filter"] is False

    controls = {control.name: control for control in v2.PROTOCOL.compositor_controls}
    same_span = controls["same-span-different-factor-overlap"]
    assert same_span.expected_status == "READY_FACTOR_SET"
    assert same_span.expected_public_factors == ("minimum", "maximum")
    canonical = controls["nested-exact-four-prompt-order"]
    permuted = controls["nested-exact-four-permuted-order"]
    assert canonical.expected_status == permuted.expected_status == "READY_FACTOR_SET"
    assert canonical.expected_factor_set_sha256 == permuted.expected_factor_set_sha256
    assert canonical.ordered_publication_sha256 != permuted.ordered_publication_sha256
    assert controls["same-factor-duplicate"].expected_status == "STOP_DUPLICATE_FACTOR"
    assert controls["same-factor-duplicate"].expected_public_factors == ()
    assert controls["fifth-distinct-factor"].expected_status == "STOP_EVENT_CAPACITY_4"
    assert controls["fifth-distinct-factor"].expected_public_factors == ()


def test_protocol_is_immutable_data_only_and_dependency_bound() -> None:
    protocol = v2.PROTOCOL
    root = protocol.root_record()
    assert root["learner_array_count"] == 0
    assert root["learning_run_count"] == 0
    assert root["electrical_execution_count"] == 0
    assert root["learner_state_constructed"] is False
    assert root["learning_executed"] is False
    assert root["electrical_controls_executed"] is False
    assert root["execution_surface_present"] is False
    assert not any(type(value).__module__.startswith("numpy") for value in _walk(protocol))

    assert dict(v2.SURFACE_BOUNDARY_RECORD) == {
        "public_train_prompt_appearance_count": 90,
        "canonical_semantic_identity_count": 42,
        "raw_public_prompt_text_count": 0,
        "dev_input_count": 0,
        "protected_input_count": 0,
        "desired_output_count": 0,
        "program_artifact_count": 0,
        "property_row_count": 0,
        "family_identifier_count": 0,
        "learner_array_count": 0,
        "learning_run_count": 0,
        "electrical_execution_count": 0,
        "learner_or_inference_interface_present": False,
        "execution_deferred_to_separately_named_learner": True,
    }
    forbidden_surfaces = {
        "resolve",
        "fit",
        "compose",
        "execute",
        "build_frozen_public_train_factor_bridge",
        "FrozenPublicTrainFactorBridge",
    }
    assert forbidden_surfaces.isdisjoint(vars(v2))
    public_functions = {
        name
        for name, value in inspect.getmembers(v2, inspect.isfunction)
        if not name.startswith("_") and value.__module__ == v2.__name__
    }
    assert public_functions == set()

    repository = Path(__file__).resolve().parents[1]
    source_roots = {
        "prompt_substrate_source_sha256": (
            repository / "experiments/r3_pilot/public_train_prompt_semantic_bridge.py"
        ),
        "prompt_manifest_source_sha256": (
            repository / "experiments/r3_pilot/public_train_prompt_factor_manifest_v1.py"
        ),
        "v5_protocol_source_sha256": (
            repository / "experiments/r3_pilot/teacher_gated_factor_workspace_v5_protocol.py"
        ),
        "v5_learner_source_sha256": (
            repository / "experiments/r3_pilot/teacher_gated_factor_workspace_v5.py"
        ),
    }
    for key, path in source_roots.items():
        assert sha256(path.read_bytes()).hexdigest() == v2.DEPENDENCY_RECORD[key]
    assert v2.DEPENDENCY_RECORD["v5_learner_commit"] == "5cb01cc"

    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "RUN"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.counterfactual_lessons[0].factor = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        v2.COMPOSITOR_TOPOLOGY_RECORD["event_capacity"] = 2  # type: ignore[index]


def test_all_protocol_roots_are_exact_and_schema_bound() -> None:
    actual_roots = {
        field.name: getattr(v2.PROTOCOL, field.name)
        for field in dataclasses.fields(v2.PROTOCOL)
        if field.name.endswith("sha256")
    }
    assert actual_roots == EXPECTED_ROOTS
    assert v2.PROTOCOL.protocol_sha256 == v2._namespaced_sha256(
        v2._PROTOCOL_NAMESPACE,
        v2.PROTOCOL.root_record(include_protocol_sha256=False),
    )
    assert all(
        len(root) == 64 and root == root.lower() and int(root, 16) >= 0
        for root in actual_roots.values()
    )
    assert (
        v2.ELECTRICAL_PREFLIGHT_CONTRACT_RECORD["controls_before_learner_state_allocation"] is True
    )
    assert v2.ELECTRICAL_PREFLIGHT_CONTRACT_RECORD["execution_deferred"] is True
