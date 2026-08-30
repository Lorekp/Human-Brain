from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_annotation_capsule_v1 as capsule,
)


def test_fit_capsule_stops_atomically_on_missing_raw_selector_geometry() -> None:
    ledger = capsule.STOP_LEDGER

    assert ledger.status == capsule.STOP_RAW_SELECTOR
    assert (
        ledger.fit_identity_count,
        ledger.fit_appearance_count,
        ledger.target_class_count,
        ledger.complete_identity_count,
        ledger.stopped_identity_count,
    ) == (9, 12, 2, 0, 9)
    assert (
        ledger.topology_node_count,
        ledger.edge_record_count,
        ledger.factor_slot_record_count,
        ledger.relation_record_count,
        ledger.domain_record_count,
        ledger.literal_copy_group_count,
        ledger.signature_record_count,
        ledger.potential_lesson_count,
    ) == (77, 68, 164, 9, 18, 18, 18, 372)
    assert (
        ledger.stable_node_carrier_count,
        ledger.stable_node_carrier_conflict_count,
        ledger.exact_physical_span_join_count,
        ledger.exact_typed_literal_event_join_count,
        ledger.required_raw_observation_count,
        ledger.raw_selector_candidate_count,
        ledger.exact_raw_observation_join_count,
    ) == (77, 0, 59, 18, 59, 472, 0)
    assert (
        ledger.missing_raw_direction_count,
        ledger.missing_raw_cue_slot_count,
        ledger.missing_raw_selected_slot_count,
        ledger.multiple_physical_span_candidate_count,
        ledger.source_order_choice_count,
        ledger.fixed_index_choice_count,
        ledger.translation_conflict_count,
    ) == (59, 59, 59, 0, 0, 0, 0)
    assert (
        ledger.emitted_target_record_count,
        ledger.curriculum_lesson_count,
        ledger.learner_array_count,
        ledger.learning_run_count,
        ledger.electrical_execution_count,
        ledger.panel_spend_count,
        ledger.artifact_export_count,
    ) == (0, 0, 0, 0, 0, 0, 0)

    rows = capsule.FIT_IDENTITY_AUDITS
    assert Counter(
        (
            row.topology_node_count,
            row.edge_record_count,
            row.factor_slot_record_count,
            row.relation_record_count,
            row.domain_record_count,
            row.literal_copy_group_count,
            row.signature_record_count,
            row.potential_lesson_count,
            row.required_raw_observation_count,
        )
        for row in rows
    ) == Counter(
        {
            (8, 7, 16, 1, 2, 2, 2, 38, 6): 4,
            (9, 8, 20, 1, 2, 2, 2, 44, 7): 5,
        }
    )
    assert sum(row.multiplicity for row in rows if row.topology_node_count == 8) == 6
    assert sum(row.multiplicity for row in rows if row.topology_node_count == 9) == 6
    assert all(
        row.missing_fields
        == (
            "raw_observation.direction",
            "raw_observation.cue_slot",
            "raw_observation.selected_slot",
        )
        and row.exact_physical_span_join_count
        == row.required_raw_observation_count
        and row.raw_selector_candidate_count
        == row.required_raw_observation_count * capsule.RAW_SELECTOR_GEOMETRY_COUNT
        and row.exact_raw_observation_join_count == 0
        for row in rows
    )

    assert capsule.CAPSULE.root_record() == {
        "scope": capsule.PROTOCOL_SCOPE,
        "status": capsule.STOP_RAW_SELECTOR,
        "dependency_sha256": "b81f1d2e64517f0f4b23a4cf5065e32966da1b59bc9e40d9c15e7cad0dd30279",
        "successor_contract_sha256": (
            "8764290fa5a1c3fda2badc06fbf400df462c14bf603bae128fdfb1394fc14cbd"
        ),
        "physical_vocabulary_sha256": (
            "28ef60d27045e5cc152021f35f81554ba00b2ced525bc96e82743ae7fa2cf806"
        ),
        "identity_audit_set_sha256": (
            "aa40aed56c2e43b89cce44cd9fa1d7ac5ba883b76b4641263b78e71def4a3669"
        ),
        "stop_ledger_sha256": "39f02d04249bbb6cb52c2d36f76a72f705ce5871f4899568377784243325e7de",
        "surface_boundary_sha256": (
            "4c7ca092916b1f4159dfbdb52005e8edc4645f3a912bef051ab2f72a25b01fba"
        ),
        "raw_source_successor_contract_sha256": (
            "fb11a848948255a8c98b10d1b50ad28e494da2082b8eab990c88de00c9c145bc"
        ),
        "fit_identity_count": 9,
        "emitted_target_record_count": 0,
        "artifact_export_count": 0,
        "capsule_sha256": "f0e1bf69b3943c7c09fa7f7322d5a91d3246db651e8ce898030b54ae6393235b",
    }
    assert (
        capsule.RAW_SOURCE_SUCCESSOR_CONTRACT_RECORD["required_record_count"]
        == 59
    )
    assert (
        capsule.RAW_SOURCE_SUCCESSOR_CONTRACT_RECORD["allocation_condition"]
        == "all-59-records-atomically-identifiable"
    )
    with pytest.raises((FrozenInstanceError, AttributeError)):
        ledger.status = "READY"  # type: ignore[misc]
    with pytest.raises(TypeError):
        capsule.RAW_SOURCE_SUCCESSOR_CONTRACT_RECORD["required_record_count"] = 0
