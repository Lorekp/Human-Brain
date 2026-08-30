from experiments.r3_pilot import public_train_prompt_selection_v5_protocol as v5


def test_public_train_prompt_selection_v5_freezes_exact_zero_allocation_stop() -> None:
    protocol = v5.PROTOCOL
    ledger = v5.STOP_LEDGER
    contract = v5.SUCCESSOR_ANNOTATION_CONTRACT

    assert protocol.status == v5.STOP_IDENTIFIABILITY
    assert protocol.curriculum == ()
    assert len(protocol.fit_identity_audits) == 9
    assert ledger.fit_identity_count == 9
    assert ledger.fit_appearance_count == 12
    assert ledger.excluded_heldout_identity_count == 6
    assert ledger.input_annotation_join_count == 9
    assert ledger.missing_input_annotation_count == 0
    assert ledger.duplicate_input_root_count == 0
    assert ledger.target_class_count == 2
    assert ledger.causal_key_target_conflict_count == 0

    assert (
        ledger.raw_span_event_count,
        ledger.typed_literal_event_count,
        ledger.legacy_relay_label_count,
        ledger.legacy_edge_label_count,
        ledger.legacy_edge_endpoint_reference_count,
        ledger.legacy_child_ordinal_field_count,
        ledger.legacy_orientation_label_count,
        ledger.legacy_parameter_label_count,
        ledger.legacy_literal_slot_label_count,
        ledger.legacy_literal_copy_label_count,
    ) == (630, 18, 217, 68, 136, 68, 54, 9, 18, 18)
    assert (
        ledger.structural_event_target_count,
        ledger.stable_node_occurrence_count,
        ledger.edge_carrier_record_count,
        ledger.parent_factor_slot_occupancy_count,
        ledger.occurrence_local_source_binding_count,
        ledger.generic_factor_slot_record_count,
        ledger.relation_slot_value_event_count,
        ledger.complete_domain_record_count,
        ledger.literal_copy_group_count,
        ledger.curriculum_lesson_count,
        ledger.learner_array_count,
        ledger.learning_run_count,
        ledger.electrical_execution_count,
        ledger.panel_spend_count,
    ) == (0,) * 14
    assert ledger.missing_requirement_identity_counts == tuple(
        (requirement, 9) for requirement in v5.MISSING_REQUIREMENTS
    )
    assert all(
        row.missing_requirements == v5.MISSING_REQUIREMENTS
        and row.explicit_stable_node_occurrence_count == 0
        and row.explicit_edge_carrier_record_count == 0
        and row.explicit_parent_factor_slot_occupancy_count == 0
        and row.explicit_occurrence_local_source_binding_count == 0
        and row.explicit_generic_factor_slot_record_count == 0
        and row.explicit_relation_slot_value_event_count == 0
        and row.explicit_complete_domain_record_count == 0
        and row.explicit_literal_copy_group_count == 0
        for row in protocol.fit_identity_audits
    )

    assert contract.factor_kinds == ("literal", "role")
    assert contract.physical_types == ("Int", "List[Int]")
    assert contract.relation_operators == ("eq", "ne", "lt", "le", "gt", "ge")
    assert {
        "node_occurrence_cells",
        "edge_parent_occurrence_cells",
        "edge_child_occurrence_cells",
        "edge_child_position_cells",
        "parent_factor_slot_occupancy_cells",
        "occurrence_source_event_sha256",
        "ordered_factor_slots",
        "relation_events",
    } <= set(contract.event_fields)
    assert set(contract.forbidden_fields) == set(v5.FORBIDDEN_FIELDS)

    assert v5.DEPENDENCY_SHA256 == (
        "82fa9955c8d3aaab624aaf9995d0ab28ee808d60ee7ad08e61451aec536787a9"
    )
    assert contract.sha256 == (
        "8764290fa5a1c3fda2badc06fbf400df462c14bf603bae128fdfb1394fc14cbd"
    )
    assert v5.FIT_ROSTER_SHA256 == (
        "02aa54781a2152a54863b8f4d456ecea70bc040e9dd3e405cc90d5e18f10ffba"
    )
    assert ledger.sha256 == (
        "14e0a033c28be009456bb76b8db460eb48db097aa2b3ae5f2b1fc4d30ad7e35a"
    )
    assert v5.SURFACE_BOUNDARY_SHA256 == (
        "dbc7f1fd455141fa199b5f0f41ab91c51756777badade130ec0c4ed730daf3b6"
    )
    assert protocol.protocol_sha256 == (
        "c6f5bf2951576a05304b4870a2dbc3fc1921cdb62c4e992fd2af176d63baa408"
    )
