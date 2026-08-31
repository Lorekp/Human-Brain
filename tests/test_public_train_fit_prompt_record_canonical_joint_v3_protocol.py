from pathlib import Path

import pytest
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v3_protocol as protocol,
)


def test_v3_data_only_roster_split_contract_and_dynamic_work_bound() -> None:
    assert protocol.PROTOCOL.status == protocol.PASS
    assert protocol.PROTOCOL.protocol_sha256 == (
        "265df3427f54bf95c1516bcf91081a6a54fef47b1b4e70bd7af8a92783d7d660"
    )

    terminal = protocol.TERMINAL_SPENT_COHORTS
    assert tuple(item["name"] for item in terminal) == ("V1", "V2")
    assert tuple(item["verdict"] for item in terminal) == (
        "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1",
        "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2",
    )
    assert all(item["terminal_and_never_reclassified"] is True for item in terminal)
    assert protocol.SPENT_APPEARANCE_INDICES == (60, 61, 62, 71)
    assert protocol._canonical_spent_appearances(tuple(reversed(terminal))) == (
        60,
        61,
        62,
        71,
    )
    assert terminal[0]["protocol_source_sha256"] == (
        "a149b3468a980517c30d7322a4247b9f7506306fc80f2ced4d0ff9ab479a97a0"
    )
    assert terminal[1]["protocol_source_sha256"] == (
        "68bdbf8a36d77b01d204dfeaec46f7cabb44f45ffcdca272c2a481b58cb71670"
    )
    assert terminal[1]["bound_engine_source_sha256"] == (
        "18f9a6205a07a3b088288bbab4f08c100c5c2ab9b84dd411a04e7e96f628a4fe"
    )

    eligibility = protocol.ELIGIBILITY_RECEIPT
    assert eligibility.source_appearance_count == 90
    assert eligibility.scanned_appearance_count == 65
    assert eligibility.selected_appearance_indices == (63, 64)
    assert eligibility.excluded_spent_appearance_indices == (60, 61, 62, 71)
    assert not set(eligibility.selected_appearance_indices) & set(
        eligibility.excluded_spent_appearance_indices
    )
    assert eligibility.visible_io_read_count_before_roster_freeze == 0
    assert eligibility.task_family_name_program_root_desired_authority_count == 0
    assert protocol.NEUTRAL_ELIGIBILITY_RULE[
        "exclusion_membership_order_independent"
    ]
    assert (
        protocol.NEUTRAL_ELIGIBILITY_RULE[
            "task_family_name_program_root_desired_authority"
        ]
        is False
    )
    assert (
        protocol.NEUTRAL_ELIGIBILITY_RULE["visible_io_opened_during_selection"]
        is False
    )

    geometries = protocol.CANDIDATE_GEOMETRY_RECEIPTS
    assert tuple(row.appearance_index for row in geometries) == (63, 64)
    assert tuple(
        (
            row.span_record_count,
            row.unique_assembly_count,
            row.unique_literal_count,
            row.raw_candidate_count,
            row.literal_candidate_count,
            row.total_candidate_count,
        )
        for row in geometries
    ) == (
        (70, 69, 2, 552, 2, 554),
        (70, 69, 2, 552, 2, 554),
    )
    assert tuple(row.candidate_events_sha256 for row in geometries) == (
        "745093727f8b68c9af4e98a833ff5eb3820fee8967dec7aa27ebbcbb94f7f369",
        "6367a5eb51e7a425d9bf873e25e656d6ccb6dfa37ee6ccbf19fe515f910ee1b7",
    )
    assert tuple(row.sha256 for row in geometries) == (
        "d5f2a9411f528f054259cb9b4e70e13be840fc5258b28fdbc2f8aa71721857cf",
        "f2e02f93654106375bca6e46ef77975a92164930228f445817eb02dc35b19896",
    )

    exact = protocol.preflight_pairing(protocol.ROWS)
    assert exact.status == protocol.PASS
    assert exact.roster_sha256 == protocol.PROTOCOL.roster_sha256
    assert exact.electrical_execution_count == 0
    assert exact.proof_materialization_count == 0
    assert exact.runtime_desired_read_count == 0

    swapped = protocol.preflight_pairing(tuple(reversed(protocol.ROWS)))
    assert swapped.status == protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
    forged_row = object.__new__(protocol.FrozenPublicTrainJointRow)
    forged = protocol.preflight_pairing((protocol.ROWS[0], forged_row))
    assert forged.status == protocol.STOP_ROW_CERTIFICATE

    payloads = tuple(
        protocol.require_build_payload(exact, row) for row in protocol.ROWS
    )
    assert tuple(payload.sha256 for payload in payloads) == tuple(
        receipt.build_payload_sha256 for receipt in protocol.PROVENANCE_RECEIPTS
    )
    assert payloads[0].cases == (([],), ([0],), ([-4, 4],), ([1, 2, 3],))
    assert payloads[1].cases == (([],), ([0],), ([-8, 8],), ([1, 2, 3],))
    assert tuple(payload.literals for payload in payloads) == (
        (-3, 0, 3),
        (-7, 0, 7),
    )
    forbidden = (
        "desired",
        "appearance_index",
        "task_id",
        "family",
        "name",
        "input_root",
        "program",
    )
    assert all(not hasattr(payload, field) for payload in payloads for field in forbidden)
    with pytest.raises(protocol.JointProtocolStop) as raised:
        protocol.require_build_payload(exact, forged_row)
    assert raised.value.status == protocol.STOP_ROW_CERTIFICATE

    execution = protocol.SOURCE_STATIC_EXECUTION_CONTRACT
    assert execution.dependency_source_sha256 == (
        "9f56d3f3f6a3a499118c6ba2d6a9f6ad7ab0a3de195b6f2d77d64d05e271cc68"
    )
    assert execution.selector_passes_per_row == 2
    assert execution.selector_gains == (0.0, 1.0)
    assert execution.expected_source_statuses == (
        "PUBLIC_TRAIN_FIT_PROMPT_RECORD_SOURCE_STATIC_SPLIT_V1_SOURCE_ZERO",
        "PUBLIC_TRAIN_FIT_PROMPT_RECORD_SOURCE_STATIC_SPLIT_V1_SOURCE_SELECTED",
    )
    assert execution.gain1_source_certificate_count_per_row == 1
    assert execution.gain1_source_certificate_required_exact is True
    assert execution.static_passes_per_row == 2
    assert execution.static_gains == (0.0, 1.0)
    assert execution.static_input_selector_passes == (1, 1)
    assert execution.same_gain1_certificate_identity_required is True
    assert execution.selection_rerun_for_static_allowed is False
    assert execution.retain_source_status_and_roots is True
    assert execution.retain_static_status_and_roots_even_on_stop is True
    assert execution.host_selected_count_admission_gate_present is False
    assert execution.authenticated_static_on_gates_downstream is True
    assert execution.gated_downstream_modes == (
        "converter-authority-off",
        "wrong-desired",
        "normal",
    )

    work = protocol.WORK_FORMULA
    assert work.per_row_authenticated_candidate_counts == (554, 554)
    assert work.source_query_cap == 2 * 2 * 554 == 2_216
    assert work.per_row_selected_count_bounds == ((1, 554), (1, 554))
    assert protocol.static_query_count_for_selected_counts((3, 5)) == 2 * (
        (3 + 6 * (3 * 3 + 18 * 3)) + (5 + 6 * (5 * 5 + 18 * 5))
    )
    assert protocol.static_query_count_for_selected_counts((554, 554)) == (
        work.static_query_cap
    ) == 7_607_528
    assert work.fitted_glif_steps_per_query == 30
    assert work.fitted_glif_step_cap == (
        work.source_query_cap + work.static_query_cap
    ) * 30 == 228_292_320
    assert work.host_selected_count_admission_gate_present is False
    assert work.cap_independent_of_desired_and_outcomes is True
    with pytest.raises(ValueError):
        protocol.static_query_count_for_selected_counts((555, 0))
    with pytest.raises(ValueError):
        protocol.static_query_count_for_selected_counts((0, 1))
    assert not hasattr(protocol, "SUPPORTED_SOURCE_EVENT_COUNTS")

    repository = Path(protocol.__file__).resolve().parents[2]
    assert not (repository / protocol.FUTURE_RESULT_PATH).exists()
    assert not (repository / protocol.FUTURE_SPENT_PATH).exists()
    assert protocol.SOURCE_SELECTION_EXECUTION_COUNT == 0
    assert protocol.STATIC_INFERENCE_EXECUTION_COUNT == 0
    assert protocol.ELECTRICAL_EXECUTION_COUNT == 0
    assert protocol.RESULT_ARTIFACT_COUNT == 0
    assert protocol.SPEND_MARKER_COUNT == 0
    assert not hasattr(protocol, "require_selection_payload")
    assert not hasattr(protocol, "open_desired")
