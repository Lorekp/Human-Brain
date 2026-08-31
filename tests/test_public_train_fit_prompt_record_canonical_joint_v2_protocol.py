from pathlib import Path

import pytest
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v2_protocol as protocol,
)


def test_fresh_v2_data_only_roster_geometry_and_exact_object_seam() -> None:
    assert protocol.PROTOCOL.status == protocol.PASS
    assert protocol.PROTOCOL.protocol_sha256 == (
        "5eb01df63fa0d0f375e5180bde91d01a395e509f2f34e6a46782fed77d53ea91"
    )
    assert len(protocol.ROWS) == 2
    assert len(protocol.PROVENANCE_RECEIPTS) == 2

    eligibility = protocol.ELIGIBILITY_RECEIPT
    assert eligibility.source_appearance_count == 90
    assert eligibility.scanned_appearance_count == 63
    assert eligibility.selected_appearance_indices == (60, 62)
    assert eligibility.excluded_spent_appearance_indices == (61, 71)
    assert not set(eligibility.selected_appearance_indices) & set(
        protocol.V1_SPENT_APPEARANCE_INDICES
    )
    assert eligibility.visible_io_read_count == 0
    assert eligibility.task_family_name_program_desired_authority_count == 0
    assert protocol.NEUTRAL_ELIGIBILITY_RULE["visible_io_opened_during_selection"] is False
    assert (
        protocol.NEUTRAL_ELIGIBILITY_RULE[
            "task_family_name_program_root_desired_authority"
        ]
        is False
    )

    geometries = protocol.CANDIDATE_GEOMETRY_RECEIPTS
    assert tuple(row.appearance_index for row in geometries) == (60, 62)
    assert tuple(
        (
            row.span_record_count,
            row.unique_assembly_count,
            row.unique_literal_count,
            row.raw_candidate_count,
            row.literal_candidate_count,
            row.total_candidate_count,
            row.collision_root_sha256,
        )
        for row in geometries
    ) == (
        (
            70,
            69,
            2,
            552,
            2,
            554,
            "81348e943876c13d60113170e61643b2b505d625842735c4484aea1567fcb90f",
        ),
        (
            70,
            69,
            2,
            552,
            2,
            554,
            "81348e943876c13d60113170e61643b2b505d625842735c4484aea1567fcb90f",
        ),
    )
    assert tuple(row.candidate_events_sha256 for row in geometries) == (
        "7a0bcd7644e12d5023ad4ec4d4b72eb89467170905bce82928f3d42e0f682868",
        "fd09c50b823b75c81c5e5f574d2840a22b2bd8dafe840a7b6f1c7a71176b6d83",
    )
    assert tuple(row.sha256 for row in geometries) == (
        "0d47bade318570106d221bb9eb66c9ef4a85f07b9ab71bd3a00ee40eb0f8c6ac",
        "0376d5119aee97701ed919aa264c9b8d9341fdd9c0af5b53c50263d4cdedbadd",
    )

    work = protocol.WORK_FORMULA
    assert work.per_row_candidate_counts == (554, 554)
    assert work.producer_query_cap == 5 * sum(work.per_row_candidate_counts) == 5_540
    assert work.supported_source_event_counts == (8, 9)
    assert work.static_query_cap == 2_934
    assert work.fitted_glif_steps_per_query == 30
    assert work.fitted_glif_step_cap == (5_540 + 2_934) * 30 == 254_220
    assert not hasattr(protocol, "PRODUCER_CANDIDATES_PER_ACTIVATION")

    frozen_v1 = protocol.V1_IMMUTABLE_SPENT_COHORT
    assert frozen_v1["appearance_indices"] == (61, 71)
    assert frozen_v1["verdict"] == (
        "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1"
    )
    assert frozen_v1["reason"] == "producer candidate count changed"
    assert frozen_v1["terminal_and_never_reclassified"] is True
    assert frozen_v1["result_artifact_sha256"] == (
        "67c261f31fa26aac6d021501878d2fd15bc1c6ea297bba3c31accd78453d843a"
    )

    repository = Path(protocol.__file__).resolve().parents[2]
    assert not (repository / protocol.FUTURE_RUNNER_PATH).exists()
    assert not (repository / protocol.FUTURE_RESULT_PATH).exists()
    assert not (repository / protocol.FUTURE_SPENT_PATH).exists()
    assert "_v2_runner" in protocol.FUTURE_RESULT_PATH
    assert "_v1_runner" not in protocol.FUTURE_RESULT_PATH
    assert protocol.RESULT_ARTIFACT_COUNT == 0
    assert protocol.SPEND_MARKER_COUNT == 0
    assert protocol.ELECTRICAL_EXECUTION_COUNT == 0
    assert protocol.PROOF_MATERIALIZATION_COUNT == 0
    assert protocol.RUNTIME_DESIRED_READ_COUNT == 0
    assert not hasattr(protocol, "require_selection_payload")
    assert not hasattr(protocol, "open_desired")

    exact = protocol.preflight_pairing(protocol.ROWS)
    assert exact.status == protocol.PASS
    assert exact.roster_sha256 == protocol.PROTOCOL.roster_sha256
    assert exact.electrical_execution_count == 0
    assert exact.proof_materialization_count == 0
    assert exact.runtime_desired_read_count == 0

    swapped = protocol.preflight_pairing(tuple(reversed(protocol.ROWS)))
    assert swapped.status == protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
    assert swapped.electrical_execution_count == 0
    assert swapped.proof_materialization_count == 0
    assert swapped.runtime_desired_read_count == 0

    forged_row = object.__new__(protocol.FrozenPublicTrainJointRow)
    forged = protocol.preflight_pairing((protocol.ROWS[0], forged_row))
    assert forged.status == protocol.STOP_ROW_CERTIFICATE
    assert forged.electrical_execution_count == 0
    assert forged.proof_materialization_count == 0
    assert forged.runtime_desired_read_count == 0

    payloads = tuple(
        protocol.require_build_payload(exact, row) for row in protocol.ROWS
    )
    assert tuple(payload.sha256 for payload in payloads) == tuple(
        receipt.build_payload_sha256 for receipt in protocol.PROVENANCE_RECEIPTS
    )
    assert payloads[0].cases == (([],), ([0],), ([-8, 8],), ([1, 2, 3],))
    assert payloads[1].cases == (([],), ([0],), ([-5, 5],), ([1, 2, 3],))
    assert tuple(payload.literals for payload in payloads) == (
        (-7, 0, 7),
        (-4, 0, 4),
    )
    assert all(payload.max_cost == 9 for payload in payloads)
    assert all(payload.signature.entry_point == "joint_entry" for payload in payloads)
    assert all(
        payload.signature.parameters[0].name == "arg0" for payload in payloads
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
