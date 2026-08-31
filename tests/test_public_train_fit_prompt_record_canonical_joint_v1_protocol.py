from __future__ import annotations

import pytest
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v1_protocol as protocol,
)


def test_two_row_public_train_joint_protocol_is_atomic_and_data_only() -> None:
    frozen = protocol.PROTOCOL
    assert frozen.status == protocol.PASS
    assert frozen.dependency_sha256 == (
        "0d123434e00d9a60d41099bca97b5b760423b179596d4f08b8f3a440c4c00b49"
    )
    assert frozen.roster_sha256 == (
        "a7b5be434bdef22bf5bccebf3e6a7c2a28001c87566bc087b409634289388084"
    )
    assert frozen.controls_sha256 == (
        "85e6e2074362992bdaf1fc539008fe33605b3387f31ff663010ff3337f8455b1"
    )
    assert frozen.outcome_contract_sha256 == (
        "3ed755cebe6bfc76c865733ee83a2edb5f712fa9885238e24e0437c904ce580c"
    )
    assert frozen.work_caps_sha256 == (
        "3705bb240fd986f85e52abe8fd990be2fe0640383a8f24f8bbe678dc60e226fa"
    )
    assert frozen.one_shot_order_sha256 == (
        "9a0b0ab0a8c34a77a777cd2ebbaaaa6455e4141c05be9e6570d5c455e2290f79"
    )
    assert frozen.surface_boundary_sha256 == (
        "026b285ce0b746856f7ae00b0c3ae3526893a1a131297f4826d14ef8e5012828"
    )
    assert frozen.protocol_sha256 == (
        "524947ef3a29dbb4a7292c4e386d69090d9027c72fccb9559409aff1cf9cc6d0"
    )

    assert tuple(
        (
            item.appearance_index,
            item.task_id,
            item.input_root_sha256,
            item.span_events_sha256,
            item.literal_events_sha256,
            item.full_signature_sha256,
            item.input_multiplicity,
            item.selection_payload_sha256,
        )
        for item in frozen.provenance
    ) == (
        (
            61,
            "simple-swe-v1/train/tier-2/t2_clamp_then_sum/001",
            "781be7355d7b577006361245a80d38817bde4b71dd970236124d15378421f577",
            "73bc5fede7406b0071c128b96be17ad027844dbc43b5f098d38588b2a84b3188",
            "14fe0583ee6933db287935a572729ef0fd930497cc54ff6fa0204635277700ed",
            "1a0056668600b40b6cebeb9388b5c1bb92abe1e31042a74bf46fe1036d0b347d",
            1,
            "5d96648943689d65efad7cdf3c53f93cb2a867197853fd67cd22c1044adef4f7",
        ),
        (
            71,
            "simple-swe-v1/train/tier-2/t2_map_then_count/001",
            "6ce57bcff03faaee3204c0806de890bdb37d87dc80ebce31dbf069c2ceabb8d6",
            "0a8ced7eb5330ba32526885f2bcd3aa005c1021b6e87d9553e1bc6b3d1740793",
            "a4e5dfc895d91453cef44e691acf9bd0d06deb53ca1605fbb52eb1a7e52df3c0",
            "266f46cda74070c57b0414ae8df8c63816b420934dfce1a2a7c326828a1e3bb9",
            2,
            "6559a86be5d485291495305780e03a659b8c0ddcd13be59da15b08b011459be9",
        ),
    )
    assert {item.signature_role_events_sha256 for item in frozen.provenance} == {
        "945c6e4b1a8c5a77d711060711bbb6f412935d3d6cd48e7a9c5b2dc7300c63e4"
    }

    forbidden_runtime_authority = (
        "appearance_index",
        "task_id",
        "family",
        "name",
        "input_root",
        "program",
        "desired",
    )
    for row in protocol.ROWS:
        assert all(not hasattr(row, name) for name in forbidden_runtime_authority)
    with pytest.raises(TypeError):
        protocol.FrozenPublicTrainJointRow()

    preflight = protocol.preflight_pairing(protocol.ROWS)
    assert preflight.status == protocol.PASS
    assert preflight.roster_sha256 == frozen.roster_sha256
    assert preflight.sha256 == (
        "2715052abd1436e489a318b3d5553a474962b1720aae68ccfd260d05cf2d5814"
    )
    assert (
        preflight.electrical_execution_count,
        preflight.proof_materialization_count,
        preflight.runtime_desired_read_count,
    ) == (0, 0, 0)

    swapped = protocol.preflight_pairing(protocol.ROWS[::-1])
    assert swapped.status == protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
    assert swapped.sha256 == (
        "e008b2f48495bc88cd2494b3c40f64e99df9cf03b06e17243a6dfa33bc62bd0d"
    )
    assert (
        swapped.electrical_execution_count,
        swapped.proof_materialization_count,
        swapped.runtime_desired_read_count,
    ) == (0, 0, 0)
    with pytest.raises(protocol.JointProtocolStop) as stopped:
        protocol.require_build_payload(swapped, protocol.ROWS[0])
    assert stopped.value.status == protocol.STOP_PROMPT_IO_PAIRING_MISMATCH

    forged = object.__new__(protocol.FrozenPublicTrainJointRow)
    rejected = protocol.preflight_pairing((forged, protocol.ROWS[1]))
    assert rejected.status == protocol.STOP_ROW_CERTIFICATE
    assert (
        rejected.electrical_execution_count,
        rejected.proof_materialization_count,
        rejected.runtime_desired_read_count,
    ) == (0, 0, 0)

    payloads = tuple(
        protocol.require_build_payload(preflight, row) for row in protocol.ROWS
    )
    assert tuple(payload.sha256 for payload in payloads) == (
        "791d9f923488234ce2b100b19009f0cb3c86aa217aad9f03d185c886c7c9cb9f",
        "e5ead95d9aa30f3b2369bb20054eaa3a0872f29c2dac1515a273e454503ddd91",
    )
    assert tuple(len(payload.span_records) for payload in payloads) == (70, 70)
    assert tuple(
        tuple(record["typed_output_active_cells"] for record in payload.literal_records)
        for payload in payloads
    ) == (((0, 59), (0, 79)), ((0, 73), (0, 71)))
    assert tuple(payload.cases for payload in payloads) == (
        (([],), ([0],), ([-11, 11],), ([1, 2, 3],)),
        (([],), ([-2],), ([-1],), ([-9, 0, 9],)),
    )
    assert tuple(payload.literals for payload in payloads) == (
        (-10, 0, 10),
        (0, 2, 4),
    )
    assert tuple(payload.max_cost for payload in payloads) == (9, 9)
    assert payloads[0].signature == payloads[1].signature
    assert payloads[0].signature.entry_point == "joint_entry"
    for payload in payloads:
        assert all(not hasattr(payload, name) for name in forbidden_runtime_authority)
        assert payload.signature_role_record["entrypoint_spelling_present"] is False
        assert payload.signature_role_record["parameter_spelling_present"] is False

    assert tuple(item.name for item in protocol.CONTROL_CONTRACTS) == (
        "producer-off",
        "static-cells-off",
        "converter-authority-off",
        "desired-permutation",
    )
    assert protocol.WORK_CAPS == (
        562,
        10,
        5_620,
        2_723,
        250_290,
        (30_000, 30_000, 80_000),
        (120_000, 120_000, 320_000),
        4,
    )
    assert protocol.ONE_SHOT_ORDER[-3:] == (
        "normal-row-0",
        "normal-row-1",
        "freeze-one-result-and-mark-spent",
    )
    assert protocol.OUTCOME_CONTRACT["desired_opening_interface_present"] is False
    assert (
        protocol.OUTCOME_CONTRACT["future_private_exact_build_binding_required"]
        is True
    )
    assert protocol.OUTCOME_CONTRACT["shape_only_frontier_binding_allowed"] is False
    assert not hasattr(protocol, "require_selection_payload")
    assert (
        protocol.LEARNER_EXECUTION_COUNT,
        protocol.ELECTRICAL_EXECUTION_COUNT,
        protocol.PROOF_MATERIALIZATION_COUNT,
        protocol.RUNTIME_DESIRED_READ_COUNT,
        protocol.OLD_PANEL_OUTCOME_READ_COUNT,
        protocol.RESULT_ARTIFACT_COUNT,
    ) == (0, 0, 0, 0, 0, 0)
    assert all(
        protocol.SURFACE_BOUNDARY[key] == 0
        for key in (
            "dev_input_count",
            "heldout_input_count",
            "m0_input_count",
            "c1_input_count",
            "g0_input_count",
            "protected_input_count",
            "program_input_count",
        )
    )
