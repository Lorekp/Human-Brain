from __future__ import annotations

from inspect import signature

from experiments.r4_integrated.persistent_multiscale_python_learning_v1 import (
    public_inference_schema,
    run_experiment,
)
from humanbrain.multiscale_source_motor_online_growth import (
    AppendOnlyMultiscaleSourceMotorStack,
)


def test_persistent_multiscale_v1_banks_exact_replay_but_not_renamed_shard_transfer() -> None:
    result = run_experiment()
    episodes = {row["phase"]: row for row in result["episodes"]}

    assert result["gate_passed"] is False
    assert result["transfer_score"] == {
        "attempted": 5,
        "passed": 2,
        "sanity_retries_excluded": 2,
    }
    assert result["failure_distribution"] == {
        "correct_ambiguity_stop": 1,
        "correct_edit": 4,
        "physical_stop": 10,
    }
    assert result["unmet_expectation_distribution"] == {
        "one_directory_fresh_transfer": 1,
        "one_directory_retention_transfer_after_update_two": 1,
        "two_directory_fresh_transfer_after_update_two": 1,
    }

    assert episodes["base_transfer_before_updates"]["outcome"] == "correct_edit"
    assert episodes["base_transfer_before_updates"]["winning_segment_ordinal"] == 0
    assert episodes["one_directory_initial"]["physical_stop"] is True
    assert episodes["one_directory_exact_retry_sanity"]["outcome"] == "correct_edit"
    assert episodes["one_directory_exact_retry_sanity"]["winning_segment_ordinal"] == 1
    assert episodes["one_directory_fresh_transfer"]["physical_stop"] is True
    assert episodes["one_directory_fresh_transfer"]["expectation_met"] is False
    assert episodes["two_directory_initial"]["physical_stop"] is True
    assert episodes["two_directory_exact_retry_sanity"]["outcome"] == "correct_edit"
    assert episodes["two_directory_exact_retry_sanity"]["winning_segment_ordinal"] == 2
    assert episodes["flat_base_retention_after_update_two"]["outcome"] == "correct_edit"
    assert episodes["flat_base_retention_after_update_two"]["winning_segment_ordinal"] == 0
    assert episodes["one_directory_retention_transfer_after_update_two"]["physical_stop"] is True
    assert episodes["two_directory_fresh_transfer_after_update_two"]["physical_stop"] is True
    assert episodes["genuine_ambiguity"]["outcome"] == "correct_ambiguity_stop"
    assert episodes["no_update_initial"]["physical_stop"] is True
    assert episodes["no_update_exact_retry"]["physical_stop"] is True
    assert episodes["final_equality_off"]["physical_stop"] is True
    assert episodes["final_propagation_off"]["physical_stop"] is True
    assert episodes["final_boundary_off"]["physical_stop"] is True

    lineage = result["checkpoint_lineage"]
    assert len(lineage) == 3
    assert lineage[0]["summary"] == {
        "segment_count": 1,
        "reserved_cell_capacity_estimate": 4381,
        "reserved_synapse_capacity_estimate": 33,
        "learned_event_bytes_retained_estimate": 145,
        "excitatory_support_tick_estimate": 64,
        "next_reserved_cell_address": 4381,
        "next_reserved_synapse_address": 33,
        "provenance_sha256": "dd0097eb84f9a351684d1941f48a6c64d31a04c787a5975459b31160a9f69e91",
    }
    first_receipt = lineage[1]["receipt"]
    second_receipt = lineage[2]["receipt"]
    assert first_receipt["status"] == second_receipt["status"] == "APPENDED"
    assert (
        first_receipt["reserved_cell_capacity_estimate_added"],
        first_receipt["reserved_synapse_capacity_estimate_added"],
        first_receipt["learned_event_bytes_retained_estimate_added"],
        first_receipt["excitatory_support_tick_estimate_added"],
    ) == (4382, 35, 198, 68)
    assert (
        second_receipt["reserved_cell_capacity_estimate_added"],
        second_receipt["reserved_synapse_capacity_estimate_added"],
        second_receipt["learned_event_bytes_retained_estimate_added"],
        second_receipt["excitatory_support_tick_estimate_added"],
    ) == (4384, 37, 251, 72)
    for receipt in (first_receipt, second_receipt):
        assert receipt["parent_experience_replay_count"] == 0
        assert receipt["raw_training_sequences_retained"] == 0
        assert receipt["source_experience_count"] == 1
        assert receipt["current_experience_training_rows"] == 2
        assert receipt["committed_evidence_tick_phases"] == 2

    assert result["final_summary"] == {
        "segment_count": 3,
        "reserved_cell_capacity_estimate": 13147,
        "reserved_synapse_capacity_estimate": 105,
        "learned_event_bytes_retained_estimate": 594,
        "excitatory_support_tick_estimate": 204,
        "next_reserved_cell_address": 13147,
        "next_reserved_synapse_address": 105,
        "provenance_sha256": "bfe3708f554599104712f71e9bb7bc7936cbd75de68f6e7a8fa732d020a64627",
    }
    assert result["no_update"]["zero_state_committed"] is True
    assert result["no_update"]["receipt"]["status"] == "NO_UPDATE"
    assert all(result["integrity"].values())
    assert public_inference_schema() == ("task_id", "issue", "files")
    assert tuple(signature(AppendOnlyMultiscaleSourceMotorStack.respond).parameters) == (
        "self",
        "query",
        "files",
        "ablation",
    )
