from __future__ import annotations

import inspect

from experiments.r3_pilot import (
    public_train_list_int_occurrence_carrier_adapter_v1 as adapter,
)


def test_public_list_int_adapter_stops_before_forging_occurrence_carriers() -> None:
    state = adapter.run_public_list_int_occurrence_carrier_adapter_v1()

    assert state.status == adapter.STOP_NO_AUTHENTICATED_JOIN
    assert (state.unique_input_count, state.appearance_count) == (15, 20)
    assert (state.prompt_emitting_input_count, state.prompt_empty_input_count) == (9, 6)
    assert (
        state.prompt_occurrence_count,
        state.prompt_relay_count,
        state.prompt_selected_cell_count,
    ) == (185, 1038, 1038)
    assert (state.edge_binding_count, state.edge_endpoint_count) == (68, 136)
    assert state.reused_edge_endpoint_count == 0
    assert state.non_edge_binding_count == 0
    assert (
        state.attention_ready_input_count,
        state.attention_zero_input_count,
        state.attention_prompt_stop_input_count,
    ) == (5, 4, 6)
    assert state.attention_factor_scan_attempt_count == 4968
    assert state.attention_membership_spike_count == 10
    assert (
        state.stable_node_occurrence_join_count,
        state.occurrence_local_source_join_count,
        state.parent_factor_slot_occupancy_join_count,
        state.conversion_candidate_count,
        state.physical_occurrence_event_build_count,
        state.v5_inference_attempt_count,
        state.caller_certificate_count,
        state.source_ordinal_read_count,
        state.runtime_relay_parse_count,
        state.host_selected_structure_count,
        state.semantic_label_input_count,
        state.evaluation_input_count,
    ) == (0,) * 12
    assert len(state.input_audits) == 15
    assert all(row.sha256 and row.attention_receipt_sha256 for row in state.input_audits)
    source = inspect.getsource(adapter)
    assert "prompt_occurrence_slot_carrier_v5" not in source
    assert "SELF_POSED_TREATMENT_EVENTS" not in source
    assert "_source_feature_cells" not in source
