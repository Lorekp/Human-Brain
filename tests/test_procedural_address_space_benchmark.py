from benchmarks.procedural_address_space import _query_topology


def test_address_query_result_uses_truthful_scale_accounting() -> None:
    result = _query_topology(
        cell_count=8,
        fanout=6,
        topology="uniform_no_self",
        seed=71,
        valid_gids=(0, 7),
        invalid_gids=(-1, 8, 9, 1 << 64),
    )

    assert result["valid_cell_queries"] == 2
    assert result["valid_edge_row_queries"] == 2
    assert result["invalid_cell_queries_rejected"] == 4
    assert result["invalid_edge_queries_rejected"] == 4
    assert result["all_invariants_pass"] is True
    accounting = result["scale_accounting"]
    assert accounting["cells"] == {
        "configured": 8,
        "addressable": 8,
        "resident_peak": 0,
        "resident_distinct": 0,
        "updated_distinct": 0,
        "spiking_distinct": 0,
    }
    assert accounting["edges"] == {
        "configured": 48,
        "resident": 0,
        "mutable": 0,
        "generated": 12,
    }
    assert accounting["work"] == {
        "cell_step_evaluations": 0,
        "spike_events": 0,
        "analytic_catchups": 0,
        "analytic_step_equivalents": 0,
    }
    assert accounting["events"] == {
        "delivered": 0,
        "queue_peak": 0,
        "sent_bytes_by_shard_pair": [],
    }
    assert accounting["interval"]["step_count"] == 0
    assert accounting["interval"]["biological_step_seconds"] is None
    assert "no resident or updated cell simulation" in accounting["claim"]["text"]
    assert "tokens_per_second" not in result
