"""G-S0 fixture wiring tests: frozen mapping + observation-sink replay driver.

Complements tests/test_segment_growth.py (D2 mechanics) with the wiring layer
dispatched for G-S0: the tokenizer-freeze §4 pool mapping, winner sets sourced
from the lane's observation-sink spike stream, and the arm-level contracts
(A2 structural zero, bit-identical reruns, burst telemetry).  Uses small
synthetic concept-id streams so no corpus or mining is required.
"""

from __future__ import annotations

import pytest

from humanbrain.r3_pilot_glue import (
    CONCEPT_ID_CEILING,
    K_POOL,
    assert_feasibility,
    gs0_health_checks,
    initial_pool,
    pool_cells_for_id,
    pool_slot,
    required_pool_cells,
    run_arm,
)


# A tiny deterministic concept-id stream with a repeated bigram and a
# same-token-novel-context contrast (ids in the frozen band [1024,1536)).
STREAM = [1024, 1105, 1024, 1105, 1199, 1024, 1301, 1105, 1450] * 2


def test_frozen_mapping_formulas():
    assert initial_pool(1024) == 0
    assert initial_pool(1025) == 1
    assert initial_pool(1055) == 31
    assert pool_slot(1024) == 32
    assert pool_slot(1055) == 32
    assert pool_slot(1535) == 47
    # Pure arithmetic bijection: id -> (pool offset, slot); reconstructible.
    for tid in (0, 255, 1023, 1024, 1055, 1535):
        assert initial_pool(tid) + K_POOL * pool_slot(tid) == tid


def test_pool_blocks_disjoint_and_sized_k():
    ids = [1024, 1025, 1056, 1535]
    blocks = {tid: pool_cells_for_id(tid, cell_offset=96) for tid in ids}
    for tid, block in blocks.items():
        assert len(block) == K_POOL
        assert len(set(block)) == K_POOL
        assert all(c >= 96 for c in block), "pools must live above lane gids"
    seen: set[int] = set()
    for block in blocks.values():
        assert seen.isdisjoint(block)
        seen.update(block)


def test_feasibility_stop_discipline():
    ceiling = CONCEPT_ID_CEILING - 1
    assert_feasibility(ceiling, 96 + required_pool_cells(ceiling))  # headroom passes
    with pytest.raises(RuntimeError, match="STOP"):
        assert_feasibility(ceiling, required_pool_cells(ceiling) - 1)


def test_driver_a2_structural_zero():
    run, recorder, diag = run_arm(STREAM, arm="A2", lane_cells=96, passes=1)
    assert len(run.rows) == len(STREAM)
    assert all(r.burst for r in run.rows)
    assert all(r.fired_cells == () for r in run.rows)
    assert run.total_segments == 0 and run.total_contacts == 0
    assert diag["tracker_growth_on"] is False
    # Election machinery still ran for diagnostics on every burst.
    assert run.elections == len(STREAM)
    # Winner sets came from the observation-sink spike stream.
    assert recorder.winner_sets and all(len(w) > 0 for w in recorder.winner_sets[1:])
    assert run.predicted_path_fraction == 0.0


def test_driver_burst_telemetry_and_f_pred_computable():
    a1, recorder, _ = run_arm(STREAM, arm="A1", lane_cells=96, passes=2)
    assert len(a1.rows) == 2 * len(STREAM)
    assert all(isinstance(r.burst, bool) for r in a1.rows)
    assert a1.total_segments >= 1, "growth-on must allocate segments on bursts"
    assert a1.total_contacts >= 1
    fp = a1.predicted_path_fraction  # computable; value itself is not gating
    assert 0.0 <= fp <= 1.0
    # W_{t-1} sizes recorded per step; step 0 runs on an empty history.
    assert a1.rows[0].winner_set_size == 0
    assert all(r.winner_set_size > 0 for r in a1.rows[1:])
    # Spike-rate pre-flight telemetry present and sane.
    assert recorder.mean_activity_density() > 0.001


def test_driver_bit_identical_reruns():
    a1, rec1, _ = run_arm(STREAM, arm="A1", lane_cells=96, passes=1)
    a1b, rec2, _ = run_arm(STREAM, arm="A1", lane_cells=96, passes=1)
    checks = gs0_health_checks(a1, run_arm(STREAM, arm="A2", lane_cells=96, passes=1)[0], a1b, rec1)
    assert checks["bit_identical_reruns"] is True
    assert rec1.winner_sets == rec2.winner_sets
