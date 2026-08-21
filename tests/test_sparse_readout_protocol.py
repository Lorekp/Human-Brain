from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "sparse_readout_protocol_v1.json"


def _sparse_readout_bytes(
    *, output_cells: int, reservoir_cells: int, fan_in: int
) -> int:
    candidate_count = min(reservoir_cells, 4 * fan_in)
    per_candidate = np.dtype(np.int32).itemsize + np.dtype(np.float64).itemsize
    per_contact = (
        np.dtype(np.int32).itemsize
        + np.dtype(np.float64).itemsize
        + np.dtype(np.uint32).itemsize
    )
    per_output = np.dtype(np.float64).itemsize + np.dtype(np.uint64).itemsize
    return output_cells * (
        candidate_count * per_candidate + fan_in * per_contact + per_output
    )


def test_sparse_readout_protocol_keeps_every_declared_memory_gate_reachable() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "preregistered-before-first-capability-run"
    assert not protocol["execution_lock"]["capability_execution_allowed"]
    case = protocol["engineering_gate"]["selected_fan_in_memory_case"]
    output_cells = int(case["output_cells"])
    reservoir_cells = int(case["reservoir_cells"])
    minimum_ratio = float(case["minimum_dense_to_actual_sparse_readout_byte_ratio"])
    dense_bytes = (
        output_cells * reservoir_cells * np.dtype(np.float64).itemsize
    )
    fan_ins = protocol["model"]["sparse_rule"]["fan_in_selection_grid"]

    for fan_in in fan_ins:
        sparse_bytes = _sparse_readout_bytes(
            output_cells=output_cells,
            reservoir_cells=reservoir_cells,
            fan_in=int(fan_in),
        )
        assert dense_bytes / sparse_bytes >= minimum_ratio
