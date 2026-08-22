import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from humanbrain.scale_accounting import (
    CellCounts,
    EdgeCounts,
    EventCounts,
    MeasurementInterval,
    MemoryCounts,
    ScaleAccounting,
    ScaleClaim,
    ShardMemory,
    ShardPairTraffic,
    WorkCounts,
)

ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_BENCHMARK = ROOT / "benchmarks" / "structured_topology_construction.py"


def _accounting(
    *,
    interval: MeasurementInterval = MeasurementInterval("sample", 0, 1, 0.001, 0.1),
    cells: CellCounts = CellCounts(2, 2, 2, 2, 0, 0),
    work: WorkCounts = WorkCounts(0, 0, 0, 0),
    edges: EdgeCounts = EdgeCounts(0, 0, 0, 0),
) -> ScaleAccounting:
    return ScaleAccounting(
        claim=ScaleClaim("scale result", "exact scale counters"),
        interval=interval,
        cells=cells,
        work=work,
        edges=edges,
        events=EventCounts(0, 0, ()),
        memory=MemoryCounts(()),
    )


def test_accounting_serializes_one_canonical_scale_claim() -> None:
    accounting = ScaleAccounting(
        claim=ScaleClaim(
            label="materialized topology construction",
            text="Configured edges were generated into resident host records.",
        ),
        interval=MeasurementInterval(
            name="topology construction",
            start_step=0,
            step_count=0,
            biological_step_seconds=None,
            wall_seconds=0.25,
        ),
        cells=CellCounts(
            configured=8,
            addressable=8,
            resident_peak=0,
            resident_distinct=0,
            updated_distinct=0,
            spiking_distinct=0,
        ),
        work=WorkCounts(
            cell_step_evaluations=0,
            spike_events=0,
            analytic_catchups=0,
            analytic_step_equivalents=0,
        ),
        edges=EdgeCounts(
            configured=48,
            resident=48,
            mutable=0,
            generated=48,
        ),
        events=EventCounts(
            delivered=0,
            queue_peak=0,
            sent_bytes_by_shard_pair=(
                ShardPairTraffic("shard-b", "shard-a", 7),
                ShardPairTraffic("shard-a", "shard-b", 5),
            ),
        ),
        memory=MemoryCounts(
            bytes_by_shard=(
                ShardMemory("shard-b", host_bytes=32, device_bytes=4),
                ShardMemory("shard-a", host_bytes=192, device_bytes=0),
            )
        ),
    )

    assert accounting.to_dict() == {
        "claim": {
            "label": "materialized topology construction",
            "text": "Configured edges were generated into resident host records.",
        },
        "interval": {
            "name": "topology construction",
            "start_step": 0,
            "step_count": 0,
            "biological_step_seconds": None,
            "wall_seconds": 0.25,
        },
        "cells": {
            "configured": 8,
            "addressable": 8,
            "resident_peak": 0,
            "resident_distinct": 0,
            "updated_distinct": 0,
            "spiking_distinct": 0,
        },
        "work": {
            "cell_step_evaluations": 0,
            "spike_events": 0,
            "analytic_catchups": 0,
            "analytic_step_equivalents": 0,
        },
        "edges": {
            "configured": 48,
            "resident": 48,
            "mutable": 0,
            "generated": 48,
        },
        "events": {
            "delivered": 0,
            "queue_peak": 0,
            "sent_bytes_by_shard_pair": [
                {
                    "source_shard": "shard-a",
                    "target_shard": "shard-b",
                    "bytes": 5,
                },
                {
                    "source_shard": "shard-b",
                    "target_shard": "shard-a",
                    "bytes": 7,
                },
            ],
        },
        "memory": {
            "host_bytes": 224,
            "device_bytes": 4,
            "bytes_by_shard": [
                {"shard": "shard-a", "host_bytes": 192, "device_bytes": 0},
                {"shard": "shard-b", "host_bytes": 32, "device_bytes": 4},
            ],
        },
    }


def test_cell_counts_reject_impossible_scale_order() -> None:
    invalid_counts = (
        CellCounts(8, 9, 0, 0, 0, 0),
        CellCounts(8, 8, 5, 4, 0, 0),
        CellCounts(8, 8, 4, 4, 5, 0),
        CellCounts(8, 8, 4, 4, 3, 4),
    )

    for counts in invalid_counts:
        with pytest.raises(ValueError, match="cell scale relation"):
            ScaleAccounting(
                claim=ScaleClaim("scale", "updated cells"),
                interval=MeasurementInterval("sample", 0, 1, 0.001, 0.1),
                cells=counts,
                work=WorkCounts(5, 4, 0, 0),
                edges=EdgeCounts(0, 0, 0, 0),
                events=EventCounts(0, 0, ()),
                memory=MemoryCounts(()),
            )


@pytest.mark.parametrize(
    ("factory", "error"),
    (
        (lambda: CellCounts(-1, 0, 0, 0, 0, 0), ValueError),
        (lambda: WorkCounts(True, 0, 0, 0), TypeError),
        (lambda: EdgeCounts(0, 0, 0, -1), ValueError),
        (lambda: EventCounts(-1, 0, ()), ValueError),
        (lambda: ShardPairTraffic("a", "b", True), TypeError),
        (lambda: ShardMemory("a", 0, -1), ValueError),
    ),
)
def test_all_counters_require_exact_nonnegative_integers(
    factory: Callable[[], object],
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="non-negative integer"):
        factory()


@pytest.mark.parametrize(
    ("interval", "error", "message"),
    (
        (MeasurementInterval("", 0, 0, None, 0.1), ValueError, "name"),
        (
            MeasurementInterval("sample", True, 0, None, 0.1),
            TypeError,
            "start_step",
        ),
        (
            MeasurementInterval("sample", 0, -1, None, 0.1),
            ValueError,
            "step_count",
        ),
        (
            MeasurementInterval("sample", 0, 1, None, 0.1),
            ValueError,
            "biological_step_seconds",
        ),
        (
            MeasurementInterval("sample", 0, 1, 0.0, 0.1),
            ValueError,
            "biological_step_seconds",
        ),
        (
            MeasurementInterval("sample", 0, 0, None, float("nan")),
            ValueError,
            "wall_seconds",
        ),
    ),
)
def test_accounting_rejects_an_invalid_measurement_interval(
    interval: MeasurementInterval,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ScaleAccounting(
            claim=ScaleClaim("scale", "configured cells"),
            interval=interval,
            cells=CellCounts(0, 0, 0, 0, 0, 0),
            work=WorkCounts(0, 0, 0, 0),
            edges=EdgeCounts(0, 0, 0, 0),
            events=EventCounts(0, 0, ()),
            memory=MemoryCounts(()),
        )


@pytest.mark.parametrize(
    "claim",
    (
        ScaleClaim("active_cells", "scale result"),
        ScaleClaim("scale result", "The run reports active cells."),
        ScaleClaim("configured active cells", "scale result"),
    ),
)
def test_accounting_rejects_unqualified_active_claim_language(
    claim: ScaleClaim,
) -> None:
    with pytest.raises(ValueError, match="unqualified 'active'"):
        ScaleAccounting(
            claim=claim,
            interval=MeasurementInterval("sample", 0, 0, None, 0.1),
            cells=CellCounts(0, 0, 0, 0, 0, 0),
            work=WorkCounts(0, 0, 0, 0),
            edges=EdgeCounts(0, 0, 0, 0),
            events=EventCounts(0, 0, ()),
            memory=MemoryCounts(()),
        )


def test_accounting_allows_explicitly_qualified_active_claim_language() -> None:
    accounting = ScaleAccounting(
        claim=ScaleClaim(
            "updated_active_cells",
            "The interval reports spiking active cells, not configured capacity.",
        ),
        interval=MeasurementInterval("sample", 0, 1, 0.001, 0.1),
        cells=CellCounts(2, 2, 2, 2, 2, 1),
        work=WorkCounts(2, 1, 0, 0),
        edges=EdgeCounts(0, 0, 0, 0),
        events=EventCounts(0, 0, ()),
        memory=MemoryCounts(()),
    )

    assert accounting.to_dict()["claim"] == {
        "label": "updated_active_cells",
        "text": "The interval reports spiking active cells, not configured capacity.",
    }


@pytest.mark.parametrize(
    ("cells", "work", "message"),
    (
        (CellCounts(2, 2, 0, 1, 0, 0), WorkCounts(0, 0, 0, 0), "resident_peak"),
        (CellCounts(2, 2, 2, 2, 1, 0), WorkCounts(0, 0, 0, 0), "updated"),
        (CellCounts(2, 2, 2, 2, 0, 0), WorkCounts(1, 0, 0, 0), "updated"),
        (CellCounts(2, 2, 2, 2, 1, 1), WorkCounts(1, 0, 0, 0), "spike"),
        (CellCounts(2, 2, 2, 2, 1, 0), WorkCounts(1, 1, 0, 0), "spike"),
        (CellCounts(2, 2, 2, 2, 1, 0), WorkCounts(0, 0, 1, 0), "analytic"),
    ),
)
def test_accounting_rejects_inconsistent_cell_work_relations(
    cells: CellCounts,
    work: WorkCounts,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _accounting(cells=cells, work=work)


def test_accounting_rejects_dynamic_work_outside_a_step_interval() -> None:
    with pytest.raises(ValueError, match="step_count"):
        _accounting(
            interval=MeasurementInterval("construction", 0, 0, None, 0.1),
            cells=CellCounts(2, 2, 2, 2, 1, 0),
            work=WorkCounts(1, 0, 0, 0),
        )


def test_accounting_rejects_more_resident_than_configured_base_edges() -> None:
    with pytest.raises(ValueError, match="resident.*configured"):
        _accounting(edges=EdgeCounts(configured=3, resident=4, mutable=7, generated=8))


def test_accounting_rejects_more_mutable_than_resident_edge_records() -> None:
    with pytest.raises(ValueError, match="mutable.*resident"):
        _accounting(edges=EdgeCounts(configured=3, resident=2, mutable=3, generated=8))


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    (
        (lambda: ShardPairTraffic("", "b", 0), ValueError, "source_shard"),
        (lambda: ShardMemory("", 0, 0), ValueError, "shard"),
        (lambda: EventCounts(0, 0, []), TypeError, "tuple"),
        (
            lambda: EventCounts(
                0,
                0,
                (ShardPairTraffic("a", "b", 1), ShardPairTraffic("a", "b", 2)),
            ),
            ValueError,
            "duplicate shard pair",
        ),
        (lambda: MemoryCounts([]), TypeError, "tuple"),
        (
            lambda: MemoryCounts((ShardMemory("a", 1, 0), ShardMemory("a", 2, 0))),
            ValueError,
            "duplicate shard",
        ),
    ),
)
def test_shard_accounting_rejects_ambiguous_or_mutable_records(
    factory: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        factory()


def test_topology_worker_emits_truthful_accounting_without_artifact_run() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(TOPOLOGY_BENCHMARK),
            "--worker",
            "8",
            "--topology",
            "uniform_no_self",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    accounting = result["scale_accounting"]
    assert accounting["cells"] == {
        "configured": 8,
        "addressable": 0,
        "resident_peak": 0,
        "resident_distinct": 0,
        "updated_distinct": 0,
        "spiking_distinct": 0,
    }
    assert accounting["work"] == {
        "cell_step_evaluations": 0,
        "spike_events": 0,
        "analytic_catchups": 0,
        "analytic_step_equivalents": 0,
    }
    assert accounting["edges"] == {
        "configured": 48,
        "resident": 48,
        "mutable": 0,
        "generated": 48,
    }
    assert accounting["events"] == {
        "delivered": 0,
        "queue_peak": 0,
        "sent_bytes_by_shard_pair": [],
    }
    assert accounting["memory"] == {
        "host_bytes": 192,
        "device_bytes": 0,
        "bytes_by_shard": [
            {"shard": "process-0", "host_bytes": 192, "device_bytes": 0}
        ],
    }
    assert accounting["interval"]["step_count"] == 0
    assert accounting["interval"]["biological_step_seconds"] is None
    assert accounting["interval"]["wall_seconds"] == result["seconds"]
