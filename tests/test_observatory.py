from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from humanbrain.observatory import (
    ACTIVITY_STREAM_FORMAT,
    ActivityRecorder,
    evenly_spaced_gids,
)
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.visualizer import load_run
from humanbrain.visualizer.activity import ActivityStreamReader
from humanbrain.visualizer.structure import (
    StructureUnavailable,
    load_structure_manifest,
    open_procedural_structure,
    open_structure,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO_RUN = ROOT / "experiments" / "visualizer_demo" / "demo-ridge-512.json"


def _activity_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "type": "header",
            "format": ACTIVITY_STREAM_FORMAT,
            "run_id": "replay/fixture",
            "configured_neurons": 1_000_000_000,
            "active_neurons": 100,
            "sample_gids": [0, 99],
            "state_stride": 3,
            "model_family": "fixture",
            "token_interval_ms": 5.0,
            "integration_dt_ms": 0.05,
        }
    ]
    spikes = (0, 1, 2, 0, 3, 2, 0, 5)
    for index, spike_count in enumerate(spikes):
        frame: dict[str, object] = {
            "type": "frame",
            "index": index,
            "phase": "training" if index < 4 else "calibration",
            "token_id": index % 3,
            "biological_time_ms": 5.0 * (index + 1),
            "spike_count": spike_count,
            "population_firing_rate_hz": 10.0 * spike_count,
            "neuron_step_activity_density": spike_count / 10_000.0,
        }
        if index % 3 == 0:
            frame.update(
                {
                    "voltage_relative_mv": [-5.0 + index, -4.0 + index],
                    "synaptic_current_pa": [1.0 + index, 2.0 + index],
                    "eligibility": [0.1 * index, 0.2 * index],
                }
            )
        records.append(frame)
    records.append(
        {
            "type": "footer",
            "complete": True,
            "frame_count": len(spikes),
            "biological_time_ms": 5.0 * len(spikes),
        }
    )
    return records


def _write_activity_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_dense_structure_matches_the_circuit_topology_and_roles() -> None:
    run = load_run(DEMO_RUN)
    structure = open_structure(run)
    circuit_config = dict(run.configuration["circuit_config"])
    circuit_config["input_fanout"] = run.configuration["input_fanout"]
    circuit = RecurrentCellCircuit(
        token_cells=run.configuration["token_cells"],
        config=RecurrentCircuitConfig(**circuit_config),
    )

    outgoing = structure.neighborhood(0, hops=1).edges

    assert [edge.target_gid for edge in outgoing] == [
        int(target) for target in circuit._recurrent_targets[0]
    ]
    expected_role = (
        "inhibitory" if circuit._recurrent_weights_pa[0, 0] < 0 else "excitatory"
    )
    assert structure.describe(0).outgoing_role == expected_role
    assert structure.info.representation.startswith("exact reconstructed")


def test_procedural_billion_node_structure_queries_stay_bounded() -> None:
    structure = open_procedural_structure(
        {
            "cell_count": 1_000_000_000,
            "fanout": 8,
            "topology": "uniform_no_self",
            "seed": 7,
        }
    )

    view = structure.neighborhood(999_999_999, hops=2, max_nodes=80)
    density = structure.density_overview(bin_count=8, sample_limit=32)

    assert structure.info.configured_neurons == 1_000_000_000
    assert len(view.nodes) <= 80
    assert all(0 <= edge.target_gid < 1_000_000_000 for edge in view.edges)
    assert density.sampled_sources == 32
    assert density.exact is False


def test_direct_structure_manifest_opens_a_procedural_oracle(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "cell_count": 1_000_000_000,
                "fanout": 8,
                "topology": "uniform_no_self",
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )

    structure = load_structure_manifest(path)

    assert structure.info.configured_neurons == 1_000_000_000
    assert len(structure.neighborhood(0, hops=1).edges) == 8


def test_materialized_large_result_requires_a_queryable_structure_artifact() -> None:
    run = load_run(DEMO_RUN)
    run = replace(
        run,
        population=replace(run.population, configured_neurons=4_097),
    )

    with pytest.raises(StructureUnavailable, match="did not record adjacency"):
        open_structure(run)


def test_evenly_spaced_gids_cover_the_full_address_range() -> None:
    assert evenly_spaced_gids(1_000_000_000, 5) == (
        0,
        249_999_999,
        499_999_999,
        749_999_999,
        999_999_999,
    )


def test_activity_stream_is_flush_visible_and_records_real_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "activity.jsonl"
    recorder = ActivityRecorder(
        path,
        run_id="live/demo",
        sample_count=4,
        state_stride=2,
    )
    circuit = RecurrentCellCircuit(
        token_cells=8,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=4,
            recurrent_fanout=4,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=4,
            random_seed=7,
        ),
        observation_sink=recorder.record,
    )
    reader = ActivityStreamReader(path)

    recorder.set_phase("training")
    first = circuit.observe(2)
    assert reader.poll() == 1
    assert reader.header is not None
    assert tuple(reader.header.sample_gids) == (0, 5, 10, 15)
    assert reader.frames[-1].spike_count == first.reservoir_spikes
    assert len(reader.frames[-1].voltage_relative_mv) == 4

    recorder.set_phase("calibration")
    circuit.observe(3)
    assert reader.poll() == 1
    assert reader.frames[-1].phase == "calibration"
    assert reader.frames[-1].has_state is False
    recorder.close()
    assert reader.poll() == 0
    assert reader.complete is True
    assert reader.declared_frame_count == 2


def test_activity_replay_indexes_the_full_trace_with_bounded_frame_objects(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replay.jsonl"
    _write_activity_records(path, _activity_records())
    reader = ActivityStreamReader(path, max_frames=2, random_access_cache=1)

    added = reader.poll(max_bytes=1)
    while not reader.caught_up:
        added += reader.poll(max_bytes=1)

    assert added == 8
    assert reader.total_frames == 8
    assert len(reader.frames) == 2
    assert reader.complete is True
    assert reader.frame_at(0).index == 0
    assert reader.frame_at(7).spike_count == 5
    assert reader.state_frame_at_or_before(5).index == 3
    assert [
        (run.start_position, run.end_position, run.phase)
        for run in reader.phase_runs
    ] == [(0, 4, "training"), (4, 8, "calibration")]


def test_activity_replay_navigates_recorded_events_and_preserves_extrema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    _write_activity_records(path, _activity_records())
    reader = ActivityStreamReader(path, max_frames=2)
    reader.poll()

    assert reader.event_at(3).kinds == ("state", "zero")
    assert reader.previous_event(5, "phase").position == 4
    assert reader.next_event(4, "state").position == 6
    assert reader.next_event(6, "zero") is None
    frame_indices, firing_rates = reader.timeline_points(max_points=4)
    assert len(frame_indices) <= 4
    assert min(firing_rates) == 0.0
    assert max(firing_rates) == 50.0


def test_activity_replay_waits_for_a_complete_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    records = _activity_records()
    header = json.dumps(records[0], sort_keys=True) + "\n"
    frame = json.dumps(records[1], sort_keys=True)
    path.write_text(header + frame, encoding="utf-8")
    reader = ActivityStreamReader(path)

    assert reader.poll() == 0
    assert reader.header is not None
    assert reader.total_frames == 0
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n")
    assert reader.poll() == 1
    assert reader.frame_at(0).index == 0


def test_capture_dynamics_rejects_non_integer_or_dormant_gids() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=8,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            active_reservoir_cells=8,
            input_fanout=4,
            recurrent_fanout=4,
        ),
    )

    with pytest.raises(ValueError, match="sample gids"):
        circuit.capture_dynamics((True,))
    with pytest.raises(ValueError, match="sample gids"):
        circuit.capture_dynamics((8,))
