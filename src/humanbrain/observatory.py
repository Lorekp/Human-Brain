"""Bounded, deterministic activity streaming for experiment observers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TextIO

from humanbrain.burst_metrics import BurstFractionTracker
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentStep


ACTIVITY_STREAM_FORMAT = "humanbrain-observatory-activity-v1"


def activity_stream_path(
    experiments_root: str | Path,
    experiment_id: str,
    arm: str,
) -> Path:
    return Path(experiments_root) / experiment_id / ".observatory" / f"{arm}.jsonl"


def evenly_spaced_gids(cell_count: int, sample_count: int) -> tuple[int, ...]:
    """Choose a deterministic, address-space-wide bounded cell sample."""
    if cell_count < 1:
        raise ValueError("cell_count must be positive")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count >= cell_count:
        return tuple(range(cell_count))
    if sample_count == 1:
        return (0,)
    return tuple(
        index * (cell_count - 1) // (sample_count - 1)
        for index in range(sample_count)
    )


class ActivityRecorder:
    """Write one flush-visible JSONL activity stream at token resolution.

    Recording is opt-in. Memory and file growth depend on recorded token frames
    and the bounded sample count, never on the configured population size.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        sample_count: int = 128,
        state_stride: int = 20,
        burst_tracker: BurstFractionTracker | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if state_stride < 1:
            raise ValueError("state_stride must be positive")
        self.path = Path(path)
        self.run_id = run_id
        self.sample_count = sample_count
        self.state_stride = state_stride
        # Burst-fraction instrumentation (DD-HTM TM surprise metric). None by
        # default: the stream schema stays byte-for-byte legacy when unset.
        self._burst_tracker = burst_tracker
        self.phase = "initializing"
        self._sample_gids: tuple[int, ...] | None = None
        self._frame_count = 0
        self._biological_time_ms = 0.0
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = self.path.open("x", encoding="utf-8", newline="\n")

    def set_phase(self, phase: str) -> None:
        if not phase:
            raise ValueError("phase must be non-empty")
        self.phase = phase

    def record(self, circuit: RecurrentCellCircuit, step: RecurrentStep) -> None:
        if self._closed:
            raise RuntimeError("activity recorder is closed")
        if self._burst_tracker is not None:
            # Instrumentation first, so a footer snapshot always reflects
            # every event this recorder consumed. The tracker only reads.
            self._burst_tracker.record(circuit, step)
        if self._sample_gids is None:
            self._sample_gids = evenly_spaced_gids(
                circuit.active_reservoir_cells,
                self.sample_count,
            )
            self._write(
                {
                    "type": "header",
                    "format": ACTIVITY_STREAM_FORMAT,
                    "run_id": self.run_id,
                    "configured_neurons": circuit.config.reservoir_cells,
                    "active_neurons": circuit.active_reservoir_cells,
                    "sample_gids": self._sample_gids,
                    "state_stride": self.state_stride,
                    "model_family": "GLIF5",
                    "token_interval_ms": step.biological_time_ms,
                    "integration_dt_ms": circuit.cell_parameters.dt_ms,
                    "structure": {
                        "kind": "dense_ladder",
                        "circuit_config": asdict(circuit.config),
                        "token_cells": circuit.token_cells,
                    },
                }
            )
        self._biological_time_ms += step.biological_time_ms
        firing_rate_hz = (
            step.reservoir_spikes
            / circuit.active_reservoir_cells
            / (step.biological_time_ms / 1_000.0)
        )
        activity_density = step.reservoir_spikes / (
            circuit.active_reservoir_cells * step.simulated_steps
        )
        record: dict[str, object] = {
            "type": "frame",
            "index": self._frame_count,
            "phase": self.phase,
            "token_id": step.token_id,
            "biological_time_ms": self._biological_time_ms,
            "spike_count": step.reservoir_spikes,
            "population_firing_rate_hz": firing_rate_hz,
            "neuron_step_activity_density": activity_density,
        }
        if self._frame_count % self.state_stride == 0:
            sample = circuit.capture_dynamics(self._sample_gids)
            record.update(
                {
                    "voltage_relative_mv": sample.voltage_relative_mv,
                    "synaptic_current_pa": sample.synaptic_current_pa,
                    "eligibility": sample.eligibility,
                }
            )
        self._write(record)
        self._frame_count += 1

    def close(self, *, complete: bool = True) -> None:
        if self._closed:
            return
        footer: dict[str, object] = {
            "type": "footer",
            "complete": bool(complete),
            "frame_count": self._frame_count,
            "biological_time_ms": self._biological_time_ms,
        }
        if self._burst_tracker is not None:
            footer["burst_metrics"] = self._burst_tracker.snapshot()
        self._write(footer)
        self._stream.close()
        self._closed = True

    def _write(self, record: dict[str, object]) -> None:
        payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._stream.write(payload + "\n")
        self._stream.flush()

    def __enter__(self) -> ActivityRecorder:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(complete=exc_type is None)
